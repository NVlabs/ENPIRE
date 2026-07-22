# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene management — YAML loading and MjSpec object injection.

Scenes are defined as YAML files in ``robot/models/objects/scenes/``.
Each file lists objects (primitives or meshes) to place on the table.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import mujoco
import yaml

_SCENES_DIR = Path(__file__).parents[2] / "robot" / "models" / "objects" / "scenes"
_MESHES_DIR = Path(__file__).parents[2] / "robot" / "models" / "objects" / "meshes"

_GEOM_TYPE_MAP = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    "mesh": mujoco.mjtGeom.mjGEOM_MESH,
}


@dataclasses.dataclass
class ContactPairDef:
    """Override contact friction between two bodies' geoms."""

    body1: str
    body2: str
    friction: list[float]  # 5-element: [tangent1, tangent2, spin, roll1, roll2]
    condim: int = 3


@dataclasses.dataclass
class SubGeomDef:
    """A sub-geom within a compound object."""

    type: str
    size: list[float]
    pos: list[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclasses.dataclass
class ObjectDef:
    name: str
    type: str  # box | sphere | cylinder | capsule | ellipsoid | mesh | compound
    size: list[float]
    pos: list[float]
    rgba: list[float] = dataclasses.field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])
    mass: float = 0.05
    quat: Optional[list[float]] = None
    friction: Optional[list[float]] = None
    mesh_file: Optional[str] = None  # relative to meshes dir
    scale: Optional[list[float]] = None  # mesh scale
    sub_geoms: Optional[list[SubGeomDef]] = None  # for compound type


@dataclasses.dataclass
class SceneDef:
    name: str
    objects: list[ObjectDef]
    contact_pairs: list[ContactPairDef] = dataclasses.field(default_factory=list)


def list_scenes() -> list[str]:
    """Return names of available scene YAML files (without extension)."""
    return sorted(p.stem for p in _SCENES_DIR.glob("*.yaml"))


def load_scene(name: str) -> SceneDef:
    """Parse a scene YAML file and return a SceneDef."""
    path = _SCENES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Scene not found: {path}")
    raw = yaml.safe_load(path.read_text())
    objects = []
    for obj in raw.get("objects", []):
        sub_geoms = None
        if "sub_geoms" in obj:
            sub_geoms = [
                SubGeomDef(
                    type=sg["type"],
                    size=sg["size"],
                    pos=sg.get("pos", [0.0, 0.0, 0.0]),
                )
                for sg in obj["sub_geoms"]
            ]
        objects.append(
            ObjectDef(
                name=obj["name"],
                type=obj["type"],
                size=obj.get("size", [0, 0, 0]),
                pos=obj["pos"],
                rgba=obj.get("rgba", [0.5, 0.5, 0.5, 1.0]),
                mass=obj.get("mass", 0.05),
                quat=obj.get("quat"),
                friction=obj.get("friction"),
                mesh_file=obj.get("mesh_file"),
                scale=obj.get("scale"),
                sub_geoms=sub_geoms,
            )
        )
    contact_pairs = []
    for cp in raw.get("contact_pairs", []):
        friction = cp["friction"]
        # Expand 3-element [slide, spin, roll] to 5-element [t1, t2, spin, r1, r2]
        if len(friction) == 3:
            friction = [friction[0], friction[0], friction[1], friction[2], friction[2]]
        contact_pairs.append(
            ContactPairDef(
                body1=cp["body1"],
                body2=cp["body2"],
                friction=friction,
                condim=cp.get("condim", 3),
            )
        )
    return SceneDef(name=name, objects=objects, contact_pairs=contact_pairs)


def add_objects_to_spec(spec: mujoco.MjSpec, scene: SceneDef) -> list[str]:
    """Add free-jointed bodies for each object in *scene* to *spec*.worldbody.

    For mesh objects, the mesh asset is added to the spec first.
    Returns list of body names added.
    """
    body_names = []
    body_geom_names: dict[str, list[str]] = {}  # body_name → [geom_name, ...]
    for obj in scene.objects:
        body = spec.worldbody.add_body(name=obj.name)
        body.pos = obj.pos
        if obj.quat is not None:
            body.quat = obj.quat
        body.add_freejoint(name=f"{obj.name}_joint")

        if obj.type == "compound":
            # Compound: multiple sub-geoms forming one rigid body
            if not obj.sub_geoms:
                raise ValueError(f"sub_geoms required for compound object {obj.name!r}")
            n = len(obj.sub_geoms)
            mass_per_geom = obj.mass / n
            for i, sg in enumerate(obj.sub_geoms):
                sg_type = _GEOM_TYPE_MAP.get(sg.type)
                if sg_type is None:
                    raise ValueError(
                        f"Unknown sub_geom type: {sg.type!r} in {obj.name!r}"
                    )
                size = list(sg.size)
                while len(size) < 3:
                    size.append(0.0)
                sg_kwargs: dict = {
                    "name": f"{obj.name}_geom_{i}",
                    "type": sg_type,
                    "size": size,
                    "pos": sg.pos,
                    "rgba": obj.rgba,
                    "mass": mass_per_geom,
                    "condim": 6,  # enable torsional/rolling friction
                }
                if obj.friction is not None:
                    sg_kwargs["friction"] = obj.friction
                body.add_geom(**sg_kwargs)
            body_geom_names[obj.name] = [f"{obj.name}_geom_{i}" for i in range(n)]
        else:
            geom_type = _GEOM_TYPE_MAP.get(obj.type)
            if geom_type is None:
                raise ValueError(
                    f"Unknown geom type: {obj.type!r} for object {obj.name!r}"
                )

            # Mesh asset
            meshname = None
            if obj.type == "mesh":
                if not obj.mesh_file:
                    raise ValueError(f"mesh_file required for mesh object {obj.name!r}")
                mesh_path = _MESHES_DIR / obj.mesh_file
                if not mesh_path.exists():
                    raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
                meshname = f"{obj.name}_mesh"
                mesh_asset = spec.add_mesh()
                mesh_asset.name = meshname
                mesh_asset.file = str(mesh_path)
                if obj.scale is not None:
                    mesh_asset.scale = obj.scale

            geom_kwargs: dict = {
                "name": f"{obj.name}_geom",
                "type": geom_type,
                "rgba": obj.rgba,
                "mass": obj.mass,
                "condim": 6,  # enable torsional/rolling friction
            }
            if obj.type == "mesh":
                geom_kwargs["meshname"] = meshname
            else:
                size = list(obj.size)
                while len(size) < 3:
                    size.append(0.0)
                geom_kwargs["size"] = size
            if obj.friction is not None:
                geom_kwargs["friction"] = obj.friction

            body.add_geom(**geom_kwargs)
            body_geom_names[obj.name] = [f"{obj.name}_geom"]

        body_names.append(obj.name)

    # Inject contact pairs with per-pair friction overrides
    for cp in scene.contact_pairs:
        geoms1 = body_geom_names.get(cp.body1, [])
        geoms2 = body_geom_names.get(cp.body2, [])
        if not geoms1:
            raise ValueError(
                f"contact_pair body1={cp.body1!r} not found in scene objects"
            )
        if not geoms2:
            raise ValueError(
                f"contact_pair body2={cp.body2!r} not found in scene objects"
            )
        for g1 in geoms1:
            for g2 in geoms2:
                pair = spec.add_pair()
                pair.geomname1 = g1
                pair.geomname2 = g2
                pair.condim = cp.condim
                pair.friction = cp.friction

    return body_names
