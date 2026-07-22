# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def inject_camera(
    base_xml: str,
    camera_name: str,
    T_base_from_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    image_size: tuple[int, int],
    output_xml: str,
    camera_mesh: str = "d405",
    camera_mesh_file: str | None = None,
) -> None:
    """Add a calibrated camera body to a MuJoCo station XML."""
    ET.register_namespace("", "")
    tree = ET.parse(base_xml)
    root = tree.getroot()

    pos = T_base_from_cam[:3, 3]
    quat_wxyz = Rotation.from_matrix(T_base_from_cam[:3, :3]).as_quat(scalar_first=True)
    w, h = image_size
    fy = K[1, 1]
    fovy = 2 * math.degrees(math.atan(h / (2 * fy)))

    # Ensure the camera mesh exists in <asset>. The Fello base description
    # already carries the d405 asset; this path is mainly for older models.
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if not any(m.get("name") == camera_mesh for m in asset.findall("mesh")):
        m = ET.SubElement(asset, "mesh")
        m.set("name", camera_mesh)
        if camera_mesh_file is not None:
            m.set("file", camera_mesh_file)

    # Build camera body and append to <worldbody>
    worldbody = root.find("worldbody")
    body = ET.SubElement(worldbody, "body")
    body.set("name", camera_name)
    body.set("pos", " ".join(f"{v:.8f}" for v in pos))
    body.set("quat", " ".join(f"{v:.8f}" for v in quat_wxyz))

    site = ET.SubElement(body, "site")
    site.set("name", f"{camera_name}_site")
    site.set("pos", "0 0 0")
    site.set("size", "0.01")
    site.set("rgba", "0 1 1 1")

    geom = ET.SubElement(body, "geom")
    geom.set("type", "mesh")
    geom.set("mesh", camera_mesh)
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
    geom.set("rgba", "0.2 0.2 0.2 1")

    cam = ET.SubElement(body, "camera")
    cam.set("name", camera_name)
    cam.set("pos", "0 0 0")
    cam.set("quat", "1 0 0 0")
    cam.set("fovy", f"{fovy:.4f}")
    cam.set("resolution", f"{w} {h}")

    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="unicode", xml_declaration=False)
    # Prepend standard XML declaration
    with open(output_xml, "r") as f:
        content = f.read()
    with open(output_xml, "w") as f:
        f.write("<?xml version='1.0' encoding='utf-8'?>\n" + content)

    print(f"  Camera '{camera_name}' injected.")
    print(f"    pos  : {pos.round(6).tolist()}")
    print(f"    quat : {quat_wxyz.round(6).tolist()}")
    print(f"    fovy : {fovy:.4f} deg")
    print(f"    res  : {w}x{h}")
    print(f"    dist : {dist.round(6).tolist()}")


def update_body_extrinsic(
    base_xml: str,
    body_name: str,
    T: np.ndarray,
    output_xml: str,
) -> None:
    """Update pos and quat of an existing named body in a MuJoCo XML."""
    tree = ET.parse(base_xml)
    root = tree.getroot()

    pos = T[:3, 3]
    quat_wxyz = Rotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True)

    body = next(
        (elem for elem in root.iter("body") if elem.get("name") == body_name),
        None,
    )
    if body is None:
        raise ValueError(f"Body '{body_name}' not found in {base_xml}")

    body.set("pos", " ".join(f"{v:.8f}" for v in pos))
    body.set("quat", " ".join(f"{v:.8f}" for v in quat_wxyz))

    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="unicode", xml_declaration=False)
    with open(output_xml, "r") as f:
        content = f.read()
    with open(output_xml, "w") as f:
        f.write("<?xml version='1.0' encoding='utf-8'?>\n" + content)

    print(f"  Body '{body_name}' extrinsic updated.")
    print(f"    pos  : {pos.round(6).tolist()}")
    print(f"    quat : {quat_wxyz.round(6).tolist()}")
