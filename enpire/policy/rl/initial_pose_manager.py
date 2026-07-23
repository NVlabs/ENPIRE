# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Mapping

import numpy as np

from enpire.env.forge.robot.constants import HOVER_EE_POSE
from enpire.policy.rl.config import DataCollectionConfig


class InitialPoseManager:
    """Owns selected initial pose, pose randomization, and pose-relative bounds."""

    def __init__(self, cfg: DataCollectionConfig):
        self.cfg = cfg
        self.poses = self._load_poses(cfg)
        self.index = 0
        self.last_offset = np.zeros(3, dtype=np.float64)
        self.last_out_of_range: dict | None = None

    @property
    def count(self) -> int:
        return len(self.poses)

    @property
    def label(self) -> str | None:
        if not self.poses:
            return None
        return f"Pose {self.index + 1}/{len(self.poses)}"

    def describe_current(self) -> str:
        base = self.current_base_pose()
        parts = []
        for side in ("left", "right"):
            if side in base:
                parts.append(f"{side}={np.round(base[side]['position'], 4)}")
        prefix = self.label or "Default hover pose"
        return f"{prefix}: {' '.join(parts)}"

    def select_next(self) -> None:
        self._select(self.index + 1)

    def select_previous(self) -> None:
        self._select(self.index - 1)

    def select_first(self) -> None:
        self._select(0)

    def select_initial_position_index(self, initial_position_index: int) -> None:
        if not 1 <= initial_position_index <= self.count:
            raise ValueError(
                f"initial_position_index must be between 1 and {self.count}"
            )
        self._select(initial_position_index - 1)

    def current_base_pose(self) -> dict:
        if self.poses:
            return deepcopy(self.poses[self.index])
        return deepcopy(HOVER_EE_POSE)

    def set_base_pose_from_observation(self, obs: Mapping) -> None:
        pose = self.current_base_pose() if self.poses else {}
        pose.update(self.pose_from_observation(obs))
        self.poses = [pose]
        self.index = 0
        self.last_offset = np.zeros(3, dtype=np.float64)

    def pose_from_observation(
        self,
        obs: Mapping,
        *,
        z_offset_m: float = 0.0,
    ) -> dict:
        pose: dict = {}
        for side in self._enabled_sides():
            pos = obs.get(f"{side}_ee_pos")
            rot6d = obs.get(f"{side}_ee_rot6d")
            if pos is None or rot6d is None:
                continue
            position = np.asarray(pos, dtype=np.float64).reshape(3).copy()
            position[2] += float(z_offset_m)
            side_pose = {
                "position": position.tolist(),
                "rot6d": np.asarray(rot6d, dtype=np.float64).reshape(6).tolist(),
            }
            gripper = obs.get(f"{side}_gripper_pos")
            if gripper is not None:
                side_pose["gripper_pos"] = (
                    np.asarray(gripper, dtype=np.float64).reshape(1).tolist()
                )
            pose[side] = side_pose
        if not pose:
            raise ValueError(
                "Could not build EE pose from observation; expected "
                f"EE pose keys for enabled_sides={self.cfg.enabled_sides!r}"
            )
        return pose

    def build_center_pose(self) -> dict:
        self.last_offset = np.zeros(3, dtype=np.float64)
        return self.current_base_pose()

    def build_episode_start_pose(self, *, randomize: bool | None = None) -> dict:
        use_random = self.cfg.randomize_initial_pose if randomize is None else randomize
        offset = self.sample_initial_offset() if use_random else np.zeros(3, dtype=np.float64)
        return self.pose_with_offset(offset)

    def sample_initial_offset(self) -> np.ndarray:
        self.last_offset = np.array(
            [
                np.random.uniform(*self.cfg.x_init_lim),
                np.random.uniform(*self.cfg.y_init_lim),
                np.random.uniform(*self.cfg.z_init_lim),
            ],
            dtype=np.float64,
        )
        return self.last_offset.copy()

    def pose_with_offset(self, offset: np.ndarray | list[float] | tuple[float, ...]) -> dict:
        offset = np.asarray(offset, dtype=np.float64).reshape(3)
        pose = self.current_base_pose()
        enabled = self._enabled_sides()
        for side, side_pose in pose.items():
            if side not in enabled:
                continue
            pos = np.asarray(side_pose["position"], dtype=np.float64).reshape(3)
            side_pose["position"] = (pos + offset).tolist()
        return pose

    def delta_from_base(self, obs: Mapping) -> dict[str, np.ndarray]:
        base = self.current_base_pose()
        deltas: dict[str, np.ndarray] = {}
        for side in self._enabled_sides():
            ee_pos = obs.get(f"{side}_ee_pos")
            side_pose = base.get(side)
            if ee_pos is None or side_pose is None:
                continue
            deltas[side] = (
                np.asarray(ee_pos, dtype=np.float64).reshape(3)
                - np.asarray(side_pose["position"], dtype=np.float64).reshape(3)
            )
        return deltas

    def is_out_of_range(self, obs: Mapping) -> bool:
        self.last_out_of_range = None
        if not self.cfg.enable_oor_check:
            return False
        limits = (self.cfg.x_oor_lim, self.cfg.y_oor_lim, self.cfg.z_oor_lim)
        axis_names = ("x", "y", "z")
        for side, delta in self.delta_from_base(obs).items():
            for axis, (lo, hi) in enumerate(limits):
                if delta[axis] < lo or delta[axis] > hi:
                    self.last_out_of_range = {
                        "side": side,
                        "axis": axis_names[axis],
                        "delta": float(delta[axis]),
                        "lo": float(lo),
                        "hi": float(hi),
                    }
                    return True
        return False

    def initial_boundary_corners(self, z_idx: int = 1) -> list[dict]:
        return [
            self.pose_with_offset(offset)
            for offset in self._box_corners(
                self.cfg.x_init_lim,
                self.cfg.y_init_lim,
                self.cfg.z_init_lim,
                z_idx,
            )
        ]

    def oor_boundary_corners(self, z_idx: int = 1) -> list[dict]:
        return [
            self.pose_with_offset(offset)
            for offset in self._box_corners(
                self.cfg.x_oor_lim,
                self.cfg.y_oor_lim,
                self.cfg.z_oor_lim,
                z_idx,
            )
        ]

    def boundary_corner_pose(
        self,
        *,
        kind: str,
        corner_idx: int,
        z_idx: int,
    ) -> dict:
        pose, _ = self._boundary_corner(kind=kind, corner_idx=corner_idx, z_idx=z_idx)
        return pose

    def boundary_corner_offset(
        self,
        *,
        kind: str,
        corner_idx: int,
        z_idx: int,
    ) -> np.ndarray:
        _, offset = self._boundary_corner(kind=kind, corner_idx=corner_idx, z_idx=z_idx)
        return offset

    def _boundary_corner(
        self,
        *,
        kind: str,
        corner_idx: int,
        z_idx: int,
    ) -> tuple[dict, np.ndarray]:
        if kind == "initial":
            x_lim, y_lim, z_lim = (
                self.cfg.x_init_lim,
                self.cfg.y_init_lim,
                self.cfg.z_init_lim,
            )
        elif kind == "oor":
            x_lim, y_lim, z_lim = (
                self.cfg.x_oor_lim,
                self.cfg.y_oor_lim,
                self.cfg.z_oor_lim,
            )
        else:
            raise ValueError(f"unknown boundary kind: {kind!r}")
        corners = self._box_corners(x_lim, y_lim, z_lim, z_idx)
        offset = corners[corner_idx % len(corners)]
        return self.pose_with_offset(offset), offset.copy()

    def _select(self, new_index: int) -> None:
        if not self.poses:
            return
        self.index = new_index % len(self.poses)

    def _enabled_sides(self) -> set[str]:
        return (
            {"left", "right"}
            if self.cfg.enabled_sides == "both"
            else {self.cfg.enabled_sides}
        )

    @staticmethod
    def _box_corners(
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
        z_lim: tuple[float, float],
        z_idx: int,
    ) -> list[np.ndarray]:
        z = z_lim[z_idx]
        return [
            np.array([x_lim[0], y_lim[0], z], dtype=np.float64),
            np.array([x_lim[1], y_lim[0], z], dtype=np.float64),
            np.array([x_lim[1], y_lim[1], z], dtype=np.float64),
            np.array([x_lim[0], y_lim[1], z], dtype=np.float64),
        ]

    @staticmethod
    def _load_poses(cfg: DataCollectionConfig) -> list[dict]:
        if not cfg.initial_positions_file:
            return []

        import yaml

        path = Path(cfg.initial_positions_file)
        if not path.is_absolute() and cfg.config_file:
            path = Path(cfg.config_file).parent / path

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        if cfg.station and isinstance(data.get(cfg.station), dict):
            data = data[cfg.station]

        keys = sorted(
            (key for key in data if key.startswith("position_")),
            key=lambda key: int(key.split("_", 1)[1]),
        )
        poses = [data[key] for key in keys]
        print(f"[INFO] Initial positions loaded from: {path}")
        if cfg.station:
            print(f"[INFO] Using station: {cfg.station}")
        if not poses:
            station_hint = f" for station '{cfg.station}'" if cfg.station else ""
            no_station_hint = " (no --station provided)" if not cfg.station else ""
            sys.exit(
                f"\033[1;31m[ERROR] No initial poses loaded from {path}"
                f"{station_hint}{no_station_hint}."
                f" Check YAML format or pass --station.\033[0m"
            )
        else:
            print(f"[INFO] Loaded {len(poses)} initial pose(s)")
        return poses

