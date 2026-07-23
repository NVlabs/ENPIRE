# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from enpire.policy.rl.initial_pose_manager import InitialPoseManager

BoundaryKind = Literal["initial", "oor"]


@dataclass
class ParkingNavigator:
    boundary_kind: BoundaryKind = "initial"
    z_idx: int = 1
    corner_idx: int = 0

    @property
    def z_label(self) -> str:
        return "high" if self.z_idx else "low"

    def select_initial_boundary(self) -> None:
        if self.boundary_kind != "initial":
            self.corner_idx = 0
        self.boundary_kind = "initial"

    def select_oor_boundary(self) -> None:
        if self.boundary_kind != "oor":
            self.corner_idx = 0
        self.boundary_kind = "oor"

    def select_z_high(self) -> None:
        self.z_idx = 1
        self.corner_idx = 0

    def select_z_low(self) -> None:
        self.z_idx = 0
        self.corner_idx = 0

    def next_corner_pose(self, pose_manager: InitialPoseManager) -> tuple[dict, str]:
        pose = pose_manager.boundary_corner_pose(
            kind=self.boundary_kind,
            corner_idx=self.corner_idx,
            z_idx=self.z_idx,
        )
        label = (
            f"{self.boundary_kind} boundary z-{self.z_label} "
            f"corner {self.corner_idx + 1}/4"
        )
        self.corner_idx = (self.corner_idx + 1) % 4
        return pose, label

