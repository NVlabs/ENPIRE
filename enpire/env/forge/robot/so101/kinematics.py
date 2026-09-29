# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward and inverse kinematics for a single SO-101 arm.

The arm has five joints, so a full 6D pose target is usually unreachable.
IK weights position above orientation by default and returns the residual
errors so callers can decide whether the solution is good enough.
"""

from __future__ import annotations

import mink
import mujoco
import numpy as np

from enpire.env.forge.robot.so101.constants import SO101_DOF, SO101_EE_SITE


def _quat_distance(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> float:
    return min(float(np.linalg.norm(a_xyzw - b_xyzw)), float(np.linalg.norm(a_xyzw + b_xyzw)))


class So101Kinematics:
    def __init__(
        self,
        position_cost: float = 1.0,
        orientation_cost: float = 0.1,
        lm_damping: float = 1.0,
    ):
        from enpire.env.forge.robot.models.so101.paths import get_so101_xml

        self._model = mujoco.MjModel.from_xml_path(str(get_so101_xml()))
        self._position_cost = position_cost
        self._orientation_cost = orientation_cost
        self.configuration = mink.Configuration(self._model)
        self.task = mink.FrameTask(
            frame_name=SO101_EE_SITE,
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping,
        )
        self._limits = [mink.ConfigurationLimit(self._model)]

    def _set_arm_q(self, joint_pos: np.ndarray) -> None:
        qpos = self.configuration.data.qpos.copy()
        qpos[:SO101_DOF] = np.asarray(joint_pos, dtype=np.float64)[:SO101_DOF]
        self.configuration.update(qpos)

    def _ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pose = self.configuration.get_transform_frame_to_world(SO101_EE_SITE, "site")
        return pose.translation().copy(), pose.rotation().wxyz[[1, 2, 3, 0]].copy()

    def forward_kinematics(self, joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the end-effector position and xyzw quaternion in the base frame."""
        self._set_arm_q(joint_pos)
        return self._ee_pose()

    def inverse_kinematics(
        self,
        ee_pos: np.ndarray,
        ee_quat_xyzw: np.ndarray | None = None,
        seed: np.ndarray | None = None,
        dt: float = 0.01,
        solver: str = "daqp",
        damping: float = 1e-3,
        pos_threshold: float = 1e-3,
        max_iters: int = 200,
    ) -> tuple[np.ndarray, float, float]:
        """Solve for arm joints reaching ``ee_pos``.

        Pass ``ee_quat_xyzw=None`` for a position-only solve. Returns the joint
        solution, the position error in meters, and the quaternion distance
        (0.0 when no orientation was requested).
        """
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        self._set_arm_q(np.zeros(SO101_DOF) if seed is None else seed)

        if ee_quat_xyzw is None:
            current_quat = self._ee_pose()[1]
            target_wxyz = current_quat[[3, 0, 1, 2]]
            self.task.set_orientation_cost(0.0)
        else:
            ee_quat_xyzw = np.asarray(ee_quat_xyzw, dtype=np.float64)
            target_wxyz = ee_quat_xyzw[[3, 0, 1, 2]]
            self.task.set_orientation_cost(self._orientation_cost)
        self.task.set_target(
            mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=target_wxyz), ee_pos)
        )

        for _ in range(max_iters):
            vel = mink.solve_ik(
                self.configuration, [self.task], dt, solver, damping, limits=self._limits
            )
            self.configuration.integrate_inplace(vel, dt)
            if np.linalg.norm(self._ee_pose()[0] - ee_pos) <= pos_threshold:
                break

        pos, quat = self._ee_pose()
        pos_err = float(np.linalg.norm(pos - ee_pos))
        rot_err = 0.0 if ee_quat_xyzw is None else _quat_distance(quat, ee_quat_xyzw)
        return self.configuration.q[:SO101_DOF].copy(), pos_err, rot_err
