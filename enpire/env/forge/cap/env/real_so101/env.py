# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real single-arm SO-101 environment for direct-mode runs.

Covers the arm API of ``RealYamEnv``: observations with FK, joint commands,
homing, and the gripper. Cameras are not wired up yet.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from enpire.env.forge.cap.env.base.profile import RobotProfile, so101_profile
from enpire.env.forge.robot.so101.constants import SO101_DOF, SO101_FOLLOWER_PORT


class RealSo101Env:
    def __init__(self, side: str = "left", port: int = SO101_FOLLOWER_PORT, arm=None, kin=None):
        self._profile: RobotProfile = so101_profile(side)
        self._side = side
        if arm is None:
            from enpire.env.forge.robot.yam.arm_client import FollowerRobotClient

            arm = FollowerRobotClient(port=port, label="SO-101 follower")
        if kin is None:
            from enpire.env.forge.robot.so101.kinematics import So101Kinematics

            kin = So101Kinematics()
        self._arm = arm
        self.kin = kin
        self._kin_lock = threading.Lock()
        self._last_cmd: np.ndarray | None = None

    @property
    def profile(self) -> RobotProfile:
        return self._profile

    def _check_side(self, side: str) -> None:
        if side != self._side:
            raise ValueError(f"SO-101 env drives a single arm named {self._side!r}, got {side!r}")

    def get_observations(self, side: str) -> dict[str, np.ndarray]:
        """Return {joint_pos(5), gripper_pos(1), ee_pos(3), ee_quat(4)}."""
        self._check_side(side)
        obs = self._arm.get_observations()
        with self._kin_lock:
            ee_pos, ee_quat = self.kin.forward_kinematics(obs["joint_pos"])
        return {
            "joint_pos": obs["joint_pos"],
            "gripper_pos": obs["gripper_pos"],
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
        }

    def command_joint_pos(self, side: str, joint_pos: np.ndarray) -> None:
        """Send a [joint x5 + gripper x1] position command."""
        self._check_side(side)
        self._send(joint_pos)

    def command_joint_state(self, side: str, state: dict) -> None:
        self._check_side(side)
        self._last_cmd = np.asarray(state["pos"], dtype=float).copy()
        self._arm.command_joint_state(state)

    def _send(self, joint_pos: np.ndarray) -> None:
        joint_pos = np.asarray(joint_pos, dtype=float)
        self._arm.command_joint_pos(joint_pos)
        self._last_cmd = joint_pos.copy()

    def move_to_joint_pos(
        self,
        side: str,
        joint_pos: np.ndarray,
        max_joint_vel: float = 0.5,
        rate_hz: float | None = None,
    ) -> None:
        """Linearly interpolate from the current state to ``joint_pos``."""
        self._check_side(side)
        target = np.asarray(joint_pos, dtype=float)
        start = self._arm.get_joint_pos()
        rate_hz = rate_hz or self._profile.control_freq_hz
        max_delta = float(np.max(np.abs(target[:SO101_DOF] - start[:SO101_DOF])))
        steps = max(1, int(np.ceil(max_delta / max_joint_vel * rate_hz)))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            self._send(start + alpha * (target - start))
            time.sleep(1.0 / rate_hz)

    def go_home(self, max_joint_vel: float = 0.5) -> None:
        arm = self._profile.arms[self._side]
        home = np.concatenate([arm.home_joint_pos, arm.home_gripper_pos])
        self.move_to_joint_pos(self._side, home, max_joint_vel=max_joint_vel)

    def set_gripper(self, side: str, value: float) -> None:
        """Set the gripper opening in [0, 1] and hold the last arm target.

        Holding the commanded target rather than the measured position keeps
        the arm from creeping down under gravity on repeated calls.
        """
        self._check_side(side)
        target = self._last_cmd if self._last_cmd is not None else self._arm.get_joint_pos()
        target = np.array(target, dtype=float)
        target[SO101_DOF] = float(np.clip(value, 0.0, 1.0))
        self._send(target)

    def set_recorder(self, recorder) -> None:
        """No-op, recording is not supported on real hardware."""

    def close(self) -> None:
        pass
