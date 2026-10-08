# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SO-101 follower and leader drivers built on LeRobot.

The follower exposes the same surface as ``YamRobot`` so the portal arm
server and ``FollowerRobotClient`` work unchanged. Joint values are radians and the
gripper is normalized to [0, 1]. LeRobot works in degrees and a 0 to 100
gripper range, so all conversion happens here.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from enpire.env.forge.robot.so101.constants import (
    SO101_ARM_JOINT_NAMES,
    SO101_DOF,
    SO101_GRIPPER_NAME,
    SO101_JOINT_LIMITS_HIGH,
    SO101_JOINT_LIMITS_LOW,
    SO101_MOTOR_NAMES,
)

logger = logging.getLogger(__name__)

_GRIPPER_SCALE = 100.0


_LEROBOT_MISSING = (
    "SO-101 control needs LeRobot with Feetech support. "
    "Install it with: uv pip install 'lerobot[feetech]'"
)


def _import_lerobot_follower() -> tuple[type, type]:
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError as exc:
            raise RuntimeError(_LEROBOT_MISSING) from exc
    return SO101Follower, SO101FollowerConfig


def _import_lerobot_leader() -> tuple[type, type]:
    try:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    except ImportError:
        try:
            from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
        except ImportError as exc:
            raise RuntimeError(_LEROBOT_MISSING) from exc
    return SO101Leader, SO101LeaderConfig


def _connect_calibrated(device: Any, port: str, robot_id: str, device_type: str) -> None:
    """Connect without LeRobot's interactive calibration prompt.

    A stored calibration file is written to the motors when they disagree
    with it. With no calibration file the arm must be calibrated first via
    ``lerobot-calibrate``.
    """
    device.connect(calibrate=False)
    if device.is_calibrated:
        return
    if not device.calibration:
        device.disconnect()
        kind = "teleop" if device_type.endswith("leader") else "robot"
        raise RuntimeError(
            f"[{port}] no LeRobot calibration for id {robot_id!r}. Run "
            f"lerobot-calibrate --{kind}.type={device_type} --{kind}.port={port} "
            f"--{kind}.id={robot_id}"
        )
    logger.info("[%s] writing stored calibration to motors", port)
    device.bus.write_calibration(device.calibration)


def _read_temperatures(device: Any) -> np.ndarray:
    temps = device.bus.sync_read("Present_Temperature", normalize=False)
    return np.asarray([float(temps[name]) for name in SO101_MOTOR_NAMES], dtype=np.float32)


def action_from_joint_pos(joint_pos: np.ndarray) -> dict[str, float]:
    """Convert ``[arm rad x5, gripper 0..1]`` into a LeRobot action dict."""
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    arm = np.clip(joint_pos[:SO101_DOF], SO101_JOINT_LIMITS_LOW, SO101_JOINT_LIMITS_HIGH)
    gripper = float(np.clip(joint_pos[SO101_DOF], 0.0, 1.0))
    action = {f"{name}.pos": float(np.rad2deg(v)) for name, v in zip(SO101_ARM_JOINT_NAMES, arm)}
    action[f"{SO101_GRIPPER_NAME}.pos"] = gripper * _GRIPPER_SCALE
    return action


def joint_pos_from_observation(obs: dict[str, Any]) -> np.ndarray:
    """Convert a LeRobot observation dict into ``[arm rad x5, gripper 0..1]``."""
    arm = np.deg2rad([float(obs[f"{name}.pos"]) for name in SO101_ARM_JOINT_NAMES])
    gripper = float(obs[f"{SO101_GRIPPER_NAME}.pos"]) / _GRIPPER_SCALE
    return np.concatenate([arm, [gripper]])


class So101Robot:
    NUM_JOINTS = SO101_DOF + 1

    def __init__(
        self,
        port: str,
        robot_id: str,
        calibration_dir: str | Path | None = None,
        max_relative_target: float | None = None,
        robot: Any = None,
    ):
        self.port = port
        self.robot_id = robot_id
        self._lock = threading.Lock()
        self._connected = False
        if robot is None:
            follower_cls, config_cls = _import_lerobot_follower()
            config = config_cls(
                port=port,
                id=robot_id,
                calibration_dir=Path(calibration_dir) if calibration_dir else None,
                max_relative_target=max_relative_target,
                use_degrees=True,
            )
            robot = follower_cls(config)
        self._robot = robot

    def connect(self) -> None:
        _connect_calibrated(self._robot, self.port, self.robot_id, "so101_follower")
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        with self._lock:
            self._robot.disconnect()
            self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(f"[{self.port}] SO-101 is not connected")

    def get_joint_pos(self) -> np.ndarray:
        """Returns arm positions (rad) + gripper normalized [0=close, 1=open]."""
        self._require_connected()
        with self._lock:
            obs = self._robot.get_observation()
        return joint_pos_from_observation(obs)

    def get_observations(self) -> dict[str, np.ndarray]:
        pos = self.get_joint_pos()
        return {"joint_pos": pos[:SO101_DOF], "gripper_pos": pos[SO101_DOF:]}

    def get_motor_temperatures(self) -> np.ndarray:
        self._require_connected()
        with self._lock:
            return _read_temperatures(self._robot)

    def command_joint_pos(
        self,
        joint_pos: np.ndarray,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
        gripper_vel_limit: float | None = None,
        gripper_torque_limit_nm: float | None = None,
    ) -> None:
        """Send a position target. Gains and gripper limits are accepted for
        ``YamRobot`` compatibility and ignored, since the servos run their own
        position loop."""
        self._require_connected()
        joint_pos = np.asarray(joint_pos, dtype=float)
        if joint_pos.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.port}] expected joint_pos shape {(self.NUM_JOINTS,)}, got {joint_pos.shape}"
            )
        if not np.all(np.isfinite(joint_pos)):
            raise ValueError(f"[{self.port}] joint_pos contains non-finite values: {joint_pos}")
        with self._lock:
            self._robot.send_action(action_from_joint_pos(joint_pos))

    def command_joint_state(self, joint_state: dict) -> None:
        self.command_joint_pos(joint_state["pos"])


class So101LeaderArm:
    """SO-101 leader arm read through the LeRobot SO leader teleoperator.

    ``get_info`` matches the YAM leader RPC. The SO-101 leader has no buttons,
    so the second element is always empty.
    """

    def __init__(
        self,
        port: str,
        robot_id: str,
        calibration_dir: str | Path | None = None,
        teleop: Any = None,
    ):
        self.port = port
        self.robot_id = robot_id
        self._lock = threading.Lock()
        if teleop is None:
            leader_cls, config_cls = _import_lerobot_leader()
            config = config_cls(
                port=port,
                id=robot_id,
                calibration_dir=Path(calibration_dir) if calibration_dir else None,
                use_degrees=True,
            )
            teleop = leader_cls(config)
        self._teleop = teleop

    def connect(self) -> None:
        _connect_calibrated(self._teleop, self.port, self.robot_id, "so101_leader")

    def disconnect(self) -> None:
        with self._lock:
            self._teleop.disconnect()

    def get_info(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            action = self._teleop.get_action()
        return joint_pos_from_observation(action), np.zeros(0, dtype=bool)

    def get_motor_temperatures(self) -> np.ndarray:
        with self._lock:
            return _read_temperatures(self._teleop)
