# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from enpire.env.forge.robot.so101.constants import SO101_JOINT_LIMITS_HIGH, SO101_MOTOR_NAMES
from enpire.env.forge.robot.so101.so101_controller import (
    So101LeaderArm,
    So101Robot,
    action_from_joint_pos,
    joint_pos_from_observation,
)


class FakeBus:
    def __init__(self):
        self.written_calibration = None

    def write_calibration(self, calibration):
        self.written_calibration = calibration

    def sync_read(self, data_name, motors=None, *, normalize=True):
        assert data_name == "Present_Temperature"
        assert normalize is False
        return {name: 30 + i for i, name in enumerate(SO101_MOTOR_NAMES)}


class FakeFollower:
    def __init__(self, *, calibrated=True, calibration=None):
        self.bus = FakeBus()
        self.is_calibrated = calibrated
        self.calibration = calibration or {}
        self.connected = False
        self.actions = []
        self.observation = {f"{name}.pos": 0.0 for name in SO101_MOTOR_NAMES}

    def connect(self, calibrate=True):
        assert calibrate is False
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        return dict(self.observation)

    def send_action(self, action):
        self.actions.append(action)
        return action


def _connected_robot(**kwargs):
    fake = FakeFollower(**kwargs)
    robot = So101Robot(port="/dev/null", robot_id="test", robot=fake)
    robot.connect()
    return robot, fake


def test_action_and_observation_conversions_round_trip():
    joint_pos = np.array([0.1, -0.5, 0.7, 0.2, -1.0, 0.25])

    action = action_from_joint_pos(joint_pos)

    assert action["shoulder_pan.pos"] == pytest.approx(np.rad2deg(0.1))
    assert action["gripper.pos"] == pytest.approx(25.0)
    np.testing.assert_allclose(joint_pos_from_observation(action), joint_pos)


def test_action_clips_to_joint_limits_and_gripper_range():
    action = action_from_joint_pos(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 1.5]))

    assert action["shoulder_pan.pos"] == pytest.approx(np.rad2deg(SO101_JOINT_LIMITS_HIGH[0]))
    assert action["gripper.pos"] == pytest.approx(100.0)


def test_observations_split_arm_and_gripper():
    robot, fake = _connected_robot()
    fake.observation["elbow_flex.pos"] = 90.0
    fake.observation["gripper.pos"] = 50.0

    obs = robot.get_observations()

    assert obs["joint_pos"].shape == (5,)
    assert obs["joint_pos"][2] == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(obs["gripper_pos"], [0.5])


def test_command_joint_state_ignores_gains():
    robot, fake = _connected_robot()

    robot.command_joint_state({"pos": np.zeros(6), "kp": np.ones(6), "kd": np.ones(6)})

    assert fake.actions == [action_from_joint_pos(np.zeros(6))]


def test_command_rejects_bad_shapes_and_non_finite_values():
    robot, _ = _connected_robot()

    with pytest.raises(ValueError, match="shape"):
        robot.command_joint_pos(np.zeros(7))
    with pytest.raises(ValueError, match="non-finite"):
        robot.command_joint_pos(np.array([np.nan, 0, 0, 0, 0, 0]))


def test_commands_require_connection():
    robot = So101Robot(port="/dev/null", robot_id="test", robot=FakeFollower())

    with pytest.raises(RuntimeError, match="not connected"):
        robot.command_joint_pos(np.zeros(6))


def test_connect_writes_stored_calibration_when_motors_disagree():
    calibration = {"shoulder_pan": object()}
    _, fake = _connected_robot(calibrated=False, calibration=calibration)

    assert fake.bus.written_calibration is calibration


def test_connect_without_calibration_file_fails_and_disconnects():
    fake = FakeFollower(calibrated=False)
    robot = So101Robot(port="/dev/ttyACM0", robot_id="arm", robot=fake)

    with pytest.raises(RuntimeError, match="lerobot-calibrate"):
        robot.connect()
    assert not fake.connected


def test_motor_temperatures_follow_motor_order():
    robot, _ = _connected_robot()

    np.testing.assert_array_equal(robot.get_motor_temperatures(), [30, 31, 32, 33, 34, 35])


class FakeLeader(FakeFollower):
    def get_action(self):
        return dict(self.observation)


def test_leader_get_info_returns_joint_pos_and_no_buttons():
    fake = FakeLeader()
    fake.observation["wrist_roll.pos"] = -90.0
    fake.observation["gripper.pos"] = 100.0
    leader = So101LeaderArm(port="/dev/null", robot_id="lead", teleop=fake)
    leader.connect()

    qpos, buttons = leader.get_info()

    assert qpos.shape == (6,)
    assert qpos[4] == pytest.approx(-np.pi / 2)
    assert qpos[5] == pytest.approx(1.0)
    assert buttons.shape == (0,)


def test_leader_calibration_hint_uses_teleop_flags():
    leader = So101LeaderArm(port="/dev/ttyACM1", robot_id="lead", teleop=FakeLeader(calibrated=False))

    with pytest.raises(RuntimeError, match="--teleop.type=so101_leader"):
        leader.connect()
