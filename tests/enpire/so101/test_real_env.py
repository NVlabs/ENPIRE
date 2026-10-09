# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from enpire.env.forge.cap.env.real_so101 import RealSo101Env
from enpire.env.forge.robot.so101.constants import SO101_HOME_JOINT_POS


class FakeArm:
    def __init__(self):
        self.q = np.zeros(6)
        self.commands = []

    def get_joint_pos(self):
        return self.q.copy()

    def get_observations(self):
        return {"joint_pos": self.q[:5].copy(), "gripper_pos": self.q[5:].copy()}

    def command_joint_pos(self, joint_pos):
        self.commands.append(np.asarray(joint_pos).copy())
        self.q = np.asarray(joint_pos).copy()

    def command_joint_state(self, state):
        self.command_joint_pos(state["pos"])


class FakeKin:
    def forward_kinematics(self, joint_pos):
        return np.asarray(joint_pos[:3]), np.array([0.0, 0.0, 0.0, 1.0])


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    return RealSo101Env(arm=FakeArm(), kin=FakeKin())


def test_observations_include_fk(env):
    env._arm.q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    obs = env.get_observations("left")

    np.testing.assert_allclose(obs["ee_pos"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(obs["gripper_pos"], [0.6])


def test_wrong_side_is_rejected(env):
    with pytest.raises(ValueError, match="single arm"):
        env.get_observations("right")


def test_go_home_interpolates_within_velocity_limit(env):
    env.go_home(max_joint_vel=0.5)

    commands = np.array(env._arm.commands)
    np.testing.assert_allclose(commands[-1][:5], SO101_HOME_JOINT_POS)
    step = np.max(np.abs(np.diff(np.vstack([np.zeros(6), commands]), axis=0))[:, :5])
    assert step <= 0.5 / env.profile.control_freq_hz + 1e-9


def test_set_gripper_holds_last_commanded_arm_target(env):
    target = np.array([0.2, -1.0, 0.8, 0.5, 0.0, 1.0])
    env.command_joint_pos("left", target)
    env._arm.q[:5] -= 0.05

    env.set_gripper("left", 0.0)

    np.testing.assert_allclose(env._arm.commands[-1], [0.2, -1.0, 0.8, 0.5, 0.0, 0.0])


def test_set_gripper_clips_value(env):
    env.set_gripper("left", 3.0)

    assert env._arm.commands[-1][5] == 1.0
