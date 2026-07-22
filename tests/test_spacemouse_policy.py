# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from enpire.env.forge.robot.spacemouse.spacemouse_policy import SpaceMouseState, SpaceMouseTeleopPolicy
from enpire.policy.rl.policy import PolicyRouter, _stub_fello_action, _stub_fello_info


class _FakeReader:
    description = "fake"

    def __init__(self, state):
        self.state = state
        self.closed = False

    def get_state(self):
        return self.state

    def close(self):
        self.closed = True


class _FakeRL:
    def get_action(self, obs):
        return (
            {
                "left_ee_pos": np.array([0.1, 0.0, 0.0], dtype=np.float32),
                "left_ee_rot6d": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
                "left_gripper_pos": obs["left_gripper_pos"].copy(),
                "right_ee_pos": np.zeros(3, dtype=np.float32),
                "right_ee_rot6d": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
                "right_gripper_pos": obs["right_gripper_pos"].copy(),
            },
            {"action_source": "rl"},
        )


def _obs():
    return {
        "left_gripper_pos": np.array([0.2], dtype=np.float32),
        "right_gripper_pos": np.array([0.8], dtype=np.float32),
    }


def _policy(state, **kwargs):
    return SpaceMouseTeleopPolicy(
        enabled_sides="left",
        control_side="auto",
        delta_ee_translation_xyz_max=(0.001, 0.002, 0.003),
        xyz_scale=(1.0, 1.0, 1.0),
        deadzone=0.05,
        axis_order=("y", "x", "z", "pitch", "roll", "yaw"),
        axis_signs=(1.0, 1.0, -1.0, 1.0, -1.0, -1.0),
        require_takeover_button=False,
        takeover_button=None,
        open_gripper_button=None,
        close_gripper_button=None,
        gripper_open_pos=1.0,
        gripper_close_pos=0.0,
        reader=_FakeReader(state),
        **kwargs,
    )


def test_spacemouse_default_mapping_matches_reference_order():
    state = SpaceMouseState(
        axes={
            "x": 0.25,
            "y": -0.5,
            "z": 0.75,
            "pitch": 0.1,
            "roll": -0.2,
            "yaw": 0.3,
        },
        buttons=(0, 0),
        age_s=0.0,
    )
    action, info = _policy(state).get_action(_obs())

    assert info["active"] is True
    np.testing.assert_allclose(
        action["left_ee_pos"],
        np.array([-0.0005, 0.0005, -0.00225], dtype=np.float32),
    )
    np.testing.assert_allclose(action["right_ee_pos"], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(action["left_gripper_pos"], np.array([0.2], dtype=np.float32))


def test_spacemouse_deadzone_holds_action_inactive():
    state = SpaceMouseState(
        axes={"x": 0.01, "y": 0.01, "z": 0.01, "pitch": 0.0, "roll": 0.0, "yaw": 0.0},
        buttons=(0, 0),
        age_s=0.0,
    )
    action, info = _policy(state).get_action(_obs())

    assert info["active"] is False
    np.testing.assert_allclose(action["left_ee_pos"], np.zeros(3, dtype=np.float32))


def test_spacemouse_buttons_are_edge_triggered():
    state = SpaceMouseState(
        axes={"x": 0.0, "y": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0, "yaw": 0.0},
        buttons=(1, 0),
        age_s=0.0,
    )
    policy = _policy(state)

    _, first_info = policy.get_action(_obs())
    _, second_info = policy.get_action(_obs())

    assert first_info["just_pressed_buttons"] == (0,)
    assert second_info["just_pressed_buttons"] == ()


def test_policy_router_uses_spacemouse_intervention_before_rl():
    state = SpaceMouseState(
        axes={"x": 1.0, "y": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0, "yaw": 0.0},
        buttons=(0, 0),
        age_s=0.0,
    )
    router = PolicyRouter(
        fello_policy=None,
        spacemouse_policy=_policy(state),
        rl_policy=_FakeRL(),
        enabled_sides="left",
        z_up_step_m=0.001,
    )
    action, _ = router.route_action(
        _obs(),
        _stub_fello_action(_obs()),
        _stub_fello_info(),
        use_rl=True,
        keyboard_info={"z": False},
    )

    assert action["source"] == "human"
    np.testing.assert_allclose(action["left_ee_pos"], np.array([0.0, 0.002, 0.0]))
