# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from enpire.env.forge.robot.fello import fello_teleop_policy as fello_module


IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _observation() -> dict[str, np.ndarray]:
    return {
        "left_joint_pos": np.arange(6, dtype=np.float32),
        "left_gripper_pos": np.array([0.25], dtype=np.float32),
        "right_joint_pos": np.arange(6, 12, dtype=np.float32),
        "right_gripper_pos": np.array([0.75], dtype=np.float32),
    }


def _defaults_from_obs(
    observation: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    if observation is None:
        return {
            "left_joint_pos": np.zeros(6, dtype=np.float32),
            "left_gripper_pos": np.array([1.0], dtype=np.float32),
            "right_joint_pos": np.zeros(6, dtype=np.float32),
            "right_gripper_pos": np.array([1.0], dtype=np.float32),
        }
    return {
        "left_joint_pos": observation["left_joint_pos"],
        "left_gripper_pos": observation["left_gripper_pos"],
        "right_joint_pos": observation["right_joint_pos"],
        "right_gripper_pos": observation["right_gripper_pos"],
    }


def _install_fake_fello_policy(monkeypatch):
    instances = {}

    class FakeFelloTeleopPolicy:
        def __init__(
            self,
            *,
            target_side,
            action_type,
            scaled_control,
            scaled_control_xyz_scale,
            delta_ee_translation_xyz_max,
            decouple_translation,
            takeover_button,
        ):
            del scaled_control_xyz_scale, delta_ee_translation_xyz_max
            self.target_side = target_side
            self.action_type = action_type
            self.calls: list[str] = []
            instances[target_side] = self

        def _defaults_from_obs(self, observation):
            return _defaults_from_obs(observation)

        def get_action(self, observation, options=None):
            self.calls.append("get_action")
            del options
            defaults = _defaults_from_obs(observation)
            if self.action_type == "delta_eef":
                action = fello_module.DualFelloPolicy._zero_delta_eef_action(defaults)
                action[f"{self.target_side}_ee_pos"] = np.full(
                    3, 2.0 if self.target_side == "right" else -2.0, dtype=np.float32
                )
                action[f"{self.target_side}_ee_rot6d"] = IDENTITY_ROT6D.copy()
                action[f"{self.target_side}_gripper_pos"] = np.array(
                    [0.9 if self.target_side == "right" else 0.1],
                    dtype=np.float32,
                )
            else:
                action = defaults.copy()
                action[f"{self.target_side}_joint_pos"] = np.full(
                    6, 2.0 if self.target_side == "right" else -2.0, dtype=np.float32
                )
                action[f"{self.target_side}_gripper_pos"] = np.array(
                    [0.9 if self.target_side == "right" else 0.1],
                    dtype=np.float32,
                )
            info = {
                "buttons": np.array(
                    [1.0, 0.0, 0.0] if self.target_side == "right" else [0.0, 1.0, 0.0],
                    dtype=np.float32,
                )
            }
            if self.target_side == "right":
                info["right_button_states"] = {
                    "start": True,
                    "pause": False,
                    "home": False,
                }
            return action, info

        def poll_button_events(self, observation=None):
            del observation
            self.calls.append("poll_button_events")
            return {"start_pressed": True}

        def slow_home(self, observation, duration_s=2.0, steps=40):
            del observation, duration_s, steps
            self.calls.append("slow_home")

        def hold_current_position(self):
            self.calls.append("hold_current_position")

        def set_external_takeover_pressed(self, pressed):
            del pressed
            self.calls.append("set_external_takeover_pressed")

        def reset(self):
            self.calls.append("reset")
            return {
                "initial_state": fello_module.DualFelloPolicy._zero_delta_eef_action(
                    _defaults_from_obs(None)
                )
            }

    monkeypatch.setattr(fello_module, "FelloTeleopPolicy", FakeFelloTeleopPolicy)
    return instances


def _make_dual_policy(enabled_sides: str):
    return fello_module.DualFelloPolicy(
        action_type="delta_eef",
        scaled_control=True,
        scaled_control_xyz_scale=(0.25, 0.25, 1.0),
        delta_ee_translation_xyz_max=(0.0003, 0.0003, 0.0006),
        enabled_sides=enabled_sides,
        decouple_translation=True,
        takeover_button=(1, 0),
    )


def test_dual_fello_policy_right_only_never_constructs_or_polls_left(monkeypatch):
    instances = _install_fake_fello_policy(monkeypatch)
    policy = _make_dual_policy("right")

    assert set(instances) == {"right"}

    action, info = policy.get_action(_observation())
    assert instances["right"].calls == ["get_action"]
    np.testing.assert_allclose(action["right_ee_pos"], np.full(3, 2.0))
    np.testing.assert_allclose(action["left_ee_pos"], np.zeros(3))
    np.testing.assert_allclose(action["left_ee_rot6d"], IDENTITY_ROT6D)
    np.testing.assert_allclose(action["left_gripper_pos"], np.array([0.25]))
    np.testing.assert_allclose(info["right_buttons"], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(info["left_buttons"], np.zeros(3))

    policy.poll_button_events(_observation())
    policy.slow_home(_observation(), duration_s=0.0, steps=1)
    policy.hold_current_position()
    policy.reset()

    assert instances["right"].calls == [
        "get_action",
        "poll_button_events",
        "slow_home",
        "hold_current_position",
        "set_external_takeover_pressed",
        "reset",
    ]
    assert set(instances) == {"right"}


def test_dual_fello_policy_left_only_never_constructs_or_polls_right(monkeypatch):
    instances = _install_fake_fello_policy(monkeypatch)
    policy = _make_dual_policy("left")

    assert set(instances) == {"left"}

    action, info = policy.get_action(_observation())
    assert instances["left"].calls == ["get_action"]
    np.testing.assert_allclose(action["left_ee_pos"], np.full(3, -2.0))
    np.testing.assert_allclose(action["right_ee_pos"], np.zeros(3))
    np.testing.assert_allclose(action["right_ee_rot6d"], IDENTITY_ROT6D)
    np.testing.assert_allclose(action["right_gripper_pos"], np.array([0.75]))
    np.testing.assert_allclose(info["left_buttons"], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(info["right_buttons"], np.zeros(3))
    assert "right_button_states" not in info

    policy.poll_button_events(_observation())
    policy.slow_home(_observation(), duration_s=0.0, steps=1)
    policy.hold_current_position()
    policy.reset()

    assert instances["left"].calls == [
        "get_action",
        "poll_button_events",
        "slow_home",
        "hold_current_position",
        "set_external_takeover_pressed",
        "reset",
    ]
    assert set(instances) == {"left"}


def _teleop_config(nominal_position=None):
    if nominal_position is None:
        nominal_position = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    return {
        "teleop": {
            "host": "localhost",
            "print_interval_s": 0.1,
            "align_interval_s": 0.0,
            "align_lpf_alpha": 1.0,
        },
        "control": {
            "kp": [3.0, 10.0, 10.0, 3.0, 3.0, 3.0, 1.5],
            "kd": [0.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.15],
        },
        "scaled_control": {
            "active_button": 1,
            "nominal_position": nominal_position,
            "rpy_scale": [1.0, 1.0, 1.0],
        },
        "hardware": {"footswitch": {"button_mode": "recording"}},
        "mapping": {"arm_index_map": [0, 1, 2, 3, 4, 5], "gripper_index_map": 6},
    }


def _install_fake_teleop_dependencies(monkeypatch, cfg):
    leaders = []

    class FakeLeaderClient:
        def __init__(self, host, port):
            del host, port
            self.qpos = np.arange(7, dtype=np.float32)
            self.buttons = np.zeros(3, dtype=np.float32)
            self.commanded_joint_pos = []
            self.modes = []
            leaders.append(self)

        def get_info(self):
            return self.qpos.copy(), self.buttons.copy()

        def command_joint_pos(self, joint_state):
            self.commanded_joint_pos.append(
                {key: np.asarray(value).copy() for key, value in joint_state.items()}
            )

        def set_mode(self, mode):
            self.modes.append(mode)

    class FakeKinematics:
        def __init__(self, side):
            del side

        def forward_kinematics(self, qpos):
            qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
            return (
                qpos[:3].copy(),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )

    monkeypatch.setattr(fello_module, "load_fello_config", lambda side=None: cfg)
    monkeypatch.setattr(fello_module, "FelloLeaderClient", FakeLeaderClient)
    monkeypatch.setattr(fello_module, "FelloKinematics", FakeKinematics)
    return leaders


def test_scaled_delta_pressed_only_commands_fello_server(monkeypatch):
    nominal = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float32)
    cfg = _teleop_config(nominal.tolist())
    leaders = _install_fake_teleop_dependencies(monkeypatch, cfg)
    policy = fello_module.FelloTeleopPolicy(
        target_side="right",
        action_type="delta_eef",
        scaled_control=True,
        scaled_control_xyz_scale=(0.25, 0.25, 1.0),
        delta_ee_translation_xyz_max=(0.0003, 0.0003, 0.0006),
        decouple_translation=True,
        takeover_button=0,
    )
    leader = leaders[0]
    leader.qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    leader.buttons = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    policy.get_action(_observation())
    leader.commanded_joint_pos.clear()
    leader.modes.clear()

    leader.qpos = np.array([0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6], dtype=np.float32)
    leader.buttons = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    action, info = policy.get_action(_observation())

    np.testing.assert_allclose(
        action["right_ee_pos"], np.array([0.0, 0.0001, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(action["right_ee_rot6d"], IDENTITY_ROT6D)
    np.testing.assert_allclose(action["right_gripper_pos"], np.array([0.6]))
    np.testing.assert_allclose(info["buttons"], np.array([1.0, 1.0, 0.0]))
    assert leader.modes == ["gravity", "position"]
    assert len(leader.commanded_joint_pos) == 1
    np.testing.assert_allclose(leader.commanded_joint_pos[0]["pos"], nominal)
    np.testing.assert_allclose(leader.commanded_joint_pos[0]["vel"], np.zeros(7))
    np.testing.assert_allclose(
        leader.commanded_joint_pos[0]["kp"], cfg["control"]["kp"]
    )
    np.testing.assert_allclose(
        leader.commanded_joint_pos[0]["kd"], cfg["control"]["kd"]
    )


def test_scaled_control_nominal_position_must_be_7d(monkeypatch):
    cfg = _teleop_config([0.0] * 6)
    _install_fake_teleop_dependencies(monkeypatch, cfg)

    try:
        fello_module.FelloTeleopPolicy(
            target_side="right",
            action_type="delta_eef",
            scaled_control=True,
            scaled_control_xyz_scale=(0.25, 0.25, 1.0),
            delta_ee_translation_xyz_max=(0.0003, 0.0003, 0.0006),
            decouple_translation=True,
            takeover_button=0,
        )
    except ValueError as exc:
        assert "scaled_control.nominal_position must have 7 finite values" in str(exc)
    else:
        raise AssertionError("Expected invalid nominal position to raise ValueError")
