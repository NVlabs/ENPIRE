# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from enpire.policy.rl.config import DataCollectionConfig
from enpire.policy.rl.handlers import do_home, do_hover


class _Env:
    def __init__(self):
        self.reset_options = []
        self.reset_obs = {}

    def reset(self, *, options):
        self.reset_options.append(options)
        return self.reset_obs, {}


class _InitialPoseManager:
    def __init__(self):
        self.center_pose = {"right": {"position": [1.0, 2.0, 3.0]}}
        self.episode_pose = {"right": {"position": [1.0, 2.0, 3.1]}}
        self.last_offset = np.zeros(3, dtype=np.float32)
        self.set_base_pose_calls = 0

    def build_center_pose(self):
        return self.center_pose

    def build_episode_start_pose(self):
        return self.episode_pose

    def pose_from_observation(self, obs, *, z_offset_m):
        return {"right": {"position": [9.0, 9.0, float(obs["right_ee_pos"][2] + z_offset_m)]}}

    def describe_current(self):
        return "Pose 1/1: right=[1. 2. 3.]"

    def set_base_pose_from_observation(self, obs):
        self.set_base_pose_calls += 1


def _home_obs(offset=0.0):
    return {
        "left_joint_pos": np.full(6, offset, dtype=np.float32),
        "left_gripper_pos": np.full(1, offset, dtype=np.float32),
        "right_joint_pos": np.full(6, offset, dtype=np.float32),
        "right_gripper_pos": np.full(1, offset, dtype=np.float32),
    }


def _ctx(obs):
    return SimpleNamespace(
        cfg=DataCollectionConfig(),
        obs=obs,
        env=_Env(),
        initial_pose_manager=_InitialPoseManager(),
        speech_announcer=SimpleNamespace(speak=lambda _: None),
        timing_log=SimpleNamespace(log=lambda *_, **__: None),
        policy_router=SimpleNamespace(rl_policy=SimpleNamespace(reset=lambda: None)),
        event_router=SimpleNamespace(reset_timer=lambda: None),
        terminal_event="timeout",
        state_machine=SimpleNamespace(state="home"),
    )


def test_do_home_skips_pause_reset_when_observation_is_already_home():
    ctx = _ctx(_home_obs(offset=0.19))

    do_home(ctx)

    assert ctx.env.reset_options == [{"alias": "home", "discard_episode": True}]
    assert ctx.terminal_event is None
    assert ctx.state_machine.state == "idle"


def test_do_home_uses_pause_reset_when_observation_is_not_home():
    ctx = _ctx(_home_obs(offset=0.21))

    do_home(ctx)

    assert ctx.env.reset_options == [
        {
            "target_ee_pose": ctx.initial_pose_manager.center_pose,
            "discard_episode": True,
        },
        {"alias": "home", "discard_episode": True},
    ]
    assert ctx.terminal_event is None
    assert ctx.state_machine.state == "idle"


def test_do_home_uses_pause_reset_when_home_cannot_be_detected():
    ctx = _ctx({})

    do_home(ctx)

    assert ctx.env.reset_options == [
        {
            "target_ee_pose": ctx.initial_pose_manager.center_pose,
            "discard_episode": True,
        },
        {"alias": "home", "discard_episode": True},
    ]


def test_hover_reset_lifts_before_returning_to_hover_and_preserves_anchor():
    ctx = _ctx(
        {
            "right_ee_pos": np.array([0.0, 0.0, 0.82], dtype=np.float32),
            "right_ee_rot6d": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
        }
    )
    ctx.cfg = DataCollectionConfig(
        task_name="gpu_insertion",
        episode_reset_lift_m=0.03,
    )
    ctx.terminal_event = "success"
    ctx.state_machine.state = "hover"

    do_hover(ctx)

    assert ctx.env.reset_options[0]["target_ee_pose"]["right"]["position"] == pytest.approx(
        [9.0, 9.0, 0.85]
    )
    assert ctx.env.reset_options == [
        {
            "target_ee_pose": ctx.env.reset_options[0]["target_ee_pose"],
            "episode_terminal_event": "success",
            "episode_terminal_reward": 1.0,
            "episode_terminal_done": True,
        },
        {
            "target_ee_pose": ctx.initial_pose_manager.episode_pose,
            "task_name": "gpu_insertion",
            "start_new_episode": True,
        },
    ]
    assert ctx.initial_pose_manager.set_base_pose_calls == 0
    assert ctx.terminal_event is None
    assert ctx.state_machine.state == "learn"


def test_gpu_hover_reset_relocalizes_then_starts_episode_at_current_pose(monkeypatch):
    ctx = _ctx(
        {
            "right_ee_pos": np.array([0.0, 0.0, 0.82], dtype=np.float32),
            "right_ee_rot6d": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
        }
    )
    ctx.cfg = DataCollectionConfig(
        task_name="gpu_insertion",
        episode_reset_lift_m=0.03,
        episode_reset_strategy="gpu_slot_hover",
        initial_pose_source="current_observation",
    )
    ctx.env.reset_obs = {
        "right_ee_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "right_ee_rot6d": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
    }
    calls = []
    monkeypatch.setattr(
        "enpire.policy.rl.handlers.check_gpu_slot_hover_dependencies",
        lambda cfg: None,
    )
    monkeypatch.setattr(
        "enpire.policy.rl.handlers.move_to_gpu_slot_hover",
        lambda ctx: calls.append(ctx.cfg.task_name),
    )
    ctx.terminal_event = "success"
    ctx.state_machine.state = "hover"

    do_hover(ctx)

    assert calls == ["gpu_insertion"]
    assert ctx.env.reset_options[0]["target_ee_pose"]["right"]["position"] == pytest.approx(
        [9.0, 9.0, 0.85]
    )
    assert ctx.env.reset_options == [
        {
            "target_ee_pose": ctx.env.reset_options[0]["target_ee_pose"],
            "episode_terminal_event": "success",
            "episode_terminal_reward": 1.0,
            "episode_terminal_done": True,
        },
        {
            "alias": "current",
            "task_name": "gpu_insertion",
            "start_new_episode": True,
        },
    ]
    assert ctx.initial_pose_manager.set_base_pose_calls == 1
    assert ctx.terminal_event is None
    assert ctx.state_machine.state == "learn"


def test_gpu_hover_reset_can_randomize_start_after_relocalization(monkeypatch):
    ctx = _ctx(
        {
            "right_ee_pos": np.array([0.0, 0.0, 0.82], dtype=np.float32),
            "right_ee_rot6d": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
        }
    )
    ctx.cfg = DataCollectionConfig(
        task_name="gpu_insertion",
        episode_reset_strategy="gpu_slot_hover",
        initial_pose_source="current_observation",
        randomize_initial_pose=True,
        x_init_lim=(-0.003, 0.003),
        y_init_lim=(-0.003, 0.003),
        z_init_lim=(0.0, 0.003),
    )
    ctx.env.reset_obs = {
        "right_ee_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "right_ee_rot6d": np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
    }
    calls = []
    monkeypatch.setattr(
        "enpire.policy.rl.handlers.check_gpu_slot_hover_dependencies",
        lambda cfg: None,
    )
    monkeypatch.setattr(
        "enpire.policy.rl.handlers.move_to_gpu_slot_hover",
        lambda ctx: calls.append(ctx.cfg.task_name),
    )
    ctx.terminal_event = None
    ctx.state_machine.state = "hover"

    do_hover(ctx)

    assert calls == ["gpu_insertion"]
    assert ctx.env.reset_options == [
        {"alias": "current"},
        {
            "target_ee_pose": ctx.initial_pose_manager.episode_pose,
            "task_name": "gpu_insertion",
            "start_new_episode": True,
        },
    ]
    assert ctx.initial_pose_manager.set_base_pose_calls == 2
    assert ctx.terminal_event is None
    assert ctx.state_machine.state == "learn"
