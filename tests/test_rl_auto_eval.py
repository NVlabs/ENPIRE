# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

import yaml

from enpire.policy.rl.auto_eval import AutoEvalController
from enpire.policy.rl.config import DataCollectionConfig
from enpire.policy.rl.state_machine import RLStateMachine


class _Env:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.last_episode_dir = None
        self.output_dirs = []
        self.reset_options = []
        self.finalize_calls = []
        output_dir.mkdir(parents=True, exist_ok=True)

    def set_output_dir(self, path):
        self.output_dir = Path(path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dirs.append(self.output_dir)

    def reset(self, *, options):
        self.reset_options.append(options)
        return {}, {}

    def finalize_episode(self, discard_episode=False, **_terminal_options):
        self.finalize_calls.append(discard_episode)


class _InitialPoseManager:
    def __init__(self, count=2):
        self.count = count
        self.selected = []

    def select_first(self):
        self.selected.append(1)

    def select_initial_position_index(self, index):
        self.selected.append(index)

    def build_center_pose(self):
        return {"right": {"position": [float(self.selected[-1]), 0.0, 0.0]}}


class _Policy:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _Handshake:
    def __init__(self):
        self.paths = []

    def set_path(self, path):
        self.paths.append(Path(path))


class _EventRouter:
    def __init__(self):
        self.reset_count = 0

    def reset_timer(self):
        self.reset_count += 1


def _ctx(tmp_path, *, episodes_per_hole=2, holes=2):
    cfg = DataCollectionConfig(
        task_name="pin_insertion",
        auto_eval_episodes_per_hole=episodes_per_hole,
        auto_eval_output_yaml=str(tmp_path / "stats"),
    )
    env = _Env(tmp_path / "pin_insertion" / "initial")
    return SimpleNamespace(
        cfg=cfg,
        env=env,
        initial_pose_manager=_InitialPoseManager(holes),
        policy_router=SimpleNamespace(rl_policy=_Policy()),
        handshake_server=_Handshake(),
        terminal_event=None,
        last_terminal_event=None,
        event_router=_EventRouter(),
        state_machine=RLStateMachine("learn"),
        external_event_queue=queue.Queue(),
    )


def test_auto_eval_starts_first_hole_and_rotates_output(tmp_path):
    ctx = _ctx(tmp_path)
    controller = AutoEvalController()

    controller.start(ctx)

    assert controller.state.active is True
    assert ctx.initial_pose_manager.selected == [1, 1]
    assert ctx.state_machine.state == "hover"
    assert list(controller.state.run_dirs) == ["hole_index_1"]
    assert ctx.handshake_server.paths == ctx.env.output_dirs
    assert ctx.env.reset_options[-1] == {
        "target_ee_pose": {"right": {"position": [1.0, 0.0, 0.0]}},
        "discard_episode": True,
    }


def test_auto_eval_counts_unique_saved_episodes_and_switches_holes(tmp_path):
    ctx = _ctx(tmp_path, episodes_per_hole=2, holes=2)
    controller = AutoEvalController()
    controller.start(ctx)
    first_run_dir = ctx.env.output_dir

    ctx.env.last_episode_dir = first_run_dir / "episode_1"
    controller.observe(ctx)
    controller.observe(ctx)

    assert controller.state.hole_index == 0
    assert controller.state.episodes_for_hole == 1

    ctx.env.last_episode_dir = first_run_dir / "episode_2"
    controller.observe(ctx)

    assert controller.state.hole_index == 1
    assert controller.state.episodes_for_hole == 0
    assert list(controller.state.run_dirs) == ["hole_index_1", "hole_index_2"]
    assert ctx.initial_pose_manager.selected[-1] == 2
    assert ctx.env.output_dir != first_run_dir


def test_auto_eval_writes_data_stats_yaml_after_last_hole(tmp_path):
    ctx = _ctx(tmp_path, episodes_per_hole=1, holes=1)
    controller = AutoEvalController()
    controller.start(ctx)
    run_dir = ctx.env.output_dir

    ctx.env.last_episode_dir = run_dir / "episode_1"
    controller.observe(ctx)

    assert controller.state.active is False
    assert controller.state.completed is True
    assert ctx.state_machine.state == "home"
    assert ctx.env.finalize_calls == [True]

    output_yaml = next((tmp_path / "stats").glob("pin_insertion_auto_eval_*.yaml"))
    payload = yaml.safe_load(output_yaml.read_text())
    assert payload == {"holes": {"hole_index_1": [str(run_dir)]}}
