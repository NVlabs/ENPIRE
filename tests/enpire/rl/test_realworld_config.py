# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from enpire.policy.rl.config import load_yaml_defaults

ROOT = Path(__file__).resolve().parents[3]
TASKS = ROOT / "enpire/env/forge/tmux/realworld_rl/tasks_config"


def test_pin_config_uses_external_station_artifacts(monkeypatch, tmp_path) -> None:
    poses = tmp_path / "poses.yaml"
    reward = tmp_path / "reward.yaml"
    poses.write_text("position_1: {}\n", encoding="utf-8")
    reward.write_text("auto_reward_z_threshold: 0.8\n", encoding="utf-8")
    monkeypatch.setenv("ENPIRE_RL_INITIAL_POSITIONS", str(poses))
    monkeypatch.setenv("ENPIRE_RL_REWARD_CONFIG", str(reward))
    monkeypatch.setenv("ENPIRE_YAM_STATION", "test")
    cfg = load_yaml_defaults(str(TASKS / "pin_insertion/pin_insertion.yaml"))
    assert cfg.task_name == "pin_insertion"
    assert cfg.initial_positions_file == str(poses)
    assert cfg.reward_config_file == str(reward)
    assert cfg.enable_auto_reward is True
    assert cfg.station == "test"


def test_gpu_config_uses_current_pose_reset() -> None:
    cfg = load_yaml_defaults(str(TASKS / "gpu_insertion/gpu_insertion.yaml"))
    assert cfg.task_name == "gpu_insertion"
    assert cfg.initial_pose_source == "current_observation"
    assert cfg.episode_reset_strategy == "gpu_slot_hover"
    assert cfg.enabled_camera_names == ("top", "left_third", "left_wrist")


def test_pusht_config_does_not_embed_station_pose() -> None:
    import yaml

    data = yaml.safe_load((TASKS / "pusht/pusht.yaml").read_text(encoding="utf-8"))
    assert data["task_name"] == "pusht"
    assert data["initial_positions_file"] is None
    assert data["pusht_reward_goal_image"] == ""
