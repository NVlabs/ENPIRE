# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from enpire.policy.rl.config import load_yaml_defaults

ROOT = Path(__file__).resolve().parents[3]
TASKS = ROOT / "enpire/env/forge/tmux/realworld_rl/tasks_config"


def test_pusht_config_does_not_embed_station_pose() -> None:
    import yaml

    data = yaml.safe_load((TASKS / "pusht/pusht.yaml").read_text(encoding="utf-8"))
    assert data["task_name"] == "pusht"
    assert data["initial_positions_file"] is None
    assert data["pusht_reward_goal_image"] == ""
