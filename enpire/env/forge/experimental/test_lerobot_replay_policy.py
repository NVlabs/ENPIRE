# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from enpire.env.forge.experimental.lerobot_replay_policy import LerobotReplayPolicy


def test_update_replay_config_rolls_back_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "actions.npy"
    original_actions = np.arange(42, dtype=np.float32).reshape(3, 14)
    np.save(dataset_path, original_actions)

    policy = LerobotReplayPolicy(dataset_path=dataset_path, control_mode="joint_position")
    policy.current_step = 1

    def _failing_reload(self: LerobotReplayPolicy) -> None:
        raise FileNotFoundError("missing norm stats")

    monkeypatch.setattr(LerobotReplayPolicy, "_load_episode_data", _failing_reload)

    with pytest.raises(FileNotFoundError, match="missing norm stats"):
        policy.update_replay_config(control_mode="umi_ee_pose", norm_stats_path="")

    assert policy.control_mode == "joint_position"
    assert policy.norm_stats_path is None
    assert policy.current_step == 1
    np.testing.assert_array_equal(policy.actions, original_actions)

    action, info = policy.get_action({})
    np.testing.assert_array_equal(action["left_joint_pos"], original_actions[1, 0:6])
    assert info["current_step"] == 1


def test_umi_initial_ee_uses_abs_targets_when_states_absent() -> None:
    policy = LerobotReplayPolicy(dataset_path="", control_mode="umi_ee_pose")
    abs_targets = np.arange(32, dtype=np.float32).reshape(2, 16)
    policy._umi_states = None
    policy._umi_abs_targets = abs_targets

    np.testing.assert_array_equal(policy.umi_initial_ee, abs_targets[0])
