# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from enpire.policy.rl.record_episode_wrapper import RecordEpisodeWrapper


def _obs(step: int) -> dict[str, np.ndarray]:
    return {
        "left_joint_pos": np.full(6, step, dtype=np.float32),
        "left_gripper_pos": np.array([0.1 * step], dtype=np.float32),
        "right_joint_pos": np.full(6, -step, dtype=np.float32),
        "right_gripper_pos": np.array([1.0 - 0.1 * step], dtype=np.float32),
    }


def _action() -> dict[str, np.ndarray | str]:
    return {
        "left_joint_pos": np.zeros(6, dtype=np.float32),
        "left_gripper_pos": np.zeros(1, dtype=np.float32),
        "right_joint_pos": np.zeros(6, dtype=np.float32),
        "right_gripper_pos": np.zeros(1, dtype=np.float32),
        "source": "rl",
    }


class _DummyYamEnv(gym.Env):
    policy_control_freq = 30.0
    control_period = 1.0 / policy_control_freq

    def __init__(self) -> None:
        super().__init__()
        self.step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        return _obs(self.step_count), {}

    def step(self, action: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        return _obs(self.step_count), 0.0, False, False, {}


def test_record_episode_writes_terminal_reward_and_dones(tmp_path) -> None:
    env = RecordEpisodeWrapper(_DummyYamEnv(), output_dir=str(tmp_path))
    try:
        env.reset(options={"start_new_episode": True, "task_name": "task"})
        for _ in range(3):
            env.step(_action())

        env.reset(
            options={
                "start_new_episode": False,
                "episode_terminal_event": "success",
                "episode_terminal_reward": 1.0,
                "episode_terminal_done": True,
            }
        )

        episode_dir = env.last_episode_dir
        assert episode_dir is not None
        env.wait_for_pending_saves()  # save is async; block until files are on disk
        rewards = np.load(episode_dir / "reward.npy")
        dones = np.load(episode_dir / "dones.npy")
        with np.load(episode_dir / "reward.npz") as labels:
            np.testing.assert_array_equal(labels["rewards"], rewards)
            np.testing.assert_array_equal(labels["dones"], dones)

        np.testing.assert_array_equal(rewards, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(dones, np.array([False, False, True]))
    finally:
        env.close()


def test_record_episode_marks_final_step_done_without_terminal_reward(tmp_path) -> None:
    env = RecordEpisodeWrapper(_DummyYamEnv(), output_dir=str(tmp_path))
    try:
        env.reset(options={"start_new_episode": True, "task_name": "task"})
        for _ in range(2):
            env.step(_action())

        env.reset(options={"start_new_episode": False})

        episode_dir = env.last_episode_dir
        assert episode_dir is not None
        env.wait_for_pending_saves()  # save is async; block until files are on disk
        rewards = np.load(episode_dir / "reward.npy")
        dones = np.load(episode_dir / "dones.npy")

        np.testing.assert_array_equal(rewards, np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(dones, np.array([False, True]))
    finally:
        env.close()

