# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import deque

import gymnasium as gym
import numpy as np


class RecordPastNStatisticsWrapper(gym.Wrapper):
    """
    Tracks the mean of the most recent N episodes for return, length, and time.

    This wrapper works in conjunction with RecordEpisodeStatistics. It assumes
    that RecordEpisodeStatistics wraps the environment *before* this wrapper.
    It reads the 'episode' dictionary from the info returned by the final step
    of an episode and calculates the moving average of 'r', 'l', and 't'
    over the last `n` episodes.

    Args:
        env (gym.Env): The environment to wrap.
        n (int): The number of past episodes to calculate the average over.
        stats_prefix (str): The prefix to use for the logged average statistics.
                            Defaults to "episode_stats_avg".
    """

    def __init__(self, env: gym.Env, n: int = 10, stats_prefix: str = "episode_stats_avg"):
        super().__init__(env)
        self.n = n
        self.stats_prefix = stats_prefix
        self.returns_deque = deque(maxlen=n)
        self.lengths_deque = deque(maxlen=n)
        self.times_deque = deque(maxlen=n)

    def step(self, action):
        """
        Steps through the environment, recording episode statistics if done.

        Args:
            action: The action to take.

        Returns:
            A tuple containing observation, reward, terminated, truncated, and info.
            The info dictionary will contain the moving averages of episode
            statistics if the episode ended.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            episode_stats = info.get("episode")
            if episode_stats:
                self.returns_deque.append(episode_stats["r"])
                self.lengths_deque.append(episode_stats["l"])
                self.times_deque.append(episode_stats["t"])

                avg_r = np.mean(self.returns_deque)
                avg_l = np.mean(self.lengths_deque)
                avg_t = np.mean(self.times_deque)

                info[f"{self.stats_prefix}/avg_return_past_{self.n}"] = avg_r
                info[f"{self.stats_prefix}/avg_length_past_{self.n}"] = avg_l
                info[f"{self.stats_prefix}/avg_time_past_{self.n}"] = avg_t
            else:
                # Warn if RecordEpisodeStatistics might be missing or incorrectly placed
                print(
                    "Warning: 'episode' key not found in info dict. "
                    "Ensure RecordEpisodeStatistics wraps the environment "
                    "before PastNStatisticsWrapper."
                )

        return observation, reward, terminated, truncated, info

    # We don't need to override reset as the statistics are only calculated at step time.

