# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
from typing import Iterable, Optional, Union

import gymnasium as gym
import jax
import numpy as np
from absl import flags
from flax.core import frozen_dict

from serl_launcher.common.env_common import calc_return_to_go
from serl_launcher.data_andy_fix.dataset import Dataset, DatasetDict

FLAGS = flags.FLAGS


def _sample(dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


def _init_replay_dict(obs_space: gym.Space, capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        include_next_actions: Optional[bool] = False,
        include_label: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
        include_mc_returns: Optional[bool] = False,
        include_base_action: Optional[bool] = False,
        include_next_base_action: Optional[bool] = False,
        discount: Optional[float] = None,
        reward_scale: Optional[float] = 1,
        reward_bias: Optional[float] = 0,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        if include_next_actions:
            dataset_dict["next_actions"] = np.empty(
                (capacity, *action_space.shape), dtype=action_space.dtype
            )
            dataset_dict["next_intvn"] = np.empty((capacity,), dtype=bool)

        if include_label:
            dataset_dict["labels"] = np.empty((capacity,), dtype=int)

        if include_grasp_penalty:
            dataset_dict["grasp_penalty"] = np.empty((capacity,), dtype=np.float32)

        if include_base_action:
            dataset_dict["base_actions"] = np.empty(
                (capacity, *action_space.shape), dtype=action_space.dtype
            )

        if include_next_base_action:
            dataset_dict["next_base_actions"] = np.empty(
                (capacity, *action_space.shape), dtype=action_space.dtype
            )

        if include_mc_returns:
            assert discount is not None
            dataset_dict["mc_returns"] = np.empty((capacity,), dtype=np.float32)
            self._allow_idxs = []
            self._traj_start_idx = 0
            self.reward_scale = reward_scale
            self.reward_bias = reward_bias
            self.discount = discount
            # Track trajectory boundaries: list of (start_idx, end_idx) tuples
            self._trajectory_boundaries = []

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._include_mc_returns = include_mc_returns

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        data_dict["mc_returns"] = None

        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)
        if "dones" not in data_dict:
            data_dict["dones"] = 1 - data_dict["masks"]
        if self._include_mc_returns and data_dict["dones"] == 1.0:
            # Handle wrap-around for trajectory data collection
            if self._traj_start_idx <= self._insert_index:
                # Normal case: trajectory doesn't wrap around
                rewards = self.dataset_dict["rewards"][
                    self._traj_start_idx : self._insert_index + 1
                ]
                masks = self.dataset_dict["masks"][
                    self._traj_start_idx : self._insert_index + 1
                ]
                indices_range = list(range(self._traj_start_idx, self._insert_index + 1))
            else:
                # Wrap-around case: trajectory wraps from end to beginning of buffer
                rewards = np.concatenate(
                    [
                        self.dataset_dict["rewards"][self._traj_start_idx :],
                        self.dataset_dict["rewards"][: self._insert_index + 1],
                    ]
                )
                masks = np.concatenate(
                    [
                        self.dataset_dict["masks"][self._traj_start_idx :],
                        self.dataset_dict["masks"][: self._insert_index + 1],
                    ]
                )
                indices_range = list(range(self._traj_start_idx, self._capacity)) + list(
                    range(0, self._insert_index + 1)
                )

            # Calculate MC returns
            mc_returns = calc_return_to_go(
                self.env_name,
                rewards,
                masks,
                self.discount,
                self.reward_scale,
                self.reward_bias,
            )

            # Store MC returns
            for i, idx in enumerate(indices_range):
                self.dataset_dict["mc_returns"][idx] = mc_returns[i]

            # Add indices to allow_idxs
            self._allow_idxs.extend(indices_range)

            # Track this trajectory's boundaries
            self._trajectory_boundaries.append((self._traj_start_idx, self._insert_index))

            # Clean up any trajectories that have been overwritten
            if self._size == self._capacity:
                self._cleanup_overwritten_trajectories()

            self._traj_start_idx = self._insert_index + 1

        # Update insert index and size
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def _cleanup_overwritten_trajectories(self):
        """Remove trajectories that have been overwritten by the newest trajectory."""
        # Get the indices of the newest trajectory
        newest_start, newest_end = self._trajectory_boundaries[-1]

        if newest_start <= newest_end:
            # Normal trajectory
            newest_indices = set(range(newest_start, newest_end + 1))
        else:
            # Wrap-around trajectory
            newest_indices = set(range(newest_start, self._capacity)) | set(
                range(0, newest_end + 1)
            )

        # Find and remove trajectories that overlap with the newest trajectory
        trajectories_to_keep = []
        removed_indices = set()

        for i, (start_idx, end_idx) in enumerate(
            self._trajectory_boundaries[:-1]
        ):  # Exclude the newest
            if start_idx <= end_idx:
                # Normal trajectory
                traj_indices = set(range(start_idx, end_idx + 1))
            else:
                # Wrap-around trajectory
                traj_indices = set(range(start_idx, self._capacity)) | set(
                    range(0, end_idx + 1)
                )

            # Check if this trajectory overlaps with the newest one
            if traj_indices.intersection(newest_indices):
                # This trajectory was overwritten
                removed_indices.update(traj_indices)
            else:
                trajectories_to_keep.append((start_idx, end_idx))

        # Add back the newest trajectory
        trajectories_to_keep.append(self._trajectory_boundaries[-1])
        self._trajectory_boundaries = trajectories_to_keep

        # Remove overwritten indices from _allow_idxs (but keep the newest trajectory's indices)
        self._allow_idxs = [
            idx
            for idx in self._allow_idxs
            if idx not in removed_indices or idx in newest_indices
        ]

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device=None):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data, device=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> frozen_dict.FrozenDict:
        if indx is None:
            if self._include_mc_returns:
                indx = self.np_random.choice(self._allow_idxs, size=batch_size, replace=True)
            else:
                if hasattr(self.np_random, "integers"):
                    indx = self.np_random.integers(len(self), size=batch_size)
                else:
                    indx = self.np_random.randint(len(self), size=batch_size)
        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)

