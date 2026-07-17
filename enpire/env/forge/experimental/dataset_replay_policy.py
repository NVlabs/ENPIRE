"""Replay policy that loads actions from a LeRobotDataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from typing import Any

from enpire.policy.legacy import Policy


class DatasetReplayPolicy(Policy):
    """Policy that replays actions from a LeRobotDataset episode.

    This policy loads action data from a LeRobotDataset parquet file and
    replays them sequentially. It's useful for verifying data transformations
    by replaying recorded actions in simulation.

    Action format (xdof, 46D):
        indices 0-6:   joint_pos_action_left (6D)
        indices 6-12:  joint_pos_action_right (6D)
        indices 12-28: ee_pose_action_left (16D) - not used
        indices 28-44: ee_pose_action_right (16D) - not used
        indices 44-45: gripper_pos_action_left (1D)
        indices 45-46: gripper_pos_action_right (1D)
    """

    def __init__(
        self,
        dataset_path: str | Path,
        episode_index: int = 0,
        chunk_size: int = 1,
    ):
        """Initialize the dataset replay policy.

        Args:
            dataset_path: Path to LeRobotDataset directory
            episode_index: Which episode to replay (0-indexed)
            chunk_size: Number of timesteps to advance per get_action() call
        """
        self.dataset_path = Path(dataset_path)
        self.episode_index = episode_index
        self.chunk_size = chunk_size

        # Load episode data from parquet
        self._load_episode_data()

        # Timestep counter
        self.current_step = 0

    def _load_episode_data(self):
        """Load action data from the dataset's parquet file."""
        # Find the parquet file for this episode
        # LeRobot stores episodes in data/chunk-XXX/episode_XXXXXX.parquet
        data_dir = self.dataset_path / "data"

        # Find episode file (may be in different chunks)
        episode_file = None
        for chunk_dir in sorted(data_dir.glob("chunk-*")):
            candidate = chunk_dir / f"episode_{self.episode_index:06d}.parquet"
            if candidate.exists():
                episode_file = candidate
                break

        if episode_file is None:
            raise FileNotFoundError(f"Episode {self.episode_index} not found in {data_dir}")

        # Load parquet file
        df = pd.read_parquet(episode_file)

        # Extract actions - stored as list of arrays in "action" column
        actions = np.stack(df["action"].values)  # Shape: (T, 46)

        self.actions = actions
        self.num_steps = len(actions)

        print(
            f"[DatasetReplayPolicy] Loaded episode {self.episode_index} "
            f"with {self.num_steps} steps from {episode_file}"
        )

    def _extract_action(self, action_46d: np.ndarray) -> dict[str, np.ndarray]:
        """Convert 46D action to policy action format.

        Args:
            action_46d: Action vector of shape (46,)

        Returns:
            Action dict with joint and gripper positions
        """
        return {
            "left_joint_pos": action_46d[0:6],  # 6D
            "right_joint_pos": action_46d[6:12],  # 6D
            "left_gripper_pos": action_46d[44:45],  # 1D
            "right_gripper_pos": action_46d[45:46],  # 1D
        }

    def _build_action_chunk(self, start_step: int, horizon: int) -> dict[str, np.ndarray]:
        """Build action chunk for info dict.

        Args:
            start_step: Starting timestep
            horizon: Number of steps in chunk

        Returns:
            Action chunk dict with shape (horizon, dim) for each key
        """
        end_step = min(start_step + horizon, self.num_steps)
        actual_horizon = end_step - start_step

        # Get actions for this chunk
        chunk_actions = self.actions[start_step:end_step]  # (actual_horizon, 46)

        # Pad if needed (repeat last action)
        if actual_horizon < horizon:
            padding = np.tile(chunk_actions[-1:], (horizon - actual_horizon, 1))
            chunk_actions = np.concatenate([chunk_actions, padding], axis=0)

        return {
            "left_joint_pos": chunk_actions[:, 0:6],
            "right_joint_pos": chunk_actions[:, 6:12],
            "left_gripper_pos": chunk_actions[:, 44:45],
            "right_gripper_pos": chunk_actions[:, 45:46],
        }

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Get the next action from the replay.

        Args:
            observation: Ignored for replay policy

        Returns:
            Tuple of (action, info) where action is the current step's action
            and info contains the action_chunk
        """
        if self.current_step >= self.num_steps:
            # Episode ended - return last action and signal done
            action = self._extract_action(self.actions[-1])
            info = {
                "action_chunk": self._build_action_chunk(self.num_steps - 1, self.chunk_size),
                "episode_done": True,
            }
            return action, info

        # Get current action
        action = self._extract_action(self.actions[self.current_step])

        # Build action chunk for info
        action_chunk = self._build_action_chunk(self.current_step, self.chunk_size)

        # Advance timestep
        self.current_step += self.chunk_size

        info = {
            "action_chunk": action_chunk,
            "episode_done": self.current_step >= self.num_steps,
            "current_step": self.current_step - self.chunk_size,  # Step we just played
            "num_steps": self.num_steps,
        }

        return action, info

    def reset(self) -> dict[str, Any] | None:
        """Reset the replay to the beginning of the episode."""
        self.current_step = 0
        return {"task_name": f"replay_episode_{self.episode_index}"}
