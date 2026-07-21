"""Replay policy that loads actions from a LeRobotDataset."""

# uv pip install pyarrow

import os
from pathlib import Path

import numpy as np
import pandas as pd

from typing import Any, Literal

from enpire.policy.legacy import Policy
from loguru import logger
DEFAULT_PARQUET_PATH = Path(
    os.environ.get("ENPIRE_REPLAY_DATASET", "")
)


class LerobotReplayPolicy(Policy):
    """Policy that replays actions from a LeRobotDataset episode.

    This policy loads action data from a LeRobotDataset parquet file and
    replays them sequentially. It's useful for verifying data transformations
    by replaying recorded actions in simulation or real environment.

    Action formats:
        14D (joint+gripper):  for joint_position, delta_joint_position
            indices 0-5:   joint_pos_action_left (6D) or difference
            index 6:       gripper_pos_action_left (1D)
            indices 7-12:  joint_pos_action_right (6D) or difference
            index 13:      gripper_pos_action_right (1D)
        
        16D (ee_pose+gripper), for cartesian_position (ee_pose), delta_ee_pose (delta_ee_pose), UMI
            indices 0-2:   left_ee_pos (3D) or difference
            indices 3-6:   left_ee_quat_xyzw (4D) or difference
            index 7:       left_gripper_pos (1D)
            indices 8-10:  right_ee_pos (3D) or difference
            indices 11-14: right_ee_quat_xyzw (4D) or difference
            index 15:      right_gripper_pos (1D)
    """

    CONTROL_MODE_DIMS = {
        "joint_position": 14,
        "delta_joint_position": 14,
        "cartesian_position": 16,
        "delta_ee_pose": 16,
        "umi_ee_pose": 16,
    }

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_PARQUET_PATH,
        replan_horizon: int = 1,
        control_mode: str = "joint_position",
        action_horizon: int = 50,
        norm_stats_path: str | Path = "",
    ):
        """Initialize the dataset replay policy.

        Args:
            dataset_path: Path to parquet/npy/npz file, or folder with action files (for raw teleop)
            replan_horizon: Number of timesteps to advance per get_action() call
            control_mode: Control mode - "joint_position" for 14D joint actions, "cartesian_position" for 16D ee_pose actions, "delta_joint_position" for delta joint actions
            norm_stats_path: Path to norm_stats.json (required for 20D UMI training-pipeline replay)
        """
        if isinstance(dataset_path, Path):
            path_str = str(dataset_path)
        else:
            path_str = str(dataset_path or "")
        path_str = path_str.strip()
        self.dataset_path = Path(path_str) if path_str else Path()
        self.replan_horizon = replan_horizon
        self.control_mode = control_mode
        self.action_horizon = max(1, int(action_horizon))
        self.norm_stats_path = Path(norm_stats_path) if str(norm_stats_path).strip() else None
        self._action_format: Literal["joint", "cartesian"] | None = None
        self._umi_states: np.ndarray | None = None
        self._umi_abs_targets: np.ndarray | None = None

        # Initial state from the recording's first frame (populated by _load_episode_data
        # when observation.state or raw teleop state files are available). Used by
        # the control loop to calibrate the env to the same starting configuration
        # as the original recording — critical for delta replay modes and Sync to Init.
        self._initial_state: dict[str, np.ndarray] | None = None
        self._recorded_states: np.ndarray | None = None

        # Load episode data (auto-detect format)
        if self._is_empty_path(self.dataset_path):
            self._set_empty_actions(reason="Replay dataset path is empty")
        elif self.dataset_path and self.dataset_path.exists():
            print(f"Loading episode data from {self.dataset_path}")
            self._load_episode_data()
        else:
            self._set_empty_actions(
                reason=f"Replay dataset path not found: {self.dataset_path}"
            )

        # Timestep counter
        self.current_step = 0

    @property
    def initial_state(self) -> dict[str, np.ndarray] | None:
        """First frame's observation.state from the recording, or ``None``.

        Returned as a dict matching the ``zero_state`` layout used by
        ``yam_control_loop``:
            ``{left_joint_pos (6,), left_gripper_pos (1,),
              right_joint_pos (6,), right_gripper_pos (1,)}``

        The control loop should pass this to ``env.reset(initial_state=...)``
        when replaying delta modes so the robot starts at the same
        configuration as the original recording.
        """
        return self._initial_state

    @property
    def umi_initial_ee(self) -> np.ndarray | None:
        """First frame's 16D EE pose from UMI data, or ``None``."""
        if self._umi_abs_targets is not None and len(self._umi_abs_targets) > 0:
            return self._umi_abs_targets[0].copy()
        if self._umi_states is not None and len(self._umi_states) > 0:
            return self._umi_states[0].copy()
        return None

    # Reasonable joint-position range for YAM arms (radians).
    _JOINT_RANGE_RAD = (-2 * np.pi, 2 * np.pi)
    _GRIPPER_RANGE = (-0.1, 1.1)

    @classmethod
    def _parse_state_vector(cls, state: np.ndarray) -> dict[str, np.ndarray] | None:
        """Parse a 14-D observation.state vector into a named dict.

        Returns ``None`` (with a warning) if the vector doesn't look like
        valid joint-format data.
        """
        if state.shape != (14,):
            raise ValueError(
                f"observation.state has shape {state.shape}, expected (14,). Cannot extract initial_state as joint position."
            )

        left_jp = state[0:6]
        left_grip = state[6:7]
        right_jp = state[7:13]
        right_grip = state[13:14]

        # Sanity: joint values should be in a reasonable radian range
        lo, hi = cls._JOINT_RANGE_RAD
        for name, arr in [("left_joint_pos", left_jp), ("right_joint_pos", right_jp)]:
            if np.any(arr < lo) or np.any(arr > hi):
                raise ValueError(
                    f"initial_state.{name} has values outside [{lo:.1f}, {hi:.1f}] rad: {arr}. This does not look like joint positions."
                )

        # Sanity: gripper should be in [0, 1] (with tolerance)
        glo, ghi = cls._GRIPPER_RANGE
        for name, val in [("left_gripper", float(left_grip[0])),
                          ("right_gripper", float(right_grip[0]))]:
            if val < glo or val > ghi:
                raise ValueError(
                    f"initial_state.{name} = {val:.4f} outside [{glo}, {ghi}]. This does not look like a gripper value."
                )

        return {
            "left_joint_pos": left_jp.copy(),
            "left_gripper_pos": left_grip.copy(),
            "right_joint_pos": right_jp.copy(),
            "right_gripper_pos": right_grip.copy(),
        }

    @staticmethod
    def _extract_first_state_part(
        values: np.ndarray, *, expected_dim: int, name: str
    ) -> np.ndarray:
        """Return the first recorded state vector for one joint/gripper stream."""
        arr = np.asarray(values)
        if arr.ndim == 0:
            first = arr.reshape(1)
        elif arr.ndim == 1:
            first = arr if expected_dim != 1 else arr[:1]
        else:
            first = np.asarray(arr[0]).reshape(-1)
        if first.shape != (expected_dim,):
            raise ValueError(
                f"Raw replay state {name} has shape {first.shape}, expected {(expected_dim,)}"
            )
        return np.asarray(first, dtype=np.float32)

    def _load_raw_folder_initial_state(self, folder: Path) -> None:
        """Populate ``_initial_state`` from raw teleop state files when present."""
        state_candidates = {
            "left_joint_pos": ["left-joint_pos.npy", "left-joint-pos.npy"],
            "left_gripper_pos": ["left-gripper_pos.npy", "left-gripper-pos.npy"],
            "right_joint_pos": ["right-joint_pos.npy", "right-joint-pos.npy"],
            "right_gripper_pos": ["right-gripper_pos.npy", "right-gripper-pos.npy"],
        }

        resolved: dict[str, Path] = {}
        for key, candidates in state_candidates.items():
            match = next((folder / name for name in candidates if (folder / name).exists()), None)
            if match is None:
                return
            resolved[key] = match

        state_vec = np.concatenate(
            [
                self._extract_first_state_part(
                    np.load(resolved["left_joint_pos"], allow_pickle=True),
                    expected_dim=6,
                    name="left_joint_pos",
                ),
                self._extract_first_state_part(
                    np.load(resolved["left_gripper_pos"], allow_pickle=True),
                    expected_dim=1,
                    name="left_gripper_pos",
                ),
                self._extract_first_state_part(
                    np.load(resolved["right_joint_pos"], allow_pickle=True),
                    expected_dim=6,
                    name="right_joint_pos",
                ),
                self._extract_first_state_part(
                    np.load(resolved["right_gripper_pos"], allow_pickle=True),
                    expected_dim=1,
                    name="right_gripper_pos",
                ),
            ],
            axis=0,
        )
        parsed = self._parse_state_vector(state_vec)
        if parsed is None:
            return
        self._initial_state = parsed
        print(
            "[LerobotReplayPolicy] Extracted initial_state from raw teleop state files"
        )
        print(f"  left_joint_pos  = {parsed['left_joint_pos']}")
        print(f"  left_gripper_pos  = {parsed['left_gripper_pos']}")
        print(f"  right_joint_pos = {parsed['right_joint_pos']}")
        print(f"  right_gripper_pos = {parsed['right_gripper_pos']}")

    def _infer_action_format(self, action_dim: int) -> Literal["joint", "cartesian"]:
        if action_dim == 14:
            return "joint"
        if action_dim == 16:
            return "cartesian"
        raise ValueError(f"Unsupported action dimension: {action_dim}. Expected 14 or 16.")

    def _set_empty_actions(self, reason: str) -> None:
        expected_dim = self.CONTROL_MODE_DIMS.get(self.control_mode, 14)
        self.actions = np.zeros((1, expected_dim), dtype=float)
        self.num_steps = 1
        self.current_step = 0
        self._action_format = self._infer_action_format(expected_dim)
        print(f"[LerobotReplayPolicy] {reason}. Using zeros until updated.")

    @staticmethod
    def _is_empty_path(path: Path) -> bool:
        if not str(path).strip():
            return True
        return path.resolve() == Path(".").resolve()

    def _load_episode_data(self):
        """Load action data from file or folder (auto-detect format: parquet, npy, or npz)."""

        if self._is_empty_path(self.dataset_path):
            self._set_empty_actions(reason="Replay dataset path is empty")
            return
        if not self.dataset_path or not self.dataset_path.exists():
            self._set_empty_actions(
                reason=f"Replay dataset path not found: {self.dataset_path}"
            )
            return

        if self.dataset_path.is_dir():  # raw teleoperation data folder
            logger.info(f"Loading raw teleoperation data from {self.dataset_path}")
            separate_action_files = ["action-left-pos.npy", "action-right-pos.npy"]
            actions = np.concatenate([np.load(self.dataset_path / action_file, allow_pickle=True) for action_file in separate_action_files], axis=1)
            self._load_raw_folder_initial_state(self.dataset_path)
            file_ext = "folder"
        else:
            file_ext = self.dataset_path.suffix.lower()
            if file_ext == ".parquet":
                # Load parquet file
                df = pd.read_parquet(self.dataset_path)
                if self.control_mode == "umi_ee_pose":
                    if "observation.state" not in df.columns:
                        raise ValueError(
                            "UMI replay requires parquet column 'observation.state'."
                        )
                    umi_states = np.stack(df["observation.state"].values)
                    if umi_states.ndim != 2 or umi_states.shape[1] not in (16, 20):
                        raise ValueError(
                            f"UMI replay expects observation.state shape (T, 16 or 20), got {umi_states.shape}"
                        )
                    if umi_states.shape[1] == 20:
                        self._umi_abs_targets = self._precompute_umi_20d_targets(umi_states)
                        self._umi_states = None
                        actions = self._umi_abs_targets.copy()
                    else:
                        self._umi_states = np.asarray(umi_states, dtype=np.float32)
                        self._umi_abs_targets = None
                        actions = self._umi_states.copy()
                else:
                    # Extract actions - stored as list of arrays in "action" column
                    if "action" not in df.columns:
                        raise ValueError(
                            "Replay parquet missing 'action' column for non-UMI control mode."
                        )
                    actions = np.stack(df["action"].values)  # Shape: (T, D)

                # Extract first-frame observation.state for delta-mode calibration.
                # This MUST be in 14D joint format regardless of action format. That should be the initial joint position.
                if "observation.state" in df.columns:
                    states = np.stack(df["observation.state"].values)
                    if states.shape[1] == 14 and self.control_mode == "delta_ee_pose":
                        self._recorded_states = states.astype(np.float64)
                    if states.shape[1] == 14:
                        parsed = self._parse_state_vector(states[0])
                        if parsed is not None:
                            self._initial_state = parsed
                            print(
                                f"[LerobotReplayPolicy] Extracted initial_state from "
                                f"first frame observation.state (14D joint format)"
                            )
                            print(
                                f"  left_joint_pos  = {parsed['left_joint_pos']}"
                            )
                            print(
                                f"  left_gripper_pos  = {parsed['left_gripper_pos']}"
                            )
                            print(
                                f"  right_joint_pos = {parsed['right_joint_pos']}"
                            )
                            print(
                                f"  right_gripper_pos = {parsed['right_gripper_pos']}"
                            )
                        # else: warning already printed by _parse_state_vector
                    elif self.control_mode in ("delta_ee_pose", "delta_joint_position"):
                        print(
                            f"[LerobotReplayPolicy] WARNING: observation.state is "
                            f"{states.shape[1]}D, expected 14D joint format. "
                            f"Cannot extract initial_state for delta-mode "
                            f"calibration. Replay will start from zero joints."
                        )
                elif self.control_mode in ("delta_ee_pose", "delta_joint_position"):
                    print(
                        f"[LerobotReplayPolicy] WARNING: parquet has no "
                        f"'observation.state' column. Cannot extract initial_state "
                        f"for {self.control_mode} calibration. "
                        f"Replay will start from zero joints."
                    )
            elif file_ext == ".npy":
                # Load numpy array file
                data = np.load(self.dataset_path, allow_pickle=True)
                if isinstance(data, np.ndarray):
                    # Direct array: assume shape is (T, D)
                    actions = data
                elif isinstance(data, dict):
                    # Dict with 'action' key
                    if "action" in data:
                        actions = data["action"]
                    else:
                        raise ValueError(
                            f".npy file must contain an array or a dict with 'action' key. "
                            f"Got keys: {list(data.keys())}"
                        )
                else:
                    raise ValueError(f"Unexpected data type in .npy file: {type(data)}")
            elif file_ext == ".npz":
                # Load compressed numpy file
                data = np.load(self.dataset_path, allow_pickle=True)
                if "action" in data:
                    actions = data["action"]
                else:
                    raise ValueError(
                        f".npz file must contain 'action' key. Got keys: {list(data.keys())}"
                    )
            else:
                raise ValueError(
                    f"Unsupported file format: {file_ext}. "
                    f"Supported formats: .parquet, .npy, .npz, or folder with action files"
                )

        # Ensure actions is 2D (T, D)
        if actions.ndim == 1:
            # Single action, reshape to (1, D)
            actions = actions.reshape(1, -1)
        elif actions.ndim != 2:
            raise ValueError(
                f"Actions must be 2D array (T, D), got shape: {actions.shape}"
            )

        # Validate action dimension matches expected control_mode (if known).
        # For umi_ee_pose with 20D source data, _precompute_umi_20d_targets already
        # converted to 16D absolute targets, so validation passes normally.
        expected_dim = self.CONTROL_MODE_DIMS.get(self.control_mode)
        if expected_dim is not None and actions.shape[1] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim}D actions for control_mode='{self.control_mode}', "
                f"but got {actions.shape[1]}D actions in {self.dataset_path}"
            )

        # Infer action format from data to support future control modes
        self._action_format = self._infer_action_format(actions.shape[1])

        self.actions = actions
        self.num_steps = len(actions)

        print(
            f"[LerobotReplayPolicy] Loaded {self.num_steps} steps from {self.dataset_path} "
            f"(format={file_ext}, control_mode={self.control_mode}, {actions.shape[1]}D)"
        )

    @staticmethod
    def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
        out = np.asarray(quat, dtype=np.float32).copy()
        out[..., :3] *= -1.0
        return out

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        q1 = np.asarray(q1, dtype=np.float32)
        q2 = np.asarray(q2, dtype=np.float32)
        x1, y1, z1, w1 = np.split(q1, 4, axis=-1)
        x2, y2, z2, w2 = np.split(q2, 4, axis=-1)
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return np.concatenate([x, y, z, w], axis=-1)

    @staticmethod
    def _quat_normalize(quat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32)
        denom = np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), eps)
        return quat / denom

    @staticmethod
    def _load_action_quantiles(path: Path) -> tuple:
        """Load action q01/q99 from norm_stats.json without heavy flash_manip imports."""
        import json, torch
        raw = json.loads(path.read_text())
        entry = raw.get("norm_stats", {}).get("action")
        if entry is None:
            raise RuntimeError(f"norm_stats.json at {path} missing 'action' key")
        q01 = torch.tensor(entry["q01"], dtype=torch.float32) if "q01" in entry else torch.full((20,), -1.0)
        q99 = torch.tensor(entry["q99"], dtype=torch.float32) if "q99" in entry else torch.full((20,), 1.0)
        return q01, q99

    def _precompute_umi_20d_targets(self, raw_states_20d: np.ndarray) -> np.ndarray:
        """Run training-pipeline round-trip on 20D UMI states and return [T, 16] absolute EE targets.

        For each action chunk: build ego-centric actions (``build_ego_action_chunk``),
        normalize, unnormalize (simulating a perfect neural net), then convert back to
        global-frame absolute targets (pos + quat_xyzw + grip per arm = 16D).
        """
        import torch
        from scipy.spatial.transform import Rotation
        from flash_manip.datasets.umi_transforms import (
            build_ego_action_chunk,
            rotmat_from_6d,
        )

        if self.norm_stats_path is None or not self.norm_stats_path.exists():
            raise FileNotFoundError(
                f"norm_stats.json required for 20D UMI replay but not found: {self.norm_stats_path}"
            )

        act_q01, act_q99 = self._load_action_quantiles(self.norm_stats_path)

        raw = torch.as_tensor(raw_states_20d, dtype=torch.float32)
        T = raw.shape[0]
        H = self.action_horizon
        targets = np.zeros((T, 16), dtype=np.float32)

        for chunk_idx in range(0, T, H):
            chunk_end = min(chunk_idx + H, T)
            anchor = raw[chunk_idx]
            future = raw[chunk_idx:chunk_end]

            ego_norm = build_ego_action_chunk(anchor, future, act_q01, act_q99)
            # Unnormalize (inverse of quantile normalization)
            q01 = act_q01[..., :ego_norm.shape[-1]]
            q99 = act_q99[..., :ego_norm.shape[-1]]
            ego_unnorm = (ego_norm + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01

            # Convert ego-centric deltas back to global-frame absolute targets
            pL_t = anchor[0:3]
            RL_t = rotmat_from_6d(anchor[3:9])
            pR_t = anchor[10:13]
            RR_t = rotmat_from_6d(anchor[13:19])

            dpL = ego_unnorm[:, 0:3]
            dRL_6d = ego_unnorm[:, 3:9]
            gL = ego_unnorm[:, 9:10]
            dpR = ego_unnorm[:, 10:13]
            dRR_6d = ego_unnorm[:, 13:19]
            gR = ego_unnorm[:, 19:20]

            pL_target = pL_t + (RL_t @ dpL.unsqueeze(-1)).squeeze(-1)
            pR_target = pR_t + (RR_t @ dpR.unsqueeze(-1)).squeeze(-1)
            dRL = rotmat_from_6d(dRL_6d)
            dRR = rotmat_from_6d(dRR_6d)
            RL_target = RL_t @ dRL
            RR_target = RR_t @ dRR

            for j in range(chunk_end - chunk_idx):
                i = chunk_idx + j
                lq = Rotation.from_matrix(RL_target[j].detach().numpy()).as_quat().astype(np.float32)
                rq = Rotation.from_matrix(RR_target[j].detach().numpy()).as_quat().astype(np.float32)
                targets[i] = np.concatenate([
                    pL_target[j].detach().numpy(),
                    lq,
                    gL[j].detach().numpy(),
                    pR_target[j].detach().numpy(),
                    rq,
                    gR[j].detach().numpy(),
                ])

        print(
            f"[LerobotReplayPolicy] Pre-computed {T} absolute EE targets "
            f"via training-pipeline round-trip (20D UMI -> ego -> norm -> unnorm -> global 16D)"
        )
        return targets

    def _umi_action_at(self, step: int) -> tuple[np.ndarray, bool]:
        """Return (16D action, is_absolute) for UMI replay at *step*.

        When 20D training-pipeline targets are available, returns the
        pre-computed absolute EE target.  Otherwise falls back to the
        legacy 16D quaternion-delta path.
        """
        step = int(np.clip(step, 0, self.num_steps - 1))

        if self._umi_abs_targets is not None:
            return self._umi_abs_targets[step].copy(), True

        if self._umi_states is None:
            raise RuntimeError("UMI states not loaded.")

        chunk_start = (step // self.action_horizon) * self.action_horizon
        s0 = self._umi_states[chunk_start]
        st = self._umi_states[step]

        left_dpos = st[0:3] - s0[0:3]
        right_dpos = st[8:11] - s0[8:11]
        left_dquat = self._quat_normalize(
            self._quat_multiply(st[3:7][None, :], self._quat_conjugate(s0[3:7][None, :]))
        )[0]
        right_dquat = self._quat_normalize(
            self._quat_multiply(st[11:15][None, :], self._quat_conjugate(s0[11:15][None, :]))
        )[0]

        delta = np.concatenate(
            [left_dpos, left_dquat, st[7:8],
             right_dpos, right_dquat, st[15:16]],
            axis=0,
        ).astype(np.float32)
        return delta, False

    def _umi_chunk_start(self, step: int) -> int:
        return (int(step) // self.action_horizon) * self.action_horizon

    def _extract_action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """Convert action to policy action format.

        Args:
            action: Action vector of shape (14,) for joint or (16,) for ee_pose

        Returns:
            Action dict with joint/gripper positions or ee_pose/gripper positions
        """
        if self._action_format == "joint":
            if action.shape[-1] == 14:
                return {
                    "left_joint_pos": action[0:6],  # 6D
                    "right_joint_pos": action[7:13],  # 6D
                    "left_gripper_pos": action[6:7],  # 1D
                    "right_gripper_pos": action[13:14],  # 1D
                }
            else:
                raise ValueError(
                    f"Expected 14D action for control_mode='joint_position' or 'delta_joint_position', got {action.shape}"
                )
        elif self._action_format == "cartesian":
            if action.shape[-1] == 16:
                return {
                    "left_ee_pos": action[0:3],  # 3D
                    "left_ee_quat_xyzw": action[3:7],  # 4D
                    "left_gripper_pos": action[7:8],  # 1D
                    "right_ee_pos": action[8:11],  # 3D
                    "right_ee_quat_xyzw": action[11:15],  # 4D
                    "right_gripper_pos": action[15:16],  # 1D
                }
            else:
                raise ValueError(
                    f"Expected 16D action for control_mode='cartesian_position', got {action.shape}"
                )
        else:
            raise ValueError(f"Unsupported action format for control_mode: {self.control_mode}")

    def _build_action_chunk(self, start_step: int, horizon: int) -> dict[str, np.ndarray]:
        """Build action chunk for info dict.

        Args:
            start_step: Starting timestep
            horizon: Number of steps in chunk

        Returns:
            Action chunk dict with shape (horizon, dim) for each key
        """
        if self.control_mode == "umi_ee_pose":
            if self._umi_states is None and self._umi_abs_targets is None:
                raise RuntimeError("UMI states not loaded.")
            start = (int(start_step) // self.action_horizon) * self.action_horizon
            end_step = min(start + self.action_horizon, self.num_steps)
            actual_horizon = end_step - start
            chunk_actions = np.stack(
                [self._umi_action_at(t)[0] for t in range(start, end_step)],
                axis=0,
            )
            target_horizon = max(horizon, self.action_horizon)
            if actual_horizon < target_horizon:
                padding = np.tile(chunk_actions[-1:], (target_horizon - actual_horizon, 1))
                chunk_actions = np.concatenate([chunk_actions, padding], axis=0)
            return {
                "left_ee_pos": chunk_actions[:, 0:3],
                "left_ee_quat_xyzw": chunk_actions[:, 3:7],
                "left_gripper_pos": chunk_actions[:, 7:8],
                "right_ee_pos": chunk_actions[:, 8:11],
                "right_ee_quat_xyzw": chunk_actions[:, 11:15],
                "right_gripper_pos": chunk_actions[:, 15:16],
            }

        end_step = min(start_step + horizon, self.num_steps)
        actual_horizon = end_step - start_step

        # Get actions for this chunk
        chunk_actions = self.actions[start_step:end_step]  # (actual_horizon, D)

        # Pad if needed (repeat last action)
        if actual_horizon < horizon:
            padding = np.tile(chunk_actions[-1:], (horizon - actual_horizon, 1))
            chunk_actions = np.concatenate([chunk_actions, padding], axis=0)

        if self._action_format == "joint":
            if chunk_actions.shape[-1] == 14:
                return {
                    "left_joint_pos": chunk_actions[:, 0:6],
                    "right_joint_pos": chunk_actions[:, 7:13],
                    "left_gripper_pos": chunk_actions[:, 6:7],
                    "right_gripper_pos": chunk_actions[:, 13:14],
                }
            else:
                raise ValueError(
                    f"Expected 14D actions for control_mode={self.control_mode}, "
                    f"got {chunk_actions.shape}"
                )
        elif self._action_format == "cartesian":
            if chunk_actions.shape[-1] == 16:
                chunk = {
                    "left_ee_pos": chunk_actions[:, 0:3],
                    "left_ee_quat_xyzw": chunk_actions[:, 3:7],
                    "left_gripper_pos": chunk_actions[:, 7:8],
                    "right_ee_pos": chunk_actions[:, 8:11],
                    "right_ee_quat_xyzw": chunk_actions[:, 11:15],
                    "right_gripper_pos": chunk_actions[:, 15:16],
                }
                if self.control_mode == "delta_ee_pose":
                    chunk["_delta_ee"] = True
                return chunk
            else:
                raise ValueError(
                    f"Expected 16D actions for control_mode={self.control_mode}, "
                    f"got {chunk_actions.shape}"
                )
        else:
            raise ValueError(f"Unsupported action format for control_mode: {self.control_mode}")

    def _build_zero_action(self) -> dict[str, np.ndarray]:
        """Build a neutral hold action for post-episode playback.

        For delta control modes this returns zero deltas so we do not keep
        applying the final non-zero command after the episode ends.
        """
        if self.control_mode == "delta_joint_position":
            return {
                "left_joint_pos": np.zeros(6, dtype=np.float32),
                "right_joint_pos": np.zeros(6, dtype=np.float32),
                "left_gripper_pos": np.zeros(1, dtype=np.float32),
                "right_gripper_pos": np.zeros(1, dtype=np.float32),
            }
        if self.control_mode in ("delta_ee_pose", "umi_ee_pose"):
            return {
                "left_ee_pos": np.zeros(3, dtype=np.float32),
                "left_ee_quat_xyzw": np.array([0, 0, 0, 1], dtype=np.float32),
                "left_gripper_pos": np.zeros(1, dtype=np.float32),
                "right_ee_pos": np.zeros(3, dtype=np.float32),
                "right_ee_quat_xyzw": np.array([0, 0, 0, 1], dtype=np.float32),
                "right_gripper_pos": np.zeros(1, dtype=np.float32),
            }
        # Absolute modes can safely hold the last action.
        return self._extract_action(self.actions[-1])

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Get the next action from the replay.

        Args:
            observation: Ignored for replay policy

        Returns:
            Tuple of (action, info) where action is the current step's action
            and info contains the action_chunk
        """
        if self.current_step >= self.num_steps:
            # Episode ended: return a neutral hold action.
            # Returning the last delta repeatedly can cause drift/instability.
            action = self._build_zero_action()
            info = {
                "action_chunk": self._build_action_chunk(self.num_steps - 1, self.replan_horizon),
                "episode_done": True,
                "current_step": self.num_steps,
                "num_steps": self.num_steps,
            }
            return action, info

        # Get current action
        if self.control_mode == "umi_ee_pose":
            vec, is_abs = self._umi_action_at(self.current_step)
            action = self._extract_action(vec)
            action["umi_absolute_target"] = is_abs
            if not is_abs:
                action["umi_chunk_start_step"] = int(self._umi_chunk_start(self.current_step))
        else:
            action = self._extract_action(self.actions[self.current_step])
            if (
                self.control_mode == "delta_ee_pose"
                and self._recorded_states is not None
                and self.current_step < len(self._recorded_states)
            ):
                s = self._recorded_states[self.current_step]
                action["_recorded_state_left_jp"] = s[0:6].copy()
                action["_recorded_state_right_jp"] = s[7:13].copy()

        # Build action chunk for info
        action_chunk = self._build_action_chunk(self.current_step, self.replan_horizon)

        # Advance timestep
        self.current_step += self.replan_horizon

        info = {
            "action_chunk": action_chunk,
            "episode_done": self.current_step >= self.num_steps,
            "current_step": self.current_step - self.replan_horizon,
            "num_steps": self.num_steps,
        }

        return action, info

    def peek_action(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Return the next replay action/info without advancing current_step.

        This is used by UI wrappers in pause mode so previewing does not consume
        real replay timesteps.
        """
        if self.current_step >= self.num_steps:
            action = self._build_zero_action()
            info = {
                "action_chunk": self._build_action_chunk(self.num_steps - 1, self.replan_horizon),
                "episode_done": True,
                "current_step": self.num_steps,
                "num_steps": self.num_steps,
            }
            return action, info

        if self.control_mode == "umi_ee_pose":
            vec, is_abs = self._umi_action_at(self.current_step)
            action = self._extract_action(vec)
            action["umi_absolute_target"] = is_abs
            if not is_abs:
                action["umi_chunk_start_step"] = int(self._umi_chunk_start(self.current_step))
        else:
            action = self._extract_action(self.actions[self.current_step])
            if (
                self.control_mode == "delta_ee_pose"
                and self._recorded_states is not None
                and self.current_step < len(self._recorded_states)
            ):
                s = self._recorded_states[self.current_step]
                action["_recorded_state_left_jp"] = s[0:6].copy()
                action["_recorded_state_right_jp"] = s[7:13].copy()
        action_chunk = self._build_action_chunk(self.current_step, self.replan_horizon)
        info = {
            "action_chunk": action_chunk,
            "episode_done": False,
            "current_step": self.current_step,
            "num_steps": self.num_steps,
        }
        return action, info

    def reset(self) -> dict[str, Any] | None:
        """Reset the replay to the beginning of the episode."""
        self.current_step = 0
        return {"task_name": f"replay_{self.dataset_path.stem}"}

    def update_replay_config(
        self,
        dataset_path: str | Path | None = None,
        control_mode: str | None = None,
        action_horizon: int | None = None,
        norm_stats_path: str | Path | None = None,
    ) -> None:
        """Update replay dataset path/control mode at runtime and reload data."""
        next_dataset_path = self.dataset_path
        if dataset_path:
            new_path = Path(dataset_path)
            if not new_path.exists():
                raise ValueError(f"Replay dataset path does not exist: {new_path}")
            next_dataset_path = new_path
        next_control_mode = control_mode or self.control_mode
        next_action_horizon = (
            max(1, int(action_horizon)) if action_horizon is not None else self.action_horizon
        )
        next_norm_stats_path = self.norm_stats_path
        if norm_stats_path is not None:
            ns = str(norm_stats_path).strip()
            next_norm_stats_path = Path(ns) if ns else None

        old_state = {
            "dataset_path": self.dataset_path,
            "control_mode": self.control_mode,
            "action_horizon": self.action_horizon,
            "norm_stats_path": self.norm_stats_path,
            "actions": self.actions,
            "num_steps": self.num_steps,
            "current_step": self.current_step,
            "action_format": self._action_format,
            "initial_state": self._initial_state,
            "umi_states": self._umi_states,
            "umi_abs_targets": self._umi_abs_targets,
            "recorded_states": self._recorded_states,
        }

        try:
            self.dataset_path = next_dataset_path
            self.control_mode = next_control_mode
            self.action_horizon = next_action_horizon
            self.norm_stats_path = next_norm_stats_path
            self._initial_state = None
            self._recorded_states = None
            self._umi_states = None
            self._umi_abs_targets = None
            self._load_episode_data()
            self.current_step = 0
        except Exception:
            self.dataset_path = old_state["dataset_path"]
            self.control_mode = old_state["control_mode"]
            self.action_horizon = old_state["action_horizon"]
            self.norm_stats_path = old_state["norm_stats_path"]
            self.actions = old_state["actions"]
            self.num_steps = old_state["num_steps"]
            self.current_step = old_state["current_step"]
            self._action_format = old_state["action_format"]
            self._initial_state = old_state["initial_state"]
            self._umi_states = old_state["umi_states"]
            self._umi_abs_targets = old_state["umi_abs_targets"]
            self._recorded_states = old_state["recorded_states"]
            raise

if __name__ =="__main__":
    replay_policy=LerobotReplayPolicy(control_mode="joint_position")
    print(f"num_steps: {replay_policy.num_steps}")
    print(f"Reset:")
    replay_policy.reset()
    test_steps = 3
    for i in range(test_steps):
        action, info = replay_policy.get_action(None)
        print(f"Action: {action}")
        print(f"Info: {info}")
    print(f"Reset:")
    replay_policy.reset()
    test_steps = 3
    for i in range(test_steps):
        action, info = replay_policy.get_action(None)
        print(f"Action: {action}")
        print(f"Info: {info}")
