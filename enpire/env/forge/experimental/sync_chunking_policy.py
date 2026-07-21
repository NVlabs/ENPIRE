from collections import deque
from typing import Any

import numpy as np

from enpire.env.forge.experimental.get_action_policy import (
    chunk_to_action_list,
)
from enpire.env.forge.experimental.key_remapping_utils import hold_action_from_proprio


class SyncChunkingPolicy:
    def __init__(
        self,
        policy,
        action_exec_horizon: int,
        require_prev_observation: bool = False,
    ):
        self.policy = policy
        self.action_queue = deque()
        self.last_info = {}
        self.action_exec_horizon = action_exec_horizon
        self.require_prev_observation = require_prev_observation
        self._last_observation: dict[str, Any] | None = None

    def reset(self) -> dict[str, Any] | None:
        self.clear_local_state()
        return self.policy.reset()

    def clear_local_state(self) -> None:
        """Clear queued actions/history without touching the inner policy."""
        self.action_queue = deque()
        self.last_info = {}
        self._last_observation = None

    def _truncate_chunk(self, action_chunk: dict[str, Any]) -> dict[str, np.ndarray]:
        """Truncate action chunk to respect action_exec_horizon."""
        truncated_chunk = {}
        for key, value in action_chunk.items():
            seq = np.asarray(value)
            # Handle both (H, D) and (B, H, D) shapes
            if seq.ndim == 3:
                truncated_chunk[key] = seq[:, : self.action_exec_horizon, :]
            elif seq.ndim == 2:
                truncated_chunk[key] = seq[: self.action_exec_horizon, :]
            else:
                truncated_chunk[key] = value
        return truncated_chunk

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        # Prime one-step history before first policy inference so obs_horizon=2
        # consumers can use a real (t-1, t) pair instead of synthetic zero-delta.
        if self.require_prev_observation and self._last_observation is None:
            self._last_observation = observation
            return hold_action_from_proprio(observation), {
                "remaining_num_action_in_chunk": 0,
                "history_priming": True,
            }

        if not self.action_queue:
            # Provide a real previous observation for chunk-boundary UMI state
            # construction after history priming has completed.
            policy_observation = observation
            if self.require_prev_observation:
                policy_observation = dict(observation)
                policy_observation["__prev_observation"] = self._last_observation
            _action, info = self.policy.get_action(policy_observation)
            print("getting new action")
            del _action
            truncated_chunk = self._truncate_chunk(info["action_chunk"])
            truncated_info = info.copy()
            truncated_info["action_chunk"] = truncated_chunk
            self.last_info = truncated_info
            self.action_queue.extend(chunk_to_action_list(truncated_chunk))
        action = self.action_queue.popleft()
        self._last_observation = observation
        self.last_info["remaining_num_action_in_chunk"] = len(self.action_queue)
        return action, self.last_info
