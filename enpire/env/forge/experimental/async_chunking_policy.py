# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
import warnings
import weakref
from collections import deque
from typing import Any, Literal

import numpy as np

from enpire.env.forge.experimental.get_action_policy import (
    chunk_to_action_list,
)
from enpire.env.forge.experimental.key_remapping_utils import hold_action_from_proprio
from enpire.env.forge.experimental.realtime_rtc_chunking_policy import (
    _action_list_to_chunk,
    _blend_action_lists,
    _clone_action,
)


class AsyncChunkingPolicy:
    def __init__(
        self,
        policy,
        action_exec_horizon: int,
        policy_latency_steps: int | None,
        max_get_action_seconds: float,
        replan_horizon: int | None = None,
        use_chunk_smoothing: bool = False,
        min_smooth_steps: int = 8,
        require_prev_observation: bool = False,
        use_torch_compile: bool = False,
        torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default",
    ):
        self.policy = policy
        self.action_queue = deque()
        self.last_info = {}
        self.action_exec_horizon = int(action_exec_horizon)
        self.replan_horizon = int(replan_horizon) if replan_horizon is not None else None
        if self.replan_horizon is not None:
            if self.replan_horizon <= 0:
                raise ValueError(f"replan_horizon must be > 0, got {self.replan_horizon}")
            if self.replan_horizon >= self.action_exec_horizon:
                raise ValueError(
                    "Async chunking requires replan_horizon < action_exec_horizon, got "
                    f"{self.replan_horizon} >= {self.action_exec_horizon}"
                )
            self.policy_latency_steps = self.action_exec_horizon - self.replan_horizon - 1
        else:
            if policy_latency_steps is None:
                raise ValueError(
                    "AsyncChunkingPolicy requires either policy_latency_steps or replan_horizon"
                )
            self.policy_latency_steps = int(policy_latency_steps)
        self.require_prev_observation = require_prev_observation
        self.use_chunk_smoothing = bool(use_chunk_smoothing)
        self.min_smooth_steps = max(1, int(min_smooth_steps))

        # For communicating with policy
        self.lock = threading.Lock()
        self.should_exit = False

        # true while self.policy.reset() should be or is currently being called
        self.is_resetting = False
        # true after reset until the first action chunk is enqueued
        self._pipeline_priming = True
        # set externally to flush the queue and recompute the next chunk from step 0
        self._needs_replan = False

        # get_action input
        self.last_obs = None
        self._last_raw_observation = None
        self._last_executed_action = None
        self._obs_event = threading.Event()  # signaled when last_obs becomes non-None
        self.max_get_action_seconds = max_get_action_seconds

        # torch.compile settings
        self.use_torch_compile = use_torch_compile
        self.torch_compile_mode = torch_compile_mode
        self._compiled = False

        self._policy_thread = threading.Thread(daemon=True, target=self._policy_loop)
        self._policy_thread.start()

    def _apply_torch_compile(self):
        """Apply torch.compile to the DiT action head.

        Must be called from the policy thread to ensure correct CUDA context.
        """
        if self._compiled or not self.use_torch_compile:
            return

        import torch

        print(f"[AsyncChunkingPolicy] Applying torch.compile (mode={self.torch_compile_mode})...")

        # Navigate to the underlying model:
        # AsyncChunkingPolicy -> GetActionPolicy -> RobotInterface/OmniDiffusionPolicy
        inner_policy = self.policy
        if hasattr(inner_policy, "policy"):
            inner_policy = inner_policy.policy

        # Try to find and compile the DiT action head
        model = None
        if hasattr(inner_policy, "model"):
            model = inner_policy.model
        elif hasattr(inner_policy, "policy") and hasattr(inner_policy.policy, "model"):
            model = inner_policy.policy.model

        if model is not None and hasattr(model, "action_head"):
            try:
                print("[AsyncChunkingPolicy] Compiling DiT action head...")
                model.action_head.get_action = torch.compile(
                    model.action_head.get_action,
                    mode=self.torch_compile_mode,
                    fullgraph=False,
                    dynamic=True,
                )
                print("[AsyncChunkingPolicy] torch.compile applied to DiT action head.")
                self._compiled = True
            except Exception as e:
                print(f"[AsyncChunkingPolicy] WARNING: torch.compile failed: {e}")
                print("[AsyncChunkingPolicy] Continuing without compilation...")
                self._compiled = True  # Don't retry
        else:
            print("[AsyncChunkingPolicy] WARNING: Could not find model.action_head to compile.")
            self._compiled = True  # Don't retry

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        with self.lock:
            prev_raw = self._last_raw_observation
            self._last_raw_observation = observation
            if self.require_prev_observation:
                # Prime one-step history before first policy inference so
                # obs_horizon=2 consumers always get a real (t-1, t) pair.
                if prev_raw is None:
                    self.last_obs = None
                else:
                    payload = dict(observation)
                    payload["__prev_observation"] = prev_raw
                    self.last_obs = payload
                    self._obs_event.set()
            else:
                self.last_obs = observation
                self._obs_event.set()
        first_chunk = self.last_obs is None
        if self.require_prev_observation and prev_raw is None:
            hold_action = hold_action_from_proprio(observation)
            return hold_action, {"remaining_num_action_in_chunk": 0, "history_priming": True}
        # After reset, the policy thread needs ~250ms for the first forward
        # pass.  Rather than spinning for max_get_action_seconds each step,
        # return a hold action immediately so the control loop stays at ~8ms.
        if self._pipeline_priming and not self.action_queue:
            hold_action = hold_action_from_proprio(observation)
            return hold_action, {"pipeline_priming": True}
        start = time.monotonic()
        while not self.action_queue:
            if not first_chunk:
                warnings.warn(
                    "AsyncChunkingPolicy waiting for policy.get_action(). "
                    + "Consider increasing policy_latency_steps or lowering replan_horizon."
                )
            now = time.monotonic()
            if now - start > self.max_get_action_seconds:
                # To avoid blocking, compute a hold action from the observation
                # if we run out of time.
                hold_action = hold_action_from_proprio(observation)
                return hold_action, {}
            time.sleep(0.001)
            if not self._policy_thread.is_alive():
                raise ValueError("AsyncChunkingPolicy thread is not alive")
            if self.last_obs is None:
                # This is mostly likely because someone called reset from
                # another thread. In this case we aren't going to ever get any
                # actions, so crash.
                raise ValueError(
                    "Observation reset to None while waiting for actions. "
                    + "Did AsyncChunkingPolicy.reset() get called from a second thread?"
                )
        with self.lock:
            action = self.action_queue.popleft()
            if self.use_chunk_smoothing:
                self._last_executed_action = _clone_action(action)
            return action, self.last_info

    def reset(self) -> dict[str, Any] | None:
        # Instead of emptying the action queue here, we wait for the policy loop
        # to empty it
        self.is_resetting = True
        start_time = time.monotonic()
        while True:
            time.sleep(0.001)
            if not self.is_resetting:
                assert len(self.action_queue) == 0, "Reset should empty the current chunk"
                return self.reset_info
            if not self._policy_thread.is_alive():
                raise ValueError("AsyncChunkingPolicy thread is not alive")
            if time.monotonic() > start_time + 1:
                print(f"Waiting for {self.policy}.reset()")
                time.sleep(1)

    def _policy_loop(self):
        # Only use a weakref, to make sure we don't keep this thread alive if
        # there's no external references to this policy
        weak_self = weakref.ref(self)
        first_chunk = True
        while True:
            self = weak_self()
            if not self or self.should_exit:
                print("AsyncChunkingPolicy thread exiting to allow GC")
                break

            # Check if we need to reset
            # This needs to happen before checking if the queue is empty, since
            # we might still have actions in the queue when we are reset
            if self.is_resetting:
                reset_info = self.policy.reset()
                first_chunk = True
                with self.lock:
                    self.last_obs = None
                    self._last_raw_observation = None
                    self._last_executed_action = None
                    self.action_queue = deque()
                    self.last_info = {}
                    self.reset_info = reset_info
                    self._obs_event.clear()
                    self._pipeline_priming = True
                    self.is_resetting = False

            if self._needs_replan:
                with self.lock:
                    self.action_queue.clear()
                    self._needs_replan = False
                first_chunk = True

            if len(self.action_queue) > self.policy_latency_steps:
                time.sleep(0.001)
                continue

            last_obs = self.last_obs

            if not last_obs:
                print("AsyncChunkingPolicy waiting for observations")
                # Allow self to get eventually garbage collected
                obs_event = self._obs_event
                self = None
                obs_event.wait(timeout=1.0)  # wakes immediately when get_action sets obs
                obs_event.clear()
                continue

            # Apply torch.compile before first inference (must be in policy thread)
            if not self._compiled and self.use_torch_compile:
                self._apply_torch_compile()

            # Run forward pass outside lock
            forward_start = time.perf_counter()
            _action, info = self.policy.get_action(last_obs)
            forward_time_ms = (time.perf_counter() - forward_start) * 1000
            if "action_chunk" not in info:
                info["action_chunk"] = {
                    key: np.asarray(value)[None] for key, value in _action.items()
                }
            is_first_chunk = first_chunk
            if first_chunk:
                prefix_steps = 0
                first_chunk = False
            else:
                # Skip actions that overlap with what's still in the queue.
                # The +1 accounts for the action currently being executed.
                prefix_steps = self.policy_latency_steps + 1
            truncated_chunk = self._truncate_chunk(info["action_chunk"], prefix_steps=prefix_steps)
            truncated_info = info.copy()
            truncated_info["action_chunk"] = truncated_chunk
            truncated_info["forward_time_ms"] = forward_time_ms

            action_list = chunk_to_action_list(truncated_chunk)
            with self.lock:
                if self.use_chunk_smoothing and not is_first_chunk and action_list:
                    smoothed_action_list, overlap_len = _blend_action_lists(
                        list(self.action_queue),
                        action_list,
                        min_smooth_steps=self.min_smooth_steps,
                        fallback_old_action=self._last_executed_action,
                    )
                    if overlap_len > 0:
                        action_list = smoothed_action_list
                        truncated_info = truncated_info.copy()
                        truncated_info["action_chunk"] = _action_list_to_chunk(action_list)
                        truncated_info["chunk_smoothing_overlap_len"] = overlap_len
                        truncated_info["chunk_smoothing_enabled"] = True
                        self.action_queue = deque(action_list)
                    else:
                        self.action_queue.extend(action_list)
                else:
                    self.action_queue.extend(action_list)
                self.last_info = truncated_info
                self._pipeline_priming = False

    def shutdown(self):
        """Shutdown the policy thread.
        This function does not need to be called during process shutdown, since
        the thread is marked daemon and will automatically shutdown if the main
        thread is shutdown.
        """
        self.should_exit = True

    def __del__(self):
        """Make sure that we don't keep the policy thread running forever if
        this class is being repeatedly constructed and destructed.
        """
        self.shutdown()

    def _truncate_chunk(
        self, action_chunk: dict[str, Any], prefix_steps: int
    ) -> dict[str, np.ndarray]:
        """Truncate action chunk to respect action_exec_horizon."""
        truncated_chunk = {}
        for key, value in action_chunk.items():
            seq = np.asarray(value)
            # Handle both (H, D) and (B, H, D) shapes
            if seq.ndim == 3:
                truncated_chunk[key] = seq[:, prefix_steps : self.action_exec_horizon, :]
            elif seq.ndim == 2:
                truncated_chunk[key] = seq[prefix_steps : self.action_exec_horizon, :]
            else:
                truncated_chunk[key] = value
        return truncated_chunk
