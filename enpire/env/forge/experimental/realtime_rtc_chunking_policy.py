from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import threading
import time
from typing import Any
import warnings

import numpy as np

from enpire.env.forge.experimental.get_action_policy import chunk_to_action_list
from enpire.env.forge.experimental.key_remapping_utils import hold_action_from_proprio


def _clone_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in chunk.items():
        if isinstance(value, np.ndarray):
            cloned[key] = np.asarray(value).copy()
        elif isinstance(value, dict):
            cloned[key] = _clone_chunk(value)
        else:
            cloned[key] = value
    return cloned


def _slice_chunk(chunk: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in chunk.items():
        if isinstance(value, dict):
            sliced[key] = _slice_chunk(value, start, end)
            continue
        seq = np.asarray(value)
        if seq.ndim == 3:
            sliced[key] = seq[:, start:end, :].copy()
        elif seq.ndim == 2:
            sliced[key] = seq[start:end, :].copy()
        else:
            sliced[key] = seq.copy()
    return sliced


def _truncate_chunk(chunk: Mapping[str, Any], start: int, end: int | None = None) -> dict[str, Any]:
    if end is None:
        end = 1_000_000_000
    return _slice_chunk(chunk, start, end)


def _clone_action(action: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in action.items()}


def _action_list_to_chunk(actions: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    chunk: dict[str, list[np.ndarray]] = {}
    for action in actions:
        for key, value in action.items():
            chunk.setdefault(key, []).append(np.asarray(value).copy())
    return {key: np.stack(values, axis=0) for key, values in chunk.items()}


def _blend_action_lists(
    old_list: list[dict[str, np.ndarray]],
    new_list: list[dict[str, np.ndarray]],
    *,
    min_smooth_steps: int,
    fallback_old_action: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, np.ndarray]], int]:
    """Blend old and new unexecuted actions with a linear overlap ramp."""
    old_local = [_clone_action(action) for action in old_list]
    new_local = [_clone_action(action) for action in new_list]

    if len(old_local) == 0 and fallback_old_action is not None:
        old_local = [_clone_action(fallback_old_action) for _ in range(min_smooth_steps)]
    elif 0 < len(old_local) < min_smooth_steps:
        pad = [_clone_action(old_local[-1]) for _ in range(min_smooth_steps - len(old_local))]
        old_local.extend(pad)

    if len(old_local) == 0 or len(new_local) == 0:
        return new_local, 0

    if len(old_local) > len(new_local):
        old_local = old_local[: len(new_local)]

    overlap_len = min(len(old_local), len(new_local))
    if overlap_len == 1:
        w_old = np.array([1.0], dtype=np.float64)
    else:
        w_old = np.linspace(1.0, 0.0, overlap_len, dtype=np.float64)
    w_new = 1.0 - w_old

    blended: list[dict[str, np.ndarray]] = []
    for i in range(overlap_len):
        blended_action: dict[str, np.ndarray] = {}
        for key in new_local[i]:
            old_val = np.asarray(old_local[i].get(key, new_local[i][key]), dtype=np.float64)
            new_val = np.asarray(new_local[i][key], dtype=np.float64)
            blended_action[key] = w_old[i] * old_val + w_new[i] * new_val
        blended.append(blended_action)

    blended.extend(new_local[overlap_len:])
    return blended, overlap_len


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return np.asarray(obj).tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


class RealtimeRTCChunkingPolicy:
    """Asynchronous chunking wrapper with client-owned realtime RTC overlap.

    The wrapper keeps executing the current chunk until ``replan_horizon``.
    At that point it asynchronously requests the next chunk using the current
    observation and an RTC prefix taken from the currently-active chunk:

        prefix = current_chunk[replan_horizon : replan_horizon + d_est]

    where ``d_est`` is derived from the previous measured end-to-end client
    inference latency.
    """

    def __init__(
        self,
        policy,
        *,
        action_horizon: int,
        replan_horizon: int,
        bootstrap_delay_steps: int,
        max_delay_steps: int,
        control_hz: float,
        max_get_action_seconds: float,
        require_prev_observation: bool = False,
        use_chunk_smoothing: bool = False,
        min_smooth_steps: int = 8,
        debug_log_path: str | None = None,
    ):
        if replan_horizon <= 0:
            raise ValueError(f"replan_horizon must be > 0, got {replan_horizon}")
        if replan_horizon >= action_horizon:
            raise ValueError(
                "Realtime RTC requires replan_horizon < action_horizon, got "
                f"{replan_horizon} >= {action_horizon}"
            )

        self.realtime_rtc_enabled = True
        self.policy = policy
        self.action_horizon = int(action_horizon)
        self.replan_horizon = int(replan_horizon)
        self.bootstrap_delay_steps = max(0, int(bootstrap_delay_steps))
        self.max_delay_steps = max(0, min(int(max_delay_steps), self.action_horizon - self.replan_horizon))
        self.control_hz = float(control_hz)
        self.max_get_action_seconds = float(max_get_action_seconds)
        self.require_prev_observation = require_prev_observation
        self.use_chunk_smoothing = bool(use_chunk_smoothing)
        self.min_smooth_steps = max(1, int(min_smooth_steps))

        self._debug_log_path = Path(debug_log_path).expanduser() if debug_log_path else None
        self._debug_log_lock = threading.Lock()
        self._step_counter = 0
        if self._debug_log_path is not None:
            self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._debug_log_path.write_text("")

        self._condition = threading.Condition()
        self._should_exit = False
        self._is_resetting = False
        self._reset_info: dict[str, Any] | None = None

        self._last_obs: dict[str, Any] | None = None
        self._last_raw_observation: dict[str, Any] | None = None
        self._last_executed_action: dict[str, np.ndarray] | None = None

        self._active_chunk: dict[str, Any] | None = None
        self._active_actions: list[dict[str, np.ndarray]] = []
        self._active_exec_index = 0
        self._active_info: dict[str, Any] = {}
        self._active_request_id: int | None = None
        self._active_switch_index: int | None = None
        self._active_switch_warning_emitted = False

        self._ready_successor: dict[str, Any] | None = None
        self._pending_request: dict[str, Any] | None = None
        self._inflight_request_id: int | None = None

        self._request_counter = 0
        self._generation = 0
        self._next_delay_steps_estimate = self.max_delay_steps
        if self.max_delay_steps > 0:
            self._next_delay_steps_estimate = min(self.bootstrap_delay_steps, self.max_delay_steps)

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        with self._condition:
            prev_raw = self._last_raw_observation
            self._last_raw_observation = observation
            if self.require_prev_observation:
                if prev_raw is None:
                    self._last_obs = None
                else:
                    payload = dict(observation)
                    payload["__prev_observation"] = prev_raw
                    self._last_obs = payload
            else:
                self._last_obs = observation

            if self.require_prev_observation and prev_raw is None:
                hold_action = hold_action_from_proprio(observation)
                self._log_event(
                    "history_priming_hold",
                    action=hold_action,
                )
                return hold_action, {
                    "remaining_num_action_in_chunk": 0,
                    "history_priming": True,
                }

            self._activate_ready_successor_locked()
            self._ensure_bootstrap_request_locked()
            self._maybe_schedule_replan_locked(self._last_obs)

            start = time.monotonic()
            while True:
                self._activate_ready_successor_locked()
                self._maybe_schedule_replan_locked(self._last_obs)
                if self._active_chunk is not None and self._active_exec_index < len(self._active_actions):
                    break
                now = time.monotonic()
                if now - start > self.max_get_action_seconds:
                    warnings.warn(
                        "RealtimeRTCChunkingPolicy timed out waiting for the next chunk. "
                        "Returning a hold action."
                    )
                    hold_action = hold_action_from_proprio(observation)
                    self._log_event(
                        "timeout_hold",
                        max_get_action_seconds=self.max_get_action_seconds,
                        action=hold_action,
                    )
                    return hold_action, {"rtc_timeout": True}
                if not self._worker.is_alive():
                    raise ValueError("RealtimeRTCChunkingPolicy worker thread is not alive")
                self._condition.wait(timeout=0.01)

            if (
                self._active_switch_index is not None
                and self._active_exec_index >= self._active_switch_index
                and self._ready_successor is None
                and not self._active_switch_warning_emitted
            ):
                self._active_switch_warning_emitted = True
                warnings.warn(
                    "Realtime RTC reached the planned switch point before the next "
                    "chunk arrived; continuing to execute the old chunk."
                )

            action = {
                key: np.asarray(value).copy()
                for key, value in self._active_actions[self._active_exec_index].items()
            }
            info = self._build_info_locked()
            self._active_exec_index += 1
            self._step_counter += 1
            if self.use_chunk_smoothing:
                self._last_executed_action = _clone_action(action)
            self._log_event(
                "step_executed",
                step_id=self._step_counter,
                active_request_id=self._active_request_id,
                chunk_local_index=self._active_exec_index - 1,
                remaining_num_action_in_chunk=info["remaining_num_action_in_chunk"],
                action=action,
                rtc_sent_prefix_steps=info.get("rtc_sent_prefix_steps"),
                rtc_client_latency_steps=info.get("rtc_client_latency_steps"),
            )
            if self._active_exec_index == self.replan_horizon:
                self._maybe_schedule_replan_locked(self._last_obs)
            return action, info

    def reset(self) -> dict[str, Any] | None:
        with self._condition:
            self._generation += 1
            self._is_resetting = True
            self._condition.notify_all()

            start_time = time.monotonic()
            while self._is_resetting:
                if not self._worker.is_alive():
                    raise ValueError("RealtimeRTCChunkingPolicy worker thread is not alive")
                if time.monotonic() > start_time + 1.0:
                    print(f"Waiting for {self.policy}.reset()")
                    start_time = time.monotonic()
                self._condition.wait(timeout=0.05)
            return self._reset_info

    def shutdown(self) -> None:
        with self._condition:
            if self._should_exit:
                return
            self._should_exit = True
            self._condition.notify_all()
        self._log_event("shutdown")

    def __del__(self):
        self.shutdown()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._should_exit and not self._is_resetting and self._pending_request is None:
                    self._condition.wait(timeout=0.1)

                if self._should_exit:
                    return

                if self._is_resetting:
                    generation = self._generation
                    request = None
                else:
                    request = self._pending_request
                    self._pending_request = None
                    assert request is not None
                    self._inflight_request_id = int(request["request_id"])
                    generation = int(request["generation"])

            if request is None:
                reset_info = self.policy.reset()
                with self._condition:
                    if generation == self._generation:
                        self._last_obs = None
                        self._last_raw_observation = None
                        self._last_executed_action = None
                        self._active_chunk = None
                        self._active_actions = []
                        self._active_exec_index = 0
                        self._active_info = {}
                        self._active_request_id = None
                        self._active_switch_index = None
                        self._active_switch_warning_emitted = False
                        self._ready_successor = None
                        self._pending_request = None
                        self._inflight_request_id = None
                        self._reset_info = reset_info
                        self._is_resetting = False
                        self._condition.notify_all()
                        self._log_event(
                            "reset_completed",
                            generation=self._generation,
                            reset_info=reset_info,
                        )
                continue

            policy_observation = request["observation"]
            request_id = int(request["request_id"])
            d_sent = int(request["d_sent"])

            forward_start = time.perf_counter()
            action, info = self.policy.get_action(policy_observation)
            latency_s = time.perf_counter() - forward_start

            if "action_chunk" not in info:
                info = dict(info)
                info["action_chunk"] = {key: np.asarray(value)[None] for key, value in action.items()}
            else:
                info = dict(info)

            response = {
                "request_id": request_id,
                "generation": generation,
                "chunk": _clone_chunk(info["action_chunk"]),
                "actions": chunk_to_action_list(info["action_chunk"]),
                "info": info,
                "latency_s": latency_s,
                "latency_ms": latency_s * 1000.0,
                "latency_steps_measured": self._latency_to_steps(latency_s),
                "d_sent": d_sent,
            }

            with self._condition:
                if generation != self._generation or self._is_resetting:
                    if self._inflight_request_id == request_id:
                        self._inflight_request_id = None
                    self._condition.notify_all()
                    self._log_event(
                        "request_discarded",
                        request_id=request_id,
                        generation=generation,
                        d_sent=d_sent,
                    )
                    continue

                self._next_delay_steps_estimate = response["latency_steps_measured"]
                self._inflight_request_id = None
                if self._active_chunk is None:
                    self._activate_response_locked(response)
                else:
                    self._ready_successor = response
                self._log_event(
                    "request_completed",
                    request_id=request_id,
                    generation=generation,
                    d_sent=d_sent,
                    latency_ms=response["latency_ms"],
                    latency_steps_measured=response["latency_steps_measured"],
                    returned_chunk_len=len(response["actions"]),
                    chunk_preview=self._chunk_preview(response["chunk"]),
                )
                self._condition.notify_all()

    def _ensure_bootstrap_request_locked(self) -> None:
        if self._active_chunk is not None:
            return
        if self._pending_request is not None or self._inflight_request_id is not None:
            return
        if self._last_obs is None:
            return
        request_observation = dict(self._last_obs)
        rtc_payload = {
            "enabled": True,
            "prefix_steps": 0,
        }
        request_observation["__rtc"] = rtc_payload
        self._schedule_request_locked(
            observation=request_observation,
            d_sent=0,
            rtc_payload=rtc_payload,
            reason="bootstrap",
        )

    def _maybe_schedule_replan_locked(self, observation: dict[str, Any] | None) -> None:
        if observation is None or self._active_chunk is None:
            return
        if self._pending_request is not None or self._inflight_request_id is not None or self._ready_successor is not None:
            return
        if self._active_exec_index < self.replan_horizon:
            return

        d_sent = self._effective_delay_steps_locked()
        rtc_payload = None
        if d_sent > 0:
            prefix_end = min(self.action_horizon, self.replan_horizon + d_sent)
            prefix_chunk = _slice_chunk(self._active_chunk, self.replan_horizon, prefix_end)
            rtc_payload = {
                "enabled": True,
                "prefix_steps": d_sent,
                "prefix_action_chunk": prefix_chunk,
            }

        request_observation = dict(observation)
        if rtc_payload is not None:
            request_observation["__rtc"] = rtc_payload
        self._schedule_request_locked(
            observation=request_observation,
            d_sent=d_sent,
            rtc_payload=rtc_payload,
            reason="replan",
        )
        self._active_switch_index = min(len(self._active_actions), self.replan_horizon + d_sent)
        self._active_switch_warning_emitted = False

    def _schedule_request_locked(
        self,
        *,
        observation: dict[str, Any],
        d_sent: int,
        rtc_payload: dict[str, Any] | None,
        reason: str,
    ) -> None:
        self._request_counter += 1
        if rtc_payload is not None:
            rtc_payload["request_id"] = self._request_counter
        self._pending_request = {
            "request_id": self._request_counter,
            "generation": self._generation,
            "observation": observation,
            "d_sent": d_sent,
            "rtc_payload": rtc_payload,
        }
        self._log_event(
            "request_scheduled",
            request_id=self._request_counter,
            generation=self._generation,
            reason=reason,
            active_request_id=self._active_request_id,
            active_exec_index=self._active_exec_index,
            replan_horizon=self.replan_horizon,
            d_sent=d_sent,
            rtc_payload=rtc_payload,
            active_chunk_len=len(self._active_actions),
        )
        self._condition.notify_all()

    def _activate_ready_successor_locked(self) -> None:
        if self._ready_successor is None:
            return
        if self._active_chunk is None:
            self._activate_response_locked(self._ready_successor)
            self._ready_successor = None
            return
        if self._active_switch_index is None:
            return
        if self._active_exec_index < self._active_switch_index:
            return
        response = self._ready_successor
        if self.use_chunk_smoothing:
            response = self._maybe_smooth_successor_locked(response)
        self._activate_response_locked(response)
        self._ready_successor = None

    def _activate_response_locked(self, response: dict[str, Any]) -> None:
        self._active_chunk = response["chunk"]
        self._active_actions = response["actions"]
        start_exec_index = int(response.get("start_exec_index", response["d_sent"]))
        self._active_exec_index = min(start_exec_index, len(self._active_actions))
        self._active_request_id = int(response["request_id"])
        self._active_switch_index = None
        self._active_switch_warning_emitted = False

        active_info = dict(response["info"])
        active_info["action_chunk"] = _clone_chunk(response["chunk"])
        active_info["forward_time_ms"] = float(response["latency_ms"])
        active_info["rtc_client_latency_ms"] = float(response["latency_ms"])
        active_info["rtc_client_latency_steps"] = int(response["latency_steps_measured"])
        active_info["rtc_sent_prefix_steps"] = int(response["d_sent"])
        if "chunk_smoothing_overlap_len" in response:
            active_info["chunk_smoothing_overlap_len"] = int(
                response["chunk_smoothing_overlap_len"]
            )
            active_info["chunk_smoothing_enabled"] = True
        self._active_info = active_info
        self._log_event(
            "request_activated",
            request_id=self._active_request_id,
            start_exec_index=self._active_exec_index,
            chunk_len=len(self._active_actions),
            d_sent=response["d_sent"],
            chunk_preview=self._chunk_preview(response["chunk"]),
        )
        if len(self._active_actions) <= self.replan_horizon:
            warnings.warn(
                "Realtime RTC received a chunk whose length is not larger than "
                "replan_horizon. Make sure the serving-side RTC/full-horizon mode "
                "is enabled before using --use-realtime-rtc."
            )
            self._log_event(
                "chunk_too_short_warning",
                request_id=self._active_request_id,
                chunk_len=len(self._active_actions),
                replan_horizon=self.replan_horizon,
            )

    def _maybe_smooth_successor_locked(self, response: dict[str, Any]) -> dict[str, Any]:
        old_future_actions = self._active_actions[self._active_exec_index :]
        start_exec_index = min(int(response["d_sent"]), len(response["actions"]))
        skipped_prefix = [_clone_action(action) for action in response["actions"][:start_exec_index]]
        new_future_actions = response["actions"][start_exec_index:]
        blended_future_actions, overlap_len = _blend_action_lists(
            old_future_actions,
            new_future_actions,
            min_smooth_steps=self.min_smooth_steps,
            fallback_old_action=self._last_executed_action,
        )
        if overlap_len <= 0:
            return response

        smoothed_response = dict(response)
        smoothed_actions = skipped_prefix + blended_future_actions
        smoothed_response["actions"] = smoothed_actions
        smoothed_response["chunk"] = _action_list_to_chunk(smoothed_actions)
        smoothed_response["chunk_smoothing_overlap_len"] = overlap_len
        self._log_event(
            "successor_smoothed",
            request_id=int(response["request_id"]),
            active_request_id=self._active_request_id,
            old_future_len=len(old_future_actions),
            new_future_len=len(new_future_actions),
            overlap_len=overlap_len,
            original_start_exec_index=start_exec_index,
        )
        return smoothed_response

    def _build_info_locked(self) -> dict[str, Any]:
        info = dict(self._active_info)
        if self._active_chunk is not None:
            info["action_chunk"] = _truncate_chunk(
                self._active_chunk,
                self._active_exec_index,
                len(self._active_actions),
            )
        remaining_after_current = max(0, len(self._active_actions) - (self._active_exec_index + 1))
        info["remaining_num_action_in_chunk"] = remaining_after_current
        return info

    def _effective_delay_steps_locked(self) -> int:
        if self.max_delay_steps <= 0:
            return 0
        d_est = int(self._next_delay_steps_estimate)
        d_est = max(0, d_est)
        d_est = min(d_est, self.max_delay_steps)
        d_est = min(d_est, self.action_horizon - self.replan_horizon)
        return d_est

    def _latency_to_steps(self, latency_s: float) -> int:
        if self.max_delay_steps <= 0:
            return 0
        return min(
            int(np.ceil(max(0.0, latency_s) * self.control_hz)),
            self.max_delay_steps,
        )

    def _chunk_preview(self, chunk: Mapping[str, Any]) -> dict[str, Any]:
        preview: dict[str, Any] = {}
        for key, value in chunk.items():
            if isinstance(value, dict):
                preview[key] = self._chunk_preview(value)
                continue
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 2:
                preview[key] = {
                    "shape": list(arr.shape),
                    "first_dim_series": arr[:, 0].tolist(),
                }
            else:
                preview[key] = {
                    "shape": list(arr.shape),
                    "value": _jsonable(arr),
                }
        return preview

    def _log_event(self, event: str, **payload: Any) -> None:
        if self._debug_log_path is None:
            return
        record = {
            "event": event,
            "wall_time": time.time(),
            **{key: _jsonable(value) for key, value in payload.items()},
        }
        with self._debug_log_lock:
            try:
                with self._debug_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
            except FileNotFoundError:
                return
