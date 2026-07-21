"""Receding-horizon action chunking with optional async / RTC / smoothing.

Default mode (sync + receding horizon):
    Backend predicts ``action_horizon`` steps.  We execute the first
    ``replan_horizon`` steps, then replan.  No threading, no smoothing.

Advanced modes (wired but disabled by default):
    use_async           — background-thread inference, queue-based
    use_rtc             — real-time latency compensation with measured delay
    use_chunk_smoothing — linear-blend at chunk boundaries
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from enpire.env.forge.cap.policy.backend import PolicyBackend


@dataclass
class ChunkingConfig:
    """Full parameter set for the chunking pipeline.

    Only ``action_horizon`` and ``replan_horizon`` matter in sync mode.
    The rest are wired for future async / RTC activation.
    """

    # ---- core (always active) ----
    action_horizon: int = 16
    """Number of steps the model predicts per forward pass."""
    replan_horizon: int = 8
    """Execute this many steps from each chunk, then replan."""

    # ---- async inference (off by default) ----
    use_async: bool = False
    """Run model inference in a background thread."""
    max_get_action_seconds: float = 5.0
    """Timeout waiting for the background thread (async mode)."""

    # ---- real-time control / latency compensation (off by default) ----
    use_rtc: bool = False
    """Enable measured-latency compensation (RTC mode)."""
    control_hz: float = 20.0
    """Control loop frequency — used by RTC to convert latency → steps."""
    rtc_bootstrap_delay_steps: int = 4
    """Initial latency estimate (steps) before first measurement."""

    # ---- chunk smoothing (off by default) ----
    use_chunk_smoothing: bool = False
    """Linear-blend overlapping tails at chunk boundaries."""
    min_smooth_steps: int = 8
    """Minimum overlap length for the blend ramp."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_to_list(
    chunk: dict[str, np.ndarray], horizon: int
) -> list[dict[str, np.ndarray]]:
    """Convert ``{key: (H, D)}`` chunk to list of ``{key: (D,)}`` actions."""
    actions: list[dict[str, np.ndarray]] = []
    for step in range(horizon):
        action: dict[str, np.ndarray] = {}
        for k, v in chunk.items():
            if v.ndim >= 2:
                action[k] = v[step]
            else:
                action[k] = v.copy()
        actions.append(action)
    return actions


def _blend(
    old: list[dict],
    new: list[dict],
    min_steps: int,
    fallback: dict | None = None,
) -> tuple[list[dict], int]:
    """Linear-blend two action lists over their overlap region."""
    if not old and fallback is not None:
        old = [fallback] * min_steps
    elif 0 < len(old) < min_steps:
        old = list(old) + [old[-1]] * (min_steps - len(old))
    if not old or not new:
        return list(new), 0
    overlap = min(len(old), len(new))
    w_old = np.linspace(1.0, 0.0, overlap) if overlap > 1 else np.array([1.0])
    w_new = 1.0 - w_old
    blended: list[dict] = []
    for i in range(overlap):
        step: dict[str, np.ndarray] = {}
        for k in new[i]:
            step[k] = w_old[i] * np.asarray(old[i][k]) + w_new[i] * np.asarray(
                new[i][k]
            )
        blended.append(step)
    blended.extend(new[overlap:])
    return blended, overlap


# ---------------------------------------------------------------------------
# Main policy class
# ---------------------------------------------------------------------------


class ChunkingPolicy:
    """Receding-horizon chunking wrapper around any :class:`PolicyBackend`.

    Sync mode (default):
        Each ``get_action()`` call either pops from the queue or, when the
        queue is empty, calls ``backend.predict()`` to refill it.

    Async mode (``use_async=True``):
        A daemon thread calls ``backend.predict()`` in the background.
        ``get_action()`` pops from the queue with a timeout; if the queue
        is empty it returns a hold-action (last executed action).

    RTC mode (``use_rtc=True``):
        Extends async with measured-latency prefixing — sends the
        unexecuted tail of the current chunk as context so the model
        can anticipate what will execute during its own inference time.
        *Not yet implemented — raises NotImplementedError.*
    """

    def __init__(self, backend: PolicyBackend, config: ChunkingConfig | None = None):
        self.backend = backend
        self.cfg = config or ChunkingConfig()
        self._queue: deque[dict[str, np.ndarray]] = deque()
        self._last_action: dict[str, np.ndarray] | None = None

        # async state
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._obs_ready = threading.Event()
        self._last_obs: dict | None = None
        self._running = False

        # per-call timing (sync path only — async queues are backgrounded and
        # per-call timing is meaningless from the caller's POV).
        self._predict_ms: list[float] = []  # backend.predict durations (chunk refills)
        self._chunk_hit_ms: list[float] = []  # pop-only durations (no refill)

        if self.cfg.use_rtc:
            raise NotImplementedError(
                "RTC mode is wired but not yet implemented. "
                "Set use_rtc=False (default) or use_async=True."
            )
        if self.cfg.use_async:
            self._start_thread()

    # -- public API ----------------------------------------------------------

    def get_action(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return a single-step action ``{key: (D,)}``.

        In sync mode this is deterministic: predict when queue is empty.
        In async mode the background thread keeps the queue filled.
        """
        if self.cfg.use_async:
            return self._get_action_async(obs)
        return self._get_action_sync(obs)

    def reset(self) -> None:
        """Clear queue, reset backend."""
        with self._lock:
            self._queue.clear()
            self._last_action = None
        self.backend.reset()

    def close(self) -> None:
        """Stop async thread if running. Does NOT close the backend."""
        self._running = False
        if self._thread is not None:
            self._obs_ready.set()
            self._thread.join(timeout=3)

    # -- sync implementation -------------------------------------------------

    def _get_action_sync(self, obs: dict) -> dict[str, np.ndarray]:
        t_call = time.monotonic()
        if not self._queue:
            t_predict = time.monotonic()
            chunk = self.backend.predict(obs)
            self._predict_ms.append((time.monotonic() - t_predict) * 1000.0)

            horizon = min(self.cfg.replan_horizon, self._chunk_len(chunk))
            new_actions = _chunk_to_list(chunk, horizon)

            if self.cfg.use_chunk_smoothing and self._last_action is not None:
                old_tail = list(self._queue)
                new_actions, _ = _blend(
                    old_tail,
                    new_actions,
                    self.cfg.min_smooth_steps,
                    self._last_action,
                )
            self._queue.extend(new_actions)
        else:
            self._chunk_hit_ms.append((time.monotonic() - t_call) * 1000.0)

        action = self._queue.popleft()
        self._last_action = action
        return action

    def get_stats(self) -> dict[str, list[float]]:
        """Return per-call timing samples collected in sync mode.

        ``predict_ms``  — durations of ``backend.predict`` calls (one per
                          chunk refill; these include ZMQ round-trip for
                          remote backends like grootpool).
        ``chunk_hit_ms`` — duration of ``get_action`` calls that popped from
                          the queue without calling the backend.
        """
        return {
            "predict_ms": list(self._predict_ms),
            "chunk_hit_ms": list(self._chunk_hit_ms),
        }

    def reset_stats(self) -> None:
        self._predict_ms.clear()
        self._chunk_hit_ms.clear()

    # -- async implementation ------------------------------------------------

    def _start_thread(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._async_loop, daemon=True)
        self._thread.start()

    def _async_loop(self) -> None:
        latency_steps = max(
            1,
            self.cfg.action_horizon - self.cfg.replan_horizon - 1,
        )
        first_chunk = True
        while self._running:
            self._obs_ready.wait()
            if not self._running:
                break

            with self._lock:
                if len(self._queue) > latency_steps:
                    self._obs_ready.clear()
                    continue
                obs = self._last_obs

            if obs is None:
                time.sleep(0.001)
                continue

            chunk = self.backend.predict(obs)
            prefix = 0 if first_chunk else latency_steps + 1
            first_chunk = False
            horizon = min(self.cfg.replan_horizon, self._chunk_len(chunk))
            actions = _chunk_to_list(chunk, horizon)
            if prefix < len(actions):
                actions = actions[prefix:]

            with self._lock:
                if self.cfg.use_chunk_smoothing:
                    old_tail = list(self._queue)
                    actions, _ = _blend(
                        old_tail,
                        actions,
                        self.cfg.min_smooth_steps,
                        self._last_action,
                    )
                    self._queue.clear()
                self._queue.extend(actions)

            self._obs_ready.clear()

    def _get_action_async(self, obs: dict) -> dict[str, np.ndarray]:
        with self._lock:
            self._last_obs = obs
        self._obs_ready.set()

        deadline = time.monotonic() + self.cfg.max_get_action_seconds
        while time.monotonic() < deadline:
            with self._lock:
                if self._queue:
                    action = self._queue.popleft()
                    self._last_action = action
                    return action
            time.sleep(0.001)

        # timeout — hold last action
        if self._last_action is not None:
            return self._last_action
        raise TimeoutError("Async policy: no action available within timeout")

    # -- util ----------------------------------------------------------------

    @staticmethod
    def _chunk_len(chunk: dict[str, np.ndarray]) -> int:
        for v in chunk.values():
            if v.ndim >= 2:
                return v.shape[0]
        return 1
