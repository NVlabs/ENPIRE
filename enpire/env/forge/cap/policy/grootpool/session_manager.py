"""Per-request dispatcher — stateless replacement for the old session manager.

Old design: open() pins a worker for an entire episode (sticky session).
New design: each step() grabs any idle worker, runs inference, releases.
            No sessions, no session_ids, no stickiness.

N1.5 panda_omron is fully stateless (observation_indices=[0]), so
stickiness added zero model-correctness value.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from enpire.env.forge.cap.policy.grootpool import protocol as P
from enpire.env.forge.cap.policy.grootpool.supervisor import WorkerHandle, WorkerSupervisor

log = logging.getLogger(__name__)


class GrootPoolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class RequestDispatcher:
    """Routes each step request to any idle worker; releases immediately after."""

    def __init__(
        self,
        supervisor: WorkerSupervisor,
        *,
        step_timeout: float = 30.0,
        no_worker_timeout: float = 3600.0,
    ) -> None:
        self._sup = supervisor
        self._idle: asyncio.Queue[int] = asyncio.Queue()
        self._step_timeout = step_timeout
        self._no_worker_timeout = no_worker_timeout
        self._inflight: int = 0
        self._step_latencies: deque[float] = deque(maxlen=1024)

    async def start(self) -> None:
        for wid in self._sup.all_worker_ids():
            await self._idle.put(wid)

    async def stop(self) -> None:
        pass

    async def step(self, obs: dict) -> dict:
        """Grab idle worker → predict → release. Raises GrootPoolError on failure."""
        try:
            worker_id = await asyncio.wait_for(
                self._idle.get(), timeout=self._no_worker_timeout
            )
        except asyncio.TimeoutError:
            raise GrootPoolError(
                P.ERR_NO_WORKER,
                f"no idle worker after {self._no_worker_timeout:.0f}s",
            )

        self._inflight += 1
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        try:
            worker = self._sup.get(worker_id)
            if not worker.alive:
                raise GrootPoolError(P.ERR_WORKER_CRASH, f"worker {worker_id} not alive")

            try:
                action = await asyncio.wait_for(
                    loop.run_in_executor(None, worker.predict, obs),
                    timeout=self._step_timeout,
                )
            except asyncio.TimeoutError:
                log.error("worker %d step timeout", worker_id)
                raise GrootPoolError(
                    P.ERR_WORKER_TIMEOUT,
                    f"worker {worker_id} did not respond within {self._step_timeout:.0f}s",
                )
            except Exception as exc:
                log.error("worker %d step failed: %s", worker_id, exc)
                raise GrootPoolError(P.ERR_WORKER_CRASH, str(exc))
        finally:
            self._inflight -= 1
            # Return worker to idle (alive or not — supervisor handles respawn)
            try:
                await self._idle.put(worker_id)
            except Exception:
                pass
            dt_ms = (time.monotonic() - t0) * 1000.0
            self._step_latencies.append(dt_ms)

        return action

    async def on_worker_crash(self, worker_id: int) -> None:
        """Called by supervisor when a worker dies (noop — worker will be respawned)."""
        pass

    async def on_worker_respawned(self, worker_id: int) -> None:
        """Called by supervisor after a worker comes back up."""
        await self._idle.put(worker_id)

    def status_snapshot(self) -> dict:
        latencies = sorted(self._step_latencies)
        return {
            "inflight": self._inflight,
            "idle_workers": self._idle.qsize(),
            "step_p50_ms": _percentile(latencies, 50),
            "step_p95_ms": _percentile(latencies, 95),
        }


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    k = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * p / 100.0)))
    return round(sorted_values[k], 2)
