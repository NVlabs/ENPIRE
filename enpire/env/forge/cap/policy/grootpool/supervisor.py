"""Worker supervisor — spawn, health-check, and respawn GR00T workers.

For N1.5 we reuse ``cap/saved_scripts/robocasa/policy_eval/gr00t/_servers/n15.py``
unchanged. The subprocess lifecycle mirrors ``ModelServer365`` at
``cap/saved_scripts/robocasa/policy_eval/gr00t/_common/robocasa365.py:103-222``
but is lifted here so the middleware doesn't import from _saved_scripts_.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zmq

log = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    worker_id: int
    gpu_id: int
    port: int
    model_path: Path
    server_python: Path
    server_script: Path
    embodiment: str = "new_embodiment"
    startup_timeout: float = 300.0
    shutdown_timeout: float = 10.0
    call_timeout: float = 30.0
    log_path: Path | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    # If external=True the worker subprocess is launched elsewhere (e.g. by a
    # tmux script); WorkerHandle will not spawn or kill it, only connect.
    external: bool = False
    host: str = "127.0.0.1"


class WorkerHandle:
    """One GR00T subprocess + a sync REQ socket to talk to it.

    Sync methods are safe to call from an executor thread; middleware bridges
    to asyncio via ``loop.run_in_executor``.
    """

    def __init__(self, cfg: WorkerConfig) -> None:
        self.cfg = cfg
        self._proc: subprocess.Popen | None = None
        self._log_fh = None
        self._ctx: zmq.Context | None = None
        self._sock: zmq.Socket | None = None
        self._sock_lock = threading.Lock()
        self._alive = False
        self._started_at: float | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if not self.cfg.external:
            self._open_log()
            self._spawn()
        if not self._wait_ready():
            if not self.cfg.external:
                self._shutdown_proc()
                self._close_log()
            raise RuntimeError(
                f"[grootpool] worker {self.cfg.worker_id} "
                f"(gpu {self.cfg.gpu_id}, {self.cfg.host}:{self.cfg.port}) "
                f"failed to become ready within {self.cfg.startup_timeout:.0f}s"
            )
        self._connect_socket()
        self._alive = True
        self._started_at = time.monotonic()
        pid = self._proc.pid if self._proc else None
        log.info(
            "worker %d ready (gpu=%d %s:%d pid=%s %s)",
            self.cfg.worker_id,
            self.cfg.gpu_id,
            self.cfg.host,
            self.cfg.port,
            pid if pid else "—",
            "external" if self.cfg.external else "spawned",
        )

    def stop(self) -> None:
        self._alive = False
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass
            self._sock = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None
        if not self.cfg.external:
            self._shutdown_proc()
            self._close_log()

    # -- health --------------------------------------------------------------

    @property
    def alive(self) -> bool:
        if not self._alive:
            return False
        # External workers: we don't own the subprocess, so rely on the last
        # successful RPC flag. Health is detected on next call failure.
        if self.cfg.external:
            return True
        if self._proc is None or self._proc.poll() is not None:
            self._alive = False
            return False
        return True

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    # -- inference -----------------------------------------------------------

    def predict(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Single-obs → un-batched action chunk.

        Mirrors ``N15PolicyBackend.predict`` at
        ``cap/saved_scripts/robocasa/policy_eval/gr00t/_workers/robocasa365.py:39-55``
        so we match the same wire format expected by the N1.5 server.
        """
        batched = _add_batch(obs)
        with self._sock_lock:
            # gr00t's BaseInferenceServer passes request["data"] DIRECTLY to the
            # handler (gr00t/eval/service.py::run line 114). Pass batched as-is;
            # don't wrap it in {"observation": batched}.
            action = self._torch_rpc_data("get_action", batched)
        return _remove_batch(action)

    def reset(self) -> None:
        try:
            with self._sock_lock:
                self._torch_rpc_data("reset", None)
        except Exception as exc:
            log.warning("worker %d reset failed: %s", self.cfg.worker_id, exc)

    # -- internals -----------------------------------------------------------

    def _open_log(self) -> None:
        if self.cfg.log_path is None:
            return
        self.cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.cfg.log_path, "w")

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            finally:
                self._log_fh = None

    def _spawn(self) -> None:
        env = {
            **os.environ,
            **self.cfg.extra_env,
            "CUDA_VISIBLE_DEVICES": str(self.cfg.gpu_id),
        }
        cmd = [
            str(self.cfg.server_python),
            str(self.cfg.server_script),
            "--model-path",
            str(self.cfg.model_path),
            "--embodiment-tag",
            self.cfg.embodiment,
            "--port",
            str(self.cfg.port),
        ]
        log.info(
            "spawning worker %d: gpu=%d port=%d",
            self.cfg.worker_id,
            self.cfg.gpu_id,
            self.cfg.port,
        )
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_fh or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=env,
        )

    def _wait_ready(self) -> bool:
        import torch  # only needed for real workers; middleware venv has it

        ctx = zmq.Context()
        deadline = time.monotonic() + self.cfg.startup_timeout
        try:
            while time.monotonic() < deadline:
                # Only managed workers have a proc we can poll for death.
                if (
                    not self.cfg.external
                    and self._proc is not None
                    and self._proc.poll() is not None
                ):
                    return False
                sock = ctx.socket(zmq.REQ)
                sock.setsockopt(zmq.RCVTIMEO, 5000)
                sock.setsockopt(zmq.LINGER, 0)
                try:
                    sock.connect(f"tcp://{self.cfg.host}:{self.cfg.port}")
                    buf = io.BytesIO()
                    torch.save({"endpoint": "ping"}, buf)
                    sock.send(buf.getvalue())
                    resp = torch.load(io.BytesIO(sock.recv()), weights_only=False)
                    if isinstance(resp, dict) and resp.get("status") == "ok":
                        return True
                except Exception:
                    pass
                finally:
                    sock.close()
                time.sleep(2)
        finally:
            ctx.term()
        return False

    def _connect_socket(self) -> None:
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, int(self.cfg.call_timeout * 1000))
        self._sock.connect(f"tcp://{self.cfg.host}:{self.cfg.port}")

    def _torch_rpc_data(self, endpoint: str, data: Any) -> Any:
        """Send one torch-serialized request.  Matches gr00t BaseInferenceClient's
        call_endpoint: wire format is ``{"endpoint": str, "data": <payload>}``
        where payload is passed verbatim to the server-side handler."""
        import torch

        req: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        buf = io.BytesIO()
        torch.save(req, buf)
        self._sock.send(buf.getvalue())
        resp_bytes = self._sock.recv()
        resp = torch.load(io.BytesIO(resp_bytes), weights_only=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"gr00t server error: {resp['error']}")
        return resp

    def _shutdown_proc(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = None
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=self.cfg.shutdown_timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        finally:
            self._proc = None


class MockWorkerHandle:
    """In-process stub for tests — no subprocess, no torch, returns zero actions."""

    def __init__(self, cfg: WorkerConfig) -> None:
        self.cfg = cfg
        self._alive = False
        self._started_at: float | None = None

    def start(self) -> None:
        self._alive = True
        self._started_at = time.monotonic()
        log.info("mock worker %d ready", self.cfg.worker_id)

    def stop(self) -> None:
        self._alive = False

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def pid(self) -> int | None:
        return None

    def predict(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        # Return a plausible-looking action chunk. action_horizon=16, dim=7.
        return {
            "action.joint_position": np.zeros((16, 7), dtype=np.float32),
            "action.gripper": np.zeros((16, 1), dtype=np.float32),
        }

    def reset(self) -> None:
        pass


class WorkerSupervisor:
    """Owns a pool of workers, handles respawn on crash."""

    def __init__(
        self,
        configs: list[WorkerConfig],
        *,
        mock: bool = False,
        respawn: bool = True,
        on_crash=None,
        on_respawn=None,
    ) -> None:
        self._configs = configs
        self._mock = mock
        self._respawn = respawn
        self._on_crash = on_crash
        self._on_respawn = on_respawn
        self._workers: dict[int, WorkerHandle | MockWorkerHandle] = {}
        self._respawning: set[int] = set()
        self._monitor_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        handles = []
        for cfg in self._configs:
            handles.append((cfg.worker_id, self._make_handle(cfg)))
        # Spawn all workers in parallel via executor (each .start blocks up to 5 min).
        await asyncio.gather(*[loop.run_in_executor(None, h.start) for _, h in handles])
        for wid, h in handles:
            self._workers[wid] = h
        self._monitor_task = asyncio.create_task(self._monitor())

    async def stop(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except (asyncio.CancelledError, Exception):
                pass
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *[loop.run_in_executor(None, w.stop) for w in self._workers.values()],
            return_exceptions=True,
        )
        self._workers.clear()

    def get(self, worker_id: int) -> WorkerHandle | MockWorkerHandle:
        return self._workers[worker_id]

    def all_worker_ids(self) -> list[int]:
        return sorted(self._workers.keys())

    def status_snapshot(self) -> list[dict]:
        out = []
        for wid in self.all_worker_ids():
            w = self._workers[wid]
            out.append(
                {
                    "id": wid,
                    "gpu": w.cfg.gpu_id,
                    "port": w.cfg.port,
                    "pid": w.pid,
                    "alive": w.alive,
                    "respawning": wid in self._respawning,
                }
            )
        return out

    # -- internals -----------------------------------------------------------

    def _make_handle(self, cfg: WorkerConfig):
        return MockWorkerHandle(cfg) if self._mock else WorkerHandle(cfg)

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            for wid, worker in list(self._workers.items()):
                if not worker.alive and wid not in self._respawning:
                    log.warning("worker %d died, scheduling respawn", wid)
                    self._respawning.add(wid)
                    if self._on_crash is not None:
                        try:
                            await self._on_crash(wid)
                        except Exception:
                            log.exception("on_crash hook failed for worker %d", wid)
                    if self._respawn:
                        asyncio.create_task(self._respawn_worker(wid))
                    else:
                        self._respawning.discard(wid)

    async def _respawn_worker(self, wid: int) -> None:
        loop = asyncio.get_running_loop()
        try:
            old = self._workers.get(wid)
            cfg = next(c for c in self._configs if c.worker_id == wid)
            if cfg.external:
                # Don't try to respawn something we don't own — operator must
                # restart the external tmux window.
                log.warning("external worker %d crashed; will retry connect", wid)
                if old is not None:
                    await loop.run_in_executor(None, old.stop)
                new = self._make_handle(cfg)
                await loop.run_in_executor(None, new.start)
                self._workers[wid] = new
                log.info("external worker %d reconnected", wid)
                if self._on_respawn is not None:
                    await self._on_respawn(wid)
                return
            if old is not None:
                await loop.run_in_executor(None, old.stop)
            new = self._make_handle(cfg)
            await loop.run_in_executor(None, new.start)
            self._workers[wid] = new
            log.info("worker %d respawned", wid)
            if self._on_respawn is not None:
                try:
                    await self._on_respawn(wid)
                except Exception:
                    log.exception("on_respawn hook failed for worker %d", wid)
        except Exception:
            log.exception("respawn failed for worker %d", wid)
        finally:
            self._respawning.discard(wid)


# -- batch helpers (mirror ZMQPolicyBackend._add_batch / _remove_batch) -------


def _add_batch(obs: dict) -> dict:
    """Add B=1, T=1 dims expected by the GR00T server."""
    out: dict[str, Any] = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v[np.newaxis, np.newaxis]
        elif isinstance(v, str):
            out[k] = np.array([v])
        else:
            out[k] = v
    return out


def _remove_batch(action: dict) -> dict[str, np.ndarray]:
    return {
        k: (v[0] if hasattr(v, "ndim") and v.ndim > 0 else v) for k, v in action.items()
    }
