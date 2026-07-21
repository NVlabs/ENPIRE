"""Shared constants and utilities for GR00T N1.6 eval scripts.

All external paths are configured via environment variables.  Set them in
your shell profile or a .env wrapper script before running the eval:

  export GROOT_ROOT=/path/to/Isaac-GR00T
  export GROOT_MODEL_PATH=/path/to/GR00T-N1.6-3B
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import msgpack
import zmq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parents[6]
LOG_ROOT = REPO_DIR / "logs" / "GR00TN1.6"


GROOT_ROOT = Path(os.environ.get(
    "GROOT_ROOT",
    "",
))
MODEL_PATH = Path(os.environ.get(
    "GROOT_MODEL_PATH",
    "",
))

SERVER_PYTHON = GROOT_ROOT / "gr00t/eval/sim/robocasa/model_server_venv/bin/python"
SERVER_SCRIPT = GROOT_ROOT / "gr00t/eval/run_gr00t_server.py"
CLIENT_PYTHON = GROOT_ROOT / "gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
EMBODIMENT = "ROBOCASA_PANDA_OMRON"


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class ModelServer:
    """Context manager for a GR00T inference server subprocess.

    Owns the subprocess and log file handle. On exit: SIGTERM, wait,
    then SIGKILL if stuck. Checks proc.poll() during startup for fast
    failure if the server crashes during model loading.
    """

    def __init__(
        self,
        gpu_id: int,
        port: int,
        log_path: Path,
        *,
        startup_timeout: float = 180.0,
        shutdown_timeout: float = 10.0,
    ) -> None:
        self._gpu_id = gpu_id
        self._port = port
        self._log_path = log_path
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._proc: subprocess.Popen | None = None
        self._log_file = None

    def __enter__(self) -> ModelServer:
        self._log_file = open(self._log_path, "w")
        try:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self._gpu_id)}
            self._proc = subprocess.Popen(
                [
                    str(SERVER_PYTHON),
                    str(SERVER_SCRIPT),
                    "--model-path",
                    str(MODEL_PATH),
                    "--embodiment-tag",
                    EMBODIMENT,
                    "--use-sim-policy-wrapper",
                    "--port",
                    str(self._port),
                ],
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            if not self._wait_for_ready():
                self._shutdown()
                raise RuntimeError(
                    f"Server on GPU {self._gpu_id} port {self._port} "
                    f"failed to start within {self._startup_timeout}s"
                )
        except BaseException:
            self._log_file.close()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self._shutdown()
        finally:
            self._log_file.close()

    @property
    def port(self) -> int:
        return self._port

    def _wait_for_ready(self) -> bool:
        ctx = zmq.Context()
        deadline = time.monotonic() + self._startup_timeout
        try:
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    return False
                sock = ctx.socket(zmq.REQ)
                sock.setsockopt(zmq.RCVTIMEO, 2000)
                sock.setsockopt(zmq.LINGER, 0)
                try:
                    sock.connect(f"tcp://127.0.0.1:{self._port}")
                    sock.send(msgpack.packb({"endpoint": "ping"}))
                    resp = msgpack.unpackb(sock.recv(), raw=False)
                    if resp.get("status") == "ok":
                        return True
                except Exception:
                    pass
                finally:
                    sock.close()
                time.sleep(2)
        finally:
            ctx.term()
        return False

    def _shutdown(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=self._shutdown_timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TeeStream:
    """Write to both a file and the original stream (real-time tee)."""

    def __init__(self, stream, log_path: Path) -> None:
        self._stream = stream
        self._file = open(log_path, "w")

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()
