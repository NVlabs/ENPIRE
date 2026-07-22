# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants and utilities for RoboCasa365 benchmark eval scripts.

Separate from _common.py to avoid conflicts with the PandaOmron24 pipeline.
Supports both GR00T N1.5 (benchmark repo) and N1.6 (existing repo).

All external paths are configured via environment variables. Set them in
your shell profile or a .env wrapper script before running the eval:

  export GROOT_ROOT=/path/to/Isaac-GR00T
  export GROOT_MODEL_PATH=/path/to/GR00T-N1.6-3B
  export GROOT_BENCHMARK_ROOT=/path/to/Isaac-GR00T-benchmark
  export GROOT_N15_MODEL_PATH=/path/to/gr00t_n1-5/multitask_learning/checkpoint-120000
  export ROBOCASA365_ROOT=/path/to/robocasa365
  export ROBOCASA365_CLIENT_PYTHON=/path/to/python  # optional worker override
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
# Path helpers
# ---------------------------------------------------------------------------



N16_GROOT_ROOT = Path(os.environ.get(
    "GROOT_ROOT",
    "",
))
N16_MODEL_PATH = Path(os.environ.get(
    "GROOT_MODEL_PATH",
    "",
))
N16_SERVER_PYTHON = N16_GROOT_ROOT / "gr00t/eval/sim/robocasa/model_server_venv/bin/python"
N16_SERVER_SCRIPT = N16_GROOT_ROOT / "gr00t/eval/run_gr00t_server.py"
N16_EMBODIMENT = "ROBOCASA_PANDA_OMRON"

N15_GROOT_ROOT = Path(os.environ.get(
    "GROOT_BENCHMARK_ROOT",
    "",
))
N15_MODEL_PATH = Path(os.environ.get(
    "GROOT_N15_MODEL_PATH",
    "",
))
N15_SERVER_PYTHON = N15_GROOT_ROOT / "model_server_venv/bin/python"
N15_EMBODIMENT = "new_embodiment"

ROBOCASA365_ROOT = Path(os.environ.get(
    "ROBOCASA365_ROOT",
    "",
))
CLIENT_PYTHON_365 = N15_GROOT_ROOT / "client_venv/bin/python"
N15_SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "_servers" / "n15.py"

CLIENT_SCRIPT_365 = Path(__file__).resolve().parent.parent / "_workers" / "robocasa365.py"

REPO_DIR = Path(__file__).resolve().parents[6]
LOG_ROOT = REPO_DIR / "logs" / "robocasa365"


def get_client_python_365(client_python: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the RoboCasa365 worker interpreter.

    Resolution order:
      1. explicit override (CLI / function arg)
      2. ROBOCASA365_CLIENT_PYTHON
      3. UV_PROJECT_ENVIRONMENT/bin/python
      4. repo-local .venv/bin/python (symlinked by setup_env_osmo.sh on OSMO)
      5. Isaac-GR00T-benchmark/client_venv/bin/python (legacy benchmark setup)
    """

    if client_python is not None:
        return Path(client_python)

    env_override = os.environ.get("ROBOCASA365_CLIENT_PYTHON")
    if env_override:
        return Path(env_override)

    uv_project_environment = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if uv_project_environment:
        uv_python = Path(uv_project_environment) / "bin/python"
        if uv_python.exists():
            return uv_python

    forge_python = REPO_DIR / ".venv" / "bin" / "python"
    if forge_python.exists():
        return forge_python

    return CLIENT_PYTHON_365


def get_model_config(model_version: str) -> dict:
    """Return paths and settings for the given model version."""
    if model_version == "n16":
        return {
            "server_python": N16_SERVER_PYTHON,
            "server_script": N16_SERVER_SCRIPT,
            "model_path": N16_MODEL_PATH,
            "embodiment": N16_EMBODIMENT,
            "model_name": "GR00T-N1.6-3B",
        }
    elif model_version == "n15":
        return {
            "server_python": N15_SERVER_PYTHON,
            "server_script": N15_SERVER_SCRIPT,
            "model_path": N15_MODEL_PATH,
            "embodiment": N15_EMBODIMENT,
            "model_name": "GR00T-N1.5-3B",
        }
    else:
        raise ValueError(f"Unknown model version: {model_version}")


# ---------------------------------------------------------------------------
# Server lifecycle (same logic as _common.py, parameterised)
# ---------------------------------------------------------------------------


class ModelServer365:
    """Context manager for a GR00T inference server subprocess."""

    def __init__(
        self,
        gpu_id: int,
        port: int,
        log_path: Path,
        model_config: dict,
        *,
        startup_timeout: float = 300.0,
        shutdown_timeout: float = 10.0,
    ) -> None:
        self._gpu_id = gpu_id
        self._port = port
        self._log_path = log_path
        self._cfg = model_config
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._proc: subprocess.Popen | None = None
        self._log_file = None

    def __enter__(self) -> ModelServer365:
        self._log_file = open(self._log_path, "w")
        try:
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(self._gpu_id),
            }
            # N1.5 server needs benchmark repo on PYTHONPATH
            if self._cfg["embodiment"] == N15_EMBODIMENT:
                env["PYTHONPATH"] = str(N15_GROOT_ROOT)
            cmd = [
                str(self._cfg["server_python"]),
                str(self._cfg["server_script"]),
                "--model-path",
                str(self._cfg["model_path"]),
                "--embodiment-tag",
                self._cfg["embodiment"],
                "--port",
                str(self._port),
            ]
            # N1.6 server needs --use-sim-policy-wrapper
            if self._cfg["embodiment"] == N16_EMBODIMENT:
                cmd.append("--use-sim-policy-wrapper")
            self._proc = subprocess.Popen(
                cmd,
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

    def _wait_for_ready(self) -> bool:
        import io

        try:
            import torch

            _has_torch = True
        except ImportError:
            _has_torch = False

        use_torch = self._cfg["embodiment"] == N15_EMBODIMENT and _has_torch
        ctx = zmq.Context()
        deadline = time.monotonic() + self._startup_timeout
        try:
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    return False
                sock = ctx.socket(zmq.REQ)
                sock.setsockopt(zmq.RCVTIMEO, 5000)
                sock.setsockopt(zmq.LINGER, 0)
                try:
                    sock.connect(f"tcp://127.0.0.1:{self._port}")
                    if use_torch:
                        buf = io.BytesIO()
                        torch.save({"endpoint": "ping"}, buf)
                        sock.send(buf.getvalue())
                        resp_bytes = sock.recv()
                        resp = torch.load(io.BytesIO(resp_bytes), weights_only=False)
                    else:
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
