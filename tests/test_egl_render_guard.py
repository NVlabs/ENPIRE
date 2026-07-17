from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pythonpath_with_repo() -> str:
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT)]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def test_egl_device_render_guard_serializes_processes_on_same_gpu(
    tmp_path: Path,
) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": _pythonpath_with_repo(),
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "5",
        "CAP_EGL_LOCK_DIR": str(tmp_path),
    }
    holder_code = """
import sys
import time

from enpire.env.forge.cap.utils.egl_render_guard import egl_device_render_guard

with egl_device_render_guard():
    print("READY", flush=True)
    time.sleep(0.5)
"""
    waiter_code = """
import time

from enpire.env.forge.cap.utils.egl_render_guard import egl_device_render_guard

t0 = time.monotonic()
with egl_device_render_guard():
    waited = time.monotonic() - t0
print(waited, flush=True)
"""

    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = holder.stdout.readline().strip()
        assert ready == "READY"

        waiter = subprocess.run(
            [sys.executable, "-c", waiter_code],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        waited = float(waiter.stdout.strip())
        assert waited >= 0.3
    finally:
        holder.wait(timeout=5)
        if holder.returncode != 0:
            raise AssertionError(holder.stderr.read())
