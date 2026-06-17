from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


def _egl_device_id() -> str | None:
    if os.environ.get("MUJOCO_GL", "egl") != "egl":
        return None
    device_id = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if device_id is None or not device_id.strip():
        return None
    return device_id.strip()


def _lock_path_for_device(device_id: str) -> Path:
    lock_root = Path(os.environ.get("CAP_EGL_LOCK_DIR", tempfile.gettempdir()))
    lock_root.mkdir(parents=True, exist_ok=True)
    return lock_root / f"forge_mujoco_egl_gpu_{device_id}.lock"


@contextmanager
def egl_device_render_guard() -> Iterator[None]:
    """Serialize EGL rendering across processes sharing one GPU.

    MuJoCo offscreen rendering under NVIDIA EGL can abort inside
    ``libnvidia-eglcore`` when multiple processes call into the same GPU at
    once. This guard provides a best-effort cross-process mutex keyed by
    ``MUJOCO_EGL_DEVICE_ID`` so render calls on the same GPU are serialized.
    """

    device_id = _egl_device_id()
    if device_id is None or fcntl is None:
        yield
        return

    lock_path = _lock_path_for_device(device_id)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
