"""Video recorder for cap scripts and agent runs.

ScriptRecorder writes camera frames to MP4 files, then re-encodes to H.264.

Two modes:

  CapServer mode — timer-based pull from server cameras (hardware/legacy):
      recorder = ScriptRecorder(server, Path("video"))

  Env push mode — frames pushed after each env.step(), no background timer:
      recorder = ScriptRecorder.from_env(camera_names, Path("video"))
      env.set_recorder(recorder)   # env calls recorder.push_frame() on step
"""

from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from enpire.env.forge.cap.server.cap_server import CapServer

_RECORD_FPS = 5


class ScriptRecorder:
    def __init__(self, server: CapServer, output_dir: Path) -> None:
        """CapServer mode — timer pulls frames from server cameras."""
        self._server = server
        self._cam_names: list[str] = list(server._cameras.keys())
        self._get_image = server.get_camera_image
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._frame_queue: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="script-recorder-capture",
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="script-recorder-writer",
        )

    @classmethod
    def from_env(
        cls,
        camera_names: list[str],
        output_dir: Path,
    ) -> ScriptRecorder:
        """Env push mode — no timer thread; frames pushed via push_frame()."""
        instance = object.__new__(cls)
        instance._server = None
        instance._cam_names = list(camera_names)
        instance._get_image = None
        instance._output_dir = output_dir
        instance._output_dir.mkdir(parents=True, exist_ok=True)
        instance._writers = {}
        instance._frame_queue = queue.Queue()
        instance._stop_event = threading.Event()
        instance._capture_thread = None  # no timer in push mode
        instance._writer_thread = threading.Thread(
            target=instance._writer_loop,
            daemon=True,
            name="script-recorder-writer",
        )
        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        print(f"[recorder] Saving to {self._output_dir}")
        if self._capture_thread is not None:
            self._capture_thread.start()
        self._writer_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join()
        self._frame_queue.put(None)  # sentinel
        self._writer_thread.join()
        for w in self._writers.values():
            w.release()
        self._reencode_videos()
        print(f"[recorder] Videos saved to {self._output_dir}")

    def push_frame(self, cam_name: str, frame: np.ndarray) -> None:
        """Push a frame directly (env push mode). Safe to call from any thread."""
        if self._stop_event.is_set():
            return
        if frame is not None and frame.ndim == 3 and frame.shape[0] > 1:
            self._frame_queue.put((cam_name, frame.copy()))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        import time
        interval = 1.0 / _RECORD_FPS
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            for cam in self._cam_names:
                try:
                    frame = self._get_image(cam)
                    if frame is not None and frame.shape[0] > 1:
                        self._frame_queue.put((cam, frame.copy()))
                except Exception:
                    pass
            elapsed = time.monotonic() - t0
            remaining = interval - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _writer_loop(self) -> None:
        while True:
            item = self._frame_queue.get()
            if item is None:
                break
            cam, frame = item
            if cam not in self._writers:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                path = str(self._output_dir / f"{cam}.mp4")
                self._writers[cam] = cv2.VideoWriter(path, fourcc, _RECORD_FPS, (w, h))
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._writers[cam].write(bgr)

    def _reencode_videos(self) -> None:
        for mp4 in sorted(self._output_dir.glob("*.mp4")):
            tmp = mp4.with_suffix(".tmp.mp4")
            mp4.rename(tmp)
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(tmp),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "30",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-pix_fmt", "yuv420p", str(mp4),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                tmp.unlink()
            except (subprocess.CalledProcessError, FileNotFoundError):
                tmp.rename(mp4)
