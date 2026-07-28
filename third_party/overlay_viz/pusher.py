# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-blocking frame pusher for overlay viz server.

Usage from the control loop::

    from third_party.overlay_viz.pusher import FramePusher

    pusher = FramePusher()  # connects to http://localhost:8888

    # In your image update loop (called at ~30Hz):
    pusher.push(bgr_frame)  # non-blocking, drops frames if busy
"""

from __future__ import annotations

import logging
import queue
import threading
from urllib.request import Request, urlopen

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FramePusher:
    """Push camera frames to the overlay viz server without blocking the caller.

    Runs a background thread that drains a single-slot queue, so at most one
    frame is in-flight at a time and the caller never blocks.
    """

    def __init__(self, url: str = "http://localhost:8888/api/cameras/push") -> None:
        self._url = url
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self._running = True
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    def push(self, frame: np.ndarray, quality: int = 70) -> None:
        """Encode *frame* (BGR uint8) as JPEG and queue it for sending.

        Non-blocking.  Drops the frame if the previous send hasn't finished.
        """
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        jpeg = buf.tobytes()
        # Replace whatever is in the queue (single-slot, drop old)
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(jpeg)
        except queue.Full:
            pass

    def close(self) -> None:
        self._running = False

    def _send_loop(self) -> None:
        while self._running:
            try:
                jpeg = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                req = Request(
                    self._url,
                    data=jpeg,
                    headers={"Content-Type": "image/jpeg"},
                    method="POST",
                )
                urlopen(req, timeout=1)
            except Exception:
                pass  # Server down or busy — silently drop
