# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Asynchronous thread of non-blocking camera
"""

import threading
import time

from enpire.env.forge.robot.camera_factory import create_camera
from enpire.env.forge.robot.constants import TOP_CAM_RESOLUTION, WRIST_CAM_RESOLUTION


class NonBlockingCamera:
    def __init__(self, camera_name: str, image_transform=None, enable_depth=False):
        resolution = TOP_CAM_RESOLUTION if camera_name == "top" else WRIST_CAM_RESOLUTION
        self.camera = create_camera(
            camera_name,
            resolution=resolution,
            fps=30,
            enable_depth=enable_depth,
        )
        self.image = None
        self.depth = None
        self.intrinsics = None
        self.image_transform = image_transform
        self._read_error_logged = False

        # Camera worker
        self.running = True
        self.worker_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.worker_thread.start()

        # Wait for worker to start
        t0 = time.time()
        while self.image is None:
            if not self.worker_thread.is_alive():
                raise RuntimeError(f"Camera worker exited before first frame for {camera_name!r}")
            if time.time() - t0 > 5.0:
                raise TimeoutError(f"Timed out waiting for first frame from camera {camera_name!r}")
            time.sleep(0.01)

        if self.image is None:
            self.running = False
            self.worker_thread.join(timeout=1.0)
            self.camera.stop()
            raise RuntimeError(f"Timed out waiting for first frame from camera {camera_name!r}")

    def _camera_worker(self):
        while self.running:
            try:
                data = self.camera.read()  # Blocks until the next frame is ready.
                self._read_error_logged = False
            except Exception as exc:
                if self.running and not self._read_error_logged:
                    print(f"[YamRealEnv] Camera read error: {exc}")
                    self._read_error_logged = True
                time.sleep(0.01)
                continue
            if data is not None and data.images["rgb"] is not None:
                image = data.images["rgb"]
                self.image = self.image_transform(image) if self.image_transform else image
                self.depth = None if data.depth is None else data.depth
                self.intrinsics = data.intrinsics

    def get_image(self):
        return self.image.copy()

    def get_depth(self):
        return None if self.depth is None else self.depth.copy()

    def get_intrinsics(self):
        if self.intrinsics is not None:
            return dict(self.intrinsics)
        return self.camera.get_intrinsics()

    def close(self):
        self.running = False
        self.worker_thread.join()
        if self.camera is not None:
            self.camera.stop()
