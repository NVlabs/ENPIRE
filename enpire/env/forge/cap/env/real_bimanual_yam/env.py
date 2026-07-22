# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real bimanual YAM environment — direct-mode, no CapServer required."""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from enpire.env.forge.cap.env.base.profile import RobotProfile, yam_profile
from enpire.env.forge.robot.camera_factory import create_camera
from enpire.env.forge.robot.constants import LEFT_FOLLOWER_PORT, RIGHT_FOLLOWER_PORT
from enpire.env.forge.robot.yam.kinematics import YamKinematics
from enpire.env.forge.robot.yam.yam_real_env import FollowerRobotClient


class _CameraThread:
    """Background daemon that reads and caches RGB, depth, and intrinsics."""

    def __init__(self, name: str):
        self._name = name
        self._camera = create_camera(name, enable_depth=True)
        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._intrinsics: dict | None = None
        self._lock = threading.Lock()
        self._running = True
        self._error_logged = False
        self._period_s = max(
            0.0,
            1.0 / float(os.environ.get("CAP_CAMERA_THREAD_MAX_FPS", "30")),
        )
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        # Block until first RGB frame arrives
        t0 = time.time()
        while True:
            with self._lock:
                if self._rgb is not None:
                    break
            if not self._thread.is_alive():
                raise RuntimeError(f"Camera thread for {name!r} died before first frame")
            if time.time() - t0 > 8.0:
                raise TimeoutError(f"Camera {name!r} timed out waiting for first frame")
            time.sleep(0.02)

    def _worker(self) -> None:
        while self._running:
            try:
                data = self._camera.read()
                self._error_logged = False
            except Exception as exc:
                if self._running and not self._error_logged:
                    print(f"[CameraThread] read error: {exc}")
                    self._error_logged = True
                time.sleep(0.01)
                continue
            if data is None:
                continue
            with self._lock:
                if data.images.get("rgb") is not None:
                    self._rgb = data.images["rgb"]
                if data.depth is not None:
                    self._depth = data.depth
                if data.intrinsics is not None:
                    self._intrinsics = dict(data.intrinsics)
            # Do not spin the RGB-D backends as fast as possible.  Without
            # throttling, ZED NEURAL depth + two RealSense depth streams can
            # saturate CPU/GIL scheduling enough to stall the script runner
            # before user code executes.
            if self._period_s > 0:
                time.sleep(self._period_s)

    def get_rgb(self) -> np.ndarray | None:
        with self._lock:
            return self._rgb.copy() if self._rgb is not None else None

    def get_depth(self) -> np.ndarray | None:
        with self._lock:
            return self._depth.copy() if self._depth is not None else None

    def get_intrinsics(self) -> dict | None:
        with self._lock:
            return dict(self._intrinsics) if self._intrinsics is not None else None

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        if hasattr(self._camera, "stop"):
            self._camera.stop()


class RealYamEnv:
    """Real bimanual YAM hardware environment for direct-mode agent runs.

    Wraps FollowerRobotClient arms, depth-enabled camera threads, and
    YamKinematics (FK/IK). Provides the same observation/camera API as
    RoboCasaEnv so skills.py tools work identically on real hardware.

    Camera names include "top", "left_third" for the fixed left/top
    third-view camera, and "left"/"left_wrist"/"right" for wrist cameras.
    """

    def __init__(self, enable_cameras: bool = True):
        self._profile: RobotProfile = yam_profile()
        self._arms: dict[str, FollowerRobotClient] = {
            "left": FollowerRobotClient(port=LEFT_FOLLOWER_PORT),
            "right": FollowerRobotClient(port=RIGHT_FOLLOWER_PORT),
        }
        self.kin = YamKinematics()
        # Serialize all YamKinematics mutations (FK in get_observations,
        # IK in freespace_move, FK for extrinsics — never concurrent).
        self._kin_lock = threading.Lock()

        self._cameras: dict[str, _CameraThread] = {}
        self._camera_retry_after_s: dict[str, float] = {}
        self._camera_lock = threading.Lock()
        self._camera_retry_cooldown_s = max(
            0.5,
            float(os.environ.get("CAP_CAMERA_RETRY_COOLDOWN_S", "2.0")),
        )
        if enable_cameras:
            startup_names_raw = os.environ.get("CAP_REAL_YAM_START_CAMERAS", "").strip()
            if startup_names_raw:
                startup_camera_names = tuple(
                    name.strip()
                    for name in startup_names_raw.replace(",", " ").split()
                    if name.strip()
                )
            else:
                startup_camera_names = tuple(
                    name for name in self._profile.camera_names if name not in {"left", "left_wrist"}
                )
            for name in startup_camera_names:
                self._try_start_camera(name, announce=True)

    def _try_start_camera(self, name: str, *, announce: bool) -> _CameraThread | None:
        try:
            cam = _CameraThread(name)
        except Exception as exc:
            self._camera_retry_after_s[name] = time.time() + self._camera_retry_cooldown_s
            if announce:
                print(f"[RealYamEnv] Camera '{name}' unavailable: {exc}")
            return None
        self._cameras[name] = cam
        self._camera_retry_after_s.pop(name, None)
        if announce:
            print(f"[RealYamEnv] Camera '{name}' ready")
        return cam

    def _ensure_camera(self, name: str) -> _CameraThread | None:
        cam = self._cameras.get(name)
        if cam is not None:
            return cam
        with self._camera_lock:
            cam = self._cameras.get(name)
            if cam is not None:
                return cam
            now = time.time()
            retry_after = float(self._camera_retry_after_s.get(name, 0.0))
            if now < retry_after:
                return None
            return self._try_start_camera(name, announce=True)

    # ------------------------------------------------------------------
    # Arm observations
    # ------------------------------------------------------------------

    def get_observations(self, side: str) -> dict[str, np.ndarray]:
        """Return {joint_pos(6), gripper_pos(1), ee_pos(3), ee_quat(4)} for one arm.

        FK is computed for both arms simultaneously (mink requires bimanual qpos).
        """
        obs_l = self._arms["left"].get_observations()
        obs_r = self._arms["right"].get_observations()
        ljp = obs_l["joint_pos"]
        rjp = obs_r["joint_pos"]
        with self._kin_lock:
            self._seed_kin(ljp, rjp)
            l_pos, l_q, r_pos, r_q = self.kin.forward_kinematics(ljp, rjp)
        side_obs = obs_l if side == "left" else obs_r
        return {
            "joint_pos": side_obs["joint_pos"],
            "gripper_pos": side_obs["gripper_pos"],
            "ee_pos": l_pos if side == "left" else r_pos,
            "ee_quat": l_q if side == "left" else r_q,
        }

    def _seed_kin(self, ljp: np.ndarray, rjp: np.ndarray) -> None:
        """Seed mink configuration with observed joint positions. Must be called inside _kin_lock."""
        self.kin.configuration.data.qpos[:6] = ljp
        self.kin.configuration.data.qpos[8:14] = rjp
        self.kin.configuration.update()

    # ------------------------------------------------------------------
    # Arm commands
    # ------------------------------------------------------------------

    def command_joint_pos(self, side: str, joint_pos_7d: np.ndarray) -> None:
        """Send a 7D [joint×6 + gripper×1] position command. Uses arm_server default gains."""
        self._arms[side].command_joint_pos(joint_pos_7d)

    def command_joint_state(self, side: str, state: dict) -> None:
        """Send a full joint state dict (pos, vel, kp, kd) to one arm."""
        self._arms[side].command_joint_state(state)

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    def render_rgb(self, camera: str) -> np.ndarray | None:
        cam = self._ensure_camera(camera)
        return cam.get_rgb() if cam is not None else None

    def render_depth(self, camera: str) -> np.ndarray | None:
        cam = self._ensure_camera(camera)
        return cam.get_depth() if cam is not None else None

    def get_camera_intrinsics(self, camera: str) -> list[float]:
        """Return [fx, fy, cx, cy]."""
        cam = self._ensure_camera(camera)
        if cam is None:
            return [0.0, 0.0, 0.0, 0.0]
        intr = cam.get_intrinsics()
        if intr is None:
            return [0.0, 0.0, 0.0, 0.0]
        return [intr["fx"], intr["fy"], intr["cx"], intr["cy"]]

    def get_camera_extrinsics(self, camera: str) -> dict:
        """Return {position, rotation, needs_optical_flip} via FK to camera body frame."""
        from enpire.env.forge.robot.models.station.paths import (
            get_camera_extrinsics_override,
            get_top_camera_frame,
            needs_optical_flip,
        )

        override = get_camera_extrinsics_override(camera)
        if override is not None:
            return override

        cam_frame_map = {
            "top": os.environ.get("CAP_TOP_CAMERA_FRAME", get_top_camera_frame()),
            "left": os.environ.get("CAP_LEFT_CAMERA_FRAME", "left_camera_d405"),
            "left_third": os.environ.get(
                "CAP_LEFT_THIRD_CAMERA_FRAME",
                "top_camera_left_d435",
            ),
            "left_fixed": os.environ.get(
                "CAP_LEFT_THIRD_CAMERA_FRAME",
                "top_camera_left_d435",
            ),
            "left_wrist": "left_camera_d405",
            "right": os.environ.get("CAP_RIGHT_CAMERA_FRAME", "right_camera_d405"),
            "right_wrist": os.environ.get("CAP_RIGHT_CAMERA_FRAME", "right_camera_d405"),
        }
        frame_name = cam_frame_map.get(camera)
        if frame_name is None:
            return {
                "position": [0.0, 0.0, 0.0],
                "rotation": np.eye(3).tolist(),
                "needs_optical_flip": True,
            }

        obs_l = self._arms["left"].get_observations()
        obs_r = self._arms["right"].get_observations()
        with self._kin_lock:
            self._seed_kin(obs_l["joint_pos"], obs_r["joint_pos"])
            T = self.kin.configuration.get_transform_frame_to_world(frame_name, "body")

        # mink's SO3 exposes ``as_matrix()`` (not ``matrix()``).  Some older
        # pose objects in this codebase / dependencies expose ``matrix`` as a
        # method or property, so keep this conversion tolerant.
        rot = T.rotation()
        if hasattr(rot, "as_matrix"):
            rot_mat = rot.as_matrix()
        else:
            maybe_matrix = getattr(rot, "matrix", rot)
            rot_mat = maybe_matrix() if callable(maybe_matrix) else maybe_matrix

        return {
            "position": T.translation().tolist(),
            "rotation": np.asarray(rot_mat, dtype=np.float64).reshape(3, 3).tolist(),
            "needs_optical_flip": needs_optical_flip(camera),
        }

    # ------------------------------------------------------------------

    def set_recorder(self, recorder) -> None:
        """No-op — recording is not supported on real hardware."""

    def close(self) -> None:
        if hasattr(self, "_dashboard"):
            self._dashboard.stop()
        for cam in self._cameras.values():
            cam.close()
