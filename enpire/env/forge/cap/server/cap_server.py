"""CAP Server — CONTROL_FREQ_HZ control loop that owns all hardware communication.

Exposes Portal RPC methods for the agent/tool layer to call:
  get_state, _ik_servo, _ik_servo_keypoints, move_joint_keypoints,
  move_bimanual_joint_keypoints,
  set_gripper, go_home, get_camera_image, get_camera_depth,
  get_camera_intrinsics, get_camera_extrinsics, execute_skill,
  start_policy_output, step_policy_output, stop_policy_output,
  use_policy_output, learn_skill, get_skill_prediction, estop, release_estop.

Run standalone:  uv run cap/server/cap_server.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `cap.*` imports work when run as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
import pink
import pinocchio as pin
import portal
from enpire.env.forge.robot.camera_factory import create_camera, get_camera_backend, get_camera_type_name

import logging

from enpire.env.forge.cap.config import (
    POLICY_MODEL_CONFIGS,
    CAP_SERVER_PORT,
    CAMERA_NAMES,
    CONTROL_FREQ_HZ,
    CONTROL_PERIOD_S,
    GO_HOME_MAX_JOINT_VEL,
    GRIPPER_POLL_S,
    GRIPPER_SETTLE_THRESH,
    GRIPPER_SETTLE_TIMEOUT_S,
    GRIPPER_STALL_THRESH,
    GRIPPER_TORQUE_LIMIT_HOLD_S,
    HIL_LOG_DIR as _HIL_LOG_DIR,
    HOME_JOINT_STATE,
    INTERP_KD,
    INTERP_KP,
    FELLO_HOST,
    LEFT_FOLLOWER_PORT,
    LEFT_LEADER_PORT,
    MOVE_EEF_MAX_DURATION_S,
    MOVE_EEF_MAX_VEL,
    POLICY_FREQ_HZ,
    POLICY_PERIOD_S,
    POLICY_SERVER_PORT,
    RIGHT_FOLLOWER_PORT,
    RIGHT_LEADER_PORT,
    ALWAYS_TAKEOVERABLE,
    REWARD_SERVER_PORT,
    RL_EPISODE_MAX_STEPS,
    RL_POLICY_HOST,
    RL_POLICY_PORT,
    USE_FELLO,
    HIL_POLICY_SLOWDOWN,
    JOINT_LIMITS_LOW,
    JOINT_LIMITS_HIGH,
    GRIPPER_MIN,
    GRIPPER_MAX,
)

logger = logging.getLogger(__name__)
from enpire.env.forge.cap.diag.emitter import emit
from enpire.env.forge.cap.server.safety import ArmSafetyZone, SafetyChecker


def _print_green(msg: str) -> None:
    print(f"\033[32m{msg}\033[0m")


def _print_red(msg: str) -> None:
    print(f"\033[31m{msg}\033[0m")


def _decode_portal_str(value: str, default: str = "") -> str:
    """Convert Portal's empty-string sentinel back to actual empty string.

    Portal's buffer layer cannot transmit zero-length strings, so the client
    sends ``"__none__"`` instead. This helper converts it back.
    """
    from enpire.env.forge.cap.config import PORTAL_EMPTY_SENTINEL

    return default if value == PORTAL_EMPTY_SENTINEL else value


def _rpc_ok(reason: str = "ok", **extra) -> dict:
    """Return a successful RPC response dict."""
    return {"success": True, "reason": reason, **extra}


def _rpc_error(reason: str, **extra) -> dict:
    """Return a failed RPC response dict."""
    return {"success": False, "reason": reason, **extra}


# ---------------------------------------------------------------------------
# Realtime file logger for HIL gripper debugging
# Writes to _HIL_LOG_DIR (from cap.config) with immediate flush so logs survive kill.
# ---------------------------------------------------------------------------
_hil_log_file = None


def _hil_log(msg: str) -> None:
    global _hil_log_file
    if _hil_log_file is None:
        try:
            _HIL_LOG_DIR.mkdir(parents=True, exist_ok=True)
            _hil_log_file = open(
                _HIL_LOG_DIR / "cap_grip.log", "a", buffering=1
            )  # line-buffered
        except Exception:
            return
    _hil_log_file.write(f"{time.time():.3f} {msg}\n")
    _hil_log_file.flush()


@dataclass
class _PolicyOutputSession:
    model: str
    policy_server: str
    task_description: str
    replan_horizon: int
    reset_behavior: str
    embodiment_tag: str
    resolution: int
    policy: Any
    dirty: bool = False
    dirty_reason: str = ""


# ---------------------------------------------------------------------------
# Hardware wrappers
# ---------------------------------------------------------------------------


class _ArmClient:
    """Portal RPC client for a single Fello follower arm."""

    def __init__(self, host: str, port: int):
        self._client = portal.Client(f"{host}:{port}")

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> dict[str, np.ndarray]:
        return self._client.get_observations().result()


class _StubArmClient:
    """Drop-in replacement for _ArmClient that returns zeros (no hardware)."""

    def get_joint_pos(self) -> np.ndarray:
        return np.zeros(6)

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        pass

    def get_observations(self) -> dict[str, np.ndarray]:
        return {"joint_pos": np.zeros(6), "gripper_pos": np.zeros(1)}


class _CameraClient:
    """Non-blocking camera reader (background thread)."""

    def __init__(self, camera_name: str):
        self._name = camera_name
        self._backend = get_camera_backend(camera_name)
        default_fps = 30 if self._backend == "zed" else 60
        self._camera = create_camera(
            camera_name,
            resolution=(640, 480),
            fps=default_fps,
            enable_depth=True,
        )
        self.camera_type = get_camera_type_name(self._camera)
        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._intrinsics: dict | None = None
        self._read_error_logged = False
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._rgb is None:
            if not self._thread.is_alive():
                raise RuntimeError(
                    f"Camera worker exited before first frame for {camera_name!r}"
                )
            if time.time() - t0 > 5.0:
                raise TimeoutError(
                    f"Timed out waiting for first frame from camera {camera_name!r}"
                )
            time.sleep(0.01)

    # -- RealSense backend --------------------------------------------------

    def _init_realsense(self, cfg: "CameraConfig | None") -> None:
        from enpire.env.forge.robot.realsense import RealSenseCamera

        if cfg is not None:
            if cfg.device_id is not None:
                serial = cfg.device_id
            elif cfg.symlink is not None:
                serial = self._resolve_serial_from_symlink(cfg.symlink)
            else:
                raise ValueError(
                    f"RealSense camera '{cfg.name}' needs device_id or symlink"
                )
        else:
            serial = self._resolve_serial_from_symlink(f"/dev/video_{self._name}")

        self._camera = RealSenseCamera(
            device_id=serial,
            resolution=(640, 480),
            fps=60,
            enable_depth=True,
        )
        self.camera_type = get_camera_type_name(self._camera)
        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._intrinsics: dict | None = None
        self._read_error_logged = False
        self._backend = "realsense"
        self._running = True
        self._thread = threading.Thread(target=self._worker_realsense, daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._rgb is None:
            if not self._thread.is_alive():
                raise RuntimeError(
                    f"Camera worker exited before first frame for {camera_name!r}"
                )
            if time.time() - t0 > 5.0:
                raise TimeoutError(
                    f"Timed out waiting for first frame from camera {camera_name!r}"
                )
            time.sleep(0.01)

    @staticmethod
    def _resolve_serial_from_symlink(symlink: str) -> str:
        import pyrealsense2 as rs

        if not os.path.exists(symlink):
            raise FileNotFoundError(f"No symlink: {symlink}")
        video = os.path.basename(os.path.realpath(symlink))
        usb_id = os.path.basename(
            os.path.realpath(f"/sys/class/video4linux/{video}/device")
        )
        ctx = rs.context()
        for dev in ctx.query_devices():
            if usb_id in dev.get_info(rs.camera_info.physical_port):
                return dev.get_info(rs.camera_info.serial_number)
        raise ValueError(f"No RealSense for symlink: {symlink}")

    def _worker(self) -> None:
        _frame_count = 0
        _last_frame_time = time.time()
        while self._running:
            _t_read_start = time.time()
            try:
                data = self._camera.read()
                self._read_error_logged = False
            except Exception as exc:
                if self._running and not self._read_error_logged:
                    print(f"[CapServer] Camera '{self._name}' read error: {exc}")
                    self._read_error_logged = True
                time.sleep(0.01)
                continue
            _read_ms = (time.time() - _t_read_start) * 1000
            if data is not None:
                if data.images.get("rgb") is not None:
                    self._rgb = data.images["rgb"]
                if data.depth is not None:
                    self._depth = data.depth
                if data.intrinsics is not None:
                    self._intrinsics = data.intrinsics
                _frame_count += 1
                if _frame_count % 10 == 0:
                    _now = time.time()
                    _dt = _now - _last_frame_time
                    emit(
                        "camera",
                        "frame",
                        camera_name=self._name,
                        dt_ms=_dt * 1000,
                        read_ms=_read_ms,
                    )
                    _last_frame_time = _now

    # -- ZED backend --------------------------------------------------------

    def _init_zed(self, cfg: "CameraConfig") -> None:
        import pyzed.sl as sl

        settings_dir = Path.home() / ".cache" / "zed" / "settings"
        settings_dir.mkdir(parents=True, exist_ok=True)

        self._sl = sl
        self._zed = sl.Camera()
        self._zed_image = sl.Mat()
        self._zed_depth = sl.Mat()

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.optional_settings_path = str(settings_dir) + "/"

        status = self._zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Failed to open ZED '{cfg.name}' (symlink {cfg.symlink}): {status}"
            )

        calib = self._zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self._intrinsics = {
            "fx": calib.fx,
            "fy": calib.fy,
            "cx": calib.cx,
            "cy": calib.cy,
        }

        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._backend = "zed"
        self._running = True
        self._thread = threading.Thread(target=self._worker_zed, daemon=True)
        self._thread.start()
        while self._rgb is None:
            time.sleep(0.01)

    def _worker_zed(self) -> None:
        sl = self._sl
        _frame_count = 0
        _last_frame_time = time.time()
        while self._running:
            _t_read_start = time.time()
            if self._zed.grab() == sl.ERROR_CODE.SUCCESS:
                self._zed.retrieve_image(self._zed_image, sl.VIEW.LEFT)
                bgra = self._zed_image.get_data()
                rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (480, 640):
                    rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
                self._rgb = rgb

                self._zed.retrieve_measure(self._zed_depth, sl.MEASURE.DEPTH)
                raw = self._zed_depth.get_data().copy()
                self._depth = np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)
                if self._depth.shape[:2] != (480, 640):
                    self._depth = cv2.resize(
                        self._depth, (640, 480), interpolation=cv2.INTER_NEAREST
                    )

                _read_ms = (time.time() - _t_read_start) * 1000
                _frame_count += 1
                if _frame_count % 10 == 0:
                    _now = time.time()
                    _dt = _now - _last_frame_time
                    emit(
                        "camera",
                        "frame",
                        camera_name=self._name,
                        dt_ms=_dt * 1000,
                        read_ms=_read_ms,
                    )
                    _last_frame_time = _now

    # -- Unified public interface -------------------------------------------

    def get_rgb(self) -> np.ndarray:
        assert self._rgb is not None
        return self._rgb.copy()

    def get_depth(self) -> np.ndarray | None:
        return self._depth.copy() if self._depth is not None else None

    def get_intrinsics(self) -> list[float] | None:
        if self._intrinsics is None:
            return None
        i = self._intrinsics
        return [i["fx"], i["fy"], i["cx"], i["cy"]]

    def get_intrinsics_full(self) -> dict | None:
        if self._intrinsics is None:
            return None
        return dict(self._intrinsics)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        stop = getattr(self._camera, "stop", None)
        if callable(stop):
            stop()


# ---------------------------------------------------------------------------
# Hold-position policy (conforms to SyncChunkingPolicy inner-policy interface)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Skill data recorder
# ---------------------------------------------------------------------------


class _SkillDataRecorder:
    """Per-episode data recorder for learn_skill, matching RecordEpisodeWrapper format.

    Records observations, commanded actions, action sources, per-step takeover flags,
    and a placeholder reward signal to disk in the same directory layout used by
    run_data_collection / RecordEpisodeWrapper.
    """

    VIDEO_FPS = POLICY_FREQ_HZ

    def __init__(self, output_dir: str | Path, task_name: str) -> None:
        self._task_name = task_name
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = tempfile.mkdtemp(prefix="cap_learn_skill_")
        self.timestamps: list[float] = []
        self._obs_buffers: dict[str, list] = {}
        self._action_left: list[np.ndarray] = []
        self._action_right: list[np.ndarray] = []
        self._action_sources: list[str] = []
        self._is_human_action: list[float] = []
        self._reward: list[float] = []
        self._video_writers: dict[str, cv2.VideoWriter] = {}
        # Background thread for video encoding — keeps cvtColor + write off the
        # control-loop thread so it doesn't cause CONTROL_FREQ_HZ overruns.
        self._frame_queue: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._frame_writer_loop,
            daemon=True,
            name="cap-video-writer",
        )
        self._writer_thread.start()

    def record_step(
        self,
        obs: dict,
        left_jp: np.ndarray,
        left_grip: np.ndarray,
        right_jp: np.ndarray,
        right_grip: np.ndarray,
        left_takeover: bool,
        right_takeover: bool,
        reward: float = 0.0,
    ) -> None:
        """Buffer one step of data."""
        self.timestamps.append(time.time())
        # Non-image observations
        for k, v in obs.items():
            arr = np.asarray(v)
            if arr.ndim != 3:
                self._obs_buffers.setdefault(k, []).append(arr)
        # Commanded actions (joint + gripper concatenated)
        self._action_left.append(np.concatenate([left_jp.ravel(), left_grip.ravel()]))
        self._action_right.append(
            np.concatenate([right_jp.ravel(), right_grip.ravel()])
        )
        # Source label and takeover flag
        is_human = left_takeover or right_takeover
        self._action_sources.append("human" if is_human else "policy")
        self._is_human_action.append(1.0 if is_human else 0.0)
        self._reward.append(reward)
        # Enqueue video frames for background encoding (copy to decouple
        # from camera buffers that may be overwritten next tick).
        for k, v in obs.items():
            arr = np.asarray(v)
            if arr.ndim == 3:
                self._frame_queue.put((k, arr.copy()))

    def _write_video_frame(self, key: str, frame: np.ndarray) -> None:
        if key not in self._video_writers:
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            name = f"{key.replace('_image', '-images-rgb')}.mp4"
            path = str(Path(self._tmp_dir) / name)
            self._video_writers[key] = cv2.VideoWriter(
                path, fourcc, self.VIDEO_FPS, (w, h)
            )
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._video_writers[key].write(bgr)

    def _frame_writer_loop(self) -> None:
        """Background thread: drain frame queue → cvtColor + VideoWriter.write."""
        while True:
            item = self._frame_queue.get()
            if item is None:  # sentinel → shutdown
                break
            key, frame = item
            self._write_video_frame(key, frame)

    def _drain_writer(self) -> None:
        """Send sentinel and wait for the background writer to finish."""
        self._frame_queue.put(None)
        self._writer_thread.join()

    def save(self) -> Path | None:
        """Finalize, save episode to disk, and return the episode directory."""
        self._drain_writer()
        for writer in self._video_writers.values():
            writer.release()
        if not self.timestamps:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            return None

        tmp = Path(self._tmp_dir)

        # Timestamps
        np.save(tmp / "timestamp.npy", np.array(self.timestamps))

        # Non-image observations
        for k, v in self._obs_buffers.items():
            if k == "annotation.task":
                continue
            fname = k.replace("left_", "left-").replace("right_", "right-")
            try:
                np.save(tmp / f"{fname}.npy", np.array(v, dtype=np.float64))
            except ValueError:
                logger.warning("learn_skill: could not save obs key %s", k)

        # Actions
        np.save(tmp / "action-left-pos.npy", np.array(self._action_left))
        np.save(tmp / "action-right-pos.npy", np.array(self._action_right))
        np.save(tmp / "action-source.npy", np.array(self._action_sources))

        # Takeover flag + placeholder reward
        np.save(
            tmp / "is-human-action.npy",
            np.array(self._is_human_action, dtype=np.float32),
        )
        np.save(tmp / "reward.npy", np.array(self._reward, dtype=np.float32))

        # Metadata
        duration = (
            self.timestamps[-1] - self.timestamps[0]
            if len(self.timestamps) > 1
            else 0.0
        )
        meta = {
            "task_name": self._task_name,
            "env_loop_frequency": self.VIDEO_FPS,
            "duration": duration,
            "station_metadata": {
                "arm_type": "yam",
                "world_frame": "left_arm",
                "extrinsics": {
                    "right_arm_extrinsic": {
                        "position": [0.0, -0.61, 0.0],
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                    }
                },
            },
        }
        with open(tmp / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._convert_videos_to_h264(tmp)

        episode_dir = self._output_dir / datetime.now().strftime("%Y%m%dT%H%M%S%f")
        shutil.move(str(tmp), episode_dir)
        logger.info(
            "learn_skill: saved episode to %s (%d steps)",
            episode_dir,
            len(self.timestamps),
        )
        return episode_dir

    def discard(self) -> None:
        self._drain_writer()
        for writer in self._video_writers.values():
            writer.release()
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _convert_videos_to_h264(self, tmp: Path) -> None:
        for video_file in sorted(tmp.glob("*-images-rgb.mp4")):
            temp_file = video_file.parent / f"{video_file.stem}_tmp{video_file.suffix}"
            video_file.rename(temp_file)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(temp_file),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                        "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-pix_fmt",
                        "yuv420p",
                        str(video_file),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                temp_file.unlink()
            except (subprocess.CalledProcessError, FileNotFoundError):
                temp_file.rename(video_file)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_se3(pos: np.ndarray, quat_xyzw: np.ndarray):
    """Build a pinocchio SE3 from a position and xyzw quaternion."""
    x, y, z, w = quat_xyzw
    return pin.SE3(pin.Quaternion(w, x, y, z).toRotationMatrix(), pos)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two unit quaternions in xyzw format. t in [0, 1]."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take the shorter arc
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:  # nearly identical — fall back to normalised lerp
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_t0 = np.sin(theta_0)
    return np.sin(theta_0 - theta) / sin_t0 * q0 + np.sin(theta) / sin_t0 * q1


# ---------------------------------------------------------------------------
# CAP Server
# ---------------------------------------------------------------------------


class CapServer:
    """CONTROL_FREQ_HZ control loop server with Portal RPC interface."""

    # --- Backward-compatible property aliases for dict-based state ---
    # These allow existing code/tests that access self.left_joint_pos etc.
    # to keep working after the migration to self._arm_jp["left"].

    @property
    def left_joint_pos(self) -> np.ndarray:
        return self._arm_jp.get("left", np.zeros(6))

    @left_joint_pos.setter
    def left_joint_pos(self, value: np.ndarray):
        self._arm_jp["left"] = value

    @property
    def right_joint_pos(self) -> np.ndarray:
        return self._arm_jp.get("right", np.zeros(6))

    @right_joint_pos.setter
    def right_joint_pos(self, value: np.ndarray):
        self._arm_jp["right"] = value

    @property
    def left_gripper_pos(self) -> np.ndarray:
        return self._arm_gp.get("left", np.zeros(1))

    @left_gripper_pos.setter
    def left_gripper_pos(self, value: np.ndarray):
        self._arm_gp["left"] = value

    @property
    def right_gripper_pos(self) -> np.ndarray:
        return self._arm_gp.get("right", np.zeros(1))

    @right_gripper_pos.setter
    def right_gripper_pos(self, value: np.ndarray):
        self._arm_gp["right"] = value

    @property
    def _left_ee_pos(self) -> np.ndarray:
        return self._ee_pos.get("left", np.zeros(3))

    @_left_ee_pos.setter
    def _left_ee_pos(self, value: np.ndarray):
        self._ee_pos["left"] = value

    @property
    def _right_ee_pos(self) -> np.ndarray:
        return self._ee_pos.get("right", np.zeros(3))

    @_right_ee_pos.setter
    def _right_ee_pos(self, value: np.ndarray):
        self._ee_pos["right"] = value

    @property
    def _left_ee_quat(self) -> np.ndarray:
        return self._ee_quat.get("left", np.array([0.0, 0.0, 0.0, 1.0]))

    @_left_ee_quat.setter
    def _left_ee_quat(self, value: np.ndarray):
        self._ee_quat["left"] = value

    @property
    def _right_ee_quat(self) -> np.ndarray:
        return self._ee_quat.get("right", np.array([0.0, 0.0, 0.0, 1.0]))

    @_right_ee_quat.setter
    def _right_ee_quat(self, value: np.ndarray):
        self._ee_quat["right"] = value

    @property
    def _cmd_left_jp(self) -> np.ndarray:
        return self._cmd_jp.get("left", np.zeros(6))

    @_cmd_left_jp.setter
    def _cmd_left_jp(self, value: np.ndarray):
        self._cmd_jp["left"] = value

    @property
    def _cmd_right_jp(self) -> np.ndarray:
        return self._cmd_jp.get("right", np.zeros(6))

    @_cmd_right_jp.setter
    def _cmd_right_jp(self, value: np.ndarray):
        self._cmd_jp["right"] = value

    @property
    def _cmd_left_gp(self) -> np.ndarray:
        return self._cmd_gp.get("left", np.zeros(1))

    @_cmd_left_gp.setter
    def _cmd_left_gp(self, value: np.ndarray):
        self._cmd_gp["left"] = value

    @property
    def _cmd_right_gp(self) -> np.ndarray:
        return self._cmd_gp.get("right", np.zeros(1))

    @_cmd_right_gp.setter
    def _cmd_right_gp(self, value: np.ndarray):
        self._cmd_gp["right"] = value

    @property
    def _cmd_left_gv(self) -> float | None:
        return self._cmd_gv.get("left")

    @_cmd_left_gv.setter
    def _cmd_left_gv(self, value: float | None):
        self._cmd_gv["left"] = value

    @property
    def _cmd_right_gv(self) -> float | None:
        return self._cmd_gv.get("right")

    @_cmd_right_gv.setter
    def _cmd_right_gv(self, value: float | None):
        self._cmd_gv["right"] = value

    @property
    def _cmd_left_gt(self) -> float | None:
        return self._cmd_gt.get("left")

    @_cmd_left_gt.setter
    def _cmd_left_gt(self, value: float | None):
        self._cmd_gt["left"] = value

    @property
    def _cmd_right_gt(self) -> float | None:
        return self._cmd_gt.get("right")

    @_cmd_right_gt.setter
    def _cmd_right_gt(self, value: float | None):
        self._cmd_gt["right"] = value

    def __init__(
        self,
        arm_host: str = "127.0.0.1",
        left_port: int = LEFT_FOLLOWER_PORT,
        right_port: int = RIGHT_FOLLOWER_PORT,
        server_port: int = CAP_SERVER_PORT,
        enable_cameras: bool = True,
        no_arms: bool = False,
        rl_host: str = RL_POLICY_HOST,
        env_name: str | None = None,
        env_viewer: bool = False,
    ):
        self._sim_mode = env_name is not None
        self._sim_backend = None
        self._robot_profile = None
        self._env_name = env_name

        if env_name is not None:
            from enpire.env.forge.cap.env import create_env
            from enpire.env.forge.cap.env.adapters.sim import (
                SimArmAdapter as SimArmClient,
                SimCameraAdapter as SimCameraClient,
            )

            self._sim_backend = create_env(env_name, viewer=env_viewer)

            # Get profile and arm/camera names from the env if available
            self._robot_profile = getattr(self._sim_backend, "_profile", None)
            _arm_names = (
                self._robot_profile.arm_names
                if self._robot_profile
                else ("left", "right")
            )
            _camera_names = (
                self._robot_profile.camera_names
                if self._robot_profile
                else CAMERA_NAMES
            )

            self._arms: dict[str, object] = {
                side: SimArmClient(self._sim_backend, side) for side in _arm_names
            }

            self._cameras: dict[str, object] = {
                name: SimCameraClient(self._sim_backend, name) for name in _camera_names
            }
            print(f"[CapServer] Running env: {env_name}")
        else:
            # Arms
            if no_arms:
                self._arms: dict[str, _ArmClient | _StubArmClient] = {
                    "left": _StubArmClient(),
                    "right": _StubArmClient(),
                }
                print("[CapServer] Running with stub arms (--no-arms)")
            else:
                self._arms: dict[str, _ArmClient | _StubArmClient] = {
                    "left": _ArmClient(arm_host, left_port),
                    "right": _ArmClient(arm_host, right_port),
                }

            # Cameras
            self._cameras: dict[str, _CameraClient] = {}
            if enable_cameras:
                for name in CAMERA_NAMES:
                    try:
                        self._cameras[name] = _CameraClient(name)
                        _print_green(
                            f"[CapServer] Camera '{name}' ready ({self._cameras[name].camera_type})"
                        )
                    except Exception as e:
                        _print_red(f"[CapServer] Camera '{name}' unavailable: {e}")

        # RL policy server host (for learn_skill_rl)
        self._rl_host = rl_host

        # Safety (estop only)
        self._safety = SafetyChecker()

        # Per-side motion cancellation — set by cancel_motion(), checked in _ik_servo()
        self._cancel_motion: dict[str, threading.Event] = {}

        # ---------------------------------------------------------------------------
        # Shared state — ONE lock protects everything below
        # ---------------------------------------------------------------------------
        self._state_lock = threading.Lock()

        # Determine DOF per arm from profile or default to 6 (YAM)
        _arm_dof: dict[str, int] = {}
        if self._robot_profile is not None:
            for side, arm in self._robot_profile.arms.items():
                _arm_dof[side] = arm.dof
        else:
            _arm_dof = {"left": 6, "right": 6}

        # Initialise per-side cancel events now that arm names are known
        self._cancel_motion = {side: threading.Event() for side in _arm_dof}

        # Actual hardware readings — updated at CONTROL_FREQ_HZ by control loop
        self._arm_jp: dict[str, np.ndarray] = {
            side: np.zeros(dof) for side, dof in _arm_dof.items()
        }
        self._arm_gp: dict[str, np.ndarray] = {side: np.zeros(1) for side in _arm_dof}

        # Cached EE poses (FK on actual, computed in control loop)
        self._ee_pos: dict[str, np.ndarray] = {side: np.zeros(3) for side in _arm_dof}
        self._ee_quat: dict[str, np.ndarray] = {
            side: np.array([0.0, 0.0, 0.0, 1.0]) for side in _arm_dof
        }

        # Commanded joint positions — written by _ik_servo / execute_skill / go_home
        self._cmd_jp: dict[str, np.ndarray] = {
            side: np.zeros(dof) for side, dof in _arm_dof.items()
        }
        self._cmd_gp: dict[str, np.ndarray] = {side: np.zeros(1) for side in _arm_dof}
        self._cmd_gv: dict[str, float | None] = {side: None for side in _arm_dof}
        self._cmd_gt: dict[str, float | None] = {side: None for side in _arm_dof}

        # Base pose for mobile-base robots (updated in control loop)

        # Active EE targets — set by _ik_servo, consumed by control loop
        self._eef_targets: dict[str, tuple[np.ndarray, np.ndarray, float | None]] = {}

        # Control period from profile
        self._control_period_s: float = (
            1.0 / self._robot_profile.control_freq_hz
            if self._robot_profile
            else CONTROL_PERIOD_S
        )

        # Seed commanded positions from hardware
        self._init_state()

        # Capture home EE poses (for go_home on EE-control envs like RoboCasa)
        self._home_ee_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in self._arms:
            try:
                obs = self._arms[side].get_observations()
                if "ee_pos" in obs and "ee_quat" in obs:
                    self._home_ee_poses[side] = (
                        np.asarray(obs["ee_pos"], dtype=np.float64).copy(),
                        np.asarray(obs["ee_quat"], dtype=np.float64).copy(),
                    )
            except Exception:
                pass

        # ---------------------------------------------------------------------------
        # Pink IK (used by _ik_servo) — only for robots with a URDF
        # ---------------------------------------------------------------------------
        # Resolve URDF path for FK/IK
        _urdf_path: str | None = None
        if (
            self._robot_profile is not None
            and self._robot_profile.urdf_path is not None
        ):
            _urdf_path = str(self._robot_profile.urdf_path)
        elif self._robot_profile is None:
            # No profile = real hardware or legacy YAM sim — load YAM URDF
            try:
                from enpire.env.forge.robot.models.station.paths import get_station_urdf

                _urdf_path = str(get_station_urdf())
            except Exception:
                pass

        self._has_urdf_ik = _urdf_path is not None
        self._pink_model = None
        self._pink_cfg = None
        self._pink_tasks: dict[str, object] = {}  # side → FrameTask
        self._posture = None
        self._pink_q_lower = None
        self._pink_q_upper = None
        self._fk_data = None
        self._fk_frame_ids: dict[str, int] = {}  # side → frame id

        # Legacy aliases for YAM code paths
        self._left_task = None
        self._right_task = None
        self._left_grasp_id = None
        self._right_grasp_id = None

        if self._has_urdf_ik:
            self._pink_model = pin.buildModelFromUrdf(_urdf_path)
            self._pink_cfg = pink.Configuration(
                self._pink_model,
                self._pink_model.createData(),
                pin.neutral(self._pink_model),
            )
            self._posture = pink.PostureTask(cost=1e-3)
            self._pink_q_lower = self._pink_model.lowerPositionLimit.copy()
            self._pink_q_upper = self._pink_model.upperPositionLimit.copy()
            self._fk_data = self._pink_model.createData()

            # Set up per-arm frame tasks and FK frame IDs
            if self._robot_profile is not None:
                for side, arm in self._robot_profile.arms.items():
                    task = pink.FrameTask(
                        arm.ee_frame_name, position_cost=1.0, orientation_cost=1.0
                    )
                    self._pink_tasks[side] = task
                    self._fk_frame_ids[side] = self._pink_model.getFrameId(
                        arm.ee_frame_name
                    )
            else:
                # Legacy YAM path
                self._pink_tasks["left"] = pink.FrameTask(
                    "left_grasp", position_cost=1.0, orientation_cost=1.0
                )
                self._pink_tasks["right"] = pink.FrameTask(
                    "right_grasp", position_cost=1.0, orientation_cost=1.0
                )
                self._fk_frame_ids["left"] = self._pink_model.getFrameId("left_grasp")
                self._fk_frame_ids["right"] = self._pink_model.getFrameId("right_grasp")

            # Legacy aliases for existing code that references these directly
            self._left_task = self._pink_tasks.get("left")
            self._right_task = self._pink_tasks.get("right")
            self._left_grasp_id = self._fk_frame_ids.get("left")
            self._right_grasp_id = self._fk_frame_ids.get("right")

        # Skill execution lock (prevents concurrent skills)
        self._skill_lock = threading.Lock()
        self._latest_action_chunk: dict | None = None
        self._policy_output_session_lock = threading.Lock()
        self._policy_output_session: _PolicyOutputSession | None = None

        # Motion lock — serialises go_home / _ik_servo / _ik_servo_keypoints /
        # move_joint_keypoints / move_bimanual_joint_keypoints so concurrent RPC
        # calls don't interleave writes to _cmd_*_jp and _pink_cfg.
        self._motion_lock = threading.Lock()

        # Fello leader arms (human-in-the-loop takeover) — None when USE_FELLO=False
        self._left_fello: object = None  # FelloLeaderClient | None
        self._right_fello: object = None  # FelloLeaderClient | None
        self._fello_arm_index_map: np.ndarray | None = None
        self._fello_gripper_index_map: int | None = None
        self._fello_align_cmd_scale: float = 1.0
        self._fello_align_kp: np.ndarray = np.ones(7, dtype=np.float32)
        self._fello_align_kd: np.ndarray = np.zeros(7, dtype=np.float32)
        self._fello_align_lpf_alpha: float = 0.3

        # Learn-skill live status — written by learn_skill loop, read by get_learn_skill_status
        self._learn_skill_status: dict = {"active": False}

        # Local reward function — set via set_reward_mode(). When not None, used by
        # learn_skill / learn_skill_policy instead of the external reward server.
        self._local_reward_fn = None  # Callable[[dict], float] | None
        self._local_reward_mode: str | None = None

        # Cached Fello state — written by _fello_loop, read by learn_skill (under _state_lock)
        self._fello_left_qpos: np.ndarray = np.zeros(7)
        self._fello_right_qpos: np.ndarray = np.zeros(7)
        self._fello_takeover_left: bool = False
        self._fello_takeover_right: bool = False

        if USE_FELLO:
            from enpire.env.forge.robot.fello.fello_config import load_fello_config
            from enpire.env.forge.robot.fello.fello_teleop_policy import FelloLeaderClient

            fello_cfg = load_fello_config()
            self._fello_arm_index_map = np.asarray(
                fello_cfg["mapping"]["arm_index_map"], dtype=int
            )
            self._fello_gripper_index_map = int(
                fello_cfg["mapping"]["gripper_index_map"]
            )
            _ctrl = fello_cfg.get("control", {})
            _tel = fello_cfg.get("teleop", {})
            self._fello_align_cmd_scale = float(_ctrl.get("command_scale", 1.0))
            self._fello_align_kp = np.asarray(
                _ctrl.get("kp", np.ones(7, dtype=np.float32)), dtype=np.float32
            )
            self._fello_align_kd = np.asarray(
                _ctrl.get("kd", np.zeros(7, dtype=np.float32)), dtype=np.float32
            )
            self._fello_align_lpf_alpha = float(_tel.get("align_lpf_alpha", 0.3))

            try:
                self._left_fello = FelloLeaderClient(
                    host=FELLO_HOST, port=LEFT_LEADER_PORT
                )
                logger.info(
                    "[CapServer] Fello left leader connected (%s:%d)",
                    FELLO_HOST,
                    LEFT_LEADER_PORT,
                )
            except Exception:
                logger.warning(
                    "[CapServer] Fello left leader not available (%s:%d)",
                    FELLO_HOST,
                    LEFT_LEADER_PORT,
                )

            try:
                self._right_fello = FelloLeaderClient(
                    host=FELLO_HOST, port=RIGHT_LEADER_PORT
                )
                logger.info(
                    "[CapServer] Fello right leader connected (%s:%d)",
                    FELLO_HOST,
                    RIGHT_LEADER_PORT,
                )
            except Exception:
                logger.warning(
                    "[CapServer] Fello right leader not available (%s:%d)",
                    FELLO_HOST,
                    RIGHT_LEADER_PORT,
                )

        # Control loop
        self._running = False
        self._loop_thread: threading.Thread | None = None
        self._fello_thread: threading.Thread | None = None

        # Portal RPC server
        self._server = portal.Server(server_port, workers=4)
        self._server.bind("get_state", self.get_state)
        self._server.bind("_ik_servo", self._ik_servo)
        self._server.bind("_ik_servo_keypoints", self._ik_servo_keypoints)
        self._server.bind("move_joint_keypoints", self.move_joint_keypoints)
        self._server.bind(
            "move_bimanual_joint_keypoints", self.move_bimanual_joint_keypoints
        )
        self._server.bind("set_gripper", self.set_gripper)
        self._server.bind("go_home", self.go_home)
        self._server.bind("list_cameras", self.list_cameras)
        self._server.bind("get_camera_image", self.get_camera_image)
        self._server.bind("get_camera_depth", self.get_camera_depth)
        self._server.bind("get_camera_intrinsics", self.get_camera_intrinsics)
        self._server.bind(
            "get_camera_intrinsics_full", self.get_camera_intrinsics_full
        )
        self._server.bind("get_camera_extrinsics", self.get_camera_extrinsics)
        self._server.bind("get_collision_geoms", self.get_collision_geoms)
        self._server.bind(
            "forward_kinematics_batch", self.forward_kinematics_batch
        )
        self._server.bind("get_visual_meshes", self.get_visual_meshes)
        self._server.bind("execute_skill", self.execute_skill, workers=1)
        self._server.bind("start_policy_output", self.start_policy_output, workers=1)
        self._server.bind("step_policy_output", self.step_policy_output, workers=1)
        self._server.bind("stop_policy_output", self.stop_policy_output, workers=1)
        self._server.bind("use_policy_output", self.use_policy_output, workers=1)
        self._server.bind("learn_skill", self.learn_skill, workers=1)
        self._server.bind("get_skill_prediction", self.get_skill_prediction)
        self._server.bind("get_learn_skill_status", self.get_learn_skill_status)
        self._server.bind("set_reward_mode", self.set_reward_mode)
        self._server.bind("cancel_motion", self.cancel_motion)
        self._server.bind("estop", self.estop)
        self._server.bind("release_estop", self.release_estop)
        self._server.bind("set_safety_zone", self.set_safety_zone)
        self._server.bind("clear_safety_zone", self.clear_safety_zone)
        self._server.bind("get_safety_zone", self.get_safety_zone)

        # Task backend RPCs — only available when backend supports tasks (RoboCasa)
        from enpire.env.forge.cap.env.base import TaskProtocol

        if self._sim_backend is not None and isinstance(
            self._sim_backend, TaskProtocol
        ):
            self._server.bind("reset_env", self._rpc_reset_env)
            self._server.bind("reset_to_initial", self._rpc_reset_to_initial)
            self._server.bind("get_task_info", self._rpc_get_task_info)
            self._server.bind("get_last_reward", self._rpc_get_last_reward)
            if hasattr(self._sim_backend, "load_task"):
                self._server.bind("load_task", self._rpc_load_task)

        # Scene management RPCs — only for envs that support SceneProtocol
        from enpire.env.forge.cap.env.base import SceneProtocol

        if self._sim_backend is not None and isinstance(
            self._sim_backend, SceneProtocol
        ):
            self._server.bind("setup_scene", self.setup_scene)
            self._server.bind("clear_table", self.clear_table)
            self._server.bind("list_scenes", self.list_scenes)
            self._server.bind("get_object_positions", self.get_object_positions)
            self._server.bind("set_body_pose", self.set_body_pose)
        else:
            _not_sim = lambda *a, **kw: {
                "ok": False,
                "error": "sim-only (real robot mode)",
            }
            self._server.bind("setup_scene", _not_sim)
            self._server.bind("clear_table", _not_sim)
            self._server.bind("list_scenes", _not_sim)
            self._server.bind("set_body_pose", _not_sim)
            # Stub so cap_agent broadcast loop doesn't crash Portal workers
            self._server.bind(
                "get_object_positions", lambda: {"ok": True, "objects": {}}
            )

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        """Seed state and commanded positions from hardware."""
        for side in self._arms:
            if side not in self._arm_jp:
                continue
            obs = self._arms[side].get_observations()
            dof = self._arm_jp[side].shape[0]
            jp = np.asarray(obs["joint_pos"], dtype=np.float64).ravel()[:dof]
            gp = np.asarray(obs["gripper_pos"], dtype=np.float64).ravel()[:1]
            with self._state_lock:
                self._arm_jp[side][:] = jp
                self._arm_gp[side][:] = gp
                self._cmd_jp[side][:] = jp
                self._cmd_gp[side][:] = gp

    # ------------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------------

    def _seed_q_from_arms(self, q: np.ndarray | None = None) -> np.ndarray:
        """Build pinocchio q-vector from current commanded joint positions."""
        if q is None:
            q = pin.neutral(self._pink_model).copy()
        if self._robot_profile is not None:
            for side, arm in self._robot_profile.arms.items():
                if arm.q_slice is not None and side in self._cmd_jp:
                    q[arm.q_slice] = self._cmd_jp[side]
        else:
            # Legacy YAM path
            if "left" in self._cmd_jp:
                q[:6] = self._cmd_jp["left"]
            if "right" in self._cmd_jp:
                q[8:14] = self._cmd_jp["right"]
        return q

    def _extract_arm_from_q(self, q: np.ndarray, side: str) -> np.ndarray:
        """Extract arm joint positions from pinocchio q-vector."""
        if self._robot_profile is not None:
            arm = self._robot_profile.arms[side]
            if arm.q_slice is not None:
                return q[arm.q_slice].copy()
        # Legacy YAM path
        if side == "left":
            return q[:6].copy()
        return q[8:14].copy()

    def _forward_kinematics(
        self, ljp: np.ndarray, rjp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """FK for both end-effectors (YAM legacy). Uses self._fk_data (control-loop thread only)."""
        q = pin.neutral(self._pink_model)
        q[:6] = ljp
        q[8:14] = rjp
        pin.forwardKinematics(self._pink_model, self._fk_data, q)
        pin.updateFramePlacements(self._pink_model, self._fk_data)
        l_T = self._fk_data.oMf[self._left_grasp_id]
        r_T = self._fk_data.oMf[self._right_grasp_id]
        return (
            l_T.translation.copy(),
            pin.Quaternion(l_T.rotation).coeffs(),  # xyzw
            r_T.translation.copy(),
            pin.Quaternion(r_T.rotation).coeffs(),  # xyzw
        )

    def _clamp_q_to_urdf(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint config to URDF limits so Pink doesn't raise."""
        return np.clip(q, self._pink_q_lower, self._pink_q_upper)

    def _inverse_kinematics(
        self,
        left_pos: np.ndarray,
        left_quat_xyzw: np.ndarray,
        right_pos: np.ndarray,
        right_quat_xyzw: np.ndarray,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """IK for both end-effectors from a seed config. Returns (left_jp, right_jp)."""
        cfg = pink.Configuration(
            self._pink_model,
            self._pink_model.createData(),
            self._clamp_q_to_urdf(q_seed),
        )
        l_task = pink.FrameTask("left_grasp", position_cost=1.0, orientation_cost=1.0)
        r_task = pink.FrameTask("right_grasp", position_cost=1.0, orientation_cost=1.0)
        posture = pink.PostureTask(cost=1e-3)
        l_task.set_target(_make_se3(left_pos, left_quat_xyzw))
        r_task.set_target(_make_se3(right_pos, right_quat_xyzw))
        posture.set_target_from_configuration(cfg)
        for _ in range(20):
            vel = pink.solve_ik(cfg, [l_task, r_task, posture], 0.01, solver="quadprog")
            cfg.integrate_inplace(vel, 0.01)
            if (
                np.linalg.norm(l_task.compute_error(cfg)) < 1e-4
                and np.linalg.norm(r_task.compute_error(cfg)) < 1e-4
            ):
                break
        return cfg.q[:6].copy(), cfg.q[8:14].copy()

    def _frame_pose(self, frame_name: str, ljp: np.ndarray, rjp: np.ndarray) -> pin.SE3:
        """FK for a single named frame. Allocates fresh data (safe to call from any thread)."""
        q = pin.neutral(self._pink_model)
        q[:6] = ljp
        q[8:14] = rjp
        fk_data = self._pink_model.createData()
        pin.forwardKinematics(self._pink_model, fk_data, q)
        pin.updateFramePlacements(self._pink_model, fk_data)
        return fk_data.oMf[self._pink_model.getFrameId(frame_name)]

    def _map_fello_qpos_to_yam(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map 7-DOF Fello qpos to YAM arm(6) + gripper(1)."""
        if self._fello_arm_index_map is None or self._fello_gripper_index_map is None:
            raise RuntimeError("Fello mapping is not initialized")
        qpos = np.asarray(qpos, dtype=np.float64).ravel()
        arm = qpos[self._fello_arm_index_map].astype(np.float64)
        grip = np.array([float(qpos[self._fello_gripper_index_map])], dtype=np.float64)
        return arm, grip

    # Number of ticks after takeover starts before the Fello gripper anchor
    # is re-sampled.  During this window the gripper motor settles after
    # switching from position → gravity mode (no holding torque).  Gripper
    # command holds yam_anchor_grip until the re-anchor, then tracks delta
    # from the settled Fello position.
    FELLO_GRIP_SETTLE_TICKS = 5
    FELLO_GRIP_TAKEOVER_DEADBAND = 0.05  # ignore Fello gripper drift below this

    def _compute_delta_takeover_cmd(
        self,
        yam_anchor_arm: np.ndarray,
        yam_anchor_grip: np.ndarray,
        fello_anchor_arm: np.ndarray,
        fello_anchor_grip: np.ndarray,
        fello_now_arm: np.ndarray,
        fello_now_grip: np.ndarray,
        grip_settled: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute press-anchored incremental takeover command in YAM joint space.

        When grip_settled=False the gripper is still settling after gravity-mode
        switch — hold yam_anchor_grip (no delta applied).
        """
        arm_cmd = np.asarray(yam_anchor_arm, dtype=np.float64) + (
            np.asarray(fello_now_arm, dtype=np.float64)
            - np.asarray(fello_anchor_arm, dtype=np.float64)
        )
        if not grip_settled:
            grip_scalar = float(
                np.asarray(yam_anchor_grip, dtype=np.float64).reshape(-1)[0]
            )
        else:
            f_now = float(np.asarray(fello_now_grip, dtype=np.float64).reshape(-1)[0])
            f_anc = float(
                np.asarray(fello_anchor_grip, dtype=np.float64).reshape(-1)[0]
            )
            y_anc = float(np.asarray(yam_anchor_grip, dtype=np.float64).reshape(-1)[0])
            fello_delta = f_now - f_anc
            # Scale Fello delta so full Fello range maps to full YAM range
            # in each direction, giving the operator [0, 1] control.
            if fello_delta >= 0:
                room_fello = max(1.0 - f_anc, 1e-4)
                room_yam = 1.0 - y_anc
                scale = room_yam / room_fello
            else:
                room_fello = max(f_anc, 1e-4)
                room_yam = y_anc
                scale = room_yam / room_fello
            grip_scalar = y_anc + fello_delta * scale
        grip_scalar = float(np.clip(grip_scalar, 0.0, 1.0))
        return arm_cmd.reshape(6), np.array([grip_scalar], dtype=np.float64)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """60Hz loop — reads hardware state, caches it, and sends commanded positions."""
        tick_count = 0
        prev_left_takeover = False
        prev_right_takeover = False
        left_yam_anchor_jp: np.ndarray | None = None
        left_yam_anchor_gp: np.ndarray | None = None
        right_yam_anchor_jp: np.ndarray | None = None
        right_yam_anchor_gp: np.ndarray | None = None
        left_fello_anchor_jp: np.ndarray | None = None
        left_fello_anchor_gp: np.ndarray | None = None
        right_fello_anchor_jp: np.ndarray | None = None
        right_fello_anchor_gp: np.ndarray | None = None
        while self._running:
            tick_start = time.time()

            # 1. Read actual state from hardware
            for side in self._arms:
                if side not in self._arm_jp:
                    continue
                try:
                    obs = self._arms[side].get_observations()
                    dof = self._arm_jp[side].shape[0]
                    jp = np.asarray(obs["joint_pos"], dtype=np.float64).ravel()[:dof]
                    gp = np.asarray(obs["gripper_pos"], dtype=np.float64).ravel()[:1]
                    with self._state_lock:
                        self._arm_jp[side][:] = jp
                        self._arm_gp[side][:] = gp
                except Exception as e:
                    logger.warning(f"[CapServer] Failed to read {side} arm: {e}")

            # 2. FK → cache EE poses
            if self._has_urdf_ik and "left" in self._arm_jp and "right" in self._arm_jp:
                # YAM path: pinocchio FK for both arms
                with self._state_lock:
                    ljp = self._arm_jp["left"].copy()
                    rjp = self._arm_jp["right"].copy()
                try:
                    l_pos, l_quat, r_pos, r_quat = self._forward_kinematics(ljp, rjp)
                    with self._state_lock:
                        self._ee_pos["left"][:] = l_pos
                        self._ee_quat["left"][:] = l_quat
                        self._ee_pos["right"][:] = r_pos
                        self._ee_quat["right"][:] = r_quat
                except Exception as e:
                    logger.warning(f"[CapServer] FK failed: {e}")
            else:
                # Backend provides EE poses directly (RoboCasa without URDF, etc.)
                for side in self._arm_jp:
                    if side not in self._arms:
                        continue
                    try:
                        obs = self._arms[side].get_observations()
                        if "ee_pos" in obs and "ee_quat" in obs:
                            with self._state_lock:
                                self._ee_pos[side][:] = obs["ee_pos"]
                                self._ee_quat[side][:] = obs["ee_quat"]
                    except Exception as e:
                        logger.warning(f"[CapServer] Backend FK for {side} failed: {e}")

            # 3. Estop → skip sending
            if self._safety.is_estopped():
                time.sleep(self._control_period_s)
                continue

            # 4. Send commanded positions
            with self._state_lock:
                ljp_cmd = self._cmd_left_jp.copy()
                lgp_cmd = self._cmd_left_gp.copy()
                rjp_cmd = self._cmd_right_jp.copy()
                rgp_cmd = self._cmd_right_gp.copy()
                lgv_cmd = self._cmd_left_gv
                rgv_cmd = self._cmd_right_gv
                lgt_cmd = self._cmd_left_gt
                rgt_cmd = self._cmd_right_gt
                left_takeover_now = False
                right_takeover_now = False
                # ALWAYS_TAKEOVERABLE: press-anchored incremental Fello takeover
                if (
                    ALWAYS_TAKEOVERABLE
                    and USE_FELLO
                    and self._fello_arm_index_map is not None
                ):
                    left_takeover_now = bool(self._fello_takeover_left)
                    right_takeover_now = bool(self._fello_takeover_right)
                    if left_takeover_now:
                        left_fello_now_jp, left_fello_now_gp = (
                            self._map_fello_qpos_to_yam(self._fello_left_qpos)
                        )
                        if (
                            not prev_left_takeover
                            or left_yam_anchor_jp is None
                            or left_yam_anchor_gp is None
                            or left_fello_anchor_jp is None
                            or left_fello_anchor_gp is None
                        ):
                            left_yam_anchor_jp = self.left_joint_pos.copy()
                            left_yam_anchor_gp = self.left_gripper_pos.copy()
                            left_fello_anchor_jp = left_fello_now_jp.copy()
                            left_fello_anchor_gp = left_fello_now_gp.copy()
                        ljp_cmd, lgp_cmd = self._compute_delta_takeover_cmd(
                            left_yam_anchor_jp,
                            left_yam_anchor_gp,
                            left_fello_anchor_jp,
                            left_fello_anchor_gp,
                            left_fello_now_jp,
                            left_fello_now_gp,
                        )
                        if tick_count % 200 == 0:
                            _tk_msg = (
                                f"[TAKEOVER-GRIP] fello_raw6={self._fello_left_qpos[6]:.3f}"
                                f" fello_gp={left_fello_now_gp[0]:.3f}"
                                f" anc_fello={left_fello_anchor_gp[0]:.3f}"
                                f" anc_yam={left_yam_anchor_gp[0]:.3f}"
                                f" lgp_cmd={lgp_cmd[0]:.3f}"
                                f" sim_gp={self.left_gripper_pos[0]:.3f}"
                            )
                            print(_tk_msg)
                            _hil_log(_tk_msg)
                    if right_takeover_now:
                        right_fello_now_jp, right_fello_now_gp = (
                            self._map_fello_qpos_to_yam(self._fello_right_qpos)
                        )
                        if (
                            not prev_right_takeover
                            or right_yam_anchor_jp is None
                            or right_yam_anchor_gp is None
                            or right_fello_anchor_jp is None
                            or right_fello_anchor_gp is None
                        ):
                            right_yam_anchor_jp = self.right_joint_pos.copy()
                            right_yam_anchor_gp = self.right_gripper_pos.copy()
                            right_fello_anchor_jp = right_fello_now_jp.copy()
                            right_fello_anchor_gp = right_fello_now_gp.copy()
                        rjp_cmd, rgp_cmd = self._compute_delta_takeover_cmd(
                            right_yam_anchor_jp,
                            right_yam_anchor_gp,
                            right_fello_anchor_jp,
                            right_fello_anchor_gp,
                            right_fello_now_jp,
                            right_fello_now_gp,
                        )

            if prev_left_takeover and not left_takeover_now:
                left_yam_anchor_jp = None
                left_yam_anchor_gp = None
                left_fello_anchor_jp = None
                left_fello_anchor_gp = None
            if prev_right_takeover and not right_takeover_now:
                right_yam_anchor_jp = None
                right_yam_anchor_gp = None
                right_fello_anchor_jp = None
                right_fello_anchor_gp = None
            prev_left_takeover = left_takeover_now
            prev_right_takeover = right_takeover_now

            # Clamp joint commands to limits
            if self._robot_profile is not None:
                for _s in self._arm_jp:
                    _ap = self._robot_profile.arms.get(_s)
                    if _ap and _s == "left":
                        ljp_cmd = np.clip(
                            ljp_cmd[: _ap.dof],
                            _ap.joint_limits_low,
                            _ap.joint_limits_high,
                        )
                        lgp_cmd = np.clip(lgp_cmd, _ap.gripper_min, _ap.gripper_max)
                    elif _ap and _s == "right":
                        rjp_cmd = np.clip(
                            rjp_cmd[: _ap.dof],
                            _ap.joint_limits_low,
                            _ap.joint_limits_high,
                        )
                        rgp_cmd = np.clip(rgp_cmd, _ap.gripper_min, _ap.gripper_max)
            else:
                ljp_cmd = np.clip(ljp_cmd, JOINT_LIMITS_LOW[:6], JOINT_LIMITS_HIGH[:6])
                rjp_cmd = np.clip(rjp_cmd, JOINT_LIMITS_LOW[6:], JOINT_LIMITS_HIGH[6:])
                lgp_cmd = np.clip(lgp_cmd, GRIPPER_MIN, GRIPPER_MAX)
                rgp_cmd = np.clip(rgp_cmd, GRIPPER_MIN, GRIPPER_MAX)

            _cmd_pairs = []
            if "left" in self._arms:
                _cmd_pairs.append(("left", ljp_cmd, lgp_cmd, lgv_cmd, lgt_cmd))
            if "right" in self._arms:
                _cmd_pairs.append(("right", rjp_cmd, rgp_cmd, rgv_cmd, rgt_cmd))
            # Check if any EE targets are active (from _ik_servo)
            from enpire.env.forge.cap.env.base import EefControlProtocol

            _has_eef_control = self._sim_backend is not None and isinstance(
                self._sim_backend, EefControlProtocol
            )

            for side, jp, gp, gv, gt in _cmd_pairs:
                # If an EE target is active for this arm, use compute_eef_action
                _eef = self._eef_targets.get(side)
                if _eef is not None and _has_eef_control:
                    _tgt_pos, _tgt_quat, _tgt_grip = _eef
                    self._sim_backend.compute_eef_action(
                        side, _tgt_pos, _tgt_quat, gripper=_tgt_grip
                    )
                    # compute_eef_action sets _pending_eef_action on the backend;
                    # step() below will use it
                else:
                    _arm_prof = (
                        self._robot_profile.arms.get(side)
                        if self._robot_profile
                        else None
                    )
                    cmd = {
                        "pos": np.concatenate([jp, gp]),
                        "vel": np.zeros(len(jp) + len(gp)),
                        "kp": _arm_prof.interp_kp if _arm_prof else INTERP_KP,
                        "kd": _arm_prof.interp_kd if _arm_prof else INTERP_KD,
                    }
                    if gv is not None:
                        cmd["gripper_vel_limit"] = float(gv)
                    if gt is not None:
                        cmd["gripper_torque_limit_nm"] = float(gt)
                    self._arms[side].command_joint_state(cmd)

            if self._sim_backend is not None:
                self._sim_backend.step()

            # Sleep until next tick.  On macOS, time.sleep() overshoots by
            # ~2 ms due to coarse timer resolution, so we sleep only up to a
            # threshold and busy-wait the remainder for precise timing.
            deadline = tick_start + self._control_period_s
            remaining = deadline - time.time()
            if remaining > 0:
                _BUSYWAIT_S = 0.002  # busy-wait the last 2 ms
                coarse = remaining - _BUSYWAIT_S
                if coarse > 0:
                    time.sleep(coarse)
                while time.time() < deadline:
                    pass  # spin
            else:
                logger.warning(
                    f"[CapServer] Loop overrun: {(time.time() - tick_start) * 1000:.1f}ms"
                )

            tick_count += 1
            if tick_count % 20 == 0:
                emit("control_loop", "tick", dt_ms=(time.time() - tick_start) * 1000)

    def _fello_loop(self) -> None:
        """~POLICY_FREQ_HZ loop — reads Fello state, caches it, and aligns non-takeover arms to robot.

        Mode logic (mirrors FelloTeleopPolicy.get_action):
          - Button NOT pressed → "position" mode: arm actively tracks robot via LPF
          - Button pressed     → "gravity" mode:  arm floats freely for human to move
        LPF is seeded from Fello's current qpos on gravity→position transition to avoid jumps.
        """
        align_filtered_left: np.ndarray | None = None
        align_filtered_right: np.ndarray | None = None
        left_mode: str | None = None  # last mode sent to left Fello
        right_mode: str | None = None  # last mode sent to right Fello

        # Probe each arm with a timeout — Portal clients block indefinitely
        # on a missing server, so we disable unreachable arms early.
        _PROBE_TIMEOUT_S = 3.0
        for _side, _attr in [("left", "_left_fello"), ("right", "_right_fello")]:
            _client = getattr(self, _attr)
            if _client is None:
                continue
            _probe_ok = [False]

            def _probe(client=_client, result=_probe_ok):
                try:
                    client.get_info()
                    result[0] = True
                except Exception:
                    pass

            _t = threading.Thread(target=_probe, daemon=True)
            _t.start()
            _t.join(timeout=_PROBE_TIMEOUT_S)
            if _probe_ok[0]:
                logger.info("[CapServer] Fello %s arm reachable", _side)
            else:
                logger.warning(
                    "[CapServer] Fello %s arm unreachable (timeout %.0fs) — disabling",
                    _side,
                    _PROBE_TIMEOUT_S,
                )
                setattr(self, _attr, None)

        while self._running:
            tick_start = time.time()

            # Read Fello state (best-effort, each arm independent)
            lq = rq = None
            left_takeover = right_takeover = False
            _t_read = time.time()

            if self._left_fello is not None:
                try:
                    lq, lb = self._left_fello.get_info()
                    lq = np.asarray(lq, dtype=np.float64).ravel()
                    lb = np.asarray(lb, dtype=np.float32).ravel()
                    left_takeover = lb.size > 1 and bool(lb[1] > 0.5)
                except Exception:
                    lq = None
                    logger.debug("[CapServer] Left Fello get_info failed")

            if self._right_fello is not None:
                try:
                    rq, rb = self._right_fello.get_info()
                    rq = np.asarray(rq, dtype=np.float64).ravel()
                    rb = np.asarray(rb, dtype=np.float32).ravel()
                    right_takeover = rb.size > 1 and bool(rb[1] > 0.5)
                except Exception:
                    rq = None
                    logger.debug("[CapServer] Right Fello get_info failed")

            if lq is None and rq is None:
                time.sleep(POLICY_PERIOD_S)
                continue
            read_elapsed = time.time() - _t_read

            # Cache state (under state lock) and snapshot robot positions
            with self._state_lock:
                if lq is not None:
                    self._fello_left_qpos[:] = lq
                if rq is not None:
                    self._fello_right_qpos[:] = rq
                self._fello_takeover_left = left_takeover
                self._fello_takeover_right = right_takeover
                robot_ljp = self.left_joint_pos.copy()
                robot_lgp = self.left_gripper_pos.copy()
                robot_rjp = self.right_joint_pos.copy()
                robot_rgp = self.right_gripper_pos.copy()

            idxmap = self._fello_arm_index_map
            gidx = self._fello_gripper_index_map
            alpha = self._fello_align_lpf_alpha

            # Left arm
            if lq is not None:
                desired_left_mode = "gravity" if left_takeover else "position"
                if desired_left_mode != left_mode:
                    if desired_left_mode == "position":
                        # Seed LPF from Fello's current position to avoid jump on release
                        align_filtered_left = lq.astype(np.float32).copy()
                    else:
                        align_filtered_left = None
                    try:
                        self._left_fello.set_mode(desired_left_mode)
                        left_mode = desired_left_mode
                    except Exception:
                        logger.debug(
                            "[CapServer] left Fello set_mode(%s) failed",
                            desired_left_mode,
                        )

                if (
                    not left_takeover
                    and align_filtered_left is not None
                    and len(robot_ljp) == len(idxmap)
                ):
                    target_l = np.zeros(7, dtype=np.float32)
                    target_l[idxmap] = robot_ljp.astype(np.float32)
                    target_l[gidx] = float(robot_lgp[0])
                    align_filtered_left = (
                        alpha * target_l + (1.0 - alpha) * align_filtered_left
                    )
                    try:
                        self._left_fello.command_joint_pos(
                            {
                                "pos": align_filtered_left
                                / self._fello_align_cmd_scale,
                                "vel": np.zeros(7, dtype=np.float32),
                                "kp": self._fello_align_kp,
                                "kd": self._fello_align_kd,
                            }
                        )
                    except Exception:
                        logger.debug("[CapServer] left Fello alignment command failed")

            # Right arm
            if rq is not None:
                desired_right_mode = "gravity" if right_takeover else "position"
                if desired_right_mode != right_mode:
                    if desired_right_mode == "position":
                        align_filtered_right = rq.astype(np.float32).copy()
                    else:
                        align_filtered_right = None
                    try:
                        self._right_fello.set_mode(desired_right_mode)
                        right_mode = desired_right_mode
                    except Exception:
                        logger.debug(
                            "[CapServer] right Fello set_mode(%s) failed",
                            desired_right_mode,
                        )

                if (
                    not right_takeover
                    and align_filtered_right is not None
                    and len(robot_rjp) == len(idxmap)
                ):
                    target_r = np.zeros(7, dtype=np.float32)
                    target_r[idxmap] = robot_rjp.astype(np.float32)
                    target_r[gidx] = float(robot_rgp[0])
                    align_filtered_right = (
                        alpha * target_r + (1.0 - alpha) * align_filtered_right
                    )
                    try:
                        self._right_fello.command_joint_pos(
                            {
                                "pos": align_filtered_right
                                / self._fello_align_cmd_scale,
                                "vel": np.zeros(7, dtype=np.float32),
                                "kp": self._fello_align_kp,
                                "kd": self._fello_align_kd,
                            }
                        )
                    except Exception:
                        logger.debug("[CapServer] right Fello alignment command failed")

            elapsed = time.time() - tick_start
            emit(
                "fello_loop",
                "tick",
                dt_ms=elapsed * 1000,
                takeover_left=self._fello_takeover_left,
                takeover_right=self._fello_takeover_right,
                read_ms=read_elapsed * 1000,
            )
            sleep = POLICY_PERIOD_S - elapsed
            if sleep > 0:
                time.sleep(sleep)

    # ------------------------------------------------------------------
    # RPC methods
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return robot state.

        For envs with native get_robot_state (RoboCasa), reads directly from
        the env observations.  For control-loop envs, returns cached state.
        """
        with self._state_lock:
            state = {}
            for side in self._arm_jp:
                state[f"{side}_joint_pos"] = self._arm_jp[side].copy()
                state[f"{side}_gripper_pos"] = self._arm_gp[side].copy()
                state[f"{side}_ee_pos"] = self._ee_pos[side].copy()
                state[f"{side}_ee_quat_xyzw"] = self._ee_quat[side].copy()
            # Include base pose for mobile-base robots
            if self._sim_backend is not None and hasattr(
                self._sim_backend, "get_base_pose"
            ):
                try:
                    bp = self._sim_backend.get_base_pose()
                    if "base_pos" in bp:
                        state["base_pos"] = bp["base_pos"]
                    if "base_quat" in bp:
                        state["base_quat_xyzw"] = bp["base_quat"]
                except Exception:
                    pass
            return state

    def get_collision_geoms(
        self,
        exclude_body_prefixes: list[str] | None = None,
        max_dist: float = 1.2,
        min_size: float = 0.03,
    ) -> dict:
        """Extract collision geometry from the sim as serializable dicts.

        Returns a dict with keys:
            base_pos: [x, y, z] arm base position in world frame
            base_quat_xyzw: [x, y, z, w] arm base quaternion
            geoms: list of {name, type, pos, rot_mat, size} dicts
        """
        import mujoco

        if self._sim_backend is None:
            return {"base_pos": [0, 0, 0], "base_quat_xyzw": [0, 0, 0, 1], "geoms": []}

        if exclude_body_prefixes is None:
            exclude_body_prefixes = ["robot0", "gripper", "mobilebase"]

        # Get base pose
        base_pos = np.zeros(3)
        base_quat = np.array([0.0, 0.0, 0.0, 1.0])
        if hasattr(self._sim_backend, "get_base_pose"):
            try:
                bp = self._sim_backend.get_base_pose()
                if "base_pos" in bp:
                    base_pos = np.asarray(bp["base_pos"], dtype=np.float64)
                if "base_quat" in bp:
                    base_quat = np.asarray(bp["base_quat"], dtype=np.float64)
            except Exception:
                pass

        # Access MuJoCo model/data
        env = getattr(self._sim_backend, "_env", None)
        sim = getattr(env, "sim", None) if env is not None else None
        if sim is None:
            return {"base_pos": base_pos.tolist(), "base_quat_xyzw": base_quat.tolist(), "geoms": []}

        model = sim.model._model
        data = sim.data._data

        names = []
        positions_list = []
        rot_mats_list = []
        dims_list = []
        for i in range(model.ngeom):
            if model.geom_group[i] != 0:
                continue
            gtype = int(model.geom_type[i])
            if gtype not in (5, 6):  # cylinder, box
                continue
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if any(body_name.startswith(p) for p in exclude_body_prefixes):
                continue
            pos_world = data.geom_xpos[i].copy()
            dist = float(np.linalg.norm(pos_world[:2] - base_pos[:2]))
            if dist > max_dist:
                continue
            size = model.geom_size[i].copy()
            if gtype == 6:
                dims = size * 2.0
            elif gtype == 5:
                dims = np.array([size[0] * 2, size[0] * 2, size[1] * 2])
            else:
                continue
            if np.max(dims) < min_size:
                continue
            rot_mat = data.geom_xmat[i].reshape(3, 3).copy()
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
            names.append(geom_name)
            positions_list.append(pos_world)
            rot_mats_list.append(rot_mat)
            dims_list.append(dims)

        # Pack as numpy arrays for Portal serialization (dicts of lists are too slow)
        n = len(names)
        if n > 0:
            positions_arr = np.array(positions_list, dtype=np.float64)
            rot_mats_arr = np.array(rot_mats_list, dtype=np.float64)
            dims_arr = np.array(dims_list, dtype=np.float64)
        else:
            positions_arr = np.empty((0, 3), dtype=np.float64)
            rot_mats_arr = np.empty((0, 3, 3), dtype=np.float64)
            dims_arr = np.empty((0, 3), dtype=np.float64)

        return {
            "base_pos": base_pos,
            "base_quat_xyzw": base_quat,
            "names": names,
            "positions": positions_arr,
            "rot_mats": rot_mats_arr,
            "dims_array": dims_arr,
            "n_geoms": n,
        }

    def forward_kinematics_batch(
        self, side: str, joint_positions: np.ndarray
    ) -> dict:
        """Compute FK for a batch of joint configurations using MuJoCo.

        Returns dict with ee_positions (N,3), ee_quats_xyzw (N,4),
        base_pos (3,), base_quat_xyzw (4,) — all in world frame.
        """
        import mujoco
        from scipy.spatial.transform import Rotation as ScipyR

        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        if joint_positions.ndim == 1:
            joint_positions = joint_positions.reshape(1, -1)
        n_configs = joint_positions.shape[0]
        dof = joint_positions.shape[1]

        empty = {
            "ee_positions": np.zeros((n_configs, 3)),
            "ee_quats_xyzw": np.tile([0, 0, 0, 1.0], (n_configs, 1)),
            "base_pos": np.zeros(3),
            "base_quat_xyzw": np.array([0, 0, 0, 1.0]),
        }

        env = getattr(self._sim_backend, "_env", None)
        sim = getattr(env, "sim", None) if env is not None else None
        if sim is None:
            return empty

        model = sim.model._model
        data = sim.data._data

        # Find EE site — try robosuite robot attribute first, fallback to grip_site
        site_id = -1
        robots = getattr(env, "robots", None)
        if robots and len(robots) > 0:
            sid = getattr(robots[0], "eef_site_id", None)
            if sid is not None:
                site_id = int(sid) if not isinstance(sid, dict) else int(
                    sid.get(side, sid.get("right", -1))
                )
        if site_id < 0:
            site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, "grip_site"
            )
        if site_id < 0:
            logger.warning("[CapServer] forward_kinematics_batch: no EE site found")
            return empty

        # Find joint qpos indices
        qpos_addrs = []
        for j in range(1, dof + 1):
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{j}"
            )
            if jid < 0:
                logger.warning("[CapServer] FK: joint robot0_joint%d not found", j)
                return empty
            qpos_addrs.append(model.jnt_qposadr[jid])

        ee_positions = np.zeros((n_configs, 3))
        ee_quats = np.zeros((n_configs, 4))

        # Use a separate MjData copy to avoid racing with the control loop.
        # Snapshot current qpos so the FK sees the full scene (base, objects).
        fk_data = mujoco.MjData(model)
        fk_data.qpos[:] = data.qpos
        fk_data.qvel[:] = data.qvel

        for i in range(n_configs):
            for j, addr in enumerate(qpos_addrs):
                fk_data.qpos[addr] = joint_positions[i, j]
            mujoco.mj_forward(model, fk_data)
            ee_positions[i] = fk_data.site_xpos[site_id].copy()
            mat = fk_data.site_xmat[site_id].reshape(3, 3)
            ee_quats[i] = ScipyR.from_matrix(mat).as_quat()

        # Get base pose
        base_pos = np.zeros(3)
        base_quat = np.array([0.0, 0.0, 0.0, 1.0])
        if hasattr(self._sim_backend, "get_base_pose"):
            try:
                bp = self._sim_backend.get_base_pose()
                if "base_pos" in bp:
                    base_pos = np.asarray(bp["base_pos"], dtype=np.float64)
                if "base_quat" in bp:
                    base_quat = np.asarray(bp["base_quat"], dtype=np.float64)
            except Exception:
                pass

        return {
            "ee_positions": ee_positions,
            "ee_quats_xyzw": ee_quats,
            "base_pos": base_pos,
            "base_quat_xyzw": base_quat,
        }

    def get_visual_meshes(
        self,
        exclude_body_prefixes: list[str] | None = None,
    ) -> dict:
        """Extract visual mesh geometry from MuJoCo for 3D rendering.

        Returns packed arrays for Portal-friendly serialization:
            n_meshes, vert_counts, face_counts, all_vertices, all_faces,
            all_colors, all_positions, all_rot_mats
        """
        import mujoco

        empty = {
            "n_meshes": 0,
            "vert_counts": np.empty(0, dtype=np.int32),
            "face_counts": np.empty(0, dtype=np.int32),
            "all_vertices": np.empty((0, 3), dtype=np.float32),
            "all_faces": np.empty((0, 3), dtype=np.int32),
            "all_colors": np.empty((0, 4), dtype=np.float32),
        }

        env = getattr(self._sim_backend, "_env", None)
        sim = getattr(env, "sim", None) if env is not None else None
        if sim is None:
            return empty

        model = sim.model._model
        data = sim.data._data

        if exclude_body_prefixes is None:
            exclude_body_prefixes = ["robot0", "gripper"]

        vert_counts = []
        face_counts = []
        all_verts = []
        all_faces = []
        all_colors = []
        vert_offset = 0

        for i in range(model.ngeom):
            gtype = int(model.geom_type[i])
            # Only mesh geoms (type 7)
            if gtype != 7:
                continue

            body_id = model.geom_bodyid[i]
            body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or ""
            )
            if any(body_name.startswith(p) for p in exclude_body_prefixes):
                continue

            mesh_id = model.geom_dataid[i]
            if mesh_id < 0:
                continue

            nv = model.mesh_vertnum[mesh_id]
            nf = model.mesh_facenum[mesh_id]
            if nv == 0 or nf == 0:
                continue

            va = model.mesh_vertadr[mesh_id]
            fa = model.mesh_faceadr[mesh_id]
            verts_local = np.ascontiguousarray(model.mesh_vert[va : va + nv])  # (nv, 3)
            faces = np.ascontiguousarray(model.mesh_face[fa : fa + nf])  # (nf, 3)

            # Transform vertices to world frame
            pos = data.geom_xpos[i]
            rot = data.geom_xmat[i].reshape(3, 3)
            verts_world = (rot @ verts_local.T).T + pos

            # Color: use geom rgba, fall back to material
            rgba = model.geom_rgba[i].copy()
            if rgba[3] == 0 and model.geom_matid[i] >= 0:
                rgba = model.mat_rgba[model.geom_matid[i]].copy()

            all_verts.append(verts_world.astype(np.float32))
            all_faces.append((faces + vert_offset).astype(np.int32))
            all_colors.append(rgba.astype(np.float32))
            vert_counts.append(nv)
            face_counts.append(nf)
            vert_offset += nv

        n = len(vert_counts)
        if n == 0:
            return empty

        return {
            "n_meshes": n,
            "vert_counts": np.ascontiguousarray(vert_counts, dtype=np.int32),
            "face_counts": np.ascontiguousarray(face_counts, dtype=np.int32),
            "all_vertices": np.ascontiguousarray(np.concatenate(all_verts, axis=0)),
            "all_faces": np.ascontiguousarray(np.concatenate(all_faces, axis=0)),
            "all_colors": np.ascontiguousarray(all_colors, dtype=np.float32),
        }

    def _ik_servo(
        self,
        side: str,
        pos: np.ndarray,
        quat: np.ndarray,
        gripper: float | None = None,
        max_duration_sec: float = MOVE_EEF_MAX_DURATION_S,
        max_vel: float = MOVE_EEF_MAX_VEL,
        check_feasibility: bool = False,
        tol: float = 0.02,
        dry_run_only: bool = False,
    ) -> dict:
        """Move end-effector to target pose. Blocking.

        If the env implements EefControlProtocol, delegates to its native controller.
        Otherwise falls back to pinocchio IK (YAM).
        """
        from enpire.env.forge.cap.env.base import EefControlProtocol

        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)  # xyzw

        # EefControlProtocol: set target, let control loop drive, wait for convergence
        if self._sim_backend is not None and isinstance(
            self._sim_backend, EefControlProtocol
        ):
            self._mark_policy_output_dirty("_ik_servo")
            cancel_event = self._cancel_motion.get(side)
            if cancel_event is not None:
                cancel_event.clear()
            self._eef_targets[side] = (pos.copy(), quat.copy(), gripper)
            start = time.time()
            while time.time() - start < max_duration_sec:
                if cancel_event is not None and cancel_event.is_set():
                    cancel_event.clear()
                    self._eef_targets.pop(side, None)
                    return {
                        "success": False,
                        "reason": "cancelled",
                        "feasible": True,
                        "moved": True,
                        "pos_err": float("inf"),
                    }
                with self._state_lock:
                    ee_now = self._ee_pos.get(side)
                if ee_now is not None and np.linalg.norm(ee_now - pos) < tol:
                    self._eef_targets.pop(side, None)
                    return {
                        "success": True,
                        "reason": "ok",
                        "feasible": True,
                        "moved": True,
                        "pos_err": float(np.linalg.norm(ee_now - pos)),
                    }
                time.sleep(self._control_period_s)
            self._eef_targets.pop(side, None)
            with self._state_lock:
                ee_now = self._ee_pos.get(side)
            pos_err = (
                float(np.linalg.norm(ee_now - pos))
                if ee_now is not None
                else float("inf")
            )
            return {
                "success": False,
                "reason": "timeout",
                "feasible": True,
                "moved": True,
                "pos_err": pos_err,
            }

        valid_sides = set(self._arm_jp.keys())
        if side not in valid_sides:
            return {
                "success": False,
                "reason": f"invalid side: {side} (available: {valid_sides})",
                "feasible": False,
                "moved": False,
            }

        if not self._has_urdf_ik:
            return {
                "success": False,
                "reason": "_ik_servo requires URDF-based IK (not available for this backend)",
                "feasible": False,
                "moved": False,
            }

        self._mark_policy_output_dirty("_ik_servo")
        with self._motion_lock:
            _cancel_event = self._cancel_motion.get(side)
            if _cancel_event is not None:
                _cancel_event.clear()

            # Feasibility threshold used for pre-check and reporting.
            _feasible_pos_err_thresh = max(1e-4, float(tol))

            # Seed pink from current state. For mobile-base robots, use actual
            # robot state (not commanded) since cmd may lag behind actual due to
            # controller dynamics, and the frame transform is computed from actual.
            with self._state_lock:
                q = pin.neutral(self._pink_model).copy()
                if self._robot_profile and self._robot_profile.is_mobile_base:
                    for _s, _arm in self._robot_profile.arms.items():
                        if _arm.q_slice is not None and _s in self._arm_jp:
                            q[_arm.q_slice] = self._arm_jp[_s].copy()
                else:
                    q = self._seed_q_from_arms(q)
            self._pink_cfg.update(self._clamp_q_to_urdf(q))

            # Hold non-moving sides at current position; move the target side
            for _s, _task in self._pink_tasks.items():
                _task.set_target_from_configuration(self._pink_cfg)
            moving_task = self._pink_tasks[side]
            moving_task.set_target(_make_se3(pos, quat))
            self._posture.set_target_from_configuration(self._pink_cfg)

            tasks = list(self._pink_tasks.values()) + [self._posture]
            max_iters = int(max_duration_sec / self._control_period_s)
            preplanned_executed = False
            preplanned_ik_iters = 0
            preplanned_stall_count = 0
            preplanned_pos_err = float("inf")

            if check_feasibility:
                _ik_iters_check = 0
                _stall_window_check = 20
                _stall_threshold_check = 1e-4
                _prev_pos_err_check = float("inf")
                _stall_count_check = 0
                _ik_err_check = float("inf")
                _pos_err_check = float("inf")
                _planned_q: list[np.ndarray] = []

                for _ in range(max_iters):
                    _ik_iters_check += 1
                    _pos_err_check = np.linalg.norm(
                        moving_task.compute_error(self._pink_cfg)[:3]
                    )
                    moving_task.gain = min(
                        1.0, max_vel * self._control_period_s / (_pos_err_check + 1e-6)
                    )
                    vel = pink.solve_ik(
                        self._pink_cfg, tasks, self._control_period_s, solver="quadprog"
                    )
                    self._pink_cfg.integrate_inplace(vel, self._control_period_s)
                    _planned_q.append(self._pink_cfg.q.copy())
                    _ik_err_check = np.linalg.norm(
                        moving_task.compute_error(self._pink_cfg)
                    )

                    if _ik_err_check < 1e-3:
                        break

                    if _ik_iters_check % _stall_window_check == 0:
                        improvement = _prev_pos_err_check - _pos_err_check
                        if improvement < _stall_threshold_check:
                            _stall_count_check += 1
                        else:
                            _stall_count_check = 0
                        _prev_pos_err_check = _pos_err_check
                        if _stall_count_check >= 2:
                            break

                _final_cmd_pos_err = np.linalg.norm(
                    moving_task.compute_error(self._pink_cfg)[:3]
                )
                feasible = bool(
                    _ik_err_check < 1e-3
                    or _final_cmd_pos_err < _feasible_pos_err_thresh
                )
                if not feasible:
                    print(
                        f"[_ik_servo:{side}] infeasible pre-check: ik_err={_ik_err_check:.4f}, "
                        f"cmd_pos_err={_final_cmd_pos_err:.4f}m (no motion executed)"
                    )
                    return {
                        "success": True,
                        "reason": "infeasible",
                        "feasible": False,
                        "moved": False,
                        "ik_err": float(_ik_err_check),
                        "cmd_pos_err": float(_final_cmd_pos_err),
                    }

                if dry_run_only:
                    return {
                        "success": True,
                        "reason": "feasible_dry_run",
                        "feasible": True,
                        "moved": False,
                        "ik_err": float(_ik_err_check),
                        "cmd_pos_err": float(_final_cmd_pos_err),
                    }
                else:
                    # Plan-then-commit: reuse solved IK path (no second IK pass).
                    for q_step in _planned_q:
                        with self._state_lock:
                            for _s in self._cmd_jp:
                                self._cmd_jp[_s][:] = self._extract_arm_from_q(
                                    q_step, _s
                                )
                            if gripper is not None:
                                if side == "left":
                                    self._cmd_left_gp[0] = float(
                                        np.clip(gripper, 0.0, 1.0)
                                    )
                                else:
                                    self._cmd_right_gp[0] = float(
                                        np.clip(gripper, 0.0, 1.0)
                                    )
                        time.sleep(self._control_period_s)
                    preplanned_executed = True
                    preplanned_ik_iters = _ik_iters_check
                    preplanned_stall_count = _stall_count_check
                    preplanned_pos_err = _final_cmd_pos_err

            ik_converged = False
            _ik_iters = 0
            _settle_iters = 0
            _stall_window = 20  # check improvement over last N iters
            _stall_threshold = 1e-4  # min pos_err improvement to not be stalled
            _prev_pos_err = float("inf")
            _stall_count = 0
            last_pos_err = float("inf")
            if preplanned_executed:
                ik_converged = True
                _ik_iters = preplanned_ik_iters
                _stall_count = preplanned_stall_count
                last_pos_err = preplanned_pos_err
            for _ in range(max_iters):
                if _cancel_event is not None and _cancel_event.is_set():
                    _cancel_event.clear()
                    return {
                        "success": False,
                        "reason": "cancelled",
                        "feasible": True,
                        "moved": True,
                        "cmd_pos_err": float(last_pos_err),
                    }
                if not ik_converged:
                    _ik_iters += 1
                    # Phase 1: step IK until commanded config reaches target
                    pos_err = np.linalg.norm(
                        moving_task.compute_error(self._pink_cfg)[:3]
                    )
                    last_pos_err = pos_err
                    moving_task.gain = min(
                        1.0, max_vel * self._control_period_s / (pos_err + 1e-6)
                    )
                    vel = pink.solve_ik(
                        self._pink_cfg, tasks, self._control_period_s, solver="quadprog"
                    )
                    self._pink_cfg.integrate_inplace(vel, self._control_period_s)
                    q = self._pink_cfg.q
                    with self._state_lock:
                        for _s in self._cmd_jp:
                            self._cmd_jp[_s][:] = self._extract_arm_from_q(q, _s)
                        if gripper is not None:
                            if side == "left":
                                self._cmd_left_gp[0] = float(np.clip(gripper, 0.0, 1.0))
                            else:
                                self._cmd_right_gp[0] = float(
                                    np.clip(gripper, 0.0, 1.0)
                                )
                    ik_err = np.linalg.norm(moving_task.compute_error(self._pink_cfg))

                    # Stall detection: if pos_err hasn't improved, stop early
                    if _ik_iters % _stall_window == 0:
                        improvement = _prev_pos_err - pos_err
                        if improvement < _stall_threshold:
                            _stall_count += 1
                        else:
                            _stall_count = 0
                        _prev_pos_err = pos_err
                        if _stall_count >= 2:
                            ik_converged = True
                            print(
                                f"[_ik_servo:{side}] IK stalled after {_ik_iters} iters "
                                f"(pos_err={pos_err:.4f}m), moving to settle phase"
                            )
                            continue

                    if ik_err < 1e-3:
                        ik_converged = True
                        print(f"[_ik_servo:{side}] IK converged after {_ik_iters} iters")
                else:
                    _settle_iters += 1
                    # Phase 2: poll actual hardware EE pose until settled
                    with self._state_lock:
                        actual_pos = (
                            self._left_ee_pos.copy()
                            if side == "left"
                            else self._right_ee_pos.copy()
                        )
                        actual_quat = (
                            self._left_ee_quat.copy()
                            if side == "left"
                            else self._right_ee_quat.copy()
                        )
                    pos_hw_err = np.linalg.norm(actual_pos - pos)
                    # angular distance: θ = 2·arccos(|q_actual · q_target|)
                    ori_hw_err = 2.0 * np.arccos(
                        np.clip(abs(np.dot(actual_quat, quat)), 0.0, 1.0)
                    )
                    if _settle_iters % 30 == 0:
                        print(
                            f"[_ik_servo:{side}] settle iter {_settle_iters}: pos_err={pos_hw_err:.4f}m ori_err={np.rad2deg(ori_hw_err):.1f}°"
                        )
                    _settle_pos_thresh = _feasible_pos_err_thresh
                    if pos_hw_err < _settle_pos_thresh and ori_hw_err < np.deg2rad(5.0):
                        break
                time.sleep(self._control_period_s)

            if not ik_converged:
                print(
                    f"[_ik_servo:{side}] TIMEOUT: IK did not converge after {_ik_iters} iters"
                )
            elif _settle_iters > 0:
                print(
                    f"[_ik_servo:{side}] settled after {_settle_iters} iters (total {_ik_iters + _settle_iters})"
                )

            final_cmd_pos_err = np.linalg.norm(
                moving_task.compute_error(self._pink_cfg)[:3]
            )
            feasible = bool(final_cmd_pos_err < _feasible_pos_err_thresh)

        return {
            "success": True,
            "reason": "ok",
            "feasible": feasible,
            "moved": True,
            "cmd_pos_err": float(final_cmd_pos_err),
        }

    def _ik_servo_keypoints(
        self,
        side: str,
        timestamps: list[float],
        keypoints: list,
        max_vel: float = MOVE_EEF_MAX_VEL,
    ) -> dict:
        """Move end-effector through timestamped EEF waypoints using pink IK.

        timestamps: monotonically increasing times in seconds, relative to call time.
        keypoints:  one per timestamp; each a 6-element sequence [px, py, pz, roll, pitch, yaw]
                    where roll/pitch/yaw are in radians (RPY / extrinsic XYZ convention).
        """
        if side not in self._arm_jp:
            return {"success": False, "reason": f"invalid side: {side}"}
        if len(timestamps) != len(keypoints) or len(timestamps) < 1:
            return {
                "success": False,
                "reason": "timestamps and keypoints must be non-empty and equal length",
            }

        def _rpy_to_quat_xyzw(rpy):
            r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
            cr, cp, cy = math.cos(r / 2), math.cos(p / 2), math.cos(y / 2)
            sr, sp, sy = math.sin(r / 2), math.sin(p / 2), math.sin(y / 2)
            return np.array(
                [
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                    cr * cp * cy + sr * sp * sy,
                ]
            )

        kp_pos, kp_quat = [], []
        for kp in keypoints:
            arr = np.asarray(kp, dtype=np.float64).ravel()
            if arr.size < 6:
                return {
                    "success": False,
                    "reason": f"each keypoint must have 6 elements [px,py,pz,roll,pitch,yaw], got {arr.size}",
                }
            p, q = arr[:3], _rpy_to_quat_xyzw(arr[3:6])
            kp_pos.append(p)
            kp_quat.append(q)

        ts = [float(t) for t in timestamps]

        self._mark_policy_output_dirty("_ik_servo_keypoints")
        with self._motion_lock:
            # Seed pink from current commanded positions
            with self._state_lock:
                q = self._seed_q_from_arms()
            self._pink_cfg.update(self._clamp_q_to_urdf(q))

            for _t in self._pink_tasks.values():
                _t.set_target_from_configuration(self._pink_cfg)
            self._posture.set_target_from_configuration(self._pink_cfg)

            tasks = list(self._pink_tasks.values()) + [self._posture]
            moving_task = self._pink_tasks[side]

            t_start = time.time()
            while True:
                t_now = time.time() - t_start

                if t_now <= ts[0]:
                    pos_target = kp_pos[0]
                    quat_target = kp_quat[0]
                elif t_now >= ts[-1]:
                    pos_target = kp_pos[-1]
                    quat_target = kp_quat[-1]
                else:
                    seg = len(ts) - 2
                    for i in range(len(ts) - 1):
                        if t_now <= ts[i + 1]:
                            seg = i
                            break
                    seg_dur = ts[seg + 1] - ts[seg]
                    alpha = np.clip(
                        (t_now - ts[seg]) / seg_dur if seg_dur > 1e-9 else 1.0, 0.0, 1.0
                    )
                    pos_target = (1.0 - alpha) * kp_pos[seg] + alpha * kp_pos[seg + 1]
                    quat_target = _slerp(kp_quat[seg], kp_quat[seg + 1], alpha)

                moving_task.set_target(_make_se3(pos_target, quat_target))
                pos_err = np.linalg.norm(moving_task.compute_error(self._pink_cfg)[:3])
                moving_task.gain = min(
                    1.0, max_vel * self._control_period_s / (pos_err + 1e-6)
                )

                vel = pink.solve_ik(
                    self._pink_cfg, tasks, self._control_period_s, solver="quadprog"
                )
                self._pink_cfg.integrate_inplace(vel, self._control_period_s)
                q = self._pink_cfg.q
                with self._state_lock:
                    for _s in self._cmd_jp:
                        self._cmd_jp[_s][:] = self._extract_arm_from_q(q, _s)

                if t_now >= ts[-1]:
                    break
                time.sleep(self._control_period_s)

            return {"success": True, "reason": "ok"}

    def _ik_servo_traj(
        self,
        side: str,
        func,
        start_time: float,
        stop_time: float,
        max_vel: float = MOVE_EEF_MAX_VEL,
    ) -> dict:
        """Track a continuous EEF trajectory defined by a callable using pink IK.

        func(t) must return (pos, euler) where:
          pos   — array-like shape (3,): [x, y, z] in metres
          euler — array-like shape (3,): [roll, pitch, yaw] in radians (extrinsic XYZ / RPY)
        Evaluated at CONTROL_PERIOD_S intervals from start_time to stop_time.

        NOTE: not exposed over RPC — callables are not serialisable.
        """
        if side not in self._arm_jp:
            return {"success": False, "reason": f"invalid side: {side}"}

        self._mark_policy_output_dirty("_ik_servo_traj")
        with self._state_lock:
            q = self._seed_q_from_arms()
        self._pink_cfg.update(self._clamp_q_to_urdf(q))

        for _t in self._pink_tasks.values():
            _t.set_target_from_configuration(self._pink_cfg)
        self._posture.set_target_from_configuration(self._pink_cfg)

        tasks = list(self._pink_tasks.values()) + [self._posture]
        moving_task = self._pink_tasks[side]

        t = start_time
        while t <= stop_time:
            if self._safety.is_estopped():
                return {"success": False, "reason": "e-stop during trajectory"}

            pos, euler = func(t)
            pos = np.asarray(pos, dtype=np.float64).ravel()
            euler = np.asarray(euler, dtype=np.float64).ravel()
            R = pin.rpy.rpyToMatrix(euler[0], euler[1], euler[2])
            target_se3 = pin.SE3(R, pos)

            moving_task.set_target(target_se3)
            pos_err = np.linalg.norm(moving_task.compute_error(self._pink_cfg)[:3])
            moving_task.gain = min(
                1.0, max_vel * self._control_period_s / (pos_err + 1e-6)
            )

            vel = pink.solve_ik(
                self._pink_cfg, tasks, self._control_period_s, solver="quadprog"
            )
            self._pink_cfg.integrate_inplace(vel, self._control_period_s)
            q = self._pink_cfg.q
            with self._state_lock:
                for _s in self._cmd_jp:
                    self._cmd_jp[_s][:] = self._extract_arm_from_q(q, _s)

            t += self._control_period_s
            time.sleep(self._control_period_s)

        return {"success": True, "reason": "ok"}

    def move_joint_traj(
        self,
        side: str,
        func,
        start_time: float,
        stop_time: float,
    ) -> dict:
        """Execute a joint-space trajectory defined by a callable.

        func(t) must return a 6-element array of joint positions for the given side.
        Evaluated at CONTROL_PERIOD_S intervals from start_time to stop_time.

        NOTE: not exposed over RPC — callables are not serialisable.
        """
        if side not in ("left", "right"):
            return {"success": False, "reason": f"invalid side: {side}"}

        self._mark_policy_output_dirty("move_joint_traj")
        t = start_time
        while t <= stop_time:
            if self._safety.is_estopped():
                return {"success": False, "reason": "e-stop during trajectory"}

            jp = np.asarray(func(t), dtype=np.float64).ravel()[:6]
            with self._state_lock:
                if side == "left":
                    self._cmd_left_jp[:] = jp
                else:
                    self._cmd_right_jp[:] = jp

            t += self._control_period_s
            time.sleep(self._control_period_s)

        return {"success": True, "reason": "ok"}

    def move_joint_keypoints(
        self,
        side: str,
        timestamps: list[float],
        joint_positions: list,
        gripper_positions: list | None = None,
    ) -> dict:
        """Execute joint-space trajectory via timestamped waypoints.

        timestamps: monotonically increasing times in seconds, relative to call time.
        joint_positions: one 6-element array per timestamp.
        gripper_positions: optional one-element gripper waypoint per timestamp.
        Linearly interpolates between waypoints.
        """
        if side not in ("left", "right"):
            return {"success": False, "reason": f"invalid side: {side}"}
        if len(timestamps) != len(joint_positions) or len(timestamps) < 1:
            return {
                "success": False,
                "reason": "timestamps and joint_positions must be non-empty and equal length",
            }

        _dof = self._arm_jp[side].shape[0] if side in self._arm_jp else 6
        jps = [np.asarray(jp, dtype=np.float64).ravel()[:_dof] for jp in joint_positions]
        gps = None
        if gripper_positions is not None:
            if len(timestamps) != len(gripper_positions):
                return {
                    "success": False,
                    "reason": (
                        "timestamps and gripper_positions must be equal length "
                        "when gripper_positions is provided"
                    ),
                }
            gps = [
                np.asarray(
                    [np.clip(np.asarray(gp, dtype=np.float64).ravel()[0], 0.0, 1.0)]
                )
                for gp in gripper_positions
            ]
        ts = [float(t) for t in timestamps]

        self._mark_policy_output_dirty("move_joint_keypoints")
        with self._motion_lock:
            # For single-arm trajectories, explicitly latch the opposite arm's
            # current measured state into the command buffer before execution.
            # Otherwise that arm may drift toward a stale previously-commanded
            # posture (for example after the robot was moved by another control
            # loop such as yam_control_loop).
            with self._state_lock:
                if side == "left":
                    self._cmd_right_jp[:] = self.right_joint_pos.copy()
                    self._cmd_right_gp[:] = self.right_gripper_pos.copy()
                else:
                    self._cmd_left_jp[:] = self.left_joint_pos.copy()
                    self._cmd_left_gp[:] = self.left_gripper_pos.copy()
            t_start = time.time()
            while True:
                if self._safety.is_estopped():
                    return {"success": False, "reason": "e-stop during trajectory"}

                t_now = time.time() - t_start
                jp = self._sample_joint_keypoints(ts, jps, t_now)
                gp = (
                    self._sample_joint_keypoints(ts, gps, t_now)
                    if gps is not None
                    else None
                )

                with self._state_lock:
                    if side == "left":
                        self._cmd_left_jp[:] = jp
                        if gp is not None:
                            self._cmd_left_gp[:] = gp
                    else:
                        self._cmd_right_jp[:] = jp
                        if gp is not None:
                            self._cmd_right_gp[:] = gp

                if t_now >= ts[-1]:
                    break
                time.sleep(self._control_period_s)

            return {"success": True, "reason": "ok"}

    @staticmethod
    def _sample_joint_keypoints(
        timestamps: list[float],
        joint_positions: list[np.ndarray],
        t_now: float,
    ) -> np.ndarray:
        if t_now <= timestamps[0]:
            return joint_positions[0]
        if t_now >= timestamps[-1]:
            return joint_positions[-1]

        seg = len(timestamps) - 2
        for i in range(len(timestamps) - 1):
            if t_now <= timestamps[i + 1]:
                seg = i
                break
        seg_dur = timestamps[seg + 1] - timestamps[seg]
        alpha = np.clip(
            (t_now - timestamps[seg]) / seg_dur if seg_dur > 1e-9 else 1.0,
            0.0,
            1.0,
        )
        return (1.0 - alpha) * joint_positions[seg] + alpha * joint_positions[seg + 1]

    def move_bimanual_joint_keypoints(
        self,
        timestamps: list[float],
        left_joint_positions: list,
        right_joint_positions: list,
        left_gripper_positions: list | None = None,
        right_gripper_positions: list | None = None,
    ) -> dict:
        """Execute a synchronized bimanual joint-space trajectory.

        timestamps: monotonically increasing times in seconds, relative to call time.
        left_joint_positions / right_joint_positions: one 6-element array per timestamp.
        left_gripper_positions / right_gripper_positions: optional one-element gripper
        waypoint per timestamp. Both arms are interpolated against the same clock
        and commanded together.
        """
        if (
            len(timestamps) != len(left_joint_positions)
            or len(timestamps) != len(right_joint_positions)
            or len(timestamps) < 1
        ):
            return {
                "success": False,
                "reason": (
                    "timestamps, left_joint_positions, and right_joint_positions "
                    "must be non-empty and equal length"
                ),
            }

        _left_dof = self._arm_jp["left"].shape[0] if "left" in self._arm_jp else 6
        _right_dof = self._arm_jp["right"].shape[0] if "right" in self._arm_jp else 6
        left_jps = [
            np.asarray(jp, dtype=np.float64).ravel()[:_left_dof] for jp in left_joint_positions
        ]
        right_jps = [
            np.asarray(jp, dtype=np.float64).ravel()[:_right_dof] for jp in right_joint_positions
        ]
        left_gps = None
        if left_gripper_positions is not None:
            if len(timestamps) != len(left_gripper_positions):
                return {
                    "success": False,
                    "reason": (
                        "timestamps and left_gripper_positions must be equal length "
                        "when left_gripper_positions is provided"
                    ),
                }
            left_gps = [
                np.asarray(
                    [np.clip(np.asarray(gp, dtype=np.float64).ravel()[0], 0.0, 1.0)]
                )
                for gp in left_gripper_positions
            ]
        right_gps = None
        if right_gripper_positions is not None:
            if len(timestamps) != len(right_gripper_positions):
                return {
                    "success": False,
                    "reason": (
                        "timestamps and right_gripper_positions must be equal length "
                        "when right_gripper_positions is provided"
                    ),
                }
            right_gps = [
                np.asarray(
                    [np.clip(np.asarray(gp, dtype=np.float64).ravel()[0], 0.0, 1.0)]
                )
                for gp in right_gripper_positions
            ]
        ts = [float(t) for t in timestamps]

        self._mark_policy_output_dirty("move_bimanual_joint_keypoints")
        with self._motion_lock:
            t_start = time.time()
            while True:
                if self._safety.is_estopped():
                    return {"success": False, "reason": "e-stop during trajectory"}

                t_now = time.time() - t_start
                left_jp = self._sample_joint_keypoints(ts, left_jps, t_now)
                right_jp = self._sample_joint_keypoints(ts, right_jps, t_now)
                left_gp = (
                    self._sample_joint_keypoints(ts, left_gps, t_now)
                    if left_gps is not None
                    else None
                )
                right_gp = (
                    self._sample_joint_keypoints(ts, right_gps, t_now)
                    if right_gps is not None
                    else None
                )

                with self._state_lock:
                    self._cmd_left_jp[:] = left_jp
                    self._cmd_right_jp[:] = right_jp
                    if left_gp is not None:
                        self._cmd_left_gp[:] = left_gp
                    if right_gp is not None:
                        self._cmd_right_gp[:] = right_gp

                if t_now >= ts[-1]:
                    break
                time.sleep(self._control_period_s)

            return {"success": True, "reason": "ok"}

    def set_gripper(
        self,
        side: str,
        value: float,
        timeout: float = GRIPPER_SETTLE_TIMEOUT_S,
        vel_limit: float | None = None,
        torque_limit: float | None = None,
    ) -> bool:
        """Set gripper target. value=1.0 open, 0.0 closed.

        Works for all backends — writes to _cmd_gp, control loop delivers it.
        """

        if side not in self._arm_jp:
            return False
        value = float(np.clip(value, 0.0, 1.0))
        vel_limit = None if vel_limit is None else float(vel_limit)
        torque_limit = None if torque_limit is None else float(torque_limit)
        self._mark_policy_output_dirty("set_gripper")
        with self._state_lock:
            if side == "left":
                self._cmd_left_gp[0] = value
                self._cmd_left_gv = vel_limit
                self._cmd_left_gt = torque_limit
            elif side == "right":
                self._cmd_right_gp[0] = value
                self._cmd_right_gv = vel_limit
                self._cmd_right_gt = torque_limit
        start = time.time()
        stall_since: float | None = None
        last_pos: float | None = None
        while time.time() - start < timeout:
            with self._state_lock:
                actual = float(
                    self.left_gripper_pos[0]
                    if side == "left"
                    else self.right_gripper_pos[0]
                )

            # Stall detection (only when torque_limit is set)
            stall_held = False
            if torque_limit is not None:
                if (
                    last_pos is not None
                    and abs(actual - last_pos) < GRIPPER_STALL_THRESH
                ):
                    if stall_since is None:
                        stall_since = time.time()
                    stall_held = (
                        time.time() - stall_since
                    ) >= GRIPPER_TORQUE_LIMIT_HOLD_S
                else:
                    stall_since = None  # position moving, reset timer

            last_pos = actual

            # Exit: position reached OR torque-limit stall held long enough
            if abs(actual - value) < GRIPPER_SETTLE_THRESH or stall_held:
                break

            time.sleep(GRIPPER_POLL_S)
        return True

    def go_home(self, max_joint_vel: float = GO_HOME_MAX_JOINT_VEL) -> bool:
        """Return to home configuration.

        For envs with native go_home (RoboCasa), delegates directly.
        For EefControlProtocol envs uses _ik_servo to the home EE pose
        captured at startup, since OSC controllers cannot accept
        direct joint targets.
        TODO: deprecate this EE path once joint-space interpolation is
        implemented for RoboCasa (requires a joint-level controller mode).
        For joint-level envs (YAM, hardware) uses joint-space interpolation.
        """
        from enpire.env.forge.cap.env.base import EefControlProtocol


        self._mark_policy_output_dirty("go_home")

        # EefControlProtocol path: use _ik_servo to cached home EE poses
        if (
            self._sim_backend is not None
            and isinstance(self._sim_backend, EefControlProtocol)
            and self._home_ee_poses
        ):
            for side, (pos, quat) in self._home_ee_poses.items():
                gripper = 1.0  # open
                self._ik_servo(side, pos, quat, gripper=gripper,
                              max_duration_sec=12.0, tol=0.03)
            return True

        # Joint-space interpolation path (YAM, hardware)
        with self._motion_lock:
            # Build per-arm targets from profile or legacy constants
            targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            starts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for side in self._arm_jp:
                if self._robot_profile and side in self._robot_profile.arms:
                    arm = self._robot_profile.arms[side]
                    targets[side] = (
                        arm.home_joint_pos.copy(),
                        arm.home_gripper_pos.copy(),
                    )
                else:
                    targets[side] = (
                        HOME_JOINT_STATE[f"{side}_joint_pos"].copy(),
                        HOME_JOINT_STATE[f"{side}_gripper_pos"].copy(),
                    )

            with self._state_lock:
                for side in self._arm_jp:
                    starts[side] = (
                        self._cmd_jp[side].copy(),
                        self._cmd_gp[side].copy(),
                    )
                    self._cmd_gv[side] = None
                    self._cmd_gt[side] = None

            # Compute duration from largest joint displacement
            max_disp = 0.0
            for side in self._arm_jp:
                disp = np.max(np.abs(targets[side][0] - starts[side][0]))
                max_disp = max(max_disp, disp)
            n_steps = max(1, int(max_disp / max_joint_vel / self._control_period_s))

            for i in range(n_steps):
                t = (i + 1) / n_steps
                with self._state_lock:
                    for side in self._arm_jp:
                        self._cmd_jp[side][:] = starts[side][0] + t * (
                            targets[side][0] - starts[side][0]
                        )
                        self._cmd_gp[side][:] = starts[side][1] + t * (
                            targets[side][1] - starts[side][1]
                        )
                time.sleep(self._control_period_s)

            return True

    def list_cameras(self) -> dict:
        """Return available camera names."""
        return {"cameras": list(self._cameras.keys())}

    def get_camera_image(self, camera: str) -> np.ndarray:
        if camera not in self._cameras:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return self._cameras[camera].get_rgb()

    def get_camera_depth(self, camera: str) -> np.ndarray:
        if camera not in self._cameras:
            return np.zeros((1, 1), dtype=np.float32)
        depth = self._cameras[camera].get_depth()
        return depth if depth is not None else np.zeros((1, 1), dtype=np.float32)

    def get_camera_intrinsics(self, camera: str) -> list[float]:
        intrinsics = self.get_camera_intrinsics_full(camera)
        try:
            return [
                intrinsics["fx"],
                intrinsics["fy"],
                intrinsics["cx"],
                intrinsics["cy"],
            ]
        except (KeyError, TypeError, ValueError):
            return [0.0, 0.0, 0.0, 0.0]

    def get_camera_intrinsics_full(self, camera: str) -> dict:
        from enpire.env.forge.robot.camera_calibration_intrinsics import (
            load_yam_camera_intrinsics,
            require_yam_camera_intrinsics,
            yam_camera_intrinsics_required,
        )

        if yam_camera_intrinsics_required(camera):
            return require_yam_camera_intrinsics(camera)
        calibrated = load_yam_camera_intrinsics(camera)
        if calibrated is not None:
            return calibrated
        if camera not in self._cameras:
            return {"fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0}
        intrinsics = self._cameras[camera].get_intrinsics_full()
        if intrinsics is None:
            return {"fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0}
        return intrinsics

    def get_camera_extrinsics(self, camera: str) -> dict:
        # In sim mode, use MuJoCo's camera pose directly (matches rendered images)
        if self._sim_mode and self._sim_backend is not None:
            extr = self._sim_backend.get_camera_extrinsics(camera)
            # All sim envs return Pinocchio convention — always need optical flip to OpenCV
            extr["needs_optical_flip"] = True
            return extr

        from enpire.env.forge.robot.models.station.paths import get_top_camera_frame, needs_optical_flip

        cam_frame_map = {
            "top": os.environ.get("CAP_TOP_CAMERA_FRAME", get_top_camera_frame()),
            "left": os.environ.get("CAP_LEFT_CAMERA_FRAME", "left_camera_d405"),
            "right": os.environ.get("CAP_RIGHT_CAMERA_FRAME", "right_camera_d405"),
        }
        frame_name = cam_frame_map.get(camera)
        if frame_name is None:
            return {
                "position": [0.0, 0.0, 0.0],
                "rotation": np.eye(3).tolist(),
                "needs_optical_flip": True,
            }

        with self._state_lock:
            ljp = self.left_joint_pos.copy()
            rjp = self.right_joint_pos.copy()
        T = self._frame_pose(frame_name, ljp, rjp)
        return {
            "position": T.translation.tolist(),
            "rotation": T.rotation.tolist(),
            "needs_optical_flip": needs_optical_flip(camera),
        }

    def _create_policy_runtime(
        self,
        server_addr: str,
        embodiment_tag: str,
        resolution: int,
        replan_horizon: int,
        require_prev_observation: bool = False,
    ) -> Any:
        from enpire.env.forge.experimental.embodiment_tags import EmbodimentTag
        from enpire.env.forge.experimental.get_action_policy import PolicyAdapters
        from enpire.env.forge.experimental.key_remapping_utils import map_action, map_observation
        from enpire.env.forge.experimental.robot_interface import RobotInterface
        from enpire.env.forge.experimental.sync_chunking_policy import SyncChunkingPolicy

        embodiment = EmbodimentTag(embodiment_tag)
        adapters = PolicyAdapters(
            map_observation=lambda obs: map_observation(obs, embodiment, resolution),
            map_action=lambda act: map_action(act, embodiment),
        )
        robot_interface = RobotInterface(
            checkpoint_dir=".",
            server_address=server_addr,
            adapters=adapters,
            embodiment_tag=embodiment,
        )
        return SyncChunkingPolicy(
            policy=robot_interface,
            action_exec_horizon=max(1, int(replan_horizon)),
            require_prev_observation=require_prev_observation,
        )

    def _clear_policy_runtime_local_state(self, policy: Any) -> None:
        clear_local_state = getattr(policy, "clear_local_state", None)
        if callable(clear_local_state):
            clear_local_state()

    def _reset_policy_output_runtime(
        self,
        session: _PolicyOutputSession,
        phase: str,
    ) -> None:
        self._clear_policy_runtime_local_state(session.policy)
        reset_behavior = str(session.reset_behavior).strip().lower()
        if reset_behavior in {"", "noop", "none", "local_only"}:
            logger.info(
                "[CapServer] policy-output reset skipped for %s during %s (%s)",
                session.model,
                phase,
                session.reset_behavior,
            )
            return
        if reset_behavior == "server":
            session.policy.reset()
            logger.info(
                "[CapServer] policy-output runtime reset for %s during %s",
                session.model,
                phase,
            )
            return
        raise ValueError(
            f"unknown reset_behavior for {session.model!r}: {session.reset_behavior!r}"
        )

    def _mark_policy_output_dirty(self, reason: str) -> None:
        with self._policy_output_session_lock:
            session = self._policy_output_session
            if session is None:
                return
            self._clear_policy_runtime_local_state(session.policy)
            session.dirty = True
            session.dirty_reason = reason

    def _policy_output_info(
        self,
        session: _PolicyOutputSession,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "model": session.model,
            "policy_server": session.policy_server,
            "replan_horizon": session.replan_horizon,
            "task_description": session.task_description,
            "reset_behavior": session.reset_behavior,
            "dirty": bool(session.dirty),
        }
        if session.dirty_reason:
            info["dirty_reason"] = session.dirty_reason
        if extra:
            info.update(extra)
        return info

    def _resolve_policy_action_targets(
        self,
        action: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if "left_joint_pos" in action and "right_joint_pos" in action:
            left_jp = np.asarray(action["left_joint_pos"]).ravel()[:6]
            right_jp = np.asarray(action["right_joint_pos"]).ravel()[:6]
        elif "left_ee_pos" in action and "right_ee_pos" in action:
            with self._state_lock:
                q_seed = pin.neutral(self._pink_model).copy()
                q_seed[:6] = self._cmd_left_jp.copy()
                q_seed[8:14] = self._cmd_right_jp.copy()
            left_jp, right_jp = self._inverse_kinematics(
                np.asarray(action["left_ee_pos"]).ravel()[:3],
                np.asarray(action["left_ee_quat_xyzw"]).ravel()[:4],
                np.asarray(action["right_ee_pos"]).ravel()[:3],
                np.asarray(action["right_ee_quat_xyzw"]).ravel()[:4],
                q_seed,
            )
        else:
            raise ValueError(f"unknown action format: {list(action.keys())}")

        with self._state_lock:
            default_left_grip = self.left_gripper_pos.copy()
            default_right_grip = self.right_gripper_pos.copy()

        left_grip = np.asarray(
            action.get("left_gripper_pos", default_left_grip)
        ).ravel()[:1]
        right_grip = np.asarray(
            action.get("right_gripper_pos", default_right_grip)
        ).ravel()[:1]
        return left_jp, left_grip, right_jp, right_grip

    def _run_policy_steps(
        self,
        policy: Any,
        task_description: str,
        max_steps: int,
    ) -> dict:
        steps = 0
        last_step_time = time.time()
        action_horizon = max(1, int(getattr(policy, "action_exec_horizon", 1)))
        last_info: dict[str, Any] = {}
        obs = self._build_skill_obs(task_description)

        while steps < max_steps:
            if self._safety.is_estopped():
                return {
                    "success": False,
                    "steps_executed": steps,
                    "reason": "e-stop during skill",
                    "info": last_info,
                }

            action, info = policy.get_action(obs)
            last_info = dict(info or {})

            remaining = last_info.get("remaining_num_action_in_chunk", -1)
            if "action_chunk" in last_info and remaining == action_horizon - 1:
                self._latest_action_chunk = last_info["action_chunk"]

            left_jp, left_grip, right_jp, right_grip = (
                self._resolve_policy_action_targets(action)
            )

            with self._state_lock:
                self._cmd_left_jp[:] = left_jp
                self._cmd_left_gp[:] = left_grip
                self._cmd_right_jp[:] = right_jp
                self._cmd_right_gp[:] = right_grip

            sleep_end = last_step_time + POLICY_PERIOD_S
            while time.time() < sleep_end:
                time.sleep(0.0001)
            last_step_time = time.time()

            obs = self._build_skill_obs(task_description)
            steps += 1

        return {
            "success": True,
            "steps_executed": steps,
            "reason": "ok",
            "info": last_info,
        }

    def start_policy_output(
        self,
        model: str,
        replan_horizon: int = 30,
        task_description: str = "",
        policy_server: str = "",
    ) -> dict:
        model_name = str(model).strip()
        task_description = _decode_portal_str(task_description)
        policy_server = _decode_portal_str(policy_server)
        model_cfg = POLICY_MODEL_CONFIGS.get(model_name)
        if model_cfg is None:
            return {
                "success": False,
                "steps_executed": 0,
                "reason": (
                    f"unknown policy model: {model_name!r}. "
                    f"Supported: {', '.join(sorted(POLICY_MODEL_CONFIGS))}"
                ),
                "info": {},
            }

        rollout_horizon = max(1, int(replan_horizon))
        server_addr = str(policy_server).strip() or str(model_cfg["server"])
        task = str(task_description).strip() or str(
            model_cfg.get("default_task_description", model_name)
        )
        embodiment_tag = str(model_cfg.get("embodiment_tag", "xdof"))
        resolution = int(model_cfg.get("resolution", 480))
        reset_behavior = str(model_cfg.get("reset_behavior", "server"))
        require_prev_observation = bool(
            model_cfg.get("require_prev_observation", False)
        )

        new_session = None
        try:
            policy = self._create_policy_runtime(
                server_addr=server_addr,
                embodiment_tag=embodiment_tag,
                resolution=resolution,
                replan_horizon=rollout_horizon,
                require_prev_observation=require_prev_observation,
            )
            new_session = _PolicyOutputSession(
                model=model_name,
                policy_server=server_addr,
                task_description=task,
                replan_horizon=rollout_horizon,
                reset_behavior=reset_behavior,
                embodiment_tag=embodiment_tag,
                resolution=resolution,
                policy=policy,
            )
            self._reset_policy_output_runtime(new_session, phase="start")
        except Exception as e:
            logger.exception("start_policy_output failed")
            if new_session is not None:
                try:
                    self._reset_policy_output_runtime(
                        new_session, phase="cleanup_on_error"
                    )
                except Exception:
                    logger.exception("cleanup after start_policy_output failed")
            return {
                "success": False,
                "steps_executed": 0,
                "reason": str(e),
                "info": {},
            }

        with self._policy_output_session_lock:
            old_session = self._policy_output_session
            self._policy_output_session = new_session
        self._latest_action_chunk = None

        if old_session is not None:
            try:
                self._reset_policy_output_runtime(old_session, phase="replace")
            except Exception:
                logger.exception("policy-output replace cleanup failed")

        return {
            "success": True,
            "steps_executed": 0,
            "reason": "ok",
            "info": self._policy_output_info(new_session, {"state": "started"}),
        }

    def step_policy_output(self, max_steps: int | None = 0) -> dict:
        with self._policy_output_session_lock:
            session = self._policy_output_session
        if session is None:
            return {
                "success": False,
                "steps_executed": 0,
                "reason": "no active policy-output session",
                "info": {},
            }

        rollout_steps = (
            max(1, int(max_steps))
            if max_steps not in (None, "") and int(max_steps) > 0
            else session.replan_horizon
        )
        if not self._skill_lock.acquire(blocking=False):
            return {
                "success": False,
                "steps_executed": 0,
                "reason": "another skill is running",
                "info": self._policy_output_info(session),
            }

        self._latest_action_chunk = None

        try:
            with self._motion_lock:
                # Re-validate session identity — a concurrent start_policy_output may
                # have replaced self._policy_output_session between our read and now.
                with self._policy_output_session_lock:
                    if self._policy_output_session is not session:
                        raise RuntimeError(
                            "policy-output session was replaced during step"
                        )
                if session.dirty:
                    self._reset_policy_output_runtime(
                        session,
                        phase=f"step_after_{session.dirty_reason or 'external_change'}",
                    )
                    with self._policy_output_session_lock:
                        if self._policy_output_session is session:
                            session.dirty = False
                            session.dirty_reason = ""

                result = self._run_policy_steps(
                    session.policy,
                    task_description=session.task_description,
                    max_steps=rollout_steps,
                )
            result["info"] = self._policy_output_info(
                session,
                {
                    "state": "stepped",
                    "max_steps": rollout_steps,
                    **dict(result.get("info") or {}),
                },
            )
            return result
        except Exception as e:
            logger.exception("step_policy_output failed")
            return {
                "success": False,
                "steps_executed": 0,
                "reason": str(e),
                "info": self._policy_output_info(
                    session,
                    {"state": "step_failed", "max_steps": rollout_steps},
                ),
            }
        finally:
            # Keep _latest_action_chunk visible so the UI poll can display it
            # between consecutive step calls.  stop_policy_output() clears it.
            self._skill_lock.release()

    def stop_policy_output(self) -> dict:
        with self._policy_output_session_lock:
            session = self._policy_output_session
            self._policy_output_session = None
        self._latest_action_chunk = None

        if session is None:
            return {
                "success": True,
                "steps_executed": 0,
                "reason": "no active policy-output session",
                "info": {"state": "idle"},
            }

        try:
            self._reset_policy_output_runtime(session, phase="stop")
            session.dirty = False
            session.dirty_reason = ""
            return {
                "success": True,
                "steps_executed": 0,
                "reason": "ok",
                "info": self._policy_output_info(session, {"state": "stopped"}),
            }
        except Exception as e:
            logger.exception("stop_policy_output failed")
            return {
                "success": False,
                "steps_executed": 0,
                "reason": str(e),
                "info": self._policy_output_info(session, {"state": "stop_failed"}),
            }

    def use_policy_output(
        self,
        model: str,
        replan_horizon: int = 30,
        max_steps: int | None = 0,
        task_description: str = "",
        policy_server: str = "",
    ) -> dict:
        """Convenience wrapper around start_policy_output + step + stop."""
        task_description = _decode_portal_str(task_description)
        policy_server = _decode_portal_str(policy_server)
        start_result = self.start_policy_output(
            model=model,
            replan_horizon=replan_horizon,
            task_description=task_description,
            policy_server=policy_server,
        )
        if not start_result.get("success", False):
            return start_result

        step_exc: Exception | None = None
        step_result: dict | None = None
        try:
            step_result = self.step_policy_output(max_steps=max_steps)
        except Exception as e:
            logger.exception("step_policy_output raised")
            step_exc = e
        finally:
            stop_result = self.stop_policy_output()

        if step_exc is not None:
            return {
                "success": False,
                "steps_executed": 0,
                "reason": f"step failed: {step_exc}",
                "info": {"stop_result": stop_result},
            }
        # Merge stop result into step result
        step_info = dict(step_result.get("info") or {})
        step_info["stop_result"] = stop_result
        step_result["info"] = step_info
        if step_result.get("success", False) and not stop_result.get("success", False):
            step_result["success"] = False
            step_result["reason"] = (
                f"policy rollout succeeded but stop_policy_output failed: "
                f"{stop_result.get('reason', 'unknown error')}"
            )
        return step_result

    def execute_skill(
        self,
        skill_name: str,
        policy_server: str = "none",
        params: dict | None = None,
    ) -> dict:
        """Execute a flow matching policy. Sets joint targets under _state_lock."""
        if not self._skill_lock.acquire(blocking=False):
            return {
                "success": False,
                "steps_executed": 0,
                "reason": "another skill is running",
            }

        self._latest_action_chunk = None

        try:
            server_addr = (
                f"localhost:{POLICY_SERVER_PORT}"
                if policy_server == "none"
                else policy_server
            )
            params = params or {}
            self._mark_policy_output_dirty("execute_skill")

            task_description = params.get("task_description", skill_name)
            resolution = int(params.get("resolution", 480))
            action_horizon = int(params.get("action_horizon", 8))
            require_prev_observation = bool(
                params.get("require_prev_observation", False)
            )
            policy = self._create_policy_runtime(
                server_addr=server_addr,
                embodiment_tag=str(params.get("embodiment_tag", "xdof")),
                resolution=resolution,
                replan_horizon=action_horizon,
                require_prev_observation=require_prev_observation,
            )
            policy.reset()

            max_steps = int(params.get("max_steps", 300))
            with self._motion_lock:
                return self._run_policy_steps(
                    policy,
                    task_description=task_description,
                    max_steps=max_steps,
                )

        except Exception as e:
            logger.exception("execute_skill failed")
            return {"success": False, "steps_executed": 0, "reason": str(e)}
        finally:
            self._latest_action_chunk = None
            self._skill_lock.release()

    def learn_skill(
        self,
        skill_name: str,
        params: dict | None = None,
    ) -> dict:
        """Run an RL policy with human Fello takeover and data recording.

        Connects to an external rl_policy_server via Portal RPC, sends
        observations and receives actions at POLICY_FREQ_HZ.  Human takeover
        via Fello is supported (press-anchored incremental control).  Each
        transition (obs, action_executed, action_source, reward, done) is sent
        back to the rl_policy_server so it can update its replay buffer / policy.
        """
        if not self._skill_lock.acquire(blocking=False):
            return {
                "success": False,
                "steps_executed": 0,
                "reason": "another skill is running",
            }

        try:
            params = params or {}
            self._mark_policy_output_dirty("learn_skill")
            task_description = params.get("task_description", skill_name)
            max_steps = params.get("max_steps", RL_EPISODE_MAX_STEPS)
            rl_host = params.get("rl_host", self._rl_host)
            rl_port = params.get("rl_port", RL_POLICY_PORT)
            control_mode = params.get("control_mode", "both")

            # --- Connect to rl_policy_server ---
            rl_client = portal.Client(f"{rl_host}:{rl_port}")
            logger.info(
                "[learn_skill] Connected to rl_policy_server at %s:%d", rl_host, rl_port
            )

            # --- Reward: local fn (set via setup_reward) takes priority over external server ---
            _local_fn = self._local_reward_fn
            if _local_fn is not None:
                logger.info(
                    "[learn_skill] Using local reward fn: mode=%r",
                    self._local_reward_mode,
                )
                _reward_client = None
            else:
                from enpire.env.forge.cap.reward.reward_client import RewardClient

                _reward_client: RewardClient | None = None
                try:
                    _reward_client = RewardClient()
                    logger.info(
                        "[learn_skill] Reward server connected on port %d",
                        REWARD_SERVER_PORT,
                    )
                except Exception:
                    logger.info(
                        "[learn_skill] No reward server available — rewards will be 0.0"
                    )

            def _get_step_reward(obs: dict) -> float:
                if _local_fn is not None:
                    try:
                        return float(_local_fn(obs))
                    except Exception:
                        return 0.0
                if _reward_client is None:
                    return 0.0
                try:
                    return _reward_client.get_reward(obs)
                except Exception:
                    return 0.0

            # --- Build initial observation for RL ---
            obs = self._build_skill_obs(task_description)
            rl_obs = self._build_rl_obs(obs, control_mode=control_mode)

            # --- Reset: send initial obs, get first action ---
            reset_result = rl_client.reset(rl_obs).result()
            rl_action = reset_result["action"]
            action_type = reset_result.get("action_type", "joint_angle")

            steps = 0
            cumulative_reward = 0.0
            last_step_time = time.time()
            episode = getattr(self, "_rl_episode_counter", 0)
            self._rl_episode_counter = episode + 1

            # HILPolicyWrapper-style takeover state
            took_control_left = False
            took_control_right = False
            prev_left_takeover = False
            prev_right_takeover = False
            last_left_jp: np.ndarray | None = None
            last_left_grip: np.ndarray | None = None
            last_right_jp: np.ndarray | None = None
            last_right_grip: np.ndarray | None = None
            left_yam_anchor_jp: np.ndarray | None = None
            left_yam_anchor_gp: np.ndarray | None = None
            right_yam_anchor_jp: np.ndarray | None = None
            right_yam_anchor_gp: np.ndarray | None = None
            left_fello_anchor_jp: np.ndarray | None = None
            left_fello_anchor_gp: np.ndarray | None = None
            right_fello_anchor_jp: np.ndarray | None = None
            right_fello_anchor_gp: np.ndarray | None = None
            left_grip_settle_ticks = 0
            right_grip_settle_ticks = 0
            left_grip_unlocked = False
            right_grip_unlocked = False
            _hil_substep = 0  # counts consecutive human-takeover steps

            from rich.progress import (
                Progress,
                BarColumn,
                TextColumn,
                TimeRemainingColumn,
                MofNCompleteColumn,
            )

            _progress = Progress(
                TextColumn("[learn_skill] ep={task.fields[ep]}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                TextColumn(
                    "r={task.fields[reward]:.2f} cumR={task.fields[cumR]:.2f} src={task.fields[src]}"
                    " rl={task.fields[rl_ms]:.0f}ms local={task.fields[local_ms]:.0f}ms"
                    " (obs={task.fields[obs_ms]:.0f} rew={task.fields[rew_ms]:.0f} fello={task.fields[fello_ms]:.0f} act={task.fields[act_ms]:.0f})"
                ),
            )
            _progress.start()
            _task_id = _progress.add_task(
                "learn_skill",
                total=max_steps,
                ep=episode,
                reward=0.0,
                cumR=0.0,
                src="rl",
                rl_ms=0.0,
                local_ms=0.0,
                obs_ms=0.0,
                rew_ms=0.0,
                fello_ms=0.0,
                act_ms=0.0,
            )
            self._learn_skill_status = {
                "active": True,
                "episode": episode,
                "step": 0,
                "max_steps": max_steps,
                "reward": 0.0,
                "cumulative_reward": 0.0,
                "action_source": "rl",
            }

            while steps < max_steps:
                if self._safety.is_estopped():
                    _progress.stop()
                    return {
                        "success": False,
                        "steps_executed": steps,
                        "reason": "e-stop during skill",
                    }

                emit(
                    "cap_server",
                    "step_start",
                    ep=episode,
                    step=steps,
                    max_steps=max_steps,
                    cumR=cumulative_reward,
                )

                # --- Read Fello state from cache (written by _fello_loop) ---
                _t_local_start = time.time()
                _t_fello_start = time.time()
                left_takeover = right_takeover = False
                lq = rq = None
                robot_ljp = np.zeros(6, dtype=np.float64)
                robot_lgp = np.zeros(1, dtype=np.float64)
                robot_rjp = np.zeros(6, dtype=np.float64)
                robot_rgp = np.zeros(1, dtype=np.float64)
                if USE_FELLO:
                    with self._state_lock:
                        left_takeover = self._fello_takeover_left
                        right_takeover = self._fello_takeover_right
                        lq = self._fello_left_qpos.copy()
                        rq = self._fello_right_qpos.copy()
                        robot_ljp = self.left_joint_pos.copy()
                        robot_lgp = self.left_gripper_pos.copy()
                        robot_rjp = self.right_joint_pos.copy()
                        robot_rgp = self.right_gripper_pos.copy()

                _fello_ms = (time.time() - _t_fello_start) * 1000
                emit(
                    "cap_server",
                    "fello_read",
                    ep=episode,
                    step=steps,
                    takeover_left=left_takeover,
                    takeover_right=right_takeover,
                )

                # --- Resolve RL action into joint targets ---
                _t_act_start = time.time()
                left_jp, left_grip, right_jp, right_grip = (
                    self._resolve_action_to_joints(rl_action, action_type)
                )
                rl_grip_L, rl_grip_R = left_grip.copy(), right_grip.copy()

                # --- Determine action source ---
                action_source = "rl"

                # --- Apply press-anchored incremental Fello takeover ---
                if left_takeover and lq is not None:
                    left_fello_now_jp, left_fello_now_grip = (
                        self._map_fello_qpos_to_yam(lq)
                    )
                    if (
                        not prev_left_takeover
                        or left_yam_anchor_jp is None
                        or left_yam_anchor_gp is None
                        or left_fello_anchor_jp is None
                        or left_fello_anchor_gp is None
                    ):
                        left_yam_anchor_jp = robot_ljp.copy()
                        left_yam_anchor_gp = self._cmd_left_gp.copy()
                        left_fello_anchor_jp = left_fello_now_jp.copy()
                        left_fello_anchor_gp = left_fello_now_grip.copy()
                        left_grip_settle_ticks = 0
                        left_grip_unlocked = False
                    left_grip_settle_ticks += 1
                    left_grip_settled = (
                        left_grip_settle_ticks > self.FELLO_GRIP_SETTLE_TICKS
                    )
                    # Re-anchor gripper once settled so delta starts from rest
                    if left_grip_settle_ticks == self.FELLO_GRIP_SETTLE_TICKS + 1:
                        left_fello_anchor_gp = left_fello_now_grip.copy()
                    left_jp, _fello_grip = self._compute_delta_takeover_cmd(
                        left_yam_anchor_jp,
                        left_yam_anchor_gp,
                        left_fello_anchor_jp,
                        left_fello_anchor_gp,
                        left_fello_now_jp,
                        left_fello_now_grip,
                        grip_settled=left_grip_settled,
                    )
                    # Hold pre-takeover gripper unless human intentionally moved Fello gripper
                    _grip_delta = abs(
                        float(left_fello_now_grip[0]) - float(left_fello_anchor_gp[0])
                    )
                    if left_grip_settled and (
                        left_grip_unlocked
                        or _grip_delta > self.FELLO_GRIP_TAKEOVER_DEADBAND
                    ):
                        left_grip_unlocked = True
                        left_grip = _fello_grip
                    else:
                        left_grip = left_yam_anchor_gp.copy()
                    took_control_left = True
                    last_left_jp = left_jp.copy()
                    last_left_grip = left_grip.copy()
                    action_source = "human"
                elif took_control_left:
                    left_jp = last_left_jp
                    left_grip = last_left_grip
                    action_source = "human"

                if right_takeover and rq is not None:
                    right_fello_now_jp, right_fello_now_grip = (
                        self._map_fello_qpos_to_yam(rq)
                    )
                    if (
                        not prev_right_takeover
                        or right_yam_anchor_jp is None
                        or right_yam_anchor_gp is None
                        or right_fello_anchor_jp is None
                        or right_fello_anchor_gp is None
                    ):
                        right_yam_anchor_jp = robot_rjp.copy()
                        right_yam_anchor_gp = self._cmd_right_gp.copy()
                        right_fello_anchor_jp = right_fello_now_jp.copy()
                        right_fello_anchor_gp = right_fello_now_grip.copy()
                        right_grip_settle_ticks = 0
                        right_grip_unlocked = False
                    right_grip_settle_ticks += 1
                    right_grip_settled = (
                        right_grip_settle_ticks > self.FELLO_GRIP_SETTLE_TICKS
                    )
                    # Re-anchor gripper once settled so delta starts from rest
                    if right_grip_settle_ticks == self.FELLO_GRIP_SETTLE_TICKS + 1:
                        right_fello_anchor_gp = right_fello_now_grip.copy()
                    right_jp, _fello_grip = self._compute_delta_takeover_cmd(
                        right_yam_anchor_jp,
                        right_yam_anchor_gp,
                        right_fello_anchor_jp,
                        right_fello_anchor_gp,
                        right_fello_now_jp,
                        right_fello_now_grip,
                        grip_settled=right_grip_settled,
                    )
                    # Hold pre-takeover gripper unless human intentionally moved Fello gripper
                    _grip_delta = abs(
                        float(right_fello_now_grip[0]) - float(right_fello_anchor_gp[0])
                    )
                    if right_grip_settled and (
                        right_grip_unlocked
                        or _grip_delta > self.FELLO_GRIP_TAKEOVER_DEADBAND
                    ):
                        right_grip_unlocked = True
                        right_grip = _fello_grip
                    else:
                        right_grip = right_yam_anchor_gp.copy()
                    took_control_right = True
                    last_right_jp = right_jp.copy()
                    last_right_grip = right_grip.copy()
                    action_source = "human"
                elif took_control_right:
                    right_jp = last_right_jp
                    right_grip = last_right_grip
                    action_source = "human"

                if prev_left_takeover and not left_takeover:
                    left_yam_anchor_jp = None
                    left_yam_anchor_gp = None
                    left_fello_anchor_jp = None
                    left_fello_anchor_gp = None
                if prev_right_takeover and not right_takeover:
                    right_yam_anchor_jp = None
                    right_yam_anchor_gp = None
                    right_fello_anchor_jp = None
                    right_fello_anchor_gp = None
                prev_left_takeover = bool(left_takeover)
                prev_right_takeover = bool(right_takeover)

                # --- Detect full-release transition ---
                releasing = (not left_takeover and not right_takeover) and (
                    took_control_left or took_control_right
                )
                if releasing:
                    took_control_left = took_control_right = False

                # --- Safety zone enforcement ---
                left_jp, left_grip, right_jp, right_grip, safety_info = (
                    self._enforce_safety_zone(left_jp, left_grip, right_jp, right_grip)
                )
                if safety_info:
                    for _side, _si in safety_info.items():
                        if _si and (
                            _si.get("pos_in_elastic")
                            or _si.get("pos_hard_clamp")
                            or _si.get("ori_in_elastic")
                            or _si.get("ori_hard_clamp")
                        ):
                            emit(
                                "cap_server",
                                "safety_enforced",
                                ep=episode,
                                step=steps,
                                side=_side,
                                **_si,
                            )

                # --- Gripper debug logging ---
                _dbg_parts = [
                    f"[GRIP] step={steps}",
                    f"takeover_L={left_takeover} takeover_R={right_takeover}",
                    f"rl_grip_L={rl_grip_L[0]:.4f} rl_grip_R={rl_grip_R[0]:.4f}",
                    f"grip_L={left_grip[0]:.4f} grip_R={right_grip[0]:.4f}",
                ]
                if left_takeover:
                    _fello_l_gp = (
                        float(left_fello_now_grip[0])
                        if left_fello_now_grip is not None
                        else -1
                    )
                    _fello_l_anc = (
                        float(left_fello_anchor_gp[0])
                        if left_fello_anchor_gp is not None
                        else -1
                    )
                    _yam_l_anc = (
                        float(left_yam_anchor_gp[0])
                        if left_yam_anchor_gp is not None
                        else -1
                    )
                    _dbg_parts.append(
                        f"settle_L={left_grip_settle_ticks} unlk_L={left_grip_unlocked} fello_gp={_fello_l_gp:.3f} fello_anc={_fello_l_anc:.3f} yam_anc={_yam_l_anc:.3f}"
                    )
                if right_takeover:
                    _fello_r_gp = (
                        float(right_fello_now_grip[0])
                        if right_fello_now_grip is not None
                        else -1
                    )
                    _fello_r_anc = (
                        float(right_fello_anchor_gp[0])
                        if right_fello_anchor_gp is not None
                        else -1
                    )
                    _yam_r_anc = (
                        float(right_yam_anchor_gp[0])
                        if right_yam_anchor_gp is not None
                        else -1
                    )
                    _dbg_parts.append(
                        f"settle_R={right_grip_settle_ticks} unlk_R={right_grip_unlocked} fello_gp={_fello_r_gp:.3f} fello_anc={_fello_r_anc:.3f} yam_anc={_yam_r_anc:.3f}"
                    )
                _grip_line = " | ".join(_dbg_parts)
                print(_grip_line)
                _hil_log(_grip_line)

                # --- Clamp grippers before writing — prevents RL policy
                # garbage from corrupting the takeover anchor on next press.
                left_grip = np.clip(left_grip, GRIPPER_MIN, GRIPPER_MAX)
                right_grip = np.clip(right_grip, GRIPPER_MIN, GRIPPER_MAX)

                _act_ms = (time.time() - _t_act_start) * 1000
                # --- Write targets under lock — control loop sends at CONTROL_FREQ_HZ ---
                with self._state_lock:
                    self._cmd_left_jp[:] = left_jp
                    self._cmd_left_gp[:] = left_grip
                    self._cmd_right_jp[:] = right_jp
                    self._cmd_right_gp[:] = right_grip

                emit(
                    "cap_server",
                    "action_applied",
                    ep=episode,
                    step=steps,
                    action_source=action_source,
                )

                # Maintain control rate at POLICY_FREQ_HZ — always, even during
                # human takeover so Fello writes are smooth (no jitter).
                sleep_end = last_step_time + POLICY_PERIOD_S
                while time.time() < sleep_end:
                    time.sleep(0.0001)
                last_step_time = time.time()

                # During human takeover, skip expensive obs/reward/RPC for
                # N-1 out of every N steps so Fello control stays at full
                # POLICY_FREQ_HZ while the RL server only gets called at
                # POLICY_FREQ_HZ / HIL_POLICY_SLOWDOWN.
                if action_source == "human":
                    _do_rl_step = _hil_substep % HIL_POLICY_SLOWDOWN == 0
                    _hil_substep += 1
                else:
                    _do_rl_step = True
                    _hil_substep = 0

                steps += 1
                done = steps >= max_steps

                if _do_rl_step:
                    # --- Build next observation ---
                    _t_obs = time.time()
                    obs = self._build_skill_obs(task_description)
                    rl_obs = self._build_rl_obs(obs, control_mode=control_mode)
                    _obs_ms = (time.time() - _t_obs) * 1000
                    emit(
                        "cap_server",
                        "obs_built",
                        ep=episode,
                        step=steps,
                        camera_read_ms=_obs_ms,
                    )

                    # --- Get reward ---
                    emit("cap_server", "reward_start", ep=episode, step=steps)
                    _t_rew = time.time()
                    reward = _get_step_reward(obs)
                    _rew_ms = (time.time() - _t_rew) * 1000
                    cumulative_reward += reward
                    emit(
                        "cap_server",
                        "reward_end",
                        ep=episode,
                        step=steps,
                        reward=reward,
                    )

                    emit("cap_server", "record_done", ep=episode, step=steps)

                    # --- Send transition to rl_policy_server, get next action ---
                    action_executed = {
                        k: np.asarray(v).copy() for k, v in rl_action.items()
                    }
                    transition = {
                        "obs": rl_obs,
                        "action_executed": action_executed,
                        "action_source": action_source,
                        "reward": reward,
                        "done": done,
                    }
                    emit("cap_server", "rpc_call_start", ep=episode, step=steps)
                    _t_rpc = time.time()
                    step_result = rl_client.step(transition).result()
                    _rpc_ms = (time.time() - _t_rpc) * 1000
                    emit("cap_server", "rpc_call_end", ep=episode, step=steps)

                    if not done:
                        rl_action = step_result["action"]
                else:
                    reward = 0.0
                    _obs_ms = _rew_ms = _rpc_ms = 0.0

                _local_ms = (time.time() - _t_local_start) * 1000 - _rpc_ms
                _total_ms = (time.time() - _t_local_start) * 1000
                logger.info(
                    "[learn_skill] ep=%d step=%d/%d src=%s rl_step=%s r=%.3f cumR=%.3f | "
                    "total=%.0fms  rpc=%.0fms local=%.0fms "
                    "(obs=%.0f rew=%.0f fello=%.0f act=%.0f)",
                    episode,
                    steps,
                    max_steps,
                    action_source,
                    _do_rl_step,
                    reward,
                    cumulative_reward,
                    _total_ms,
                    _rpc_ms,
                    _local_ms,
                    _obs_ms,
                    _rew_ms,
                    _fello_ms,
                    _act_ms,
                )
                _progress.update(
                    _task_id,
                    completed=steps,
                    reward=reward,
                    cumR=cumulative_reward,
                    src=action_source,
                    rl_ms=_rpc_ms,
                    local_ms=_local_ms,
                    obs_ms=_obs_ms,
                    rew_ms=_rew_ms,
                    fello_ms=_fello_ms,
                    act_ms=_act_ms,
                )
                self._learn_skill_status = {
                    "active": True,
                    "episode": int(episode),
                    "step": int(steps),
                    "max_steps": int(max_steps),
                    "reward": float(reward),
                    "cumulative_reward": float(cumulative_reward),
                    "action_source": str(action_source),
                }

                emit("cap_server", "step_end", ep=episode, step=steps)

            _progress.update(_task_id, completed=max_steps)
            _progress.stop()
            self._learn_skill_status = {"active": False}

            return {
                "success": True,
                "steps_executed": steps,
            }

        except Exception as e:
            if "_progress" in locals():
                _progress.stop()
            self._learn_skill_status = {"active": False}
            logger.exception("learn_skill failed")
            return {"success": False, "steps_executed": 0, "reason": str(e)}
        finally:
            self._skill_lock.release()

    def get_learn_skill_status(self) -> dict:
        """Return live learn_skill status for the UI."""
        return self._learn_skill_status

    def set_reward_mode(self, mode: str) -> dict:
        """Set the local reward function used by learn_skill / learn_skill_policy.

        When set, the local function is called in-process each step instead of
        contacting the external reward server.  Pass mode="" to clear and fall
        back to the external reward server.

        Available modes are the same as reward_server --mode:
            constant-0, constant-1, random, insert_usb, gemini

        Returns {"ok": True, "mode": mode} or {"ok": False, "error": ...}.
        """
        from enpire.env.forge.cap.reward.reward_server import get_modes

        if mode == "":
            self._local_reward_fn = None
            self._local_reward_mode = None
            logger.info(
                "[CapServer] Local reward cleared — will use external reward server"
            )
            return {"ok": True, "mode": None}

        all_modes = get_modes()
        fn = all_modes.get(mode)
        if fn is None:
            available = list(all_modes.keys())
            logger.warning(
                "[CapServer] Unknown reward mode %r (available: %s)", mode, available
            )
            return {
                "ok": False,
                "error": f"Unknown mode {mode!r}. Available: {available}",
            }

        self._local_reward_fn = fn
        self._local_reward_mode = mode
        logger.info("[CapServer] Local reward set to mode=%r", mode)
        return {"ok": True, "mode": mode}

    # ------------------------------------------------------------------
    # Action type resolvers — convert RL action dict to joint targets
    # ------------------------------------------------------------------

    def _resolve_action_to_joints(
        self, action: dict, action_type: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Dispatch to the correct resolver based on action_type.

        Returns (left_jp[6], left_grip[1], right_jp[6], right_grip[1]).
        """
        if action_type == "joint_angle":
            return self._resolve_joint_angle(action)
        elif action_type == "delta_joint_angle":
            return self._resolve_delta_joint_angle(action)
        elif action_type == "eef_pose":
            return self._resolve_eef_pose(action)
        elif action_type == "delta_eef_pose":
            return self._resolve_delta_eef_pose(action)
        else:
            raise ValueError(f"Unknown action_type: {action_type!r}")

    def _resolve_joint_angle(
        self, action: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Absolute joint positions — write directly (current default behavior).

        Supports partial action dicts: if left/right keys are missing, that arm
        holds its current commanded position (for single-arm control_mode).
        """
        if "left_joint_pos" in action:
            left_jp = np.asarray(action["left_joint_pos"]).ravel()[:6]
            left_grip = np.asarray(action["left_gripper_pos"]).ravel()[:1]
        else:
            with self._state_lock:
                left_jp = self._cmd_left_jp.copy()
                left_grip = self._cmd_left_gp.copy()
        if "right_joint_pos" in action:
            right_jp = np.asarray(action["right_joint_pos"]).ravel()[:6]
            right_grip = np.asarray(action["right_gripper_pos"]).ravel()[:1]
        else:
            with self._state_lock:
                right_jp = self._cmd_right_jp.copy()
                right_grip = self._cmd_right_gp.copy()
        return left_jp, left_grip, right_jp, right_grip

    def _resolve_delta_joint_angle(
        self, action: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Delta joint angles — add delta to current joint positions.

        Missing arm keys → that arm holds (zero delta).
        """
        with self._state_lock:
            cur_ljp = self._cmd_left_jp.copy()
            cur_lgp = self._cmd_left_gp.copy()
            cur_rjp = self._cmd_right_jp.copy()
            cur_rgp = self._cmd_right_gp.copy()
        if "left_joint_pos" in action:
            left_jp = cur_ljp + np.asarray(action["left_joint_pos"]).ravel()[:6]
            left_grip = cur_lgp + np.asarray(action["left_gripper_pos"]).ravel()[:1]
        else:
            left_jp, left_grip = cur_ljp, cur_lgp
        if "right_joint_pos" in action:
            right_jp = cur_rjp + np.asarray(action["right_joint_pos"]).ravel()[:6]
            right_grip = cur_rgp + np.asarray(action["right_gripper_pos"]).ravel()[:1]
        else:
            right_jp, right_grip = cur_rjp, cur_rgp
        return left_jp, left_grip, right_jp, right_grip

    def _resolve_eef_pose(
        self, action: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Absolute EE pose — run IK from current joint seed.

        Missing arm keys → that arm holds current position.
        """
        has_left = "left_ee_pos" in action
        has_right = "right_ee_pos" in action
        with self._state_lock:
            q_seed = pin.neutral(self._pink_model).copy()
            q_seed[:6] = self._cmd_left_jp.copy()
            q_seed[8:14] = self._cmd_right_jp.copy()
            hold_ljp = self._cmd_left_jp.copy()
            hold_lgp = self._cmd_left_gp.copy()
            hold_rjp = self._cmd_right_jp.copy()
            hold_rgp = self._cmd_right_gp.copy()
        if has_left and has_right:
            l_pos = np.asarray(action["left_ee_pos"]).ravel()[:3]
            l_quat = np.asarray(action["left_ee_quat_xyzw"]).ravel()[:4]
            r_pos = np.asarray(action["right_ee_pos"]).ravel()[:3]
            r_quat = np.asarray(action["right_ee_quat_xyzw"]).ravel()[:4]
            left_jp, right_jp = self._inverse_kinematics(
                l_pos, l_quat, r_pos, r_quat, q_seed
            )
            left_grip = np.asarray(action["left_gripper_pos"]).ravel()[:1]
            right_grip = np.asarray(action["right_gripper_pos"]).ravel()[:1]
        elif has_left:
            l_pos = np.asarray(action["left_ee_pos"]).ravel()[:3]
            l_quat = np.asarray(action["left_ee_quat_xyzw"]).ravel()[:4]
            left_jp, _ = self._inverse_kinematics(
                l_pos, l_quat, hold_rjp[:3], hold_rjp[:4], q_seed
            )
            left_grip = np.asarray(action["left_gripper_pos"]).ravel()[:1]
            right_jp, right_grip = hold_rjp, hold_rgp
        elif has_right:
            r_pos = np.asarray(action["right_ee_pos"]).ravel()[:3]
            r_quat = np.asarray(action["right_ee_quat_xyzw"]).ravel()[:4]
            _, right_jp = self._inverse_kinematics(
                hold_ljp[:3], hold_ljp[:4], r_pos, r_quat, q_seed
            )
            right_grip = np.asarray(action["right_gripper_pos"]).ravel()[:1]
            left_jp, left_grip = hold_ljp, hold_lgp
        else:
            left_jp, left_grip = hold_ljp, hold_lgp
            right_jp, right_grip = hold_rjp, hold_rgp
        return left_jp, left_grip, right_jp, right_grip

    def _resolve_delta_eef_pose(
        self, action: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Delta EE pose — FK current, apply delta, then IK.

        Missing arm keys → that arm holds (zero delta).
        """
        from scipy.spatial.transform import Rotation

        has_left = "left_ee_delta_pos" in action
        has_right = "right_ee_delta_pos" in action

        with self._state_lock:
            cur_l_pos = self._left_ee_pos.copy()
            cur_l_quat = self._left_ee_quat.copy()
            cur_r_pos = self._right_ee_pos.copy()
            cur_r_quat = self._right_ee_quat.copy()
            q_seed = pin.neutral(self._pink_model).copy()
            q_seed[:6] = self._cmd_left_jp.copy()
            q_seed[8:14] = self._cmd_right_jp.copy()
            hold_ljp = self._cmd_left_jp.copy()
            hold_lgp = self._cmd_left_gp.copy()
            hold_rjp = self._cmd_right_jp.copy()
            hold_rgp = self._cmd_right_gp.copy()

        # Apply deltas for present arms, hold for missing
        if has_left:
            l_delta_pos = np.asarray(action["left_ee_delta_pos"]).ravel()[:3]
            l_delta_euler = np.asarray(action["left_ee_delta_euler"]).ravel()[:3]
            new_l_pos = cur_l_pos + l_delta_pos
            new_l_rot = Rotation.from_euler("xyz", l_delta_euler) * Rotation.from_quat(
                cur_l_quat
            )
            new_l_quat = new_l_rot.as_quat()
            left_grip = np.asarray(action["left_gripper_pos"]).ravel()[:1]
        else:
            new_l_pos, new_l_quat = cur_l_pos, cur_l_quat
            left_grip = hold_lgp

        if has_right:
            r_delta_pos = np.asarray(action["right_ee_delta_pos"]).ravel()[:3]
            r_delta_euler = np.asarray(action["right_ee_delta_euler"]).ravel()[:3]
            new_r_pos = cur_r_pos + r_delta_pos
            new_r_rot = Rotation.from_euler("xyz", r_delta_euler) * Rotation.from_quat(
                cur_r_quat
            )
            new_r_quat = new_r_rot.as_quat()
            right_grip = np.asarray(action["right_gripper_pos"]).ravel()[:1]
        else:
            new_r_pos, new_r_quat = cur_r_pos, cur_r_quat
            right_grip = hold_rgp

        if has_left or has_right:
            left_jp, right_jp = self._inverse_kinematics(
                new_l_pos, new_l_quat, new_r_pos, new_r_quat, q_seed
            )
        else:
            left_jp, right_jp = hold_ljp, hold_rjp

        if not has_left:
            left_jp = hold_ljp
        if not has_right:
            right_jp = hold_rjp

        return left_jp, left_grip, right_jp, right_grip

    _RL_IMAGE_SIZE = 256

    # proprio_keys order must match remote_deployment.yaml / SERLObsWrapper
    _RL_PROPRIO_KEYS = [
        ("left_joint_pos", 6),
        ("left_gripper_pos", 1),
        ("right_joint_pos", 6),
        ("right_gripper_pos", 1),
    ]

    _RL_PROPRIO_KEYS_LEFT = [
        ("left_joint_pos", 6),
        ("left_gripper_pos", 1),
    ]

    _RL_PROPRIO_KEYS_RIGHT = [
        ("right_joint_pos", 6),
        ("right_gripper_pos", 1),
    ]

    def _get_rl_proprio_keys(self, control_mode: str = "both"):
        if control_mode == "left":
            return self._RL_PROPRIO_KEYS_LEFT
        elif control_mode == "right":
            return self._RL_PROPRIO_KEYS_RIGHT
        return self._RL_PROPRIO_KEYS

    def _build_rl_obs(self, obs: dict, control_mode: str = "both") -> dict:
        """Build RL observation matching the env format (SERLObsWrapper output).

        Images are resized to _RL_IMAGE_SIZE×_RL_IMAGE_SIZE here to keep
        Portal RPC payloads small (~192KB instead of ~2.7MB).

        Returns:
            {
                "state": ndarray(14),              # flat proprio vector
                "top_camera_image": ndarray(256,256,3),
                "left_camera_image": ndarray(256,256,3),
                ...
            }
        """
        sz = self._RL_IMAGE_SIZE
        rl_obs = {}
        for cam_name in CAMERA_NAMES:
            key = f"{cam_name}_camera_image"
            if key in obs:
                img = np.asarray(obs[key], dtype=np.uint8)
                if img.shape[0] != sz or img.shape[1] != sz:
                    img = cv2.resize(img, (sz, sz))
                rl_obs[key] = img
        # Concatenate proprio into flat state vector
        state_parts = []
        for key, dim in self._get_rl_proprio_keys(control_mode):
            state_parts.append(np.asarray(obs[key], dtype=np.float32).ravel()[:dim])
        rl_obs["state"] = np.concatenate(state_parts, axis=0)
        return rl_obs

    def _build_skill_obs(self, task_description: str) -> dict:
        """Build observation dict matching gym env format."""
        with self._state_lock:
            obs = {
                "left_joint_pos": self.left_joint_pos.copy(),
                "left_gripper_pos": self.left_gripper_pos.copy(),
                "right_joint_pos": self.right_joint_pos.copy(),
                "right_gripper_pos": self.right_gripper_pos.copy(),
            }
        for cam_name in CAMERA_NAMES:
            img = self.get_camera_image(cam_name)
            if img is not None:
                obs[f"{cam_name}_camera_image"] = img
        obs["annotation.task"] = task_description
        return obs

    def get_skill_prediction(self) -> dict | None:
        return self._latest_action_chunk

    def cancel_motion(self, side: str) -> dict:
        """Cancel an ongoing _ik_servo on *side*. Thread-safe, non-blocking.

        Sets a per-side threading.Event that _ik_servo checks each control cycle.
        Also clears any pending EEF target so the control loop stops driving.
        """
        event = self._cancel_motion.get(side)
        if event is not None:
            event.set()
        self._eef_targets.pop(side, None)
        return {"success": True, "side": side}

    def estop(self) -> bool:
        self._mark_policy_output_dirty("estop")
        self._safety.trigger_estop()
        print("[CapServer] E-STOP triggered")
        return True

    def release_estop(self) -> bool:
        self._mark_policy_output_dirty("release_estop")
        self._safety.release_estop()
        print("[CapServer] E-STOP released")
        return True

    # ------------------------------------------------------------------
    # Safety zone RPC endpoints
    # ------------------------------------------------------------------

    def set_safety_zone(
        self, side: str, keyposes: list, pos_margin: float, ori_margin: float
    ) -> dict:
        """Set a task-aware EE safety zone for one arm (Portal RPC)."""
        try:
            arm_zone = ArmSafetyZone(
                keyposes=[np.asarray(kp, dtype=np.float64) for kp in keyposes],
                pos_margin=float(pos_margin),
                ori_margin=float(ori_margin),
                elastic_band=float(pos_margin) / 2.0,
                elastic_band_ori=float(ori_margin) / 2.0,
            )
            self._safety.set_arm_zone(side, arm_zone)
            return {"success": True}
        except Exception as e:
            logger.exception("set_safety_zone failed")
            return {"success": False, "error": str(e)}

    def clear_safety_zone(self, side: str = "") -> dict:
        """Clear safety zone(s) (Portal RPC).  Empty string = clear both."""
        try:
            self._safety.clear_task_zone(side if side else None)
            return {"success": True}
        except Exception as e:
            logger.exception("clear_safety_zone failed")
            return {"success": False, "error": str(e)}

    def get_safety_zone(self) -> dict:
        """Return current safety zone config (Portal RPC)."""
        return self._safety.get_zone_config()

    # ------------------------------------------------------------------
    # Safety zone enforcement helper
    # ------------------------------------------------------------------

    _safety_enforce_log_count: int = 0

    def _enforce_safety_zone(
        self,
        left_jp: np.ndarray,
        left_grip: np.ndarray,
        right_jp: np.ndarray,
        right_grip: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Apply task safety zone constraints to proposed joint targets.

        Runs FK on the proposed joints, checks against the zone, and
        interpolates back toward the current commanded position if needed.

        Returns (left_jp, left_grip, right_jp, right_grip, info).
        """
        if not self._safety.has_task_zone():
            return left_jp, left_grip, right_jp, right_grip, {}

        try:
            # Clamp proposed joints to URDF limits before FK (pinocchio
            # raises if joints are out of bounds during forwardKinematics).
            q_lo = self._pink_model.lowerPositionLimit
            q_hi = self._pink_model.upperPositionLimit
            left_jp_fk = np.clip(left_jp, q_lo[:6], q_hi[:6])
            right_jp_fk = np.clip(right_jp, q_lo[8:14], q_hi[8:14])

            info: dict = {}
            for side in ("left", "right"):
                jp = left_jp if side == "left" else right_jp
                grip = left_grip if side == "left" else right_grip

                # FK to get proposed EE pose (using clamped joints for FK only)
                frame = "left_grasp" if side == "left" else "right_grasp"
                se3 = self._frame_pose(frame, left_jp_fk, right_jp_fk)
                ee_pos = se3.translation.copy()
                ee_quat = pin.Quaternion(se3.rotation).coeffs().copy()  # xyzw

                factor, side_info = self._safety.enforce_ee(side, ee_pos, ee_quat)
                info[side] = side_info

                if factor < 1.0:
                    with self._state_lock:
                        if side == "left":
                            cur_jp = self._cmd_left_jp.copy()
                        else:
                            cur_jp = self._cmd_right_jp.copy()

                    # Only interpolate joint positions — gripper is independent
                    # of EE safety zone and should pass through unmodified.
                    safe_jp = cur_jp + factor * (jp - cur_jp)

                    if side == "left":
                        left_jp = safe_jp
                    else:
                        right_jp = safe_jp

            # Log first 5 calls and then every 100th
            self._safety_enforce_log_count += 1
            if (
                self._safety_enforce_log_count <= 5
                or self._safety_enforce_log_count % 100 == 0
            ):
                logger.info(
                    "[safety_enforce] call #%d  left=%s  right=%s",
                    self._safety_enforce_log_count,
                    {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in info.get("left", {}).items()
                    },
                    {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in info.get("right", {}).items()
                    },
                )

            return left_jp, left_grip, right_jp, right_grip, info
        except Exception:
            logger.exception("[safety_enforce] _enforce_safety_zone CRASHED")
            return left_jp, left_grip, right_jp, right_grip, {}

    # ------------------------------------------------------------------
    # Task backend RPCs (RoboCasa)
    # ------------------------------------------------------------------

    def _rpc_reset_env(self) -> dict:
        """Reset the environment for a new episode (RoboCasa only).

        After env reset, re-sync cap_server command state from the fresh
        observations so the control loop doesn't send stale targets.
        """
        from enpire.env.forge.cap.env.base import TaskProtocol

        if self._sim_backend is not None and isinstance(
            self._sim_backend, TaskProtocol
        ):
            result = self._sim_backend.reset_env()
            # Re-sync commanded positions from fresh observations
            self._init_state()
            return result
        return {"ok": False, "error": "not a task backend"}

    def _rpc_reset_to_initial(self) -> dict:
        """Deterministic reset — same scene, same objects, same positions.

        Uses robosuite's ``deterministic_reset`` path to restore the initial
        MuJoCo state without re-randomizing layout/objects.  Suitable for
        agent retry loops where each iteration attempts the same task.
        """
        if self._sim_backend is not None and hasattr(
            self._sim_backend, "reset_to_initial"
        ):
            result = self._sim_backend.reset_to_initial()
            self._init_state()
            return result
        # Fallback: full reset
        return self._rpc_reset_env()

    def _rpc_get_task_info(self) -> dict:
        """Return task info (reward, success, done) from the backend."""
        from enpire.env.forge.cap.env.base import TaskProtocol

        if self._sim_backend is not None and isinstance(
            self._sim_backend, TaskProtocol
        ):
            return self._sim_backend.get_task_info()
        return {"done": False, "reward": 0.0, "success": False}

    def _rpc_get_last_reward(self) -> dict:
        """Return the reward from the most recent step."""
        from enpire.env.forge.cap.env.base import TaskProtocol

        if self._sim_backend is not None and isinstance(
            self._sim_backend, TaskProtocol
        ):
            return {"reward": self._sim_backend.get_last_reward()}
        return {"reward": 0.0}

    def _rpc_load_task(self, task_name: str) -> dict:
        """Load a new task/scene on the env (e.g. RoboCasa task switch)."""
        if self._sim_backend is not None and hasattr(self._sim_backend, "load_task"):
            return self._sim_backend.load_task(task_name)
        return {"ok": False, "error": "env does not support load_task"}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._loop_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._loop_thread.start()
        print(f"[CapServer] Control loop started at {CONTROL_FREQ_HZ}Hz")

        if USE_FELLO and (
            self._left_fello is not None or self._right_fello is not None
        ):
            self._fello_thread = threading.Thread(
                target=self._fello_loop, daemon=True, name="fello-loop"
            )
            self._fello_thread.start()
            print(f"[CapServer] Fello loop started at {POLICY_FREQ_HZ}Hz")

        self._server.start(block=False)
        print(f"[CapServer] Portal RPC server listening on port {CAP_SERVER_PORT}")

    # ------------------------------------------------------------------
    # Scene management (sim-only)
    # ------------------------------------------------------------------

    def setup_scene(self, name: str) -> dict:
        """Load a named scene YAML and inject objects into the sim."""
        result = self._sim_backend.setup_scene(name)
        self._init_state()  # re-sync commanded positions with new sim state
        return result

    def clear_table(self) -> dict:
        """Remove all scene objects from the sim."""
        result = self._sim_backend.clear_table()
        self._init_state()  # re-sync commanded positions with new sim state
        return result

    def list_scenes(self) -> dict:
        """List available scene files and the active scene."""
        return self._sim_backend.get_scenes()

    def get_object_positions(self) -> dict:
        """Return positions of all scene objects (debug)."""
        return self._sim_backend.get_object_positions()

    def set_body_pose(
        self, name: str, pos: list, quat_wxyz: list, gravity_comp: bool = True
    ) -> dict:
        """Set a scene body's pose (sim-only). quat is [w,x,y,z]."""
        return self._sim_backend.set_body_pose(name, pos, quat_wxyz, gravity_comp)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
        try:
            self._server.close(timeout=2.0)
        except Exception:
            pass
        for cam in self._cameras.values():
            cam.close()
        if self._sim_backend is not None:
            self._sim_backend.close()
        print("[CapServer] Stopped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAP Server — CONTROL_FREQ_HZ control loop"
    )
    parser.add_argument("--arm-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=CAP_SERVER_PORT)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument(
        "--no-arms",
        action="store_true",
        help="Use stub arms (no hardware); frees Portal workers for camera/detection",
    )
    parser.add_argument(
        "--rl-host",
        default=RL_POLICY_HOST,
        help="RL policy server host (default: config.RL_POLICY_HOST)",
    )
    parser.add_argument(
        "--env",
        default=None,
        help=(
            "Environment to use (replaces --sim/--robocasa). Examples: "
            "'yam', 'yam-warp', 'robocasa', 'robocasa:PickPlaceCounterToCabinet', "
            "'robocasa:PickPlaceCounterToCabinet:GR1ArmsOnly'"
        ),
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch viewer window (sim envs only)",
    )
    parser.add_argument(
        "--use-fello",
        action="store_true",
        help="Enable Fello HIL takeover in sim mode (normally disabled in sim)",
    )
    parser.add_argument(
        "--always-takeoverable",
        action="store_true",
        help="Enable Fello takeover for ALL commands, not just learn_skill. Implies --use-fello.",
    )
    args = parser.parse_args()

    if args.env == "yam-warp" and args.viewer:
        parser.error(
            "--viewer is not supported with --env yam-warp (no passive viewer)"
        )

    if args.env is not None:
        import enpire.env.forge.cap.config as _cfg

        if not args.use_fello and not args.always_takeoverable:
            _cfg.USE_FELLO = False
        _cfg.ALWAYS_TAKEOVERABLE = bool(args.always_takeoverable)
        global USE_FELLO, ALWAYS_TAKEOVERABLE
        if not args.use_fello and not args.always_takeoverable:
            USE_FELLO = False
        ALWAYS_TAKEOVERABLE = bool(args.always_takeoverable)

    server = CapServer(
        arm_host=args.arm_host,
        server_port=args.port,
        enable_cameras=not args.no_cameras,
        no_arms=args.no_arms,
        rl_host=args.rl_host,
        env_name=args.env,
        env_viewer=args.viewer,
    )

    try:
        server.start()
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[CapServer] Shutting down...")
        server.stop()


if __name__ == "__main__":
    main()
