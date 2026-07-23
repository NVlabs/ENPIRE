# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import logging
import queue
import re
import threading
import time
import typing
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import portal  # noqa: E402
import trimesh  # noqa: E402
import viser  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from viser.extras import ViserUrdf  # noqa: E402
from yourdfpy import URDF  # noqa: E402

from enpire.env.forge.experimental._pyroki_compat import import_pyroki  # noqa: E402

pk = import_pyroki()

# Import kinematics for converting ee_pose to joint positions
import sys

from enpire.env.forge.experimental.embodiment_tags import EmbodimentTag
from enpire.env.forge.experimental.get_action_policy import PolicyAdapters
from enpire.env.forge.experimental.key_remapping_utils import (
    _make_arrays_contiguous,
    hold_action_from_proprio,
)

_TBD_ROOT = Path(__file__).resolve().parents[1]
if str(_TBD_ROOT) not in sys.path:
    sys.path.insert(0, str(_TBD_ROOT))
from enpire.env.forge.robot.yam.kinematics import YamKinematics

# Suppress Portal's verbose logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("portal").setLevel(logging.CRITICAL)

SCRIPTED_PLANNER_MAX_JOINT_VEL_LIMIT = 3.0


# Minimal overlay helper to avoid UI crashes when overlay assets are unavailable.
class FrameOverlay:
    # Expected camera mp4 filenames inside a raw episode folder.
    _CAMERA_MP4S = (
        "top_camera-images-rgb.mp4",
        "left_camera-images-rgb.mp4",
        "right_camera-images-rgb.mp4",
    )

    def __init__(
        self,
        folder_path: str,
        on_progress: typing.Callable[[int, int], None] | None = None,
        on_ready: typing.Callable[[], None] | None = None,
    ):
        # Accepts either an individual episode folder (contains the camera mp4s
        # directly) or a session folder (contains episode subfolders).
        self._folder_path = folder_path
        self._frame_idx = 0
        self._frame_counter = 0
        self._frame: np.ndarray | None = None
        # Pre-scanned list of (folder, video_path) tuples — only folders with valid videos.
        self._entries: list[tuple[Path, Path]] = []
        # Per-episode extra first-frame cache for left/right cameras, so we can
        # present all three camera views side-by-side in the popup.
        self._side_cams: dict[int, dict[str, np.ndarray]] = {}
        self._frame_cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
        self._max_cache = 64  # ~60MB at 640x480 BGR
        self._ready = False
        self._on_progress = on_progress
        self._on_ready = on_ready
        self._last_progress_time = 0.0
        self._scan_thread = threading.Thread(target=self._bg_init, daemon=True)
        self._scan_thread.start()

    @property
    def ready(self) -> bool:
        return self._ready

    def _bg_init(self) -> None:
        """Background thread: scan folders, load first frame, then mark ready."""
        self._init_entries()
        if self._entries:
            self._load_frame(0)
        self._ready = True
        if self._on_ready:
            self._on_ready()

    def total_frames(self) -> int:
        return len(self._entries)

    def get_frame_idx(self) -> int:
        return self._frame_idx

    def get_current_frame_counter(self) -> int:
        return self._frame_counter

    def increment_frame_counter(self) -> None:
        self._frame_counter += 1

    def get_frame(self) -> np.ndarray | None:
        return self._frame

    def get_next_frame(self) -> np.ndarray | None:
        if not self._entries:
            return self._frame
        new_idx = self._frame_idx + 1
        if new_idx >= len(self._entries):
            return self._frame
        return self._load_frame(new_idx)

    def get_prev_frame(self) -> np.ndarray | None:
        if not self._entries:
            return self._frame
        new_idx = self._frame_idx - 1
        if new_idx < 0:
            return self._frame
        return self._load_frame(new_idx)

    def _init_entries(self) -> None:
        """Pre-scan folders and cache video paths. Skips folders without videos.

        If ``folder_path`` contains the camera mp4s directly it's a single
        episode; otherwise its immediate subfolders are scanned.
        """
        if not self._folder_path:
            return
        p = Path(self._folder_path).expanduser()
        if not p.exists():
            print(f"[FrameOverlay] Folder path not found: {p}")
            return
        # Episode folder? (cameras live directly inside)
        top_here = p / self._CAMERA_MP4S[0]
        if top_here.exists():
            self._entries.append((p, top_here))
            print(f"[FrameOverlay] Loaded single-episode folder: {p}")
            return
        root = p

        folders = sorted([p for p in root.iterdir() if p.is_dir()])
        total = len(folders)
        print(f"[FrameOverlay] Scanning {total} folders under {root} ...")
        for i, folder in enumerate(folders):
            matches = sorted(folder.rglob(self._CAMERA_MP4S[0]))
            if matches:
                self._entries.append((folder, matches[0]))
            done = i + 1
            pct = int(done / total * 100) if total else 100
            print(f"\r[FrameOverlay] {done}/{total} ({pct}%)", end="", flush=True)
            now = time.time()
            if (
                self._on_progress
                and total > 0
                and (now - self._last_progress_time > 1.0 or done == total)
            ):
                self._last_progress_time = now
                self._on_progress(done, total)
        print()
        if not self._entries:
            print(f"[FrameOverlay] No valid episodes found under: {root}")
        else:
            print(f"[FrameOverlay] Found {len(self._entries)} episodes with video")

    def _load_frame(self, idx: int) -> np.ndarray | None:
        """Load frame at idx, using bounded cache to avoid repeated I/O."""
        if idx in self._frame_cache:
            self._frame = self._frame_cache[idx]
            self._frame_idx = idx
            return self._frame
        _, video_path = self._entries[idx]
        frame = self._read_first_frame(video_path)
        if frame is not None:
            # Evict oldest entry if cache is full
            if len(self._frame_cache) >= self._max_cache:
                evict = self._cache_order.pop(0)
                self._frame_cache.pop(evict, None)
            self._frame_cache[idx] = frame
            self._cache_order.append(idx)
            self._frame = frame
            self._frame_idx = idx
        return self._frame

    def _read_first_frame(self, video_path: Path) -> np.ndarray | None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[FrameOverlay] Failed to open video: {video_path}")
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"[FrameOverlay] Failed to read first frame: {video_path}")
            return None
        return frame

    def get_triptych_cams(self, idx: int | None = None) -> dict[str, np.ndarray]:
        """Return {short_name: first_frame} for top/left/right cameras.

        Lazy-loads and caches per-episode. Missing cameras are omitted from
        the returned dict. ``short_name`` is one of {"top", "left", "right"}.
        """
        if not self._entries:
            return {}
        if idx is None:
            idx = self._frame_idx
        if idx not in self._side_cams:
            folder, _ = self._entries[idx]
            cams: dict[str, np.ndarray] = {}
            for name in self._CAMERA_MP4S:
                p = folder / name
                if not p.exists():
                    hits = sorted(folder.rglob(name))
                    if hits:
                        p = hits[0]
                    else:
                        continue
                f = self._read_first_frame(p)
                if f is not None:
                    cams[name] = f
            self._side_cams[idx] = cams
        # Map long filenames to short labels top/left/right for readable keys.
        short_map = {
            "top_camera-images-rgb.mp4": "top",
            "left_camera-images-rgb.mp4": "left",
            "right_camera-images-rgb.mp4": "right",
        }
        return {
            short_map[name]: frame
            for name, frame in self._side_cams[idx].items()
            if name in short_map
        }

    def get_triptych_frame(self, idx: int | None = None) -> np.ndarray | None:
        """Return a 3-camera side-by-side composite (top | left | right).

        First frames of each camera in the current entry's folder. Lazily
        loaded and cached per-episode. Any missing camera is replaced with a
        black panel sized to match the available frames.
        """
        if not self._entries:
            return None
        if idx is None:
            idx = self._frame_idx
        if idx not in self._side_cams:
            folder, _ = self._entries[idx]
            cams: dict[str, np.ndarray] = {}
            for name in self._CAMERA_MP4S:
                p = folder / name
                if not p.exists():
                    # Look recursively one level down (e.g. session folders).
                    hits = sorted(folder.rglob(name))
                    if hits:
                        p = hits[0]
                    else:
                        continue
                f = self._read_first_frame(p)
                if f is not None:
                    cams[name] = f
            self._side_cams[idx] = cams
        cams = self._side_cams[idx]
        if not cams:
            return None
        ordered = [cams.get(n) for n in self._CAMERA_MP4S]
        h = max(f.shape[0] for f in ordered if f is not None)
        w = max(f.shape[1] for f in ordered if f is not None)
        panels: list[np.ndarray] = []
        for f in ordered:
            if f is None:
                panels.append(np.zeros((h, w, 3), dtype=np.uint8))
            else:
                if f.shape[:2] != (h, w):
                    f = cv2.resize(f, (w, h))
                panels.append(f)
        return np.concatenate(panels, axis=1)


# Global dictionary to track exception counts per function
_exception_counts: dict[str, int] = {}


def safe_call(func):
    """Decorator that wraps functions with error handling and logging.

    Features:
    - Catches and logs all exceptions except KeyboardInterrupt
    - Re-raises KeyboardInterrupt to allow graceful shutdown
    - Prints descriptive error messages with function name
    - Tracks and logs cumulative exception counts per function
    - Returns None on error to allow graceful degradation
    """
    import traceback
    from functools import wraps

    func_key = f"{func.__module__}.{func.__qualname__}"

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise  # Always re-raise KeyboardInterrupt for clean shutdown
        except Exception as e:
            # Track exception count for this function
            _exception_counts[func_key] = _exception_counts.get(func_key, 0) + 1
            count = _exception_counts[func_key]

            print(f"[{func.__name__}] Error (#{count}): {e}")
            traceback.print_exc()
            return None

    return wrapper


State = Literal[
    "pause",
    "start",
    "step_once",
    "home",
    "reset_policy",
    "go_to_target",
    "sync_to_init",
]
MotionPlannerBackend = Literal["rrtconnect", "curobo"]
MotionPlannerSolverSpeed = Literal["slow", "fast"]
MIN_REAL_ACTIONS_TO_RECORD = 3
_MOTION_PLANNER_BACKENDS: tuple[str, ...] = ("rrtconnect", "curobo")
_MOTION_PLANNER_SOLVER_SPEEDS: tuple[str, ...] = ("slow", "fast")


def _normalize_motion_planner_backend(value: str | None) -> MotionPlannerBackend:
    backend = (value or "curobo").strip().lower()
    if backend not in _MOTION_PLANNER_BACKENDS:
        print(f"[Planner] Unknown backend '{value}', falling back to curobo")
        backend = "curobo"
    return typing.cast(MotionPlannerBackend, backend)


def _normalize_motion_planner_solver_speed(
    value: str | None,
) -> MotionPlannerSolverSpeed:
    solver_speed = (value or "fast").strip().lower()
    if solver_speed not in _MOTION_PLANNER_SOLVER_SPEEDS:
        print(f"[Planner] Unknown solver speed '{value}', falling back to fast")
        solver_speed = "fast"
    return typing.cast(MotionPlannerSolverSpeed, solver_speed)


def _create_motion_planner(
    backend: str,
    *,
    shared_port: int | None = None,
    start_server: bool = True,
    solver_speed: str | None = None,
) -> Any:
    backend = _normalize_motion_planner_backend(backend)
    if backend == "curobo":
        from enpire.env.forge.experimental.portal_motion_planner import PortalMotionPlanner

        kwargs: dict[str, Any] = {
            "backend": "curobo",
            "port": shared_port,
            "start_server": start_server,
        }
        if solver_speed is not None:
            kwargs["solver_speed"] = _normalize_motion_planner_solver_speed(
                solver_speed
            )
        try:
            return PortalMotionPlanner(**kwargs)
        except TypeError:
            kwargs.pop("solver_speed", None)
            return PortalMotionPlanner(**kwargs)
    from enpire.env.forge.experimental.motion_planner import YamMotionPlanner

    return YamMotionPlanner()


def _format_planner_timing(result: dict[str, Any]) -> str:
    total = result.get("curobo_total_time_ms")
    solve = result.get("curobo_solve_time_ms")
    ik = result.get("curobo_ik_time_ms")
    trajopt = result.get("curobo_trajopt_time_ms")
    finetune = result.get("curobo_finetune_time_ms")
    parts = []
    if total is not None:
        parts.append(f"total={float(total):.1f} ms")
    if solve is not None:
        parts.append(f"solve={float(solve):.1f} ms")
    if ik is not None:
        parts.append(f"ik={float(ik):.1f} ms")
    if trajopt is not None:
        parts.append(f"trajopt={float(trajopt):.1f} ms")
    if finetune is not None:
        parts.append(f"finetune={float(finetune):.1f} ms")
    return ", ".join(parts)


class StartStopPlayPolicyWrapper:
    def __init__(
        self,
        policy: Any,  # RobotInterface | PortalPolicy
        adapters: PolicyAdapters,
        embodiment_tag: EmbodimentTag | str,
        policy_port: int = 8009,
        viser_host: str = "localhost",
        viser_port: int = 8010,
        motion_planner_backend: MotionPlannerBackend = "curobo",
        default_motion_planner_solver_speed: MotionPlannerSolverSpeed = "fast",
        preload_motion_planner: bool = False,
        shared_motion_planner_port: int | None = None,
    ):
        """
        This is a wrapper class over any policy that supports start, stop, play, and pause features.
        Uses Portal for subprocess communication with ViserUI.
        """
        self.policy = policy
        self.adapters = adapters
        self._embodiment_tag = embodiment_tag

        # A simple state machine driven by enter_state() and get_action().
        self._execution_state: State = "pause"

        # Episode recording state (set by the control loop, forwarded to ViserUI)
        self._is_recording: bool = False
        self._record_enabled: bool = False
        self._auto_record_start: bool = False
        self._auto_record_save: bool = False
        self._last_episode_dir: str = ""
        self._uploaded_episode_dirs: set[str] = set()
        self._right_button_states: dict[str, bool] = {
            "start": False,
            "pause": False,
            "home": False,
        }

        # Language command currently being run
        self._task_command: str | None = None
        self._pending_task_command: str | None = None
        self._pending_task_command_ts: float = 0.0
        self._task_command_debounce_s: float = 0.35

        # Action to hold the robot in place
        # Computed from the observation when entering pause state, and kept
        # constant to avoid drifting while paused
        self._hold_action: dict[str, Any] | None = None

        # Information to send to ViserUI
        self._current_obs_action: dict[str, Any] | None = None
        # Only counts external / Homing resets
        self._reset_count: int = 0
        self._real_actions_this_episode = 0
        self._pending_replay_config: dict[str, Any] | None = None
        self._pending_sync_to_init = False
        self._control_mode_changed: str | None = None
        self._voice_enabled = False
        self._voice_once_event = threading.Event()
        self._object_detections: list[dict] | None = None
        self._target_ee_pose: dict[str, Any] | None = None
        self._kinematics: Any | None = None
        self._pending_scripted_commands: list[dict[str, Any]] = []
        self._motion_planner_backend = _normalize_motion_planner_backend(
            motion_planner_backend
        )
        self._default_motion_planner_solver_speed = (
            _normalize_motion_planner_solver_speed(default_motion_planner_solver_speed)
        )
        self._motion_planners: dict[str, Any] = {}
        self._shared_motion_planner_port = shared_motion_planner_port

        # Portal IPC: Server to receive commands from ViserUI
        self._server = portal.Server(policy_port)
        self._server.bind("enter_state", self._portal_enter_state)
        self._server.bind("set_task_command", self._portal_set_task_command)
        self._server.bind("reset", self._portal_reset)
        self._server.bind("set_replay_config", self._portal_set_replay_config)
        self._server.bind("set_record_enabled", self._portal_set_record_enabled)
        self._server.bind("set_voice_enabled", self._portal_set_voice_enabled)
        self._server.bind("trigger_voice_once", self._portal_trigger_voice_once)
        self._server.bind("go_to_target", self._portal_go_to_target)
        self._server.bind("scripted_nudge", self._portal_scripted_nudge)
        self._server.bind("scripted_set_gripper", self._portal_scripted_set_gripper)
        self._server.bind("scripted_set_target", self._portal_scripted_set_target)
        self._server.bind("scripted_get_ee_poses", self._portal_scripted_get_ee_poses)
        self._server.bind("scripted_move_to", self._portal_scripted_move_to)
        self._server.bind(
            "scripted_execute_trajectory", self._portal_scripted_execute_trajectory
        )
        self._server.bind("sync_to_init", self._portal_sync_to_init)

        # Start portal server in background thread
        server_thread = threading.Thread(target=self._server.start, daemon=True)
        server_thread.start()
        time.sleep(0.1)  # Give server time to start

        # Portal IPC: Client to send state updates to ViserUI
        self._client = portal.Client(f"{viser_host}:{viser_port}")

        print(
            f"[Policy] Portal IPC ready (port {policy_port} ← commands, {viser_host}:{viser_port} → updates)",
            flush=True,
        )

        if preload_motion_planner:
            try:
                print(
                    f"[Policy] Preloading {self._motion_planner_backend} motion planner "
                    f"(solver_speed={self._default_motion_planner_solver_speed})...",
                    flush=True,
                )
                preload_key = (
                    self._motion_planner_backend
                    if self._motion_planner_backend != "curobo"
                    else f"{self._motion_planner_backend}:{self._default_motion_planner_solver_speed}"
                )
                planner = _create_motion_planner(
                    self._motion_planner_backend,
                    shared_port=self._shared_motion_planner_port,
                    start_server=self._shared_motion_planner_port is None,
                    solver_speed=self._default_motion_planner_solver_speed,
                )
                self._motion_planners[preload_key] = planner
                if hasattr(planner, "port"):
                    self._shared_motion_planner_port = int(planner.port)
            except Exception as exc:
                print(
                    f"[Policy] Motion planner preload failed ({self._motion_planner_backend}): {exc}",
                    flush=True,
                )

        self.lock = threading.RLock()

    @staticmethod
    def _clamp_planner_max_joint_vel(value: float | None) -> float | None:
        if value is None:
            return None
        return min(
            SCRIPTED_PLANNER_MAX_JOINT_VEL_LIMIT,
            max(0.0, float(value)),
        )

    @staticmethod
    def _clamp_planner_ik_error_threshold(value: float | None) -> float | None:
        if value is None:
            return None
        return min(0.10, max(0.001, float(value)))

    @staticmethod
    def _clamp_planner_ik_xyz_weight(value: float | None) -> float | None:
        if value is None:
            return None
        return min(20.0, max(0.001, float(value)))

    @staticmethod
    def _clamp_planner_ik_rpy_weight(value: float | None) -> float | None:
        if value is None:
            return None
        return min(20.0, max(0.0001, float(value)))

    @property
    def shared_motion_planner_port(self) -> int | None:
        return self._shared_motion_planner_port

    def _get_motion_planner(
        self,
        *,
        position_cost: float = 1.0,
        orientation_cost: float = 0.05,
    ) -> Any:
        if not hasattr(self, "_motion_planner_cache"):
            self._motion_planner_cache: dict[tuple[float, float], Any] = {}
        key = (float(position_cost), float(orientation_cost))
        planner = self._motion_planner_cache.get(key)
        if planner is None:
            from enpire.env.forge.experimental.motion_planner import YamMotionPlanner

            planner = YamMotionPlanner(
                position_cost=float(position_cost),
                orientation_cost=float(orientation_cost),
            )
            self._motion_planner_cache[key] = planner
        return planner

    def _portal_enter_state(self, state: str) -> bool:
        """Portal RPC handler for enter_state command from ViserUI."""
        self.enter_state(typing.cast(State, state))
        return True

    def _portal_set_task_command(self, task_command: str) -> bool:
        """Portal RPC handler for set_task_command from ViserUI."""
        self.set_task_command(task_command)
        return True

    def _portal_reset(self) -> bool:
        """Portal RPC handler for reset command from ViserUI."""
        # This is called from another thread, so we can't actually reset here.
        self.enter_state("reset_policy")
        return True

    def _portal_set_replay_config(self, config: dict[str, Any]) -> bool:
        """Portal RPC handler for replay config updates from ViserUI."""
        dataset_path = config.get("dataset_path")
        control_mode = config.get("control_mode")
        norm_stats_path = config.get("norm_stats_path")
        action_horizon = config.get("action_horizon")
        self.set_replay_config(
            dataset_path=dataset_path,
            control_mode=control_mode,
            norm_stats_path=norm_stats_path,
            action_horizon=action_horizon,
        )
        return True

    def _portal_set_record_enabled(self, enabled: bool) -> bool:
        """Portal RPC handler for toggling auto-record from ViserUI checkbox."""
        with self.lock:
            self._record_enabled = bool(enabled)
        print(f"[Policy] Record enabled: {self._record_enabled}")
        return True

    def _portal_set_voice_enabled(self, enabled: bool) -> bool:
        """Portal RPC handler for enabling/disabling voice capture."""
        self.set_voice_enabled(bool(enabled))
        return True

    def _portal_trigger_voice_once(self) -> bool:
        """Portal RPC handler for one-shot voice capture."""
        self.trigger_voice_once()
        return True

    def _portal_go_to_target(self, pose: dict) -> bool:
        """Portal RPC handler: move right EE to a target pose."""
        self._target_ee_pose = pose
        self.enter_state("go_to_target")
        print(f"[Policy] Go to target: {pose}")
        return True

    def _portal_scripted_nudge(self, payload: dict[str, Any]) -> bool:
        """Portal RPC handler for scripted SE(3) nudges."""
        self.enqueue_scripted_command({"type": "nudge", **(payload or {})})
        return True

    def _portal_scripted_set_gripper(self, payload: dict[str, Any]) -> bool:
        """Portal RPC handler for scripted gripper commands."""
        self.enqueue_scripted_command({"type": "gripper", **(payload or {})})
        return True

    def _portal_scripted_set_target(self, payload: dict[str, Any]) -> bool:
        """Portal RPC handler for absolute EE target from 6D gizmo."""
        self.enqueue_scripted_command({"type": "set_target", **(payload or {})})
        return True

    def _portal_scripted_get_ee_poses(self) -> dict:
        """Portal RPC handler: return current EE poses from ScriptedPolicy FK."""
        if hasattr(self.policy, "get_current_ee_poses"):
            result = self.policy.get_current_ee_poses()
            if result is not None:
                return result
        return {}

    def _portal_scripted_move_to(self, payload: dict[str, Any]) -> bool:
        """Portal RPC handler for scripted absolute-EE move-to commands."""
        self.enqueue_scripted_command({"type": "move_to", **(payload or {})})
        return True

    def _portal_scripted_execute_trajectory(self, payload: dict[str, Any]) -> bool:
        """Portal RPC handler for scripted joint-trajectory execution."""
        self.enqueue_scripted_command({"type": "execute_trajectory", **(payload or {})})
        return True

    def _portal_sync_to_init(self) -> bool:
        """Portal RPC handler: move robot to the replay episode's first frame state.

        Signals the control loop to run ``_calibrate_for_delta_replay()`` (which
        works for *any* control mode, not just delta modes), then auto-pauses so
        the operator can inspect the pose before pressing Play.
        """
        self._pending_sync_to_init = True
        if (
            self._pending_replay_config is not None
            or self.execution_state == "reset_policy"
        ):
            print("[Policy] Queued sync_to_init until replay config reset completes")
        else:
            self.enter_state("sync_to_init")
        return True

    @property
    def execution_state(self) -> State:
        return self._execution_state

    def enter_state(self, state: State):
        with self.lock:
            prev = self._execution_state
            print("Policy entering state:", state)
            self._execution_state = state
            if state == "pause":
                self._hold_action = None
                self._current_obs_action = None
                if self._record_enabled and prev == "start" and self._is_recording:
                    self._auto_record_save = True
            elif state == "start":
                if self._record_enabled and not self._is_recording:
                    self._auto_record_start = True

    def _update_right_button_states(self, info: dict[str, Any] | None) -> None:
        if not isinstance(info, dict):
            return
        states = info.get("right_button_states")
        if not isinstance(states, dict):
            return
        self._right_button_states = {
            "start": bool(states.get("start", False)),
            "pause": bool(states.get("pause", False)),
            "home": bool(states.get("home", False)),
        }

    def _policy_get_action(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Wrapper around self.policy.get_action containing extra logic we always want to run"""
        if self._task_command and len(self._task_command) > 1:
            observation["annotation.task"] = self._task_command
        action, info = self.policy.get_action(observation)
        self._update_right_button_states(info)
        # NOTE: Do NOT consume _control_mode_changed here.
        # _pop_control_mode_info() is the ONLY place that should consume it,
        # because _policy_get_action is also called during the "pause" state
        # (to populate the UI chunk) where its return value is discarded.
        # Consuming it here would silently eat the mode-change before the
        # control loop ever sees it.
        self._current_obs_action = {
            "action": action,
            "info": info,
            "observation": observation,
            "reset_count": self._reset_count,
        }
        self._send_viser_message(observation)
        # Send state update to ViserUI via Portal
        return action, info

    def _pop_control_mode_info(self) -> dict[str, Any]:
        if self._control_mode_changed:
            payload = {"control_mode_changed": self._control_mode_changed}
            self._control_mode_changed = None
            return payload
        return {}

    def _policy_reset(self) -> dict[str, Any]:
        """Wrapper around self.policy.reset() containing extra logic we always want to run"""
        self.enter_state("pause")
        policy_info = self.policy.reset() or {}
        policy_info["task_name"] = self._task_command
        return policy_info

    @safe_call
    def _send_viser_message(self, observation: dict[str, Any]):
        if self._current_obs_action is not None:
            message = self._current_obs_action.copy()
            # Extract attention data from info to top level for viser panel
            if "info" in message and "attention" in message["info"]:
                message["attention"] = message["info"]["attention"]
        else:
            message = {}
        message["reset_count"] = self._reset_count
        message["observation"] = observation
        message["is_recording"] = self._is_recording
        message["right_button_states"] = self._right_button_states
        if self._last_episode_dir:
            message["last_episode_dir"] = self._last_episode_dir
        message["voice_enabled"] = self._voice_enabled
        if self._task_command:
            message["task_command"] = self._task_command
        if self._object_detections is not None:
            message["object_detections"] = self._object_detections
        message_clean = _make_arrays_contiguous(message)
        future = self._client.update_state(message_clean)
        # This probably blocks on sending the message to the viser process
        # It might make sense to remove this if it's causing latency, at the
        # cost of potentially more confusing debugging:
        future.result()

    def set_task_command(self, task_command: str):
        # This method may be called by another thread by the viser UI
        # To avoid racing get_action()'s state machine, only queue the latest
        # task command here and let the get_action() thread apply it under the
        # wrapper lock on the next tick.
        cleaned = (task_command or "").strip()
        if not cleaned:
            print("[Policy] Ignored empty task command")
            return
        with self.lock:
            self._pending_task_command = cleaned
            self._pending_task_command_ts = time.monotonic()

    def save_attention_recording(self) -> dict[str, Any]:
        """Manually trigger attention recording save on the policy server."""
        if hasattr(self.policy, "save_attention_recording"):
            return self.policy.save_attention_recording()
        print("[Policy] save_attention_recording not supported by underlying policy")
        return {"success": False, "message": "Not supported"}

    def new_attention_session(self) -> dict[str, Any]:
        """Start a new attention recording session on the policy server."""
        if hasattr(self.policy, "new_attention_session"):
            return self.policy.new_attention_session()
        print("[Policy] new_attention_session not supported by underlying policy")
        return {"success": False, "message": "Not supported"}

    def set_replay_config(
        self,
        dataset_path: str | None = None,
        control_mode: str | None = None,
        norm_stats_path: str | None = None,
        action_horizon: int | None = None,
    ) -> None:
        # Called by ViserUI thread; defer to get_action() thread for safety.
        # No lock — dict assignment is atomic in CPython; get_action() reads
        # and clears _pending_replay_config at the top of each iteration.
        self._pending_replay_config = {
            "dataset_path": dataset_path,
            "control_mode": control_mode,
            "norm_stats_path": norm_stats_path,
            "action_horizon": action_horizon,
        }

    def set_voice_enabled(self, enabled: bool) -> None:
        self._voice_enabled = bool(enabled)
        print(f"[Policy] Voice enabled: {self._voice_enabled}")

    def get_voice_enabled(self) -> bool:
        return self._voice_enabled

    def trigger_voice_once(self) -> None:
        self._voice_once_event.set()
        print("[Policy] Voice capture requested")

    def set_object_detections(self, detections: list[dict]) -> None:
        self._object_detections = detections

    def enqueue_scripted_command(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self._pending_scripted_commands.append(payload)

    def consume_voice_once(self) -> bool:
        with self.lock:
            if self._voice_once_event.is_set():
                self._voice_once_event.clear()
                return True
        return False

    def get_action(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self.lock:
            if self._pending_task_command is not None:
                pending_age = time.monotonic() - self._pending_task_command_ts
                if pending_age >= self._task_command_debounce_s:
                    cleaned = self._pending_task_command
                    self._pending_task_command = None
                    if cleaned != self._task_command:
                        print(f"[Policy] Task command set: {cleaned}")
                        self._task_command = cleaned
                        self.enter_state("reset_policy")
            # this assumes the underlying policy is responsible for data conversion
            # (policy_adapters.map_observation and policy_adapters.map_action)
            if self._pending_scripted_commands:
                pending = self._pending_scripted_commands
                self._pending_scripted_commands = []
                if hasattr(self.policy, "ensure_initialized"):
                    try:
                        self.policy.ensure_initialized(observation)
                    except Exception as exc:
                        print(f"[Policy] Failed to initialize scripted policy: {exc}")
                for cmd in pending:
                    self._apply_scripted_command(cmd, observation)
            if self._pending_replay_config is not None:
                pending = self._pending_replay_config
                self._pending_replay_config = None
                if hasattr(self.policy, "update_replay_config"):
                    try:
                        self.policy.update_replay_config(
                            dataset_path=pending.get("dataset_path"),
                            control_mode=pending.get("control_mode"),
                            norm_stats_path=pending.get("norm_stats_path"),
                            action_horizon=pending.get("action_horizon"),
                        )
                        self.enter_state("reset_policy")
                        if pending.get("control_mode"):
                            self._control_mode_changed = str(pending["control_mode"])
                        print("[Policy] Replay config updated")
                    except Exception as exc:
                        print(f"[Policy] Failed to update replay config: {exc}")
                else:
                    print("[Policy] Underlying policy does not support replay updates")

            if self.execution_state == "reset_policy":
                # This enters the pause state
                # We enter this state whenever the policy logically resets (e.g.
                # changing the language command), so we call the internal api to
                # avoid counting them.
                self._policy_reset()

            if self._pending_sync_to_init and self.execution_state == "pause":
                print("[Policy] Processing queued sync_to_init request")
                self._pending_sync_to_init = False
                self.enter_state("sync_to_init")

            if self.execution_state == "pause":
                # Poll for button events from inner policy (if supported)
                pause_button_info: dict[str, Any] = {}
                if hasattr(self.policy, "poll_button_events"):
                    pause_button_info = self.policy.poll_button_events(observation)
                    self._update_right_button_states(pause_button_info)
                    if pause_button_info.get("ui_home"):
                        print("[Policy] Button: entering home state")
                        self.enter_state("home")
                    elif pause_button_info.get("pause_toggle") or pause_button_info.get(
                        "start_pressed"
                    ) or pause_button_info.get("ui_start"):
                        print("[Policy] Start request detected, entering start state")
                        self.enter_state("start")
                is_scripted_like = hasattr(
                    self.policy, "execute_trajectory"
                ) or hasattr(self.policy, "move_to")
                if not is_scripted_like and (
                    self._current_obs_action is None
                    or "action_chunk" not in self._current_obs_action["info"]
                ):
                    # Even though we are paused, we need to call get_action once to
                    # have a chunk to show in the UI
                    self._policy_get_action(observation)
                else:
                    self._send_viser_message(observation)
                if self._hold_action is None:
                    _, proprio = self.adapters.map_observation(observation)  # type: ignore[misc]
                    self._hold_action = hold_action_from_proprio(proprio)
                # We don't update the inner policy here to keep the chunk "frozen"
                pause_info = self._pop_control_mode_info()
                # Forward footswitch events so the control loop can act on them
                if self._auto_record_save:
                    pause_info["save_pressed"] = True
                    self._auto_record_save = False
                for key in ("save_pressed", "start_pressed"):
                    if pause_button_info.get(key):
                        pause_info[key] = True
                return self._hold_action, pause_info
            elif self.execution_state == "start":
                self._hold_action = None
                self._real_actions_this_episode += 1
                action, info = self._policy_get_action(observation)
                if self._auto_record_start:
                    info["start_pressed"] = True
                    self._auto_record_start = False
                if "source" not in action:
                    action["source"] = "policy"
                # Check for pause toggle from inner policy
                if info.get("pause_toggle"):
                    print("[Policy] Pause toggle detected, entering pause state")
                    self.enter_state("pause")
                # Auto-pause when replay episode finishes (all steps played).
                # This prevents the last delta action from being applied
                # indefinitely, which would cause the robot to drift.
                if info.get("episode_done"):
                    print(
                        "[Policy] Replay episode done — auto-pausing. "
                        "Press Home to reset."
                    )
                    self.enter_state("pause")
                if info.get("ui_pause"):
                    print("[Policy] Button: entering pause state")
                    self.enter_state("pause")
                if info.get("ui_home"):
                    print("[Policy] Button: entering home state")
                    self.enter_state("home")
                info.update(self._pop_control_mode_info())
                return action, info
            elif self.execution_state == "step_once":
                self._real_actions_this_episode += 1
                action, info = self._policy_get_action(observation)
                if "source" not in action:
                    action["source"] = "policy"
                self.enter_state("pause")
                info.update(self._pop_control_mode_info())
                return action, info
            elif self.execution_state == "go_to_target":
                if self._target_ee_pose is None:
                    print("[Policy] go_to_target: no target set, pausing")
                    self.enter_state("pause")
                    return self._hold_action or hold_action_from_proprio(
                        self.adapters.map_observation(observation)[1]
                    ), self._pop_control_mode_info()

                # Get current joint positions from observation
                _, proprio = self.adapters.map_observation(observation)
                if self._kinematics is None:
                    self._kinematics = YamKinematics()

                left_jp = np.asarray(
                    proprio.get(
                        "joint_pos_obs_left", proprio.get("left_joint_pos", np.zeros(6))
                    ),
                    dtype=np.float32,
                ).reshape(-1)[:6]
                right_jp = np.asarray(
                    proprio.get(
                        "joint_pos_obs_right",
                        proprio.get("right_joint_pos", np.zeros(6)),
                    ),
                    dtype=np.float32,
                ).reshape(-1)[:6]

                # Compute current left EE pose (to keep it still)
                left_ee_pos, left_ee_quat, _, _ = self._kinematics.forward_kinematics(
                    left_jp, right_jp
                )

                target_pos = np.asarray(
                    self._target_ee_pose["right_ee_pos"], dtype=np.float32
                )
                target_quat = np.asarray(
                    self._target_ee_pose["right_ee_quat_xyzw"], dtype=np.float32
                )

                # IK: compute target joint angles
                _, target_right_jp = self._kinematics.inverse_kinematics(
                    left_ee_pos,
                    left_ee_quat,
                    target_pos,
                    target_quat,
                )

                left_grip = np.asarray(
                    proprio.get(
                        "gripper_pos_obs_left",
                        proprio.get("left_gripper_pos", np.ones(1)),
                    ),
                    dtype=np.float32,
                ).reshape(-1)[:1]
                right_grip = np.asarray(
                    proprio.get(
                        "gripper_pos_obs_right",
                        proprio.get("right_gripper_pos", np.ones(1)),
                    ),
                    dtype=np.float32,
                ).reshape(-1)[:1]

                # Store target joint state for env.reset (control loop reads this)
                self._go_to_target_joint_state = {
                    "left_joint_pos": left_jp.astype(np.float32),
                    "left_gripper_pos": left_grip,
                    "right_joint_pos": target_right_jp.astype(np.float32),
                    "right_gripper_pos": right_grip,
                }

                print(
                    f"[Policy] go_to_target: target_ee={target_pos.tolist()}, "
                    f"target_right_jp={target_right_jp.tolist()}"
                )

                # Reset inner policy so it generates fresh chunks from the new position
                self.policy.reset()
                print("[Policy] go_to_target: inner policy reset")

                # Return hold action + event for control loop to call env.reset
                hold = hold_action_from_proprio(proprio)
                self._send_viser_message(observation)
                self.enter_state("pause")
                self._target_ee_pose = None
                info = self._pop_control_mode_info()
                info["event"] = "go_to_target"
                return hold, info
            elif self.execution_state == "home":
                action, info = self._policy_get_action(observation)
                if "source" not in action:
                    action["source"] = "policy"
                self._reset_count += 1
                # Immediately inform ViserUI that the episode has restarted,
                # before the robot starts moving back to home.
                # This avoids a race condition with the operator where they hit "Home",
                # then view the next overlay while the robot is going home.
                # Without this the reset would count for the new frame instead of the current one.
                self._send_viser_message(observation)
                # _policy_reset() sets self._current_obs_action = None, so we need to send the viser message first
                self._policy_reset()
                info["event"] = "home"
                info.update(self._pop_control_mode_info())
                return action, info
            elif self.execution_state == "sync_to_init":
                # Signal the control loop to interpolate to the episode's first frame,
                # then immediately transition to pause so the operator can review
                # the pose before pressing Play.
                self._pending_sync_to_init = False
                if self._hold_action is None:
                    _, proprio = self.adapters.map_observation(observation)  # type: ignore[misc]
                    self._hold_action = hold_action_from_proprio(proprio)
                self.enter_state("pause")
                info = self._pop_control_mode_info()
                info["event"] = "sync_to_init"
                return self._hold_action, info
            else:
                # Handle rare races where another thread requested a logical
                # reset between state checks.
                if self.execution_state == "reset_policy":
                    self._policy_reset()
                    if self._hold_action is None:
                        _, proprio = self.adapters.map_observation(observation)  # type: ignore[misc]
                        self._hold_action = hold_action_from_proprio(proprio)
                    return self._hold_action, self._pop_control_mode_info()
                # Fallback for unexpected state
                raise ValueError(f"Unknown execution state: {self.execution_state}")

    def _apply_scripted_command(
        self, payload: dict[str, Any], observation: dict[str, Any]
    ) -> None:
        cmd_type = payload.get("type")
        if cmd_type == "nudge":
            if hasattr(self.policy, "apply_nudge"):
                side = payload.get("side")
                delta_pos = payload.get("delta_pos")
                delta_quat = payload.get("delta_quat_xyzw")
                try:
                    self.policy.apply_nudge(
                        side=side,
                        delta_pos=np.asarray(delta_pos)
                        if delta_pos is not None
                        else None,
                        delta_quat_xyzw=np.asarray(delta_quat)
                        if delta_quat is not None
                        else None,
                    )
                except Exception as exc:
                    print(f"[Policy] Failed to apply scripted nudge: {exc}")
            else:
                print("[Policy] Scripted nudge not supported by underlying policy")
        elif cmd_type == "gripper":
            if hasattr(self.policy, "set_gripper"):
                side = payload.get("side")
                value = payload.get("value")
                try:
                    self.policy.set_gripper(side=side, value=float(value))
                except Exception as exc:
                    print(f"[Policy] Failed to set scripted gripper: {exc}")
            else:
                print("[Policy] Scripted gripper not supported by underlying policy")
        elif cmd_type == "move_to":
            if hasattr(self.policy, "move_to"):
                left_pos = payload.get("left_pos")
                right_pos = payload.get("right_pos")
                left_quat = payload.get("left_quat")
                right_quat = payload.get("right_quat")
                left_gripper = payload.get("left_gripper")
                right_gripper = payload.get("right_gripper")
                left_distance_per_step = payload.get("left_distance_per_step")
                right_distance_per_step = payload.get("right_distance_per_step")
                use_planner = payload.get("use_planner", False)
                planner_max_joint_vel = self._clamp_planner_max_joint_vel(
                    payload.get("planner_max_joint_vel")
                )
                planner_ik_error_threshold = self._clamp_planner_ik_error_threshold(
                    payload.get("planner_ik_error_threshold")
                )
                planner_ik_xyz_weight = self._clamp_planner_ik_xyz_weight(
                    payload.get("planner_ik_xyz_weight")
                )
                planner_ik_rpy_weight = self._clamp_planner_ik_rpy_weight(
                    payload.get("planner_ik_rpy_weight")
                )
                planner_backend = payload.get("planner_backend")
                planner_solver_speed = payload.get("planner_solver_speed")
                curobo_finetune = payload.get("curobo_finetune")
                try:
                    if use_planner and hasattr(self.policy, "execute_trajectory"):
                        return self._apply_planned_move_to(
                            left_pos=left_pos,
                            right_pos=right_pos,
                            left_quat=left_quat,
                            right_quat=right_quat,
                            left_gripper=left_gripper,
                            right_gripper=right_gripper,
                            observation=observation,
                            planner_max_joint_vel=planner_max_joint_vel,
                            planner_ik_error_threshold=planner_ik_error_threshold,
                            planner_ik_xyz_weight=planner_ik_xyz_weight,
                            planner_ik_rpy_weight=planner_ik_rpy_weight,
                            planner_backend=planner_backend,
                            planner_solver_speed=planner_solver_speed,
                            curobo_finetune=curobo_finetune,
                        )
                    accepted = self.policy.move_to(
                        left_target_pos=np.asarray(left_pos, dtype=np.float64)
                        if left_pos is not None
                        else None,
                        left_target_quat_xyzw=np.asarray(left_quat, dtype=np.float64)
                        if left_quat is not None
                        else None,
                        right_target_pos=np.asarray(right_pos, dtype=np.float64)
                        if right_pos is not None
                        else None,
                        right_target_quat_xyzw=np.asarray(right_quat, dtype=np.float64)
                        if right_quat is not None
                        else None,
                        left_target_gripper=float(left_gripper)
                        if left_gripper is not None
                        else None,
                        right_target_gripper=float(right_gripper)
                        if right_gripper is not None
                        else None,
                        left_distance_per_step=float(left_distance_per_step)
                        if left_distance_per_step is not None
                        else None,
                        right_distance_per_step=float(right_distance_per_step)
                        if right_distance_per_step is not None
                        else None,
                    )
                    if not accepted:
                        print("[Policy] move_to rejected (kinematically infeasible)")
                except Exception as exc:
                    print(f"[Policy] Failed to apply scripted move_to: {exc}")
            else:
                print("[Policy] move_to not supported by underlying policy")
        elif cmd_type == "execute_trajectory":
            if hasattr(self.policy, "execute_trajectory"):
                left_positions = payload.get("left_positions")
                right_positions = payload.get("right_positions")
                planner_max_joint_vel = self._clamp_planner_max_joint_vel(
                    payload.get("planner_max_joint_vel")
                )
                if left_positions is None or right_positions is None:
                    print("[Policy] execute_trajectory missing joint positions")
                    return
                try:
                    self.policy.execute_trajectory(
                        left_positions=np.asarray(left_positions, dtype=np.float64),
                        right_positions=np.asarray(right_positions, dtype=np.float64),
                        left_gripper=(
                            float(payload["left_gripper"])
                            if payload.get("left_gripper") is not None
                            else None
                        ),
                        right_gripper=(
                            float(payload["right_gripper"])
                            if payload.get("right_gripper") is not None
                            else None
                        ),
                        max_joint_vel=planner_max_joint_vel,
                        current_left_joint_pos=np.asarray(
                            observation["left_joint_pos"], dtype=np.float64
                        ),
                        current_right_joint_pos=np.asarray(
                            observation["right_joint_pos"], dtype=np.float64
                        ),
                    )
                except Exception as exc:
                    print(f"[Policy] Failed to execute scripted trajectory: {exc}")
            else:
                print("[Policy] execute_trajectory not supported by underlying policy")
        elif cmd_type == "set_target":
            if hasattr(self.policy, "set_target_pose_bimanual"):
                try:
                    self.policy.set_target_pose_bimanual(
                        left_pos=np.asarray(payload["left_pos"]),
                        left_quat_xyzw=np.asarray(payload["left_quat_xyzw"]),
                        right_pos=np.asarray(payload["right_pos"]),
                        right_quat_xyzw=np.asarray(payload["right_quat_xyzw"]),
                    )
                except Exception as exc:
                    print(f"[Policy] Failed to set gizmo target: {exc}")
            else:
                print("[Policy] Scripted set_target not supported by underlying policy")
        else:
            print(f"[Policy] Unknown scripted command type: {cmd_type}")

    def _apply_planned_move_to(
        self,
        left_pos=None,
        right_pos=None,
        left_quat=None,
        right_quat=None,
        left_gripper=None,
        right_gripper=None,
        observation: dict[str, Any] | None = None,
        planner_max_joint_vel: float | None = None,
        planner_solver_speed: str | None = None,
        planner_ik_error_threshold: float | None = None,
        planner_ik_xyz_weight: float | None = None,
        planner_ik_rpy_weight: float | None = None,
        planner_backend: str | None = None,
        curobo_finetune: bool | None = None,
    ) -> bool:
        """Use the selected planner backend to generate a collision-free trajectory."""
        planner_max_joint_vel = self._clamp_planner_max_joint_vel(planner_max_joint_vel)
        planner_solver_speed = _normalize_motion_planner_solver_speed(
            planner_solver_speed
        )
        planner_ik_error_threshold = self._clamp_planner_ik_error_threshold(
            planner_ik_error_threshold
        )
        planner_ik_xyz_weight = self._clamp_planner_ik_xyz_weight(planner_ik_xyz_weight)
        planner_ik_rpy_weight = self._clamp_planner_ik_rpy_weight(planner_ik_rpy_weight)

        state = None
        if observation is not None:
            try:
                state = {
                    "left_joint_pos": np.asarray(
                        observation["left_joint_pos"], dtype=np.float64
                    ),
                    "right_joint_pos": np.asarray(
                        observation["right_joint_pos"], dtype=np.float64
                    ),
                    "left_gripper": float(
                        np.asarray(
                            observation.get("left_gripper_pos", [1.0]), dtype=np.float64
                        ).reshape(-1)[0]
                    ),
                    "right_gripper": float(
                        np.asarray(
                            observation.get("right_gripper_pos", [1.0]),
                            dtype=np.float64,
                        ).reshape(-1)[0]
                    ),
                }
            except Exception:
                state = None
        if state is None and hasattr(self.policy, "get_current_state"):
            state = self.policy.get_current_state()
        if state is None:
            print("[Policy] Not yet initialized, cannot plan")
            return False

        backend = _normalize_motion_planner_backend(
            planner_backend or self._motion_planner_backend
        )
        planner_solver_speed = _normalize_motion_planner_solver_speed(
            planner_solver_speed
        )
        self._motion_planner_backend = backend
        if backend == "rrtconnect":
            planner = self._get_motion_planner(
                position_cost=(
                    float(planner_ik_xyz_weight)
                    if planner_ik_xyz_weight is not None
                    else 1.0
                ),
                orientation_cost=(
                    float(planner_ik_rpy_weight)
                    if planner_ik_rpy_weight is not None
                    else 0.05
                ),
            )
        else:
            planner_key = (
                backend if backend != "curobo" else f"{backend}:{planner_solver_speed}"
            )
            if planner_key not in self._motion_planners:
                shared_port = (
                    self._shared_motion_planner_port
                    if (
                        backend == "curobo"
                        and planner_solver_speed
                        == self._default_motion_planner_solver_speed
                    )
                    else None
                )
                self._motion_planners[planner_key] = _create_motion_planner(
                    backend,
                    shared_port=shared_port,
                    start_server=shared_port is None,
                    solver_speed=planner_solver_speed if backend == "curobo" else None,
                )
            planner = self._motion_planners[planner_key]
            if (
                backend == "curobo"
                and curobo_finetune is not None
                and hasattr(planner, "set_finetune_enabled")
            ):
                planner.set_finetune_enabled(bool(curobo_finetune))

        cur_left = state["left_joint_pos"]
        cur_right = state["right_joint_pos"]
        cur_left_gripper = state["left_gripper"]
        cur_right_gripper = state["right_gripper"]

        has_left = left_pos is not None
        has_right = right_pos is not None
        if has_left and has_right:
            side = "both"
        elif has_left:
            side = "left"
        elif has_right:
            side = "right"
        else:
            print("[Policy] No target specified for motion planner")
            return False

        if hasattr(planner, "set_gripper_qpos"):
            planner.set_gripper_qpos(cur_left_gripper, cur_right_gripper)

        plan_kwargs: dict[str, Any] = dict(
            current_left_jp=cur_left,
            current_right_jp=cur_right,
            target_left_pos=np.asarray(left_pos, dtype=np.float64)
            if left_pos is not None
            else None,
            target_left_quat_xyzw=np.asarray(left_quat, dtype=np.float64)
            if left_quat is not None
            else None,
            target_right_pos=np.asarray(right_pos, dtype=np.float64)
            if right_pos is not None
            else None,
            target_right_quat_xyzw=np.asarray(right_quat, dtype=np.float64)
            if right_quat is not None
            else None,
            side=side,
            left_gripper=cur_left_gripper,
            right_gripper=cur_right_gripper,
        )

        if backend == "rrtconnect":
            if planner_max_joint_vel is not None:
                plan_kwargs["max_joint_vel"] = float(planner_max_joint_vel)
            if planner_ik_error_threshold is not None:
                plan_kwargs["ik_error_threshold"] = float(planner_ik_error_threshold)
            speed_msg = (
                f", max_joint_vel={float(planner_max_joint_vel):.2f} rad/s"
                if planner_max_joint_vel is not None
                else ""
            )
            ik_msg = (
                f", ik_thresh={float(planner_ik_error_threshold):.4f}m"
                if planner_ik_error_threshold is not None
                else ""
            )
            weight_msg = (
                f", xyz_w={float(planner_ik_xyz_weight):.3f}, rpy_w={float(planner_ik_rpy_weight):.3f}"
                if planner_ik_xyz_weight is not None
                or planner_ik_rpy_weight is not None
                else ""
            )
            print(
                f"[Policy] Planning with RRTConnect (side={side}{speed_msg}{ik_msg}{weight_msg})..."
            )
        else:
            finetune_msg = (
                f", finetune={'on' if bool(curobo_finetune) else 'off'}"
                if backend == "curobo" and curobo_finetune is not None
                else ""
            )
            solver_msg = (
                f", solver_speed={planner_solver_speed}" if backend == "curobo" else ""
            )
            print(
                f"[Policy] Planning with {backend} (side={side}{solver_msg}{finetune_msg})..."
            )

        if backend == "curobo":
            plan_kwargs["solver_speed"] = planner_solver_speed
        try:
            result = planner.plan_to_pose(**plan_kwargs)
        except TypeError:
            if backend == "curobo" and "solver_speed" in plan_kwargs:
                plan_kwargs = dict(plan_kwargs)
                plan_kwargs.pop("solver_speed", None)
                result = planner.plan_to_pose(**plan_kwargs)
            else:
                raise

        if result["status"] != "Success":
            detail = result.get("status_detail")
            timing = _format_planner_timing(result)
            suffix = f" | {timing}" if timing else ""
            print(
                f"[Policy] Motion planning failed: {result['status']}"
                + (f" ({detail})" if detail else "")
                + suffix
            )
            return False

        n_steps = len(result["left_positions"])
        timing = _format_planner_timing(result)
        suffix = f" | {timing}" if timing else ""
        print(
            f"[Policy] Motion plan ({backend}): {n_steps} steps, executing trajectory...{suffix}"
        )
        self.policy.execute_trajectory(
            left_positions=result["left_positions"],
            right_positions=result["right_positions"],
            left_gripper=float(left_gripper) if left_gripper is not None else None,
            right_gripper=float(right_gripper) if right_gripper is not None else None,
            max_joint_vel=planner_max_joint_vel,
            current_left_joint_pos=cur_left,
            current_right_joint_pos=cur_right,
        )
        return True

    def reset(self):
        # We only count "external" resets
        self._reset_count += 1
        policy_info = self._policy_reset()
        policy_info["discard_episode"] = (
            self._real_actions_this_episode < MIN_REAL_ACTIONS_TO_RECORD
        )
        print("Counted", self._real_actions_this_episode, "real actions")
        if policy_info["discard_episode"]:
            print(
                "Discarding episodes due to having only",
                self._real_actions_this_episode,
                "real actions.",
            )
        # self._real_actions_this_episode is only reset to zero here, to make
        # sure it's in the correct order with the above line.
        self._real_actions_this_episode = 0
        return policy_info


def _make_detection_arrow(
    length: float = 0.10,
    shaft_radius: float = 0.004,
    head_radius: float = 0.010,
    head_fraction: float = 0.3,
    color: tuple[int, int, int] = (0, 200, 0),
) -> trimesh.Trimesh:
    """Create a pin-style arrow mesh using trimesh. Tip at origin, shaft in +Z."""
    head_length = length * head_fraction
    shaft_length = length - head_length

    # Cone arrowhead: apex at origin, base at z=head_length
    head = trimesh.creation.cone(radius=head_radius, height=head_length, sections=16)
    head.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    head.apply_translation([0, 0, head_length])

    # Cylinder shaft from z=head_length to z=length
    shaft = trimesh.creation.cylinder(
        radius=shaft_radius, height=shaft_length, sections=16
    )
    shaft.apply_translation([0, 0, head_length + shaft_length / 2])

    arrow = trimesh.util.concatenate([shaft, head])
    color_rgba = np.array([color[0], color[1], color[2], 255], dtype=np.uint8)
    arrow.visual.vertex_colors = np.tile(color_rgba, (len(arrow.vertices), 1))
    return arrow


def _direction_to_wxyz(direction: np.ndarray) -> tuple[float, float, float, float]:
    """Return viser wxyz rotating +Z onto *direction*."""
    dir_vec = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(dir_vec))
    if norm <= 1e-9:
        raise ValueError("direction must be non-zero")
    rot = R.align_vectors([dir_vec / norm], [np.array([0.0, 0.0, 1.0])])[0]
    quat_xyzw = rot.as_quat()
    return (
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def _display_rpy_to_quat_xyzw(rpy_deg: list[float] | np.ndarray) -> list[float]:
    """Convert planner/gripper-frame display RPY (deg) to quat xyzw."""
    roll, pitch, yaw = np.asarray(rpy_deg, dtype=np.float64)
    euler_xyz = [-pitch, roll, -yaw - 90.0]
    quat = R.from_euler("xyz", euler_xyz, degrees=True).as_quat()
    return [float(v) for v in quat]


def _quat_xyzw_to_global_rpy_deg(quat_xyzw: list[float] | np.ndarray) -> np.ndarray:
    """Convert quat xyzw to world/global xyz Euler RPY (deg)."""
    return R.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_euler(
        "xyz", degrees=True
    )


def _quat_xyzw_to_display_rpy_deg(quat_xyzw: list[float] | np.ndarray) -> np.ndarray:
    """Convert quat xyzw to planner/gripper-frame display RPY (deg)."""
    ex, ey, ez = R.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_euler(
        "xyz", degrees=True
    )
    disp = np.array([ey, -ex, -ez - 90.0], dtype=np.float64)
    return (disp + 180.0) % 360.0 - 180.0


class ViserUI:
    grasp_site_offset_m = 0.1347
    urdf_gripper_open_m = 0.037524

    def __init__(
        self,
        adapters: PolicyAdapters,
        task_command: str,
        replay_action_horizon: int = 50,
        embodiment_tag: EmbodimentTag | str = "XDOF",
        urdf_path: Path = None,
        ik_tcp_link_names: tuple[str, str] = ("right_tcp", "left_tcp"),
        policy_host: str = "localhost",
        policy_port: int = 8009,
        viser_port: int = 8010,
        viser_web_port: int = 8080,
        action_type: Literal["absolute", "relative"] = "absolute",
        video_enabled: bool = True,
        video_fps: int = 30,
        video_realtime: bool = True,
        video_queue_size: int = 512,
        top_cam_to_world: np.ndarray | None = None,
        show_scripted_controls: bool = False,
        default_motion_planner_backend: MotionPlannerBackend = "curobo",
        default_motion_planner_solver_speed: MotionPlannerSolverSpeed = "fast",
        scripted_use_planner_default: bool = True,
        scripted_planner_max_joint_vel: float = 0.2,
        show_mp_feasible_region: bool = False,
        recompute_mp: bool = False,
        shared_motion_planner_port: int | None = None,
        eval_run_name: str = "",
    ):
        if urdf_path is None:
            from enpire.env.forge.robot.models.station.paths import get_station_urdf

            urdf_path = get_station_urdf()
        self._adapters = adapters
        self._embodiment_tag = embodiment_tag
        self._action_type = action_type
        # Identity until URDF FK overwrites it at the end of __init__.
        self._top_cam_to_world = np.eye(4)
        self._top_cam_to_world_override = top_cam_to_world

        # Latest state from policy
        self._latest_state: dict[str, Any] | None = None
        self._reset_count = 0
        self._state_lock = threading.Lock()
        self._suppress_cmd_update = False
        self._suppress_voice_update = False
        self._last_cmd_local_update_ts = 0.0
        self._cmd_remote_update_cooldown_s = 1.0

        # Cached joint state for Move-To panel (updated in update_ui)
        self._cached_left_jp: np.ndarray | None = None
        self._cached_right_jp: np.ndarray | None = None
        self._cached_left_gripper: float | None = None
        self._cached_right_gripper: float | None = None
        self._moveto_follow_current_side: str | None = None
        self._scripted_use_planner_default = bool(scripted_use_planner_default)
        self._scripted_planner_max_joint_vel_default = min(
            SCRIPTED_PLANNER_MAX_JOINT_VEL_LIMIT,
            max(0.0, float(scripted_planner_max_joint_vel)),
        )
        self._scripted_planner_speed_widget: Any | None = None
        self._scripted_planner_solver_speed_widget: Any | None = None
        self._scripted_planner_ik_threshold_default = 0.005
        self._scripted_planner_ik_xyz_weight_default = 1.0
        self._scripted_planner_ik_rpy_weight_default = 0.05
        self._scripted_planner_ik_threshold_widget: Any | None = None
        self._scripted_planner_ik_xyz_weight_widget: Any | None = None
        self._scripted_planner_ik_rpy_weight_widget: Any | None = None

        # Motion-planner feasible-region preview state.
        self._show_mp_feasible_region = bool(show_mp_feasible_region)
        self._mp_region_visible = bool(show_mp_feasible_region)
        self._mp_region_lock = threading.Lock()
        self._mp_region_status_value = (
            "Waiting for first robot state..."
            if self._show_mp_feasible_region
            else "Disabled"
        )
        self._mp_region_summary_value = "Red=0%, green=100% planner success rate"
        self._mp_region_pending_point_updates: dict[
            str, list[tuple[int, np.ndarray, np.ndarray, Any]]
        ] = {
            "left": [],
            "right": [],
        }
        self._mp_region_handles: dict[str, Any] = {}
        self._mp_region_marker_handles: dict[str, dict[int, Any]] = {
            "left": {},
            "right": {},
        }
        self._mp_region_points: dict[str, dict[int, np.ndarray]] = {
            "left": {},
            "right": {},
        }
        self._mp_region_colors: dict[str, dict[int, np.ndarray]] = {
            "left": {},
            "right": {},
        }
        self._mp_region_trajectories: dict[str, dict[int, Any]] = {
            "left": {},
            "right": {},
        }
        self._mp_region_target_quats: dict[str, np.ndarray] = {}
        self._mp_region_scan_start_left_jp: np.ndarray | None = None
        self._mp_region_scan_start_right_jp: np.ndarray | None = None
        self._mp_region_scan_start_left_gripper: float | None = None
        self._mp_region_scan_start_right_gripper: float | None = None
        self._mp_region_active_cache_key: str | None = None
        self._mp_region_worker: threading.Thread | None = None
        self._mp_region_stop = threading.Event()
        self._mp_region_last_progress_ts = 0.0
        self._mp_region_auto_start = bool(show_mp_feasible_region)
        self._mp_region_force_recompute = bool(recompute_mp)
        self._mp_region_recompute_requested = bool(recompute_mp)
        self._mp_region_status_text: Any | None = None
        self._mp_region_summary_text: Any | None = None
        self._mp_region_exec_traj_handles: dict[str, list[Any]] = {
            "left": [],
            "right": [],
        }
        self._mp_region_point_cloud_size_m = 0.022
        self._mp_region_click_radius_m = 0.006
        self._mp_region_click_opacity = 0.32
        self._default_motion_planner_backend = _normalize_motion_planner_backend(
            default_motion_planner_backend
        )
        self._default_motion_planner_solver_speed = (
            _normalize_motion_planner_solver_speed(default_motion_planner_solver_speed)
        )
        self._shared_motion_planner_port = shared_motion_planner_port

        # Portal IPC: Server to receive state updates from policy
        self._server = portal.Server(viser_port)
        self._server.bind("update_state", self._portal_update_state)

        # Start server in background thread
        server_thread = threading.Thread(target=self._server.start, daemon=True)
        server_thread.start()
        time.sleep(0.1)  # Give server time to start

        # Portal IPC: Client to send commands to policy
        self._client = portal.Client(f"{policy_host}:{policy_port}")

        print(
            f"[Viser] Portal IPC ready (port {viser_port} ← updates, {policy_host}:{policy_port} → commands)",
            flush=True,
        )

        self.server = viser.ViserServer(host="0.0.0.0", port=viser_web_port)
        print(
            f"[Viser] UI server running on http://localhost:{viser_web_port}",
            flush=True,
        )
        self._video_enabled = bool(video_enabled)
        self._video_fps = int(video_fps)
        self._video_realtime = bool(video_realtime)
        self._video_queue_size = int(video_queue_size)
        self._video_dir: Path | None = None
        self._video_writer: Any | None = None
        self._video_path: Path | None = None
        self._video_lock = threading.Lock()
        self._video_queue: queue.Queue[tuple[np.ndarray, int]] | None = None
        self._video_thread: threading.Thread | None = None
        self._video_stop = threading.Event()
        self._video_drop_count = 0
        self._imageio = None
        self._last_video_ts: float | None = None
        self._font = ImageFont.load_default()
        self._title_font = self._load_title_font()
        self._init_video_output()
        atexit.register(self.close)

        # Optional overlay viz frame pusher (non-blocking, best-effort)
        self._overlay_pusher = None
        try:
            from third_party.overlay_viz.pusher import FramePusher

            self._overlay_pusher = FramePusher()
            print("[Viser] Overlay viz pusher enabled (http://localhost:8888)")
        except Exception:
            pass  # overlay viz not available, skip silently

        self._eval_run_name = eval_run_name
        self._last_episode_dir: str = ""
        self._uploaded_episode_dirs: set[str] = set()

        self._recording_label = self.server.gui.add_markdown("# ⏸ Not Recording")

        # Default Record on when eval_run_name is set (i.e. --record-episode).
        _record_default = bool(eval_run_name)
        self._record_cb = self.server.gui.add_checkbox(
            label="Record",
            initial_value=_record_default,
        )

        @self._record_cb.on_update  # type: ignore[attr-defined]
        @safe_call
        def _update_record_enabled(_e: Any) -> None:
            self._client.set_record_enabled(bool(self._record_cb.value)).result()

        # If defaulting to on, push the initial state to the policy wrapper.
        if _record_default:
            try:
                self._client.set_record_enabled(True).result()
            except Exception:
                pass

        with self.server.gui.add_folder("Episode upload"):
            self._upload_auto_cb = self.server.gui.add_checkbox(
                label="Auto-upload episodes",
                initial_value=False,
            )
            self._upload_run_name = self.server.gui.add_text(
                label="Run name",
                initial_value=self._eval_run_name,
            )
            self._upload_dest = self.server.gui.add_text(
                label="rclone dest",
                initial_value="gdrive:eval_logs/",
            )
            self._upload_status = self.server.gui.add_markdown("No episode yet")

        with self.server.gui.add_folder("Policy control"):
            start_btn = self.server.gui.add_button("Start")
            pause_btn = self.server.gui.add_button("Pause")
            home_btn = self.server.gui.add_button("Home")
            step_once_btn = self.server.gui.add_button("Step once")

        @start_btn.on_click
        @safe_call
        def _click_start(_e: Any) -> None:
            self._client.enter_state("start").result()

        @pause_btn.on_click
        @safe_call
        def _click_pause(_e: Any) -> None:
            self._client.enter_state("pause").result()

        @home_btn.on_click
        @safe_call
        def _click_home(_e: Any) -> None:
            # Upload the last saved episode if not already uploaded
            if (
                self._last_episode_dir
                and self._last_episode_dir not in self._uploaded_episode_dirs
            ):
                self._upload_episode(self._last_episode_dir)
            self._moveto_follow_current_side = None
            if hasattr(self, "_clear_moveto_visuals"):
                self._clear_moveto_visuals()
            if hasattr(self, "_hide_mp_region_exec_trajectory"):
                self._hide_mp_region_exec_trajectory()
            self._client.enter_state("home").result()

        @step_once_btn.on_click
        @safe_call
        def _click_step_once(_e: Any) -> None:
            self._client.enter_state("step_once").result()

        self._cmd_input = self.server.gui.add_text(
            label="Task command", initial_value=task_command
        )

        # Send initial task command
        @safe_call
        def _send_initial_task_command():
            self._client.set_task_command(task_command).result()

        _send_initial_task_command()

        @self._cmd_input.on_update  # type: ignore[attr-defined]
        @safe_call
        def _update_cmd(_e: Any) -> None:
            if self._suppress_cmd_update:
                return
            cleaned = (self._cmd_input.value or "").strip()
            if not cleaned:
                print("[ViserUI] Ignoring empty task command")
                return
            self._last_cmd_local_update_ts = time.monotonic()
            self._client.set_task_command(cleaned).result()

        # Voice controls
        with self.server.gui.add_folder("Voice control"):
            self._voice_enabled_cb = self.server.gui.add_checkbox(
                label="Voice enabled",
                initial_value=False,
            )
            voice_once_btn = self.server.gui.add_button("Record once")

        @self._voice_enabled_cb.on_update  # type: ignore[attr-defined]
        @safe_call
        def _update_voice_enabled(_e: Any) -> None:
            if self._suppress_voice_update:
                return
            self._client.set_voice_enabled(bool(self._voice_enabled_cb.value)).result()

        @voice_once_btn.on_click
        @safe_call
        def _record_once(_e: Any) -> None:
            self._client.trigger_voice_once().result()

        # Replay controls
        default_replay_action_horizon = max(1, int(replay_action_horizon))
        umi_replay_action_horizon = 15
        with self.server.gui.add_folder("Replay control"):
            self._replay_control_mode = self.server.gui.add_dropdown(
                label="Control mode",
                options=[
                    "joint_position",
                    "cartesian_position",
                    "delta_joint_position",
                    "delta_ee_pose",
                    "umi_ee_pose",
                ],
                initial_value="joint_position",
            )
            self._replay_dataset_path = self.server.gui.add_text(
                label="Replay dataset path",
                initial_value="",
            )
            self._replay_norm_stats_path = self.server.gui.add_text(
                label="Norm stats path (20D UMI)",
                initial_value="",
            )
            self._replay_action_horizon = self.server.gui.add_text(
                label="Action horizon",
                initial_value=str(default_replay_action_horizon),
            )
            replay_confirm_btn = self.server.gui.add_button("Confirm replay config")

        def _sync_replay_mode_fields() -> None:
            is_umi_mode = self._replay_control_mode.value == "umi_ee_pose"
            current_horizon = self._replay_action_horizon.value.strip()
            if is_umi_mode and current_horizon == str(default_replay_action_horizon):
                self._replay_action_horizon.value = str(umi_replay_action_horizon)
            elif not is_umi_mode and current_horizon == str(umi_replay_action_horizon):
                self._replay_action_horizon.value = str(default_replay_action_horizon)
            self._replay_norm_stats_path.visible = is_umi_mode
            self._replay_action_horizon.visible = is_umi_mode

        _sync_replay_mode_fields()

        @self._replay_control_mode.on_update  # type: ignore[attr-defined]
        @safe_call
        def _update_replay_mode_fields(_e: Any) -> None:
            _sync_replay_mode_fields()

        @replay_confirm_btn.on_click
        @safe_call
        def _apply_replay_config(_e: Any) -> None:
            dataset_path = self._replay_dataset_path.value.strip()
            control_mode = self._replay_control_mode.value
            norm_stats_path = self._replay_norm_stats_path.value.strip()
            action_horizon_raw = self._replay_action_horizon.value.strip()
            if not dataset_path:
                print("[ViserUI] Please enter a replay dataset path before confirming")
                return
            path_obj = Path(dataset_path)
            if not path_obj.exists():
                print(f"[ViserUI] Replay dataset path not found: {path_obj}")
                return
            try:
                action_horizon = int(action_horizon_raw)
                if action_horizon < 1:
                    raise ValueError
            except ValueError:
                print(
                    f"[ViserUI] Action horizon must be a positive integer, got: {action_horizon_raw!r}"
                )
                return
            if norm_stats_path:
                norm_path_obj = Path(norm_stats_path)
                if not norm_path_obj.exists():
                    print(f"[ViserUI] Norm stats path not found: {norm_path_obj}")
                    return
            payload = {
                "dataset_path": dataset_path,
                "control_mode": control_mode,
                "norm_stats_path": norm_stats_path or None,
                "action_horizon": action_horizon,
            }
            self._client.set_replay_config(payload).result()
            print(f"[ViserUI] Replay config sent: {payload}")

        # 6D Gizmo controls.
        # _gizmo_visible: gizmos shown in 3D scene (for both live teleop and Move-To).
        # _gizmo_live_teleop: when True, arm follows gizmo in real-time (~20Hz).
        #                     when False, gizmos are passive waypoint markers.
        # _gizmos_snapped: True only after gizmo positions have been set from FK.
        #                  Prevents sending stale/default positions to the robot.
        self._gizmo_visible = False
        self._gizmo_live_teleop = False
        self._gizmos_snapped = False
        self.ik_left = self.server.scene.add_transform_controls(
            "/ik_target_left",
            scale=0.12,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=False,
        )
        self.ik_right = self.server.scene.add_transform_controls(
            "/ik_target_right",
            scale=0.12,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=False,
        )

        if show_scripted_controls:
            with self.server.gui.add_folder("6D Gizmo Control"):
                gizmo_show_cb = self.server.gui.add_checkbox("Show gizmos", False)
                gizmo_live_cb = self.server.gui.add_checkbox(
                    "Live teleop (arm follows gizmo)", False
                )
                gizmo_snap_btn = self.server.gui.add_button("Snap gizmos to current EE")

                @gizmo_show_cb.on_update  # type: ignore[attr-defined]
                @safe_call
                def _toggle_gizmo_vis(_e: Any) -> None:
                    self._gizmo_visible = bool(gizmo_show_cb.value)
                    self.ik_left.visible = self._gizmo_visible
                    self.ik_right.visible = self._gizmo_visible
                    if self._gizmo_visible:
                        self._snap_gizmos_to_current()
                    if not self._gizmo_visible:
                        # Hide gizmos also disables live teleop
                        self._gizmo_live_teleop = False
                        gizmo_live_cb.value = False

                @gizmo_live_cb.on_update  # type: ignore[attr-defined]
                @safe_call
                def _toggle_live_teleop(_e: Any) -> None:
                    want_live = bool(gizmo_live_cb.value)
                    if want_live:
                        # SAFETY: snap gizmos to FK BEFORE enabling live teleop.
                        # Auto-show gizmos when enabling live teleop
                        if not self._gizmo_visible:
                            self._gizmo_visible = True
                            gizmo_show_cb.value = True
                            self.ik_left.visible = True
                            self.ik_right.visible = True
                        self._snap_gizmos_to_current()
                        if not self._gizmos_snapped:
                            print(
                                "[ViserUI] SAFETY: Cannot enable live teleop — "
                                "gizmo snap failed (no joint data yet). "
                                "Wait for first observation."
                            )
                            gizmo_live_cb.value = False
                            self._gizmo_live_teleop = False
                            return
                        self._gizmo_live_teleop = True
                    else:
                        self._gizmo_live_teleop = False

                @gizmo_snap_btn.on_click
                @safe_call
                def _snap_gizmos(_e: Any) -> None:
                    self._snap_gizmos_to_current()

                # Pose readouts (updated in _update_gizmo_pose_display)
                self._gizmo_pose_texts: dict[str, Any] = {}
                for _disp_side in ("left", "right"):
                    self._gizmo_pose_texts[f"{_disp_side}_pos"] = (
                        self.server.gui.add_text(
                            label=f"{_disp_side.title()} XYZ (m)",
                            initial_value="—",
                            disabled=True,
                        )
                    )
                    self._gizmo_pose_texts[f"{_disp_side}_rpy_global"] = (
                        self.server.gui.add_text(
                            label=f"{_disp_side.title()} RPY (deg, global)",
                            initial_value="—",
                            disabled=True,
                        )
                    )
                    self._gizmo_pose_texts[f"{_disp_side}_rpy_gripper"] = (
                        self.server.gui.add_text(
                            label=f"{_disp_side.title()} RPY (deg, gripper)",
                            initial_value="—",
                            disabled=True,
                        )
                    )

                def _send_gripper(side: str, value: float) -> None:
                    payload = {"side": side, "value": float(value)}
                    self._client.scripted_set_gripper(payload).result()

                def _install_gripper_controls(side: str) -> None:
                    with self.server.gui.add_folder(f"{side.title()} arm"):
                        gripper_slider = self.server.gui.add_slider(
                            "Gripper (0=close, 1=open)",
                            0.0,
                            1.0,
                            0.01,
                            1.0,
                        )
                        open_btn = self.server.gui.add_button("Open gripper")
                        close_btn = self.server.gui.add_button("Close gripper")

                        @gripper_slider.on_update  # type: ignore[attr-defined]
                        @safe_call
                        def _(_e: Any) -> None:
                            _send_gripper(side, float(gripper_slider.value))

                        @open_btn.on_click
                        @safe_call
                        def _(_e: Any) -> None:
                            gripper_slider.value = 1.0
                            _send_gripper(side, 1.0)

                        @close_btn.on_click
                        @safe_call
                        def _(_e: Any) -> None:
                            gripper_slider.value = 0.0
                            _send_gripper(side, 0.0)

                _install_gripper_controls("left")
                _install_gripper_controls("right")

                # ---- Move-To panel (reads target from gizmos instead of sliders) ----

                def _read_gizmo_pose(side: str) -> tuple[list[float], list[float]]:
                    """Read gizmo pose and return (pos, quat_xyzw)."""
                    handle = self.ik_left if side == "left" else self.ik_right
                    pos = list(handle.position)
                    wxyz = np.array(handle.wxyz)
                    quat_xyzw = [
                        float(wxyz[1]),
                        float(wxyz[2]),
                        float(wxyz[3]),
                        float(wxyz[0]),
                    ]
                    return pos, quat_xyzw

                _mt_widgets: dict[str, Any] = {}

                # Trajectory visualization helpers (dots + planned path)
                _mt_dot_radius = 0.012
                _mt_start_sphere = trimesh.creation.icosphere(
                    subdivisions=2, radius=_mt_dot_radius
                )
                _mt_start_vc = np.tile(
                    np.array([255, 50, 50, 255], dtype=np.uint8),
                    (_mt_start_sphere.vertices.shape[0], 1),
                )
                _mt_start_sphere.visual.vertex_colors = _mt_start_vc  # type: ignore[union-attr]
                _mt_end_sphere = trimesh.creation.icosphere(
                    subdivisions=2, radius=_mt_dot_radius
                )
                _mt_end_vc = np.tile(
                    np.array([50, 100, 255, 255], dtype=np.uint8),
                    (_mt_end_sphere.vertices.shape[0], 1),
                )
                _mt_end_sphere.visual.vertex_colors = _mt_end_vc  # type: ignore[union-attr]

                _mt_dots: dict[str, Any] = {}
                for _dot_side in ("left", "right"):
                    _mt_dots[f"{_dot_side}_start"] = self.server.scene.add_mesh_trimesh(
                        f"/moveto/{_dot_side}_start",
                        _mt_start_sphere,
                        position=(0.0, 0.0, 0.0),
                        visible=False,
                    )
                    _mt_dots[f"{_dot_side}_end"] = self.server.scene.add_mesh_trimesh(
                        f"/moveto/{_dot_side}_end",
                        _mt_end_sphere,
                        position=(0.0, 0.0, 0.0),
                        visible=False,
                    )

                def _show_moveto_dots(
                    side: str,
                    start_pos: tuple[float, float, float] | None,
                    end_pos: tuple[float, float, float] | None,
                ) -> None:
                    other = "right" if side == "left" else "left"
                    for suffix in ("_start", "_end"):
                        dot = _mt_dots.get(f"{other}{suffix}")
                        if dot is not None:
                            dot.visible = False
                    if start_pos is not None:
                        dot_start = _mt_dots.get(f"{side}_start")
                        if dot_start is not None:
                            dot_start.position = start_pos
                            dot_start.visible = True
                    if end_pos is not None:
                        dot_end = _mt_dots.get(f"{side}_end")
                        if dot_end is not None:
                            dot_end.position = end_pos
                            dot_end.visible = True

                def _show_moveto_dots_both(
                    left_start,
                    left_end,
                    right_start,
                    right_end,
                ) -> None:
                    for side_key, s_pos, e_pos in [
                        ("left", left_start, left_end),
                        ("right", right_start, right_end),
                    ]:
                        ds = _mt_dots.get(f"{side_key}_start")
                        de = _mt_dots.get(f"{side_key}_end")
                        if ds is not None:
                            ds.position = s_pos if s_pos else (0, 0, 0)
                            ds.visible = s_pos is not None
                        if de is not None:
                            de.position = e_pos if e_pos else (0, 0, 0)
                            de.visible = e_pos is not None

                def _hide_moveto_dots() -> None:
                    for side_key in ("left", "right"):
                        for suffix in ("_start", "_end"):
                            handle = _mt_dots.get(f"{side_key}{suffix}")
                            if handle is not None:
                                handle.visible = False

                _mt_traj_handles: dict[str, list[Any]] = {"left": [], "right": []}
                _MT_MAX_TRAJ_PTS = 40
                _mt_planner_cache: dict[str, Any] = {}

                def _ensure_traj_handles(side: str, count: int) -> None:
                    handles = _mt_traj_handles[side]
                    while len(handles) < count:
                        i = len(handles)
                        t = i / max(count - 1, 1)
                        rgba = np.array(
                            [int(50 + 205 * t), int(220 - 60 * t), 50, 200],
                            dtype=np.uint8,
                        )
                        sph = trimesh.creation.icosphere(subdivisions=2, radius=0.007)
                        sph.visual.vertex_colors = np.tile(
                            rgba, (sph.vertices.shape[0], 1)
                        )  # type: ignore[union-attr]
                        h = self.server.scene.add_mesh_trimesh(
                            f"/moveto/traj/{side}_{i}",
                            sph,
                            position=(0.0, 0.0, 0.0),
                            visible=False,
                        )
                        handles.append(h)

                def _hide_planned_trajectory() -> None:
                    for side_key in ("left", "right"):
                        for h in _mt_traj_handles[side_key]:
                            h.visible = False

                def _get_selected_moveto_side() -> str:
                    side_widget = _mt_widgets.get("_txt_side")
                    if side_widget is None:
                        return "both"
                    return str(side_widget.value)

                def _show_current_tcp_endpoint(side: str | None = None) -> None:
                    side = side or _get_selected_moveto_side()
                    _hide_planned_trajectory()
                    _hide_moveto_dots()
                    with self._state_lock:
                        ljp = self._cached_left_jp
                        rjp = self._cached_right_jp
                    if ljp is None or rjp is None:
                        return
                    if not hasattr(self, "_kinematics"):
                        self._kinematics = YamKinematics()
                    l_pos, _, r_pos, _ = self._kinematics.forward_kinematics(ljp, rjp)
                    if side in ("left", "both"):
                        dot_end = _mt_dots.get("left_end")
                        if dot_end is not None:
                            dot_end.position = tuple(float(v) for v in l_pos)
                            dot_end.visible = True
                    if side in ("right", "both"):
                        dot_end = _mt_dots.get("right_end")
                        if dot_end is not None:
                            dot_end.position = tuple(float(v) for v in r_pos)
                            dot_end.visible = True

                def _clear_moveto_visuals() -> None:
                    _hide_planned_trajectory()
                    _hide_moveto_dots()

                self._get_moveto_selected_side = _get_selected_moveto_side
                self._refresh_moveto_current_endpoint = _show_current_tcp_endpoint
                self._clear_moveto_visuals = _clear_moveto_visuals
                _mt_status: dict[str, Any] = {"widget": None}

                def _set_moveto_status(message: str) -> None:
                    widget = _mt_status.get("widget")
                    if widget is not None:
                        try:
                            widget.value = message
                        except Exception:
                            pass

                def _show_planned_trajectory(
                    side: str, left_jp_traj: np.ndarray, right_jp_traj: np.ndarray
                ) -> None:
                    if not hasattr(self, "_kinematics"):
                        self._kinematics = YamKinematics()
                    n_steps = len(left_jp_traj)
                    if n_steps == 0:
                        _hide_planned_trajectory()
                        return
                    idxs = np.linspace(
                        0, n_steps - 1, min(n_steps, _MT_MAX_TRAJ_PTS), dtype=int
                    )
                    n_vis = len(idxs)
                    left_ee, right_ee = [], []
                    for idx in idxs:
                        l_pos, _, r_pos, _ = self._kinematics.forward_kinematics(
                            left_jp_traj[idx], right_jp_traj[idx]
                        )
                        left_ee.append(l_pos)
                        right_ee.append(r_pos)
                    for arm in ("left", "right"):
                        if side not in (arm, "both"):
                            for h in _mt_traj_handles[arm]:
                                h.visible = False
                            continue
                        ee_list = left_ee if arm == "left" else right_ee
                        _ensure_traj_handles(arm, n_vis)
                        for i in range(n_vis):
                            _mt_traj_handles[arm][i].position = tuple(
                                float(v) for v in ee_list[i]
                            )
                            _mt_traj_handles[arm][i].visible = True
                        for i in range(n_vis, len(_mt_traj_handles[arm])):
                            _mt_traj_handles[arm][i].visible = False
                    print(f"[ViserUI] Showing planned trajectory: {n_vis} waypoints")

                def _plan_and_visualize(
                    side: str,
                    left_jp: np.ndarray,
                    right_jp: np.ndarray,
                    left_pos,
                    right_pos,
                    left_quat,
                    right_quat,
                    left_gripper: float,
                    right_gripper: float,
                    planner_backend: str,
                    curobo_finetune: bool | None = None,
                ) -> None:
                    try:
                        from scipy.spatial.transform import Rotation

                        planner_ik_xyz_weight = (
                            self._get_scripted_planner_ik_xyz_weight()
                        )
                        planner_ik_rpy_weight = (
                            self._get_scripted_planner_ik_rpy_weight()
                        )
                        planner_ik_error_threshold = (
                            self._get_scripted_planner_ik_error_threshold()
                        )
                        planner_solver_speed = self._get_scripted_planner_solver_speed()
                        planner_key = (
                            float(planner_ik_xyz_weight),
                            float(planner_ik_rpy_weight),
                        )
                        if planner_key not in _mt_planner_cache:
                            from enpire.env.forge.experimental.motion_planner import (
                                YamMotionPlanner,
                            )

                            _mt_planner_cache[planner_key] = YamMotionPlanner(
                                position_cost=float(planner_ik_xyz_weight),
                                orientation_cost=float(planner_ik_rpy_weight),
                            )
                        planner = _mt_planner_cache[planner_key]
                        planner.set_gripper_qpos(left_gripper, right_gripper)
                        planner_max_joint_vel = (
                            self._get_scripted_planner_max_joint_vel()
                        )
                        backend = _normalize_motion_planner_backend(planner_backend)
                        backend_key = (
                            backend
                            if backend != "curobo"
                            else f"{backend}:{planner_solver_speed}"
                        )
                        if backend_key not in _mt_planner_cache:
                            shared_port = (
                                self._shared_motion_planner_port
                                if (
                                    backend == "curobo"
                                    and planner_solver_speed
                                    == self._default_motion_planner_solver_speed
                                )
                                else None
                            )
                            _mt_planner_cache[backend_key] = _create_motion_planner(
                                backend,
                                shared_port=shared_port,
                                start_server=shared_port is None,
                                solver_speed=planner_solver_speed
                                if backend == "curobo"
                                else None,
                            )
                        planner = _mt_planner_cache[backend_key]
                        if (
                            backend == "curobo"
                            and curobo_finetune is not None
                            and hasattr(planner, "set_finetune_enabled")
                        ):
                            planner.set_finetune_enabled(bool(curobo_finetune))
                        if hasattr(planner, "set_gripper_qpos"):
                            planner.set_gripper_qpos(left_gripper, right_gripper)
                        result = planner.plan_to_pose(
                            current_left_jp=left_jp,
                            current_right_jp=right_jp,
                            target_left_pos=np.asarray(left_pos, dtype=np.float64)
                            if left_pos is not None
                            else None,
                            target_left_quat_xyzw=np.asarray(
                                left_quat, dtype=np.float64
                            )
                            if left_quat is not None
                            else None,
                            target_right_pos=np.asarray(right_pos, dtype=np.float64)
                            if right_pos is not None
                            else None,
                            target_right_quat_xyzw=np.asarray(
                                right_quat, dtype=np.float64
                            )
                            if right_quat is not None
                            else None,
                            side=side,
                            left_gripper=left_gripper,
                            right_gripper=right_gripper,
                            max_joint_vel=planner_max_joint_vel,
                            ik_error_threshold=planner_ik_error_threshold,
                        )
                        if result["status"] == "Success":
                            timing = _format_planner_timing(result)
                            if timing:
                                print(
                                    f"[ViserUI] Trajectory viz ({backend}) timing: {timing}"
                                )
                            _show_planned_trajectory(
                                side,
                                result["left_positions"],
                                result["right_positions"],
                            )
                            final_left = np.asarray(
                                result["left_positions"][-1], dtype=np.float64
                            )
                            final_right = np.asarray(
                                result["right_positions"][-1], dtype=np.float64
                            )
                            got_l_pos, got_l_q, got_r_pos, got_r_q = (
                                planner._kin.forward_kinematics(final_left, final_right)
                            )
                            arm_metrics: list[tuple[float, float]] = []
                            if (
                                side in ("left", "both")
                                and left_pos is not None
                                and left_quat is not None
                            ):
                                pos_err = float(
                                    np.linalg.norm(
                                        got_l_pos
                                        - np.asarray(left_pos, dtype=np.float64)
                                    )
                                )
                                rot_err = float(
                                    np.degrees(
                                        (
                                            Rotation.from_quat(
                                                np.asarray(left_quat, dtype=np.float64)
                                            ).inv()
                                            * Rotation.from_quat(
                                                np.asarray(got_l_q, dtype=np.float64)
                                            )
                                        ).magnitude()
                                    )
                                )
                                arm_metrics.append((pos_err, rot_err))
                            if (
                                side in ("right", "both")
                                and right_pos is not None
                                and right_quat is not None
                            ):
                                pos_err = float(
                                    np.linalg.norm(
                                        got_r_pos
                                        - np.asarray(right_pos, dtype=np.float64)
                                    )
                                )
                                rot_err = float(
                                    np.degrees(
                                        (
                                            Rotation.from_quat(
                                                np.asarray(right_quat, dtype=np.float64)
                                            ).inv()
                                            * Rotation.from_quat(
                                                np.asarray(got_r_q, dtype=np.float64)
                                            )
                                        ).magnitude()
                                    )
                                )
                                arm_metrics.append((pos_err, rot_err))
                            max_pos_err = max((m[0] for m in arm_metrics), default=0.0)
                            max_rot_err = max((m[1] for m in arm_metrics), default=0.0)
                            _set_moveto_status(
                                f"Planner preview: {result['status']} · steps={len(result['left_positions'])} · "
                                f"pos err={max_pos_err:.4f} m · rot err={max_rot_err:.2f} deg"
                            )
                        else:
                            _hide_planned_trajectory()
                            _set_moveto_status(
                                f"Planner preview failed: {result['status']}"
                            )
                            detail = result.get("status_detail")
                            timing = _format_planner_timing(result)
                            print(
                                f"[ViserUI] Trajectory viz ({backend}): planning returned {result['status']}"
                                + (f" ({detail})" if detail else "")
                                + (f" | {timing}" if timing else "")
                            )
                    except Exception as exc:
                        _hide_planned_trajectory()
                        _set_moveto_status(f"Planner preview failed: {exc}")
                        print(f"[ViserUI] Trajectory viz failed: {exc}")

                # ---- Move-To GUI: gizmo-based + exact text input ----
                with self.server.gui.add_folder("Move To Target"):
                    with self.server.gui.add_folder("Speed"):
                        _mt_widgets["_left_dist"] = self.server.gui.add_slider(
                            "Left arm (m/step)",
                            0.002,
                            0.05,
                            0.001,
                            0.01,
                        )
                        _mt_widgets["_right_dist"] = self.server.gui.add_slider(
                            "Right arm (m/step)",
                            0.002,
                            0.05,
                            0.001,
                            0.01,
                        )
                        self._scripted_planner_speed_widget = (
                            self.server.gui.add_slider(
                                "Planner speed (rad/s)",
                                0.0,
                                SCRIPTED_PLANNER_MAX_JOINT_VEL_LIMIT,
                                0.05,
                                self._scripted_planner_max_joint_vel_default,
                            )
                        )
                        self._scripted_planner_solver_speed_widget = (
                            self.server.gui.add_dropdown(
                                "Solver speed",
                                options=list(_MOTION_PLANNER_SOLVER_SPEEDS),
                                initial_value=self._default_motion_planner_solver_speed,
                            )
                        )
                        self._scripted_planner_ik_threshold_widget = self.server.gui.add_text(
                            label="Planner IK threshold (m)",
                            initial_value=f"{self._scripted_planner_ik_threshold_default:.4f}",
                        )
                        self._scripted_planner_ik_xyz_weight_widget = self.server.gui.add_text(
                            label="Planner XYZ weight",
                            initial_value=f"{self._scripted_planner_ik_xyz_weight_default:.3f}",
                        )
                        self._scripted_planner_ik_rpy_weight_widget = self.server.gui.add_text(
                            label="Planner RPY weight",
                            initial_value=f"{self._scripted_planner_ik_rpy_weight_default:.3f}",
                        )
                    _mt_widgets["_use_planner"] = self.server.gui.add_checkbox(
                        "Use motion planner",
                        self._scripted_use_planner_default,
                    )
                    _mt_widgets["_planner_backend"] = self.server.gui.add_dropdown(
                        "Planner backend",
                        options=list(_MOTION_PLANNER_BACKENDS),
                        initial_value=self._default_motion_planner_backend,
                    )
                    _mt_widgets["_curobo_finetune"] = self.server.gui.add_checkbox(
                        "cuRobo finetune",
                        self._scripted_use_planner_default,
                    )
                    _mt_status["widget"] = self.server.gui.add_text(
                        label="Planner status",
                        initial_value="Planner preview: not run yet",
                    )

                    def _sync_moveto_speed_widgets() -> None:
                        planner_on = bool(_mt_widgets["_use_planner"].value)
                        _mt_widgets["_left_dist"].visible = not planner_on
                        _mt_widgets["_right_dist"].visible = not planner_on
                        if self._scripted_planner_speed_widget is not None:
                            self._scripted_planner_speed_widget.visible = planner_on
                        if self._scripted_planner_solver_speed_widget is not None:
                            self._scripted_planner_solver_speed_widget.visible = (
                                planner_on
                            )
                        if self._scripted_planner_ik_threshold_widget is not None:
                            self._scripted_planner_ik_threshold_widget.visible = (
                                planner_on
                            )
                        if self._scripted_planner_ik_xyz_weight_widget is not None:
                            self._scripted_planner_ik_xyz_weight_widget.visible = (
                                planner_on
                            )
                        if self._scripted_planner_ik_rpy_weight_widget is not None:
                            self._scripted_planner_ik_rpy_weight_widget.visible = (
                                planner_on
                            )

                    _sync_moveto_speed_widgets()

                    @_mt_widgets["_use_planner"].on_update  # type: ignore[attr-defined]
                    @safe_call
                    def _toggle_moveto_speed_widgets(_e: Any) -> None:
                        _sync_moveto_speed_widgets()

                    def _sync_moveto_speed_widgets() -> None:
                        planner_on = bool(_mt_widgets["_use_planner"].value)
                        _mt_widgets["_left_dist"].visible = not planner_on
                        _mt_widgets["_right_dist"].visible = not planner_on
                        if self._scripted_planner_speed_widget is not None:
                            self._scripted_planner_speed_widget.visible = planner_on
                        if self._scripted_planner_solver_speed_widget is not None:
                            self._scripted_planner_solver_speed_widget.visible = (
                                planner_on
                            )
                        if self._scripted_planner_ik_threshold_widget is not None:
                            self._scripted_planner_ik_threshold_widget.visible = (
                                planner_on
                            )
                        if self._scripted_planner_ik_xyz_weight_widget is not None:
                            self._scripted_planner_ik_xyz_weight_widget.visible = (
                                planner_on
                            )
                        if self._scripted_planner_ik_rpy_weight_widget is not None:
                            self._scripted_planner_ik_rpy_weight_widget.visible = (
                                planner_on
                            )

                    _sync_moveto_speed_widgets()

                    @_mt_widgets["_use_planner"].on_update  # type: ignore[attr-defined]
                    @safe_call
                    def _toggle_moveto_speed_widgets(_e: Any) -> None:
                        _sync_moveto_speed_widgets()

                    # -- Exact text inputs for Move-To (compact) --
                    with self.server.gui.add_folder("Exact Input"):
                        _mt_widgets["_txt_side"] = self.server.gui.add_dropdown(
                            "Side",
                            options=["left", "right", "both"],
                            initial_value="left",
                        )
                        _mt_widgets["_txt_pos"] = self.server.gui.add_text(
                            label="Left XYZ (m)",
                            initial_value="0.4979, 0.3114, 0.9195",
                        )
                        _mt_widgets["_txt_rpy"] = self.server.gui.add_text(
                            label="Left RPY (deg, global)",
                            initial_value="-90, 0, -90",
                        )
                        _mt_widgets["_txt_rpy_gripper"] = self.server.gui.add_text(
                            label="Left RPY (deg, gripper)",
                            initial_value="0, 0, 0",
                            disabled=True,
                        )
                        _mt_widgets["_txt_pos_r"] = self.server.gui.add_text(
                            label="Right XYZ (m)",
                            initial_value="0.4979, -0.3163, 0.9183",
                        )
                        _mt_widgets["_txt_rpy_r"] = self.server.gui.add_text(
                            label="Right RPY (deg, global)",
                            initial_value="-90, 0, -90",
                        )
                        _mt_widgets["_txt_rpy_r_gripper"] = self.server.gui.add_text(
                            label="Right RPY (deg, gripper)",
                            initial_value="0, 0, 0",
                            disabled=True,
                        )

                        def _sync_exact_input_visibility() -> None:
                            side = str(_mt_widgets["_txt_side"].value)
                            show_left = side in ("left", "both")
                            show_right = side in ("right", "both")
                            _mt_widgets["_txt_pos"].visible = show_left
                            _mt_widgets["_txt_rpy"].visible = show_left
                            _mt_widgets["_txt_rpy_gripper"].visible = show_left
                            _mt_widgets["_txt_pos_r"].visible = show_right
                            _mt_widgets["_txt_rpy_r"].visible = show_right
                            _mt_widgets["_txt_rpy_r_gripper"].visible = show_right

                        _sync_exact_input_visibility()

                        @_mt_widgets["_txt_side"].on_update  # type: ignore[attr-defined]
                        @safe_call
                        def _toggle_exact_input_visibility(_e: Any) -> None:
                            _sync_exact_input_visibility()

                        def _parse_floats(s: str) -> list[float]:
                            return [float(v.strip()) for v in s.split(",")]

                        def _sync_exact_rpy_readouts() -> None:
                            for global_key, gripper_key in (
                                ("_txt_rpy", "_txt_rpy_gripper"),
                                ("_txt_rpy_r", "_txt_rpy_r_gripper"),
                            ):
                                try:
                                    global_rpy = _parse_floats(
                                        str(_mt_widgets[global_key].value)
                                    )
                                    if len(global_rpy) != 3:
                                        raise ValueError
                                    quat = R.from_euler(
                                        "xyz", global_rpy, degrees=True
                                    ).as_quat()
                                    gripper_rpy = _quat_xyzw_to_display_rpy_deg(quat)
                                    _mt_widgets[gripper_key].value = ", ".join(
                                        f"{v:.1f}" for v in gripper_rpy
                                    )
                                except Exception:
                                    _mt_widgets[gripper_key].value = "—"

                        _sync_exact_rpy_readouts()

                        @_mt_widgets["_txt_rpy"].on_update  # type: ignore[attr-defined]
                        @safe_call
                        def _sync_left_exact_rpy(_e: Any) -> None:
                            _sync_exact_rpy_readouts()

                        @_mt_widgets["_txt_rpy_r"].on_update  # type: ignore[attr-defined]
                        @safe_call
                        def _sync_right_exact_rpy(_e: Any) -> None:
                            _sync_exact_rpy_readouts()

                        def _apply_planner_settings(payload: dict[str, Any]) -> None:
                            payload["planner_max_joint_vel"] = (
                                self._get_scripted_planner_max_joint_vel()
                            )
                            payload["planner_solver_speed"] = (
                                self._get_scripted_planner_solver_speed()
                            )
                            payload["planner_ik_error_threshold"] = (
                                self._get_scripted_planner_ik_error_threshold()
                            )
                            payload["planner_ik_xyz_weight"] = (
                                self._get_scripted_planner_ik_xyz_weight()
                            )
                            payload["planner_ik_rpy_weight"] = (
                                self._get_scripted_planner_ik_rpy_weight()
                            )

                        def _rpy_to_quat_xyzw(rpy_deg: list[float]) -> list[float]:
                            q = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()
                            return [float(v) for v in q]

                        _txt_snap_btn = self.server.gui.add_button(
                            "Fill from current EE"
                        )

                        @_txt_snap_btn.on_click
                        @safe_call
                        def _snap_txt(_e: Any) -> None:

                            with self._state_lock:
                                ljp = self._cached_left_jp
                                rjp = self._cached_right_jp
                            if ljp is None or rjp is None:
                                return
                            if not hasattr(self, "_kinematics"):
                                self._kinematics = YamKinematics()
                            l_pos, l_q, r_pos, r_q = (
                                self._kinematics.forward_kinematics(ljp, rjp)
                            )
                            l_rpy = _quat_xyzw_to_global_rpy_deg(l_q)
                            r_rpy = _quat_xyzw_to_global_rpy_deg(r_q)
                            _mt_widgets["_txt_pos"].value = ", ".join(
                                f"{v:.4f}" for v in l_pos
                            )
                            _mt_widgets["_txt_rpy"].value = ", ".join(
                                f"{v:.1f}" for v in l_rpy
                            )
                            _mt_widgets["_txt_pos_r"].value = ", ".join(
                                f"{v:.4f}" for v in r_pos
                            )
                            _mt_widgets["_txt_rpy_r"].value = ", ".join(
                                f"{v:.1f}" for v in r_rpy
                            )
                            _sync_exact_rpy_readouts()
                            self._moveto_follow_current_side = (
                                _get_selected_moveto_side()
                            )
                            _show_current_tcp_endpoint(self._moveto_follow_current_side)

                        _txt_move_btn = self.server.gui.add_button(
                            "Move to exact target"
                        )

                        @_txt_move_btn.on_click
                        @safe_call
                        def _move_exact(_e: Any) -> None:
                            side = str(_mt_widgets["_txt_side"].value)
                            self._moveto_follow_current_side = None
                            dist_l = float(_mt_widgets["_left_dist"].value)
                            dist_r = float(_mt_widgets["_right_dist"].value)
                            use_planner = bool(_mt_widgets["_use_planner"].value)
                            planner_backend = _normalize_motion_planner_backend(
                                str(_mt_widgets["_planner_backend"].value)
                            )
                            payload: dict[str, Any] = {
                                "use_planner": use_planner,
                                "planner_backend": planner_backend,
                                "curobo_finetune": bool(
                                    _mt_widgets["_curobo_finetune"].value
                                ),
                            }
                            if use_planner:
                                _apply_planner_settings(payload)
                            if side in ("left", "both"):
                                pos = _parse_floats(str(_mt_widgets["_txt_pos"].value))
                                rpy = _parse_floats(str(_mt_widgets["_txt_rpy"].value))
                                quat = _rpy_to_quat_xyzw(rpy)
                                payload["left_pos"] = pos
                                payload["left_quat"] = quat
                                payload["left_distance_per_step"] = dist_l
                                payload["left_gripper"] = 1.0
                            if side == "right":
                                pos = _parse_floats(
                                    str(_mt_widgets["_txt_pos_r"].value)
                                )
                                rpy = _parse_floats(
                                    str(_mt_widgets["_txt_rpy_r"].value)
                                )
                                quat = _rpy_to_quat_xyzw(rpy)
                                payload["right_pos"] = pos
                                payload["right_quat"] = quat
                                payload["right_distance_per_step"] = dist_r
                                payload["right_gripper"] = 1.0
                            if side == "both":
                                pos_r = _parse_floats(
                                    str(_mt_widgets["_txt_pos_r"].value)
                                )
                                rpy_r = _parse_floats(
                                    str(_mt_widgets["_txt_rpy_r"].value)
                                )
                                quat_r = _rpy_to_quat_xyzw(rpy_r)
                                payload["right_pos"] = pos_r
                                payload["right_quat"] = quat_r
                                payload["right_distance_per_step"] = dist_r
                                payload["right_gripper"] = 1.0
                            # Visualize dots
                            with self._state_lock:
                                ljp = self._cached_left_jp
                                rjp = self._cached_right_jp
                            if ljp is not None and rjp is not None:
                                if not hasattr(self, "_kinematics"):
                                    self._kinematics = YamKinematics()
                                l_cur, _, r_cur, _ = (
                                    self._kinematics.forward_kinematics(ljp, rjp)
                                )
                                if side == "both":
                                    _show_moveto_dots_both(
                                        tuple(float(v) for v in l_cur),
                                        tuple(pos),
                                        tuple(float(v) for v in r_cur),
                                        tuple(pos_r),
                                    )
                                else:
                                    cur = l_cur if side == "left" else r_cur
                                    _show_moveto_dots(
                                        side,
                                        tuple(float(v) for v in cur),
                                        tuple(pos if side == "left" else pos),
                                    )
                                if use_planner:
                                    _plan_and_visualize(
                                        side=side,
                                        left_jp=ljp.copy(),
                                        right_jp=rjp.copy(),
                                        left_pos=payload.get("left_pos"),
                                        right_pos=payload.get("right_pos"),
                                        left_quat=payload.get("left_quat"),
                                        right_quat=payload.get("right_quat"),
                                        left_gripper=1.0,
                                        right_gripper=1.0,
                                        planner_backend=planner_backend,
                                        curobo_finetune=payload.get("curobo_finetune"),
                                    )
                                else:
                                    _hide_planned_trajectory()
                            tag = f" [{planner_backend}]" if use_planner else ""
                            print(f"[ViserUI] move_to {side}{tag} -> exact {payload}")
                            self._client.scripted_move_to(payload).result()

                    # -- Gizmo-based Move-To buttons --
                    def _make_move_gizmo_cb(_side: str) -> Any:
                        def _move(_e: Any) -> None:
                            self._moveto_follow_current_side = None
                            pos, quat_xyzw = _read_gizmo_pose(_side)
                            gripper = 1.0  # gripper unchanged during Move-To
                            dist_per_step = float(
                                _mt_widgets["_left_dist"].value
                                if _side == "left"
                                else _mt_widgets["_right_dist"].value
                            )
                            use_planner = bool(_mt_widgets["_use_planner"].value)
                            planner_backend = _normalize_motion_planner_backend(
                                str(_mt_widgets["_planner_backend"].value)
                            )
                            # Show start/end dots
                            with self._state_lock:
                                ljp = self._cached_left_jp
                                rjp = self._cached_right_jp
                            if ljp is not None and rjp is not None:
                                if not hasattr(self, "_kinematics"):
                                    self._kinematics = YamKinematics()
                                l_pos_cur, _, r_pos_cur, _ = (
                                    self._kinematics.forward_kinematics(ljp, rjp)
                                )
                                cur_pos = l_pos_cur if _side == "left" else r_pos_cur
                                _show_moveto_dots(
                                    _side, tuple(float(v) for v in cur_pos), tuple(pos)
                                )
                                if use_planner:
                                    _plan_and_visualize(
                                        side=_side,
                                        left_jp=ljp.copy(),
                                        right_jp=rjp.copy(),
                                        left_pos=pos if _side == "left" else None,
                                        right_pos=pos if _side == "right" else None,
                                        left_quat=quat_xyzw
                                        if _side == "left"
                                        else None,
                                        right_quat=quat_xyzw
                                        if _side == "right"
                                        else None,
                                        left_gripper=gripper
                                        if _side == "left"
                                        else 1.0,
                                        right_gripper=gripper
                                        if _side == "right"
                                        else 1.0,
                                        planner_backend=planner_backend,
                                        curobo_finetune=bool(
                                            _mt_widgets["_curobo_finetune"].value
                                        ),
                                    )
                                else:
                                    _hide_planned_trajectory()
                            payload: dict[str, Any] = {
                                "left_pos" if _side == "left" else "right_pos": pos,
                                "left_quat"
                                if _side == "left"
                                else "right_quat": quat_xyzw,
                                "left_gripper"
                                if _side == "left"
                                else "right_gripper": gripper,
                                "left_distance_per_step"
                                if _side == "left"
                                else "right_distance_per_step": dist_per_step,
                                "use_planner": use_planner,
                                "planner_backend": planner_backend,
                                "curobo_finetune": bool(
                                    _mt_widgets["_curobo_finetune"].value
                                ),
                            }
                            if use_planner:
                                _apply_planner_settings(payload)
                            planner_tag = f" [{planner_backend}]" if use_planner else ""
                            print(
                                f"[ViserUI] move_to {_side}{planner_tag} -> gizmo {pos}"
                            )
                            self._client.scripted_move_to(payload).result()

                        return _move

                    move_left_btn = self.server.gui.add_button("Move LEFT to gizmo")
                    move_left_btn.on_click(safe_call(_make_move_gizmo_cb("left")))
                    move_right_btn = self.server.gui.add_button("Move RIGHT to gizmo")
                    move_right_btn.on_click(safe_call(_make_move_gizmo_cb("right")))

                    move_both_btn = self.server.gui.add_button("Move BOTH to gizmos")

                    @move_both_btn.on_click
                    @safe_call
                    def _move_both(_e: Any) -> None:
                        self._moveto_follow_current_side = None
                        left_pos, left_quat = _read_gizmo_pose("left")
                        right_pos, right_quat = _read_gizmo_pose("right")
                        left_gripper = 1.0
                        right_gripper = 1.0
                        left_dist = float(_mt_widgets["_left_dist"].value)
                        right_dist = float(_mt_widgets["_right_dist"].value)
                        use_planner = bool(_mt_widgets["_use_planner"].value)
                        planner_backend = _normalize_motion_planner_backend(
                            str(_mt_widgets["_planner_backend"].value)
                        )
                        with self._state_lock:
                            ljp = self._cached_left_jp
                            rjp = self._cached_right_jp
                        if ljp is not None and rjp is not None:
                            if not hasattr(self, "_kinematics"):
                                self._kinematics = YamKinematics()
                            l_pos_cur, _, r_pos_cur, _ = (
                                self._kinematics.forward_kinematics(ljp, rjp)
                            )
                            _show_moveto_dots_both(
                                tuple(float(v) for v in l_pos_cur),
                                tuple(left_pos),
                                tuple(float(v) for v in r_pos_cur),
                                tuple(right_pos),
                            )
                            if use_planner:
                                _plan_and_visualize(
                                    side="both",
                                    left_jp=ljp.copy(),
                                    right_jp=rjp.copy(),
                                    left_pos=left_pos,
                                    right_pos=right_pos,
                                    left_quat=left_quat,
                                    right_quat=right_quat,
                                    left_gripper=left_gripper,
                                    right_gripper=right_gripper,
                                    planner_backend=planner_backend,
                                    curobo_finetune=bool(
                                        _mt_widgets["_curobo_finetune"].value
                                    ),
                                )
                            else:
                                _hide_planned_trajectory()
                        payload = {
                            "left_pos": left_pos,
                            "right_pos": right_pos,
                            "left_quat": left_quat,
                            "right_quat": right_quat,
                            "left_gripper": left_gripper,
                            "right_gripper": right_gripper,
                            "left_distance_per_step": left_dist,
                            "right_distance_per_step": right_dist,
                            "use_planner": use_planner,
                            "planner_backend": planner_backend,
                            "curobo_finetune": bool(
                                _mt_widgets["_curobo_finetune"].value
                            ),
                        }
                        if use_planner:
                            _apply_planner_settings(payload)
                        planner_tag = f" [{planner_backend}]" if use_planner else ""
                        print(
                            f"[ViserUI] move_to BOTH{planner_tag} -> gizmos L={left_pos} R={right_pos}"
                        )
                        self._client.scripted_move_to(payload).result()

                # ---- Exact Nudge (text input) ----
                with self.server.gui.add_folder("Exact Nudge"):
                    _nudge_side = self.server.gui.add_dropdown(
                        "Side",
                        options=["left", "right"],
                        initial_value="left",
                    )
                    _nudge_dpos = self.server.gui.add_text(
                        label="Delta XYZ (m)",
                        initial_value="0, 0, 0.01",
                    )
                    _nudge_drpy = self.server.gui.add_text(
                        label="Delta RPY (deg)",
                        initial_value="0, 0, 0",
                    )
                    _nudge_btn = self.server.gui.add_button("Apply nudge")

                    @_nudge_btn.on_click
                    @safe_call
                    def _apply_nudge_txt(_e: Any) -> None:
                        side = str(_nudge_side.value)
                        dpos = [
                            float(v.strip()) for v in str(_nudge_dpos.value).split(",")
                        ]
                        drpy = [
                            float(v.strip()) for v in str(_nudge_drpy.value).split(",")
                        ]
                        # Convert RPY delta to quaternion delta
                        from scipy.spatial.transform import Rotation

                        dquat = (
                            Rotation.from_euler("xyz", drpy, degrees=True)
                            .as_quat()
                            .tolist()
                        )
                        payload = {
                            "side": side,
                            "delta_pos": dpos,
                            "delta_quat_xyzw": dquat,
                        }
                        print(f"[ViserUI] nudge {side} dpos={dpos} drpy={drpy}")
                        self._client.scripted_nudge(payload).result()

        # Overlay state — overlay is opt-in via the folder path input below.
        self.overlay: Any | None = None
        self.overlay_image_popup: Any | None = None

        self._overlay_folder = self.server.gui.add_folder(
            "Overlay Controls", visible=True
        )
        with self._overlay_folder:
            self._overlay_folder_path_input = self.server.gui.add_text(
                label="Raw episode/session folder",
                initial_value="",
                hint=(
                    "Path to a raw episode folder (contains top/left/right "
                    "camera .mp4) or a session folder of such episodes."
                ),
            )
            overlay_load_btn = self.server.gui.add_button("Load overlay from folder")
            overlay_popup_btn = self.server.gui.add_button("Overlay Popup")
            overlay_browser_btn = self.server.gui.add_button(
                "Open overlay in browser (new tab)"
            )
            self._overlay_browser_status = self.server.gui.add_markdown("")

        # Lazy state for the HTTP overlay server (started on first click).
        self._overlay_http_server = None
        self._overlay_http_port = 8081
        self._overlay_http_thread: threading.Thread | None = None

        @overlay_browser_btn.on_click  # type: ignore[attr-defined]
        @safe_call
        def _open_overlay_browser(_e: Any) -> None:
            if self.overlay is None:
                print(
                    "[ViserUI] Overlay not loaded — set the folder path and click "
                    "'Load overlay from folder' first."
                )
                return
            self._ensure_overlay_http_server()
            url = f"http://localhost:{self._overlay_http_port}/"
            print(f"[ViserUI] Overlay browser URL: {url}")
            self._overlay_browser_status.content = (
                f"Open [`{url}`]({url}) in a new browser tab."
            )

        @overlay_load_btn.on_click  # type: ignore[attr-defined]
        @safe_call
        def _load_overlay_from_path(_e: Any) -> None:
            path = (self._overlay_folder_path_input.value or "").strip()
            if not path:
                print("[ViserUI] Overlay load: empty path; ignoring.")
                return
            print(f"[ViserUI] Loading overlay from folder: {path}")
            self.overlay = FrameOverlay(path)
            # Give the background scan a moment to finish (fine for single-
            # episode folders; for session folders it may still be loading).
            time.sleep(0.1)
            total = self.overlay.total_frames()
            print(f"[ViserUI] Overlay: {total} episode(s) loaded from {path}")

        @overlay_popup_btn.on_click
        def _show_popup(_e: Any) -> None:
            if self.overlay is None:
                print(
                    "[ViserUI] No overlay loaded. Type a folder path and click "
                    "'Load overlay from folder', or pick a task from the dropdown."
                )
                return

            # Prefer the 3-camera composite (top | left | right) when available.
            triptych = None
            try:
                triptych = self.overlay.get_triptych_frame()
            except Exception as exc:
                print(f"[ViserUI] triptych load failed: {exc}")
            overlay_frame = triptych if triptych is not None else self._get_overlay_frame()
            if overlay_frame is None:
                print("[ViserUI] ERROR: No overlay frame available!")
                return

            print(
                "[ViserUI] Opening popup with overlay frame shape:", overlay_frame.shape
            )

            with self.server.gui.add_modal(title="Overlay (top | left | right)") as modal:
                cams = {}
                try:
                    cams = self.overlay.get_triptych_cams()
                except Exception as exc:
                    print(f"[ViserUI] triptych cams fetch failed: {exc}")

                # Build one horizontally-stitched composite so Viser renders
                # all three cameras in a single row. Per-panel size kept small
                # so the modal doesn't clip the width.
                panel_w, panel_h = 320, 240
                placeholder = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                panels = []
                for cam_name in ("top", "left", "right"):
                    f = cams.get(cam_name)
                    panels.append(
                        cv2.resize(f, (panel_w, panel_h)) if f is not None else placeholder
                    )
                composite = np.concatenate(panels, axis=1)
                self.overlay_image_popup = self.server.gui.add_image(
                    label="top | left | right (first frames)",
                    image=composite.astype(np.uint8),
                )

                next_frame_btn = self.server.gui.add_button("Next Frame")
                prev_frame_btn = self.server.gui.add_button("Previous Frame")
                close_overlay_btn = self.server.gui.add_button("Close")

                # Initialize with current overlay info
                initial_frame_info = "0/0"
                initial_count = "0"
                if self.overlay is not None:
                    initial_frame_info = f"{self.overlay.get_frame_idx() + 1}/{self.overlay.total_frames()}"
                    initial_count = str(self.overlay.get_current_frame_counter())

                frame_numb_txt = self.server.gui.add_text(
                    label="Frame number", initial_value=initial_frame_info
                )
                frame_count_txt = self.server.gui.add_text(
                    label="Episodes Complete on this frame", initial_value=initial_count
                )

                def _update_overlay_info_and_display():
                    frame_numb_txt.value = f"{self.overlay.get_frame_idx() + 1}/{self.overlay.total_frames()}"
                    frame_count_txt.value = str(
                        self.overlay.get_current_frame_counter()
                    )
                    try:
                        cams_now = self.overlay.get_triptych_cams()
                    except Exception as exc:
                        print(f"[ViserUI] triptych cams refresh failed: {exc}")
                        cams_now = {}
                    new_panels = []
                    for cam_name in ("top", "left", "right"):
                        f = cams_now.get(cam_name)
                        new_panels.append(
                            cv2.resize(f, (panel_w, panel_h)) if f is not None else placeholder
                        )
                    composite_new = np.concatenate(new_panels, axis=1)
                    if self.overlay_image_popup is not None:
                        self.overlay_image_popup.image = composite_new.astype(np.uint8)
                    print(
                        f"[ViserUI] Updated overlay to frame "
                        f"{self.overlay.get_frame_idx() + 1}"
                    )

                @close_overlay_btn.on_click
                def _click_close(_e: Any) -> None:
                    print("[ViserUI] Closing overlay popup")
                    self.overlay_image_popup = None
                    modal.close()

                @next_frame_btn.on_click
                def _click_next_frame(_e: Any) -> None:
                    if not self.overlay:
                        return
                    _ = self.overlay.get_next_frame()
                    _update_overlay_info_and_display()

                @prev_frame_btn.on_click
                def _click_prev_frame(_e: Any) -> None:
                    if self.overlay is None:
                        return
                    _ = self.overlay.get_prev_frame()
                    _update_overlay_info_and_display()

                # Initialize overlay info display
                _update_overlay_info_and_display()

        # Image panes (optional, to avoid duplication if a host already provides them)
        self.gui_images: dict[str, Any] = {}
        self.gui_future_images: dict[str, Any] = {}
        self._last_images: dict[str, np.ndarray] = {}
        self._images_folder = self.server.gui.add_folder("Images")
        with self._images_folder:
            # Current observations (always shown)
            self.gui_images["top"] = self.server.gui.add_image(
                label="top", image=np.zeros((240, 320, 3), dtype=np.uint8)
            )
            self.gui_images["left"] = self.server.gui.add_image(
                label="left", image=np.zeros((240, 320, 3), dtype=np.uint8)
            )
            self.gui_images["right"] = self.server.gui.add_image(
                label="right", image=np.zeros((240, 320, 3), dtype=np.uint8)
            )
            # Predicted images created on-demand when first prediction is available

        # Attention visualization panels (updated when attention data is received)
        self.gui_attention_images: dict[str, Any] = {}
        self._attention_folder = self.server.gui.add_folder("Attention Visualization")
        with self._attention_folder:
            self._attention_info = self.server.gui.add_text(
                label="Info",
                initial_value="Waiting for attention data...",
                disabled=True,
            )
            # Camera attention overlays
            self.gui_attention_images["top"] = self.server.gui.add_image(
                label="Top Camera Attention",
                image=np.zeros((240, 320, 3), dtype=np.uint8),
            )
            self.gui_attention_images["left"] = self.server.gui.add_image(
                label="Left Wrist Attention",
                image=np.zeros((240, 320, 3), dtype=np.uint8),
            )
            self.gui_attention_images["right"] = self.server.gui.add_image(
                label="Right Wrist Attention",
                image=np.zeros((240, 320, 3), dtype=np.uint8),
            )
            # Token attention bar chart
            self.gui_attention_images["token_bar"] = self.server.gui.add_image(
                label="Language Token Attention",
                image=np.zeros((150, 800, 3), dtype=np.uint8),
            )

        # Visualization toggles
        with self.server.gui.add_folder("Visualization"):
            self.ee_as_poses_cb = self.server.gui.add_checkbox(
                "EE as poses (frames)", False
            )

        if self._show_mp_feasible_region:
            with self.server.gui.add_folder("Motion Planner Region"):
                self._mp_region_status_text = self.server.gui.add_text(
                    label="Status",
                    initial_value=self._mp_region_status_value,
                    disabled=True,
                )
                self._mp_region_summary_text = self.server.gui.add_text(
                    label="Summary",
                    initial_value=self._mp_region_summary_value,
                    disabled=True,
                )
                mp_region_visible_cb = self.server.gui.add_checkbox(
                    "Visible",
                    initial_value=True,
                )
                mp_region_recompute_btn = self.server.gui.add_button(
                    "Recompute from current pose"
                )

                @mp_region_visible_cb.on_update  # type: ignore[attr-defined]
                @safe_call
                def _toggle_mp_region_visibility(_e: Any) -> None:
                    self._mp_region_visible = bool(mp_region_visible_cb.value)
                    for handle in self._mp_region_handles.values():
                        handle.visible = self._mp_region_visible
                    for side in ("left", "right"):
                        for handle in self._mp_region_marker_handles[side].values():
                            handle.visible = self._mp_region_visible

                @mp_region_recompute_btn.on_click
                @safe_call
                def _recompute_mp_region(_e: Any) -> None:
                    with self._mp_region_lock:
                        self._mp_region_force_recompute = True
                        self._mp_region_recompute_requested = True
                        self._mp_region_status_value = (
                            "Queued recompute from current pose..."
                        )

        # Object detection visualization
        with self.server.gui.add_folder("Object Detection"):
            self._show_objects_cb = self.server.gui.add_checkbox(
                "Show detected objects", True
            )
            self._detection_query_input = self.server.gui.add_text(
                label="Detection query", initial_value="an object"
            )
            go_to_target_btn = self.server.gui.add_button("Go to target")
            self._offset_x = self.server.gui.add_slider(
                "Offset X (m)",
                min=-0.3,
                max=0.3,
                step=0.01,
                initial_value=0.15,
            )
            self._offset_y = self.server.gui.add_slider(
                "Offset Y (m)",
                min=-0.3,
                max=0.3,
                step=0.01,
                initial_value=0.0,
            )
            self._offset_z = self.server.gui.add_slider(
                "Offset Z (m)",
                min=-0.3,
                max=0.3,
                step=0.01,
                initial_value=-0.02,
            )

        @go_to_target_btn.on_click
        @safe_call
        def _click_go_to_target(_e: Any) -> None:
            if not self._latest_det_world_positions:
                print("[ViserUI] No detections with world positions available")
                return
            # Pick highest-scoring detection
            best = max(
                self._latest_det_world_positions,
                key=lambda d: d["score"],
            )
            pos = best["world_pos"]
            offset = np.array(
                [
                    self._offset_x.value,
                    self._offset_y.value,
                    self._offset_z.value,
                ]
            )
            pos = pos + offset
            print(
                f"[ViserUI] Go to target: {best['label']} at {pos.tolist()} (offset={offset.tolist()})"
            )
            self._client.go_to_target(
                {
                    "right_ee_pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "right_ee_quat_xyzw": [
                        0.7071,
                        -0.7071,
                        0.0,
                        0.0,
                    ],  # down + CW 90° around Z
                }
            ).result()

        # Object detection scene handles (arrows + labels)
        self._object_markers: list[Any] = []
        self._object_labels: list[Any] = []
        # Store latest detections with computed world positions for "Go to target"
        self._latest_det_world_positions: list[dict] = []
        self._station_origin_axis_handles: list[Any] = []

        # URDF visualization and IK/FK robot
        self.ik_tcp_link_names = ik_tcp_link_names
        if not urdf_path.exists():
            print(f"[Viser] WARNING: URDF path does not exist: {urdf_path}")
        self.urdf_vis = ViserUrdf(self.server, urdf_or_path=urdf_path, load_meshes=True)
        self._urdf_joint_limits = dict(self.urdf_vis.get_actuated_joint_limits())  # type: ignore[arg-type]
        self.urdf_joint_names = list(self._urdf_joint_limits.keys())
        if self.urdf_joint_names:
            self.urdf_vis.update_cfg(np.zeros(len(self.urdf_joint_names), dtype=float))

        # Build IK robot regardless of server usage if possible
        urdf_model = URDF.load(str(urdf_path))
        self.ik_robot = pk.Robot.from_urdf(urdf_model)
        self._link_names: list[str] = list(self.ik_robot.links.names)  # type: ignore[attr-defined]
        self._ik_actuated_names: list[str] = list(self.ik_robot.joints.actuated_names)  # type: ignore[attr-defined]
        self._ik_idx_right: list[int] = [
            i
            for i, n in enumerate(self._ik_actuated_names)
            if n.startswith("right_joint")
        ][:6]
        self._ik_idx_left: list[int] = [
            i
            for i, n in enumerate(self._ik_actuated_names)
            if n.startswith("left_joint")
        ][:6]

        if show_scripted_controls:
            self._init_station_origin_axes()

        # EE visualization handles
        self.ee_points_left: list[Any] = []
        self.ee_points_right: list[Any] = []
        self.ee_frames_left: list[Any] = []
        self.ee_frames_right: list[Any] = []

        # cache tcp link indices if available
        self._tcp_link_idx: dict[str, int] = {}
        for name in self.ik_tcp_link_names:
            if hasattr(self, "_link_names") and name in getattr(
                self, "_link_names", []
            ):
                self._tcp_link_idx[name] = self._link_names.index(name)

        # Compute top camera optical → world transform from URDF FK
        from enpire.env.forge.robot.models.station.paths import get_top_camera_frame

        _top_cam_frame = get_top_camera_frame()
        if self._top_cam_to_world_override is not None:
            self._top_cam_to_world = self._top_cam_to_world_override
        elif _top_cam_frame in self._link_names:
            zero_cfg = np.zeros((1, len(self._ik_actuated_names)), dtype=float)
            fk = np.asarray(self.ik_robot.forward_kinematics(zero_cfg))  # (1, L, 7)
            cam_idx = self._link_names.index(_top_cam_frame)
            wxyz_xyz = fk[0, cam_idx]
            quat_wxyz = wxyz_xyz[:4]
            pos = wxyz_xyz[4:7]
            R_mat = R.from_quat(
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            ).as_matrix()
            T_world_cam = np.eye(4)
            T_world_cam[:3, :3] = R_mat
            T_world_cam[:3, 3] = pos
            # URDF link X and Y are flipped vs optical convention (camera is
            # rotated 180° around its optical axis). Negate both columns.
            T_world_cam[:3, 0] *= -1
            T_world_cam[:3, 1] *= -1
            self._top_cam_to_world = T_world_cam
            print(
                f"[Viser] Top camera pos={pos.tolist()}, "
                f"X-axis(right)={T_world_cam[:3, 0].tolist()}, "
                f"Z-axis(look)={T_world_cam[:3, 2].tolist()}"
            )
            print(f"[Viser] T_world_cam:\n{self._top_cam_to_world}")
        else:
            self._top_cam_to_world = np.eye(4)
            print(f"[Viser] WARNING: {_top_cam_frame} not in URDF, using identity")

    def _init_station_origin_axes(self) -> None:
        """Visualize world/station origin axes for scripted-policy debugging."""
        axis_length = 0.18
        shaft_radius = 0.010
        head_radius = 0.022
        label_offset = 0.03
        origin = np.array([0.0, 0.0, 0.0], dtype=float)
        origin_label = self.server.scene.add_label(
            "/station_origin_axes/origin_label",
            text="station origin",
            position=(0.0, 0.0, 0.035),
        )
        self._station_origin_axis_handles.append(origin_label)
        axis_specs = (
            ("x", np.array([1.0, 0.0, 0.0]), (255, 64, 64), "X forward"),
            ("y", np.array([0.0, 1.0, 0.0]), (64, 220, 64), "Y left"),
            ("z", np.array([0.0, 0.0, 1.0]), (80, 140, 255), "Z up"),
        )

        for axis_name, direction, color, label_text in axis_specs:
            arrow_mesh = _make_detection_arrow(
                length=axis_length,
                shaft_radius=shaft_radius,
                head_radius=head_radius,
                head_fraction=0.32,
                color=color,
            )
            handle = self.server.scene.add_mesh_trimesh(
                f"/station_origin_axes/{axis_name}_arrow",
                arrow_mesh,
                position=tuple((origin + direction * axis_length).tolist()),
                wxyz=_direction_to_wxyz(-direction),
            )
            label_pos = origin + direction * (axis_length + label_offset)
            label = self.server.scene.add_label(
                f"/station_origin_axes/{axis_name}_label",
                text=label_text,
                position=(
                    float(label_pos[0]),
                    float(label_pos[1]),
                    float(label_pos[2]),
                ),
            )
            self._station_origin_axis_handles.extend([handle, label])

    def _portal_update_state(self, state: dict[str, Any]) -> bool:
        """Portal RPC handler for state updates from policy."""
        with self._state_lock:
            if state and self._reset_count != state["reset_count"]:
                print("[Viser] Incrementing frame counter")
                if self.overlay is not None:
                    self.overlay.increment_frame_counter()
                self._reset_count = state["reset_count"]
            self._latest_state = state

            # Cache joint positions and gripper for Move-To FK sync
            obs = state.get("observation") if state else None
            if obs is not None:
                if "left_joint_pos" in obs:
                    self._cached_left_jp = np.asarray(
                        obs["left_joint_pos"], dtype=np.float64
                    )
                if "right_joint_pos" in obs:
                    self._cached_right_jp = np.asarray(
                        obs["right_joint_pos"], dtype=np.float64
                    )
                for key in ("left_gripper_pos", "gripper_pos_obs_left"):
                    if key in obs:
                        arr = np.asarray(obs[key], dtype=np.float64)
                        self._cached_left_gripper = float(arr.reshape(-1)[0])
                        break
                for key in ("right_gripper_pos", "gripper_pos_obs_right"):
                    if key in obs:
                        arr = np.asarray(obs[key], dtype=np.float64)
                        self._cached_right_gripper = float(arr.reshape(-1)[0])
                        break

            # Update attention visualization if data is present
            attention_data = state.get("attention")
            if attention_data:
                self._update_attention_visualization(attention_data)

            # Update object detection visualization if data is present
            object_detections = state.get("object_detections")
            if object_detections is not None:
                self._update_object_detections(object_detections)
        return True

    def _update_attention_visualization(self, attention_data: dict[str, Any]) -> None:
        """Update attention visualization panels with new data.

        Expected attention_data format:
        {
            "top": np.ndarray (H, W, 3) - attention overlay for top camera
            "left": np.ndarray (H, W, 3) - attention overlay for left camera
            "right": np.ndarray (H, W, 3) - attention overlay for right camera
            "token_bar": np.ndarray (H, W, 3) - token attention bar chart
            "step_idx": int - current step index
        }
        """
        import time as _time

        # Update camera attention images
        for key in ["top", "left", "right", "token_bar"]:
            if key in attention_data and attention_data[key] is not None:
                img = attention_data[key]
                if isinstance(img, np.ndarray) and key in self.gui_attention_images:
                    self.gui_attention_images[key].image = img.astype(np.uint8)

        # Update info text
        step_idx = attention_data.get("step_idx", "?")
        if hasattr(self, "_attention_info"):
            self._attention_info.value = (
                f"Step {step_idx} | {_time.strftime('%H:%M:%S')}"
            )

    # Table pose from station URDF (play_table visual origin)
    _TABLE_CENTER_XY = (0.6, 0.0)
    _TABLE_Z = 0.745
    _OBJECT_Z = _TABLE_Z + 0.10  # 10 cm above table
    # Default D405 intrinsics at 640x480 (used for 2D→3D fallback)
    _D405_FX = 430.0
    _D405_FY = 430.0
    _D405_CX = 320.0
    _D405_CY = 240.0

    def _box2d_to_world(self, box_2d: list[float]) -> np.ndarray:
        """Convert 2D bbox center to 3D world position on the table plane.

        Uses the camera-to-world transform (from URDF FK) and the known
        table height to backproject the 2D center with a fixed depth
        (camera-to-table distance).
        """
        u = (box_2d[0] + box_2d[2]) / 2.0
        v = (box_2d[1] + box_2d[3]) / 2.0
        # Camera origin in world frame
        cam_z_world = self._top_cam_to_world[2, 3]
        approx_depth = cam_z_world - self._OBJECT_Z  # camera-to-table distance
        if approx_depth <= 0:
            approx_depth = 0.9  # fallback ~0.9m
        # Point in camera optical frame at that depth
        p_cam = np.array(
            [
                (u - self._D405_CX) / self._D405_FX * approx_depth,
                (v - self._D405_CY) / self._D405_FY * approx_depth,
                approx_depth,
                1.0,
            ]
        )
        p_world = (self._top_cam_to_world @ p_cam)[:3]
        # Clamp z to the object height on the table
        p_world[2] = self._OBJECT_Z
        return p_world

    def _update_object_detections(self, detections: list[dict]) -> None:
        """Update 3D object detection arrows and labels in the viser scene."""
        # Remove old handles
        for h in self._object_markers:
            h.remove()
        for h in self._object_labels:
            h.remove()
        self._object_markers.clear()
        self._object_labels.clear()
        self._latest_det_world_positions.clear()

        if not detections:
            return

        has_3d = sum(1 for d in detections if "position_3d" in d)
        print(
            f"[Viser] _update_object_detections: {len(detections)} total, "
            f"{has_3d} with 3D, show_cb={self._show_objects_cb.value}"
        )

        # Compute world positions for all detections (needed for "Go to target" button)
        resolved: list[tuple[dict, np.ndarray, str]] = []
        for i, det in enumerate(detections):
            if "position_3d" in det:
                pos_cam = np.array(det["position_3d"], dtype=float)
                pos_h = np.append(pos_cam, 1.0)
                pos_world = (self._top_cam_to_world @ pos_h)[:3]
                src = "3D"
            elif "box_2d" in det:
                pos_world = self._box2d_to_world(det["box_2d"])
                src = "2D→table"
            else:
                print(
                    f"[Viser]   det[{i}] '{det.get('label')}': SKIPPED (no position_3d or box_2d)"
                )
                continue

            score = det.get("score", 0.0)
            print(
                f"[Viser]   det[{i}] '{det.get('label')}' ({src}): "
                f"score={score:.2f} world={pos_world.tolist()}"
            )
            resolved.append((det, pos_world, src))
            self._latest_det_world_positions.append(
                {
                    "label": det.get("label", "?"),
                    "score": score,
                    "world_pos": pos_world.copy(),
                }
            )

        if not self._show_objects_cb.value:
            return

        arrow_length = 0.10  # 10 cm tall pin arrow

        for i, (det, pos_world, src) in enumerate(resolved):
            score = det.get("score", 0.0)

            if score > 0.5:
                color = (0, 200, 0)
            elif score > 0.2:
                color = (200, 200, 0)
            else:
                color = (200, 0, 0)

            arrow_mesh = _make_detection_arrow(
                length=arrow_length,
                shaft_radius=0.004,
                head_radius=0.010,
                head_fraction=0.3,
                color=color,
            )
            marker = self.server.scene.add_mesh_trimesh(
                f"/objects/arrow_{i}",
                arrow_mesh,
                position=(
                    float(pos_world[0]),
                    float(pos_world[1]),
                    float(pos_world[2]),
                ),
            )
            # Place label above the arrow
            label_pos = pos_world + np.array([0.0, 0.0, arrow_length + 0.02])
            label = self.server.scene.add_label(
                f"/objects/label_{i}",
                text=f"{det['label']} ({score:.2f})",
                position=(
                    float(label_pos[0]),
                    float(label_pos[1]),
                    float(label_pos[2]),
                ),
            )
            self._object_markers.append(marker)
            self._object_labels.append(label)

    def _set_mp_region_status(
        self,
        *,
        status: str | None = None,
        summary: str | None = None,
    ) -> None:
        if not self._show_mp_feasible_region:
            return
        with self._mp_region_lock:
            if status is not None:
                self._mp_region_status_value = status
            if summary is not None:
                self._mp_region_summary_value = summary

    def _queue_mp_region_point_update(
        self,
        side: str,
        point_idx: int,
        point: np.ndarray,
        color: np.ndarray,
        trajectory: Any,
    ) -> None:
        if not self._show_mp_feasible_region:
            return
        with self._mp_region_lock:
            self._mp_region_pending_point_updates[side].append(
                (
                    int(point_idx),
                    np.asarray(point, dtype=np.float32).copy(),
                    np.asarray(color, dtype=np.uint8).copy(),
                    trajectory,
                )
            )

    def _clear_mp_region_visuals(self) -> None:
        if not self._show_mp_feasible_region:
            return
        for handle in self._mp_region_handles.values():
            try:
                handle.remove()
            except Exception:
                pass
        self._mp_region_handles.clear()
        for side in ("left", "right"):
            for handle in self._mp_region_marker_handles[side].values():
                try:
                    handle.remove()
                except Exception:
                    pass
            self._mp_region_marker_handles[side] = {}
            self._mp_region_points[side] = {}
            self._mp_region_colors[side] = {}
            self._mp_region_trajectories[side] = {}
            self._mp_region_pending_point_updates[side] = []
        self._mp_region_scan_start_left_jp = None
        self._mp_region_scan_start_right_jp = None
        self._mp_region_scan_start_left_gripper = None
        self._mp_region_scan_start_right_gripper = None
        self._mp_region_active_cache_key = None
        self._mp_region_target_quats = {}
        self._hide_mp_region_exec_trajectory()

    def _ensure_mp_region_exec_traj_handles(self, side: str, count: int) -> None:
        handles = self._mp_region_exec_traj_handles[side]
        while len(handles) < count:
            i = len(handles)
            t_norm = i / max(count - 1, 1)
            rgba = self._point_color(t_norm)
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.007)
            vc = (np.array(rgba) * 255).astype(np.uint8)
            vis = getattr(sphere, "visual", None)
            if vis is not None and hasattr(vis, "vertex_colors"):
                vis.vertex_colors = np.tile(vc, (sphere.vertices.shape[0], 1))
            handle = self.server.scene.add_mesh_trimesh(
                f"/motion_planner_region_exec/{side}_{i}",
                sphere,
                position=(0.0, 0.0, 0.0),
                visible=False,
            )
            handles.append(handle)

    def _hide_mp_region_exec_trajectory(self) -> None:
        for side in ("left", "right"):
            for handle in self._mp_region_exec_traj_handles[side]:
                handle.visible = False

    def _show_mp_region_exec_trajectory(
        self,
        side: str,
        left_jp_traj: np.ndarray,
        right_jp_traj: np.ndarray,
    ) -> None:
        if not hasattr(self, "_kinematics"):
            self._kinematics = YamKinematics()

        n_steps = len(left_jp_traj)
        if n_steps <= 0:
            self._hide_mp_region_exec_trajectory()
            return

        idxs = np.linspace(0, n_steps - 1, min(n_steps, 40), dtype=int)
        n_vis = len(idxs)
        points_by_side: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        for idx in idxs:
            l_pos, _, r_pos, _ = self._kinematics.forward_kinematics(
                left_jp_traj[idx],
                right_jp_traj[idx],
            )
            points_by_side["left"].append(l_pos)
            points_by_side["right"].append(r_pos)

        for arm in ("left", "right"):
            if arm != side:
                for handle in self._mp_region_exec_traj_handles[arm]:
                    handle.visible = False
                continue
            self._ensure_mp_region_exec_traj_handles(arm, n_vis)
            for i in range(n_vis):
                point = points_by_side[arm][i]
                handle = self._mp_region_exec_traj_handles[arm][i]
                handle.position = (float(point[0]), float(point[1]), float(point[2]))
                handle.visible = True
            for i in range(n_vis, len(self._mp_region_exec_traj_handles[arm])):
                self._mp_region_exec_traj_handles[arm][i].visible = False

    def _get_scripted_planner_max_joint_vel(self) -> float:
        widget = getattr(self, "_scripted_planner_speed_widget", None)
        if widget is not None:
            try:
                return min(
                    SCRIPTED_PLANNER_MAX_JOINT_VEL_LIMIT,
                    max(0.0, float(widget.value)),
                )
            except Exception:
                pass
        return min(
            SCRIPTED_PLANNER_MAX_JOINT_VEL_LIMIT,
            max(0.0, float(self._scripted_planner_max_joint_vel_default)),
        )

    def _get_scripted_planner_solver_speed(self) -> MotionPlannerSolverSpeed:
        widget = getattr(self, "_scripted_planner_solver_speed_widget", None)
        if widget is not None:
            try:
                return _normalize_motion_planner_solver_speed(str(widget.value))
            except Exception:
                pass
        return _normalize_motion_planner_solver_speed(
            self._default_motion_planner_solver_speed
        )

    def _get_scripted_planner_ik_error_threshold(self) -> float:
        widget = getattr(self, "_scripted_planner_ik_threshold_widget", None)
        if widget is not None:
            try:
                return min(0.10, max(0.001, float(widget.value)))
            except Exception:
                pass
        return min(0.10, max(0.001, float(self._scripted_planner_ik_threshold_default)))

    def _get_scripted_planner_ik_xyz_weight(self) -> float:
        widget = getattr(self, "_scripted_planner_ik_xyz_weight_widget", None)
        if widget is not None:
            try:
                return min(20.0, max(0.001, float(widget.value)))
            except Exception:
                pass
        return min(
            20.0, max(0.001, float(self._scripted_planner_ik_xyz_weight_default))
        )

    def _get_scripted_planner_ik_rpy_weight(self) -> float:
        widget = getattr(self, "_scripted_planner_ik_rpy_weight_widget", None)
        if widget is not None:
            try:
                return min(20.0, max(0.0001, float(widget.value)))
            except Exception:
                pass
        return min(
            20.0, max(0.0001, float(self._scripted_planner_ik_rpy_weight_default))
        )

    def _get_scripted_motion_planner(
        self,
        *,
        position_cost: float,
        orientation_cost: float,
        solver_speed: str | None = None,
    ) -> Any:
        if not hasattr(self, "_scripted_motion_planner_cache"):
            self._scripted_motion_planner_cache: dict[
                tuple[float, float, str], Any
            ] = {}
        normalized_solver_speed = _normalize_motion_planner_solver_speed(solver_speed)
        key = (float(position_cost), float(orientation_cost), normalized_solver_speed)
        planner = self._scripted_motion_planner_cache.get(key)
        if planner is None:
            from enpire.env.forge.experimental.motion_planner import YamMotionPlanner

            planner = YamMotionPlanner(
                position_cost=float(position_cost),
                orientation_cost=float(orientation_cost),
            )
            self._scripted_motion_planner_cache[key] = planner
        return planner

    def _retime_joint_trajectory(
        self,
        left_positions: np.ndarray,
        right_positions: np.ndarray,
        *,
        max_joint_vel: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if left_positions.shape[0] <= 1:
            return left_positions, right_positions

        full_path = np.concatenate([left_positions, right_positions], axis=1)
        positions: list[np.ndarray] = []
        dt = 1.0 / 30.0
        safe_vel = max(0.05, float(max_joint_vel))
        for k in range(len(full_path) - 1):
            diff = full_path[k + 1] - full_path[k]
            max_delta = float(np.max(np.abs(diff)))
            seg_time = max(max_delta / safe_vel, dt)
            n_steps = max(1, int(np.ceil(seg_time / dt)))
            for s in range(n_steps):
                t = s / n_steps
                positions.append(full_path[k] + t * diff)
        positions.append(full_path[-1].copy())
        retimed = np.asarray(positions, dtype=np.float32)
        return retimed[:, :6], retimed[:, 6:]

    def _handle_mp_region_point_click(self, side: str, point_idx: int) -> None:
        with self._state_lock:
            left_jp = (
                None if self._cached_left_jp is None else self._cached_left_jp.copy()
            )
            right_jp = (
                None if self._cached_right_jp is None else self._cached_right_jp.copy()
            )
            left_gripper = (
                1.0
                if self._cached_left_gripper is None
                else float(self._cached_left_gripper)
            )
            right_gripper = (
                1.0
                if self._cached_right_gripper is None
                else float(self._cached_right_gripper)
            )

        if left_jp is None or right_jp is None:
            self._set_mp_region_status(
                status="Cannot execute point: robot state unavailable"
            )
            print(
                "[ViserUI] Motion-planner point click ignored: robot state unavailable"
            )
            return

        target_pos = self._mp_region_points[side].get(point_idx)
        if target_pos is None:
            self._set_mp_region_status(status="Clicked point is not ready yet")
            return

        target_quat = self._mp_region_target_quats.get(side)
        if target_quat is None:
            if not hasattr(self, "_kinematics"):
                self._kinematics = YamKinematics()
            cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = (
                self._kinematics.forward_kinematics(
                    left_jp,
                    right_jp,
                )
            )
            _ = cur_l_pos, cur_r_pos
            target_quat = cur_l_q if side == "left" else cur_r_q

        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(-1)[:3].copy()
        target_quat = np.asarray(target_quat, dtype=np.float64).reshape(-1)[:4].copy()
        planner_ik_xyz_weight = self._get_scripted_planner_ik_xyz_weight()
        planner_ik_rpy_weight = self._get_scripted_planner_ik_rpy_weight()
        planner_ik_error_threshold = self._get_scripted_planner_ik_error_threshold()

        cached_traj = self._mp_region_trajectories[side].get(point_idx)
        scan_matches_current = (
            self._mp_region_scan_start_left_jp is not None
            and self._mp_region_scan_start_right_jp is not None
            and np.max(np.abs(left_jp - self._mp_region_scan_start_left_jp)) < 1e-3
            and np.max(np.abs(right_jp - self._mp_region_scan_start_right_jp)) < 1e-3
        )
        if cached_traj is not None and scan_matches_current:
            left_positions = np.asarray(cached_traj["left_positions"], dtype=np.float32)
            right_positions = np.asarray(
                cached_traj["right_positions"], dtype=np.float32
            )
            left_positions, right_positions = self._retime_joint_trajectory(
                left_positions,
                right_positions,
                max_joint_vel=self._get_scripted_planner_max_joint_vel(),
            )
            self._show_mp_region_exec_trajectory(side, left_positions, right_positions)
            payload = {
                "left_positions": left_positions.tolist(),
                "right_positions": right_positions.tolist(),
                "left_gripper": left_gripper,
                "right_gripper": right_gripper,
                "planner_max_joint_vel": self._get_scripted_planner_max_joint_vel(),
            }
            self._client.scripted_execute_trajectory(payload).result()
            self._set_mp_region_status(
                status=(
                    f"Executing cached {side} plan to "
                    f"[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]"
                )
            )
            return

        self._set_mp_region_status(
            status=(
                f"Planning {side} arm to "
                f"[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]"
            )
        )
        print(
            f"[ViserUI] Motion-planner region click: side={side} "
            f"target={target_pos.tolist()}"
        )

        planner = self._get_scripted_motion_planner(
            position_cost=planner_ik_xyz_weight,
            orientation_cost=planner_ik_rpy_weight,
            solver_speed=self._get_scripted_planner_solver_speed(),
        )
        plan_kwargs = dict(
            current_left_jp=left_jp,
            current_right_jp=right_jp,
            side=side,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
            max_joint_vel=self._get_scripted_planner_max_joint_vel(),
            ik_error_threshold=planner_ik_error_threshold,
            verbose=False,
        )
        if side == "left":
            plan_kwargs["target_left_pos"] = target_pos
            plan_kwargs["target_left_quat_xyzw"] = target_quat
        else:
            plan_kwargs["target_right_pos"] = target_pos
            plan_kwargs["target_right_quat_xyzw"] = target_quat

        try:
            result = planner.plan_to_pose(**plan_kwargs)
        except TypeError:
            if "solver_speed" in plan_kwargs:
                plan_kwargs = dict(plan_kwargs)
                plan_kwargs.pop("solver_speed", None)
                result = planner.plan_to_pose(**plan_kwargs)
            else:
                raise
        if result["status"] != "Success":
            self._hide_mp_region_exec_trajectory()
            self._set_mp_region_status(
                status=f"Clicked point is not executable: {result['status']}"
            )
            print(f"[ViserUI] Motion-planner region click failed: {result['status']}")
            return

        self._show_mp_region_exec_trajectory(
            side,
            result["left_positions"],
            result["right_positions"],
        )
        payload = {
            "left_positions": result["left_positions"].tolist(),
            "right_positions": result["right_positions"].tolist(),
            "left_gripper": left_gripper,
            "right_gripper": right_gripper,
            "planner_max_joint_vel": self._get_scripted_planner_max_joint_vel(),
        }
        self._client.scripted_execute_trajectory(payload).result()
        self._set_mp_region_status(
            status=(
                f"Executing {side} plan to "
                f"[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]"
            )
        )

    def _upsert_mp_region_clickable_marker(
        self,
        side: str,
        point_idx: int,
        point: np.ndarray,
        color: np.ndarray,
    ) -> None:
        handle = self._mp_region_marker_handles[side].get(point_idx)
        color_tuple = tuple(
            int(v) for v in np.asarray(color, dtype=np.uint8).reshape(-1)[:3]
        )
        if handle is None:
            marker = self.server.scene.add_icosphere(
                f"/motion_planner_region_click/{side}_{point_idx}",
                radius=self._mp_region_click_radius_m,
                color=color_tuple,
                subdivisions=2,
                opacity=self._mp_region_click_opacity,
                cast_shadow=False,
                receive_shadow=False,
                position=(float(point[0]), float(point[1]), float(point[2])),
                visible=self._mp_region_visible,
            )

            @marker.on_click
            @safe_call
            def _click(
                _event: Any, side: str = side, point_idx: int = point_idx
            ) -> None:
                self._handle_mp_region_point_click(side, point_idx)

            self._mp_region_marker_handles[side][point_idx] = marker
        else:
            handle.position = (float(point[0]), float(point[1]), float(point[2]))
            handle.visible = self._mp_region_visible
            try:
                handle.color = color_tuple
                handle.opacity = self._mp_region_click_opacity
            except Exception:
                pass

        self._mp_region_marker_handles[side][
            point_idx
        ].visible = self._mp_region_visible

    def _format_mp_region_summary(self, results: dict[str, Any], config: Any) -> str:
        if not results:
            return (
                f"{config.positions_per_side} positions/arm, fixed current orientation"
            )
        parts = [
            f"{config.positions_per_side} positions/arm",
            "fixed current orientation",
        ]
        for side in ("left", "right"):
            result = results.get(side)
            if result is None:
                continue
            reachable = int(np.count_nonzero(result.scores > 0.0))
            mean_score = float(np.mean(result.scores)) if len(result.scores) else 0.0
            parts.append(
                f"{side.title()} reachable {reachable}/{len(result.scores)}, mean {mean_score:.2f}"
            )
        return " | ".join(parts)

    def _update_mp_region_progress(self, side: str, done: int, total: int) -> None:
        if not self._show_mp_feasible_region or total <= 0:
            return
        now = time.time()
        with self._mp_region_lock:
            if done < total and (now - self._mp_region_last_progress_ts) < 0.25:
                return
            self._mp_region_last_progress_ts = now
            pct = 100.0 * float(done) / float(total)
            self._mp_region_status_value = (
                f"Sampling {side.upper()} region: {done}/{total} ({pct:.0f}%)"
            )

    def _maybe_start_mp_region_compute(self) -> None:
        if not self._show_mp_feasible_region:
            return

        with self._mp_region_lock:
            worker = self._mp_region_worker
            force_recompute = self._mp_region_force_recompute
            should_start = (
                self._mp_region_auto_start
                or self._mp_region_recompute_requested
                or self._mp_region_force_recompute
            )

        if worker is not None and worker.is_alive():
            return
        if not should_start:
            return

        with self._state_lock:
            left_jp = (
                None if self._cached_left_jp is None else self._cached_left_jp.copy()
            )
            right_jp = (
                None if self._cached_right_jp is None else self._cached_right_jp.copy()
            )
            left_gripper = (
                1.0 if self._cached_left_gripper is None else self._cached_left_gripper
            )
            right_gripper = (
                1.0
                if self._cached_right_gripper is None
                else self._cached_right_gripper
            )

        if left_jp is None or right_jp is None:
            self._set_mp_region_status(status="Waiting for first robot state...")
            return

        self._clear_mp_region_visuals()
        self._set_mp_region_status(
            status=(
                "Initializing motion-planner region scan..."
                if force_recompute
                else "Loading cached motion-planner region or computing if missing..."
            )
        )
        with self._mp_region_lock:
            self._mp_region_auto_start = False
            self._mp_region_force_recompute = False
            self._mp_region_recompute_requested = False
            self._mp_region_stop.clear()
            self._mp_region_last_progress_ts = 0.0
            worker = threading.Thread(
                target=self._mp_region_worker_main,
                args=(
                    left_jp,
                    right_jp,
                    float(left_gripper),
                    float(right_gripper),
                    bool(force_recompute),
                ),
                daemon=True,
            )
            self._mp_region_worker = worker
        worker.start()

    def _mp_region_worker_main(
        self,
        start_left_jp: np.ndarray,
        start_right_jp: np.ndarray,
        left_gripper: float,
        right_gripper: float,
        force_recompute: bool,
    ) -> None:
        try:
            from enpire.env.forge.experimental.motion_planner import YamMotionPlanner
            from enpire.env.forge.experimental.motion_planner_feasible_region import (
                DEFAULT_FEASIBLE_REGION_CONFIG,
                SamplingInterrupted,
                compute_feasible_region_cache_key,
                compute_side_feasible_region,
                load_feasible_region_cache,
                save_feasible_region_cache,
                scores_to_rgb,
                side_result_from_cache,
            )

            config = DEFAULT_FEASIBLE_REGION_CONFIG
            kin = YamKinematics()
            cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = kin.forward_kinematics(
                start_left_jp,
                start_right_jp,
            )
            _ = cur_l_pos, cur_r_pos
            with self._mp_region_lock:
                self._mp_region_target_quats["left"] = np.asarray(
                    cur_l_q, dtype=np.float64
                ).copy()
                self._mp_region_target_quats["right"] = np.asarray(
                    cur_r_q, dtype=np.float64
                ).copy()
                self._mp_region_scan_start_left_jp = np.asarray(
                    start_left_jp, dtype=np.float64
                ).copy()
                self._mp_region_scan_start_right_jp = np.asarray(
                    start_right_jp, dtype=np.float64
                ).copy()
                self._mp_region_scan_start_left_gripper = float(left_gripper)
                self._mp_region_scan_start_right_gripper = float(right_gripper)
            cache_key = compute_feasible_region_cache_key(
                current_left_jp=start_left_jp,
                current_right_jp=start_right_jp,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                left_target_quat_xyzw=cur_l_q,
                right_target_quat_xyzw=cur_r_q,
                config=config,
            )
            with self._mp_region_lock:
                self._mp_region_active_cache_key = cache_key

            if not force_recompute:
                cached = load_feasible_region_cache(cache_key)
                if cached is not None:
                    results: dict[str, Any] = {}
                    target_quats = cached.get("target_quats", {})
                    start_state = cached.get("start_state", {})
                    with self._mp_region_lock:
                        if "left" in target_quats:
                            self._mp_region_target_quats["left"] = np.asarray(
                                target_quats["left"], dtype=np.float64
                            ).copy()
                        if "right" in target_quats:
                            self._mp_region_target_quats["right"] = np.asarray(
                                target_quats["right"], dtype=np.float64
                            ).copy()
                        if "left_joint_pos" in start_state:
                            self._mp_region_scan_start_left_jp = np.asarray(
                                start_state["left_joint_pos"], dtype=np.float64
                            ).copy()
                        if "right_joint_pos" in start_state:
                            self._mp_region_scan_start_right_jp = np.asarray(
                                start_state["right_joint_pos"], dtype=np.float64
                            ).copy()
                        if "left_gripper" in start_state:
                            self._mp_region_scan_start_left_gripper = float(
                                start_state["left_gripper"]
                            )
                        if "right_gripper" in start_state:
                            self._mp_region_scan_start_right_gripper = float(
                                start_state["right_gripper"]
                            )
                    for side in ("left", "right"):
                        side_payload = cached.get("sides", {}).get(side)
                        if side_payload is None:
                            continue
                        result = side_result_from_cache(side, side_payload)
                        results[side] = result
                        side_colors = scores_to_rgb(result.scores)
                        for point_idx in range(len(result.points)):
                            self._queue_mp_region_point_update(
                                side,
                                point_idx,
                                result.points[point_idx],
                                side_colors[point_idx],
                                result.trajectories[point_idx],
                            )
                    self._set_mp_region_status(
                        status=f"Loaded cached feasible region ({cache_key})",
                        summary=self._format_mp_region_summary(results, config),
                    )
                    return

            planner = YamMotionPlanner()
            self._set_mp_region_status(
                status="Sampling planner feasible region...",
                summary=self._format_mp_region_summary({}, config),
            )
            start_time = time.time()
            results: dict[str, Any] = {}

            for side, target_quat_xyzw in (
                ("left", cur_l_q),
                ("right", cur_r_q),
            ):
                if self._mp_region_stop.is_set():
                    raise SamplingInterrupted("Feasible-region scan cancelled")

                result = compute_side_feasible_region(
                    planner,
                    side=typing.cast(Literal["left", "right"], side),
                    current_left_jp=start_left_jp,
                    current_right_jp=start_right_jp,
                    target_quat_xyzw=target_quat_xyzw,
                    left_gripper=left_gripper,
                    right_gripper=right_gripper,
                    config=config,
                    progress_callback=lambda done, total, side=side: (
                        self._update_mp_region_progress(side, done, total)
                    ),
                    point_result_callback=lambda point_idx, point, score, trajectory, side=side: (
                        self._queue_mp_region_point_update(
                            side,
                            point_idx,
                            point,
                            scores_to_rgb(np.array([score], dtype=np.float32))[0],
                            trajectory,
                        )
                    ),
                    stop_event=self._mp_region_stop,
                )
                results[side] = result
                self._set_mp_region_status(
                    summary=self._format_mp_region_summary(results, config)
                )

            save_feasible_region_cache(
                cache_key=cache_key,
                config=config,
                start_left_jp=start_left_jp,
                start_right_jp=start_right_jp,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                left_target_quat_xyzw=cur_l_q,
                right_target_quat_xyzw=cur_r_q,
                results=results,
            )
            elapsed_s = time.time() - start_time
            self._set_mp_region_status(
                status=f"Ready in {elapsed_s:.1f}s (cache saved: {cache_key})",
                summary=self._format_mp_region_summary(results, config),
            )
        except SamplingInterrupted:
            self._set_mp_region_status(status="Feasible-region scan stopped")
        except Exception as exc:
            import traceback

            print(f"[ViserUI] Motion-planner feasible-region scan failed: {exc}")
            traceback.print_exc()
            self._set_mp_region_status(status=f"Feasible-region scan failed: {exc}")
        finally:
            with self._mp_region_lock:
                self._mp_region_worker = None

    def _sync_mp_region_visuals(self) -> None:
        if not self._show_mp_feasible_region:
            return

        with self._mp_region_lock:
            pending = self._mp_region_pending_point_updates
            self._mp_region_pending_point_updates = {"left": [], "right": []}
            status_value = self._mp_region_status_value
            summary_value = self._mp_region_summary_value

        if self._mp_region_status_text is not None:
            self._mp_region_status_text.value = status_value
        if self._mp_region_summary_text is not None:
            self._mp_region_summary_text.value = summary_value

        for side, updates in pending.items():
            for point_idx, point, color, trajectory in updates:
                self._mp_region_points[side][point_idx] = np.asarray(
                    point, dtype=np.float32
                ).copy()
                self._mp_region_colors[side][point_idx] = np.asarray(
                    color, dtype=np.uint8
                ).copy()
                self._mp_region_trajectories[side][point_idx] = trajectory
                self._upsert_mp_region_clickable_marker(
                    side,
                    point_idx,
                    self._mp_region_points[side][point_idx],
                    self._mp_region_colors[side][point_idx],
                )

        for side in ("left", "right"):
            point_records = self._mp_region_points[side]
            if not point_records:
                continue
            point_indices = sorted(point_records.keys())
            points = np.stack([point_records[idx] for idx in point_indices]).astype(
                np.float32
            )
            colors = np.stack(
                [self._mp_region_colors[side][idx] for idx in point_indices]
            ).astype(np.uint8)
            handle = self._mp_region_handles.get(side)
            if handle is None:
                point_shape = "circle" if side == "left" else "diamond"
                self._mp_region_handles[side] = self.server.scene.add_point_cloud(
                    f"/motion_planner_region/{side}",
                    points=points,
                    colors=colors,
                    point_size=self._mp_region_point_cloud_size_m,
                    point_shape=point_shape,
                    visible=self._mp_region_visible,
                )
            else:
                handle.points = points
                handle.colors = colors
                handle.visible = self._mp_region_visible

    def _upload_episode(self, ep_dir: str) -> None:
        """Upload an episode directory via rclone in a background thread."""
        dest = self._upload_dest.value.strip()
        if not dest:
            self._upload_status.content = "**No upload destination set**"
            return
        run_name = self._upload_run_name.value.strip()
        if not run_name:
            run_name = Path(ep_dir).parent.name
        ep_name = Path(ep_dir).name
        full_dest = f"{dest.rstrip('/')}/{run_name}/{ep_name}"
        self._upload_status.content = f"⏳ Uploading `{ep_name}`..."

        def _do_upload():
            try:
                import subprocess as _sp

                result = _sp.run(
                    ["rclone", "copy", ep_dir, full_dest],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode == 0:
                    self._uploaded_episode_dirs.add(ep_dir)
                    self._upload_status.content = (
                        f"Uploaded `{ep_name}` to `{run_name}/`"
                    )
                else:
                    err = result.stderr.strip()[:200]
                    self._upload_status.content = f"**Upload failed**: {err}"
            except FileNotFoundError:
                self._upload_status.content = (
                    "**rclone not found**. Install: https://rclone.org/install/ "
                    "then run `rclone config` to set up a Google Drive remote."
                )
            except Exception as exc:
                self._upload_status.content = f"**Upload error**: {exc}"

        threading.Thread(target=_do_upload, daemon=True).start()

    @safe_call
    def update_ui(self):
        """Update visualization from latest policy state."""
        with self._state_lock:
            state = self._latest_state
            self._latest_state = None  # Clear after reading

        self._sync_mp_region_visuals()
        self._maybe_start_mp_region_compute()

        if not state:
            return

        is_recording = state.get("is_recording", False)
        if is_recording:
            self._recording_label.content = "# 🟢 Recording"
        else:
            self._recording_label.content = "# ⏸ Not Recording"

        last_ep = state.get("last_episode_dir")
        if isinstance(last_ep, str) and last_ep and last_ep != self._last_episode_dir:
            self._last_episode_dir = last_ep
            ep_name = Path(last_ep).name
            if self._upload_auto_cb.value:
                self._upload_episode(last_ep)
            else:
                self._upload_status.content = (
                    f"Saved: `{ep_name}` — press **Home** to upload"
                )

        voice_enabled = state.get("voice_enabled")
        if isinstance(voice_enabled, bool):
            if self._voice_enabled_cb.value != voice_enabled:
                self._suppress_voice_update = True
                try:
                    self._voice_enabled_cb.value = voice_enabled
                finally:
                    self._suppress_voice_update = False

        task_command = state.get("task_command")
        if isinstance(task_command, str) and task_command:
            if (
                time.monotonic() - self._last_cmd_local_update_ts
            ) < self._cmd_remote_update_cooldown_s:
                # Avoid fighting user typing; ignore remote updates for a short window.
                pass
            elif self._cmd_input.value != task_command:
                self._suppress_cmd_update = True
                try:
                    self._cmd_input.value = task_command
                finally:
                    self._suppress_cmd_update = False

        observation = state.get("observation")
        if observation is None:
            return
        info = state.get("info", {})
        image, proprio = self._adapters.map_observation(observation)  # type: ignore[misc]
        self._update_images(image)
        action_chunk = info.get("action_chunk")
        if action_chunk:
            if self._action_type == "relative":
                action_chunk = self._convert_relative_chunk(action_chunk, observation)
            if action_chunk.get("_delta_ee"):
                action_chunk = self._convert_delta_ee_chunk_to_absolute(
                    action_chunk, observation
                )
            self._update_prediction(action_chunk)
            # Display future state predictions if available
            future_predictions = action_chunk.get("future_image_predictions")
            if future_predictions:
                self._update_future_images(future_predictions)
        self._update_urdf_cfg_from_proprio(proprio)
        if self._moveto_follow_current_side and hasattr(
            self, "_refresh_moveto_current_endpoint"
        ):
            self._refresh_moveto_current_endpoint(self._moveto_follow_current_side)
        self._poll_and_send_gizmo_targets()
        self._update_gizmo_pose_display()
        self._sync_mp_region_visuals()

    # ----------------------- internal methods for viz ---------------------------------------
    def _convert_relative_chunk(
        self, action_chunk: dict[str, Any], observation: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert relative joint deltas to absolute using current observation."""

        def _accumulate(seq: np.ndarray, base: np.ndarray) -> np.ndarray:
            if seq.ndim == 3:
                return np.cumsum(seq, axis=1) + base[None, None, :]
            return np.cumsum(seq, axis=0) + base[None, :]

        current_left = np.asarray(
            observation.get("left_joint_pos", np.zeros(6)), dtype=float
        )[:6]
        current_right = np.asarray(
            observation.get("right_joint_pos", np.zeros(6)), dtype=float
        )[:6]

        converted: dict[str, Any] = {}
        for key, value in action_chunk.items():
            seq = np.asarray(value, dtype=float)
            if key == "left_joint_pos":
                converted[key] = _accumulate(seq, current_left)
            elif key == "right_joint_pos":
                converted[key] = _accumulate(seq, current_right)
            else:
                # Leave gripper and unknown keys as-is.
                converted[key] = value
        return converted

    def _convert_delta_ee_chunk_to_absolute(
        self, action_chunk: dict[str, Any], observation: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert a delta EE action chunk to absolute EE poses for visualization."""
        if not hasattr(self, "_kinematics"):
            self._kinematics = YamKinematics()

        cur_left_jp = np.asarray(
            observation.get("left_joint_pos", np.zeros(6)), dtype=np.float64
        )[:6]
        cur_right_jp = np.asarray(
            observation.get("right_joint_pos", np.zeros(6)), dtype=np.float64
        )[:6]

        cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = self._kinematics.forward_kinematics(
            cur_left_jp, cur_right_jp
        )

        left_dpos = np.asarray(action_chunk["left_ee_pos"], dtype=np.float64)
        left_dquat = np.asarray(action_chunk["left_ee_quat_xyzw"], dtype=np.float64)
        right_dpos = np.asarray(action_chunk["right_ee_pos"], dtype=np.float64)
        right_dquat = np.asarray(action_chunk["right_ee_quat_xyzw"], dtype=np.float64)

        H = left_dpos.shape[0]
        abs_left_pos = np.zeros_like(left_dpos)
        abs_left_quat = np.zeros_like(left_dquat)
        abs_right_pos = np.zeros_like(right_dpos)
        abs_right_quat = np.zeros_like(right_dquat)

        lp, lq = cur_l_pos.copy(), cur_l_q.copy()
        rp, rq = cur_r_pos.copy(), cur_r_q.copy()
        for i in range(H):
            lp = lp + left_dpos[i]
            lq = (R.from_quat(left_dquat[i]) * R.from_quat(lq)).as_quat()
            rp = rp + right_dpos[i]
            rq = (R.from_quat(right_dquat[i]) * R.from_quat(rq)).as_quat()
            abs_left_pos[i] = lp
            abs_left_quat[i] = lq
            abs_right_pos[i] = rp
            abs_right_quat[i] = rq

        converted = dict(action_chunk)
        converted["left_ee_pos"] = abs_left_pos
        converted["left_ee_quat_xyzw"] = abs_left_quat
        converted["right_ee_pos"] = abs_right_pos
        converted["right_ee_quat_xyzw"] = abs_right_quat
        converted.pop("_delta_ee", None)
        return converted

    @safe_call
    def _get_overlay_frame(self) -> np.ndarray | None:
        """Get current overlay frame if available."""
        if self.overlay is None:
            return None
        return self.overlay.get_frame()

    def _ensure_overlay_http_server(self) -> None:
        """Start the stdlib HTTP server that serves the 3-camera overlay.

        Idempotent — does nothing if already running. Listens on
        ``self._overlay_http_port`` (default 8081). Routes:
          * ``/``               → HTML page with 3 <img> refreshing every 1s.
          * ``/cam/<name>.jpg`` → current first frame for top/left/right.
        """
        if self._overlay_http_server is not None:
            return

        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        ui_self = self  # capture for the handler closure

        class _OverlayHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                # silence access logs to stderr
                return

            def _send(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/" or self.path.startswith("/?"):
                    html = (
                        "<!doctype html><html><head><meta charset='utf-8'>"
                        "<title>YAM overlay (top | left | right)</title>"
                        "<style>body{margin:0;background:#111;color:#ccc;"
                        "font-family:sans-serif}"
                        ".row{display:flex;gap:8px;padding:8px;"
                        "justify-content:center;align-items:flex-start}"
                        ".panel{text-align:center}"
                        ".panel img{display:block;max-width:32vw;height:auto;"
                        "background:#000}"
                        "</style></head><body>"
                        "<div class='row'>"
                        "<div class='panel'><div>top</div>"
                        "<img id='top' src='/cam/top.jpg'></div>"
                        "<div class='panel'><div>left</div>"
                        "<img id='left' src='/cam/left.jpg'></div>"
                        "<div class='panel'><div>right</div>"
                        "<img id='right' src='/cam/right.jpg'></div>"
                        "</div>"
                        "<script>"
                        "setInterval(()=>{"
                        "for(const id of ['top','left','right']){"
                        "const e=document.getElementById(id);"
                        "e.src='/cam/'+id+'.jpg?'+Date.now();}"
                        "},250);"
                        "</script></body></html>"
                    ).encode("utf-8")
                    self._send(200, html, "text/html; charset=utf-8")
                    return
                if self.path.startswith("/cam/"):
                    name = self.path[len("/cam/"):].split("?", 1)[0]
                    short = name.rsplit(".", 1)[0]
                    if short not in ("top", "left", "right"):
                        self._send(404, b"not found", "text/plain")
                        return

                    # Live image from the current observation (BGR if from
                    # OpenCV, RGB if from the policy — we normalise to BGR
                    # for cv2.imencode).
                    live = ui_self._last_images.get(short)
                    if live is not None:
                        live = np.ascontiguousarray(live)
                        # The Viser sidebar stores frames as RGB uint8. cv2
                        # wants BGR for imencode, so swap.
                        if live.ndim == 3 and live.shape[2] == 3:
                            live = live[:, :, ::-1]

                    # Reference first-frame from the loaded recording.
                    ov = ui_self.overlay
                    ref_cams = ov.get_triptych_cams() if ov is not None else {}
                    ref = ref_cams.get(short)

                    # Compose: blend live and reference at 50/50. Fall back to
                    # whichever one is available if the other isn't.
                    composite: np.ndarray | None = None
                    if live is not None and ref is not None:
                        h = min(live.shape[0], ref.shape[0])
                        w = min(live.shape[1], ref.shape[1])
                        l = cv2.resize(live, (w, h))
                        r = cv2.resize(ref, (w, h))
                        composite = cv2.addWeighted(l, 0.5, r, 0.5, 0.0)
                    elif live is not None:
                        composite = live
                    elif ref is not None:
                        composite = ref

                    if composite is None:
                        composite = np.zeros((240, 320, 3), dtype=np.uint8)

                    ok, enc = cv2.imencode(
                        ".jpg",
                        composite,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                    )
                    if not ok:
                        self._send(500, b"encode failed", "text/plain")
                        return
                    self._send(200, bytes(enc), "image/jpeg")
                    return
                self._send(404, b"not found", "text/plain")

        srv = ThreadingHTTPServer(("0.0.0.0", self._overlay_http_port), _OverlayHandler)
        self._overlay_http_server = srv
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        self._overlay_http_thread = t
        print(
            f"[Viser] Overlay HTTP server listening on "
            f"http://localhost:{self._overlay_http_port}/"
        )

    def _update_images(self, image: dict[str, Any]):
        # Hacky way to handle key renaming. This should be done when
        # the obs is passed by the policy and its adapters
        def _normalize_view_key(raw_key: str) -> str | None:
            if raw_key in ("top", "left", "right"):
                return raw_key
            if "top" in raw_key and "camera" in raw_key:
                return "top"
            if "left" in raw_key and "camera" in raw_key:
                return "left"
            if "right" in raw_key and "camera" in raw_key:
                return "right"
            return None

        renamed_image: dict[str, Any] = {}
        for key, value in image.items():
            match = re.match(r"observation\.images\.([^_]*)", key)
            if match is not None:
                renamed_key = match.group(1)
                normalized = _normalize_view_key(renamed_key)
                if normalized is not None:
                    renamed_image[normalized] = value
            else:
                normalized = _normalize_view_key(key)
                if normalized is not None:
                    renamed_image[normalized] = value

        overlay_frame = self._get_overlay_frame()

        normalized_images: dict[str, np.ndarray] = {}
        normalized_labels: dict[str, str] = {}
        for view in ("top", "left", "right"):
            try:
                arr_value = renamed_image.get(view)
                arr = None
                if arr_value is not None:
                    arr = np.asarray(arr_value).astype(np.uint8)
                    if arr.ndim == 2:
                        arr = np.stack([arr, arr, arr], axis=-1)
                    if arr.size == 0 or not np.any(arr):
                        arr = None

                if arr is None:
                    arr = self._last_images.get(view)
                else:
                    self._last_images[view] = arr

                if arr is None:
                    continue

                # Update main camera view (always update for real-time display)
                self.gui_images[view].image = arr
                normalized_images[view] = arr
                raw_label = next((k for k in renamed_image.keys() if view in k), view)
                normalized_labels[view] = raw_label

                # Add overlay to popup if available and popup is open
                if (
                    view == "top"
                    and overlay_frame is not None
                    and self.overlay_image_popup is not None
                ):
                    resized_arr = cv2.resize(arr, (320 * 2, 240 * 2))
                    overlay_frame_resized = cv2.resize(
                        overlay_frame, (320 * 2, 240 * 2)
                    )
                    blended = cv2.addWeighted(
                        resized_arr, 0.5, overlay_frame_resized, 0.5, 0
                    ).astype(np.uint8)
                    self.overlay_image_popup.image = blended

            except KeyError:
                print(f"[update_images] Could not find camera view: {view}")
            except Exception as e:
                print(f"[update_images] Error updating view {view}: {e}")
        self._maybe_write_video_frame(normalized_images, normalized_labels)

        # Push top camera frame to overlay viz (non-blocking, best-effort)
        if self._overlay_pusher and "top" in normalized_images:
            top_frame = normalized_images["top"]
            # Convert RGB to BGR for cv2.imencode in the pusher
            self._overlay_pusher.push(cv2.cvtColor(top_frame, cv2.COLOR_RGB2BGR))

    def _init_video_output(self) -> None:
        if not self._video_enabled:
            return
        try:
            import imageio.v2 as imageio  # type: ignore
        except ImportError:
            print("[Viser] imageio missing; video recording disabled.", flush=True)
            self._video_enabled = False
            return
        self._imageio = imageio
        repo_root = Path(__file__).resolve().parents[1]
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._video_dir = repo_root / "log" / "deploy" / timestamp
        self._video_dir.mkdir(parents=True, exist_ok=True)
        print(f"\033[34m[Viser] Video output dir: {self._video_dir}\033[0m", flush=True)
        self._video_queue = queue.Queue(maxsize=self._video_queue_size)
        self._video_thread = threading.Thread(
            target=self._video_worker, name="viser-video-writer", daemon=True
        )
        self._video_thread.start()

    def _maybe_write_video_frame(
        self, images: dict[str, np.ndarray], labels: dict[str, str]
    ) -> None:
        if (
            not self._video_enabled
            or self._video_dir is None
            or self._imageio is None
            or self._video_queue is None
        ):
            return
        frame = self._compose_video_frame(images, labels)
        if frame is None:
            return
        repeats = 1
        if self._video_realtime:
            now = time.time()
            if self._last_video_ts is not None:
                dt = max(0.0, now - self._last_video_ts)
                repeats = max(1, int(round(dt * self._video_fps)))
            self._last_video_ts = now
        try:
            self._video_queue.put_nowait((frame, repeats))
        except queue.Full:
            self._video_drop_count += 1
            if self._video_drop_count % 100 == 1:
                print(
                    f"[Viser] Video queue full; dropping frames (dropped={self._video_drop_count}).",
                    flush=True,
                )

    def _compose_video_frame(
        self, images: dict[str, np.ndarray], labels: dict[str, str]
    ) -> np.ndarray | None:
        order = ["top", "left", "right"]
        ordered: list[tuple[str, np.ndarray]] = []
        for key in order:
            if key in images:
                ordered.append((key, images[key]))
        if not ordered:
            return None
        target_h = ordered[0][1].shape[0]
        resized: list[tuple[str, np.ndarray]] = []
        for key, image_np in ordered:
            if image_np.shape[0] != target_h:
                scale = target_h / float(image_np.shape[0])
                target_w = int(round(image_np.shape[1] * scale))
                image_np = cv2.resize(
                    image_np, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
            resized.append((key, image_np))
        total_w = sum(img.shape[1] for _, img in resized)
        font_size = getattr(self._title_font, "size", 36)
        title_h = max(36, int(font_size) + 12)
        canvas = Image.new("RGB", (total_w, target_h + title_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        title = self._cmd_input.value or "Task command"
        left, top, right, bottom = draw.textbbox((0, 0), title, font=self._title_font)
        title_w, title_h_text = right - left, bottom - top
        draw.text(
            ((canvas.width - title_w) // 2, (title_h - title_h_text) // 2),
            title,
            fill=(255, 255, 255),
            font=self._title_font,
        )
        x0 = 0
        for key, image_np in resized:
            canvas.paste(Image.fromarray(image_np), (x0, title_h))
            label = labels.get(key, key)
            left, top, right, bottom = draw.textbbox((0, 0), label, font=self._font)
            label_w, label_h = right - left, bottom - top
            label_x = x0 + 6
            label_y = title_h + 6
            draw.rectangle(
                (
                    label_x - 2,
                    label_y - 2,
                    label_x + label_w + 2,
                    label_y + label_h + 2,
                ),
                fill=(0, 0, 0),
            )
            draw.text((label_x, label_y), label, fill=(255, 255, 255), font=self._font)
            x0 += image_np.shape[1]
        return np.asarray(canvas)

    def _load_title_font(self) -> ImageFont.ImageFont:
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(path, size=36)
            except Exception:
                continue
        return ImageFont.load_default()

    def _video_worker(self) -> None:
        if (
            self._video_dir is None
            or self._imageio is None
            or self._video_queue is None
        ):
            return
        while not self._video_stop.is_set() or not self._video_queue.empty():
            try:
                frame, repeats = self._video_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._video_lock:
                if self._video_writer is None:
                    self._video_path = self._video_dir / "viser_views.mp4"
                    self._video_writer = self._imageio.get_writer(
                        str(self._video_path), fps=self._video_fps
                    )
                    print(
                        f"\033[34m[Viser] Saving video stream to {self._video_path}\033[0m",
                        flush=True,
                    )
                for _ in range(repeats):
                    self._video_writer.append_data(frame)
            self._video_queue.task_done()

    def close(self) -> None:
        self._mp_region_stop.set()
        if self._mp_region_worker is not None:
            self._mp_region_worker.join(timeout=2.0)
        self._shutdown_viser_server()
        if self._video_thread is not None:
            self._video_stop.set()
            self._video_thread.join(timeout=2.0)
        if self._video_writer is None:
            return
        with self._video_lock:
            try:
                self._video_writer.close()
            except Exception:
                print("[Viser] Failed to close video writer", flush=True)
            self._video_writer = None

    def _shutdown_viser_server(self) -> None:
        server = getattr(self, "server", None)
        if server is None:
            return
        for attr in ("shutdown", "close", "stop"):
            fn = getattr(server, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:
                    print(f"[Viser] Failed to {attr} server: {exc}", flush=True)
                break
        nested = getattr(server, "server", None) or getattr(server, "_server", None)
        if nested is not None:
            for attr in ("shutdown", "close", "stop"):
                fn = getattr(nested, attr, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as exc:
                        print(f"[Viser] Failed to {attr} server: {exc}", flush=True)
                    break
        time.sleep(0.1)

    @safe_call
    def _update_future_images(self, future_predictions: dict[str, Any]):
        """Display predicted future state images side-by-side with current observations."""
        # Map cosmos_utils keys to our gui_future_images keys
        key_mapping = {
            "future_image": "top",
            "future_wrist_image": "left",
            "future_wrist_image2": "right",
        }

        for cosmos_key, view in key_mapping.items():
            if cosmos_key in future_predictions:
                try:
                    arr = np.asarray(future_predictions[cosmos_key])

                    # Handle shape (B, H, W, C) or (H, W, C)
                    if arr.ndim == 4:
                        arr = arr[0]  # Take first batch element

                    # Convert to uint8 if needed
                    if arr.dtype != np.uint8:
                        if arr.max() <= 1.0:
                            arr = (arr * 255).astype(np.uint8)
                        else:
                            arr = arr.astype(np.uint8)

                    # Convert grayscale to RGB if needed
                    if arr.ndim == 2:
                        arr = np.stack([arr, arr, arr], axis=-1)

                    # Create GUI element on first use
                    if view not in self.gui_future_images and arr.size > 0:
                        with self._images_folder:
                            self.gui_future_images[view] = self.server.gui.add_image(
                                label=f"{view} (predicted)", image=arr
                            )
                    # Update existing GUI image
                    elif arr.size > 0:
                        self.gui_future_images[view].image = arr
                except Exception as exc:
                    print(
                        f"[Viser] Failed to update {view} future image: {exc}",
                        file=sys.stderr,
                    )

    def _point_color(self, t_norm: float) -> tuple[float, float, float, float]:
        r = float(t_norm)
        g = 0.2
        b = float(1.0 - t_norm)
        return (r, g, b, 1.0)

    def _update_prediction(self, action_chunk: dict[str, Any]) -> None:
        if self.ik_robot is None:
            return

        # Extract sequences
        def _get_seq(key_candidates: list[str]) -> np.ndarray | None:
            for k in key_candidates:
                if k in action_chunk:
                    seq = np.asarray(action_chunk[k], dtype=float)
                    if seq.ndim == 3:
                        seq = seq[0]
                    return seq
            return None

        # Check if action chunk is in ee_pose format and convert to joint positions
        if "left_ee_pos" in action_chunk and "right_ee_pos" in action_chunk:
            # Convert ee_pose action chunk to joint positions using IK
            if not hasattr(self, "_kinematics"):
                self._kinematics = YamKinematics()

            left_ee_pos_seq = _get_seq(["left_ee_pos"])
            left_ee_quat_seq = _get_seq(["left_ee_quat_xyzw"])
            right_ee_pos_seq = _get_seq(["right_ee_pos"])
            right_ee_quat_seq = _get_seq(["right_ee_quat_xyzw"])

            if (
                left_ee_pos_seq is not None
                and left_ee_quat_seq is not None
                and right_ee_pos_seq is not None
                and right_ee_quat_seq is not None
            ):
                # Convert each timestep in the chunk
                horizon = left_ee_pos_seq.shape[0]
                left_joint_seq = np.zeros((horizon, 6))
                right_joint_seq = np.zeros((horizon, 6))

                for s in range(horizon):
                    left_joint_seq[s], right_joint_seq[s] = (
                        self._kinematics.inverse_kinematics(
                            left_ee_pos_seq[s],
                            left_ee_quat_seq[s],
                            right_ee_pos_seq[s],
                            right_ee_quat_seq[s],
                            seeded=(s > 0),  # Seed from previous step for efficiency
                        )
                    )

                # Use converted joint sequences
                left_seq = left_joint_seq
                right_seq = right_joint_seq
            else:
                # Can't convert, skip visualization
                return
        else:
            # Action chunk is already in joint format
            left_seq = _get_seq(
                ["left_joint_pos", "joint_pos_action_left", "left_arm_joints"]
            )  # (H,6)
            right_seq = _get_seq(
                ["right_joint_pos", "joint_pos_action_right", "right_arm_joints"]
            )  # (H,6)
        steps = max(
            (left_seq.shape[0] if left_seq is not None else 0),
            (right_seq.shape[0] if right_seq is not None else 0),
        )
        if steps <= 0:
            return
        idxs = np.linspace(0, steps - 1, num=int(steps), dtype=int)

        # Prepare outputs
        left_positions: list[np.ndarray] = []
        right_positions: list[np.ndarray] = []
        left_wxyzs: list[np.ndarray] = []
        right_wxyzs: list[np.ndarray] = []

        for s in idxs:
            # Build actuated cfg
            cfg = np.zeros((len(self._ik_actuated_names),), dtype=float)
            if right_seq is not None and len(self._ik_idx_right) >= 6:
                cfg[np.array(self._ik_idx_right[: right_seq.shape[-1]])] = right_seq[s]
            if left_seq is not None and len(self._ik_idx_left) >= 6:
                cfg[np.array(self._ik_idx_left[: left_seq.shape[-1]])] = left_seq[s]
            Ts_world_links = np.asarray(
                self.ik_robot.forward_kinematics(cfg[None, ...])
            )  # type: ignore[arg-type]  # (1, L, 7)
            for side, acc in (
                ("right", (right_positions, right_wxyzs)),
                ("left", (left_positions, left_wxyzs)),
            ):
                name = (
                    self.ik_tcp_link_names[0]
                    if side == "right"
                    else self.ik_tcp_link_names[1]
                )
                link_idx = self._tcp_link_idx.get(name, None)
                if link_idx is None:
                    continue
                wxyz_xyz = Ts_world_links[0, link_idx]
                p_tcp = np.asarray(wxyz_xyz[4:7], dtype=float)
                q_tcp = np.asarray(wxyz_xyz[0:4], dtype=float)
                # Apply grasp-site offset along local +Z of TCP to align with MJCF grasp site
                R_tcp = R.from_quat(q_tcp, scalar_first=True).as_matrix()
                p_grasp = p_tcp + R_tcp[:, 2] * float(self.grasp_site_offset_m)
                acc[0].append(p_grasp)
                acc[1].append(q_tcp)

        npts = len(idxs)
        # for this POC we're not doing ee_as_pose. Can uncomment we're doing real thing.
        if self.ee_as_poses_cb is not None and bool(self.ee_as_poses_cb.value):  # type: ignore[attr-defined]
            # Frames mode
            self._ensure_ee_frames(npts)
            # Hide points
            for h in self.ee_points_left:
                h.visible = False
            for h in self.ee_points_right:
                h.visible = False
            # Update frames
            for i in range(npts):
                if i < len(left_positions):
                    p = left_positions[i]
                    q = (
                        left_wxyzs[i]
                        if i < len(left_wxyzs)
                        else np.array([1, 0, 0, 0], dtype=float)
                    )
                    hl = self.ee_frames_left[i]
                    hl.position = (float(p[0]), float(p[1]), float(p[2]))
                    hl.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    hl.visible = True
                if i < len(right_positions):
                    p = right_positions[i]
                    q = (
                        right_wxyzs[i]
                        if i < len(right_wxyzs)
                        else np.array([1, 0, 0, 0], dtype=float)
                    )
                    hr = self.ee_frames_right[i]
                    hr.position = (float(p[0]), float(p[1]), float(p[2]))
                    hr.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    hr.visible = True
            # Hide any extra frames beyond npts
            for i in range(npts, len(self.ee_frames_left)):
                self.ee_frames_left[i].visible = False
            for i in range(npts, len(self.ee_frames_right)):
                self.ee_frames_right[i].visible = False
        else:
            # Points mode
            self._ensure_ee_points(npts)
            # Hide frames
            for h in self.ee_frames_left:
                h.visible = False
            for h in self.ee_frames_right:
                h.visible = False
            # Update points
            for i in range(npts):
                if i < len(left_positions):
                    p = left_positions[i]
                    hl = self.ee_points_left[i]
                    hl.position = (float(p[0]), float(p[1]), float(p[2]))
                    hl.visible = True
                if i < len(right_positions):
                    p = right_positions[i]
                    hr = self.ee_points_right[i]
                    hr.position = (float(p[0]), float(p[1]), float(p[2]))
                    hr.visible = True
        # Hide extras
        for i in range(npts, len(self.ee_points_left)):
            self.ee_points_left[i].visible = False
        for i in range(npts, len(self.ee_points_right)):
            self.ee_points_right[i].visible = False

    def _ensure_ee_frames(self, count: int) -> None:
        while len(self.ee_frames_left) < count:
            i = len(self.ee_frames_left)
            t_norm = i / max(count - 1, 1)
            rgb = self._point_color(t_norm)[:3]
            rgb255 = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            h = self.server.scene.add_frame(
                f"/pred/left_pose_{i}",
                show_axes=True,
                axes_length=0.02,
                axes_radius=0.002,
                origin_radius=0.005,
                origin_color=rgb255,
                visible=False,
            )
            self.ee_frames_left.append(h)
        while len(self.ee_frames_right) < count:
            i = len(self.ee_frames_right)
            t_norm = i / max(count - 1, 1)
            rgb = self._point_color(t_norm)[:3]
            rgb255 = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            h = self.server.scene.add_frame(
                f"/pred/right_pose_{i}",
                show_axes=True,
                axes_length=0.02,
                axes_radius=0.002,
                origin_radius=0.005,
                origin_color=rgb255,
                visible=False,
            )
            self.ee_frames_right.append(h)

    def _ensure_ee_points(self, count: int) -> None:
        radius = 0.008
        while len(self.ee_points_left) < count:
            i = len(self.ee_points_left)
            t_norm = i / max(count - 1, 1)
            rgba = self._point_color(t_norm)
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            vc = (np.array(rgba) * 255).astype(np.uint8)
            vis = getattr(sphere, "visual", None)
            if vis is not None and hasattr(vis, "vertex_colors"):
                vis.vertex_colors = np.tile(vc, (sphere.vertices.shape[0], 1))
            h = self.server.scene.add_mesh_trimesh(
                f"/pred/left_pt_{i}", sphere, position=(0.0, 0.0, 0.0)
            )
            self.ee_points_left.append(h)
        while len(self.ee_points_right) < count:
            i = len(self.ee_points_right)
            t_norm = i / max(count - 1, 1)
            rgba = self._point_color(t_norm)
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            vc = (np.array(rgba) * 255).astype(np.uint8)
            vis = getattr(sphere, "visual", None)
            if vis is not None and hasattr(vis, "vertex_colors"):
                vis.vertex_colors = np.tile(vc, (sphere.vertices.shape[0], 1))
            h = self.server.scene.add_mesh_trimesh(
                f"/pred/right_pt_{i}", sphere, position=(0.0, 0.0, 0.0)
            )
            self.ee_points_right.append(h)

    # ----------------------- 6D gizmo teleop -------------------------------------------------

    def _snap_gizmos_to_current(self) -> None:
        """Snap gizmo positions to current EE poses computed via FK.

        PRIMARY method: compute FK from cached joint positions (updated every
        observation frame in ``_portal_update_state``).  This avoids depending
        on the ScriptedPolicy RPC which can return None after homing.

        Sets ``_gizmos_snapped = True`` on success so that
        ``_poll_and_send_gizmo_targets`` is allowed to send commands.
        """
        left_jp = self._cached_left_jp
        right_jp = self._cached_right_jp
        if left_jp is None or right_jp is None:
            print(
                "[ViserUI] SAFETY: Cannot snap gizmos — no cached joint positions yet. "
                "Waiting for first observation frame."
            )
            return

        try:
            if not hasattr(self, "_kinematics"):
                self._kinematics = YamKinematics()
            left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw = (
                self._kinematics.forward_kinematics(left_jp[:6], right_jp[:6])
            )
        except Exception as exc:
            print(f"[ViserUI] SAFETY: FK computation failed, gizmos NOT snapped: {exc}")
            return

        for side, ee_pos, ee_quat_xyzw in (
            ("left", left_ee_pos, left_ee_quat_xyzw),
            ("right", right_ee_pos, right_ee_quat_xyzw),
        ):
            wxyz = (
                float(ee_quat_xyzw[3]),
                float(ee_quat_xyzw[0]),
                float(ee_quat_xyzw[1]),
                float(ee_quat_xyzw[2]),
            )
            handle = self.ik_left if side == "left" else self.ik_right
            handle.position = (float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2]))
            handle.wxyz = wxyz

        self._gizmos_snapped = True
        self._update_gizmo_pose_display()

    def _update_gizmo_pose_display(self) -> None:
        """Update the pose readout texts from current gizmo positions."""
        if not self._gizmo_visible or not hasattr(self, "_gizmo_pose_texts"):
            return
        for side in ("left", "right"):
            handle = self.ik_left if side == "left" else self.ik_right
            pos = handle.position
            wxyz = np.array(handle.wxyz)
            xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
            global_rpy = _quat_xyzw_to_global_rpy_deg(xyzw)
            gripper_rpy = _quat_xyzw_to_display_rpy_deg(xyzw)
            self._gizmo_pose_texts[
                f"{side}_pos"
            ].value = f"{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}"
            self._gizmo_pose_texts[
                f"{side}_rpy_global"
            ].value = f"{global_rpy[0]:.1f}, {global_rpy[1]:.1f}, {global_rpy[2]:.1f}"
            self._gizmo_pose_texts[
                f"{side}_rpy_gripper"
            ].value = (
                f"{gripper_rpy[0]:.1f}, {gripper_rpy[1]:.1f}, {gripper_rpy[2]:.1f}"
            )

    def _poll_and_send_gizmo_targets(self) -> None:
        """Read gizmo poses, convert wxyz->xyzw, send to ScriptedPolicy.

        Only active when live teleop is enabled. When disabled, gizmos are
        passive waypoint markers for the Move-To buttons.

        SAFETY: refuses to send targets if gizmos have never been snapped to
        FK-computed positions (``_gizmos_snapped`` is False).
        """
        if not self._gizmo_live_teleop:
            return
        if not self._gizmos_snapped:
            print(
                "[ViserUI] SAFETY: Blocking gizmo target send — gizmos not yet "
                "snapped to FK. This should not happen; snap before enabling teleop."
            )
            return

        left_pos = np.array(self.ik_left.position)
        left_wxyz = np.array(self.ik_left.wxyz)
        left_xyzw = np.array([left_wxyz[1], left_wxyz[2], left_wxyz[3], left_wxyz[0]])

        right_pos = np.array(self.ik_right.position)
        right_wxyz = np.array(self.ik_right.wxyz)
        right_xyzw = np.array(
            [right_wxyz[1], right_wxyz[2], right_wxyz[3], right_wxyz[0]]
        )

        payload = {
            "left_pos": left_pos.tolist(),
            "left_quat_xyzw": left_xyzw.tolist(),
            "right_pos": right_pos.tolist(),
            "right_quat_xyzw": right_xyzw.tolist(),
        }
        try:
            self._client.scripted_set_target(payload).result()
        except Exception:
            pass  # best-effort, don't block UI

    # ----------------------- URDF update ---------------------------------------------------

    def _gripper_joint_target(
        self, joint_name: str, gripper_open: float
    ) -> float | None:
        limits = self._urdf_joint_limits.get(joint_name)
        if limits is None:
            return None
        limit_arr = np.asarray(limits, dtype=float).reshape(-1)
        if limit_arr.size < 2:
            return None
        open_extent = float(limit_arr[:2][np.argmax(np.abs(limit_arr[:2]))])
        return float(np.clip(gripper_open, 0.0, 1.0) * open_extent)

    @safe_call
    def _update_urdf_cfg_from_proprio(self, proprio: dict[str, Any]) -> None:
        if self.urdf_vis is None:
            return
        # Try to assemble full actuated joint vector in the URDF's name order
        q_map = hold_action_from_proprio(proprio)

        # Handle both joint and ee_pose formats
        if "left_joint_pos" in q_map and "right_joint_pos" in q_map:
            # Already in joint format
            left_joint_pos = q_map["left_joint_pos"]
            right_joint_pos = q_map["right_joint_pos"]
        elif "left_ee_pos" in q_map and "right_ee_pos" in q_map:
            # Need to convert from ee_pose to joint positions using IK
            if not hasattr(self, "_kinematics"):
                self._kinematics = YamKinematics()
            left_joint_pos, right_joint_pos = self._kinematics.inverse_kinematics(
                q_map["left_ee_pos"],
                q_map["left_ee_quat_xyzw"],
                q_map["right_ee_pos"],
                q_map["right_ee_quat_xyzw"],
                seeded=True,
            )
        else:
            # Can't determine format, skip update
            return

        joint_names = self.urdf_joint_names or (
            list(self._ik_actuated_names)
            if hasattr(self, "_ik_actuated_names") and self._ik_actuated_names
            else [
                *[f"right_joint{i}" for i in range(1, 7)],
                *[f"left_joint{i}" for i in range(1, 7)],
            ]
        )
        name_to_val: dict[str, float] = {}
        for i in range(6):
            name_to_val[f"left_joint{i + 1}"] = float(left_joint_pos[i])
            name_to_val[f"right_joint{i + 1}"] = float(right_joint_pos[i])
        left_gripper = float(
            np.clip(
                np.asarray(
                    q_map.get("left_gripper_pos", np.zeros(1)), dtype=float
                ).reshape(-1)[0],
                0.0,
                1.0,
            )
        )
        right_gripper = float(
            np.clip(
                np.asarray(
                    q_map.get("right_gripper_pos", np.zeros(1)), dtype=float
                ).reshape(-1)[0],
                0.0,
                1.0,
            )
        )
        left_slide = left_gripper * float(self.urdf_gripper_open_m)
        right_slide = right_gripper * float(self.urdf_gripper_open_m)
        name_to_val["left_left_finger_joint"] = left_slide
        name_to_val["left_right_finger_joint"] = -left_slide
        name_to_val["right_left_finger_joint"] = right_slide
        name_to_val["right_right_finger_joint"] = -right_slide
        cfg = np.array([name_to_val.get(n, 0.0) for n in joint_names], dtype=float)
        self.urdf_vis.update_cfg(cfg)


def dummy_env_loop(policy: StartStopPlayPolicyWrapper, viser_ui: ViserUI):
    """Dummy environment loop for testing."""
    home_observation = {
        "left_joint_pos": np.zeros(7),
        "left_gripper_pos": np.zeros(1),
        "right_joint_pos": np.zeros(7),
        "right_gripper_pos": np.zeros(1),
        "top_camera_image": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),
        "left_camera_image": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),
        "right_camera_image": (np.random.rand(480, 640, 3) * 255).astype(np.uint8),
        "timestamp": time.time(),
    }
    observation = home_observation.copy()
    while True:
        action, info = policy.get_action(observation)
        if info.get("event", None) == "home":
            observation = home_observation.copy()
            continue
        observation = action.copy()
        observation["top_camera_image"] = (np.random.rand(480, 640, 3) * 255).astype(
            np.uint8
        )
        observation["left_camera_image"] = (np.random.rand(480, 640, 3) * 255).astype(
            np.uint8
        )
        observation["right_camera_image"] = (np.random.rand(480, 640, 3) * 255).astype(
            np.uint8
        )
        observation["timestamp"] = time.time()

        # Update viser visualization
        viser_ui.update_ui()
        time.sleep(1 / 30)


def run_viser_subprocess(
    adapters: PolicyAdapters,
    task_command: str,
    replay_action_horizon: int = 50,
    embodiment_tag: EmbodimentTag | str = "XDOF",
    urdf_path: Path | None = None,
    policy_host: str = "localhost",
    policy_port: int = 8009,
    viser_port: int = 8010,
    viser_web_port: int = 8080,
    action_type: Literal["absolute", "relative"] = "absolute",
    video_enabled: bool = True,
    video_fps: int = 30,
    video_realtime: bool = True,
    video_queue_size: int = 512,
    top_cam_to_world: np.ndarray | None = None,
    show_scripted_controls: bool = False,
    scripted_use_planner_default: bool = True,
    scripted_planner_solver_speed: MotionPlannerSolverSpeed = "fast",
    scripted_planner_max_joint_vel: float = 0.2,
    show_mp_feasible_region: bool = False,
    recompute_mp: bool = False,
    default_motion_planner_backend: MotionPlannerBackend = "curobo",
    shared_motion_planner_port: int | None = None,
    eval_run_name: str = "",
):
    """
    Run ViserUI in a subprocess.
    This function is meant to be called by portal.Process.

    Args:
        adapters: PolicyAdapters for observation/action mapping
        task_command: Initial task command
        replay_action_horizon: Initial replay action horizon shown in the UI
        embodiment_tag: Robot embodiment tag
        urdf_path: Path to URDF file (defaults to station.urdf)
        policy_host: Host where policy server is running
        policy_port: Port where policy server is listening
        viser_port: Port for ViserUI server
        viser_web_port: Port for Viser web UI
        top_cam_to_world: 4x4 camera-to-world transform for top camera detections.
    """
    if urdf_path is None:
        from enpire.env.forge.robot.models.station.paths import get_station_urdf

        urdf_path = get_station_urdf()

    print(f"[ViserUI Subprocess] URDF path: {urdf_path}")

    print("[ViserUI Subprocess] Starting ViserUI...")
    viser_ui = ViserUI(
        adapters=adapters,
        task_command=task_command,
        replay_action_horizon=replay_action_horizon,
        embodiment_tag=embodiment_tag,
        urdf_path=urdf_path,
        policy_host=policy_host,
        policy_port=policy_port,
        viser_port=viser_port,
        viser_web_port=viser_web_port,
        action_type=action_type,
        video_enabled=video_enabled,
        video_fps=video_fps,
        video_realtime=video_realtime,
        video_queue_size=video_queue_size,
        top_cam_to_world=top_cam_to_world,
        show_scripted_controls=show_scripted_controls,
        scripted_use_planner_default=scripted_use_planner_default,
        default_motion_planner_solver_speed=scripted_planner_solver_speed,
        scripted_planner_max_joint_vel=scripted_planner_max_joint_vel,
        show_mp_feasible_region=show_mp_feasible_region,
        recompute_mp=recompute_mp,
        default_motion_planner_backend=default_motion_planner_backend,
        shared_motion_planner_port=shared_motion_planner_port,
        eval_run_name=eval_run_name,
    )

    print("[ViserUI Subprocess] ViserUI initialized, starting update loop...")
    # Run update loop
    try:
        while True:
            viser_ui.update_ui()
            time.sleep(1 / 20)  # 20 Hz update rate
    except KeyboardInterrupt:
        pass
    finally:
        viser_ui.close()
