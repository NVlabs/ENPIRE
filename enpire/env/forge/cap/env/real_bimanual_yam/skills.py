"""Direct-env tool namespace for real bimanual YAM.

This module is intentionally a thin shim: user-facing tool behavior lives in
``cap.agent.tools``.  ``make_namespace`` instantiates those tools with
``env=env`` so direct script execution talks to the provided environment.
"""

from __future__ import annotations

import signal
import threading
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

# Imported by YamDashboard.  Tool implementations own motion behavior; these
# flags remain here for UI compatibility with the direct runner.
_stop_requested = threading.Event()
_pause_requested = threading.Event()


def _sigint_handler(sig, frame):
    if not _stop_requested.is_set():
        print("\n[YAM] Stop requested.")
    _stop_requested.set()


signal.signal(signal.SIGINT, _sigint_handler)


def display_rpy_to_quat(rpy_deg: list[float] | np.ndarray) -> np.ndarray:
    """Convert planner/display RPY degrees to quaternion xyzw."""
    roll, pitch, yaw = [float(x) for x in rpy_deg]
    return Rotation.from_euler(
        "xyz", [-pitch, roll, -yaw - 90.0], degrees=True
    ).as_quat()


def _normalize_quat_xyzw(quat: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        raise ValueError("quaternion norm is zero")
    return arr / norm


def _arm(env: Any, side: str):
    return env._profile.arms[side]


def _sample_keypoints(
    timestamps: np.ndarray,
    values: np.ndarray,
    t_now: float,
) -> np.ndarray:
    if t_now <= float(timestamps[0]):
        return values[0]
    if t_now >= float(timestamps[-1]):
        return values[-1]
    idx = int(np.searchsorted(timestamps, t_now, side="right") - 1)
    idx = max(0, min(idx, len(timestamps) - 2))
    t0 = float(timestamps[idx])
    t1 = float(timestamps[idx + 1])
    alpha = 1.0 if t1 <= t0 + 1e-9 else (float(t_now) - t0) / (t1 - t0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * values[idx] + alpha * values[idx + 1]


def _command_bimanual_joint7(env: Any, left_cmd: np.ndarray, right_cmd: np.ndarray) -> None:
    left_p = _arm(env, "left")
    right_p = _arm(env, "right")
    env.command_joint_state(
        "left",
        {
            "pos": np.asarray(left_cmd, dtype=np.float64).reshape(7),
            "vel": np.zeros(7),
            "kp": left_p.interp_kp,
            "kd": left_p.interp_kd,
        },
    )
    env.command_joint_state(
        "right",
        {
            "pos": np.asarray(right_cmd, dtype=np.float64).reshape(7),
            "vel": np.zeros(7),
            "kp": right_p.interp_kp,
            "kd": right_p.interp_kd,
        },
    )


def _prepare_joint7_waypoints(
    env: Any,
    name: str,
    joint_positions: list,
    gripper_positions: list | None,
    n: int,
) -> np.ndarray:
    arr = np.asarray(joint_positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 6:
        raise ValueError(
            f"{name}_joint_positions must have shape (N,6) or (N,7); got {arr.shape}"
        )
    joints = arr[:, :6]
    if gripper_positions is None:
        if arr.shape[1] >= 7:
            gripper = arr[:, 6]
        else:
            cur = env._arms[name].get_observations()["gripper_pos"]
            gripper = np.full(n, float(np.asarray(cur).ravel()[0]), dtype=np.float64)
    else:
        gripper = np.asarray(gripper_positions, dtype=np.float64).reshape(n, -1)[:, 0]
    gripper = np.clip(gripper, 0.0, 1.0)
    return np.column_stack([joints, gripper]).astype(np.float64)


def _move_bimanual_joint_keypoints(
    env: Any,
    timestamps: list[float] | np.ndarray,
    left_joint_positions: list | np.ndarray,
    right_joint_positions: list | np.ndarray,
    left_gripper_positions: list | np.ndarray | None = None,
    right_gripper_positions: list | np.ndarray | None = None,
    playback_speed: float = 1.0,
    command_hz: float = 60.0,
    start_interp_s: float = 0.0,
) -> dict[str, Any]:
    """Replay synchronized bimanual joint waypoints directly on RealYamEnv.

    This is private support for direct-env FreespaceMoveTool execution.  It is
    deliberately not exported into the script namespace.
    """
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size < 1:
        return {"success": False, "reason": "empty timestamps"}
    if not np.all(np.isfinite(ts)):
        return {"success": False, "reason": "timestamps contain non-finite values"}
    ts = ts - float(ts[0])
    if np.any(np.diff(ts) < -1e-9):
        return {"success": False, "reason": "timestamps must be monotonically increasing"}

    keep = np.ones(ts.shape[0], dtype=bool)
    keep[1:] = np.diff(ts) > 1e-9
    original_n = int(len(keep))
    ts = ts[keep]
    n = int(ts.size)
    try:
        left7_all = _prepare_joint7_waypoints(
            env, "left", left_joint_positions, left_gripper_positions, original_n
        )[keep]
        right7_all = _prepare_joint7_waypoints(
            env, "right", right_joint_positions, right_gripper_positions, original_n
        )[keep]
    except Exception as exc:
        return {"success": False, "reason": str(exc)}
    if left7_all.shape[0] != n or right7_all.shape[0] != n:
        return {"success": False, "reason": "waypoint count mismatch after timestamp filtering"}

    speed = max(0.05, float(playback_speed))
    ts = ts / speed
    duration_s = float(ts[-1]) if ts.size else 0.0
    dt = 1.0 / max(1.0, float(command_hz))

    obs_l = env._arms["left"].get_observations()
    obs_r = env._arms["right"].get_observations()
    cur_left7 = np.concatenate([obs_l["joint_pos"], obs_l["gripper_pos"]]).astype(np.float64)
    cur_right7 = np.concatenate([obs_r["joint_pos"], obs_r["gripper_pos"]]).astype(np.float64)
    first_left7 = left7_all[0]
    first_right7 = right7_all[0]

    interp_s = max(0.0, float(start_interp_s))
    interp_steps = int(np.ceil(interp_s / dt)) if interp_s > 1e-9 else 0
    for step in range(1, interp_steps + 1):
        if _stop_requested.is_set():
            raise KeyboardInterrupt("stop during replay start interpolation")
        while _pause_requested.is_set():
            if _stop_requested.is_set():
                raise KeyboardInterrupt("stop while paused during replay")
            time.sleep(dt)
        alpha = float(step) / float(max(interp_steps, 1))
        _command_bimanual_joint7(
            env,
            (1.0 - alpha) * cur_left7 + alpha * first_left7,
            (1.0 - alpha) * cur_right7 + alpha * first_right7,
        )
        time.sleep(dt)

    t0 = time.time()
    command_count = 0
    while True:
        if _stop_requested.is_set():
            raise KeyboardInterrupt("stop during bimanual trajectory replay")
        while _pause_requested.is_set():
            if _stop_requested.is_set():
                raise KeyboardInterrupt("stop while paused during replay")
            time.sleep(dt)

        t_now = time.time() - t0
        left_cmd = _sample_keypoints(ts, left7_all, t_now)
        right_cmd = _sample_keypoints(ts, right7_all, t_now)
        _command_bimanual_joint7(env, left_cmd, right_cmd)
        command_count += 1
        if t_now >= duration_s:
            break
        time.sleep(dt)

    settle_steps = max(1, int(round(0.2 / dt)))
    for _ in range(settle_steps):
        if _stop_requested.is_set():
            raise KeyboardInterrupt("stop during replay settle")
        _command_bimanual_joint7(env, left7_all[-1], right7_all[-1])
        time.sleep(dt)

    return {
        "success": True,
        "reason": "ok",
        "waypoints": int(n),
        "duration_s": round(duration_s, 4),
        "playback_speed": float(speed),
        "command_hz": float(command_hz),
        "command_count": int(command_count),
        "start_interp_s": float(interp_s),
        "final_left_gripper": float(left7_all[-1, 6]),
        "final_right_gripper": float(right7_all[-1, 6]),
    }


def _install_direct_helpers(env: Any) -> None:
    if not hasattr(env, "_move_bimanual_joint_keypoints"):
        setattr(
            env,
            "_move_bimanual_joint_keypoints",
            lambda *args, **kwargs: _move_bimanual_joint_keypoints(env, *args, **kwargs),
        )
    if not hasattr(env, "move_bimanual_joint_keypoints"):
        setattr(
            env,
            "move_bimanual_joint_keypoints",
            lambda *args, **kwargs: _move_bimanual_joint_keypoints(env, *args, **kwargs),
        )


def _tool_with_env(tool_cls: type, env: Any, **kwargs: Any):
    try:
        return tool_cls(env=env, **kwargs)
    except TypeError as exc:
        raise TypeError(
            f"{tool_cls.__module__}.{tool_cls.__name__} must support env= for "
            "real_bimanual_yam direct mode; refusing to fall back to remote transport."
        ) from exc


def _call_tool(tool: Any, *args: Any, **kwargs: Any) -> Any:
    param_names = [p.name for p in getattr(tool, "parameters", [])]
    for idx, value in enumerate(args):
        if idx < len(param_names):
            kwargs[param_names[idx]] = value
    result = tool.execute(**kwargs)
    if not result.success:
        raise RuntimeError(f"Tool {tool.name} failed: {result.error}")
    return result.data


def _tool_callable(tool: Any) -> Callable[..., Any]:
    def fn(*args: Any, **kwargs: Any) -> Any:
        return _call_tool(tool, *args, **kwargs)

    fn.__name__ = getattr(tool, "name", tool.__class__.__name__)
    fn.__doc__ = getattr(tool, "description", None)
    return fn


def _cfg_select(cfg: Any, path: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(cfg, path, default=default)
    except Exception:
        cur = cfg
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                return default
        return cur


def make_namespace(env, vlm_backend: str = "gemini", cfg: Any = None) -> dict[str, Any]:
    """Build the real-YAM direct-mode namespace from shared tool classes."""
    from enpire.env.forge.cap.agent.tools.bundlesdf_track import EndDetectionTool, ListDetectionsTool
    from enpire.env.forge.cap.agent.tools.camera import (
        GetCameraExtrinsicsTool,
        GetCameraIntrinsicsTool,
        RenderDepthTool,
        RenderRgbTool,
    )
    from enpire.env.forge.cap.agent.tools.detection import DetectObjectTool, DetectObjectsOneshotTool
    from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool
    from enpire.env.forge.cap.agent.tools.grasp_2d import SampleGraspPose2DTool
    from enpire.env.forge.cap.agent.tools.grasp_3d_bb import SampleGraspPose3DBBoxTool
    from enpire.env.forge.cap.agent.tools.grasp_anygrasp import SampleGraspPoseAnyGraspTool
    from enpire.env.forge.cap.agent.tools.native import (
        CloseGripperTool,
        GetCameraImageTool,
        GetRobotStateTool,
        GoHomeTool,
        OpenGripperTool,
        SetGripperTool,
    )
    from enpire.env.forge.cap.agent.tools.nudge import NudgeTool
    from enpire.env.forge.cap.agent.tools.segmentation import SegmentAllObjectsTool, SegmentObjectTool
    from enpire.env.forge.cap.agent.tools.vlm_query import VlmQueryTool

    _install_direct_helpers(env)

    freespace = _tool_with_env(FreespaceMoveTool, env)
    nudge = _tool_with_env(NudgeTool, env)
    segment = _tool_with_env(SegmentObjectTool, env)
    segment_all = _tool_with_env(SegmentAllObjectsTool, env)
    detect = _tool_with_env(DetectObjectTool, env)
    detect_oneshot = _tool_with_env(DetectObjectsOneshotTool, env, detect_tool=detect)
    end_detection = EndDetectionTool()
    list_detections = ListDetectionsTool()
    anygrasp = _tool_with_env(SampleGraspPoseAnyGraspTool, env)
    grasp_3d_bb = _tool_with_env(SampleGraspPose3DBBoxTool, env)
    grasp_2d = _tool_with_env(SampleGraspPose2DTool, env)
    vlm = _tool_with_env(VlmQueryTool, env, default_backend=vlm_backend)
    camera_image = _tool_with_env(GetCameraImageTool, env)
    camera_intrinsics = _tool_with_env(GetCameraIntrinsicsTool, env)
    camera_extrinsics = _tool_with_env(GetCameraExtrinsicsTool, env)
    render_rgb = _tool_with_env(RenderRgbTool, env)
    render_depth = _tool_with_env(RenderDepthTool, env)
    robot_state = _tool_with_env(GetRobotStateTool, env)
    set_gripper = _tool_with_env(SetGripperTool, env)
    open_gripper = _tool_with_env(OpenGripperTool, env)
    close_gripper = _tool_with_env(CloseGripperTool, env)
    go_home = _tool_with_env(GoHomeTool, env)

    def set_gripper_fast(
        side: str,
        pos: float,
        vel_limit: float | None = 30.0,
        torque_limit: float | None = None,
        timeout: float = 0.25,
    ) -> dict[str, Any]:
        """Set one YAM gripper with a short blocking timeout."""
        side = str(side).strip().lower()
        if side not in {"left", "right"}:
            raise RuntimeError(f"invalid side: {side}")

        target_pos = float(np.clip(float(pos), 0.0, 1.0))
        timeout = max(0.0, float(timeout))
        vel = None if vel_limit is None else float(vel_limit)
        torque = None if torque_limit is None else float(torque_limit)

        if hasattr(env, "set_gripper"):
            result = env.set_gripper(side, target_pos, timeout, vel, torque)
            if isinstance(result, dict) and not bool(result.get("success", result.get("ok", True))):
                raise RuntimeError(result.get("reason", "set_gripper_fast failed"))
            return {
                "success": True,
                "side": side,
                "gripper": target_pos,
                "vel_limit": vel,
                "torque_limit": torque,
                "timeout": timeout,
                "result": result,
            }

        obs = env.get_observations(side)
        joint_pos = np.asarray(obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        target = np.concatenate([joint_pos, [target_pos]])
        profile = _arm(env, side)
        cmd = {
            "pos": target,
            "vel": np.zeros(7),
            "kp": profile.interp_kp,
            "kd": profile.interp_kd,
            "gripper_vel_limit": vel,
            "gripper_torque_limit_nm": torque,
        }

        dt = 1.0 / 60.0
        t0 = time.time()
        env.command_joint_state(side, cmd)
        while time.time() - t0 < timeout:
            if _stop_requested.is_set():
                raise KeyboardInterrupt("stop during set_gripper_fast")
            while _pause_requested.is_set():
                if _stop_requested.is_set():
                    raise KeyboardInterrupt("stop while paused during set_gripper_fast")
                time.sleep(dt)
            env.command_joint_state(side, cmd)
            time.sleep(dt)

        final_pos = None
        try:
            final_obs = env.get_observations(side)
            final_pos = float(np.asarray(final_obs["gripper_pos"]).reshape(-1)[0])
        except Exception:
            final_pos = None

        return {
            "success": True,
            "side": side,
            "gripper": target_pos,
            "final_gripper_pos": final_pos,
            "vel_limit": vel,
            "torque_limit": torque,
            "timeout": timeout,
        }

    def open_gripper_fast(
        side: str,
        vel_limit: float | None = 30.0,
        torque_limit: float | None = None,
        timeout: float = 0.25,
    ) -> dict[str, Any]:
        """Open one YAM gripper with a short blocking timeout."""
        return set_gripper_fast(
            side,
            1.0,
            vel_limit=vel_limit,
            torque_limit=torque_limit,
            timeout=timeout,
        )

    def go_home_fast(
        max_joint_vel: float = 3.0,
        min_duration_s: float = 0.8,
        settle_s: float = 0.1,
        command_hz: float = 60.0,
        keep_grippers: bool = True,
    ) -> dict[str, Any]:
        """Move both arms home quickly while preserving current gripper positions."""
        left_p = _arm(env, "left")
        right_p = _arm(env, "right")
        obs_l = env.get_observations("left")
        obs_r = env.get_observations("right")
        cur_left = np.concatenate([obs_l["joint_pos"], obs_l["gripper_pos"]]).astype(np.float64)
        cur_right = np.concatenate([obs_r["joint_pos"], obs_r["gripper_pos"]]).astype(np.float64)
        target_left_gripper = (
            cur_left[6:7]
            if bool(keep_grippers)
            else np.asarray(left_p.home_gripper_pos, dtype=np.float64).reshape(1)
        )
        target_right_gripper = (
            cur_right[6:7]
            if bool(keep_grippers)
            else np.asarray(right_p.home_gripper_pos, dtype=np.float64).reshape(1)
        )
        target_left = np.concatenate(
            [
                np.asarray(left_p.home_joint_pos, dtype=np.float64).reshape(-1)[:6],
                target_left_gripper,
            ]
        )
        target_right = np.concatenate(
            [
                np.asarray(right_p.home_joint_pos, dtype=np.float64).reshape(-1)[:6],
                target_right_gripper,
            ]
        )

        max_disp = max(
            float(np.max(np.abs(target_left[:6] - cur_left[:6]))),
            float(np.max(np.abs(target_right[:6] - cur_right[:6]))),
        )
        duration = max(
            float(min_duration_s),
            max_disp / max(1e-6, float(max_joint_vel)),
        )
        dt = 1.0 / max(1.0, float(command_hz))
        n_steps = max(2, int(np.ceil(duration / dt)) + 1)
        t_norm = np.linspace(0.0, 1.0, n_steps)
        smooth = 3.0 * t_norm**2 - 2.0 * t_norm**3

        t0 = time.time()
        command_count = 0
        for alpha in smooth:
            if _stop_requested.is_set():
                raise KeyboardInterrupt("stop during go_home_fast")
            while _pause_requested.is_set():
                if _stop_requested.is_set():
                    raise KeyboardInterrupt("stop while paused during go_home_fast")
                time.sleep(dt)

            cmd_l = {
                "pos": ((1.0 - alpha) * cur_left + alpha * target_left).astype(np.float64),
                "vel": np.zeros(7),
                "kp": left_p.interp_kp,
                "kd": left_p.interp_kd,
            }
            cmd_r = {
                "pos": ((1.0 - alpha) * cur_right + alpha * target_right).astype(np.float64),
                "vel": np.zeros(7),
                "kp": right_p.interp_kp,
                "kd": right_p.interp_kd,
            }
            env.command_joint_state("left", cmd_l)
            env.command_joint_state("right", cmd_r)
            command_count += 1
            time.sleep(dt)

        settle_t0 = time.time()
        while time.time() - settle_t0 < float(settle_s):
            if _stop_requested.is_set():
                raise KeyboardInterrupt("stop during go_home_fast settle")
            while _pause_requested.is_set():
                if _stop_requested.is_set():
                    raise KeyboardInterrupt("stop while paused during go_home_fast settle")
                time.sleep(dt)
            env.command_joint_state(
                "left",
                {"pos": target_left, "vel": np.zeros(7), "kp": left_p.interp_kp, "kd": left_p.interp_kd},
            )
            env.command_joint_state(
                "right",
                {"pos": target_right, "vel": np.zeros(7), "kp": right_p.interp_kp, "kd": right_p.interp_kd},
            )
            command_count += 1
            time.sleep(dt)

        elapsed = time.time() - t0
        return {
            "success": True,
            "duration_s": float(elapsed),
            "command_count": int(command_count),
            "target_left": target_left.tolist(),
            "target_right": target_right.tolist(),
        }

    def select_best_grasp(
        grasp_candidates: list[Any],
        side: str = "right",
        **kwargs: Any,
    ) -> Any:
        kwargs.setdefault("batch_side", side)
        return _call_tool(freespace, grasp_candidates=grasp_candidates, **kwargs)

    def nudge_world_pose_preserving(
        side: str,
        world_delta_m: list[float],
        initial_pos_m: list[float] | None = None,
        initial_quat_xyzw: list[float] | None = None,
        speed: float = 0.3,
        ik_rpy_weight: float = 10.0,
        pos_tol_m: float = 0.005,
        rot_tol_deg: float = 3.0,
        max_attempts: int = 3,
        command_hz: float = 60.0,
        command_fixed_arm: bool = False,
        preview_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """World-frame nudge to an absolute target while preserving orientation.

        Kept as a real-YAM compatibility helper, but execution goes through the
        shared cuRobo-backed ``freespace_move`` tool instead of the removed
        Cartesian servo loop.
        """
        side = str(side).strip().lower()
        if side not in {"left", "right"}:
            return {"success": False, "status": "invalid_side", "side": side}
        try:
            delta = np.asarray(world_delta_m, dtype=np.float64).reshape(3)
        except Exception as exc:
            return {
                "success": False,
                "status": "invalid_delta",
                "side": side,
                "error": str(exc),
            }

        obs = env.get_observations(side)
        start_pos = np.asarray(obs["ee_pos"], dtype=np.float64).reshape(3)
        start_quat = _normalize_quat_xyzw(obs["ee_quat"])
        base_pos = (
            np.asarray(initial_pos_m, dtype=np.float64).reshape(3)
            if initial_pos_m is not None
            else start_pos.copy()
        )
        target_quat = (
            _normalize_quat_xyzw(initial_quat_xyzw)
            if initial_quat_xyzw is not None
            else start_quat.copy()
        )
        target_pos = base_pos + delta

        move_kwargs: dict[str, Any] = {
            f"{side}_target_pos": target_pos.tolist(),
            f"{side}_target_quat": target_quat.tolist(),
            "planning_speed": float(speed),
            "ik_rpy_weight": float(ik_rpy_weight),
            "ik_error_threshold": float(pos_tol_m),
            "ik_rot_threshold_deg": float(rot_tol_deg),
            "preview_only": bool(preview_only),
        }
        move_kwargs.update(kwargs)
        result = freespace.execute(**move_kwargs)
        data = result.data
        final_obs = obs if preview_only else env.get_observations(side)
        final_pos = np.asarray(final_obs.get("ee_pos", target_pos), dtype=np.float64).reshape(3)
        final_quat = _normalize_quat_xyzw(final_obs.get("ee_quat", target_quat))
        log = {
            "success": bool(result.success),
            "status": getattr(data, "status", "success" if result.success else "failed"),
            "side": side,
            "world_delta_m": [float(v) for v in delta],
            "initial_pos_m": [float(v) for v in base_pos],
            "uses_initial_pos": bool(initial_pos_m is not None),
            "start_pos": [float(v) for v in start_pos],
            "start_quat_xyzw": [float(v) for v in start_quat],
            "target_pos": [float(v) for v in target_pos],
            "target_quat_xyzw": [float(v) for v in target_quat],
            "uses_initial_quat": bool(initial_quat_xyzw is not None),
            "final_pos": [float(v) for v in final_pos],
            "final_quat_xyzw": [float(v) for v in final_quat],
            "move_status": getattr(data, "status", None),
            "move_final_pos_error_m": getattr(data, "final_pos_error_m", None),
            "move_final_rot_error_deg": getattr(data, "final_rot_error_deg", None),
            "move_trajectory_steps": getattr(data, "trajectory_steps", None),
            "preview_only": bool(preview_only),
            "max_attempts": int(max_attempts),
            "command_hz": float(command_hz),
            "command_fixed_arm": bool(command_fixed_arm),
        }
        if not result.success:
            log["error"] = result.error or getattr(data, "reason", None)
        return log

    def get_task_info() -> dict[str, Any]:
        """Evaluate real-hardware task success with a configured VLM reward."""
        from enpire.env.forge.cap.agent.tools.vlm import query as _vlm_query

        task = str(_cfg_select(cfg, "task", "Complete the real-YAM task.") or "")
        backend = str(_cfg_select(cfg, "reward.vlm_backend", "nvidia") or "nvidia")
        model = _cfg_select(
            cfg, "reward.vlm_model", "gcp/google/gemini-3.1-pro-preview"
        )
        camera = str(_cfg_select(cfg, "reward.vlm_camera", "top") or "top")
        reasoning_effort = str(
            _cfg_select(cfg, "reward.vlm_reasoning_effort", "high") or "high"
        )

        img = env.render_rgb(camera)
        if img is None:
            return {
                "success": False,
                "reward": 0.0,
                "method": "vlm_reward",
                "error": f"No image available from camera={camera!r}",
            }

        prompt = f"""
You are evaluating whether a real robot task is complete from the camera image.

Task:
{task}

For the nail-bussing task, success means every visible small black nail/screw is
on or inside the blue plate. Failure means at least one visible nail/screw is
still outside the blue plate. If the image is ambiguous, occluded, or the blue
plate is not clearly visible, answer UNSURE.

Answer with exactly one first-line token:
SUCCESS
FAILURE
UNSURE

After the first-line token, add one short sentence explaining the visible
evidence.
""".strip()

        try:
            response = _vlm_query(
                backend=backend,
                text=prompt,
                images=[img],
                model=model,
                temperature=0.0,
                reasoning_effort=reasoning_effort,
                telemetry_source="real_yam_task_reward",
            )
        except Exception as exc:
            return {
                "success": False,
                "reward": 0.0,
                "method": "vlm_reward",
                "backend": backend,
                "model": model,
                "camera": camera,
                "error": str(exc),
            }

        text = str(response).strip()
        first = text.splitlines()[0].strip().upper() if text else "UNSURE"
        success = first.startswith("SUCCESS")
        status = (
            "success"
            if success
            else "failure"
            if first.startswith("FAILURE")
            else "unsure"
        )
        return {
            "success": success,
            "reward": 1.0 if success else 0.0,
            "method": "vlm_reward",
            "status": status,
            "task": task,
            "backend": backend,
            "model": model,
            "camera": camera,
            "vlm_response": text,
        }

    return {
        "freespace_move": _tool_callable(freespace),
        "select_best_grasp": select_best_grasp,
        "nudge_world_pose_preserving": nudge_world_pose_preserving,
        "nudge": _tool_callable(nudge),
        "segment_object": _tool_callable(segment),
        "segment_all_objects": _tool_callable(segment_all),
        "detect_object": _tool_callable(detect),
        "detect_objects_oneshot": _tool_callable(detect_oneshot),
        "end_detection": _tool_callable(end_detection),
        "list_detections": _tool_callable(list_detections),
        "sample_grasp_pose_anygrasp": _tool_callable(anygrasp),
        "sample_grasp_pose_3d_bb": _tool_callable(grasp_3d_bb),
        "sample_grasp_pose_2d": _tool_callable(grasp_2d),
        "vlm_query": _tool_callable(vlm),
        "get_camera_image": _tool_callable(camera_image),
        "get_camera_intrinsics": _tool_callable(camera_intrinsics),
        "get_camera_extrinsics": _tool_callable(camera_extrinsics),
        "render_rgb": _tool_callable(render_rgb),
        "render_depth": _tool_callable(render_depth),
        "get_robot_state": _tool_callable(robot_state),
        "set_gripper": _tool_callable(set_gripper),
        "set_gripper_fast": set_gripper_fast,
        "open_gripper": _tool_callable(open_gripper),
        "open_gripper_fast": open_gripper_fast,
        "close_gripper": _tool_callable(close_gripper),
        "go_home": _tool_callable(go_home),
        "go_home_fast": go_home_fast,
        "display_rpy_to_quat": display_rpy_to_quat,
        "get_task_info": get_task_info,
    }
