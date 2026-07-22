# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nudge the motherboard until its long/bottom edge is parallel to the table edge.

Minimal top-camera-first utility for real YAM.  It segments the motherboard
from the top camera, estimates the board long-edge orientation in world XY,
and uses the right gripper as a small pusher if the angle is outside tolerance.

Run with:
  uv run python run_script.py robot=real_yam \
      script_file=cap/saved_scripts/gpu/gpu_nudge_motherboard_parallel.py \
      env.name=yam-real skill_library_path=cap/saved_scripts/skill_library
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
from pathlib import Path
import time

import numpy as np


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default if raw is None or raw == "" else raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default if raw is None or raw == "" else raw)


def _env_float_list(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return [float(v) for v in default]
    values = [float(part) for part in raw.replace(",", " ").split()]
    if len(values) != len(default):
        raise RuntimeError(f"{name} must contain {len(default)} values, got {raw!r}")
    return values


def _load_tool_namespace() -> None:
    try:
        ns = importlib.import_module("skill_library.namespace")
    except Exception:
        ns = None
    for name in [
        "close_gripper",
        "freespace_move",
        "get_camera_extrinsics",
        "get_camera_image",
        "get_camera_intrinsics",
        "get_robot_state",
        "go_home",
        "open_gripper",
        "segment_object",
        "set_gripper",
    ]:
        if name not in globals() and ns is not None and hasattr(ns, name):
            globals()[name] = getattr(ns, name)


_load_tool_namespace()


CAMERA = os.environ.get("GPU_MB_NUDGE_CAMERA", "top").strip() or "top"
MOTHERBOARD_QUERIES = [
    q.strip()
    for q in os.environ.get(
        "GPU_MB_NUDGE_MOTHERBOARD_QUERIES",
        "large mother board,motherboard,large motherboard",
    ).split(",")
    if q.strip()
]

# The target long/bottom edge direction in world XY.  Default is world +X.
TARGET_EDGE_AXIS_WORLD = np.asarray(
    _env_float_list("GPU_MB_NUDGE_TARGET_EDGE_AXIS_WORLD", [1.0, 0.0, 0.0]),
    dtype=np.float64,
)
BOARD_PLANE_Z_M = _env_float("GPU_MB_NUDGE_BOARD_PLANE_Z_M", 0.773)
TABLE_SURFACE_Z_M = _env_float(
    "GPU_MB_NUDGE_TABLE_SURFACE_Z_M",
    _env_float("GPU_TABLE_SURFACE_Z_M", 0.75),
)
ANGLE_TOL_DEG = _env_float("GPU_MB_NUDGE_ANGLE_TOL_DEG", 2.0)
MAX_ATTEMPTS = max(0, _env_int("GPU_MB_NUDGE_MAX_ATTEMPTS", 4))

RIGHT_RPY = _env_float_list("GPU_MB_NUDGE_RIGHT_RPY", [0.0, 125.0, -20.0])
RIGHT_GRIPPER_POS = _env_float("GPU_MB_NUDGE_RIGHT_GRIPPER_POS", 0.08)
RIGHT_GRIPPER_VEL_LIMIT = _env_float("GPU_MB_NUDGE_RIGHT_GRIPPER_VEL_LIMIT", 2.0)
RIGHT_GRIPPER_TORQUE_LIMIT = _env_float("GPU_MB_NUDGE_RIGHT_GRIPPER_TORQUE_LIMIT", 0.6)

CONTACT_Z_M = _env_float("GPU_MB_NUDGE_CONTACT_Z_M", TABLE_SURFACE_Z_M + 0.008)
CONTACT_PLANNER_BACKEND = os.environ.get(
    "GPU_MB_NUDGE_CONTACT_PLANNER_BACKEND",
    "rrtconnect",
).strip()
CARTESIAN_STEP_M = _env_float("GPU_MB_NUDGE_CARTESIAN_STEP_M", 0.006)
CARTESIAN_SPEED_MPS = _env_float("GPU_MB_NUDGE_CARTESIAN_SPEED_MPS", 0.040)
HOVER_Z_OFFSET_M = _env_float("GPU_MB_NUDGE_HOVER_Z_OFFSET_M", 0.085)
OUTSIDE_EDGE_M = _env_float("GPU_MB_NUDGE_OUTSIDE_EDGE_M", 0.012)
PUSH_THROUGH_M = _env_float("GPU_MB_NUDGE_PUSH_THROUGH_M", 0.008)
END_FRACTION = _env_float("GPU_MB_NUDGE_END_FRACTION", 0.72)
PUSH_SCALE = _env_float("GPU_MB_NUDGE_PUSH_SCALE", 1.0)
PUSH_DIR_WORLD = np.asarray(
    _env_float_list("GPU_MB_NUDGE_PUSH_DIR_WORLD", [0.0, 1.0, 0.0]),
    dtype=np.float64,
)
SETTLE_S = _env_float("GPU_MB_NUDGE_SETTLE_S", 0.40)
PLANNING_SPEED = _env_float("GPU_MB_NUDGE_PLANNING_SPEED", 0.45)
IK_ERROR_THRESHOLD = _env_float("GPU_MB_NUDGE_IK_ERROR_THRESHOLD", 0.008)
RIGHT_REACH_MAX_Y_M = _env_float("GPU_MB_NUDGE_RIGHT_REACH_MAX_Y_M", 0.18)
RIGHT_REACH_MIN_X_M = _env_float("GPU_MB_NUDGE_RIGHT_REACH_MIN_X_M", 0.25)
RIGHT_REACH_MAX_X_M = _env_float("GPU_MB_NUDGE_RIGHT_REACH_MAX_X_M", 0.70)
PREVIEW_CANDIDATES = _env_flag("GPU_MB_NUDGE_PREVIEW_CANDIDATES", True)

GO_HOME_ON_START = _env_flag("GPU_MB_NUDGE_GO_HOME_ON_START", True)
GO_HOME_AFTER_EACH_NUDGE = _env_flag(
    "GPU_MB_NUDGE_GO_HOME_AFTER_EACH_NUDGE",
    True,
)
GO_HOME_ON_DONE = _env_flag("GPU_MB_NUDGE_GO_HOME_ON_DONE", False)
DRY_RUN = _env_flag("GPU_MB_NUDGE_DRY_RUN", False)
SAVE_ARTIFACTS = _env_flag("GPU_MB_NUDGE_SAVE_ARTIFACTS", True)


def _unit_xy(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3).copy()
    arr[2] = 0.0
    norm = float(np.linalg.norm(arr[:2]))
    if norm < 1e-9:
        if fallback is None:
            raise RuntimeError(f"zero XY vector: {vec}")
        arr = np.asarray(fallback, dtype=np.float64).reshape(3).copy()
        arr[2] = 0.0
        norm = float(np.linalg.norm(arr[:2]))
        if norm < 1e-9:
            raise RuntimeError(f"invalid XY fallback vector: {fallback}")
    return arr / norm


TARGET_AXIS = _unit_xy(TARGET_EDGE_AXIS_WORLD, fallback=np.array([1.0, 0.0, 0.0]))


def _cross_z(a: np.ndarray, b: np.ndarray) -> float:
    return float(float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]))


def _seg_attr(seg, name: str, default=None):
    if isinstance(seg, dict):
        return seg.get(name, default)
    return getattr(seg, name, default)


def _as_uint8_rgb(image):
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise RuntimeError("camera image is not an RGB-like array")
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr.copy()


def _camera_matrix(camera: str) -> np.ndarray:
    intr = get_camera_intrinsics(camera=camera)
    if isinstance(intr, dict):
        if "K" in intr:
            return np.asarray(intr["K"], dtype=np.float64).reshape(3, 3)
        if "intrinsics" in intr:
            fx, fy, cx, cy = [float(v) for v in intr["intrinsics"]]
        else:
            fx = float(intr["fx"])
            fy = float(intr["fy"])
            cx = float(intr["cx"])
            cy = float(intr["cy"])
    else:
        fx, fy, cx, cy = [
            float(v)
            for v in np.asarray(intr, dtype=np.float64).reshape(-1)[:4]
        ]
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera_T_cam_world(camera: str) -> np.ndarray:
    extr = get_camera_extrinsics(camera=camera)
    if isinstance(extr, dict) and "T_cam_world" in extr:
        return np.asarray(extr["T_cam_world"], dtype=np.float64).reshape(4, 4)
    rot = np.asarray(extr["rotation"], dtype=np.float64).reshape(3, 3)
    pos = np.asarray(extr["position"], dtype=np.float64).reshape(3)
    needs_optical_flip = bool(extr.get("needs_optical_flip", True))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot @ np.diag([-1.0, -1.0, 1.0]) if needs_optical_flip else rot
    T[:3, 3] = pos
    return T


def _project_pixels_to_plane_world(
    pixels_xy: np.ndarray,
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
    plane_z_m: float,
) -> np.ndarray:
    pix = np.asarray(pixels_xy, dtype=np.float64).reshape(-1, 2)
    fx = float(cam_K[0, 0])
    fy = float(cam_K[1, 1])
    cx = float(cam_K[0, 2])
    cy = float(cam_K[1, 2])
    rays_cam = np.column_stack(
        [
            (pix[:, 0] - cx) / fx,
            (pix[:, 1] - cy) / fy,
            np.ones(pix.shape[0], dtype=np.float64),
        ]
    )
    origin_world = np.asarray(T_cam_world[:3, 3], dtype=np.float64).reshape(3)
    rays_world = (np.asarray(T_cam_world[:3, :3], dtype=np.float64) @ rays_cam.T).T
    dz = rays_world[:, 2]
    valid = np.abs(dz) > 1e-9
    scale = np.full(pix.shape[0], np.nan, dtype=np.float64)
    scale[valid] = (float(plane_z_m) - float(origin_world[2])) / dz[valid]
    valid &= scale > 0.0
    points = origin_world[None, :] + rays_world * scale[:, None]
    points = points[valid]
    if points.shape[0] < 20:
        raise RuntimeError("too few projected motherboard mask pixels")
    return points


def _segment_motherboard(camera: str):
    last_error = None
    for query in MOTHERBOARD_QUERIES:
        try:
            seg = segment_object(query=query, camera=camera, score_thresh=0.1)
            mask = _seg_attr(seg, "mask")
            mask_bool = np.asarray(mask) > 0 if mask is not None else None
            if mask_bool is None or mask_bool.ndim != 2 or not np.any(mask_bool):
                raise RuntimeError("empty motherboard mask")
            return {
                "query": query,
                "mask": mask_bool,
                "score": _seg_attr(seg, "score"),
                "bbox_xywh": _seg_attr(seg, "bbox_xywh"),
            }
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"could not segment motherboard; last_error={last_error}")


def _fit_board_pose(points_world: np.ndarray) -> dict:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    xy = points[:, :2]
    centroid = np.median(xy, axis=0)
    centered = xy - centroid
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    major = evecs[:, order[0]]
    if float(np.dot(major, TARGET_AXIS[:2])) < 0.0:
        major = -major
    minor = np.asarray([-major[1], major[0]], dtype=np.float64)

    proj_major = centered @ major
    proj_minor = centered @ minor
    major_lo, major_hi = np.percentile(proj_major, [2.0, 98.0])
    minor_lo, minor_hi = np.percentile(proj_minor, [2.0, 98.0])
    center_xy = (
        centroid
        + 0.5 * (major_lo + major_hi) * major
        + 0.5 * (minor_lo + minor_hi) * minor
    )
    major3 = np.array([major[0], major[1], 0.0], dtype=np.float64)
    minor3 = np.array([minor[0], minor[1], 0.0], dtype=np.float64)
    cross_z = float(TARGET_AXIS[0] * major3[1] - TARGET_AXIS[1] * major3[0])
    dot = float(np.clip(np.dot(TARGET_AXIS[:2], major3[:2]), -1.0, 1.0))
    angle_deg = math.degrees(math.atan2(cross_z, dot))
    return {
        "center_world": np.array([center_xy[0], center_xy[1], BOARD_PLANE_Z_M]),
        "major_axis_world": major3,
        "minor_axis_world": minor3,
        "half_major_m": 0.5 * float(major_hi - major_lo),
        "half_minor_m": 0.5 * float(minor_hi - minor_lo),
        "angle_error_deg": float(angle_deg),
        "num_points": int(points.shape[0]),
    }


def _observe_board() -> dict:
    rgb = _as_uint8_rgb(get_camera_image(camera=CAMERA))
    seg = _segment_motherboard(CAMERA)
    mask = seg["mask"]
    ys, xs = np.where(mask)
    if xs.size > 9000:
        step = max(1, int(math.ceil(xs.size / 9000)))
        xs = xs[::step]
        ys = ys[::step]
    pixels = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    cam_K = _camera_matrix(CAMERA)
    T_cam_world = _camera_T_cam_world(CAMERA)
    points = _project_pixels_to_plane_world(
        pixels,
        cam_K,
        T_cam_world,
        BOARD_PLANE_Z_M,
    )
    pose = _fit_board_pose(points)
    pose["seg"] = seg
    pose["rgb"] = rgb
    return pose


def _move_right(
    pos,
    rpy=RIGHT_RPY,
    *,
    speed=PLANNING_SPEED,
    preview_only=False,
    planner_backend: str | None = None,
):
    pos = [float(v) for v in pos]
    rpy = [float(v) for v in rpy]
    print(
        "[gpu_mb_nudge] move right: "
        f"pos={[round(v, 4) for v in pos]} rpy={[round(v, 2) for v in rpy]} "
        f"speed={float(speed):.2f} preview={bool(preview_only)} "
        f"planner_backend={planner_backend or 'default'}"
    )
    if DRY_RUN:
        return
    kwargs = {
        "right_target_pos": pos,
        "right_target_rpy": rpy,
        "right_gripper": float(RIGHT_GRIPPER_POS),
        "planning_speed": float(speed),
        "ik_error_threshold": float(IK_ERROR_THRESHOLD),
        "preview_only": bool(preview_only),
    }
    if planner_backend:
        kwargs["planner_backend"] = planner_backend
    freespace_move(**kwargs)


def _prepare_right_pusher():
    print(
        "[gpu_mb_nudge] prepare right gripper pusher: "
        f"pos={RIGHT_GRIPPER_POS:.3f} torque_limit={RIGHT_GRIPPER_TORQUE_LIMIT:.3f}"
    )
    if DRY_RUN:
        return
    if "set_gripper" in globals():
        set_gripper(
            "right",
            float(RIGHT_GRIPPER_POS),
            vel_limit=float(RIGHT_GRIPPER_VEL_LIMIT),
            torque_limit=float(RIGHT_GRIPPER_TORQUE_LIMIT),
        )
    elif RIGHT_GRIPPER_POS <= 0.2 and "close_gripper" in globals():
        close_gripper(
            "right",
            vel_limit=float(RIGHT_GRIPPER_VEL_LIMIT),
            torque_limit=float(RIGHT_GRIPPER_TORQUE_LIMIT),
        )
    elif "open_gripper" in globals():
        open_gripper(
            "right",
            vel_limit=float(RIGHT_GRIPPER_VEL_LIMIT),
            torque_limit=float(RIGHT_GRIPPER_TORQUE_LIMIT),
        )


def _tool_env_from_callable(fn):
    seen = set()

    def _search(obj):
        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)

        env = getattr(obj, "_env", None)
        if env is not None:
            return env

        wrapped = getattr(obj, "__wrapped__", None)
        if wrapped is not None:
            found = _search(wrapped)
            if found is not None:
                return found

        closure = getattr(obj, "__closure__", None) or ()
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            found = _search(value)
            if found is not None:
                return found

        func = getattr(obj, "func", None)
        if func is not None:
            found = _search(func)
            if found is not None:
                return found

        return None

    return _search(inspect.unwrap(fn))


def _guided_right_cartesian_path(target_points, *, label: str) -> None:
    points = [np.asarray(point, dtype=np.float64).reshape(3) for point in target_points]
    if DRY_RUN or not points:
        return
    env = _tool_env_from_callable(get_robot_state)
    if env is None or not hasattr(env, "move_bimanual_joint_keypoints"):
        raise RuntimeError("direct YAM env not available for guided motherboard nudge")

    obs_left = env.get_observations("left")
    obs_right = env.get_observations("right")
    left_jp = np.asarray(obs_left["joint_pos"], dtype=np.float64).reshape(6)
    right_jp = np.asarray(obs_right["joint_pos"], dtype=np.float64).reshape(6)
    left_gp = float(np.asarray(obs_left["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    right_gp = float(RIGHT_GRIPPER_POS)
    left_pos = np.asarray(obs_left["ee_pos"], dtype=np.float64).reshape(3)
    left_quat = np.asarray(obs_left["ee_quat"], dtype=np.float64).reshape(4)
    right_pos = np.asarray(obs_right["ee_pos"], dtype=np.float64).reshape(3)
    right_quat = np.asarray(obs_right["ee_quat"], dtype=np.float64).reshape(4)

    expanded_points = []
    cursor = right_pos.copy()
    for point in points:
        delta = point - cursor
        dist = float(np.linalg.norm(delta))
        segments = max(1, int(math.ceil(dist / max(float(CARTESIAN_STEP_M), 1e-4))))
        for index in range(1, segments + 1):
            alpha = float(index) / float(segments)
            expanded_points.append(cursor + alpha * delta)
        cursor = point

    left_waypoints = []
    right_waypoints = []
    left_grippers = []
    right_grippers = []
    timestamps = []
    elapsed_s = 0.0
    prev_point = right_pos.copy()

    with env._kin_lock:
        env.kin.forward_kinematics(left_jp, right_jp)
        cur_left_jp = left_jp.copy()
        cur_right_jp = right_jp.copy()
        for point in expanded_points:
            elapsed_s += float(np.linalg.norm(point - prev_point)) / max(
                float(CARTESIAN_SPEED_MPS),
                1e-3,
            )
            env.kin.forward_kinematics(cur_left_jp, cur_right_jp)
            next_left_jp, next_right_jp = env.kin.inverse_kinematics(
                left_pos,
                left_quat,
                point,
                right_quat,
                seeded=True,
                dt=0.01,
                solver="daqp",
                damping=1e-3,
                err_threshold=1e-4,
                max_iters=40,
            )
            cur_left_jp = np.asarray(next_left_jp, dtype=np.float64).reshape(6)
            cur_right_jp = np.asarray(next_right_jp, dtype=np.float64).reshape(6)
            left_waypoints.append(cur_left_jp.copy())
            right_waypoints.append(cur_right_jp.copy())
            left_grippers.append([left_gp])
            right_grippers.append([right_gp])
            timestamps.append(float(elapsed_s))
            prev_point = point

    print(
        "[gpu_mb_nudge] guided right Cartesian nudge: "
        f"label={label!r} waypoints={len(timestamps)} "
        f"speed_mps={float(CARTESIAN_SPEED_MPS):.3f} "
        f"step_m={float(CARTESIAN_STEP_M):.4f} "
        f"duration_s={(timestamps[-1] if timestamps else 0.0):.2f}"
    )
    result = env.move_bimanual_joint_keypoints(
        timestamps=timestamps,
        left_joint_positions=left_waypoints,
        right_joint_positions=right_waypoints,
        left_gripper_positions=left_grippers,
        right_gripper_positions=right_grippers,
        playback_speed=1.0,
        command_hz=60.0,
        start_interp_s=0.15,
    )
    if not bool(result.get("success", False)):
        print(
            "[gpu_mb_nudge] guided right Cartesian nudge failed: "
            f"label={label!r} reason={result.get('reason', 'unknown')}"
        )
        raise RuntimeError(result.get("reason", "guided right Cartesian nudge failed"))


def _go_home_for_clear_view(context: str, *, force: bool = False) -> None:
    if DRY_RUN or (not force and not GO_HOME_AFTER_EACH_NUDGE):
        return
    if "go_home" not in globals():
        print(f"[gpu_mb_nudge] {context}: go_home unavailable; skipping clear-view home")
        return
    print(f"[gpu_mb_nudge] {context}: go home before next top-camera observation")
    go_home()
    if SETTLE_S > 0.0:
        time.sleep(float(SETTLE_S))


def _point_reach_penalty(point: np.ndarray) -> float:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    penalty = 0.0
    penalty += 1000.0 * max(0.0, float(p[1]) - float(RIGHT_REACH_MAX_Y_M))
    penalty += 1000.0 * max(0.0, float(RIGHT_REACH_MIN_X_M) - float(p[0]))
    penalty += 1000.0 * max(0.0, float(p[0]) - float(RIGHT_REACH_MAX_X_M))
    return float(penalty)


def _candidate_reach_penalty(candidate: dict) -> float:
    return max(
        _point_reach_penalty(candidate["hover_start"]),
        _point_reach_penalty(candidate["contact_start"]),
        _point_reach_penalty(candidate["contact_end"]),
    )


def _preview_candidate(candidate: dict) -> tuple[bool, str]:
    if DRY_RUN or not PREVIEW_CANDIDATES:
        return True, "preview_disabled"
    try:
        _move_right(
            candidate["hover_start"],
            speed=min(float(PLANNING_SPEED), 0.25),
            preview_only=True,
        )
    except Exception as exc:
        return False, f"hover_start: {exc}"
    return True, "preview_ok"


def _build_nudge_candidates(pose: dict) -> list[dict]:
    angle_deg = float(pose["angle_error_deg"])
    center = np.asarray(pose["center_world"], dtype=np.float64).reshape(3)
    major = _unit_xy(pose["major_axis_world"], fallback=TARGET_AXIS)
    minor = _unit_xy(pose["minor_axis_world"], fallback=np.array([0.0, 1.0, 0.0]))
    half_major = float(pose["half_major_m"])
    half_minor = float(pose["half_minor_m"])
    outside_m = max(0.0, float(OUTSIDE_EDGE_M))
    push_m = max(0.0, float(PUSH_THROUGH_M))

    desired_tau_sign = -1.0 if angle_deg > 0.0 else 1.0
    correction_mode = "clockwise" if desired_tau_sign < 0.0 else "counter_clockwise"
    push_dir = _unit_xy(
        PUSH_DIR_WORLD * float(PUSH_SCALE),
        fallback=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    table_side_dir = minor if float(np.dot(minor, push_dir)) < 0.0 else -minor
    fractions = [
        max(0.15, min(0.95, float(END_FRACTION))),
        0.55,
        0.35,
        0.85,
    ]
    side_fractions = [0.0, 0.5, 1.0]

    candidates = []
    rejected = []
    for end_sign in (-1.0, 1.0):
        for fraction in fractions:
            for side_fraction in side_fractions:
                end_point = center + end_sign * major * half_major * fraction
                edge_point = end_point + table_side_dir * half_minor * side_fraction
                start = edge_point - push_dir * outside_m
                contact = edge_point + push_dir * push_m
                hover_start = start.copy()
                hover_start[2] = float(BOARD_PLANE_Z_M + HOVER_Z_OFFSET_M)
                start[2] = float(CONTACT_Z_M)
                contact[2] = float(CONTACT_Z_M)
                hover_end = contact.copy()
                hover_end[2] = float(BOARD_PLANE_Z_M + HOVER_Z_OFFSET_M)
                torque = _cross_z(edge_point - center, push_dir)
                if torque * desired_tau_sign <= 1e-7:
                    rejected.append((end_sign, fraction, side_fraction, torque))
                    continue
                reach_penalty = _candidate_reach_penalty(
                    {
                        "hover_start": hover_start,
                        "contact_start": start,
                        "contact_end": contact,
                    }
                )
                target_side = float(np.dot(edge_point - center, TARGET_AXIS))
                score = reach_penalty - abs(float(torque)) - 0.002 * float(side_fraction)
                candidates.append(
                    {
                        "score": float(score),
                        "correction_mode": correction_mode,
                        "desired_tau_z_sign": float(desired_tau_sign),
                        "torque_z": float(torque),
                        "reach_penalty": float(reach_penalty),
                        "end_label": (
                            "target_positive" if target_side > 0.0 else "target_negative"
                        ),
                        "end_sign": float(end_sign),
                        "fraction": float(fraction),
                        "side_fraction": float(side_fraction),
                        "push_dir": push_dir,
                        "table_side_dir": table_side_dir,
                        "edge_point": edge_point,
                        "hover_start": hover_start,
                        "contact_start": start,
                        "contact_end": contact,
                        "hover_end": hover_end,
                    }
                )
    if not candidates:
        details = ", ".join(
            f"end={end_sign:+.0f}/frac={fraction:.2f}/side={side_fraction:.1f}/tau={torque:+.5f}"
            for end_sign, fraction, side_fraction, torque in rejected[:8]
        )
        raise RuntimeError(
            "no motherboard nudge candidate produces the desired corrective torque; "
            f"angle_error_deg={angle_deg:+.2f} desired_tau_sign={desired_tau_sign:+.0f} "
            f"rejected=[{details}]"
        )
    candidates.sort(key=lambda item: item["score"])
    return candidates


def _select_nudge_candidate(pose: dict) -> dict:
    candidates = _build_nudge_candidates(pose)
    preview_failures = []
    for idx, candidate in enumerate(candidates):
        ok, reason = _preview_candidate(candidate)
        if ok:
            print(
                "[gpu_mb_nudge] selected right-arm nudge candidate "
                f"idx={idx} score={candidate['score']:.3f} "
                f"mode={candidate['correction_mode']} "
                f"end={candidate['end_label']} "
                f"desired_tau_sign={candidate['desired_tau_z_sign']:+.0f} "
                f"torque_z={candidate['torque_z']:+.5f} "
                f"reach_penalty={candidate['reach_penalty']:.3f} "
                f"fraction={candidate['fraction']:.2f} "
                f"side_fraction={candidate['side_fraction']:.1f}"
            )
            return candidate
        preview_failures.append(
            f"candidate {idx}: {reason} "
            f"hover_start={[round(float(v), 4) for v in candidate['hover_start'].tolist()]}"
        )
    raise RuntimeError(
        "no reachable right-arm motherboard nudge candidate; "
        + "; ".join(preview_failures[:6])
    )


def _nudge_once(pose: dict, attempt: int) -> dict:
    angle_deg = float(pose["angle_error_deg"])
    candidate = _select_nudge_candidate(pose)
    push_dir = np.asarray(candidate["push_dir"], dtype=np.float64).reshape(3)
    hover_start = np.asarray(candidate["hover_start"], dtype=np.float64).reshape(3)
    start = np.asarray(candidate["contact_start"], dtype=np.float64).reshape(3)
    contact = np.asarray(candidate["contact_end"], dtype=np.float64).reshape(3)
    hover_end = np.asarray(candidate["hover_end"], dtype=np.float64).reshape(3)

    print(
        "[gpu_mb_nudge] correction "
        f"attempt={attempt} angle_error_deg={angle_deg:+.2f} "
        f"mode={candidate['correction_mode']} "
        f"end={candidate['end_label']} "
        f"desired_tau_sign={candidate['desired_tau_z_sign']:+.0f} "
        f"torque_z={candidate['torque_z']:+.5f} "
        f"push_dir={[round(float(v), 4) for v in push_dir.tolist()]} "
        f"side_fraction={candidate['side_fraction']:.1f} "
        f"hover_start={[round(float(v), 4) for v in hover_start.tolist()]} "
        f"contact_start={[round(float(v), 4) for v in start.tolist()]} "
        f"contact_end={[round(float(v), 4) for v in contact.tolist()]}"
    )
    _prepare_right_pusher()
    _move_right(hover_start)
    _guided_right_cartesian_path(
        [start, contact, hover_end],
        label=f"attempt {attempt} contact push",
    )
    _go_home_for_clear_view(f"post-nudge attempt {attempt}")
    if SETTLE_S > 0.0 and not DRY_RUN and not GO_HOME_AFTER_EACH_NUDGE:
        time.sleep(float(SETTLE_S))
    return {
        "attempt": int(attempt),
        "angle_error_deg": angle_deg,
        "correction_mode": str(candidate["correction_mode"]),
        "desired_tau_z_sign": float(candidate["desired_tau_z_sign"]),
        "end_label": str(candidate["end_label"]),
        "fraction": float(candidate["fraction"]),
        "side_fraction": float(candidate["side_fraction"]),
        "push_dir": [float(v) for v in push_dir.tolist()],
        "hover_start": [float(v) for v in hover_start.tolist()],
        "contact_start": [float(v) for v in start.tolist()],
        "contact_end": [float(v) for v in contact.tolist()],
        "hover_end": [float(v) for v in hover_end.tolist()],
    }


def _draw_line_rgb(canvas, p0, p1, color, width=3):
    arr = np.asarray(canvas)
    h, w = arr.shape[:2]
    steps = max(
        1,
        int(
            math.ceil(
                max(abs(float(p1[0]) - float(p0[0])), abs(float(p1[1]) - float(p0[1])))
            )
        ),
    )
    xs = np.linspace(float(p0[0]), float(p1[0]), steps + 1)
    ys = np.linspace(float(p0[1]), float(p1[1]), steps + 1)
    r = max(0, int(width) // 2)
    for x_f, y_f in zip(xs, ys):
        x = int(round(x_f))
        y = int(round(y_f))
        if not (0 <= x < w and 0 <= y < h):
            continue
        arr[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)] = color


def _project_world_to_pixel(point_world, cam_K, T_cam_world):
    T_world_cam = np.linalg.inv(T_cam_world)
    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = np.asarray(point_world, dtype=np.float64).reshape(3)
    point_cam = T_world_cam @ point_h
    if float(point_cam[2]) <= 1e-6:
        return None
    u = float(cam_K[0, 0]) * float(point_cam[0]) / float(point_cam[2]) + float(cam_K[0, 2])
    v = float(cam_K[1, 1]) * float(point_cam[1]) / float(point_cam[2]) + float(cam_K[1, 2])
    return np.asarray([u, v], dtype=np.float64)


def _save_ppm(path: Path, rgb: np.ndarray) -> Path:
    arr = _as_uint8_rgb(rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape[:2]
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr.tobytes())
    return path


def _save_artifacts(records: list[dict], final_pose: dict, nudges: list[dict]) -> None:
    if not SAVE_ARTIFACTS:
        return
    out_dir = Path("logs") / "gpu_motherboard_nudge_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    report_path = out_dir / f"motherboard_nudge_{ts}.json"
    image_path = out_dir / f"motherboard_nudge_{ts}.ppm"

    cam_K = _camera_matrix(CAMERA)
    T_cam_world = _camera_T_cam_world(CAMERA)
    overlay = _as_uint8_rgb(final_pose["rgb"])
    center = np.asarray(final_pose["center_world"], dtype=np.float64).reshape(3)
    major = _unit_xy(final_pose["major_axis_world"], fallback=TARGET_AXIS)
    half_major = float(final_pose["half_major_m"])
    p0 = _project_world_to_pixel(center - major * half_major, cam_K, T_cam_world)
    p1 = _project_world_to_pixel(center + major * half_major, cam_K, T_cam_world)
    if p0 is not None and p1 is not None:
        _draw_line_rgb(overlay, p0, p1, np.array([80, 255, 80], dtype=np.uint8), width=4)

    serializable_records = []
    for rec in records:
        serializable_records.append(
            {
                "attempt": int(rec["attempt"]),
                "angle_error_deg": round(float(rec["angle_error_deg"]), 4),
                "center_world": [
                    round(float(v), 5)
                    for v in np.asarray(rec["center_world"], dtype=np.float64).reshape(3)
                ],
                "major_axis_world": [
                    round(float(v), 5)
                    for v in np.asarray(rec["major_axis_world"], dtype=np.float64).reshape(3)
                ],
                "half_major_m": round(float(rec["half_major_m"]), 5),
                "half_minor_m": round(float(rec["half_minor_m"]), 5),
                "query": rec.get("query"),
                "seg_score": rec.get("seg_score"),
            }
        )
    report = {
        "camera": CAMERA,
        "target_edge_axis_world": [float(v) for v in TARGET_AXIS.tolist()],
        "board_plane_z_m": float(BOARD_PLANE_Z_M),
        "angle_tol_deg": float(ANGLE_TOL_DEG),
        "dry_run": bool(DRY_RUN),
        "success": abs(float(final_pose["angle_error_deg"])) <= float(ANGLE_TOL_DEG),
        "records": serializable_records,
        "nudges": nudges,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _save_ppm(image_path, overlay)
    print(f"[gpu_mb_nudge] artifacts: report={report_path} overlay={image_path}")


def _record_pose(attempt: int, pose: dict) -> dict:
    return {
        "attempt": int(attempt),
        "angle_error_deg": float(pose["angle_error_deg"]),
        "center_world": np.asarray(pose["center_world"], dtype=np.float64).reshape(3),
        "major_axis_world": np.asarray(pose["major_axis_world"], dtype=np.float64).reshape(3),
        "half_major_m": float(pose["half_major_m"]),
        "half_minor_m": float(pose["half_minor_m"]),
        "query": pose["seg"]["query"],
        "seg_score": pose["seg"].get("score"),
    }


def main() -> None:
    print(
        "[gpu_mb_nudge] Starting motherboard parallel nudge loop: "
        f"camera={CAMERA!r} target_axis={[round(float(v), 3) for v in TARGET_AXIS.tolist()]} "
        f"tol_deg={ANGLE_TOL_DEG:.2f} max_attempts={MAX_ATTEMPTS} dry_run={DRY_RUN}"
    )
    if GO_HOME_ON_START:
        _go_home_for_clear_view("start", force=True)
    records = []
    nudges = []
    final_pose = None
    for attempt in range(0, MAX_ATTEMPTS + 1):
        pose = _observe_board()
        final_pose = pose
        records.append(_record_pose(attempt, pose))
        print(
            "[gpu_mb_nudge] observation "
            f"attempt={attempt} query={pose['seg']['query']!r} "
            f"angle_error_deg={float(pose['angle_error_deg']):+.2f} "
            f"center={[round(float(v), 4) for v in pose['center_world'].tolist()]} "
            f"major={[round(float(v), 4) for v in pose['major_axis_world'].tolist()]}"
        )
        if abs(float(pose["angle_error_deg"])) <= float(ANGLE_TOL_DEG):
            print("[gpu_mb_nudge] Success: motherboard edge is within tolerance.")
            break
        if attempt >= MAX_ATTEMPTS:
            print("[gpu_mb_nudge] Stop: max correction attempts reached.")
            break
        nudges.append(_nudge_once(pose, attempt + 1))

    if final_pose is not None:
        _save_artifacts(records, final_pose, nudges)
    if GO_HOME_ON_DONE and not DRY_RUN:
        print("[gpu_mb_nudge] go_home at end")
        go_home()


main()

