#!/usr/bin/env python3
"""Standalone AnyGrasp debug UI.

This intentionally shows two pose views for each grasp candidate:

1. Raw / native AnyGrasp pose in world frame.
2. Planner-aligned pose using the same convention expected by this repo
   (the one intended for freespace_move).

The overlay image remains native AnyGrasp output, while the pose table makes the
robot-specific transform explicit instead of hiding it.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import inspect
import io
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enpire.env.forge.tools._bootstrap import maybe_reexec_with_uv

maybe_reexec_with_uv(
    __file__, REPO_ROOT, required_modules=["cv2", "requests", "uvicorn"]
)

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from enpire.env.forge.cap.config import (
    ANYGRASP_MIN_PLANNER_Z_M,
    GRIPPER_SETTLE_TIMEOUT_S,
    GRIPPER_TCP_OFFSET_Z_M,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message="Using Portal from plain Python threads is discouraged because they can cause hangs during shutdown.",
    category=UserWarning,
)

DEBUG_PORT = int(os.getenv("DEBUG_PORT", "8121"))
ANYGRASP_URL = os.getenv(
    "ANYGRASP_URL", os.getenv("ANYGRASP_SERVICE_URL", "http://localhost:8122")
).rstrip("/")
SAM3_URL = os.getenv("SAM3_URL", "http://localhost:6767").rstrip("/")
DEFAULT_OBJECT_INPUT_MODE = (
    os.getenv("ANYGRASP_OBJECT_INPUT_MODE", "segmented_object_cloud").strip().lower()
)
CAP_HOST = os.getenv("CAP_HOST", "localhost")
CAP_PORT = int(os.getenv("CAP_PORT", "8300"))
TWO_D_TOP_DOWN_Z_M = float(os.getenv("ANYGRASP_2D_TOP_DOWN_Z_M", "0.79"))
LEFT_HOME_XYZ = [0.4975, 0.31, 0.914]
RIGHT_HOME_XYZ = [0.4975, -0.31, 0.914]
HOME_VIEW_Z_OFFSET = 0.15
LEFT_BIRDEYE_VIEW_RPY = [0.0, 125.0, 20.0]
RIGHT_BIRDEYE_VIEW_RPY = [0.0, 125.0, -20.0]
DEFAULT_IK_THRESHOLD_M = 0.005
MAX_SORT_IK_GRASPS = 16
DEFAULT_SOLVER_SPEED = "fast"
_PREVIEW_STATE_TOLERANCE_RAD = float(
    os.getenv("ANYGRASP_PREVIEW_STATE_TOLERANCE_RAD", "0.05")
)
_TWO_D_MAJOR_AXIS_RATIO_THRESHOLD = 1.25
_TWO_D_CENTER_ONLY_YAWS_DEG = (0.0, 180.0, 90.0, -90.0)
_TWO_D_LOCAL_AXIS_MIN_RADIUS_PX = 6.0
_TWO_D_LOCAL_AXIS_MAX_RADIUS_PX = 32.0
_TWO_D_LOCAL_AXIS_RADIUS_FRACTION = 0.18
_TWO_D_LOCAL_AXIS_MIN_NEIGHBOR_POINTS = 12
_LOCAL_POSE_SCRIPT_PATH = REPO_ROOT / "cap/saved_scripts/small_object/local_pose.py"
_LOCAL_POSE_DEFAULTS: dict[str, Any] = {
    "LEFT_TARGET_POS": [0.6, 0.15, 1.00],
    "LEFT_TARGET_RPY": [0.0, 160.0, 0.0],
    "RIGHT_TARGET_POS": [0.6, -0.15, 1.00],
    "RIGHT_TARGET_RPY": [0.0, 160.0, 0.0],
    "PLANNING_SPEED": 1.5,
    "IK_ERROR_THRESHOLD_M": 0.01,
    "IK_XYZ_WEIGHT": 1.0,
    "IK_RPY_WEIGHT": 0.3,
    "MOTION_PLANNER_BACKEND": "curobo",
}
_local_pose_config_cache: dict[str, Any] | None = None

# See cap/agent/tools/grasp_anygrasp.py for derivation.
_ANYGRASP_TO_GRIPPER = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
_ANYGRASP_TO_GRIPPER_T = np.eye(4, dtype=np.float64)
_ANYGRASP_TO_GRIPPER_T[:3, :3] = _ANYGRASP_TO_GRIPPER


def _quat_xyzw_to_display_rpy_deg(quat_xyzw: np.ndarray) -> list[float]:
    """Inverse of freespace_move._display_rpy_to_quat for planner-facing UI output."""
    from scipy.spatial.transform import Rotation

    ex, ey, ez = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_euler(
        "xyz", degrees=True
    )
    disp = np.array([ey, -ex, -ez - 90.0], dtype=np.float64)
    disp = (disp + 180.0) % 360.0 - 180.0
    return disp.tolist()


_executor = ThreadPoolExecutor(max_workers=1)
_portal_client = None
_preview_trajectory_cache_lock = threading.Lock()
_preview_trajectory_cache: dict[tuple[Any, ...], dict[str, Any]] = {}


class GraspPoseRow(BaseModel):
    rank: int
    score: float
    width: float
    raw_xyz: list[float] = Field(default_factory=list)
    raw_rpy: list[float] = Field(default_factory=list)
    planner_xyz: list[float] = Field(default_factory=list)
    planner_rpy: list[float] = Field(default_factory=list)
    two_d_xyz: list[float] = Field(default_factory=list)
    two_d_rpy: list[float] = Field(default_factory=list)
    ik_error_m: float | None = None
    ik_rot_error_deg: float | None = None
    within_ik_threshold: bool | None = None
    motion_plan_error: bool | None = None
    motion_plan_reason: str | None = None
    trajectory_cache_key: str | None = None
    trajectory_cache_pose_mode: str | None = None
    trajectory_steps: int = 0


class RunRequest(BaseModel):
    prompt: str
    camera: str = "top"
    max_grasps: int = 10
    z_range_min: float = 1e-6
    z_range_max: float = 1.5
    workspace_margin: float = 0.02
    collision_detection: bool = True
    object_input_mode: str = "segmented_object_cloud"
    tcp_offset_z_m: float = GRIPPER_TCP_OFFSET_Z_M
    disable_planner_z_clipping: bool = False


class RunResponse(BaseModel):
    status: str  # ok | no_grasps | error
    error: str | None = None
    n_grasps: int = 0
    best_score: float | None = None
    overlay_b64: str | None = None
    cloud_preview_url: str | None = None
    image_width: int = 0
    image_height: int = 0
    latency_ms: float = 0.0
    object_input_mode: str = "segmented_object_cloud"
    segmented_cloud_z_min_m: float | None = None
    segmented_cloud_z_max_m: float | None = None
    segmented_cloud_thickness_m: float | None = None
    two_d_top_down_z_m: float = TWO_D_TOP_DOWN_Z_M
    tcp_offset_z_m: float = GRIPPER_TCP_OFFSET_Z_M
    planner_z_floor_m: float = ANYGRASP_MIN_PLANNER_Z_M
    planner_z_clipping_enabled: bool = True
    n_planner_z_clipped: int = 0
    grasps: list[GraspPoseRow] = Field(default_factory=list)


class PlannerRequest(BaseModel):
    side: str = "left"
    xyz: list[float]
    rpy: list[float]
    pose_mode: str = "planner"
    solver_speed: str = DEFAULT_SOLVER_SPEED
    planning_speed: float = 3.0
    ik_error_threshold: float = DEFAULT_IK_THRESHOLD_M
    ik_xyz_weight: float = 1.0
    ik_rpy_weight: float = 0.3
    execute: bool = False
    execute_fraction: float = 1.0
    trajectory_cache_key: str | None = None


class PlannerResponse(BaseModel):
    ok: bool
    status: str
    executed: bool = False
    used_cached_preview: bool = False
    execute_fraction: float = 1.0
    preview_cache_state: str = "none"
    preview_cache_message: str | None = None
    trajectory_steps: int = 0
    final_pos_error_m: float = 0.0
    final_rot_error_deg: float = 0.0
    final_pose_error: float = 0.0
    reason: str | None = None
    error: str | None = None
    side: str | None = None
    pose_mode: str | None = None
    start_xyz: list[float] = Field(default_factory=list)
    goal_xyz: list[float] = Field(default_factory=list)
    path_xyz: list[list[float]] = Field(default_factory=list)
    preview_url: str | None = None


class HomeResponse(BaseModel):
    ok: bool
    error: str | None = None


class SortIkRequest(BaseModel):
    side: str = "left"
    pose_mode: str = "2d"
    solver_speed: str = DEFAULT_SOLVER_SPEED
    planning_speed: float = 3.0
    ik_error_threshold: float = DEFAULT_IK_THRESHOLD_M
    ik_xyz_weight: float = 1.0
    ik_rpy_weight: float = 0.3
    grasps: list[GraspPoseRow] = Field(default_factory=list)


class SortIkResponse(BaseModel):
    ok: bool
    error: str | None = None
    side: str | None = None
    pose_mode: str | None = None
    ik_error_threshold: float = DEFAULT_IK_THRESHOLD_M
    grasps: list[GraspPoseRow] = Field(default_factory=list)
    timing_total_ms: float = 0.0
    timing_plan_eval_ms: float = 0.0
    timing_postprocess_ms: float = 0.0
    timing_rank_ms: float = 0.0
    curobo_solve_time_ms: float = 0.0
    curobo_total_time_ms: float = 0.0
    curobo_graph_time_ms: float = 0.0
    curobo_ik_time_ms: float = 0.0
    planning_mode: str = "unknown"
    input_grasp_count: int = 0
    evaluated_grasp_count: int = 0
    truncated_input_count: int = 0
    batch_attempted: bool = False
    batch_error: str | None = None


class PipelineUserError(RuntimeError):
    def __init__(self, status: str, message: str, *, log_level: str = "info"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.log_level = log_level


def _np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_np(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)))


def _round_list(arr: np.ndarray | list[float], digits: int) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(arr, dtype=np.float64).tolist()]


def _normalize_solver_speed(value: Any | None) -> str:
    speed = str(value or DEFAULT_SOLVER_SPEED).strip().lower().replace("_", "-")
    if speed not in {"fast", "slow"}:
        logger.warning(
            "Unknown solver speed %r; falling back to %s",
            value,
            DEFAULT_SOLVER_SPEED,
        )
        return DEFAULT_SOLVER_SPEED
    return speed


def _tool_get_planner_with_solver_speed(
    tool: Any,
    *,
    planner_backend: str = "curobo",
    ik_xyz_weight: float = 1.0,
    ik_rpy_weight: float = 0.3,
    solver_speed: str = DEFAULT_SOLVER_SPEED,
):
    planner_kwargs: dict[str, Any] = {
        "planner_backend": planner_backend,
        "ik_xyz_weight": float(ik_xyz_weight),
        "ik_rpy_weight": float(ik_rpy_weight),
    }
    try:
        sig = inspect.signature(tool._get_planner)
        if "solver_speed" in sig.parameters:
            planner_kwargs["solver_speed"] = _normalize_solver_speed(solver_speed)
    except Exception:
        pass
    return tool._get_planner(**planner_kwargs)


def _depth_to_points(depth: np.ndarray, cam_K: np.ndarray) -> np.ndarray:
    fx = float(cam_K[0, 0])
    fy = float(cam_K[1, 1])
    cx = float(cam_K[0, 2])
    cy = float(cam_K[1, 2])

    xmap = np.arange(depth.shape[1], dtype=np.float32)
    ymap = np.arange(depth.shape[0], dtype=np.float32)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth.astype(np.float32)
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    return np.stack([points_x, points_y, points_z], axis=-1)


def _frame_to_scene(
    rgb: np.ndarray,
    depth: np.ndarray,
    cam_K: np.ndarray,
    *,
    z_range: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points_all = _depth_to_points(depth, cam_K)
    mask = (
        np.isfinite(points_all).all(axis=-1)
        & (points_all[:, :, 2] > float(z_range[0]))
        & (points_all[:, :, 2] < float(z_range[1]))
    )
    points = points_all[mask].astype(np.float32)
    colors = (rgb[mask].astype(np.float32) / 255.0).astype(np.float32)
    return points_all, mask, points, colors


def _render_depth_preview_bgr(
    depth: np.ndarray,
    *,
    zmin: float | None = None,
    zmax: float | None = None,
) -> np.ndarray:
    """Render a human-readable depth preview for the exact map used for cloud generation.

    Invalid pixels are black. When ``zmin``/``zmax`` are provided, points outside the
    AnyGrasp z-range are dimmed so it is easier to compare the depth input with the
    resulting point cloud preview.
    """
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0.0)]

    h, w = depth.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if finite.size == 0:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "No valid depth",
            (16, max(28, h // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return canvas

    lo = (
        float(zmin)
        if zmin is not None and np.isfinite(zmin)
        else float(np.percentile(finite, 5))
    )
    hi = (
        float(zmax)
        if zmax is not None and np.isfinite(zmax)
        else float(np.percentile(finite, 95))
    )
    if hi <= lo:
        hi = lo + 1e-6

    clipped = np.clip(depth, lo, hi)
    normalized = np.nan_to_num(
        (clipped - lo) / (hi - lo) * 255.0,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    ).astype(np.uint8)
    preview = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)

    invalid_mask = ~np.isfinite(depth) | (depth <= 0.0)
    preview[invalid_mask] = (0, 0, 0)

    range_mask = None
    if zmin is not None and zmax is not None and float(zmax) > float(zmin):
        range_mask = (
            np.isfinite(depth) & (depth >= float(zmin)) & (depth <= float(zmax))
        )
        preview[~range_mask & ~invalid_mask] = (
            preview[~range_mask & ~invalid_mask] * 0.25
        ).astype(np.uint8)
        preview[range_mask] = cv2.addWeighted(
            preview[range_mask],
            0.75,
            np.full((np.count_nonzero(range_mask), 3), (80, 255, 80), dtype=np.uint8),
            0.25,
            0.0,
        )

    min_d = float(np.min(finite))
    med_d = float(np.median(finite))
    max_d = float(np.max(finite))
    lines = [
        f"valid depth: min {min_d:.3f} m  med {med_d:.3f} m  max {max_d:.3f} m",
    ]
    if zmin is not None and zmax is not None and float(zmax) > float(zmin):
        kept_ratio = float(np.count_nonzero(range_mask)) / float(depth.size)
        lines.append(
            f"AnyGrasp z-range: [{float(zmin):.3f}, {float(zmax):.3f}] m   kept {kept_ratio:.1%}"
        )
    else:
        lines.append(f"preview scale: [{lo:.3f}, {hi:.3f}] m")

    y = 22
    for line in lines:
        cv2.putText(
            preview,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 20
    return preview


def _render_point_cloud_jpeg(
    points: np.ndarray,
    colors: np.ndarray,
    object_flags: np.ndarray,
    *,
    out_h: int,
    out_w: int,
) -> bytes:
    canvas = np.zeros((max(out_h, 1), max(out_w, 1), 3), dtype=np.uint8)
    if len(points) == 0:
        cv2.putText(
            canvas,
            "No scene points in selected z-range",
            (16, max(32, out_h // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Failed to encode point-cloud JPEG")
        return buf.tobytes()

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    object_flags = np.asarray(object_flags, dtype=bool)
    if object_flags.shape[0] != points.shape[0]:
        object_flags = np.zeros((points.shape[0],), dtype=bool)
    n_points_total = int(points.shape[0])
    n_object_total = int(object_flags.sum())

    max_points = 80_000
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[idx]
        colors = colors[idx]
        object_flags = object_flags[idx]

    focus_pts = points[object_flags] if np.any(object_flags) else points
    center = np.median(focus_pts, axis=0)
    pts = points - center[None, :]

    # The AnyGrasp cloud is in camera coordinates. For the top-down debug view we
    # want a 180° spin around the camera optical / top-down axis so left-right
    # orientation matches the live preview more intuitively.
    rot_z_pi = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pts = pts @ rot_z_pi.T

    yaw = np.deg2rad(-42.0)
    pitch = np.deg2rad(28.0)
    rot_y = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    view = pts @ (rot_y @ rot_x).T

    xy = np.stack([view[:, 0], -view[:, 1]], axis=1)
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-4)
    scale = min((out_w * 0.88) / spans[0], (out_h * 0.80) / spans[1])

    px = np.round((xy[:, 0] - (mins[0] + maxs[0]) * 0.5) * scale + out_w * 0.5).astype(
        np.int32
    )
    py = np.round((xy[:, 1] - (mins[1] + maxs[1]) * 0.5) * scale + out_h * 0.5).astype(
        np.int32
    )

    valid = (px >= 0) & (px < out_w) & (py >= 0) & (py < out_h)
    if not np.any(valid):
        cv2.putText(
            canvas,
            "Point cloud view is empty",
            (16, max(32, out_h // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Failed to encode point-cloud JPEG")
        return buf.tobytes()

    px = px[valid]
    py = py[valid]
    view = view[valid]
    colors = colors[valid]
    object_flags = object_flags[valid]

    order = np.argsort(view[:, 2], kind="stable")
    px = px[order]
    py = py[order]
    colors = colors[order]
    object_flags = object_flags[order]

    bg_layer = np.zeros_like(canvas)
    fg_layer = np.zeros_like(canvas)
    color_bgr = np.clip(colors[:, ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)
    if np.any(~object_flags):
        bg_layer[py[~object_flags], px[~object_flags]] = color_bgr[~object_flags]
    if np.any(object_flags):
        fg_layer[py[object_flags], px[object_flags]] = np.array(
            [90, 255, 120], dtype=np.uint8
        )

    bg_layer = cv2.dilate(bg_layer, np.ones((2, 2), dtype=np.uint8), iterations=1)
    fg_layer = cv2.dilate(fg_layer, np.ones((3, 3), dtype=np.uint8), iterations=1)
    canvas = np.maximum(bg_layer, fg_layer)

    label = f"Scene cloud sent to AnyGrasp: {n_points_total:,} pts" + (
        f"  |  object: {n_object_total:,} pts" if n_object_total > 0 else ""
    )
    cv2.putText(
        canvas,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    if np.any(object_flags):
        cv2.putText(
            canvas,
            "green = SAM3-selected object points",
            (10, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 255, 120),
            1,
            cv2.LINE_AA,
        )

    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Failed to encode point-cloud JPEG")
    return buf.tobytes()


def _clip_planner_z_in_place(grasps_world: np.ndarray) -> int:
    """Clip planner-facing AnyGrasp Z to the configured table-safe floor."""
    if grasps_world.size == 0:
        return 0
    z = grasps_world[:, 2, 3]
    mask = z < float(ANYGRASP_MIN_PLANNER_Z_M)
    if not np.any(mask):
        return 0
    grasps_world[mask, 2, 3] = float(ANYGRASP_MIN_PLANNER_Z_M)
    return int(np.count_nonzero(mask))


def _point_cloud_z_extent(
    points: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Return min(z), max(z), and thickness=max(z)-min(z) for a point cloud."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return None, None, None
    z = points[:, 2]
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None, None, None
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    return z_min, z_max, float(z_max - z_min)


def _normalize_angle_deg(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _angle_diff_deg(a: float, b: float) -> float:
    return abs(_normalize_angle_deg(float(a) - float(b)))


def _principal_axis_from_mask(
    mask: np.ndarray,
) -> tuple[
    np.ndarray | None, np.ndarray | None, float, tuple[np.ndarray, np.ndarray] | None
]:
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None, None, 1.0, None

    coords = np.column_stack([xs, ys]).astype(np.float64)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    if len(coords) < 2:
        return (
            centroid,
            np.array([1.0, 0.0], dtype=np.float64),
            1.0,
            (centroid.copy(), centroid.copy()),
        )

    cov = np.cov(centered, rowvar=False, bias=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major_var = float(max(eigvals[order[0]], 0.0))
    minor_var = float(max(eigvals[order[1]], 0.0))
    ratio = float(major_var / max(minor_var, 1e-12)) if major_var > 0.0 else 1.0

    axis = np.asarray(eigvecs[:, order[0]], dtype=np.float64)
    if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
        axis = -axis
    projections = centered @ axis
    lo_idx = int(np.argmin(projections))
    hi_idx = int(np.argmax(projections))
    return centroid, axis, ratio, (coords[lo_idx].copy(), coords[hi_idx].copy())


def _project_pixel_to_plane_world(
    pixel_xy: np.ndarray | list[float] | tuple[float, float],
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
    *,
    plane_z_m: float = TWO_D_TOP_DOWN_Z_M,
) -> np.ndarray | None:
    u, v = [float(x) for x in pixel_xy]
    fx = float(cam_K[0, 0])
    fy = float(cam_K[1, 1])
    cx = float(cam_K[0, 2])
    cy = float(cam_K[1, 2])
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)

    origin_world = np.asarray(T_cam_world[:3, 3], dtype=np.float64)
    ray_world = np.asarray(T_cam_world[:3, :3], dtype=np.float64) @ ray_cam
    dz = float(ray_world[2])
    if abs(dz) < 1e-9:
        return None

    scale = (float(plane_z_m) - float(origin_world[2])) / dz
    if scale <= 0.0:
        return None
    return origin_world + scale * ray_world


def _project_world_to_pixel(
    point_world: np.ndarray | list[float] | tuple[float, float, float],
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
) -> tuple[int, int] | None:
    point_world = np.asarray(point_world, dtype=np.float64).reshape(3)
    R_cam_world = np.asarray(T_cam_world[:3, :3], dtype=np.float64)
    t_cam_world = np.asarray(T_cam_world[:3, 3], dtype=np.float64)
    point_cam = R_cam_world.T @ (point_world - t_cam_world)
    z = float(point_cam[2])
    if z <= 1e-9:
        return None

    fx = float(cam_K[0, 0])
    fy = float(cam_K[1, 1])
    cx = float(cam_K[0, 2])
    cy = float(cam_K[1, 2])
    u = int(round(fx * float(point_cam[0]) / z + cx))
    v = int(round(fy * float(point_cam[1]) / z + cy))
    return u, v


def _estimate_local_tangent_from_mask(
    mask: np.ndarray,
    pixel_xy: np.ndarray | list[float] | tuple[float, float],
    *,
    reference_axis_px: np.ndarray | None = None,
    radius_px: float | None = None,
) -> tuple[np.ndarray | None, tuple[np.ndarray, np.ndarray] | None]:
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None, None

    coords = np.column_stack([xs, ys]).astype(np.float64)
    anchor = np.asarray(pixel_xy, dtype=np.float64).reshape(2)

    if radius_px is None:
        width_px = float(np.max(xs) - np.min(xs) + 1)
        height_px = float(np.max(ys) - np.min(ys) + 1)
        radius_px = np.clip(
            _TWO_D_LOCAL_AXIS_RADIUS_FRACTION * max(width_px, height_px),
            _TWO_D_LOCAL_AXIS_MIN_RADIUS_PX,
            _TWO_D_LOCAL_AXIS_MAX_RADIUS_PX,
        )

    deltas = coords - anchor
    dist2 = np.sum(deltas * deltas, axis=1)
    local_coords = coords[dist2 <= float(radius_px) ** 2]
    if len(local_coords) < _TWO_D_LOCAL_AXIS_MIN_NEIGHBOR_POINTS:
        nearest_order = np.argsort(dist2)
        k = min(len(coords), _TWO_D_LOCAL_AXIS_MIN_NEIGHBOR_POINTS)
        local_coords = coords[nearest_order[:k]]

    if len(local_coords) < 2:
        return None, None

    local_centroid = local_coords.mean(axis=0)
    centered = local_coords - local_centroid
    cov = np.cov(centered, rowvar=False, bias=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    tangent_px = np.asarray(eigvecs[:, order[0]], dtype=np.float64)
    if reference_axis_px is not None and np.dot(tangent_px, reference_axis_px) < 0.0:
        tangent_px = -tangent_px
    elif tangent_px[0] < 0.0 or (abs(tangent_px[0]) < 1e-9 and tangent_px[1] < 0.0):
        tangent_px = -tangent_px

    projections = centered @ tangent_px
    endpoints = (
        local_centroid + tangent_px * float(np.min(projections)),
        local_centroid + tangent_px * float(np.max(projections)),
    )
    return tangent_px, endpoints


def _top_down_yaw_from_world_axis(axis_world_xy: np.ndarray) -> float:
    theta_from_x = np.degrees(
        np.arctan2(float(axis_world_xy[1]), float(axis_world_xy[0]))
    )
    return _normalize_angle_deg(-(theta_from_x + 90.0))


def _perpendicular_world_axis(axis_world_xy: np.ndarray) -> np.ndarray:
    axis_world_xy = np.asarray(axis_world_xy, dtype=np.float64).reshape(2)
    perp = np.array([-axis_world_xy[1], axis_world_xy[0]], dtype=np.float64)
    norm = float(np.linalg.norm(perp))
    if norm <= 1e-12:
        return np.array([0.0, -1.0], dtype=np.float64)
    return perp / norm


def _derive_two_d_pose(
    *,
    mask: np.ndarray,
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
    planner_xyz: np.ndarray,
    planner_yaw_deg: float,
    plane_z_m: float = TWO_D_TOP_DOWN_Z_M,
) -> tuple[list[float], list[float]]:
    mask01 = (np.asarray(mask) > 0).astype(np.uint8)
    planner_xyz = np.asarray(planner_xyz, dtype=np.float64).reshape(3)

    _centroid_px, global_axis_px, global_ratio, global_endpoints_px = (
        _principal_axis_from_mask(mask01)
    )
    global_axis_world = None
    if (
        global_axis_px is not None
        and global_endpoints_px is not None
        and global_ratio >= float(_TWO_D_MAJOR_AXIS_RATIO_THRESHOLD)
    ):
        a_world = _project_pixel_to_plane_world(
            global_endpoints_px[0], cam_K, T_cam_world, plane_z_m=plane_z_m
        )
        b_world = _project_pixel_to_plane_world(
            global_endpoints_px[1], cam_K, T_cam_world, plane_z_m=plane_z_m
        )
        if a_world is not None and b_world is not None:
            axis_xy = np.asarray(b_world[:2] - a_world[:2], dtype=np.float64)
            axis_norm = float(np.linalg.norm(axis_xy))
            if axis_norm > 1e-9:
                global_axis_world = axis_xy / axis_norm

    anchor_world = np.array(
        [float(planner_xyz[0]), float(planner_xyz[1]), float(plane_z_m)],
        dtype=np.float64,
    )
    anchor_px = _project_world_to_pixel(anchor_world, cam_K, T_cam_world)
    local_tangent_world = None
    if anchor_px is not None:
        local_tangent_px, local_endpoints_px = _estimate_local_tangent_from_mask(
            mask01, anchor_px, reference_axis_px=global_axis_px
        )
        if local_endpoints_px is not None:
            local_a_world = _project_pixel_to_plane_world(
                local_endpoints_px[0], cam_K, T_cam_world, plane_z_m=plane_z_m
            )
            local_b_world = _project_pixel_to_plane_world(
                local_endpoints_px[1], cam_K, T_cam_world, plane_z_m=plane_z_m
            )
            if local_a_world is not None and local_b_world is not None:
                tangent_xy = np.asarray(
                    local_b_world[:2] - local_a_world[:2], dtype=np.float64
                )
                tangent_norm = float(np.linalg.norm(tangent_xy))
                if tangent_norm > 1e-9:
                    local_tangent_world = tangent_xy / tangent_norm
        elif local_tangent_px is not None:
            local_a_world = _project_pixel_to_plane_world(
                np.asarray(anchor_px, dtype=np.float64)
                - 4.0 * np.asarray(local_tangent_px, dtype=np.float64),
                cam_K,
                T_cam_world,
                plane_z_m=plane_z_m,
            )
            local_b_world = _project_pixel_to_plane_world(
                np.asarray(anchor_px, dtype=np.float64)
                + 4.0 * np.asarray(local_tangent_px, dtype=np.float64),
                cam_K,
                T_cam_world,
                plane_z_m=plane_z_m,
            )
            if local_a_world is not None and local_b_world is not None:
                tangent_xy = np.asarray(
                    local_b_world[:2] - local_a_world[:2], dtype=np.float64
                )
                tangent_norm = float(np.linalg.norm(tangent_xy))
                if tangent_norm > 1e-9:
                    local_tangent_world = tangent_xy / tangent_norm

    yaw_candidates = list(_TWO_D_CENTER_ONLY_YAWS_DEG)
    if (
        global_ratio >= float(_TWO_D_MAJOR_AXIS_RATIO_THRESHOLD)
        and local_tangent_world is None
    ):
        local_tangent_world = global_axis_world
    if (
        global_ratio >= float(_TWO_D_MAJOR_AXIS_RATIO_THRESHOLD)
        and local_tangent_world is not None
    ):
        cross_section_world = _perpendicular_world_axis(local_tangent_world)
        primary_yaw = _top_down_yaw_from_world_axis(cross_section_world)
        backup_yaw = _top_down_yaw_from_world_axis(local_tangent_world)
        yaw_candidates = [
            primary_yaw,
            _normalize_angle_deg(primary_yaw + 180.0),
            backup_yaw,
            _normalize_angle_deg(backup_yaw + 180.0),
        ]

    selected_yaw = min(
        yaw_candidates,
        key=lambda yaw: (_angle_diff_deg(yaw, planner_yaw_deg), abs(float(yaw))),
    )
    two_d_xyz = [
        round(float(anchor_world[0]), 5),
        round(float(anchor_world[1]), 5),
        round(float(anchor_world[2]), 5),
    ]
    two_d_rpy = [0.0, 180.0, round(float(_normalize_angle_deg(selected_yaw)), 3)]
    return two_d_xyz, two_d_rpy


def _get_portal():
    global _portal_client
    if _portal_client is None:
        import portal

        _portal_client = portal.Client(f"{CAP_HOST}:{CAP_PORT}")
    return _portal_client


def _planner_request_cache_key(req: PlannerRequest) -> tuple[Any, ...]:
    return (
        str(req.side).strip().lower(),
        str(req.pose_mode).strip().lower(),
        _normalize_solver_speed(req.solver_speed),
        round(float(req.planning_speed), 6),
        round(float(req.ik_error_threshold), 6),
        round(float(req.ik_xyz_weight), 6),
        round(float(req.ik_rpy_weight), 6),
        tuple(round(float(v), 6) for v in req.xyz),
        tuple(round(float(v), 6) for v in req.rpy),
    )


def _store_preview_trajectory(
    req: PlannerRequest,
    *,
    current_left_jp: np.ndarray,
    current_right_jp: np.ndarray,
    current_left_gp: float,
    current_right_gp: float,
    left_positions: np.ndarray,
    right_positions: np.ndarray,
    timestamps: list[float],
    path_xyz: list[list[float]],
    target_left_pos: np.ndarray,
    target_left_quat_xyzw: np.ndarray,
    target_right_pos: np.ndarray,
    target_right_quat_xyzw: np.ndarray,
) -> None:
    cache_key = _planner_request_cache_key(req)
    cache_entry = {
        "cache_key": cache_key,
        "side": str(req.side).strip().lower(),
        "pose_mode": str(req.pose_mode).strip().lower(),
        "solver_speed": _normalize_solver_speed(req.solver_speed),
        "planning_speed": float(req.planning_speed),
        "ik_error_threshold": float(req.ik_error_threshold),
        "ik_xyz_weight": float(req.ik_xyz_weight),
        "ik_rpy_weight": float(req.ik_rpy_weight),
        "xyz": [float(v) for v in req.xyz],
        "rpy": [float(v) for v in req.rpy],
        "current_left_jp": np.asarray(current_left_jp, dtype=np.float64).copy(),
        "current_right_jp": np.asarray(current_right_jp, dtype=np.float64).copy(),
        "current_left_gp": float(current_left_gp),
        "current_right_gp": float(current_right_gp),
        "left_positions": np.asarray(left_positions, dtype=np.float64).copy(),
        "right_positions": np.asarray(right_positions, dtype=np.float64).copy(),
        "timestamps": [float(t) for t in timestamps],
        "path_xyz": [list(map(float, row)) for row in path_xyz],
        "target_left_pos": np.asarray(target_left_pos, dtype=np.float64).copy(),
        "target_left_quat_xyzw": np.asarray(
            target_left_quat_xyzw, dtype=np.float64
        ).copy(),
        "target_right_pos": np.asarray(target_right_pos, dtype=np.float64).copy(),
        "target_right_quat_xyzw": np.asarray(
            target_right_quat_xyzw, dtype=np.float64
        ).copy(),
    }
    with _preview_trajectory_cache_lock:
        _preview_trajectory_cache[cache_key] = cache_entry


def _get_cached_preview_trajectory(req: PlannerRequest) -> dict[str, Any] | None:
    with _preview_trajectory_cache_lock:
        entry = _preview_trajectory_cache.get(_planner_request_cache_key(req))
    if entry is None:
        return None
    return dict(entry)


def _build_cached_batch_preview_entry(req: PlannerRequest) -> dict[str, Any] | None:
    tool = _get_freespace_tool()
    cache_entry = tool._get_cached_trajectory(req.trajectory_cache_key)
    if cache_entry is None:
        return None

    side = str(req.side).strip().lower()
    kin = tool._get_diagnostic_kinematics(
        position_cost=float(req.ik_xyz_weight),
        orientation_cost=float(req.ik_rpy_weight),
    )
    left_positions = np.asarray(
        cache_entry["left_positions"], dtype=np.float64
    ).reshape(-1, 6)
    right_positions = np.asarray(
        cache_entry["right_positions"], dtype=np.float64
    ).reshape(-1, 6)
    path_xyz: list[list[float]] = []
    for lp, rp in zip(left_positions, right_positions, strict=False):
        l_pos, _, r_pos, _ = kin.forward_kinematics(lp, rp)
        pos = l_pos if side == "left" else r_pos
        path_xyz.append(_round_list(pos, 5))

    preview_entry = dict(cache_entry)
    preview_entry["path_xyz"] = path_xyz
    return preview_entry


def _preview_state_matches_current(
    cache_entry: dict[str, Any],
    current_left_jp: np.ndarray,
    current_right_jp: np.ndarray,
) -> bool:
    cached_left = np.asarray(cache_entry["current_left_jp"], dtype=np.float64).reshape(
        -1
    )
    cached_right = np.asarray(
        cache_entry["current_right_jp"], dtype=np.float64
    ).reshape(-1)
    current_left = np.asarray(current_left_jp, dtype=np.float64).reshape(-1)
    current_right = np.asarray(current_right_jp, dtype=np.float64).reshape(-1)
    return bool(
        np.allclose(
            cached_left,
            current_left,
            atol=_PREVIEW_STATE_TOLERANCE_RAD,
            rtol=0.0,
        )
        and np.allclose(
            cached_right,
            current_right,
            atol=_PREVIEW_STATE_TOLERANCE_RAD,
            rtol=0.0,
        )
    )


def _truncate_preview_trajectory(
    cache_entry: dict[str, Any], execute_fraction: float
) -> tuple[np.ndarray, np.ndarray, list[float], list[list[float]], int]:
    left_positions = np.asarray(
        cache_entry["left_positions"], dtype=np.float64
    ).reshape(-1, 6)
    right_positions = np.asarray(
        cache_entry["right_positions"], dtype=np.float64
    ).reshape(-1, 6)
    timestamps = [float(t) for t in cache_entry["timestamps"]]
    path_xyz = [list(map(float, row)) for row in cache_entry["path_xyz"]]
    n_steps = len(left_positions)
    if n_steps == 0:
        return left_positions, right_positions, [], [], 0

    frac = float(np.clip(float(execute_fraction), 0.0, 1.0))
    if frac >= 1.0 or n_steps <= 1:
        keep = n_steps
    elif frac <= 0.0:
        keep = 1
    else:
        keep = max(2, 1 + int(np.ceil(frac * max(n_steps - 1, 0))))
        keep = min(keep, n_steps)
    return (
        left_positions[:keep].copy(),
        right_positions[:keep].copy(),
        list(timestamps[:keep]),
        list(path_xyz[:keep]),
        keep,
    )


def _planner_target_quats(
    tool,
    *,
    side: str,
    rpy: np.ndarray,
    pose_mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if pose_mode in {"planner", "2d"}:
        left_target_quat = tool._display_rpy_to_quat(rpy) if side == "left" else None
        right_target_quat = tool._display_rpy_to_quat(rpy) if side == "right" else None
        return left_target_quat, right_target_quat
    if pose_mode == "raw":
        from scipy.spatial.transform import Rotation

        raw_quat = Rotation.from_euler("xyz", rpy, degrees=True).as_quat()
        left_target_quat = raw_quat if side == "left" else None
        right_target_quat = raw_quat if side == "right" else None
        return left_target_quat, right_target_quat
    raise ValueError(
        f"Unsupported pose_mode: {pose_mode!r}. Use 'planner', '2d', or 'raw'."
    )


def _load_local_pose_config() -> dict[str, Any]:
    global _local_pose_config_cache
    if _local_pose_config_cache is not None:
        return dict(_local_pose_config_cache)

    config = dict(_LOCAL_POSE_DEFAULTS)
    try:
        source = _LOCAL_POSE_SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(_LOCAL_POSE_SCRIPT_PATH))
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                name = node.targets[0].id
                if name not in config:
                    continue
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    continue
                config[name] = value
    except Exception:
        logger.exception(
            "[anygrasp-debug] Failed to parse %s; using baked defaults",
            _LOCAL_POSE_SCRIPT_PATH,
        )

    _local_pose_config_cache = dict(config)
    return dict(config)


_freespace_tool = None
_preview_viser = None
_preview_viser_lock = threading.Lock()
_cloud_viser = None
_cloud_viser_lock = threading.Lock()


def _get_freespace_tool():
    global _freespace_tool
    if _freespace_tool is None:
        from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool

        _freespace_tool = FreespaceMoveTool(host=CAP_HOST, port=CAP_PORT)
    return _freespace_tool


def _preload_motion_planner(
    planner_backend: str = "curobo",
    *,
    solver_speed: str = DEFAULT_SOLVER_SPEED,
    ik_xyz_weight: float = 1.0,
    ik_rpy_weight: float = 0.3,
) -> None:
    start = time.perf_counter()
    logger.info(
        "[anygrasp-debug] Preloading motion planner backend=%s solver_speed=%s (ik_xyz_weight=%.3f, ik_rpy_weight=%.3f)",
        planner_backend,
        _normalize_solver_speed(solver_speed),
        float(ik_xyz_weight),
        float(ik_rpy_weight),
    )
    tool = _get_freespace_tool()
    _tool_get_planner_with_solver_speed(
        tool,
        planner_backend=planner_backend,
        ik_xyz_weight=float(ik_xyz_weight),
        ik_rpy_weight=float(ik_rpy_weight),
        solver_speed=solver_speed,
    )
    logger.info(
        "[anygrasp-debug] Motion planner preload complete in %.2fs",
        time.perf_counter() - start,
    )


class _StandaloneCloudViser:
    def __init__(self, host: str = "0.0.0.0", port: int = 8092):
        import viser

        self._host = host
        self._port = int(port)
        self.server = viser.ViserServer(host=host, port=self._port)
        self._scene_handle = None
        self._object_handle = None
        self._grasp_handles: list[Any] = []
        self.server.scene.add_frame(
            "/anygrasp_cloud/camera_frame",
            show_axes=True,
            axes_length=0.12,
            axes_radius=0.004,
            origin_radius=0.007,
            visible=True,
        )

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}"

    @staticmethod
    def _sample(
        points: np.ndarray,
        colors: np.ndarray,
        *,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(points) <= max_points:
            return points.astype(np.float32), colors.astype(np.uint8)
        idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        return points[idx].astype(np.float32), colors[idx].astype(np.uint8)

    def update(
        self,
        *,
        scene_points: np.ndarray,
        scene_colors: np.ndarray,
        object_points: np.ndarray,
        grasp_poses_cam: np.ndarray | None = None,
        grasp_scores: np.ndarray | None = None,
    ) -> str:
        from scipy.spatial.transform import Rotation

        scene_points = np.asarray(scene_points, dtype=np.float32)
        scene_colors = np.asarray(scene_colors, dtype=np.float32)
        object_points = np.asarray(object_points, dtype=np.float32)
        grasp_poses_cam = (
            np.asarray(grasp_poses_cam, dtype=np.float64)
            if grasp_poses_cam is not None and np.asarray(grasp_poses_cam).size > 0
            else np.empty((0, 4, 4), dtype=np.float64)
        )
        grasp_scores = (
            np.asarray(grasp_scores, dtype=np.float64).reshape(-1)
            if grasp_scores is not None and np.asarray(grasp_scores).size > 0
            else np.empty((0,), dtype=np.float64)
        )

        # Display-only flip around the optical axis so the initial top-down
        # appearance matches the RGB preview more intuitively.
        display_rot = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        scene_points = scene_points @ display_rot.T
        object_points = (
            object_points @ display_rot.T if len(object_points) else object_points
        )

        scene_rgb = np.clip(scene_colors * 255.0, 0.0, 255.0).astype(np.uint8)
        obj_rgb = np.tile(
            np.array([[90, 255, 120]], dtype=np.uint8), (len(object_points), 1)
        )

        scene_points, scene_rgb = self._sample(
            scene_points, scene_rgb, max_points=120_000
        )
        object_points, obj_rgb = self._sample(object_points, obj_rgb, max_points=25_000)

        if self._scene_handle is None:
            self._scene_handle = self.server.scene.add_point_cloud(
                "/anygrasp_cloud/scene",
                points=scene_points,
                colors=scene_rgb,
                point_size=0.003,
                point_shape="circle",
                visible=True,
            )
        else:
            self._scene_handle.points = scene_points
            self._scene_handle.colors = scene_rgb
            self._scene_handle.visible = len(scene_points) > 0

        if self._object_handle is None:
            self._object_handle = self.server.scene.add_point_cloud(
                "/anygrasp_cloud/object",
                points=object_points
                if len(object_points)
                else np.zeros((1, 3), dtype=np.float32),
                colors=obj_rgb
                if len(obj_rgb)
                else np.array([[90, 255, 120]], dtype=np.uint8),
                point_size=0.008,
                point_shape="sparkle",
                visible=len(object_points) > 0,
            )
        else:
            if len(object_points) > 0:
                self._object_handle.points = object_points
                self._object_handle.colors = obj_rgb
            self._object_handle.visible = len(object_points) > 0

        for h in self._grasp_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._grasp_handles.clear()

        if len(grasp_poses_cam) > 0:
            F = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
            for i, T_cam in enumerate(grasp_poses_cam):
                T_disp = np.eye(4, dtype=np.float64)
                T_disp[:3, :3] = F @ T_cam[:3, :3]
                T_disp[:3, 3] = F @ T_cam[:3, 3]
                quat = Rotation.from_matrix(T_disp[:3, :3]).as_quat()
                score = float(grasp_scores[i]) if i < len(grasp_scores) else 0.0
                scale = 0.08 if i == 0 else (0.06 if i == 1 else 0.045)
                frame = self.server.scene.add_frame(
                    f"/anygrasp_cloud/grasp_{i}",
                    position=tuple(float(v) for v in T_disp[:3, 3]),
                    wxyz=(
                        float(quat[3]),
                        float(quat[0]),
                        float(quat[1]),
                        float(quat[2]),
                    ),
                    axes_length=scale,
                    axes_radius=scale * 0.08,
                    origin_radius=scale * 0.1,
                    visible=True,
                )
                label = self.server.scene.add_label(
                    f"/anygrasp_cloud/grasp_{i}/label",
                    text=f"#{i + 1} ({score:.0%})",
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=(0.0, 0.0, scale * 1.5),
                )
                self._grasp_handles.extend([frame, label])

        logger.info(
            "[anygrasp-debug] Updated cloud preview: scene=%d object=%d grasps=%d url=%s",
            len(scene_points),
            len(object_points),
            len(grasp_poses_cam),
            self.url,
        )
        return self.url


def _get_cloud_viser() -> _StandaloneCloudViser:
    global _cloud_viser
    if _cloud_viser is None:
        with _cloud_viser_lock:
            if _cloud_viser is None:
                port = int(os.environ.get("ANYGRASP_VISER_CLOUD_PORT", "8092"))
                _cloud_viser = _StandaloneCloudViser(port=port)
    return _cloud_viser


class _StandalonePreviewViser:
    def __init__(self, host: str = "0.0.0.0", port: int = 8091):
        import trimesh
        import viser
        from viser.extras import ViserUrdf

        self._trimesh = trimesh
        self._host = host
        self._port = int(port)
        self.server = viser.ViserServer(host=host, port=self._port)
        from enpire.env.forge.robot.models.station.paths import get_station_urdf

        urdf_path = get_station_urdf()
        self.urdf_vis = ViserUrdf(self.server, urdf_or_path=urdf_path, load_meshes=True)
        self.urdf_joint_names = list(
            dict(self.urdf_vis.get_actuated_joint_limits()).keys()
        )  # type: ignore[arg-type]
        if self.urdf_joint_names:
            self.urdf_vis.update_cfg(np.zeros(len(self.urdf_joint_names), dtype=float))
        from enpire.env.forge.robot.yam.kinematics import YamKinematics

        self._kin = YamKinematics()
        self._traj_handles: dict[str, list[Any]] = {"left": [], "right": []}
        self._start_handles: dict[str, Any] = {}
        self._goal_handles: dict[str, Any] = {}
        self._goal_frames: dict[str, Any] = {}
        self._max_traj_pts = 48
        self._init_scene()

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}"

    def _joint_cfg_array(self, left_jp: np.ndarray, right_jp: np.ndarray) -> np.ndarray:
        mapping = {f"left_joint{i + 1}": float(left_jp[i]) for i in range(6)}
        mapping.update({f"right_joint{i + 1}": float(right_jp[i]) for i in range(6)})
        return np.asarray(
            [mapping.get(name, 0.0) for name in self.urdf_joint_names], dtype=float
        )

    def _init_scene(self) -> None:
        trimesh = self._trimesh
        start_sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.012)
        start_sphere.visual.vertex_colors = np.tile(
            np.array([50, 220, 80, 220], dtype=np.uint8),
            (start_sphere.vertices.shape[0], 1),
        )  # type: ignore[union-attr]
        goal_sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.012)
        goal_sphere.visual.vertex_colors = np.tile(
            np.array([255, 95, 87, 220], dtype=np.uint8),
            (goal_sphere.vertices.shape[0], 1),
        )  # type: ignore[union-attr]
        for side in ("left", "right"):
            self._start_handles[side] = self.server.scene.add_mesh_trimesh(
                f"/anygrasp_preview/{side}_start",
                start_sphere,
                position=(0.0, 0.0, 0.0),
                visible=False,
            )
            self._goal_handles[side] = self.server.scene.add_mesh_trimesh(
                f"/anygrasp_preview/{side}_goal",
                goal_sphere,
                position=(0.0, 0.0, 0.0),
                visible=False,
            )
            self._goal_frames[side] = self.server.scene.add_frame(
                f"/anygrasp_preview/{side}_goal_frame",
                show_axes=True,
                axes_length=0.08,
                axes_radius=0.003,
                origin_radius=0.006,
                visible=False,
            )

    def _ensure_traj_handles(self, side: str, count: int) -> None:
        trimesh = self._trimesh
        handles = self._traj_handles[side]
        while len(handles) < count:
            i = len(handles)
            t = i / max(count - 1, 1)
            rgba = np.array(
                [int(50 + 205 * t), int(220 - 60 * t), 50, 200], dtype=np.uint8
            )
            sph = trimesh.creation.icosphere(subdivisions=2, radius=0.008)
            sph.visual.vertex_colors = np.tile(rgba, (sph.vertices.shape[0], 1))  # type: ignore[union-attr]
            h = self.server.scene.add_mesh_trimesh(
                f"/anygrasp_preview/traj/{side}_{i}",
                sph,
                position=(0.0, 0.0, 0.0),
                visible=False,
            )
            handles.append(h)

    def _hide_side(self, side: str) -> None:
        self._start_handles[side].visible = False
        self._goal_handles[side].visible = False
        self._goal_frames[side].visible = False
        for h in self._traj_handles[side]:
            h.visible = False

    def update(
        self,
        *,
        side: str,
        pose_mode: str,
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
        left_positions: np.ndarray,
        right_positions: np.ndarray,
        target_quat_xyzw: np.ndarray,
    ) -> str:
        self.urdf_vis.update_cfg(
            self._joint_cfg_array(current_left_jp, current_right_jp)
        )
        inactive = "right" if side == "left" else "left"
        self._hide_side(inactive)

        n_steps = len(left_positions)
        idxs = (
            np.linspace(0, n_steps - 1, min(n_steps, self._max_traj_pts), dtype=int)
            if n_steps
            else np.empty((0,), dtype=int)
        )
        ee_list: list[np.ndarray] = []
        for idx in idxs:
            l_pos, _, r_pos, _ = self._kin.forward_kinematics(
                left_positions[idx], right_positions[idx]
            )
            ee_list.append(l_pos if side == "left" else r_pos)

        cur_l_pos, _, cur_r_pos, _ = self._kin.forward_kinematics(
            current_left_jp, current_right_jp
        )
        start_pos = cur_l_pos if side == "left" else cur_r_pos
        goal_pos = ee_list[-1] if ee_list else start_pos

        self._start_handles[side].position = tuple(float(v) for v in start_pos)
        self._start_handles[side].visible = True
        self._goal_handles[side].position = tuple(float(v) for v in goal_pos)
        self._goal_handles[side].visible = True
        self._goal_frames[side].position = tuple(float(v) for v in goal_pos)
        quat_xyzw = np.asarray(target_quat_xyzw, dtype=np.float64).reshape(4)
        self._goal_frames[side].wxyz = tuple(float(v) for v in quat_xyzw[[3, 0, 1, 2]])
        self._goal_frames[side].visible = True

        self._ensure_traj_handles(side, len(ee_list))
        for i, pos in enumerate(ee_list):
            self._traj_handles[side][i].position = tuple(float(v) for v in pos)
            self._traj_handles[side][i].visible = True
        for i in range(len(ee_list), len(self._traj_handles[side])):
            self._traj_handles[side][i].visible = False

        logger.info(
            "[anygrasp-debug] Updated 3D Viser preview: side=%s mode=%s points=%d url=%s",
            side,
            pose_mode,
            len(ee_list),
            self.url,
        )
        return self.url


def _get_preview_viser() -> _StandalonePreviewViser:
    global _preview_viser
    if _preview_viser is None:
        with _preview_viser_lock:
            if _preview_viser is None:
                port = int(os.environ.get("ANYGRASP_VISER_PREVIEW_PORT", "8091"))
                _preview_viser = _StandalonePreviewViser(port=port)
    return _preview_viser


def _preview_pose_with_path(req: PlannerRequest) -> PlannerResponse:
    tool = _get_freespace_tool()
    side = req.side.strip().lower()
    pose_mode = req.pose_mode.strip().lower()
    solver_speed = _normalize_solver_speed(req.solver_speed)
    xyz = np.asarray(req.xyz, dtype=np.float64)
    rpy = np.asarray(req.rpy, dtype=np.float64)
    planning_speed = float(req.planning_speed)
    ik_threshold = float(req.ik_error_threshold)
    ik_xyz_weight = float(req.ik_xyz_weight)
    ik_rpy_weight = float(req.ik_rpy_weight)

    client = tool._get_client()
    state = client.get_state().result()
    cur_left_jp = np.asarray(state["left_joint_pos"], dtype=np.float64)
    cur_right_jp = np.asarray(state["right_joint_pos"], dtype=np.float64)
    cur_left_gp = float(state["left_gripper_pos"][0])
    cur_right_gp = float(state["right_gripper_pos"][0])

    planner = _tool_get_planner_with_solver_speed(
        tool,
        ik_xyz_weight=ik_xyz_weight,
        ik_rpy_weight=ik_rpy_weight,
        solver_speed=solver_speed,
    )
    diagnostic_kin = tool._get_diagnostic_kinematics(
        position_cost=ik_xyz_weight,
        orientation_cost=ik_rpy_weight,
    )
    try:
        left_target_quat, right_target_quat = _planner_target_quats(
            tool,
            side=side,
            rpy=rpy,
            pose_mode=pose_mode,
        )
    except ValueError:
        return PlannerResponse(
            ok=False,
            status="Invalid",
            executed=False,
            side=side,
            pose_mode=pose_mode,
            goal_xyz=_round_list(xyz, 5),
            reason=f"Unsupported pose_mode: {req.pose_mode!r}. Use 'planner', '2d', or 'raw'.",
            error="Invalid pose mode",
        )

    with tool._planner_lock:
        cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = planner._kin.forward_kinematics(
            cur_left_jp, cur_right_jp
        )
        tgt_l_pos = xyz if side == "left" else np.asarray(cur_l_pos, dtype=np.float64)
        tgt_r_pos = xyz if side == "right" else np.asarray(cur_r_pos, dtype=np.float64)
        tgt_l_q = (
            np.asarray(left_target_quat, dtype=np.float64)
            if left_target_quat is not None
            else np.asarray(cur_l_q, dtype=np.float64)
        )
        tgt_r_q = (
            np.asarray(right_target_quat, dtype=np.float64)
            if right_target_quat is not None
            else np.asarray(cur_r_q, dtype=np.float64)
        )

        plan_result = planner.plan_to_pose(
            current_left_jp=cur_left_jp,
            current_right_jp=cur_right_jp,
            target_left_pos=xyz if side == "left" else None,
            target_left_quat_xyzw=left_target_quat if side == "left" else None,
            target_right_pos=xyz if side == "right" else None,
            target_right_quat_xyzw=right_target_quat if side == "right" else None,
            side=side,
            ik_error_threshold=ik_threshold,
            left_gripper=cur_left_gp,
            right_gripper=cur_right_gp,
            max_joint_vel=planning_speed,
            verbose=False,
        )

        status = plan_result["status"]
        if status == "IK_Failed":
            kin = diagnostic_kin
            kin.forward_kinematics(cur_left_jp, cur_right_jp)
            gl, gr = kin.inverse_kinematics(
                tgt_l_pos, tgt_l_q, tgt_r_pos, tgt_r_q, seeded=True
            )
            diag = tool._compute_plan_diagnostics(
                diagnostic_kin,
                planner,
                side=side,
                final_left_joint_pos=gl,
                final_right_joint_pos=gr,
                target_left_pos=tgt_l_pos,
                target_left_quat_xyzw=tgt_l_q,
                target_right_pos=tgt_r_pos,
                target_right_quat_xyzw=tgt_r_q,
            )
            return PlannerResponse(
                ok=False,
                status="IK_Failed",
                executed=False,
                side=side,
                pose_mode=pose_mode,
                start_xyz=_round_list(cur_l_pos if side == "left" else cur_r_pos, 5),
                goal_xyz=_round_list(xyz, 5),
                final_pos_error_m=float(diag["final_pos_error_m"]),
                final_rot_error_deg=float(diag["final_rot_error_deg"]),
                final_pose_error=float(diag["final_pose_error"]),
                reason=(
                    f"IK did not converge: pos_err={float(diag['final_pos_error_m']):.4f} m, "
                    f"rot_err={float(diag['final_rot_error_deg']):.2f} deg "
                    f"(threshold={ik_threshold:.4f} m)."
                ),
                error="IK failed",
            )
        if status == "Planning_Failed":
            status_detail = str(plan_result.get("status_detail") or "").strip()
            reason = (
                f"cuRobo preview failed: {status_detail}"
                if status_detail
                else "cuRobo could not find a collision-free path."
            )
            return PlannerResponse(
                ok=False,
                status="Planning_Failed",
                executed=False,
                side=side,
                pose_mode=pose_mode,
                start_xyz=_round_list(cur_l_pos if side == "left" else cur_r_pos, 5),
                goal_xyz=_round_list(xyz, 5),
                reason=reason,
                error="Motion planning failed",
            )

        left_waypoints = np.asarray(
            plan_result.get("left_waypoints", plan_result["left_positions"]),
            dtype=np.float64,
        )
        right_waypoints = np.asarray(
            plan_result.get("right_waypoints", plan_result["right_positions"]),
            dtype=np.float64,
        )
        left_positions, right_positions = tool._densify_joint_waypoints(
            left_waypoints,
            right_waypoints,
        )
        timestamps = tool._timestamps_from_waypoints(
            left_positions,
            right_positions,
            planning_speed=planning_speed,
        )
        final_left_joint_pos = (
            left_positions[-1] if len(left_positions) else cur_left_jp
        )
        final_right_joint_pos = (
            right_positions[-1] if len(right_positions) else cur_right_jp
        )
        diag = tool._compute_plan_diagnostics(
            diagnostic_kin,
            planner,
            side=side,
            final_left_joint_pos=final_left_joint_pos,
            final_right_joint_pos=final_right_joint_pos,
            target_left_pos=tgt_l_pos,
            target_left_quat_xyzw=tgt_l_q,
            target_right_pos=tgt_r_pos,
            target_right_quat_xyzw=tgt_r_q,
        )

        path_xyz: list[list[float]] = []
        for lp, rp in zip(left_positions, right_positions, strict=False):
            l_pos, _, r_pos, _ = planner._kin.forward_kinematics(lp, rp)
            pos = l_pos if side == "left" else r_pos
            path_xyz.append(_round_list(pos, 5))

        target_quat_xyzw = tgt_l_q if side == "left" else tgt_r_q
        preview_url = None
        try:
            preview_url = _get_preview_viser().update(
                side=side,
                pose_mode=pose_mode,
                current_left_jp=cur_left_jp,
                current_right_jp=cur_right_jp,
                left_positions=left_positions,
                right_positions=right_positions,
                target_quat_xyzw=np.asarray(target_quat_xyzw, dtype=np.float64),
            )
        except Exception:
            logger.exception("[anygrasp-debug] Failed to update 3D Viser preview")

        _store_preview_trajectory(
            req,
            current_left_jp=cur_left_jp,
            current_right_jp=cur_right_jp,
            current_left_gp=cur_left_gp,
            current_right_gp=cur_right_gp,
            left_positions=left_positions,
            right_positions=right_positions,
            timestamps=timestamps,
            path_xyz=path_xyz,
            target_left_pos=tgt_l_pos,
            target_left_quat_xyzw=tgt_l_q,
            target_right_pos=tgt_r_pos,
            target_right_quat_xyzw=tgt_r_q,
        )

    return PlannerResponse(
        ok=True,
        status="Success",
        executed=False,
        used_cached_preview=False,
        execute_fraction=1.0,
        preview_cache_state="cached",
        preview_cache_message="Cached execution trajectory ready. Execute will reuse this exact path with no replanning.",
        trajectory_steps=int(len(path_xyz)),
        final_pos_error_m=float(diag["final_pos_error_m"]),
        final_rot_error_deg=float(diag["final_rot_error_deg"]),
        final_pose_error=float(diag["final_pose_error"]),
        side=side,
        pose_mode=pose_mode,
        start_xyz=_round_list(
            path_xyz[0] if path_xyz else (cur_l_pos if side == "left" else cur_r_pos), 5
        ),
        goal_xyz=_round_list(xyz, 5),
        path_xyz=path_xyz,
        preview_url=preview_url,
    )


def _preview_cached_trajectory(
    req: PlannerRequest, cache_entry: dict[str, Any]
) -> PlannerResponse:
    tool = _get_freespace_tool()
    side = req.side.strip().lower()
    pose_mode = req.pose_mode.strip().lower()
    xyz = np.asarray(req.xyz, dtype=np.float64)
    rpy = np.asarray(req.rpy, dtype=np.float64)

    client = tool._get_client()
    state = client.get_state().result()
    cur_left_jp = np.asarray(state["left_joint_pos"], dtype=np.float64)
    cur_right_jp = np.asarray(state["right_joint_pos"], dtype=np.float64)
    if not _preview_state_matches_current(cache_entry, cur_left_jp, cur_right_jp):
        return PlannerResponse(
            ok=False,
            status="Preview_Stale",
            executed=False,
            used_cached_preview=True,
            preview_cache_state="stale",
            preview_cache_message="Cached batch trajectory is stale because the robot state changed.",
            side=side,
            pose_mode=pose_mode,
            goal_xyz=_round_list(xyz, 5),
            reason=(
                "Robot state changed since batch preview. Re-run the batch preview/sort before "
                "executing to keep execution identical to the previewed trajectory."
            ),
            error="Batch trajectory is stale",
        )

    try:
        left_target_quat, right_target_quat = _planner_target_quats(
            tool,
            side=side,
            rpy=rpy,
            pose_mode=pose_mode,
        )
    except ValueError:
        return PlannerResponse(
            ok=False,
            status="Invalid",
            executed=False,
            used_cached_preview=True,
            side=side,
            pose_mode=pose_mode,
            goal_xyz=_round_list(xyz, 5),
            reason=f"Unsupported pose_mode: {req.pose_mode!r}. Use 'planner', '2d', or 'raw'.",
            error="Invalid pose mode",
        )

    left_positions = np.asarray(
        cache_entry["left_positions"], dtype=np.float64
    ).reshape(-1, 6)
    right_positions = np.asarray(
        cache_entry["right_positions"], dtype=np.float64
    ).reshape(-1, 6)
    path_xyz = [list(map(float, row)) for row in cache_entry.get("path_xyz", [])]
    target_quat_xyzw = (
        np.asarray(left_target_quat, dtype=np.float64)
        if side == "left"
        else np.asarray(right_target_quat, dtype=np.float64)
    )
    preview_url = None
    try:
        preview_url = _get_preview_viser().update(
            side=side,
            pose_mode=pose_mode,
            current_left_jp=cur_left_jp,
            current_right_jp=cur_right_jp,
            left_positions=left_positions,
            right_positions=right_positions,
            target_quat_xyzw=target_quat_xyzw,
        )
    except Exception:
        logger.exception("[anygrasp-debug] Failed to update cached 3D Viser preview")

    return PlannerResponse(
        ok=True,
        status="Success",
        executed=False,
        used_cached_preview=True,
        execute_fraction=1.0,
        preview_cache_state="cached",
        preview_cache_message="Using the exact cached batch trajectory; execute will not replan.",
        trajectory_steps=int(cache_entry.get("trajectory_steps", len(left_positions))),
        final_pos_error_m=float(cache_entry.get("final_pos_error_m", 0.0)),
        final_rot_error_deg=float(cache_entry.get("final_rot_error_deg", 0.0)),
        final_pose_error=float(cache_entry.get("final_pose_error", 0.0)),
        side=side,
        pose_mode=pose_mode,
        start_xyz=_round_list(path_xyz[0], 5) if path_xyz else [],
        goal_xyz=_round_list(xyz, 5),
        path_xyz=path_xyz,
        preview_url=preview_url,
    )


def _run_go_home() -> HomeResponse:
    try:
        client = _get_portal()
        ok = bool(client.go_home().result())
        return HomeResponse(ok=ok, error=None if ok else "go_home returned false")
    except Exception as e:
        logger.exception("[anygrasp-debug] go_home error")
        return HomeResponse(ok=False, error=str(e))


def _run_gripper(side: str, value: float, action: str) -> HomeResponse:
    try:
        side = str(side).strip().lower()
        if side not in {"left", "right"}:
            return HomeResponse(ok=False, error=f"unsupported side: {side!r}")
        client = _get_portal()
        ok = bool(
            client.set_gripper(
                side, value, GRIPPER_SETTLE_TIMEOUT_S, None, None
            ).result()
        )
        return HomeResponse(
            ok=ok,
            error=None if ok else f"{action} returned false for side={side}",
        )
    except Exception as e:
        logger.exception("[anygrasp-debug] %s error", action)
        return HomeResponse(ok=False, error=str(e))


def _run_open_gripper(side: str) -> HomeResponse:
    return _run_gripper(side, 1.0, "open_gripper")


def _run_close_gripper(side: str) -> HomeResponse:
    return _run_gripper(side, 0.0, "close_gripper")


def _execute_cached_preview(
    req: PlannerRequest,
    cache_entry: dict[str, Any],
    *,
    used_cached_preview: bool,
) -> PlannerResponse:
    tool = _get_freespace_tool()
    side = str(req.side).strip().lower()
    pose_mode = str(req.pose_mode).strip().lower()
    execute_fraction = float(np.clip(float(req.execute_fraction), 0.0, 1.0))
    client = _get_portal()

    if execute_fraction <= 0.0:
        return PlannerResponse(
            ok=True,
            status="Success",
            executed=False,
            used_cached_preview=used_cached_preview,
            execute_fraction=0.0,
            preview_cache_state="cached",
            preview_cache_message="Cached execution trajectory kept; execute fraction was 0.0 so no motion was sent.",
            trajectory_steps=0,
            reason="Execute fraction is 0.0; nothing executed.",
            side=side,
            pose_mode=pose_mode,
            start_xyz=_round_list(cache_entry["path_xyz"][0], 5)
            if cache_entry.get("path_xyz")
            else [],
            goal_xyz=_round_list(req.xyz, 5),
            path_xyz=[],
        )

    current_state = client.get_state().result()
    cur_left_jp = np.asarray(current_state["left_joint_pos"], dtype=np.float64)
    cur_right_jp = np.asarray(current_state["right_joint_pos"], dtype=np.float64)
    if not _preview_state_matches_current(cache_entry, cur_left_jp, cur_right_jp):
        return PlannerResponse(
            ok=False,
            status="Preview_Stale",
            executed=False,
            used_cached_preview=used_cached_preview,
            execute_fraction=execute_fraction,
            preview_cache_state="stale",
            preview_cache_message="Cached execution trajectory is stale because the robot state changed.",
            side=side,
            pose_mode=pose_mode,
            goal_xyz=_round_list(req.xyz, 5),
            reason=(
                "Robot state changed since preview. Re-preview this grasp before executing "
                "to keep execution consistent with the previewed trajectory."
            ),
            error="Previewed trajectory is stale",
        )

    left_positions, right_positions, timestamps, path_xyz, keep = (
        _truncate_preview_trajectory(cache_entry, execute_fraction)
    )
    if keep <= 1 or len(timestamps) <= 1:
        return PlannerResponse(
            ok=True,
            status="Success",
            executed=False,
            used_cached_preview=used_cached_preview,
            execute_fraction=execute_fraction,
            preview_cache_state="cached",
            preview_cache_message="Cached execution trajectory kept; increase t above 0 to execute a motion segment.",
            trajectory_steps=keep,
            reason=(
                "Execute fraction is too small to include any motion segment; "
                "increase t above 0 to move."
            ),
            side=side,
            pose_mode=pose_mode,
            start_xyz=_round_list(path_xyz[0], 5) if path_xyz else [],
            goal_xyz=_round_list(req.xyz, 5),
            path_xyz=path_xyz,
        )

    gripper_ok = bool(client.set_gripper(side, 1.0).result())
    if not gripper_ok:
        return PlannerResponse(
            ok=False,
            status="Gripper_Failed",
            executed=False,
            used_cached_preview=used_cached_preview,
            execute_fraction=execute_fraction,
            preview_cache_state="cached",
            preview_cache_message="Cached execution trajectory is still available, but opening the gripper failed.",
            side=side,
            pose_mode=pose_mode,
            goal_xyz=_round_list(req.xyz, 5),
            reason=f"Failed to open {side} gripper before execution.",
            error="open_gripper failed",
        )

    exec_err = tool._execute_trajectory(
        client,
        side,
        timestamps,
        left_positions,
        right_positions,
        None,
        None,
    )
    if exec_err:
        return PlannerResponse(
            ok=False,
            status="Execution_Failed",
            executed=False,
            used_cached_preview=used_cached_preview,
            execute_fraction=execute_fraction,
            preview_cache_state="cached",
            preview_cache_message="Cached execution trajectory remained valid, but trajectory execution failed.",
            trajectory_steps=keep,
            side=side,
            pose_mode=pose_mode,
            start_xyz=_round_list(path_xyz[0], 5) if path_xyz else [],
            goal_xyz=_round_list(req.xyz, 5),
            path_xyz=path_xyz,
            reason=str(exec_err),
            error=str(exec_err),
        )

    if all(
        key in cache_entry
        for key in (
            "ik_xyz_weight",
            "ik_rpy_weight",
            "solver_speed",
            "target_left_pos",
            "target_left_quat_xyzw",
            "target_right_pos",
            "target_right_quat_xyzw",
        )
    ):
        planner = _tool_get_planner_with_solver_speed(
            tool,
            ik_xyz_weight=float(cache_entry["ik_xyz_weight"]),
            ik_rpy_weight=float(cache_entry["ik_rpy_weight"]),
            solver_speed=str(cache_entry["solver_speed"]),
        )
        diagnostic_kin = tool._get_diagnostic_kinematics(
            position_cost=float(cache_entry["ik_xyz_weight"]),
            orientation_cost=float(cache_entry["ik_rpy_weight"]),
        )
        diag = tool._compute_plan_diagnostics(
            diagnostic_kin,
            planner,
            side=side,
            final_left_joint_pos=left_positions[-1],
            final_right_joint_pos=right_positions[-1],
            target_left_pos=np.asarray(
                cache_entry["target_left_pos"], dtype=np.float64
            ),
            target_left_quat_xyzw=np.asarray(
                cache_entry["target_left_quat_xyzw"], dtype=np.float64
            ),
            target_right_pos=np.asarray(
                cache_entry["target_right_pos"], dtype=np.float64
            ),
            target_right_quat_xyzw=np.asarray(
                cache_entry["target_right_quat_xyzw"], dtype=np.float64
            ),
        )
        final_pos_error_m = float(diag["final_pos_error_m"])
        final_rot_error_deg = float(diag["final_rot_error_deg"])
        final_pose_error = float(diag["final_pose_error"])
    else:
        final_pos_error_m = float(cache_entry.get("final_pos_error_m", 0.0))
        final_rot_error_deg = float(cache_entry.get("final_rot_error_deg", 0.0))
        final_pose_error = float(cache_entry.get("final_pose_error", 0.0))

    partial_reason = None
    if execute_fraction < 1.0:
        partial_reason = (
            f"Executed the first {execute_fraction:.3f} fraction of the cached execution "
            f"trajectory ({keep}/{len(cache_entry['timestamps'])} waypoints)."
        )

    return PlannerResponse(
        ok=True,
        status="Success",
        executed=True,
        used_cached_preview=used_cached_preview,
        execute_fraction=execute_fraction,
        preview_cache_state="cached",
        preview_cache_message=(
            "Executed using the cached execution trajectory with no replanning."
            if used_cached_preview
            else "Executed using the trajectory cached immediately before execution."
        ),
        trajectory_steps=keep,
        final_pos_error_m=final_pos_error_m,
        final_rot_error_deg=final_rot_error_deg,
        final_pose_error=final_pose_error,
        reason=partial_reason,
        side=side,
        pose_mode=pose_mode,
        start_xyz=_round_list(path_xyz[0], 5) if path_xyz else [],
        goal_xyz=_round_list(req.xyz, 5),
        path_xyz=path_xyz,
    )


def _run_sort_grasps_by_ik_error(req: SortIkRequest) -> SortIkResponse:
    total_start = time.perf_counter()
    try:
        side = str(req.side).strip().lower()
        pose_mode = str(req.pose_mode or "2d").strip().lower()
        solver_speed = _normalize_solver_speed(req.solver_speed)
        if side not in {"left", "right"}:
            return SortIkResponse(ok=False, error=f"unsupported side: {req.side!r}")
        if pose_mode not in {"planner", "2d"}:
            return SortIkResponse(
                ok=False, error=f"unsupported pose_mode: {req.pose_mode!r}"
            )
        if not req.grasps:
            return SortIkResponse(
                ok=True,
                side=side,
                pose_mode=pose_mode,
                ik_error_threshold=float(req.ik_error_threshold),
                grasps=[],
                planning_mode="none",
            )

        tool = _get_freespace_tool()
        plan_eval_start = time.perf_counter()
        batch_candidates = []
        for idx, grasp in enumerate(req.grasps, start=1):
            xyz = grasp.two_d_xyz if pose_mode == "2d" else grasp.planner_xyz
            rpy = grasp.two_d_rpy if pose_mode == "2d" else grasp.planner_rpy
            if not (
                isinstance(xyz, list)
                and len(xyz) == 3
                and isinstance(rpy, list)
                and len(rpy) == 3
            ):
                return SortIkResponse(
                    ok=False,
                    side=side,
                    pose_mode=pose_mode,
                    error=(
                        f"grasp #{idx} is missing a valid {pose_mode} pose; rerun AnyGrasp before "
                        "batch previewing this pose mode."
                    ),
                )
            batch_candidates.append(
                {
                    "position": list(map(float, xyz)),
                    "rpy": list(map(float, rpy)),
                    "score": float(grasp.score),
                    "width": float(grasp.width),
                }
            )
        batch_tool_result = tool.execute(
            grasp_candidates=batch_candidates,
            batch_side=side,
            batch_top_k=int(MAX_SORT_IK_GRASPS),
            solver_speed=solver_speed,
            planning_speed=float(req.planning_speed),
            ik_error_threshold=float(req.ik_error_threshold),
            ik_xyz_weight=float(req.ik_xyz_weight),
            ik_rpy_weight=float(req.ik_rpy_weight),
            planner_backend="curobo",
            batch_validate_trajectory=False,
        )
        plan_eval_elapsed_ms = (time.perf_counter() - plan_eval_start) * 1000.0
        batch_data = batch_tool_result.data
        if not batch_tool_result.success:
            total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            error_msg = str(
                getattr(batch_data, "reason", "")
                or batch_tool_result.error
                or "batch cuRobo planning did not return a usable result"
            )
            logger.error(
                "[anygrasp-debug] batch-only sort_by_ik_error failed side=%s input=%d plan=%.1fms reason=%s",
                side,
                len(req.grasps),
                plan_eval_elapsed_ms,
                error_msg,
            )
            return SortIkResponse(
                ok=False,
                error=f"Batch cuRobo sort failed: {error_msg}",
                side=side,
                pose_mode=pose_mode,
                ik_error_threshold=float(req.ik_error_threshold),
                timing_total_ms=round(total_elapsed_ms, 3),
                timing_plan_eval_ms=round(plan_eval_elapsed_ms, 3),
                curobo_solve_time_ms=round(
                    float(getattr(batch_data, "curobo_solve_time_ms", 0.0)), 3
                ),
                curobo_total_time_ms=round(
                    float(getattr(batch_data, "curobo_total_time_ms", 0.0)), 3
                ),
                curobo_graph_time_ms=round(
                    float(getattr(batch_data, "curobo_graph_time_ms", 0.0)), 3
                ),
                curobo_ik_time_ms=round(
                    float(getattr(batch_data, "curobo_ik_time_ms", 0.0)), 3
                ),
                planning_mode="batch_error",
                input_grasp_count=int(
                    getattr(batch_data, "input_candidate_count", len(req.grasps))
                ),
                evaluated_grasp_count=int(
                    getattr(batch_data, "evaluated_candidate_count", 0)
                ),
                truncated_input_count=int(
                    getattr(batch_data, "truncated_input_count", 0)
                ),
                batch_attempted=bool(getattr(batch_data, "batch_attempted", False)),
                batch_error=error_msg,
            )

        postprocess_start = time.perf_counter()
        ranked_rows: list[GraspPoseRow] = []
        for candidate in list(getattr(batch_data, "batch_candidates", []) or []):
            source_index = int(getattr(candidate, "source_index", -1))
            if not (0 <= source_index < len(req.grasps)):
                continue
            row_data = req.grasps[source_index].model_dump()
            row_data["rank"] = int(getattr(candidate, "rank", 0))
            row_data["ik_error_m"] = getattr(candidate, "ik_error_m", None)
            row_data["ik_rot_error_deg"] = getattr(candidate, "ik_rot_error_deg", None)
            row_data["within_ik_threshold"] = getattr(
                candidate, "within_ik_threshold", None
            )
            row_data["motion_plan_error"] = getattr(
                candidate, "motion_plan_error", None
            )
            row_data["motion_plan_reason"] = getattr(
                candidate, "motion_plan_reason", None
            )
            row_data["trajectory_cache_key"] = getattr(
                candidate, "trajectory_cache_key", None
            )
            row_data["trajectory_cache_pose_mode"] = (
                pose_mode if row_data["trajectory_cache_key"] else None
            )
            row_data["trajectory_steps"] = int(
                getattr(candidate, "trajectory_steps", 0) or 0
            )
            ranked_rows.append(GraspPoseRow(**row_data))
        postprocess_elapsed_ms = (time.perf_counter() - postprocess_start) * 1000.0
        rank_elapsed_ms = 0.0
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        logger.info(
            "[anygrasp-debug] sort_by_ik_error side=%s mode=%s input=%d evaluated=%d total=%.1fms plan=%.1fms post=%.1fms rank=%.1fms curobo_solve=%.1fms curobo_graph=%.1fms curobo_ik=%.1fms",
            side,
            pose_mode,
            len(req.grasps),
            int(getattr(batch_data, "evaluated_candidate_count", len(ranked_rows))),
            total_elapsed_ms,
            plan_eval_elapsed_ms,
            postprocess_elapsed_ms,
            rank_elapsed_ms,
            float(getattr(batch_data, "curobo_solve_time_ms", 0.0)),
            float(getattr(batch_data, "curobo_graph_time_ms", 0.0)),
            float(getattr(batch_data, "curobo_ik_time_ms", 0.0)),
        )

        return SortIkResponse(
            ok=True,
            side=side,
            pose_mode=pose_mode,
            ik_error_threshold=float(req.ik_error_threshold),
            grasps=ranked_rows,
            timing_total_ms=round(total_elapsed_ms, 3),
            timing_plan_eval_ms=round(plan_eval_elapsed_ms, 3),
            timing_postprocess_ms=round(postprocess_elapsed_ms, 3),
            timing_rank_ms=round(rank_elapsed_ms, 3),
            curobo_solve_time_ms=round(
                float(getattr(batch_data, "curobo_solve_time_ms", 0.0)), 3
            ),
            curobo_total_time_ms=round(
                float(getattr(batch_data, "curobo_total_time_ms", 0.0)), 3
            ),
            curobo_graph_time_ms=round(
                float(getattr(batch_data, "curobo_graph_time_ms", 0.0)), 3
            ),
            curobo_ik_time_ms=round(
                float(getattr(batch_data, "curobo_ik_time_ms", 0.0)), 3
            ),
            planning_mode=str(getattr(batch_data, "planning_mode", "unknown")),
            input_grasp_count=int(
                getattr(batch_data, "input_candidate_count", len(req.grasps))
            ),
            evaluated_grasp_count=int(
                getattr(batch_data, "evaluated_candidate_count", len(ranked_rows))
            ),
            truncated_input_count=int(getattr(batch_data, "truncated_input_count", 0)),
            batch_attempted=bool(getattr(batch_data, "batch_attempted", False)),
            batch_error=getattr(batch_data, "batch_error", None),
        )
    except Exception as e:
        logger.exception("[anygrasp-debug] sort_by_ik_error error")
        return SortIkResponse(ok=False, error=str(e))


def _run_birdeyeview(planning_speed: float | None = None) -> HomeResponse:
    try:
        config = _load_local_pose_config()
        tool = _get_freespace_tool()
        speed = float(
            planning_speed
            if planning_speed is not None
            else config.get("PLANNING_SPEED", _LOCAL_POSE_DEFAULTS["PLANNING_SPEED"])
        )
        if not np.isfinite(speed) or speed <= 0.0:
            speed = float(
                config.get("PLANNING_SPEED", _LOCAL_POSE_DEFAULTS["PLANNING_SPEED"])
            )
        result = tool.execute(
            left_target_pos=[
                LEFT_HOME_XYZ[0],
                LEFT_HOME_XYZ[1],
                LEFT_HOME_XYZ[2] + HOME_VIEW_Z_OFFSET,
            ],
            left_target_rpy=LEFT_BIRDEYE_VIEW_RPY,
            right_target_pos=[
                RIGHT_HOME_XYZ[0],
                RIGHT_HOME_XYZ[1],
                RIGHT_HOME_XYZ[2] + HOME_VIEW_Z_OFFSET,
            ],
            right_target_rpy=RIGHT_BIRDEYE_VIEW_RPY,
            left_gripper_target_width=1.0,
            right_gripper_target_width=1.0,
            planning_speed=speed,
            ik_error_threshold=DEFAULT_IK_THRESHOLD_M,
            ik_xyz_weight=1.0,
            ik_rpy_weight=0.3,
        )
        return HomeResponse(ok=bool(result.success), error=result.error)
    except Exception as e:
        logger.exception("[anygrasp-debug] birdeyeview error")
        return HomeResponse(ok=False, error=str(e))


def _run_local_pose(planning_speed: float | None = None) -> HomeResponse:
    try:
        config = _load_local_pose_config()
        tool = _get_freespace_tool()
        speed = float(
            planning_speed
            if planning_speed is not None
            else config.get("PLANNING_SPEED", _LOCAL_POSE_DEFAULTS["PLANNING_SPEED"])
        )
        if not np.isfinite(speed) or speed <= 0.0:
            speed = float(
                config.get("PLANNING_SPEED", _LOCAL_POSE_DEFAULTS["PLANNING_SPEED"])
            )

        result = tool.execute(
            left_target_pos=[float(v) for v in config["LEFT_TARGET_POS"]],
            left_target_rpy=[float(v) for v in config["LEFT_TARGET_RPY"]],
            right_target_pos=[float(v) for v in config["RIGHT_TARGET_POS"]],
            right_target_rpy=[float(v) for v in config["RIGHT_TARGET_RPY"]],
            left_gripper_target_width=1.0,
            right_gripper_target_width=1.0,
            planning_speed=speed,
            ik_error_threshold=float(config["IK_ERROR_THRESHOLD_M"]),
            ik_xyz_weight=float(config["IK_XYZ_WEIGHT"]),
            ik_rpy_weight=float(config["IK_RPY_WEIGHT"]),
            planner_backend=str(config["MOTION_PLANNER_BACKEND"]),
        )
        return HomeResponse(ok=bool(result.success), error=result.error)
    except Exception as e:
        logger.exception("[anygrasp-debug] local_pose error")
        return HomeResponse(ok=False, error=str(e))


def _get_camera_data(
    camera: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    client = _get_portal()
    rgb = np.asarray(client.get_camera_image(camera).result())
    depth = np.asarray(client.get_camera_depth(camera).result()).astype(np.float32)
    intr_raw = client.get_camera_intrinsics(camera).result()
    fx, fy, cx, cy = [float(x) for x in intr_raw]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    extr = client.get_camera_extrinsics(camera).result()
    R = np.asarray(extr["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(extr["position"], dtype=np.float64)
    T_cam_world = np.eye(4, dtype=np.float64)
    T_cam_world[:3, 3] = t

    # The D405 URDF body frame has X/Y axes flipped vs the optical (OpenCV)
    # convention, so a diag(-1,-1,1) correction is needed.  The ZED 2i
    # calibration already produces the optical frame — no flip required.
    from enpire.env.forge.robot.models.station.paths import needs_optical_flip

    if needs_optical_flip(camera):
        F = np.diag([-1.0, -1.0, 1.0])
        T_cam_world[:3, :3] = R @ F
    else:
        T_cam_world[:3, :3] = R

    return rgb, depth, K, T_cam_world


def _segment_object(rgb: np.ndarray, prompt: str) -> np.ndarray:
    buf = io.BytesIO()
    np.save(buf, rgb)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = json.dumps({"text": prompt, "image_b64": image_b64}).encode()
    req = urllib.request.Request(
        f"{SAM3_URL}/segment",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except Exception:
            detail = body
        if int(e.code) == 404:
            raise PipelineUserError(
                "object_not_found", str(detail), log_level="info"
            ) from None
        raise PipelineUserError(
            "sam3_error",
            f"SAM3 /segment failed ({e.code}): {detail}",
            log_level="warning",
        ) from None

    data = json.loads(resp.read())
    mask_bytes = base64.b64decode(data["mask_b64"])
    return np.load(io.BytesIO(mask_bytes)).astype(np.int32)


def _call_anygrasp_viz(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    segmap: np.ndarray,
    *,
    max_grasps: int,
    z_range: list[float],
    workspace_margin: float,
    collision_detection: bool,
    object_input_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes, float | None]:
    payload = {
        "rgb_base64": _np_to_b64(rgb),
        "depth_base64": _np_to_b64(depth),
        "cam_K_base64": _np_to_b64(K),
        "segmap_base64": _np_to_b64(segmap),
        "segmap_id": 1,
        "z_range": z_range,
        "max_grasps": max_grasps,
        "workspace_margin": workspace_margin,
        "collision_detection": collision_detection,
        "object_input_mode": object_input_mode,
    }
    try:
        resp = requests.post(f"{ANYGRASP_URL}/plan_viz", json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise PipelineUserError(
            "backend_unreachable",
            f"Cannot reach AnyGrasp at {ANYGRASP_URL}/plan_viz: {e}. "
            "tools/vision/serve_anygrasp.py listens on http://localhost:8122 by default.",
            log_level="warning",
        ) from None
    except requests.exceptions.HTTPError as e:
        detail = e.response.text if e.response is not None else str(e)
        try:
            parsed = json.loads(detail)
            detail = parsed.get("detail", detail)
        except Exception:
            pass
        raise PipelineUserError(
            "anygrasp_error",
            f"AnyGrasp /plan_viz failed ({e.response.status_code if e.response else 'error'}): {detail}",
            log_level="warning",
        ) from None
    except requests.exceptions.RequestException as e:
        raise PipelineUserError(
            "anygrasp_error",
            f"AnyGrasp /plan_viz request failed: {e}",
            log_level="warning",
        ) from None

    data = resp.json()
    grasps = (
        _b64_to_np(data["grasps_base64"])
        if data.get("grasps_base64")
        else np.empty((0, 4, 4), dtype=np.float64)
    )
    scores = (
        _b64_to_np(data["scores_base64"])
        if data.get("scores_base64")
        else np.empty((0,), dtype=np.float64)
    )
    widths = (
        _b64_to_np(data["widths_base64"])
        if data.get("widths_base64")
        else np.empty((0,), dtype=np.float64)
    )
    overlay_bytes = base64.b64decode(data["overlay_jpeg_base64"])
    return grasps, scores, widths, overlay_bytes, data.get("best_score")


def _pose_rows(
    grasps_cam_raw: np.ndarray,
    scores: np.ndarray,
    widths: np.ndarray,
    T_cam_world: np.ndarray,
    segmap: np.ndarray,
    cam_K: np.ndarray,
    *,
    tcp_offset_z_m: float,
    disable_planner_z_clipping: bool,
) -> tuple[list[GraspPoseRow], int]:
    from scipy.spatial.transform import Rotation

    if grasps_cam_raw.size == 0:
        return [], 0

    order = np.argsort(-scores)
    grasps_cam_raw = grasps_cam_raw[order]
    scores = scores[order]
    widths = widths[order] if len(widths) >= len(order) else widths

    raw_world = np.matmul(T_cam_world, grasps_cam_raw)
    planner_world = np.matmul(raw_world, _ANYGRASP_TO_GRIPPER_T)
    planner_translated = planner_world.copy()
    planner_translated[:, :3, 3] = (
        planner_world[:, :3, 3] + float(tcp_offset_z_m) * planner_world[:, :3, 2]
    )
    n_z_clipped = 0
    if not disable_planner_z_clipping:
        n_z_clipped = _clip_planner_z_in_place(planner_translated)
    if n_z_clipped:
        logger.info(
            "[anygrasp-debug] Clipped %d planner-facing grasp z value(s) to floor %.4f m",
            n_z_clipped,
            float(ANYGRASP_MIN_PLANNER_Z_M),
        )

    rows: list[GraspPoseRow] = []
    for i in range(len(raw_world)):
        raw_T = raw_world[i]
        plan_T = planner_translated[i]
        plan_rpy = _round_list(
            _quat_xyzw_to_display_rpy_deg(
                Rotation.from_matrix(plan_T[:3, :3]).as_quat()
            ),
            3,
        )
        two_d_xyz, two_d_rpy = _derive_two_d_pose(
            mask=segmap,
            cam_K=cam_K,
            T_cam_world=T_cam_world,
            planner_xyz=np.asarray(plan_T[:3, 3], dtype=np.float64),
            planner_yaw_deg=float(plan_rpy[2]),
            plane_z_m=float(TWO_D_TOP_DOWN_Z_M),
        )
        width = float(widths[i]) if i < len(widths) else 0.08
        rows.append(
            GraspPoseRow(
                rank=i + 1,
                score=round(float(scores[i]), 4),
                width=round(width, 5),
                raw_xyz=_round_list(raw_T[:3, 3], 5),
                raw_rpy=_round_list(
                    Rotation.from_matrix(raw_T[:3, :3]).as_euler("xyz", degrees=True), 3
                ),
                planner_xyz=_round_list(plan_T[:3, 3], 5),
                planner_rpy=plan_rpy,
                two_d_xyz=two_d_xyz,
                two_d_rpy=two_d_rpy,
            )
        )
    return rows, int(n_z_clipped)


def _run_pipeline(req: RunRequest) -> RunResponse:
    try:
        t0 = time.perf_counter()
        tcp_offset_z_m = float(req.tcp_offset_z_m)
        disable_planner_z_clipping = bool(req.disable_planner_z_clipping)
        rgb, depth, K, T_cam_world = _get_camera_data(req.camera)
        h, w = rgb.shape[:2]
        segmap = _segment_object(rgb, req.prompt)
        _points_all, scene_mask, scene_points, scene_colors = _frame_to_scene(
            rgb,
            depth,
            K,
            z_range=[req.z_range_min, req.z_range_max],
        )
        object_flags = ((segmap == 1) & scene_mask)[scene_mask]
        object_points = (
            scene_points[object_flags]
            if np.any(object_flags)
            else np.empty((0, 3), dtype=np.float32)
        )
        (
            segmented_cloud_z_min_m,
            segmented_cloud_z_max_m,
            segmented_cloud_thickness_m,
        ) = _point_cloud_z_extent(object_points)
        cloud_preview_url = None
        try:
            cloud_preview_url = _get_cloud_viser().update(
                scene_points=scene_points,
                scene_colors=scene_colors,
                object_points=object_points,
                grasp_poses_cam=np.empty((0, 4, 4), dtype=np.float64),
                grasp_scores=np.empty((0,), dtype=np.float64),
            )
        except Exception:
            logger.exception("[anygrasp-debug] Failed to update cloud preview")
        grasps, scores, widths, overlay_bytes, best_score = _call_anygrasp_viz(
            rgb,
            depth,
            K,
            segmap,
            max_grasps=req.max_grasps,
            z_range=[req.z_range_min, req.z_range_max],
            workspace_margin=req.workspace_margin,
            collision_detection=req.collision_detection,
            object_input_mode=req.object_input_mode,
        )
        pose_rows, n_z_clipped = _pose_rows(
            grasps,
            scores,
            widths,
            T_cam_world,
            segmap,
            K,
            tcp_offset_z_m=tcp_offset_z_m,
            disable_planner_z_clipping=disable_planner_z_clipping,
        )
        try:
            cloud_preview_url = _get_cloud_viser().update(
                scene_points=scene_points,
                scene_colors=scene_colors,
                object_points=object_points,
                grasp_poses_cam=grasps,
                grasp_scores=scores,
            )
        except Exception:
            logger.exception(
                "[anygrasp-debug] Failed to update cloud preview with grasps"
            )
        return RunResponse(
            status="ok" if len(scores) > 0 else "no_grasps",
            n_grasps=int(len(scores)),
            best_score=float(best_score)
            if best_score is not None
            else (float(scores[0]) if len(scores) else None),
            overlay_b64=base64.b64encode(overlay_bytes).decode(),
            cloud_preview_url=cloud_preview_url,
            image_width=w,
            image_height=h,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            object_input_mode=str(req.object_input_mode),
            segmented_cloud_z_min_m=segmented_cloud_z_min_m,
            segmented_cloud_z_max_m=segmented_cloud_z_max_m,
            segmented_cloud_thickness_m=segmented_cloud_thickness_m,
            two_d_top_down_z_m=float(TWO_D_TOP_DOWN_Z_M),
            tcp_offset_z_m=tcp_offset_z_m,
            planner_z_floor_m=float(ANYGRASP_MIN_PLANNER_Z_M),
            planner_z_clipping_enabled=not disable_planner_z_clipping,
            n_planner_z_clipped=int(n_z_clipped),
            grasps=pose_rows,
        )
    except PipelineUserError as e:
        log_fn = logger.warning if e.log_level == "warning" else logger.info
        log_fn("[anygrasp-debug] %s", e.message)
        return RunResponse(status=e.status, error=e.message)
    except Exception as e:
        logger.exception("[anygrasp-debug] Pipeline error")
        return RunResponse(status="error", error=str(e))


def _run_freespace(req: PlannerRequest) -> PlannerResponse:
    try:
        side = req.side.strip().lower()
        solver_speed = _normalize_solver_speed(req.solver_speed)
        if side not in {"left", "right"}:
            return PlannerResponse(
                ok=False,
                status="Invalid",
                executed=False,
                reason=f"Unsupported side: {req.side!r}. Use 'left' or 'right'.",
                error="Invalid side",
            )

        xyz = [float(v) for v in req.xyz]
        rpy = [float(v) for v in req.rpy]
        if len(xyz) != 3 or len(rpy) != 3:
            return PlannerResponse(
                ok=False,
                status="Invalid",
                executed=False,
                reason="Expected xyz and rpy to each have exactly 3 numbers.",
                error="Invalid pose length",
            )

        if req.trajectory_cache_key:
            cache_entry = _build_cached_batch_preview_entry(req)
            if cache_entry is None:
                return PlannerResponse(
                    ok=False,
                    status="Preview_Missing",
                    executed=False,
                    used_cached_preview=False,
                    execute_fraction=float(
                        np.clip(float(req.execute_fraction), 0.0, 1.0)
                    ),
                    side=side,
                    pose_mode=req.pose_mode.strip().lower(),
                    goal_xyz=_round_list(xyz, 5),
                    reason=(
                        "Cached batch trajectory was not found. Re-run batch preview/sort before "
                        "executing."
                    ),
                    error="Batch trajectory cache missing",
                )
            if not bool(req.execute):
                return _preview_cached_trajectory(req, cache_entry)
            return _execute_cached_preview(
                req,
                cache_entry,
                used_cached_preview=True,
            )

        if not bool(req.execute):
            return _preview_pose_with_path(req)

        if side not in {"left", "right"}:
            return PlannerResponse(
                ok=False,
                status="Invalid",
                executed=False,
                reason=f"Unsupported side: {req.side!r}. Use 'left' or 'right'.",
                error="Invalid side",
            )
        if req.pose_mode.strip().lower() not in {"planner", "2d"}:
            return PlannerResponse(
                ok=False,
                status="Invalid",
                executed=False,
                side=side,
                pose_mode=req.pose_mode.strip().lower(),
                goal_xyz=_round_list(xyz, 5),
                reason="Execute is only enabled for planner-facing poses ('planner' or '2d').",
                error="Raw execute disabled",
            )
        cache_entry = _get_cached_preview_trajectory(req)
        if cache_entry is None:
            return PlannerResponse(
                ok=False,
                status="Preview_Missing",
                executed=False,
                used_cached_preview=False,
                execute_fraction=float(np.clip(float(req.execute_fraction), 0.0, 1.0)),
                side=side,
                pose_mode=req.pose_mode.strip().lower(),
                goal_xyz=_round_list(xyz, 5),
                reason=(
                    "Execute never replans. Preview this grasp first, then execute the cached "
                    "trajectory."
                ),
                error="Preview trajectory cache missing",
            )
        return _execute_cached_preview(
            req,
            cache_entry,
            used_cached_preview=True,
        )
    except Exception as e:
        logger.exception("[anygrasp-debug] freespace_move error")
        return PlannerResponse(
            ok=False,
            status="Error",
            executed=False,
            used_cached_preview=False,
            execute_fraction=float(np.clip(float(req.execute_fraction), 0.0, 1.0)),
            reason=str(e),
            error=str(e),
        )


_DEBUG_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AnyGrasp Debug</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; background: #111; color: #eee; font-family: sans-serif; }
  #control-bar { padding: 10px; background: #222; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; border-bottom: 1px solid #2a2a2a; }
  .inp { padding: 7px 11px; background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; font-size: 13px; outline: none; }
  .inp:focus { border-color: #0a84ff; }
  #prompt-inp { flex: 2; min-width: 220px; }
  #camera-sel { min-width: 75px; }
  #run-btn { padding: 7px 18px; background: #0a84ff; color: #fff; border: none; border-radius: 4px; font-size: 13px; cursor: pointer; white-space: nowrap; }
  #run-btn:disabled { background: #555; cursor: default; }
  #status { font-size: 12px; color: #aaa; flex: 1; min-width: 160px; }
  #params-bar { padding: 8px 12px; background: #1a1a1a; border-bottom: 1px solid #222; }
  #params-toggle { background: none; border: none; color: #888; font-size: 12px; cursor: pointer; padding: 0; }
  #params-panel { display: none; margin-top: 10px; gap: 18px; flex-wrap: wrap; align-items: center; }
  #params-panel.open { display: flex; }
  .param-group { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #aaa; }
  .param-group .inp { width: 82px; padding: 4px 7px; font-size: 12px; }
  #body {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    grid-template-areas:
      "preview overlay"
      "depth cloud"
      "planner planner";
    gap: 12px;
    padding: 12px;
    align-items: stretch;
  }
  .panel { background: #181818; border: 1px solid #252525; border-radius: 8px; padding: 12px; min-width: 0; min-height: 0; }
  .hidden { display: none !important; }
  .media-panel { display: flex; flex-direction: column; }
  .panel-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 8px; }
  .img-label { font-size: 11px; color: #777; text-transform: uppercase; letter-spacing: 0.06em; }
  .panel-meta { font-size: 12px; color: #aaa; text-align: right; }
  img { max-width: 100%; border-radius: 4px; background: #000; display: block; }
  .viz-media { width: 100%; aspect-ratio: 4 / 3; border-radius: 4px; background: #000; display: block; object-fit: contain; }
  #preview-panel { grid-area: preview; }
  #cloud-panel { grid-area: cloud; }
  #depth-panel { grid-area: depth; }
  #overlay-panel { grid-area: overlay; }
  #planner-panel { grid-area: planner; display: flex; flex-direction: column; min-height: 0; }
  #cloud-frame { border: 0; }
  #cloud-note, #depth-note { font-size: 12px; color: #888; margin-top: 8px; line-height: 1.45; }
  #results-label { font-size: 12px; color: #aaa; }
  #pose-note { font-size: 12px; color: #aaa; margin: 12px 0; line-height: 1.5; }
  #planner-bar { margin: 12px 0; display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  #planner-status { font-size: 12px; color: #aaa; min-height: 18px; }
  .status-stack { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 8px; border-radius: 999px; font-size: 11px; letter-spacing: 0.02em; border: 1px solid #3a3a3a; color: #cfcfcf; background: #1e1e1e; }
  .status-badge.cached { border-color: #2a8f4a; color: #b9f5c8; background: rgba(46, 160, 67, 0.14); }
  .status-badge.used { border-color: #0a84ff; color: #b7d8ff; background: rgba(10, 132, 255, 0.14); }
  .status-badge.stale { border-color: #d29922; color: #ffd891; background: rgba(210, 153, 34, 0.16); }
  .status-badge.none { border-color: #4a4a4a; color: #b8b8b8; background: rgba(120, 120, 120, 0.10); }
  #planner-table-wrap { margin-top: 12px; border-top: 1px solid #2a2a2a; padding-top: 10px; overflow: auto; max-height: clamp(260px, 42vh, 560px); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #2a2a2a; vertical-align: top; }
  th { color: #999; font-weight: 600; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .planner-btn { padding: 5px 9px; background: #2d2d2d; color: #eee; border: 1px solid #555; border-radius: 4px; cursor: pointer; font-size: 12px; }
  .planner-btn:hover { border-color: #0a84ff; }
  .planner-btn:disabled { opacity: 0.55; cursor: default; }
  .planner-actions { display: flex; flex-wrap: wrap; gap: 6px; }
  tr.ik-pass td { background: rgba(46, 160, 67, 0.16); }
  tr.ik-fail td { background: rgba(248, 81, 73, 0.14); }
  .ik-cell-pass { color: #8df0a8; font-weight: 700; }
  .ik-cell-fail { color: #ff9a9a; font-weight: 700; }
  @media (max-width: 1200px) {
    #body {
      grid-template-columns: 1fr;
      grid-template-areas:
        "preview"
        "cloud"
        "depth"
        "overlay"
        "planner";
    }
    #planner-table-wrap { max-height: none; }
  }
</style>
</head>
<body>
<div id="control-bar">
  <input id="prompt-inp" class="inp" type="text" placeholder="Object to grasp (e.g. red mug)" oninput="onInput()" onkeydown="onKey(event)">
  <select id="camera-sel" class="inp">
    <option value="top">top</option>
    <option value="left">left</option>
    <option value="right">right</option>
  </select>
  <button id="run-btn" disabled onclick="runGrasps()">Run AnyGrasp</button>
  <button id="close-gripper-btn" class="planner-btn" onclick="closeGripper()">close_gripper</button>
  <button id="open-gripper-btn" class="planner-btn" onclick="openGripper()">open_gripper</button>
  <button id="home-btn" class="planner-btn" onclick="goHome()">Go Home</button>
  <button id="birdeye-btn" class="planner-btn" onclick="birdEyeView()">BirdEyeView</button>
  <button id="local-pose-btn" class="planner-btn" onclick="localPose()">LocalPose</button>
  <span id="status">Describe an object and click Run AnyGrasp.</span>
</div>
<div id="params-bar">
  <button id="params-toggle" onclick="toggleParams()">&#9881; Parameters &#9660;</button>
  <div id="params-panel">
    <div class="param-group"><label>Max grasps</label><input id="maxg-inp" class="inp" type="number" value="10" min="1" max="50"></div>
    <div class="param-group"><label>Z range (m)</label><input id="zmin-inp" class="inp" type="number" value="0.000001" step="0.05" min="0"><span style="color:#555">&#8212;</span><input id="zmax-inp" class="inp" type="number" value="1.5" step="0.1" min="0"></div>
    <div class="param-group"><label>Workspace margin</label><input id="margin-inp" class="inp" type="number" value="0.02" step="0.01" min="0"></div>
    <div class="param-group"><label><input id="collision-detection-inp" type="checkbox" checked style="margin-right:6px;">Collision detection</label></div>
    <div class="param-group"><label>TCP offset Z (m)</label><input id="tcp-offset-inp" class="inp" type="number" value="0.0" step="0.005"></div>
    <div class="param-group"><label><input id="disable-zclip-inp" type="checkbox" style="margin-right:6px;">Disable planner Z clipping</label></div>
  </div>
</div>
<div id="body">
  <div class="panel media-panel" id="preview-panel">
    <div class="panel-head">
      <div class="img-label">Live Preview</div>
    </div>
    <img id="preview-stream" class="viz-media" src="/preview/top" alt="camera preview">
  </div>
  <div class="panel media-panel" id="cloud-panel">
    <div class="panel-head">
      <div class="img-label">Point Cloud Sent to AnyGrasp</div>
    </div>
    <iframe id="cloud-frame" class="viz-media" title="AnyGrasp point cloud preview"></iframe>
    <div id="cloud-note">
      Interactive 3D preview of the same depth + intrinsics + z-range-filtered cloud sent to AnyGrasp.
      Drag to orbit, scroll to zoom. SAM3-selected object points are highlighted in green.
    </div>
  </div>
  <div class="panel media-panel" id="depth-panel">
    <div class="panel-head" style="margin-top:10px">
      <div class="img-label">Depth Used for Point Cloud</div>
    </div>
    <img id="depth-stream" class="viz-media" src="/preview_depth/top?zmin=0.000001&zmax=1.5" alt="depth preview">
    <div id="depth-note">
      This is the exact depth map returned by <code>get_camera_depth()</code> for the selected camera.
      Pixels outside the current AnyGrasp z-range are dimmed in the preview.
    </div>
  </div>
  <div class="panel media-panel" id="overlay-panel">
    <div class="panel-head">
      <div class="img-label">AnyGrasp Pose Overlay</div>
    </div>
    <img id="overlay-img" class="viz-media" alt="overlay" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==">
  </div>
  <div class="panel" id="planner-panel">
    <div class="panel-head">
      <div class="img-label">Motion Planner</div>
      <div class="status-stack">
        <span id="preview-cache-badge" class="status-badge none" title="No cached execution trajectory yet.">no cached exec traj</span>
        <div id="results-label" class="panel-meta">Awaiting AnyGrasp run.</div>
      </div>
    </div>
    <div id="pose-note">
      <b>Raw pose</b> = native AnyGrasp world pose.<br>
      <b>Planner pose</b> = planner-facing remap intended for <code>freespace_move</code>.<br>
      <b>2D pose</b> = Jalen-style top-down hybrid: keep AnyGrasp planner XY, use SAM3-mask yaw, and fix Z to the configured table-touch height.<br>
      Planner-facing AnyGrasp Z is clipped to <code>&gt;= 0.800 m</code> for table safety.
    </div>
  <div id="planner-bar">
      <div class="param-group"><label>Planner side</label><select id="planner-side" class="inp"><option value="left">left</option><option value="right">right</option></select></div>
      <div class="param-group"><label>Grasp mode</label><select id="grasp-mode" class="inp"><option value="2d" selected>2d</option><option value="planner">planner</option></select></div>
      <div class="param-group"><label>Solver speed</label><select id="solver-speed" class="inp"><option value="fast" selected>fast</option><option value="slow">slow</option></select></div>
      <div class="param-group"><label>Planner speed</label><input id="planner-speed" class="inp" type="number" value="0.5" min="0.05" max="3.0" step="0.05"></div>
      <div class="param-group"><label>IK thresh (m)</label><input id="planner-ik" class="inp" type="number" value="0.005" min="0.005" max="0.10" step="0.005"></div>
      <div class="param-group"><label>XYZ weight</label><input id="planner-xyz-weight" class="inp" type="number" value="1.0" min="0.001" max="20.0" step="0.05"></div>
      <div class="param-group"><label>RPY weight</label><input id="planner-rpy-weight" class="inp" type="number" value="0.3" min="0.0001" max="20.0" step="0.01"></div>
      <div class="param-group"><label>Execute fraction t</label><input id="execute-fraction" class="inp" type="number" value="1.0" min="0.0" max="1.0" step="0.05"></div>
      <div class="param-group"><label>Object input</label><select id="object-input-mode" class="inp"><option value="segmented_object_cloud" {"selected" if DEFAULT_OBJECT_INPUT_MODE == "segmented_object_cloud" else ""}>segmented object cloud</option><option value="roi_workspace" {"selected" if DEFAULT_OBJECT_INPUT_MODE == "roi_workspace" else ""}>roi workspace</option></select></div>
      <button id="sort-ik-btn" class="planner-btn" onclick="sortGraspsByIkError()">Batch Preview + Cache</button>
      <button id="execute-preview-prefix-btn" class="planner-btn" onclick="executeLastPreviewPrefix()">Execute Cached Prefix</button>
    </div>
    <div id="planner-status">
      <b>Preview Raw</b> plans the native AnyGrasp world pose directly and opens a 3D Viser page.<br>
      <b>Grasp mode = 2D</b> mirrors the current Python scripts: batch-preview, cache, and execute only the top-down transformed <code>2d xyz/rpy</code> grasps.<br>
      <b>Grasp mode = planner</b> uses the converted planner-facing AnyGrasp poses instead.<br>
      <b>Execute</b> only runs a cached trajectory. It does <b>not</b> preview again or replan.<br>
      <b>Execute Cached Prefix</b> executes only the first <code>t</code> fraction of the most recently cached planner/2D trajectory.<br>
      <b>Batch Preview + Cache</b> batch-plans the current grasps in the selected grasp mode, ranks them, and caches executable trajectories for the feasible ones. Collision / planning failures are treated as <code>∞ (collision)</code> and sorted to the bottom.
    </div>
    <div id="planner-table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Score</th><th>Width</th>
            <th>Raw XYZ</th><th>Raw RPY</th>
            <th>Planner XYZ</th><th>Planner RPY</th>
            <th>2D XYZ</th><th>2D RPY</th>
            <th>IK err (m)</th>
            <th>Motion Planner</th>
          </tr>
        </thead>
        <tbody id="grasps-body"></tbody>
      </table>
    </div>
  </div>
</div>
<script>
const promptInp = document.getElementById('prompt-inp');
const cameraSel = document.getElementById('camera-sel');
const runBtn = document.getElementById('run-btn');
const statusEl = document.getElementById('status');
const paramsPanel = document.getElementById('params-panel');
const cloudFrame = document.getElementById('cloud-frame');
const overlayPanel = document.getElementById('overlay-panel');
const plannerPanel = document.getElementById('planner-panel');
const overlayImg = document.getElementById('overlay-img');
const resultsLabel = document.getElementById('results-label');
const zminInp = document.getElementById('zmin-inp');
const zmaxInp = document.getElementById('zmax-inp');
const graspModeSel = document.getElementById('grasp-mode');
const previewImg = document.getElementById('preview-stream');
const depthImg = document.getElementById('depth-stream');
const previewCacheBadge = document.getElementById('preview-cache-badge');
const EMPTY_IMAGE = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==';
let running = false;
let plannerBusy = false;
let currentGrasps = [];
let lastPreviewRequest = null;
function refreshPreviewStreams() {
  const ts = Date.now();
  previewImg.src = `/preview/${cameraSel.value}?_ts=${ts}`;
  const zmin = encodeURIComponent(zminInp.value || '');
  const zmax = encodeURIComponent(zmaxInp.value || '');
  depthImg.src = `/preview_depth/${cameraSel.value}?zmin=${zmin}&zmax=${zmax}&_ts=${ts}`;
}
cameraSel.addEventListener('change', refreshPreviewStreams);
zminInp.addEventListener('change', refreshPreviewStreams);
zmaxInp.addEventListener('change', refreshPreviewStreams);
previewImg.onerror = () => { setTimeout(refreshPreviewStreams, 1000); };
depthImg.onerror = () => { setTimeout(refreshPreviewStreams, 1000); };
window.addEventListener('load', () => {
  refreshPreviewStreams();
  setTimeout(refreshPreviewStreams, 1500);
  setPreviewCacheBadge('none', 'no cached exec traj', 'No cached execution trajectory yet.');
});
if (graspModeSel) graspModeSel.addEventListener('change', onGraspModeChanged);
function onInput() { runBtn.disabled = running || promptInp.value.trim().length === 0; }
function onKey(e) { if (e.key === 'Enter' && !runBtn.disabled) runGrasps(); }
function toggleParams() { paramsPanel.classList.toggle('open'); }
function fmtList(arr, digits) { return '[' + arr.map(v => Number(v).toFixed(digits)).join(', ') + ']'; }
function setResultsVisible(visible) {
  overlayPanel.classList.remove('hidden');
  plannerPanel.classList.remove('hidden');
}
function resetResults(summary='Awaiting AnyGrasp run.') {
  resultsLabel.textContent = summary;
  overlayImg.src = EMPTY_IMAGE;
  currentGrasps = [];
  lastPreviewRequest = null;
  setPreviewCacheBadge('none', 'no cached exec traj', 'No cached execution trajectory yet.');
  renderRows([]);
}
function setPlannerStatus(msg, isError=false) {
  const el = document.getElementById('planner-status');
  el.style.color = isError ? '#ff7b7b' : '#aaa';
  el.innerHTML = msg;
}
function setPreviewCacheBadge(state='none', label='no cached exec traj', title='') {
  if (!previewCacheBadge) return;
  previewCacheBadge.className = `status-badge ${state || 'none'}`;
  previewCacheBadge.textContent = label || 'cache';
  previewCacheBadge.title = title || '';
}
function getSelectedGraspMode() {
  return graspModeSel ? String(graspModeSel.value || '2d') : '2d';
}
function cacheKeyForMode(grasp, mode) {
  if (!grasp || grasp.trajectory_cache_pose_mode !== mode) return null;
  return grasp.trajectory_cache_key || null;
}
function onGraspModeChanged() {
  lastPreviewRequest = null;
  setPreviewCacheBadge(
    'none',
    'no cached exec traj',
    `No cached execution trajectory for the current ${getSelectedGraspMode()} grasp mode.`
  );
  renderRows(currentGrasps);
  setPlannerStatus(
    `Grasp mode switched to <b>${getSelectedGraspMode()}</b>. Re-run Batch Preview + Cache so ranking and execution use that same pose family.`
  );
}
function markPreviewStaleUi(title='Cached execution trajectory no longer matches the current robot state. Preview or batch-preview again before executing.') {
  lastPreviewRequest = null;
  setPreviewCacheBadge('stale', 'exec traj stale', title);
}
function getExecuteFraction() {
  const raw = parseFloat(document.getElementById('execute-fraction').value);
  if (!Number.isFinite(raw)) return 1.0;
  return Math.min(1.0, Math.max(0.0, raw));
}
function samePoseRequest(a, side, xyz, rpy, poseMode) {
  if (!a) return false;
  if (a.trajectoryCacheKey) return false;
  if (a.side !== side || a.poseMode !== poseMode) return false;
  return JSON.stringify(a.xyz) === JSON.stringify(xyz) && JSON.stringify(a.rpy) === JSON.stringify(rpy);
}
function executeLastPreviewPrefix() {
  if (plannerBusy) return;
  if (!lastPreviewRequest) {
    setPlannerStatus('Cache a planner or 2D trajectory first, then Execute Cached Prefix will reuse that exact trajectory.', true);
    return;
  }
  if (!['planner', '2d'].includes(String(lastPreviewRequest.poseMode || ''))) {
    setPlannerStatus('Execute Cached Prefix only supports the last cached planner-facing or 2D trajectory.', true);
    return;
  }
  const t = getExecuteFraction();
  const rankLabel = lastPreviewRequest.rank == null ? 'last previewed trajectory' : `grasp #${lastPreviewRequest.rank}`;
  if (!window.confirm(`Execute the first ${(t * 100).toFixed(1)}% of ${rankLabel} on ${lastPreviewRequest.side}?`)) return;
  runPlanner(
    lastPreviewRequest.rank ?? '?',
    lastPreviewRequest.xyz,
    lastPreviewRequest.rpy,
    true,
    lastPreviewRequest.poseMode,
    t,
    lastPreviewRequest,
    true
  );
}
function renderRows(rows) {
  currentGrasps = Array.isArray(rows) ? rows : [];
  const activeMode = getSelectedGraspMode();
  const plannerModeActive = activeMode === 'planner';
  const twoDModeActive = activeMode === '2d';
  const body = document.getElementById('grasps-body');
  body.innerHTML = '';
  currentGrasps.forEach((g) => {
    const tr = document.createElement('tr');
    const hasMotionPlanError = g.motion_plan_error === true;
    const ikErr = hasMotionPlanError
      ? '∞ (collision)'
      : (g.ik_error_m == null ? '—' : Number(g.ik_error_m).toFixed(4));
    const ikRot = g.ik_rot_error_deg == null ? null : Number(g.ik_rot_error_deg).toFixed(2);
    if (g.within_ik_threshold === true) tr.classList.add('ik-pass');
    if (g.within_ik_threshold === false) tr.classList.add('ik-fail');
    const ikCellClass = g.within_ik_threshold === true
      ? 'ik-cell-pass'
      : (g.within_ik_threshold === false ? 'ik-cell-fail' : '');
    const ikTitle = hasMotionPlanError
      ? (g.motion_plan_reason || 'Motion plan error')
      : (ikRot == null
          ? 'IK error not computed yet'
          : `IK pos err ${ikErr} m · rot err ${ikRot} deg`);
    const has2dPose = Array.isArray(g.two_d_xyz) && g.two_d_xyz.length === 3 && Array.isArray(g.two_d_rpy) && g.two_d_rpy.length === 3;
    const fmtMaybeList = (arr, digits) => Array.isArray(arr) && arr.length === 3 ? fmtList(arr, digits) : '—';
    tr.innerHTML = `<td>${g.rank}</td><td>${Number(g.score).toFixed(4)}</td><td>${Number(g.width).toFixed(4)}</td><td class="mono">${fmtList(g.raw_xyz, 4)}</td><td class="mono">${fmtList(g.raw_rpy, 1)}</td><td class="mono">${fmtList(g.planner_xyz, 4)}</td><td class="mono">${fmtList(g.planner_rpy, 1)}</td><td class="mono">${fmtMaybeList(g.two_d_xyz, 4)}</td><td class="mono">${fmtMaybeList(g.two_d_rpy, 1)}</td><td class="mono ${ikCellClass}" title="${ikTitle}">${ikErr}</td>`;
    const actionTd = document.createElement('td');
    actionTd.className = 'planner-actions';
    const previewRawBtn = document.createElement('button');
    previewRawBtn.className = 'planner-btn';
    previewRawBtn.textContent = 'Preview Raw';
    previewRawBtn.onclick = () => runPlanner(g.rank, g.raw_xyz, g.raw_rpy, false, 'raw');
    const previewBtn = document.createElement('button');
    previewBtn.className = 'planner-btn';
    previewBtn.textContent = plannerModeActive ? 'Preview' : 'Preview Planner';
    previewBtn.disabled = !plannerModeActive;
    previewBtn.onclick = () => runPlanner(
      g.rank,
      g.planner_xyz,
      g.planner_rpy,
      false,
      'planner',
      1.0,
      cacheKeyForMode(g, 'planner') ? { trajectoryCacheKey: cacheKeyForMode(g, 'planner') } : null
    );
    const preview2dBtn = document.createElement('button');
    preview2dBtn.className = 'planner-btn';
    preview2dBtn.textContent = twoDModeActive ? 'Preview' : 'Preview 2D';
    preview2dBtn.disabled = !has2dPose || !twoDModeActive;
    preview2dBtn.onclick = () => runPlanner(g.rank, g.two_d_xyz, g.two_d_rpy, false, '2d');
    const execBtn = document.createElement('button');
    execBtn.className = 'planner-btn';
    execBtn.textContent = plannerModeActive ? 'Execute' : 'Execute Planner';
    execBtn.disabled = !plannerModeActive;
    execBtn.onclick = () => runPlanner(
      g.rank,
      g.planner_xyz,
      g.planner_rpy,
      true,
      'planner',
      1.0,
      cacheKeyForMode(g, 'planner') ? { trajectoryCacheKey: cacheKeyForMode(g, 'planner') } : null
    );
    const exec2dBtn = document.createElement('button');
    exec2dBtn.className = 'planner-btn';
    exec2dBtn.textContent = twoDModeActive ? 'Execute' : 'Execute 2D';
    exec2dBtn.disabled = !has2dPose || !twoDModeActive;
    exec2dBtn.onclick = () => runPlanner(
      g.rank,
      g.two_d_xyz,
      g.two_d_rpy,
      true,
      '2d',
      1.0,
      cacheKeyForMode(g, '2d') ? { trajectoryCacheKey: cacheKeyForMode(g, '2d') } : null
    );
    actionTd.appendChild(previewRawBtn);
    actionTd.appendChild(previewBtn);
    actionTd.appendChild(preview2dBtn);
    actionTd.appendChild(execBtn);
    actionTd.appendChild(exec2dBtn);
    tr.appendChild(actionTd);
    body.appendChild(tr);
  });
}
function sortGraspsByIkError() {
  if (plannerBusy) return;
  if (!Array.isArray(currentGrasps) || currentGrasps.length === 0) {
    setPlannerStatus('Run AnyGrasp first to compute and sort grasp IK errors.', true);
    return;
  }
  plannerBusy = true;
  const graspMode = getSelectedGraspMode();
  const side = document.getElementById('planner-side').value;
  const solverSpeed = document.getElementById('solver-speed').value || 'fast';
  const ikThresh = parseFloat(document.getElementById('planner-ik').value) || 0.005;
  const ikXyzWeight = parseFloat(document.getElementById('planner-xyz-weight').value) || 1.0;
  const ikRpyWeight = parseFloat(document.getElementById('planner-rpy-weight').value) || 0.3;
  const planningSpeed = parseFloat(document.getElementById('planner-speed').value) || 3.0;
  setPlannerStatus(`Batch-previewing ${currentGrasps.length} <b>${graspMode}</b> grasp(s) on <b>${side}</b> with <b>${solverSpeed}</b> solver and caching feasible trajectories ...`);
  fetch('/planner/sort_by_ik_error', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      side: side,
      pose_mode: graspMode,
      solver_speed: solverSpeed,
      planning_speed: planningSpeed,
      ik_error_threshold: ikThresh,
      ik_xyz_weight: ikXyzWeight,
      ik_rpy_weight: ikRpyWeight,
      grasps: currentGrasps,
    }),
  })
    .then(r => r.json())
    .then(data => {
      plannerBusy = false;
      if (!data.ok) {
        setPlannerStatus(`Sort with IK error failed: ${data.error || 'unknown error'}`, true);
        return;
      }
      const sortedRows = Array.isArray(data.grasps) ? data.grasps : [];
      renderRows(sortedRows);
      const passCount = sortedRows.filter(g => g.within_ik_threshold === true).length;
      const collisionCount = sortedRows.filter(g => g.motion_plan_error === true).length;
      const totalMs = Number(data.timing_total_ms || 0);
      const planMs = Number(data.timing_plan_eval_ms || 0);
      const postMs = Number(data.timing_postprocess_ms || 0);
      const rankMs = Number(data.timing_rank_ms || 0);
      const solveMs = Number(data.curobo_solve_time_ms || 0);
      const graphMs = Number(data.curobo_graph_time_ms || 0);
      const ikMs = Number(data.curobo_ik_time_ms || 0);
      const planningMode = String(data.planning_mode || 'unknown');
      const poseMode = String(data.pose_mode || graspMode);
      const inputCount = Number(data.input_grasp_count || sortedRows.length);
      const evaluatedCount = Number(data.evaluated_grasp_count || sortedRows.length);
      const truncatedCount = Number(data.truncated_input_count || 0);
      const cachedCount = sortedRows.filter(g => !!g.trajectory_cache_key).length;
      const truncMsg = truncatedCount > 0 ? ` Evaluated top ${evaluatedCount}/${inputCount} by score.` : '';
      const batchErr = data.batch_error ? ` Batch error: ${String(data.batch_error)}.` : '';
      setPlannerStatus(
        `Batch-previewed ${sortedRows.length} <b>${poseMode}</b> grasp(s) on <b>${data.side || side}</b>. ` +
        `<span class="ik-cell-pass">${passCount}</span> within threshold ${Number(data.ik_error_threshold ?? ikThresh).toFixed(4)} m; ` +
        `<span class="ik-cell-fail">${collisionCount}</span> motion plan error; ` +
        `<span class="ik-cell-fail">${Math.max(sortedRows.length - passCount - collisionCount, 0)}</span> above threshold. ` +
        `<b>${cachedCount}</b> cached executable ${poseMode} trajectory(s). ` +
        `Time: <b>${totalMs.toFixed(1)} ms</b> total = ${planMs.toFixed(1)} ms plan (${planningMode}) + ${postMs.toFixed(1)} ms map + ${rankMs.toFixed(1)} ms rank. ` +
        `cuRobo: solve ${solveMs.toFixed(1)} ms, graph ${graphMs.toFixed(1)} ms, ik ${ikMs.toFixed(1)} ms.` +
        truncMsg + batchErr
      );
    })
    .catch(err => {
      plannerBusy = false;
      setPlannerStatus(`Sort with IK error request failed: ${err}`, true);
    });
}
function runPlanner(rank, xyz, rpy, execute, poseMode, executeFraction=1.0, requestOverrides=null, skipConfirm=false) {
  let previewWin = null;
  if (!execute) {
    previewWin = window.open('about:blank', 'anygrasp_viser_preview');
    if (previewWin) {
      previewWin.document.open();
      previewWin.document.write('<!DOCTYPE html><html><body style="background:#111;color:#eee;font-family:sans-serif;padding:16px">Preparing 3D Viser preview…</body></html>');
      previewWin.document.close();
    }
  }
  if (execute && !skipConfirm && !window.confirm(`Execute freespace_move for grasp #${rank} on the current ${document.getElementById('planner-side').value} arm?`)) return;
  if (plannerBusy) return;
  plannerBusy = true;
  const selectedSide = document.getElementById('planner-side').value;
  const cachedMatch = execute && !requestOverrides && samePoseRequest(lastPreviewRequest, selectedSide, xyz, rpy, poseMode)
    ? lastPreviewRequest
    : null;
  const effectiveRequest = requestOverrides || cachedMatch;
  const side = effectiveRequest?.side || selectedSide;
  const solverSpeed = effectiveRequest?.solverSpeed || document.getElementById('solver-speed').value || 'fast';
  const trajectoryCacheKey = effectiveRequest?.trajectoryCacheKey || null;
  const planningSpeed = Number(
    effectiveRequest?.planningSpeed ?? (parseFloat(document.getElementById('planner-speed').value) || 3.0)
  );
  const ikThresh = Number(
    effectiveRequest?.ikThresh ?? (parseFloat(document.getElementById('planner-ik').value) || 0.005)
  );
  const ikXyzWeight = Number(
    effectiveRequest?.ikXyzWeight ?? (parseFloat(document.getElementById('planner-xyz-weight').value) || 1.0)
  );
  const ikRpyWeight = Number(
    effectiveRequest?.ikRpyWeight ?? (parseFloat(document.getElementById('planner-rpy-weight').value) || 0.3)
  );
  const safeExecuteFraction = Math.min(1.0, Math.max(0.0, Number(executeFraction)));
  const verb = execute ? 'Executing' : 'Previewing';
  const fracMsg = execute && safeExecuteFraction < 1.0 ? ` · t=${safeExecuteFraction.toFixed(3)}` : '';
  setPlannerStatus(`${verb} grasp #${rank} on <b>${side}</b> using <b>${poseMode}</b> pose with <b>${solverSpeed}</b> solver${fracMsg} ${fmtList(xyz, 4)} / ${fmtList(rpy, 1)} ...`);
  fetch('/planner/freespace', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      side: side,
      solver_speed: solverSpeed,
      xyz: xyz,
      rpy: rpy,
      pose_mode: poseMode,
      planning_speed: planningSpeed,
      ik_error_threshold: ikThresh,
      ik_xyz_weight: ikXyzWeight,
      ik_rpy_weight: ikRpyWeight,
      execute: execute,
      execute_fraction: safeExecuteFraction,
      trajectory_cache_key: trajectoryCacheKey,
    }),
  })
    .then(r => r.json())
    .then(data => {
      plannerBusy = false;
      if (!data.ok) {
        if (previewWin && !previewWin.closed) previewWin.close();
        if (String(data.preview_cache_state || '') === 'stale') {
          markPreviewStaleUi(
            data.preview_cache_message || 'Cached preview is stale because the robot state changed.'
          );
        }
        setPlannerStatus(`${execute ? 'Execute' : 'Preview'} failed: <b>${data.status}</b>${data.reason ? ' — ' + data.reason : ''}`, true);
        return;
      }
      if (!execute && ['planner', '2d'].includes(String(poseMode || ''))) {
        lastPreviewRequest = {
          rank: rank,
          side: side,
          xyz: Array.isArray(xyz) ? [...xyz] : [],
          rpy: Array.isArray(rpy) ? [...rpy] : [],
          poseMode: poseMode,
          solverSpeed: solverSpeed,
          planningSpeed: planningSpeed,
          ikThresh: ikThresh,
          ikXyzWeight: ikXyzWeight,
          ikRpyWeight: ikRpyWeight,
          trajectoryCacheKey: trajectoryCacheKey,
        };
      }
      if (data.executed) {
        markPreviewStaleUi(
          'The previewed trajectory was executed, so the robot state changed. Preview again before executing another full or partial trajectory.'
        );
      } else if (!execute && ['planner', '2d'].includes(String(poseMode || '')) && String(data.preview_cache_state || '') === 'cached') {
        setPreviewCacheBadge(
          data.used_cached_preview ? 'used' : 'cached',
          'cached exec traj ready',
          data.preview_cache_message || 'Cached execution trajectory ready; execute will reuse it with no replanning.'
        );
      } else if (String(data.preview_cache_state || '') === 'stale') {
        markPreviewStaleUi(
          data.preview_cache_message || 'Cached preview is stale because the robot state changed.'
        );
      }
      const actionWord = data.executed ? 'Executed' : 'Previewed';
      const cacheMsg = data.used_cached_preview ? ' · cached exec traj' : '';
      const fractionMsg = execute && Number(data.execute_fraction || 1.0) < 1.0
        ? ` · t=${Number(data.execute_fraction).toFixed(3)}`
        : '';
      setPlannerStatus(
        `${actionWord} grasp #${rank} (${data.pose_mode || poseMode}): <b>${data.status}</b>${cacheMsg}${fractionMsg} · steps ${data.trajectory_steps} · pos err ${Number(data.final_pos_error_m).toFixed(4)} m · rot err ${Number(data.final_rot_error_deg).toFixed(2)} deg${data.reason ? ' — ' + data.reason : ''}`
      );
      if (!execute && data.preview_url && previewWin) {
        previewWin.location.href = data.preview_url;
      } else if (!execute && previewWin && !previewWin.closed) {
        previewWin.document.open();
        previewWin.document.write('<!DOCTYPE html><html><body style="background:#111;color:#eee;font-family:sans-serif;padding:16px">Planner preview succeeded, but the 3D Viser page could not be started. Check the UI server log for details.</body></html>');
        previewWin.document.close();
      }
    })
    .catch(err => {
      plannerBusy = false;
      if (previewWin && !previewWin.closed) previewWin.close();
      setPlannerStatus(`${execute ? 'Execute' : 'Preview'} request failed: ${err}`, true);
    });
}

function goHome() {
  setPlannerStatus('Sending both arms home...');
  fetch('/robot/go_home', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (!data.ok) {
        setPlannerStatus(`Go Home failed: ${data.error || 'unknown error'}`, true);
        return;
      }
      markPreviewStaleUi('Both arms moved, so any cached execution trajectory is no longer valid.');
      setPlannerStatus('Both arms returned to home.');
    })
    .catch(err => {
      setPlannerStatus(`Go Home request failed: ${err}`, true);
    });
}
function robotPlannerSide() {
  const plannerSideEl = document.getElementById('planner-side');
  return plannerSideEl ? plannerSideEl.value : 'left';
}
function setGripper(action) {
  const side = robotPlannerSide();
  setPlannerStatus(`Sending ${action}(${side})...`);
  fetch(`/robot/${action}?side=${encodeURIComponent(side)}`, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (!data.ok) {
        setPlannerStatus(`${action} failed on ${side}: ${data.error || 'unknown error'}`, true);
        return;
      }
      setPlannerStatus(`${action}(${side}) completed.`);
    })
    .catch(err => {
      setPlannerStatus(`${action} request failed on ${side}: ${err}`, true);
    });
}
function openGripper() {
  setGripper('open_gripper');
}
function closeGripper() {
  setGripper('close_gripper');
}
function birdEyeView() {
  const planningSpeed = parseFloat(document.getElementById('planner-speed').value) || 1.5;
  setPlannerStatus(`Sending both arms to bird-eye view with planning speed ${planningSpeed.toFixed(2)}...`);
  fetch(`/robot/birdeyeview?planning_speed=${encodeURIComponent(planningSpeed)}`, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (!data.ok) {
        setPlannerStatus(`BirdEyeView failed: ${data.error || 'unknown error'}`, true);
        return;
      }
      markPreviewStaleUi('Both arms moved, so any cached execution trajectory is no longer valid.');
      setPlannerStatus(`Both arms moved to bird-eye view at planning speed ${planningSpeed.toFixed(2)}.`);
    })
    .catch(err => {
      setPlannerStatus(`BirdEyeView request failed: ${err}`, true);
    });
}
function localPose() {
  const planningSpeed = parseFloat(document.getElementById('planner-speed').value) || 1.5;
  setPlannerStatus(`Sending both arms to local pose with planning speed ${planningSpeed.toFixed(2)}...`);
  fetch(`/robot/local_pose?planning_speed=${encodeURIComponent(planningSpeed)}`, { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (!data.ok) {
        setPlannerStatus(`LocalPose failed: ${data.error || 'unknown error'}`, true);
        return;
      }
      markPreviewStaleUi('Both arms moved, so any cached execution trajectory is no longer valid.');
      setPlannerStatus(`Both arms moved to local pose at planning speed ${planningSpeed.toFixed(2)}.`);
    })
    .catch(err => {
      setPlannerStatus(`LocalPose request failed: ${err}`, true);
    });
}
function runGrasps() {
  if (running || !promptInp.value.trim()) return;
  running = true; runBtn.disabled = true; runBtn.textContent = 'Running…'; statusEl.textContent = 'Running AnyGrasp…';
  const body = {
    prompt: promptInp.value.trim(),
    camera: cameraSel.value,
    max_grasps: Math.max(1, parseInt(document.getElementById('maxg-inp').value) || 10),
    z_range_min: parseFloat(document.getElementById('zmin-inp').value) || 1e-6,
    z_range_max: parseFloat(document.getElementById('zmax-inp').value) || 1.5,
    workspace_margin: parseFloat(document.getElementById('margin-inp').value) || 0.02,
    collision_detection: !!document.getElementById('collision-detection-inp').checked,
    object_input_mode: document.getElementById('object-input-mode').value || 'segmented_object_cloud',
    tcp_offset_z_m: parseFloat(document.getElementById('tcp-offset-inp').value) || 0.0,
    disable_planner_z_clipping: !!document.getElementById('disable-zclip-inp').checked,
  };
  fetch('/run', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) })
    .then(r => r.json())
    .then(data => {
      running = false; runBtn.disabled = promptInp.value.trim().length === 0; runBtn.textContent = 'Run AnyGrasp';
      if (data.status === 'object_not_found') {
        resetResults('Object not found.');
        statusEl.textContent = data.error || 'SAM3 could not find the object.';
        return;
      }
      if (data.status === 'sam3_error' || data.status === 'anygrasp_error' || data.status === 'backend_unreachable') {
        resetResults('Backend request failed.');
        statusEl.textContent = data.error || 'Backend request failed.';
        return;
      }
      if (data.status === 'error') {
        resetResults('Unexpected error.');
        statusEl.textContent = 'Unexpected error: ' + (data.error || 'unknown');
        return;
      }
      setResultsVisible(true);
      const segmentedThickness = data.segmented_cloud_thickness_m;
      const segmentedZMin = data.segmented_cloud_z_min_m;
      const segmentedZMax = data.segmented_cloud_z_max_m;
      const thicknessSummary = segmentedThickness == null
        ? ' · segmented h unavailable'
        : ` · segmented h ${(Number(segmentedThickness) * 100.0).toFixed(2)} cm`;
      resultsLabel.textContent = data.status === 'no_grasps'
        ? `No grasps found.${thicknessSummary}`
        : `Found ${data.n_grasps} grasp(s)` + (data.best_score != null ? ` · best ${Number(data.best_score).toFixed(4)}` : '') + thicknessSummary + ` · ${Number(data.latency_ms).toFixed(0)} ms`;
      statusEl.textContent = data.status === 'no_grasps'
        ? `No grasps found for “${body.prompt}”`
        : `Found ${data.n_grasps} grasp(s) for “${body.prompt}”`;
      if (data.overlay_b64) {
        overlayImg.src = 'data:image/jpeg;base64,' + data.overlay_b64;
      } else {
        overlayImg.src = EMPTY_IMAGE;
      }
      if (data.cloud_preview_url && cloudFrame && cloudFrame.src !== data.cloud_preview_url) {
        cloudFrame.src = data.cloud_preview_url;
      }
      const plannerNote = document.getElementById('pose-note');
      if (plannerNote) {
        plannerNote.innerHTML =
          `<b>Raw pose</b> = native AnyGrasp world pose.<br>` +
          `<b>Planner pose</b> = planner-facing remap intended for <code>freespace_move</code>.<br>` +
          `<b>2D pose</b> = Jalen-style top-down hybrid using the AnyGrasp planner XY, SAM3-mask local yaw, and fixed Z = <code>${Number(data.two_d_top_down_z_m ?? 0.79).toFixed(3)} m</code>.<br>` +
          `Object input mode = <code>${String(data.object_input_mode || body.object_input_mode || 'segmented_object_cloud')}</code>.<br>` +
          `Segmented cloud z-range = ` +
          (segmentedZMin == null || segmentedZMax == null
            ? `<b>unavailable</b>`
            : `<code>[${Number(segmentedZMin).toFixed(4)}, ${Number(segmentedZMax).toFixed(4)}] m</code>`) +
          `.<br>` +
          `Segmented cloud thickness <code>h = max(z) - min(z)</code> = ` +
          (segmentedThickness == null
            ? `<b>unavailable</b>`
            : `<code>${(Number(segmentedThickness) * 100.0).toFixed(2)} cm</code>`) +
          `.<br>` +
          `Using TCP offset Z = <code>${Number(data.tcp_offset_z_m ?? body.tcp_offset_z_m).toFixed(4)} m</code>. ` +
          (data.planner_z_clipping_enabled
            ? `Planner-facing Z is clipped to <code>&gt;= ${Number(data.planner_z_floor_m).toFixed(3)} m</code> ` +
              `(${Number(data.n_planner_z_clipped || 0)} grasp(s) clipped).`
            : `Planner-facing Z clipping is <b>disabled</b>.`);
      }
      renderRows(data.grasps || []);
    })
    .catch(err => {
      running = false; runBtn.disabled = promptInp.value.trim().length === 0; runBtn.textContent = 'Run AnyGrasp';
      statusEl.textContent = 'Error: ' + err;
    });
}
</script>
</body>
</html>
"""

app = FastAPI(title="AnyGrasp Debug Server")


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(_DEBUG_HTML)


@app.post("/run", response_model=RunResponse)
async def run_grasps(req: RunRequest) -> RunResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_pipeline, req)


@app.post("/planner/freespace", response_model=PlannerResponse)
async def planner_freespace(req: PlannerRequest) -> PlannerResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_freespace, req)


@app.post("/planner/sort_by_ik_error", response_model=SortIkResponse)
async def planner_sort_by_ik_error(req: SortIkRequest) -> SortIkResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_sort_grasps_by_ik_error, req)


@app.post("/robot/go_home", response_model=HomeResponse)
async def robot_go_home() -> HomeResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_go_home)


@app.post("/robot/open_gripper", response_model=HomeResponse)
async def robot_open_gripper(side: str = "left") -> HomeResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_open_gripper, side)


@app.post("/robot/close_gripper", response_model=HomeResponse)
async def robot_close_gripper(side: str = "left") -> HomeResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_close_gripper, side)


@app.post("/robot/birdeyeview", response_model=HomeResponse)
async def robot_birdeyeview(planning_speed: float | None = None) -> HomeResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_birdeyeview, planning_speed)


@app.post("/robot/local_pose", response_model=HomeResponse)
async def robot_local_pose(planning_speed: float | None = None) -> HomeResponse:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_local_pose, planning_speed)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "anygrasp_url": ANYGRASP_URL}


@app.get("/preview/{camera}")
def preview(camera: str):
    def _gen():
        while True:
            try:
                client = _get_portal()
                rgb = np.asarray(client.get_camera_image(camera).result())
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpg = buf.tobytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            except Exception:
                time.sleep(0.1)
            time.sleep(0.033)

    return StreamingResponse(
        _gen(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/preview_depth/{camera}")
def preview_depth(
    camera: str,
    zmin: float | None = None,
    zmax: float | None = None,
):
    def _gen():
        while True:
            try:
                client = _get_portal()
                depth = np.asarray(client.get_camera_depth(camera).result()).astype(
                    np.float32
                )
                depth_bgr = _render_depth_preview_bgr(depth, zmin=zmin, zmax=zmax)
                _, buf = cv2.imencode(".jpg", depth_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpg = buf.tobytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            except Exception:
                time.sleep(0.1)
            time.sleep(0.033)

    return StreamingResponse(
        _gen(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


def main() -> None:
    global CAP_HOST, CAP_PORT, SAM3_URL, ANYGRASP_URL, DEFAULT_OBJECT_INPUT_MODE

    import argparse

    p = argparse.ArgumentParser(description="AnyGrasp debug UI server")
    p.add_argument(
        "--port", type=int, default=DEBUG_PORT, help="HTTP port (default: 8121)"
    )
    p.add_argument("--cap-host", default=CAP_HOST, help="cap_server hostname")
    p.add_argument(
        "--cap-port", type=int, default=CAP_PORT, help="cap_server Portal RPC port"
    )
    p.add_argument("--sam3-url", default=SAM3_URL, help="SAM3 server URL")
    p.add_argument("--anygrasp-url", default=ANYGRASP_URL, help="AnyGrasp server URL")
    p.add_argument(
        "--preload-motion-planner",
        action="store_true",
        help="Preload the default fast cuRobo motion planner on startup so the first planner action is warm.",
    )
    p.add_argument(
        "--default-object-input-mode",
        default=DEFAULT_OBJECT_INPUT_MODE,
        help="Default UI object input mode: segmented_object_cloud or roi_workspace",
    )
    args = p.parse_args()

    CAP_HOST = args.cap_host
    CAP_PORT = args.cap_port
    SAM3_URL = args.sam3_url.rstrip("/")
    ANYGRASP_URL = args.anygrasp_url.rstrip("/")
    DEFAULT_OBJECT_INPUT_MODE = str(args.default_object_input_mode).strip().lower()

    logger.info(f"AnyGrasp debug server → http://0.0.0.0:{args.port}")
    logger.info(f"  cap_server : {CAP_HOST}:{CAP_PORT}")
    logger.info(f"  sam3       : {SAM3_URL}")
    logger.info(f"  anygrasp   : {ANYGRASP_URL}")
    if args.preload_motion_planner:
        logger.info("[anygrasp-debug] scheduling motion planner preload in background")
        threading.Thread(
            target=_preload_motion_planner,
            args=("curobo",),
            kwargs={"solver_speed": "fast", "ik_xyz_weight": 1.0, "ik_rpy_weight": 0.3},
            daemon=True,
            name="anygrasp-preload-motion-planner",
        ).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
