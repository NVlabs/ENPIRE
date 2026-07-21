"""GPU pick + handover + insertion-prep hover on real YAM.

Steps:
  1. Right arm picks the GPU with a top-down 3D-BB grasp.
  2. Right arm lifts and moves the GPU to a center handover pose.
  3. Top camera re-detects the held GPU with 3D-BB.
  4. Left arm side-grasps a board face using the live OBB axes.
  5. Right arm releases and retreats.
  6. Left arm poses the GPU into an insertion-ready hover.

Run with:
  source .forge_env && uv run python run_script.py robot=real_yam \
      script_file=cap/saved_scripts/gpu/gpu_handover.py \
      env.name=yam-real skill_library_path=cap/saved_scripts/skill_library

By default the script ends with the left arm holding the GPU in an insertion-ready hover.
Set `GPU_HANDOVER_GO_HOME_ON_EXIT=1` to return both arms home afterward.

Working pipeline summary:
  - Steps 1-7 do right-pick, center handover, left regrasp, right release, and left reorientation.
  - Step 8 follows `gpu_debug_left_slot_hover.py`: initial SAM3 + depth motherboard reference, one hover acquire, then a reactive SAM3 refresh loop.
  - The motherboard plane z and board-frame geometry are fixed from the initial scene.
  - Step 8 fails closed if the initial hover cannot be acquired.
  - Reactive re-hover updates refresh motherboard xy only and move the left arm in the commanded hover plane.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
import re
import time
import types
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import enpire.env.forge.cap.agent.tools.segmentation as _segmentation_tools

from skill_library.constants.sorting import TABLE_SORT_RUN_CONFIG
from skill_library.namespace import (
    close_gripper,
    detect_object,
    end_detection,
    freespace_move,
    get_camera_extrinsics,
    get_camera_image,
    get_camera_intrinsics,
    get_robot_state,
    go_home,
    open_gripper,
    render_depth,
    sample_grasp_pose_3d_bb,
    segment_object,
    vlm_query,
)
from skill_library.pick_place import pick_object

OBJECT_NAME = "graphics card"
GPU_QUERIES = [
    "graphics card with fan",
    "small graphics card with fan",
    "black graphics card with fan",
    "small graphics card",
    "black gpu card",
    "graphics card",
    "video card",
    "pcie graphics card",
    "black graphics card",
    "pcie expansion card",
    "small gpu",
]
GPU_PICK_RIGHT_ROI_QUERIES = [
    q.strip()
    for q in os.environ.get(
        "GPU_PICK_RIGHT_ROI_QUERIES",
        (
            "black gpu card,"
            "black rectangular gpu,"
            "black rectangular graphics card,"
            "black computer expansion card,"
            "black rectangular object,"
            "black object"
        ),
    ).split(",")
    if q.strip()
]
MOTHERBOARD_QUERIES = [
    "large mother board",
    "motherboard",
    "mother board",
    "mainboard",
]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default if raw is None or raw == "" else raw)


def _env_optional_float(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_rpy(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    parts = [segment.strip() for segment in str(raw).split(",")]
    if len(parts) != 3 or any(part == "" for part in parts):
        raise ValueError(
            "expected three comma-separated values for RPY override, "
            f"got {raw!r}"
        )
    return [float(part) for part in parts]


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return tuple(float(value) for value in default)
    values = [float(part) for part in raw.replace(",", " ").split()]
    if len(values) != len(default):
        raise ValueError(
            f"{name} must contain {len(default)} numeric values, got {raw!r}"
        )
    return tuple(values)


if not _env_flag("GPU_SOCKET_HOVER_SAVE_SEGMENT_LOGS", False):
    _segmentation_tools.log_mask = lambda *args, **kwargs: None


HANDOVER_POS = [
    _env_float("GPU_HANDOVER_X", 0.50),
    _env_float("GPU_HANDOVER_Y", -0.02),
    _env_float("GPU_HANDOVER_Z", 0.92),
]
RECEIVER_HANDOVER_Y_BIAS_M = _env_float("GPU_RECEIVER_HANDOVER_Y_BIAS_M", 0.03)
HANDOVER_REGRASP_Z_M = _env_float("GPU_HANDOVER_REGRASP_Z_M", 0.96)
RIGHT_HANDOVER_REGRASP_Y_M = _env_float("GPU_RIGHT_HANDOVER_REGRASP_Y_M", -0.06)
RIGHT_HANDOVER_STAGE_X_BIAS_M = _env_float("GPU_RIGHT_HANDOVER_STAGE_X_BIAS_M", 0.02)
RIGHT_HANDOVER_RECEIVER_CLEARANCE_M = _env_float(
    "GPU_RIGHT_HANDOVER_RECEIVER_CLEARANCE_M",
    min(0.02, max(0.0, RECEIVER_HANDOVER_Y_BIAS_M)),
)
RIGHT_HANDOVER_ROLL_SEARCH_OFFSETS_DEG = [0.0, -8.0, 8.0, -12.0, 12.0]
LIFT_Z_M = _env_float("GPU_LIFT_Z_M", HANDOVER_POS[2])
INITIAL_PICK_MAX_ATTEMPTS = max(
    1,
    int(_env_float("GPU_INITIAL_PICK_MAX_ATTEMPTS", 5)),
)
RIGHT_PICK_LOWER_EDGE_BIAS_M = _env_float("GPU_RIGHT_PICK_LOWER_EDGE_BIAS_M", 0.015)
RIGHT_PICK_MAIN_BODY_BIAS_M = _env_float(
    "GPU_RIGHT_PICK_MAIN_BODY_BIAS_M",
    max(0.025, float(RIGHT_PICK_LOWER_EDGE_BIAS_M)),
)
RIGHT_PICK_MAIN_BODY_IMAGE_SIDE = (
    os.environ.get("GPU_RIGHT_PICK_MAIN_BODY_IMAGE_SIDE", "lower").strip().lower()
)
RIGHT_PICK_EXTRA_Z_OFFSET_M = _env_float("GPU_RIGHT_PICK_EXTRA_Z_OFFSET_M", 0.003)
RIGHT_PICK_ABOVE_TOP_SURFACE_M = _env_float(
    "GPU_RIGHT_PICK_ABOVE_TOP_SURFACE_M",
    0.0,
)
RIGHT_PICK_USE_FIXED_Z = _env_flag("GPU_RIGHT_PICK_USE_FIXED_Z", True)
RIGHT_PICK_FIXED_Z_M = _env_optional_float("GPU_RIGHT_PICK_FIXED_Z_M", 0.796)
RIGHT_PICK_APPROACH_CLEARANCE_M = _env_float("GPU_RIGHT_PICK_APPROACH_CLEARANCE_M", 0.07)
RIGHT_PICK_APPROACH_DESCEND_DURATION_S = _env_float(
    "GPU_RIGHT_PICK_APPROACH_DESCEND_DURATION_S",
    0.75,
)
RIGHT_PICK_APPROACH_DESCEND_STEPS = max(
    2,
    int(_env_float("GPU_RIGHT_PICK_APPROACH_DESCEND_STEPS", 8)),
)
RIGHT_PICK_CANONICALIZE_YAW = _env_flag("GPU_RIGHT_PICK_CANONICALIZE_YAW", True)
RIGHT_PICK_MAX_ABS_YAW_DEG = _env_float("GPU_RIGHT_PICK_MAX_ABS_YAW_DEG", 90.0)
GRASP_Z_OFFSET_M = _env_float("GPU_GRASP_Z_OFFSET_M", -0.0065)
LEFT_REGRASP_STANDOFF_M = _env_float("GPU_LEFT_REGRASP_STANDOFF_M", 0.10)
LEFT_REGRASP_INSET_M = _env_float("GPU_LEFT_REGRASP_INSET_M", 0.004)
LEFT_REGRASP_DEEP_INSET_M = _env_float("GPU_LEFT_REGRASP_DEEP_INSET_M", 0.012)
LEFT_REGRASP_MAX_INSET_M = _env_float("GPU_LEFT_REGRASP_MAX_INSET_M", 0.018)
LEFT_REGRASP_EXTRA_INSET_M = _env_float("GPU_LEFT_REGRASP_EXTRA_INSET_M", 0.015)
LEFT_REGRASP_FORWARD_X_BIAS_M = _env_float(
    "GPU_LEFT_REGRASP_FORWARD_X_BIAS_M",
    0.025,
)
LEFT_REGRASP_Z_BIAS_M = _env_float("GPU_LEFT_REGRASP_Z_BIAS_M", -0.007)
LEFT_REGRASP_RECEIVER_EDGE_BIAS_M = _env_float(
    "GPU_LEFT_REGRASP_RECEIVER_EDGE_BIAS_M",
    min(max(0.0, RECEIVER_HANDOVER_Y_BIAS_M), max(0.0, LEFT_REGRASP_EXTRA_INSET_M)),
)
LEFT_REGRASP_FACE_MARGIN_M = _env_float("GPU_LEFT_REGRASP_FACE_MARGIN_M", 0.004)
LEFT_REGRASP_RIGHT_EE_MIN_OUTWARD_CLEARANCE_M = _env_float(
    "GPU_LEFT_REGRASP_RIGHT_EE_MIN_OUTWARD_CLEARANCE_M",
    0.04,
)
LEFT_REGRASP_RIGHT_EE_MIN_XY_CLEARANCE_M = _env_float(
    "GPU_LEFT_REGRASP_RIGHT_EE_MIN_XY_CLEARANCE_M",
    0.04,
)
LEFT_REGRASP_RIGHT_EE_MIN_FORWARD_X_CLEARANCE_M = _env_float(
    "GPU_LEFT_REGRASP_RIGHT_EE_MIN_FORWARD_X_CLEARANCE_M",
    0.005,
)
LEFT_REGRASP_REQUIRE_RIGHT_CLEARANCE = _env_flag(
    "GPU_LEFT_REGRASP_REQUIRE_RIGHT_CLEARANCE",
    True,
)
LEFT_REGRASP_GUIDED_APPROACH = _env_flag("GPU_LEFT_REGRASP_GUIDED_APPROACH", True)
LEFT_REGRASP_GUIDED_APPROACH_DURATION_S = _env_float(
    "GPU_LEFT_REGRASP_GUIDED_APPROACH_DURATION_S",
    0.65,
)
LEFT_REGRASP_GUIDED_APPROACH_STEPS = max(
    2,
    int(_env_float("GPU_LEFT_REGRASP_GUIDED_APPROACH_STEPS", 8)),
)
LEFT_REGRASP_PREGRASP_TRANSIT_LIFT_M = _env_float(
    "GPU_LEFT_REGRASP_PREGRASP_TRANSIT_LIFT_M",
    0.05,
)
LEFT_REGRASP_PREGRASP_LATE_DESCENT_M = _env_float(
    "GPU_LEFT_REGRASP_PREGRASP_LATE_DESCENT_M",
    0.045,
)
LEFT_REGRASP_FINAL_APPROACH_MIN_M = _env_float(
    "GPU_LEFT_REGRASP_FINAL_APPROACH_MIN_M",
    0.035,
)
LEFT_REGRASP_MIN_GRIPPER_POS = _env_float("GPU_LEFT_REGRASP_MIN_GRIPPER_POS", 0.003)
LEFT_REGRASP_STABLE_MIN_GRIPPER_POS = _env_float(
    "GPU_LEFT_REGRASP_STABLE_MIN_GRIPPER_POS",
    0.006,
)
LEFT_REGRASP_RELEASE_MAX_GRIPPER_POS = _env_float(
    "GPU_LEFT_REGRASP_RELEASE_MAX_GRIPPER_POS",
    0.65,
)
LEFT_REGRASP_RELEASE_MAX_GRIPPER_DELTA = _env_float(
    "GPU_LEFT_REGRASP_RELEASE_MAX_GRIPPER_DELTA",
    0.035,
)
LEFT_REGRASP_SETTLE_S = _env_float("GPU_LEFT_REGRASP_SETTLE_S", 0.45)
LEFT_REGRASP_SETTLE_POLLS = max(
    2,
    int(_env_float("GPU_LEFT_REGRASP_SETTLE_POLLS", 4)),
)
LEFT_PRE_RELEASE_DWELL_S = _env_float("GPU_LEFT_PRE_RELEASE_DWELL_S", 0.35)
LEFT_PRE_RELEASE_DWELL_POLLS = max(
    2,
    int(_env_float("GPU_LEFT_PRE_RELEASE_DWELL_POLLS", 3)),
)
LEFT_REGRASP_CONFIRM_ATTEMPTS = max(
    1,
    int(_env_float("GPU_LEFT_REGRASP_CONFIRM_ATTEMPTS", 2)),
)
GPU_HALF_THICKNESS_M = _env_float("GPU_HALF_THICKNESS_M", 0.014)
GPU_SHORT_EDGE_M = _env_float("GPU_SHORT_EDGE_M", 0.10)
GPU_LONG_EDGE_M = _env_float("GPU_LONG_EDGE_M", 0.20)
LIVE_BBOX_MAX_CENTER_XY_DIST_M = _env_float(
    "GPU_LIVE_BBOX_MAX_CENTER_XY_DIST_M",
    0.12,
)
LIVE_BBOX_MAX_TOP_NORMAL_EXTENT_M = _env_float(
    "GPU_LIVE_BBOX_MAX_TOP_NORMAL_EXTENT_M",
    0.06,
)
LIVE_BBOX_MAX_FACE_EXTENT_M = _env_float(
    "GPU_LIVE_BBOX_MAX_FACE_EXTENT_M",
    0.16,
)
PICK_BBOX_MAX_TOP_NORMAL_EXTENT_M = _env_float(
    "GPU_PICK_BBOX_MAX_TOP_NORMAL_EXTENT_M",
    LIVE_BBOX_MAX_TOP_NORMAL_EXTENT_M,
)
PICK_ROI_BBOX_MAX_TOP_NORMAL_EXTENT_M = _env_float(
    "GPU_PICK_ROI_BBOX_MAX_TOP_NORMAL_EXTENT_M",
    max(PICK_BBOX_MAX_TOP_NORMAL_EXTENT_M, 0.09),
)
PICK_BBOX_MAX_SHORT_EXTENT_M = _env_float(
    "GPU_PICK_BBOX_MAX_SHORT_EXTENT_M",
    max(LIVE_BBOX_MAX_FACE_EXTENT_M, 0.18),
)
PICK_BBOX_MAX_LONG_EXTENT_M = _env_float(
    "GPU_PICK_BBOX_MAX_LONG_EXTENT_M",
    0.34,
)
PICK_MAX_ADJUSTED_Z_ABOVE_MOTHERBOARD_TOP_M = _env_float(
    "GPU_PICK_MAX_ADJUSTED_Z_ABOVE_MOTHERBOARD_TOP_M",
    0.058,
)
PICK_CLAMP_HIGH_Z_TO_TABLE = _env_flag("GPU_PICK_CLAMP_HIGH_Z_TO_TABLE", True)
MOTHERBOARD_HOVER_CLEARANCE_M = _env_float(
    "GPU_MOTHERBOARD_HOVER_CLEARANCE_M",
    0.10,
)
MOTHERBOARD_HOVER_MIN_Z = _env_float("GPU_MOTHERBOARD_HOVER_MIN_Z", 0.87)
SOCKET_HOVER_CLEARANCE_M = _env_float(
    "GPU_SOCKET_HOVER_CLEARANCE_M",
    max(0.0, MOTHERBOARD_HOVER_CLEARANCE_M),
)
SOCKET_HOVER_EFFECTIVE_CLEARANCE_M = _env_float(
    "GPU_SOCKET_HOVER_EFFECTIVE_CLEARANCE_M",
    max(0.0, float(SOCKET_HOVER_CLEARANCE_M) + 0.01),
)
SOCKET_HOVER_EFFECTIVE_MIN_Z = _env_float(
    "GPU_SOCKET_HOVER_EFFECTIVE_MIN_Z",
    0.0,
)
SOCKET_HOVER_MIN_MOVE_M = _env_float(
    "GPU_SOCKET_HOVER_MIN_MOVE_M",
    0.004,
)
SOCKET_HOVER_Z_TOL_M = _env_float(
    "GPU_SOCKET_HOVER_Z_TOL_M",
    0.005,
)
SOCKET_HOVER_X_OFFSET_M = _env_float(
    "GPU_SOCKET_HOVER_X_OFFSET_M",
    -0.04,
)
SOCKET_HOVER_Y_TRIM_M = _env_float(
    "GPU_SOCKET_HOVER_Y_TRIM_M",
    0.01,
)
SOCKET_HOVER_BOARD_PLANE_Z_OFFSET_M = _env_float(
    "GPU_SOCKET_HOVER_BOARD_PLANE_Z_OFFSET_M",
    0.01,
)
SOCKET_TARGET_RIGHT_EDGE_OFFSETS_M = _env_float_tuple(
    "GPU_SOCKET_TARGET_RIGHT_EDGE_OFFSETS_M",
    (
        0.11,
        0.06,
        0.01,
    ),
)
TARGET_SOCKET_NUMBER = max(
    1,
    min(3, int(_env_float("GPU_TARGET_SOCKET_NUMBER", 1))),
)
SOCKET_HOVER_AUX_CAMERA = (
    os.environ.get("GPU_SOCKET_HOVER_AUX_CAMERA", "left_third").strip().lower() or None
)
SOCKET_HOVER_TRACK_CAMERA = (
    os.environ.get(
        "GPU_SOCKET_HOVER_TRACK_CAMERA",
        os.environ.get("GPU_SOCKET_HOVER_CAMERA", "top"),
    )
    .strip()
    .lower()
    or "top"
)
if SOCKET_HOVER_AUX_CAMERA == SOCKET_HOVER_TRACK_CAMERA:
    SOCKET_HOVER_AUX_CAMERA = None
SOCKET_HOVER_AUX_CAMERA_PREFER_WORLD_POSE = _env_flag(
    "GPU_SOCKET_HOVER_AUX_CAMERA_PREFER_WORLD_POSE",
    True,
)
SOCKET_HOVER_AUX_CAMERA_REQUIRED = _env_flag(
    "GPU_SOCKET_HOVER_AUX_CAMERA_REQUIRED",
    bool(SOCKET_HOVER_AUX_CAMERA and SOCKET_HOVER_AUX_CAMERA_PREFER_WORLD_POSE),
)
SOCKET_HOVER_PRIMARY_MIN_SCORE = _env_float(
    "GPU_SOCKET_HOVER_PRIMARY_MIN_SCORE",
    0.55,
)
SOCKET_HOVER_AUX_MIN_SCORE = _env_float(
    "GPU_SOCKET_HOVER_AUX_MIN_SCORE",
    0.55,
)
SOCKET_HOVER_AUX_BLEND_MAX_SHIFT_M = _env_float(
    "GPU_SOCKET_HOVER_AUX_BLEND_MAX_SHIFT_M",
    0.03,
)
SOCKET_HOVER_AUX_MAX_POSE_DIFF_M = _env_float(
    "GPU_SOCKET_HOVER_AUX_MAX_POSE_DIFF_M",
    0.12,
)
EXPLICIT_SOCKET_HOVER_RPY = _parse_optional_rpy(os.environ.get("GPU_SOCKET_HOVER_RPY"))
MOTHERBOARD_SLOT_BBOX_MARGIN_PX = max(
    8,
    int(_env_float("GPU_MOTHERBOARD_SLOT_BBOX_MARGIN_PX", 20)),
)
SLOT_CENTER_DEPTH_WINDOW_RADIUS_PX = max(
    1,
    int(_env_float("GPU_SLOT_CENTER_DEPTH_WINDOW_RADIUS_PX", 4)),
)
ENABLE_SOCKET_HOVER = os.environ.get("GPU_ENABLE_SOCKET_HOVER", "1") == "1"
USE_INITIAL_SCENE_SOCKET_TARGET = (
    os.environ.get("GPU_USE_INITIAL_SCENE_SOCKET_TARGET", "1") == "1"
)
REACTIVE_SOCKET_HOVER = (
    os.environ.get("GPU_REACTIVE_SOCKET_HOVER", "1") == "1"
)
REACTIVE_SOCKET_HOVER_PERIOD_S = _env_float(
    "GPU_REACTIVE_SOCKET_HOVER_PERIOD_S",
    0.0,
)
SOCKET_HOVER_REACTIVE_GUIDED_DURATION_S = _env_float(
    "GPU_SOCKET_HOVER_REACTIVE_GUIDED_DURATION_S",
    0.18,
)
SOCKET_HOVER_REACTIVE_GUIDED_STEPS = max(
    2,
    int(_env_float("GPU_SOCKET_HOVER_REACTIVE_GUIDED_STEPS", 6)),
)
SOCKET_HOVER_GUIDED_MAX_XY_SPEED_MPS = _env_float(
    "GPU_SOCKET_HOVER_GUIDED_MAX_XY_SPEED_MPS",
    0.20,
)
SOCKET_HOVER_GUIDED_MAX_Z_SPEED_MPS = _env_float(
    "GPU_SOCKET_HOVER_GUIDED_MAX_Z_SPEED_MPS",
    0.10,
)
SOCKET_HOVER_INITIAL_GUIDED_DURATION_S = _env_float(
    "GPU_SOCKET_HOVER_INITIAL_GUIDED_DURATION_S",
    1.2,
)
SOCKET_HOVER_INITIAL_GUIDED_STEPS = max(
    2,
    int(_env_float("GPU_SOCKET_HOVER_INITIAL_GUIDED_STEPS", 12)),
)
SOCKET_HOVER_INITIAL_MAX_ATTEMPTS = max(
    3,
    int(_env_float("GPU_SOCKET_HOVER_INITIAL_MAX_ATTEMPTS", 5)),
)
USE_MOTHERBOARD_TRACKING = (
    os.environ.get("GPU_USE_MOTHERBOARD_TRACKING", "0") == "1"
)
SOCKET_HOVER_SAVE_REFRESH_ARTIFACTS = _env_flag(
    "GPU_SOCKET_HOVER_SAVE_REFRESH_ARTIFACTS",
    False,
)
MOTHERBOARD_TRACK_QUERY = (
    os.environ.get("GPU_MOTHERBOARD_TRACK_QUERY", "").strip()
)
MOTHERBOARD_TRACK_SESSION_NAME = (
    os.environ.get(
        "GPU_MOTHERBOARD_TRACK_SESSION_NAME",
        "gpu_handover_motherboard",
    ).strip()
    or "gpu_handover_motherboard"
)
MOTHERBOARD_TRACK_MAX_RETRIES = max(
    1,
    int(_env_float("GPU_MOTHERBOARD_TRACK_MAX_RETRIES", 1)),
)
REFRESH_VIEW_CLEAR_X_M = _env_float("GPU_REFRESH_VIEW_CLEAR_X_M", 0.0)
REFRESH_VIEW_CLEAR_Y_M = _env_float("GPU_REFRESH_VIEW_CLEAR_Y_M", 0.02)
REFRESH_VIEW_CLEAR_Z_M = _env_float("GPU_REFRESH_VIEW_CLEAR_Z_M", 0.03)
REFRESH_MOTHERBOARD_Z_CLIP_MARGIN_M = _env_float(
    "GPU_REFRESH_MOTHERBOARD_Z_CLIP_MARGIN_M",
    0.04,
)
REFRESH_MOTHERBOARD_CENTER_SHIFT_LIMIT_M = _env_float(
    "GPU_REFRESH_MOTHERBOARD_CENTER_SHIFT_LIMIT_M",
    0.0,
)
REFRESH_MOTHERBOARD_CENTER_UPDATE_MIN_SHIFT_M = _env_float(
    "GPU_REFRESH_MOTHERBOARD_CENTER_UPDATE_MIN_SHIFT_M",
    0.004,
)
REFRESH_MOTHERBOARD_CENTER_UPDATE_MAX_SHIFT_M = _env_float(
    "GPU_REFRESH_MOTHERBOARD_CENTER_UPDATE_MAX_SHIFT_M",
    0.12,
)
REFRESH_MOTHERBOARD_EDGE_ANCHOR_SOURCE = (
    os.environ.get("GPU_REFRESH_MOTHERBOARD_EDGE_ANCHOR_SOURCE", "current")
    .strip()
    .lower()
)
REFRESH_MOTHERBOARD_REFERENCE_ANCHOR_BLEND_MAX_PX = _env_float(
    "GPU_REFRESH_MOTHERBOARD_REFERENCE_ANCHOR_BLEND_MAX_PX",
    18.0,
)
REFRESH_MOTHERBOARD_REFERENCE_ANCHOR_BLEND_GAIN = _env_float(
    "GPU_REFRESH_MOTHERBOARD_REFERENCE_ANCHOR_BLEND_GAIN",
    0.25,
)
_SOCKET_HOVER_AUX_XML_T_CAM_WORLD_CACHE = {}


def _socket_hover_env_value(*names, default):
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw != "":
            return str(raw)
    return str(default)


def _set_slot_debug_env_default(name, *source_names, default):
    if name in os.environ:
        return
    os.environ[name] = _socket_hover_env_value(*source_names, default=default)


def _configure_reset_aligned_slot_hover_env():
    """Configure gpu_debug_left_slot_hover.py exactly like gpu_reset.py by default."""
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_CAMERA",
        "GPU_SOCKET_HOVER_TRACK_CAMERA",
        "GPU_SOCKET_HOVER_CAMERA",
        default="top",
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_AUX_CAMERA",
        "GPU_SOCKET_HOVER_AUX_CAMERA",
        default="left",
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_AUX_CAMERA_PREFER_WORLD_POSE",
        "GPU_SOCKET_HOVER_AUX_CAMERA_PREFER_WORLD_POSE",
        default="1",
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_REQUIRE_AUX_CAMERA",
        "GPU_SOCKET_HOVER_AUX_CAMERA_REQUIRED",
        default="0",
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_BOARD_PLANE_Z_OFFSET_M",
        "GPU_SOCKET_HOVER_BOARD_PLANE_Z_OFFSET_M",
        default=SOCKET_HOVER_BOARD_PLANE_Z_OFFSET_M,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_HOVER_CLEARANCE_M",
        "GPU_SOCKET_HOVER_EFFECTIVE_CLEARANCE_M",
        "GPU_SOCKET_HOVER_CLEARANCE_M",
        default=SOCKET_HOVER_EFFECTIVE_CLEARANCE_M,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_HOVER_MIN_Z",
        "GPU_SOCKET_HOVER_EFFECTIVE_MIN_Z",
        default=SOCKET_HOVER_EFFECTIVE_MIN_Z,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_MIN_MOVE_M",
        "GPU_SOCKET_HOVER_MIN_MOVE_M",
        default=SOCKET_HOVER_MIN_MOVE_M,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_HOVER_Z_TOL_M",
        "GPU_SOCKET_HOVER_Z_TOL_M",
        default=SOCKET_HOVER_Z_TOL_M,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_PRIMARY_CAMERA_MIN_SCORE",
        "GPU_SOCKET_HOVER_PRIMARY_MIN_SCORE",
        default=SOCKET_HOVER_PRIMARY_MIN_SCORE,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_AUX_CAMERA_MIN_SCORE",
        "GPU_SOCKET_HOVER_AUX_MIN_SCORE",
        default=SOCKET_HOVER_AUX_MIN_SCORE,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_AUX_CAMERA_BLEND_MAX_SHIFT_M",
        "GPU_SOCKET_HOVER_AUX_BLEND_MAX_SHIFT_M",
        default=SOCKET_HOVER_AUX_BLEND_MAX_SHIFT_M,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_AUX_CAMERA_MAX_POSE_DIFF_M",
        "GPU_SOCKET_HOVER_AUX_MAX_POSE_DIFF_M",
        default=SOCKET_HOVER_AUX_MAX_POSE_DIFF_M,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_INITIAL_GUIDED_DURATION_S",
        "GPU_SOCKET_HOVER_INITIAL_GUIDED_DURATION_S",
        default=SOCKET_HOVER_INITIAL_GUIDED_DURATION_S,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_INITIAL_GUIDED_STEPS",
        "GPU_SOCKET_HOVER_INITIAL_GUIDED_STEPS",
        default=SOCKET_HOVER_INITIAL_GUIDED_STEPS,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_INITIAL_HOVER_MAX_ATTEMPTS",
        "GPU_SOCKET_HOVER_INITIAL_MAX_ATTEMPTS",
        default=SOCKET_HOVER_INITIAL_MAX_ATTEMPTS,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_REACTIVE_GUIDED_DURATION_S",
        "GPU_SOCKET_HOVER_REACTIVE_GUIDED_DURATION_S",
        default=SOCKET_HOVER_REACTIVE_GUIDED_DURATION_S,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_REACTIVE_GUIDED_STEPS",
        "GPU_SOCKET_HOVER_REACTIVE_GUIDED_STEPS",
        default=SOCKET_HOVER_REACTIVE_GUIDED_STEPS,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_GUIDED_MAX_XY_SPEED_MPS",
        "GPU_SOCKET_HOVER_GUIDED_MAX_XY_SPEED_MPS",
        default=SOCKET_HOVER_GUIDED_MAX_XY_SPEED_MPS,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_GUIDED_MAX_Z_SPEED_MPS",
        "GPU_SOCKET_HOVER_GUIDED_MAX_Z_SPEED_MPS",
        default=SOCKET_HOVER_GUIDED_MAX_Z_SPEED_MPS,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_CENTER_UPDATE_MIN_SHIFT_M",
        "GPU_REFRESH_MOTHERBOARD_CENTER_UPDATE_MIN_SHIFT_M",
        default=REFRESH_MOTHERBOARD_CENTER_UPDATE_MIN_SHIFT_M,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_CENTER_UPDATE_MAX_SHIFT_M",
        "GPU_REFRESH_MOTHERBOARD_CENTER_UPDATE_MAX_SHIFT_M",
        default=REFRESH_MOTHERBOARD_CENTER_UPDATE_MAX_SHIFT_M,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_REFERENCE_ANCHOR_BLEND_MAX_PX",
        "GPU_REFRESH_MOTHERBOARD_REFERENCE_ANCHOR_BLEND_MAX_PX",
        default=REFRESH_MOTHERBOARD_REFERENCE_ANCHOR_BLEND_MAX_PX,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_REFERENCE_ANCHOR_BLEND_GAIN",
        "GPU_REFRESH_MOTHERBOARD_REFERENCE_ANCHOR_BLEND_GAIN",
        default=REFRESH_MOTHERBOARD_REFERENCE_ANCHOR_BLEND_GAIN,
    )
    _set_slot_debug_env_default(
        "GPU_SLOT_DEBUG_SAVE_ARTIFACTS",
        "GPU_SOCKET_HOVER_SAVE_REFRESH_ARTIFACTS",
        default="1",
    )


def _load_reset_aligned_slot_hover_helpers():
    _configure_reset_aligned_slot_hover_env()
    script_path = Path.cwd() / "cap" / "saved_scripts" / "gpu" / "gpu_debug_left_slot_hover.py"
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    keep = []
    for idx, node in enumerate(tree.body):
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            keep.append(node)
        elif (
            idx == 0
            and isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str)
        ):
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    helpers = types.ModuleType("gpu_left_slot_hover_helpers")
    helpers.__file__ = str(script_path)
    helpers.__dict__.update({"__builtins__": __builtins__})
    for name in [
        "close_gripper",
        "freespace_move",
        "get_camera_extrinsics",
        "get_camera_image",
        "get_camera_intrinsics",
        "get_robot_state",
        "go_home",
        "open_gripper",
        "render_depth",
        "sample_grasp_pose_3d_bb",
        "segment_object",
    ]:
        if name in globals():
            helpers.__dict__[name] = globals()[name]
    exec(compile(module, str(script_path), "exec"), helpers.__dict__)  # noqa: S102
    return helpers


SOCKET_VLM_BACKEND = os.environ.get("GPU_SOCKET_VLM_BACKEND", "nvidia").strip() or "nvidia"
SOCKET_VLM_MODEL = (
    os.environ.get("GPU_SOCKET_VLM_MODEL", "gcp/google/gemini-2.5-flash").strip()
    or "gcp/google/gemini-2.5-flash"
)
PICK_VLM_ROI_FALLBACK = (
    os.environ.get("GPU_PICK_VLM_ROI_FALLBACK", "1") == "1"
)
PICK_VLM_BACKEND = (
    os.environ.get("GPU_PICK_VLM_BACKEND", SOCKET_VLM_BACKEND).strip()
    or SOCKET_VLM_BACKEND
)
PICK_VLM_MODEL = (
    os.environ.get("GPU_PICK_VLM_MODEL", SOCKET_VLM_MODEL).strip()
    or SOCKET_VLM_MODEL
)
PICK_VLM_BBOX_MARGIN_PX = max(
    8,
    int(_env_float("GPU_PICK_VLM_BBOX_MARGIN_PX", 24)),
)
PICK_MOTHERBOARD_RIGHT_ROI_FALLBACK = (
    os.environ.get("GPU_PICK_MOTHERBOARD_RIGHT_ROI_FALLBACK", "1") == "1"
)
PICK_RIGHT_ROI_X_GAP_PX = max(
    0,
    int(_env_float("GPU_PICK_RIGHT_ROI_X_GAP_PX", 28)),
)
PICK_RIGHT_ROI_MIN_WIDTH_PX = max(
    80,
    int(_env_float("GPU_PICK_RIGHT_ROI_MIN_WIDTH_PX", 240)),
)
PICK_RIGHT_ROI_WIDTH_SCALE = max(
    0.5,
    float(_env_float("GPU_PICK_RIGHT_ROI_WIDTH_SCALE", 0.90)),
)
PICK_RIGHT_ROI_TOP_PAD_PX = max(
    0,
    int(_env_float("GPU_PICK_RIGHT_ROI_TOP_PAD_PX", 110)),
)
PICK_RIGHT_ROI_BOTTOM_PAD_PX = max(
    0,
    int(_env_float("GPU_PICK_RIGHT_ROI_BOTTOM_PAD_PX", 65)),
)
PICK_RIGHT_ROI_DARK_THRESHOLD = max(
    1,
    min(255, int(_env_float("GPU_PICK_RIGHT_ROI_DARK_THRESHOLD", 85))),
)
PICK_RIGHT_ROI_MIN_COMPONENT_AREA_PX = max(
    20,
    int(_env_float("GPU_PICK_RIGHT_ROI_MIN_COMPONENT_AREA_PX", 120)),
)
PICK_RIGHT_ROI_TIGHT_PAD_PX = max(
    0,
    int(_env_float("GPU_PICK_RIGHT_ROI_TIGHT_PAD_PX", 14)),
)
PICK_RIGHT_ROI_TOP_BORDER_REJECT_PX = max(
    0,
    int(_env_float("GPU_PICK_RIGHT_ROI_TOP_BORDER_REJECT_PX", 6)),
)
PICK_RIGHT_ROI_MIN_BOTTOM_REL_MB_TOP_PX = int(
    _env_float("GPU_PICK_RIGHT_ROI_MIN_BOTTOM_REL_MB_TOP_PX", -10)
)
PCIE_SLOT_CENTER_MAX_Y_SPREAD_PX = max(
    8,
    int(_env_float("GPU_PCIE_SLOT_CENTER_MAX_Y_SPREAD_PX", 25)),
)
PCIE_SLOT_CENTER_MIN_X_SPREAD_PX = max(
    40,
    int(_env_float("GPU_PCIE_SLOT_CENTER_MIN_X_SPREAD_PX", 80)),
)
POST_HANDOVER_HOVER_MIN_LIFT_M = _env_float("GPU_POST_HANDOVER_HOVER_MIN_LIFT_M", 0.06)
POST_HANDOVER_REORIENT_DURATION_S = _env_float("GPU_POST_HANDOVER_REORIENT_DURATION_S", 1.7)
POST_HANDOVER_REORIENT_STEPS = max(
    4,
    int(_env_float("GPU_POST_HANDOVER_REORIENT_STEPS", 10)),
)
POST_HANDOVER_COMBINE_REORIENT_WITH_HOVER = _env_flag(
    "GPU_POST_HANDOVER_COMBINE_REORIENT_WITH_HOVER",
    True,
)
MOVE_PLANNING_SPEED = _env_float("GPU_MOVE_PLANNING_SPEED", 1.0)
POST_PICK_PLANNING_SPEED = _env_float(
    "GPU_POST_PICK_PLANNING_SPEED",
    min(MOVE_PLANNING_SPEED, 0.85),
)
HANDOVER_BATCH_PLANNING_SPEED = _env_float(
    "GPU_HANDOVER_BATCH_PLANNING_SPEED",
    POST_PICK_PLANNING_SPEED,
)
RIGHT_HOLD_MIN_GRIPPER_POS = _env_float("GPU_RIGHT_HOLD_MIN_GRIPPER_POS", 0.1)
RIGHT_HOLD_MIN_RATIO = _env_float("GPU_RIGHT_HOLD_MIN_RATIO", 0.35)
RIGHT_IN_HAND_MAX_XY_DIST_M = _env_float("GPU_RIGHT_IN_HAND_MAX_XY_DIST_M", 0.07)
RIGHT_IN_HAND_MAX_BELOW_EE_Z_M = _env_float(
    "GPU_RIGHT_IN_HAND_MAX_BELOW_EE_Z_M",
    0.05,
)
PRE_HANDOVER_REPICK_LIMIT = int(_env_float("GPU_PRE_HANDOVER_REPICK_LIMIT", 2))
TABLE_SURFACE_Z_M = _env_float("GPU_TABLE_SURFACE_Z_M", 0.75)
GPU_TABLE_LIKE_Z_MAX_M = _env_float("GPU_TABLE_LIKE_Z_MAX_M", 0.83)
RIGHT_DROP_NEAR_XY_DIST_M = _env_float("GPU_RIGHT_DROP_NEAR_XY_DIST_M", 0.10)
RIGHT_DROP_MIN_BELOW_EE_Z_M = _env_float("GPU_RIGHT_DROP_MIN_BELOW_EE_Z_M", 0.12)
RIGHT_RETRY_CLEAR_POS = [
    _env_float("GPU_RIGHT_RETRY_CLEAR_X", 0.60),
    _env_float("GPU_RIGHT_RETRY_CLEAR_Y", -0.18),
    _env_float("GPU_RIGHT_RETRY_CLEAR_Z", 0.92),
]
PICK_GRIPPER_OPEN_VEL_LIMIT = _env_float("GPU_PICK_GRIPPER_OPEN_VEL_LIMIT", 6.0)
PICK_GRIPPER_CLOSE_VEL_LIMIT = _env_float("GPU_PICK_GRIPPER_CLOSE_VEL_LIMIT", 2.0)
PICK_GRIPPER_CLOSE_TORQUE_LIMIT = _env_float(
    "GPU_PICK_GRIPPER_CLOSE_TORQUE_LIMIT",
    0.4,
)
LEFT_REGRASP_OPEN_VEL_LIMIT = _env_float("GPU_LEFT_REGRASP_OPEN_VEL_LIMIT", 6.0)
LEFT_REGRASP_CLOSE_VEL_LIMIT = _env_float("GPU_LEFT_REGRASP_CLOSE_VEL_LIMIT", 1.6)
LEFT_REGRASP_CLOSE_TORQUE_LIMIT = _env_float(
    "GPU_LEFT_REGRASP_CLOSE_TORQUE_LIMIT",
    0.35,
)
LEFT_POST_RELEASE_RECLAMP = os.environ.get("GPU_LEFT_POST_RELEASE_RECLAMP", "1") == "1"
LEFT_POST_RELEASE_RECLAMP_VEL_LIMIT = _env_float(
    "GPU_LEFT_POST_RELEASE_RECLAMP_VEL_LIMIT",
    1.0,
)
LEFT_POST_RELEASE_RECLAMP_TORQUE_LIMIT = _env_float(
    "GPU_LEFT_POST_RELEASE_RECLAMP_TORQUE_LIMIT",
    0.30,
)
RIGHT_RELEASE_OPEN_VEL_LIMIT = _env_float("GPU_RIGHT_RELEASE_OPEN_VEL_LIMIT", 2.5)
RIGHT_POST_RELEASE_LIFT_M = _env_float("GPU_RIGHT_POST_RELEASE_LIFT_M", 0.13)
RIGHT_POST_RELEASE_CARTESIAN_STEP_M = _env_float(
    "GPU_RIGHT_POST_RELEASE_CARTESIAN_STEP_M",
    0.015,
)
RIGHT_POST_RELEASE_CARTESIAN_SPEED_MPS = _env_float(
    "GPU_RIGHT_POST_RELEASE_CARTESIAN_SPEED_MPS",
    0.14,
)
RIGHT_POST_RELEASE_CLEAR_X_M = _env_float("GPU_RIGHT_POST_RELEASE_CLEAR_X_M", -0.05)
RIGHT_POST_RELEASE_CLEAR_Y_M = _env_float("GPU_RIGHT_POST_RELEASE_CLEAR_Y_M", -0.18)
RIGHT_POST_RELEASE_EXTRACT_Z_MARGIN_M = _env_float(
    "GPU_RIGHT_POST_RELEASE_EXTRACT_Z_MARGIN_M",
    0.03,
)
RIGHT_POST_RELEASE_CLOSE_VEL_LIMIT = _env_float(
    "GPU_RIGHT_POST_RELEASE_CLOSE_VEL_LIMIT",
    2.0,
)
RIGHT_POST_RELEASE_CLOSE_TORQUE_LIMIT = _env_optional_float(
    "GPU_RIGHT_POST_RELEASE_CLOSE_TORQUE_LIMIT",
    0.8,
)
RECOVERY_OPEN_VEL_LIMIT = _env_float("GPU_RECOVERY_OPEN_VEL_LIMIT", 6.0)
RIGHT_POST_RELEASE_CLOSE = os.environ.get("GPU_RIGHT_POST_RELEASE_CLOSE", "1") == "1"
GO_HOME_ON_START = os.environ.get("GPU_HANDOVER_GO_HOME_ON_START", "1") == "1"
GO_HOME_ON_EXIT = os.environ.get("GPU_HANDOVER_GO_HOME_ON_EXIT", "0") == "1"

RUN_CONFIG = dict(TABLE_SORT_RUN_CONFIG)
RUN_CONFIG["tcp_offset_z_m"] = GRASP_Z_OFFSET_M
RUN_CONFIG["gripper_open_vel_limit"] = PICK_GRIPPER_OPEN_VEL_LIMIT
RUN_CONFIG["gripper_close_vel_limit"] = PICK_GRIPPER_CLOSE_VEL_LIMIT
RUN_CONFIG["gripper_close_torque_limit"] = PICK_GRIPPER_CLOSE_TORQUE_LIMIT

WORLD_LEFT = np.array([0.0, 1.0, 0.0], dtype=float)
WORLD_DOWN = np.array([0.0, 0.0, -1.0], dtype=float)


def _normalize_display_rpy(rpy_deg):
    arr = (np.asarray(rpy_deg, dtype=float) + 180.0) % 360.0 - 180.0
    return [float(round(v, 4)) for v in arr]


def _ordered_unique_positive(values):
    seen = set()
    ordered = []
    for value in values:
        val = round(float(value), 4)
        if val <= 0.0 or val in seen:
            continue
        seen.add(val)
        ordered.append(val)
    return ordered


def _unit(vec):
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        raise RuntimeError(f"zero-length vector: {vec}")
    return arr / norm


def _rotation_to_display_rpy(rot: Rotation):
    ex, ey, ez = rot.as_euler("xyz", degrees=True)
    return _normalize_display_rpy([ey, -ex, -ez - 90.0])


def _display_rpy_to_rotation(rpy_deg):
    roll, pitch, yaw = [float(v) for v in rpy_deg]
    return Rotation.from_euler("xyz", [-pitch, roll, -yaw - 90.0], degrees=True)


def _front_handover_yaw_deg(yaw_deg):
    yaw = float(_normalize_display_rpy([0.0, 0.0, yaw_deg])[2])
    candidates = [0.0, 180.0, -180.0]
    best = min(
        candidates,
        key=lambda cand: abs(float(_normalize_display_rpy([0.0, 0.0, yaw - cand])[2])),
    )
    return float(best)


def _display_rpy_from_axes(jaw_axis_world, approach_world):
    jaw_axis = _unit(jaw_axis_world)
    approach = _unit(approach_world)
    third = np.cross(approach, jaw_axis)
    third_norm = float(np.linalg.norm(third))
    if third_norm < 1e-9:
        raise RuntimeError(
            f"degenerate side-grasp axes: jaw={jaw_axis.tolist()} "
            f"approach={approach.tolist()}"
        )
    third = third / third_norm
    jaw_axis = np.cross(third, approach)
    jaw_axis = jaw_axis / max(float(np.linalg.norm(jaw_axis)), 1e-9)
    rot = Rotation.from_matrix(np.column_stack([jaw_axis, third, approach]))
    return _rotation_to_display_rpy(rot)


def _robot_vec(state, side, field):
    value = getattr(state, f"{side}_{field}", None)
    if value is None:
        arms = getattr(state, "arms", None)
        if isinstance(arms, dict) and side in arms:
            value = getattr(arms[side], field, None)
    if value is None:
        raise AttributeError(f"robot state missing {side}_{field}")
    arr = np.asarray(value, dtype=float).reshape(-1)
    return [float(v) for v in arr]


def _gripper_pos(side):
    state = get_robot_state()
    value = getattr(state, f"{side}_gripper_pos", None)
    if value is None:
        arms = getattr(state, "arms", None)
        if isinstance(arms, dict) and side in arms:
            value = getattr(arms[side], "gripper_pos", None)
    if value is None:
        raise AttributeError(f"robot state missing {side}_gripper_pos")
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _gripper_call_kwargs(vel_limit=None, torque_limit=None):
    kwargs = {}
    if vel_limit is not None:
        kwargs["vel_limit"] = float(vel_limit)
    if torque_limit is not None:
        kwargs["torque_limit"] = float(torque_limit)
    return kwargs


def _open_gripper(side, vel_limit=None, torque_limit=None):
    kwargs = _gripper_call_kwargs(vel_limit=vel_limit, torque_limit=torque_limit)
    if kwargs:
        print(f"[gpu_handover] open_gripper({side!r}, {kwargs})")
    return open_gripper(side, **kwargs)


def _close_gripper(side, vel_limit=None, torque_limit=None):
    kwargs = _gripper_call_kwargs(vel_limit=vel_limit, torque_limit=torque_limit)
    if kwargs:
        print(f"[gpu_handover] close_gripper({side!r}, {kwargs})")
    return close_gripper(side, **kwargs)


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


def _xy_unit(vec, fallback=None):
    arr = np.asarray(vec, dtype=float).reshape(3).copy()
    arr[2] = 0.0
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        if fallback is None:
            raise RuntimeError(f"cannot project vector to XY plane: {vec}")
        arr = np.asarray(fallback, dtype=float).reshape(3).copy()
        arr[2] = 0.0
        norm = float(np.linalg.norm(arr))
        if norm < 1e-9:
            raise RuntimeError(f"invalid XY fallback vector: {fallback}")
    return arr / norm


def _perp_xy(vec):
    base = _xy_unit(vec)
    return _xy_unit([-base[1], base[0], 0.0], fallback=[0.0, 1.0, 0.0])


def _toward_left(axis_xy):
    axis = _xy_unit(axis_xy, fallback=WORLD_LEFT)
    return axis if float(axis[1]) >= 0.0 else -axis


def _move(side, pos, rpy):
    return _move_with_speed(side, pos, rpy, planning_speed=MOVE_PLANNING_SPEED)


def _move_holding_right(pos, rpy):
    return _move_with_speed(
        "right",
        pos,
        rpy,
        planning_speed=POST_PICK_PLANNING_SPEED,
    )


def _home_pose_for_side(side):
    pos_key = f"{side}_home_xyz"
    rpy_key = f"{side}_birdeye_view_rpy"
    return (
        [float(v) for v in RUN_CONFIG[pos_key]],
        [float(v) for v in RUN_CONFIG[rpy_key]],
    )


def _as_uint8_rgb(image):
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        return None
    if arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr.copy()


def _camera_matrix(camera):
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
            for v in np.asarray(intr, dtype=float).reshape(-1)[:4]
        ]
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _camera_T_cam_world(camera):
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


def _project_pixel_to_plane_world(pixel_xy, cam_K, T_cam_world, plane_z_m):
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


def _median_depth_at_pixel(depth_image, pixel_xy, radius_px=SLOT_CENTER_DEPTH_WINDOW_RADIUS_PX):
    depth = np.asarray(depth_image, dtype=np.float64)
    if depth.ndim != 2:
        return None
    u = int(round(float(pixel_xy[0])))
    v = int(round(float(pixel_xy[1])))
    h, w = depth.shape[:2]
    if w <= 0 or h <= 0:
        return None
    x0 = max(0, u - int(radius_px))
    x1 = min(w, u + int(radius_px) + 1)
    y0 = max(0, v - int(radius_px))
    y1 = min(h, v + int(radius_px) + 1)
    patch = depth[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0.05)
    if not np.any(valid):
        return None
    return float(np.median(patch[valid]))


def _project_pixel_to_depth_world(pixel_xy, depth_m, cam_K, T_cam_world):
    if depth_m is None or not np.isfinite(depth_m) or float(depth_m) <= 0.05:
        return None
    u, v = [float(x) for x in pixel_xy]
    fx = float(cam_K[0, 0])
    fy = float(cam_K[1, 1])
    cx = float(cam_K[0, 2])
    cy = float(cam_K[1, 2])
    point_cam = np.asarray(
        [
            (u - cx) * float(depth_m) / fx,
            (v - cy) * float(depth_m) / fy,
            float(depth_m),
        ],
        dtype=np.float64,
    )
    return (
        np.asarray(T_cam_world[:3, :3], dtype=np.float64) @ point_cam
        + np.asarray(T_cam_world[:3, 3], dtype=np.float64)
    )


def _estimate_mask_translation_px(reference_mask, current_mask, max_shift_px=120):
    ref = np.asarray(reference_mask).astype(bool)
    cur = np.asarray(current_mask).astype(bool)
    if ref.ndim != 2 or cur.ndim != 2 or ref.shape != cur.shape:
        raise RuntimeError("reference/current motherboard mask shape mismatch")
    if not np.any(ref) or not np.any(cur):
        raise RuntimeError("empty reference/current motherboard mask for registration")
    try:
        return _estimate_mask_translation_from_edges_px(ref, cur)
    except Exception:
        pass
    corr = np.fft.ifftn(
        np.fft.fftn(cur.astype(np.float32))
        * np.conj(np.fft.fftn(ref.astype(np.float32)))
    ).real
    corr = np.fft.fftshift(corr)
    h, w = corr.shape
    cy = h // 2
    cx = w // 2
    max_shift = max(1, int(max_shift_px))
    y0 = max(0, cy - max_shift)
    y1 = min(h, cy + max_shift + 1)
    x0 = max(0, cx - max_shift)
    x1 = min(w, cx + max_shift + 1)
    window = corr[y0:y1, x0:x1]
    if window.size <= 0:
        raise RuntimeError("empty translation search window for motherboard mask registration")
    best_idx = np.unravel_index(int(np.argmax(window)), window.shape)
    best_y = y0 + int(best_idx[0])
    best_x = x0 + int(best_idx[1])
    dy_px = int(best_y - cy)
    dx_px = int(best_x - cx)
    return {
        "dx_px": dx_px,
        "dy_px": dy_px,
        "score": float(window[best_idx]),
        "method": "fft_full_mask",
    }


def _mask_bbox_xyxy(mask):
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.ndim != 2 or not np.any(mask_bool):
        raise RuntimeError("empty motherboard mask for bbox extraction")
    ys, xs = np.where(mask_bool)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _bbox_center_px_from_xywh(bbox_xywh):
    if bbox_xywh is None or len(bbox_xywh) != 4:
        return None
    x, y, w_box, h_box = [float(v) for v in bbox_xywh]
    if w_box <= 0.0 or h_box <= 0.0:
        return None
    return np.asarray(
        [x + 0.5 * w_box, y + 0.5 * h_box],
        dtype=np.float64,
    )


def _edge_support_counts(mask, bbox_xyxy, band_px=4):
    mask_bool = np.asarray(mask).astype(bool)
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    h, w = mask_bool.shape
    band = max(1, int(band_px))
    left = mask_bool[:, max(0, x0 - band) : min(w, x0 + band + 1)]
    right = mask_bool[:, max(0, x1 - band) : min(w, x1 + band + 1)]
    top = mask_bool[max(0, y0 - band) : min(h, y0 + band + 1), :]
    bottom = mask_bool[max(0, y1 - band) : min(h, y1 + band + 1), :]
    return {
        "left": int(left.sum()),
        "right": int(right.sum()),
        "top": int(top.sum()),
        "bottom": int(bottom.sum()),
    }


def _estimate_mask_translation_from_edges_px(
    reference_mask,
    current_mask,
    min_edge_support_px=80,
):
    ref = np.asarray(reference_mask).astype(bool)
    cur = np.asarray(current_mask).astype(bool)
    ref_bbox = _mask_bbox_xyxy(ref)
    cur_bbox = _mask_bbox_xyxy(cur)
    ref_support = _edge_support_counts(ref, ref_bbox)
    cur_support = _edge_support_counts(cur, cur_bbox)

    def _weighted_shift(candidates):
        if not candidates:
            raise RuntimeError("no usable edge candidates for motherboard registration")
        shifts = np.asarray([float(shift) for shift, _ in candidates], dtype=np.float64)
        weights = np.asarray([float(weight) for _, weight in candidates], dtype=np.float64)
        return float(np.sum(shifts * weights) / max(np.sum(weights), 1e-9))

    min_support = max(1, int(min_edge_support_px))
    x_candidates = []
    y_candidates = []
    if min(ref_support["left"], cur_support["left"]) >= min_support:
        x_candidates.append((cur_bbox[0] - ref_bbox[0], min(ref_support["left"], cur_support["left"])))
    if min(ref_support["right"], cur_support["right"]) >= min_support:
        x_candidates.append((cur_bbox[2] - ref_bbox[2], min(ref_support["right"], cur_support["right"])))
    if min(ref_support["top"], cur_support["top"]) >= min_support:
        y_candidates.append((cur_bbox[1] - ref_bbox[1], min(ref_support["top"], cur_support["top"])))
    if min(ref_support["bottom"], cur_support["bottom"]) >= min_support:
        y_candidates.append((cur_bbox[3] - ref_bbox[3], min(ref_support["bottom"], cur_support["bottom"])))

    dx_px = int(round(_weighted_shift(x_candidates)))
    dy_px = int(round(_weighted_shift(y_candidates)))
    return {
        "dx_px": dx_px,
        "dy_px": dy_px,
        "score": float(sum(weight for _, weight in x_candidates + y_candidates)),
        "method": "edge_bbox",
        "reference_bbox_xyxy": ref_bbox,
        "current_bbox_xyxy": cur_bbox,
        "reference_edge_support": ref_support,
        "current_edge_support": cur_support,
    }


def _seg_attr(seg, name, default=None):
    if isinstance(seg, dict):
        return seg.get(name, default)
    return getattr(seg, name, default)


def _artifact_output_dir(name="gpu_handover"):
    try:
        from enpire.env.forge.cap.agent.tools._artifact_log import _vis_subdir

        out = _vis_subdir(name)
        if out is not None:
            return Path(out)
    except Exception:
        pass
    fallback = Path("logs") / f"{Path(__file__).stem}_artifacts"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _save_ppm(path, rgb):
    arr = _as_uint8_rgb(rgb)
    if arr is None:
        raise RuntimeError(f"cannot save invalid RGB image to {path}")
    h, w = arr.shape[:2]
    path = Path(path)
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr.tobytes())
    return path


def _save_overlay_artifact(rgb, *, fallback_path, tag, label, subdir):
    arr = _as_uint8_rgb(rgb)
    if arr is None:
        raise RuntimeError(f"cannot save invalid RGB image to {fallback_path}")
    try:
        from enpire.env.forge.cap.agent.tools._artifact_log import log_image

        saved = log_image(
            arr,
            tag=tag,
            label=label,
            subdir=subdir,
        )
        if saved is not None:
            return Path(saved)
    except Exception:
        pass
    return _save_ppm(fallback_path, arr)


def _draw_rect_rgb(canvas, bbox, color, width=2):
    arr = np.asarray(canvas)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    h, w = arr.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return
    for k in range(max(1, int(width))):
        yy0 = max(0, min(h - 1, y0 + k))
        yy1 = max(0, min(h - 1, y1 - k))
        xx0 = max(0, min(w - 1, x0 + k))
        xx1 = max(0, min(w - 1, x1 - k))
        arr[yy0, xx0 : xx1 + 1] = color
        arr[yy1, xx0 : xx1 + 1] = color
        arr[yy0 : yy1 + 1, xx0] = color
        arr[yy0 : yy1 + 1, xx1] = color


def _draw_cross_rgb(canvas, center_xy, color, radius=6):
    arr = np.asarray(canvas)
    h, w = arr.shape[:2]
    x_f = float(center_xy[0])
    y_f = float(center_xy[1])
    if not (np.isfinite(x_f) and np.isfinite(y_f)):
        return
    x = int(round(x_f))
    y = int(round(y_f))
    x0 = max(0, x - int(radius))
    x1 = min(w - 1, x + int(radius))
    y0 = max(0, y - int(radius))
    y1 = min(h - 1, y + int(radius))
    if 0 <= y < h and x0 <= x1:
        arr[y, x0 : x1 + 1] = color
    if 0 <= x < w and y0 <= y1:
        arr[y0 : y1 + 1, x] = color


def _draw_line_rgb(canvas, p0, p1, color, width=2):
    arr = np.asarray(canvas)
    h, w = arr.shape[:2]
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    steps = max(int(np.ceil(max(abs(x1 - x0), abs(y1 - y0)))), 1)
    xs = np.linspace(x0, x1, steps + 1)
    ys = np.linspace(y0, y1, steps + 1)
    r = max(0, int(width) // 2)
    for x_f, y_f in zip(xs, ys):
        x = int(round(x_f))
        y = int(round(y_f))
        xx0 = max(0, x - r)
        xx1 = min(w - 1, x + r)
        yy0 = max(0, y - r)
        yy1 = min(h - 1, y + r)
        arr[yy0 : yy1 + 1, xx0 : xx1 + 1] = color


def _segment_motherboard_mask(camera="top"):
    last_error = None
    for query in MOTHERBOARD_QUERIES:
        try:
            seg = segment_object(query=query, camera=camera, score_thresh=0.1)
            mask = _seg_attr(seg, "mask")
            mask_bool = np.asarray(mask) > 0 if mask is not None else None
            if mask_bool is None or mask_bool.ndim != 2 or not np.any(mask_bool):
                raise RuntimeError("empty segmentation mask")
            return {
                "query": query,
                "mask": mask_bool,
                "score": _seg_attr(seg, "score"),
                "mask_area": _seg_attr(seg, "mask_area"),
                "bbox_xywh": _seg_attr(seg, "bbox_xywh"),
            }
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"could not segment motherboard; last_error={last_error}")


def _extract_mask_world_points(mask, depth_image, cam_K, T_cam_world, z_clip_max_m=None):
    depth = np.asarray(depth_image, dtype=np.float64)
    mask_bool = np.asarray(mask).astype(bool)
    if depth.ndim != 2 or mask_bool.ndim != 2 or depth.shape != mask_bool.shape:
        raise RuntimeError("depth/mask shape mismatch for motherboard debug cloud")
    valid = mask_bool & np.isfinite(depth) & (depth > 0.05)
    if not np.any(valid):
        raise RuntimeError("no valid depth points inside motherboard mask")
    ys, xs = np.where(valid)
    zs = depth[valid]
    fx = float(cam_K[0, 0])
    fy = float(cam_K[1, 1])
    cx = float(cam_K[0, 2])
    cy = float(cam_K[1, 2])
    pts_cam = np.stack(
        [
            (xs.astype(np.float64) - cx) * zs / fx,
            (ys.astype(np.float64) - cy) * zs / fy,
            zs,
        ],
        axis=1,
    )
    R = np.asarray(T_cam_world[:3, :3], dtype=np.float64)
    t = np.asarray(T_cam_world[:3, 3], dtype=np.float64)
    world_points = (R @ pts_cam.T).T + t
    if z_clip_max_m is not None:
        world_points = world_points[world_points[:, 2] <= float(z_clip_max_m)]
        if world_points.shape[0] < 20:
            raise RuntimeError(
                "too few clipped depth points inside motherboard mask after z filtering"
            )
    return world_points


def _fit_xy_obb(points_world):
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 20:
        raise RuntimeError("too few points for motherboard debug OBB")
    xy = points[:, :2]
    centroid = np.median(xy, axis=0)
    centered = xy - centroid
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    major = evecs[:, order[0]]
    minor = evecs[:, order[1]]
    if np.linalg.norm(major) < 1e-9 or np.linalg.norm(minor) < 1e-9:
        raise RuntimeError("degenerate motherboard debug OBB axes")
    if np.cross(np.array([major[0], major[1], 0.0]), np.array([minor[0], minor[1], 0.0]))[2] < 0:
        minor = -minor
    proj_major = centered @ major
    proj_minor = centered @ minor
    major_lo, major_hi = np.percentile(proj_major, [2.0, 98.0])
    minor_lo, minor_hi = np.percentile(proj_minor, [2.0, 98.0])
    center_xy = (
        centroid
        + 0.5 * (major_lo + major_hi) * major
        + 0.5 * (minor_lo + minor_hi) * minor
    )
    z_lo, z_hi = np.percentile(points[:, 2], [5.0, 95.0])
    return {
        "center_world": np.array([center_xy[0], center_xy[1], 0.5 * (z_lo + z_hi)], dtype=np.float64),
        "major_axis_world": np.array([major[0], major[1], 0.0], dtype=np.float64),
        "minor_axis_world": np.array([minor[0], minor[1], 0.0], dtype=np.float64),
        "half_major_m": 0.5 * float(major_hi - major_lo),
        "half_minor_m": 0.5 * float(minor_hi - minor_lo),
        "z_min": float(z_lo),
        "z_max": float(z_hi),
        "num_points": int(points.shape[0]),
    }


def _copy_motherboard_obb(obb):
    copied = {}
    for key, value in dict(obb).items():
        key_str = str(key)
        if key_str.endswith("_world"):
            copied[key] = np.asarray(value, dtype=np.float64).copy()
        elif key_str in {"half_major_m", "half_minor_m", "z_min", "z_max"}:
            copied[key] = float(value)
        elif key_str == "num_points":
            copied[key] = int(value)
        else:
            copied[key] = value
    return copied


def _override_obb_center_xy(obb, center_world_xy):
    if obb is None or center_world_xy is None:
        return obb
    updated = _copy_motherboard_obb(obb)
    target_center = np.asarray(center_world_xy, dtype=np.float64).reshape(3)
    center = np.asarray(updated["center_world"], dtype=np.float64).reshape(3).copy()
    center[0] = float(target_center[0])
    center[1] = float(target_center[1])
    updated["center_world"] = center
    return updated


def _select_stable_refresh_center_world(
    reference_center_world,
    *,
    tracked_center_world=None,
    registered_center_world=None,
    shift_limit_m=None,
):
    ref_center = np.asarray(reference_center_world, dtype=np.float64).reshape(3)
    shift_limit = float(
        REFRESH_MOTHERBOARD_CENTER_SHIFT_LIMIT_M
        if shift_limit_m is None
        else shift_limit_m
    )
    shift_limit = max(1e-6, shift_limit)
    proposals = []
    if tracked_center_world is not None:
        proposals.append(
            (
                "tracking",
                np.asarray(tracked_center_world, dtype=np.float64).reshape(3),
            )
        )
    if registered_center_world is not None:
        proposals.append(
            (
                "registration",
                np.asarray(registered_center_world, dtype=np.float64).reshape(3),
            )
        )
    if not proposals:
        return ref_center.copy(), {
            "method": "hold_reference",
            "shift_m": 0.0,
            "clamped": False,
            "proposal_count": 0,
        }

    chosen_label, chosen_center = proposals[0]
    if len(proposals) == 2:
        tracked = proposals[0][1]
        registered = proposals[1][1]
        pair_dist_m = float(np.linalg.norm(tracked[:2] - registered[:2]))
        if pair_dist_m <= max(0.015, 1.5 * shift_limit):
            chosen_label = "tracking_registration_mean"
            chosen_center = tracked.copy()
            chosen_center[:2] = 0.5 * (tracked[:2] + registered[:2])
        else:
            candidates = []
            for label, proposal in proposals:
                proposal_shift = float(np.linalg.norm(proposal[:2] - ref_center[:2]))
                candidates.append((proposal_shift, label, proposal))
            candidates.sort(key=lambda item: item[0])
            chosen_shift, chosen_label, chosen_center = candidates[0]
            if chosen_shift > shift_limit:
                return ref_center.copy(), {
                    "method": "hold_reference_divergent",
                    "shift_m": 0.0,
                    "clamped": False,
                    "proposal_count": 2,
                    "pair_dist_m": pair_dist_m,
                    "tracked_shift_m": float(np.linalg.norm(tracked[:2] - ref_center[:2])),
                    "registered_shift_m": float(
                        np.linalg.norm(registered[:2] - ref_center[:2])
                    ),
                }

    delta_xy = chosen_center[:2] - ref_center[:2]
    shift_m = float(np.linalg.norm(delta_xy))
    clamped = False
    if shift_m > shift_limit:
        chosen_center = chosen_center.copy()
        chosen_center[:2] = ref_center[:2] + delta_xy * (shift_limit / max(shift_m, 1e-9))
        shift_m = shift_limit
        clamped = True
    return chosen_center, {
        "method": chosen_label,
        "shift_m": shift_m,
        "clamped": clamped,
        "proposal_count": len(proposals),
        "tracked_shift_m": None
        if tracked_center_world is None
        else float(
            np.linalg.norm(
                np.asarray(tracked_center_world, dtype=np.float64).reshape(3)[:2]
                - ref_center[:2]
            )
        ),
        "registered_shift_m": None
        if registered_center_world is None
        else float(
            np.linalg.norm(
                np.asarray(registered_center_world, dtype=np.float64).reshape(3)[:2]
                - ref_center[:2]
            )
        ),
    }


def _extract_detection3d_fields(det):
    if det is None:
        raise RuntimeError("empty BundleSDF detection")
    if isinstance(det, dict):
        position_3d = det.get("position_3d")
        score = det.get("score", 0.0)
        rpy = det.get("rpy", [])
        half_extents = det.get("half_extents", [])
    else:
        position_3d = getattr(det, "position_3d", None)
        score = getattr(det, "score", 0.0)
        rpy = getattr(det, "rpy", [])
        half_extents = getattr(det, "half_extents", [])
    if position_3d is None:
        raise RuntimeError("BundleSDF detection did not return position_3d")
    return {
        "position_3d": np.asarray(position_3d, dtype=np.float64).reshape(3),
        "score": float(score),
        "rpy": [float(v) for v in (rpy or [])],
        "half_extents": [float(v) for v in (half_extents or [])],
    }


def _start_motherboard_tracking(reference_scene, camera="top"):
    if not USE_MOTHERBOARD_TRACKING:
        return None
    track_query = (
        MOTHERBOARD_TRACK_QUERY
        or str(reference_scene.get("motherboard_debug_3d_bbox", {}).get("query", "")).strip()
        or MOTHERBOARD_QUERIES[0]
    )
    print(
        "[gpu_handover] Step 0b: start BundleSDF motherboard tracking "
        f"query={track_query!r} camera={camera!r} session={MOTHERBOARD_TRACK_SESSION_NAME!r}"
    )
    detections = detect_object(
        query=track_query,
        camera=camera,
        backend="bundlesdf",
        max_retries=MOTHERBOARD_TRACK_MAX_RETRIES,
        name=MOTHERBOARD_TRACK_SESSION_NAME,
    )
    if not detections:
        raise RuntimeError("BundleSDF motherboard tracking returned no detections")
    det_fields = _extract_detection3d_fields(detections[0])
    return {
        "query": track_query,
        "camera": str(camera),
        "session_name": MOTHERBOARD_TRACK_SESSION_NAME,
        "center_world": det_fields["position_3d"],
        "score": det_fields["score"],
        "rpy": det_fields["rpy"],
        "half_extents": det_fields["half_extents"],
        "backend": "bundlesdf",
    }


def _stop_motherboard_tracking(reference_scene):
    tracking = None if reference_scene is None else reference_scene.get("motherboard_tracking")
    if not tracking:
        return
    try:
        end_detection(
            object=str(tracking["query"]),
            name=str(tracking["session_name"]),
        )
    except Exception as exc:
        print(f"[gpu_handover] BundleSDF motherboard tracking cleanup failed: {exc}")


def _fit_xy_obb_with_fixed_reference(points_world, reference_obb):
    if not reference_obb:
        return _fit_xy_obb(points_world)
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 20:
        raise RuntimeError("too few points for constrained motherboard refresh OBB")
    ref_center = np.asarray(reference_obb["center_world"], dtype=np.float64).reshape(3)
    ref_major = _xy_unit(reference_obb["major_axis_world"], fallback=WORLD_LEFT)
    ref_minor = _xy_unit(reference_obb["minor_axis_world"], fallback=_perp_xy(ref_major))
    half_major = float(reference_obb["half_major_m"])
    half_minor = float(reference_obb["half_minor_m"])
    xy = points[:, :2] - ref_center[:2]
    proj_major = xy @ ref_major[:2]
    proj_minor = xy @ ref_minor[:2]
    q_major_lo, q_major_hi = np.percentile(proj_major, [2.0, 98.0])
    q_minor_lo, q_minor_hi = np.percentile(proj_minor, [2.0, 98.0])
    raw_major_shift = 0.5 * float(q_major_lo + q_major_hi)
    raw_minor_shift = 0.5 * float(q_minor_lo + q_minor_hi)
    pad_m = 0.005
    feasible_major_lo = float(q_major_hi - (half_major + pad_m))
    feasible_major_hi = float(q_major_lo + (half_major + pad_m))
    feasible_minor_lo = float(q_minor_hi - (half_minor + pad_m))
    feasible_minor_hi = float(q_minor_lo + (half_minor + pad_m))
    if feasible_major_lo <= feasible_major_hi:
        major_shift = float(np.clip(raw_major_shift, feasible_major_lo, feasible_major_hi))
    else:
        major_shift = raw_major_shift
    if feasible_minor_lo <= feasible_minor_hi:
        minor_shift = float(np.clip(raw_minor_shift, feasible_minor_lo, feasible_minor_hi))
    else:
        minor_shift = raw_minor_shift
    shift_limit_m = float(REFRESH_MOTHERBOARD_CENTER_SHIFT_LIMIT_M)
    major_shift = float(np.clip(major_shift, -shift_limit_m, shift_limit_m))
    minor_shift = float(np.clip(minor_shift, -shift_limit_m, shift_limit_m))
    center_xy = (
        ref_center[:2]
        + major_shift * ref_major[:2]
        + minor_shift * ref_minor[:2]
    )
    z_lo, z_hi = np.percentile(points[:, 2], [5.0, 95.0])
    return {
        "center_world": np.array([center_xy[0], center_xy[1], 0.5 * (z_lo + z_hi)], dtype=np.float64),
        "major_axis_world": np.array([ref_major[0], ref_major[1], 0.0], dtype=np.float64),
        "minor_axis_world": np.array([ref_minor[0], ref_minor[1], 0.0], dtype=np.float64),
        "half_major_m": half_major,
        "half_minor_m": half_minor,
        "z_min": float(z_lo),
        "z_max": float(z_hi),
        "num_points": int(points.shape[0]),
        "raw_major_shift_m": float(raw_major_shift),
        "raw_minor_shift_m": float(raw_minor_shift),
        "major_shift_m": float(major_shift),
        "minor_shift_m": float(minor_shift),
    }


def _project_debug_3d_bbox_pixels(obb, cam_K, T_cam_world):
    center = np.asarray(obb["center_world"], dtype=np.float64).reshape(3)
    major = np.asarray(obb["major_axis_world"], dtype=np.float64).reshape(3)
    minor = np.asarray(obb["minor_axis_world"], dtype=np.float64).reshape(3)
    half_major = float(obb["half_major_m"])
    half_minor = float(obb["half_minor_m"])
    z_min = float(obb["z_min"])
    z_max = float(obb["z_max"])
    base_xy = center[:2]
    corners_world = []
    for z in (z_min, z_max):
        for s0, s1 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            xy = base_xy + s0 * half_major * major[:2] + s1 * half_minor * minor[:2]
            corners_world.append(np.array([xy[0], xy[1], z], dtype=np.float64))
    pixels = []
    for point in corners_world:
        px = _project_world_to_pixel(point, cam_K, T_cam_world)
        if px is None:
            return None
        pixels.append(px)
    return np.asarray(pixels, dtype=np.float64).reshape(8, 2)


def _compute_initial_motherboard_3d_bbox(
    camera,
    depth=None,
    cam_K=None,
    T_cam_world=None,
    z_clip_max_m=None,
    reference_obb=None,
):
    seg_record = _segment_motherboard_mask(camera=camera)
    if depth is None:
        depth = render_depth(camera=camera)
    if depth is None:
        raise RuntimeError("no top depth image available for motherboard debug OBB")
    cam_K = _camera_matrix(camera) if cam_K is None else cam_K
    T_cam_world = _camera_T_cam_world(camera) if T_cam_world is None else T_cam_world
    world_points = _extract_mask_world_points(
        seg_record["mask"],
        depth,
        cam_K,
        T_cam_world,
        z_clip_max_m=z_clip_max_m,
    )
    obb = (
        _fit_xy_obb_with_fixed_reference(world_points, reference_obb)
        if reference_obb is not None
        else _fit_xy_obb(world_points)
    )
    return {
        "seg_record": seg_record,
        "obb": obb,
        "depth": depth,
        "cam_K": cam_K,
        "T_cam_world": T_cam_world,
    }


def _save_initial_motherboard_3d_bbox_artifacts(
    rgb,
    camera,
    scene,
    artifact_prefix="initial",
    obb_record=None,
):
    try:
        if obb_record is None:
            obb_record = _compute_initial_motherboard_3d_bbox(camera=camera)
        seg_record = obb_record["seg_record"]
        obb = obb_record["obb"]
        cam_K = obb_record["cam_K"]
        T_cam_world = obb_record["T_cam_world"]
        pixels = _project_debug_3d_bbox_pixels(obb, cam_K, T_cam_world)
        if pixels is None:
            raise RuntimeError("could not project motherboard debug OBB")
        overlay = _as_uint8_rgb(rgb)
        if overlay is None:
            raise RuntimeError("invalid RGB image for motherboard debug OBB overlay")
        reference_obb = scene.get("motherboard_reference_3d_bbox")
        if reference_obb is not None:
            ref_pixels = _project_debug_3d_bbox_pixels(reference_obb, cam_K, T_cam_world)
            if ref_pixels is not None:
                ref_bottom = ref_pixels[:4]
                ref_top = ref_pixels[4:]
                ref_color = np.array([255, 80, 255], dtype=np.uint8)
                for quad in (ref_bottom, ref_top):
                    for i in range(4):
                        _draw_line_rgb(overlay, quad[i], quad[(i + 1) % 4], ref_color, width=1)
                for i in range(4):
                    _draw_line_rgb(overlay, ref_bottom[i], ref_top[i], ref_color, width=1)
        bottom = pixels[:4]
        top = pixels[4:]
        box_color = np.array([80, 220, 255], dtype=np.uint8)
        axis_major_color = np.array([40, 80, 255], dtype=np.uint8)
        axis_minor_color = np.array([80, 255, 80], dtype=np.uint8)
        for quad in (bottom, top):
            for i in range(4):
                _draw_line_rgb(overlay, quad[i], quad[(i + 1) % 4], box_color, width=2)
        for i in range(4):
            _draw_line_rgb(overlay, bottom[i], top[i], box_color, width=2)
        center_top = np.array(
            [
                float(np.mean(top[:, 0])),
                float(np.mean(top[:, 1])),
            ],
            dtype=np.float64,
        )
        major_tip_world = np.asarray(obb["center_world"], dtype=np.float64) + np.asarray(
            obb["major_axis_world"], dtype=np.float64
        ) * float(obb["half_major_m"])
        minor_tip_world = np.asarray(obb["center_world"], dtype=np.float64) + np.asarray(
            obb["minor_axis_world"], dtype=np.float64
        ) * float(obb["half_minor_m"])
        major_tip_px = _project_world_to_pixel(
            np.array([major_tip_world[0], major_tip_world[1], float(obb["z_max"])], dtype=np.float64),
            cam_K,
            T_cam_world,
        )
        minor_tip_px = _project_world_to_pixel(
            np.array([minor_tip_world[0], minor_tip_world[1], float(obb["z_max"])], dtype=np.float64),
            cam_K,
            T_cam_world,
        )
        if major_tip_px is not None:
            _draw_line_rgb(overlay, center_top, major_tip_px, axis_major_color, width=3)
        if minor_tip_px is not None:
            _draw_line_rgb(overlay, center_top, minor_tip_px, axis_minor_color, width=3)
        _draw_cross_rgb(overlay, center_top, np.array([255, 255, 80], dtype=np.uint8), radius=8)
        bbox_xywh = seg_record.get("bbox_xywh")
        if bbox_xywh is not None and len(bbox_xywh) == 4:
            x, y, w_box, h_box = [int(v) for v in bbox_xywh]
            _draw_rect_rgb(
                overlay,
                [x, y, x + w_box, y + h_box],
                np.array([255, 80, 80], dtype=np.uint8),
                width=2,
            )
        out_dir = _artifact_output_dir("gpu_handover")
        image_path = out_dir / f"{artifact_prefix}_motherboard_3d_bbox_debug.ppm"
        report_path = out_dir / f"{artifact_prefix}_motherboard_3d_bbox_debug.json"
        image_path = _save_overlay_artifact(
            overlay,
            fallback_path=image_path,
            tag="motherboard_3d_bbox_debug",
            label="motherboard",
            subdir="grasp",
        )
        report = {
            "query": seg_record["query"],
            "seg_score": None if seg_record.get("score") is None else float(seg_record["score"]),
            "mask_area": seg_record.get("mask_area"),
            "seg_bbox_center_px": (
                None
                if _bbox_center_px_from_xywh(seg_record.get("bbox_xywh")) is None
                else [
                    round(float(v), 2)
                    for v in _bbox_center_px_from_xywh(seg_record.get("bbox_xywh")).tolist()
                ]
            ),
            "center_world": [round(float(v), 5) for v in np.asarray(obb["center_world"], dtype=float).tolist()],
            "major_axis_world": [round(float(v), 5) for v in np.asarray(obb["major_axis_world"], dtype=float).tolist()],
            "minor_axis_world": [round(float(v), 5) for v in np.asarray(obb["minor_axis_world"], dtype=float).tolist()],
            "half_major_m": round(float(obb["half_major_m"]), 5),
            "half_minor_m": round(float(obb["half_minor_m"]), 5),
            "z_min": round(float(obb["z_min"]), 5),
            "z_max": round(float(obb["z_max"]), 5),
            "num_points": int(obb["num_points"]),
            "seg_bbox_xywh": seg_record.get("bbox_xywh"),
            "registration": scene.get("motherboard_registration"),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"[gpu_handover] Saved {artifact_prefix} motherboard 3D bbox debug artifacts: "
            f"overlay={image_path} report={report_path}"
        )
        scene["motherboard_debug_3d_bbox"] = report
        scene["motherboard_debug_3d_bbox_artifacts"] = {
            "overlay": str(image_path),
            "report": str(report_path),
        }
    except Exception as exc:
        print(f"[gpu_handover] Motherboard 3D bbox debug artifact failed: {exc}")


def _save_initial_scene_detection_artifacts(
    rgb,
    motherboard_det,
    slot_record,
    scene,
    artifact_prefix="initial",
):
    out_dir = _artifact_output_dir("gpu_handover")
    overlay = _as_uint8_rgb(rgb)
    if overlay is None:
        return None
    _draw_rect_rgb(overlay, motherboard_det["bbox"], np.array([255, 80, 80], dtype=np.uint8), width=3)
    for center in slot_record.get("selected_centers", []):
        _draw_cross_rgb(
            overlay,
            [float(center["x"]), float(center["y"])],
            np.array([80, 220, 255], dtype=np.uint8),
            radius=7,
        )
    _draw_cross_rgb(
        overlay,
        scene["socket_record"]["center_px"],
        np.array([80, 255, 80], dtype=np.uint8),
        radius=9,
    )
    _draw_cross_rgb(
        overlay,
        motherboard_det["center_px"],
        np.array([255, 220, 80], dtype=np.uint8),
        radius=9,
    )
    image_path = out_dir / f"{artifact_prefix}_motherboard_socket_detection.ppm"
    report_path = out_dir / f"{artifact_prefix}_motherboard_socket_detection.json"
    image_path = _save_overlay_artifact(
        overlay,
        fallback_path=image_path,
        tag="motherboard_socket_detection",
        label="motherboard",
        subdir="grasp",
    )
    report = {
        "motherboard_bbox": [int(v) for v in motherboard_det["bbox"]],
        "motherboard_center_px": [
            round(float(v), 2) for v in np.asarray(motherboard_det["center_px"], dtype=float).tolist()
        ],
        "motherboard_seg_bbox_xywh": scene.get("motherboard_seg_bbox_xywh"),
        "motherboard_seg_center_px": (
            None
            if scene.get("motherboard_seg_center_px") is None
            else [
                round(float(v), 2)
                for v in np.asarray(scene["motherboard_seg_center_px"], dtype=float).tolist()
            ]
        ),
        "motherboard_center_world": (
            None
            if scene.get("motherboard_center_world") is None
            else [round(float(v), 5) for v in np.asarray(scene["motherboard_center_world"], dtype=float).tolist()]
        ),
        "motherboard_right_edge_anchor_px": (
            None
            if scene.get("motherboard_right_edge_anchor_px") is None
            else [
                round(float(v), 2)
                for v in np.asarray(
                    scene["motherboard_right_edge_anchor_px"],
                    dtype=float,
                ).tolist()
            ]
        ),
        "motherboard_right_edge_anchor_world": (
            None
            if scene.get("motherboard_right_edge_anchor_world") is None
            else [
                round(float(v), 5)
                for v in np.asarray(
                    scene["motherboard_right_edge_anchor_world"],
                    dtype=float,
                ).tolist()
            ]
        ),
        "motherboard_top_z": round(float(scene["motherboard_top_z"]), 5),
        "slot_centers_px": [
            [int(center["x"]), int(center["y"])]
            for center in slot_record.get("selected_centers", [])
        ],
        "target_socket_number": int(scene["socket_record"].get("target_socket_number", 1)),
        "target_socket_index": int(scene["socket_record"].get("target_socket_index", 0)),
        "socket_center_px": [
            round(float(v), 2)
            for v in np.asarray(scene["socket_record"]["center_px"], dtype=float).tolist()
        ],
        "socket_world": [
            round(float(v), 5)
            for v in np.asarray(scene["socket_world"], dtype=float).tolist()
        ],
        "socket_hover_world": (
            None
            if scene.get("socket_hover_world") is None
            else [
                round(float(v), 5)
                for v in np.asarray(scene["socket_hover_world"], dtype=float).tolist()
            ]
        ),
        "socket_hover_row_major_coord_m": (
            None
            if scene.get("socket_hover_row_major_coord_m") is None
            else round(float(scene["socket_hover_row_major_coord_m"]), 5)
        ),
        "socket_hover_edge_offset_m": (
            None
            if scene.get("socket_hover_edge_offset_m") is None
            else round(float(scene["socket_hover_edge_offset_m"]), 5)
        ),
        "socket_hover_x_from_right_edge_m": (
            None
            if scene.get("socket_hover_x_from_right_edge_m") is None
            else round(float(scene["socket_hover_x_from_right_edge_m"]), 5)
        ),
        "socket_hover_side_sign": (
            None
            if scene.get("socket_hover_side_sign") is None
            else round(float(scene["socket_hover_side_sign"]), 3)
        ),
        "socket_hover_rpy": (
            None
            if scene.get("socket_hover_rpy") is None
            else [
                round(float(v), 3)
                for v in np.asarray(scene["socket_hover_rpy"], dtype=float).tolist()
            ]
        ),
        "motherboard_tracking": (
            None
            if scene.get("motherboard_tracking") is None
            else {
                "query": str(scene["motherboard_tracking"].get("query", "")),
                "camera": str(scene["motherboard_tracking"].get("camera", "")),
                "session_name": str(scene["motherboard_tracking"].get("session_name", "")),
                "score": float(scene["motherboard_tracking"].get("score", 0.0)),
                "center_world": [
                    round(float(v), 5)
                    for v in np.asarray(
                        scene["motherboard_tracking"].get("center_world", [0.0, 0.0, 0.0]),
                        dtype=float,
                    ).tolist()
                ],
            }
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[gpu_handover] Saved {artifact_prefix} motherboard/socket detection artifacts: "
        f"overlay={image_path} report={report_path}"
    )
    return {"overlay": str(image_path), "report": str(report_path)}


def _motherboard_right_edge_sign(center_world, minor_axis_world, half_minor_m, cam_K, T_cam_world):
    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    minor = _xy_unit(minor_axis_world, fallback=-WORLD_LEFT)
    edge_extent_m = max(0.0, float(half_minor_m))
    plus_edge_world = center + edge_extent_m * minor
    minus_edge_world = center - edge_extent_m * minor
    plus_px = _project_world_to_pixel(plus_edge_world, cam_K, T_cam_world)
    minus_px = _project_world_to_pixel(minus_edge_world, cam_K, T_cam_world)
    if plus_px is None and minus_px is None:
        return 1.0
    if plus_px is None:
        return -1.0
    if minus_px is None:
        return 1.0
    return 1.0 if float(plus_px[0]) >= float(minus_px[0]) else -1.0


def _target_socket_right_edge_offset_m(target_socket_number):
    idx = max(0, min(len(SOCKET_TARGET_RIGHT_EDGE_OFFSETS_M) - 1, int(target_socket_number) - 1))
    return float(SOCKET_TARGET_RIGHT_EDGE_OFFSETS_M[idx])


def _is_left_third_view_camera(camera):
    return str(camera).strip().lower() in {"left_fixed", "left_third"}


def _right_edge_half_extent_world_y(obb):
    major = _xy_unit(obb["major_axis_world"], fallback=WORLD_LEFT)
    minor = _xy_unit(obb["minor_axis_world"], fallback=_perp_xy(major))
    world_y = _xy_unit(WORLD_LEFT, fallback=WORLD_LEFT)
    return (
        abs(float(np.dot(major[:2], world_y[:2]))) * float(obb["half_major_m"])
        + abs(float(np.dot(minor[:2], world_y[:2]))) * float(obb["half_minor_m"])
    )


def _socket_hover_aux_xml_body_candidates(camera):
    if not _is_left_third_view_camera(camera):
        return []
    env_name = os.environ.get(
        "CAP_LEFT_THIRD_CAMERA_FRAME",
        "",
    ).strip()
    names = [env_name, "top_camera_left_d435", "top_camera_left_d405"]
    seen = set()
    ordered = []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _xml_body_world_pose(xml_path: Path, body_name: str):
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return None

    def _quat_wxyz_to_rotmat(quat):
        w, x, y, z = quat
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _parse_vec(attr, default):
        if not attr:
            return np.asarray(default, dtype=np.float64)
        return np.asarray([float(v) for v in attr.split()], dtype=np.float64)

    def _find(body, parent_pos, parent_rot):
        local_pos = _parse_vec(body.get("pos"), (0.0, 0.0, 0.0))
        local_quat = _parse_vec(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
        local_rot = _quat_wxyz_to_rotmat(local_quat)
        world_pos = parent_pos + parent_rot @ local_pos
        world_rot = parent_rot @ local_rot
        if body.get("name") == body_name:
            return world_pos, world_rot
        for child in body.findall("body"):
            found = _find(child, world_pos, world_rot)
            if found is not None:
                return found
        return None

    for body in worldbody.findall("body"):
        found = _find(body, np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64))
        if found is not None:
            return found
    return None


def _socket_hover_calibrated_aux_camera_T_cam_world(camera):
    cached = _SOCKET_HOVER_AUX_XML_T_CAM_WORLD_CACHE.get(camera)
    if cached is not None:
        return cached
    try:
        from enpire.env.forge.robot.models.station.paths import get_station_xml
    except Exception:
        _SOCKET_HOVER_AUX_XML_T_CAM_WORLD_CACHE[camera] = None
        return None
    xml_path = Path(get_station_xml()).expanduser()
    if not xml_path.is_file():
        _SOCKET_HOVER_AUX_XML_T_CAM_WORLD_CACHE[camera] = None
        return None
    for body_name in _socket_hover_aux_xml_body_candidates(camera):
        try:
            pose = _xml_body_world_pose(xml_path, body_name)
        except Exception:
            pose = None
        if pose is None:
            continue
        position, rotation = pose
        T_cam_world = np.eye(4, dtype=np.float64)
        T_cam_world[:3, :3] = rotation
        T_cam_world[:3, 3] = position
        _SOCKET_HOVER_AUX_XML_T_CAM_WORLD_CACHE[camera] = T_cam_world
        return T_cam_world
    _SOCKET_HOVER_AUX_XML_T_CAM_WORLD_CACHE[camera] = None
    return None


def _socket_hover_tracking_camera_T_cam_world(camera):
    if camera != SOCKET_HOVER_AUX_CAMERA:
        return _camera_T_cam_world(camera)
    return _socket_hover_calibrated_aux_camera_T_cam_world(camera)


def _camera_board_right_edge_image_side(camera):
    return "left" if _is_left_third_view_camera(camera) else "right"


def _mask_right_edge_anchor_px(mask, *, camera=None):
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2 or not np.any(mask_bool):
        return None
    valid_rows = np.flatnonzero(np.any(mask_bool, axis=1))
    if valid_rows.size == 0:
        return None
    if valid_rows.size >= 5:
        lo = int(np.floor(0.15 * (valid_rows.size - 1)))
        hi = int(np.ceil(0.85 * (valid_rows.size - 1)))
        valid_rows = valid_rows[lo : hi + 1]
    edge_xs = []
    edge_ys = []
    image_side = _camera_board_right_edge_image_side(camera)
    for row_idx in valid_rows.tolist():
        row_cols = np.flatnonzero(mask_bool[int(row_idx)])
        if row_cols.size == 0:
            continue
        edge_xs.append(float(row_cols.min() if image_side == "left" else row_cols.max()))
        edge_ys.append(float(row_idx))
    if not edge_xs:
        return None
    return np.asarray(
        [float(np.median(edge_xs)), float(np.median(edge_ys))],
        dtype=np.float64,
    )


def _project_right_edge_anchor_world(anchor_px, cam_K, T_cam_world, top_z):
    if anchor_px is None or cam_K is None or T_cam_world is None:
        return None
    return _project_pixel_to_plane_world(
        anchor_px,
        cam_K,
        T_cam_world,
        top_z,
    )


def _derive_socket_hover_world_from_scene_geometry(scene, motherboard_top_z):
    localization_mode = str(scene.get("socket_hover_localization_mode", "edge")).strip().lower()
    if localization_mode == "center":
        center_world = scene.get("motherboard_center_world")
        center_offset_world = scene.get("socket_hover_center_offset_world")
        if center_world is None or center_offset_world is None:
            raise RuntimeError("motherboard center-based socket hover localization is unavailable")
        hover_world = np.asarray(center_world, dtype=np.float64).reshape(3).copy()
        hover_world += np.asarray(center_offset_world, dtype=np.float64).reshape(3)
    else:
        right_edge_anchor_world = scene.get("motherboard_right_edge_anchor_world")
        if right_edge_anchor_world is None:
            raise RuntimeError("motherboard right-edge anchor world is unavailable")
        hover_world = np.asarray(right_edge_anchor_world, dtype=np.float64).reshape(3).copy()
        hover_world[0] += float(scene["socket_hover_x_from_right_edge_m"])
        hover_world[1] -= float(scene["socket_hover_side_sign"]) * float(
            scene["socket_hover_edge_offset_m"]
        )
    hover_world[1] += float(SOCKET_HOVER_Y_TRIM_M)
    hover_world[2] = float(motherboard_top_z)
    return hover_world


def _capture_socket_hover_aux_observation(camera, motherboard_top_z):
    T_cam_world = _socket_hover_tracking_camera_T_cam_world(camera)
    rgb = _as_uint8_rgb(get_camera_image(camera=camera))
    if rgb is None:
        raise RuntimeError(f"could not capture {camera!r} RGB image")
    cam_K = _camera_matrix(camera) if T_cam_world is not None else None
    seg_record = _segment_motherboard_mask(camera=camera)
    score = float(seg_record.get("score", 0.0) or 0.0)
    mask_area = int(seg_record.get("mask_area", 0) or 0)
    center_px = _bbox_center_px_from_xywh(seg_record.get("bbox_xywh"))
    right_edge_anchor_px = _mask_right_edge_anchor_px(seg_record.get("mask"), camera=camera)
    center_world = None
    right_edge_anchor_world = None
    world_valid = False
    if T_cam_world is not None and cam_K is not None and center_px is not None:
        center_world = _project_pixel_to_plane_world(
            center_px,
            cam_K,
            T_cam_world,
            motherboard_top_z,
        )
        if right_edge_anchor_px is not None:
            right_edge_anchor_world = _project_right_edge_anchor_world(
                right_edge_anchor_px,
                cam_K,
                T_cam_world,
                motherboard_top_z,
            )
        if _is_left_third_view_camera(camera):
            world_valid = center_world is not None
        else:
            world_valid = center_world is not None and right_edge_anchor_world is not None
    return {
        "camera": camera,
        "rgb": rgb,
        "cam_K": cam_K,
        "T_cam_world": T_cam_world,
        "seg_record": seg_record,
        "score": score,
        "mask_area": mask_area,
        "center_px": center_px,
        "right_edge_anchor_px": right_edge_anchor_px,
        "center_world": center_world,
        "right_edge_anchor_world": right_edge_anchor_world,
        "guard_valid": (
            center_px is not None
            and (_is_left_third_view_camera(camera) or right_edge_anchor_px is not None)
            and score >= float(SOCKET_HOVER_AUX_MIN_SCORE)
        ),
        "world_valid": bool(world_valid and score >= float(SOCKET_HOVER_AUX_MIN_SCORE)),
    }


def _apply_socket_hover_world_estimate_to_scene(
    reference_scene,
    scene,
    *,
    center_world,
    right_edge_anchor_world,
    registration,
    cam_K,
    T_cam_world,
    center_update=None,
    pose_source_camera=None,
    aux_score=None,
    localization_mode=None,
):
    updated = dict(scene)
    updated["motherboard_center_world"] = np.asarray(
        center_world,
        dtype=np.float64,
    ).reshape(3)
    updated["motherboard_right_edge_anchor_world"] = np.asarray(
        right_edge_anchor_world,
        dtype=np.float64,
    ).reshape(3)
    updated["motherboard_registration"] = registration
    updated["motherboard_debug_3d_bbox"] = _copy_motherboard_obb(
        _override_obb_center_xy(
            reference_scene["motherboard_reference_3d_bbox"],
            updated["motherboard_center_world"],
        )
    )
    if center_update is not None:
        updated["motherboard_center_update"] = center_update
    if pose_source_camera is not None:
        updated["pose_source_camera"] = str(pose_source_camera)
    if aux_score is not None:
        updated["aux_camera_last_score"] = float(aux_score)
    if localization_mode is not None:
        updated["socket_hover_localization_mode"] = str(localization_mode)
    _set_scene_socket_hover_target(
        updated,
        _derive_socket_hover_world_from_scene_geometry(
            updated,
            float(reference_scene["motherboard_top_z"]),
        ),
        cam_K,
        T_cam_world,
    )
    return updated


def _maybe_apply_socket_hover_aux_pose(
    reference_scene,
    scene,
    *,
    primary_score,
    aux_obs,
    base_scene,
    cam_K,
    T_cam_world,
    stage_label,
):
    if aux_obs is None:
        return scene
    scene["aux_camera_name"] = aux_obs["camera"]
    scene["aux_camera_last_score"] = float(aux_obs["score"])
    scene["aux_camera_world_enabled"] = bool(aux_obs["world_valid"])
    if not aux_obs["guard_valid"] or not aux_obs["world_valid"]:
        return scene
    top_center_world = np.asarray(scene["motherboard_center_world"], dtype=np.float64).reshape(3)
    aux_center_world = np.asarray(aux_obs["center_world"], dtype=np.float64).reshape(3)
    aux_anchor_world = aux_center_world + np.asarray(
        reference_scene["motherboard_right_edge_anchor_from_center_offset_world"],
        dtype=np.float64,
    ).reshape(3)
    pose_diff_m = float(np.linalg.norm(aux_center_world[:2] - top_center_world[:2]))
    if (
        float(SOCKET_HOVER_AUX_MAX_POSE_DIFF_M) > 0.0
        and pose_diff_m > float(SOCKET_HOVER_AUX_MAX_POSE_DIFF_M)
    ):
        msg = (
            f"pose_diff_m={pose_diff_m:.4f} "
            f"max_pose_diff_m={float(SOCKET_HOVER_AUX_MAX_POSE_DIFF_M):.4f} "
            f"primary_score={float(primary_score):.3f} "
            f"aux_score={float(aux_obs['score']):.3f}"
        )
        if float(primary_score) >= float(SOCKET_HOVER_PRIMARY_MIN_SCORE):
            print(
                "[gpu_handover] Auxiliary camera "
                f"{aux_obs['camera']!r} {stage_label} pose rejected; "
                f"keeping primary motherboard pose: {msg}"
            )
            return scene
        raise RuntimeError(
            "auxiliary camera motherboard pose disagrees with primary camera "
            f"and primary score is below threshold: {msg}"
        )
    if float(primary_score) < float(SOCKET_HOVER_PRIMARY_MIN_SCORE):
        use_mode = "aux"
    elif pose_diff_m <= float(SOCKET_HOVER_AUX_BLEND_MAX_SHIFT_M):
        use_mode = "blend"
    elif bool(SOCKET_HOVER_AUX_CAMERA_PREFER_WORLD_POSE):
        use_mode = "aux"
    else:
        use_mode = "primary"
    if use_mode == "primary":
        return scene
    if use_mode == "aux":
        chosen_center_world = aux_center_world
        registration = {
            "method": "aux_camera_world",
            "camera": aux_obs["camera"],
            "score": float(aux_obs["score"]),
            "dx_px": 0,
            "dy_px": 0,
        }
    else:
        top_w = max(1e-6, float(primary_score))
        aux_w = max(1e-6, float(aux_obs["score"]))
        chosen_center_world = (top_w * top_center_world + aux_w * aux_center_world) / (top_w + aux_w)
        registration = {
            "method": "top_aux_blend",
            "camera": aux_obs["camera"],
            "score": float(aux_obs["score"]),
            "dx_px": 0,
            "dy_px": 0,
        }
    chosen_anchor_world = chosen_center_world + np.asarray(
        reference_scene["motherboard_right_edge_anchor_from_center_offset_world"],
        dtype=np.float64,
    ).reshape(3)

    center_update = scene.get("motherboard_center_update")
    if base_scene is not None:
        previous_pose_world = np.asarray(
            base_scene["motherboard_center_world"],
            dtype=np.float64,
        ).reshape(3)
        next_pose_world = chosen_center_world
        applied_shift_m = float(
            np.linalg.norm(next_pose_world[:2] - previous_pose_world[:2])
        )
        center_update = {
            "proposed_shift_m": applied_shift_m,
            "applied_shift_m": applied_shift_m,
            "held": False,
            "min_shift_m": float(REFRESH_MOTHERBOARD_CENTER_UPDATE_MIN_SHIFT_M),
            "max_shift_m": float(REFRESH_MOTHERBOARD_CENTER_UPDATE_MAX_SHIFT_M),
            "hold_reason": "",
        }

    print(
        "[gpu_handover] Auxiliary camera "
        f"{aux_obs['camera']!r} {stage_label} pose update: "
        f"mode={use_mode} "
        f"primary_score={float(primary_score):.3f} "
        f"aux_score={float(aux_obs['score']):.3f} "
        f"pose_diff_m={pose_diff_m:.4f} "
        "localization=center"
    )
    return _apply_socket_hover_world_estimate_to_scene(
        reference_scene,
        scene,
        center_world=chosen_center_world,
        right_edge_anchor_world=chosen_anchor_world,
        registration=registration,
        cam_K=cam_K,
        T_cam_world=T_cam_world,
        center_update=center_update,
        pose_source_camera=(
            aux_obs["camera"]
            if use_mode == "aux"
            else f"{scene.get('pose_source_camera', 'top')}+{aux_obs['camera']}"
        ),
        aux_score=aux_obs["score"],
        localization_mode="center",
    )


def _align_motherboard_obb_axes(obb, reference_obb):
    aligned = {
        key: (np.asarray(value, dtype=np.float64).copy() if key.endswith("_world") else value)
        for key, value in obb.items()
    }
    if not reference_obb:
        return aligned
    ref_major = _xy_unit(reference_obb["major_axis_world"], fallback=WORLD_LEFT)
    ref_minor = _xy_unit(reference_obb["minor_axis_world"], fallback=_perp_xy(ref_major))
    major = _xy_unit(aligned["major_axis_world"], fallback=ref_major)
    minor = _xy_unit(aligned["minor_axis_world"], fallback=ref_minor)
    if float(np.dot(major[:2], ref_major[:2])) < 0.0:
        major = -major
    if float(np.dot(minor[:2], ref_minor[:2])) < 0.0:
        minor = -minor
    aligned["major_axis_world"] = major
    aligned["minor_axis_world"] = minor
    return aligned


def _build_socket_hover_scene_record(
    *,
    top_z,
    hover_z,
    center_px,
    center_world,
    seg_center_px,
    seg_bbox_xywh,
    reference_mask,
    reference_obb,
    debug_obb,
    registration,
    x_offset_m,
    x_from_right_edge_m,
    edge_offset_m,
    side_sign,
    target_socket_number,
    right_edge_anchor_px,
    right_edge_anchor_world,
    center_offset_world,
    right_edge_anchor_from_center_offset_world,
    localization_mode="edge",
):
    return {
        "motherboard_top_z": float(top_z),
        "hover_z": float(hover_z),
        "motherboard_center_px": np.asarray(center_px, dtype=np.float64).reshape(2),
        "motherboard_center_world": np.asarray(center_world, dtype=np.float64).reshape(3),
        "motherboard_seg_center_px": (
            None
            if seg_center_px is None
            else np.asarray(seg_center_px, dtype=np.float64).reshape(2)
        ),
        "motherboard_seg_bbox_xywh": seg_bbox_xywh,
        "motherboard_reference_mask": reference_mask,
        "motherboard_reference_3d_bbox": _copy_motherboard_obb(reference_obb),
        "motherboard_debug_3d_bbox": _copy_motherboard_obb(debug_obb),
        "motherboard_registration": registration,
        "socket_hover_x_offset_m": float(x_offset_m),
        "socket_hover_x_from_right_edge_m": float(x_from_right_edge_m),
        "socket_hover_center_offset_world": np.asarray(
            center_offset_world,
            dtype=np.float64,
        ).reshape(3),
        "socket_hover_localization_mode": str(localization_mode),
        "socket_hover_edge_offset_m": float(edge_offset_m),
        "socket_hover_side_sign": float(side_sign),
        "socket_hover_target_socket_number": int(target_socket_number),
        "motherboard_right_edge_anchor_px": (
            None
            if right_edge_anchor_px is None
            else np.asarray(right_edge_anchor_px, dtype=np.float64).reshape(2)
        ),
        "motherboard_right_edge_anchor_world": (
            None
            if right_edge_anchor_world is None
            else np.asarray(right_edge_anchor_world, dtype=np.float64).reshape(3)
        ),
        "motherboard_right_edge_anchor_from_center_offset_world": np.asarray(
            right_edge_anchor_from_center_offset_world,
            dtype=np.float64,
        ).reshape(3),
    }


def _set_scene_socket_hover_target(scene, hover_world, cam_K=None, T_cam_world=None):
    scene["socket_hover_world"] = np.asarray(hover_world, dtype=float).reshape(3)
    scene["socket_world"] = np.asarray(scene["socket_hover_world"], dtype=float).reshape(3)
    if cam_K is not None and T_cam_world is not None:
        scene["socket_hover_px"] = _project_world_to_pixel(
            scene["socket_hover_world"],
            cam_K,
            T_cam_world,
        )
    if scene.get("socket_record") is not None:
        if scene.get("socket_hover_px") is not None:
            scene["socket_record"]["center_px"] = np.asarray(
                scene["socket_hover_px"],
                dtype=float,
            ).reshape(2)
        scene["socket_record"]["world"] = np.asarray(
            scene["socket_hover_world"],
            dtype=float,
        ).reshape(3)


def _move_with_speed(side, pos, rpy, planning_speed):
    pos = [float(v) for v in pos]
    rpy = [float(v) for v in rpy]
    planning_speed = float(planning_speed)
    print(
        f"[gpu_handover] move {side}: pos={pos} rpy={rpy} "
        f"planning_speed={planning_speed:.2f}"
    )
    freespace_move(
        **{
            f"{side}_target_pos": pos,
            f"{side}_target_rpy": rpy,
        },
        planning_speed=planning_speed,
        ik_error_threshold=RUN_CONFIG["ik_error_threshold"],
        ik_xyz_weight=RUN_CONFIG["ik_xyz_weight"],
        ik_rpy_weight=RUN_CONFIG["ik_rpy_weight"],
        planner_backend=RUN_CONFIG["planner_backend"],
    )


def _cartesian_retract_up_after_release(
    side,
    lift_m,
    speed_mps=None,
    step_m=None,
):
    env = _tool_env_from_callable(get_robot_state)
    if env is None or not hasattr(env, "move_bimanual_joint_keypoints"):
        raise RuntimeError("direct YAM env not available for Cartesian retract")
    side = str(side).strip().lower()
    if side not in {"left", "right"}:
        raise RuntimeError(f"Cartesian retract only implemented for left/right, got {side!r}")
    other = "left" if side == "right" else "right"

    speed_mps = float(
        RIGHT_POST_RELEASE_CARTESIAN_SPEED_MPS if speed_mps is None else speed_mps
    )
    step_m = float(RIGHT_POST_RELEASE_CARTESIAN_STEP_M if step_m is None else step_m)
    lift_m = float(lift_m)
    if lift_m <= 1e-6:
        return

    obs_left = env.get_observations("left")
    obs_right = env.get_observations("right")
    left_jp = np.asarray(obs_left["joint_pos"], dtype=np.float64).reshape(6)
    right_jp = np.asarray(obs_right["joint_pos"], dtype=np.float64).reshape(6)
    left_gp = float(np.asarray(obs_left["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    right_gp = float(np.asarray(obs_right["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    left_pos = np.asarray(obs_left["ee_pos"], dtype=np.float64).reshape(3)
    left_quat = np.asarray(obs_left["ee_quat"], dtype=np.float64).reshape(4)
    right_pos = np.asarray(obs_right["ee_pos"], dtype=np.float64).reshape(3)
    right_quat = np.asarray(obs_right["ee_quat"], dtype=np.float64).reshape(4)
    move_start_pos = left_pos if side == "left" else right_pos

    num_segments = max(2, int(np.ceil(lift_m / max(step_m, 1e-4))))
    left_waypoints = []
    right_waypoints = []
    left_grippers = []
    right_grippers = []
    timestamps = []

    with env._kin_lock:
        env.kin.forward_kinematics(left_jp, right_jp)
        cur_left_jp = left_jp.copy()
        cur_right_jp = right_jp.copy()
        for waypoint_index in range(1, num_segments + 1):
            alpha = float(waypoint_index) / float(num_segments)
            target_move_pos = move_start_pos + np.array([0.0, 0.0, alpha * lift_m])
            env.kin.forward_kinematics(cur_left_jp, cur_right_jp)
            if side == "left":
                next_left_jp, next_right_jp = env.kin.inverse_kinematics(
                    target_move_pos,
                    left_quat,
                    right_pos,
                    right_quat,
                    seeded=True,
                    dt=0.01,
                    solver="daqp",
                    damping=1e-3,
                    err_threshold=1e-4,
                    max_iters=40,
                )
            else:
                next_left_jp, next_right_jp = env.kin.inverse_kinematics(
                    left_pos,
                    left_quat,
                    target_move_pos,
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
            timestamps.append(alpha * lift_m / max(speed_mps, 1e-3))

    print(
        "[gpu_handover] Cartesian retract: "
        f"side={side} lift_m={lift_m:.4f} speed_mps={speed_mps:.3f} "
        f"waypoints={len(timestamps)}"
    )
    result = env.move_bimanual_joint_keypoints(
        timestamps=timestamps,
        left_joint_positions=left_waypoints,
        right_joint_positions=right_waypoints,
        left_gripper_positions=left_grippers,
        right_gripper_positions=right_grippers,
        playback_speed=1.0,
        command_hz=60.0,
        start_interp_s=0.2,
    )
    if not bool(result.get("success", False)):
        print(
            "[gpu_handover] Cartesian retract failed: "
            f"side={side} reason={result.get('reason', 'unknown')}"
        )
        raise RuntimeError(result.get("reason", "Cartesian retract failed"))


def _guided_side_pose_move(side, target_pos, target_rpy, duration_s=None, num_steps=None):
    env = _tool_env_from_callable(get_robot_state)
    if env is None or not hasattr(env, "move_bimanual_joint_keypoints"):
        raise RuntimeError("direct YAM env not available for guided pose move")
    side = str(side).strip().lower()
    if side not in {"left", "right"}:
        raise RuntimeError(f"guided pose move only implemented for left/right, got {side!r}")

    duration_s = float(
        POST_HANDOVER_REORIENT_DURATION_S if duration_s is None else duration_s
    )
    num_steps = int(POST_HANDOVER_REORIENT_STEPS if num_steps is None else num_steps)
    num_steps = max(2, num_steps)
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _display_rpy_to_rotation(target_rpy).as_quat().astype(np.float64)

    obs_left = env.get_observations("left")
    obs_right = env.get_observations("right")
    left_jp = np.asarray(obs_left["joint_pos"], dtype=np.float64).reshape(6)
    right_jp = np.asarray(obs_right["joint_pos"], dtype=np.float64).reshape(6)
    left_gp = float(np.asarray(obs_left["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    right_gp = float(np.asarray(obs_right["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    left_start_pos = np.asarray(obs_left["ee_pos"], dtype=np.float64).reshape(3)
    left_start_quat = np.asarray(obs_left["ee_quat"], dtype=np.float64).reshape(4)
    right_start_pos = np.asarray(obs_right["ee_pos"], dtype=np.float64).reshape(3)
    right_start_quat = np.asarray(obs_right["ee_quat"], dtype=np.float64).reshape(4)

    if side == "left":
        move_start_pos = left_start_pos
        move_start_quat = left_start_quat
        hold_pos = right_start_pos
        hold_quat = right_start_quat
    else:
        move_start_pos = right_start_pos
        move_start_quat = right_start_quat
        hold_pos = left_start_pos
        hold_quat = left_start_quat

    start_rot = Rotation.from_quat(move_start_quat)
    target_rot = Rotation.from_quat(target_quat)
    delta_rot = target_rot * start_rot.inv()

    left_waypoints = []
    right_waypoints = []
    left_grippers = []
    right_grippers = []
    timestamps = []

    with env._kin_lock:
        env.kin.forward_kinematics(left_jp, right_jp)
        cur_left_jp = left_jp.copy()
        cur_right_jp = right_jp.copy()
        for step_index in range(1, num_steps + 1):
            alpha = float(step_index) / float(num_steps)
            interp_pos = move_start_pos + alpha * (target_pos - move_start_pos)
            interp_rot = Rotation.from_rotvec(delta_rot.as_rotvec() * alpha) * start_rot
            interp_quat = interp_rot.as_quat().astype(np.float64)
            env.kin.forward_kinematics(cur_left_jp, cur_right_jp)
            if side == "left":
                next_left_jp, next_right_jp = env.kin.inverse_kinematics(
                    interp_pos,
                    interp_quat,
                    hold_pos,
                    hold_quat,
                    seeded=True,
                    dt=0.01,
                    solver="daqp",
                    damping=1e-3,
                    err_threshold=1e-4,
                    max_iters=40,
                )
            else:
                next_left_jp, next_right_jp = env.kin.inverse_kinematics(
                    hold_pos,
                    hold_quat,
                    interp_pos,
                    interp_quat,
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
            timestamps.append(alpha * duration_s)

    print(
        "[gpu_handover] Guided pose move: "
        f"side={side} target_pos={[round(float(v), 4) for v in target_pos.tolist()]} "
        f"target_rpy={[round(float(v), 1) for v in target_rpy]} "
        f"steps={num_steps} duration_s={duration_s:.2f}"
    )
    result = env.move_bimanual_joint_keypoints(
        timestamps=timestamps,
        left_joint_positions=left_waypoints,
        right_joint_positions=right_waypoints,
        left_gripper_positions=left_grippers,
        right_gripper_positions=right_grippers,
        playback_speed=1.0,
        command_hz=60.0,
        start_interp_s=0.2,
    )
    if not bool(result.get("success", False)):
        print(
            "[gpu_handover] Guided pose move failed: "
            f"side={side} reason={result.get('reason', 'unknown')}"
        )
        raise RuntimeError(result.get("reason", "guided pose move failed"))


def _move_side_to_joint_home(side, close_gripper_after=False):
    env = _tool_env_from_callable(get_robot_state)
    if env is None or not hasattr(env, "move_bimanual_joint_keypoints"):
        raise RuntimeError("direct YAM env not available for joint-home move")
    side = str(side).strip().lower()
    if side not in {"left", "right"}:
        raise RuntimeError(f"invalid side for joint-home move: {side!r}")
    other = "left" if side == "right" else "right"

    obs_side = env.get_observations(side)
    obs_other = env.get_observations(other)
    home_arm = env._profile.arms[side]
    side_start_jp = np.asarray(obs_side["joint_pos"], dtype=np.float64).reshape(6)
    side_start_gp = float(np.asarray(obs_side["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    other_jp = np.asarray(obs_other["joint_pos"], dtype=np.float64).reshape(6)
    other_gp = float(np.asarray(obs_other["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    side_home_jp = np.asarray(home_arm.home_joint_pos, dtype=np.float64).reshape(6)
    side_home_gp = float(
        np.asarray(home_arm.home_gripper_pos, dtype=np.float64).reshape(-1)[0]
    )
    target_gp = side_home_gp if close_gripper_after else side_start_gp
    max_joint_delta = float(np.max(np.abs(side_home_jp - side_start_jp)))
    duration_s = max(1.0, min(3.0, 1.2 + 1.4 * max_joint_delta))
    timestamps = [0.0, duration_s]

    if side == "right":
        left_joint_positions = [other_jp.tolist(), other_jp.tolist()]
        right_joint_positions = [side_start_jp.tolist(), side_home_jp.tolist()]
        left_gripper_positions = [[other_gp], [other_gp]]
        right_gripper_positions = [[side_start_gp], [target_gp]]
    else:
        left_joint_positions = [side_start_jp.tolist(), side_home_jp.tolist()]
        right_joint_positions = [other_jp.tolist(), other_jp.tolist()]
        left_gripper_positions = [[side_start_gp], [target_gp]]
        right_gripper_positions = [[other_gp], [other_gp]]

    print(
        "[gpu_handover] Joint-home move: "
        f"side={side} duration_s={duration_s:.2f} close_gripper_after={close_gripper_after}"
    )
    result = env.move_bimanual_joint_keypoints(
        timestamps=timestamps,
        left_joint_positions=left_joint_positions,
        right_joint_positions=right_joint_positions,
        left_gripper_positions=left_gripper_positions,
        right_gripper_positions=right_gripper_positions,
        playback_speed=1.0,
        command_hz=60.0,
        start_interp_s=0.2,
    )
    if not bool(result.get("success", False)):
        print(
            "[gpu_handover] Joint-home move failed: "
            f"side={side} reason={result.get('reason', 'unknown')}"
        )
        raise RuntimeError(result.get("reason", "joint-home move failed"))


def _vertical_insertion_rpy_from_grasp_axes(jaw_axis_world, approach_world, axis_kind):
    jaw_axis = _unit(jaw_axis_world)
    approach = _unit(approach_world)
    tangent = _unit(np.cross(approach, jaw_axis))
    axis_kind = str(axis_kind).strip().lower()

    if axis_kind == "short":
        # Short-axis approach means the gripper is approaching a long-edge side face.
        # Making the short axis vertical places one long edge at the bottom.
        final_z = WORLD_DOWN
        final_y = _xy_unit(tangent, fallback=_perp_xy(approach))
    elif axis_kind == "long":
        # Long-axis approach means the gripper is approaching a short-edge side face.
        # Making the short axis vertical again places one long edge at the bottom.
        final_y = WORLD_DOWN
        final_z = _xy_unit(approach, fallback=WORLD_LEFT)
    else:
        raise RuntimeError(f"unknown insertion axis kind: {axis_kind!r}")

    final_x = np.cross(final_y, final_z)
    final_x_norm = float(np.linalg.norm(final_x))
    if final_x_norm < 1e-9:
        raise RuntimeError(
            "degenerate insertion axes: "
            f"axis_kind={axis_kind!r} final_y={final_y.tolist()} final_z={final_z.tolist()}"
        )
    final_x = final_x / final_x_norm
    final_y = np.cross(final_z, final_x)
    final_y = final_y / max(float(np.linalg.norm(final_y)), 1e-9)
    rot = Rotation.from_matrix(np.column_stack([final_x, final_y, final_z]))
    return _rotation_to_display_rpy(rot)

def _sample_3d_bb_candidates_with_queries(queries, camera="top", tcp_offset_z_m=0.0):
    last_error = None
    records = []
    for query in queries:
        try:
            grasps = sample_grasp_pose_3d_bb(
                object_name=query,
                camera=camera,
                tcp_offset_z_m=tcp_offset_z_m,
            )
        except Exception as exc:
            last_error = exc
            print(f"[gpu_handover] 3D-BB query failed for {query!r}: {exc}")
            continue
        if grasps:
            print(
                f"[gpu_handover] 3D-BB query {query!r} returned {len(grasps)} candidates"
            )
            for grasp in grasps:
                records.append({"query": query, "grasp": grasp})
    if records:
        return records
    if last_error is not None:
        raise RuntimeError(
            f"all 3D-BB queries failed for {queries!r}; last_error={last_error}"
        )
    raise RuntimeError(f"no 3D-BB candidates for {queries!r}")


def _bbox_semantic_extents(bbox):
    if bbox is None:
        return 0.0, 0.0, 0.0
    return (
        float(getattr(bbox, "top_normal_extent", 0.0) or 0.0),
        float(getattr(bbox, "top_face_short_extent", 0.0) or 0.0),
        float(getattr(bbox, "top_face_long_extent", 0.0) or 0.0),
    )


def _initial_gpu_pick_bbox_rejection_reasons(
    bbox,
    *,
    max_top_normal_extent_m=None,
    max_short_extent_m=None,
    max_long_extent_m=None,
):
    max_top_normal_extent_m = float(
        PICK_BBOX_MAX_TOP_NORMAL_EXTENT_M
        if max_top_normal_extent_m is None
        else max_top_normal_extent_m
    )
    max_short_extent_m = float(
        PICK_BBOX_MAX_SHORT_EXTENT_M
        if max_short_extent_m is None
        else max_short_extent_m
    )
    max_long_extent_m = float(
        PICK_BBOX_MAX_LONG_EXTENT_M
        if max_long_extent_m is None
        else max_long_extent_m
    )
    top_normal_extent, short_extent, long_extent = _bbox_semantic_extents(bbox)
    reasons = []
    if bbox is None:
        reasons.append("missing bbox_result")
        return reasons
    if top_normal_extent <= 0.0 or short_extent <= 0.0 or long_extent <= 0.0:
        reasons.append(
            "invalid_extents="
            f"{[round(top_normal_extent, 4), round(short_extent, 4), round(long_extent, 4)]}"
        )
    if top_normal_extent > max_top_normal_extent_m:
        reasons.append(
            "top_normal_extent="
            f"{top_normal_extent:.3f}>{max_top_normal_extent_m:.3f}"
        )
    if short_extent > max_short_extent_m:
        reasons.append(
            f"short_extent={short_extent:.3f}>{max_short_extent_m:.3f}"
        )
    if long_extent > max_long_extent_m:
        reasons.append(
            f"long_extent={long_extent:.3f}>{max_long_extent_m:.3f}"
        )
    return reasons


def _filter_initial_gpu_pick_candidates_with_limits(
    grasps,
    query=None,
    object_name=None,
    camera=None,
    *,
    log_prefix="Initial GPU-pick",
    max_top_normal_extent_m=None,
    max_adjusted_pick_z=None,
):
    grasps = list(grasps or [])
    kept = []
    rejected = 0
    query_label = repr(query) if query is not None else repr(object_name or "gpu")
    for grasp in grasps:
        reasons = _initial_gpu_pick_bbox_rejection_reasons(
            getattr(grasp, "bbox_result", None),
            max_top_normal_extent_m=max_top_normal_extent_m,
        )
        if max_adjusted_pick_z is not None:
            try:
                pos = np.asarray(getattr(grasp, "position", []), dtype=float).reshape(-1)
                candidate_z = float(pos[2]) + float(RIGHT_PICK_EXTRA_Z_OFFSET_M)
                bbox = getattr(grasp, "bbox_result", None)
                top_surface_z = None if bbox is None else getattr(bbox, "top_surface_z", None)
                if top_surface_z is not None:
                    candidate_z = float(top_surface_z) + max(
                        0.0,
                        float(RIGHT_PICK_ABOVE_TOP_SURFACE_M),
                    )
                if candidate_z > float(max_adjusted_pick_z):
                    if PICK_CLAMP_HIGH_Z_TO_TABLE:
                        print(
                            f"[gpu_handover] {log_prefix} candidate high-Z will be clamped: "
                            f"query={query_label} "
                            f"adjusted_pick_z={candidate_z:.3f} "
                            f"max_adjusted_pick_z={float(max_adjusted_pick_z):.3f}"
                        )
                    else:
                        reasons.append(
                            f"adjusted_pick_z={candidate_z:.3f}>{float(max_adjusted_pick_z):.3f}"
                        )
            except Exception as exc:
                reasons.append(f"invalid_adjusted_pick_z={exc}")
        if reasons:
            rejected += 1
            print(
                f"[gpu_handover] {log_prefix} candidate rejected: "
                f"query={query_label} reasons={'; '.join(reasons)}"
            )
            continue
        kept.append(grasp)
    print(
        f"[gpu_handover] {log_prefix} candidate filter: "
        f"query={query_label} kept={len(kept)}/{len(grasps)} "
        f"rejected={rejected}"
    )
    return kept


def _filter_initial_gpu_pick_candidates(grasps, query=None, object_name=None, camera=None):
    return _filter_initial_gpu_pick_candidates_with_limits(
        grasps,
        query=query,
        object_name=object_name,
        camera=camera,
    )


def _filter_initial_gpu_pick_roi_candidates(grasps, query=None, object_name=None, camera=None):
    return _filter_initial_gpu_pick_candidates_with_limits(
        grasps,
        query=query,
        object_name=object_name,
        camera=camera,
        log_prefix="Initial GPU-pick ROI",
        max_top_normal_extent_m=PICK_ROI_BBOX_MAX_TOP_NORMAL_EXTENT_M,
    )


def _parse_socket_bbox(raw, width, height):
    txt = str(raw).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", txt, re.DOTALL)
    if fence:
        txt = fence.group(1).strip()
    match = re.search(r"(\{.*\}|\[.*\])", txt, re.DOTALL)
    if match:
        txt = match.group(1)
    payload = json.loads(txt)
    source = "bbox"
    if isinstance(payload, list):
        if len(payload) == 2 and isinstance(payload[0], str) and isinstance(payload[1], list):
            payload = {"bbox": payload[1]}
        elif len(payload) == 4:
            payload = {"bbox": payload}
        else:
            raise RuntimeError(
                f"socket detector returned invalid list payload: {payload!r}"
            )
    if not isinstance(payload, dict):
        raise RuntimeError(f"socket detector returned invalid payload: {payload!r}")
    if "bbox" in payload:
        box = payload["bbox"]
        source = "bbox"
    elif "box" in payload:
        box = payload["box"]
        source = "box"
    else:
        box = payload.get("box_2d") or []
        source = "box_2d"
    if len(box) != 4:
        raise RuntimeError(f"socket detector returned invalid bbox payload: {payload!r}")
    coords = [float(v) for v in box]
    if source == "box_2d":
        ymin, xmin, ymax, xmax = coords
        if max(coords) > max(width, height):
            scale_x = float(width) / 1000.0
            scale_y = float(height) / 1000.0
            x0, y0, x1, y1 = (
                xmin * scale_x,
                ymin * scale_y,
                xmax * scale_x,
                ymax * scale_y,
            )
        else:
            x0, y0, x1, y1 = xmin, ymin, xmax, ymax
    else:
        x0, y0, x1, y1 = coords
    x0 = max(0, min(width - 1, x0))
    x1 = max(0, min(width - 1, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(0, min(height - 1, y1))
    x0, y0, x1, y1 = [int(round(v)) for v in (x0, y0, x1, y1)]
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"socket detector returned degenerate bbox: {[x0, y0, x1, y1]}")
    return [x0, y0, x1, y1]


def _expand_bbox_xyxy(bbox, width, height, margin_px):
    x0, y0, x1, y1 = [int(v) for v in bbox]
    margin_px = max(0, int(margin_px))
    return [
        max(0, x0 - margin_px),
        max(0, y0 - margin_px),
        min(int(width), x1 + margin_px),
        min(int(height), y1 + margin_px),
    ]


def _detect_gpu_bbox_2d(camera="top", image=None):
    rgb = _as_uint8_rgb(get_camera_image(camera=camera) if image is None else image)
    if rgb is None:
        raise RuntimeError(f"could not capture {camera!r} RGB image for GPU bbox")
    height, width = rgb.shape[:2]
    prompt = (
        f"You are given a top-down {width}x{height} image. Find the single discrete GPU / "
        "graphics card device lying on the table. The GPU may be black, may have one or more "
        "cooling fans, and may not have any visible NVIDIA text. Return the tight 2D bounding box "
        "for only that small GPU card device. Ignore the large motherboard, CPU socket, RAM slots, "
        "table, hands, and grippers. Respond with ONLY JSON in the form "
        '{"label":"graphics_card","bbox":[x_min,y_min,x_max,y_max]} using integer pixel coordinates.'
    )
    raw = vlm_query(
        text=prompt,
        image=rgb,
        backend=PICK_VLM_BACKEND,
        model=PICK_VLM_MODEL,
        mode="json",
        temperature=0.0,
    )
    bbox = _parse_socket_bbox(raw, width, height)
    print(
        "[gpu_handover] Detected GPU 2D bbox from VLM: "
        f"bbox={bbox} backend={PICK_VLM_BACKEND!r} model={PICK_VLM_MODEL!r}"
    )
    return bbox, width, height


def _motherboard_right_gpu_pick_roi(scene_targets, camera="top"):
    if not PICK_MOTHERBOARD_RIGHT_ROI_FALLBACK:
        return None
    if scene_targets is None:
        return None
    motherboard_bbox = scene_targets.get("motherboard_bbox")
    if motherboard_bbox is None or len(motherboard_bbox) != 4:
        return None
    rgb = _as_uint8_rgb(get_camera_image(camera=camera))
    if rgb is None:
        raise RuntimeError(f"could not capture {camera!r} RGB image for GPU pick ROI")
    height, width = rgb.shape[:2]
    mb_x0, mb_y0, mb_x1, mb_y1 = [int(round(float(v))) for v in motherboard_bbox]
    mb_w = max(1, mb_x1 - mb_x0)
    roi_x0 = max(0, min(width - 2, mb_x1 + int(PICK_RIGHT_ROI_X_GAP_PX)))
    roi_width = max(
        int(PICK_RIGHT_ROI_MIN_WIDTH_PX),
        int(round(float(PICK_RIGHT_ROI_WIDTH_SCALE) * float(mb_w))),
    )
    roi_x1 = max(roi_x0 + 2, min(width, roi_x0 + roi_width))
    roi_y0 = max(0, mb_y0 - int(PICK_RIGHT_ROI_TOP_PAD_PX))
    roi_y1 = min(height, mb_y1 + int(PICK_RIGHT_ROI_BOTTOM_PAD_PX))
    if roi_y1 <= roi_y0 + 2 or roi_x1 <= roi_x0 + 2:
        raise RuntimeError(
            "degenerate motherboard-right GPU pick ROI: "
            f"motherboard_bbox={motherboard_bbox} roi={[roi_x0, roi_y0, roi_x1, roi_y1]} "
            f"image_shape={(height, width)}"
        )
    roi = [int(roi_x0), int(roi_y0), int(roi_x1), int(roi_y1)]
    crop = rgb[roi_y0:roi_y1, roi_x0:roi_x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    dark_mask = (gray < int(PICK_RIGHT_ROI_DARK_THRESHOLD)).astype(np.uint8)
    num_labels, _labels, stats, cents = cv2.connectedComponentsWithStats(
        dark_mask,
        8,
    )
    components = []
    rejected_components = []
    for idx in range(1, int(num_labels)):
        lx, ly, lw, lh, area = [int(v) for v in stats[idx]]
        image_x0 = int(roi_x0 + lx)
        image_x1 = int(roi_x0 + lx + lw)
        image_y0 = int(roi_y0 + ly)
        image_y1 = int(roi_y0 + ly + lh)
        reject_reasons = []
        if area < int(PICK_RIGHT_ROI_MIN_COMPONENT_AREA_PX):
            continue
        if lw < 10 or lh < 10:
            continue
        if (
            ly <= int(PICK_RIGHT_ROI_TOP_BORDER_REJECT_PX)
            and image_y1 < mb_y0
        ):
            reject_reasons.append("top_border_background")
        if image_y1 < mb_y0 + int(PICK_RIGHT_ROI_MIN_BOTTOM_REL_MB_TOP_PX):
            reject_reasons.append(
                f"above_table_region(y1={image_y1}<mb_y0{int(PICK_RIGHT_ROI_MIN_BOTTOM_REL_MB_TOP_PX):+d})"
            )
        if reject_reasons:
            rejected_components.append(
                (
                    area,
                    lx,
                    ly,
                    lw,
                    lh,
                    [float(v) for v in cents[idx]],
                    reject_reasons,
                )
            )
            continue
        components.append((area, lx, ly, lw, lh, [float(v) for v in cents[idx]]))
    components.sort(reverse=True)
    if not components:
        raise RuntimeError(
            "no dark loose-GPU component found in motherboard-right ROI: "
            f"motherboard_bbox={motherboard_bbox} roi={roi} "
            f"dark_threshold={int(PICK_RIGHT_ROI_DARK_THRESHOLD)} "
            f"rejected_components={rejected_components[:4]}"
        )
    _area, lx, ly, lw, lh, _cent = components[0]
    pad = int(PICK_RIGHT_ROI_TIGHT_PAD_PX)
    tight = [
        max(0, int(roi_x0 + lx - pad)),
        max(0, int(roi_y0 + ly - pad)),
        min(width, int(roi_x0 + lx + lw + pad)),
        min(height, int(roi_y0 + ly + lh + pad)),
    ]
    print(
        "[gpu_handover] Motherboard-right loose-GPU pick ROI: "
        f"motherboard_bbox={[int(v) for v in motherboard_bbox]} "
        f"broad_roi={roi} tight_bbox={tight} "
        f"components={components[:4]} image_shape={(height, width)} "
        f"rejected_components={rejected_components[:3]} "
        f"queries={GPU_PICK_RIGHT_ROI_QUERIES}"
    )
    return tight


def _parse_pcie_slot_center_list(raw, width, height):
    txt = str(raw).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", txt, re.DOTALL)
    if fence:
        txt = fence.group(1).strip()
    match = re.search(r"(\{.*\}|\[.*\])", txt, re.DOTALL)
    if match:
        txt = match.group(1)
    payload = json.loads(txt)
    if not isinstance(payload, list):
        raise RuntimeError(f"slot center detector returned invalid payload: {payload!r}")
    centers = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "x" in item and "y" in item:
            x = item["x"]
            y = item["y"]
        else:
            point = item.get("point")
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        x_i = int(round(float(x)))
        y_i = int(round(float(y)))
        if 0 <= x_i < width and 0 <= y_i < height:
            centers.append({"x": x_i, "y": y_i})
    if not centers:
        raise RuntimeError(f"slot center detector returned no valid centers: {payload!r}")
    return centers


def _select_slot_center_triplet(centers):
    ordered = sorted(
        (
            {"x": int(center["x"]), "y": int(center["y"])}
            for center in centers
            if "x" in center and "y" in center
        ),
        key=lambda center: (int(center["x"]), int(center["y"])),
    )
    if len(ordered) < 3:
        raise RuntimeError(f"need at least 3 PCIe slot centers, got {ordered!r}")
    shortlist = []
    for start in range(len(ordered) - 2):
        triplet = ordered[start : start + 3]
        xs = [int(center["x"]) for center in triplet]
        ys = [int(center["y"]) for center in triplet]
        x_spread = int(max(xs) - min(xs))
        y_spread = int(max(ys) - min(ys))
        if x_spread < PCIE_SLOT_CENTER_MIN_X_SPREAD_PX:
            continue
        if y_spread > PCIE_SLOT_CENTER_MAX_Y_SPREAD_PX:
            continue
        shortlist.append(
            {
                "centers": triplet,
                "x_spread": x_spread,
                "y_spread": y_spread,
                "score": float(x_spread) - 2.0 * float(y_spread),
            }
        )
    if not shortlist:
        raise RuntimeError(
            "no left-to-right PCIe slot-center triplet passed validation: "
            f"centers={ordered!r}"
        )
    shortlist.sort(
        key=lambda item: (
            -float(item["score"]),
            int(item["y_spread"]),
            int(item["centers"][0]["x"]),
        )
    )
    selected = shortlist[0]
    print(
        "[gpu_handover] PCIe slot-center candidates: "
        + "; ".join(
            f"triplet={item['centers']} x_spread={item['x_spread']} y_spread={item['y_spread']}"
            for item in shortlist[:5]
        )
    )
    return selected["centers"]


def _detect_pcie_slot_centers(camera="top", image=None):
    rgb = _as_uint8_rgb(image if image is not None else get_camera_image(camera=camera))
    if rgb is None:
        raise RuntimeError(f"could not capture {camera!r} RGB image for PCIe slot centers")
    height, width = rgb.shape[:2]
    prompt = (
        "Find the centers of the visible full-length PCIe expansion slots that a GPU "
        "could plug into. In this image, the target slots are three long vertical "
        "socket bodies arranged LEFT-TO-RIGHT across the motherboard, so their "
        "centers should have similar y values and different x values. Do not return "
        "a top-to-bottom stack. Return ONLY JSON array of objects like "
        '{"label":"pcie_slot_center","x":123,"y":456}. '
        f"Use {width}x{height} image pixel coordinates."
    )
    raw = vlm_query(
        text=prompt,
        backend=SOCKET_VLM_BACKEND,
        model=SOCKET_VLM_MODEL,
        image=rgb,
    )
    centers = _parse_pcie_slot_center_list(raw, width, height)
    triplet = _select_slot_center_triplet(centers)
    print(
        "[gpu_handover] Detected PCIe slot centers: "
        f"all={centers} selected={triplet}"
    )
    return {
        "image_shape": [height, width],
        "all_centers": centers,
        "selected_centers": triplet,
        "leftmost_center_px": np.asarray(
            [float(triplet[0]["x"]), float(triplet[0]["y"])],
            dtype=np.float64,
        ),
        "raw": raw,
    }


def _detect_motherboard_bbox_2d(camera="top", image=None, slot_centers=None):
    rgb = _as_uint8_rgb(image if image is not None else get_camera_image(camera=camera))
    if rgb is None:
        raise RuntimeError(f"could not capture {camera!r} RGB image for motherboard bbox")
    height, width = rgb.shape[:2]
    if not slot_centers:
        raise RuntimeError("motherboard bbox detection requires validated PCIe slot centers")
    prompt = (
        f"You are given a top-down {width}x{height} image. The motherboard contains "
        f"three PCIe slot centers near {slot_centers}. Return the tight 2D bounding box "
        "for the single motherboard that contains those three slot centers. Exclude the "
        "separate loose GPU on the right, robot arms, and empty table. Respond ONLY JSON "
        'in the form {"label":"motherboard","bbox":[x_min,y_min,x_max,y_max]} using '
        f"pixel coordinates in the {width}x{height} image."
    )
    raw = vlm_query(
        text=prompt,
        backend=SOCKET_VLM_BACKEND,
        model=SOCKET_VLM_MODEL,
        image=rgb,
    )
    bbox = _parse_socket_bbox(raw, width, height)
    center_px = np.asarray(
        [
            0.5 * float(bbox[0] + bbox[2]),
            0.5 * float(bbox[1] + bbox[3]),
        ],
        dtype=np.float64,
    )
    print(
        "[gpu_handover] Detected motherboard 2D bbox: "
        f"bbox={bbox} center_px={[round(float(v), 1) for v in center_px.tolist()]}"
    )
    return {
        "bbox": bbox,
        "center_px": center_px,
        "raw": raw,
    }


def _capture_initial_scene_targets(camera="top", artifact_prefix="initial"):
    rgb = _as_uint8_rgb(get_camera_image(camera=camera))
    if rgb is None:
        raise RuntimeError(f"could not capture {camera!r} RGB image for initial scene cache")
    depth = render_depth(camera=camera)
    if depth is None:
        raise RuntimeError(f"could not capture {camera!r} depth image for initial scene cache")
    cam_K = _camera_matrix(camera)
    T_cam_world = _camera_T_cam_world(camera)
    obb_record = _compute_initial_motherboard_3d_bbox(
        camera=camera,
        depth=depth,
        cam_K=cam_K,
        T_cam_world=T_cam_world,
    )
    obb = obb_record["obb"]
    seg_record = obb_record["seg_record"]
    seg_bbox_xywh = seg_record.get("bbox_xywh")
    seg_center_px = _bbox_center_px_from_xywh(seg_bbox_xywh)
    if seg_center_px is None:
        raise RuntimeError("motherboard segmentation bbox center is unavailable")
    motherboard_top_z = float(obb["z_min"]) + float(SOCKET_HOVER_BOARD_PLANE_Z_OFFSET_M)
    center_world = _project_pixel_to_plane_world(
        seg_center_px,
        cam_K,
        T_cam_world,
        motherboard_top_z,
    )
    if center_world is None:
        raise RuntimeError("could not project motherboard segmentation center to fixed board plane")
    reference_obb = _override_obb_center_xy(obb, center_world)
    reference_obb = _copy_motherboard_obb(reference_obb)
    right_edge_anchor_px = _mask_right_edge_anchor_px(
        seg_record.get("mask"),
        camera=camera,
    )
    if right_edge_anchor_px is None:
        raise RuntimeError("motherboard right-edge anchor pixel is unavailable")
    right_edge_anchor_world = _project_right_edge_anchor_world(
        right_edge_anchor_px,
        cam_K,
        T_cam_world,
        motherboard_top_z,
    )
    if right_edge_anchor_world is None:
        raise RuntimeError("could not project motherboard right-edge anchor to fixed board plane")
    right_edge_half_extent_m = _right_edge_half_extent_world_y(reference_obb)
    side_sign = _motherboard_right_edge_sign(
        reference_obb["center_world"],
        WORLD_LEFT,
        right_edge_half_extent_m,
        cam_K,
        T_cam_world,
    )
    edge_offset_m = _target_socket_right_edge_offset_m(int(TARGET_SOCKET_NUMBER))
    scene = _build_socket_hover_scene_record(
        top_z=motherboard_top_z,
        hover_z=max(
            motherboard_top_z + float(SOCKET_HOVER_EFFECTIVE_CLEARANCE_M),
            float(SOCKET_HOVER_EFFECTIVE_MIN_Z),
        ),
        center_px=seg_center_px,
        center_world=center_world,
        seg_center_px=seg_center_px,
        seg_bbox_xywh=seg_bbox_xywh,
        reference_mask=np.asarray(seg_record["mask"], dtype=bool),
        reference_obb=reference_obb,
        debug_obb=reference_obb,
        registration=None,
        x_offset_m=float(SOCKET_HOVER_X_OFFSET_M),
        x_from_right_edge_m=0.0,
        edge_offset_m=float(edge_offset_m),
        side_sign=float(side_sign),
        target_socket_number=int(TARGET_SOCKET_NUMBER),
        right_edge_anchor_px=right_edge_anchor_px,
        right_edge_anchor_world=right_edge_anchor_world,
        center_offset_world=np.zeros(3, dtype=np.float64),
        right_edge_anchor_from_center_offset_world=(
            np.asarray(right_edge_anchor_world, dtype=np.float64).reshape(3)
            - np.asarray(center_world, dtype=np.float64).reshape(3)
        ),
        localization_mode="edge",
    )
    scene["slot_record"] = {
        "image_shape": list(rgb.shape[:2]),
        "all_centers": [],
        "selected_centers": [],
        "leftmost_center_px": None,
        "raw": None,
        "method": "motherboard_geometry_only",
    }
    initial_hover_world = np.asarray(center_world, dtype=np.float64).reshape(3).copy()
    initial_hover_world[0] += float(scene["socket_hover_x_offset_m"])
    initial_hover_world[1] = float(right_edge_anchor_world[1]) - float(
        scene["socket_hover_side_sign"]
    ) * float(scene["socket_hover_edge_offset_m"])
    initial_hover_world[2] = float(motherboard_top_z)
    scene["socket_hover_x_from_right_edge_m"] = float(
        initial_hover_world[0] - float(right_edge_anchor_world[0])
    )
    scene["socket_hover_center_offset_world"] = (
        np.asarray(initial_hover_world, dtype=np.float64).reshape(3)
        - np.asarray(center_world, dtype=np.float64).reshape(3)
    )
    scene["pose_source_camera"] = str(camera)
    _set_scene_socket_hover_target(
        scene,
        _derive_socket_hover_world_from_scene_geometry(scene, motherboard_top_z),
        cam_K,
        T_cam_world,
    )
    seeded_hover_world = np.asarray(scene["socket_hover_world"], dtype=np.float64).reshape(3).copy()
    seeded_right_edge_anchor_world = np.asarray(
        scene["motherboard_right_edge_anchor_world"],
        dtype=np.float64,
    ).reshape(3).copy()
    if SOCKET_HOVER_AUX_CAMERA is not None:
        try:
            aux_obs = _capture_socket_hover_aux_observation(
                SOCKET_HOVER_AUX_CAMERA,
                motherboard_top_z,
            )
            use_aux_initial_world = (
                bool(SOCKET_HOVER_AUX_CAMERA_PREFER_WORLD_POSE)
                and _is_left_third_view_camera(aux_obs["camera"])
                and aux_obs.get("center_world") is not None
                and bool(aux_obs.get("world_valid"))
            )
            if use_aux_initial_world:
                aux_center_world = np.asarray(
                    aux_obs["center_world"],
                    dtype=np.float64,
                ).reshape(3)
                primary_score = float(seg_record.get("score", 0.0) or 0.0)
                aux_pose_diff_m = float(
                    np.linalg.norm(
                        aux_center_world[:2]
                        - np.asarray(center_world, dtype=np.float64).reshape(3)[:2]
                    )
                )
                if (
                    float(SOCKET_HOVER_AUX_MAX_POSE_DIFF_M) > 0.0
                    and aux_pose_diff_m > float(SOCKET_HOVER_AUX_MAX_POSE_DIFF_M)
                    and primary_score >= float(SOCKET_HOVER_PRIMARY_MIN_SCORE)
                ):
                    print(
                        "[gpu_handover] Auxiliary camera "
                        f"{aux_obs['camera']!r} initial world pose rejected; "
                        "keeping primary motherboard pose: "
                        f"pose_diff_m={aux_pose_diff_m:.4f} "
                        f"max_pose_diff_m={float(SOCKET_HOVER_AUX_MAX_POSE_DIFF_M):.4f} "
                        f"primary_score={primary_score:.3f} "
                        f"aux_score={float(aux_obs['score']):.3f}"
                    )
                    use_aux_initial_world = False
            if use_aux_initial_world:
                aux_center_world = np.asarray(
                    aux_obs["center_world"],
                    dtype=np.float64,
                ).reshape(3)
                aux_right_edge_anchor_world = seeded_right_edge_anchor_world.copy()
                scene["motherboard_center_world"] = aux_center_world
                scene["motherboard_right_edge_anchor_world"] = aux_right_edge_anchor_world
                scene["motherboard_registration"] = {
                    "method": "aux_camera_initial_world_center",
                    "camera": aux_obs["camera"],
                    "score": float(aux_obs["score"]),
                    "dx_px": 0,
                    "dy_px": 0,
                }
                scene["motherboard_debug_3d_bbox"] = _copy_motherboard_obb(
                    _override_obb_center_xy(reference_obb, aux_center_world)
                )
                recentered_hover_offset_world = seeded_hover_world - aux_center_world
                recentered_hover_offset_world = np.asarray(
                    recentered_hover_offset_world,
                    dtype=np.float64,
                ).reshape(3)
                recentered_hover_offset_world[0] = float(scene["socket_hover_x_offset_m"])
                recentered_hover_offset_world[2] = 0.0
                scene["socket_hover_center_offset_world"] = recentered_hover_offset_world
                scene["motherboard_right_edge_anchor_from_center_offset_world"] = (
                    aux_right_edge_anchor_world - aux_center_world
                )
                scene["pose_source_camera"] = str(aux_obs["camera"])
                scene["socket_hover_localization_mode"] = "center"
                scene["aux_camera_name"] = aux_obs["camera"]
                scene["aux_camera_last_score"] = float(aux_obs["score"])
                scene["aux_camera_world_enabled"] = True
                _set_scene_socket_hover_target(
                    scene,
                    _derive_socket_hover_world_from_scene_geometry(scene, motherboard_top_z),
                    cam_K,
                    T_cam_world,
                )
            else:
                scene = _maybe_apply_socket_hover_aux_pose(
                    scene,
                    scene,
                    primary_score=float(seg_record.get("score", 0.0) or 0.0),
                    aux_obs=aux_obs,
                    base_scene=None,
                    cam_K=cam_K,
                    T_cam_world=T_cam_world,
                    stage_label="initial",
                )
        except Exception as exc:
            if SOCKET_HOVER_AUX_CAMERA_REQUIRED:
                raise RuntimeError(
                    f"required auxiliary camera {SOCKET_HOVER_AUX_CAMERA!r} is unavailable: {exc}"
                ) from exc
            print(
                "[gpu_handover] Auxiliary camera initial observation unavailable: "
                f"{exc}"
            )
    scene["socket_record"] = {
        "center_px": (
            np.asarray(scene["socket_hover_px"], dtype=float).reshape(2)
            if scene.get("socket_hover_px") is not None
            else np.asarray(seg_center_px, dtype=float).reshape(2)
        ),
        "all_centers": [],
        "selected_centers": [],
        "target_socket_number": int(TARGET_SOCKET_NUMBER),
        "target_socket_index": int(TARGET_SOCKET_NUMBER) - 1,
        "world": np.asarray(scene["socket_world"], dtype=float).reshape(3),
        "method": "motherboard_geometry_only",
    }
    x0 = int(seg_bbox_xywh[0])
    y0 = int(seg_bbox_xywh[1])
    x1 = int(seg_bbox_xywh[0] + seg_bbox_xywh[2])
    y1 = int(seg_bbox_xywh[1] + seg_bbox_xywh[3])
    motherboard_det = {
        "bbox": [x0, y0, x1, y1],
        "center_px": np.asarray(seg_center_px, dtype=float).reshape(2),
        "raw": None,
    }
    scene["motherboard_bbox"] = motherboard_det["bbox"]
    print(
        "[gpu_handover] Derived motherboard-aligned socket hover target: "
        f"world={[round(float(v), 4) for v in scene['socket_hover_world'].tolist()]} "
        f"target_socket={int(TARGET_SOCKET_NUMBER)} "
        f"edge_offset_m={float(scene['socket_hover_edge_offset_m']):.3f} "
        f"x_offset_m={float(scene['socket_hover_x_offset_m']):.4f} "
        f"x_from_right_edge_m={float(scene['socket_hover_x_from_right_edge_m']):.4f} "
        f"side_sign={float(scene['socket_hover_side_sign']):.0f} "
        f"method={scene['socket_record'].get('method', 'unknown')}"
    )
    scene["artifact_paths"] = _save_initial_scene_detection_artifacts(
        rgb,
        motherboard_det,
        scene["slot_record"],
        scene,
        artifact_prefix=artifact_prefix,
    )
    _save_initial_motherboard_3d_bbox_artifacts(
        rgb,
        camera,
        scene,
        artifact_prefix=artifact_prefix,
        obb_record=obb_record,
    )
    print(
        f"[gpu_handover] Cached {artifact_prefix} scene targets: "
        f"motherboard_bbox={scene['motherboard_bbox']} "
        f"motherboard_top_z={float(scene['motherboard_top_z']):.4f} "
        f"target_socket={int(TARGET_SOCKET_NUMBER)} "
        f"socket_world={[round(float(v), 4) for v in scene['socket_world'].tolist()]} "
        f"socket_hover_world="
        f"{None if scene.get('socket_hover_world') is None else [round(float(v), 4) for v in np.asarray(scene['socket_hover_world'], dtype=float).tolist()]} "
        f"socket_method={scene['socket_record'].get('method', 'unknown')}"
    )
    return scene


def _move_to_cached_socket_hover_pose(
    side,
    hover_rpy,
    scene_targets,
    axis_kind,
    *,
    fixed_plane_z=None,
    guided=False,
    guided_duration_s=None,
    guided_steps=None,
):
    socket_world = np.asarray(scene_targets["socket_hover_world"], dtype=float).reshape(3)
    current_pos = _robot_vec(get_robot_state(), side, "ee_pos")
    if fixed_plane_z is None:
        hover_z = float(
            scene_targets.get(
                "hover_z",
                max(
                    float(scene_targets["motherboard_top_z"]) + SOCKET_HOVER_EFFECTIVE_CLEARANCE_M,
                    SOCKET_HOVER_EFFECTIVE_MIN_Z,
                ),
            )
        )
    else:
        hover_z = float(fixed_plane_z)
    hover_pos = [
        float(socket_world[0]),
        float(socket_world[1]),
        float(hover_z),
    ]
    target_vec = np.asarray(hover_pos, dtype=np.float64)
    current_vec = np.asarray(current_pos, dtype=np.float64)
    move_delta_m = float(np.linalg.norm(target_vec[:2] - current_vec[:2]))
    z_delta_m = abs(float(target_vec[2] - current_vec[2]))
    if (
        move_delta_m <= float(SOCKET_HOVER_MIN_MOVE_M)
        and z_delta_m <= float(SOCKET_HOVER_Z_TOL_M)
    ):
        print(
            "[gpu_handover] Step 8a: skipping hover move because target is already close "
            f"target_pos={[round(float(v), 4) for v in hover_pos]} "
            f"current_pos={[round(float(v), 4) for v in current_pos]} "
            f"xy_delta_m={move_delta_m:.4f} "
            f"z_delta_m={z_delta_m:.4f} "
            f"threshold_m={float(SOCKET_HOVER_MIN_MOVE_M):.4f}"
        )
        scene_targets["socket_hover_rpy"] = [float(v) for v in hover_rpy]
        return [float(v) for v in current_pos]
    print(
        "[gpu_handover] Step 8a: cached hover over motherboard socket "
        f"target_pos={[round(float(v), 4) for v in hover_pos]} "
        f"target_rpy={[round(float(v), 1) for v in hover_rpy]} "
        f"guided={guided} "
        f"fixed_plane_z={None if fixed_plane_z is None else round(float(fixed_plane_z), 4)} "
        f"xy_delta_m={move_delta_m:.4f} "
        f"z_delta_m={z_delta_m:.4f}"
    )
    if guided:
        base_duration_s = float(
            SOCKET_HOVER_REACTIVE_GUIDED_DURATION_S
            if guided_duration_s is None
            else guided_duration_s
        )
        duration_s = float(base_duration_s)
        if float(SOCKET_HOVER_GUIDED_MAX_XY_SPEED_MPS) > 0.0:
            duration_s = max(duration_s, move_delta_m / float(SOCKET_HOVER_GUIDED_MAX_XY_SPEED_MPS))
        if float(SOCKET_HOVER_GUIDED_MAX_Z_SPEED_MPS) > 0.0:
            duration_s = max(duration_s, z_delta_m / float(SOCKET_HOVER_GUIDED_MAX_Z_SPEED_MPS))
        if duration_s > base_duration_s + 1e-6:
            print(
                "[gpu_handover] Guided hover speed clamp: "
                f"base_duration_s={base_duration_s:.2f} "
                f"clamped_duration_s={duration_s:.2f} "
                f"max_xy_speed_mps={float(SOCKET_HOVER_GUIDED_MAX_XY_SPEED_MPS):.3f} "
                f"max_z_speed_mps={float(SOCKET_HOVER_GUIDED_MAX_Z_SPEED_MPS):.3f}"
            )
        _guided_hover_plane_move(
            side,
            hover_pos,
            hover_rpy,
            duration_s=duration_s,
            num_steps=int(
                SOCKET_HOVER_REACTIVE_GUIDED_STEPS
                if guided_steps is None
                else guided_steps
            ),
        )
    else:
        _move(side, hover_pos, hover_rpy)
    scene_targets["socket_hover_rpy"] = [float(v) for v in hover_rpy]
    return hover_pos


def _guided_hover_plane_move(side, target_pos, target_rpy, duration_s=None, num_steps=None):
    env = _tool_env_from_callable(get_robot_state)
    if env is None or not hasattr(env, "move_bimanual_joint_keypoints"):
        raise RuntimeError("direct YAM env not available for guided hover move")
    side = str(side).strip().lower()
    if side not in {"left", "right"}:
        raise RuntimeError(f"guided hover move only implemented for left/right, got {side!r}")
    other = "left" if side == "right" else "right"

    duration_s = float(
        SOCKET_HOVER_REACTIVE_GUIDED_DURATION_S if duration_s is None else duration_s
    )
    num_steps = max(
        2,
        int(SOCKET_HOVER_REACTIVE_GUIDED_STEPS if num_steps is None else num_steps),
    )
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = _display_rpy_to_rotation(target_rpy).as_quat().astype(np.float64)

    obs_side = env.get_observations(side)
    obs_other = env.get_observations(other)
    side_jp = np.asarray(obs_side["joint_pos"], dtype=np.float64).reshape(6)
    other_jp = np.asarray(obs_other["joint_pos"], dtype=np.float64).reshape(6)
    side_gp = float(np.asarray(obs_side["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    other_gp = float(np.asarray(obs_other["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    side_start_pos = np.asarray(obs_side["ee_pos"], dtype=np.float64).reshape(3)
    side_start_quat = np.asarray(obs_side["ee_quat"], dtype=np.float64).reshape(4)
    other_hold_pos = np.asarray(obs_other["ee_pos"], dtype=np.float64).reshape(3)
    other_hold_quat = np.asarray(obs_other["ee_quat"], dtype=np.float64).reshape(4)

    start_rot = Rotation.from_quat(side_start_quat)
    target_rot = Rotation.from_quat(target_quat)
    delta_rot = target_rot * start_rot.inv()

    side_waypoints = [side_jp.copy()]
    other_waypoints = [other_jp.copy()]
    side_grippers = [[side_gp]]
    other_grippers = [[other_gp]]
    timestamps = [0.0]

    with env._kin_lock:
        env.kin.forward_kinematics(
            side_jp if side == "left" else other_jp,
            other_jp if side == "left" else side_jp,
        )
        cur_side_jp = side_jp.copy()
        cur_other_jp = other_jp.copy()
        for step_index in range(1, num_steps + 1):
            alpha = float(step_index) / float(num_steps)
            interp_pos = side_start_pos + alpha * (target_pos - side_start_pos)
            interp_rot = Rotation.from_rotvec(delta_rot.as_rotvec() * alpha) * start_rot
            interp_quat = interp_rot.as_quat().astype(np.float64)
            if side == "left":
                env.kin.forward_kinematics(cur_side_jp, cur_other_jp)
                next_side_jp, next_other_jp = env.kin.inverse_kinematics(
                    interp_pos,
                    interp_quat,
                    other_hold_pos,
                    other_hold_quat,
                    seeded=True,
                    dt=0.01,
                    solver="daqp",
                    damping=1e-3,
                    err_threshold=1e-4,
                    max_iters=40,
                )
            else:
                env.kin.forward_kinematics(cur_other_jp, cur_side_jp)
                next_other_jp, next_side_jp = env.kin.inverse_kinematics(
                    other_hold_pos,
                    other_hold_quat,
                    interp_pos,
                    interp_quat,
                    seeded=True,
                    dt=0.01,
                    solver="daqp",
                    damping=1e-3,
                    err_threshold=1e-4,
                    max_iters=40,
                )
            cur_side_jp = np.asarray(next_side_jp, dtype=np.float64).reshape(6)
            cur_other_jp = np.asarray(next_other_jp, dtype=np.float64).reshape(6)
            side_waypoints.append(cur_side_jp.copy())
            other_waypoints.append(cur_other_jp.copy())
            side_grippers.append([side_gp])
            other_grippers.append([other_gp])
            timestamps.append(alpha * duration_s)

    print(
        "[gpu_handover] Guided hover move: "
        f"side={side} target_pos={[round(float(v), 4) for v in target_pos.tolist()]} "
        f"target_rpy={[round(float(v), 1) for v in target_rpy]} "
        f"steps={num_steps} duration_s={duration_s:.2f}"
    )
    if side == "left":
        left_joint_positions = side_waypoints
        right_joint_positions = other_waypoints
        left_gripper_positions = side_grippers
        right_gripper_positions = other_grippers
    else:
        left_joint_positions = other_waypoints
        right_joint_positions = side_waypoints
        left_gripper_positions = other_grippers
        right_gripper_positions = side_grippers

    result = env.move_bimanual_joint_keypoints(
        timestamps=timestamps,
        left_joint_positions=left_joint_positions,
        right_joint_positions=right_joint_positions,
        left_gripper_positions=left_gripper_positions,
        right_gripper_positions=right_gripper_positions,
        playback_speed=1.0,
        command_hz=60.0,
        start_interp_s=0.0,
    )
    if not bool(result.get("success", False)):
        raise RuntimeError(result.get("reason", "guided hover move failed"))


def _refresh_motherboard_targets_from_current_view(
    reference_scene,
    base_scene=None,
    camera="top",
    artifact_prefix="refined",
):
    if base_scene is None:
        base_scene = reference_scene
    rgb = _as_uint8_rgb(get_camera_image(camera=camera))
    if rgb is None:
        raise RuntimeError(f"could not capture {camera!r} RGB image for motherboard refresh")
    cam_K = _camera_matrix(camera)
    T_cam_world = _camera_T_cam_world(camera)
    reference_top_z = float(reference_scene["motherboard_top_z"])
    print(
        "[gpu_handover] Refresh motherboard hover reference using fixed board plane: "
        f"reference_top_z={reference_top_z:.4f}"
    )
    reference_obb = reference_scene.get(
        "motherboard_reference_3d_bbox",
        reference_scene.get("motherboard_debug_3d_bbox"),
    )
    reference_center_world = np.asarray(
        reference_obb["center_world"],
        dtype=np.float64,
    ).reshape(3)
    prior_center_world = np.asarray(
        base_scene["motherboard_center_world"],
        dtype=np.float64,
    ).reshape(3)
    previous_right_edge_anchor_world = np.asarray(
        base_scene["motherboard_right_edge_anchor_world"],
        dtype=np.float64,
    ).reshape(3)
    seg_record = _segment_motherboard_mask(camera=camera)
    primary_score = float(seg_record.get("score", 0.0) or 0.0)
    seg_center_px = _bbox_center_px_from_xywh(seg_record.get("bbox_xywh"))
    current_right_edge_anchor_px = _mask_right_edge_anchor_px(
        seg_record.get("mask"),
        camera=camera,
    )
    registration = None
    registered_center_px = None
    registered_right_edge_anchor_px = None
    anchor_center_px = None
    tracking_mask = base_scene.get("motherboard_reference_mask")
    tracking_center_px = base_scene.get("motherboard_seg_center_px")
    tracking_right_edge_anchor_px = base_scene.get("motherboard_right_edge_anchor_px")
    current_mask = seg_record.get("mask")
    if tracking_mask is not None and tracking_center_px is not None and current_mask is not None:
        try:
            registration = _estimate_mask_translation_px(tracking_mask, current_mask)
            registered_center_px = np.asarray(
                tracking_center_px,
                dtype=np.float64,
            ).reshape(2) + np.asarray(
                [float(registration["dx_px"]), float(registration["dy_px"])],
                dtype=np.float64,
            )
            if tracking_right_edge_anchor_px is not None:
                registered_right_edge_anchor_px = np.asarray(
                    tracking_right_edge_anchor_px,
                    dtype=np.float64,
                ).reshape(2) + np.asarray(
                    [float(registration["dx_px"]), float(registration["dy_px"])],
                    dtype=np.float64,
                )
        except Exception as reg_exc:
            print(
                "[gpu_handover] Motherboard mask registration unavailable: "
                f"{reg_exc}. Falling back to SAM3 bbox center."
            )
    reference_mask = reference_scene.get("motherboard_reference_mask")
    reference_seg_center_px = reference_scene.get("motherboard_seg_center_px")
    if reference_mask is not None and reference_seg_center_px is not None and current_mask is not None:
        try:
            anchor_registration = _estimate_mask_translation_px(reference_mask, current_mask)
            anchor_center_px = np.asarray(
                reference_seg_center_px,
                dtype=np.float64,
            ).reshape(2) + np.asarray(
                [float(anchor_registration["dx_px"]), float(anchor_registration["dy_px"])],
                dtype=np.float64,
            )
            if registration is None:
                registration = anchor_registration
        except Exception:
            anchor_center_px = None
    chosen_center_px = registered_center_px if registered_center_px is not None else seg_center_px
    tracking_reference_center_px = chosen_center_px
    if registered_center_px is not None and anchor_center_px is not None:
        anchor_diff_px = float(
            np.linalg.norm(
                np.asarray(anchor_center_px, dtype=np.float64).reshape(2)
                - np.asarray(registered_center_px, dtype=np.float64).reshape(2)
            )
        )
        if anchor_diff_px <= float(REFRESH_MOTHERBOARD_REFERENCE_ANCHOR_BLEND_MAX_PX):
            tracking_reference_center_px = (
                (1.0 - float(REFRESH_MOTHERBOARD_REFERENCE_ANCHOR_BLEND_GAIN))
                * np.asarray(registered_center_px, dtype=np.float64).reshape(2)
                + float(REFRESH_MOTHERBOARD_REFERENCE_ANCHOR_BLEND_GAIN)
                * np.asarray(anchor_center_px, dtype=np.float64).reshape(2)
            )
    if chosen_center_px is None:
        raise RuntimeError("motherboard center pixel unavailable during refresh")
    edge_anchor_source = ""
    if REFRESH_MOTHERBOARD_EDGE_ANCHOR_SOURCE in {
        "registered",
        "registration",
        "tracked",
    }:
        chosen_right_edge_anchor_px = (
            registered_right_edge_anchor_px
            if registered_right_edge_anchor_px is not None
            else current_right_edge_anchor_px
        )
        edge_anchor_source = (
            "registered_right_edge"
            if registered_right_edge_anchor_px is not None
            else "current_right_edge"
        )
    else:
        chosen_right_edge_anchor_px = (
            current_right_edge_anchor_px
            if current_right_edge_anchor_px is not None
            else registered_right_edge_anchor_px
        )
        edge_anchor_source = (
            "current_right_edge"
            if current_right_edge_anchor_px is not None
            else "registered_right_edge"
        )
    if chosen_right_edge_anchor_px is None:
        raise RuntimeError("motherboard right-edge anchor pixel unavailable during refresh")
    edge_anchor_diff_px = None
    if (
        current_right_edge_anchor_px is not None
        and registered_right_edge_anchor_px is not None
    ):
        edge_anchor_diff_px = float(
            np.linalg.norm(
                np.asarray(current_right_edge_anchor_px, dtype=np.float64).reshape(2)
                - np.asarray(registered_right_edge_anchor_px, dtype=np.float64).reshape(2)
            )
        )
    proposed_center_world = _project_pixel_to_plane_world(
        chosen_center_px,
        cam_K,
        T_cam_world,
        reference_top_z,
    )
    if proposed_center_world is None:
        raise RuntimeError("could not project refreshed motherboard center to fixed board plane")
    proposed_center_world = np.asarray(proposed_center_world, dtype=np.float64).reshape(3)
    proposed_right_edge_anchor_world = _project_right_edge_anchor_world(
        chosen_right_edge_anchor_px,
        cam_K,
        T_cam_world,
        reference_top_z,
    )
    if proposed_right_edge_anchor_world is None:
        raise RuntimeError("could not project refreshed motherboard right-edge anchor to fixed board plane")
    proposed_right_edge_anchor_world = np.asarray(
        proposed_right_edge_anchor_world,
        dtype=np.float64,
    ).reshape(3)
    proposed_shift_m = float(
        np.linalg.norm(
            proposed_right_edge_anchor_world[:2] - previous_right_edge_anchor_world[:2]
        )
    )
    center_update = {
        "proposed_shift_m": proposed_shift_m,
        "applied_shift_m": 0.0,
        "held": False,
        "min_shift_m": float(REFRESH_MOTHERBOARD_CENTER_UPDATE_MIN_SHIFT_M),
        "max_shift_m": float(REFRESH_MOTHERBOARD_CENTER_UPDATE_MAX_SHIFT_M),
        "hold_reason": "",
        "edge_anchor_source": edge_anchor_source,
        "edge_anchor_diff_px": edge_anchor_diff_px,
    }
    accepted_center_px = chosen_center_px
    accepted_tracking_center_px = tracking_reference_center_px
    accepted_right_edge_anchor_px = chosen_right_edge_anchor_px
    accepted_right_edge_anchor_world = proposed_right_edge_anchor_world
    accepted_tracking_mask = (
        None if current_mask is None else np.asarray(current_mask, dtype=bool)
    )
    if proposed_shift_m > float(REFRESH_MOTHERBOARD_CENTER_UPDATE_MAX_SHIFT_M):
        chosen_center_world = prior_center_world.copy()
        chosen_right_edge_anchor_world = previous_right_edge_anchor_world.copy()
        center_update["held"] = True
        center_update["hold_reason"] = "large_shift"
        if base_scene.get("motherboard_center_px") is not None:
            accepted_center_px = np.asarray(
                base_scene["motherboard_center_px"],
                dtype=np.float64,
            ).reshape(2)
        accepted_tracking_center_px = tracking_center_px
        accepted_right_edge_anchor_px = base_scene.get("motherboard_right_edge_anchor_px")
        accepted_right_edge_anchor_world = previous_right_edge_anchor_world
        accepted_tracking_mask = tracking_mask
    else:
        chosen_center_world = proposed_center_world
        chosen_right_edge_anchor_world = proposed_right_edge_anchor_world
        center_update["applied_shift_m"] = proposed_shift_m
    reference_shift_m = float(
        np.linalg.norm(chosen_center_world[:2] - reference_center_world[:2])
    )
    current_obb = _override_obb_center_xy(reference_obb, chosen_center_world)
    scene = _build_socket_hover_scene_record(
        top_z=reference_top_z,
        hover_z=float(reference_scene["hover_z"]),
        center_px=accepted_center_px,
        center_world=chosen_center_world,
        seg_center_px=accepted_tracking_center_px,
        seg_bbox_xywh=seg_record.get("bbox_xywh"),
        reference_mask=accepted_tracking_mask,
        reference_obb=reference_scene["motherboard_reference_3d_bbox"],
        debug_obb=current_obb,
        registration=registration,
        x_offset_m=reference_scene["socket_hover_x_offset_m"],
        x_from_right_edge_m=reference_scene["socket_hover_x_from_right_edge_m"],
        edge_offset_m=reference_scene["socket_hover_edge_offset_m"],
        side_sign=reference_scene["socket_hover_side_sign"],
        target_socket_number=reference_scene["socket_hover_target_socket_number"],
        right_edge_anchor_px=accepted_right_edge_anchor_px,
        right_edge_anchor_world=accepted_right_edge_anchor_world,
        center_offset_world=reference_scene["socket_hover_center_offset_world"],
        right_edge_anchor_from_center_offset_world=reference_scene[
            "motherboard_right_edge_anchor_from_center_offset_world"
        ],
        localization_mode=reference_scene.get("socket_hover_localization_mode", "edge"),
    )
    scene["slot_record"] = reference_scene.get("slot_record")
    scene["socket_record"] = dict(
        base_scene.get("socket_record", reference_scene.get("socket_record", {}))
    )
    scene["socket_world"] = np.asarray(
        base_scene.get("socket_world", reference_scene["socket_world"]),
        dtype=float,
    ).reshape(3)
    scene["motherboard_center_update"] = center_update
    scene["pose_source_camera"] = str(camera)
    print(
        "[gpu_handover] Refresh center selection: "
        f"method={'mask_registration' if registered_center_px is not None else 'sam_bbox_center'} "
        f"proposed_shift_m={center_update['proposed_shift_m']:.4f} "
        f"applied_shift_m={center_update['applied_shift_m']:.4f} "
        f"held={center_update['held']} "
        f"hold_reason={center_update['hold_reason'] or 'none'} "
        f"edge_anchor_source={edge_anchor_source} "
        f"edge_anchor_diff_px={None if edge_anchor_diff_px is None else round(float(edge_anchor_diff_px), 1)} "
        f"reference_shift_m={reference_shift_m:.4f} "
        f"step_shift_m={float(np.linalg.norm(chosen_center_world[:2] - prior_center_world[:2])):.4f}"
    )
    if seg_center_px is not None:
        print(
            "[gpu_handover] Motherboard segmentation bbox center: "
            f"center_px={[round(float(v), 1) for v in np.asarray(seg_center_px, dtype=float).tolist()]}"
        )
    if registration is not None and registered_center_px is not None:
        print(
            "[gpu_handover] Motherboard mask registration override: "
            f"method={registration.get('method', 'unknown')} "
            f"dx_px={int(registration.get('dx_px', 0))} "
            f"dy_px={int(registration.get('dy_px', 0))} "
            f"score={float(registration.get('score', 0.0)):.1f} "
            f"registered_center_px="
            f"{[round(float(v), 1) for v in np.asarray(registered_center_px, dtype=float).tolist()]}"
        )
    _set_scene_socket_hover_target(
        scene,
        _derive_socket_hover_world_from_scene_geometry(scene, reference_top_z),
        cam_K,
        T_cam_world,
    )
    if SOCKET_HOVER_AUX_CAMERA is not None:
        try:
            aux_obs = _capture_socket_hover_aux_observation(
                SOCKET_HOVER_AUX_CAMERA,
                reference_top_z,
            )
            scene = _maybe_apply_socket_hover_aux_pose(
                reference_scene,
                scene,
                primary_score=primary_score,
                aux_obs=aux_obs,
                base_scene=base_scene,
                cam_K=cam_K,
                T_cam_world=T_cam_world,
                stage_label="reactive",
            )
        except Exception as exc:
            print(
                "[gpu_handover] Auxiliary camera refresh unavailable: "
                f"{exc}"
            )
    _set_scene_socket_hover_target(
        scene,
        _derive_socket_hover_world_from_scene_geometry(scene, reference_top_z),
        cam_K,
        T_cam_world,
    )
    print(
        "[gpu_handover] Refreshed motherboard-aligned socket hover target: "
        f"world={[round(float(v), 4) for v in scene['socket_hover_world'].tolist()]} "
        f"edge_offset_m={float(scene['socket_hover_edge_offset_m']):.3f} "
        f"x_offset_m={float(scene['socket_hover_x_offset_m']):.4f} "
        f"x_from_right_edge_m={float(scene['socket_hover_x_from_right_edge_m']):.4f} "
        f"pose_source={scene.get('pose_source_camera', camera)!r} "
        f"localization_mode={scene.get('socket_hover_localization_mode', 'edge')!r}"
    )
    if SOCKET_HOVER_SAVE_REFRESH_ARTIFACTS:
        _save_initial_motherboard_3d_bbox_artifacts(
            rgb,
            camera,
            scene,
            artifact_prefix=artifact_prefix,
            obb_record={
                "seg_record": seg_record,
                "obb": scene["motherboard_debug_3d_bbox"],
            },
        )
    return scene


def _acquire_socket_hover_with_refresh_retries(
    side,
    hover_rpy,
    scene_targets,
    axis_kind,
    camera="top",
    max_attempts=None,
):
    attempts = max(
        1,
        int(
            SOCKET_HOVER_INITIAL_MAX_ATTEMPTS if max_attempts is None else max_attempts
        ),
    )
    reference_scene_targets = scene_targets
    current_scene_targets = scene_targets
    last_exc = None
    for attempt_idx in range(attempts):
        attempt_num = attempt_idx + 1
        print(
            "[gpu_handover] Step 8a hover attempt "
            f"{attempt_num}/{attempts}"
        )
        try:
            hover_pos = _move_to_cached_socket_hover_pose(
                side,
                hover_rpy,
                current_scene_targets,
                axis_kind,
                guided=True,
                guided_duration_s=float(SOCKET_HOVER_INITIAL_GUIDED_DURATION_S),
                guided_steps=int(SOCKET_HOVER_INITIAL_GUIDED_STEPS),
            )
            return hover_pos, current_scene_targets
        except Exception as exc:
            last_exc = exc
            if attempt_num >= attempts:
                break
            print(
                "[gpu_handover] Step 8a hover attempt "
                f"{attempt_num}/{attempts} failed: {exc}. "
                "Refreshing motherboard pose and retrying hover."
            )
            current_scene_targets = _refresh_motherboard_targets_from_current_view(
                reference_scene=reference_scene_targets,
                base_scene=current_scene_targets,
                camera=camera,
                artifact_prefix=f"refined_attempt_{attempt_num:02d}",
            )
    raise RuntimeError(
        f"socket hover failed after {attempts} attempt(s): {last_exc}"
    )


def _run_reactive_socket_hover_loop(
    side,
    hover_rpy,
    scene_targets,
    axis_kind,
    camera="top",
    initial_delay_s=None,
    reactive_plane_z=None,
):
    period_s = max(0.0, float(REACTIVE_SOCKET_HOVER_PERIOD_S))
    reference_scene_targets = scene_targets
    current_scene_targets = scene_targets
    current_hover_pos = _robot_vec(get_robot_state(), side, "ee_pos")
    cycle_index = 0
    print(
        "[gpu_handover] Step 8 reactive loop enabled: "
        f"retry_period_s={period_s:.2f} target_socket={int(TARGET_SOCKET_NUMBER)}"
    )
    while True:
        cycle_index += 1
        delay_s = period_s if initial_delay_s is None or cycle_index > 1 else max(0.0, float(initial_delay_s))
        print(f"[gpu_handover] Step 8 reactive cycle {cycle_index}: refresh motherboard and adjust hover")
        if delay_s > 0.0:
            time.sleep(delay_s)
        try:
            refined_scene_targets = _refresh_motherboard_targets_from_current_view(
                reference_scene=reference_scene_targets,
                base_scene=current_scene_targets,
                camera=camera,
                artifact_prefix=f"refined_cycle_{cycle_index:03d}",
            )
            current_hover_pos = _move_to_cached_socket_hover_pose(
                side,
                hover_rpy,
                refined_scene_targets,
                axis_kind,
                fixed_plane_z=reactive_plane_z,
                guided=True,
            )
            current_scene_targets = refined_scene_targets
            print(
                "[gpu_handover] Step 8 reactive cycle "
                f"{cycle_index}: updated hover pos="
                f"{[round(float(v), 4) for v in current_hover_pos]}"
            )
        except Exception as exc:
            print(
                "[gpu_handover] Step 8 reactive cycle "
                f"{cycle_index} failed: {exc}. Keeping current hover and retrying."
            )


def _batch_select(side, candidates, label, planning_speed=None):
    if not candidates:
        raise RuntimeError(f"no candidates for {label}")
    planning_speed = (
        HANDOVER_BATCH_PLANNING_SPEED if planning_speed is None else float(planning_speed)
    )
    batch = freespace_move(
        grasp_candidates=list(candidates),
        batch_side=side,
        batch_top_k=max(int(RUN_CONFIG["batch_top_k"]), len(candidates)),
        solver_speed=RUN_CONFIG["solver_speed"],
        batch_validate_trajectory=RUN_CONFIG["batch_validate_trajectory"],
        planning_speed=planning_speed,
        ik_error_threshold=RUN_CONFIG["ik_error_threshold"],
        ik_xyz_weight=RUN_CONFIG["ik_xyz_weight"],
        ik_rpy_weight=RUN_CONFIG["ik_rpy_weight"],
        planner_backend=RUN_CONFIG["planner_backend"],
    )
    print(
        f"[gpu_handover] Batched {label} [{side}]: "
        f"input={int(getattr(batch, 'input_candidate_count', len(candidates)))}, "
        f"evaluated={int(getattr(batch, 'evaluated_candidate_count', 0))}, "
        f"truncated={int(getattr(batch, 'truncated_input_count', 0))}, "
        f"solve={float(getattr(batch, 'curobo_solve_time_ms', 0.0)):.1f}ms, "
        f"graph={float(getattr(batch, 'curobo_graph_time_ms', 0.0)):.1f}ms, "
        f"ik={float(getattr(batch, 'curobo_ik_time_ms', 0.0)):.1f}ms, "
        f"mode={getattr(batch, 'planning_mode', 'unknown')}"
    )
    best = getattr(batch, "best_candidate", None)
    if best is not None and getattr(best, "motion_plan_error", True) is False:
        source_index = int(getattr(best, "source_index", 0) or 0)
        source = candidates[source_index] if source_index < len(candidates) else {}
        print(
            f"[gpu_handover] Selected {label}: rank={int(getattr(best, 'rank', 0))} "
            f"label={source.get('label', 'unknown')} "
            f"xyz={[round(float(x), 4) for x in getattr(best, 'position', [])]} "
            f"rpy={[round(float(x), 1) for x in getattr(best, 'rpy', [])]}"
        )
        return best, source

    failures = []
    for candidate in list(getattr(batch, "batch_candidates", []) or [])[:5]:
        if getattr(candidate, "motion_plan_error", False):
            failures.append(
                f"rank={int(getattr(candidate, 'rank', 0))}: "
                f"{getattr(candidate, 'motion_plan_reason', None) or 'Motion plan error'}"
            )
    raise RuntimeError(
        f"Batch cuRobo returned no feasible {label} on {side}. "
        + (" Top failures: " + "; ".join(failures) if failures else "")
    )


def _execute_batch_candidate(best_candidate):
    key = getattr(best_candidate, "trajectory_cache_key", None)
    if not key:
        raise RuntimeError("batch candidate missing trajectory_cache_key")
    freespace_move(trajectory_cache_key=key)


def _live_handover_geometry(grasp, ref_top_pos):
    if grasp is None:
        return None
    bbox = getattr(grasp, "bbox_result", None)
    if bbox is None:
        return None

    center = np.asarray(bbox.obb_center_world, dtype=float)
    normal = _unit(bbox.top_normal_world)
    if normal[2] < 0.0:
        normal = -normal
    center_xy_dist = float(
        np.linalg.norm(center[:2] - np.asarray(ref_top_pos, dtype=float)[:2])
    )
    top_normal_extent, short_extent, long_extent = _bbox_semantic_extents(bbox)

    reasons = []
    if center_xy_dist > LIVE_BBOX_MAX_CENTER_XY_DIST_M:
        reasons.append(
            f"center_xy_dist={center_xy_dist:.3f}>{LIVE_BBOX_MAX_CENTER_XY_DIST_M:.3f}"
        )
    if top_normal_extent > LIVE_BBOX_MAX_TOP_NORMAL_EXTENT_M:
        reasons.append(
            "top_normal_extent="
            f"{top_normal_extent:.3f}>{LIVE_BBOX_MAX_TOP_NORMAL_EXTENT_M:.3f}"
        )
    if short_extent > LIVE_BBOX_MAX_FACE_EXTENT_M:
        reasons.append(
            f"short_extent={short_extent:.3f}>{LIVE_BBOX_MAX_FACE_EXTENT_M:.3f}"
        )
    if abs(float(normal[2])) < 0.80:
        reasons.append(f"normal_z={float(normal[2]):.3f}<0.800")

    if reasons:
        print(
            "[gpu_handover] Live held-GPU OBB rejected: "
            + ", ".join(reasons)
        )
        return None

    print(
        "[gpu_handover] Live held-GPU OBB accepted: "
        f"center={[round(float(v), 4) for v in center.tolist()]} "
        f"extents={[round(top_normal_extent, 4), round(short_extent, 4), round(long_extent, 4)]}"
    )
    return {
        "source": "live_obb",
        "center": center,
        "normal": normal,
        "families": [
            {
                "name": "live_short_axis",
                "axis_kind": "short",
                "axis_xy": _toward_left(bbox.short_axis_world),
                "face_extent_m": short_extent,
                "score_bias": 1.00,
            },
            {
                "name": "live_long_axis",
                "axis_kind": "long",
                "axis_xy": _toward_left(bbox.long_axis_world),
                "face_extent_m": long_extent,
                "score_bias": 0.96,
            },
        ],
    }


def _fallback_handover_geometry(ref_top_pos, ref_rpy):
    top_pos = np.asarray(ref_top_pos, dtype=float)
    center = top_pos + np.array([0.0, 0.0, -GPU_HALF_THICKNESS_M], dtype=float)
    jaw_axis = _display_rpy_to_rotation(ref_rpy).as_matrix()[:, 0]
    short_axis_xy = _toward_left(jaw_axis)
    long_axis_xy = _toward_left(_perp_xy(short_axis_xy))
    print(
        "[gpu_handover] Using kinematic handover estimate: "
        f"top_pos={[round(float(v), 4) for v in top_pos.tolist()]} "
        f"center={[round(float(v), 4) for v in center.tolist()]} "
        f"short_axis_xy={[round(float(v), 4) for v in short_axis_xy.tolist()]}"
    )
    return {
        "source": "handover_estimate",
        "center": center,
        "normal": np.array([0.0, 0.0, 1.0], dtype=float),
        "families": [
            {
                "name": "estimated_short_axis",
                "axis_kind": "short",
                "axis_xy": short_axis_xy,
                "face_extent_m": GPU_SHORT_EDGE_M,
                "score_bias": 1.00,
            },
            {
                "name": "estimated_long_axis",
                "axis_kind": "long",
                "axis_xy": long_axis_xy,
                "face_extent_m": GPU_LONG_EDGE_M,
                "score_bias": 0.93,
            },
        ],
    }


def _select_handover_geometry(grasp, ref_top_pos, ref_rpy):
    live = _live_handover_geometry(grasp, ref_top_pos)
    if live is not None:
        return live
    return _fallback_handover_geometry(ref_top_pos, ref_rpy)


def _build_left_board_face_candidates(geometry, right_handover_pos=None):
    center = np.asarray(geometry["center"], dtype=float)
    normal = _unit(geometry["normal"])
    if normal[2] < 0.0:
        normal = -normal
    all_families = list(geometry["families"])
    families = [
        family
        for family in all_families
        if str(family.get("axis_kind", "")).strip().lower() == "short"
    ]
    if families:
        skipped = len(all_families) - len(families)
        if skipped > 0:
            print(
                "[gpu_handover] Left regrasp constrained to long-edge family: "
                f"kept={len(families)} skipped={skipped}"
            )
    else:
        print(
            "[gpu_handover] Left regrasp family fallback: "
            "no long-edge family found, keeping original family set"
        )
        families = all_families
    inset_values = _ordered_unique_positive(
        [
            LEFT_REGRASP_INSET_M,
            LEFT_REGRASP_DEEP_INSET_M,
            LEFT_REGRASP_MAX_INSET_M,
        ]
    )
    right_ref = (
        None
        if right_handover_pos is None
        else np.asarray(right_handover_pos, dtype=float).reshape(3)
    )
    receiver_edge_bias_m = max(0.0, float(LEFT_REGRASP_RECEIVER_EDGE_BIAS_M))
    forward_x_bias_m = float(LEFT_REGRASP_FORWARD_X_BIAS_M)
    forward_x_bias = np.array([forward_x_bias_m, 0.0, 0.0], dtype=float)
    face_margin_m = max(0.0005, float(LEFT_REGRASP_FACE_MARGIN_M))
    min_outward_clearance_m = max(
        0.0,
        float(LEFT_REGRASP_RIGHT_EE_MIN_OUTWARD_CLEARANCE_M),
    )
    min_xy_clearance_m = max(0.0, float(LEFT_REGRASP_RIGHT_EE_MIN_XY_CLEARANCE_M))
    min_forward_x_clearance_m = float(
        LEFT_REGRASP_RIGHT_EE_MIN_FORWARD_X_CLEARANCE_M
    )

    all_pregrasp_candidates = []
    all_grasp_candidates = []
    safe_pregrasp_candidates = []
    safe_grasp_candidates = []
    unsafe_pair_count = 0
    for family_index, family in enumerate(families):
        outward = _toward_left(family["axis_xy"])
        approach = -outward
        axis_kind = str(family.get("axis_kind", "")).strip().lower() or (
            "short" if "short" in str(family.get("name", "")).lower() else "long"
        )
        half_extent = 0.5 * float(family["face_extent_m"])
        base_pregrasp = center + outward * (half_extent + LEFT_REGRASP_STANDOFF_M)
        max_grasp_offset_m = max(0.0, half_extent - face_margin_m)
        z_variants = (
            (-0.005, "zm5", 0.0),
            (0.010, "zp10", -0.01),
            (0.025, "zp25", -0.02),
        )
        for jaw_index, jaw_axis in enumerate((normal, -normal)):
            rpy = _display_rpy_from_axes(jaw_axis, approach)
            insertion_hold_rpy = _vertical_insertion_rpy_from_grasp_axes(
                jaw_axis,
                approach,
                axis_kind,
            )
            for z_offset_base, z_label_base, z_score_bias in z_variants:
                z_offset = float(z_offset_base) + float(LEFT_REGRASP_Z_BIAS_M)
                z_label = (
                    f"{z_label_base}_zb"
                    f"{int(round(1000.0 * LEFT_REGRASP_Z_BIAS_M)):+d}"
                )
                pregrasp_pos = base_pregrasp.copy() + forward_x_bias
                pregrasp_pos[2] += z_offset
                for inset_index, inset_m in enumerate(inset_values):
                    raw_inset_m = inset_m + LEFT_REGRASP_EXTRA_INSET_M
                    effective_inset_m = min(
                        half_extent - 1e-4,
                        max(face_margin_m, raw_inset_m - receiver_edge_bias_m),
                    )
                    grasp_offset_m = min(
                        max_grasp_offset_m,
                        half_extent - effective_inset_m,
                    )
                    grasp_pos = center + outward * grasp_offset_m + forward_x_bias
                    grasp_pos[2] += z_offset
                    right_outward_clearance_m = None
                    right_xy_clearance_m = None
                    right_forward_x_clearance_m = None
                    right_clearance_ok = True
                    if right_ref is not None:
                        delta_from_right = grasp_pos - right_ref
                        right_forward_x_clearance_m = float(delta_from_right[0])
                        right_outward_clearance_m = float(
                            np.dot(delta_from_right[:2], outward[:2])
                        )
                        right_xy_clearance_m = float(
                            np.linalg.norm(delta_from_right[:2])
                        )
                        right_clearance_ok = bool(
                            right_outward_clearance_m >= min_outward_clearance_m
                            and right_xy_clearance_m >= min_xy_clearance_m
                            and right_forward_x_clearance_m >= min_forward_x_clearance_m
                        )
                    score = (
                        float(family["score_bias"])
                        - 0.03 * jaw_index
                        + float(z_score_bias)
                        - 0.01 * inset_index
                    )
                    label_prefix = (
                        f"{geometry['source']}_{family['name']}_jaw{jaw_index}"
                        f"_{z_label}_inset{int(round(1000.0 * inset_m))}"
                    )
                    pair_id = label_prefix
                    base_payload = {
                        "pair_id": pair_id,
                        "axis_kind": axis_kind,
                        "approach_world": [float(v) for v in approach.tolist()],
                        "insertion_hold_rpy": [float(v) for v in insertion_hold_rpy],
                        "raw_inset_m": float(raw_inset_m),
                        "effective_inset_m": float(effective_inset_m),
                        "forward_x_bias_m": float(forward_x_bias_m),
                        "receiver_edge_bias_m": float(receiver_edge_bias_m),
                        "right_forward_x_clearance_m": right_forward_x_clearance_m,
                        "right_outward_clearance_m": right_outward_clearance_m,
                        "right_xy_clearance_m": right_xy_clearance_m,
                    }
                    pregrasp_candidate = {
                        **base_payload,
                        "position": [float(v) for v in pregrasp_pos.tolist()],
                        "rpy": [float(v) for v in rpy],
                        "score": score,
                        "label": f"pre_{label_prefix}",
                    }
                    grasp_candidate = {
                        **base_payload,
                        "position": [float(v) for v in grasp_pos.tolist()],
                        "rpy": [float(v) for v in rpy],
                        "score": score,
                        "label": f"grasp_{label_prefix}",
                    }
                    all_pregrasp_candidates.append(pregrasp_candidate)
                    all_grasp_candidates.append(grasp_candidate)
                    if right_clearance_ok:
                        safe_pregrasp_candidates.append(pregrasp_candidate)
                        safe_grasp_candidates.append(grasp_candidate)
                    else:
                        unsafe_pair_count += 1
        print(
            "[gpu_handover] Candidate family "
            f"{family['name']}: outward={[round(float(v), 4) for v in outward.tolist()]} "
            f"axis_kind={axis_kind} "
            f"face_extent={float(family['face_extent_m']):.4f} "
            f"insets_mm={[int(round(1000.0 * inset)) for inset in inset_values]} "
            f"z_bias_mm={int(round(1000.0 * LEFT_REGRASP_Z_BIAS_M))} "
            f"forward_x_bias_mm={int(round(1000.0 * forward_x_bias_m))} "
            f"receiver_bias_mm={int(round(1000.0 * receiver_edge_bias_m))}"
        )
    if safe_grasp_candidates:
        if unsafe_pair_count > 0:
            print(
                "[gpu_handover] Left regrasp right-hand clearance filter: "
                f"kept={len(safe_grasp_candidates)} rejected={unsafe_pair_count} "
                f"min_outward={min_outward_clearance_m:.3f} "
                f"min_xy={min_xy_clearance_m:.3f} "
                f"min_forward_x={min_forward_x_clearance_m:.3f}"
            )
        return safe_pregrasp_candidates, safe_grasp_candidates
    if right_ref is not None and LEFT_REGRASP_REQUIRE_RIGHT_CLEARANCE:
        raise RuntimeError(
            "no left handover grasp candidate keeps enough clearance from the right gripper "
            f"(required outward>={min_outward_clearance_m:.3f}m, "
            f"xy>={min_xy_clearance_m:.3f}m, "
            f"forward_x>={min_forward_x_clearance_m:.3f}m)"
        )
    if right_ref is not None:
        print(
            "[gpu_handover] Left regrasp right-hand clearance filter: "
            "no safe candidate found; falling back because "
            "GPU_LEFT_REGRASP_REQUIRE_RIGHT_CLEARANCE=0"
        )
    return all_pregrasp_candidates, all_grasp_candidates


def _build_right_handover_stage_candidates(base_rpy):
    preferred_receiver_clear_y = (
        RIGHT_HANDOVER_REGRASP_Y_M - max(0.0, RIGHT_HANDOVER_RECEIVER_CLEARANCE_M)
    )
    z_targets = _ordered_unique_positive(
        [
            HANDOVER_POS[2],
            min(HANDOVER_REGRASP_Z_M, HANDOVER_POS[2] + 0.02),
            min(HANDOVER_REGRASP_Z_M, HANDOVER_POS[2] + 0.04),
            HANDOVER_REGRASP_Z_M,
        ]
    )
    xy_targets = [
        (
            HANDOVER_POS[0] + RIGHT_HANDOVER_STAGE_X_BIAS_M,
            min(preferred_receiver_clear_y, -0.06),
        ),
        (
            HANDOVER_POS[0],
            min(preferred_receiver_clear_y, -0.06),
        ),
        (
            HANDOVER_POS[0] + RIGHT_HANDOVER_STAGE_X_BIAS_M,
            min(RIGHT_HANDOVER_REGRASP_Y_M, -0.06),
        ),
        (
            HANDOVER_POS[0],
            min(RIGHT_HANDOVER_REGRASP_Y_M, -0.06),
        ),
        (
            HANDOVER_POS[0] + RIGHT_HANDOVER_STAGE_X_BIAS_M,
            min(RIGHT_HANDOVER_REGRASP_Y_M, -0.04),
        ),
        (
            HANDOVER_POS[0],
            min(RIGHT_HANDOVER_REGRASP_Y_M, -0.04),
        ),
        (
            HANDOVER_POS[0] + RIGHT_HANDOVER_STAGE_X_BIAS_M,
            min(RIGHT_HANDOVER_REGRASP_Y_M, -0.02),
        ),
        (HANDOVER_POS[0], RIGHT_HANDOVER_REGRASP_Y_M),
        (HANDOVER_POS[0] - 0.02, RIGHT_HANDOVER_REGRASP_Y_M),
        (HANDOVER_POS[0], HANDOVER_POS[1]),
    ]
    candidates = []
    seen = set()
    for z_index, z_target in enumerate(z_targets):
        for xy_index, (x_target, y_target) in enumerate(xy_targets):
            pos = (
                round(float(x_target), 4),
                round(float(y_target), 4),
                round(float(z_target), 4),
            )
            if pos in seen:
                continue
            seen.add(pos)
            score = 1.0 - 0.03 * z_index - 0.02 * xy_index
            candidates.append(
                {
                    "position": [float(v) for v in pos],
                    "rpy": [float(v) for v in base_rpy],
                    "score": score,
                    "label": f"right_handover_stage_z{z_index}_xy{xy_index}",
                }
            )
    return candidates


def _choose_right_handover_move_rpy(handover_yaw):
    preview_candidates = []
    for roll_rank, roll_offset_deg in enumerate(RIGHT_HANDOVER_ROLL_SEARCH_OFFSETS_DEG):
        candidate_rpy = [float(roll_offset_deg), 180.0, float(handover_yaw)]
        for stage_candidate in _build_right_handover_stage_candidates(candidate_rpy):
            candidate = dict(stage_candidate)
            candidate["score"] = float(stage_candidate.get("score", 0.0)) - 0.01 * float(
                roll_rank
            )
            candidate["label"] = (
                f"{stage_candidate.get('label', 'right_handover_stage')}"
                f"_roll{roll_offset_deg:+.1f}"
            )
            preview_candidates.append(candidate)
    best_preview, best_preview_src = _batch_select(
        "right",
        preview_candidates,
        label="right handover orientation preview",
        planning_speed=HANDOVER_BATCH_PLANNING_SPEED,
    )
    selected_rpy = [float(v) for v in best_preview_src.get("rpy", [0.0, 180.0, handover_yaw])]
    print(
        "[gpu_handover] Right handover orientation selected: "
        f"label={best_preview_src.get('label', 'unknown')} "
        f"rpy={[round(v, 1) for v in selected_rpy]}"
    )
    return selected_rpy


def _right_pick_main_body_dir(grasp, camera="top"):
    image_side = str(RIGHT_PICK_MAIN_BODY_IMAGE_SIDE or "lower").strip().lower()
    bbox = getattr(grasp, "bbox_result", None)
    if bbox is not None:
        long_axis_world = getattr(bbox, "long_axis_world", None)
        center_world = getattr(bbox, "obb_center_world", None)
        long_extent = float(getattr(bbox, "top_face_long_extent", 0.0) or 0.0)
        if long_axis_world is not None and center_world is not None:
            try:
                axis_world = _unit(long_axis_world)
            except Exception:
                axis_world = np.asarray([0.0, 1.0, 0.0], dtype=float)
            center = np.asarray(center_world, dtype=float).reshape(-1)
            if center.size == 3:
                span_m = max(0.02, 0.5 * long_extent)
                plus_world = center + span_m * axis_world
                minus_world = center - span_m * axis_world
                try:
                    cam_K = _camera_matrix(camera)
                    T_cam_world = _camera_T_cam_world(camera)
                    plus_px = _project_world_to_pixel(plus_world, cam_K, T_cam_world)
                    minus_px = _project_world_to_pixel(minus_world, cam_K, T_cam_world)
                except Exception:
                    plus_px = None
                    minus_px = None
                axis_xy = _xy_unit(axis_world, fallback=[0.0, 1.0, 0.0])
                if plus_px is not None and minus_px is not None:
                    plus_is_lower = float(plus_px[1]) >= float(minus_px[1])
                    prefer_lower = image_side not in {"upper", "top"}
                    return (
                        axis_xy
                        if plus_is_lower == prefer_lower
                        else -axis_xy
                    )
                return axis_xy if float(axis_xy[1]) >= 0.0 else -axis_xy
    grasp_rpy = [float(v) for v in getattr(grasp, "rpy", [0.0, 180.0, 0.0])]
    jaw_axis = _display_rpy_to_rotation(grasp_rpy).as_matrix()[:, 0]
    axis = _perp_xy(jaw_axis)
    return axis if float(axis[1]) >= 0.0 else -axis


def _right_pick_canonical_rpy(rpy, *, enabled=None):
    out = [float(v) for v in rpy]
    if enabled is None:
        enabled = RIGHT_PICK_CANONICALIZE_YAW
    if not enabled or len(out) != 3:
        yaw_text = ""
        if len(out) == 3:
            yaw_text = (
                "rpy_yaw_canonicalized=disabled "
                f"yaw={float(_normalize_display_rpy([0.0, 0.0, out[2]])[2]):.1f} "
            )
        return out, yaw_text
    old_yaw = float(_normalize_display_rpy([0.0, 0.0, out[2]])[2])
    candidates = []
    for shift_deg in (0.0, 180.0, -180.0):
        yaw = float(_normalize_display_rpy([0.0, 0.0, old_yaw + shift_deg])[2])
        if all(abs(yaw - existing) > 1e-4 for existing in candidates):
            candidates.append(yaw)
    max_abs_yaw = min(180.0, max(0.0, float(RIGHT_PICK_MAX_ABS_YAW_DEG)))
    in_range = [yaw for yaw in candidates if abs(yaw) <= max_abs_yaw + 1e-6]
    best_yaw = min(in_range or candidates, key=lambda yaw: abs(yaw))
    out[2] = float(best_yaw)
    if abs(best_yaw - old_yaw) > 1e-4:
        return (
            out,
            "rpy_yaw_canonicalized_from="
            f"{old_yaw:.1f} to={best_yaw:.1f} "
            f"max_abs_yaw_deg={max_abs_yaw:.1f} ",
        )
    return (
        out,
        "rpy_yaw_canonicalized=unchanged "
        f"yaw={old_yaw:.1f} max_abs_yaw_deg={max_abs_yaw:.1f} ",
    )


def _replace_grasp_pose(grasp, position_xyz=None, rpy=None):
    updates = {"trajectory_cache_key": None}
    if position_xyz is not None:
        updates["position"] = [
            float(v) for v in np.asarray(position_xyz, dtype=float).reshape(3).tolist()
        ]
    if rpy is not None:
        updates["rpy"] = [
            float(v) for v in np.asarray(rpy, dtype=float).reshape(3).tolist()
        ]
    if hasattr(grasp, "_replace"):
        return grasp._replace(**updates)
    out = dict(grasp) if isinstance(grasp, dict) else None
    if out is None:
        return grasp
    out.update(updates)
    return out


def _replace_grasp_position(grasp, position_xyz):
    return _replace_grasp_pose(grasp, position_xyz=position_xyz)


def _adjust_initial_right_pick_grasp(
    grasp,
    side=None,
    label=None,
    max_adjusted_pick_z=None,
    canonicalize_yaw=None,
):
    if side != "right":
        return grasp
    bias_m = max(0.0, float(RIGHT_PICK_MAIN_BODY_BIAS_M))
    extra_z_m = float(RIGHT_PICK_EXTRA_Z_OFFSET_M)
    above_top_surface_m = max(0.0, float(RIGHT_PICK_ABOVE_TOP_SURFACE_M))
    bias_dir = _right_pick_main_body_dir(grasp) if bias_m > 0.0 else None
    pos = np.asarray(getattr(grasp, "position", []), dtype=float).reshape(-1)
    rpy = [float(v) for v in getattr(grasp, "rpy", [])]
    if pos.size != 3 or len(rpy) != 3:
        return grasp
    adjusted_rpy, rpy_text = _right_pick_canonical_rpy(
        rpy,
        enabled=(
            RIGHT_PICK_CANONICALIZE_YAW
            if canonicalize_yaw is None
            else bool(canonicalize_yaw)
        ),
    )
    adjusted_pos = pos.copy()
    if abs(extra_z_m) > 1e-9:
        adjusted_pos[2] += extra_z_m
    bbox = getattr(grasp, "bbox_result", None)
    top_surface_z = (
        None
        if bbox is None
        else getattr(bbox, "top_surface_z", None)
    )
    top_surface_retargeted = False
    if top_surface_z is not None:
        try:
            adjusted_pos[2] = float(top_surface_z) + above_top_surface_m
            top_surface_retargeted = True
        except Exception:
            top_surface_z = None
    top_surface_text = ""
    if top_surface_z is not None:
        top_surface_text = (
            f"top_surface_z={float(top_surface_z):.4f} "
            f"above_top_surface_m={above_top_surface_m:.4f} "
            f"top_surface_retargeted={top_surface_retargeted} "
        )
    fixed_z_text = ""
    if RIGHT_PICK_USE_FIXED_Z and RIGHT_PICK_FIXED_Z_M is not None:
        old_z = float(adjusted_pos[2])
        adjusted_pos[2] = float(RIGHT_PICK_FIXED_Z_M)
        fixed_z_text = (
            f"fixed_pick_z_from={old_z:.4f} "
            f"fixed_pick_z={float(RIGHT_PICK_FIXED_Z_M):.4f} "
        )
    z_clamp_text = ""
    if PICK_CLAMP_HIGH_Z_TO_TABLE and max_adjusted_pick_z is not None:
        try:
            max_z = float(max_adjusted_pick_z)
            if float(adjusted_pos[2]) > max_z:
                old_z = float(adjusted_pos[2])
                adjusted_pos[2] = max_z
                z_clamp_text = (
                    f"z_clamped_from={old_z:.4f} "
                    f"max_adjusted_pick_z={max_z:.4f} "
                )
        except Exception as exc:
            z_clamp_text = f"z_clamp_skipped={exc} "
    center_world = getattr(bbox, "obb_center_world", None) if bbox is not None else None
    if center_world is None:
        print(
            "[gpu_handover] Initial right-pick adjustment: "
            f"label={label or 'grasp'} "
            f"old_pos={[round(float(v), 4) for v in pos.tolist()]} "
            f"new_pos={[round(float(v), 4) for v in adjusted_pos.tolist()]} "
            f"old_rpy={[round(float(v), 1) for v in rpy]} "
            f"new_rpy={[round(float(v), 1) for v in adjusted_rpy]} "
            f"z_offset_m={extra_z_m:.4f} "
            f"{top_surface_text}{fixed_z_text}{z_clamp_text}{rpy_text}"
            "body_bias=skipped(missing bbox center)"
        )
        return _replace_grasp_pose(grasp, position_xyz=adjusted_pos, rpy=adjusted_rpy)
    center = np.asarray(center_world, dtype=float).reshape(-1)
    if center.size != 3:
        print(
            "[gpu_handover] Initial right-pick adjustment: "
            f"label={label or 'grasp'} "
            f"old_pos={[round(float(v), 4) for v in pos.tolist()]} "
            f"new_pos={[round(float(v), 4) for v in adjusted_pos.tolist()]} "
            f"old_rpy={[round(float(v), 1) for v in rpy]} "
            f"new_rpy={[round(float(v), 1) for v in adjusted_rpy]} "
            f"z_offset_m={extra_z_m:.4f} "
            f"{top_surface_text}{fixed_z_text}{z_clamp_text}{rpy_text}"
            f"body_bias=skipped(invalid bbox center={center_world})"
        )
        return _replace_grasp_pose(grasp, position_xyz=adjusted_pos, rpy=adjusted_rpy)
    signed_long_axis_m = None
    target_body_shift_m = None
    if bias_dir is not None:
        signed_long_axis_m = float(np.dot(adjusted_pos[:2] - center[:2], bias_dir[:2]))
        long_extent_m = float(getattr(bbox, "top_face_long_extent", 0.0) or 0.0)
        target_body_shift_m = (
            min(bias_m, 0.25 * long_extent_m) if long_extent_m > 0.0 else bias_m
        )
    if (
        bias_dir is None
        or signed_long_axis_m is None
        or target_body_shift_m is None
        or target_body_shift_m <= signed_long_axis_m + 1e-6
    ):
        print(
            "[gpu_handover] Initial right-pick adjustment: "
            f"label={label or 'grasp'} "
            f"old_pos={[round(float(v), 4) for v in pos.tolist()]} "
            f"new_pos={[round(float(v), 4) for v in adjusted_pos.tolist()]} "
            f"old_rpy={[round(float(v), 1) for v in rpy]} "
            f"new_rpy={[round(float(v), 1) for v in adjusted_rpy]} "
            f"z_offset_m={extra_z_m:.4f} "
            f"{top_surface_text}{fixed_z_text}{z_clamp_text}{rpy_text}"
            + (
                "body_bias=disabled"
                if bias_dir is None or signed_long_axis_m is None
                else (
                    "body_bias=skipped(already on main-body half "
                    f"signed_long_axis_m={signed_long_axis_m:.4f} "
                    f"target_body_shift_m={target_body_shift_m:.4f})"
                )
            )
        )
        return _replace_grasp_pose(grasp, position_xyz=adjusted_pos, rpy=adjusted_rpy)
    applied_bias_m = max(0.0, float(target_body_shift_m) - float(signed_long_axis_m))
    adjusted_pos[:2] += applied_bias_m * bias_dir[:2]
    print(
        "[gpu_handover] Initial right-pick adjustment: "
        f"label={label or 'grasp'} "
        f"old_pos={[round(float(v), 4) for v in pos.tolist()]} "
        f"new_pos={[round(float(v), 4) for v in adjusted_pos.tolist()]} "
        f"old_rpy={[round(float(v), 1) for v in rpy]} "
        f"new_rpy={[round(float(v), 1) for v in adjusted_rpy]} "
        f"z_offset_m={extra_z_m:.4f} "
        f"{top_surface_text}{fixed_z_text}{z_clamp_text}{rpy_text}"
        f"dir={[round(float(v), 4) for v in bias_dir.tolist()]} "
        f"signed_long_axis_m={signed_long_axis_m:.4f} "
        f"target_body_shift_m={target_body_shift_m:.4f} "
        f"applied_bias_m={applied_bias_m:.4f}"
    )
    return _replace_grasp_pose(grasp, position_xyz=adjusted_pos, rpy=adjusted_rpy)


def _execute_initial_right_pick_grasp(
    side,
    grasp,
    label=None,
    open_kwargs=None,
    close_kwargs=None,
    config=None,
):
    if side != "right":
        print(
            "[gpu_handover] Initial GPU pick executor only accepts the right arm; "
            f"got side={side!r}"
        )
        return None
    open_kwargs = dict(open_kwargs or {})
    close_kwargs = dict(close_kwargs or {})
    config = dict(config or {})
    final_pos = np.asarray(grasp.position, dtype=float).reshape(3)
    final_rpy = [float(v) for v in grasp.rpy]
    approach_pos = final_pos.copy()
    approach_pos[2] += max(0.0, float(RIGHT_PICK_APPROACH_CLEARANCE_M))
    planning_speed = float(config.get("planning_speed", MOVE_PLANNING_SPEED))

    open_gripper(side, **open_kwargs)
    print(
        "[gpu_handover] Initial right pick approach: "
        f"label={label or 'grasp'} "
        f"approach_pos={[round(float(v), 4) for v in approach_pos.tolist()]} "
        f"final_pos={[round(float(v), 4) for v in final_pos.tolist()]} "
        f"rpy={[round(float(v), 1) for v in final_rpy]} "
        f"clearance_m={float(RIGHT_PICK_APPROACH_CLEARANCE_M):.4f}"
    )
    try:
        _move_with_speed(
            side,
            approach_pos.tolist(),
            final_rpy,
            planning_speed=planning_speed,
        )
    except Exception as exc:
        print(
            "[gpu_handover] Initial right pick failed before descent: "
            f"approach move failed: {exc}"
        )
        return None
    try:
        _guided_side_pose_move(
            side,
            final_pos.tolist(),
            final_rpy,
            duration_s=float(RIGHT_PICK_APPROACH_DESCEND_DURATION_S),
            num_steps=int(RIGHT_PICK_APPROACH_DESCEND_STEPS),
        )
    except Exception as exc:
        print(
            "[gpu_handover] Initial right pick Cartesian descend failed; "
            f"falling back to direct move from approach pose: {exc}"
        )
        _move_with_speed(side, final_pos.tolist(), final_rpy, planning_speed=planning_speed)

    close_gripper(side, **close_kwargs)
    gripper_pos = _gripper_pos(side)
    print(f"[gpu_handover] Initial right pick gripper pos after close: {gripper_pos:.4f}")
    if gripper_pos > 0.0:
        print("[gpu_handover] Initial right pick grasp check passed")
        return True
    print("[gpu_handover] Initial right pick grasp check failed (gripper at/near zero)")
    open_gripper(side, **open_kwargs)
    return False


def _pick_and_stage_right_gpu(initial_scene_targets=None):
    print("[gpu_handover] Step 1: pick GPU")
    pick_config = dict(RUN_CONFIG)
    pick_config.pop("max_attempts", None)
    pick_config["planning_speed"] = float(
        RUN_CONFIG.get("planning_speed", MOVE_PLANNING_SPEED)
    )
    max_adjusted_pick_z = None
    if initial_scene_targets is not None:
        try:
            motherboard_top_z = float(initial_scene_targets.get("motherboard_top_z"))
            max_adjusted_pick_z = (
                motherboard_top_z
                + float(PICK_MAX_ADJUSTED_Z_ABOVE_MOTHERBOARD_TOP_M)
            )
            print(
                "[gpu_handover] Initial GPU-pick adjusted-Z filter: "
                f"motherboard_top_z={motherboard_top_z:.4f} "
                f"max_adjusted_pick_z={max_adjusted_pick_z:.4f}"
            )
        except Exception:
            max_adjusted_pick_z = None

    def _filter_pick_candidates(grasps, query=None, object_name=None, camera=None):
        return _filter_initial_gpu_pick_candidates_with_limits(
            grasps,
            query=query,
            object_name=object_name,
            camera=camera,
            max_adjusted_pick_z=max_adjusted_pick_z,
        )

    def _filter_pick_roi_candidates(grasps, query=None, object_name=None, camera=None):
        return _filter_initial_gpu_pick_candidates_with_limits(
            grasps,
            query=query,
            object_name=object_name,
            camera=camera,
            log_prefix="Initial GPU-pick ROI",
            max_top_normal_extent_m=PICK_ROI_BBOX_MAX_TOP_NORMAL_EXTENT_M,
            max_adjusted_pick_z=max_adjusted_pick_z,
        )

    def _adjust_pick_grasp_for_scene(grasp, side=None, label=None):
        return _adjust_initial_right_pick_grasp(
            grasp,
            side=side,
            label=label,
            max_adjusted_pick_z=max_adjusted_pick_z,
            canonicalize_yaw=RIGHT_PICK_CANONICALIZE_YAW,
        )

    def _adjust_pick_roi_grasp_for_scene(grasp, side=None, label=None):
        return _adjust_initial_right_pick_grasp(
            grasp,
            side=side,
            label=label,
            max_adjusted_pick_z=max_adjusted_pick_z,
            canonicalize_yaw=RIGHT_PICK_CANONICALIZE_YAW,
        )

    pick_kwargs = dict(
        grasp_mode="3d_bb",
        queries=GPU_QUERIES,
        max_attempts=INITIAL_PICK_MAX_ATTEMPTS,
        candidate_filter_fn=_filter_pick_candidates,
        selected_grasp_adjust_fn=_adjust_pick_grasp_for_scene,
        selected_grasp_execute_fn=_execute_initial_right_pick_grasp,
        **pick_config,
    )

    def _attempt_pick(
        image_bbox=None,
        queries=None,
        candidate_filter_fn=None,
        selected_grasp_adjust_fn=None,
    ):
        kwargs = dict(pick_kwargs)
        if image_bbox is not None:
            kwargs["image_bbox"] = image_bbox
        if queries is not None:
            kwargs["queries"] = queries
        if candidate_filter_fn is not None:
            kwargs["candidate_filter_fn"] = candidate_filter_fn
        if selected_grasp_adjust_fn is not None:
            kwargs["selected_grasp_adjust_fn"] = selected_grasp_adjust_fn
        return pick_object(OBJECT_NAME, **kwargs)

    pick_side = None
    initial_pick_error = None
    try:
        pick_side = _attempt_pick()
    except Exception as exc:
        initial_pick_error = exc
        print(
            "[gpu_handover] Step 1 initial GPU pick failed before ROI fallback: "
            f"{exc}"
        )
    roi_fallback_error = None
    if pick_side is None and PICK_MOTHERBOARD_RIGHT_ROI_FALLBACK:
        try:
            roi_bbox = _motherboard_right_gpu_pick_roi(
                initial_scene_targets,
                camera="top",
            )
            if roi_bbox is None:
                raise RuntimeError("initial motherboard scene bbox unavailable")
            print(
                "[gpu_handover] Step 1 fallback: retry GPU pick in "
                f"motherboard-right ROI bbox={roi_bbox}"
            )
            pick_side = _attempt_pick(
                image_bbox=roi_bbox,
                queries=GPU_PICK_RIGHT_ROI_QUERIES,
                candidate_filter_fn=_filter_pick_roi_candidates,
                selected_grasp_adjust_fn=_adjust_pick_roi_grasp_for_scene,
            )
        except Exception as exc:
            roi_fallback_error = exc
            print(
                "[gpu_handover] Step 1 motherboard-right ROI fallback unavailable: "
                f"{exc}"
            )
    fallback_error = None
    if pick_side is None and PICK_VLM_ROI_FALLBACK:
        try:
            gpu_bbox, width, height = _detect_gpu_bbox_2d(camera="top")
            roi_bbox = _expand_bbox_xyxy(
                gpu_bbox,
                width=width,
                height=height,
                margin_px=PICK_VLM_BBOX_MARGIN_PX,
            )
            print(
                "[gpu_handover] Step 1 fallback: retry GPU pick inside VLM ROI "
                f"bbox={gpu_bbox} expanded_bbox={roi_bbox}"
            )
            pick_side = _attempt_pick(image_bbox=roi_bbox)
        except Exception as exc:
            fallback_error = exc
            print(
                "[gpu_handover] Step 1 VLM ROI fallback unavailable: "
                f"{exc}"
            )
    if pick_side != "right":
        details = []
        if initial_pick_error is not None:
            details.append(f"initial_error={initial_pick_error}")
        if roi_fallback_error is not None:
            details.append(f"roi_fallback_error={roi_fallback_error}")
        if fallback_error is not None:
            details.append(f"fallback_error={fallback_error}")
        suffix = f": {'; '.join(details)}" if details else ""
        raise RuntimeError(f"pick failed{suffix}")

    state = get_robot_state()
    ee_pos = _robot_vec(state, pick_side, "ee_pos")
    ee_rpy_raw = _robot_vec(state, pick_side, "ee_rpy")
    handover_yaw = _front_handover_yaw_deg(float(ee_rpy_raw[2]))
    base_move_rpy = [0.0, 180.0, handover_yaw]
    try:
        move_rpy = _choose_right_handover_move_rpy(handover_yaw)
    except Exception as exc:
        move_rpy = list(base_move_rpy)
        print(
            "[gpu_handover] Right handover orientation preview fallback: "
            f"keeping nominal rpy={[round(float(v), 1) for v in move_rpy]} "
            f"because selection failed: {exc}"
        )
    print(
        f"[gpu_handover] Grasp EE pos={ee_pos} "
        f"rpy_raw={ee_rpy_raw} handover_yaw={handover_yaw:.1f} "
        f"nominal_move_rpy={base_move_rpy} clean_move_rpy={move_rpy}"
    )

    lift_pos = [ee_pos[0], ee_pos[1], LIFT_Z_M]
    print(f"[gpu_handover] Step 2: lift to {lift_pos}")
    _move_holding_right(lift_pos, move_rpy)

    print("[gpu_handover] Step 3: select and move directly to right-arm handover staging pose")
    right_stage_candidates = _build_right_handover_stage_candidates(move_rpy)
    try:
        best_right_stage, best_right_stage_src = _batch_select(
            "right",
            right_stage_candidates,
            label="right handover staging",
            planning_speed=HANDOVER_BATCH_PLANNING_SPEED,
        )
        _execute_batch_candidate(best_right_stage)
        print(
            "[gpu_handover] Right handover staging pose reached directly: "
            f"{best_right_stage_src.get('label', 'unknown')}"
        )
    except Exception as exc:
        print(
            "[gpu_handover] Right handover staging fallback: "
            f"hovering to handover transit pos {HANDOVER_POS} "
            f"because batch selection failed: {exc}"
        )
        _move_holding_right(HANDOVER_POS, move_rpy)

    state = get_robot_state()
    return {
        "move_rpy": move_rpy,
        "right_handover_pos": _robot_vec(state, "right", "ee_pos"),
        "right_handover_rpy": _robot_vec(state, "right", "ee_rpy"),
        "right_gripper_pos": _gripper_pos("right"),
    }


def _check_gpu_in_right_hand(label, ref_pos=None, grip_reference=None):
    state = get_robot_state()
    ref = np.asarray(
        _robot_vec(state, "right", "ee_pos") if ref_pos is None else ref_pos,
        dtype=float,
    )
    grip = _gripper_pos("right")
    grip_reference = None if grip_reference is None else float(grip_reference)
    grip_rel_threshold = (
        None
        if grip_reference is None
        else max(RIGHT_HOLD_MIN_GRIPPER_POS, grip_reference * RIGHT_HOLD_MIN_RATIO)
    )
    grip_abs_ok = grip > RIGHT_HOLD_MIN_GRIPPER_POS
    grip_rel_ok = grip_rel_threshold is None or grip >= grip_rel_threshold
    grip_ok = bool(grip_abs_ok and grip_rel_ok)

    try:
        detections = _sample_3d_bb_candidates_with_queries(GPU_QUERIES, camera="top")
    except Exception as exc:
        detections = []
        print(f"[gpu_handover] {label}: live GPU check could not detect any 3D-BB: {exc}")

    best = None
    best_in_hand = None
    best_drop = None
    for det in detections:
        grasp = det["grasp"]
        pos = np.asarray(getattr(grasp, "position", []), dtype=float).reshape(-1)
        if pos.size != 3:
            continue
        xy_dist = float(np.linalg.norm(pos[:2] - ref[:2]))
        below_ee_z = float(ref[2] - pos[2])
        z_dist = float(abs(pos[2] - ref[2]))
        xyz_dist = float(np.linalg.norm(pos - ref))
        candidate = dict(
            det,
            pos=pos,
            xy_dist=xy_dist,
            below_ee_z=below_ee_z,
            z_dist=z_dist,
            xyz_dist=xyz_dist,
        )
        candidate["table_like"] = bool(pos[2] <= GPU_TABLE_LIKE_Z_MAX_M)
        candidate["in_hand_like"] = bool(
            candidate["xy_dist"] <= RIGHT_IN_HAND_MAX_XY_DIST_M
            and candidate["below_ee_z"] <= RIGHT_IN_HAND_MAX_BELOW_EE_Z_M
            and not candidate["table_like"]
        )
        candidate["drop_like"] = bool(
            candidate["table_like"]
            and candidate["xy_dist"] <= RIGHT_DROP_NEAR_XY_DIST_M
            and candidate["below_ee_z"] >= RIGHT_DROP_MIN_BELOW_EE_Z_M
        )
        if best is None or (
            candidate["xy_dist"],
            max(candidate["below_ee_z"], 0.0),
            candidate["xyz_dist"],
        ) < (
            best["xy_dist"],
            max(best["below_ee_z"], 0.0),
            best["xyz_dist"],
        ):
            best = candidate
        if candidate["in_hand_like"] and (
            best_in_hand is None
            or (
                candidate["xy_dist"],
                candidate["xyz_dist"],
            )
            < (
                best_in_hand["xy_dist"],
                best_in_hand["xyz_dist"],
            )
        ):
            best_in_hand = candidate
        if candidate["drop_like"] and (
            best_drop is None
            or (
                candidate["xy_dist"],
                candidate["below_ee_z"],
            )
            < (
                best_drop["xy_dist"],
                best_drop["below_ee_z"],
            )
        ):
            best_drop = candidate

    hold_confirmed = bool(grip_ok and best_in_hand is not None)
    drop_detected = bool(best_drop is not None and best_in_hand is None)
    ok = bool(grip_ok and not drop_detected)

    if best is not None:
        print(
            f"[gpu_handover] {label}: grip={grip:.4f} grip_abs_ok={grip_abs_ok} "
            f"grip_rel_ok={grip_rel_ok} "
            f"grip_ref={None if grip_reference is None else round(grip_reference, 4)} "
            f"best_query={best['query']!r} "
            f"best_pos={[round(float(v), 4) for v in best['pos'].tolist()]} "
            f"xy_dist={best['xy_dist']:.4f} below_ee_z={best['below_ee_z']:.4f} "
            f"z_dist={best['z_dist']:.4f} xyz_dist={best['xyz_dist']:.4f} "
            f"in_hand_like={best['in_hand_like']} drop_like={best['drop_like']} "
            f"hold_confirmed={hold_confirmed} drop_detected={drop_detected} ok={ok}"
        )
    else:
        print(
            f"[gpu_handover] {label}: grip={grip:.4f} grip_abs_ok={grip_abs_ok} "
            f"grip_rel_ok={grip_rel_ok} "
            f"grip_ref={None if grip_reference is None else round(grip_reference, 4)} "
            f"hold_confirmed={hold_confirmed} drop_detected={drop_detected} "
            "no GPU candidate found"
        )

    if ok and hold_confirmed:
        reason = ""
    elif not grip_abs_ok:
        reason = f"right gripper hold looks empty (gripper_pos={grip:.4f})"
    elif not grip_rel_ok:
        reason = (
            "right gripper opened too much relative to the original pick "
            f"(current={grip:.4f}, required>={grip_rel_threshold:.4f})"
        )
    elif drop_detected:
        reason = (
            "vision saw the GPU below the right hand in a table-like pose "
            f"(xy={best_drop['xy_dist']:.4f}, below_ee_z={best_drop['below_ee_z']:.4f}, "
            f"z={float(best_drop['pos'][2]):.4f})"
        )
    elif best is None:
        reason = "vision could not confirm the GPU, but no drop evidence was seen"
    else:
        reason = (
            "vision did not positively lock the GPU to the right hand, "
            "but gripper hold is still strong"
        )

    return {
        "ok": ok,
        "reason": reason,
        "gripper_pos": grip,
        "grip_ok": grip_ok,
        "hold_confirmed": hold_confirmed,
        "drop_detected": drop_detected,
        "best_candidate": None if best_in_hand is None else best_in_hand["grasp"],
        "best_query": None if best_in_hand is None else best_in_hand["query"],
        "best_xy_dist_m": None if best is None else best["xy_dist"],
        "best_z_dist_m": None if best is None else best["z_dist"],
        "best_below_ee_z_m": None if best is None else best["below_ee_z"],
        "ref_pos": [float(v) for v in ref.tolist()],
    }


def _double_check_gpu_in_right_hand(ref_pos=None, grip_reference=None):
    accepted = None
    checks = []
    for check_index in (1, 2):
        check = _check_gpu_in_right_hand(
            label=f"pre-handover hold check {check_index}/2",
            ref_pos=ref_pos,
            grip_reference=grip_reference,
        )
        checks.append(check)
        if check["hold_confirmed"] and accepted is None:
            accepted = check
        if not check["grip_ok"] or check["drop_detected"]:
            return check
    if accepted is not None:
        accepted["ok"] = True
        return accepted
    if all(check["grip_ok"] and not check["drop_detected"] for check in checks):
        print(
            "[gpu_handover] pre-handover hold verification: "
            "accepting based on stable gripper hold across both checks"
        )
        fallback = dict(checks[-1])
        fallback["ok"] = True
        fallback["reason"] = ""
        return fallback
    return {
        "ok": False,
        "reason": "internal error: hold check did not run",
        "best_candidate": None,
    }


def _prepare_for_right_repicking(reason, clear_rpy=None):
    print(f"[gpu_handover] Hold verification failed before handover: {reason}")
    try:
        _open_gripper("right", vel_limit=RECOVERY_OPEN_VEL_LIMIT)
    except Exception as exc:
        print(f"[gpu_handover] open_gripper('right') during recovery failed: {exc}")
    try:
        _open_gripper("left", vel_limit=LEFT_REGRASP_OPEN_VEL_LIMIT)
    except Exception:
        pass
    if clear_rpy is None or len(clear_rpy) != 3:
        try:
            clear_rpy = _robot_vec(get_robot_state(), "right", "ee_rpy")
        except Exception:
            clear_rpy = [0.0, 180.0, 0.0]
    print(
        "[gpu_handover] Local repick recovery: moving empty right arm to clear pose "
        f"{RIGHT_RETRY_CLEAR_POS}"
    )
    try:
        _move("right", RIGHT_RETRY_CLEAR_POS, clear_rpy)
    except Exception as exc:
        print(
            "[gpu_handover] Local right-arm clear move failed; "
            f"retrying pick from current pose: {exc}"
        )


def _confirm_left_regrasp_stable():
    def _sample_status(samples):
        sample_min = float(min(samples))
        sample_max = float(max(samples))
        sample_delta = float(sample_max - sample_min)
        contact_ok = sample_min > float(LEFT_REGRASP_STABLE_MIN_GRIPPER_POS)
        width_ok = sample_max <= float(LEFT_REGRASP_RELEASE_MAX_GRIPPER_POS)
        settled_ok = sample_delta <= float(LEFT_REGRASP_RELEASE_MAX_GRIPPER_DELTA)
        ready = bool(contact_ok and width_ok and settled_ok)
        return sample_min, sample_max, sample_delta, contact_ok, width_ok, settled_ok, ready

    for attempt in range(1, LEFT_REGRASP_CONFIRM_ATTEMPTS + 1):
        samples = []
        for poll_index in range(LEFT_REGRASP_SETTLE_POLLS):
            samples.append(_gripper_pos("left"))
            if poll_index + 1 < LEFT_REGRASP_SETTLE_POLLS:
                time.sleep(LEFT_REGRASP_SETTLE_S / LEFT_REGRASP_SETTLE_POLLS)
        (
            min_gp,
            max_gp,
            delta_gp,
            contact_ok,
            width_ok,
            settled_ok,
            ready,
        ) = _sample_status(samples)
        print(
            "[gpu_handover] Left handover hold confirm "
            f"{attempt}/{LEFT_REGRASP_CONFIRM_ATTEMPTS}: "
            f"samples={[round(float(v), 4) for v in samples]} "
            f"min={min_gp:.4f} "
            f"max={max_gp:.4f} "
            f"delta={delta_gp:.4f} "
            f"contact_ok={contact_ok} "
            f"width_ok={width_ok} "
            f"settled_ok={settled_ok} "
            f"release_ready={ready}"
        )
        if ready:
            dwell_samples = []
            for poll_index in range(LEFT_PRE_RELEASE_DWELL_POLLS):
                dwell_samples.append(_gripper_pos("left"))
                if poll_index + 1 < LEFT_PRE_RELEASE_DWELL_POLLS:
                    time.sleep(
                        LEFT_PRE_RELEASE_DWELL_S / LEFT_PRE_RELEASE_DWELL_POLLS
                    )
            (
                dwell_min_gp,
                dwell_max_gp,
                dwell_delta_gp,
                dwell_contact_ok,
                dwell_width_ok,
                dwell_settled_ok,
                dwell_ready,
            ) = _sample_status(dwell_samples)
            print(
                "[gpu_handover] Left pre-release dwell: "
                f"samples={[round(float(v), 4) for v in dwell_samples]} "
                f"min={dwell_min_gp:.4f} "
                f"max={dwell_max_gp:.4f} "
                f"delta={dwell_delta_gp:.4f} "
                f"contact_ok={dwell_contact_ok} "
                f"width_ok={dwell_width_ok} "
                f"settled_ok={dwell_settled_ok} "
                f"dwell_ok={dwell_ready}"
            )
            if dwell_ready:
                return float(dwell_samples[-1])
        if attempt < LEFT_REGRASP_CONFIRM_ATTEMPTS:
            print(
                "[gpu_handover] Left hold is not ready for right release; keeping right hand closed "
                "and re-closing left gripper before release"
            )
            _close_gripper(
                "left",
                vel_limit=LEFT_REGRASP_CLOSE_VEL_LIMIT,
                torque_limit=LEFT_REGRASP_CLOSE_TORQUE_LIMIT,
            )
    _open_gripper("left", vel_limit=LEFT_REGRASP_OPEN_VEL_LIMIT)
    raise RuntimeError(
        "left-arm handover grasp was not ready before right release; "
        f"required contact>{LEFT_REGRASP_STABLE_MIN_GRIPPER_POS:.4f}, "
        f"not wide-open max_gripper<={LEFT_REGRASP_RELEASE_MAX_GRIPPER_POS:.4f}, "
        f"delta<={LEFT_REGRASP_RELEASE_MAX_GRIPPER_DELTA:.4f}"
    )


def _best_effort_go_home(context):
    try:
        go_home()
        return True
    except Exception as exc:
        print(f"[gpu_handover] go_home failed during {context}: {exc}")
        return False


def _locally_reorient_left_held_gpu(left_insertion_rpy, context):
    reorient_pos = _robot_vec(get_robot_state(), "left", "ee_pos")
    print(
        f"[gpu_handover] {context}: guided local left-hand insertion reorientation "
        f"at pos={[round(float(v), 4) for v in reorient_pos]} "
        f"target_rpy={[round(float(v), 1) for v in left_insertion_rpy]}"
    )
    _guided_side_pose_move(
        "left",
        reorient_pos,
        left_insertion_rpy,
    )
    return [float(v) for v in reorient_pos], [float(v) for v in left_insertion_rpy]


success = False
left_holding_gpu = False
final_hover_pos = None
final_hover_rpy = None


def get_task_info():
    return {
        "success": bool(success),
        "reward": 1.0 if success else 0.0,
        "left_holding_gpu": bool(left_holding_gpu),
        "target_socket": int(TARGET_SOCKET_NUMBER),
        "final_hover_pos": final_hover_pos,
        "final_hover_rpy": final_hover_rpy,
        "go_home_on_exit": bool(GO_HOME_ON_EXIT),
    }

try:
    stage_info = None
    hold_check = None
    held_gpu_grasp = None
    initial_scene_targets = None
    if GO_HOME_ON_START:
        print("[gpu_handover] Step -1: go home before starting a new run")
        if not _best_effort_go_home("script start"):
            raise RuntimeError("could not go home at script start")
    else:
        print(
            "[gpu_handover] Step -1: skipping go_home before starting "
            "(GPU_HANDOVER_GO_HOME_ON_START=0)"
        )
    if ENABLE_SOCKET_HOVER and USE_INITIAL_SCENE_SOCKET_TARGET:
        print(
            "[gpu_handover] Step 0: detect and cache initial motherboard/socket targets before pick "
            f"using camera={SOCKET_HOVER_TRACK_CAMERA!r}"
        )
        try:
            initial_scene_targets = _capture_initial_scene_targets(
                camera=SOCKET_HOVER_TRACK_CAMERA
            )
        except Exception as exc:
            print(
                "[gpu_handover] Initial scene cache failed: "
                f"{exc}. Step 8 will keep the insertion-ready pose from Step 7."
            )
    for repick_round in range(PRE_HANDOVER_REPICK_LIMIT + 1):
        if repick_round > 0:
            print(
                f"[gpu_handover] Recovery repick attempt {repick_round}/"
                f"{PRE_HANDOVER_REPICK_LIMIT}"
            )
        stage_info = _pick_and_stage_right_gpu(
            initial_scene_targets=initial_scene_targets,
        )
        hold_check = _double_check_gpu_in_right_hand(
            ref_pos=stage_info["right_handover_pos"],
            grip_reference=stage_info["right_gripper_pos"],
        )
        if hold_check["ok"]:
            held_gpu_grasp = hold_check["best_candidate"]
            break
        if repick_round >= PRE_HANDOVER_REPICK_LIMIT:
            raise RuntimeError(
                "GPU could not be verified in the right hand before handover: "
                f"{hold_check['reason']}"
            )
        _prepare_for_right_repicking(
            hold_check["reason"],
            clear_rpy=stage_info["move_rpy"],
        )

    if stage_info is None:
        raise RuntimeError("GPU handover staging did not execute")

    right_handover_pos = list(stage_info["right_handover_pos"])
    right_handover_rpy = list(stage_info["right_handover_rpy"])
    move_rpy = list(stage_info["move_rpy"])

    print("[gpu_handover] Step 4: estimate held GPU pose for left-arm board-face grasp")
    handover_geometry = _select_handover_geometry(
        held_gpu_grasp,
        right_handover_pos,
        right_handover_rpy,
    )
    left_pregrasp_candidates, left_grasp_candidates = _build_left_board_face_candidates(
        handover_geometry,
        right_handover_pos=right_handover_pos,
    )

    print("[gpu_handover] Step 5: left-arm regrasp")
    _open_gripper("left", vel_limit=LEFT_REGRASP_OPEN_VEL_LIMIT)
    best_pregrasp, _best_pregrasp_src = _batch_select(
        "left",
        left_pregrasp_candidates,
        label="left board-face pregrasp",
    )
    pregrasp_pair_id = str(_best_pregrasp_src.get("pair_id", "")).strip()
    paired_left_grasp_candidates = [
        candidate
        for candidate in left_grasp_candidates
        if str(candidate.get("pair_id", "")).strip() == pregrasp_pair_id
    ]
    pregrasp_reached = False
    if float(LEFT_REGRASP_PREGRASP_TRANSIT_LIFT_M) > 1e-6:
        pregrasp_pos = np.asarray(getattr(best_pregrasp, "position", []), dtype=float).reshape(3)
        pregrasp_rpy = [float(v) for v in getattr(best_pregrasp, "rpy", [])]
        current_left_pos = _robot_vec(get_robot_state(), "left", "ee_pos")
        low_descend_pos = pregrasp_pos.copy()
        late_offset_m = max(0.0, float(LEFT_REGRASP_PREGRASP_LATE_DESCENT_M))
        if late_offset_m > 1e-6:
            approach_dir = np.asarray(
                _best_pregrasp_src.get("approach_world", [0.0, -1.0, 0.0]),
                dtype=float,
            ).reshape(3)
            approach_dir[2] = 0.0
            paired_grasp_pos = None
            if paired_left_grasp_candidates:
                paired_grasp_pos = np.asarray(
                    paired_left_grasp_candidates[0].get("position", []),
                    dtype=float,
                ).reshape(3)
                to_grasp = paired_grasp_pos - pregrasp_pos
                if float(np.linalg.norm(to_grasp[:2])) > 1e-6:
                    approach_dir = to_grasp
                    approach_dir[2] = 0.0
                    max_late_offset_m = max(
                        0.0,
                        float(np.linalg.norm(to_grasp[:2]))
                        - max(0.0, float(LEFT_REGRASP_FINAL_APPROACH_MIN_M)),
                    )
                    late_offset_m = min(late_offset_m, max_late_offset_m)
            approach_norm = float(np.linalg.norm(approach_dir[:2]))
            if approach_norm > 1e-6 and late_offset_m > 1e-6:
                low_descend_pos[:2] += (approach_dir[:2] / approach_norm) * late_offset_m
        transit_pos = [float(v) for v in low_descend_pos.tolist()]
        transit_pos[2] = max(
            float(transit_pos[2] + LEFT_REGRASP_PREGRASP_TRANSIT_LIFT_M),
            float(current_left_pos[2]),
        )
        print(
            "[gpu_handover] Step 5a: high late left pregrasp transit to avoid inserted GPU "
            f"current_pos={[round(float(v), 4) for v in current_left_pos]} "
            f"transit_pos={[round(float(v), 4) for v in transit_pos]} "
            f"selected_pregrasp_pos={[round(float(v), 4) for v in pregrasp_pos.tolist()]} "
            f"late_descent_offset_m={late_offset_m:.4f}"
        )
        _move_with_speed(
            "left",
            transit_pos,
            pregrasp_rpy,
            planning_speed=POST_PICK_PLANNING_SPEED,
        )
        print(
            "[gpu_handover] Step 5b: descend near handover before final left approach "
            "with a fresh plan"
        )
        _move_with_speed(
            "left",
            low_descend_pos.tolist(),
            pregrasp_rpy,
            planning_speed=POST_PICK_PLANNING_SPEED,
        )
        pregrasp_reached = True
    if not pregrasp_reached:
        _execute_batch_candidate(best_pregrasp)
    if paired_left_grasp_candidates:
        grasp_selection_candidates = paired_left_grasp_candidates
        print(
            "[gpu_handover] Left final approach paired to selected pregrasp: "
            f"pair_id={pregrasp_pair_id}"
        )
    else:
        grasp_selection_candidates = left_grasp_candidates
        print(
            "[gpu_handover] Left final approach pairing unavailable; "
            "falling back to all grasp candidates"
        )
    best_grasp, best_grasp_src = _batch_select(
        "left",
        grasp_selection_candidates,
        label="left board-face grasp",
    )
    if LEFT_REGRASP_GUIDED_APPROACH:
        _guided_side_pose_move(
            "left",
            getattr(best_grasp, "position", []),
            getattr(best_grasp, "rpy", []),
            duration_s=LEFT_REGRASP_GUIDED_APPROACH_DURATION_S,
            num_steps=LEFT_REGRASP_GUIDED_APPROACH_STEPS,
        )
    else:
        _execute_batch_candidate(best_grasp)
    _close_gripper(
        "left",
        vel_limit=LEFT_REGRASP_CLOSE_VEL_LIMIT,
        torque_limit=LEFT_REGRASP_CLOSE_TORQUE_LIMIT,
    )
    left_gp = _gripper_pos("left")
    print(f"[gpu_handover] Left gripper pos after regrasp: {left_gp:.4f}")
    if left_gp <= LEFT_REGRASP_MIN_GRIPPER_POS:
        _open_gripper("left", vel_limit=LEFT_REGRASP_OPEN_VEL_LIMIT)
        raise RuntimeError(
            "left-arm handover grasp did not register contact; "
            f"gripper_pos={left_gp:.4f}"
        )
    left_gp = _confirm_left_regrasp_stable()
    print(
        "[gpu_handover] Left handover grasp verified before right release: "
        f"gripper_pos={left_gp:.4f}"
    )
    left_hold_rpy = [float(v) for v in best_grasp_src.get("rpy", [])]
    if len(left_hold_rpy) != 3:
        left_hold_rpy = _robot_vec(get_robot_state(), "left", "ee_rpy")
    left_insertion_rpy = [float(v) for v in best_grasp_src.get("insertion_hold_rpy", [])]
    if len(left_insertion_rpy) != 3:
        left_insertion_rpy = list(left_hold_rpy)
    left_axis_kind = str(best_grasp_src.get("axis_kind", "")).strip().lower() or "unknown"
    print(
        "[gpu_handover] Left post-handover orientations: "
        f"axis_kind={left_axis_kind} "
        f"hold_rpy={[round(v, 1) for v in left_hold_rpy]} "
        f"insertion_rpy={[round(v, 1) for v in left_insertion_rpy]}"
    )

    print("[gpu_handover] Step 6: right-arm release, left-hand re-clamp, and retreat")
    _open_gripper("right", vel_limit=RIGHT_RELEASE_OPEN_VEL_LIMIT)
    if LEFT_POST_RELEASE_RECLAMP:
        _close_gripper(
            "left",
            vel_limit=LEFT_POST_RELEASE_RECLAMP_VEL_LIMIT,
            torque_limit=LEFT_POST_RELEASE_RECLAMP_TORQUE_LIMIT,
        )
        left_gp = _gripper_pos("left")
        print(
            "[gpu_handover] Left gripper pos after post-release re-clamp: "
            f"{left_gp:.4f}"
        )
        if left_gp <= LEFT_REGRASP_MIN_GRIPPER_POS:
            raise RuntimeError(
                "left-arm grip was lost during post-release re-clamp; "
                f"gripper_pos={left_gp:.4f}"
            )
    else:
        left_gp = _gripper_pos("left")
        if left_gp <= LEFT_REGRASP_MIN_GRIPPER_POS:
            raise RuntimeError(
                "left-arm grip was lost immediately after right release; "
                f"gripper_pos={left_gp:.4f}"
            )
    left_holding_gpu = True
    print(
        "[gpu_handover] Left hold confirmed after right release: "
        f"gripper_pos={left_gp:.4f}"
    )
    right_release_pos = _robot_vec(get_robot_state(), "right", "ee_pos")
    right_lift_pos = [
        float(right_release_pos[0]),
        float(right_release_pos[1]),
        float(right_release_pos[2] + RIGHT_POST_RELEASE_LIFT_M),
    ]
    print(
        "[gpu_handover] Step 6a: lift released right arm clear of left-held GPU "
        f"to {right_lift_pos}"
    )
    try:
        _cartesian_retract_up_after_release(
            "right",
            RIGHT_POST_RELEASE_LIFT_M,
        )
    except Exception as exc:
        print(
            "[gpu_handover] Cartesian retract fallback: "
            f"using freespace lift because direct retract failed: {exc}"
        )
        _move_with_speed(
            "right",
            right_lift_pos,
            move_rpy,
            planning_speed=POST_PICK_PLANNING_SPEED,
        )
    print(
        "[gpu_handover] Step 6b: send released right arm directly to joint-home "
        "after vertical clear"
    )
    _move_side_to_joint_home(
        "right",
        close_gripper_after=RIGHT_POST_RELEASE_CLOSE,
    )

    print("[gpu_handover] Step 7: lift left-held GPU before insertion-ready motion")
    left_current_pos = _robot_vec(get_robot_state(), "left", "ee_pos")
    left_current_rpy = _robot_vec(get_robot_state(), "left", "ee_rpy")
    reorient_pos = [
        float(left_current_pos[0]),
        float(left_current_pos[1]),
        float(left_current_pos[2] + POST_HANDOVER_HOVER_MIN_LIFT_M),
    ]
    print(
        "[gpu_handover] Step 7a: lift left-held GPU before reorientation "
        f"from pos={[round(float(v), 4) for v in left_current_pos]} "
        f"to pos={reorient_pos} "
        f"keeping_rpy={[round(float(v), 1) for v in left_current_rpy]}"
    )
    _cartesian_retract_up_after_release(
        "left",
        float(reorient_pos[2] - float(left_current_pos[2])),
    )
    left_lifted_pos = _robot_vec(get_robot_state(), "left", "ee_pos")
    reorient_pos[0] = float(left_lifted_pos[0])
    reorient_pos[1] = float(left_lifted_pos[1])
    reorient_pos[2] = float(left_lifted_pos[2])
    left_lifted_rpy = _robot_vec(get_robot_state(), "left", "ee_rpy")
    combined_reorient_with_hover = bool(
        ENABLE_SOCKET_HOVER and POST_HANDOVER_COMBINE_REORIENT_WITH_HOVER
    )
    final_hover_pos = [float(v) for v in reorient_pos]
    final_hover_rpy = [float(v) for v in left_lifted_rpy]
    if combined_reorient_with_hover:
        print(
            "[gpu_handover] Step 7b: defer insertion reorientation into the guided "
            "socket-hover move so translation and rotation happen together "
            f"target_rpy={[round(float(v), 1) for v in left_insertion_rpy]}"
        )
        if initial_scene_targets is not None:
            try:
                print(
                    "[gpu_handover] Step 7c: coarse cached socket-hover move while "
                    "reorienting left-held GPU; Step 8 will refine with fresh vision"
                )
                cached_hover_pos = _move_to_cached_socket_hover_pose(
                    "left",
                    left_insertion_rpy,
                    initial_scene_targets,
                    left_axis_kind,
                    guided=True,
                    guided_duration_s=float(SOCKET_HOVER_INITIAL_GUIDED_DURATION_S),
                    guided_steps=int(SOCKET_HOVER_INITIAL_GUIDED_STEPS),
                )
                final_hover_pos = [float(v) for v in cached_hover_pos]
                final_hover_rpy = [float(v) for v in left_insertion_rpy]
                success = True
            except Exception as exc:
                print(
                    "[gpu_handover] Step 7c cached hover/reorientation skipped: "
                    f"{exc}. Step 8 will use fresh vision from the current lifted pose."
                )
        else:
            print(
                "[gpu_handover] Step 7c cached hover/reorientation unavailable: "
                "no initial socket scene cache"
            )
    else:
        final_hover_pos, final_hover_rpy = _locally_reorient_left_held_gpu(
            left_insertion_rpy,
            "Step 7b",
        )
        success = True
    if ENABLE_SOCKET_HOVER:
        print("[gpu_handover] Step 8: detect motherboard socket and hover reoriented GPU")
        try:
            reactive_plane_z = None
            reactive_hover_rpy = None
            step8_hover_rpy = (
                [float(v) for v in EXPLICIT_SOCKET_HOVER_RPY]
                if EXPLICIT_SOCKET_HOVER_RPY is not None
                else [float(v) for v in left_insertion_rpy]
            )
            hover = _load_reset_aligned_slot_hover_helpers()
            print(
                "[gpu_handover] Step 8 uses reset-aligned socket hover helper: "
                f"camera={hover.CAMERA!r} aux_camera={hover.AUX_CAMERA!r} "
                f"aux_prefer_world_pose={bool(hover.AUX_CAMERA_PREFER_WORLD_POSE)}"
            )
            reference_scene_targets = hover._build_initial_reference_scene(
                camera=hover.CAMERA
            )
            hover_pos, _ = hover._acquire_initial_hover(
                reference_scene=reference_scene_targets,
                hover_rpy=step8_hover_rpy,
                camera=hover.CAMERA,
            )
            final_hover_pos = [float(v) for v in hover_pos]
            final_hover_rpy = [float(v) for v in step8_hover_rpy]

            reactive_plane_z = float(reference_scene_targets["hover_z"])
            reactive_hover_rpy = [float(v) for v in step8_hover_rpy]
            print(
                "[gpu_handover] Step 8b: lock reactive hover pose "
                f"z={reactive_plane_z:.4f} "
                f"rpy={[round(float(v), 1) for v in reactive_hover_rpy]}"
            )
            success = True
            if REACTIVE_SOCKET_HOVER:
                hover._run_reactive_hover_loop(
                    reference_scene=reference_scene_targets,
                    reactive_hover_rpy=reactive_hover_rpy,
                    reactive_plane_z=reactive_plane_z,
                    camera=hover.CAMERA,
                )
        except Exception as exc:
            if combined_reorient_with_hover:
                print(
                    "[gpu_handover] Step 8 skipped before combined hover/reorientation completed: "
                    f"{exc}. Falling back to local insertion reorientation."
                )
                final_hover_pos, final_hover_rpy = _locally_reorient_left_held_gpu(
                    left_insertion_rpy,
                    "Step 8 fallback",
                )
                success = True
            else:
                print(
                    "[gpu_handover] Step 8 skipped: "
                    f"{exc}. Keeping insertion-ready pose from Step 7."
                )
            print(
                "[gpu_handover] Success: left arm is holding the GPU "
                "in an insertion-ready hover pose"
            )
        else:
            print(
                "[gpu_handover] Success: left arm is holding the GPU "
                "above the motherboard socket in insertion-ready orientation "
                f"at pos={[round(float(v), 4) for v in hover_pos]} "
                f"target_socket={int(TARGET_SOCKET_NUMBER)}"
            )
    else:
        print(
            "[gpu_handover] Step 8 disabled via GPU_ENABLE_SOCKET_HOVER=0. "
            "Keeping insertion-ready pose from Step 7."
        )
        print(
            "[gpu_handover] Success: left arm is holding the GPU "
            "in an insertion-ready hover pose"
        )

finally:
    _stop_motherboard_tracking(initial_scene_targets)
    if success and not GO_HOME_ON_EXIT:
        print(
            "[gpu_handover] Leaving robot in final hover pose. "
            "Set GPU_HANDOVER_GO_HOME_ON_EXIT=1 to auto-home instead."
        )
    elif left_holding_gpu and not GO_HOME_ON_EXIT:
        print(
            "[gpu_handover] Post-handover failure: keeping current left-hand hold pose "
            "instead of auto-homing with the GPU in hand."
        )
    else:
        _best_effort_go_home("script cleanup")

