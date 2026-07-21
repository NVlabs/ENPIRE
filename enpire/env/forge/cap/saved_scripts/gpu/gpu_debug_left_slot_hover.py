"""Standalone left-arm motherboard slot hover tracker.

This script debugs only the final hover behavior:
  1. Detect the motherboard from the top camera.
  2. Build a fixed motherboard reference once from SAM3 + depth.
  3. Define slot hover geometry in the motherboard frame.
  4. Move only the left arm to the hover pose.
  5. Keep updating the hover pose if the motherboard translates.

It does not pick the GPU, does not move the right arm, and does not use the
initial VLM slot query path.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default if raw is None or raw == "" else raw)


def _parse_optional_rpy(raw: str | None):
    if raw is None or raw.strip() == "":
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise RuntimeError(
            "GPU_SLOT_DEBUG_HOVER_RPY must contain exactly three comma-separated values"
    )
    return [float(part) for part in parts]


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return tuple(float(value) for value in default)
    values = [float(part) for part in raw.replace(",", " ").split()]
    if len(values) != len(default):
        raise RuntimeError(
            f"{name} must contain {len(default)} numeric values, got {raw!r}"
        )
    return tuple(values)


import enpire.env.forge.cap.agent.tools.segmentation as _segmentation_tools


_ORIGINAL_SEGMENT_LOG_MASK = _segmentation_tools.log_mask

if not _env_flag("GPU_SLOT_DEBUG_SAVE_SEGMENT_LOGS", True):
    _segmentation_tools.log_mask = lambda *args, **kwargs: None


def _load_handover_helpers():
    try:
        skill_sorting = importlib.import_module("skill_library.constants.sorting")
        skill_constants = importlib.import_module("skill_library.constants")
        skill_library = importlib.import_module("skill_library")
    except Exception:
        skill_library = sys.modules.setdefault(
            "skill_library",
            types.ModuleType("skill_library"),
        )
        skill_constants = sys.modules.setdefault(
            "skill_library.constants",
            types.ModuleType("skill_library.constants"),
        )
        skill_sorting = sys.modules.setdefault(
            "skill_library.constants.sorting",
            types.ModuleType("skill_library.constants.sorting"),
        )
        if not hasattr(skill_sorting, "TABLE_SORT_RUN_CONFIG"):
            skill_sorting.TABLE_SORT_RUN_CONFIG = {}
    skill_namespace = sys.modules.setdefault(
        "skill_library.namespace",
        types.ModuleType("skill_library.namespace"),
    )
    skill_pick_place = sys.modules.setdefault(
        "skill_library.pick_place",
        types.ModuleType("skill_library.pick_place"),
    )
    skill_library.constants = skill_constants
    skill_constants.sorting = skill_sorting
    skill_library.namespace = skill_namespace
    skill_library.pick_place = skill_pick_place
    for name in [
        "close_gripper",
        "freespace_move",
        "get_camera_extrinsics",
        "get_camera_image",
        "get_camera_intrinsics",
        "get_robot_state",
        "go_home",
        "render_depth",
        "segment_object",
    ]:
        if name in globals():
            setattr(skill_namespace, name, globals()[name])
    if not hasattr(skill_pick_place, "pick_object"):
        def _unused_pick_object(*args, **kwargs):
            raise RuntimeError("pick_object is unavailable in left-slot hover debug mode")

        skill_pick_place.pick_object = _unused_pick_object

    script_path = Path.cwd() / "cap" / "saved_scripts" / "gpu" / "gpu_handover.py"
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
            # Keep only the module docstring; drop other top-level expressions.
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    helpers = types.ModuleType("gpu_handover_helpers")
    helpers.__file__ = str(script_path)
    helpers.__dict__.update(
        {
            "__builtins__": __builtins__,
            "ast": ast,
            "json": json,
            "np": np,
            "os": os,
            "Path": Path,
            "Rotation": Rotation,
            "sys": sys,
            "time": time,
            "types": types,
        }
    )
    exec(compile(module, str(script_path), "exec"), helpers.__dict__)  # noqa: S102
    return helpers


gh = _load_handover_helpers()

CAMERA = os.environ.get("GPU_SLOT_DEBUG_CAMERA", "top").strip() or "top"
AUX_CAMERA = os.environ.get("GPU_SLOT_DEBUG_AUX_CAMERA", "left_third").strip().lower() or None
if AUX_CAMERA == CAMERA:
    AUX_CAMERA = None
AUX_CAMERA_PREFER_WORLD_POSE = _env_flag(
    "GPU_SLOT_DEBUG_AUX_CAMERA_PREFER_WORLD_POSE",
    True,
)
AUX_CAMERA_REQUIRED = _env_flag(
    "GPU_SLOT_DEBUG_REQUIRE_AUX_CAMERA",
    bool(AUX_CAMERA is not None and AUX_CAMERA_PREFER_WORLD_POSE),
)
CLOSE_LEFT_GRIPPER_ON_START = _env_flag(
    "GPU_SLOT_DEBUG_CLOSE_LEFT_GRIPPER_ON_START",
    True,
)
GO_HOME_ON_START = _env_flag(
    "GPU_SLOT_DEBUG_GO_HOME_ON_START",
    False,
)
BOARD_PLANE_Z_OFFSET_M = _env_float(
    "GPU_SLOT_DEBUG_BOARD_PLANE_Z_OFFSET_M",
    float(gh.SOCKET_HOVER_BOARD_PLANE_Z_OFFSET_M),
)
TARGET_SOCKET_NUMBER = max(
    1,
    min(3, int(_env_float("GPU_TARGET_SOCKET_NUMBER", 1))),
)
SOCKET_TARGET_RIGHT_EDGE_OFFSETS_M = _env_float_tuple(
    "GPU_SOCKET_TARGET_RIGHT_EDGE_OFFSETS_M",
    tuple(float(v) for v in gh.SOCKET_TARGET_RIGHT_EDGE_OFFSETS_M),
)
SOCKET_HOVER_X_OFFSET_M = _env_float(
    "GPU_SOCKET_HOVER_X_OFFSET_M",
    -0.04,
)
SOCKET_HOVER_Y_TRIM_M = _env_float(
    "GPU_SOCKET_HOVER_Y_TRIM_M",
    0.01,
)
EXPLICIT_HOVER_RPY = _parse_optional_rpy(os.environ.get("GPU_SLOT_DEBUG_HOVER_RPY"))
DEFAULT_HOVER_RPY = [0.0, -185.0, 0.0]
HOVER_CLEARANCE_M = _env_float(
    "GPU_SLOT_DEBUG_HOVER_CLEARANCE_M",
    max(0.0, float(gh.SOCKET_HOVER_CLEARANCE_M) + 0.01),
)
HOVER_MIN_Z = _env_float(
    "GPU_SLOT_DEBUG_HOVER_MIN_Z",
    0.0,
)
PRIMARY_CAMERA_MIN_SCORE = _env_float(
    "GPU_SLOT_DEBUG_PRIMARY_CAMERA_MIN_SCORE",
    0.55,
)
AUX_CAMERA_MIN_SCORE = _env_float(
    "GPU_SLOT_DEBUG_AUX_CAMERA_MIN_SCORE",
    0.55,
)
AUX_CAMERA_BLEND_MAX_SHIFT_M = _env_float(
    "GPU_SLOT_DEBUG_AUX_CAMERA_BLEND_MAX_SHIFT_M",
    0.03,
)
AUX_CAMERA_MAX_POSE_DIFF_M = _env_float(
    "GPU_SLOT_DEBUG_AUX_CAMERA_MAX_POSE_DIFF_M",
    0.12,
)
MIN_MOVE_M = _env_float("GPU_SLOT_DEBUG_MIN_MOVE_M", 0.008)
HOVER_Z_TOL_M = _env_float("GPU_SLOT_DEBUG_HOVER_Z_TOL_M", 0.005)
CENTER_UPDATE_MIN_SHIFT_M = _env_float(
    "GPU_SLOT_DEBUG_CENTER_UPDATE_MIN_SHIFT_M",
    0.004,
)
CENTER_UPDATE_MAX_SHIFT_M = _env_float(
    "GPU_SLOT_DEBUG_CENTER_UPDATE_MAX_SHIFT_M",
    0.05,
)
SAVE_DEBUG_ARTIFACTS = _env_flag("GPU_SLOT_DEBUG_SAVE_ARTIFACTS", False)
REACTIVE_GUIDED_DURATION_S = _env_float(
    "GPU_SLOT_DEBUG_REACTIVE_GUIDED_DURATION_S",
    0.18,
)
REACTIVE_GUIDED_STEPS = max(
    2,
    int(_env_float("GPU_SLOT_DEBUG_REACTIVE_GUIDED_STEPS", 6)),
)
GUIDED_MAX_XY_SPEED_MPS = _env_float(
    "GPU_SLOT_DEBUG_GUIDED_MAX_XY_SPEED_MPS",
    0.20,
)
GUIDED_MAX_Z_SPEED_MPS = _env_float(
    "GPU_SLOT_DEBUG_GUIDED_MAX_Z_SPEED_MPS",
    0.10,
)
INITIAL_GUIDED_DURATION_S = _env_float(
    "GPU_SLOT_DEBUG_INITIAL_GUIDED_DURATION_S",
    1.2,
)
INITIAL_GUIDED_STEPS = max(
    2,
    int(_env_float("GPU_SLOT_DEBUG_INITIAL_GUIDED_STEPS", 12)),
)
INITIAL_HOVER_MAX_ATTEMPTS = max(
    3,
    int(_env_float("GPU_SLOT_DEBUG_INITIAL_HOVER_MAX_ATTEMPTS", 5)),
)
MAX_HOVER_XY_DELTA_M = _env_float(
    "GPU_SLOT_DEBUG_MAX_HOVER_XY_DELTA_M",
    0.24,
)
MIN_MOTHERBOARD_BBOX_AREA_PX = int(
    _env_float("GPU_SLOT_DEBUG_MIN_MOTHERBOARD_BBOX_AREA_PX", 15000)
)
MIN_MOTHERBOARD_BBOX_WIDTH_PX = int(
    _env_float("GPU_SLOT_DEBUG_MIN_MOTHERBOARD_BBOX_WIDTH_PX", 110)
)
MIN_MOTHERBOARD_BBOX_HEIGHT_PX = int(
    _env_float("GPU_SLOT_DEBUG_MIN_MOTHERBOARD_BBOX_HEIGHT_PX", 100)
)
REACTIVE_PERIOD_S = max(
    0.0,
    _env_float("GPU_REACTIVE_SOCKET_HOVER_PERIOD_S", 0.0),
)
REFERENCE_ANCHOR_BLEND_MAX_PX = _env_float(
    "GPU_SLOT_DEBUG_REFERENCE_ANCHOR_BLEND_MAX_PX",
    18.0,
)
REFERENCE_ANCHOR_BLEND_GAIN = _env_float(
    "GPU_SLOT_DEBUG_REFERENCE_ANCHOR_BLEND_GAIN",
    0.25,
)

_AUX_CAMERA_XML_T_CAM_WORLD_CACHE = {}


def _is_left_third_view_camera(camera):
    return str(camera).strip().lower() in {"left_fixed", "left_third"}


def _copy_scene_obb(obb, query):
    copied = gh._copy_motherboard_obb(obb)
    copied["query"] = str(query)
    return copied


def _capture_camera_bundle(camera, *, include_depth):
    rgb = gh._as_uint8_rgb(gh.get_camera_image(camera=camera))
    if rgb is None:
        raise RuntimeError(f"could not capture {camera!r} RGB image")
    depth = None
    if include_depth:
        depth = gh.render_depth(camera=camera)
        if depth is None:
            raise RuntimeError(f"could not capture {camera!r} depth image")
    T_cam_world = _tracking_camera_T_cam_world(camera)
    cam_K = gh._camera_matrix(camera) if (include_depth or T_cam_world is not None) else None
    return rgb, depth, cam_K, T_cam_world
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

    parent_pos = np.zeros(3, dtype=np.float64)
    parent_rot = np.eye(3, dtype=np.float64)
    for body in worldbody.findall("body"):
        found = _find(body, parent_pos, parent_rot)
        if found is not None:
            return found
    return None


def _aux_camera_xml_body_candidates(camera):
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


def _calibrated_aux_camera_T_cam_world(camera):
    cached = _AUX_CAMERA_XML_T_CAM_WORLD_CACHE.get(camera)
    if cached is not None:
        return cached
    try:
        from enpire.env.forge.robot.models.station.paths import get_station_xml
    except Exception:
        _AUX_CAMERA_XML_T_CAM_WORLD_CACHE[camera] = None
        return None
    xml_path = Path(get_station_xml()).expanduser()
    if not xml_path.is_file():
        _AUX_CAMERA_XML_T_CAM_WORLD_CACHE[camera] = None
        return None
    for body_name in _aux_camera_xml_body_candidates(camera):
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
        _AUX_CAMERA_XML_T_CAM_WORLD_CACHE[camera] = T_cam_world
        return T_cam_world
    _AUX_CAMERA_XML_T_CAM_WORLD_CACHE[camera] = None
    return None


def _tracking_camera_T_cam_world(camera):
    if camera != AUX_CAMERA:
        return gh._camera_T_cam_world(camera)
    calibrated_T_cam_world = _calibrated_aux_camera_T_cam_world(camera)
    return calibrated_T_cam_world


def _target_socket_right_edge_offset_m(target_socket_number):
    idx = max(0, min(len(SOCKET_TARGET_RIGHT_EDGE_OFFSETS_M) - 1, int(target_socket_number) - 1))
    return float(SOCKET_TARGET_RIGHT_EDGE_OFFSETS_M[idx])


def _right_edge_half_extent_world_y(obb):
    major = gh._xy_unit(obb["major_axis_world"], fallback=gh.WORLD_LEFT)
    minor = gh._xy_unit(obb["minor_axis_world"], fallback=gh._perp_xy(major))
    world_y = gh._xy_unit(gh.WORLD_LEFT, fallback=gh.WORLD_LEFT)
    return (
        abs(float(np.dot(major[:2], world_y[:2]))) * float(obb["half_major_m"])
        + abs(float(np.dot(minor[:2], world_y[:2]))) * float(obb["half_minor_m"])
    )


def _camera_board_right_edge_image_side(camera):
    if _is_left_third_view_camera(camera):
        return "left"
    return "right"


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
        if image_side == "left":
            edge_xs.append(float(row_cols.min()))
        else:
            edge_xs.append(float(row_cols.max()))
        edge_ys.append(float(row_idx))
    if not edge_xs:
        return None
    return np.asarray(
        [float(np.median(edge_xs)), float(np.median(edge_ys))],
        dtype=np.float64,
    )


def _mask_right_edge_trace_px(mask, *, camera=None, max_points=24):
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2 or not np.any(mask_bool):
        return []
    valid_rows = np.flatnonzero(np.any(mask_bool, axis=1))
    if valid_rows.size == 0:
        return []
    if valid_rows.size > max_points:
        sample_idx = np.linspace(0, valid_rows.size - 1, num=max_points)
        valid_rows = valid_rows[np.round(sample_idx).astype(int)]
    trace = []
    image_side = _camera_board_right_edge_image_side(camera)
    for row_idx in valid_rows.tolist():
        row_cols = np.flatnonzero(mask_bool[int(row_idx)])
        if row_cols.size == 0:
            continue
        edge_x = float(row_cols.min()) if image_side == "left" else float(row_cols.max())
        trace.append(
            np.asarray(
                [edge_x, float(row_idx)],
                dtype=np.float64,
            )
        )
    return trace


def _project_right_edge_anchor_world(anchor_px, cam_K, T_cam_world, top_z):
    if anchor_px is None:
        return None
    if cam_K is None or T_cam_world is None:
        return None
    return gh._project_pixel_to_plane_world(
        anchor_px,
        cam_K,
        T_cam_world,
        top_z,
    )


def _log_mask_with_right_edge(rgb, mask, *, query="", tag="segment", alpha=0.45, camera=None):
    path = _ORIGINAL_SEGMENT_LOG_MASK(
        rgb,
        mask,
        query=query,
        tag=tag,
        alpha=alpha,
    )
    query_text = str(query).lower()
    if path is None or "mother" not in query_text or "board" not in query_text:
        return path
    right_edge_trace_px = _mask_right_edge_trace_px(mask, camera=camera)
    right_edge_anchor_px = _mask_right_edge_anchor_px(mask, camera=camera)
    if len(right_edge_trace_px) < 2 and right_edge_anchor_px is None:
        return path
    try:
        from PIL import Image, ImageDraw

        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        edge_color = (80, 220, 255)
        for p0, p1 in zip(right_edge_trace_px, right_edge_trace_px[1:]):
            draw.line(
                (
                    float(p0[0]),
                    float(p0[1]),
                    float(p1[0]),
                    float(p1[1]),
                ),
                fill=edge_color,
                width=3,
            )
        if right_edge_anchor_px is not None:
            x = float(right_edge_anchor_px[0])
            y = float(right_edge_anchor_px[1])
            r = 7.0
            draw.line((x - r, y, x + r, y), fill=edge_color, width=3)
            draw.line((x, y - r, x, y + r), fill=edge_color, width=3)
        img.save(path)
    except Exception:
        pass
    return path


if _env_flag("GPU_SLOT_DEBUG_SAVE_SEGMENT_LOGS", True):
    _segmentation_tools.log_mask = _log_mask_with_right_edge


def _derive_biased_socket_hover_world(geometry_scene, top_z):
    localization_mode = str(
        geometry_scene.get("socket_hover_localization_mode", "edge")
    ).strip().lower()
    if localization_mode == "center":
        center_world = geometry_scene.get("motherboard_center_world")
        center_offset_world = geometry_scene.get("socket_hover_center_offset_world")
        if center_world is None or center_offset_world is None:
            raise RuntimeError("motherboard center-based hover localization is unavailable")
        hover_world = np.asarray(center_world, dtype=np.float64).reshape(3).copy()
        hover_world += np.asarray(center_offset_world, dtype=np.float64).reshape(3)
    else:
        right_edge_anchor_world = geometry_scene.get("motherboard_right_edge_anchor_world")
        if right_edge_anchor_world is None:
            raise RuntimeError("motherboard right-edge anchor world is unavailable")
        hover_world = np.asarray(right_edge_anchor_world, dtype=np.float64).reshape(3).copy()
        hover_world[0] += float(geometry_scene["socket_hover_x_from_right_edge_m"])
        hover_world[1] -= float(geometry_scene["socket_hover_side_sign"]) * float(
            geometry_scene["socket_hover_edge_offset_m"]
        )
    hover_world[1] += float(SOCKET_HOVER_Y_TRIM_M)
    hover_world[2] = float(top_z)
    return hover_world


def _camera_segmentation_observation(camera, top_z):
    allow_world = _tracking_camera_T_cam_world(camera) is not None
    rgb, _, cam_K, T_cam_world = _capture_camera_bundle(
        camera,
        include_depth=False,
    )
    seg_record = gh._segment_motherboard_mask(camera=camera)
    score = float(seg_record.get("score", 0.0) or 0.0)
    mask_area = int(seg_record.get("mask_area", 0) or 0)
    center_px = gh._bbox_center_px_from_xywh(seg_record.get("bbox_xywh"))
    right_edge_anchor_px = _mask_right_edge_anchor_px(
        seg_record.get("mask"),
        camera=camera,
    )
    center_world = None
    right_edge_anchor_world = None
    world_valid = False
    if allow_world and center_px is not None:
        center_world = gh._project_pixel_to_plane_world(
            center_px,
            cam_K,
            T_cam_world,
            top_z,
        )
        if right_edge_anchor_px is not None:
            right_edge_anchor_world = _project_right_edge_anchor_world(
                right_edge_anchor_px,
                cam_K,
                T_cam_world,
                top_z,
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
            and score >= float(AUX_CAMERA_MIN_SCORE)
        ),
        "world_valid": bool(world_valid and score >= float(AUX_CAMERA_MIN_SCORE)),
    }


def _apply_world_estimate_to_scene(
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
    updated["motherboard_debug_3d_bbox"] = _copy_scene_obb(
        gh._override_obb_center_xy(
            reference_scene["motherboard_reference_3d_bbox"],
            updated["motherboard_center_world"],
        ),
        reference_scene["motherboard_reference_3d_bbox"]["query"],
    )
    if center_update is not None:
        updated["motherboard_center_update"] = center_update
    if pose_source_camera is not None:
        updated["pose_source_camera"] = str(pose_source_camera)
    if aux_score is not None:
        updated["aux_camera_last_score"] = float(aux_score)
    if localization_mode is not None:
        updated["socket_hover_localization_mode"] = str(localization_mode)
    _set_scene_hover_target(
        updated,
        _derive_biased_socket_hover_world(updated, float(reference_scene["motherboard_top_z"])),
        cam_K,
        T_cam_world,
    )
    return updated


def _maybe_apply_aux_camera_pose(
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
    top_center_world = np.asarray(
        scene["motherboard_center_world"],
        dtype=np.float64,
    ).reshape(3)
    aux_center_world = np.asarray(
        aux_obs["center_world"],
        dtype=np.float64,
    ).reshape(3)
    aux_anchor_world = aux_center_world + np.asarray(
        reference_scene["motherboard_right_edge_anchor_from_center_offset_world"],
        dtype=np.float64,
    ).reshape(3)
    pose_diff_m = float(
        np.linalg.norm(aux_center_world[:2] - top_center_world[:2])
    )
    if (
        float(AUX_CAMERA_MAX_POSE_DIFF_M) > 0.0
        and pose_diff_m > float(AUX_CAMERA_MAX_POSE_DIFF_M)
    ):
        msg = (
            f"pose_diff_m={pose_diff_m:.4f} "
            f"max_pose_diff_m={float(AUX_CAMERA_MAX_POSE_DIFF_M):.4f} "
            f"primary_score={float(primary_score):.3f} "
            f"aux_score={float(aux_obs['score']):.3f}"
        )
        if float(primary_score) >= float(PRIMARY_CAMERA_MIN_SCORE):
            print(
                "[gpu_left_slot_hover] Auxiliary camera "
                f"{aux_obs['camera']!r} {stage_label} pose rejected; "
                f"keeping primary motherboard pose: {msg}"
            )
            return scene
        raise RuntimeError(
            "auxiliary camera motherboard pose disagrees with primary camera "
            f"and primary score is below threshold: {msg}"
        )
    if float(primary_score) < float(PRIMARY_CAMERA_MIN_SCORE):
        use_mode = "aux"
    elif pose_diff_m <= float(AUX_CAMERA_BLEND_MAX_SHIFT_M):
        use_mode = "blend"
    elif bool(AUX_CAMERA_PREFER_WORLD_POSE):
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
        weight_sum = top_w + aux_w
        chosen_center_world = (top_w * top_center_world + aux_w * aux_center_world) / weight_sum
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
            "min_shift_m": float(CENTER_UPDATE_MIN_SHIFT_M),
            "max_shift_m": float(CENTER_UPDATE_MAX_SHIFT_M),
            "hold_reason": "",
        }

    print(
        "[gpu_left_slot_hover] Auxiliary camera "
        f"{aux_obs['camera']!r} {stage_label} pose update: "
        f"mode={use_mode} "
        f"primary_score={float(primary_score):.3f} "
        f"aux_score={float(aux_obs['score']):.3f} "
        f"pose_diff_m={pose_diff_m:.4f} "
        "localization=center"
    )
    return _apply_world_estimate_to_scene(
        reference_scene,
        scene,
        center_world=chosen_center_world,
        right_edge_anchor_world=chosen_anchor_world,
        registration=registration,
        cam_K=cam_K,
        T_cam_world=T_cam_world,
        center_update=center_update,
        pose_source_camera=aux_obs["camera"] if use_mode == "aux" else f"{CAMERA}+{aux_obs['camera']}",
        aux_score=aux_obs["score"],
        localization_mode="center",
    )


def _set_scene_hover_target(scene, hover_world, cam_K, T_cam_world):
    scene["socket_hover_world"] = np.asarray(hover_world, dtype=np.float64).reshape(3)
    scene["socket_hover_px"] = gh._project_world_to_pixel(
        scene["socket_hover_world"],
        cam_K,
        T_cam_world,
    )


def _build_scene_record(
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
        "motherboard_reference_3d_bbox": reference_obb,
        "motherboard_debug_3d_bbox": debug_obb,
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


def _save_hover_debug_artifacts(rgb, camera, scene, seg_record, artifact_prefix):
    overlay = gh._as_uint8_rgb(rgb)
    if overlay is None:
        return
    right_edge_color = np.array([80, 220, 255], dtype=np.uint8)
    bbox_xywh = seg_record.get("bbox_xywh")
    if bbox_xywh is not None and len(bbox_xywh) == 4:
        x, y, w_box, h_box = [int(v) for v in bbox_xywh]
        gh._draw_rect_rgb(
            overlay,
            [x, y, x + w_box, y + h_box],
            np.array([255, 80, 80], dtype=np.uint8),
            width=3,
        )
    if scene.get("motherboard_center_px") is not None:
        gh._draw_cross_rgb(
            overlay,
            scene["motherboard_center_px"],
            np.array([255, 220, 80], dtype=np.uint8),
            radius=8,
        )
    right_edge_trace_px = _mask_right_edge_trace_px(
        seg_record.get("mask"),
        camera=camera,
    )
    for p0, p1 in zip(right_edge_trace_px, right_edge_trace_px[1:]):
        gh._draw_line_rgb(
            overlay,
            p0,
            p1,
            right_edge_color,
            width=2,
        )
    if scene.get("motherboard_right_edge_anchor_px") is not None:
        gh._draw_cross_rgb(
            overlay,
            scene["motherboard_right_edge_anchor_px"],
            right_edge_color,
            radius=8,
        )
    hover_px = scene.get("socket_hover_px")
    if hover_px is not None:
        gh._draw_cross_rgb(
            overlay,
            hover_px,
            np.array([80, 255, 80], dtype=np.uint8),
            radius=9,
        )
    current_obb_record = {
        "obb": scene["motherboard_debug_3d_bbox"],
        "seg_record": seg_record,
    }
    gh._save_initial_motherboard_3d_bbox_artifacts(
        overlay,
        camera,
        scene,
        artifact_prefix=artifact_prefix,
        obb_record=current_obb_record,
    )
    out_dir = gh._artifact_output_dir("gpu_handover")
    report_path = out_dir / f"{artifact_prefix}_motherboard_hover_debug.json"
    current_left = gh.get_robot_state()
    left_pos = gh._robot_vec(current_left, "left", "ee_pos")
    left_rpy = gh._robot_vec(current_left, "left", "ee_rpy")
    report = {
        "target_socket_number": int(TARGET_SOCKET_NUMBER),
        "pose_source_camera": scene.get("pose_source_camera", CAMERA),
        "socket_hover_localization_mode": scene.get("socket_hover_localization_mode", "edge"),
        "aux_camera_name": scene.get("aux_camera_name"),
        "aux_camera_last_score": (
            None
            if scene.get("aux_camera_last_score") is None
            else round(float(scene["aux_camera_last_score"]), 4)
        ),
        "aux_camera_world_enabled": bool(scene.get("aux_camera_world_enabled", False)),
        "motherboard_seg_bbox_xywh": bbox_xywh,
        "motherboard_center_px": (
            None
            if scene.get("motherboard_center_px") is None
            else [
                round(float(v), 2)
                for v in np.asarray(scene["motherboard_center_px"], dtype=float).tolist()
            ]
        ),
        "motherboard_center_world": [
            round(float(v), 5)
            for v in np.asarray(scene["motherboard_center_world"], dtype=float).tolist()
        ],
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
        "hover_z": round(float(scene["hover_z"]), 5),
        "socket_hover_world": [
            round(float(v), 5)
            for v in np.asarray(scene["socket_hover_world"], dtype=float).tolist()
        ],
        "socket_hover_x_offset_m": round(
            float(scene["socket_hover_x_offset_m"]),
            5,
        ),
        "socket_hover_y_trim_m": round(float(SOCKET_HOVER_Y_TRIM_M), 5),
        "socket_hover_edge_offset_m": round(
            float(scene["socket_hover_edge_offset_m"]),
            5,
        ),
        "socket_hover_x_from_right_edge_m": (
            None
            if scene.get("socket_hover_x_from_right_edge_m") is None
            else round(float(scene["socket_hover_x_from_right_edge_m"]), 5)
        ),
        "socket_hover_side_sign": round(float(scene["socket_hover_side_sign"]), 3),
        "socket_target_right_edge_offsets_m": [
            round(float(v), 5) for v in SOCKET_TARGET_RIGHT_EDGE_OFFSETS_M
        ],
        "socket_hover_x_offset_m_default": round(float(SOCKET_HOVER_X_OFFSET_M), 5),
        "board_plane_z_offset_m": round(float(BOARD_PLANE_Z_OFFSET_M), 5),
        "registration": scene.get("motherboard_registration"),
        "left_ee_pos": [round(float(v), 5) for v in np.asarray(left_pos, dtype=float).tolist()],
        "left_ee_rpy": [round(float(v), 3) for v in np.asarray(left_rpy, dtype=float).tolist()],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "[gpu_left_slot_hover] Saved hover debug artifacts: "
        f"report={report_path}"
    )


def _maybe_save_hover_debug_artifacts(rgb, camera, scene, seg_record, artifact_prefix):
    if not SAVE_DEBUG_ARTIFACTS:
        return
    _save_hover_debug_artifacts(
        rgb,
        camera,
        scene,
        seg_record=seg_record,
        artifact_prefix=artifact_prefix,
    )


def _build_initial_reference_scene(camera):
    rgb, depth, cam_K, T_cam_world = _capture_camera_bundle(
        camera,
        include_depth=True,
    )
    obb_record = gh._compute_initial_motherboard_3d_bbox(
        camera=camera,
        depth=depth,
        cam_K=cam_K,
        T_cam_world=T_cam_world,
    )
    seg_record = obb_record["seg_record"]
    obb = obb_record["obb"]
    seg_bbox_xywh = seg_record.get("bbox_xywh")
    if seg_bbox_xywh is not None and len(seg_bbox_xywh) == 4:
        _x, _y, bbox_w, bbox_h = [int(round(float(v))) for v in seg_bbox_xywh]
        bbox_area = int(bbox_w) * int(bbox_h)
        if (
            bbox_area < int(MIN_MOTHERBOARD_BBOX_AREA_PX)
            or int(bbox_w) < int(MIN_MOTHERBOARD_BBOX_WIDTH_PX)
            or int(bbox_h) < int(MIN_MOTHERBOARD_BBOX_HEIGHT_PX)
        ):
            raise RuntimeError(
                "motherboard segmentation bbox is too small for a reliable socket hover: "
                f"bbox_xywh={seg_bbox_xywh} area_px={bbox_area} "
                f"min_area_px={int(MIN_MOTHERBOARD_BBOX_AREA_PX)} "
                f"min_width_px={int(MIN_MOTHERBOARD_BBOX_WIDTH_PX)} "
                f"min_height_px={int(MIN_MOTHERBOARD_BBOX_HEIGHT_PX)}"
            )
    seg_center_px = gh._bbox_center_px_from_xywh(seg_record.get("bbox_xywh"))
    if seg_center_px is None:
        raise RuntimeError("motherboard segmentation bbox center is unavailable")
    motherboard_top_z = float(obb["z_min"]) + float(BOARD_PLANE_Z_OFFSET_M)
    center_world = gh._project_pixel_to_plane_world(
        seg_center_px,
        cam_K,
        T_cam_world,
        motherboard_top_z,
    )
    if center_world is None:
        raise RuntimeError("could not project motherboard segmentation center to fixed board plane")
    reference_obb = gh._override_obb_center_xy(obb, center_world)
    reference_obb = _copy_scene_obb(reference_obb, seg_record["query"])
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
    side_sign = gh._motherboard_right_edge_sign(
        reference_obb["center_world"],
        gh.WORLD_LEFT,
        right_edge_half_extent_m,
        cam_K,
        T_cam_world,
    )
    scene = _build_scene_record(
        top_z=motherboard_top_z,
        hover_z=max(
            motherboard_top_z + float(HOVER_CLEARANCE_M),
            float(HOVER_MIN_Z),
        ),
        center_px=seg_center_px,
        center_world=center_world,
        seg_center_px=seg_center_px,
        seg_bbox_xywh=seg_record.get("bbox_xywh"),
        reference_mask=np.asarray(seg_record["mask"], dtype=bool),
        reference_obb=reference_obb,
        debug_obb=_copy_scene_obb(reference_obb, seg_record["query"]),
        registration=None,
        x_offset_m=float(SOCKET_HOVER_X_OFFSET_M),
        x_from_right_edge_m=0.0,
        edge_offset_m=_target_socket_right_edge_offset_m(int(TARGET_SOCKET_NUMBER)),
        side_sign=side_sign,
        target_socket_number=int(TARGET_SOCKET_NUMBER),
        right_edge_anchor_px=right_edge_anchor_px,
        right_edge_anchor_world=right_edge_anchor_world,
        center_offset_world=np.zeros(3, dtype=np.float64),
        right_edge_anchor_from_center_offset_world=(
            np.asarray(right_edge_anchor_world, dtype=np.float64).reshape(3)
            - np.asarray(center_world, dtype=np.float64).reshape(3)
        ),
    )
    initial_hover_world = center_world.copy()
    initial_hover_world[0] += float(scene["socket_hover_x_offset_m"])
    initial_hover_world[1] = float(right_edge_anchor_world[1]) - float(scene["socket_hover_side_sign"]) * float(
        scene["socket_hover_edge_offset_m"]
    )
    initial_hover_world[2] = float(motherboard_top_z)
    scene["socket_hover_x_from_right_edge_m"] = float(
        initial_hover_world[0] - float(right_edge_anchor_world[0])
    )
    scene["socket_hover_center_offset_world"] = (
        np.asarray(initial_hover_world, dtype=np.float64).reshape(3)
        - np.asarray(center_world, dtype=np.float64).reshape(3)
    )
    scene["pose_source_camera"] = str(camera)
    _set_scene_hover_target(
        scene,
        _derive_biased_socket_hover_world(scene, motherboard_top_z),
        cam_K,
        T_cam_world,
    )
    seeded_hover_world = np.asarray(
        scene["socket_hover_world"],
        dtype=np.float64,
    ).reshape(3).copy()
    seeded_right_edge_anchor_world = np.asarray(
        scene["motherboard_right_edge_anchor_world"],
        dtype=np.float64,
    ).reshape(3).copy()
    if AUX_CAMERA is not None:
        try:
            aux_obs = _camera_segmentation_observation(
                AUX_CAMERA,
                motherboard_top_z,
            )
            use_aux_initial_world = (
                bool(AUX_CAMERA_PREFER_WORLD_POSE)
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
                    float(AUX_CAMERA_MAX_POSE_DIFF_M) > 0.0
                    and aux_pose_diff_m > float(AUX_CAMERA_MAX_POSE_DIFF_M)
                    and primary_score >= float(PRIMARY_CAMERA_MIN_SCORE)
                ):
                    print(
                        "[gpu_left_slot_hover] Auxiliary camera "
                        f"{aux_obs['camera']!r} initial world pose rejected; "
                        "keeping primary motherboard pose: "
                        f"pose_diff_m={aux_pose_diff_m:.4f} "
                        f"max_pose_diff_m={float(AUX_CAMERA_MAX_POSE_DIFF_M):.4f} "
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
                scene["motherboard_debug_3d_bbox"] = _copy_scene_obb(
                    gh._override_obb_center_xy(reference_obb, aux_center_world),
                    seg_record["query"],
                )
                recentered_hover_offset_world = (
                    seeded_hover_world - aux_center_world
                )
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
                _set_scene_hover_target(
                    scene,
                    _derive_biased_socket_hover_world(scene, motherboard_top_z),
                    cam_K,
                    T_cam_world,
                )
                print(
                    "[gpu_left_slot_hover] Auxiliary camera "
                    f"{aux_obs['camera']!r} initial world pose seeded the hover geometry"
                )
            else:
                scene = _maybe_apply_aux_camera_pose(
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
            if AUX_CAMERA_REQUIRED:
                raise RuntimeError(
                    f"required auxiliary camera {AUX_CAMERA!r} is unavailable: {exc}"
                ) from exc
            print(
                "[gpu_left_slot_hover] Auxiliary camera initial observation unavailable: "
                f"{exc}"
            )
    _maybe_save_hover_debug_artifacts(
        rgb,
        camera,
        scene,
        seg_record=seg_record,
        artifact_prefix="initial",
    )
    print(
        "[gpu_left_slot_hover] Built initial reference: "
        f"target_socket={int(TARGET_SOCKET_NUMBER)} "
        f"top_z={float(motherboard_top_z):.4f} "
        f"hover_z={float(scene['hover_z']):.4f} "
        f"x_offset_m={float(scene['socket_hover_x_offset_m']):.4f} "
        f"x_from_right_edge_m={float(scene['socket_hover_x_from_right_edge_m']):.4f} "
        f"edge_offset_m={float(scene['socket_hover_edge_offset_m']):.3f} "
        f"pose_source={scene.get('pose_source_camera', camera)!r} "
        f"localization_mode={scene.get('socket_hover_localization_mode', 'edge')!r} "
        f"center_world={[round(float(v), 4) for v in scene['motherboard_center_world'].tolist()]} "
        f"right_edge_anchor_world={[round(float(v), 4) for v in scene['motherboard_right_edge_anchor_world'].tolist()]} "
        f"hover_world={[round(float(v), 4) for v in scene['socket_hover_world'].tolist()]}"
    )
    return scene


def _resolve_hover_rpy():
    if EXPLICIT_HOVER_RPY is not None:
        return [float(v) for v in EXPLICIT_HOVER_RPY]
    return [float(v) for v in DEFAULT_HOVER_RPY]


def _refresh_scene_from_sam3(reference_scene, base_scene, camera, artifact_prefix):
    rgb, _, cam_K, T_cam_world = _capture_camera_bundle(camera, include_depth=False)
    seg_record = gh._segment_motherboard_mask(camera=camera)
    primary_score = float(seg_record.get("score", 0.0) or 0.0)
    reference_top_z = float(reference_scene["motherboard_top_z"])
    seg_center_px = gh._bbox_center_px_from_xywh(seg_record.get("bbox_xywh"))
    current_right_edge_anchor_px = _mask_right_edge_anchor_px(
        seg_record.get("mask"),
        camera=camera,
    )
    registered_center_px = None
    registered_right_edge_anchor_px = None
    anchor_center_px = None
    registration = None
    tracking_mask = base_scene.get("motherboard_reference_mask")
    tracking_center_px = base_scene.get("motherboard_seg_center_px")
    tracking_right_edge_anchor_px = base_scene.get("motherboard_right_edge_anchor_px")
    if (
        tracking_mask is not None
        and tracking_center_px is not None
        and seg_record.get("mask") is not None
    ):
        try:
            registration = gh._estimate_mask_translation_px(
                tracking_mask,
                seg_record["mask"],
            )
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
        except Exception as exc:
            print(
                "[gpu_left_slot_hover] Mask registration unavailable: "
                f"{exc}. Falling back to current SAM bbox center."
            )
    anchor_mask = reference_scene.get("motherboard_reference_mask")
    anchor_tracking_center_px = reference_scene.get("motherboard_seg_center_px")
    if (
        anchor_mask is not None
        and anchor_tracking_center_px is not None
        and seg_record.get("mask") is not None
    ):
        try:
            anchor_registration = gh._estimate_mask_translation_px(
                anchor_mask,
                seg_record["mask"],
            )
            anchor_center_px = np.asarray(
                anchor_tracking_center_px,
                dtype=np.float64,
            ).reshape(2) + np.asarray(
                [
                    float(anchor_registration["dx_px"]),
                    float(anchor_registration["dy_px"]),
                ],
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
        if anchor_diff_px <= float(REFERENCE_ANCHOR_BLEND_MAX_PX):
            tracking_reference_center_px = (
                (1.0 - float(REFERENCE_ANCHOR_BLEND_GAIN))
                * np.asarray(registered_center_px, dtype=np.float64).reshape(2)
                + float(REFERENCE_ANCHOR_BLEND_GAIN)
                * np.asarray(anchor_center_px, dtype=np.float64).reshape(2)
            )
    if chosen_center_px is None:
        raise RuntimeError("motherboard center pixel unavailable during refresh")
    chosen_right_edge_anchor_px = (
        registered_right_edge_anchor_px
        if registered_right_edge_anchor_px is not None
        else current_right_edge_anchor_px
    )
    if chosen_right_edge_anchor_px is None:
        raise RuntimeError("motherboard right-edge anchor pixel unavailable during refresh")
    center_world = gh._project_pixel_to_plane_world(
        chosen_center_px,
        cam_K,
        T_cam_world,
        reference_top_z,
    )
    if center_world is None:
        raise RuntimeError("could not project refreshed motherboard center to fixed board plane")
    right_edge_anchor_world = _project_right_edge_anchor_world(
        chosen_right_edge_anchor_px,
        cam_K,
        T_cam_world,
        reference_top_z,
    )
    if right_edge_anchor_world is None:
        raise RuntimeError("could not project refreshed motherboard right-edge anchor to fixed board plane")
    previous_center_world = np.asarray(
        base_scene["motherboard_center_world"],
        dtype=np.float64,
    ).reshape(3)
    previous_right_edge_anchor_world = np.asarray(
        base_scene["motherboard_right_edge_anchor_world"],
        dtype=np.float64,
    ).reshape(3)
    proposed_center_world = np.asarray(center_world, dtype=np.float64).reshape(3)
    proposed_right_edge_anchor_world = np.asarray(
        right_edge_anchor_world,
        dtype=np.float64,
    ).reshape(3)
    proposed_delta_xy = (
        proposed_right_edge_anchor_world[:2] - previous_right_edge_anchor_world[:2]
    )
    proposed_shift_m = float(np.linalg.norm(proposed_delta_xy))
    center_update = {
        "proposed_shift_m": proposed_shift_m,
        "applied_shift_m": 0.0,
        "held": False,
        "min_shift_m": float(CENTER_UPDATE_MIN_SHIFT_M),
        "max_shift_m": float(CENTER_UPDATE_MAX_SHIFT_M),
        "hold_reason": "",
    }
    accepted_center_px = chosen_center_px
    accepted_tracking_center_px = tracking_reference_center_px
    accepted_right_edge_anchor_px = chosen_right_edge_anchor_px
    accepted_right_edge_anchor_world = proposed_right_edge_anchor_world
    accepted_tracking_mask = (
        None
        if seg_record.get("mask") is None
        else np.asarray(seg_record["mask"], dtype=bool)
    )
    if proposed_shift_m > float(CENTER_UPDATE_MAX_SHIFT_M):
        center_world = previous_center_world.copy()
        right_edge_anchor_world = previous_right_edge_anchor_world.copy()
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
        center_world = proposed_center_world
        right_edge_anchor_world = proposed_right_edge_anchor_world
        center_update["applied_shift_m"] = proposed_shift_m
    current_obb = gh._override_obb_center_xy(
        reference_scene["motherboard_reference_3d_bbox"],
        center_world,
    )
    current_obb = _copy_scene_obb(current_obb, seg_record["query"])
    scene = _build_scene_record(
        top_z=reference_top_z,
        hover_z=reference_scene["hover_z"],
        center_px=accepted_center_px,
        center_world=center_world,
        seg_center_px=accepted_tracking_center_px,
        seg_bbox_xywh=seg_record.get("bbox_xywh"),
        reference_mask=accepted_tracking_mask,
        reference_obb=_copy_scene_obb(
            reference_scene["motherboard_reference_3d_bbox"],
            reference_scene["motherboard_reference_3d_bbox"]["query"],
        ),
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
    scene["motherboard_center_update"] = center_update
    scene["pose_source_camera"] = str(camera)
    _set_scene_hover_target(
        scene,
        _derive_biased_socket_hover_world(scene, reference_top_z),
        cam_K,
        T_cam_world,
    )
    if AUX_CAMERA is not None:
        try:
            aux_obs = _camera_segmentation_observation(
                AUX_CAMERA,
                reference_top_z,
            )
            scene = _maybe_apply_aux_camera_pose(
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
                "[gpu_left_slot_hover] Auxiliary camera refresh unavailable: "
                f"{exc}"
            )
    _maybe_save_hover_debug_artifacts(
        rgb,
        camera,
        scene,
        seg_record=seg_record,
        artifact_prefix=artifact_prefix,
    )
    reg_text = (
        "none"
        if registration is None
        else (
            f"method={registration.get('method', 'unknown')} "
            f"dx_px={int(registration.get('dx_px', 0))} "
            f"dy_px={int(registration.get('dy_px', 0))} "
            f"score={float(registration.get('score', 0.0)):.1f}"
        )
    )
    print(
        "[gpu_left_slot_hover] Refreshed motherboard from SAM3: "
        f"pose_source={scene.get('pose_source_camera', camera)!r} "
        f"localization_mode={scene.get('socket_hover_localization_mode', 'edge')!r} "
        f"primary_score={primary_score:.3f} "
        f"center_world={[round(float(v), 4) for v in scene['motherboard_center_world'].tolist()]} "
        f"right_edge_anchor_world={[round(float(v), 4) for v in scene['motherboard_right_edge_anchor_world'].tolist()]} "
        f"hover_world={[round(float(v), 4) for v in scene['socket_hover_world'].tolist()]} "
        f"registration={reg_text} "
        f"proposed_shift_m={center_update['proposed_shift_m']:.4f} "
        f"applied_shift_m={center_update['applied_shift_m']:.4f} "
        f"held={center_update['held']} "
        f"hold_reason={center_update['hold_reason'] or 'none'}"
    )
    return scene


def _move_left_to_hover(
    scene,
    hover_rpy,
    *,
    fixed_plane_z=None,
    guided=False,
    guided_duration_s=None,
    guided_steps=None,
):
    target_xy = np.asarray(scene["socket_hover_world"], dtype=np.float64).reshape(3)
    current_state = gh.get_robot_state()
    current_pos = np.asarray(gh._robot_vec(current_state, "left", "ee_pos"), dtype=np.float64)
    target_pos = np.asarray(
        [
            float(target_xy[0]),
            float(target_xy[1]),
            float(scene["hover_z"] if fixed_plane_z is None else fixed_plane_z),
        ],
        dtype=np.float64,
    )
    target_rpy = [float(v) for v in hover_rpy]
    move_delta_m = float(np.linalg.norm(target_pos[:2] - current_pos[:2]))
    z_delta_m = abs(float(target_pos[2] - current_pos[2]))
    print(
        "[gpu_left_slot_hover] Evaluating hover move: "
        f"current_pos={[round(float(v), 4) for v in current_pos.tolist()]} "
        f"target_pos={[round(float(v), 4) for v in target_pos.tolist()]} "
        f"target_rpy={[round(float(v), 1) for v in target_rpy]} "
        f"guided={guided} "
        f"fixed_plane_z={None if fixed_plane_z is None else round(float(fixed_plane_z), 4)} "
        f"xy_delta_m={move_delta_m:.4f} "
        f"z_delta_m={z_delta_m:.4f}"
    )
    if float(MAX_HOVER_XY_DELTA_M) > 0.0 and move_delta_m > float(MAX_HOVER_XY_DELTA_M):
        raise RuntimeError(
            "slot hover target is too far from the current left-held GPU pose; "
            "refusing likely bad motherboard localization: "
            f"xy_delta_m={move_delta_m:.4f} "
            f"max_xy_delta_m={float(MAX_HOVER_XY_DELTA_M):.4f} "
            f"current_pos={[round(float(v), 4) for v in current_pos.tolist()]} "
            f"target_pos={[round(float(v), 4) for v in target_pos.tolist()]}"
        )
    if move_delta_m <= float(MIN_MOVE_M) and z_delta_m <= float(HOVER_Z_TOL_M):
        print(
            "[gpu_left_slot_hover] Skipping replanning because target is already close "
            f"(xy_delta_m={move_delta_m:.4f}, z_delta_m={z_delta_m:.4f})"
        )
        return [float(v) for v in current_pos.tolist()]

    print(
        "[gpu_left_slot_hover] XY hover move: "
        f"pos={[round(float(v), 4) for v in target_pos.tolist()]} "
        f"rpy={[round(float(v), 1) for v in target_rpy]}"
    )
    if guided:
        base_duration_s = float(
            REACTIVE_GUIDED_DURATION_S if guided_duration_s is None else guided_duration_s
        )
        duration_s = float(base_duration_s)
        if float(GUIDED_MAX_XY_SPEED_MPS) > 0.0:
            duration_s = max(duration_s, move_delta_m / float(GUIDED_MAX_XY_SPEED_MPS))
        if float(GUIDED_MAX_Z_SPEED_MPS) > 0.0:
            duration_s = max(duration_s, z_delta_m / float(GUIDED_MAX_Z_SPEED_MPS))
        if duration_s > base_duration_s + 1e-6:
            print(
                "[gpu_left_slot_hover] Guided hover speed clamp: "
                f"base_duration_s={base_duration_s:.2f} "
                f"clamped_duration_s={duration_s:.2f} "
                f"max_xy_speed_mps={float(GUIDED_MAX_XY_SPEED_MPS):.3f} "
                f"max_z_speed_mps={float(GUIDED_MAX_Z_SPEED_MPS):.3f}"
            )
        _guided_left_xy_plane_move(
            target_pos.tolist(),
            target_rpy,
            duration_s=duration_s,
            num_steps=int(REACTIVE_GUIDED_STEPS if guided_steps is None else guided_steps),
        )
    else:
        gh._move("left", target_pos.tolist(), target_rpy)
    return [float(v) for v in target_pos.tolist()]


def _guided_left_xy_plane_move(target_pos, target_rpy, *, duration_s, num_steps):
    env = gh._tool_env_from_callable(get_robot_state)
    if env is None or not hasattr(env, "move_bimanual_joint_keypoints"):
        raise RuntimeError("direct YAM env not available for guided XY hover move")

    duration_s = float(duration_s)
    num_steps = max(2, int(num_steps))
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = gh._display_rpy_to_rotation(target_rpy).as_quat().astype(np.float64)

    obs_left = env.get_observations("left")
    obs_right = env.get_observations("right")
    left_jp = np.asarray(obs_left["joint_pos"], dtype=np.float64).reshape(6)
    right_jp = np.asarray(obs_right["joint_pos"], dtype=np.float64).reshape(6)
    left_gp = float(np.asarray(obs_left["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    right_gp = float(np.asarray(obs_right["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    left_start_pos = np.asarray(obs_left["ee_pos"], dtype=np.float64).reshape(3)
    left_start_quat = np.asarray(obs_left["ee_quat"], dtype=np.float64).reshape(4)
    right_hold_pos = np.asarray(obs_right["ee_pos"], dtype=np.float64).reshape(3)
    right_hold_quat = np.asarray(obs_right["ee_quat"], dtype=np.float64).reshape(4)

    start_rot = Rotation.from_quat(left_start_quat)
    target_rot = Rotation.from_quat(target_quat)
    delta_rot = target_rot * start_rot.inv()

    left_waypoints = [left_jp.copy()]
    right_waypoints = [right_jp.copy()]
    left_grippers = [[left_gp]]
    right_grippers = [[right_gp]]
    timestamps = [0.0]

    with env._kin_lock:
        env.kin.forward_kinematics(left_jp, right_jp)
        cur_left_jp = left_jp.copy()
        cur_right_jp = right_jp.copy()
        for step_index in range(1, num_steps + 1):
            alpha = float(step_index) / float(num_steps)
            interp_pos = left_start_pos + alpha * (target_pos - left_start_pos)
            interp_rot = Rotation.from_rotvec(delta_rot.as_rotvec() * alpha) * start_rot
            interp_quat = interp_rot.as_quat().astype(np.float64)
            env.kin.forward_kinematics(cur_left_jp, cur_right_jp)
            next_left_jp, next_right_jp = env.kin.inverse_kinematics(
                interp_pos,
                interp_quat,
                right_hold_pos,
                right_hold_quat,
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
        "[gpu_left_slot_hover] Guided XY hover move: "
        f"target_pos={[round(float(v), 4) for v in target_pos.tolist()]} "
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
        start_interp_s=0.0,
    )
    if not bool(result.get("success", False)):
        raise RuntimeError(result.get("reason", "guided XY hover move failed"))


def _acquire_initial_hover(reference_scene, hover_rpy, camera):
    current_scene = reference_scene
    last_exc = None
    for attempt_idx in range(INITIAL_HOVER_MAX_ATTEMPTS):
        attempt_num = attempt_idx + 1
        print(
            "[gpu_left_slot_hover] Initial hover attempt "
            f"{attempt_num}/{int(INITIAL_HOVER_MAX_ATTEMPTS)}"
        )
        try:
            hover_pos = _move_left_to_hover(
                current_scene,
                hover_rpy,
                guided=True,
                guided_duration_s=float(INITIAL_GUIDED_DURATION_S),
                guided_steps=int(INITIAL_GUIDED_STEPS),
            )
            return hover_pos, current_scene
        except Exception as exc:
            last_exc = exc
            print(
                "[gpu_left_slot_hover] Initial hover attempt failed: "
                f"{exc}"
            )
            if attempt_num >= int(INITIAL_HOVER_MAX_ATTEMPTS):
                break
            if "slot hover target is too far" in str(exc):
                print(
                    "[gpu_left_slot_hover] Rebuilding initial motherboard reference "
                    "after far hover target rejection"
                )
                try:
                    current_scene = _build_initial_reference_scene(camera)
                    reference_scene = current_scene
                    continue
                except Exception as refresh_exc:
                    print(
                        "[gpu_left_slot_hover] Full motherboard redetection failed; "
                        f"falling back to tracked SAM3 refresh: {refresh_exc}"
                    )
            current_scene = _refresh_scene_from_sam3(
                reference_scene,
                current_scene,
                camera=camera,
                artifact_prefix=f"refined_attempt_{attempt_num:02d}",
            )
    raise RuntimeError(
        f"socket hover failed after {int(INITIAL_HOVER_MAX_ATTEMPTS)} attempt(s): {last_exc}"
    )


def _run_reactive_hover_loop(reference_scene, reactive_hover_rpy, reactive_plane_z, camera):
    cycle_index = 0
    current_scene = reference_scene
    while True:
        cycle_index += 1
        print(
            "[gpu_left_slot_hover] Reactive cycle "
            f"{cycle_index}: refresh motherboard and adjust hover"
        )
        try:
            current_scene = _refresh_scene_from_sam3(
                reference_scene,
                current_scene,
                camera=camera,
                artifact_prefix=f"refined_cycle_{cycle_index:03d}",
            )
            hover_pos = _move_left_to_hover(
                current_scene,
                reactive_hover_rpy,
                fixed_plane_z=reactive_plane_z,
                guided=True,
            )
            print(
                "[gpu_left_slot_hover] Reactive cycle "
                f"{cycle_index}: hover pos="
                f"{[round(float(v), 4) for v in hover_pos]}"
            )
        except Exception as exc:
            print(
                "[gpu_left_slot_hover] Reactive cycle "
                f"{cycle_index} failed: {exc}"
            )
        time.sleep(float(REACTIVE_PERIOD_S))


print(
    "[gpu_left_slot_hover] Config: "
    f"camera={CAMERA!r} "
    f"aux_camera={AUX_CAMERA!r} "
    f"target_socket={int(TARGET_SOCKET_NUMBER)} "
    f"x_offset_m={float(SOCKET_HOVER_X_OFFSET_M):.3f} "
    f"y_trim_m={float(SOCKET_HOVER_Y_TRIM_M):.3f} "
    f"edge_offsets_m={[round(float(v), 3) for v in SOCKET_TARGET_RIGHT_EDGE_OFFSETS_M]} "
    f"go_home_on_start={bool(GO_HOME_ON_START)} "
    f"hover_clearance_m={float(HOVER_CLEARANCE_M):.3f} "
    f"hover_min_z={float(HOVER_MIN_Z):.3f} "
    f"board_plane_z_offset_m={float(BOARD_PLANE_Z_OFFSET_M):.3f} "
    f"primary_min_score={float(PRIMARY_CAMERA_MIN_SCORE):.3f} "
    f"aux_min_score={float(AUX_CAMERA_MIN_SCORE):.3f} "
    f"aux_max_pose_diff_m={float(AUX_CAMERA_MAX_POSE_DIFF_M):.3f} "
    f"aux_prefer_world_pose={bool(AUX_CAMERA_PREFER_WORLD_POSE)} "
    f"aux_required={bool(AUX_CAMERA_REQUIRED)} "
    f"aux_has_calibrated_xml_world={bool(_calibrated_aux_camera_T_cam_world(AUX_CAMERA) is not None)} "
    f"min_move_m={float(MIN_MOVE_M):.4f} "
    f"hover_z_tol_m={float(HOVER_Z_TOL_M):.4f} "
    f"center_update_min_shift_m={float(CENTER_UPDATE_MIN_SHIFT_M):.4f} "
    f"center_update_max_shift_m={float(CENTER_UPDATE_MAX_SHIFT_M):.4f} "
    f"save_debug_artifacts={bool(SAVE_DEBUG_ARTIFACTS)} "
    f"initial_guided_duration_s={float(INITIAL_GUIDED_DURATION_S):.2f} "
    f"initial_guided_steps={int(INITIAL_GUIDED_STEPS)} "
    f"reactive_guided_duration_s={float(REACTIVE_GUIDED_DURATION_S):.2f} "
    f"reactive_guided_steps={int(REACTIVE_GUIDED_STEPS)} "
    f"guided_max_xy_speed_mps={float(GUIDED_MAX_XY_SPEED_MPS):.3f} "
    f"guided_max_z_speed_mps={float(GUIDED_MAX_Z_SPEED_MPS):.3f} "
    f"initial_hover_max_attempts={int(INITIAL_HOVER_MAX_ATTEMPTS)} "
    f"reactive_period_s={float(REACTIVE_PERIOD_S):.2f}"
)

try:
    if GO_HOME_ON_START:
        print("[gpu_left_slot_hover] Step -1: go home before starting hover debug")
        gh.go_home()

    if CLOSE_LEFT_GRIPPER_ON_START:
        print("[gpu_left_slot_hover] Step 0: close left gripper")
        gh.close_gripper("left")

    left_pos = gh._robot_vec(gh.get_robot_state(), "left", "ee_pos")
    left_rpy = gh._robot_vec(gh.get_robot_state(), "left", "ee_rpy")
    print(
        "[gpu_left_slot_hover] Step 1: build initial motherboard reference "
        f"from left_pos={[round(float(v), 4) for v in left_pos]} "
        f"left_rpy={[round(float(v), 1) for v in left_rpy]}"
    )
    reference_scene = _build_initial_reference_scene(camera=CAMERA)
    hover_rpy = _resolve_hover_rpy()
    print(
        "[gpu_left_slot_hover] Using hover orientation: "
        f"{[round(float(v), 1) for v in hover_rpy]}"
    )

    print("[gpu_left_slot_hover] Step 2: acquire initial hover")
    try:
        hover_pos, _ = _acquire_initial_hover(
            reference_scene=reference_scene,
            hover_rpy=hover_rpy,
            camera=CAMERA,
        )
        print(
            "[gpu_left_slot_hover] Initial hover acquired: "
            f"pos={[round(float(v), 4) for v in hover_pos]}"
        )
    except Exception as exc:
        print(
            "[gpu_left_slot_hover] Initial hover acquisition failed: "
            f"{exc}"
        )
        raise

    reactive_plane_z = float(reference_scene["hover_z"])
    reactive_hover_rpy = [float(v) for v in hover_rpy]
    print(
        "[gpu_left_slot_hover] Step 2b: lock reactive hover pose "
        f"z={reactive_plane_z:.4f} "
        f"rpy={[round(float(v), 1) for v in reactive_hover_rpy]}"
    )

    print("[gpu_left_slot_hover] Step 3: enter reactive hover loop")
    _run_reactive_hover_loop(
        reference_scene=reference_scene,
        reactive_hover_rpy=reactive_hover_rpy,
        reactive_plane_z=reactive_plane_z,
        camera=CAMERA,
    )
except KeyboardInterrupt:
    print("[gpu_left_slot_hover] Interrupted. Keeping current pose.")

