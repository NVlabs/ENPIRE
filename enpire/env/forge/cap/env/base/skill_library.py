"""Shared skill library — stateless service calls to external servers.

These functions take raw data (images, state) and call external APIs.
They are env-agnostic — each env's skills.py provides the I/O adapter.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import requests

logger = logging.getLogger(__name__)


# VLM routing was previously duplicated here; it now lives in the unified
# transport at ``cap.agent.tools.vlm`` (one registry, one dispatch point
# for all backends). Import ``cap.agent.tools.vlm.query`` if you need a
# VLM call from an env-side helper.


# ---------------------------------------------------------------------------
# Oracle Detection
# ---------------------------------------------------------------------------


def detect_oracle(
    query: str,
    objects: dict[str, dict],
) -> list[dict]:
    """Match query against scene object positions. Returns list of detection dicts.

    Args:
        query: Object name to search for (fuzzy substring match, case-insensitive).
        objects: Dict of object data from the sim, keyed by name. Each value must
            have ``"pos"`` ([x, y, z]) and ``"quat"`` ([w, x, y, z] MuJoCo convention).
            Optionally ``"size"`` (geom half-extents).

    Returns:
        List of dicts with keys: label, score, box_2d, position_3d,
        quaternion_xyzw, half_extents.

    Raises:
        ValueError: If no objects match the query.
    """
    if not objects:
        raise ValueError("No scene objects found in simulation")

    # Fuzzy match: case-insensitive substring
    query_lower = query.lower()
    matches: list[tuple[str, dict]] = [
        (name, data)
        for name, data in objects.items()
        if query_lower in name.lower() or name.lower() in query_lower
    ]

    if not matches:
        available = ", ".join(objects.keys())
        raise ValueError(f"No object matching '{query}'. Available: {available}")

    detections: list[dict] = []
    for name, data in matches:
        pos = data["pos"]  # [x, y, z]
        quat_wxyz = data["quat"]  # MuJoCo convention: [w, x, y, z]
        quat_xyzw = quat_wxyz[1:] + quat_wxyz[:1]  # -> [x, y, z, w]
        size = data.get("size", [])  # geom half-extents from MuJoCo
        detections.append(
            {
                "label": name,
                "score": 1.0,
                "box_2d": [],
                "position_3d": [round(float(x), 4) for x in pos],
                "quaternion_xyzw": [round(float(x), 4) for x in quat_xyzw],
                "half_extents": [round(float(x), 4) for x in size],
            }
        )

    return detections


# ---------------------------------------------------------------------------
# Nudge (delta EE move)
# ---------------------------------------------------------------------------


def compute_nudge_target(
    cur_pos: np.ndarray,
    cur_quat_xyzw: np.ndarray,
    delta_pos: list[float] | np.ndarray | None = None,
    delta_rpy_deg: list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute new EE target from current pose + deltas in world frame.

    Args:
        cur_pos: Current EE position [x, y, z], shape (3,).
        cur_quat_xyzw: Current EE orientation as quaternion [x, y, z, w], shape (4,).
        delta_pos: Position offset [dx, dy, dz] in metres (world frame).
            Defaults to [0, 0, 0].
        delta_rpy_deg: Orientation offset [droll, dpitch, dyaw] in degrees
            (world frame, extrinsic XYZ). Composed as ``delta_rot @ current_rot``.
            Defaults to [0, 0, 0].

    Returns:
        (new_pos, new_quat_xyzw): New EE target position and orientation.
    """
    from scipy.spatial.transform import Rotation

    d_pos = np.asarray(
        delta_pos if delta_pos is not None else [0, 0, 0], dtype=np.float64
    )
    d_rpy = np.asarray(
        delta_rpy_deg if delta_rpy_deg is not None else [0, 0, 0], dtype=np.float64
    )

    cur_pos = np.asarray(cur_pos, dtype=np.float64)
    cur_quat_xyzw = np.asarray(cur_quat_xyzw, dtype=np.float64)

    # New position: current + delta (world frame)
    new_pos = cur_pos + d_pos

    # New orientation: delta_rot @ current_rot (world frame rotation)
    cur_rot = Rotation.from_quat(cur_quat_xyzw)  # scipy uses xyzw
    delta_rot = Rotation.from_euler("xyz", d_rpy, degrees=True)
    new_rot = delta_rot * cur_rot
    new_quat_xyzw = new_rot.as_quat()  # scipy returns xyzw

    return new_pos, new_quat_xyzw


# ---------------------------------------------------------------------------
# AnyGrasp Service
# ---------------------------------------------------------------------------

# AnyGrasp vendor frame: X=approach, Y=opening, Z=height
# Right-multiplied onto 4x4 grasp poses: columns select old→new axis mapping
# Original (grip_site): col0=[0,1,0] col1=[0,0,1] col2=[1,0,0]
#   → new_X=old_Y(opening), new_Y=old_Z(height), new_Z=old_X(approach)
# ee_quat has X/Y swapped vs grip_site, so swap columns 0 and 1:
#   → new_X=old_Z(height), new_Y=old_Y(opening), new_Z=old_X(approach)
_ANYGRASP_TO_GRIPPER = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)
_ANYGRASP_TO_GRIPPER_T = np.eye(4, dtype=np.float64)
_ANYGRASP_TO_GRIPPER_T[:3, :3] = _ANYGRASP_TO_GRIPPER


def _np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_np(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)))


def _quat_xyzw_to_display_rpy_deg(quat_xyzw: np.ndarray) -> list[float]:
    """Inverse of freespace_move._display_rpy_to_quat for planner output."""
    from scipy.spatial.transform import Rotation

    ex, ey, ez = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_euler(
        "xyz", degrees=True
    )
    disp = np.array([ey, -ex, -ez - 90.0], dtype=np.float64)
    disp = (disp + 180.0) % 360.0 - 180.0
    return disp.tolist()


@dataclass
class GraspCandidate:
    """Motion-planner-compatible grasp candidate."""

    position: list[float]  # [x, y, z] in world frame
    rpy: list[float]  # display RPY [roll, pitch, yaw] in degrees
    score: float
    width: float = 0.08


def segment_object_sam3(
    rgb: np.ndarray,
    object_name: str,
    sam3_url: str,
    score_threshold: float = 0.2,
) -> tuple[np.ndarray, float]:
    """Segment an object using SAM3 server. Returns (int32 mask, confidence score)."""
    import urllib.error
    import urllib.request

    buf = io.BytesIO()
    np.save(buf, rgb)
    image_b64 = base64.b64encode(buf.getvalue()).decode()
    payload = json.dumps(
        {
            "text": object_name,
            "image_b64": image_b64,
            "score_threshold": float(score_threshold),
        }
    ).encode()
    req = urllib.request.Request(
        f"{sam3_url.rstrip('/')}/segment",
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
        raise RuntimeError(f"SAM3 /segment failed ({e.code}): {detail}") from None
    data = json.loads(resp.read())
    mask_bytes = base64.b64decode(data["mask_b64"])
    mask = np.load(io.BytesIO(mask_bytes)).astype(np.int32)
    score = float(data.get("score", 1.0))
    return mask, score


def segment_all_objects_sam3(
    rgb: np.ndarray,
    object_name: str,
    sam3_url: str,
    score_threshold: float = 0.1,
) -> list:
    """Segment all matching objects with SAM3.

    Calls the SAM3 server's ``/segment_all`` endpoint and returns a list of
    ``SegmentationResult`` objects sorted by descending confidence by the
    server. Returns an empty list when SAM3 finds no masks above threshold.
    """
    import urllib.error
    import urllib.request

    from enpire.env.forge.cap.agent.tools.base import SegmentationResult

    buf = io.BytesIO()
    np.save(buf, rgb)
    image_b64 = base64.b64encode(buf.getvalue()).decode()
    payload = json.dumps(
        {
            "text": object_name,
            "image_b64": image_b64,
            "score_threshold": float(score_threshold),
        }
    ).encode()
    req = urllib.request.Request(
        f"{sam3_url.rstrip('/')}/segment_all",
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
        raise RuntimeError(f"SAM3 /segment_all failed ({e.code}): {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"SAM3 /segment_all connection failed: {e}") from None

    data = json.loads(resp.read())
    detections = []
    for det in data.get("detections", []):
        mask = _b64_to_np(det["mask_b64"]).astype(np.int32)
        detections.append(
            SegmentationResult(
                mask=mask,
                bbox_xywh=[int(x) for x in det["bbox_xywh"]],
                score=float(det["score"]),
                mask_area=int(det["mask_area"]),
            )
        )
    return detections


def sample_grasp_anygrasp(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_cam_world: np.ndarray,
    object_name: str,
    *,
    sam3_url: str | None = None,
    anygrasp_url: str | None = None,
    max_grasps: int = 10,
    top_down_only: bool = False,
    vertical_threshold: float = 0.8,
    object_input_mode: str = "segmented_object_cloud",
    tcp_offset_z_m: float = 0.0,
    min_planner_z_m: float | None = None,
    gripper_default_width_m: float = 0.08,
) -> list[GraspCandidate]:
    """Plan grasp poses using SAM3 segmentation + AnyGrasp server.

    This is a stateless function — all camera data is passed in.
    The env's skills.py adapter calls this with data from the env.

    Args:
        rgb: RGB image (H, W, 3) uint8.
        depth: Depth image (H, W) float32 in metres.
        K: Camera intrinsics (3, 3).
        T_cam_world: Camera extrinsics 4x4 (camera-to-world transform).
        object_name: Name of the object to grasp.
        sam3_url: SAM3 segmentation server URL.
        anygrasp_url: AnyGrasp planning server URL.
        max_grasps: Max candidates to return.
        top_down_only: Filter for top-down grasps only.
        vertical_threshold: Dot-product threshold for top-down filtering.
        object_input_mode: "segmented_object_cloud" or "roi_workspace".
        tcp_offset_z_m: TCP offset along local +Z.
        min_planner_z_m: Floor for planner Z clipping (None = no clipping).
        gripper_default_width_m: Default gripper width.

    Returns:
        List of GraspCandidate sorted by score (best first).

    Raises:
        RuntimeError: On server connection or segmentation errors.
    """
    from scipy.spatial.transform import Rotation

    _sam3_url = sam3_url or os.environ.get(
        "SAM3_SERVER_URL",
        f"http://{os.environ.get('SAM3_SERVER_HOST', 'localhost')}:{os.environ.get('SAM3_SERVER_PORT', '6767')}",
    )
    _anygrasp_url = anygrasp_url or os.environ.get(
        "ANYGRASP_SERVICE_URL",
        f"http://{os.environ.get('ANYGRASP_SERVER_HOST', 'localhost')}:{os.environ.get('ANYGRASP_SERVER_PORT', '8122')}",
    )
    _anygrasp_timeout = float(os.environ.get("ANYGRASP_PLAN_TIMEOUT_S", "300"))

    # 1) Segment object
    logger.info("[anygrasp] Segmenting %r ...", object_name)
    mask, _seg_score = segment_object_sam3(rgb, object_name, _sam3_url)
    segmap = mask.astype(np.int32)

    # 2) Call AnyGrasp /plan_viz
    logger.info("[anygrasp] Planning grasps for %r ...", object_name)
    payload = {
        "rgb_base64": _np_to_b64(rgb),
        "depth_base64": _np_to_b64(depth.astype(np.float32)),
        "cam_K_base64": _np_to_b64(np.asarray(K, dtype=np.float64)),
        "segmap_base64": _np_to_b64(segmap),
        "segmap_id": 1,
        "z_range": [1e-6, 1.5],
        "max_grasps": max_grasps,
        "workspace_margin": 0.02,
        "object_input_mode": object_input_mode,
    }
    t0 = time.perf_counter()
    resp = requests.post(
        f"{_anygrasp_url.rstrip('/')}/plan_viz",
        json=payload,
        timeout=_anygrasp_timeout,
    )
    resp.raise_for_status()
    logger.info("[anygrasp] /plan_viz returned in %.1fs", time.perf_counter() - t0)
    data = resp.json()

    grasps_cam_raw = (
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

    if grasps_cam_raw.size == 0:
        raise RuntimeError(f"AnyGrasp found no grasps for {object_name!r}")

    # 3) Camera frame AnyGrasp → planner/gripper frame
    grasps_cam = np.matmul(grasps_cam_raw, _ANYGRASP_TO_GRIPPER_T)
    # 4) Camera → world
    T_cam = np.asarray(T_cam_world, dtype=np.float64)
    grasps_world = np.matmul(T_cam, grasps_cam)

    # 5) TCP offset along local +Z
    translated = grasps_world.copy()
    translated[:, :3, 3] = (
        grasps_world[:, :3, 3] + tcp_offset_z_m * grasps_world[:, :3, 2]
    )

    # 6) Planner Z clipping
    if min_planner_z_m is not None:
        z = translated[:, 2, 3]
        z[z < min_planner_z_m] = min_planner_z_m

    # 7) Top-down filter
    if top_down_only:
        keep = [
            i
            for i in range(len(translated))
            if float(np.dot(translated[i, :3, 2], [0, 0, -1])) > vertical_threshold
        ]
        if not keep:
            raise RuntimeError(
                f"No top-down grasps for {object_name!r} (threshold={vertical_threshold})"
            )
        translated = translated[keep]
        scores = scores[keep]
        widths = widths[keep] if len(widths) >= len(keep) else widths

    # 8) Filter grasps wider than the gripper can open
    width_keep = [
        i for i in range(len(translated))
        if (float(widths[i]) if i < len(widths) else gripper_default_width_m) <= gripper_default_width_m
    ]
    if width_keep:
        translated = translated[width_keep]
        scores = scores[width_keep]
        widths = widths[width_keep] if len(widths) >= len(width_keep) else widths

    # 9) Sort by score, limit
    order = np.argsort(-scores)[:max_grasps]
    translated = translated[order]
    scores = scores[order]
    widths = widths[order] if len(widths) >= len(order) else widths

    # 9) Build candidates
    candidates: list[GraspCandidate] = []
    for i, T in enumerate(translated):
        pos = T[:3, 3].tolist()
        quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
        rpy_deg = _quat_xyzw_to_display_rpy_deg(quat_xyzw)
        w = float(widths[i]) if i < len(widths) else gripper_default_width_m
        candidates.append(
            GraspCandidate(
                position=[round(float(x), 5) for x in pos],
                rpy=[round(float(x), 4) for x in rpy_deg],
                score=round(float(scores[i]), 4),
                width=round(w, 5),
            )
        )

    overlay_jpeg = (
        base64.b64decode(data["overlay_jpeg_base64"])
        if data.get("overlay_jpeg_base64")
        else None
    )

    logger.info(
        "[anygrasp] %d grasps for %r (best=%.4f)",
        len(candidates),
        object_name,
        candidates[0].score if candidates else 0,
    )
    return candidates, {"mask": mask, "overlay_jpeg": overlay_jpeg}
