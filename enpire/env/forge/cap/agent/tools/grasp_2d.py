"""Top-down 2D grasp planning tool for the CAP agent.

This planner uses only a SAM3 segmentation mask plus camera intrinsics/extrinsics
to propose top-down grasps on the table plane. It is intended as a fallback for
short or depth-sparse objects that segment cleanly in 2D but fail to produce a
usable segmented point cloud for AnyGrasp.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from enpire.env.forge.cap.agent.tools.base import Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import (
    CAP_SERVER_PORT,
    SAM3_SERVER_HOST,
    SAM3_SERVER_PORT,
    TABLE_SURFACE_Z_M,
)

logger = logging.getLogger(__name__)

_ANYGRASP_Z_RANGE = (1e-6, 1.5)
_TWO_D_GRASP_PLANNER_Z_M = float(TABLE_SURFACE_Z_M) + 0.03
_TWO_D_MAJOR_AXIS_RATIO_THRESHOLD = 1.10
_TWO_D_GRASP_WIDTH_M = 0.08
_TWO_D_CENTER_ONLY_YAWS_DEG = (0.0, 180.0, 90.0, -90.0)
_TWO_D_AXIS_SAMPLE_FRACTIONS = (0.15, 0.35, 0.50, 0.65, 0.85)
_TWO_D_LOCAL_AXIS_MIN_RADIUS_PX = 6.0
_TWO_D_LOCAL_AXIS_MAX_RADIUS_PX = 32.0
_TWO_D_LOCAL_AXIS_RADIUS_FRACTION = 0.18
_TWO_D_LOCAL_AXIS_MIN_NEIGHBOR_POINTS = 12


@dataclass
class GraspCandidate:
    """Motion-planner-compatible grasp candidate."""

    position: list[float]  # [x, y, z] in world frame
    rpy: list[float]  # display RPY [roll, pitch, yaw] in degrees
    score: float
    width: float = _TWO_D_GRASP_WIDTH_M


@dataclass
class Grasp2DPlanResult:
    """Result bundle returned by the pure 2D planning helpers."""

    candidates: list[GraspCandidate]
    grasp_rows: list[dict[str, Any]]
    overlay_jpeg: bytes | None
    debug: dict[str, Any]


def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize an angle to [-180, 180)."""

    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


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


def _project_world_points_to_pixels(
    points_world: np.ndarray,
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    R_cam_world = np.asarray(T_cam_world[:3, :3], dtype=np.float64)
    t_cam_world = np.asarray(T_cam_world[:3, 3], dtype=np.float64)
    pts_cam = (R_cam_world.T @ (pts - t_cam_world).T).T
    z = pts_cam[:, 2]
    valid = np.isfinite(pts_cam).all(axis=1) & (z > 1e-9)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)
    pts_cam = pts_cam[valid]
    z = pts_cam[:, 2]
    fx = float(cam_K[0, 0])
    fy = float(cam_K[1, 1])
    cx = float(cam_K[0, 2])
    cy = float(cam_K[1, 2])
    u = fx * pts_cam[:, 0] / z + cx
    v = fy * pts_cam[:, 1] / z + cy
    return np.column_stack([u, v]).astype(np.float64)


def _compute_pixel_ransac_obb(
    px: np.ndarray,
    *,
    n_iters: int = 200,
    inlier_thresh_px: float = 2.5,
    source: str = "segmented_region_2d_ransac_obb",
) -> dict[str, Any] | None:
    pts = np.asarray(px, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return None

    pts_i = np.round(pts).astype(np.int32)
    min_xy = np.min(pts_i, axis=0)
    max_xy = np.max(pts_i, axis=0)
    w = int(max_xy[0] - min_xy[0] + 5)
    h = int(max_xy[1] - min_xy[1] + 5)
    if w < 3 or h < 3:
        return None

    contour_pts = pts
    try:
        import cv2

        mask = np.zeros((h, w), dtype=np.uint8)
        local = pts_i - min_xy + 2
        valid = (
            (local[:, 0] >= 0)
            & (local[:, 0] < w)
            & (local[:, 1] >= 0)
            & (local[:, 1] < h)
        )
        local = local[valid]
        if local.shape[0] >= 4:
            mask[local[:, 1], local[:, 0]] = 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if contours:
                contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                if contour.shape[0] >= 4:
                    contour_pts = contour.astype(np.float64) + min_xy.astype(np.float64) - 2.0
    except Exception:
        contour_pts = pts

    def _axis_from_points(points_2d: np.ndarray) -> np.ndarray | None:
        centered = points_2d - np.mean(points_2d, axis=0)
        cov = centered.T @ centered / max(len(points_2d), 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = np.asarray(eigvecs[:, int(np.argmax(eigvals))], dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return None
        axis = axis / norm
        if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
            axis = -axis
        return axis

    rng = np.random.default_rng(0)
    best_inliers = None
    best_count = 0
    best_err = float("inf")
    n = int(contour_pts.shape[0])
    for _ in range(int(max(16, n_iters))):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        a = contour_pts[i]
        b = contour_pts[j]
        axis = np.asarray(b - a, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-6:
            continue
        axis = axis / axis_norm
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)
        d = np.abs((contour_pts - a) @ normal)
        inliers = d <= float(inlier_thresh_px)
        count = int(np.count_nonzero(inliers))
        if count <= 1:
            continue
        err = float(np.mean(d[inliers]))
        if count > best_count or (count == best_count and err < best_err):
            best_inliers = inliers
            best_count = count
            best_err = err

    if best_inliers is not None and int(np.count_nonzero(best_inliers)) >= 4:
        major = _axis_from_points(contour_pts[best_inliers])
    else:
        major = _axis_from_points(contour_pts)
    if major is None:
        return None

    centroid = pts.mean(axis=0)
    minor = np.array([-major[1], major[0]], dtype=np.float64)
    centered = pts - centroid
    proj_major = centered @ major
    proj_minor = centered @ minor
    min_major = float(np.min(proj_major))
    max_major = float(np.max(proj_major))
    min_minor = float(np.min(proj_minor))
    max_minor = float(np.max(proj_minor))
    corners = [
        centroid + max_major * major + max_minor * minor,
        centroid + max_major * major + min_minor * minor,
        centroid + min_major * major + min_minor * minor,
        centroid + min_major * major + max_minor * minor,
    ]
    x0 = int(np.floor(np.min(pts[:, 0])))
    y0 = int(np.floor(np.min(pts[:, 1])))
    x1 = int(np.ceil(np.max(pts[:, 0])))
    y1 = int(np.ceil(np.max(pts[:, 1])))
    return {
        "bbox_xywh": [x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
        "bbox_corners_px": [[round(float(v), 3) for v in c] for c in corners],
        "bbox_center_px": [round(float(v), 3) for v in centroid],
        "bbox_major_axis_px": [round(float(v), 6) for v in major],
        "bbox_minor_axis_px": [round(float(v), 6) for v in minor],
        "bbox_source": source,
        "bbox_ransac_inlier_count": int(best_count),
        "bbox_ransac_inlier_ratio": round(float(best_count) / float(max(n, 1)), 6),
    }


def _masked_valid_pixel_coords(
    depth: np.ndarray,
    cam_K: np.ndarray,
    mask: np.ndarray,
    *,
    z_range: tuple[float, float] = _ANYGRASP_Z_RANGE,
) -> np.ndarray:
    points_all = _depth_to_points(depth, cam_K)
    valid = (
        np.isfinite(points_all).all(axis=-1)
        & (points_all[:, :, 2] > float(z_range[0]))
        & (points_all[:, :, 2] < float(z_range[1]))
    )
    obj_mask_2d = (np.asarray(mask) > 0) & valid
    ys, xs = np.nonzero(obj_mask_2d)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([xs, ys]).astype(np.float64)


def _compute_projected_cloud_obb(
    points_world: np.ndarray,
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
) -> dict[str, Any] | None:
    px = _project_world_points_to_pixels(points_world, cam_K, T_cam_world)
    return _compute_pixel_ransac_obb(
        px,
        source="projected_segmented_cloud_2d_ransac_obb",
    )


def extract_segmented_object_world_points(
    depth: np.ndarray,
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
    mask: np.ndarray,
    *,
    z_range: tuple[float, float] = _ANYGRASP_Z_RANGE,
) -> np.ndarray:
    """Return the masked object point cloud in world frame.

    The point filtering matches the AnyGrasp server path: valid finite camera-frame
    points whose camera Z falls inside the demo-style z-range.
    """

    points_all = _depth_to_points(depth, cam_K)
    valid = (
        np.isfinite(points_all).all(axis=-1)
        & (points_all[:, :, 2] > float(z_range[0]))
        & (points_all[:, :, 2] < float(z_range[1]))
    )
    object_mask = (np.asarray(mask) > 0) & valid
    if not np.any(object_mask):
        return np.empty((0, 3), dtype=np.float64)

    object_points_cam = points_all[object_mask].astype(np.float64)
    R = np.asarray(T_cam_world[:3, :3], dtype=np.float64)
    t = np.asarray(T_cam_world[:3, 3], dtype=np.float64)
    return (R @ object_points_cam.T).T + t


def compute_segmented_cloud_height_m(points_world: np.ndarray) -> float:
    """Return the world-frame vertical extent of a segmented cloud."""

    if points_world.ndim != 2 or points_world.shape[0] < 2:
        return 0.0
    z = np.asarray(points_world[:, 2], dtype=np.float64)
    if z.size < 2 or not np.isfinite(z).any():
        return 0.0
    return float(np.max(z) - np.min(z))


def compute_mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Compute the binary-mask centroid via image moments."""

    mask01 = (np.asarray(mask) > 0).astype(np.float64)
    m00 = float(mask01.sum())
    if m00 <= 0.0:
        raise ValueError("Mask is empty")

    ys, xs = np.nonzero(mask01)
    m10 = float(xs.sum())
    m01 = float(ys.sum())
    return m10 / m00, m01 / m00


def principal_axis_from_mask(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return mask centroid, major axis unit vector, projections, and anisotropy ratio."""

    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty")

    coords = np.column_stack([xs, ys]).astype(np.float64)
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    if len(coords) < 2:
        return centroid, np.array([1.0, 0.0], dtype=np.float64), np.zeros((1,), dtype=np.float64), 1.0

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
    return centroid, axis, projections, ratio


def estimate_local_tangent_from_mask(
    mask: np.ndarray,
    pixel_xy: np.ndarray | list[float] | tuple[float, float],
    *,
    reference_axis_px: np.ndarray | None = None,
    radius_px: float | None = None,
) -> tuple[np.ndarray | None, tuple[np.ndarray, np.ndarray] | None, float]:
    """Estimate a local tangent around a sampled point using nearby mask pixels.

    This is a local-PCA approximation over nearby in-mask pixels. It lets each
    sampled grasp point use its own local orientation instead of sharing one
    global mask PCA angle.
    """

    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None, None, 1.0

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
        return None, None, 1.0

    local_centroid = local_coords.mean(axis=0)
    centered = local_coords - local_centroid
    cov = np.cov(centered, rowvar=False, bias=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major_var = float(max(eigvals[order[0]], 0.0))
    minor_var = float(max(eigvals[order[1]], 0.0))
    ratio = float(major_var / max(minor_var, 1e-12)) if major_var > 0.0 else 1.0

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
    return tangent_px, endpoints, ratio


def sample_pixels_along_major_axis(
    mask: np.ndarray,
    fractions: tuple[float, ...] = _TWO_D_AXIS_SAMPLE_FRACTIONS,
) -> tuple[list[np.ndarray], list[float], tuple[np.ndarray, np.ndarray] | None, float]:
    """Sample actual in-mask pixels along the PCA major-axis span.

    Returns (sample_points, kept_fractions, endpoints, ratio). `kept_fractions`
    has the same length as `sample_points` and records which input fraction
    each surviving point came from.
    """

    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) == 0:
        return [], [], None, 1.0

    coords = np.column_stack([xs, ys]).astype(np.float64)
    centroid, axis, projections, ratio = principal_axis_from_mask(mask)
    min_proj = float(np.min(projections))
    max_proj = float(np.max(projections))

    _MIN_SAMPLE_DIST_PX = 3.0
    sample_points: list[np.ndarray] = []
    kept_fractions: list[float] = []
    for frac in fractions:
        target = min_proj + float(frac) * (max_proj - min_proj)
        idx = int(np.argmin(np.abs(projections - target)))
        px = np.asarray(coords[idx], dtype=np.float64)
        if any(np.linalg.norm(px - existing) < _MIN_SAMPLE_DIST_PX for existing in sample_points):
            continue
        sample_points.append(px)
        kept_fractions.append(float(frac))

    lo_idx = int(np.argmin(projections))
    hi_idx = int(np.argmax(projections))
    endpoints = (coords[lo_idx].copy(), coords[hi_idx].copy())
    if not sample_points:
        sample_points = [centroid]
        kept_fractions = [0.5]
    return sample_points, kept_fractions, endpoints, ratio


def project_pixel_to_plane_world(
    pixel_xy: np.ndarray | list[float] | tuple[float, float],
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
    *,
    plane_z_m: float = TABLE_SURFACE_Z_M,
) -> np.ndarray | None:
    """Project an image pixel ray to the horizontal table plane in world frame."""

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


def project_world_to_pixel(
    point_world: np.ndarray | list[float] | tuple[float, float, float],
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
) -> tuple[int, int] | None:
    """Project a world-frame 3D point into image pixel coordinates."""

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


def _world_major_axis_deg(axis_world_xy: np.ndarray) -> float:
    """Return the axis angle in the repo's top-down yaw reference frame.

    Top-down yaw 0 aligns the gripper opening axis with world -Y in this repo.
    Measuring the world major axis relative to that -Y baseline lets us use the
    repo-level rule ``yaw = -major_axis_world_deg``.
    """

    theta_from_x = np.degrees(np.arctan2(float(axis_world_xy[1]), float(axis_world_xy[0])))
    return normalize_angle_deg(theta_from_x + 90.0)


def top_down_yaw_from_world_axis(axis_world_xy: np.ndarray) -> float:
    """Convert a world-frame XY axis into the repo's top-down yaw convention."""

    return normalize_angle_deg(-_world_major_axis_deg(axis_world_xy))


def world_axis_from_top_down_yaw(yaw_deg: float) -> np.ndarray:
    """Inverse of ``top_down_yaw_from_world_axis`` for the XY opening axis."""

    theta_from_x = np.radians(-float(yaw_deg) - 90.0)
    axis = np.array([np.cos(theta_from_x), np.sin(theta_from_x)], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.array([0.0, -1.0], dtype=np.float64)
    return axis / norm


def perpendicular_world_axis(axis_world_xy: np.ndarray) -> np.ndarray:
    """Return a normalized XY axis perpendicular to the given world-frame axis."""

    axis_world_xy = np.asarray(axis_world_xy, dtype=np.float64).reshape(2)
    perp = np.array([-axis_world_xy[1], axis_world_xy[0]], dtype=np.float64)
    norm = float(np.linalg.norm(perp))
    if norm <= 1e-12:
        return np.array([0.0, -1.0], dtype=np.float64)
    return perp / norm


def _score_for_rank(rank_zero_based: int) -> float:
    return round(max(0.0, 1.0 - 0.05 * float(rank_zero_based)), 4)


def _score_for_axis_position(fraction: float, yaw_index: int) -> float:
    """Bias scoring toward the center of the object's major axis.

    fraction = 0.5 gets the highest score; 0.0 and 1.0 get the lowest.
    Within the same sample point, the primary yaw (yaw_index=0) scores
    slightly higher than the flipped yaw (yaw_index=1).
    """
    center_bias = 1.0 - 2.0 * abs(float(fraction) - 0.5)  # 1.0 at center, 0.0 at ends
    yaw_bonus = 0.02 if int(yaw_index) == 0 else 0.0
    return round(max(0.0, center_bias + yaw_bonus), 4)


def _render_overlay_jpeg(
    rgb: np.ndarray,
    mask: np.ndarray,
    centroid_px: np.ndarray,
    sample_pixels: list[np.ndarray],
    major_axis_endpoints: tuple[np.ndarray, np.ndarray] | None,
    *,
    candidates: list[GraspCandidate],
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
) -> bytes | None:
    try:
        import cv2
    except Exception:
        return None

    img = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR).copy()
    mask_bool = np.asarray(mask) > 0
    if np.any(mask_bool):
        tint = np.zeros_like(img, dtype=np.uint8)
        tint[mask_bool] = (80, 220, 80)
        img[mask_bool] = cv2.addWeighted(img[mask_bool], 0.6, tint[mask_bool], 0.4, 0.0)
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 200, 0), 2)

    if major_axis_endpoints is not None:
        a = tuple(int(v) for v in np.round(major_axis_endpoints[0]).astype(int).tolist())
        b = tuple(int(v) for v in np.round(major_axis_endpoints[1]).astype(int).tolist())
        cv2.line(img, a, b, (255, 180, 0), 2, cv2.LINE_AA)

    h, w = img.shape[:2]

    def _draw_grasp_figure(
        rank_zero_based: int,
        candidate: GraspCandidate,
    ) -> None:
        axis_world_xy = world_axis_from_top_down_yaw(float(candidate.rpy[2]))
        center_world = np.array(
            [
                float(candidate.position[0]),
                float(candidate.position[1]),
                float(candidate.position[2]) if len(candidate.position) >= 3 else float(_TWO_D_GRASP_PLANNER_Z_M),
            ],
            dtype=np.float64,
        )
        half_line_len_m = max(0.015, 0.55 * float(candidate.width))
        line_a_world = center_world.copy()
        line_a_world[:2] -= half_line_len_m * axis_world_xy
        line_b_world = center_world.copy()
        line_b_world[:2] += half_line_len_m * axis_world_xy
        center_px = project_world_to_pixel(center_world, cam_K, T_cam_world)
        line_a_px = project_world_to_pixel(line_a_world, cam_K, T_cam_world)
        line_b_px = project_world_to_pixel(line_b_world, cam_K, T_cam_world)
        if center_px is None or line_a_px is None or line_b_px is None:
            return
        points = [center_px, line_a_px, line_b_px]
        if not all(0 <= p[0] < w and 0 <= p[1] < h for p in points):
            return

        if rank_zero_based == 0:
            color = (255, 0, 0)
            thickness = 2
        elif rank_zero_based == 1:
            color = (0, 255, 0)
            thickness = 2
        else:
            alpha = 1.0 - 0.6 * (rank_zero_based / max(len(candidates) - 1, 1))
            color = tuple(int(c * alpha) for c in (0, 220, 255))
            thickness = 2

        cv2.line(img, line_a_px, line_b_px, color, thickness, cv2.LINE_AA)
        cv2.circle(img, center_px, 4, color, -1, cv2.LINE_AA)
        cv2.circle(img, center_px, 6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            str(rank_zero_based + 1),
            (center_px[0] + 6, center_px[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if candidates:
        for idx, candidate in enumerate(candidates):
            _draw_grasp_figure(idx, candidate)
    else:
        c = tuple(int(v) for v in np.round(centroid_px).astype(int).tolist())
        cv2.circle(img, c, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, 8, (32, 32, 32), 1, cv2.LINE_AA)

        for idx, px in enumerate(sample_pixels, start=1):
            p = tuple(int(v) for v in np.round(px).astype(int).tolist())
            cv2.circle(img, p, 6, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(
                img,
                str(idx),
                (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    return buf.tobytes()


def plan_top_down_grasps_from_mask(
    *,
    rgb: np.ndarray,
    mask: np.ndarray,
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
    object_name: str,
    camera: str,
    max_grasps: int,
    projection_z_m: float | None = None,
) -> Grasp2DPlanResult:
    """Generate top-down grasp candidates directly from a 2D segmentation mask."""

    mask01 = (np.asarray(mask) > 0).astype(np.uint8)
    if mask01.sum() <= 0:
        raise ValueError(f"2dgrasp requires a non-empty mask for {object_name!r}")
    projection_z = (
        float(TABLE_SURFACE_Z_M)
        if projection_z_m is None
        else float(projection_z_m)
    )

    centroid_px = np.asarray(compute_mask_centroid(mask01), dtype=np.float64)
    sample_pixels, sample_fractions, axis_endpoints_px, ratio = sample_pixels_along_major_axis(mask01)
    global_axis_px: np.ndarray | None = None
    if axis_endpoints_px is not None:
        axis_vec_px = np.asarray(axis_endpoints_px[1] - axis_endpoints_px[0], dtype=np.float64)
        axis_norm_px = float(np.linalg.norm(axis_vec_px))
        if axis_norm_px > 1e-9:
            global_axis_px = axis_vec_px / axis_norm_px
    center_world = project_pixel_to_plane_world(
        centroid_px,
        cam_K,
        T_cam_world,
        plane_z_m=projection_z,
    )
    if center_world is None:
        raise ValueError("2dgrasp could not project the mask centroid onto the table plane")

    axis_world_vec: np.ndarray | None = None
    if axis_endpoints_px is not None:
        a_world = project_pixel_to_plane_world(axis_endpoints_px[0], cam_K, T_cam_world, plane_z_m=projection_z)
        b_world = project_pixel_to_plane_world(axis_endpoints_px[1], cam_K, T_cam_world, plane_z_m=projection_z)
        if a_world is not None and b_world is not None:
            axis_xy = np.asarray(b_world[:2] - a_world[:2], dtype=np.float64)
            if np.linalg.norm(axis_xy) > 1e-9:
                axis_world_vec = axis_xy / np.linalg.norm(axis_xy)

    is_axis_guided = bool(ratio >= _TWO_D_MAJOR_AXIS_RATIO_THRESHOLD and axis_world_vec is not None)
    candidates: list[GraspCandidate] = []
    grasp_rows: list[dict[str, Any]] = []

    if is_axis_guided:
        projected_sample_worlds: list[tuple[np.ndarray, tuple[float, ...], float]] = []
        kept_sample_pixels: list[np.ndarray] = []
        for px, frac in zip(sample_pixels, sample_fractions):
            world = project_pixel_to_plane_world(px, cam_K, T_cam_world, plane_z_m=projection_z)
            if world is None:
                continue
            local_tangent_px, local_endpoints_px, _local_ratio = estimate_local_tangent_from_mask(
                mask01,
                px,
                reference_axis_px=global_axis_px,
            )
            local_tangent_world: np.ndarray | None = None
            if local_endpoints_px is not None:
                local_a_world = project_pixel_to_plane_world(
                    local_endpoints_px[0],
                    cam_K,
                    T_cam_world,
                    plane_z_m=projection_z,
                )
                local_b_world = project_pixel_to_plane_world(
                    local_endpoints_px[1],
                    cam_K,
                    T_cam_world,
                    plane_z_m=projection_z,
                )
                if local_a_world is not None and local_b_world is not None:
                    tangent_xy = np.asarray(local_b_world[:2] - local_a_world[:2], dtype=np.float64)
                    tangent_norm = float(np.linalg.norm(tangent_xy))
                    if tangent_norm > 1e-9:
                        local_tangent_world = tangent_xy / tangent_norm

            if local_tangent_world is None and local_tangent_px is not None:
                local_a_world = project_pixel_to_plane_world(
                    np.asarray(px, dtype=np.float64) - 4.0 * np.asarray(local_tangent_px, dtype=np.float64),
                    cam_K,
                    T_cam_world,
                    plane_z_m=projection_z,
                )
                local_b_world = project_pixel_to_plane_world(
                    np.asarray(px, dtype=np.float64) + 4.0 * np.asarray(local_tangent_px, dtype=np.float64),
                    cam_K,
                    T_cam_world,
                    plane_z_m=projection_z,
                )
                if local_a_world is not None and local_b_world is not None:
                    tangent_xy = np.asarray(local_b_world[:2] - local_a_world[:2], dtype=np.float64)
                    tangent_norm = float(np.linalg.norm(tangent_xy))
                    if tangent_norm > 1e-9:
                        local_tangent_world = tangent_xy / tangent_norm

            if local_tangent_world is None:
                local_tangent_world = axis_world_vec

            assert local_tangent_world is not None
            cross_section_world = perpendicular_world_axis(local_tangent_world)
            primary_yaw = top_down_yaw_from_world_axis(cross_section_world)
            # Only keep yaws perpendicular to object axis. Yaws parallel to the
            # axis would slip the gripper off elongated objects even though they
            # may have lower IK error during batch ranking.
            yaw_sequence = (
                primary_yaw,
                normalize_angle_deg(primary_yaw + 180.0),
            )
            kept_sample_pixels.append(np.asarray(px, dtype=np.float64))
            projected_sample_worlds.append((world, yaw_sequence, frac))

        if not projected_sample_worlds:
            is_axis_guided = False
            sample_pixels = [centroid_px]
        else:
            sample_pixels = kept_sample_pixels
            for world, yaw_sequence, frac in projected_sample_worlds:
                for yaw_idx, yaw in enumerate(yaw_sequence):
                    if len(candidates) >= int(max_grasps):
                        break
                    rank = len(candidates)
                    planner_xyz = [
                        round(float(world[0]), 5),
                        round(float(world[1]), 5),
                        round(float(_TWO_D_GRASP_PLANNER_Z_M), 5),
                    ]
                    planner_rpy = [0.0, 180.0, round(float(normalize_angle_deg(yaw)), 4)]
                    # Bias scoring toward the center of the object's major axis
                    score = _score_for_axis_position(frac, yaw_idx)
                    candidates.append(
                        GraspCandidate(
                            position=planner_xyz,
                            rpy=planner_rpy,
                            score=score,
                            width=_TWO_D_GRASP_WIDTH_M,
                        )
                    )
                    grasp_rows.append(
                        {
                            "rank": rank + 1,
                            "score": score,
                            "width": _TWO_D_GRASP_WIDTH_M,
                            "raw_xyz": planner_xyz,
                            "raw_rpy": planner_rpy,
                            "planner_xyz": planner_xyz,
                            "planner_rpy": planner_rpy,
                            "status": "returned",
                            "status_reason": "2D top-down candidate generated from the SAM3 mask.",
                            "thumbnail_b64": None,
                        }
                    )
                if len(candidates) >= int(max_grasps):
                    break

    if not is_axis_guided:
        sample_pixels = [centroid_px]
        for yaw in _TWO_D_CENTER_ONLY_YAWS_DEG[: max(0, int(max_grasps))]:
            rank = len(candidates)
            planner_xyz = [
                round(float(center_world[0]), 5),
                round(float(center_world[1]), 5),
                round(float(_TWO_D_GRASP_PLANNER_Z_M), 5),
            ]
            planner_rpy = [0.0, 180.0, round(float(normalize_angle_deg(yaw)), 4)]
            score = _score_for_rank(rank)
            candidates.append(
                GraspCandidate(
                    position=planner_xyz,
                    rpy=planner_rpy,
                    score=score,
                    width=_TWO_D_GRASP_WIDTH_M,
                )
            )
            grasp_rows.append(
                {
                    "rank": rank + 1,
                    "score": score,
                    "width": _TWO_D_GRASP_WIDTH_M,
                    "raw_xyz": planner_xyz,
                    "raw_rpy": planner_rpy,
                    "planner_xyz": planner_xyz,
                    "planner_rpy": planner_rpy,
                    "status": "returned",
                    "status_reason": "Center-only 2D top-down candidate generated from an isotropic SAM3 mask.",
                    "thumbnail_b64": None,
                }
            )

    overlay_jpeg = _render_overlay_jpeg(
        rgb,
        mask01,
        centroid_px=centroid_px,
        sample_pixels=sample_pixels,
        major_axis_endpoints=axis_endpoints_px,
        candidates=candidates,
        cam_K=cam_K,
        T_cam_world=T_cam_world,
    )
    debug = {
        "camera": camera,
        "object_name": object_name,
        "n_grasps": int(len(candidates)),
        "status": "ok" if candidates else "no_grasps",
        "backend": "2dgrasp",
        "best_score": float(candidates[0].score) if candidates else None,
        "grasps": grasp_rows,
        "major_axis_ratio": float(ratio),
        "axis_guided": bool(is_axis_guided),
        "yaw_mode": "local_cross_section" if is_axis_guided else "center_only",
        "projection_z_m": projection_z,
    }
    return Grasp2DPlanResult(
        candidates=candidates,
        grasp_rows=grasp_rows,
        overlay_jpeg=overlay_jpeg,
        debug=debug,
    )


class SampleGraspPose2DTool(Tool):
    """Plan top-down grasps from a 2D mask only."""

    name = "sample_grasp_pose_2d"
    description = (
        "Generate top-down 2D grasp pose candidates for a named object using only "
        "the SAM3 segmentation mask from the top camera. Returns a list of "
        "GraspCandidate(position, rpy, score, width) in the same planner-facing "
        "format as sample_grasp_pose_anygrasp."
    )
    parameters = [
        ToolParameter("object_name", "str", "Name of the object to grasp (e.g. 'pliers')."),
        ToolParameter(
            "camera",
            "str",
            "Camera to use ('top', 'left', or 'right').",
            required=False,
            default="top",
        ),
        ToolParameter(
            "max_grasps",
            "int",
            "Max number of grasp candidates to return.",
            required=False,
            default=10,
        ),
        ToolParameter(
            "mask",
            "numpy.ndarray",
            "Pre-computed binary mask (uint8 HxW, 0/1). If provided, skip SAM3 segmentation.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "grasp_z_m",
            "float",
            "Override world-frame Z (meters) for all grasp candidates. Defaults to library default (~table + 3cm).",
            required=False,
            default=None,
        ),
        ToolParameter(
            "projection_z_m",
            "float",
            "World-frame Z plane used to project mask pixels into XY. Defaults to the table surface. "
            "Use the same value as grasp_z_m for elevated edges/rims.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "return_debug",
            "bool",
            "When true, return a dict with grasps plus debug fields like the 2D bbox from the projected segmented cloud.",
            required=False,
            default=False,
        ),
        ToolParameter(
            "reuse_cached_frame",
            "bool",
            "When true, reuse the tool's most recently captured RGB/depth/intrinsics/extrinsics for the same camera instead of grabbing a new frame.",
            required=False,
            default=False,
        ),
        ToolParameter(
            "publish_2d_bbox_viz",
            "bool",
            "When true (default), CAP UI auto-publishes the standard 2D grasp overlay to the 2dBBBox panel. "
            "Set false when a custom script will publish its own derived overlay instead.",
            required=False,
            default=True,
        ),
    ]

    def __init__(
        self,
        cap_server_host: str = "localhost",
        cap_server_port: int = CAP_SERVER_PORT,
        sam3_url: str = f"http://{SAM3_SERVER_HOST}:{SAM3_SERVER_PORT}",
        env=None,
    ):
        self._env = env
        self._cap_host = cap_server_host
        self._cap_port = cap_server_port
        self._sam3_url = sam3_url.rstrip("/")
        self._portal_client = None
        self.last_rgb: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        self.last_overlay_jpeg: bytes | None = None
        self.last_native_viz_png: bytes | None = None
        self.last_graspnet_debug: dict[str, Any] | None = None
        self.last_grasp_debug: dict[str, Any] | None = None
        self.last_bbox_xywh: list[int] | None = None
        self.last_bbox_corners_px: list[list[float]] | None = None
        self.last_depth: np.ndarray | None = None
        self.last_cam_K: np.ndarray | None = None
        self.last_T_cam_world: np.ndarray | None = None
        self.last_camera: str | None = None

    def _get_portal(self):
        if self._portal_client is None:
            import portal

            self._portal_client = portal.Client(f"{self._cap_host}:{self._cap_port}")
        return self._portal_client

    def _get_rgb_depth_intrinsics(
        self, camera: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._env is not None:
            rgb_raw = self._env.render_rgb(camera)
            depth_raw = self._env.render_depth(camera)
            intr_raw = self._env.get_camera_intrinsics(camera)
        else:
            client = self._get_portal()
            rgb_raw = client.get_camera_image(camera).result()
            depth_raw = client.get_camera_depth(camera).result()
            intr_raw = client.get_camera_intrinsics(camera).result()
        rgb = np.asarray(rgb_raw)
        depth = np.asarray(depth_raw).astype(np.float32)
        if rgb.size < 100:
            raise ValueError(f"No image returned for camera {camera!r}")
        if depth.size < 100:
            raise ValueError(f"No depth returned for camera {camera!r}")
        fx, fy, cx, cy = [float(x) for x in intr_raw]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return rgb, depth, K

    def _get_extrinsics(self, camera: str) -> np.ndarray:
        if self._env is not None:
            extr = self._env.get_camera_extrinsics(camera)
        else:
            client = self._get_portal()
            extr = client.get_camera_extrinsics(camera).result()
        R = np.asarray(extr["rotation"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(extr["position"], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        if extr.get("needs_optical_flip", True):
            F = np.diag([-1.0, -1.0, 1.0])
            T[:3, :3] = R @ F
        else:
            T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _segment_object(self, rgb: np.ndarray, object_name: str) -> np.ndarray:
        import urllib.error
        import urllib.request

        buf = io.BytesIO()
        np.save(buf, rgb)
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = json.dumps({"text": object_name, "image_b64": image_b64}).encode()
        req = urllib.request.Request(
            f"{self._sam3_url}/segment",
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
        return np.load(io.BytesIO(mask_bytes)).astype(np.int32)

    def execute(self, **kwargs: Any) -> ToolResult:
        object_name: str = kwargs["object_name"]
        camera: str = kwargs.get("camera", "top")
        max_grasps: int = int(kwargs.get("max_grasps", 10))
        pre_mask = kwargs.get("mask", None)
        grasp_z_m = kwargs.get("grasp_z_m", None)
        projection_z_m = kwargs.get("projection_z_m", None)
        return_debug = bool(kwargs.get("return_debug", False))
        reuse_cached_frame = bool(kwargs.get("reuse_cached_frame", False))

        self.last_mask = None
        self.last_overlay_jpeg = None
        self.last_native_viz_png = None
        self.last_graspnet_debug = {
            "camera": camera,
            "object_name": object_name,
            "n_grasps": 0,
            "status": "error",
            "backend": "2dgrasp",
        }
        self.last_grasp_debug = self.last_graspnet_debug
        self.last_bbox_xywh = None
        self.last_bbox_corners_px = None

        try:
            use_cached_frame = (
                reuse_cached_frame
                and self.last_camera == camera
                and self.last_rgb is not None
                and self.last_depth is not None
                and self.last_cam_K is not None
                and self.last_T_cam_world is not None
            )
            if use_cached_frame:
                rgb = np.asarray(self.last_rgb)
                depth = np.asarray(self.last_depth)
                K = np.asarray(self.last_cam_K, dtype=np.float64)
                T_cam_world = np.asarray(self.last_T_cam_world, dtype=np.float64)
            else:
                rgb, depth, K = self._get_rgb_depth_intrinsics(camera)
                T_cam_world = self._get_extrinsics(camera)
            if pre_mask is not None:
                mask = np.asarray(pre_mask, dtype=np.int32)
            else:
                mask = self._segment_object(rgb, object_name)

            self.last_rgb = rgb
            self.last_depth = depth
            self.last_cam_K = K
            self.last_T_cam_world = T_cam_world
            self.last_camera = camera
            self.last_mask = mask

            points_world = extract_segmented_object_world_points(
                depth,
                K,
                T_cam_world,
                mask,
                z_range=_ANYGRASP_Z_RANGE,
            )
            segmented_cloud_point_count = int(points_world.shape[0])
            segmented_cloud_height_m = compute_segmented_cloud_height_m(points_world)
            bbox_debug = _compute_pixel_ransac_obb(
                _masked_valid_pixel_coords(
                    depth,
                    K,
                    mask,
                    z_range=_ANYGRASP_Z_RANGE,
                ),
                source="tool_segmented_region_2d_ransac_obb",
            )

            result = plan_top_down_grasps_from_mask(
                rgb=rgb,
                mask=mask,
                cam_K=K,
                T_cam_world=T_cam_world,
                object_name=object_name,
                camera=camera,
                max_grasps=max_grasps,
                projection_z_m=projection_z_m,
            )
            self.last_overlay_jpeg = result.overlay_jpeg
            self.last_graspnet_debug = {
                **result.debug,
                "segmented_cloud_point_count": segmented_cloud_point_count,
                "segmented_cloud_height_m": segmented_cloud_height_m,
                **(bbox_debug or {}),
            }
            self.last_grasp_debug = self.last_graspnet_debug
            if bbox_debug is not None:
                self.last_bbox_xywh = list(bbox_debug["bbox_xywh"])
                self.last_bbox_corners_px = [list(c) for c in bbox_debug["bbox_corners_px"]]
            if not result.candidates:
                return ToolResult(
                    success=False,
                    error=f"2dgrasp found no grasps for {object_name!r}",
                )
            if grasp_z_m is not None:
                override_z = float(grasp_z_m)
                for cand in result.candidates:
                    cand.position[2] = round(override_z, 5)
            try:
                from enpire.env.forge.cap.agent.tools._artifact_log import log_grasp

                log_grasp(
                    rgb,
                    mask,
                    result.candidates,
                    query=object_name,
                    tag="grasp_2d",
                )
            except Exception:
                logger.debug("Failed to save 2D grasp artifact", exc_info=True)
            logger.info(
                "[2dgrasp] %d top-down grasps for %r (best score=%s)",
                len(result.candidates),
                object_name,
                result.candidates[0].score if result.candidates else "n/a",
            )
            if return_debug:
                return ToolResult(
                    success=True,
                    data={
                        "grasps": result.candidates,
                        "debug": dict(self.last_graspnet_debug or {}),
                        "overlay_jpeg": result.overlay_jpeg,
                        "rgb": rgb,
                        "mask": mask,
                    },
                )
            return ToolResult(success=True, data=result.candidates)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
