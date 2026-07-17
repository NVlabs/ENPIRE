"""3D bounding-box grasp planning tool for the CAP agent.

Uses RANSAC plane fitting on the segmented object point cloud to compute an
oriented bounding box (OBB), identifies the true top face using the world-up
direction, and returns a top-down grasp perpendicular to the longest axis of
that top face.  No remote AnyGrasp server is needed — only SAM3 for
segmentation and the cap_server for camera data.
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
from enpire.env.forge.cap.agent.tools.grasp_2d import (
    GraspCandidate,
)
from enpire.env.forge.cap.config import (
    CAP_SERVER_PORT,
    GRIPPER_TCP_OFFSET_Z_M,
    SAM3_SERVER_HOST,
    SAM3_SERVER_PORT,
)

logger = logging.getLogger(__name__)

_ANYGRASP_Z_RANGE = (1e-6, 1.5)
_DEFAULT_RELAX_XYZ_OFFSETS_M = (
    (0.0, 0.0, 0.0),
    (0.003, 0.0, 0.0),
    (-0.003, 0.0, 0.0),
    (0.0, 0.003, 0.0),
    (0.0, -0.003, 0.0),
    (0.003, 0.003, 0.0),
    (-0.003, -0.003, 0.0),
    (-0.003, 0.003, 0.0),
)
_DEFAULT_RELAX_YAW_OFFSETS_DEG = (0.0, 8.0, -8.0)


def _normalize_angle_deg(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _depth_to_points(depth: np.ndarray, cam_K: np.ndarray) -> np.ndarray:
    fx, fy = float(cam_K[0, 0]), float(cam_K[1, 1])
    cx, cy = float(cam_K[0, 2]), float(cam_K[1, 2])
    xmap = np.arange(depth.shape[1], dtype=np.float32)
    ymap = np.arange(depth.shape[0], dtype=np.float32)
    xmap, ymap = np.meshgrid(xmap, ymap)
    pz = depth.astype(np.float32)
    return np.stack([(xmap - cx) / fx * pz, (ymap - cy) / fy * pz, pz], axis=-1)


def _ransac_plane_fit(
    pts: np.ndarray,
    *,
    distance_thresh: float = 0.002,
    n_iterations: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(pts)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(n_iterations, 3))
    p0, p1, p2 = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms.ravel() > 1e-12
    norms = np.where(norms > 1e-12, norms, 1.0)
    normals = normals / norms
    dists = np.abs(
        (pts[np.newaxis, :, :] - p0[:, np.newaxis, :]) @ normals[:, :, np.newaxis]
    ).squeeze(-1)
    counts = np.where(valid, (dists < distance_thresh).sum(axis=1), 0)
    best_idx = int(np.argmax(counts))
    if int(counts[best_idx]) < 3:
        return np.array([0.0, 0.0, 1.0]), np.ones(n, dtype=bool)
    best_normal = normals[best_idx]
    best_mask = np.abs((pts - p0[best_idx]) @ best_normal) < distance_thresh
    return best_normal, best_mask


def _compute_ransac_obb(
    pts: np.ndarray,
    *,
    distance_thresh: float = 0.002,
    n_iterations: int = 200,
    second_plane_min_fraction: float = 0.08,
    second_plane_angle_min_deg: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-plane RANSAC OBB. Returns (center, axes_3x3, extents)."""
    pts = np.asarray(pts, dtype=np.float64)
    n1, inlier1 = _ransac_plane_fit(
        pts, distance_thresh=distance_thresh, n_iterations=n_iterations
    )
    outlier_pts = pts[~inlier1]
    two_plane = False
    if len(outlier_pts) >= max(10, int(len(pts) * second_plane_min_fraction)):
        n2, inlier2_local = _ransac_plane_fit(
            outlier_pts, distance_thresh=distance_thresh, n_iterations=n_iterations
        )
        angle_between = np.degrees(np.arccos(np.clip(abs(float(n1 @ n2)), 0.0, 1.0)))
        n_inlier2 = int(inlier2_local.sum())
        if angle_between >= second_plane_angle_min_deg and n_inlier2 >= max(
            6, int(len(pts) * 0.03)
        ):
            two_plane = True

    if two_plane:
        edge_axis = np.cross(n1, n2)
        edge_norm = np.linalg.norm(edge_axis)
        if edge_norm < 1e-9:
            two_plane = False
        else:
            edge_axis /= edge_norm
            ax_perp = np.cross(edge_axis, n1)
            ax_perp /= np.linalg.norm(ax_perp)
            axes = np.column_stack([edge_axis, ax_perp, n1])

    if not two_plane:
        inliers = pts[inlier1]
        centroid_plane = inliers.mean(axis=0)
        projected = inliers - np.outer(((inliers - centroid_plane) @ n1), n1)
        centered_2d = projected - centroid_plane
        cov_2d = centered_2d.T @ centered_2d
        _eigvals, eigvecs = np.linalg.eigh(cov_2d)
        ax0 = eigvecs[:, 1]
        ax1 = np.cross(n1, ax0)
        ax1 /= np.linalg.norm(ax1)
        axes = np.column_stack([ax1, ax0, n1])

    centroid = pts.mean(axis=0)
    all_proj = (pts - centroid) @ axes
    mins, maxs = all_proj.min(axis=0), all_proj.max(axis=0)
    extents = maxs - mins
    center = centroid + axes @ ((mins + maxs) / 2.0)
    return center, axes, extents


def _find_top_face(
    axes_cam: np.ndarray, extents: np.ndarray, up_in_cam: np.ndarray
) -> tuple[int, float, int, int]:
    """Find which OBB axis is the top-face normal using world-up direction.

    Returns (normal_idx, normal_sign, long_idx, short_idx).
    """
    best_dot = -np.inf
    normal_idx, normal_sign = 2, 1.0
    for i in range(3):
        d = float(axes_cam[:, i] @ up_in_cam)
        if d > best_dot:
            best_dot, normal_idx, normal_sign = d, i, 1.0
        if -d > best_dot:
            best_dot, normal_idx, normal_sign = -d, i, -1.0
    in_plane = [i for i in range(3) if i != normal_idx]
    if extents[in_plane[0]] >= extents[in_plane[1]]:
        long_idx, short_idx = in_plane[0], in_plane[1]
    else:
        long_idx, short_idx = in_plane[1], in_plane[0]
    return normal_idx, normal_sign, long_idx, short_idx


@dataclass
class BBoxGraspResult:
    candidate: GraspCandidate
    obb_center_world: list[float]
    obb_extents: list[float]
    top_surface_z: float
    n_points: int
    top_normal_world: list[float] | None = None
    long_axis_world: list[float] | None = None
    short_axis_world: list[float] | None = None
    top_center_world: list[float] | None = None
    # Extents mapped to the semantic axes above.  These make downstream state
    # checks independent of the raw OBB axis ordering.
    top_normal_extent: float | None = None
    top_face_long_extent: float | None = None
    top_face_short_extent: float | None = None
    obb_axes_world: list[list[float]] | None = None


def _quat_xyzw_to_display_rpy(quat_xyzw):
    from scipy.spatial.transform import Rotation

    ex, ey, ez = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_euler(
        "xyz", degrees=True
    )
    disp = np.array([ey, -ex, -ez - 90.0], dtype=np.float64)
    disp = (disp + 180.0) % 360.0 - 180.0
    return disp.tolist()


def _as_offset_triplets(value: Any, default: tuple[tuple[float, float, float], ...]):
    if value is None:
        return default
    return tuple(tuple(float(x) for x in offset[:3]) for offset in value)


def _as_float_offsets(value: Any, default: tuple[float, ...]):
    if value is None:
        return default
    return tuple(float(x) for x in value)


def _topdown_candidates_from_result(
    result: "BBoxGraspResult",
    *,
    relax: bool = False,
    relax_xyz_offsets_m: Any = None,
    relax_yaw_offsets_deg: Any = None,
    score_decay: float = 0.95,
) -> list[GraspCandidate]:
    """Return strict top-down candidates, optionally with small XY/yaw fallbacks."""
    base = result.candidate
    pos = [float(v) for v in base.position]
    yaw = float(base.rpy[2])
    strict_yaws = (yaw, yaw + 180.0)
    xyz_offsets = (
        _as_offset_triplets(relax_xyz_offsets_m, _DEFAULT_RELAX_XYZ_OFFSETS_M)
        if relax
        else ((0.0, 0.0, 0.0),)
    )
    yaw_offsets = (
        _as_float_offsets(relax_yaw_offsets_deg, _DEFAULT_RELAX_YAW_OFFSETS_DEG)
        if relax
        else (0.0,)
    )
    candidates: list[GraspCandidate] = []
    for base_yaw in strict_yaws:
        for pi, xyz_offset in enumerate(xyz_offsets):
            for yi, yaw_offset in enumerate(yaw_offsets):
                penalty = int(pi != 0) + int(yi != 0)
                candidates.append(
                    GraspCandidate(
                        position=[
                            round(float(pos[i]) + float(xyz_offset[i]), 5)
                            for i in range(3)
                        ],
                        rpy=[
                            0.0,
                            180.0,
                            round(_normalize_angle_deg(base_yaw + yaw_offset), 4),
                        ],
                        score=float(base.score) * (float(score_decay) ** penalty),
                        width=float(base.width),
                    )
                )
    return sorted(candidates, key=lambda c: -float(c.score))


def plan_grasp_from_3d_bbox(
    object_points_cam: np.ndarray,
    T_cam_world: np.ndarray,
    *,
    tcp_offset_z_m: float = 0.0,
) -> BBoxGraspResult | None:
    """Compute a grasp approaching along the surface normal (into the top face).

    Returns a single BBoxGraspResult or None if there aren't enough points.
    """
    from scipy.spatial.transform import Rotation

    pts = np.asarray(object_points_cam, dtype=np.float64)
    if pts.shape[0] < 4:
        return None

    center_cam, axes_cam, extents = _compute_ransac_obb(pts)
    R_cw = T_cam_world[:3, :3]
    t_cw = T_cam_world[:3, 3]
    up_in_cam = R_cw.T @ np.array([0.0, 0.0, 1.0])

    normal_idx, normal_sign, long_idx, short_idx = _find_top_face(
        axes_cam, extents, up_in_cam
    )

    pts_world = (R_cw @ pts.T).T + t_cw
    cloud_center_xy = pts_world[:, :2].mean(axis=0)
    top_surface_z = float(np.max(pts_world[:, 2]))

    axes_world = R_cw @ axes_cam
    normal_world = axes_world[:, normal_idx] * normal_sign
    if normal_world[2] < 0:
        normal_world = -normal_world

    approach = -normal_world
    approach /= np.linalg.norm(approach)

    long_axis_world = axes_world[:, long_idx]
    jaw_axis = np.cross(long_axis_world, approach)
    jaw_norm = np.linalg.norm(jaw_axis)
    if jaw_norm < 1e-9:
        jaw_axis = np.array([1.0, 0.0, 0.0])
    else:
        jaw_axis /= jaw_norm
    third = np.cross(approach, jaw_axis)
    third /= np.linalg.norm(third)

    gripper_R = np.column_stack([jaw_axis, third, approach])
    quat_xyzw = Rotation.from_matrix(gripper_R).as_quat()
    planner_rpy = _quat_xyzw_to_display_rpy(quat_xyzw)

    grasp_pos = np.array([cloud_center_xy[0], cloud_center_xy[1], top_surface_z])
    grasp_pos = grasp_pos + tcp_offset_z_m * approach

    center_world = R_cw @ center_cam + t_cw
    half = extents / 2.0
    top_center_cam = (
        center_cam + axes_cam[:, normal_idx] * normal_sign * half[normal_idx]
    )
    top_center_world = R_cw @ top_center_cam + t_cw

    candidate = GraspCandidate(
        position=[round(float(v), 5) for v in grasp_pos],
        rpy=[round(float(v), 4) for v in planner_rpy],
        score=1.0,
        width=round(float(extents[short_idx]), 5),
    )
    return BBoxGraspResult(
        candidate=candidate,
        obb_center_world=[round(float(v), 5) for v in center_world],
        obb_extents=[round(float(v), 5) for v in extents],
        top_surface_z=round(top_surface_z, 5),
        n_points=int(pts.shape[0]),
        top_normal_world=[round(float(v), 5) for v in normal_world],
        long_axis_world=[round(float(v), 5) for v in axes_world[:, long_idx]],
        short_axis_world=[round(float(v), 5) for v in axes_world[:, short_idx]],
        top_center_world=[round(float(v), 5) for v in top_center_world],
        top_normal_extent=round(float(extents[normal_idx]), 5),
        top_face_long_extent=round(float(extents[long_idx]), 5),
        top_face_short_extent=round(float(extents[short_idx]), 5),
        obb_axes_world=[
            [round(float(x), 5) for x in axes_world[:, i]] for i in range(3)
        ],
    )


def _render_overlay(
    rgb: np.ndarray,
    cam_K: np.ndarray,
    T_cam_world: np.ndarray,
    obb_center_cam: np.ndarray,
    obb_axes_cam: np.ndarray,
    obb_extents: np.ndarray,
) -> bytes:
    """Render OBB wireframe + top face overlay on RGB, return JPEG bytes."""
    import cv2

    h, w = rgb.shape[:2]
    canvas = rgb.copy()
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
    up_in_cam = T_cam_world[:3, :3].T @ np.array([0.0, 0.0, 1.0])

    def proj(pt3d):
        if pt3d[2] <= 0:
            return None
        u = int(round(fx * pt3d[0] / pt3d[2] + cx))
        v = int(round(fy * pt3d[1] / pt3d[2] + cy))
        if 0 <= u < w and 0 <= v < h:
            return (u, v)
        return None

    half = obb_extents / 2.0
    corners_local = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=np.float64,
    )
    corners = obb_center_cam + (corners_local * half) @ obb_axes_cam.T
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        pa, pb = proj(corners[a]), proj(corners[b])
        if pa and pb:
            cv2.line(canvas, pa, pb, (0, 255, 255), 2, cv2.LINE_AA)

    normal_idx, normal_sign, long_idx, short_idx = _find_top_face(
        obb_axes_cam, obb_extents, up_in_cam
    )
    normal_up = obb_axes_cam[:, normal_idx] * normal_sign
    fc = obb_center_cam + normal_up * half[normal_idx]
    long_vec = obb_axes_cam[:, long_idx] * half[long_idx]
    short_vec = obb_axes_cam[:, short_idx] * half[short_idx]
    top_corners = [
        fc + long_vec + short_vec,
        fc + long_vec - short_vec,
        fc - long_vec - short_vec,
        fc - long_vec + short_vec,
    ]
    px_top = [proj(c) for c in top_corners]
    valid_top = [p for p in px_top if p is not None]
    if len(valid_top) >= 3:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [np.array(valid_top, dtype=np.int32)], (200, 150, 50))
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
    for ea, eb in [
        (fc + short_vec - long_vec, fc + short_vec + long_vec),
        (fc - short_vec - long_vec, fc - short_vec + long_vec),
    ]:
        pa, pb = proj(ea), proj(eb)
        if pa and pb:
            cv2.line(canvas, pa, pb, (0, 0, 255), 3, cv2.LINE_AA)

    arrow_len = float(max(obb_extents)) * 0.5
    long_unit = obb_axes_cam[:, long_idx]
    short_unit = obb_axes_cam[:, short_idx]
    fc_px = proj(fc)

    # +X axis (long, red)
    x_tip = fc + long_unit * arrow_len
    x_tip_px = proj(x_tip)
    if fc_px and x_tip_px:
        cv2.arrowedLine(
            canvas, fc_px, x_tip_px, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.2
        )
        if x_tip_px:
            cv2.putText(
                canvas,
                "+X",
                (x_tip_px[0] + 5, x_tip_px[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    # +Y axis (short, green)
    y_tip = fc + short_unit * arrow_len
    y_tip_px = proj(y_tip)
    if fc_px and y_tip_px:
        cv2.arrowedLine(
            canvas, fc_px, y_tip_px, (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.2
        )
        if y_tip_px:
            cv2.putText(
                canvas,
                "+Y",
                (y_tip_px[0] + 5, y_tip_px[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


class SampleGraspPose3DBBoxTool(Tool):
    """Plan a top-down grasp from a RANSAC 3D bounding box."""

    name = "sample_grasp_pose_3d_bb"
    description = (
        "Generate a top-down grasp pose for a named object using RANSAC plane "
        "fitting on the segmented point cloud. Returns a GraspCandidate with "
        "yaw perpendicular to the longest visible-face axis. No AnyGrasp server "
        "needed — only SAM3 and cap_server."
    )
    parameters = [
        ToolParameter(
            "object_name", "str", "Name of the object to grasp (e.g. 'black block')."
        ),
        ToolParameter(
            "camera",
            "str",
            "Camera to use (default: 'top').",
            required=False,
            default="top",
        ),
        ToolParameter(
            "tcp_offset_z_m",
            "float",
            "TCP offset along approach direction (meters).",
            required=False,
            default=GRIPPER_TCP_OFFSET_Z_M,
        ),
        ToolParameter(
            "relax",
            "bool",
            "When true, return strict top-down candidates plus small XY/yaw fallback candidates.",
            required=False,
            default=False,
        ),
        ToolParameter(
            "relax_xyz_offsets_m",
            "list",
            "Optional XYZ offsets for relaxed candidates. Defaults to small +/-3mm XY offsets.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "relax_yaw_offsets_deg",
            "list",
            "Optional yaw offsets for relaxed candidates. Defaults to 0,+8,-8 degrees.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "image_bbox",
            "list",
            "Optional [x_min,y_min,x_max,y_max] pixel ROI. Pixels outside are zeroed in both RGB and depth before SAM3 / point-cloud fitting.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "min_world_z",
            "float",
            "Optional minimum world Z for segmented depth points before OBB fitting.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "max_world_z",
            "float",
            "Optional maximum world Z for segmented depth points before OBB fitting.",
            required=False,
            default=None,
        ),
    ]

    def __init__(
        self,
        cap_server_host: str = "localhost",
        cap_server_port: int = CAP_SERVER_PORT,
        sam3_url: str = f"http://{SAM3_SERVER_HOST}:{SAM3_SERVER_PORT}",
        env=None,
    ):
        self._cap_host = cap_server_host
        self._cap_port = cap_server_port
        self._sam3_url = sam3_url.rstrip("/")
        self._env = env
        self._portal_client = None
        self.last_overlay_jpeg: bytes | None = None
        self.last_rgb: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        self.last_grasp_debug: dict[str, Any] | None = None
        self.last_bbox_result: BBoxGraspResult | None = None

    def _get_portal(self):
        if self._portal_client is None:
            import portal

            self._portal_client = portal.Client(f"{self._cap_host}:{self._cap_port}")
        return self._portal_client

    def _get_rgb_depth_intrinsics(
        self, camera: str, image_bbox=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from enpire.env.forge.cap.agent.tools.rgbd import get_rgb_depth_intrinsics

        return get_rgb_depth_intrinsics(
            camera, env=self._env, get_portal=self._get_portal, image_bbox=image_bbox
        )

    def _get_extrinsics(self, camera: str) -> np.ndarray:
        from enpire.env.forge.cap.agent.tools.rgbd import get_extrinsics

        return get_extrinsics(camera, env=self._env, get_portal=self._get_portal)

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
        tcp_offset_z_m: float = float(
            kwargs.get("tcp_offset_z_m", GRIPPER_TCP_OFFSET_Z_M)
        )
        relax = bool(kwargs.get("relax", False))
        image_bbox = kwargs.get("image_bbox")
        min_world_z = kwargs.get("min_world_z")
        max_world_z = kwargs.get("max_world_z")
        min_world_z = None if min_world_z is None else float(min_world_z)
        max_world_z = None if max_world_z is None else float(max_world_z)
        self.last_overlay_jpeg = None
        self.last_rgb = None
        self.last_mask = None
        self.last_grasp_debug = {
            "camera": camera,
            "object_name": object_name,
            "n_grasps": 0,
            "status": "error",
            "backend": "3dbbgrasp",
        }

        try:
            rgb, depth, K = self._get_rgb_depth_intrinsics(
                camera, image_bbox=image_bbox
            )
            T_cam_world = self._get_extrinsics(camera)
            mask = self._segment_object(rgb, object_name)
            self.last_rgb = rgb
            self.last_mask = mask

            points_all = _depth_to_points(depth, K)
            valid = (
                np.isfinite(points_all).all(axis=-1)
                & (points_all[:, :, 2] > _ANYGRASP_Z_RANGE[0])
                & (points_all[:, :, 2] < _ANYGRASP_Z_RANGE[1])
            )
            object_mask = (mask > 0) & valid
            object_points_cam = points_all[object_mask].astype(np.float64)
            raw_n_points = int(object_points_cam.shape[0])

            if (
                (min_world_z is not None or max_world_z is not None)
                and raw_n_points > 0
            ):
                R_cw = T_cam_world[:3, :3]
                t_cw = T_cam_world[:3, 3]
                pts_world = (R_cw @ object_points_cam.T).T + t_cw
                keep = np.ones(raw_n_points, dtype=bool)
                if min_world_z is not None:
                    keep &= pts_world[:, 2] >= min_world_z
                if max_world_z is not None:
                    keep &= pts_world[:, 2] <= max_world_z
                object_points_cam = object_points_cam[keep]

            if object_points_cam.shape[0] < 4:
                return ToolResult(
                    success=False,
                    error=(
                        f"Not enough points for 3D bounding box of {object_name!r} "
                        f"after world-Z filter: kept={int(object_points_cam.shape[0])} "
                        f"raw={raw_n_points} min_world_z={min_world_z} "
                        f"max_world_z={max_world_z}"
                    ),
                )

            center_cam, axes_cam, extents = _compute_ransac_obb(object_points_cam)
            result = plan_grasp_from_3d_bbox(
                object_points_cam,
                T_cam_world,
                tcp_offset_z_m=tcp_offset_z_m,
            )
            if result is None:
                return ToolResult(
                    success=False,
                    error=f"3D BB planning failed for {object_name!r}",
                )

            try:
                self.last_overlay_jpeg = _render_overlay(
                    rgb, K, T_cam_world, center_cam, axes_cam, extents
                )
                import cv2

                bgr = cv2.imdecode(
                    np.frombuffer(self.last_overlay_jpeg, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if bgr is not None:
                    from enpire.env.forge.cap.agent.tools._artifact_log import log_image

                    log_image(
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                        tag="grasp_3d_bb",
                        label=object_name,
                        subdir="grasp",
                    )
            except Exception:
                logger.exception("[3d-bb-grasp] overlay render failed")

            candidates = _topdown_candidates_from_result(
                result,
                relax=relax,
                relax_xyz_offsets_m=kwargs.get("relax_xyz_offsets_m"),
                relax_yaw_offsets_deg=kwargs.get("relax_yaw_offsets_deg"),
            )
            for candidate in candidates:
                candidate.bbox_result = result
            grasp_rows = [
                {
                    "rank": rank,
                    "score": float(c.score),
                    "width": float(c.width),
                    "planner_xyz": [float(v) for v in c.position],
                    "planner_rpy": [float(v) for v in c.rpy],
                    "status": "candidate",
                    "status_reason": (
                        "3D-BB relaxed top-down grasp candidate."
                        if relax
                        else "3D-BB strict top-down grasp candidate."
                    ),
                }
                for rank, c in enumerate(candidates, start=1)
            ]
            self.last_grasp_debug = {
                "camera": camera,
                "object_name": object_name,
                "n_grasps": len(candidates),
                "status": "ok",
                "backend": "3dbbgrasp",
                "view": "bev",
                "relax": relax,
                "n_points": result.n_points,
                "n_points_before_world_z_filter": raw_n_points,
                "min_world_z": min_world_z,
                "max_world_z": max_world_z,
                "top_surface_z": result.top_surface_z,
                "obb_extents": result.obb_extents,
                "top_normal_extent": result.top_normal_extent,
                "top_face_long_extent": result.top_face_long_extent,
                "top_face_short_extent": result.top_face_short_extent,
                "grasps": grasp_rows,
            }
            logger.info(
                "[3d-bb-grasp] %s: candidates=%d relax=%s pos=%s rpy=%s width=%.4f top_z=%.4f n_pts=%d raw_pts=%d z_filter=%s..%s",
                object_name,
                len(candidates),
                relax,
                candidates[0].position if candidates else None,
                candidates[0].rpy if candidates else None,
                candidates[0].width if candidates else 0.0,
                result.top_surface_z,
                result.n_points,
                raw_n_points,
                min_world_z,
                max_world_z,
            )
            self.last_bbox_result = result
            return ToolResult(success=True, data=candidates)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
