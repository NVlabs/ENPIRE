"""Shared RGB+depth+intrinsics+extrinsics fetch for grasp tools.

Centralizes the env-direct vs portal-RPC dispatch that was previously copy-
pasted across grasp_2d / grasp_3d_bb / grasp_anygrasp, and adds an optional
image_bbox crop that zeros pixels outside the box (so SAM3 sees only the ROI
and depth outside the ROI drops out of the point cloud).

image_bbox = (x_min, y_min, x_max, y_max) in image pixel coordinates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _zero_outside_bbox(
    rgb: np.ndarray, depth: np.ndarray, image_bbox
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in image_bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"image_bbox out of frame: {image_bbox} for image ({h}, {w})")
    rgb_out = np.zeros_like(rgb)
    rgb_out[y0:y1, x0:x1] = rgb[y0:y1, x0:x1]
    depth_out = np.zeros_like(depth)
    depth_out[y0:y1, x0:x1] = depth[y0:y1, x0:x1]
    return rgb_out, depth_out


def get_rgb_depth_intrinsics(
    camera: str,
    *,
    env: Any = None,
    get_portal: Callable[[], Any] | None = None,
    image_bbox: tuple[int, int, int, int] | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rgb, depth, K). If image_bbox is given, zero pixels outside it."""
    if env is not None:
        rgb_raw = env.render_rgb(camera)
        depth_raw = env.render_depth(camera)
        intr_raw = env.get_camera_intrinsics(camera)
    elif get_portal is not None:
        client = get_portal()
        rgb_raw = client.get_camera_image(camera).result()
        depth_raw = client.get_camera_depth(camera).result()
        intr_raw = client.get_camera_intrinsics(camera).result()
    else:
        raise RuntimeError(
            "get_rgb_depth_intrinsics: pass env=... (direct mode) or get_portal=... (RPC)"
        )
    if rgb_raw is None:
        raise RuntimeError(f"No RGB image available from camera={camera!r}")
    if depth_raw is None:
        raise RuntimeError(f"No depth image available from camera={camera!r}")
    rgb = np.asarray(rgb_raw)
    depth = np.asarray(depth_raw).astype(np.float32)
    if rgb.size < 100:
        raise ValueError(f"No image returned for camera {camera!r}")
    if depth.size < 100:
        raise ValueError(f"No depth returned for camera {camera!r}")
    fx, fy, cx, cy = (float(x) for x in intr_raw)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if image_bbox is not None:
        rgb, depth = _zero_outside_bbox(rgb, depth, image_bbox)
    return rgb, depth, K


def get_extrinsics(
    camera: str,
    *,
    env: Any = None,
    get_portal: Callable[[], Any] | None = None,
) -> np.ndarray:
    """Return T_cam_world (4x4). Applies the optical-axis flip when requested."""
    if env is not None:
        extr = env.get_camera_extrinsics(camera)
    elif get_portal is not None:
        extr = get_portal().get_camera_extrinsics(camera).result()
    else:
        raise RuntimeError(
            "get_extrinsics: pass env=... (direct mode) or get_portal=... (RPC)"
        )
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
