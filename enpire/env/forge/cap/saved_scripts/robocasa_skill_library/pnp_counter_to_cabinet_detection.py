# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pnp_counter_to_cabinet_detection.py — PickPlaceCounterToCabinet vision helpers
import re

import numpy as np
from skill_library.namespace import *  # noqa: F401, F403

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.detection import detect_object_v1


def object_query_from_task_v1(default="object"):
    try:
        info = get_task_info()
        obj_query = str(info.get("obj_name") or "").strip()
        if obj_query:
            return obj_query
    except Exception:
        pass

    try:
        task_desc = get_task_description()
        obj_query, _ = parse_counter_to_cabinet_task_description_v1(task_desc)
        return obj_query or default
    except Exception:
        return default


def parse_counter_to_cabinet_task_description_v1(desc):
    m = re.search(
        r"[Pp]ick (?:up )?(?:the )?(.+?) from .+ place "
        r"(?:it )?(?:on |in |into )?(?:the )?(.+?)(?:\.|$)",
        desc.strip(),
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", "open cabinet"


def refine_object_position_from_segment_v1(
    query,
    camera,
    *,
    min_depth_m=0.10,
    min_valid_points=30,
    tall_z_range_m=0.05,
    z_band_m=0.003,
):
    """Refine object xyz from the segmented mask and depth.

    For tall masks, use the upper world-z band so grasp targets land near the
    object top instead of the middle of the full mask.
    """
    query = str(query or "").strip() or "object"
    seg = segment_object(query, camera=camera, score_thresh=0.05)
    depth = np.asarray(get_camera_depth(camera), dtype=np.float32)
    transform, fx, fy, cx, cy = _camera_transform(camera)
    points_world = _mask_points_world(
        seg["mask"],
        depth,
        transform,
        fx,
        fy,
        cx,
        cy,
        min_depth_m=min_depth_m,
        min_valid_points=min_valid_points,
    )

    z_values = points_world[:, 2]
    z_range = float(np.max(z_values) - np.min(z_values))
    mode = "mask_median"
    selected = points_world
    z_upper = None

    if z_range > float(tall_z_range_m):
        z_upper = float(np.max(z_values))
        band = points_world[np.abs(z_values - z_upper) <= float(z_band_m)]
        if len(band) > 0:
            selected = band
            mode = "upper_z_band"

    pos = np.median(selected, axis=0)
    return {
        "query": query,
        "camera": camera,
        "pos": np.asarray(pos, dtype=float),
        "score": float(seg.get("score", 0.0)),
        "mode": mode,
        "z_range_m": z_range,
        "z_upper_m": z_upper,
        "n_valid": int(len(points_world)),
        "n_used": int(len(selected)),
    }


def estimate_segment_grasp_orientation_v1(
    query,
    *,
    cameras=("top",),
    required=True,
    min_depth_m=0.10,
    min_valid_points=30,
    min_aspect_ratio=1.20,
):
    """Estimate long/short mask axes for a parallel-jaw grasp.

    The long side is the first PCA axis of the SAM mask. For bread-like objects
    the gripper should usually close across the short side, so motion code uses
    both local-X-short and local-Y-short quaternion candidates.
    """
    last_error = None
    for camera in cameras:
        try:
            seg = segment_object(query, camera=camera, score_thresh=0.05)
            axis_info = _mask_pca_axes(seg["mask"])
            if axis_info["aspect_ratio"] < float(min_aspect_ratio):
                raise RuntimeError(
                    "mask is not elongated enough for stable grasp orientation "
                    f"(aspect={axis_info['aspect_ratio']:.2f})"
                )
            depth = np.asarray(get_camera_depth(camera), dtype=np.float32)
            transform, fx, fy, cx, cy = _camera_transform(camera)
            long_axis_world = _pixel_axis_to_world_xy(
                axis_info["center_px"],
                axis_info["long_axis_px"],
                depth,
                seg["mask"],
                transform,
                fx,
                fy,
                cx,
                cy,
                min_depth_m=min_depth_m,
                min_valid_points=min_valid_points,
            )
            short_axis_world = _pixel_axis_to_world_xy(
                axis_info["center_px"],
                axis_info["short_axis_px"],
                depth,
                seg["mask"],
                transform,
                fx,
                fy,
                cx,
                cy,
                min_depth_m=min_depth_m,
                min_valid_points=min_valid_points,
            )
            if long_axis_world is None:
                raise RuntimeError("could not lift mask long axis to world XY")
            if short_axis_world is None:
                short_axis_world = np.array(
                    [-long_axis_world[1], long_axis_world[0], 0.0], dtype=float
                )
                short_axis_world = _normalize(short_axis_world)

            out = {
                "query": query,
                "camera": camera,
                "score": float(seg.get("score", 0.0)),
                "mask_area": int(seg.get("mask_area", 0)),
                "bbox_xywh": seg.get("bbox_xywh"),
                **axis_info,
                "long_axis_world": long_axis_world,
                "short_axis_world": short_axis_world,
            }
            print(
                "  Grasp orientation from mask: "
                f"camera={camera} aspect={out['aspect_ratio']:.2f} "
                f"long_px={_fmt(out['long_axis_px'], 3)} "
                f"short_px={_fmt(out['short_axis_px'], 3)} "
                f"long_world={_fmt(long_axis_world)} "
                f"short_world={_fmt(short_axis_world)}"
            )
            return out
        except Exception as exc:
            last_error = exc
            print(
                f"  Grasp orientation from {camera} failed: "
                f"{type(exc).__name__}: {exc}"
            )

    if required:
        raise RuntimeError(f"could not estimate grasp orientation: {last_error}")
    return None


def detect_open_cabinet_top_right_v1(
    *,
    target_query="open cabinet",
    fallback_queries=("cabinet", "shelf", "cupboard"),
):
    views = []
    queries = [target_query, *fallback_queries]
    for camera in ("top", "right"):
        view = detect_object_v1(queries, cameras=camera)
        views.append(view)
        print(
            f"  Cabinet {view['query']!r} from {camera}: "
            f"pos={_fmt(view['pos'])} score={view['score']:.3f}"
        )

    pos = np.mean([view["pos"] for view in views], axis=0)
    print(f"  Fused cabinet pos={_fmt(pos)}")
    return {
        "query": target_query,
        "camera": "top+right",
        "pos": pos,
        "views": views,
        "score": float(np.mean([view["score"] for view in views])),
    }


def _grasp_orientation_camera_order(best_camera, cameras):
    ordered = []
    if "top" in cameras:
        ordered.append("top")
    if best_camera not in ordered:
        ordered.append(best_camera)
    for camera in cameras:
        if camera not in ordered:
            ordered.append(camera)
    return tuple(ordered)


def _mask_pca_axes(mask):
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) < 2:
        raise RuntimeError("mask has too few pixels for PCA")

    coords = np.column_stack([xs, ys]).astype(float)
    center = np.median(coords, axis=0)
    centered = coords - center
    cov = np.cov(centered, rowvar=False, bias=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]

    long_axis = np.asarray(eigvecs[:, order[0]], dtype=float)
    if long_axis[0] < 0.0 or (abs(long_axis[0]) < 1e-9 and long_axis[1] < 0.0):
        long_axis = -long_axis
    short_axis = np.array([-long_axis[1], long_axis[0]], dtype=float)

    long_proj = centered @ long_axis
    short_proj = centered @ short_axis
    long_len = _robust_extent(long_proj)
    short_len = _robust_extent(short_proj)
    aspect_ratio = float(long_len / max(short_len, 1e-6))

    return {
        "center_px": center,
        "long_axis_px": long_axis,
        "short_axis_px": short_axis,
        "long_length_px": float(long_len),
        "short_length_px": float(short_len),
        "aspect_ratio": aspect_ratio,
    }


def _robust_extent(values):
    lo, hi = np.percentile(np.asarray(values, dtype=float), [2.0, 98.0])
    return float(max(hi - lo, 0.0))


def _camera_transform(camera):
    fx, fy, cx, cy = [float(x) for x in get_camera_intrinsics(camera)]
    extr = get_camera_extrinsics(camera)
    rot = np.asarray(extr["rotation"], dtype=float).reshape(3, 3)
    pos = np.asarray(extr["position"], dtype=float)
    transform = np.eye(4, dtype=float)
    if extr.get("needs_optical_flip", True):
        transform[:3, :3] = rot @ np.diag([-1.0, -1.0, 1.0])
    else:
        transform[:3, :3] = rot
    transform[:3, 3] = pos
    return transform, fx, fy, cx, cy


def _pixel_axis_to_world_xy(
    center_px,
    axis_px,
    depth,
    mask,
    transform,
    fx,
    fy,
    cx,
    cy,
    *,
    min_depth_m,
    min_valid_points,
):
    mask_bool = np.asarray(mask) > 0
    valid = mask_bool & np.isfinite(depth) & (depth > float(min_depth_m))
    if int(valid.sum()) < int(min_valid_points):
        raise RuntimeError(f"only {int(valid.sum())} valid mask depth points")

    depth_m = float(np.median(depth[valid].astype(float)))
    axis_px = _normalize(np.asarray(axis_px, dtype=float))
    if float(np.linalg.norm(axis_px)) < 1e-6:
        return None

    ys, xs = np.where(mask_bool)
    span_px = max(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
    step_px = max(8.0, min(40.0, span_px * 0.25))
    center_px = np.asarray(center_px, dtype=float)
    p0 = _pixel_to_world_at_depth(
        center_px - axis_px * step_px,
        depth_m,
        transform,
        fx,
        fy,
        cx,
        cy,
    )
    p1 = _pixel_to_world_at_depth(
        center_px + axis_px * step_px,
        depth_m,
        transform,
        fx,
        fy,
        cx,
        cy,
    )
    axis_world = np.asarray(p1 - p0, dtype=float)
    axis_world[2] = 0.0
    axis_world = _normalize(axis_world)
    if float(np.linalg.norm(axis_world)) < 1e-6:
        return None
    return axis_world


def _mask_points_world(
    mask,
    depth,
    transform,
    fx,
    fy,
    cx,
    cy,
    *,
    min_depth_m,
    min_valid_points,
):
    valid = (
        (np.asarray(mask) > 0)
        & np.isfinite(depth)
        & (depth > float(min_depth_m))
    )
    if int(valid.sum()) < int(min_valid_points):
        raise RuntimeError(f"only {int(valid.sum())} valid mask depth points")
    vs, us = np.where(valid)
    zs = depth[valid].astype(float)
    xs = (us.astype(float) - float(cx)) * zs / float(fx)
    ys = (vs.astype(float) - float(cy)) * zs / float(fy)
    points_cam = np.stack([xs, ys, zs], axis=1)
    return (transform[:3, :3] @ points_cam.T).T + transform[:3, 3]


def _pixel_to_world_at_depth(pixel_xy, depth_m, transform, fx, fy, cx, cy):
    u, v = [float(x) for x in np.asarray(pixel_xy, dtype=float)]
    z = float(depth_m)
    x = (u - float(cx)) * z / float(fx)
    y = (v - float(cy)) * z / float(fy)
    pt_cam = np.array([x, y, z], dtype=float)
    return transform[:3, :3] @ pt_cam + transform[:3, 3]


def _normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros_like(v)


def _fmt(v, ndigits=3):
    return [round(float(x), ndigits) for x in np.asarray(v, dtype=float)]
