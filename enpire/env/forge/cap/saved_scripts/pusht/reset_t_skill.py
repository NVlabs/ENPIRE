# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: F405
"""PushT red-T pose detection and reset skills.

This file carries a local copy of the red-T detector primitives so saved
scripts do not depend on the standalone ``bc.detect_red_t`` module at runtime.
The detector fits the full red silhouette and uses top-camera extrinsics to
choose desk/top-face corner heights.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from skill_library.namespace import *  # noqa: F401, F403

from enpire.env.forge.cap.agent.skill_registry import skill

DEFAULT_CAMERA = "top"
DEFAULT_PLANE_PATH = Path("tmp/desk_plane.json")

# PushT reset target recorded from logs/grasp_t_20260517T190353 after the T was
# manually placed in the reset pose.
DEFAULT_TARGET_XY = [0.4225568770347879, 0.0106398977622877]
DEFAULT_PUSH_SIDE = "right"
DEFAULT_PUSH_RPY = [0.0, 125.0, -20.0]
DEFAULT_PUSH_CLEARANCE_M = 0.16
DEFAULT_PUSH_CONTACT_Z_OFFSET_M = 0.045
DEFAULT_PRE_PUSH_STANDOFF_M = 0.10
DEFAULT_POST_PUSH_OVERSHOOT_M = 0.035
DEFAULT_POS_TOL_M = 0.025
DEFAULT_MAX_PUSH_M = 0.25
PLANNING_SPEED = 4.0
MIN_SCORE_RMSE_MM = 80.0
DEFAULT_TARGET_YAW_DEG = -85.91082681079712
ALIGN_GRIPPER_TO_T_YAW = True
PICK_GRASP_OFFSET_M = 0.0
PICK_GRASP_OBJECT_OFFSET_XY_M = [0.0, -0.035]
PICK_GRASP_Z_OFFSETS_M = [0.070, 0.060, 0.050]
PICK_HOVER_Z_M = 0.90
PICK_LIFT_Z_M = 0.90
PICK_PLACE_CLEARANCE_M = 0.12
PICK_GRASP_Z_OFFSET_M = 0.070
PICK_PLACE_Z_OFFSET_M = 0.045
PICK_TOPDOWN_RPY = [0.0, -180.0, 0.0]
PICK_GRIPPER_OPEN = 1.0
PICK_GRIPPER_CLOSED = 0.0
PICK_EMPTY_GRIPPER_POS_MAX = 0.02
PICK_CAMERA_CLEAR_X_M = 0.45
PICK_CAMERA_CLEAR_Y_M = 0.30
PICK_CAMERA_CLEAR_Z_M = 0.95
PICK_CHECKPOINT_IMAGES = True
PICK_VISUAL_SERVO_BEFORE_CLOSE = True
PICK_VISUAL_SERVO_MAX_NUDGE_M = 0.040
PICK_VISUAL_SERVO_DEADBAND_M = 0.006
PICK_VISUAL_SERVO_DARK_THRESHOLD = 125
RESET_T_RELEASE_EE_POS = [
    0.3884698030094744,
    -0.005656966961950395,
    0.8169382532228835,
]
RESET_T_RELEASE_EE_RPY = [0.0, -180.0, -4.0891731892028815]
LEFT_HANDOFF_RELEASE_EE_POS = [
    float(v)
    for v in os.environ.get(
        "PUSHT_LEFT_HANDOFF_RELEASE_EE_POS",
        "0.4650891,0.07250419,0.8169382532228835",
    ).split(",")
]
LEFT_HANDOFF_RELEASE_EE_RPY = [
    float(v)
    for v in os.environ.get(
        "PUSHT_LEFT_HANDOFF_RELEASE_EE_RPY",
        "0.0,-180.0,-4.0891731892028815",
    ).split(",")
]
RESET_T_HOLD_GRIPPER_POS = 0.35
STANDARD_WAIT_FOR_GRASP_YAW_DEG = -148.00959532689163
UPSIDE_DOWN_YAW_THRESHOLD_DEG = 75.0
NORMALIZE_RELEASE_Z_EXTRA_M = 0.04
POST_RELEASE_LIFT_CLEARANCE_M = 0.03
PUSHT_OPEN_GRIPPER_VEL_LIMIT = 30.0
RESET_OK_REFERENCE_IMAGE = Path("cap/tasks/pusht/standard_initial_top.png")
RESET_OK_REFERENCE_META = Path("cap/tasks/pusht/standard_initial_top_meta.json")
RESET_OK_THRESHOLD = 0.5
RESET_OK_CENTROID_SIGMA_PX = 35.0
RESET_OK_ANGLE_SIGMA_DEG = 25.0
RESET_OK_MIN_DOMINANT_COMPONENT_FRACTION = 0.75

T_TOTAL_WIDTH_M = 0.16
T_TOTAL_HEIGHT_M = 0.160
T_STEM_WIDTH_M = 0.040
T_CROSSBAR_H_M = 0.040
T_THICKNESS_M = 0.030
MIN_CONTOUR_AREA_PX = 500
MODEL_FALLBACK_MIN_VISIBLE_FRACTION = float(
    os.environ.get("PUSHT_MODEL_FALLBACK_MIN_VISIBLE_FRACTION", "0.25")
)
MODEL_FALLBACK_MIN_SCORE = float(
    os.environ.get("PUSHT_MODEL_FALLBACK_MIN_SCORE", "0.18")
)
RED_HSV_LOW1 = np.array([0, 80, 80], dtype=np.uint8)
RED_HSV_HIGH1 = np.array([10, 255, 255], dtype=np.uint8)
RED_HSV_LOW2 = np.array([165, 80, 80], dtype=np.uint8)
RED_HSV_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)
_POLY_EPSILONS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
WORLD_FRAME = "base_link"
_CAMERA_TRANSFORM_LOGGED = set()


def _as_rgb_uint8(image):
    if image is None:
        raise RuntimeError("No camera image returned")
    arr = np.asarray(image)
    if arr.dtype == object:
        raise RuntimeError(f"Expected numeric RGB image, got object array: {arr!r}")
    if arr.dtype != np.uint8:
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise RuntimeError(f"Expected RGB image, got shape={arr.shape}")
    return arr[:, :, :3]


def _log_debug_image(image, *, tag, label=""):
    try:
        from enpire.env.forge.cap.agent.tools._artifact_log import log_image

        path = log_image(image, tag=tag, label=label, subdir="pusht")
        return None if path is None else str(path)
    except Exception as exc:
        print(f"  reset_t: failed to save debug image {tag}: {exc}")
        return None


def _draw_debug_overlay(
    rgb,
    *,
    mask=None,
    corners=None,
    ordered=None,
    contour=None,
    title="PushT debug",
    lines=(),
):
    import cv2

    canvas = rgb.copy()
    if mask is not None:
        mask_bool = np.asarray(mask).astype(bool)
        tint = canvas.copy()
        tint[mask_bool] = np.array([255, 70, 70], dtype=np.uint8)
        canvas = (
            canvas.astype(np.float32) * 0.62 + tint.astype(np.float32) * 0.38
        ).astype(np.uint8)

    if contour is not None:
        cv2.drawContours(canvas, [np.asarray(contour, dtype=np.int32)], -1, (255, 230, 0), 2)

    pts = ordered if ordered is not None else corners
    if pts is not None:
        pts_i = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(canvas, [pts_i], isClosed=True, color=(0, 255, 255), thickness=2)
        for idx, (x, y) in enumerate(pts_i):
            cv2.circle(canvas, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                str(idx),
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    text_lines = [title, *[str(line) for line in lines if line]]
    y = 24
    for line in text_lines:
        cv2.putText(
            canvas,
            line[:120],
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line[:120],
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return canvas


def _load_t_world_cam_from_station():
    import yourdfpy

    from enpire.env.forge.robot.models.station.paths import (
        get_station_urdf,
        get_top_camera_frame,
        needs_optical_flip,
    )

    urdf_path = get_station_urdf()
    if not urdf_path.exists():
        raise FileNotFoundError(f"Station URDF not found: {urdf_path}")
    print(f"  reset_t: fallback loading station URDF camera transform from {urdf_path}")
    urdf = yourdfpy.URDF.load(str(urdf_path))
    cam_frame = get_top_camera_frame()
    T_world_urdf = np.asarray(
        urdf.get_transform(cam_frame, WORLD_FRAME), dtype=np.float64
    )
    if needs_optical_flip("top"):
        T_world_cam = T_world_urdf @ np.diag([-1.0, -1.0, 1.0, 1.0])
    else:
        T_world_cam = T_world_urdf.copy()
    print(
        f"  reset_t: station camera frame={cam_frame} "
        f"origin={T_world_cam[:3, 3].tolist()}"
    )
    return T_world_cam, cam_frame


def _build_model_points_8():
    W, H, sw, ch = T_TOTAL_WIDTH_M, T_TOTAL_HEIGHT_M, T_STEM_WIDTH_M, T_CROSSBAR_H_M
    return np.array(
        [
            [-W / 2, H / 2, 0.0],
            [W / 2, H / 2, 0.0],
            [W / 2, H / 2 - ch, 0.0],
            [sw / 2, H / 2 - ch, 0.0],
            [sw / 2, -H / 2, 0.0],
            [-sw / 2, -H / 2, 0.0],
            [-sw / 2, H / 2 - ch, 0.0],
            [-W / 2, H / 2 - ch, 0.0],
        ],
        dtype=np.float64,
    )


def _red_mask(
    bgr,
    h1_lo=int(RED_HSV_LOW1[0]),
    h1_hi=int(RED_HSV_HIGH1[0]),
    h2_lo=int(RED_HSV_LOW2[0]),
    h2_hi=int(RED_HSV_HIGH2[0]),
    s_min=int(RED_HSV_LOW1[1]),
    v_min=int(RED_HSV_LOW1[2]),
    lab_a_min=0,
    r_min=195,
    g_max=255,
    b_max=255,
):
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.bitwise_or(
        cv2.inRange(
            hsv,
            np.array([h1_lo, s_min, v_min], dtype=np.uint8),
            np.array([h1_hi, 255, 255], dtype=np.uint8),
        ),
        cv2.inRange(
            hsv,
            np.array([h2_lo, s_min, v_min], dtype=np.uint8),
            np.array([h2_hi, 255, 255], dtype=np.uint8),
        ),
    )
    if lab_a_min > 0:
        a_star = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 1]
        mask = cv2.bitwise_or(hsv_mask, cv2.inRange(a_star, int(lab_a_min), 255))
    else:
        mask = hsv_mask
    if r_min > 0 or g_max < 255 or b_max < 255:
        rgb_mask = cv2.inRange(
            bgr,
            np.array([0, 0, r_min], dtype=np.uint8),
            np.array([b_max, g_max, 255], dtype=np.uint8),
        )
        mask = cv2.bitwise_and(mask, rgb_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _find_best_t_contour(mask):
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA_PX]
    if not candidates:
        return None, None
    best_contour = max(candidates, key=cv2.contourArea)
    arc = cv2.arcLength(best_contour, closed=True)
    for eps_frac in _POLY_EPSILONS:
        approx = cv2.approxPolyDP(best_contour, eps_frac * arc, closed=True)
        if len(approx) == 8:
            return approx.reshape(-1, 2).astype(np.float32), best_contour
    return None, best_contour


def _reset_ok_dominant_component(mask):
    """Copied connected-component red-T scoring primitive for reset_ok_v1."""
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    info = {
        "valid_component": 0.0,
        "component_count": 0.0,
        "dominant_fraction": 0.0,
        "area": 0.0,
        "cx": 0.0,
        "cy": 0.0,
        "angle_deg": 0.0,
        "topology_ok": 0.0,
    }
    if n_labels <= 1:
        return out, info

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = areas >= float(MIN_CONTOUR_AREA_PX)
    if not np.any(keep):
        return out, info

    kept_indices = np.flatnonzero(keep) + 1
    component_count = int(kept_indices.size)
    largest_label = int(kept_indices[np.argmax(stats[kept_indices, cv2.CC_STAT_AREA])])
    out[labels == largest_label] = 255

    total_area = float(np.sum(stats[kept_indices, cv2.CC_STAT_AREA]))
    area = float(stats[largest_label, cv2.CC_STAT_AREA])
    cx, cy = [float(v) for v in centroids[largest_label]]
    ys, xs = np.where(out > 0)
    angle_deg = 0.0
    if xs.size >= 5:
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        pts -= np.mean(pts, axis=0, keepdims=True)
        cov = np.cov(pts, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        angle_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))

    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    topology_ok = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        arc = cv2.arcLength(contour, True)
        for eps_frac in _POLY_EPSILONS:
            approx = cv2.approxPolyDP(contour, eps_frac * arc, True)
            if len(approx) == 8:
                topology_ok = 1.0
                break

    info.update(
        {
            "valid_component": 1.0,
            "component_count": float(component_count),
            "dominant_fraction": float(area / total_area) if total_area else 0.0,
            "area": area,
            "cx": cx,
            "cy": cy,
            "angle_deg": angle_deg,
            "topology_ok": topology_ok,
        }
    )
    return out, info


def _reset_ok_bbox_from_mask(mask, margin):
    ys, xs = np.where(np.asarray(mask) > 0)
    h, w = mask.shape[:2]
    if xs.size == 0:
        return 0, 0, w, h
    x0 = max(0, int(xs.min()) - int(margin))
    y0 = max(0, int(ys.min()) - int(margin))
    x1 = min(w, int(xs.max()) + int(margin) + 1)
    y1 = min(h, int(ys.max()) + int(margin) + 1)
    return x0, y0, x1 - x0, y1 - y0


def _reset_ok_clamp_crop(crop, image_shape):
    x, y, w, h = [int(round(float(v))) for v in crop]
    img_h, img_w = image_shape[:2]
    x0 = max(0, min(img_w, x))
    y0 = max(0, min(img_h, y))
    x1 = max(x0, min(img_w, x + max(0, w)))
    y1 = max(y0, min(img_h, y + max(0, h)))
    return x0, y0, x1 - x0, y1 - y0


def _reset_ok_crop(arr, crop):
    x, y, w, h = _reset_ok_clamp_crop(crop, arr.shape)
    return arr[y : y + h, x : x + w]


def _reset_ok_angle_error_deg(a, b):
    return float(abs((float(a) - float(b) + 90.0) % 180.0 - 90.0))


def _reset_ok_mask_score(reference, current, reference_info, current_info):
    g = reference > 0
    c = current > 0
    intersection = int(np.count_nonzero(g & c))
    union = int(np.count_nonzero(g | c))
    reference_count = int(np.count_nonzero(g))
    current_count = int(np.count_nonzero(c))
    iou = float(intersection / union) if union else 0.0
    precision = float(intersection / current_count) if current_count else 0.0
    recall = float(intersection / reference_count) if reference_count else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    overlap_score = float(min(precision, recall))
    area_score = (
        min(current_info["area"], reference_info["area"])
        / max(current_info["area"], reference_info["area"])
        if current_info["area"] > 0 and reference_info["area"] > 0
        else 0.0
    )
    centroid_dist_px = float(
        np.hypot(
            current_info["cx"] - reference_info["cx"],
            current_info["cy"] - reference_info["cy"],
        )
    )
    centroid_score = float(
        np.exp(-0.5 * (centroid_dist_px / RESET_OK_CENTROID_SIGMA_PX) ** 2)
    )
    angle_error = _reset_ok_angle_error_deg(
        current_info["angle_deg"], reference_info["angle_deg"]
    )
    angle_score = float(
        np.exp(-0.5 * (angle_error / RESET_OK_ANGLE_SIGMA_DEG) ** 2)
    )
    component_gate = float(
        current_info["valid_component"] > 0.0
        and current_info["dominant_fraction"] >= RESET_OK_MIN_DOMINANT_COMPONENT_FRACTION
    )
    topology_score = 0.5 + 0.5 * float(current_info["topology_ok"])
    score = float(component_gate * f1 * area_score * centroid_score * angle_score * topology_score)
    return {
        "score": score,
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "overlap_score": overlap_score,
        "area_score": float(area_score),
        "centroid_score": centroid_score,
        "centroid_dist_px": centroid_dist_px,
        "angle_score": angle_score,
        "angle_error_deg": angle_error,
        "component_gate": component_gate,
        "topology_score": topology_score,
        "dominant_fraction": float(current_info["dominant_fraction"]),
        "component_count": float(current_info["component_count"]),
        "reference_px": float(reference_count),
        "current_px": float(current_count),
        "intersection_px": float(intersection),
    }


def _reset_ok_load_reference(reference_image, reference_meta, margin, target_shape=None):
    import cv2

    reference_image = Path(reference_image)
    reference_meta = Path(reference_meta)
    reference_bgr = cv2.imread(str(reference_image), cv2.IMREAD_COLOR)
    if reference_bgr is None:
        raise FileNotFoundError(f"reset_ok reference image not found: {reference_image}")
    original_h, original_w = reference_bgr.shape[:2]

    crop = None
    if reference_meta.exists():
        try:
            meta = json.loads(reference_meta.read_text(encoding="utf-8"))
            crop_vals = meta.get("crop_xywh")
            if crop_vals is not None and len(crop_vals) == 4:
                crop = tuple(int(v) for v in crop_vals)
        except Exception as exc:
            print(f"  reset_ok: ignoring unreadable reference meta {reference_meta}: {exc}")

    if target_shape is not None:
        target_h, target_w = target_shape[:2]
        if (target_h, target_w) != (original_h, original_w):
            sx = float(target_w) / float(original_w)
            sy = float(target_h) / float(original_h)
            reference_bgr = cv2.resize(
                reference_bgr,
                (int(target_w), int(target_h)),
                interpolation=cv2.INTER_AREA,
            )
            if crop is not None:
                x, y, w, h = crop
                crop = (
                    int(round(x * sx)),
                    int(round(y * sy)),
                    int(round(w * sx)),
                    int(round(h * sy)),
                )
            margin = int(round(float(margin) * max(sx, sy)))

    reference_mask_full, _ = _reset_ok_dominant_component(_red_mask(reference_bgr))
    if np.count_nonzero(reference_mask_full) == 0:
        raise RuntimeError(f"reset_ok reference has no red T component: {reference_image}")
    if crop is None:
        crop = _reset_ok_bbox_from_mask(reference_mask_full, int(margin))
    crop = _reset_ok_clamp_crop(crop, reference_bgr.shape)

    reference_mask, reference_info = _reset_ok_dominant_component(
        _reset_ok_crop(reference_mask_full, crop)
    )
    reference_crop = _reset_ok_crop(reference_bgr, crop)
    return reference_crop, reference_mask, reference_info, crop


def _reset_ok_overlay(crop_bgr, reference_mask, current_mask, score, crop):
    import cv2

    vis = crop_bgr.copy()
    g = reference_mask > 0
    c = current_mask > 0
    only_reference = g & ~c
    only_current = c & ~g
    both = g & c
    vis[only_reference] = (
        0.35 * vis[only_reference] + 0.65 * np.array([0, 255, 0])
    ).astype(np.uint8)
    vis[only_current] = (
        0.35 * vis[only_current] + 0.65 * np.array([255, 0, 255])
    ).astype(np.uint8)
    vis[both] = (0.25 * vis[both] + 0.75 * np.array([0, 255, 255])).astype(np.uint8)
    lines = [
        f"reset_ok={score['score']:.3f} th={RESET_OK_THRESHOLD:.2f} F1={score['f1']:.3f} IoU={score['iou']:.3f}",
        f"area={score['area_score']:.3f} center={score['centroid_score']:.3f} ({score['centroid_dist_px']:.1f}px) angle={score['angle_score']:.3f} ({score['angle_error_deg']:.1f}deg)",
        f"gate={score['component_gate']:.0f} dominant={score['dominant_fraction']:.2f} comps={int(score['component_count'])} topology={score['topology_score']:.2f}",
        f"green=reference magenta=current yellow=overlap crop={crop}",
    ]
    y = 24
    for line in lines:
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return vis


def _longest_edge_index(pts):
    edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    return int(np.argmax(edges))


def _t_topology_ok(pts, longest_idx):
    edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    longest = edges[longest_idx]
    if longest <= 1e-6:
        return False
    return (
        edges[(longest_idx - 1) % 8] <= longest / 3.0
        and edges[(longest_idx + 1) % 8] <= longest / 3.0
    )


def _order_from_long_edge(pts, longest_idx, forward):
    n = len(pts)
    if forward:
        indices = [(longest_idx + k) % n for k in range(n)]
    else:
        indices = [(longest_idx + 1 - k) % n for k in range(n)]
    return pts[indices].astype(np.float32)


def _backproject_pixels_to_plane_frame(pixels_uv, K, dist, T_cam_plane):
    import cv2

    pixels_uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    norm = cv2.undistortPoints(pixels_uv.reshape(-1, 1, 2), K, dist, P=None).reshape(-1, 2)
    rays = np.column_stack([norm, np.ones(len(norm))])
    R_cp = T_cam_plane[:3, :3]
    t_cp = T_cam_plane[:3, 3]
    n_cam = R_cp[:, 2]
    denom = rays @ n_cam
    if np.any(np.abs(denom) < 1e-9):
        raise RuntimeError("Ray is parallel to the desk plane")
    s = (n_cam @ t_cp) / denom
    P_cam = rays * s[:, None]
    P_plane = (R_cp.T @ (P_cam - t_cp).T).T
    return P_plane[:, :2]


def _backproject_pixels_to_plane_heights(pixels_uv, heights_m, K, dist, T_cam_plane):
    import cv2

    pixels_uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    heights_m = np.asarray(heights_m, dtype=np.float64).reshape(-1)
    norm = cv2.undistortPoints(pixels_uv.reshape(-1, 1, 2), K, dist, P=None).reshape(-1, 2)
    rays = np.column_stack([norm, np.ones(len(norm))])
    R_cp = T_cam_plane[:3, :3]
    t_cp = T_cam_plane[:3, 3]
    n_cam = R_cp[:, 2]
    denom = rays @ n_cam
    if np.any(np.abs(denom) < 1e-9):
        raise RuntimeError("Ray is parallel to the desk plane")
    s = (n_cam @ t_cp + heights_m) / denom
    P_cam = rays * s[:, None]
    P_plane = (R_cp.T @ (P_cam - t_cp).T).T
    return P_plane[:, :2]


def _fit_2d_rigid(model_xy, observed_xy):
    P = np.asarray(model_xy, dtype=np.float64)
    Q = np.asarray(observed_xy, dtype=np.float64)
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _S, Vt = np.linalg.svd(H)
    D = np.eye(2)
    D[1, 1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R_2d = Vt.T @ D @ U.T
    t_2d = cQ - R_2d @ cP
    yaw = float(np.arctan2(R_2d[1, 0], R_2d[0, 0]))
    residuals = (P @ R_2d.T + t_2d) - Q
    rmse_mm = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))) * 1000.0)
    return R_2d, t_2d, yaw, rmse_mm


def _rotation_2d(yaw):
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _pose_metric_rmse_mm(model_xy, observed_xy, R_2d, t_2d):
    residuals = (model_xy @ R_2d.T + t_2d) - observed_xy
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))) * 1000.0)


def _project_t_pixels(model_xy, heights_m, R_2d, t_2d, K, dist, T_world_plane, T_world_cam):
    import cv2

    model_3d = np.column_stack([model_xy, heights_m]).astype(np.float64)
    T_plane_t = np.eye(4, dtype=np.float64)
    T_plane_t[:2, :2] = R_2d
    T_plane_t[0, 3] = t_2d[0]
    T_plane_t[1, 3] = t_2d[1]
    T_cam_t = np.linalg.inv(T_world_cam) @ T_world_plane @ T_plane_t
    rvec, _ = cv2.Rodrigues(T_cam_t[:3, :3])
    projected, _ = cv2.projectPoints(model_3d, rvec, T_cam_t[:3, 3].reshape(3, 1), K, dist)
    return projected.reshape(-1, 2)


def _refine_pose_in_pixels(
    pixels_uv,
    model_xy,
    heights_m,
    R_2d,
    t_2d,
    yaw,
    K,
    dist,
    T_cam_plane,
    T_world_plane,
    T_world_cam,
):
    try:
        from scipy.optimize import least_squares
    except Exception:
        observed_xy = _backproject_pixels_to_plane_heights(pixels_uv, heights_m, K, dist, T_cam_plane)
        return R_2d, t_2d, yaw, _pose_metric_rmse_mm(model_xy, observed_xy, R_2d, t_2d)

    pixels_uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)

    def residual(params):
        tx, ty, yaw_p = params
        proj = _project_t_pixels(
            model_xy,
            heights_m,
            _rotation_2d(float(yaw_p)),
            np.array([tx, ty], dtype=np.float64),
            K,
            dist,
            T_world_plane,
            T_world_cam,
        )
        return (proj - pixels_uv).reshape(-1)

    result = least_squares(
        residual,
        np.array([t_2d[0], t_2d[1], yaw], dtype=np.float64),
        x_scale=np.array([0.05, 0.05, 0.2], dtype=np.float64),
        max_nfev=25,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )
    t_refined = result.x[:2].astype(np.float64)
    yaw_refined = float(result.x[2])
    R_refined = _rotation_2d(yaw_refined)
    observed_xy = _backproject_pixels_to_plane_heights(pixels_uv, heights_m, K, dist, T_cam_plane)
    return R_refined, t_refined, yaw_refined, _pose_metric_rmse_mm(
        model_xy, observed_xy, R_refined, t_refined
    )


def _corner_silhouette_heights(model_xy, R_2d, t_2d, T_world_plane, T_world_cam, invert=False):
    cam_in_plane = (T_world_plane[:3, :3].T @ (T_world_cam[:3, 3] - T_world_plane[:3, 3]))[:2]
    cam_in_t = R_2d.T @ (cam_in_plane - t_2d)
    cam_dist_xy = float(np.linalg.norm(cam_in_t))
    if cam_dist_xy < 1e-6:
        return np.full(len(model_xy), T_THICKNESS_M, dtype=np.float64)
    cam_dir = cam_in_t / cam_dist_xy
    depth_along_view = model_xy @ cam_dir
    heights = np.where(
        depth_along_view >= float(np.median(depth_along_view)),
        0.0,
        T_THICKNESS_M,
    )
    if invert:
        heights = T_THICKNESS_M - heights
    return heights


def _height_assignment_candidates(model_xy, R_2d, t_2d, T_world_plane, T_world_cam):
    return [
        _corner_silhouette_heights(model_xy, R_2d, t_2d, T_world_plane, T_world_cam, invert=False),
        _corner_silhouette_heights(model_xy, R_2d, t_2d, T_world_plane, T_world_cam, invert=True),
        np.zeros(len(model_xy), dtype=np.float64),
        np.full(len(model_xy), T_THICKNESS_M, dtype=np.float64),
    ]


def _all_corner_height_assignments(n_corners):
    assignments = np.zeros((1 << n_corners, n_corners), dtype=np.float64)
    for mask in range(1 << n_corners):
        for idx in range(n_corners):
            if mask & (1 << idx):
                assignments[mask, idx] = T_THICKNESS_M
    return assignments


def _fit_extrinsic_silhouette(
    pixels_uv,
    model_xy,
    K,
    dist,
    T_cam_plane,
    T_world_plane,
    T_world_cam,
    n_iters=3,
):
    observed_xy = _backproject_pixels_to_plane_frame(pixels_uv, K, dist, T_cam_plane)
    R_2d, t_2d, yaw, rmse_mm = _fit_2d_rigid(model_xy, observed_xy)
    corner_heights = np.zeros(len(model_xy), dtype=np.float64)

    for heights in _all_corner_height_assignments(len(model_xy)):
        observed_xy = _backproject_pixels_to_plane_heights(pixels_uv, heights, K, dist, T_cam_plane)
        R_next, t_next, yaw_next, rmse_next = _fit_2d_rigid(model_xy, observed_xy)
        if rmse_next < rmse_mm:
            R_2d, t_2d, yaw, rmse_mm = R_next, t_next, yaw_next, rmse_next
            corner_heights = heights

    for _ in range(n_iters):
        best = (rmse_mm, R_2d, t_2d, yaw, corner_heights)
        for heights in _height_assignment_candidates(model_xy, R_2d, t_2d, T_world_plane, T_world_cam):
            observed_xy = _backproject_pixels_to_plane_heights(pixels_uv, heights, K, dist, T_cam_plane)
            R_next, t_next, yaw_next, rmse_next = _fit_2d_rigid(model_xy, observed_xy)
            if rmse_next < best[0]:
                best = (rmse_next, R_next, t_next, yaw_next, heights)
        rmse_mm, R_2d, t_2d, yaw, corner_heights = best

    R_refined, t_refined, yaw_refined, rmse_refined = _refine_pose_in_pixels(
        pixels_uv,
        model_xy,
        corner_heights,
        R_2d,
        t_2d,
        yaw,
        K,
        dist,
        T_cam_plane,
        T_world_plane,
        T_world_cam,
    )
    if rmse_refined < rmse_mm:
        R_2d, t_2d, yaw, rmse_mm = R_refined, t_refined, yaw_refined, rmse_refined
    return R_2d, t_2d, yaw, rmse_mm, corner_heights


def _compose_t_in_world(R_2d, t_2d, T_world_plane):
    T_plane_t = np.eye(4, dtype=np.float64)
    T_plane_t[:2, :2] = R_2d
    T_plane_t[0, 3] = t_2d[0]
    T_plane_t[1, 3] = t_2d[1]
    return T_world_plane @ T_plane_t


def _world_to_cam_pose(T_world_t, T_cam_world):
    import cv2

    T_cam_t = T_cam_world @ T_world_t
    rvec, _ = cv2.Rodrigues(T_cam_t[:3, :3])
    tvec = T_cam_t[:3, 3].reshape(3, 1)
    return rvec, tvec


def _draw_track_overlay(
    frame,
    corners,
    raw_contour,
    rvec,
    tvec,
    K,
    dist,
    world_xyz,
    yaw_deg,
    rmse_mm,
    model_corner_heights=None,
):
    import cv2

    out = frame.copy()
    if raw_contour is not None:
        cv2.drawContours(out, [raw_contour], -1, (0, 255, 80), 2)
    colors = [
        (0, 255, 0),
        (0, 200, 0),
        (255, 128, 0),
        (255, 64, 0),
        (0, 0, 255),
        (0, 0, 200),
        (200, 0, 255),
        (128, 0, 255),
    ]
    for i, pt in enumerate(corners):
        x, y = int(pt[0]), int(pt[1])
        color = colors[i % len(colors)]
        cv2.circle(out, (x, y), 6, color, -1)
        cv2.putText(out, str(i), (x + 8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if rvec is not None and tvec is not None and world_xyz is not None:
        model_3d = _build_model_points_8()
        if model_corner_heights is not None:
            model_3d[:, 2] = np.asarray(model_corner_heights, dtype=np.float64).reshape(-1)
        projected, _ = cv2.projectPoints(model_3d, rvec, tvec, K, dist)
        outline = projected.reshape(-1, 2).astype(np.int32)
        cv2.polylines(out, [outline], isClosed=True, color=(255, 255, 0), thickness=2)
        stem_tip = (model_3d[4] + model_3d[5]) / 2.0
        crossbar_mid = (model_3d[0] + model_3d[1]) / 2.0
        axis_pts, _ = cv2.projectPoints(np.array([stem_tip, crossbar_mid]), rvec, tvec, K, dist)
        ap = axis_pts.reshape(-1, 2).astype(np.int32)
        cv2.arrowedLine(out, tuple(ap[0]), tuple(ap[1]), (255, 0, 255), 2, tipLength=0.2)
        cv2.drawFrameAxes(out, K, dist, rvec, tvec, T_TOTAL_WIDTH_M * 0.5, thickness=2)
        cv2.putText(
            out,
            f"world x={world_xyz[0]:+.3f} y={world_xyz[1]:+.3f} z={world_xyz[2]:+.3f} m",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"yaw={yaw_deg:+.1f} deg rmse={rmse_mm:.1f} mm",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        status, color = f"POSE OK ({len(corners)} pts)", (0, 255, 0)
    else:
        status, color = "NO POSE", (0, 80, 255)
    cv2.putText(out, status, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def _camera_matrix(camera):
    intr = get_camera_intrinsics(camera)
    if isinstance(intr, dict):
        if "K" in intr:
            return np.asarray(intr["K"], dtype=float).reshape(3, 3)
        intr = intr.get("intrinsics", [intr["fx"], intr["fy"], intr["cx"], intr["cy"]])
    fx, fy, cx, cy = [float(v) for v in intr]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def _top_camera_to_world_from_station():
    T_world_cam, _frame = _load_t_world_cam_from_station()
    return np.asarray(T_world_cam, dtype=float).reshape(4, 4)


def _camera_to_world(camera):
    try:
        extr = get_camera_extrinsics(camera)
    except Exception as exc:
        if camera != "top":
            raise
        print(f"  reset_t: get_camera_extrinsics(top) unavailable: {exc}")
        return _top_camera_to_world_from_station()
    if isinstance(extr, dict) and "T_cam_world" in extr:
        return np.asarray(extr["T_cam_world"], dtype=float).reshape(4, 4)
    if isinstance(extr, dict) and "T_world_cam" in extr:
        return np.asarray(extr["T_world_cam"], dtype=float).reshape(4, 4)
    rot = np.asarray(extr["rotation"], dtype=float).reshape(3, 3)
    needs_flip = bool(extr.get("needs_optical_flip", True))
    if needs_flip:
        rot = rot @ np.diag([-1.0, -1.0, 1.0])
    T = np.eye(4, dtype=float)
    T[:3, :3] = rot
    T[:3, 3] = np.asarray(extr["position"], dtype=float).reshape(3)
    log_key = (str(camera), tuple(np.round(T[:3, 3], 4)), needs_flip)
    if log_key not in _CAMERA_TRANSFORM_LOGGED:
        _CAMERA_TRANSFORM_LOGGED.add(log_key)
        print(
            f"  reset_t: camera={camera} extrinsics position="
            f"{[round(float(v), 4) for v in T[:3, 3]]} "
            f"needs_optical_flip={needs_flip}"
        )
    return T


def _project_world_points(camera, points_world):
    import cv2

    points = np.asarray(points_world, dtype=float).reshape(-1, 3)
    T_cam_world = np.linalg.inv(_camera_to_world(camera))
    points_cam = (T_cam_world[:3, :3] @ points.T + T_cam_world[:3, 3:4]).T
    image_points, _ = cv2.projectPoints(
        points_cam,
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
        _camera_matrix(camera),
        np.zeros((5, 1), dtype=float),
    )
    return image_points.reshape(-1, 2), points_cam[:, 2]


def _arm_ee_pos_from_state(state, side):
    arms = state.get("arms", state) if isinstance(state, dict) else getattr(state, "arms", {})
    arm = arms.get(side) if isinstance(arms, dict) else getattr(arms, side, None)
    if arm is None:
        return None
    ee_pos = arm.get("ee_pos") if isinstance(arm, dict) else getattr(arm, "ee_pos", None)
    if ee_pos is None:
        return None
    return np.asarray(ee_pos, dtype=float).reshape(-1)[:3]


def _draw_projected_marker(canvas, camera, xyz, *, label, color, radius=7):
    import cv2

    try:
        uv, depth = _project_world_points(camera, [xyz])
    except Exception as exc:
        return f"{label}: projection failed: {exc}"
    u, v = [int(round(float(x))) for x in uv[0]]
    if float(depth[0]) <= 0:
        return f"{label}: behind camera xyz={[round(float(x), 4) for x in xyz]}"
    h, w = canvas.shape[:2]
    if -50 <= u < w + 50 and -50 <= v < h + 50:
        cv2.circle(canvas, (u, v), int(radius), color, -1)
        cv2.circle(canvas, (u, v), int(radius) + 2, (255, 255, 255), 1)
        cv2.putText(
            canvas,
            label,
            (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return (
        f"{label}: xyz={[round(float(x), 4) for x in xyz]} "
        f"uv=({u},{v}) depth={float(depth[0]):.3f}"
    )


def _draw_projected_gripper_axes(canvas, camera, pos, rpy):
    import cv2

    pos = np.asarray(pos, dtype=float).reshape(3)
    rot = _display_rpy_rotation(rpy)
    axis_len = 0.055
    axes = [
        ("jaw-x", rot.apply([axis_len, 0.0, 0.0]), (255, 170, 0)),
        ("jaw-y", rot.apply([0.0, axis_len, 0.0]), (0, 170, 255)),
    ]
    for label, delta, color in axes:
        try:
            uv, depth = _project_world_points(camera, [pos, pos + delta])
        except Exception:
            continue
        if np.any(np.asarray(depth) <= 0):
            continue
        p0 = tuple(int(round(float(x))) for x in uv[0])
        p1 = tuple(int(round(float(x))) for x in uv[1])
        cv2.arrowedLine(canvas, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(
            canvas,
            label,
            (p1[0] + 4, p1[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _object_axes_from_yaw(yaw_deg):
    yaw = np.deg2rad(float(yaw_deg))
    x_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
    y_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=float)
    return x_axis, y_axis


def _draw_projected_segment(canvas, camera, p0, p1, *, label, color):
    import cv2

    try:
        uv, depth = _project_world_points(camera, [p0, p1])
    except Exception:
        return
    if np.any(np.asarray(depth, dtype=float) <= 0.0):
        return
    a = tuple(int(round(float(x))) for x in uv[0])
    b = tuple(int(round(float(x))) for x in uv[1])
    cv2.arrowedLine(canvas, a, b, color, 2, cv2.LINE_AA, tipLength=0.2)
    cv2.putText(
        canvas,
        label,
        (b[0] + 4, b[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_t_grasp_guides(canvas, camera, pose, plan):
    if pose is None or plan is None:
        return []
    if pose.get("world_xyz") is None or pose.get("yaw_deg") is None:
        return []
    if plan.get("grasp_center") is None or plan.get("grasp_rpy") is None:
        return []

    center = np.asarray(pose["world_xyz"], dtype=float).reshape(3)
    grasp_center = np.asarray(plan["grasp_center"], dtype=float).reshape(3)
    z = float(center[2])
    object_x, object_y = _object_axes_from_yaw(float(pose["yaw_deg"]))
    grasp_plane = grasp_center.copy()
    grasp_plane[2] = z

    _draw_projected_segment(
        canvas,
        camera,
        center - object_y * 0.080,
        center + object_y * 0.080,
        label="T-handle-axis",
        color=(0, 255, 120),
    )
    _draw_projected_segment(
        canvas,
        camera,
        grasp_plane - object_x * 0.045,
        grasp_plane + object_x * 0.045,
        label="handle-width-axis",
        color=(255, 80, 80),
    )

    rot = _display_rpy_rotation(plan["grasp_rpy"])
    open_axis = rot.apply([1.0, 0.0, 0.0])
    along_axis = rot.apply([0.0, 1.0, 0.0])
    _draw_projected_segment(
        canvas,
        camera,
        grasp_plane - open_axis * 0.045,
        grasp_plane + open_axis * 0.045,
        label="jaw-open-axis",
        color=(255, 170, 0),
    )
    _draw_projected_segment(
        canvas,
        camera,
        grasp_plane - along_axis * 0.060,
        grasp_plane + along_axis * 0.060,
        label="jaw-along-axis",
        color=(0, 170, 255),
    )

    open_xy = np.asarray(open_axis[:2], dtype=float)
    along_xy = np.asarray(along_axis[:2], dtype=float)
    open_xy /= max(float(np.linalg.norm(open_xy)), 1e-9)
    along_xy /= max(float(np.linalg.norm(along_xy)), 1e-9)
    object_x_xy = object_x[:2] / max(float(np.linalg.norm(object_x[:2])), 1e-9)
    object_y_xy = object_y[:2] / max(float(np.linalg.norm(object_y[:2])), 1e-9)

    open_err = np.degrees(np.arccos(np.clip(abs(float(np.dot(open_xy, object_x_xy))), -1.0, 1.0)))
    along_err = np.degrees(np.arccos(np.clip(abs(float(np.dot(along_xy, object_y_xy))), -1.0, 1.0)))
    return [
        f"grasp_yaw_deg={float(plan['grasp_rpy'][2]):+.1f} object_yaw_deg={float(pose['yaw_deg']):+.1f}",
        f"jaw_open_vs_handle_width_err_deg={float(open_err):.1f}",
        f"jaw_along_vs_handle_axis_err_deg={float(along_err):.1f}",
    ]


def _log_motion_checkpoint(stage, *, camera, side, plan=None, pose=None, waypoint=None, attempt=None):
    if not PICK_CHECKPOINT_IMAGES:
        return None

    try:
        rgb = _as_rgb_uint8(get_camera_image(camera))
    except Exception as exc:
        print(f"  reset_t checkpoint {stage}: camera capture failed: {exc}")
        return {"stage": stage, "error": str(exc)}

    canvas = rgb.copy()
    lines = [f"stage={stage} side={side}"]
    if pose is not None and pose.get("world_xyz") is not None:
        lines.append(
            _draw_projected_marker(
                canvas,
                camera,
                pose["world_xyz"],
                label="T-center",
                color=(0, 255, 0),
                radius=8,
            )
        )
    if plan is not None and plan.get("grasp_center") is not None:
        lines.append(
            _draw_projected_marker(
                canvas,
                camera,
                plan["grasp_center"],
                label="grasp-center",
                color=(255, 0, 0),
                radius=7,
            )
        )
    lines.extend(_draw_t_grasp_guides(canvas, camera, pose, plan))
    if waypoint is not None and waypoint.get("pos") is not None:
        lines.append(
            _draw_projected_marker(
                canvas,
                camera,
                waypoint["pos"],
                label=f"cmd-{waypoint.get('name', 'wp')}",
                color=(255, 220, 0),
                radius=6,
            )
        )
        if waypoint.get("rpy") is not None:
            _draw_projected_gripper_axes(canvas, camera, waypoint["pos"], waypoint["rpy"])
    try:
        ee_pos = _arm_ee_pos_from_state(get_robot_state(), side)
    except Exception as exc:
        ee_pos = None
        lines.append(f"ee: state read failed: {exc}")
    if ee_pos is not None:
        lines.append(
            _draw_projected_marker(
                canvas,
                camera,
                ee_pos,
                label="actual-ee",
                color=(255, 0, 255),
                radius=7,
            )
        )
        if waypoint is not None and waypoint.get("pos") is not None:
            err = np.linalg.norm(np.asarray(ee_pos, dtype=float) - np.asarray(waypoint["pos"], dtype=float))
            lines.append(f"ee_to_cmd_error_m={float(err):.4f}")
    if attempt is not None:
        lines.append(f"attempt={attempt}")

    overlay = _draw_debug_overlay(
        canvas,
        title=f"PushT checkpoint: {stage}",
        lines=lines,
    )
    tag_stage = str(stage).replace(" ", "_").replace("/", "_")
    path = _log_debug_image(
        overlay,
        tag=f"pusht_checkpoint_{tag_stage}_{camera}",
        label="checkpoint",
    )
    if path:
        print(f"  reset_t checkpoint {stage}: saved to {path}")
    return {"stage": stage, "path": path, "lines": lines}


def _image_gripper_axes(camera, pos, rpy):
    pos = np.asarray(pos, dtype=float).reshape(3)
    rot = _display_rpy_rotation(rpy)
    axis_len = 0.055
    uv, depth = _project_world_points(
        camera,
        [
            pos,
            pos + rot.apply([axis_len, 0.0, 0.0]),
            pos + rot.apply([0.0, axis_len, 0.0]),
        ],
    )
    if np.any(np.asarray(depth, dtype=float) <= 0):
        raise RuntimeError("Projected gripper axes are behind camera")

    origin = np.asarray(uv[0], dtype=float)
    x_axis = np.asarray(uv[1], dtype=float) - origin
    y_axis = np.asarray(uv[2], dtype=float) - origin
    x_norm = float(np.linalg.norm(x_axis))
    y_norm = float(np.linalg.norm(y_axis))
    if x_norm < 1e-6 or y_norm < 1e-6:
        raise RuntimeError("Projected gripper axes are degenerate")
    return origin, x_axis / x_norm, y_axis / y_norm


def _backproject_world_xy_at_z(camera, pixels_uv, z_world):
    K = _camera_matrix(camera)
    dist = np.zeros((5, 1), dtype=float)
    T_cam_world = np.linalg.inv(_camera_to_world(camera))
    T_cam_plane = T_cam_world @ _horizontal_plane_at_z(float(z_world))
    xy = _backproject_pixels_to_plane_frame(pixels_uv, K, dist, T_cam_plane)
    return np.asarray(xy, dtype=float).reshape(-1, 2)


def _world_point_on_camera_ray_at_z(camera, point_world, z_world):
    uv, depth = _project_world_points(camera, [point_world])
    if float(depth[0]) <= 0.0:
        raise RuntimeError("Reference point projects behind camera")
    xy = _backproject_world_xy_at_z(camera, [uv[0]], float(z_world))[0]
    return np.array([float(xy[0]), float(xy[1]), float(z_world)], dtype=float), [
        float(v) for v in uv[0]
    ]


def _find_top_gripper_contact_midpoint(rgb, *, camera, target_world, waypoint):
    import cv2

    arr = _as_rgb_uint8(rgb)
    height, width = arr.shape[:2]
    target_uv, depth = _project_world_points(camera, [target_world])
    if float(depth[0]) <= 0.0:
        raise RuntimeError("Grasp target projects behind camera")
    target_px = np.asarray(target_uv[0], dtype=float)
    _, x_axis, y_axis = _image_gripper_axes(camera, waypoint["pos"], waypoint["rpy"])

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = _red_mask(bgr) > 0
    yy, xx = np.indices((height, width))
    points = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    rel = points - target_px.reshape(1, 2)
    local_x = rel @ x_axis
    local_y = rel @ y_axis
    radial = np.linalg.norm(rel, axis=1)

    dark = (
        (gray.ravel() < int(PICK_VISUAL_SERVO_DARK_THRESHOLD))
        & (hsv[:, :, 1].ravel() < 120)
        & (~red.ravel())
        & (radial < 145.0)
        & (np.abs(local_y) < 95.0)
        & (np.abs(local_x) > 8.0)
    )
    dark_img = dark.reshape(height, width).astype(np.uint8)
    dark_img = cv2.morphologyEx(
        dark_img,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    dark_points = np.column_stack(np.nonzero(dark_img))[:, ::-1].astype(float)
    if len(dark_points) < 20:
        raise RuntimeError("Could not find enough dark gripper pixels near target")
    rel_dark = dark_points - target_px.reshape(1, 2)
    lx = rel_dark @ x_axis
    ly = rel_dark @ y_axis
    near_gap = np.abs(ly) < 75.0
    left_candidates = np.flatnonzero(near_gap & (lx < -8.0))
    right_candidates = np.flatnonzero(near_gap & (lx > 8.0))
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        raise RuntimeError(
            "Could not find dark gripper pixels on both sides of the grasp target"
        )

    def choose(indices):
        # Pick the visible pad point closest to the target while mildly
        # preferring points that are close to the projected jaw line.
        pts = dark_points[indices]
        r = rel_dark[indices]
        lx_i = r @ x_axis
        ly_i = r @ y_axis
        score = np.abs(lx_i) + 0.45 * np.abs(ly_i)
        chosen_idx = indices[int(np.argmin(score))]
        return dark_points[chosen_idx]

    left_tip = choose(left_candidates)
    right_tip = choose(right_candidates)
    midpoint = 0.5 * (left_tip + right_tip)
    return {
        "target_px": [float(v) for v in target_px],
        "left_tip_px": [float(v) for v in left_tip],
        "right_tip_px": [float(v) for v in right_tip],
        "contact_midpoint_px": [float(v) for v in midpoint],
        "dark_pixel_count": int(len(dark_points)),
        "axis_x_px": [float(v) for v in x_axis],
        "axis_y_px": [float(v) for v in y_axis],
    }


def _draw_visual_servo_overlay(rgb, *, record, world_delta_xy, stage):
    import cv2

    canvas = _as_rgb_uint8(rgb).copy()
    target = tuple(int(round(v)) for v in record["target_px"])
    left = tuple(int(round(v)) for v in record["left_tip_px"])
    right = tuple(int(round(v)) for v in record["right_tip_px"])
    mid = tuple(int(round(v)) for v in record["contact_midpoint_px"])
    cv2.line(canvas, left, right, (255, 220, 0), 2, cv2.LINE_AA)
    cv2.circle(canvas, left, 5, (255, 0, 255), -1)
    cv2.circle(canvas, right, 5, (255, 0, 255), -1)
    cv2.circle(canvas, mid, 6, (0, 180, 255), -1)
    cv2.circle(canvas, target, 7, (0, 255, 0), 2)
    cv2.arrowedLine(canvas, mid, target, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.25)
    lines = [
        f"stage={stage}",
        f"target_px={[round(float(v), 1) for v in record['target_px']]}",
        f"contact_mid_px={[round(float(v), 1) for v in record['contact_midpoint_px']]}",
        f"world_delta_xy_m={[round(float(v), 4) for v in world_delta_xy]}",
        f"dark_px={int(record['dark_pixel_count'])}",
    ]
    return _draw_debug_overlay(canvas, title="PushT visual-servo preclose", lines=lines)


def _visual_servo_grasp_alignment(plan, *, camera, side, pose, waypoint, attempt):
    if not PICK_VISUAL_SERVO_BEFORE_CLOSE:
        return None, waypoint

    target_world = np.asarray(plan["grasp_center"], dtype=float).reshape(3)
    try:
        rgb = _as_rgb_uint8(get_camera_image(camera))
        record = _find_top_gripper_contact_midpoint(
            rgb,
            camera=camera,
            target_world=target_world,
            waypoint=waypoint,
        )
        target_xy, contact_xy = _backproject_world_xy_at_z(
            camera,
            [record["target_px"], record["contact_midpoint_px"]],
            float(target_world[2]),
        )
        delta_xy = np.asarray(target_xy - contact_xy, dtype=float).reshape(2)
        norm = float(np.linalg.norm(delta_xy))
        unclipped = delta_xy.copy()
        if norm > float(PICK_VISUAL_SERVO_MAX_NUDGE_M) > 0.0:
            delta_xy *= float(PICK_VISUAL_SERVO_MAX_NUDGE_M) / norm
            norm = float(np.linalg.norm(delta_xy))

        overlay = _draw_visual_servo_overlay(
            rgb,
            record=record,
            world_delta_xy=delta_xy,
            stage=f"attempt_{attempt}",
        )
        path = _log_debug_image(
            overlay,
            tag=f"pusht_visual_servo_attempt_{attempt}_{camera}",
            label="preclose",
        )
        print(
            "  reset_t visual_servo: "
            f"attempt={attempt} delta_xy={[round(float(v), 4) for v in delta_xy]} "
            f"unclipped={[round(float(v), 4) for v in unclipped]} path={path}"
        )

        entry = {
            "name": f"visual_servo_attempt_{attempt}",
            "type": "visual_servo",
            "status": "aligned" if norm <= PICK_VISUAL_SERVO_DEADBAND_M else "nudged",
            "path": path,
            "target_world_xy": [float(v) for v in target_xy],
            "contact_world_xy": [float(v) for v in contact_xy],
            "world_delta_xy_m": [float(v) for v in delta_xy],
            "unclipped_world_delta_xy_m": [float(v) for v in unclipped],
            "record": record,
        }
        if norm <= PICK_VISUAL_SERVO_DEADBAND_M:
            return entry, waypoint

        adjusted = dict(waypoint)
        adjusted_pos = np.asarray(adjusted["pos"], dtype=float).reshape(3)
        adjusted_pos[:2] += delta_xy
        adjusted["pos"] = [float(v) for v in adjusted_pos]
        kwargs = _move_kwargs(side, adjusted["pos"], adjusted["rpy"], preview_only=False)
        kwargs.update(_gripper_width_arg(side, adjusted.get("gripper", PICK_GRIPPER_OPEN)))
        move_result = freespace_move(**kwargs)
        entry["move_status"] = getattr(move_result, "status", None)
        entry["move_result"] = str(move_result)
        if not _result_ok(move_result):
            entry["status"] = "move_failed"
            return entry, waypoint

        checkpoint = _log_motion_checkpoint(
            f"after_visual_servo_{attempt}",
            camera=camera,
            side=side,
            plan=plan,
            pose=pose,
            waypoint={
                "name": "visual_servo",
                "pos": adjusted["pos"],
                "rpy": adjusted["rpy"],
            },
            attempt=attempt,
        )
        if checkpoint is not None:
            entry["checkpoint"] = checkpoint
        return entry, adjusted
    except Exception as exc:
        print(f"  reset_t visual_servo: attempt={attempt} failed: {exc}")
        return {
            "name": f"visual_servo_attempt_{attempt}",
            "type": "visual_servo",
            "status": "exception",
            "error": str(exc),
        }, waypoint


def _table_surface_z_from_urdf():
    import xml.etree.ElementTree as ET

    from enpire.env.forge.robot.models.station.paths import get_station_urdf

    tree = ET.parse(str(get_station_urdf()))
    for link in tree.getroot().findall("link"):
        if link.get("name") != "play_table":
            continue
        visual = link.find("visual")
        if visual is None:
            break
        origin = visual.find("origin")
        box = visual.find("geometry/box")
        if origin is None or box is None:
            break
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        size = [float(v) for v in box.get("size", "0 0 0").split()]
        return float(xyz[2] + size[2] / 2.0)
    raise RuntimeError("play_table link with box geometry not found in station URDF")


def _horizontal_plane_at_z(z_world):
    T = np.eye(4, dtype=float)
    T[2, 3] = float(z_world)
    return T


def _load_plane(path=DEFAULT_PLANE_PATH):
    if os.environ.get("PUSHT_USE_SAVED_DESK_PLANE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        payload = json.loads(Path(path).read_text())
        return np.asarray(payload["T_world_plane"], dtype=float).reshape(4, 4)

    z_world = _table_surface_z_from_urdf()
    print(f"  reset_t: using URDF play_table plane z={z_world:.3f} m")
    return _horizontal_plane_at_z(z_world)


def _red_contour_centroid_pose(
    *,
    rgb,
    mask,
    raw_contour,
    camera,
    K,
    dist,
    T_cam_plane,
    raw_path,
    mask_path,
    reason,
):
    import cv2

    if raw_contour is None or cv2.contourArea(raw_contour) < MIN_CONTOUR_AREA_PX:
        raise RuntimeError(f"Cannot use red-centroid fallback: {reason}")
    moments = cv2.moments(raw_contour)
    if abs(float(moments["m00"])) < 1e-9:
        raise RuntimeError(f"Cannot use red-centroid fallback with zero contour area: {reason}")
    centroid_px = np.array(
        [
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        ],
        dtype=np.float64,
    )
    xy = _backproject_pixels_to_plane_frame([centroid_px], K, dist, T_cam_plane)[0]
    xyz = np.array([float(xy[0]), float(xy[1]), _table_surface_z_from_urdf()], dtype=float)
    overlay = _draw_debug_overlay(
        rgb,
        mask=mask,
        contour=raw_contour,
        title=f"PushT centroid fallback camera={camera}",
        lines=(
            reason,
            f"centroid_px={[round(float(v), 1) for v in centroid_px]}",
            f"world_xyz={[round(float(v), 4) for v in xyz]}",
            f"mask={mask_path or 'not saved'}",
        ),
    )
    cv2.circle(
        overlay,
        (int(round(float(centroid_px[0]))), int(round(float(centroid_px[1])))),
        8,
        (0, 255, 0),
        2,
    )
    pose_path = _log_debug_image(
        overlay,
        tag=f"pusht_centroid_pose_{camera}",
        label="centroid",
    )
    if pose_path:
        print(f"  reset_t: centroid fallback overlay saved to {pose_path}")
    return {
        "success": True,
        "camera": camera,
        "world_xyz": [float(v) for v in xyz],
        "xy": [float(xyz[0]), float(xyz[1])],
        "z": float(xyz[2]),
        "yaw_rad": 0.0,
        "yaw_deg": 0.0,
        "rmse_mm": 0.0,
        "corner_heights": [],
        "contour_area_px": float(cv2.contourArea(raw_contour)),
        "debug_raw_path": raw_path,
        "debug_mask_path": mask_path,
        "debug_pose_path": pose_path,
        "fallback": "red_contour_centroid",
        "fallback_reason": str(reason),
        "centroid_px": [float(v) for v in centroid_px],
    }


def _project_t_outline_pixels(model_xy, R_2d, t_2d, K, dist, T_world_plane, T_world_cam):
    model_3d = np.column_stack(
        [np.asarray(model_xy, dtype=np.float64), np.full(len(model_xy), T_THICKNESS_M)]
    )
    T_plane_t = np.eye(4, dtype=np.float64)
    T_plane_t[:2, :2] = R_2d
    T_plane_t[0, 3] = float(t_2d[0])
    T_plane_t[1, 3] = float(t_2d[1])
    T_world_t = T_world_plane @ T_plane_t
    T_cam_t = np.linalg.inv(T_world_cam) @ T_world_t
    pts_cam = (T_cam_t[:3, :3] @ model_3d.T + T_cam_t[:3, 3:4]).T
    import cv2

    projected, _ = cv2.projectPoints(
        pts_cam,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        dist,
    )
    return projected.reshape(-1, 2)


def _score_projected_t_mask(observed, projected):
    import cv2

    h, w = observed.shape[:2]
    pts = np.asarray(projected, dtype=np.int32).reshape(-1, 1, 2)
    x, y, bw, bh = cv2.boundingRect(pts)
    pad = 12
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return None

    local_pts = pts.copy()
    local_pts[:, 0, 0] -= x0
    local_pts[:, 0, 1] -= y0
    model = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(model, [local_pts], 255)
    model_bool = model > 0
    model_count = int(np.count_nonzero(model_bool))
    if model_count <= 0:
        return None

    observed_roi = observed[y0:y1, x0:x1] > 0
    intersection = int(np.count_nonzero(model_bool & observed_roi))
    observed_count = int(np.count_nonzero(observed_roi))
    if observed_count <= 0:
        return None

    precision = float(intersection / observed_count)
    visible_fraction = float(intersection / model_count)
    score = float(precision * min(1.0, visible_fraction / 0.65))
    return {
        "score": score,
        "precision": precision,
        "visible_fraction": visible_fraction,
        "intersection_px": float(intersection),
        "observed_px": float(observed_count),
        "model_px": float(model_count),
        "roi_xywh": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
    }


def _red_t_model_fallback_pose(
    *,
    rgb,
    mask,
    raw_contour,
    camera,
    K,
    dist,
    T_cam_plane,
    T_world_plane,
    T_world_cam,
    raw_path,
    mask_path,
    reason,
):
    import cv2

    if raw_contour is None or cv2.contourArea(raw_contour) < MIN_CONTOUR_AREA_PX:
        raise RuntimeError(f"Cannot use model fallback: {reason}")

    contour_px = np.asarray(raw_contour, dtype=np.float64).reshape(-1, 2)
    observed_xy = _backproject_pixels_to_plane_frame(contour_px, K, dist, T_cam_plane)
    center_xy = np.mean(observed_xy, axis=0)
    centered = observed_xy - center_xy
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    pca_yaw = float(np.arctan2(axis[1], axis[0]))

    model_xy = _build_model_points_8()[:, :2]
    observed = np.asarray(mask) > 0
    yaw_offsets = np.deg2rad(np.arange(-90.0, 91.0, 15.0))
    yaw_candidates = []
    for base in (pca_yaw, pca_yaw + np.pi / 2.0):
        yaw_candidates.extend(float(base + off) for off in yaw_offsets)
    trans_offsets = [-0.06, -0.03, 0.0, 0.03, 0.06]

    best = None
    for yaw in yaw_candidates:
        R_2d = _rotation_2d(yaw)
        for dx in trans_offsets:
            for dy in trans_offsets:
                t_2d = np.asarray(center_xy + np.array([dx, dy]), dtype=np.float64)
                projected = _project_t_outline_pixels(
                    model_xy, R_2d, t_2d, K, dist, T_world_plane, T_world_cam
                )
                score = _score_projected_t_mask(observed, projected)
                if score is None:
                    continue
                if best is None or score["score"] > best["score"]["score"]:
                    best = {
                        "yaw": float(yaw),
                        "R_2d": R_2d,
                        "t_2d": t_2d,
                        "projected": projected,
                        "score": score,
                    }

    if best is None:
        raise RuntimeError(f"Model fallback found no valid projected T: {reason}")
    if (
        best["score"]["score"] < MODEL_FALLBACK_MIN_SCORE
        or best["score"]["visible_fraction"] < MODEL_FALLBACK_MIN_VISIBLE_FRACTION
    ):
        raise RuntimeError(
            "Model fallback score too low: "
            f"score={best['score']['score']:.3f} "
            f"visible={best['score']['visible_fraction']:.3f}; {reason}"
        )

    T_world_t = _compose_t_in_world(best["R_2d"], best["t_2d"], T_world_plane)
    xyz = T_world_t[:3, 3]
    yaw_deg = float(np.degrees(best["yaw"]))
    overlay = _draw_debug_overlay(
        rgb,
        mask=mask,
        contour=raw_contour,
        title=f"PushT model fallback camera={camera}",
        lines=(
            reason,
            f"score={best['score']['score']:.3f} visible={best['score']['visible_fraction']:.3f} precision={best['score']['precision']:.3f}",
            f"world_xyz={[round(float(v), 4) for v in xyz]} yaw={yaw_deg:+.1f}",
            f"mask={mask_path or 'not saved'}",
        ),
    )
    cv2.polylines(
        overlay,
        [np.asarray(best["projected"], dtype=np.int32).reshape(-1, 1, 2)],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2,
    )
    pose_path = _log_debug_image(
        overlay,
        tag=f"pusht_model_pose_{camera}",
        label="model fallback",
    )
    if pose_path:
        print(f"  reset_t: model fallback overlay saved to {pose_path}")
    return {
        "success": True,
        "camera": camera,
        "world_xyz": [float(v) for v in xyz],
        "xy": [float(xyz[0]), float(xyz[1])],
        "z": float(xyz[2]),
        "yaw_rad": float(best["yaw"]),
        "yaw_deg": yaw_deg,
        "rmse_mm": 0.0,
        "corner_heights": [float(T_THICKNESS_M)] * len(model_xy),
        "contour_area_px": float(cv2.contourArea(raw_contour)),
        "debug_raw_path": raw_path,
        "debug_mask_path": mask_path,
        "debug_pose_path": pose_path,
        "fallback": "known_t_model_mask_fit",
        "fallback_reason": str(reason),
        "model_score": best["score"],
    }


def _detect_red_t_pose(
    camera=DEFAULT_CAMERA,
    plane_path=DEFAULT_PLANE_PATH,
    allow_centroid_fallback=False,
    allow_model_fallback=True,
):
    import cv2

    rgb = _as_rgb_uint8(get_camera_image(camera))
    raw_path = _log_debug_image(rgb, tag=f"pusht_raw_{camera}", label="raw")
    if raw_path:
        print(f"  reset_t: raw camera image saved to {raw_path}")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    mask = _red_mask(bgr)
    mask_overlay = _draw_debug_overlay(
        rgb,
        mask=mask,
        title=f"PushT red mask camera={camera}",
        lines=(f"raw={raw_path or 'not saved'}",),
    )
    mask_path = _log_debug_image(
        mask_overlay,
        tag=f"pusht_red_mask_{camera}",
        label="mask",
    )
    if mask_path:
        print(f"  reset_t: red-mask overlay saved to {mask_path}")

    K = _camera_matrix(camera)
    dist = np.zeros((5, 1), dtype=float)
    T_world_cam = _camera_to_world(camera)
    T_cam_world = np.linalg.inv(T_world_cam)
    T_world_plane = _load_plane(plane_path)
    T_cam_plane = T_cam_world @ T_world_plane

    corners_2d, raw_contour = _find_best_t_contour(mask)
    if corners_2d is None:
        if allow_model_fallback and raw_contour is not None:
            try:
                return _red_t_model_fallback_pose(
                    rgb=rgb,
                    mask=mask,
                    raw_contour=raw_contour,
                    camera=camera,
                    K=K,
                    dist=dist,
                    T_cam_plane=T_cam_plane,
                    T_world_plane=T_world_plane,
                    T_world_cam=T_world_cam,
                    raw_path=raw_path,
                    mask_path=mask_path,
                    reason="No 8-corner red T contour found",
                )
            except RuntimeError as exc:
                print(f"  reset_t: model fallback failed: {exc}")
        if allow_centroid_fallback and raw_contour is not None:
            return _red_contour_centroid_pose(
                rgb=rgb,
                mask=mask,
                raw_contour=raw_contour,
                camera=camera,
                K=K,
                dist=dist,
                T_cam_plane=T_cam_plane,
                raw_path=raw_path,
                mask_path=mask_path,
                reason="No 8-corner red T contour found",
            )
        fail_overlay = _draw_debug_overlay(
            rgb,
            mask=mask,
            contour=raw_contour,
            title=f"PushT detection failed camera={camera}",
            lines=("No 8-corner red T contour found", f"mask={mask_path or 'not saved'}"),
        )
        fail_path = _log_debug_image(
            fail_overlay,
            tag=f"pusht_no_contour_{camera}",
            label="fail",
        )
        raise RuntimeError(f"No 8-corner red T contour found; debug={fail_path}")

    longest_idx = _longest_edge_index(corners_2d)
    if not _t_topology_ok(corners_2d, longest_idx):
        if allow_model_fallback:
            try:
                return _red_t_model_fallback_pose(
                    rgb=rgb,
                    mask=mask,
                    raw_contour=raw_contour,
                    camera=camera,
                    K=K,
                    dist=dist,
                    T_cam_plane=T_cam_plane,
                    T_world_plane=T_world_plane,
                    T_world_cam=T_world_cam,
                    raw_path=raw_path,
                    mask_path=mask_path,
                    reason=f"T topology failed longest_edge_index={longest_idx}",
                )
            except RuntimeError as exc:
                print(f"  reset_t: model fallback failed: {exc}")
        if allow_centroid_fallback:
            return _red_contour_centroid_pose(
                rgb=rgb,
                mask=mask,
                raw_contour=raw_contour,
                camera=camera,
                K=K,
                dist=dist,
                T_cam_plane=T_cam_plane,
                raw_path=raw_path,
                mask_path=mask_path,
                reason=f"T topology failed longest_edge_index={longest_idx}",
            )
        fail_overlay = _draw_debug_overlay(
            rgb,
            mask=mask,
            corners=corners_2d,
            contour=raw_contour,
            title=f"PushT topology failed camera={camera}",
            lines=(f"longest_edge_index={longest_idx}", f"mask={mask_path or 'not saved'}"),
        )
        fail_path = _log_debug_image(
            fail_overlay,
            tag=f"pusht_bad_topology_{camera}",
            label="fail",
        )
        raise RuntimeError(f"Largest red contour does not match T topology; debug={fail_path}")

    model_xy = _build_model_points_8()[:, :2]
    best = None
    for forward in (True, False):
        cand = _order_from_long_edge(corners_2d, longest_idx, forward)
        R_2d, t_2d, yaw, rmse_mm, heights = _fit_extrinsic_silhouette(
            cand,
            model_xy,
            K,
            dist,
            T_cam_plane,
            T_world_plane,
            T_world_cam,
        )
        if best is None or rmse_mm < best["rmse_mm"]:
            best = {
                "ordered": cand,
                "R_2d": R_2d,
                "t_2d": t_2d,
                "yaw": yaw,
                "rmse_mm": float(rmse_mm),
                "corner_heights": heights,
            }

    assert best is not None
    T_world_t = _compose_t_in_world(best["R_2d"], best["t_2d"], T_world_plane)
    xyz = T_world_t[:3, 3]
    yaw_deg = float(np.degrees(best["yaw"]))
    area_px = float(cv2.contourArea(raw_contour)) if raw_contour is not None else 0.0
    rvec_draw, tvec_draw = _world_to_cam_pose(T_world_t, T_cam_world)
    pose_overlay_bgr = _draw_track_overlay(
        bgr,
        best["ordered"],
        raw_contour,
        rvec_draw,
        tvec_draw,
        K,
        dist,
        xyz,
        yaw_deg,
        float(best["rmse_mm"]),
        best["corner_heights"],
    )
    pose_path = _log_debug_image(
        cv2.cvtColor(pose_overlay_bgr, cv2.COLOR_BGR2RGB),
        tag=f"pusht_pose_{camera}",
        label="pose",
    )
    if pose_path:
        print(f"  reset_t: pose overlay saved to {pose_path}")
    return {
        "success": True,
        "camera": camera,
        "world_xyz": [float(v) for v in xyz],
        "xy": [float(xyz[0]), float(xyz[1])],
        "z": float(xyz[2]),
        "yaw_rad": float(best["yaw"]),
        "yaw_deg": yaw_deg,
        "rmse_mm": float(best["rmse_mm"]),
        "corner_heights": [float(v) for v in best["corner_heights"]],
        "contour_area_px": area_px,
        "debug_raw_path": raw_path,
        "debug_mask_path": mask_path,
        "debug_pose_path": pose_path,
    }


def _move_kwargs(side, pos, rpy, preview_only=False):
    return {
        f"{side}_target_pos": [float(v) for v in pos],
        f"{side}_target_rpy": [float(v) for v in rpy],
        "planning_speed": PLANNING_SPEED,
        "backend": "rrt-connect",
        "planner_backend": "rrtconnect",
        "ik_error_threshold": 0.025,
        "ik_rpy_weight": 0.01,
        "ik_rot_threshold_deg": 45.0,
        "preview_only": bool(preview_only),
    }


def _gripper_width_arg(side, value):
    return {f"{side}_gripper_target_width": float(value)}


def _result_ok(result):
    status = getattr(result, "status", None)
    return status in {None, "Success", "success", "done"}


def _auto_side_for_xy(xy, requested_side):
    side = str(requested_side or "auto").strip().lower()
    if side in {"left", "right"}:
        return side
    xy = np.asarray(xy, dtype=float).reshape(2)
    try:
        state = get_robot_state()
        distances = {}
        for candidate in ("left", "right"):
            ee_xy = _arm_ee_xy_from_state(state, candidate)
            if ee_xy is not None:
                distances[candidate] = float(np.linalg.norm(ee_xy - xy))
        if distances:
            chosen = min(distances, key=distances.get)
            print(
                "  reset_t: auto side="
                f"{chosen} distances={{{', '.join(f'{k}: {v:.3f}' for k, v in distances.items())}}}"
            )
            return chosen
    except Exception as exc:
        print(f"  reset_t: auto side could not read robot state: {exc}")
    return "left" if float(xy[1]) >= 0.0 else "right"


def _topdown_rpy_for_yaw(yaw_deg):
    rpy = list(PICK_TOPDOWN_RPY)
    # Display yaw alpha maps to world opening-axis angle -alpha - 90 deg.
    # The T detector yaw is the object +X / handle-width axis.  For a handle
    # grasp the gripper opening axis should cross the handle, so set alpha to
    # make the gripper +X axis parallel to object +X.
    rpy[2] = _wrap_deg(float(PICK_TOPDOWN_RPY[2] - yaw_deg - 90.0))
    return rpy


def _wrap_deg(angle):
    return float((float(angle) + 180.0) % 360.0 - 180.0)


def _angle_abs_diff_deg(a, b):
    return abs(_wrap_deg(float(a) - float(b)))


def _classify_t_orientation(yaw_deg):
    yaw_deg = float(yaw_deg)
    standard_err = _angle_abs_diff_deg(yaw_deg, STANDARD_WAIT_FOR_GRASP_YAW_DEG)
    upside_yaw = _wrap_deg(STANDARD_WAIT_FOR_GRASP_YAW_DEG + 180.0)
    upside_err = _angle_abs_diff_deg(yaw_deg, upside_yaw)
    is_upside_down = (
        upside_err <= float(UPSIDE_DOWN_YAW_THRESHOLD_DEG)
        and upside_err < standard_err
    )
    return {
        "orientation": "upside_down" if is_upside_down else "standard",
        "yaw_deg": yaw_deg,
        "standard_yaw_deg": float(STANDARD_WAIT_FOR_GRASP_YAW_DEG),
        "upside_down_yaw_deg": float(upside_yaw),
        "standard_error_deg": float(standard_err),
        "upside_down_error_deg": float(upside_err),
        "threshold_deg": float(UPSIDE_DOWN_YAW_THRESHOLD_DEG),
    }


def _display_rpy_rotation(rpy):
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    euler_xyz = [-pitch, roll, -yaw - 90.0]
    return Rotation.from_euler("xyz", euler_xyz, degrees=True)


def _topdown_fingertip_offset(rpy):
    return PICK_GRASP_OFFSET_M * _display_rpy_rotation(rpy).apply([0.0, -1.0, 0.0])


def _object_xy_offset_world(yaw_deg, offset_xy):
    R = _rotation_2d(np.deg2rad(float(yaw_deg)))
    return R @ np.asarray(offset_xy, dtype=float).reshape(2)


def _arm_ee_xy_from_state(state, side):
    arms = state.get("arms", state) if isinstance(state, dict) else getattr(state, "arms", {})
    arm = arms.get(side) if isinstance(arms, dict) else getattr(arms, side, None)
    if arm is None:
        return None
    ee_pos = arm.get("ee_pos") if isinstance(arm, dict) else getattr(arm, "ee_pos", None)
    if ee_pos is None:
        return None
    return np.asarray(ee_pos, dtype=float).reshape(-1)[:2]


def _arm_gripper_pos_from_state(state, side):
    arms = state.get("arms", state) if isinstance(state, dict) else getattr(state, "arms", {})
    arm = arms.get(side) if isinstance(arms, dict) else getattr(arms, side, None)
    if arm is None:
        return None
    value = arm.get("gripper_pos") if isinstance(arm, dict) else getattr(arm, "gripper_pos", None)
    if value is None:
        return None
    return float(value)


def _open_gripper_if_needed(side, threshold=0.95):
    try:
        gripper_pos = _arm_gripper_pos_from_state(get_robot_state(), side)
    except Exception as exc:
        print(f"  reset_t open_gripper: could not read gripper pos, opening anyway: {exc}")
        gripper_pos = None
    if gripper_pos is not None and float(gripper_pos) >= float(threshold):
        print(f"  reset_t open_gripper: skip; {side} already open at {gripper_pos:.3f}")
        return {
            "success": True,
            "side": side,
            "gripper": float(gripper_pos),
            "skipped": True,
            "reason": "already_open",
        }
    fast_open = globals().get("open_gripper_fast")
    if callable(fast_open):
        return fast_open(side)
    return open_gripper(side, vel_limit=PUSHT_OPEN_GRIPPER_VEL_LIMIT)


def _set_gripper_fast_if_available(side, pos, timeout=0.75):
    fast_set = globals().get("set_gripper_fast")
    if callable(fast_set):
        return fast_set(side, pos, timeout=timeout)
    return set_gripper(side, pos)


def _go_home_fast_if_available():
    fast_home = globals().get("go_home_fast")
    if callable(fast_home):
        return fast_home()
    return go_home()


def _choose_pick_side(pose, requested_side):
    side = str(requested_side or "auto").strip().lower()
    if side in {"left", "right"}:
        return side
    if side not in {"auto", "closest"}:
        raise ValueError(f"Unsupported reset side {requested_side!r}; use left, right, or auto")

    obj_xy = np.asarray(pose["xy"], dtype=float).reshape(2)
    try:
        state = get_robot_state()
        distances = {}
        for candidate in ("left", "right"):
            ee_xy = _arm_ee_xy_from_state(state, candidate)
            if ee_xy is not None:
                distances[candidate] = float(np.linalg.norm(ee_xy - obj_xy))
        if distances:
            chosen = min(distances, key=distances.get)
            print(
                "  reset_t: auto side="
                f"{chosen} distances={{{', '.join(f'{k}: {v:.3f}' for k, v in distances.items())}}}"
            )
            return chosen
    except Exception as exc:
        print(f"  reset_t: auto side could not read robot state: {exc}")

    chosen = "left" if float(obj_xy[1]) >= 0.0 else "right"
    print(f"  reset_t: auto side fallback={chosen} from object y={float(obj_xy[1]):+.3f}")
    return chosen


def _choose_holding_side(requested_side):
    side = str(requested_side or "auto").strip().lower()
    if side in {"left", "right"}:
        return side
    if side not in {"auto", "closest"}:
        raise ValueError(f"Unsupported holding side {requested_side!r}; use left, right, or auto")

    try:
        state = get_robot_state()
        grip = {}
        for candidate in ("left", "right"):
            value = _arm_gripper_pos_from_state(state, candidate)
            if value is not None:
                grip[candidate] = float(value)
        if grip:
            chosen = min(grip, key=grip.get)
            print(
                "  reset_t: holding side="
                f"{chosen} grippers={{{', '.join(f'{k}: {v:.3f}' for k, v in grip.items())}}}"
            )
            return chosen
    except Exception as exc:
        print(f"  reset_t: holding side could not read robot state: {exc}")

    print("  reset_t: holding side fallback=left")
    return "left"


def _pick_place_plan(pose, target_xy, target_yaw_deg, side):
    current = np.asarray(pose["world_xyz"], dtype=float).reshape(3)
    target_xy = np.asarray(target_xy, dtype=float).reshape(2)
    grasp_yaw = float(pose["yaw_deg"]) if ALIGN_GRIPPER_TO_T_YAW else 0.0
    place_yaw = float(target_yaw_deg) if ALIGN_GRIPPER_TO_T_YAW else 0.0
    grasp_rpy = _topdown_rpy_for_yaw(grasp_yaw)
    place_rpy = _topdown_rpy_for_yaw(place_yaw)
    grasp_offset = _topdown_fingertip_offset(grasp_rpy)
    place_offset = _topdown_fingertip_offset(place_rpy)

    grasp_xy = current[:2] + _object_xy_offset_world(
        float(pose["yaw_deg"]), PICK_GRASP_OBJECT_OFFSET_XY_M
    )
    grasp_center = current.copy()
    grasp_center[:2] = grasp_xy
    grasp_center[2] = float(current[2] + PICK_GRASP_Z_OFFSET_M)
    grasp_attempts = []
    for idx, z_offset in enumerate(PICK_GRASP_Z_OFFSETS_M):
        attempt = grasp_center.copy()
        attempt[2] = float(current[2] + float(z_offset))
        grasp_attempts.append(
            {
                "index": int(idx),
                "z_offset_m": float(z_offset),
                "pos": (attempt + grasp_offset).tolist(),
                "rpy": [float(v) for v in grasp_rpy],
                "gripper": PICK_GRIPPER_OPEN,
            }
        )
    place_center = np.array(
        [float(target_xy[0]), float(target_xy[1]), float(current[2] + PICK_PLACE_Z_OFFSET_M)],
        dtype=float,
    )

    pick_hover = grasp_center.copy()
    pick_hover[2] = max(float(PICK_HOVER_Z_M), float(grasp_center[2] + PICK_PLACE_CLEARANCE_M))
    lift = grasp_center.copy()
    lift[2] = max(float(PICK_LIFT_Z_M), float(grasp_center[2] + PICK_PLACE_CLEARANCE_M))
    place_hover = place_center.copy()
    place_hover[2] = max(float(PICK_LIFT_Z_M), float(place_center[2] + PICK_PLACE_CLEARANCE_M))
    camera_clear = np.array(
        [
            float(PICK_CAMERA_CLEAR_X_M),
            float(PICK_CAMERA_CLEAR_Y_M if side == "left" else -PICK_CAMERA_CLEAR_Y_M),
            float(PICK_CAMERA_CLEAR_Z_M),
        ],
        dtype=float,
    )

    return {
        "side": side,
        "current_xyz": current.tolist(),
        "target_xy": target_xy.tolist(),
        "target_yaw_deg": float(target_yaw_deg),
        "align_gripper_to_t_yaw": bool(ALIGN_GRIPPER_TO_T_YAW),
        "grasp_object_offset_xy_m": [float(v) for v in PICK_GRASP_OBJECT_OFFSET_XY_M],
        "grasp_center": grasp_center.tolist(),
        "grasp_attempts": grasp_attempts,
        "place_center": place_center.tolist(),
        "grasp_rpy": [float(v) for v in grasp_rpy],
        "place_rpy": [float(v) for v in place_rpy],
        "grasp_offset": [float(v) for v in grasp_offset],
        "place_offset": [float(v) for v in place_offset],
        "camera_clear": camera_clear.tolist(),
        "waypoints": [
            {
                "name": "pick_hover",
                "pos": (pick_hover + grasp_offset).tolist(),
                "rpy": [float(v) for v in grasp_rpy],
                "gripper": PICK_GRIPPER_OPEN,
                "motion": True,
            },
            {
                "name": "pick_descend",
                "pos": grasp_attempts[0]["pos"],
                "rpy": [float(v) for v in grasp_rpy],
                "gripper": PICK_GRIPPER_OPEN,
                "motion": True,
            },
            {
                "name": "close_gripper",
                "gripper": PICK_GRIPPER_CLOSED,
                "motion": False,
            },
            {
                "name": "lift",
                "pos": (lift + grasp_offset).tolist(),
                "rpy": [float(v) for v in grasp_rpy],
                "gripper": PICK_GRIPPER_CLOSED,
                "motion": True,
            },
            {
                "name": "place_hover",
                "pos": (place_hover + place_offset).tolist(),
                "rpy": [float(v) for v in place_rpy],
                "gripper": PICK_GRIPPER_CLOSED,
                "motion": True,
            },
            {
                "name": "place_descend",
                "pos": (place_center + place_offset).tolist(),
                "rpy": [float(v) for v in place_rpy],
                "gripper": PICK_GRIPPER_CLOSED,
                "motion": True,
            },
            {
                "name": "open_gripper",
                "gripper": PICK_GRIPPER_OPEN,
                "motion": False,
            },
            {
                "name": "retract",
                "pos": (place_hover + place_offset).tolist(),
                "rpy": [float(v) for v in place_rpy],
                "gripper": PICK_GRIPPER_OPEN,
                "motion": True,
            },
            {
                "name": "camera_clear",
                "pos": camera_clear.tolist(),
                "rpy": [float(v) for v in place_rpy],
                "gripper": PICK_GRIPPER_OPEN,
                "motion": True,
            },
        ],
    }


def _run_pick_place_plan(plan, *, preview_only, camera=DEFAULT_CAMERA, pose=None):
    side = plan["side"]
    results = []
    grasp_attempt_idx = 0
    for waypoint in plan["waypoints"]:
        name = waypoint["name"]
        if waypoint.get("motion", False):
            kwargs = _move_kwargs(
                side,
                waypoint["pos"],
                waypoint["rpy"],
                preview_only=preview_only,
            )
            kwargs.update(_gripper_width_arg(side, waypoint["gripper"]))
            print(
                f"  reset_t {name}: preview={bool(preview_only)} "
                f"xyz={[round(float(v), 4) for v in waypoint['pos']]} "
                f"rpy={[round(float(v), 1) for v in waypoint['rpy']]} "
                f"gripper={float(waypoint['gripper']):.2f}"
            )
            if not preview_only and name in {"pick_hover", "pick_descend"}:
                checkpoint = _log_motion_checkpoint(
                    f"before_{name}",
                    camera=camera,
                    side=side,
                    plan=plan,
                    pose=pose,
                    waypoint=waypoint,
                    attempt=grasp_attempt_idx,
                )
                if checkpoint is not None:
                    results.append({"name": f"checkpoint_before_{name}", **checkpoint})
            try:
                result = freespace_move(**kwargs)
            except Exception as exc:
                results.append(
                    {
                        "name": name,
                        "type": "motion_preview" if preview_only else "motion",
                        "status": "exception",
                        "error": str(exc),
                    }
                )
                return False, results
            entry = {
                "name": name,
                "type": "motion_preview" if preview_only else "motion",
                "status": getattr(result, "status", None),
                "result": str(result),
            }
            results.append(entry)
            if not preview_only and name in {"pick_hover", "pick_descend", "lift", "camera_clear"}:
                checkpoint = _log_motion_checkpoint(
                    f"after_{name}",
                    camera=camera,
                    side=side,
                    plan=plan,
                    pose=pose,
                    waypoint=waypoint,
                    attempt=grasp_attempt_idx,
                )
                if checkpoint is not None:
                    results.append({"name": f"checkpoint_after_{name}", **checkpoint})
            if not _result_ok(result):
                return False, results
        elif preview_only:
            results.append(
                {
                    "name": name,
                    "type": "gripper_preview",
                    "target": float(waypoint["gripper"]),
                    "status": "preview",
                }
            )
        else:
            if float(waypoint["gripper"]) >= 0.5:
                result = _open_gripper_if_needed(side)
            else:
                attempts = list(plan.get("grasp_attempts") or [])
                while True:
                    active_grasp = (
                        dict(attempts[grasp_attempt_idx])
                        if grasp_attempt_idx < len(attempts)
                        else {
                            "pos": plan.get("grasp_center"),
                            "rpy": plan.get("grasp_rpy"),
                            "gripper": PICK_GRIPPER_OPEN,
                        }
                    )
                    servo_entry, active_grasp = _visual_servo_grasp_alignment(
                        plan,
                        camera=camera,
                        side=side,
                        pose=pose,
                        waypoint=active_grasp,
                        attempt=grasp_attempt_idx,
                    )
                    if servo_entry is not None:
                        results.append(servo_entry)
                        if servo_entry.get("status") == "move_failed":
                            return False, results
                    result = close_gripper(side)
                    try:
                        gripper_pos = _arm_gripper_pos_from_state(get_robot_state(), side)
                    except Exception as exc:
                        gripper_pos = None
                        print(f"  reset_t close_gripper: could not read gripper pos: {exc}")
                    if gripper_pos is not None:
                        print(
                            "  reset_t close_gripper: "
                            f"{side} attempt={grasp_attempt_idx} "
                            f"gripper_pos={gripper_pos:.4f}"
                        )
                    if gripper_pos is None or gripper_pos > PICK_EMPTY_GRIPPER_POS_MAX:
                        checkpoint = _log_motion_checkpoint(
                            f"after_close_attempt_{grasp_attempt_idx}",
                            camera=camera,
                            side=side,
                            plan=plan,
                            pose=pose,
                            waypoint={
                                "name": "closed_grasp",
                                "pos": active_grasp.get("pos"),
                                "rpy": active_grasp.get("rpy"),
                            },
                            attempt=grasp_attempt_idx,
                        )
                        if checkpoint is not None:
                            results.append(
                                {
                                    "name": f"checkpoint_after_close_attempt_{grasp_attempt_idx}",
                                    **checkpoint,
                                }
                            )
                        break
                    checkpoint = _log_motion_checkpoint(
                        f"empty_close_attempt_{grasp_attempt_idx}",
                        camera=camera,
                        side=side,
                        plan=plan,
                        pose=pose,
                        waypoint={
                            "name": "empty_grasp",
                            "pos": active_grasp.get("pos"),
                            "rpy": active_grasp.get("rpy"),
                        },
                        attempt=grasp_attempt_idx,
                    )
                    if checkpoint is not None:
                        results.append(
                            {
                                "name": f"checkpoint_empty_close_attempt_{grasp_attempt_idx}",
                                **checkpoint,
                            }
                        )
                    results.append(
                        {
                            "name": name,
                            "type": "gripper",
                            "target": float(waypoint["gripper"]),
                            "result": str(result),
                            "gripper_pos": float(gripper_pos),
                            "status": "empty_grasp",
                            "attempt": int(grasp_attempt_idx),
                        }
                    )
                    grasp_attempt_idx += 1
                    if grasp_attempt_idx >= len(attempts):
                        return False, results
                    retry = attempts[grasp_attempt_idx]
                    open_result = _open_gripper_if_needed(side)
                    results.append(
                        {
                            "name": "open_for_grasp_retry",
                            "type": "gripper",
                            "target": PICK_GRIPPER_OPEN,
                            "result": str(open_result),
                            "attempt": int(grasp_attempt_idx),
                        }
                    )
                    kwargs = _move_kwargs(
                        side,
                        retry["pos"],
                        retry["rpy"],
                        preview_only=preview_only,
                    )
                    kwargs.update(_gripper_width_arg(side, retry["gripper"]))
                    print(
                        "  reset_t pick_descend_retry: "
                        f"attempt={int(retry['index'])} "
                        f"z_offset={float(retry['z_offset_m']):+.3f} "
                        f"xyz={[round(float(v), 4) for v in retry['pos']]} "
                        f"rpy={[round(float(v), 1) for v in retry['rpy']]}"
                    )
                    checkpoint = _log_motion_checkpoint(
                        f"before_pick_descend_retry_{grasp_attempt_idx}",
                        camera=camera,
                        side=side,
                        plan=plan,
                        pose=pose,
                        waypoint={
                            "name": "pick_descend_retry",
                            "pos": retry["pos"],
                            "rpy": retry["rpy"],
                        },
                        attempt=grasp_attempt_idx,
                    )
                    if checkpoint is not None:
                        results.append(
                            {
                                "name": f"checkpoint_before_pick_descend_retry_{grasp_attempt_idx}",
                                **checkpoint,
                            }
                        )
                    try:
                        move_result = freespace_move(**kwargs)
                    except Exception as exc:
                        results.append(
                            {
                                "name": "pick_descend_retry",
                                "type": "motion",
                                "status": "exception",
                                "error": str(exc),
                                "attempt": int(grasp_attempt_idx),
                            }
                        )
                        return False, results
                    results.append(
                        {
                            "name": "pick_descend_retry",
                            "type": "motion",
                            "status": getattr(move_result, "status", None),
                            "result": str(move_result),
                            "attempt": int(grasp_attempt_idx),
                        }
                    )
                    checkpoint = _log_motion_checkpoint(
                        f"after_pick_descend_retry_{grasp_attempt_idx}",
                        camera=camera,
                        side=side,
                        plan=plan,
                        pose=pose,
                        waypoint={
                            "name": "pick_descend_retry",
                            "pos": retry["pos"],
                            "rpy": retry["rpy"],
                        },
                        attempt=grasp_attempt_idx,
                    )
                    if checkpoint is not None:
                        results.append(
                            {
                                "name": f"checkpoint_after_pick_descend_retry_{grasp_attempt_idx}",
                                **checkpoint,
                            }
                        )
                    if not _result_ok(move_result):
                        return False, results
            results.append(
                {
                    "name": name,
                    "type": "gripper",
                    "target": float(waypoint["gripper"]),
                    "result": str(result),
                }
            )
    return True, results


def _push_plan(pose, target_xy, push_side, push_rpy):
    current_xy = np.asarray(pose["xy"], dtype=float)
    target_xy = np.asarray(target_xy, dtype=float).reshape(2)
    delta = target_xy - current_xy
    dist = float(np.linalg.norm(delta))
    if dist < 1e-9:
        direction = np.array([1.0, 0.0], dtype=float)
    else:
        direction = delta / dist
    if dist > DEFAULT_MAX_PUSH_M:
        target_xy = current_xy + direction * DEFAULT_MAX_PUSH_M
        delta = target_xy - current_xy
        dist = float(np.linalg.norm(delta))

    z = float(pose["z"]) + DEFAULT_PUSH_CONTACT_Z_OFFSET_M
    pre_xy = current_xy - direction * DEFAULT_PRE_PUSH_STANDOFF_M
    contact_xy = current_xy - direction * 0.5 * DEFAULT_PRE_PUSH_STANDOFF_M
    finish_xy = target_xy + direction * DEFAULT_POST_PUSH_OVERSHOOT_M
    retreat_xy = finish_xy - direction * 0.02

    return {
        "current_xy": current_xy.tolist(),
        "target_xy": target_xy.tolist(),
        "delta_xy": delta.tolist(),
        "distance_m": dist,
        "direction_xy": direction.tolist(),
        "poses": [
            {
                "name": "pre_push",
                "pos": [float(pre_xy[0]), float(pre_xy[1]), z + DEFAULT_PUSH_CLEARANCE_M],
                "rpy": list(push_rpy),
            },
            {
                "name": "contact",
                "pos": [float(contact_xy[0]), float(contact_xy[1]), z],
                "rpy": list(push_rpy),
            },
            {
                "name": "finish",
                "pos": [float(finish_xy[0]), float(finish_xy[1]), z],
                "rpy": list(push_rpy),
            },
            {
                "name": "retreat",
                "pos": [float(retreat_xy[0]), float(retreat_xy[1]), z + DEFAULT_PUSH_CLEARANCE_M],
                "rpy": list(push_rpy),
            },
        ],
        "side": push_side,
    }


@skill
def reset_ok_v1(
    camera=DEFAULT_CAMERA,
    threshold=RESET_OK_THRESHOLD,
    reference_image=str(RESET_OK_REFERENCE_IMAGE),
    reference_meta=str(RESET_OK_REFERENCE_META),
    margin=90,
):
    """Score whether the red T is in the saved standard initial/reset pose.

    This is the reset-position detector, separate from the PushT goal reward.
    It uses the saved standard-initial reference and succeeds when
    ``score > threshold``.
    """
    import cv2

    rgb = _as_rgb_uint8(get_camera_image(camera))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    reference_crop, reference_mask, reference_info, crop = _reset_ok_load_reference(
        reference_image,
        reference_meta,
        margin,
        target_shape=bgr.shape,
    )
    current_mask_full, _ = _reset_ok_dominant_component(_red_mask(bgr))
    current_crop = _reset_ok_crop(bgr, crop)
    current_mask, current_info = _reset_ok_dominant_component(
        _reset_ok_crop(current_mask_full, crop)
    )
    score = _reset_ok_mask_score(reference_mask, current_mask, reference_info, current_info)
    ok = bool(float(score["score"]) > float(threshold))

    overlay = _reset_ok_overlay(current_crop, reference_mask, current_mask, score, crop)
    # Stack reference on the left for human debugging, matching the standalone
    # viewer layout. log_image expects RGB.
    reference_view = _reset_ok_overlay(
        reference_crop,
        reference_mask,
        np.zeros_like(reference_mask),
        {**score, "score": 0.0},
        crop,
    )
    panel_bgr = np.hstack([reference_view, overlay])
    overlay_path = _log_debug_image(
        cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB),
        tag=f"pusht_reset_ok_{camera}",
        label="reset_ok overlay",
    )
    print(
        "  reset_ok: "
        f"score={float(score['score']):.3f} threshold={float(threshold):.3f} "
        f"ok={ok} crop={crop} overlay={overlay_path}"
    )
    log = {
        "success": ok,
        "reset_ok": ok,
        "score": float(score["score"]),
        "threshold": float(threshold),
        "camera": camera,
        "reference_image": str(reference_image),
        "reference_meta": str(reference_meta),
        "crop_xywh": [int(v) for v in crop],
        "overlay_path": overlay_path,
        "score_terms": score,
        "reference_info": reference_info,
        "current_info": current_info,
    }
    return ok, log


@skill
def detect_red_t_pose_v1(camera=DEFAULT_CAMERA, plane_path=str(DEFAULT_PLANE_PATH)):
    """Detect the red PushT T pose from the top camera.

    Returns world xyz, planar xy, yaw, fit RMSE, and the desk/top height labels
    selected by the red-T silhouette detector.
    """
    pose = _detect_red_t_pose(
        camera=camera,
        plane_path=plane_path,
        allow_centroid_fallback=True,
    )
    success = bool(pose["success"]) and pose["rmse_mm"] <= MIN_SCORE_RMSE_MM
    pose["success"] = success
    return success, pose


@skill
def hover_over_t_v1(
    side=DEFAULT_PUSH_SIDE,
    camera=DEFAULT_CAMERA,
    plane_path=str(DEFAULT_PLANE_PATH),
    hover_z=PICK_HOVER_Z_M,
    target="center",
    dry_run=False,
):
    """Locate the red T and move only to a top-down hover over it.

    ``target="center"`` puts the gripper frame above the detected T center.
    ``target="grasp"`` uses the same planned grasp XY as the full reset, but
    still stops at hover height.
    """
    pose = _detect_red_t_pose(camera=camera, plane_path=plane_path)
    side = _choose_pick_side(pose, side)
    plan = _pick_place_plan(
        pose,
        target_xy=np.asarray(pose["xy"], dtype=float),
        target_yaw_deg=float(pose["yaw_deg"]),
        side=side,
    )
    hover_target = str(target or "center").strip().lower()
    rpy = plan["grasp_rpy"]
    if hover_target == "grasp":
        marker_pos = plan["grasp_center"]
        hover_pos, target_uv = _world_point_on_camera_ray_at_z(
            camera,
            marker_pos,
            float(hover_z),
        )
    elif hover_target == "center":
        marker_pos = pose["world_xyz"]
        hover_pos, target_uv = _world_point_on_camera_ray_at_z(
            camera,
            marker_pos,
            float(hover_z),
        )
    else:
        raise ValueError("target must be 'center' or 'grasp'")

    waypoint = {
        "name": f"hover_{hover_target}",
        "pos": [float(v) for v in hover_pos],
        "rpy": [float(v) for v in rpy],
        "gripper": PICK_GRIPPER_OPEN,
    }
    before_checkpoint = _log_motion_checkpoint(
        f"before_hover_{hover_target}",
        camera=camera,
        side=side,
        plan={**plan, "grasp_center": [float(v) for v in marker_pos]},
        pose=pose,
        waypoint=waypoint,
        attempt=0,
    )
    kwargs = _move_kwargs(side, waypoint["pos"], waypoint["rpy"], preview_only=bool(dry_run))
    kwargs.update(_gripper_width_arg(side, PICK_GRIPPER_OPEN))
    result = freespace_move(**kwargs)
    after_checkpoint = None
    if not dry_run:
        after_checkpoint = _log_motion_checkpoint(
            f"after_hover_{hover_target}",
            camera=camera,
            side=side,
            plan={**plan, "grasp_center": [float(v) for v in marker_pos]},
            pose=pose,
            waypoint=waypoint,
            attempt=0,
        )
    ok = _result_ok(result)
    return ok, {
        "success": bool(ok),
        "side": side,
        "target": hover_target,
        "pose": pose,
        "hover_waypoint": waypoint,
        "target_pixel_uv": [float(v) for v in target_uv],
        "hover_mode": "camera_ray_at_hover_z",
        "motion_status": getattr(result, "status", None),
        "motion_result": str(result),
        "before_checkpoint": before_checkpoint,
        "after_checkpoint": after_checkpoint,
    }


@skill
def move_to_t_grasp_pose_v1(
    side=DEFAULT_PUSH_SIDE,
    camera=DEFAULT_CAMERA,
    plane_path=str(DEFAULT_PLANE_PATH),
    hover_z=PICK_HOVER_Z_M,
    wait_s=15.0,
    close_after_wait=False,
    close_pos=0.75,
    grasp_pos=None,
    grasp_rpy=None,
    skip_hover=False,
    dry_run=False,
    grasp_yaw_offset_deg=0.0,
    detected_pose=None,
):
    """Locate the red T, hover over the handle grasp, descend, then hold there.

    This is the test step between hover-only and full reset: it does not lift or
    place the T.  Set ``close_after_wait=True`` only when the grasp alignment has
    been visually checked.
    """
    explicit_grasp = grasp_pos is not None or grasp_rpy is not None
    if explicit_grasp:
        if grasp_pos is None or grasp_rpy is None:
            raise ValueError("grasp_pos and grasp_rpy must be provided together")
        descend_pos = np.asarray(grasp_pos, dtype=float).reshape(3)
        descend_rpy = [float(v) for v in np.asarray(grasp_rpy, dtype=float).reshape(3)]
        side = _auto_side_for_xy(descend_pos[:2], side)
        pose = {
            "success": False,
            "camera": camera,
            "world_xyz": descend_pos.tolist(),
            "xy": [float(descend_pos[0]), float(descend_pos[1])],
            "z": float(descend_pos[2] - PICK_GRASP_Z_OFFSETS_M[0]),
            "yaw_deg": float(-descend_rpy[2] - 90.0),
            "note": "explicit_grasp_pose_no_redetect",
        }
        plan = {
            "side": side,
            "grasp_center": descend_pos.tolist(),
            "grasp_rpy": descend_rpy,
            "grasp_attempts": [
                {
                    "index": 0,
                    "z_offset_m": float(PICK_GRASP_Z_OFFSETS_M[0]),
                    "pos": descend_pos.tolist(),
                    "rpy": descend_rpy,
                    "gripper": PICK_GRIPPER_OPEN,
                }
            ],
        }
        hover_waypoint = None
        target_uv = [float("nan"), float("nan")]
    else:
        pose = (
            detected_pose
            if detected_pose is not None
            else _detect_red_t_pose(camera=camera, plane_path=plane_path)
        )
        side = _choose_pick_side(pose, side)
        plan = _pick_place_plan(
            pose,
            target_xy=np.asarray(pose["xy"], dtype=float),
            target_yaw_deg=float(pose["yaw_deg"]),
            side=side,
        )
        grasp_yaw_offset_deg = float(grasp_yaw_offset_deg)
        if abs(grasp_yaw_offset_deg) > 1e-6:
            grasp_rpy = _topdown_rpy_for_yaw(float(pose["yaw_deg"]) + grasp_yaw_offset_deg)
            plan["grasp_rpy"] = [float(v) for v in grasp_rpy]
            plan["grasp_yaw_offset_deg"] = float(grasp_yaw_offset_deg)
            for attempt in plan.get("grasp_attempts", []):
                attempt["rpy"] = [float(v) for v in grasp_rpy]

        marker_pos = plan["grasp_center"]
        hover_pos, target_uv = _world_point_on_camera_ray_at_z(
            camera,
            marker_pos,
            float(hover_z),
        )
        hover_waypoint = {
            "name": "hover_grasp",
            "pos": [float(v) for v in hover_pos],
            "rpy": [float(v) for v in plan["grasp_rpy"]],
            "gripper": PICK_GRIPPER_OPEN,
        }
    descend_waypoint = {
        "name": "grasp_descend",
        "pos": [float(v) for v in plan["grasp_attempts"][0]["pos"]],
        "rpy": [float(v) for v in plan["grasp_attempts"][0]["rpy"]],
        "gripper": PICK_GRIPPER_OPEN,
    }

    results = []
    open_result = None
    if not dry_run:
        open_result = _open_gripper_if_needed(side)
        results.append({"name": "open_gripper", "result": str(open_result)})

    stages = []
    if hover_waypoint is not None and not bool(skip_hover):
        stages.append(("hover_grasp", hover_waypoint))
    stages.append(("grasp_descend", descend_waypoint))
    for stage, waypoint in stages:
        before_checkpoint = _log_motion_checkpoint(
            f"before_{stage}",
            camera=camera,
            side=side,
            plan=plan,
            pose=pose,
            waypoint=waypoint,
            attempt=0,
        )
        kwargs = _move_kwargs(side, waypoint["pos"], waypoint["rpy"], preview_only=bool(dry_run))
        kwargs.update(_gripper_width_arg(side, waypoint["gripper"]))
        print(
            f"  reset_t {stage}: preview={bool(dry_run)} "
            f"xyz={[round(float(v), 4) for v in waypoint['pos']]} "
            f"rpy={[round(float(v), 1) for v in waypoint['rpy']]} "
            f"gripper={float(waypoint['gripper']):.2f}"
        )
        result = freespace_move(**kwargs)
        after_checkpoint = None
        if not dry_run:
            after_checkpoint = _log_motion_checkpoint(
                f"after_{stage}",
                camera=camera,
                side=side,
                plan=plan,
                pose=pose,
                waypoint=waypoint,
                attempt=0,
            )
        results.append(
            {
                "name": stage,
                "status": getattr(result, "status", None),
                "result": str(result),
                "before_checkpoint": before_checkpoint,
                "after_checkpoint": after_checkpoint,
            }
        )
        if not _result_ok(result):
            return False, {
                "success": False,
                "side": side,
                "pose": pose,
                "plan": plan,
                "target_pixel_uv": [float(v) for v in target_uv],
                "results": results,
            }

    wait_s = max(0.0, float(wait_s))
    if wait_s > 0.0 and not dry_run:
        print(
            "  reset_t grasp_descend: holding at grasp pose before close "
            f"for {wait_s:.1f}s"
        )
        time.sleep(wait_s)

    close_result = None
    close_gripper_pos = None
    if bool(close_after_wait) and not dry_run:
        close_pos = float(np.clip(float(close_pos), 0.0, 1.0))
        close_result = _set_gripper_fast_if_available(side, close_pos)
        try:
            close_gripper_pos = _arm_gripper_pos_from_state(get_robot_state(), side)
        except Exception as exc:
            print(f"  reset_t partial_close_after_wait: could not read gripper pos: {exc}")
        close_checkpoint = _log_motion_checkpoint(
            "after_manual_wait_partial_close",
            camera=camera,
            side=side,
            plan=plan,
            pose=pose,
            waypoint={
                "name": "partial_closed_grasp",
                "pos": descend_waypoint["pos"],
                "rpy": descend_waypoint["rpy"],
            },
            attempt=0,
        )
        results.append(
            {
                "name": "partial_close_gripper",
                "target_pos": close_pos,
                "result": str(close_result),
                "gripper_pos": None if close_gripper_pos is None else float(close_gripper_pos),
                "checkpoint": close_checkpoint,
            }
        )

    return True, {
        "success": True,
        "side": side,
        "pose": pose,
        "plan": plan,
        "hover_waypoint": hover_waypoint,
        "descend_waypoint": descend_waypoint,
        "target_pixel_uv": [float(v) for v in target_uv],
        "wait_s": wait_s,
        "close_after_wait": bool(close_after_wait),
        "close_pos": float(close_pos),
        "open_result": None if open_result is None else str(open_result),
        "close_result": None if close_result is None else str(close_result),
        "close_gripper_pos": close_gripper_pos,
        "results": results,
    }


@skill
def normalize_upside_down_t_v1(
    side=DEFAULT_PUSH_SIDE,
    camera=DEFAULT_CAMERA,
    plane_path=str(DEFAULT_PLANE_PATH),
    hover_z=PICK_HOVER_Z_M,
    wait_s=1.0,
    close_pos=RESET_T_HOLD_GRIPPER_POS,
    dry_run=False,
):
    """If the T is upside down, rotate it 180 degrees and put it back down."""
    pose = _detect_red_t_pose(camera=camera, plane_path=plane_path)
    orientation = _classify_t_orientation(float(pose["yaw_deg"]))
    print(
        "  reset_t orientation: "
        f"{orientation['orientation']} yaw={orientation['yaw_deg']:+.1f} "
        f"standard_err={orientation['standard_error_deg']:.1f} "
        f"upside_err={orientation['upside_down_error_deg']:.1f}"
    )
    if orientation["orientation"] != "upside_down":
        return True, {
            "success": True,
            "normalized": False,
            "reason": "already standard",
            "pose": pose,
            "orientation": orientation,
        }

    grasp_ok, grasp_log = move_to_t_grasp_pose_v1(
        side=side,
        camera=camera,
        plane_path=plane_path,
        hover_z=hover_z,
        wait_s=wait_s,
        close_after_wait=True,
        close_pos=close_pos,
        dry_run=dry_run,
        grasp_yaw_offset_deg=180.0,
    )
    if not grasp_ok:
        return False, {
            "success": False,
            "normalized": False,
            "reason": "reverse grasp failed",
            "pose": pose,
            "orientation": orientation,
            "grasp_log": grasp_log,
        }

    active_pose = grasp_log.get("pose", pose)
    side = grasp_log.get("side", side)
    normalized_yaw = _wrap_deg(float(active_pose["yaw_deg"]) + 180.0)
    place_plan = _pick_place_plan(
        active_pose,
        target_xy=np.asarray(active_pose["xy"], dtype=float),
        target_yaw_deg=normalized_yaw,
        side=side,
    )
    grasp_rpy = [float(v) for v in grasp_log["descend_waypoint"]["rpy"]]
    rotate_rpy = list(grasp_rpy)
    rotate_rpy[2] = _wrap_deg(float(rotate_rpy[2]) + 180.0)
    place_plan["normalize_reverse_grasp_rpy"] = [float(v) for v in grasp_rpy]
    place_plan["normalize_rotated_release_rpy"] = [float(v) for v in rotate_rpy]
    place_plan["normalize_rotation_delta_deg"] = 180.0
    current_pos = None
    if not dry_run:
        current_pos = _arm_ee_pos_from_state(get_robot_state(), side)
    if current_pos is None:
        current_pos = np.asarray(grasp_log["descend_waypoint"]["pos"], dtype=float).reshape(3)
    else:
        current_pos = np.asarray(current_pos, dtype=float).reshape(3)
    lift_z = max(float(PICK_LIFT_Z_M), float(current_pos[2] + PICK_PLACE_CLEARANCE_M))
    lift_pos = current_pos.copy()
    lift_pos[2] = lift_z
    release_pos = np.asarray(
        place_plan["waypoints"][5]["pos"],
        dtype=float,
    ).reshape(3)
    release_pos[2] = float(release_pos[2] + NORMALIZE_RELEASE_Z_EXTRA_M)
    place_plan["normalize_release_z_extra_m"] = float(NORMALIZE_RELEASE_Z_EXTRA_M)
    release_hover = release_pos.copy()
    release_hover[2] = lift_z

    waypoints = [
        {
            "name": "normalize_lift_rotate_180",
            "pos": [float(v) for v in lift_pos],
            "rpy": rotate_rpy,
            "gripper": float(close_pos),
        },
        {
            "name": "normalize_release_hover",
            "pos": [float(v) for v in release_hover],
            "rpy": rotate_rpy,
            "gripper": float(close_pos),
        },
        {
            "name": "normalize_release_descend",
            "pos": [float(v) for v in release_pos],
            "rpy": rotate_rpy,
            "gripper": float(close_pos),
        },
    ]
    results = []
    for waypoint in waypoints:
        before_checkpoint = _log_motion_checkpoint(
            f"before_{waypoint['name']}",
            camera=camera,
            side=side,
            plan=place_plan,
            pose=active_pose,
            waypoint=waypoint,
            attempt=0,
        )
        kwargs = _move_kwargs(side, waypoint["pos"], waypoint["rpy"], preview_only=bool(dry_run))
        kwargs.update(_gripper_width_arg(side, waypoint["gripper"]))
        print(
            f"  reset_t {waypoint['name']}: preview={bool(dry_run)} "
            f"xyz={[round(float(v), 4) for v in waypoint['pos']]} "
            f"rpy={[round(float(v), 1) for v in waypoint['rpy']]} "
            f"gripper={float(waypoint['gripper']):.2f}"
        )
        result = freespace_move(**kwargs)
        after_checkpoint = None
        if not dry_run:
            after_checkpoint = _log_motion_checkpoint(
                f"after_{waypoint['name']}",
                camera=camera,
                side=side,
                plan=place_plan,
                pose=active_pose,
                waypoint=waypoint,
                attempt=0,
            )
        results.append(
            {
                "name": waypoint["name"],
                "status": getattr(result, "status", None),
                "result": str(result),
                "before_checkpoint": before_checkpoint,
                "after_checkpoint": after_checkpoint,
            }
        )
        if not _result_ok(result):
            return False, {
                "success": False,
                "normalized": False,
                "reason": f"{waypoint['name']} failed",
                "pose": active_pose,
                "orientation": orientation,
                "place_plan": place_plan,
                "grasp_log": grasp_log,
                "results": results,
            }

    if not dry_run:
        open_result = _open_gripper_if_needed(side)
        results.append({"name": "normalize_open_gripper", "result": str(open_result)})
    else:
        results.append({"name": "normalize_open_gripper", "status": "preview"})

    clear_waypoint = {
        "name": "normalize_camera_clear",
        "pos": [float(v) for v in place_plan["camera_clear"]],
        "rpy": rotate_rpy,
        "gripper": PICK_GRIPPER_OPEN,
    }
    kwargs = _move_kwargs(
        side,
        clear_waypoint["pos"],
        clear_waypoint["rpy"],
        preview_only=bool(dry_run),
    )
    kwargs.update(_gripper_width_arg(side, PICK_GRIPPER_OPEN))
    print(
        f"  reset_t {clear_waypoint['name']}: preview={bool(dry_run)} "
        f"xyz={[round(float(v), 4) for v in clear_waypoint['pos']]} "
        f"rpy={[round(float(v), 1) for v in clear_waypoint['rpy']]} "
        f"gripper={float(clear_waypoint['gripper']):.2f}"
    )
    clear_result = freespace_move(**kwargs)
    clear_checkpoint = None
    if not dry_run:
        clear_checkpoint = _log_motion_checkpoint(
            "after_normalize_camera_clear",
            camera=camera,
            side=side,
            plan=place_plan,
            pose=active_pose,
            waypoint=clear_waypoint,
            attempt=0,
        )
    results.append(
        {
            "name": "normalize_camera_clear",
            "status": getattr(clear_result, "status", None),
            "result": str(clear_result),
            "after_checkpoint": clear_checkpoint,
        }
    )
    if not _result_ok(clear_result):
        return False, {
            "success": False,
            "normalized": True,
            "reason": "camera-clear failed after normalization release",
            "pose": active_pose,
            "orientation": orientation,
            "place_plan": place_plan,
            "grasp_log": grasp_log,
            "results": results,
        }

    return True, {
        "success": True,
        "normalized": True,
        "pose": active_pose,
        "orientation": orientation,
        "normalized_yaw_deg": float(normalized_yaw),
        "place_plan": place_plan,
        "grasp_log": grasp_log,
        "results": results,
    }


@skill
def place_grasped_t_for_left_handoff_v1(
    side="right",
    camera=DEFAULT_CAMERA,
    lift_z=PICK_LIFT_Z_M,
    release_pos=LEFT_HANDOFF_RELEASE_EE_POS,
    release_rpy=LEFT_HANDOFF_RELEASE_EE_RPY,
    hold_gripper_pos=None,
    dry_run=False,
):
    """Put a right-arm-held T on the table where the left arm can re-grasp it."""
    result_ok, log = place_grasped_t_at_reset_v1(
        side=side,
        camera=camera,
        lift_z=lift_z,
        release_pos=release_pos,
        release_rpy=release_rpy,
        hold_gripper_pos=hold_gripper_pos,
        dry_run=dry_run,
    )
    log = dict(log)
    log["handoff"] = True
    log["handoff_target_side"] = "left"
    log["handoff_release_ee_pos"] = [float(v) for v in release_pos]
    log["handoff_release_ee_rpy"] = [float(v) for v in release_rpy]
    return result_ok, log


@skill
def place_grasped_t_at_reset_v1(
    side="auto",
    camera=DEFAULT_CAMERA,
    lift_z=PICK_LIFT_Z_M,
    release_pos=RESET_T_RELEASE_EE_POS,
    release_rpy=RESET_T_RELEASE_EE_RPY,
    hold_gripper_pos=None,
    dry_run=False,
):
    """Place an already-grasped T at the recorded reset pose, then go home."""
    side = _choose_holding_side(side)
    state = None
    current_pos = None
    current_gripper = None
    if not dry_run:
        state = get_robot_state()
        current_pos = _arm_ee_pos_from_state(state, side)
        current_gripper = _arm_gripper_pos_from_state(state, side)
    if current_pos is None:
        current_pos = np.asarray(release_pos, dtype=float).reshape(3)
    else:
        current_pos = np.asarray(current_pos, dtype=float).reshape(3)
    if hold_gripper_pos is None:
        hold_gripper_pos = (
            RESET_T_HOLD_GRIPPER_POS if current_gripper is None else float(current_gripper)
        )
    hold_gripper_pos = float(np.clip(float(hold_gripper_pos), 0.0, 1.0))

    release_pos_arr = np.asarray(release_pos, dtype=float).reshape(3)
    release_rpy = [float(v) for v in np.asarray(release_rpy, dtype=float).reshape(3)]
    lift_z = max(float(lift_z), float(current_pos[2] + PICK_PLACE_CLEARANCE_M))
    place_hover = release_pos_arr.copy()
    place_hover[2] = lift_z

    pose = {
        "success": True,
        "camera": camera,
        "world_xyz": [DEFAULT_TARGET_XY[0], DEFAULT_TARGET_XY[1], 0.75],
        "xy": [float(v) for v in DEFAULT_TARGET_XY],
        "z": 0.75,
        "yaw_deg": float(DEFAULT_TARGET_YAW_DEG),
        "note": "recorded_reset_pose_from_logs/grasp_t_20260517T190353",
    }
    plan = {
        "side": side,
        "target_xy": [float(v) for v in DEFAULT_TARGET_XY],
        "target_yaw_deg": float(DEFAULT_TARGET_YAW_DEG),
        "grasp_center": [float(v) for v in release_pos_arr],
        "place_center": [float(v) for v in release_pos_arr],
        "grasp_rpy": [float(v) for v in release_rpy],
        "place_rpy": [float(v) for v in release_rpy],
        "recorded_release_ee_pos": [float(v) for v in release_pos_arr],
        "recorded_release_ee_rpy": [float(v) for v in release_rpy],
        "hold_gripper_pos": float(hold_gripper_pos),
    }
    waypoints = [
        {
            "name": "reset_hover",
            "pos": [float(v) for v in place_hover],
            "rpy": release_rpy,
            "gripper": hold_gripper_pos,
        },
        {
            "name": "reset_descend",
            "pos": [float(v) for v in release_pos_arr],
            "rpy": release_rpy,
            "gripper": hold_gripper_pos,
        },
    ]

    results = []
    for waypoint in waypoints:
        before_checkpoint = _log_motion_checkpoint(
            f"before_{waypoint['name']}",
            camera=camera,
            side=side,
            plan=plan,
            pose=pose,
            waypoint=waypoint,
            attempt=0,
        )
        kwargs = _move_kwargs(side, waypoint["pos"], waypoint["rpy"], preview_only=bool(dry_run))
        kwargs.update(_gripper_width_arg(side, waypoint["gripper"]))
        print(
            f"  reset_t {waypoint['name']}: preview={bool(dry_run)} "
            f"xyz={[round(float(v), 4) for v in waypoint['pos']]} "
            f"rpy={[round(float(v), 1) for v in waypoint['rpy']]} "
            f"gripper={float(waypoint['gripper']):.2f}"
        )
        result = freespace_move(**kwargs)
        after_checkpoint = None
        if not dry_run:
            after_checkpoint = _log_motion_checkpoint(
                f"after_{waypoint['name']}",
                camera=camera,
                side=side,
                plan=plan,
                pose=pose,
                waypoint=waypoint,
                attempt=0,
            )
        results.append(
            {
                "name": waypoint["name"],
                "status": getattr(result, "status", None),
                "result": str(result),
                "before_checkpoint": before_checkpoint,
                "after_checkpoint": after_checkpoint,
            }
        )
        if not _result_ok(result):
            return False, {
                "success": False,
                "side": side,
                "plan": plan,
                "waypoints": waypoints,
                "results": results,
            }

    open_result = None
    post_release_lift_result = None
    home_result = None
    post_release_lift_pos = release_pos_arr.copy()
    post_release_lift_pos[2] = float(
        release_pos_arr[2] + POST_RELEASE_LIFT_CLEARANCE_M
    )
    post_release_lift = {
        "name": "post_release_lift",
        "pos": [float(v) for v in post_release_lift_pos],
        "rpy": release_rpy,
        "gripper": 1.0,
    }
    if not dry_run:
        open_result = _open_gripper_if_needed(side)
        results.append({"name": "open_gripper", "result": str(open_result)})
        before_checkpoint = _log_motion_checkpoint(
            "before_post_release_lift",
            camera=camera,
            side=side,
            plan=plan,
            pose=pose,
            waypoint=post_release_lift,
            attempt=0,
        )
        kwargs = _move_kwargs(
            side,
            post_release_lift["pos"],
            post_release_lift["rpy"],
            preview_only=False,
        )
        kwargs.update(_gripper_width_arg(side, post_release_lift["gripper"]))
        print(
            "  reset_t post_release_lift: preview=False "
            f"xyz={[round(float(v), 4) for v in post_release_lift['pos']]} "
            f"rpy={[round(float(v), 1) for v in post_release_lift['rpy']]} "
            f"gripper={float(post_release_lift['gripper']):.2f}"
        )
        post_release_lift_result = freespace_move(**kwargs)
        after_checkpoint = _log_motion_checkpoint(
            "after_post_release_lift",
            camera=camera,
            side=side,
            plan=plan,
            pose=pose,
            waypoint=post_release_lift,
            attempt=0,
        )
        results.append(
            {
                "name": "post_release_lift",
                "status": getattr(post_release_lift_result, "status", None),
                "result": str(post_release_lift_result),
                "before_checkpoint": before_checkpoint,
                "after_checkpoint": after_checkpoint,
            }
        )
        if not _result_ok(post_release_lift_result):
            return False, {
                "success": False,
                "side": side,
                "plan": plan,
                "waypoints": waypoints + [post_release_lift],
                "results": results,
            }
        home_result = _go_home_fast_if_available()
        results.append({"name": "go_home", "result": str(home_result)})
    else:
        results.append({"name": "open_gripper", "status": "preview"})
        kwargs = _move_kwargs(
            side,
            post_release_lift["pos"],
            post_release_lift["rpy"],
            preview_only=True,
        )
        kwargs.update(_gripper_width_arg(side, post_release_lift["gripper"]))
        post_release_lift_result = freespace_move(**kwargs)
        results.append(
            {
                "name": "post_release_lift",
                "status": getattr(post_release_lift_result, "status", "preview"),
                "result": str(post_release_lift_result),
            }
        )
        results.append({"name": "go_home", "status": "preview"})

    return True, {
        "success": True,
        "side": side,
        "plan": plan,
        "waypoints": waypoints + [post_release_lift],
        "open_result": None if open_result is None else str(open_result),
        "post_release_lift_result": (
            None if post_release_lift_result is None else str(post_release_lift_result)
        ),
        "home_result": None if home_result is None else str(home_result),
        "results": results,
    }


@skill
def reset_t_v1(
    target_xy=DEFAULT_TARGET_XY,
    target_yaw_deg=DEFAULT_TARGET_YAW_DEG,
    side=DEFAULT_PUSH_SIDE,
    camera=DEFAULT_CAMERA,
    plane_path=str(DEFAULT_PLANE_PATH),
    dry_run=False,
):
    """Reset the red PushT T by locating, picking it up, and placing it.

    Dry-run mode still calls freespace_move with preview_only=True for every
    motion waypoint, so it validates reachability without closing the gripper.
    """
    pose = _detect_red_t_pose(camera=camera, plane_path=plane_path)
    side = _choose_pick_side(pose, side)
    plan = _pick_place_plan(pose, target_xy, target_yaw_deg, side)
    initial_checkpoint = _log_motion_checkpoint(
        "after_locate_plan",
        camera=camera,
        side=side,
        plan=plan,
        pose=pose,
        waypoint={
            "name": "first_grasp",
            "pos": plan["grasp_attempts"][0]["pos"],
            "rpy": plan["grasp_attempts"][0]["rpy"],
        },
        attempt=0,
    )
    if pose["rmse_mm"] > MIN_SCORE_RMSE_MM:
        return False, {
            "success": False,
            "reason": "red T detection RMSE too high",
            "pose": pose,
            "plan": plan,
            "initial_checkpoint": initial_checkpoint,
        }

    current_xy = np.asarray(pose["xy"], dtype=float).reshape(2)
    target_xy_arr = np.asarray(target_xy, dtype=float).reshape(2)
    initial_err = float(np.linalg.norm(current_xy - target_xy_arr))
    if initial_err <= DEFAULT_POS_TOL_M and abs(float(pose["yaw_deg"]) - float(target_yaw_deg)) <= 5.0:
        return True, {
            "success": True,
            "reason": "already within tolerance",
            "pose": pose,
            "plan": plan,
            "initial_checkpoint": initial_checkpoint,
            "tolerance_m": DEFAULT_POS_TOL_M,
        }

    ok, steps = _run_pick_place_plan(
        plan,
        preview_only=bool(dry_run),
        camera=camera,
        pose=pose,
    )
    if dry_run:
        return ok, {
            "success": bool(ok),
            "dry_run": True,
            "reason": None if ok else "preview motion planning failed",
            "pose": pose,
            "plan": plan,
            "steps": steps,
            "initial_checkpoint": initial_checkpoint,
            "tolerance_m": DEFAULT_POS_TOL_M,
        }

    if not ok:
        recovery_steps = []
        try:
            open_result = _open_gripper_if_needed(side)
            recovery_steps.append(
                {
                    "name": "failure_open_gripper",
                    "type": "gripper",
                    "result": str(open_result),
                }
            )
        except Exception as exc:
            recovery_steps.append(
                {
                    "name": "failure_open_gripper",
                    "type": "gripper",
                    "error": str(exc),
                }
            )
        try:
            camera_clear = plan.get("camera_clear")
            if camera_clear is not None:
                clear_kwargs = _move_kwargs(
                    side,
                    camera_clear,
                    plan["place_rpy"],
                    preview_only=False,
                )
                clear_kwargs.update(_gripper_width_arg(side, PICK_GRIPPER_OPEN))
                clear_result = freespace_move(**clear_kwargs)
                recovery_steps.append(
                    {
                        "name": "failure_camera_clear",
                        "type": "motion",
                        "status": getattr(clear_result, "status", None),
                        "result": str(clear_result),
                    }
                )
        except Exception as exc:
            recovery_steps.append(
                {
                    "name": "failure_camera_clear",
                    "type": "motion",
                    "error": str(exc),
                }
            )
        try:
            final_pose = _detect_red_t_pose(camera=camera, plane_path=plane_path)
        except Exception as exc:
            final_pose = {"success": False, "reason": str(exc)}
        return False, {
            "success": False,
            "reason": "pick/place execution failed",
            "pose": pose,
            "final_pose": final_pose,
            "plan": plan,
            "steps": steps,
            "initial_checkpoint": initial_checkpoint,
            "recovery_steps": recovery_steps,
        }

    try:
        final_pose = _detect_red_t_pose(camera=camera, plane_path=plane_path)
    except Exception as exc:
        return False, {
            "success": False,
            "reason": f"final red T detection failed: {exc}",
            "pose": pose,
            "target_xy": [float(v) for v in target_xy_arr],
            "tolerance_m": DEFAULT_POS_TOL_M,
            "plan": plan,
            "steps": steps,
            "initial_checkpoint": initial_checkpoint,
        }
    final_err = float(
        np.linalg.norm(np.asarray(final_pose["xy"], dtype=float) - target_xy_arr)
    )
    success = final_err <= DEFAULT_POS_TOL_M
    return success, {
        "success": success,
        "pose": pose,
        "final_pose": final_pose,
        "target_xy": [float(v) for v in target_xy_arr],
        "final_error_m": final_err,
        "tolerance_m": DEFAULT_POS_TOL_M,
        "plan": plan,
        "steps": steps,
        "initial_checkpoint": initial_checkpoint,
    }

