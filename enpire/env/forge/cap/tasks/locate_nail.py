# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline locate-nail task harness.

This module intentionally contains only stable I/O and reward plumbing.  The
agent-generated script is expected to implement the actual image annotation
function and then call these helpers to load fixed wrist-camera examples, save
artifacts into the run's ``vis/`` directory, and ask Gemini to critique the
annotations as a reward signal.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

INPUT_IMAGE_PATHS: list[Path] = []  # populate with local test images to run standalone

DEFAULT_REWARD_BACKEND = "nvidia"
DEFAULT_REWARD_MODEL = "gcp/google/gemini-3.1-pro-preview"


def load_locate_nail_images() -> list[tuple[str, np.ndarray]]:
    """Return ``[(label, rgb), ...]`` for the fixed wrist-camera examples."""
    pairs: list[tuple[str, np.ndarray]] = []
    for idx, path in enumerate(INPUT_IMAGE_PATHS, start=1):
        if not path.exists():
            raise FileNotFoundError(f"locate_nail input image missing: {path}")
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        pairs.append((f"wrist_example_{idx}", rgb))
    return pairs


def _clip_bbox_xyxy(
    bbox_xyxy: list[int],
    width: int,
    height: int,
) -> list[int]:
    """Clip an inclusive ``[x1, y1, x2, y2]`` bbox to image bounds."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _point_int(point: tuple[float, float] | list[float] | np.ndarray) -> list[int]:
    return [int(round(float(point[0]))), int(round(float(point[1])))]


def _normalize_min_area_rect_angle(rect: tuple[Any, Any, float]) -> float:
    """Return the long-axis angle in degrees, roughly in ``[-90, 90]``."""
    (_center, size, angle) = rect
    width, height = float(size[0]), float(size[1])
    long_axis_angle = float(angle)
    if width < height:
        long_axis_angle += 90.0
    while long_axis_angle > 90.0:
        long_axis_angle -= 180.0
    while long_axis_angle <= -90.0:
        long_axis_angle += 180.0
    return long_axis_angle


def _nail_axis_points_from_rect(rect: tuple[Any, Any, float]) -> dict[str, Any]:
    """Return upper/lower nail long-axis endpoints and the lower-3/4 target."""
    center = np.asarray(rect[0], dtype=float)
    size = rect[1]
    width, height = float(size[0]), float(size[1])
    long_length = max(width, height)
    angle_deg = _normalize_min_area_rect_angle(rect)
    angle_rad = np.deg2rad(angle_deg)
    axis = np.asarray([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        axis = np.asarray([0.0, 1.0], dtype=float)
    else:
        axis /= norm

    endpoint_a = center - 0.5 * long_length * axis
    endpoint_b = center + 0.5 * long_length * axis
    if endpoint_a[1] <= endpoint_b[1]:
        upper = endpoint_a
        lower = endpoint_b
    else:
        upper = endpoint_b
        lower = endpoint_a

    lower_3_4 = upper + 0.75 * (lower - upper)
    return {
        "axis_unit_image": [float(axis[0]), float(axis[1])],
        "long_length_px": float(long_length),
        "upper_endpoint_px": _point_int(upper),
        "lower_endpoint_px": _point_int(lower),
        "lower_3_4_point_px": _point_int(lower_3_4),
        "upper_endpoint_float_px": [float(upper[0]), float(upper[1])],
        "lower_endpoint_float_px": [float(lower[0]), float(lower[1])],
        "lower_3_4_point_float_px": [float(lower_3_4[0]), float(lower_3_4[1])],
    }


def _find_nail_component(rgb: np.ndarray) -> dict[str, Any]:
    """Find the small bright metal nail/pin near the center of a wrist image.

    This is intentionally deterministic and uses only local image geometry.  In
    the captured wrist frames, the nail is the most compact, high-contrast,
    vertically elongated edge component near the center; the gripper pads are
    also elongated but farther from the center, so they lose on the center
    prior.
    """
    import cv2

    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 125)

    roi = np.zeros((height, width), dtype=bool)
    roi[int(0.30 * height) : int(0.80 * height), int(0.25 * width) : int(0.75 * width)] = True
    edge_roi = (edges > 0) & roi
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        edge_roi.astype(np.uint8),
        connectivity=8,
    )

    best: tuple[float, int] | None = None
    candidates: list[dict[str, Any]] = []
    for component_id in range(1, num_labels):
        x, y, w_box, h_box, area = [int(v) for v in stats[component_id]]
        if area < 18 or area > 800:
            continue
        if w_box < 3 or h_box < 10 or w_box > 60 or h_box > 110:
            continue
        aspect = h_box / max(float(w_box), 1.0)
        if aspect < 1.0:
            continue

        cx, cy = [float(v) for v in centroids[component_id]]
        center_dist = ((cx - width * 0.525) / (width * 0.18)) ** 2 + (
            (cy - height * 0.61) / (height * 0.20)
        ) ** 2
        score = area * 0.8 + h_box * 5.0 + min(aspect, 8.0) * 18.0 - center_dist * 80.0
        candidate = {
            "component_id": int(component_id),
            "score": float(score),
            "edge_area_px": int(area),
            "bbox_xywh": [x, y, w_box, h_box],
            "centroid_px": [cx, cy],
            "aspect": float(aspect),
            "center_dist": float(center_dist),
        }
        candidates.append(candidate)
        if best is None or score > best[0]:
            best = (score, component_id)

    if best is None:
        # Conservative fallback for these wrist frames: a small centered box.
        cx = int(round(width * 0.53))
        cy = int(round(height * 0.62))
        bbox = _clip_bbox_xyxy([cx - 12, cy - 35, cx + 12, cy + 35], width, height)
        upper = [cx, cy - 35]
        lower = [cx, cy + 35]
        lower_3_4 = _point_int([cx, cy - 35 + 0.75 * 70])
        return {
            "bbox_xyxy": bbox,
            "midpoint_float_px": [float(cx), float(cy)],
            "midpoint_px": [cx, cy],
            "axis_unit_image": [0.0, 1.0],
            "long_axis_length_px": 70.0,
            "upper_endpoint_px": upper,
            "lower_endpoint_px": lower,
            "lower_3_4_point_px": lower_3_4,
            "upper_endpoint_float_px": [float(upper[0]), float(upper[1])],
            "lower_endpoint_float_px": [float(lower[0]), float(lower[1])],
            "lower_3_4_point_float_px": [float(lower_3_4[0]), float(lower_3_4[1])],
            "angle_deg": 90.0,
            "edge_area_px": 0,
            "mask": np.zeros((height, width), dtype=bool),
            "candidates": candidates,
            "method": "fallback_center_box",
        }

    selected_id = best[1]
    x, y, w_box, h_box, area = [int(v) for v in stats[selected_id]]
    component_mask = labels == selected_id
    ys, xs = np.nonzero(component_mask)
    points = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(points)
    rect_center = [float(rect[0][0]), float(rect[0][1])]
    angle_deg = _normalize_min_area_rect_angle(rect)
    axis_points = _nail_axis_points_from_rect(rect)

    # Inclusive XYXY box with a single-pixel margin to keep the anti-aliased
    # nail edge visible without swallowing background/table pixels.
    bbox = _clip_bbox_xyxy([x - 1, y - 1, x + w_box, y + h_box], width, height)
    return {
        "bbox_xyxy": bbox,
        "midpoint_float_px": rect_center,
        "midpoint_px": _point_int(rect_center),
        "axis_unit_image": axis_points["axis_unit_image"],
        "long_axis_length_px": axis_points["long_length_px"],
        "upper_endpoint_px": axis_points["upper_endpoint_px"],
        "lower_endpoint_px": axis_points["lower_endpoint_px"],
        "lower_3_4_point_px": axis_points["lower_3_4_point_px"],
        "upper_endpoint_float_px": axis_points["upper_endpoint_float_px"],
        "lower_endpoint_float_px": axis_points["lower_endpoint_float_px"],
        "lower_3_4_point_float_px": axis_points["lower_3_4_point_float_px"],
        "angle_deg": float(angle_deg),
        "edge_area_px": int(area),
        "mask": component_mask,
        "candidates": sorted(candidates, key=lambda item: item["score"], reverse=True)[:5],
        "method": "central_edge_component",
    }


def _find_gripper_tip(
    rgb: np.ndarray,
    nail_midpoint_px: list[int],
    *,
    side: str,
) -> dict[str, Any]:
    """Find the visible inner black fingertip/pad point for one gripper jaw."""
    import cv2

    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    yy, xx = np.indices((height, width))

    # The pads are the large, dark, textured components in the lower side
    # regions.  Closing absorbs small highlights in the rubber texture while
    # preserving the outer pad silhouette.
    dark = (gray < 88) & (yy > int(0.48 * height))
    if side == "left":
        region = xx < int(0.42 * width)
    else:
        region = xx > int(0.58 * width)
    dark &= region
    dark = (
        cv2.morphologyEx(
            dark.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        > 0
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark.astype(np.uint8),
        connectivity=8,
    )
    candidates: list[dict[str, Any]] = []
    for component_id in range(1, num_labels):
        x, y, w_box, h_box, area = [int(v) for v in stats[component_id]]
        if area < 500 or w_box < 8 or h_box < 35:
            continue
        inner_x = x + w_box - 1 if side == "left" else x
        # Prefer the innermost large/tall dark pad; this rejects screw heads
        # and the outer gripper handle/holes.
        inward_score = inner_x if side == "left" else width - inner_x
        score = float(inward_score + min(area, 2500) * 0.005 + h_box * 0.2)
        candidates.append(
            {
                "component_id": int(component_id),
                "score": score,
                "area_px": int(area),
                "bbox_xywh": [x, y, w_box, h_box],
                "inner_x": int(inner_x),
                "centroid_px": [float(centroids[component_id][0]), float(centroids[component_id][1])],
            }
        )

    if not candidates:
        # Geometric fallback to an approximate inner-pad location.
        fallback_x = int(round(width * (0.29 if side == "left" else 0.74)))
        fallback_y = int(round(height * 0.64))
        return {
            "tip_px": [fallback_x, fallback_y],
            "component_bbox_xyxy": _clip_bbox_xyxy(
                [fallback_x - 10, fallback_y - 20, fallback_x + 10, fallback_y + 20],
                width,
                height,
            ),
            "method": f"{side}_fallback",
            "candidates": [],
        }

    selected = max(candidates, key=lambda item: item["score"])
    component_mask = labels == selected["component_id"]
    ys, xs = np.nonzero(component_mask)
    points = np.column_stack([xs, ys])
    nail = np.asarray(nail_midpoint_px, dtype=np.float32)
    # The contact point should be the pixel on the chosen black pad closest to
    # the nail center: it lands on the pad's inner face instead of on screws or
    # the outer handle.
    distances = (points[:, 0] - nail[0]) ** 2 + (points[:, 1] - nail[1]) ** 2
    tip = _point_int(points[int(np.argmin(distances))])
    x, y, w_box, h_box = selected["bbox_xywh"]
    selected["component_bbox_xyxy"] = _clip_bbox_xyxy(
        [x, y, x + w_box - 1, y + h_box - 1],
        width,
        height,
    )
    return {
        "tip_px": tip,
        "component_bbox_xyxy": selected["component_bbox_xyxy"],
        "method": f"{side}_dark_pad_closest_to_nail",
        "selected": selected,
        "candidates": sorted(candidates, key=lambda item: item["score"], reverse=True)[:5],
    }


def decide_gripper_centering_nudge(
    record: dict[str, Any],
    *,
    target_point_px: list[float] | tuple[float, float] | np.ndarray | None = None,
    target_key: str = "pin_lower_3_4_point_px",
    px_to_m: float | None = None,
    gain: float = 1.0,
    deadband_px: float = 2.0,
    max_nudge_px: float | None = 50.0,
    max_nudge_m: float | None = None,
    command_sign: float = 1.0,
) -> dict[str, Any]:
    """Compute a gripper-frame XY nudge that puts the grasp target on the tip line.

    The local image/gripper frame is defined from the detected tips:

    * ``+X`` points from the left gripper tip to the right gripper tip.
    * ``+Y`` is perpendicular to ``+X`` and chosen to face down in the image.

    By default the alignment target is ``record["pin_lower_3_4_point_px"]``:
    the point 75% of the way from the upper/free end of the visible pin toward
    the lower/base end.  This intentionally grasps low on the pin and leaves the
    upper part exposed for handover.  If that key is absent, the function falls
    back to ``pin_midpoint_px``.

    The returned error is ``target_point - gripper_line_midpoint`` projected
    into the gripper frame.  With the default ``command_sign=+1``, the
    recommended nudge moves the gripper-line midpoint toward the target point in
    this image/gripper frame.  If a later wrist-camera calibration shows that a
    physical robot nudge produces the opposite image motion, call this with
    ``command_sign=-1`` at the robot-control boundary.

    ``px_to_m`` is optional because the vision-only study task has no metric
    calibration yet.  When provided, the same nudge is also returned in meters.
    """
    if target_point_px is None:
        target_point_px = record.get(target_key) or record["pin_midpoint_px"]
    target = np.asarray(target_point_px, dtype=float)
    left = np.asarray(record["left_gripper_tip_px"], dtype=float)
    right = np.asarray(record["right_gripper_tip_px"], dtype=float)

    line = right - left
    line_len = float(np.linalg.norm(line))
    if line_len <= 1e-6:
        raise ValueError(f"gripper tip line is degenerate: left={left}, right={right}")

    x_axis = line / line_len
    y_axis = np.asarray([-x_axis[1], x_axis[0]], dtype=float)
    if y_axis[1] < 0:
        y_axis *= -1.0

    line_mid = 0.5 * (left + right)
    error_image = target - line_mid
    error_xy = np.asarray(
        [float(np.dot(error_image, x_axis)), float(np.dot(error_image, y_axis))],
        dtype=float,
    )

    aligned = bool(abs(error_xy[0]) <= deadband_px and abs(error_xy[1]) <= deadband_px)
    nudge_px = np.zeros(2, dtype=float) if aligned else float(command_sign) * float(gain) * error_xy
    unclipped_nudge_px = nudge_px.copy()

    if max_nudge_px is not None:
        max_px = float(max_nudge_px)
        norm_px = float(np.linalg.norm(nudge_px))
        if norm_px > max_px > 0.0:
            nudge_px *= max_px / norm_px

    nudge_m: list[float] | None = None
    if px_to_m is not None:
        nudge_m_arr = nudge_px * float(px_to_m)
        if max_nudge_m is not None:
            max_m = float(max_nudge_m)
            norm_m = float(np.linalg.norm(nudge_m_arr))
            if norm_m > max_m > 0.0:
                nudge_m_arr *= max_m / norm_m
                if abs(float(px_to_m)) > 1e-12:
                    nudge_px = nudge_m_arr / float(px_to_m)
        nudge_m = [float(nudge_m_arr[0]), float(nudge_m_arr[1])]

    return {
        "aligned": aligned,
        "deadband_px": float(deadband_px),
        "gain": float(gain),
        "command_sign": float(command_sign),
        "px_to_m": None if px_to_m is None else float(px_to_m),
        "max_nudge_px": None if max_nudge_px is None else float(max_nudge_px),
        "max_nudge_m": None if max_nudge_m is None else float(max_nudge_m),
        "target_key": str(target_key),
        "target_point_px": [float(target[0]), float(target[1])],
        "gripper_line_midpoint_px": [float(line_mid[0]), float(line_mid[1])],
        "gripper_line_length_px": line_len,
        "gripper_x_axis_image_unit": [float(x_axis[0]), float(x_axis[1])],
        "gripper_y_axis_down_image_unit": [float(y_axis[0]), float(y_axis[1])],
        "target_error_image_px": [float(error_image[0]), float(error_image[1])],
        "target_error_gripper_xy_px": [float(error_xy[0]), float(error_xy[1])],
        "recommended_nudge_gripper_xy_px": [float(nudge_px[0]), float(nudge_px[1])],
        "unclipped_recommended_nudge_gripper_xy_px": [
            float(unclipped_nudge_px[0]),
            float(unclipped_nudge_px[1]),
        ],
        "recommended_nudge_gripper_xy_m": nudge_m,
        "command_convention": (
            "+X is left_tip->right_tip; +Y is perpendicular and down-facing in "
            "the image; default nudge moves the gripper-line midpoint toward "
            "the selected grasp target in this frame."
        ),
    }


def locate_nail_annotation(label: str, rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Annotate one wrist-camera frame with nail and gripper geometry.

    Parameters
    ----------
    label:
        Stable image label, included in the returned JSON record.
    rgb:
        ``H×W×3`` uint8 RGB image.

    Returns
    -------
    annotated_rgb, record:
        ``annotated_rgb`` is a copy of the input with a nail bbox, nail
        midpoint, and gripper-tip line.  ``record`` is JSON-serializable and
        contains the coordinates needed by downstream pre-grasp alignment code.
    """
    import cv2

    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"rgb must have shape HxWx3, got {arr.shape}")
    annotated = arr.copy()

    nail = _find_nail_component(arr)
    bbox = [int(v) for v in nail["bbox_xyxy"]]
    midpoint = [int(v) for v in nail["midpoint_px"]]
    lower_3_4_point = [int(v) for v in nail["lower_3_4_point_px"]]
    left_tip = _find_gripper_tip(arr, midpoint, side="left")
    right_tip = _find_gripper_tip(arr, midpoint, side="right")
    left_point = [int(v) for v in left_tip["tip_px"]]
    right_point = [int(v) for v in right_tip["tip_px"]]

    # Draw in RGB color tuples.  OpenCV writes the tuple values verbatim, so no
    # BGR conversion is needed while the array remains RGB.
    x1, y1, x2, y2 = bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(
        annotated,
        tuple([int(v) for v in nail["upper_endpoint_px"]]),
        tuple([int(v) for v in nail["lower_endpoint_px"]]),
        (0, 255, 255),
        1,
    )
    cv2.circle(annotated, tuple(midpoint), 4, (255, 0, 0), -1)
    cv2.circle(annotated, tuple(lower_3_4_point), 6, (0, 180, 255), -1)
    cv2.line(annotated, tuple(left_point), tuple(right_point), (255, 220, 0), 2)
    cv2.circle(annotated, tuple(left_point), 5, (255, 0, 255), -1)
    cv2.circle(annotated, tuple(right_point), 5, (255, 0, 255), -1)
    line_midpoint = _point_int(
        [(left_point[0] + right_point[0]) / 2.0, (left_point[1] + right_point[1]) / 2.0]
    )
    cv2.circle(annotated, tuple(line_midpoint), 4, (0, 180, 255), -1)
    cv2.arrowedLine(annotated, tuple(line_midpoint), tuple(lower_3_4_point), (0, 180, 255), 2, tipLength=0.35)

    # Keep labels short so they do not obscure the small nail.
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, "nail", (x1, max(15, y1 - 6)), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(
        annotated,
        "pin mid",
        (midpoint[0] + 7, midpoint[1] - 7),
        font,
        0.42,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "lower 3/4 target",
        (lower_3_4_point[0] + 7, lower_3_4_point[1] + 14),
        font,
        0.42,
        (0, 180, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "gripper tips",
        (min(left_point[0], right_point[0]) + 8, max(15, min(left_point[1], right_point[1]) - 10)),
        font,
        0.45,
        (255, 220, 0),
        1,
        cv2.LINE_AA,
    )

    dx = float(right_point[0] - left_point[0])
    dy = float(right_point[1] - left_point[1])
    record: dict[str, Any] = {
        "label": str(label),
        "bbox_xyxy": bbox,
        "pin_midpoint_px": midpoint,
        "pin_upper_endpoint_px": [int(v) for v in nail["upper_endpoint_px"]],
        "pin_lower_endpoint_px": [int(v) for v in nail["lower_endpoint_px"]],
        "pin_lower_3_4_point_px": lower_3_4_point,
        "grasp_target_px": lower_3_4_point,
        "alignment_target": "pin_lower_3_4_point_px",
        "left_gripper_tip_px": left_point,
        "right_gripper_tip_px": right_point,
        "gripper_line_midpoint_px": line_midpoint,
        "gripper_line_angle_deg": float(np.degrees(np.arctan2(dy, dx))) if dx or dy else 0.0,
        "nail_angle_deg": float(nail["angle_deg"]),
        "nail_edge_area_px": int(nail["edge_area_px"]),
        "nail_method": str(nail["method"]),
        "left_gripper_method": str(left_tip["method"]),
        "right_gripper_method": str(right_tip["method"]),
        "debug": {
            "nail_midpoint_float_px": [float(v) for v in nail["midpoint_float_px"]],
            "pin_upper_endpoint_float_px": [float(v) for v in nail["upper_endpoint_float_px"]],
            "pin_lower_endpoint_float_px": [float(v) for v in nail["lower_endpoint_float_px"]],
            "pin_lower_3_4_point_float_px": [float(v) for v in nail["lower_3_4_point_float_px"]],
            "pin_axis_unit_image": [float(v) for v in nail["axis_unit_image"]],
            "pin_long_axis_length_px": float(nail["long_axis_length_px"]),
            "nail_candidates": nail["candidates"],
            "left_gripper_component_bbox_xyxy": left_tip["component_bbox_xyxy"],
            "right_gripper_component_bbox_xyxy": right_tip["component_bbox_xyxy"],
        },
    }
    centering = decide_gripper_centering_nudge(record)
    record["target_error_image_px"] = centering["target_error_image_px"]
    record["target_error_gripper_xy_px"] = centering["target_error_gripper_xy_px"]
    record["recommended_nudge_gripper_xy_px"] = centering["recommended_nudge_gripper_xy_px"]
    record["nudge_to_align_grasp_target"] = centering
    return annotated.astype(np.uint8, copy=False), record


def run_locate_nail_baseline() -> dict[str, Any]:
    """Run the built-in deterministic baseline and save annotations."""
    pairs = load_locate_nail_images()
    annotated: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for label, rgb in pairs:
        ann, rec = locate_nail_annotation(label, rgb)
        annotated[label] = ann
        records.append(rec)
        print(json.dumps(rec, indent=2))
    saved_paths = save_locate_nail_outputs(annotated, records)
    return {"records": records, "saved_paths": saved_paths}


def locate_nail_output_dir() -> Path:
    """Return the current run_script output directory when available."""
    for arg in sys.argv:
        if arg.startswith("script_output_dir="):
            value = arg.split("=", 1)[1].strip()
            if value:
                return Path(value)
    return Path("logs/locate_nail_manual")


def save_locate_nail_outputs(
    annotated_images: dict[str, np.ndarray],
    records: list[dict[str, Any]],
) -> dict[str, str]:
    """Save annotated images and coordinate JSON into the execution directory."""
    out_dir = locate_nail_output_dir()
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}
    for label, image in annotated_images.items():
        arr = np.asarray(image, dtype=np.uint8)
        path = vis_dir / f"locate_nail_{label}_annotated.png"
        Image.fromarray(arr).save(path)
        saved[label] = str(path)
        try:
            from enpire.env.forge.cap.agent.tools._artifact_log import log_image

            log_image(arr, tag="locate_nail_annotated", label=label)
        except Exception:
            pass

    result_path = out_dir / "locate_nail_records.json"
    result_path.write_text(
        json.dumps({"records": records, "annotated_images": saved}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    saved["records_json"] = str(result_path)
    return saved


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S)
    if fenced:
        return json.loads(fenced.group(1))

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError(f"Gemini response did not contain JSON: {text[:300]}")


def evaluate_locate_nail_annotations(
    vlm_query_fn,
    originals: list[tuple[str, np.ndarray]],
    annotated_images: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    *,
    backend: str = DEFAULT_REWARD_BACKEND,
    model: str = DEFAULT_REWARD_MODEL,
) -> dict[str, Any]:
    """Ask Gemini to score the nail/gripper annotations and return task info."""
    images: list[np.ndarray] = []
    labels: list[str] = []
    for label, rgb in originals:
        if label not in annotated_images:
            continue
        images.append(np.asarray(rgb, dtype=np.uint8))
        labels.append(f"{label}: original")
        images.append(np.asarray(annotated_images[label], dtype=np.uint8))
        labels.append(f"{label}: annotated")

    if not images:
        return {
            "success": False,
            "reward": 0.0,
            "status": "no_annotated_images",
            "feedback": "No annotated images were provided for Gemini critique.",
            "records": records,
        }

    prompt = f"""
You are the reward function for an offline robot-vision task.

For each pair, compare the original wrist-camera image with the annotated image.
The annotation should:
1. Draw a tight bounding box around only the visible metal nail/pin. The box
   should not include the gripper fingers, the table, or large extra margin.
2. Mark the lower 3/4 grasp target on the nail/pin body: the point 75% of
   the way from the upper/free visible end toward the lower/base end. This
   target should be low enough to leave the upper pin exposed for handover.
3. Draw a straight line connecting the two visible gripper-tip contact points
   (the inner black fingertip/pad tips on the left and right gripper jaws).
4. Keep all marks clearly visible and geometrically plausible.

The agent reported these coordinate records:
{json.dumps(records, indent=2, default=str)}

Return ONLY valid JSON with this schema:
{{
  "score": 0.0,
  "success": false,
  "critique": "short actionable critique",
  "per_image": [
    {{
      "label": "wrist_example_1",
      "bbox_ok": true,
      "grasp_target_ok": true,
      "gripper_tip_line_ok": true,
      "notes": "brief"
    }}
  ]
}}

Scoring guidance:
- 0.90-1.00: all boxes/lower-3/4 targets/tip-lines are excellent on both images.
- 0.70-0.89: usable but one mark is slightly loose or imprecise.
- 0.40-0.69: nail is roughly found but important geometry is wrong/missing.
- <0.40: wrong object, missing nail, missing gripper-tip line, or unreadable.
Set success=true only if score >= 0.85 and every image is usable.
""".strip()

    response = vlm_query_fn(
        text=prompt,
        backend=backend,
        model=model,
        image=images,
        image_labels=labels,
        temperature=0.0,
        reasoning_effort="high",
        telemetry_source="locate_nail_reward",
    )
    parsed = _extract_json_object(str(response))
    score = float(parsed.get("score", 0.0))
    success = bool(parsed.get("success", False)) and score >= 0.85
    return {
        "success": success,
        "reward": max(0.0, min(1.0, score)),
        "status": "success" if success else "needs_improvement",
        "feedback": str(parsed.get("critique", "")),
        "gemini_response": parsed,
        "records": records,
        "image_labels": labels,
    }


if __name__ == "__main__":
    print(json.dumps(run_locate_nail_baseline(), indent=2))
