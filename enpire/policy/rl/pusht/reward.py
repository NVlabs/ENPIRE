# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PushT red-T success detection.

Success is binary:
  1. connected red T has a downtop pose
     (crossbar image-horizontal within tolerance and below the stem)
  2. connected red T bbox is fully inside the saved success rectangle
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

T_TOTAL_WIDTH_M = 0.16
T_TOTAL_HEIGHT_M = 0.160
T_STEM_WIDTH_M = 0.040
T_CROSSBAR_H_M = 0.040


SUCCESS_RECT_KEY = "success_rect_xywh"
SUCCESS_ORIENTATION = "downtop"
CROSSBAR_TABLE_PARALLEL_TOL_DEG = 10.0
CROSSBAR_BELOW_STEM_MIN_FRAC = 0.12
CROSSBAR_BELOW_STEM_MIN_PX = 8.0

RED_HSV_LOW1 = np.array([0, 80, 80], dtype=np.uint8)
RED_HSV_HIGH1 = np.array([10, 255, 255], dtype=np.uint8)
RED_HSV_LOW2 = np.array([165, 80, 80], dtype=np.uint8)
RED_HSV_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)
MIN_CONTOUR_AREA_PX = 500
POLY_EPSILONS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)


def _as_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxWx3 image, got shape={arr.shape}")
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)


def _red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, RED_HSV_LOW1, RED_HSV_HIGH1),
        cv2.inRange(hsv, RED_HSV_LOW2, RED_HSV_HIGH2),
    )
    rgb_gate = cv2.inRange(
        bgr,
        np.array([0, 0, 195], dtype=np.uint8),
        np.array([255, 255, 255], dtype=np.uint8),
    )
    mask = cv2.bitwise_and(mask, rgb_gate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _dominant_component(mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    info: dict[str, Any] = {
        "valid_component": 0.0,
        "component_count": 0.0,
        "dominant_fraction": 0.0,
        "area": 0.0,
        "cx": 0.0,
        "cy": 0.0,
        "topology_ok": 0.0,
        "directed_valid": 0.0,
        "directed_angle_deg": 0.0,
        "directed_rmse_px": 0.0,
        "crossbar_line": [],
        "stem_line": [],
        "model_corners": [],
    }
    if n_labels <= 1:
        return out, info

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = areas >= float(MIN_CONTOUR_AREA_PX)
    if not np.any(keep):
        return out, info

    kept = np.flatnonzero(keep) + 1
    largest = int(kept[np.argmax(stats[kept, cv2.CC_STAT_AREA])])
    out[labels == largest] = 255

    total_area = float(np.sum(stats[kept, cv2.CC_STAT_AREA]))
    area = float(stats[largest, cv2.CC_STAT_AREA])
    cx, cy = [float(v) for v in centroids[largest]]
    fit = _fit_t_pose(out)
    info.update(
        {
            "valid_component": 1.0,
            "component_count": float(kept.size),
            "dominant_fraction": float(area / total_area) if total_area else 0.0,
            "area": area,
            "cx": cx,
            "cy": cy,
            "topology_ok": float(fit["valid"]),
            "directed_valid": float(fit["valid"]),
            "directed_angle_deg": float(fit["angle_deg"]),
            "directed_rmse_px": float(fit["rmse_px"]),
            "crossbar_line": fit["crossbar_line"],
            "stem_line": fit["stem_line"],
            "model_corners": fit["model_corners"],
        }
    )
    return out, info


def _t_model_points() -> np.ndarray:
    W, H, sw, ch = T_TOTAL_WIDTH_M, T_TOTAL_HEIGHT_M, T_STEM_WIDTH_M, T_CROSSBAR_H_M
    return np.array(
        [
            [-W / 2, H / 2],
            [W / 2, H / 2],
            [W / 2, H / 2 - ch],
            [sw / 2, H / 2 - ch],
            [sw / 2, -H / 2],
            [-sw / 2, -H / 2],
            [-sw / 2, H / 2 - ch],
            [-W / 2, H / 2 - ch],
        ],
        dtype=np.float64,
    )


def _fit_t_pose(mask: np.ndarray) -> dict[str, Any]:
    empty = {
        "valid": 0.0,
        "angle_deg": 0.0,
        "rmse_px": 0.0,
        "crossbar_line": [],
        "stem_line": [],
        "model_corners": [],
    }
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA_PX]
    model = _t_model_points()
    best: dict[str, Any] | None = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        arc = cv2.arcLength(contour, True)
        if arc <= 0.0:
            continue
        for eps_frac in POLY_EPSILONS:
            pts = cv2.approxPolyDP(contour, eps_frac * arc, True).reshape(-1, 2).astype(np.float64)
            if len(pts) != 8:
                continue
            longest_idx = _longest_edge_idx(pts)
            if not _topology_ok(pts, longest_idx):
                continue
            for ordered in (_order_from_edge(pts, longest_idx, True), _order_from_edge(pts, longest_idx, False)):
                scale, R, t, rmse = _fit_similarity(model, ordered)
                fitted = scale * (model @ R.T) + t
                candidate = {
                    "valid": 1.0,
                    "angle_deg": float(np.degrees(np.arctan2(R[1, 1], R[0, 1])) % 360.0),
                    "rmse_px": rmse,
                    "crossbar_line": [fitted[0].tolist(), fitted[1].tolist()],
                    "stem_line": [
                        (0.5 * (fitted[4] + fitted[5])).tolist(),
                        (0.5 * (fitted[0] + fitted[1])).tolist(),
                    ],
                    "model_corners": fitted.tolist(),
                }
                if best is None or candidate["rmse_px"] < best["rmse_px"]:
                    best = candidate
    return best if best is not None else empty


def _longest_edge_idx(pts: np.ndarray) -> int:
    return int(np.argmax(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))


def _topology_ok(pts: np.ndarray, longest_idx: int) -> bool:
    edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    longest = float(edges[longest_idx])
    return bool(
        len(pts) == 8
        and longest > 1e-6
        and float(edges[(longest_idx - 1) % 8]) <= longest / 3.0
        and float(edges[(longest_idx + 1) % 8]) <= longest / 3.0
    )


def _order_from_edge(pts: np.ndarray, edge_idx: int, forward: bool) -> np.ndarray:
    if forward:
        return pts[[(edge_idx + k) % len(pts) for k in range(len(pts))]].astype(np.float64)
    return pts[[(edge_idx + 1 - k) % len(pts) for k in range(len(pts))]].astype(np.float64)


def _fit_similarity(model_xy: np.ndarray, observed_xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    P = np.asarray(model_xy, dtype=np.float64)
    Q = np.asarray(observed_xy, dtype=np.float64)
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - cP, Q - cQ
    U, S, Vt = np.linalg.svd(P0.T @ Q0)
    D = np.eye(2)
    D[1, 1] = -1.0 if np.linalg.det(Vt.T @ U.T) < 0.0 else 1.0
    R = Vt.T @ D @ U.T
    scale = float(np.sum(S * np.diag(D)) / max(float(np.sum(P0 * P0)), 1e-9))
    t = cQ - scale * (R @ cP)
    residuals = scale * (P @ R.T) + t - Q
    return scale, R, t, float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def _score_success(
    cur_mask: np.ndarray,
    cur_info: dict[str, Any],
    success_rect: tuple[int, int, int, int] | list[int] | None,
) -> dict[str, Any]:
    success_rect = _clamp_rect(success_rect, cur_mask.shape[:2])
    bbox = _component_bbox(cur_mask)
    crossbar_angle = _line_angle_deg(cur_info.get("crossbar_line", []))
    crossbar_error = _axis_parallel_error_deg(crossbar_angle)
    crossbar_mid_y = _line_midpoint_y(cur_info.get("crossbar_line", []))
    stem_mid_y = _line_midpoint_y(cur_info.get("stem_line", []))
    parallel_ok = crossbar_error <= CROSSBAR_TABLE_PARALLEL_TOL_DEG
    below_delta_px = _crossbar_below_stem_delta_px(crossbar_mid_y, stem_mid_y)
    below_margin_px = _crossbar_below_stem_margin_px(bbox)
    below_ok = below_delta_px is not None and below_delta_px >= below_margin_px
    fit_ok = bool(cur_info.get("valid_component", 0.0) > 0.0 and cur_info.get("topology_ok", 0.0) > 0.0)
    orientation_ok = bool(fit_ok and parallel_ok and below_ok)
    range_ok = _rect_contains(success_rect, bbox)
    success = bool(orientation_ok and range_ok)
    return {
        "score": 1.0 if success else 0.0,
        "success": success,
        "orientation_ok": orientation_ok,
        "range_ok": range_ok,
        "success_orientation": SUCCESS_ORIENTATION,
        "success_orientation_definition": "crossbar horizontal within tolerance and meaningfully below stem in image",
        "crossbar_angle_deg": float(crossbar_angle) if crossbar_angle is not None else 0.0,
        "crossbar_parallel_error_deg": float(crossbar_error),
        "crossbar_parallel_ok": bool(parallel_ok),
        "crossbar_parallel_tol_deg": float(CROSSBAR_TABLE_PARALLEL_TOL_DEG),
        "crossbar_mid_y": float(crossbar_mid_y) if crossbar_mid_y is not None else 0.0,
        "stem_mid_y": float(stem_mid_y) if stem_mid_y is not None else 0.0,
        "crossbar_below_stem_ok": bool(below_ok),
        "crossbar_below_stem_delta_px": float(below_delta_px) if below_delta_px is not None else 0.0,
        "crossbar_below_stem_margin_px": float(below_margin_px),
        "success_rect_xywh": [int(v) for v in success_rect] if success_rect else None,
        "current_bbox_xywh": [int(v) for v in bbox] if bbox else None,
        "current_crossbar_line": cur_info.get("crossbar_line", []),
        "current_stem_line": cur_info.get("stem_line", []),
        "current_model_corners": cur_info.get("model_corners", []),
        "valid_component": float(cur_info.get("valid_component", 0.0)),
        "topology_ok": float(cur_info.get("topology_ok", 0.0)),
        "directed_valid": float(cur_info.get("directed_valid", 0.0)),
        "current_directed_angle_deg": float(cur_info.get("directed_angle_deg", 0.0)),
        "current_directed_rmse_px": float(cur_info.get("directed_rmse_px", 0.0)),
        "dominant_fraction": float(cur_info.get("dominant_fraction", 0.0)),
        "component_count": float(cur_info.get("component_count", 0.0)),
        "current_px": float(np.count_nonzero(cur_mask > 0)),
    }


def gate_score_by_avoid(score: dict[str, Any], *, avoid_active: bool) -> dict[str, Any]:
    if avoid_active:
        score["reward_gated_by_avoid"] = False
        score["reward_gate_reason"] = "avoid"
        return score
    score["score"] = 0.0
    score["success"] = False
    score["orientation_ok"] = False
    score["range_ok"] = False
    score["crossbar_parallel_ok"] = False
    score["crossbar_below_stem_ok"] = False
    score["reward_gated_by_avoid"] = True
    score["reward_gate_reason"] = "not_avoid"
    return score


def _draw_reward_overlay(bgr: np.ndarray, cur_mask: np.ndarray, score: dict[str, Any]) -> np.ndarray:
    vis = bgr.copy()
    red = cur_mask > 0
    vis[red] = (0.35 * vis[red] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    _draw_rects(vis, score)
    _draw_fit(vis, score)
    return vis


def _draw_rects(panel: np.ndarray, score: dict[str, Any]) -> None:
    rect = score.get("success_rect_xywh")
    if rect is not None:
        x, y, w, h = [int(v) for v in rect]
        color = (0, 255, 0) if score.get("range_ok") else (0, 180, 255)
        cv2.rectangle(panel, (x, y), (x + w, y + h), color, 2)
        cv2.putText(panel, "success range", (x + 4, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    bbox = score.get("current_bbox_xywh")
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(panel, (x, y), (x + w, y + h), (255, 255, 255), 1)


def _draw_fit(panel: np.ndarray, score: dict[str, Any]) -> None:
    corners = score.get("current_model_corners") or []
    if len(corners) >= 8:
        cv2.polylines(panel, [np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)], True, (255, 220, 0), 1, cv2.LINE_AA)
    _draw_line(panel, score.get("current_crossbar_line") or [], (0, 255, 255), "crossbar", arrow=False)
    _draw_line(panel, score.get("current_stem_line") or [], (255, 0, 255), "stem", arrow=True)


def _draw_line(panel: np.ndarray, line: Any, color: tuple[int, int, int], label: str, *, arrow: bool) -> None:
    if not isinstance(line, (list, tuple)) or len(line) != 2:
        return
    a = tuple(int(round(float(v))) for v in line[0])
    b = tuple(int(round(float(v))) for v in line[1])
    if arrow:
        cv2.arrowedLine(panel, a, b, color, 3, cv2.LINE_AA, tipLength=0.18)
    else:
        cv2.line(panel, a, b, color, 3, cv2.LINE_AA)
    cv2.putText(panel, label, a, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def _component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    return x0, y0, int(xs.max()) + 1 - x0, int(ys.max()) + 1 - y0


def _clamp_rect(rect: tuple[int, int, int, int] | list[int] | None, shape_hw: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if rect is None:
        return None
    h, w = shape_hw
    x, y, rw, rh = [int(round(float(v))) for v in rect]
    if rw < 0:
        x, rw = x + rw, -rw
    if rh < 0:
        y, rh = y + rh, -rh
    x0, y0 = int(np.clip(x, 0, max(0, w - 1))), int(np.clip(y, 0, max(0, h - 1)))
    x1, y1 = int(np.clip(x + rw, x0 + 1, w)), int(np.clip(y + rh, y0 + 1, h))
    return x0, y0, x1 - x0, y1 - y0


def _rect_contains(outer: tuple[int, int, int, int] | None, inner: tuple[int, int, int, int] | None) -> bool:
    if outer is None or inner is None:
        return False
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return bool(ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh)


def _rect_full_to_crop(rect: tuple[int, int, int, int] | list[int] | None, crop: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    if rect is None:
        return None
    x, y, w, h = [int(v) for v in rect]
    cx, cy, cw, ch = crop
    ix0, iy0 = max(x, cx), max(y, cy)
    ix1, iy1 = min(x + w, cx + cw), min(y + h, cy + ch)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0 - cx, iy0 - cy, ix1 - ix0, iy1 - iy0


def _scale_rect(rect: tuple[int, int, int, int] | list[int] | None, from_shape_hw: tuple[int, int], to_shape_hw: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if rect is None:
        return None
    from_h, from_w = from_shape_hw
    to_h, to_w = to_shape_hw
    if from_w <= 0 or from_h <= 0:
        return None
    sx, sy = float(to_w) / float(from_w), float(to_h) / float(from_h)
    x, y, w, h = [float(v) for v in rect]
    return _clamp_rect((x * sx, y * sy, w * sx, h * sy), to_shape_hw)


def _load_meta(meta_path: Path) -> dict[str, Any]:
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _crop(arr: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = crop
    return arr[y : y + h, x : x + w]


def _line_angle_deg(line: Any) -> float | None:
    if not isinstance(line, (list, tuple)) or len(line) != 2:
        return None
    a = np.asarray(line[0], dtype=np.float64).reshape(2)
    b = np.asarray(line[1], dtype=np.float64).reshape(2)
    delta = b - a
    if float(np.linalg.norm(delta)) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _axis_parallel_error_deg(angle_deg: float | None) -> float:
    if angle_deg is None:
        return 180.0
    return float(abs((float(angle_deg) + 90.0) % 180.0 - 90.0))


def _line_midpoint_y(line: Any) -> float | None:
    if not isinstance(line, (list, tuple)) or len(line) != 2:
        return None
    a = np.asarray(line[0], dtype=np.float64).reshape(2)
    b = np.asarray(line[1], dtype=np.float64).reshape(2)
    return float(0.5 * (a[1] + b[1]))


def _crossbar_below_stem_delta_px(crossbar_mid_y: float | None, stem_mid_y: float | None) -> float | None:
    if crossbar_mid_y is None or stem_mid_y is None:
        return None
    return float(crossbar_mid_y - stem_mid_y)


def _crossbar_below_stem_margin_px(bbox: tuple[int, int, int, int] | None) -> float:
    if bbox is None:
        return CROSSBAR_BELOW_STEM_MIN_PX
    return float(max(CROSSBAR_BELOW_STEM_MIN_PX, bbox[3] * CROSSBAR_BELOW_STEM_MIN_FRAC))

