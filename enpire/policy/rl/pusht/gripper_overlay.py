# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def normalize_rgb_frame(frame: Any) -> np.ndarray | None:
    arr = np.asarray(frame).copy()
    if arr.dtype != np.uint8:
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    return arr[:, :, :3].copy()


def draw_green_dot(
    frame: Any,
    uv: Any,
    *,
    radius_px: int = 12,
    label: str | None = None,
) -> np.ndarray | None:
    out = normalize_rgb_frame(frame)
    if out is None:
        return None
    try:
        u = int(round(float(uv[0])))
        v = int(round(float(uv[1])))
    except (TypeError, ValueError, IndexError):
        return out
    h, w = out.shape[:2]
    if -50 <= u < w + 50 and -50 <= v < h + 50:
        center = (int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1)))
        cv2.circle(out, center, int(radius_px), (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out, center, int(radius_px) + 3, (255, 255, 255), 2, cv2.LINE_AA)
    if label:
        cv2.putText(
            out,
            label,
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def sample_jsonl_path(sample_dir_or_file: Path) -> Path:
    path = Path(sample_dir_or_file)
    if path.suffix == ".jsonl":
        return path
    return path / "samples.jsonl"


def load_samples(sample_dir_or_file: Path) -> list[dict[str, Any]]:
    path = sample_jsonl_path(sample_dir_or_file)
    if not path.exists():
        return []
    samples: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return samples


def sample_file_key(sample_dir_or_file: Path) -> tuple[str, int, int] | None:
    path = sample_jsonl_path(sample_dir_or_file)
    if not path.exists():
        return None
    stat = path.stat()
    return str(path), int(stat.st_mtime_ns), int(stat.st_size)


def build_table_cache(
    sample_dir_or_file: Path,
    *,
    grid_size: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> dict[str, Any]:
    grid_size = max(2, int(grid_size))
    xlim_arr = np.asarray(xlim, dtype=np.float64)
    ylim_arr = np.asarray(ylim, dtype=np.float64)
    cell_sums: dict[tuple[int, int], np.ndarray] = {}
    cell_xy_sums: dict[tuple[int, int], np.ndarray] = {}
    cell_counts: dict[tuple[int, int], int] = {}
    sample_count = 0
    for sample in load_samples(sample_dir_or_file):
        try:
            tip = np.asarray(sample["tip_world"], dtype=np.float64).reshape(3)
            offset = np.asarray(sample["offset_px"], dtype=np.float64).reshape(2)
        except Exception:
            continue
        if not np.all(np.isfinite(tip[:2])) or not np.all(np.isfinite(offset)):
            continue
        x = float(np.clip(tip[0], xlim_arr[0], xlim_arr[1]))
        y = float(np.clip(tip[1], ylim_arr[0], ylim_arr[1]))
        cell = table_cell(x, y, xlim_arr, ylim_arr, grid_size)
        cell_sums[cell] = cell_sums.get(cell, np.zeros(2, dtype=np.float64)) + offset
        cell_xy_sums[cell] = (
            cell_xy_sums.get(cell, np.zeros(2, dtype=np.float64))
            + np.asarray([x, y], dtype=np.float64)
        )
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        sample_count += 1

    points: list[list[float]] = []
    offsets: list[list[float]] = []
    for cell, count in cell_counts.items():
        points.append((cell_xy_sums[cell] / float(count)).tolist())
        offsets.append((cell_sums[cell] / float(count)).tolist())
    return {
        "sample_count": sample_count,
        "occupied_cells": len(cell_counts),
        "grid_size": grid_size,
        "xlim": xlim_arr.tolist(),
        "ylim": ylim_arr.tolist(),
        "points": points,
        "offsets": offsets,
    }


def table_prediction(
    cache: dict[str, Any],
    tip_world: Any,
    *,
    min_samples: int,
    neighbors: int,
    power: float,
) -> dict[str, Any]:
    sample_count = int(cache.get("sample_count", 0))
    xlim = np.asarray(cache["xlim"], dtype=np.float64)
    ylim = np.asarray(cache["ylim"], dtype=np.float64)
    grid_size = int(cache["grid_size"])
    tip = np.asarray(tip_world, dtype=np.float64).reshape(3)
    x = float(np.clip(tip[0], xlim[0], xlim[1]))
    y = float(np.clip(tip[1], ylim[0], ylim[1]))
    cell = table_cell(x, y, xlim, ylim, grid_size)
    result = {
        "enabled": True,
        "offset_px": [0.0, 0.0],
        "sample_count": sample_count,
        "occupied_cells": int(cache.get("occupied_cells", 0)),
        "cell": [int(cell[0]), int(cell[1])],
        "in_bounds": bool(xlim[0] <= tip[0] <= xlim[1] and ylim[0] <= tip[1] <= ylim[1]),
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [float(ylim[0]), float(ylim[1])],
        "grid_size": grid_size,
    }
    if sample_count < int(min_samples) or not cache["points"]:
        return result
    points = np.asarray(cache["points"], dtype=np.float64)
    offsets = np.asarray(cache["offsets"], dtype=np.float64)
    dists = np.linalg.norm(points - np.asarray([x, y], dtype=np.float64), axis=1)
    if dists.size == 0:
        return result
    exact = np.flatnonzero(dists < 1e-9)
    if exact.size:
        offset = offsets[int(exact[0])]
        result["source"] = "cell"
        result["nearest_distance_m"] = 0.0
    else:
        k = min(int(neighbors), dists.size)
        idx = np.argpartition(dists, k - 1)[:k]
        weights = 1.0 / np.maximum(dists[idx], 1e-6) ** float(power)
        offset = np.sum(offsets[idx] * weights[:, None], axis=0) / np.sum(weights)
        result["source"] = "idw"
        result["nearest_distance_m"] = float(np.min(dists[idx]))
        result["neighbors"] = int(k)
    result["offset_px"] = [float(offset[0]), float(offset[1])]
    return result


def table_cell(
    x: float,
    y: float,
    xlim: np.ndarray,
    ylim: np.ndarray,
    grid_size: int,
) -> tuple[int, int]:
    x_span = max(float(xlim[1] - xlim[0]), 1e-9)
    y_span = max(float(ylim[1] - ylim[0]), 1e-9)
    ix = int(np.floor((float(x) - float(xlim[0])) / x_span * grid_size))
    iy = int(np.floor((float(y) - float(ylim[0])) / y_span * grid_size))
    return int(np.clip(ix, 0, grid_size - 1)), int(np.clip(iy, 0, grid_size - 1))


def coverage_frame(frame: Any, sample_dir_or_file: Path) -> np.ndarray | None:
    out = normalize_rgb_frame(frame)
    if out is None:
        return None
    h, w = out.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    samples = load_samples(sample_dir_or_file)
    radius_px = max(18, int(round(min(h, w) * 0.045)))
    for sample in samples:
        uv = sample.get("corrected_uv")
        if uv is None:
            continue
        try:
            u = int(round(float(uv[0])))
            v = int(round(float(uv[1])))
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(heat, (u, v), radius_px, 1.0, -1, cv2.LINE_AA)
    if heat.max(initial=0.0) > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=radius_px * 0.45)
        heat = heat / max(float(heat.max()), 1e-6)
        color = cv2.applyColorMap((heat * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        alpha = np.clip(heat * 0.42, 0.0, 0.42)[:, :, None]
        out = (out.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha)
        out = np.clip(out, 0, 255).astype(np.uint8)
    for sample in samples:
        uv = sample.get("corrected_uv")
        if uv is None:
            continue
        try:
            u = int(round(float(uv[0])))
            v = int(round(float(uv[1])))
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(out, (u, v), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (u, v), 5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(
        out,
        f"coverage samples={len(samples)} radius={radius_px}px",
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out

