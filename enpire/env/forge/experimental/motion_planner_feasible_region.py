"""Helpers for visualizing the motion planner's feasible region.

This module scans a dense XYZ grid with fixed per-arm orientation, stores the
full success/failure map plus successful joint trajectories, and persists the
result to disk so later sessions can load it immediately.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path
import threading
from typing import Any, Callable, Literal

import numpy as np


PlannerSide = Literal["left", "right"]
TrajectoryData = dict[str, np.ndarray]

_CACHE_VERSION = 2
_CACHE_DIR = (
    Path(__file__).resolve().parents[1] / ".cache" / "motion_planner_feasible_region"
)

# Coarser default XYZ grid so feasible-region scans finish in a reasonable
# amount of time while still covering the forward workspace.
_DEFAULT_X_VALUES = tuple(float(v) for v in np.linspace(0.10, 0.82, 13))
_DEFAULT_Y_VALUES = tuple(float(v) for v in np.linspace(-0.45, 0.45, 13))
_DEFAULT_Z_VALUES = tuple(float(v) for v in np.linspace(0.88, 1.32, 6))
# Solve multiple XYZ points concurrently; each worker owns its own planner.
_DEFAULT_NUM_WORKERS = max(1, min(12, os.cpu_count() or 1))


class SamplingInterrupted(RuntimeError):
    """Raised when a feasible-region scan is cancelled."""


@dataclass(frozen=True)
class FeasibleRegionSampleConfig:
    """Sampling configuration for the Viser feasible-region preview."""

    x_values: tuple[float, ...] = _DEFAULT_X_VALUES
    y_values: tuple[float, ...] = _DEFAULT_Y_VALUES
    z_values: tuple[float, ...] = _DEFAULT_Z_VALUES
    max_rrt_iters: int = 800
    step_size: float = 0.15
    point_size_m: float = 0.03
    num_workers: int = _DEFAULT_NUM_WORKERS

    @property
    def positions_per_side(self) -> int:
        return len(self.x_values) * len(self.y_values) * len(self.z_values)


@dataclass(frozen=True)
class FeasibleRegionSideResult:
    """Feasible-region sample results for one arm."""

    side: PlannerSide
    points: np.ndarray
    scores: np.ndarray
    successes: np.ndarray
    totals: np.ndarray
    trajectories: tuple[TrajectoryData | None, ...]


DEFAULT_FEASIBLE_REGION_CONFIG = FeasibleRegionSampleConfig()


def scores_to_rgb(scores: np.ndarray) -> np.ndarray:
    """Map feasibility scores in [0, 1] to red→green colors."""

    clamped = np.clip(np.asarray(scores, dtype=np.float32).reshape(-1), 0.0, 1.0)
    colors = np.zeros((len(clamped), 3), dtype=np.uint8)
    colors[:, 0] = np.round(255.0 * (1.0 - clamped)).astype(np.uint8)
    colors[:, 1] = np.round(255.0 * clamped).astype(np.uint8)
    colors[:, 2] = 48
    return colors


def build_side_points(
    side: PlannerSide,
    config: FeasibleRegionSampleConfig = DEFAULT_FEASIBLE_REGION_CONFIG,
) -> np.ndarray:
    """Build the stable point order for one arm's XYZ scan."""

    del side  # side is currently only kept for API symmetry / future use.
    points: list[list[float]] = []
    for x in config.x_values:
        for y in config.y_values:
            for z in config.z_values:
                points.append([float(x), float(y), float(z)])
    return np.asarray(points, dtype=np.float32)


def _rounded_list(arr: np.ndarray, decimals: int = 4) -> list[float]:
    return np.round(np.asarray(arr, dtype=np.float64).reshape(-1), decimals).tolist()


def _serialize_config(config: FeasibleRegionSampleConfig) -> dict[str, Any]:
    return {
        "x_values": [float(v) for v in config.x_values],
        "y_values": [float(v) for v in config.y_values],
        "z_values": [float(v) for v in config.z_values],
        "max_rrt_iters": int(config.max_rrt_iters),
        "step_size": float(config.step_size),
        "point_size_m": float(config.point_size_m),
        "num_workers": int(config.num_workers),
    }


def compute_feasible_region_cache_key(
    *,
    current_left_jp: np.ndarray,
    current_right_jp: np.ndarray,
    left_gripper: float,
    right_gripper: float,
    left_target_quat_xyzw: np.ndarray,
    right_target_quat_xyzw: np.ndarray,
    config: FeasibleRegionSampleConfig = DEFAULT_FEASIBLE_REGION_CONFIG,
) -> str:
    """Create a deterministic cache key for a scan from a specific start state."""

    payload = {
        "version": _CACHE_VERSION,
        "current_left_jp": _rounded_list(current_left_jp),
        "current_right_jp": _rounded_list(current_right_jp),
        "left_gripper": round(float(left_gripper), 4),
        "right_gripper": round(float(right_gripper), 4),
        "left_target_quat_xyzw": _rounded_list(left_target_quat_xyzw),
        "right_target_quat_xyzw": _rounded_list(right_target_quat_xyzw),
        "config": _serialize_config(config),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:24]


def feasible_region_cache_path(cache_key: str) -> Path:
    return _CACHE_DIR / f"{cache_key}.pkl.gz"


def save_feasible_region_cache(
    *,
    cache_key: str,
    config: FeasibleRegionSampleConfig,
    start_left_jp: np.ndarray,
    start_right_jp: np.ndarray,
    left_gripper: float,
    right_gripper: float,
    left_target_quat_xyzw: np.ndarray,
    right_target_quat_xyzw: np.ndarray,
    results: dict[str, FeasibleRegionSideResult],
) -> Path:
    """Persist a feasible-region scan to disk."""

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _CACHE_VERSION,
        "cache_key": cache_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": _serialize_config(config),
        "start_state": {
            "left_joint_pos": np.asarray(start_left_jp, dtype=np.float32),
            "right_joint_pos": np.asarray(start_right_jp, dtype=np.float32),
            "left_gripper": float(left_gripper),
            "right_gripper": float(right_gripper),
        },
        "target_quats": {
            "left": np.asarray(left_target_quat_xyzw, dtype=np.float32),
            "right": np.asarray(right_target_quat_xyzw, dtype=np.float32),
        },
        "sides": {
            side: {
                "points": result.points.astype(np.float32),
                "scores": result.scores.astype(np.float32),
                "successes": result.successes.astype(np.int8),
                "totals": result.totals.astype(np.int8),
                "trajectories": list(result.trajectories),
            }
            for side, result in results.items()
        },
    }
    path = feasible_region_cache_path(cache_key)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    return path


def load_feasible_region_cache(cache_key: str) -> dict[str, Any] | None:
    """Load a feasible-region cache payload if present and compatible."""

    path = feasible_region_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _CACHE_VERSION:
        return None
    return payload


def side_result_from_cache(side: str, side_payload: dict[str, Any]) -> FeasibleRegionSideResult:
    """Rehydrate a ``FeasibleRegionSideResult`` from cached payload data."""

    trajectories = tuple(
        None if traj is None else {
            "left_positions": np.asarray(traj["left_positions"], dtype=np.float32),
            "right_positions": np.asarray(traj["right_positions"], dtype=np.float32),
        }
        for traj in side_payload["trajectories"]
    )
    return FeasibleRegionSideResult(
        side=side,  # type: ignore[arg-type]
        points=np.asarray(side_payload["points"], dtype=np.float32),
        scores=np.asarray(side_payload["scores"], dtype=np.float32),
        successes=np.asarray(side_payload["successes"], dtype=np.int32),
        totals=np.asarray(side_payload["totals"], dtype=np.int32),
        trajectories=trajectories,
    )


def compute_side_feasible_region(
    planner,
    *,
    side: PlannerSide,
    current_left_jp: np.ndarray,
    current_right_jp: np.ndarray,
    target_quat_xyzw: np.ndarray,
    left_gripper: float = 1.0,
    right_gripper: float = 1.0,
    config: FeasibleRegionSampleConfig = DEFAULT_FEASIBLE_REGION_CONFIG,
    progress_callback: Callable[[int, int], None] | None = None,
    point_result_callback: Callable[[int, np.ndarray, float, TrajectoryData | None], None]
    | None = None,
    stop_event=None,
) -> FeasibleRegionSideResult:
    """Evaluate motion-planner feasibility over an XYZ grid for one arm.

    Orientation is held fixed at ``target_quat_xyzw`` for the whole scan.
    Successful trajectories are kept so clicked points can execute instantly.
    """

    current_left_jp = np.asarray(current_left_jp, dtype=np.float64).reshape(-1)[:6].copy()
    current_right_jp = np.asarray(current_right_jp, dtype=np.float64).reshape(-1)[:6].copy()
    target_quat_xyzw = (
        np.asarray(target_quat_xyzw, dtype=np.float64).reshape(-1)[:4].copy()
    )
    points = build_side_points(side, config)
    n_points = int(points.shape[0])
    successes = np.zeros((n_points,), dtype=np.int32)
    totals = np.ones((n_points,), dtype=np.int32)
    trajectories: list[TrajectoryData | None] = [None] * n_points
    planner_cls = planner.__class__
    planner_model_xml = getattr(planner, "_model_xml_path", None)
    thread_local = threading.local()

    def _get_thread_planner():
        planner_local = getattr(thread_local, "planner", None)
        if planner_local is None:
            if planner_model_xml is None:
                planner_local = planner_cls()
            else:
                planner_local = planner_cls(model_xml=planner_model_xml)
            thread_local.planner = planner_local
        return planner_local

    def _solve_point(point_idx: int) -> tuple[int, int, TrajectoryData | None]:
        if stop_event is not None and stop_event.is_set():
            raise SamplingInterrupted("Feasible-region scan cancelled")
        local_planner = _get_thread_planner()
        target_pos = np.asarray(points[point_idx], dtype=np.float64)
        kwargs = dict(
            current_left_jp=current_left_jp,
            current_right_jp=current_right_jp,
            side=side,
            max_iters=config.max_rrt_iters,
            step_size=config.step_size,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
            verbose=False,
        )
        if side == "left":
            kwargs["target_left_pos"] = target_pos
            kwargs["target_left_quat_xyzw"] = target_quat_xyzw
        else:
            kwargs["target_right_pos"] = target_pos
            kwargs["target_right_quat_xyzw"] = target_quat_xyzw
        result = local_planner.plan_to_pose(**kwargs)
        if result["status"] != "Success":
            return point_idx, 0, None
        traj = {
            "left_positions": np.asarray(result["left_positions"], dtype=np.float32),
            "right_positions": np.asarray(result["right_positions"], dtype=np.float32),
        }
        return point_idx, 1, traj

    completed = 0
    max_workers = max(1, min(int(config.num_workers), n_points))
    if max_workers == 1:
        for point_idx in range(n_points):
            if stop_event is not None and stop_event.is_set():
                raise SamplingInterrupted("Feasible-region scan cancelled")
            point_idx, success, traj = _solve_point(point_idx)
            successes[point_idx] = success
            trajectories[point_idx] = traj
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, n_points)
            if point_result_callback is not None:
                point_result_callback(point_idx, points[point_idx], float(success), traj)
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mp-region") as executor:
            futures = [executor.submit(_solve_point, point_idx) for point_idx in range(n_points)]
            for future in as_completed(futures):
                if stop_event is not None and stop_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise SamplingInterrupted("Feasible-region scan cancelled")
                point_idx, success, traj = future.result()
                successes[point_idx] = success
                trajectories[point_idx] = traj
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, n_points)
                if point_result_callback is not None:
                    point_result_callback(point_idx, points[point_idx], float(success), traj)

    scores = successes.astype(np.float32) / np.maximum(totals, 1)
    return FeasibleRegionSideResult(
        side=side,
        points=points,
        scores=scores,
        successes=successes,
        totals=totals,
        trajectories=tuple(trajectories),
    )
