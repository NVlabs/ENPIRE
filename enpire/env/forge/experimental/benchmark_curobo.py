"""Benchmark cuRobo planning speed and reachable workspace.

This script evaluates the current cuRobo integration in two ways:

1. Planning speed:
   - Samples random Cartesian targets in a configurable workspace box.
   - Measures wall-clock latency of ``planner.plan_to_pose(...)``.
   - Reports success rate and timing percentiles.

2. Reachable workspace:
   - Scans a Cartesian grid with a fixed orientation.
   - Records whether cuRobo can find a plan from a fixed start state.
   - Writes per-sample CSV data plus a JSON summary for later plotting.

Examples:
    # Quick single-arm benchmark from the current zero start state:
    uv run python -m experimental.benchmark_curobo \
      --mode both --side left --speed-samples 20 --grid-x 6 --grid-y 6 --grid-z 4

    # Symmetric bimanual benchmark:
    uv run python -m experimental.benchmark_curobo \
      --mode both --side both --start-state zero

    # Use the exact deployed planner wrapper, including MuJoCo post-validation:
    uv run python -m experimental.benchmark_curobo \
      --mode both --side both --validate-with-mujoco
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from enpire.env.forge.experimental.motion_planner_curobo import YamMotionPlannerCurobo
from enpire.env.forge.robot.yam.kinematics import YamKinematics


HOME_LEFT = np.array([-0.3, 1.35, 1.6, -0.8, 0.3, -0.25], dtype=np.float64)
HOME_RIGHT = np.array([0.3, 1.35, 1.6, -0.8, -0.3, 0.25], dtype=np.float64)
ZERO_LEFT = np.zeros(6, dtype=np.float64)
ZERO_RIGHT = np.zeros(6, dtype=np.float64)
IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class StartState:
    name: str
    left: np.ndarray
    right: np.ndarray


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["speed", "reachability", "both"], default="both")
    parser.add_argument("--side", choices=["left", "right", "both"], default="left")
    parser.add_argument("--start-state", choices=["zero", "home"], default="zero")
    parser.add_argument(
        "--orientation",
        choices=["current", "identity", "any_roll"],
        default="current",
        help=(
            "Target orientation policy. 'current' and 'identity' use one fixed quaternion. "
            "'any_roll' approximates position-only reachability by sweeping wrist-roll-like "
            "rotations around the current end-effector local z-axis and accepting any success."
        ),
    )
    parser.add_argument(
        "--orientation-samples",
        type=int,
        default=12,
        help="Number of roll orientations to test when --orientation=any_roll.",
    )
    parser.add_argument(
        "--orientation-extra-rpy-deg",
        type=str,
        default="",
        help=(
            "Optional semicolon-separated absolute XYZ Euler orientations in degrees to test in "
            "addition to the default quaternion when --orientation=any_roll. "
            'Example: --orientation-extra-rpy-deg "-180,0,-90"'
        ),
    )
    parser.add_argument("--robot-cfg-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--validate-with-mujoco",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run the existing MuJoCo collision validator on cuRobo trajectories.",
    )

    parser.add_argument("--speed-samples", type=int, default=50)
    parser.add_argument("--speed-warmup-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--grid-x", type=int, default=8)
    parser.add_argument("--grid-y", type=int, default=8)
    parser.add_argument("--grid-z", type=int, default=5)
    parser.add_argument("--x-min", type=float, default=0.35)
    parser.add_argument("--x-max", type=float, default=0.80)
    parser.add_argument("--y-abs-min", type=float, default=0.05)
    parser.add_argument("--y-abs-max", type=float, default=0.45)
    parser.add_argument("--z-min", type=float, default=0.85)
    parser.add_argument("--z-max", type=float, default=1.30)

    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/curobo_benchmark"))
    parser.add_argument("--tag", type=str, default="")
    return parser


def _get_start_state(name: str) -> StartState:
    if name == "zero":
        return StartState("zero", ZERO_LEFT.copy(), ZERO_RIGHT.copy())
    if name == "home":
        return StartState("home", HOME_LEFT.copy(), HOME_RIGHT.copy())
    raise ValueError(f"Unknown start state: {name}")


def _quantile(arr_ms: np.ndarray, q: float) -> float | None:
    if arr_ms.size == 0:
        return None
    return float(np.quantile(arr_ms, q))


def _stats_ms(values_ms: list[float]) -> dict[str, Any]:
    if not values_ms:
        return {
            "count": 0,
            "mean_ms": None,
            "std_ms": None,
            "min_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "p50_ms": _quantile(arr, 0.50),
        "p90_ms": _quantile(arr, 0.90),
        "p95_ms": _quantile(arr, 0.95),
        "p99_ms": _quantile(arr, 0.99),
        "max_ms": float(arr.max()),
    }


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _print_header(args: argparse.Namespace, start: StartState, kin: YamKinematics) -> dict[str, Any]:
    cur_l_pos, cur_l_quat, cur_r_pos, cur_r_quat = kin.forward_kinematics(start.left, start.right)
    print("[cuRobo Benchmark] Configuration")
    print(f"  mode:                  {args.mode}")
    print(f"  side:                  {args.side}")
    print(f"  start_state:           {start.name}")
    print(f"  orientation:           {args.orientation}")
    print(f"  validate_with_mujoco:  {args.validate_with_mujoco}")
    print(f"  robot_cfg_path:        {args.robot_cfg_path or '(planner default)'}")
    print(f"  device:                {args.device}")
    print("  start left joints:    ", np.round(start.left, 4))
    print("  start right joints:   ", np.round(start.right, 4))
    print("  start left pose:      ", np.round(cur_l_pos, 4), np.round(cur_l_quat, 4))
    print("  start right pose:     ", np.round(cur_r_pos, 4), np.round(cur_r_quat, 4))
    return {
        "start_left_pose": {"position": cur_l_pos, "quat_xyzw": cur_l_quat},
        "start_right_pose": {"position": cur_r_pos, "quat_xyzw": cur_r_quat},
    }


def _orientation_targets(
    orientation_mode: str,
    current_left_quat: np.ndarray,
    current_right_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if orientation_mode == "current":
        return current_left_quat.copy(), current_right_quat.copy()
    if orientation_mode == "identity":
        return IDENTITY_QUAT_XYZW.copy(), IDENTITY_QUAT_XYZW.copy()
    if orientation_mode == "any_roll":
        return current_left_quat.copy(), current_right_quat.copy()
    raise ValueError(f"Unknown orientation mode: {orientation_mode}")


def _quat_xyzw_with_local_roll(quat_xyzw: np.ndarray, roll_rad: float) -> np.ndarray:
    base = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64))
    rolled = base * Rotation.from_rotvec(np.array([0.0, 0.0, roll_rad], dtype=np.float64))
    return rolled.as_quat().astype(np.float64)


def _parse_extra_rpy_deg_specs(spec: str) -> list[np.ndarray]:
    text = (spec or "").strip()
    if not text:
        return []
    out: list[np.ndarray] = []
    for chunk in text.split(";"):
        vals = [v.strip() for v in chunk.split(",") if v.strip()]
        if len(vals) != 3:
            raise ValueError(
                "--orientation-extra-rpy-deg entries must have exactly 3 comma-separated values"
            )
        out.append(np.asarray([float(v) for v in vals], dtype=np.float64))
    return out


def _orientation_candidate_specs(
    orientation_mode: str,
    base_left_quat: np.ndarray,
    base_right_quat: np.ndarray,
    orientation_samples: int,
    extra_rpy_deg_specs: list[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    if orientation_mode != "any_roll":
        return [
            {
                "candidate_index": 0,
                "roll_rad": 0.0,
                "left_quat_xyzw": np.asarray(base_left_quat, dtype=np.float64).copy(),
                "right_quat_xyzw": np.asarray(base_right_quat, dtype=np.float64).copy(),
            }
        ]

    extras = list(extra_rpy_deg_specs or [])
    if extras:
        candidates = [
            {
                "candidate_index": 0,
                "roll_rad": 0.0,
                "left_quat_xyzw": np.asarray(base_left_quat, dtype=np.float64).copy(),
                "right_quat_xyzw": np.asarray(base_right_quat, dtype=np.float64).copy(),
                "source": "default",
            }
        ]
        for i, rpy_deg in enumerate(extras, start=1):
            quat_xyzw = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_quat().astype(np.float64)
            candidates.append(
                {
                    "candidate_index": i,
                    "roll_rad": None,
                    "rpy_deg_xyz": rpy_deg.tolist(),
                    "left_quat_xyzw": quat_xyzw.copy(),
                    "right_quat_xyzw": quat_xyzw.copy(),
                    "source": "extra_rpy_deg",
                }
            )
        return candidates

    n = max(1, int(orientation_samples))
    if n == 1:
        roll_values = np.array([0.0], dtype=np.float64)
    else:
        roll_values = np.linspace(-math.pi, math.pi, n, endpoint=False, dtype=np.float64)
    return [
        {
            "candidate_index": int(i),
            "roll_rad": float(roll_rad),
            "left_quat_xyzw": _quat_xyzw_with_local_roll(base_left_quat, float(roll_rad)),
            "right_quat_xyzw": _quat_xyzw_with_local_roll(base_right_quat, float(roll_rad)),
            "source": "roll_sweep",
        }
        for i, roll_rad in enumerate(roll_values)
    ]


def _single_arm_target(
    side: str,
    position: np.ndarray,
    left_quat: np.ndarray,
    right_quat: np.ndarray,
) -> dict[str, Any]:
    if side == "left":
        return {
            "side": "left",
            "target_left_pos": position,
            "target_left_quat_xyzw": left_quat,
            "target_right_pos": None,
            "target_right_quat_xyzw": None,
        }
    if side == "right":
        return {
            "side": "right",
            "target_left_pos": None,
            "target_left_quat_xyzw": None,
            "target_right_pos": position,
            "target_right_quat_xyzw": right_quat,
        }
    raise ValueError(f"Expected single-arm side, got: {side}")


def _bimanual_target(
    x: float,
    y_abs: float,
    z: float,
    left_quat: np.ndarray,
    right_quat: np.ndarray,
) -> dict[str, Any]:
    return {
        "side": "both",
        "target_left_pos": np.array([x, +y_abs, z], dtype=np.float64),
        "target_left_quat_xyzw": left_quat,
        "target_right_pos": np.array([x, -y_abs, z], dtype=np.float64),
        "target_right_quat_xyzw": right_quat,
    }


def _plan_once(
    planner: YamMotionPlannerCurobo,
    start: StartState,
    request: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    t0 = perf_counter()
    result = planner.plan_to_pose(
        current_left_jp=start.left,
        current_right_jp=start.right,
        target_left_pos=request["target_left_pos"],
        target_left_quat_xyzw=request["target_left_quat_xyzw"],
        target_right_pos=request["target_right_pos"],
        target_right_quat_xyzw=request["target_right_quat_xyzw"],
        side=request["side"],
        left_gripper=1.0,
        right_gripper=1.0,
    )
    elapsed_ms = 1000.0 * (perf_counter() - t0)
    return result, elapsed_ms


def _workspace_values(lo: float, hi: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([(lo + hi) / 2.0], dtype=np.float64)
    return np.linspace(lo, hi, n, dtype=np.float64)


def _speed_samples(
    args: argparse.Namespace,
    rng: np.random.Generator,
    left_quat: np.ndarray,
    right_quat: np.ndarray,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for _ in range(args.speed_samples + args.speed_warmup_samples):
        x = float(rng.uniform(args.x_min, args.x_max))
        y_abs = float(rng.uniform(args.y_abs_min, args.y_abs_max))
        z = float(rng.uniform(args.z_min, args.z_max))
        if args.side == "both":
            request = _bimanual_target(x, y_abs, z, left_quat, right_quat)
            request["sample_meta"] = {"x": x, "y_abs": y_abs, "z": z}
        else:
            sign = 1.0 if args.side == "left" else -1.0
            pos = np.array([x, sign * y_abs, z], dtype=np.float64)
            request = _single_arm_target(args.side, pos, left_quat, right_quat)
            request["sample_meta"] = {"x": x, "y": float(pos[1]), "z": z}
        samples.append(request)
    return samples


def _run_speed_benchmark(
    args: argparse.Namespace,
    planner: YamMotionPlannerCurobo,
    start: StartState,
    left_quat: np.ndarray,
    right_quat: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    samples = _speed_samples(args, rng, left_quat, right_quat)
    rows: list[dict[str, Any]] = []
    all_times_ms: list[float] = []
    success_times_ms: list[float] = []
    failure_reasons: dict[str, int] = {}

    print(
        f"[Speed] Running {args.speed_samples} measured samples "
        f"(+{args.speed_warmup_samples} warmup) for side={args.side}..."
    )

    for idx, request in enumerate(samples):
        result, elapsed_ms = _plan_once(planner, start, request)
        warmup = idx < args.speed_warmup_samples
        success = result["status"] == "Success"
        status_detail = result.get("status_detail")
        num_waypoints = int(result["position"].shape[0]) if "position" in result else 0

        row = {
            "sample_index": idx,
            "warmup": warmup,
            "side": request["side"],
            "success": success,
            "status": result["status"],
            "status_detail": status_detail,
            "elapsed_ms": elapsed_ms,
            "num_waypoints": num_waypoints,
            **request["sample_meta"],
            **{
                k: result.get(k)
                for k in (
                    "curobo_solve_time_ms",
                    "curobo_total_time_ms",
                    "curobo_ik_time_ms",
                    "curobo_graph_time_ms",
                    "curobo_trajopt_time_ms",
                    "curobo_finetune_time_ms",
                    "curobo_attempts",
                    "curobo_trajopt_attempts",
                    "curobo_used_graph",
                )
            },
        }
        rows.append(row)

        if not warmup:
            all_times_ms.append(elapsed_ms)
            if success:
                success_times_ms.append(elapsed_ms)
            else:
                key = status_detail or result["status"]
                failure_reasons[key] = failure_reasons.get(key, 0) + 1

        if (idx + 1) % 10 == 0 or idx + 1 == len(samples):
            print(f"[Speed] {idx + 1}/{len(samples)} samples complete")

    measured_rows = [r for r in rows if not r["warmup"]]
    success_count = sum(int(r["success"]) for r in measured_rows)
    summary = {
        "num_samples": args.speed_samples,
        "num_warmup_samples": args.speed_warmup_samples,
        "success_count": success_count,
        "failure_count": args.speed_samples - success_count,
        "success_rate": float(success_count / max(1, args.speed_samples)),
        "all_samples_ms": _stats_ms(all_times_ms),
        "successful_samples_ms": _stats_ms(success_times_ms),
        "failure_reasons": failure_reasons,
        "curobo_successful_samples_ms": {
            key: _stats_ms(
                [
                    float(r[key])
                    for r in measured_rows
                    if r["success"] and key in r and r[key] is not None
                ]
            )
            for key in (
                "curobo_solve_time_ms",
                "curobo_total_time_ms",
                "curobo_ik_time_ms",
                "curobo_graph_time_ms",
                "curobo_trajopt_time_ms",
                "curobo_finetune_time_ms",
            )
        },
    }

    csv_path = output_dir / "speed_samples.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[Speed] Wrote per-sample CSV to {csv_path}")
    print(
        "[Speed] Summary: "
        f"success_rate={100.0 * summary['success_rate']:.1f}% "
        f"p50={summary['all_samples_ms']['p50_ms']:.1f} ms "
        f"p95={summary['all_samples_ms']['p95_ms']:.1f} ms"
    )
    return summary


def _reachability_grid_requests(
    args: argparse.Namespace,
    left_quat: np.ndarray,
    right_quat: np.ndarray,
) -> list[dict[str, Any]]:
    xs = _workspace_values(args.x_min, args.x_max, args.grid_x)
    ys_abs = _workspace_values(args.y_abs_min, args.y_abs_max, args.grid_y)
    zs = _workspace_values(args.z_min, args.z_max, args.grid_z)

    requests: list[dict[str, Any]] = []
    for z in zs:
        for x in xs:
            for y_abs in ys_abs:
                if args.side == "both":
                    request = _bimanual_target(float(x), float(y_abs), float(z), left_quat, right_quat)
                    request["sample_meta"] = {"x": float(x), "y_abs": float(y_abs), "z": float(z)}
                else:
                    sign = 1.0 if args.side == "left" else -1.0
                    pos = np.array([x, sign * y_abs, z], dtype=np.float64)
                    request = _single_arm_target(args.side, pos, left_quat, right_quat)
                    request["sample_meta"] = {"x": float(x), "y": float(pos[1]), "z": float(z)}
                requests.append(request)
    return requests


def _run_reachability_scan(
    args: argparse.Namespace,
    planner: YamMotionPlannerCurobo,
    start: StartState,
    left_quat: np.ndarray,
    right_quat: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    extra_rpy_deg_specs = _parse_extra_rpy_deg_specs(args.orientation_extra_rpy_deg)
    requests = _reachability_grid_requests(args, left_quat, right_quat)
    orientation_candidates = _orientation_candidate_specs(
        args.orientation,
        left_quat,
        right_quat,
        args.orientation_samples,
        extra_rpy_deg_specs,
    )
    rows: list[dict[str, Any]] = []
    success_positions: list[np.ndarray] = []
    per_z: dict[float, list[bool]] = {}

    print(
        f"[Reachability] Scanning {len(requests)} targets "
        f"({args.grid_x} x {args.grid_y} x {args.grid_z}) for side={args.side}..."
    )
    if args.orientation == "any_roll":
        print(
            "[Reachability] Using position-only approximation via roll sweep: "
            f"{len(orientation_candidates)} orientation candidates per target"
        )

    for idx, request in enumerate(requests):
        total_elapsed_ms = 0.0
        attempts_for_point = 0
        best_result = None
        best_candidate = None
        final_status_detail = None
        for candidate in orientation_candidates:
            attempts_for_point += 1
            req = dict(request)
            req["target_left_quat_xyzw"] = candidate["left_quat_xyzw"]
            req["target_right_quat_xyzw"] = candidate["right_quat_xyzw"]
            result, elapsed_ms = _plan_once(planner, start, req)
            total_elapsed_ms += elapsed_ms
            final_status_detail = result.get("status_detail")
            if result["status"] == "Success":
                best_result = result
                best_candidate = candidate
                break
            if best_result is None:
                best_result = result

        assert best_result is not None
        success = best_result["status"] == "Success"
        status_detail = best_result.get("status_detail") or final_status_detail
        z_key = float(request["sample_meta"]["z"])
        per_z.setdefault(z_key, []).append(success)

        if success:
            if args.side == "both":
                pos = np.array(
                    [
                        request["sample_meta"]["x"],
                        request["sample_meta"]["y_abs"],
                        request["sample_meta"]["z"],
                    ],
                    dtype=np.float64,
                )
            else:
                pos = np.array(
                    [
                        request["sample_meta"]["x"],
                        request["sample_meta"]["y"],
                        request["sample_meta"]["z"],
                    ],
                    dtype=np.float64,
                )
            success_positions.append(pos)

        row = {
            "sample_index": idx,
            "side": request["side"],
            "success": success,
            "status": best_result["status"],
            "status_detail": status_detail,
            "elapsed_ms": total_elapsed_ms,
            "num_waypoints": int(best_result["position"].shape[0]) if "position" in best_result else 0,
            "orientation_attempts": attempts_for_point,
            "orientation_candidate_count": len(orientation_candidates),
            "best_roll_rad": None
            if best_candidate is None or best_candidate.get("roll_rad") is None
            else float(best_candidate["roll_rad"]),
            "best_roll_deg": None
            if best_candidate is None or best_candidate.get("roll_rad") is None
            else float(np.rad2deg(best_candidate["roll_rad"])),
            "best_orientation_source": None if best_candidate is None else best_candidate.get("source"),
            **request["sample_meta"],
        }
        if best_candidate is not None and "rpy_deg_xyz" in best_candidate:
            row["best_rpy_deg_xyz"] = np.asarray(best_candidate["rpy_deg_xyz"], dtype=np.float64).tolist()
        if best_candidate is not None:
            if request["side"] in ("left", "both"):
                row["best_left_quat_xyzw"] = np.asarray(
                    best_candidate["left_quat_xyzw"], dtype=np.float64
                ).tolist()
            if request["side"] in ("right", "both"):
                row["best_right_quat_xyzw"] = np.asarray(
                    best_candidate["right_quat_xyzw"], dtype=np.float64
                ).tolist()
        rows.append(row)
        if (idx + 1) % 25 == 0 or idx + 1 == len(requests):
            print(f"[Reachability] {idx + 1}/{len(requests)} samples complete")

    total = len(rows)
    success_count = sum(int(r["success"]) for r in rows)
    success_rate = float(success_count / max(1, total))
    bbox = None
    if success_positions:
        arr = np.stack(success_positions, axis=0)
        bbox = {
            "min": arr.min(axis=0),
            "max": arr.max(axis=0),
        }

    per_z_success = {
        f"{z:.4f}": float(np.mean(vals)) if vals else None for z, vals in sorted(per_z.items())
    }
    summary = {
        "num_samples": total,
        "success_count": success_count,
        "failure_count": total - success_count,
        "success_rate": success_rate,
        "latency_ms": _stats_ms([float(r["elapsed_ms"]) for r in rows]),
        "orientation_mode": args.orientation,
        "orientation_candidate_count": len(orientation_candidates),
        "orientation_extra_rpy_deg": [rpy.tolist() for rpy in extra_rpy_deg_specs],
        "orientation_attempts_per_point": _stats_ms(
            [float(r["orientation_attempts"]) for r in rows]
        ),
        "per_z_success_rate": per_z_success,
        "successful_region_bbox": bbox,
    }

    csv_path = output_dir / "reachability_samples.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[Reachability] Wrote per-sample CSV to {csv_path}")
    print(
        "[Reachability] Summary: "
        f"success_rate={100.0 * success_rate:.1f}% "
        f"({success_count}/{total})"
    )
    if bbox is not None:
        print(
            "[Reachability] Successful region bbox: "
            f"min={np.round(bbox['min'], 4)} max={np.round(bbox['max'], 4)}"
        )
    return summary


def main() -> None:
    args = _make_parser().parse_args()
    if args.speed_samples < 1:
        raise ValueError("--speed-samples must be >= 1")
    if min(args.grid_x, args.grid_y, args.grid_z) < 1:
        raise ValueError("--grid-x, --grid-y, and --grid-z must be >= 1")
    if args.orientation_samples < 1:
        raise ValueError("--orientation-samples must be >= 1")
    if not (args.x_min < args.x_max and args.y_abs_min < args.y_abs_max and args.z_min < args.z_max):
        raise ValueError("Workspace bounds must satisfy min < max")

    start = _get_start_state(args.start_state)
    kin = YamKinematics()
    pose_info = _print_header(args, start, kin)
    left_quat, right_quat = _orientation_targets(
        args.orientation,
        np.asarray(pose_info["start_left_pose"]["quat_xyzw"], dtype=np.float64),
        np.asarray(pose_info["start_right_pose"]["quat_xyzw"], dtype=np.float64),
    )

    planner = YamMotionPlannerCurobo(
        robot_cfg_path=args.robot_cfg_path,
        validate_with_mujoco=args.validate_with_mujoco,
        device=args.device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    run_dir = args.output_dir / f"{timestamp}_{args.mode}_{args.side}_{args.start_state}{tag_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "args": vars(args),
        "start_state": {
            "name": start.name,
            "left_joint_pos": start.left,
            "right_joint_pos": start.right,
        },
        "start_pose": pose_info,
        "orientation_targets": {
            "left_quat_xyzw": left_quat,
            "right_quat_xyzw": right_quat,
        },
        "planner": {
            "robot_cfg_path": planner._robot_cfg_path,
            "urdf_path": planner._urdf_path,
        },
    }

    if args.mode in ("speed", "both"):
        summary["speed"] = _run_speed_benchmark(
            args=args,
            planner=planner,
            start=start,
            left_quat=left_quat,
            right_quat=right_quat,
            output_dir=run_dir,
        )

    if args.mode in ("reachability", "both"):
        summary["reachability"] = _run_reachability_scan(
            args=args,
            planner=planner,
            start=start,
            left_quat=left_quat,
            right_quat=right_quat,
            output_dir=run_dir,
        )

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonify(summary), indent=2))
    print(f"[cuRobo Benchmark] Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
