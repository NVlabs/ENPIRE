# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark RRT-Connect vs cuRobo across threshold presets.

This compares end-to-end planning time for:
- ``experimental.motion_planner.YamMotionPlanner`` (RRT-Connect)
- ``experimental.motion_planner_curobo.YamMotionPlannerCurobo`` (cuRobo)

For cuRobo, each preset sets:
- position_threshold
- rotation_threshold
- cspace_threshold

For RRT, only the preset's position threshold is used, mapped to
``ik_error_threshold`` for IK acceptance.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enpire.env.forge.experimental.motion_planner import YamMotionPlanner
from enpire.env.forge.experimental.motion_planner_curobo import YamMotionPlannerCurobo
from enpire.env.forge.robot.yam.kinematics import YamKinematics


HOME_LEFT = np.array([-0.3, 1.35, 1.6, -0.8, 0.3, -0.25], dtype=np.float64)
HOME_RIGHT = np.array([0.3, 1.35, 1.6, -0.8, -0.3, 0.25], dtype=np.float64)
ZERO_LEFT = np.zeros(6, dtype=np.float64)
ZERO_RIGHT = np.zeros(6, dtype=np.float64)


@dataclass(frozen=True)
class StartState:
    name: str
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class ThresholdPreset:
    label: str
    position_threshold: float
    rotation_threshold: float
    cspace_threshold: float


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", choices=["left", "right", "both"], default="both")
    p.add_argument("--start-state", choices=["zero", "home"], default="zero")
    p.add_argument("--speed-samples", type=int, default=30)
    p.add_argument("--speed-warmup-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--robot-cfg-path", type=str, default=None)
    p.add_argument("--validate-with-mujoco", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--enable-finetune", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--threshold-presets",
        type=str,
        default="strict:0.005,0.05,0.05;default:0.02,0.15,0.10;loose:0.03,0.20,0.15",
    )
    p.add_argument("--x-min", type=float, default=0.35)
    p.add_argument("--x-max", type=float, default=0.80)
    p.add_argument("--y-abs-min", type=float, default=0.05)
    p.add_argument("--y-abs-max", type=float, default=0.45)
    p.add_argument("--z-min", type=float, default=0.85)
    p.add_argument("--z-max", type=float, default=1.30)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/rrt_vs_curobo_benchmark"))
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--dpi", type=int, default=180)
    return p


def _get_start_state(name: str) -> StartState:
    if name == "zero":
        return StartState("zero", ZERO_LEFT.copy(), ZERO_RIGHT.copy())
    if name == "home":
        return StartState("home", HOME_LEFT.copy(), HOME_RIGHT.copy())
    raise ValueError(name)


def _parse_threshold_presets(text: str) -> list[ThresholdPreset]:
    out: list[ThresholdPreset] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        label, vals = chunk.split(":", 1)
        parts = [float(v.strip()) for v in vals.split(",") if v.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid threshold preset: {chunk}")
        out.append(ThresholdPreset(label.strip(), parts[0], parts[1], parts[2]))
    if not out:
        raise ValueError("No threshold presets")
    return out


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _stats_ms(values_ms: list[float]) -> dict[str, Any]:
    if not values_ms:
        return {"count": 0, "mean_ms": None, "std_ms": None, "p50_ms": None, "p95_ms": None}
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.quantile(arr, 0.50)),
        "p95_ms": float(np.quantile(arr, 0.95)),
    }


def _orientation_targets(start: StartState) -> tuple[np.ndarray, np.ndarray]:
    kin = YamKinematics()
    _, lq, _, rq = kin.forward_kinematics(start.left, start.right)
    return np.asarray(lq, dtype=np.float64), np.asarray(rq, dtype=np.float64)


def _build_request(side: str, x: float, y_abs: float, z: float, left_quat: np.ndarray, right_quat: np.ndarray) -> dict[str, Any]:
    if side == "both":
        return {
            "side": "both",
            "target_left_pos": np.array([x, +y_abs, z], dtype=np.float64),
            "target_left_quat_xyzw": left_quat.copy(),
            "target_right_pos": np.array([x, -y_abs, z], dtype=np.float64),
            "target_right_quat_xyzw": right_quat.copy(),
            "sample_meta": {"x": x, "y_abs": y_abs, "z": z},
        }
    sign = 1.0 if side == "left" else -1.0
    pos = np.array([x, sign * y_abs, z], dtype=np.float64)
    return {
        "side": side,
        "target_left_pos": pos if side == "left" else None,
        "target_left_quat_xyzw": left_quat.copy() if side == "left" else None,
        "target_right_pos": pos if side == "right" else None,
        "target_right_quat_xyzw": right_quat.copy() if side == "right" else None,
        "sample_meta": {"x": x, "y": float(pos[1]), "z": z},
    }


def _sample_requests(args: argparse.Namespace, left_quat: np.ndarray, right_quat: np.ndarray) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    reqs = []
    for _ in range(args.speed_samples + args.speed_warmup_samples):
        reqs.append(
            _build_request(
                args.side,
                float(rng.uniform(args.x_min, args.x_max)),
                float(rng.uniform(args.y_abs_min, args.y_abs_max)),
                float(rng.uniform(args.z_min, args.z_max)),
                left_quat,
                right_quat,
            )
        )
    return reqs


def _plan_once(planner: Any, planner_name: str, start: StartState, request: dict[str, Any], preset: ThresholdPreset) -> tuple[dict[str, Any], float]:
    t0 = perf_counter()
    kwargs = dict(
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
    if planner_name == "rrtconnect":
        kwargs["ik_error_threshold"] = preset.position_threshold
    result = planner.plan_to_pose(**kwargs)
    elapsed_ms = 1000.0 * (perf_counter() - t0)
    return result, elapsed_ms


def _make_curobo(args: argparse.Namespace, preset: ThresholdPreset) -> YamMotionPlannerCurobo:
    return YamMotionPlannerCurobo(
        robot_cfg_path=args.robot_cfg_path,
        validate_with_mujoco=args.validate_with_mujoco,
        device=args.device,
        enable_finetune_trajopt=args.enable_finetune,
        position_threshold=preset.position_threshold,
        rotation_threshold=preset.rotation_threshold,
        cspace_threshold=preset.cspace_threshold,
    )


def _make_rrt() -> YamMotionPlanner:
    return YamMotionPlanner()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    succ = [r for r in rows if r["success"]]
    return {
        "num_samples": len(rows),
        "success_rate": float(len(succ) / max(1, len(rows))),
        "elapsed_ms": _stats_ms([float(r["elapsed_ms"]) for r in rows]),
        "successful_elapsed_ms": _stats_ms([float(r["elapsed_ms"]) for r in succ]),
        "ik_ms": _stats_ms([float(r["curobo_ik_time_ms"]) for r in succ if r.get("curobo_ik_time_ms") is not None]),
        "trajopt_ms": _stats_ms([float(r["curobo_trajopt_time_ms"]) for r in succ if r.get("curobo_trajopt_time_ms") is not None]),
        "finetune_ms": _stats_ms([float(r["curobo_finetune_time_ms"]) for r in succ if r.get("curobo_finetune_time_ms") is not None]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_grouped(summary_rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    presets = [r["label"] for r in summary_rows if r["planner"] == "curobo"]
    planners = ["rrtconnect", "curobo"]
    x = np.arange(len(presets))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    ax = axes[0]
    for i, planner in enumerate(planners):
        rows = [r for r in summary_rows if r["planner"] == planner]
        means = [float("nan") if r["elapsed_mean_ms"] is None else float(r["elapsed_mean_ms"]) for r in rows]
        stds = [float("nan") if r["elapsed_std_ms"] is None else float(r["elapsed_std_ms"]) for r in rows]
        bars = ax.bar(x + (i - 0.5) * width, means, width=width, yerr=stds, capsize=4, label=planner)
        for j, (bar, row) in enumerate(zip(bars, rows)):
            txt = (
                f"p={row['position_threshold']:.3f}\n"
                f"r={row['rotation_threshold']:.3f}\n"
                f"c={row['cspace_threshold']:.3f}"
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                (0.0 if np.isnan(means[j]) else means[j]) + (0.0 if np.isnan(stds[j]) else stds[j]) + 5.0,
                txt,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(presets)
    ax.set_ylabel("mean elapsed ms")
    ax.set_title("RRT vs cuRobo planning time across threshold presets")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    rows = [r for r in summary_rows if r["planner"] == "curobo"]
    ik = [float("nan") if r["ik_mean_ms"] is None else float(r["ik_mean_ms"]) for r in rows]
    traj = [float("nan") if r["trajopt_mean_ms"] is None else float(r["trajopt_mean_ms"]) for r in rows]
    fine = [float("nan") if r["finetune_mean_ms"] is None else float(r["finetune_mean_ms"]) for r in rows]
    ax.bar(x, ik, label="IK", color="#54a24b")
    ax.bar(x, traj, bottom=ik, label="trajopt", color="#e45756")
    ax.bar(x, fine, bottom=np.asarray(ik) + np.asarray(traj), label="finetune", color="#b279a2")
    ax.set_xticks(x)
    ax.set_xticklabels(presets)
    ax.set_ylabel("mean successful cuRobo time ms")
    ax.set_title("cuRobo internal time breakdown")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "rrt_vs_curobo_time.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Wrote {path}")


def main() -> None:
    args = _make_parser().parse_args()
    presets = _parse_threshold_presets(args.threshold_presets)
    start = _get_start_state(args.start_state)
    left_quat, right_quat = _orientation_targets(start)
    requests = _sample_requests(args, left_quat, right_quat)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    run_dir = args.output_dir / f"{timestamp}_{args.side}_{args.start_state}{tag_suffix}"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for preset in presets:
        print(
            f"[RRTvscuRobo] preset={preset.label} "
            f"(p={preset.position_threshold:.3f}, r={preset.rotation_threshold:.3f}, c={preset.cspace_threshold:.3f})"
        )
        for planner_name, planner in (
            ("rrtconnect", _make_rrt()),
            ("curobo", _make_curobo(args, preset)),
        ):
            rows: list[dict[str, Any]] = []
            for idx, request in enumerate(requests):
                result, elapsed_ms = _plan_once(planner, planner_name, start, request, preset)
                warmup = idx < args.speed_warmup_samples
                row = {
                    "planner": planner_name,
                    "label": preset.label,
                    "side": args.side,
                    "warmup": warmup,
                    "sample_index": idx,
                    "success": result["status"] == "Success",
                    "status": result["status"],
                    "status_detail": result.get("status_detail"),
                    "elapsed_ms": elapsed_ms,
                    "position_threshold": preset.position_threshold,
                    "rotation_threshold": preset.rotation_threshold,
                    "cspace_threshold": preset.cspace_threshold,
                    **request["sample_meta"],
                }
                if planner_name == "curobo":
                    for k in (
                        "curobo_total_time_ms",
                        "curobo_solve_time_ms",
                        "curobo_ik_time_ms",
                        "curobo_graph_time_ms",
                        "curobo_trajopt_time_ms",
                        "curobo_finetune_time_ms",
                    ):
                        row[k] = result.get(k)
                rows.append(row)
            measured = [r for r in rows if not r["warmup"]]
            all_rows.extend(measured)
            summary = _summarize(measured)
            summary_rows.append(
                {
                    "planner": planner_name,
                    "label": preset.label,
                    "side": args.side,
                    "position_threshold": preset.position_threshold,
                    "rotation_threshold": preset.rotation_threshold,
                    "cspace_threshold": preset.cspace_threshold,
                    "success_rate": summary["success_rate"],
                    "elapsed_mean_ms": summary["elapsed_ms"]["mean_ms"],
                    "elapsed_std_ms": summary["elapsed_ms"]["std_ms"],
                    "elapsed_p50_ms": summary["elapsed_ms"]["p50_ms"],
                    "elapsed_p95_ms": summary["elapsed_ms"]["p95_ms"],
                    "ik_mean_ms": summary["ik_ms"]["mean_ms"],
                    "trajopt_mean_ms": summary["trajopt_ms"]["mean_ms"],
                    "finetune_mean_ms": summary["finetune_ms"]["mean_ms"],
                }
            )

    _write_csv(run_dir / "samples.csv", all_rows)
    _write_csv(run_dir / "summary_rows.csv", summary_rows)
    _plot_grouped(summary_rows, plots_dir, args.dpi)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            _jsonify(
                {
                    "timestamp": timestamp,
                    "args": vars(args),
                    "start_state": {"name": start.name, "left_joint_pos": start.left, "right_joint_pos": start.right},
                    "summary_rows": summary_rows,
                }
            ),
            indent=2,
        )
    )
    print(f"[RRTvscuRobo] Wrote {summary_path}")


if __name__ == "__main__":
    main()
