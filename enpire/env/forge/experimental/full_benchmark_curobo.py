"""Run a fuller cuRobo speed benchmark and generate summary plots.

This script runs two experiments:

1. Threshold-vs-time:
   Sweep several (position, rotation, cspace) convergence thresholds and measure
   planning latency / success for a fixed side (default: both arms).

2. Single-vs-dual timing:
   Compare planning latency for left-arm, right-arm, and dual-arm planning under
   one threshold setting.

Outputs are written to ``artifacts/curobo_full_benchmark/<run_dir>``:
    - threshold_sweep_samples.csv
    - threshold_sweep_summary.csv
    - arm_comparison_samples.csv
    - arm_comparison_summary.csv
    - summary.json
    - plots/*.png
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-state", choices=["zero", "home"], default="zero")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--robot-cfg-path", type=str, default=None)
    parser.add_argument("--validate-with-mujoco", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-finetune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-samples", type=int, default=30)
    parser.add_argument("--speed-warmup-samples", type=int, default=3)
    parser.add_argument("--threshold-side", choices=["left", "right", "both"], default="both")
    parser.add_argument(
        "--threshold-presets",
        type=str,
        default="strict:0.005,0.05,0.05;medium:0.01,0.10,0.075;default:0.02,0.15,0.10;loose:0.03,0.20,0.15",
        help=(
            "Semicolon-separated presets in the form "
            "'label:position_threshold,rotation_threshold,cspace_threshold'."
        ),
    )
    parser.add_argument(
        "--comparison-threshold",
        type=str,
        default="0.02,0.15,0.10",
        help="Threshold triple used for left/right/both comparison: pos,rot,cspace.",
    )
    parser.add_argument("--x-min", type=float, default=0.35)
    parser.add_argument("--x-max", type=float, default=0.80)
    parser.add_argument("--y-abs-min", type=float, default=0.05)
    parser.add_argument("--y-abs-max", type=float, default=0.45)
    parser.add_argument("--z-min", type=float, default=0.85)
    parser.add_argument("--z-max", type=float, default=1.30)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/curobo_full_benchmark"))
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _get_start_state(name: str) -> StartState:
    if name == "zero":
        return StartState("zero", ZERO_LEFT.copy(), ZERO_RIGHT.copy())
    if name == "home":
        return StartState("home", HOME_LEFT.copy(), HOME_RIGHT.copy())
    raise ValueError(f"Unknown start state: {name}")


def _parse_threshold_presets(text: str) -> list[ThresholdPreset]:
    presets: list[ThresholdPreset] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid threshold preset '{chunk}'")
        label, vals = chunk.split(":", 1)
        parts = [float(v.strip()) for v in vals.split(",") if v.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid threshold preset '{chunk}', expected 3 values")
        presets.append(ThresholdPreset(label.strip(), parts[0], parts[1], parts[2]))
    if not presets:
        raise ValueError("No valid threshold presets provided")
    return presets


def _parse_threshold_triple(text: str) -> ThresholdPreset:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) != 3:
        raise ValueError("--comparison-threshold must have 3 comma-separated values")
    return ThresholdPreset("comparison", vals[0], vals[1], vals[2])


def _stats_ms(values_ms: list[float]) -> dict[str, Any]:
    if not values_ms:
        return {
            "count": 0,
            "mean_ms": None,
            "std_ms": None,
            "p50_ms": None,
            "p95_ms": None,
        }
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.quantile(arr, 0.50)),
        "p95_ms": float(np.quantile(arr, 0.95)),
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


def _orientation_targets(start: StartState) -> tuple[np.ndarray, np.ndarray]:
    kin = YamKinematics()
    _, left_quat, _, right_quat = kin.forward_kinematics(start.left, start.right)
    return np.asarray(left_quat, dtype=np.float64), np.asarray(right_quat, dtype=np.float64)


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


def _sample_requests(
    side: str,
    n_measured: int,
    n_warmup: int,
    rng: np.random.Generator,
    left_quat: np.ndarray,
    right_quat: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    reqs = []
    for _ in range(n_measured + n_warmup):
        reqs.append(
            _build_request(
                side=side,
                x=float(rng.uniform(args.x_min, args.x_max)),
                y_abs=float(rng.uniform(args.y_abs_min, args.y_abs_max)),
                z=float(rng.uniform(args.z_min, args.z_max)),
                left_quat=left_quat,
                right_quat=right_quat,
            )
        )
    return reqs


def _plan_once(planner: YamMotionPlannerCurobo, start: StartState, request: dict[str, Any]) -> tuple[dict[str, Any], float]:
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


def _make_planner(args: argparse.Namespace, preset: ThresholdPreset) -> YamMotionPlannerCurobo:
    return YamMotionPlannerCurobo(
        robot_cfg_path=args.robot_cfg_path,
        validate_with_mujoco=args.validate_with_mujoco,
        device=args.device,
        enable_finetune_trajopt=args.enable_finetune,
        position_threshold=preset.position_threshold,
        rotation_threshold=preset.rotation_threshold,
        cspace_threshold=preset.cspace_threshold,
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [r for r in rows if r["success"]]
    return {
        "num_samples": len(rows),
        "success_count": len(success_rows),
        "failure_count": len(rows) - len(success_rows),
        "success_rate": float(len(success_rows) / max(1, len(rows))),
        "elapsed_ms": _stats_ms([float(r["elapsed_ms"]) for r in rows]),
        "successful_elapsed_ms": _stats_ms([float(r["elapsed_ms"]) for r in success_rows]),
        "ik_ms": _stats_ms([float(r["curobo_ik_time_ms"]) for r in success_rows if r["curobo_ik_time_ms"] is not None]),
        "trajopt_ms": _stats_ms(
            [float(r["curobo_trajopt_time_ms"]) for r in success_rows if r["curobo_trajopt_time_ms"] is not None]
        ),
        "finetune_ms": _stats_ms(
            [float(r["curobo_finetune_time_ms"]) for r in success_rows if r["curobo_finetune_time_ms"] is not None]
        ),
    }


def _run_speed_case(
    case_name: str,
    side: str,
    planner: YamMotionPlannerCurobo,
    start: StartState,
    requests: list[dict[str, Any]],
    n_warmup: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, request in enumerate(requests):
        result, elapsed_ms = _plan_once(planner, start, request)
        warmup = idx < n_warmup
        row = {
            "case": case_name,
            "side": side,
            "sample_index": idx,
            "warmup": warmup,
            "success": result["status"] == "Success",
            "status": result["status"],
            "status_detail": result.get("status_detail"),
            "elapsed_ms": elapsed_ms,
            **request["sample_meta"],
            **{
                k: result.get(k)
                for k in (
                    "curobo_total_time_ms",
                    "curobo_solve_time_ms",
                    "curobo_ik_time_ms",
                    "curobo_graph_time_ms",
                    "curobo_trajopt_time_ms",
                    "curobo_finetune_time_ms",
                )
            },
        }
        rows.append(row)
    measured_rows = [r for r in rows if not r["warmup"]]
    return rows, _summarize_rows(measured_rows)


def _plot_threshold_sweep(summary_rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    def _v(key: str) -> list[float]:
        return [float("nan") if r[key] is None else float(r[key]) for r in summary_rows]

    labels = [r["label"] for r in summary_rows]
    x = np.arange(len(labels))
    mean_total = _v("elapsed_mean_ms")
    std_total = _v("elapsed_std_ms")
    success_rate = [100.0 * r["success_rate"] for r in summary_rows]
    ik_mean = _v("ik_mean_ms")
    trajopt_mean = _v("trajopt_mean_ms")
    finetune_mean = _v("finetune_mean_ms")

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    ax = axes[0]
    bars = ax.bar(x, mean_total, yerr=std_total, capsize=5, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean elapsed ms")
    ax.set_title("Threshold vs planning latency")
    ax.grid(True, axis="y", alpha=0.3)
    for i, (bar, row) in enumerate(zip(bars, summary_rows)):
        txt = (
            f"p={row['position_threshold']:.3f}\n"
            f"r={row['rotation_threshold']:.3f}\n"
            f"c={row['cspace_threshold']:.3f}"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + (std_total[i] if not np.isnan(std_total[i]) else 0.0) + 5.0,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2 = ax.twinx()
    ax2.plot(x, success_rate, color="#f58518", marker="o")
    ax2.set_ylabel("success rate (%)")
    ax2.set_ylim(0, 100)

    ax = axes[1]
    ax.bar(x, ik_mean, label="IK", color="#54a24b")
    ax.bar(x, trajopt_mean, bottom=ik_mean, label="trajopt", color="#e45756")
    ax.bar(
        x,
        finetune_mean,
        bottom=np.asarray(ik_mean) + np.asarray(trajopt_mean),
        label="finetune",
        color="#b279a2",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean successful cuRobo time ms")
    ax.set_title("Successful-plan time breakdown by threshold preset")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "threshold_vs_time.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Wrote {path}")


def _plot_arm_comparison(summary_rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    def _v(key: str) -> list[float]:
        return [float("nan") if r[key] is None else float(r[key]) for r in summary_rows]

    labels = [r["side"] for r in summary_rows]
    x = np.arange(len(labels))
    mean_total = _v("elapsed_mean_ms")
    std_total = _v("elapsed_std_ms")
    success_rate = [100.0 * r["success_rate"] for r in summary_rows]
    ik_mean = _v("ik_mean_ms")
    trajopt_mean = _v("trajopt_mean_ms")
    finetune_mean = _v("finetune_mean_ms")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ax = axes[0]
    ax.bar(x, mean_total, yerr=std_total, capsize=5, color=["#72b7b2", "#54a24b", "#e45756"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean elapsed ms")
    ax.set_title("Single-arm vs dual-arm planning latency")
    ax.grid(True, axis="y", alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, success_rate, color="#4c78a8", marker="o")
    ax2.set_ylabel("success rate (%)")
    ax2.set_ylim(0, 100)

    ax = axes[1]
    ax.bar(x, ik_mean, label="IK", color="#54a24b")
    ax.bar(x, trajopt_mean, bottom=ik_mean, label="trajopt", color="#e45756")
    ax.bar(
        x,
        finetune_mean,
        bottom=np.asarray(ik_mean) + np.asarray(trajopt_mean),
        label="finetune",
        color="#b279a2",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean successful cuRobo time ms")
    ax.set_title("Successful-plan time breakdown")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "single_vs_dual_time.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Wrote {path}")


def main() -> None:
    args = _make_parser().parse_args()
    if args.speed_samples < 1 or args.speed_warmup_samples < 0:
        raise ValueError("Invalid speed sample counts")

    threshold_presets = _parse_threshold_presets(args.threshold_presets)
    comparison_threshold = _parse_threshold_triple(args.comparison_threshold)
    start = _get_start_state(args.start_state)
    left_quat, right_quat = _orientation_targets(start)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    run_dir = args.output_dir / f"{timestamp}_full{tag_suffix}"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Pre-sample requests for fairness across settings.
    rng_threshold = np.random.default_rng(args.seed)
    threshold_requests = _sample_requests(
        side=args.threshold_side,
        n_measured=args.speed_samples,
        n_warmup=args.speed_warmup_samples,
        rng=rng_threshold,
        left_quat=left_quat,
        right_quat=right_quat,
        args=args,
    )
    rng_by_side = {
        side: _sample_requests(
            side=side,
            n_measured=args.speed_samples,
            n_warmup=args.speed_warmup_samples,
            rng=np.random.default_rng(args.seed),
            left_quat=left_quat,
            right_quat=right_quat,
            args=args,
        )
        for side in ("left", "right", "both")
    }

    threshold_sample_rows: list[dict[str, Any]] = []
    threshold_summary_rows: list[dict[str, Any]] = []
    print(f"[FullBenchmark] Threshold sweep on side={args.threshold_side}")
    for preset in threshold_presets:
        planner = _make_planner(args, preset)
        print(
            f"  - {preset.label}: pos={preset.position_threshold:.3f} m, "
            f"rot={preset.rotation_threshold:.3f} rad, cspace={preset.cspace_threshold:.3f} rad"
        )
        rows, summary = _run_speed_case(
            preset.label, args.threshold_side, planner, start, threshold_requests, args.speed_warmup_samples
        )
        measured = [r for r in rows if not r["warmup"]]
        for row in measured:
            row["position_threshold"] = preset.position_threshold
            row["rotation_threshold"] = preset.rotation_threshold
            row["cspace_threshold"] = preset.cspace_threshold
            threshold_sample_rows.append(row)
        threshold_summary_rows.append(
            {
                "label": preset.label,
                "side": args.threshold_side,
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

    arm_sample_rows: list[dict[str, Any]] = []
    arm_summary_rows: list[dict[str, Any]] = []
    print("[FullBenchmark] Single-vs-dual timing comparison")
    for side in ("left", "right", "both"):
        planner = _make_planner(args, comparison_threshold)
        rows, summary = _run_speed_case(side, side, planner, start, rng_by_side[side], args.speed_warmup_samples)
        measured = [r for r in rows if not r["warmup"]]
        for row in measured:
            row["position_threshold"] = comparison_threshold.position_threshold
            row["rotation_threshold"] = comparison_threshold.rotation_threshold
            row["cspace_threshold"] = comparison_threshold.cspace_threshold
            arm_sample_rows.append(row)
        arm_summary_rows.append(
            {
                "side": side,
                "position_threshold": comparison_threshold.position_threshold,
                "rotation_threshold": comparison_threshold.rotation_threshold,
                "cspace_threshold": comparison_threshold.cspace_threshold,
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

    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"[FullBenchmark] Wrote {path}")

    _write_csv(run_dir / "threshold_sweep_samples.csv", threshold_sample_rows)
    _write_csv(run_dir / "threshold_sweep_summary.csv", threshold_summary_rows)
    _write_csv(run_dir / "arm_comparison_samples.csv", arm_sample_rows)
    _write_csv(run_dir / "arm_comparison_summary.csv", arm_summary_rows)

    _plot_threshold_sweep(threshold_summary_rows, plots_dir, args.dpi)
    _plot_arm_comparison(arm_summary_rows, plots_dir, args.dpi)

    summary = {
        "timestamp": timestamp,
        "args": vars(args),
        "start_state": {"name": start.name, "left_joint_pos": start.left, "right_joint_pos": start.right},
        "orientation_targets": {"left_quat_xyzw": left_quat, "right_quat_xyzw": right_quat},
        "threshold_sweep_summary": threshold_summary_rows,
        "arm_comparison_summary": arm_summary_rows,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(_jsonify(summary), indent=2))
    print(f"[FullBenchmark] Wrote {summary_path}")


if __name__ == "__main__":
    main()
