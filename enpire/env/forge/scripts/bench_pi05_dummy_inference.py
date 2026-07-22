#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Pi05 YAM policy-server chunk latency with dummy inputs.

This sends dummy RGB images and zero proprioception directly to the Portal
`step(payload)` RPC exposed by `yam_policy_server_correct.py`.

The script writes one JSON record per timed inference to a log file, then reads
that log back to compute latency statistics in both milliseconds and outer-loop
action-step units at a specified control frequency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import portal

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_FILE = SCRIPT_DIR / "pi05_dummy_inference_bench.jsonl"
DEFAULT_FIGURE_FILE = SCRIPT_DIR / "pi05_dummy_inference_bench.png"


def _dummy_image(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Structured-but-cheap image so we are not benchmarking a degenerate empty object.
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return np.ascontiguousarray(image)


def _build_payload(height: int, width: int, text: str) -> dict[str, Any]:
    images = {
        "top_camera-images-rgb": _dummy_image(height, width, seed=1),
        "left_camera-images-rgb": _dummy_image(height, width, seed=2),
        "right_camera-images-rgb": _dummy_image(height, width, seed=3),
    }
    states = {
        "joint_pos_obs_left": np.zeros(6, dtype=np.float32),
        "gripper_pos_obs_left": np.zeros(1, dtype=np.float32),
        "joint_pos_obs_right": np.zeros(6, dtype=np.float32),
        "gripper_pos_obs_right": np.zeros(1, dtype=np.float32),
    }
    return {
        "images": images,
        "states": states,
        "actions": {},
        "text": text,
        "rl_info": None,
        "embodiment": "xdof",
        "is_demonstration": False,
        "metadata": {},
    }


def _infer_horizon(action_chunk: dict[str, Any]) -> int:
    for value in action_chunk.values():
        arr = np.asarray(value)
        if arr.ndim >= 1:
            return int(arr.shape[0])
    raise ValueError("Empty action chunk returned by server")


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _run_benchmark(
    *,
    server: str,
    height: int,
    width: int,
    text: str,
    warmup: int,
    runs: int,
    control_hz: float,
    log_file: Path,
) -> None:
    client = portal.Client(server)
    ok = client.health_check().result(timeout=10.0)
    if not ok:
        raise RuntimeError(f"Health check failed for {server}")

    payload = _build_payload(height, width, text)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("", encoding="utf-8")
    _write_jsonl(
        log_file,
        {
            "type": "meta",
            "server": server,
            "height": height,
            "width": width,
            "text": text,
            "warmup": warmup,
            "runs": runs,
            "control_hz": control_hz,
            "started_at_unix_s": time.time(),
        },
    )

    for idx in range(warmup):
        _ = client.step(payload).result()
        print(f"warmup[{idx}] done")

    for idx in range(runs):
        start = time.perf_counter()
        result = client.step(payload).result()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        horizon = _infer_horizon(result)
        _write_jsonl(
            log_file,
            {
                "type": "run",
                "run_idx": idx,
                "roundtrip_ms": elapsed_ms,
                "chunk_horizon": horizon,
                "logged_at_unix_s": time.time(),
                "control_hz": control_hz,
            },
        )
        print(
            f"run[{idx}] roundtrip_ms={elapsed_ms:.2f} "
            f"chunk_horizon={horizon} log={log_file}"
        )


def _load_runs_from_log(log_file: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] | None = None
    runs: list[dict[str, Any]] = []
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "meta":
                meta = record
            elif record.get("type") == "run":
                runs.append(record)
    if meta is None:
        raise ValueError(f"Missing meta record in log: {log_file}")
    if not runs:
        raise ValueError(f"No run records found in log: {log_file}")
    return meta, runs


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q99": float(np.quantile(values, 0.99)),
    }


def _plot_distribution(
    ax: plt.Axes, values: np.ndarray, title: str, xlabel: str, bins: int
) -> None:
    counts, edges = np.histogram(values, bins=bins)
    percentages = (counts / max(1, counts.sum())) * 100.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    ax.bar(centers, percentages, width=widths * 0.95, align="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Percentage")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.3)


def _make_figure(
    log_file: Path,
    ms_values: np.ndarray,
    step_values: np.ndarray,
    figure_file: Path,
    bins: int,
) -> None:
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    _plot_distribution(
        axes[0],
        ms_values,
        title="Inference Latency Distribution",
        xlabel="Latency (ms)",
        bins=bins,
    )
    _plot_distribution(
        axes[1],
        step_values,
        title="Inference Latency Distribution",
        xlabel="Latency (action steps @30Hz)",
        bins=bins,
    )
    fig.suptitle(f"Pi05 Dummy Benchmark from {log_file.name}")
    fig.savefig(figure_file, dpi=180)
    plt.close(fig)


def _compute_and_print_stats(log_file: Path, control_hz: float) -> None:
    meta, runs = _load_runs_from_log(log_file)
    ms_values = np.asarray([float(run["roundtrip_ms"]) for run in runs], dtype=np.float64)
    step_period_ms = 1000.0 / float(control_hz)
    step_values = ms_values / step_period_ms
    horizon_values = np.asarray([int(run["chunk_horizon"]) for run in runs], dtype=np.int64)

    ms_stats = _summary(ms_values)
    step_stats = _summary(step_values)
    horizon_mode = int(statistics.mode(horizon_values.tolist()))

    stats_record = {
        "type": "stats",
        "log_file": str(log_file),
        "num_runs": int(len(runs)),
        "control_hz": float(control_hz),
        "chunk_horizon_mode": horizon_mode,
        "ms": ms_stats,
        "action_steps": step_stats,
        "source_meta": meta,
    }
    stats_path = log_file.with_suffix(log_file.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats_record, indent=2), encoding="utf-8")

    print(f"summary log_file={log_file}")
    print(f"summary stats_file={stats_path}")
    print(f"summary num_runs={len(runs)}")
    print(f"summary chunk_horizon_mode={horizon_mode}")
    print(f"summary ms_mean={ms_stats['mean']:.6f}")
    print(f"summary ms_std={ms_stats['std']:.6f}")
    print(f"summary ms_median={ms_stats['median']:.6f}")
    print(f"summary ms_q01={ms_stats['q01']:.6f}")
    print(f"summary ms_q99={ms_stats['q99']:.6f}")
    print(f"summary steps_mean={step_stats['mean']:.6f}")
    print(f"summary steps_std={step_stats['std']:.6f}")
    print(f"summary steps_median={step_stats['median']:.6f}")
    print(f"summary steps_q01={step_stats['q01']:.6f}")
    print(f"summary steps_q99={step_stats['q99']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Pi05 YAM policy-server chunk latency with dummy inputs."
    )
    parser.add_argument("--server", default="localhost:8964", help="Portal server address")
    parser.add_argument("--height", type=int, default=480, help="Dummy image height")
    parser.add_argument("--width", type=int, default=640, help="Dummy image width")
    parser.add_argument("--text", default="functional grasp", help="Task text")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests")
    parser.add_argument("--runs", type=int, default=50, help="Timed requests")
    parser.add_argument(
        "--control-hz",
        type=float,
        default=30.0,
        help="Outer-loop control rate used to convert latency to action-step units",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="JSONL file used to log individual benchmark runs",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip benchmarking and only compute stats from an existing log file",
    )
    parser.add_argument(
        "--figure-file",
        default=str(DEFAULT_FIGURE_FILE),
        help="PNG file for the latency distribution figure",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of histogram bins used for the percentage bar plots",
    )
    args = parser.parse_args()

    log_file = Path(args.log_file).expanduser().resolve()
    figure_file = Path(args.figure_file).expanduser().resolve()
    if not args.stats_only:
        _run_benchmark(
            server=args.server,
            height=args.height,
            width=args.width,
            text=args.text,
            warmup=args.warmup,
            runs=args.runs,
            control_hz=args.control_hz,
            log_file=log_file,
        )
    _compute_and_print_stats(log_file, control_hz=args.control_hz)
    _, runs = _load_runs_from_log(log_file)
    ms_values = np.asarray([float(run["roundtrip_ms"]) for run in runs], dtype=np.float64)
    step_values = ms_values / (1000.0 / float(args.control_hz))
    _make_figure(
        log_file=log_file,
        ms_values=ms_values,
        step_values=step_values,
        figure_file=figure_file,
        bins=args.bins,
    )
    print(f"summary figure_file={figure_file}")


if __name__ == "__main__":
    main()
