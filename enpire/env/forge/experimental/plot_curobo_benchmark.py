# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize cuRobo benchmark outputs.

This script reads the output directory produced by
``python -m experimental.benchmark_curobo`` and generates PNG plots.

Examples:
    # Plot a specific run directory:
    uv run python -m experimental.plot_curobo_benchmark \
      artifacts/curobo_benchmark/20260326_123456_both_left_zero

    # Plot the latest run under the default root:
    uv run python -m experimental.plot_curobo_benchmark
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=Path("artifacts/curobo_benchmark"),
        type=Path,
        help="Benchmark run directory, or a root directory containing multiple run directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _resolve_run_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        if path.name == "summary.json":
            return path.parent
        if (path.parent / "summary.json").exists():
            return path.parent
        raise FileNotFoundError(
            f"Expected a benchmark run directory or a file inside one, got file: {path}"
        )
    if (path / "summary.json").exists():
        return path
    candidates = sorted(
        [p.parent for p in path.glob("**/summary.json")],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No benchmark run directories found under: {path}")
    return candidates[-1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _safe_title(summary: dict) -> str:
    args = summary.get("args", {})
    mode = args.get("mode", "?")
    side = args.get("side", "?")
    start = args.get("start_state", "?")
    orientation = args.get("orientation", "?")
    return f"cuRobo Benchmark — mode={mode}, side={side}, start={start}, orientation={orientation}"


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Wrote {path}")


def _plot_speed(speed_csv: Path, summary: dict, output_dir: Path, dpi: int) -> None:
    df = pd.read_csv(speed_csv)
    if df.empty:
        return

    measured = df[~df["warmup"].astype(bool)].copy()
    success = measured[measured["success"].astype(bool)]
    failure = measured[~measured["success"].astype(bool)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    warmup = df[df["warmup"].astype(bool)]
    if not warmup.empty:
        ax.scatter(warmup["sample_index"], warmup["elapsed_ms"], c="#999999", label="warmup", marker="x")
    if not success.empty:
        ax.scatter(success["sample_index"], success["elapsed_ms"], c="#2ca02c", label="success")
    if not failure.empty:
        ax.scatter(failure["sample_index"], failure["elapsed_ms"], c="#d62728", label="failure")
    ax.set_title("Latency by sample")
    ax.set_xlabel("sample index")
    ax.set_ylabel("elapsed ms")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[0, 1]
    if not measured.empty:
        ax.hist(
            [success["elapsed_ms"], failure["elapsed_ms"]],
            bins=min(20, max(5, len(measured))),
            stacked=True,
            color=["#2ca02c", "#d62728"],
            label=["success", "failure"],
        )
    ax.set_title("Latency histogram")
    ax.set_xlabel("elapsed ms")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1, 0]
    if not measured.empty:
        ordered = np.sort(measured["elapsed_ms"].to_numpy(dtype=float))
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, cdf, color="#1f77b4")
    ax.set_title("Latency CDF")
    ax.set_xlabel("elapsed ms")
    ax.set_ylabel("fraction <= x")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    speed_summary = summary.get("speed", {})
    all_stats = speed_summary.get("all_samples_ms", {})
    failure_reasons = speed_summary.get("failure_reasons", {})
    lines = [
        _safe_title(summary),
        "",
        f"samples:       {speed_summary.get('num_samples')}",
        f"success rate:  {100.0 * speed_summary.get('success_rate', 0.0):.1f}%",
        f"mean ms:       {all_stats.get('mean_ms')}",
        f"p50 ms:        {all_stats.get('p50_ms')}",
        f"p95 ms:        {all_stats.get('p95_ms')}",
        f"max ms:        {all_stats.get('max_ms')}",
        "",
        "failure reasons:",
    ]
    if failure_reasons:
        for k, v in failure_reasons.items():
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("  none")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace")

    _save(fig, output_dir / "speed_overview.png", dpi)


def _plot_reachability_heatmaps(
    reachability_csv: Path,
    summary: dict,
    output_dir: Path,
    dpi: int,
) -> None:
    df = pd.read_csv(reachability_csv)
    if df.empty:
        return

    y_col = "y_abs" if "y_abs" in df.columns else "y"
    z_values = sorted(df["z"].unique())
    n = len(z_values)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)

    cmap = matplotlib.colors.ListedColormap(["#d62728", "#2ca02c"])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    for idx, z in enumerate(z_values):
        ax = axes[idx // cols][idx % cols]
        sub = df[df["z"] == z].copy()
        xs = sorted(sub["x"].unique())
        ys = sorted(sub[y_col].unique())
        pivot = (
            sub.pivot_table(index=y_col, columns="x", values="success", aggfunc="first")
            .reindex(index=ys, columns=xs)
            .astype(float)
        )
        img = ax.imshow(
            pivot.to_numpy(),
            cmap=cmap,
            norm=norm,
            origin="lower",
            aspect="auto",
        )
        ax.set_title(f"z = {z:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel(y_col)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{x:.2f}" for x in xs], rotation=45)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels([f"{y:.2f}" for y in ys])
        success_rate = float(sub["success"].mean())
        ax.text(
            0.02,
            0.98,
            f"success={100.0 * success_rate:.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes.ravel().tolist(),
        shrink=0.85,
    )
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["fail", "success"])
    fig.suptitle(_safe_title(summary))
    _save(fig, output_dir / "reachability_heatmaps.png", dpi)


def _plot_reachability_scatter(
    reachability_csv: Path,
    summary: dict,
    output_dir: Path,
    dpi: int,
) -> None:
    df = pd.read_csv(reachability_csv)
    if df.empty:
        return

    y_col = "y_abs" if "y_abs" in df.columns else "y"
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    success = df[df["success"].astype(bool)]
    failure = df[~df["success"].astype(bool)]
    if not success.empty:
        ax.scatter(success["x"], success[y_col], success["z"], c="#2ca02c", label="success", s=30)
    if not failure.empty:
        ax.scatter(failure["x"], failure[y_col], failure["z"], c="#d62728", label="failure", s=30)
    ax.set_xlabel("x")
    ax.set_ylabel(y_col)
    ax.set_zlabel("z")
    ax.set_title(_safe_title(summary))
    ax.legend(loc="best")
    _save(fig, output_dir / "reachability_scatter3d.png", dpi)


def main() -> None:
    args = _make_parser().parse_args()
    run_dir = _resolve_run_dir(args.path)
    summary = _load_json(run_dir / "summary.json")
    output_dir = (args.output_dir or (run_dir / "plots")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Plot] Using run directory: {run_dir}")

    speed_csv = run_dir / "speed_samples.csv"
    if speed_csv.exists():
        _plot_speed(speed_csv, summary, output_dir, args.dpi)

    reachability_csv = run_dir / "reachability_samples.csv"
    if reachability_csv.exists():
        _plot_reachability_heatmaps(reachability_csv, summary, output_dir, args.dpi)
        _plot_reachability_scatter(reachability_csv, summary, output_dir, args.dpi)


if __name__ == "__main__":
    main()
