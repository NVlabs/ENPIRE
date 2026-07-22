# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute rolling-window success rate for a gearraw data folder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ACTION_SOURCE_JSON = "action-source.json"
ACTION_SOURCE_NPY = "action-source.npy"
CHECKPOINT_METADATA = "_CHECKPOINT_METADATA"
PLD_CHECKPOINTS_DIRNAME = "pld_checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing episode subdirectories with metadata.json.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window size in episodes.",
    )
    parser.add_argument(
        "--success-event",
        default="success",
        help="metadata.json terminal_event value counted as success.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/sr_rolling_window/<data-dir-name>.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Write CSV only; skip PNG plot.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as f:
            data = json.load(f)
    except JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_action_sources(ep_dir: Path) -> list[str]:
    json_path = ep_dir / ACTION_SOURCE_JSON
    npy_path = ep_dir / ACTION_SOURCE_NPY

    if json_path.exists():
        try:
            with json_path.open() as f:
                values = json.load(f)
        except JSONDecodeError as exc:
            raise ValueError(f"{json_path}: invalid JSON: {exc}") from exc
    elif npy_path.exists():
        values = np.load(npy_path, allow_pickle=False).tolist()
    else:
        raise FileNotFoundError(f"missing {ACTION_SOURCE_JSON} and {ACTION_SOURCE_NPY}")

    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise ValueError(f"{ep_dir}: action-source must be a list or 1-D array")
    return [str(value) for value in values]


def is_pure_rl_episode(ep_dir: Path) -> bool:
    sources = load_action_sources(ep_dir)
    return bool(sources) and all(source == "rl" for source in sources)


def episode_dirs(data_dir: Path) -> list[Path]:
    data_dir = data_dir.expanduser().resolve()
    episodes: list[Path] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "metadata.json").exists():
            episodes.append(child)
            continue
        episodes.extend(
            ep_dir
            for ep_dir in sorted(child.iterdir())
            if ep_dir.is_dir() and (ep_dir / "metadata.json").exists()
        )
    return episodes


def parse_episode_wall_time(ep_dir: Path) -> datetime | None:
    try:
        return datetime.strptime(ep_dir.name, "%Y%m%dT%H%M%S%f")
    except ValueError:
        return None


def parse_run_start_time(run_dir: Path) -> datetime | None:
    """Parse a training-run dir name like `insert_pin_run_20260523_033936`."""
    _, _, ts = run_dir.name.rpartition("_run_")
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def parse_data_dir_start_time(data_dir: Path) -> datetime | None:
    """Parse a data-dir name like `20260523-0339` or `20260523-023521_auto_eval_hole_1`."""
    name = data_dir.name
    for length, fmt in ((15, "%Y%m%d-%H%M%S"), (13, "%Y%m%d-%H%M")):
        try:
            return datetime.strptime(name[:length], fmt)
        except ValueError:
            continue
    return None


def load_checkpoint_save_time(ckpt_dir: Path) -> datetime | None:
    """Use commit_timestamp_nsecs from _CHECKPOINT_METADATA, falling back to mtime."""
    metadata_path = ckpt_dir / CHECKPOINT_METADATA
    if metadata_path.exists():
        try:
            with metadata_path.open() as f:
                data = json.load(f)
        except (OSError, JSONDecodeError):
            data = None
        if isinstance(data, dict):
            nsecs = data.get("commit_timestamp_nsecs")
            if isinstance(nsecs, (int, float)):
                return datetime.fromtimestamp(nsecs / 1e9)
    try:
        return datetime.fromtimestamp(ckpt_dir.stat().st_mtime)
    except OSError:
        return None


def discover_checkpoints(
    pld_checkpoints_dir: Path,
    after: datetime | None,
) -> list[tuple[str, list[tuple[str, datetime]]]]:
    """Return [(run_name, [(checkpoint_name, save_time), ...]), ...] for runs after `after`."""
    if not pld_checkpoints_dir.is_dir():
        return []
    results: list[tuple[str, list[tuple[str, datetime]]]] = []
    for run_dir in sorted(pld_checkpoints_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_start = parse_run_start_time(run_dir)
        if after is not None and run_start is not None and run_start < after:
            continue
        checkpoints: list[tuple[str, datetime]] = []
        for ckpt_dir in sorted(run_dir.iterdir()):
            if not ckpt_dir.is_dir() or not ckpt_dir.name.startswith("checkpoint_"):
                continue
            save_time = load_checkpoint_save_time(ckpt_dir)
            if save_time is None:
                continue
            checkpoints.append((ckpt_dir.name, save_time))
        checkpoints.sort(key=lambda item: item[1])
        if checkpoints:
            results.append((run_dir.name, checkpoints))
    return results


def first_wall_time_from_rows(rows: list[dict[str, Any]]) -> datetime | None:
    for row in rows:
        wall = row.get("wall_time")
        if wall:
            try:
                return datetime.fromisoformat(str(wall))
            except ValueError:
                continue
    return None


def find_closest_episode_index(
    rows: list[dict[str, Any]], target_elapsed_s: float
) -> int | None:
    best_idx: int | None = None
    best_diff: float | None = None
    for row in rows:
        elapsed = row.get("wall_time_elapsed_s")
        if elapsed is None:
            continue
        diff = abs(float(elapsed) - target_elapsed_s)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_idx = int(row["episode_index"])
    return best_idx


def compute_rows(
    data_dir: Path,
    window: int,
    success_event: str,
) -> list[dict[str, Any]]:
    if window <= 0:
        raise ValueError("--window must be positive")
    if not data_dir.expanduser().is_dir():
        raise ValueError(f"not a directory: {data_dir}")

    recent: deque[int] = deque(maxlen=window)
    pure_rl_recent: deque[int] = deque(maxlen=window)
    cumulative_successes = 0
    pure_rl_cumulative_successes = 0
    pure_rl_total = 0
    first_wall_time: datetime | None = None
    rows: list[dict[str, Any]] = []

    for ep_dir in episode_dirs(data_dir):
        try:
            metadata = load_json(ep_dir / "metadata.json")
        except (OSError, ValueError) as exc:
            print(f"warning: skipping {ep_dir}: {exc}", file=sys.stderr)
            continue

        idx = len(rows) + 1
        wall_time = parse_episode_wall_time(ep_dir)
        if first_wall_time is None and wall_time is not None:
            first_wall_time = wall_time
        wall_time_elapsed_s = (
            (wall_time - first_wall_time).total_seconds()
            if wall_time is not None and first_wall_time is not None
            else None
        )
        terminal_event = str(metadata.get("terminal_event", "unknown"))
        success = int(terminal_event == success_event)
        try:
            is_pure_rl = is_pure_rl_episode(ep_dir)
        except (OSError, ValueError) as exc:
            print(f"warning: excluding {ep_dir} from pure-RL SR: {exc}", file=sys.stderr)
            is_pure_rl = False

        cumulative_successes += success
        recent.append(success)

        rolling_total = len(recent)
        rolling_successes = sum(recent)
        rolling_success_rate = (
            rolling_successes / rolling_total if rolling_total >= window else None
        )

        pure_rl_episode_index = None
        pure_rl_rolling_total = None
        pure_rl_rolling_successes = None
        pure_rl_rolling_success_rate = None
        pure_rl_cumulative_success_rate = None
        if is_pure_rl:
            pure_rl_total += 1
            pure_rl_episode_index = pure_rl_total
            pure_rl_cumulative_successes += success
            pure_rl_recent.append(success)
            pure_rl_rolling_total = len(pure_rl_recent)
            pure_rl_rolling_successes = sum(pure_rl_recent)
            pure_rl_rolling_success_rate = (
                pure_rl_rolling_successes / pure_rl_rolling_total
                if pure_rl_rolling_total >= window
                else None
            )
            pure_rl_cumulative_success_rate = pure_rl_cumulative_successes / pure_rl_total

        rows.append(
            {
                "episode_index": idx,
                "episode_name": str(ep_dir.relative_to(data_dir)),
                "wall_time": wall_time.isoformat(sep=" ") if wall_time is not None else "",
                "wall_time_elapsed_s": wall_time_elapsed_s,
                "terminal_event": terminal_event,
                "success": success,
                "is_pure_rl": int(is_pure_rl),
                "rolling_window": window,
                "rolling_total": rolling_total,
                "rolling_successes": rolling_successes,
                "rolling_success_rate": rolling_success_rate,
                "cumulative_successes": cumulative_successes,
                "cumulative_success_rate": cumulative_successes / idx,
                "pure_rl_episode_index": pure_rl_episode_index,
                "pure_rl_rolling_total": pure_rl_rolling_total,
                "pure_rl_rolling_successes": pure_rl_rolling_successes,
                "pure_rl_rolling_success_rate": pure_rl_rolling_success_rate,
                "pure_rl_cumulative_successes": pure_rl_cumulative_successes
                if is_pure_rl
                else None,
                "pure_rl_cumulative_success_rate": pure_rl_cumulative_success_rate,
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_index",
        "episode_name",
        "wall_time",
        "wall_time_elapsed_s",
        "terminal_event",
        "success",
        "is_pure_rl",
        "rolling_window",
        "rolling_total",
        "rolling_successes",
        "rolling_success_rate",
        "cumulative_successes",
        "cumulative_success_rate",
        "pure_rl_episode_index",
        "pure_rl_rolling_total",
        "pure_rl_rolling_successes",
        "pure_rl_rolling_success_rate",
        "pure_rl_cumulative_successes",
        "pure_rl_cumulative_success_rate",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _overlay_checkpoint_lines(
    axes: Any,
    rows: list[dict[str, Any]],
    checkpoints: list[tuple[str, list[tuple[str, datetime]]]],
    first_wall_time: datetime,
) -> bool:
    cmap = plt.get_cmap("tab10")
    drew_any = False
    for run_idx, (run_name, ckpts) in enumerate(checkpoints):
        color = cmap(run_idx % 10)
        for ckpt_idx, (_ckpt_name, save_time) in enumerate(ckpts):
            elapsed_s = (save_time - first_wall_time).total_seconds()
            label = f"checkpoint: {run_name}" if ckpt_idx == 0 else None
            ep_idx = find_closest_episode_index(rows, elapsed_s)
            if ep_idx is not None:
                axes[0].axvline(
                    ep_idx,
                    linestyle="--",
                    color=color,
                    alpha=0.55,
                    linewidth=1.0,
                    label=label,
                )
            axes[1].axvline(
                elapsed_s / 3600.0,
                linestyle="--",
                color=color,
                alpha=0.55,
                linewidth=1.0,
                label=label,
            )
            drew_any = True
    return drew_any


def write_plot(
    path: Path,
    data_dir: Path,
    rows: list[dict[str, Any]],
    window: int,
    checkpoints: list[tuple[str, list[tuple[str, datetime]]]] | None = None,
    first_wall_time: datetime | None = None,
) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    rolling_rows = [row for row in rows if row["rolling_success_rate"] is not None]
    x = [int(row["episode_index"]) for row in rolling_rows]
    wall_rows = [row for row in rows if row["wall_time_elapsed_s"] is not None]
    wall_rolling_rows = [
        row
        for row in wall_rows
        if row["rolling_success_rate"] is not None
    ]
    wall_x = [float(row["wall_time_elapsed_s"]) / 3600.0 for row in wall_rolling_rows]
    rolling = [float(row["rolling_success_rate"]) for row in rolling_rows]
    cumulative_x = [int(row["episode_index"]) for row in rows]
    cumulative = [float(row["cumulative_success_rate"]) for row in rows]
    pure_rl_rows = [row for row in rows if row["pure_rl_rolling_success_rate"] is not None]
    pure_rl_x = [int(row["episode_index"]) for row in pure_rl_rows]
    pure_rl_rolling = [float(row["pure_rl_rolling_success_rate"]) for row in pure_rl_rows]
    pure_rl_cumulative_rows = [
        row for row in rows if row["pure_rl_cumulative_success_rate"] is not None
    ]
    pure_rl_cumulative_x = [
        int(row["episode_index"]) for row in pure_rl_cumulative_rows
    ]
    pure_rl_cumulative = [
        float(row["pure_rl_cumulative_success_rate"]) for row in pure_rl_cumulative_rows
    ]
    successes_x = [
        int(row["episode_index"])
        for row in rolling_rows
        if int(row["success"])
    ]
    successes_y = [
        float(row["rolling_success_rate"])
        for row in rolling_rows
        if int(row["success"])
    ]
    wall_rolling = [float(row["rolling_success_rate"]) for row in wall_rolling_rows]
    wall_cumulative_x = [
        float(row["wall_time_elapsed_s"]) / 3600.0 for row in wall_rows
    ]
    wall_cumulative = [float(row["cumulative_success_rate"]) for row in wall_rows]
    pure_rl_wall_rows = [row for row in pure_rl_rows if row["wall_time_elapsed_s"] is not None]
    pure_rl_wall_x = [float(row["wall_time_elapsed_s"]) / 3600.0 for row in pure_rl_wall_rows]
    pure_rl_wall_rolling = [
        float(row["pure_rl_rolling_success_rate"]) for row in pure_rl_wall_rows
    ]
    pure_rl_cumulative_wall_rows = [
        row
        for row in wall_rows
        if row["pure_rl_cumulative_success_rate"] is not None
    ]
    pure_rl_cumulative_wall_x = [
        float(row["wall_time_elapsed_s"]) / 3600.0
        for row in pure_rl_cumulative_wall_rows
    ]
    pure_rl_wall_cumulative = [
        float(row["pure_rl_cumulative_success_rate"])
        for row in pure_rl_cumulative_wall_rows
    ]
    wall_successes_x = [
        float(row["wall_time_elapsed_s"]) / 3600.0
        for row in wall_rolling_rows
        if int(row["success"])
    ]
    wall_successes_y = [
        float(row["rolling_success_rate"])
        for row in wall_rolling_rows
        if int(row["success"])
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"Rolling Success Rate - {data_dir.name}")

    axes[0].plot(x, rolling, label=f"all rolling SR, window={window}", linewidth=2)
    axes[0].plot(
        pure_rl_x,
        pure_rl_rolling,
        label=f"pure RL rolling SR, window={window}",
        linewidth=2,
    )
    axes[0].plot(
        cumulative_x,
        cumulative,
        label="all cumulative SR",
        linewidth=1.5,
        alpha=0.8,
    )
    axes[0].plot(
        pure_rl_cumulative_x,
        pure_rl_cumulative,
        label="pure RL cumulative SR",
        linewidth=1.5,
        alpha=0.8,
    )
    axes[0].scatter(successes_x, successes_y, s=12, alpha=0.35, label="success episode")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(wall_x, wall_rolling, label=f"all rolling SR, window={window}", linewidth=2)
    axes[1].plot(
        pure_rl_wall_x,
        pure_rl_wall_rolling,
        label=f"pure RL rolling SR, window={window}",
        linewidth=2,
    )
    axes[1].plot(
        wall_cumulative_x,
        wall_cumulative,
        label="all cumulative SR",
        linewidth=1.5,
        alpha=0.8,
    )
    axes[1].plot(
        pure_rl_cumulative_wall_x,
        pure_rl_wall_cumulative,
        label="pure RL cumulative SR",
        linewidth=1.5,
        alpha=0.8,
    )
    axes[1].scatter(
        wall_successes_x,
        wall_successes_y,
        s=12,
        alpha=0.35,
        label="success episode",
    )
    axes[1].set_xlabel("wall time elapsed (hours)")
    axes[1].set_ylabel("success rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(True, alpha=0.25)

    if checkpoints and first_wall_time is not None:
        _overlay_checkpoint_lines(axes, rows, checkpoints, first_wall_time)

    axes[0].legend(loc="best", fontsize="small")
    axes[1].legend(loc="best", fontsize="small")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_rate(rate: float | None) -> str:
    return f"{rate:.1%}" if rate is not None else "n/a"


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else Path("outputs") / "sr_rolling_window" / data_dir.name
    )

    rows = compute_rows(data_dir, args.window, args.success_event)
    if not rows:
        raise SystemExit(f"No episodes with metadata.json found under {data_dir}")

    csv_path = output_dir / f"sr_rolling_window_{args.window}.csv"
    write_csv(csv_path, rows)

    first_wall_time = first_wall_time_from_rows(rows)
    data_dir_start = parse_data_dir_start_time(data_dir) or first_wall_time
    pld_checkpoints_dir = data_dir.parent / PLD_CHECKPOINTS_DIRNAME
    checkpoints = discover_checkpoints(pld_checkpoints_dir, data_dir_start)
    if checkpoints:
        total_ckpts = sum(len(ckpts) for _, ckpts in checkpoints)
        print(
            f"found {total_ckpts} checkpoint(s) across {len(checkpoints)} run(s) "
            f"under {pld_checkpoints_dir}"
        )

    if not args.no_plot:
        write_plot(
            output_dir / f"sr_rolling_window_{args.window}.png",
            data_dir,
            rows,
            args.window,
            checkpoints=checkpoints,
            first_wall_time=first_wall_time,
        )

    total = len(rows)
    successes = int(rows[-1]["cumulative_successes"])
    all_pure_rl_rows = [row for row in rows if int(row["is_pure_rl"])]
    pure_rl_total = len(all_pure_rl_rows)
    pure_rl_successes = (
        int(all_pure_rl_rows[-1]["pure_rl_cumulative_successes"])
        if all_pure_rl_rows
        else 0
    )
    final_rolling = (
        float(rows[-1]["rolling_success_rate"])
        if rows[-1]["rolling_success_rate"] is not None
        else None
    )
    final_cumulative = float(rows[-1]["cumulative_success_rate"])
    final_pure_rl_rolling = (
        float(all_pure_rl_rows[-1]["pure_rl_rolling_success_rate"])
        if all_pure_rl_rows
        and all_pure_rl_rows[-1]["pure_rl_rolling_success_rate"] is not None
        else None
    )
    final_pure_rl_cumulative = (
        float(all_pure_rl_rows[-1]["pure_rl_cumulative_success_rate"])
        if all_pure_rl_rows
        else 0.0
    )
    full_pure_rl_window_rows = [
        row
        for row in all_pure_rl_rows
        if row["pure_rl_rolling_success_rate"] is not None
    ]
    peak_pure_rl_rolling = (
        max(float(r["pure_rl_rolling_success_rate"]) for r in full_pure_rl_window_rows)
        if full_pure_rl_window_rows
        else None
    )
    print(f"episodes: {total}")
    print(f"success: {successes} / {total} ({final_cumulative:.1%})")
    print(
        f"final rolling success rate, window={args.window}: {format_rate(final_rolling)}"
    )
    print(
        f"pure RL success: {pure_rl_successes} / {pure_rl_total} "
        f"({final_pure_rl_cumulative:.1%})"
    )
    print(
        f"final pure RL rolling success rate, window={args.window}: "
        f"{format_rate(final_pure_rl_rolling)}"
    )
    print(
        f"peak pure RL rolling success rate, window={args.window}: "
        f"{format_rate(peak_pure_rl_rolling)}"
    )
    print(f"wrote: {csv_path}")
    if not args.no_plot:
        print(f"wrote: {output_dir / f'sr_rolling_window_{args.window}.png'}")


if __name__ == "__main__":
    main()
