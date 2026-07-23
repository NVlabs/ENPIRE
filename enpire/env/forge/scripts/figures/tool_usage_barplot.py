# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tool call time breakdown: total duration per primitive tool across all seeds.

Fetches the ``tool_profiling`` wandb Table logged by WandbLogger, aggregates
total wall-clock time per tool, and renders a horizontal bar chart sorted by
total duration (longest first).

Usage::

    python scripts/figures/tool_usage_barplot.py --run XXX
    python scripts/figures/tool_usage_barplot.py --run XXX --out out/tool_usage.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import artistic as art
from base import WandbRunFetcher

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENTITY  = None           # defaults to logged-in user's entity via wandb.Api().default_entity
PROJECT = "cap-robocasa"

RUN_NAME = "XXX"   # replace with actual wandb run name / ID

# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class ToolProfilingFetcher(WandbRunFetcher):
    """Download the tool_profiling Table and namespace tool list from a wandb run."""

    def fetch_namespace_tools(self, run_name: str) -> list[str]:
        """Return the full list of namespace tool names from run config, or []."""
        run = self.get_run(run_name)
        return list(run.config.get("namespace_tools", []))

    def fetch(self, run_name: str) -> pd.DataFrame:
        run = self.get_run(run_name)
        artifacts = run.logged_artifacts()

        # tool_profiling is logged as a wandb.Table; retrieve via artifact
        # or fall back to scanning history for the table artifact reference.
        table_artifact = None
        for art_obj in artifacts:
            if "tool_profiling" in art_obj.name:
                table_artifact = art_obj
                break

        if table_artifact is None:
            # Try scanning history for the table key
            history = run.history(keys=["tool_profiling"])
            if history.empty or "tool_profiling" not in history.columns:
                raise ValueError(
                    f"No 'tool_profiling' table found in run '{run_name}'. "
                    "Make sure the run used wandb.enabled=true and completed at least one iteration."
                )
            # W&B returns artifact references as dicts; resolve
            rows = []
            for cell in history["tool_profiling"].dropna():
                if hasattr(cell, "get_dataframe"):
                    rows.append(cell.get_dataframe())
                elif isinstance(cell, dict) and "artifact_path" in cell:
                    api = self.api
                    t = api.artifact(cell["artifact_path"]).get("tool_profiling")
                    rows.append(t.get_dataframe())
            if not rows:
                raise ValueError("Could not parse tool_profiling table cells.")
            return pd.concat(rows, ignore_index=True)

        table = table_artifact.get("tool_profiling")
        return table.get_dataframe()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_tool_time(df: pd.DataFrame, all_tools: list[str] | None = None) -> pd.DataFrame:
    """Sum duration_ms per tool. Includes zero-call rows for tools in all_tools."""
    agg = (
        df.groupby("tool")["duration_ms"]
        .agg(total_ms="sum", calls="count", mean_ms="mean")
        .reset_index()
    )
    if all_tools:
        # Add zero rows for tools that were never called
        called = set(agg["tool"])
        zeros = pd.DataFrame({
            "tool": [t for t in all_tools if t not in called],
            "total_ms": 0.0,
            "calls": 0,
            "mean_ms": 0.0,
        })
        agg = pd.concat([agg, zeros], ignore_index=True)

    agg = agg.sort_values("total_ms", ascending=False).reset_index(drop=True)
    agg["total_s"] = agg["total_ms"] / 1000.0
    return agg


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_tool_usage(agg: pd.DataFrame, run_name: str, out: Path | None) -> None:
    art.apply_style()

    fig, ax = plt.subplots(figsize=(7, max(3.5, len(agg) * 0.35)))

    tools = agg["tool"].tolist()
    total_s = agg["total_s"].tolist()
    calls = agg["calls"].tolist()

    y = np.arange(len(tools))
    colors = [
        art.PALETTE["navy"] if t > 0 else art.PALETTE["ash"]
        for t in total_s
    ]
    bars = ax.barh(
        y,
        total_s,
        color=colors,
        alpha=0.85,
        height=0.6,
    )

    # Annotate each bar with call count
    for bar, n in zip(bars, calls):
        ax.text(
            bar.get_width() + max(total_s) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"n={n}",
            va="center",
            ha="left",
            fontsize=art.FONTSIZE["annot"],
            color=art.PALETTE["slate"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(tools, fontsize=art.FONTSIZE["tick"])
    ax.invert_yaxis()  # longest bar on top
    ax.set_xlabel("Total wall-clock time (s)", fontsize=art.FONTSIZE["label"])
    ax.set_title(f"Tool call time — {run_name}", fontsize=art.FONTSIZE["title"])
    ax.margins(x=0.15)

    plt.tight_layout()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out)
        print(f"Saved: {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tool call time bar plot from wandb run.")
    parser.add_argument("--run", default=RUN_NAME, help="wandb run name or ID (default: XXX)")
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--out", default=None, help="Output path (e.g. out/tool_usage.pdf)")
    args = parser.parse_args()

    fetcher = ToolProfilingFetcher(entity=args.entity, project=args.project)

    all_tools = fetcher.fetch_namespace_tools(args.run)
    if all_tools:
        print(f"  namespace_tools in run config: {len(all_tools)} tools")
    else:
        print("  namespace_tools not found in run config (run with updated run_agent.py to enable)")

    print(f"Fetching tool_profiling table from run '{args.run}' …")
    df = fetcher.fetch(args.run)
    print(f"  {len(df)} call records, {df['tool'].nunique()} unique tools called")

    agg = aggregate_tool_time(df, all_tools=all_tools or None)
    print(agg[["tool", "calls", "total_s", "mean_ms"]].to_string(index=False))

    out = Path(args.out) if args.out else None
    plot_tool_usage(agg, args.run, out)


if __name__ == "__main__":
    main()
