# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-tool profiling logs aggregated across iterations and exec seeds.

Creates one JSON per tool under ``{run_dir}/tool_log/``, accumulating call
records tagged with ``iteration`` and ``exec_id`` (seed) so downstream
visualizations (e.g. wandb line plots) can trend duration/success per tool
across the whole run.

Records are appended incrementally at the end of each iteration; the input
is the list of per-seed ``profiling.json`` files already produced by
``cap.agent.profiler``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def tool_log_dir(run_dir: Path | str) -> Path:
    d = Path(run_dir) / "tool_log"
    d.mkdir(parents=True, exist_ok=True)
    return d


def update_tool_logs(
    run_dir: Path | str,
    iteration: int,
    profiling_entries: list[tuple[int, Path | str]],
) -> None:
    """Append calls from this iteration's profiling.json files to per-tool JSONs.

    Args:
        run_dir: The agent session's run directory.
        iteration: Current iteration index (for tagging records).
        profiling_entries: ``(seed, profiling_json_path)`` tuples from all
            parallel execs in this iteration.
    """
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for seed, p in profiling_entries:
        path = Path(p)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for call in data.get("calls", []):
            tool = call.get("tool", "unknown")
            by_tool.setdefault(tool, []).append(
                {
                    "iteration": iteration,
                    "exec_id": seed,
                    "call_id": call.get("call_id"),
                    "duration_ms": call.get("duration_ms", 0.0),
                    "idle_gap_ms": call.get("idle_gap_ms", 0.0),
                    "error": call.get("error"),
                    "timestamp": call.get("timestamp"),
                }
            )

    if not by_tool:
        return

    out_dir = tool_log_dir(run_dir)
    for tool, new_records in by_tool.items():
        tool_path = out_dir / f"{_safe_name(tool)}.json"
        existing: dict[str, Any] = {"tool": tool, "calls": []}
        if tool_path.exists():
            try:
                existing = json.loads(tool_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        existing["tool"] = tool
        existing["calls"].extend(new_records)
        existing["summary"] = _summarize(existing["calls"])
        tool_path.write_text(
            json.dumps(existing, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def read_tool_logs(run_dir: Path | str) -> dict[str, list[dict[str, Any]]]:
    """Return ``{tool_name: [calls...]}`` across all per-tool JSONs."""
    d = Path(run_dir) / "tool_log"
    if not d.exists():
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out[data.get("tool", p.stem)] = data.get("calls", [])
    return out


def _summarize(calls: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(calls)
    if count == 0:
        return {"count": 0}
    errors = sum(1 for c in calls if c.get("error"))
    total_ms = sum(c.get("duration_ms", 0.0) for c in calls)
    return {
        "count": count,
        "errors": errors,
        "success_rate": round((count - errors) / count, 3),
        "total_duration_ms": round(total_ms, 1),
        "mean_duration_ms": round(total_ms / count, 1),
    }


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
