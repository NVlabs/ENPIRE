#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live rich TUI dashboard for the grootpool middleware.

Polls GET /status every --interval seconds and renders a full-screen panel
showing per-worker inference state, active sessions, and queue depth.

Usage (standalone):
    python scripts/grootpool_dashboard.py [--host localhost] [--port 7071]

Invoked automatically by tmux/launch_grootpool.sh in the overview window.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _fetch(url: str, timeout: float = 2.0) -> tuple[dict | None, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read()), None
    except urllib.error.URLError as e:
        return None, str(e.reason)
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _ms(val: float | None) -> str:
    if val is None:
        return "—"
    if val < 1000:
        return f"{val:.0f}ms"
    return f"{val/1000:.1f}s"


def _age_text(secs: float) -> Text:
    if secs < 2:
        return Text("now", style="bold green")
    if secs < 10:
        return Text(f"{secs:.0f}s", style="yellow")
    if secs < 60:
        return Text(f"{secs:.0f}s", style="dim")
    return Text(f"{secs/60:.1f}m", style="red")


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _build_panel(status: dict | None, url: str, err: str | None) -> Panel:
    now = datetime.now().strftime("%H:%M:%S")

    # ── unreachable ──────────────────────────────────────────────────────────
    if status is None:
        return Panel(
            Text.from_markup(
                f"[bold red]● Middleware unreachable[/bold red]\n"
                f"[dim]{url}[/dim]\n"
                f"[dim]{err or ''}[/dim]"
            ),
            title=f"[bold]GR00T Pool[/bold]  [dim]{now}[/dim]",
            border_style="red",
        )

    workers = status.get("workers", [])
    idle_n = status.get("idle_workers", 0)
    inflight = status.get("inflight", 0)
    total_n = len(workers)
    p50 = status.get("step_p50_ms")
    p95 = status.get("step_p95_ms")

    # ── workers table ─────────────────────────────────────────────────────────
    t = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        show_edge=False,
        header_style="bold dim",
        show_lines=False,
        padding=(0, 1),
    )
    t.add_column("W", width=3, justify="right")
    t.add_column("GPU", width=4, justify="right")
    t.add_column("Port", width=6, justify="right")
    t.add_column("State", min_width=14)
    t.add_column("PID", width=8, justify="right")

    for w in sorted(workers, key=lambda x: x["id"]):
        wid = w["id"]
        alive = w.get("alive", False)
        respawning = w.get("respawning", False)

        if respawning:
            state = Text("⟳ respawn", style="yellow")
        elif not alive:
            state = Text("✗ dead", style="bold red")
        else:
            state = Text("● ready", style="bold green")

        t.add_row(
            str(wid),
            str(w.get("gpu", "?")),
            str(w.get("port", "?")),
            state,
            str(w.get("pid", "—")),
            style="" if alive else "dim",
        )

    # ── title / summary bar ───────────────────────────────────────────────────
    idle_col = "green" if idle_n > 0 else "red"
    inf_col = "bold green" if inflight > 0 else "dim"
    summary = (
        f"[{idle_col}]{idle_n}/{total_n} idle[/{idle_col}]  "
        f"[{inf_col}]{inflight} inflight[/{inf_col}]  "
        f"[dim]p50 {_ms(p50)}  p95 {_ms(p95)}[/dim]  "
        f"[dim]{now}[/dim]"
    )

    border = "green" if inflight > 0 else "blue"
    return Panel(t, title=f"[bold]GR00T Pool[/bold]  {summary}", border_style=border)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Live grootpool status dashboard")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=7071)
    ap.add_argument("--interval", type=float, default=0.5, help="Poll interval in seconds")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/status"

    console = Console(
        highlight=False,
        soft_wrap=False,
        force_terminal=True,
    )

    last_status: dict | None = None
    last_err: str | None = "connecting…"

    with Live(
        _build_panel(None, url, last_err),
        console=console,
        refresh_per_second=max(1, int(1 / args.interval)),
        screen=True,
    ) as live:
        while True:
            data, err = _fetch(url)
            if data is not None:
                last_status = data
                last_err = None
            else:
                last_err = err

            live.update(
                _build_panel(
                    last_status if last_err is None else None,
                    url,
                    last_err,
                )
            )
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
