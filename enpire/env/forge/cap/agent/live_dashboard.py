"""Rich live dashboard for ``run_script.py``.

Renders a tool-call-centric view while a generated / saved script runs:
header (task · env · seed · iteration · elapsed), the currently-executing
tool call with live-ticking elapsed, and a history panel of recent
completed calls color-coded by status.

Wires into the existing profiler hooks (``set_tool_event_hooks``) so no
profiler changes are required.  Safely falls back to a no-op when not on a
TTY or when ``rich`` is not importable.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


def _rich_available() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


def dashboard_enabled(stream: Any) -> bool:
    """Return True when a rich dashboard should be rendered on *stream*."""
    if not _rich_available():
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class LiveDashboard:
    """Live tool-call dashboard.  Use as a context manager.

    Example::

        with LiveDashboard(title="seed=42", header={"env": "..."}) as dash:
            set_tool_event_hooks(on_start=dash.on_tool_start,
                                 on_end=dash.on_tool_end)
            exec(code, namespace)
    """

    HISTORY_LEN = 20
    TICK_HZ = 5.0
    SLOW_MS = 1000.0  # yellow when current call exceeds this
    HANG_MS = 30000.0  # red + "HANGING" badge when current call exceeds this

    def __init__(
        self,
        title: str = "",
        header: dict[str, str] | None = None,
        history_len: int = HISTORY_LEN,
    ) -> None:
        import sys as _sys

        from rich.console import Console
        from rich.live import Live

        # Bind to the real terminal — by the time we get here ``sys.stdout``
        # may already be wrapped by the profiler's StdoutTee, whose
        # ``isatty()`` returns False, which would make rich.Live fall back
        # to non-interactive rendering (no live updates).
        self._console = Console(
            file=_sys.__stdout__,
            highlight=False,
            soft_wrap=False,
            force_terminal=True,
        )
        self._title = title
        self._header: dict[str, str] = dict(header or {})
        self._history: deque[dict[str, Any]] = deque(maxlen=history_len)
        self._current: dict[str, Any] | None = None
        self._totals = {"calls": 0, "errors": 0, "total_ms": 0.0}
        self._run_t0 = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        self._live: Live | None = Live(
            self._render(),
            console=self._console,
            refresh_per_second=self.TICK_HZ,
            transient=False,
        )

    # ----- lifecycle -----

    def __enter__(self) -> LiveDashboard:
        assert self._live is not None
        self._live.__enter__()
        self._ticker = threading.Thread(target=self._tick_loop, daemon=True)
        self._ticker.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join(timeout=1.0)
        if self._live is not None:
            try:
                self._live.update(self._render(final=True))
            finally:
                self._live.__exit__(exc_type, exc, tb)

    def _tick_loop(self) -> None:
        interval = 1.0 / self.TICK_HZ
        while not self._stop.wait(interval):
            if self._live is None:
                return
            try:
                self._live.update(self._render())
            except Exception:
                return

    # ----- profiler hook callbacks -----

    def on_tool_start(
        self,
        name: str,
        call_id: int,
        args: tuple,
        kwargs: dict,
    ) -> None:
        with self._lock:
            self._current = {
                "call_id": call_id,
                "tool": name,
                "args": _short_args(args, kwargs),
                "t0": time.time(),
            }

    def on_tool_end(
        self,
        name: str,
        call_id: int,
        result: Any,
        error: Exception | None,
        elapsed_ms: float,
    ) -> None:
        with self._lock:
            args_str = ""
            if self._current and self._current.get("call_id") == call_id:
                args_str = self._current.get("args", "")
            self._current = None
            self._history.append(
                {
                    "call_id": call_id,
                    "tool": name,
                    "args": args_str,
                    "duration_ms": elapsed_ms,
                    "error": None
                    if error is None
                    else f"{type(error).__name__}: {error}",
                }
            )
            self._totals["calls"] += 1
            self._totals["total_ms"] += elapsed_ms
            if error is not None:
                self._totals["errors"] += 1

    # ----- rendering -----

    def _render(self, final: bool = False):
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        with self._lock:
            header = self._render_header(final)
            current = self._render_current()
            history = self._render_history()
            footer = self._render_footer()

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(header)
        grid.add_row(current)
        grid.add_row(history)
        grid.add_row(footer)
        title = Text(self._title or "run_script", style="bold")
        return Panel(grid, title=title, border_style="cyan")

    def _render_header(self, final: bool):
        from rich.table import Table
        from rich.text import Text

        elapsed = time.time() - self._run_t0
        t = Table.grid(expand=True)
        t.add_column(ratio=1)
        t.add_column(justify="right")
        kv_parts = []
        for k, v in self._header.items():
            kv_parts.append(Text.assemble((f"{k}=", "dim"), (str(v), "white")))
        left = Text(" · ").join(kv_parts) if kv_parts else Text("")
        status = "DONE" if final else "RUNNING"
        right = Text.assemble(
            (f"{status} ", "bold green" if final else "bold yellow"),
            (f"· {elapsed:6.1f}s", "dim"),
        )
        t.add_row(left, right)
        return t

    def _render_current(self):
        from rich.panel import Panel
        from rich.text import Text

        border = "blue"
        title = "in-flight"
        if self._current is None:
            body = Text("(idle)", style="dim")
        else:
            live_ms = (time.time() - self._current["t0"]) * 1000
            if live_ms > self.HANG_MS:
                color = "red"
                border = "red"
                title = "in-flight ⚠ HANGING"
            elif live_ms > self.SLOW_MS:
                color = "yellow"
            else:
                color = "cyan"
            parts: list = [
                ("▶ ", color),
                (f"#{self._current['call_id']} ", "dim"),
                (self._current["tool"], f"bold {color}"),
                (f"({self._current['args']})", "dim"),
                ("  ", ""),
                (f"{live_ms:7.0f}ms", color),
            ]
            if live_ms > self.HANG_MS:
                secs = live_ms / 1000.0
                parts.append(("  ", ""))
                parts.append((f"hanging… {secs:.0f}s (check server?)", "bold red blink"))
            body = Text.assemble(*parts)
        return Panel(body, title=title, border_style=border, padding=(0, 1))

    def _render_history(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        tbl = Table(expand=True, box=None, show_header=True, header_style="dim")
        tbl.add_column("#", justify="right", width=5, style="dim")
        tbl.add_column("tool", overflow="crop")
        tbl.add_column("args", overflow="ellipsis", ratio=1)
        tbl.add_column("ms", justify="right", width=9)
        tbl.add_column("", justify="right", width=2)

        for h in reversed(self._history):
            if h["error"]:
                status = Text("✗", style="bold red")
                tool_style = "red"
            elif h["duration_ms"] > 1000:
                status = Text("✓", style="yellow")
                tool_style = "yellow"
            else:
                status = Text("✓", style="green")
                tool_style = "green"
            tbl.add_row(
                str(h["call_id"]),
                Text(h["tool"], style=tool_style),
                h["args"],
                f"{h['duration_ms']:.0f}",
                status,
            )
        return Panel(
            tbl, title=f"recent ({len(self._history)})", border_style="magenta"
        )

    def _render_footer(self):
        from rich.text import Text

        return Text.assemble(
            ("calls=", "dim"),
            (str(self._totals["calls"]), "white"),
            ("  errors=", "dim"),
            (
                str(self._totals["errors"]),
                "red" if self._totals["errors"] else "green",
            ),
            ("  total_tool_ms=", "dim"),
            (f"{self._totals['total_ms']:.0f}", "white"),
        )


def _short_args(args: tuple, kwargs: dict, max_len: int = 60) -> str:
    parts = [_repr(a) for a in args]
    parts += [f"{k}={_repr(v)}" for k, v in kwargs.items()]
    s = ", ".join(parts)
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def _repr(v: Any) -> str:
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            return f"ndarray{tuple(v.shape)}"
    except ImportError:
        pass
    r = repr(v)
    return r if len(r) <= 40 else r[:39] + "…"
