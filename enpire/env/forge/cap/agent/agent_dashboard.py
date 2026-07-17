"""Rich live dashboard for ``run_agent.py`` pipeline loop.

Renders parent-side progress across iterations:

- Header: task · iter N/M · wall-clock elapsed · status.
- Current step panel: which pipeline step is running, live-ticking ms;
  turns yellow/red on slow/hang thresholds.
- Seed progress (only during the executor step): progress bar + per-seed
  status rows. Populated by a background poller that watches
  ``{iter_dir}/exec_*/result.json`` and ``.tmux_done`` sentinels.
- Iteration history: compact table of completed iterations with step
  timings and S/All success rate.

Sibling to ``live_dashboard.py`` (which renders the per-seed
``run_script.py`` dashboard inside each tmux window). This module's
dashboard renders in the parent terminal where ``run_agent.py`` runs.

Activate iff ``sys.__stdout__.isatty()`` and ``rich`` is importable;
otherwise it's a no-op (caller keeps today's plain prints).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

_active = [False]
_current: list[Any] = [None]


def is_active() -> bool:
    """True while an AgentDashboard is rendering — guards plain prints."""
    return _active[0]


def current() -> Any:
    """Return the active ``AgentDashboard`` instance, or ``None`` if none.

    Used by pipeline steps that want to publish sub-phase progress
    (e.g. the parallel VLM reflection inside SubprocessExecutorStep)
    without threading a dashboard ref through every function signature.
    """
    return _current[0]


def _rich_available() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


def dashboard_enabled(stream: Any) -> bool:
    if not _rich_available():
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class AgentDashboard:
    TICK_HZ = 4.0
    POLL_HZ = 2.0
    SEED_TAIL = 12  # rows shown in seed panel
    HISTORY_LEN = 20
    SLOW_MS = 10_000.0  # step yellow at >10s
    HANG_MS = 120_000.0  # step red at >2min
    NVIDIA_KEY_ROWS = 32
    NVIDIA_RPM_WINDOW_S = 60.0

    def __init__(
        self,
        task: str,
        max_iterations: int,
        session: Any,
        *,
        code_backend: str | None = None,
        code_model: str | None = None,
        reflect_backend: str | None = None,
        reflect_model: str | None = None,
    ) -> None:
        import sys as _sys

        from rich.console import Console
        from rich.live import Live

        self._task = task
        self._max_iters = max_iterations
        self._session = session
        self._code_backend = code_backend
        self._code_model = code_model
        self._reflect_backend = reflect_backend
        self._reflect_model = reflect_model

        self._console = Console(
            file=_sys.__stdout__,
            highlight=False,
            soft_wrap=False,
            force_terminal=True,
        )
        self._t0 = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._tick_thread: threading.Thread | None = None

        self._iter = 0
        self._step_name: str | None = None
        self._step_t0: float | None = None
        self._iter_history: deque[dict[str, Any]] = deque(maxlen=self.HISTORY_LEN)
        self._iter_step_times: dict[str, float] = {}
        self._n_skills: int = 0
        self._n_families: int = 0
        self._service_health: list[dict[str, Any]] = []

        # Executor seed polling
        self._n_seeds: int = 0
        self._seed_statuses: dict[int, dict[str, Any]] = {}

        # Post-exec VLM reflection progress (parallel fanout across seeds)
        self._reflect_total: int = 0
        self._reflect_done: int = 0
        self._reflect_failed: int = 0
        self._reflect_t0: float | None = None
        self._reflect_backend_live: str | None = None

        # Provider-level NVIDIA telemetry. The provider writes process-safe
        # JSONL events so child run_script.py processes are visible here too.
        telemetry_file = os.environ.get("CAP_NVIDIA_TELEMETRY_FILE")
        self._nvidia_telemetry_path: Path | None = (
            Path(telemetry_file)
            if telemetry_file
            else (
                Path(session.run_dir) / "nvidia_requests.jsonl"
                if session is not None and hasattr(session, "run_dir")
                else None
            )
        )
        backend_names = [
            str(v).lower()
            for v in (code_backend, reflect_backend)
            if v is not None
        ]
        self._nvidia_provider_url = os.environ.get("CAP_NVIDIA_PROVIDER_URL", "").strip()
        self._show_nvidia_telemetry = (
            "nvidia" in backend_names and not self._nvidia_provider_url
        )
        self._nvidia_telemetry_pos: int = 0
        self._nvidia_active_requests: dict[str, str] = {}
        self._nvidia_key_stats: dict[str, dict[str, Any]] = {}
        self._nvidia_recent_errors: deque[dict[str, Any]] = deque(maxlen=4)
        self._nvidia_recent_request_ts: deque[float] = deque()
        self._nvidia_recent_token_events: deque[tuple[float, int]] = deque()
        self._poller_thread: threading.Thread | None = None
        self._poller_stop: threading.Event | None = None

        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=self.TICK_HZ,
            transient=False,
        )

    # ----- lifecycle -----

    def __enter__(self) -> AgentDashboard:
        _active[0] = True
        _current[0] = self
        self._live.__enter__()
        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        _active[0] = False
        _current[0] = None
        self._stop.set()
        if self._poller_stop is not None:
            self._poller_stop.set()
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=1.0)
        try:
            self._live.update(self._render(final=True))
        finally:
            self._live.__exit__(exc_type, exc, tb)

    def _tick_loop(self) -> None:
        interval = 1.0 / self.TICK_HZ
        while not self._stop.wait(interval):
            try:
                self._live.update(self._render())
            except Exception:
                return

    # ----- hooks called from agent_pipeline -----

    def on_skills_updated(self, n_skills: int, n_families: int | None = None) -> None:
        with self._lock:
            self._n_skills = n_skills
            if n_families is not None:
                self._n_families = n_families

    def on_service_health(self, rows: list[dict[str, Any]]) -> None:
        """Publish startup runtime-service health rows to the dashboard."""
        with self._lock:
            self._service_health = [dict(row) for row in rows]

    # ---- per-seed VLM reflection (post-executor) ----

    def on_reflect_start(self, n_seeds: int, backend: str | None = None) -> None:
        with self._lock:
            self._reflect_total = n_seeds
            self._reflect_done = 0
            self._reflect_failed = 0
            self._reflect_t0 = time.time()
            self._reflect_backend_live = backend

    def on_reflect_tick(self, failed: bool = False) -> None:
        """Called once per seed as its reflection future resolves."""
        with self._lock:
            self._reflect_done += 1
            if failed:
                self._reflect_failed += 1

    def on_reflect_end(self) -> None:
        with self._lock:
            # Keep totals visible until next iter — they'll be overwritten
            # by on_reflect_start on the next post-exec phase. Clearing t0
            # stops the ticking clock in the render.
            self._reflect_t0 = None

    def on_iteration_start(self, iteration: int) -> None:
        with self._lock:
            self._iter = iteration
            self._iter_step_times = {}
            self._seed_statuses = {}

    def on_step_start(self, name: str, n_seeds: int = 0) -> None:
        with self._lock:
            self._step_name = name
            self._step_t0 = time.time()
            if name == "executor":
                self._n_seeds = n_seeds
                self._seed_statuses = {}
        if name == "executor":
            self._start_seed_poller()

    def on_seed_queued(self, seed_idx: int, gpu_id: int) -> None:
        """Called when a seed thread is waiting for its GPU semaphore."""
        with self._lock:
            existing = self._seed_statuses.get(seed_idx, {})
            if existing.get("status") not in ("running", "ok", "fail"):
                self._seed_statuses[seed_idx] = {
                    **existing,
                    "status": "que",
                    "gpu_id": gpu_id,
                    "start_time": existing.get("start_time") or time.time(),
                }

    def on_seed_running(self, seed_idx: int, gpu_id: int) -> None:
        """Called when a seed has acquired its GPU semaphore and started."""
        with self._lock:
            existing = self._seed_statuses.get(seed_idx, {})
            if existing.get("status") not in ("ok", "fail"):
                self._seed_statuses[seed_idx] = {
                    **existing,
                    "status": "running",
                    "gpu_id": gpu_id,
                    "start_time": time.time(),
                }

    def on_step_end(self, name: str, elapsed_ms: float) -> None:
        if name == "executor":
            self._stop_seed_poller()
        with self._lock:
            self._iter_step_times[name] = elapsed_ms
            self._step_name = None
            self._step_t0 = None

    def on_iteration_end(
        self,
        iteration: int,
        success: bool,
        n_seeds: int,
        successes: int,
        success_rate: float,
        avg_score: float,
    ) -> None:
        with self._lock:
            self._iter_history.append(
                {
                    "iter": iteration,
                    "step_times": dict(self._iter_step_times),
                    "n_seeds": n_seeds,
                    "successes": successes,
                    "success_rate": success_rate,
                    "avg_score": avg_score,
                    "success": success,
                }
            )

    # ----- seed poller -----

    def _start_seed_poller(self) -> None:
        self._poller_stop = threading.Event()
        self._poller_thread = threading.Thread(target=self._poll_seeds, daemon=True)
        self._poller_thread.start()

    def _stop_seed_poller(self) -> None:
        if self._poller_stop is not None:
            self._poller_stop.set()
        if self._poller_thread is not None:
            self._poller_thread.join(timeout=1.0)
        self._poller_thread = None
        self._poller_stop = None

    def _poll_seeds(self) -> None:
        import json as _json

        interval = 1.0 / self.POLL_HZ
        assert self._poller_stop is not None
        while not self._poller_stop.wait(interval):
            if self._session is None:
                return
            try:
                iter_dir = self._session.iterations_dir(self._iter)
            except Exception:
                continue
            for exec_subdir in sorted(iter_dir.glob("exec_*")):
                try:
                    exec_id = int(exec_subdir.name.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                result_path = exec_subdir / "result.json"
                with self._lock:
                    existing = self._seed_statuses.get(exec_id, {})
                    start_time = existing.get("start_time") or time.time()
                    prev_status = existing.get("status", "running")

                    status = prev_status if prev_status in ("ok", "fail") else "running"
                    score = existing.get("score")
                    elapsed_s = existing.get("elapsed_s")

                    if prev_status not in ("ok", "fail") and result_path.exists():
                        try:
                            data = _json.loads(result_path.read_text())
                            status = "ok" if data.get("success") else "fail"
                            score = float(data.get("score", 0.0))
                            elapsed_s = time.time() - start_time
                        except Exception:
                            pass
                    elif status == "running":
                        elapsed_s = time.time() - start_time

                    self._seed_statuses[exec_id] = {
                        "status": status,
                        "gpu_id": existing.get("gpu_id"),
                        "score": score,
                        "elapsed_s": elapsed_s,
                        "start_time": start_time,
                    }

    # ----- rendering -----

    def _render(self, final: bool = False):
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        with self._lock:
            elapsed = time.time() - self._t0
            step_name = self._step_name
            step_elapsed = (
                (time.time() - self._step_t0) if self._step_t0 is not None else 0.0
            )
            header = self._render_header(elapsed, final)
            models = self._render_models()
            services = self._render_service_health()
            current = self._render_current_step(step_name, step_elapsed)
            nvidia = self._render_nvidia_telemetry()
            seeds = self._render_seed_progress() if step_name == "executor" else None
            reflect = (
                self._render_reflect_progress() if step_name == "executor" else None
            )
            history = self._render_history()

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(header)
        if models is not None:
            grid.add_row(models)
        if services is not None:
            grid.add_row(services)
        grid.add_row(current)
        if nvidia is not None:
            grid.add_row(nvidia)
        if seeds is not None:
            grid.add_row(seeds)
        if reflect is not None:
            grid.add_row(reflect)
        grid.add_row(history)
        title = Text(
            f"run_agent · {self._task[:60]}",
            style="bold cyan",
        )
        return Panel(grid, title=title, border_style="cyan")

    def _render_header(self, elapsed: float, final: bool):
        from rich.table import Table
        from rich.text import Text

        t = Table.grid(expand=True)
        t.add_column(ratio=1)
        t.add_column(justify="right")
        # "skills N (F fam)" when families < total (i.e. some versions exist);
        # otherwise just "skills N" since families == versions makes the extra
        # segment redundant noise.
        left_parts: list[tuple[str, str]] = [
            ("iter ", "dim"),
            (f"{self._iter}", "white"),
            ("/", "dim"),
            (f"{self._max_iters}", "white"),
            ("  skills ", "dim"),
            (f"{self._n_skills}", "magenta"),
        ]
        if self._n_families and self._n_families < self._n_skills:
            left_parts.append((f" ({self._n_families} fam)", "dim"))
        left = Text.assemble(*left_parts)
        status = "DONE" if final else "RUNNING"
        right = Text.assemble(
            (f"{status} ", "bold green" if final else "bold yellow"),
            (f"· {elapsed:7.1f}s", "dim"),
        )
        t.add_row(left, right)
        return t

    def _render_models(self):
        """One-line model provenance: `code-gen: <backend>/<model>  reflect: ...`.

        Returns None when neither field was supplied (e.g. dashboard was
        constructed without config context).
        """
        from rich.text import Text

        def _fmt(backend: str | None, model: str | None) -> str | None:
            if not backend and not model:
                return None
            if backend and model:
                return f"{backend}/{model}"
            return backend or model

        code = _fmt(self._code_backend, self._code_model)
        reflect = _fmt(self._reflect_backend, self._reflect_model)
        if not code and not reflect:
            return None

        # Key count — shown when nvidia backend is active so the user knows
        # how many API keys are in the rotation pool.
        n_keys = 0
        try:
            from enpire.env.forge.cap.agent.tools.vlm.backends.nvidia import list_nvidia_keys
            n_keys = len(list_nvidia_keys())
        except Exception:
            pass

        parts: list[tuple[str, str]] = []
        if code:
            parts.extend([("code-gen: ", "dim"), (code, "cyan")])
        if code and reflect:
            parts.append(("  ", "dim"))
        if reflect:
            parts.extend([("reflect: ", "dim"), (reflect, "magenta")])
        if n_keys > 0:
            parts.extend([("  🔑", "dim"), (f"×{n_keys}", "yellow")])
        return Text.assemble(*parts)

    def _render_service_health(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        rows = list(self._service_health)
        if not rows:
            return None

        unhealthy = [r for r in rows if r.get("status") != "healthy"]
        summary_style = "green" if not unhealthy else "red"
        summary = Text.assemble(
            ("services ", "dim"),
            (f"{len(rows) - len(unhealthy)}/{len(rows)} healthy", summary_style),
        )

        tbl = Table(expand=True, box=None, show_header=True, header_style="dim")
        tbl.add_column("service", width=10)
        tbl.add_column("status", width=12)
        tbl.add_column("endpoint", width=34, overflow="fold")
        tbl.add_column("lat", width=8, justify="right")
        tbl.add_column("detail", overflow="fold")

        for row in rows:
            status = str(row.get("status") or "unknown")
            if status == "healthy":
                status_t = Text("✓ healthy", style="green")
            elif status == "unhealthy":
                status_t = Text("✗ bad", style="red")
            else:
                status_t = Text("✗ no conn", style="red")
            latency = row.get("latency_ms")
            detail = row.get("error") or row.get("detail") or ""
            tbl.add_row(
                str(row.get("name") or ""),
                status_t,
                str(row.get("endpoint") or ""),
                f"{float(latency):.0f}ms" if isinstance(latency, (int, float)) else "—",
                str(detail)[:160],
            )

        body = Table.grid(expand=True)
        body.add_column()
        body.add_row(summary)
        body.add_row(tbl)
        return Panel(
            body,
            title="startup service health",
            border_style="green" if not unhealthy else "red",
            padding=(0, 1),
        )

    def _render_reflect_progress(self):
        """Compact progress panel for the post-exec parallel VLM reflection.

        Returns None when no reflection is active (``_reflect_total == 0``).
        Clears cleanly between iterations; on_reflect_start wipes state.
        """
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        total = self._reflect_total
        if total <= 0:
            return None

        done = self._reflect_done
        failed = self._reflect_failed
        pending = max(total - done, 0)
        frac = done / total if total > 0 else 1.0

        bar_w = 24
        filled = int(bar_w * frac)
        bar = "█" * filled + "░" * (bar_w - filled)

        elapsed: float | None = None
        if self._reflect_t0 is not None:
            elapsed = time.time() - self._reflect_t0

        ok = done - failed
        line = Text.assemble(
            (bar + "  ", "magenta"),
            (f"{done}/{total}", "white"),
            ("  ✓", "dim"),
            (f"{ok}", "green"),
            (" ✗", "dim"),
            (f"{failed}", "red"),
            (" ⋯", "dim"),
            (f"{pending}", "yellow"),
            (
                f"  · {elapsed:5.1f}s" if elapsed is not None else "",
                "dim",
            ),
        )
        title = "reflection (parallel)"
        if self._reflect_backend_live:
            title = f"reflection (parallel · {self._reflect_backend_live})"

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(line)
        return Panel(grid, title=title, border_style="magenta", padding=(0, 1))

    def _nvidia_stats_for_key(self, key: str) -> dict[str, Any]:
        return self._nvidia_key_stats.setdefault(
            key,
            {
                "active": 0,
                "ok": 0,
                "error": 0,
                "last_ms": None,
                "last_source": "",
                "last_model": "",
                "last_error": "",
                "status": "unknown",
            },
        )

    def _ensure_known_nvidia_keys_locked(self) -> None:
        try:
            from enpire.env.forge.cap.agent.providers.nvidia import list_nvidia_keys, nvidia_key_label

            for key in list_nvidia_keys():
                self._nvidia_stats_for_key(nvidia_key_label(key))
        except Exception:
            pass

    def _poll_nvidia_telemetry_locked(self) -> None:
        path = self._nvidia_telemetry_path
        if path is None or not path.exists():
            return
        try:
            size = path.stat().st_size
            if size < self._nvidia_telemetry_pos:
                self._nvidia_telemetry_pos = 0
                self._nvidia_active_requests = {}
                self._nvidia_key_stats = {}
                self._nvidia_recent_errors.clear()
                self._nvidia_recent_request_ts.clear()
                self._nvidia_recent_token_events.clear()
            with path.open("r", encoding="utf-8") as f:
                f.seek(self._nvidia_telemetry_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._apply_nvidia_telemetry_event_locked(event)
                self._nvidia_telemetry_pos = f.tell()
        except Exception:
            return

    def _merge_nvidia_scheduler_stats_locked(self) -> None:
        try:
            from enpire.env.forge.cap.agent.providers.nvidia import read_nvidia_scheduler_stats

            rows = read_nvidia_scheduler_stats()
        except Exception:
            return
        for row in rows:
            key = str(row.get("key") or "")
            if not key:
                continue
            stats = self._nvidia_stats_for_key(key)
            stats["status"] = str(row.get("status") or "unknown")
            stats["active"] = int(row.get("active", 0) or 0)
            stats["ok"] = max(
                int(stats.get("ok", 0) or 0), int(row.get("ok_count", 0) or 0)
            )
            stats["error"] = max(
                int(stats.get("error", 0) or 0), int(row.get("error_count", 0) or 0)
            )
            last_ms = row.get("last_latency_ms")
            if isinstance(last_ms, (int, float)):
                stats["last_ms"] = float(last_ms)
            last_error = str(row.get("last_error") or "").strip()
            if last_error:
                http_status = row.get("last_http_status")
                prefix = f"HTTP {http_status}: " if http_status else ""
                stats["last_error"] = prefix + last_error

    def _apply_nvidia_telemetry_event_locked(self, event: dict[str, Any]) -> None:
        key = str(event.get("key") or "key?:unknown")
        request_id = str(event.get("request_id") or "")
        stats = self._nvidia_stats_for_key(key)
        source = str(event.get("source") or "")
        model = str(event.get("model") or "")
        if source:
            stats["last_source"] = source
        if model:
            stats["last_model"] = model

        event_name = event.get("event")
        if event_name == "start":
            if request_id:
                self._nvidia_active_requests[request_id] = key
            stats["active"] = int(stats.get("active", 0) or 0) + 1
            return
        if event_name != "end":
            return

        event_ts = event.get("ts")
        if not isinstance(event_ts, (int, float)):
            event_ts = time.time()
        self._nvidia_recent_request_ts.append(float(event_ts))
        key_recent_ts = stats.setdefault("_recent_request_ts", deque())
        if isinstance(key_recent_ts, deque):
            key_recent_ts.append(float(event_ts))
        total_tokens = event.get("total_tokens")
        if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
            self._nvidia_recent_token_events.append((float(event_ts), total_tokens))

        start_key = self._nvidia_active_requests.pop(request_id, None)
        if start_key is not None:
            start_stats = self._nvidia_stats_for_key(start_key)
            start_stats["active"] = max(int(start_stats.get("active", 0) or 0) - 1, 0)
        elif stats.get("active"):
            stats["active"] = max(int(stats.get("active", 0) or 0) - 1, 0)

        latency_ms = event.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            stats["last_ms"] = float(latency_ms)

        if event.get("status") == "ok":
            stats["ok"] = int(stats.get("ok", 0) or 0) + 1
            return

        stats["error"] = int(stats.get("error", 0) or 0) + 1
        error_text = str(event.get("error") or "").strip()
        http_status = event.get("http_status")
        prefix = f"HTTP {http_status}: " if http_status else ""
        stats["last_error"] = (prefix + error_text).strip()
        self._nvidia_recent_errors.appendleft(
            {
                "key": key,
                "source": source,
                "http_status": http_status,
                "error": error_text,
            }
        )

    def _prune_nvidia_rpm_window_locked(self, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        cutoff = now - self.NVIDIA_RPM_WINDOW_S
        while self._nvidia_recent_request_ts and self._nvidia_recent_request_ts[0] < cutoff:
            self._nvidia_recent_request_ts.popleft()
        while (
            self._nvidia_recent_token_events
            and self._nvidia_recent_token_events[0][0] < cutoff
        ):
            self._nvidia_recent_token_events.popleft()
        for stats in self._nvidia_key_stats.values():
            recent_ts = stats.get("_recent_request_ts")
            if not isinstance(recent_ts, deque):
                continue
            while recent_ts and recent_ts[0] < cutoff:
                recent_ts.popleft()

    def _nvidia_rpm_for_timestamps(self, timestamps: Any) -> float:
        if not isinstance(timestamps, deque):
            return 0.0
        return len(timestamps) * (60.0 / self.NVIDIA_RPM_WINDOW_S)

    def _nvidia_tpm_for_events(self, events: Any) -> float:
        if not isinstance(events, deque):
            return 0.0
        return sum(tokens for _ts, tokens in events) * (60.0 / self.NVIDIA_RPM_WINDOW_S)

    @staticmethod
    def _nvidia_key_sort_value(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        key = item[0]
        if key.startswith("key"):
            digits = ""
            for ch in key[3:]:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                return (int(digits), key)
        return (10_000, key)

    def _render_nvidia_telemetry(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        if self._nvidia_provider_url:
            return None

        self._poll_nvidia_telemetry_locked()
        self._merge_nvidia_scheduler_stats_locked()
        self._prune_nvidia_rpm_window_locked()
        if (
            not self._show_nvidia_telemetry
            and not self._nvidia_key_stats
            and not self._nvidia_recent_errors
        ):
            return None

        self._ensure_known_nvidia_keys_locked()
        if not self._nvidia_key_stats:
            return None

        total_active = sum(
            int(s.get("active", 0) or 0) for s in self._nvidia_key_stats.values()
        )
        total_ok = sum(
            int(s.get("ok", 0) or 0) for s in self._nvidia_key_stats.values()
        )
        total_error = sum(
            int(s.get("error", 0) or 0) for s in self._nvidia_key_stats.values()
        )
        total_healthy = sum(
            1 for s in self._nvidia_key_stats.values() if s.get("status") == "healthy"
        )
        total_rpm = self._nvidia_rpm_for_timestamps(self._nvidia_recent_request_ts)
        total_tpm = self._nvidia_tpm_for_events(self._nvidia_recent_token_events)
        summary = Text.assemble(
            ("active ", "dim"),
            (str(total_active), "cyan" if total_active else "white"),
            ("  rpm ", "dim"),
            (f"{total_rpm:.0f}", "cyan" if total_rpm else "white"),
            ("  tpm ", "dim"),
            (f"{total_tpm:.0f}", "cyan" if total_tpm else "white"),
            ("  ok ", "dim"),
            (str(total_ok), "green"),
            ("  err ", "dim"),
            (str(total_error), "red" if total_error else "white"),
            ("  keys ", "dim"),
            (str(len(self._nvidia_key_stats)), "yellow"),
            ("  healthy ", "dim"),
            (str(total_healthy), "green"),
        )

        tbl = Table(expand=True, box=None, show_header=True, header_style="dim")
        tbl.add_column("key", width=12)
        tbl.add_column("state", width=12)
        tbl.add_column("act", width=4, justify="right")
        tbl.add_column("rpm", width=5, justify="right")
        tbl.add_column("ok", width=5, justify="right")
        tbl.add_column("err", width=5, justify="right")
        tbl.add_column("last", width=8, justify="right")
        tbl.add_column("source", width=18)
        tbl.add_column("last error", overflow="fold")

        rows = sorted(self._nvidia_key_stats.items(), key=self._nvidia_key_sort_value)
        for key, stats in rows[: self.NVIDIA_KEY_ROWS]:
            active = int(stats.get("active", 0) or 0)
            errors = int(stats.get("error", 0) or 0)
            last_ms = stats.get("last_ms")
            state = str(stats.get("status") or "unknown")
            state_style = "green" if state == "healthy" else "red"
            key_rpm = self._nvidia_rpm_for_timestamps(stats.get("_recent_request_ts"))
            tbl.add_row(
                Text(key, style="yellow"),
                Text(state, style=state_style),
                Text(str(active), style="cyan" if active else "dim"),
                Text(f"{key_rpm:.0f}", style="cyan" if key_rpm else "dim"),
                Text(str(int(stats.get("ok", 0) or 0)), style="green"),
                Text(str(errors), style="red" if errors else "dim"),
                f"{last_ms:.0f}ms" if isinstance(last_ms, (int, float)) else "-",
                str(stats.get("last_source") or "-"),
                str(stats.get("last_error") or "-"),
            )
        if len(rows) > self.NVIDIA_KEY_ROWS:
            tbl.add_row(
                Text(f"+{len(rows) - self.NVIDIA_KEY_ROWS} more", style="dim"),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            )

        body = Table.grid(expand=True)
        body.add_column()
        body.add_row(summary)
        body.add_row(tbl)
        return Panel(
            body,
            title="nvidia provider requests",
            border_style="yellow" if total_error else "cyan",
            padding=(0, 1),
        )

    def _render_current_step(self, step_name: str | None, step_elapsed: float):
        from rich.panel import Panel
        from rich.text import Text

        if step_name is None:
            return Panel(
                Text("(between steps)", style="dim"),
                title="current step",
                border_style="blue",
                padding=(0, 1),
            )
        live_ms = step_elapsed * 1000
        if live_ms > self.HANG_MS:
            color = "red"
            border = "red"
            title = "current step ⚠ HANG"
        elif live_ms > self.SLOW_MS:
            color = "yellow"
            border = "yellow"
            title = "current step"
        else:
            color = "cyan"
            border = "blue"
            title = "current step"
        body = Text.assemble(
            ("▶ ", color),
            (step_name, f"bold {color}"),
            ("   ", ""),
            (f"{step_elapsed:6.1f}s", color),
        )
        return Panel(body, title=title, border_style=border, padding=(0, 1))

    def _render_seed_progress(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        n = self._n_seeds
        if n == 0:
            return Text("")
        statuses = list(self._seed_statuses.values())
        done = sum(1 for s in statuses if s["status"] in ("ok", "fail"))
        ok = sum(1 for s in statuses if s["status"] == "ok")
        fail = sum(1 for s in statuses if s["status"] == "fail")
        running = sum(1 for s in statuses if s["status"] == "running")
        queued = sum(1 for s in statuses if s["status"] == "que")
        not_started = max(0, n - len(statuses))

        bar_width = 24
        filled = int(bar_width * done / n) if n else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        summary = Text.assemble(
            (bar, "green"),
            ("  ", ""),
            (f"{done}/{n}", "white"),
            ("  ✓", "dim"),
            (f"{ok}", "green"),
            ("  ✗", "dim"),
            (f"{fail}", "red"),
            ("  ▶", "dim"),
            (f"{running}", "cyan"),
            ("  ⏳", "dim"),
            (f"{queued + not_started}", "yellow"),
        )

        tbl = Table(expand=True, box=None, show_header=True, header_style="dim")
        tbl.add_column("exec", width=5, justify="right")
        tbl.add_column("GPU", width=5, justify="center")
        tbl.add_column("status", width=8)
        tbl.add_column("score", width=7, justify="right")
        tbl.add_column("elapsed", width=10, justify="right")

        # Running first, then queued, then finished (most recent first)
        items = list(self._seed_statuses.items())
        _status_order = {"running": 0, "que": 1, "ok": 2, "fail": 2}
        items.sort(
            key=lambda kv: (
                _status_order.get(kv[1]["status"], 3),
                -(kv[1].get("start_time", 0) or 0),
            )
        )
        for exec_id, s in items[: self.SEED_TAIL]:
            status = s["status"]
            if status == "ok":
                status_t = Text("✓ ok", style="green")
            elif status == "fail":
                status_t = Text("✗ fail", style="red")
            elif status == "running":
                status_t = Text("▶ run", style="cyan")
            else:
                status_t = Text("⏳ que", style="yellow")
            gpu_id = s.get("gpu_id")
            gpu_t = Text(f"G{gpu_id}", style="dim") if gpu_id is not None else Text("—", style="dim")
            score = s.get("score")
            elapsed_s = s.get("elapsed_s")
            tbl.add_row(
                str(exec_id),
                gpu_t,
                status_t,
                f"{score:.3f}" if score is not None else "—",
                f"{elapsed_s:5.1f}s" if elapsed_s is not None else "—",
            )

        body = Table.grid(expand=True)
        body.add_column()
        body.add_row(summary)
        body.add_row(tbl)
        return Panel(body, title=f"seeds (parallel · {n})", border_style="magenta")

    def _render_history(self):
        from rich.panel import Panel
        from rich.table import Table

        tbl = Table(expand=True, box=None, show_header=True, header_style="dim")
        tbl.add_column("iter", width=4, justify="right")
        tbl.add_column("obs", width=7, justify="right")
        tbl.add_column("author", width=8, justify="right")
        tbl.add_column("assembly", width=8, justify="right")
        tbl.add_column("exec", width=7, justify="right")
        tbl.add_column("refl", width=7, justify="right")
        tbl.add_column("S/All", width=8, justify="right")
        tbl.add_column("rate", width=5, justify="right")
        tbl.add_column("avg", width=5, justify="right")
        for h in self._iter_history:
            st = h["step_times"]
            # Back-compat: older runs emit "code_generator"; two-stage runs
            # emit "skill_author" + "assembly_generator". Fall back so the
            # single-stage time still surfaces under `author` if present.
            author_ms = st.get("skill_author", st.get("code_generator"))
            assembly_ms = st.get("assembly_generator")
            tbl.add_row(
                str(h["iter"]),
                _fmt_ms(st.get("observer")),
                _fmt_ms(author_ms),
                _fmt_ms(assembly_ms),
                _fmt_ms(st.get("executor")),
                _fmt_ms(st.get("self_reflection")),
                f"{h['successes']}/{h['n_seeds']}",
                f"{h['success_rate']:.2f}",
                f"{h['avg_score']:.2f}",
            )
        return Panel(tbl, title="iteration history", border_style="blue")


def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"
