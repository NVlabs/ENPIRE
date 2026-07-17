"""Rich terminal dashboard for the PLD learner."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _tui_console() -> Console:
    fd_text = os.environ.get("PLD_RUN_LOG_ORIGINAL_STDERR_FD")
    if fd_text:
        try:
            stream = os.fdopen(os.dup(int(fd_text)), "w", buffering=1)
            force_terminal = os.environ.get("PLD_RUN_LOG_STDERR_ISATTY") == "1"
            return Console(file=stream, force_terminal=force_terminal)
        except Exception:
            pass
    return Console(stderr=True)


class LearnerTUI:
    """Small live dashboard for learner-side data flow.

    The dashboard is intentionally read-only: it samples buffer lengths,
    ingestor stats, disk inventory, and callbacks from the Agentlace request
    handler without changing the training loop.
    """

    def __init__(
        self,
        *,
        snapshot_fn: Callable[[], dict[str, Any]],
        refresh_hz: float = 2.0,
        actor_timeout_s: float = 15.0,
        enabled: bool = True,
    ) -> None:
        self.snapshot_fn = snapshot_fn
        self.refresh_hz = max(0.2, float(refresh_hz))
        self.actor_timeout_s = float(actor_timeout_s)
        self.enabled = bool(enabled) and sys.stderr.isatty()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._phase = "starting"
        self._step = 0
        self._last_actor_stats_t: float | None = None
        self._last_actor_payload: dict[str, Any] | None = None
        self._last_actor_action_t: float | None = None
        self._last_actor_action: dict[str, Any] | None = None
        self._last_update_t: float | None = None
        self._last_update_step: int | None = None
        self._last_update_metrics: dict[str, float] = {}
        self._last_replay_n = 0
        self._last_demo_n = 0
        self._last_replay_growth_t: float | None = None
        self._last_demo_growth_t: float | None = None
        self._started_t = time.monotonic()

    @staticmethod
    def from_config(config: Any, snapshot_fn: Callable[[], dict[str, Any]]) -> "LearnerTUI | None":
        if os.environ.get("PLD_LEARNER_TUI", "").strip() == "0":
            return None
        enabled = bool(config.get("learner_tui", True))
        refresh_hz = float(config.get("learner_tui_refresh_hz", 2.0))
        actor_timeout_s = float(config.get("learner_tui_actor_timeout_s", 15.0))
        tui = LearnerTUI(
            snapshot_fn=snapshot_fn,
            refresh_hz=refresh_hz,
            actor_timeout_s=actor_timeout_s,
            enabled=enabled,
        )
        return tui if tui.enabled else None

    def start(self) -> None:
        if not self.enabled or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pld-learner-tui")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase)

    def set_step(self, step: int) -> None:
        with self._lock:
            self._step = int(step)

    def record_actor_stats(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._last_actor_stats_t = time.monotonic()
            self._last_actor_payload = payload
            actor_action = payload.get("actor_action")
            if isinstance(actor_action, dict):
                self._last_actor_action_t = self._last_actor_stats_t
                self._last_actor_action = actor_action

    def record_update_info(self, step: int, update_info: dict[str, Any]) -> None:
        with self._lock:
            self._last_update_t = time.monotonic()
            self._last_update_step = int(step)
            self._last_update_metrics = self._numeric_metrics(update_info)

    def _run(self) -> None:
        console = _tui_console()
        with Live(
            self._render(),
            refresh_per_second=self.refresh_hz,
            transient=False,
            screen=False,
            console=console,
        ) as live:
            while not self._stop.wait(1.0 / self.refresh_hz):
                live.update(self._render())

    def _render(self) -> Panel:
        snapshot = self._safe_snapshot()
        now = time.monotonic()
        with self._lock:
            phase = self._phase
            step = self._step
            last_actor_stats_t = self._last_actor_stats_t
            last_actor_payload = self._last_actor_payload
            last_actor_action_t = self._last_actor_action_t
            last_actor_action = self._last_actor_action
            last_update_t = self._last_update_t
            last_update_step = self._last_update_step
            last_update_metrics = dict(self._last_update_metrics)

        replay_n = int(snapshot.get("replay_size", 0))
        demo_n = int(snapshot.get("demo_size", 0))
        if replay_n > self._last_replay_n:
            self._last_replay_growth_t = now
        if demo_n > self._last_demo_n:
            self._last_demo_growth_t = now
        self._last_replay_n = replay_n
        self._last_demo_n = demo_n

        actor_recent = self._recent(last_actor_stats_t, now)
        replay_recent = self._recent(self._last_replay_growth_t, now)
        demo_recent = self._recent(self._last_demo_growth_t, now)
        # Replay growth can come from DiskBufferIngestor, so do not treat it as
        # actor connectivity. Actor status is based only on explicit actor stats.
        actor_status = "connected" if actor_recent else "no stats"
        actor_style = "green" if actor_recent else "yellow"

        disk = snapshot.get("disk", {}) or {}
        rsync = snapshot.get("rsync", {}) or {}
        ingestor = snapshot.get("ingestor", {}) or {}
        profile = snapshot.get("profile", {}) or {}
        routed = ingestor.get("routed_per_label", {}) or {}
        action_range = ingestor.get("action_range", {}) or {}
        train_gate = snapshot.get("train_gate", {}) or {}
        gate_started = bool(train_gate.get("started", False))
        replay_ready = bool(train_gate.get("replay_ready", False))
        mc_required = bool(train_gate.get("mc_required", False))
        mc_ready = bool(train_gate.get("mc_ready", False))
        if gate_started:
            gate_text = "[green]training enabled[/green]"
        elif not replay_ready:
            gate_text = "[yellow]waiting replay[/yellow]"
        elif mc_required and not mc_ready:
            gate_text = "[yellow]waiting complete episode[/yellow]"
        else:
            gate_text = "[yellow]initializing[/yellow]"

        last_error = ingestor.get("last_error") or disk.get("last_error")
        loss_panel = Panel(
            self._render_loss_info(last_update_metrics, last_update_step, last_update_t, now),
            title="Losses",
            border_style="green" if self._recent(last_update_t, now) else "yellow",
        )
        training_panel = Panel(
            self._render_training_status(
                phase=phase,
                step=step,
                gate_text=gate_text,
                train_gate=train_gate,
                mc_ready=mc_ready,
                actor_status=actor_status,
                actor_style=actor_style,
                last_actor_stats_t=last_actor_stats_t,
                profile=profile,
                sync_jax=snapshot.get("profile_sync_jax", False),
                now=now,
            ),
            title="Learning",
            border_style="cyan",
        )
        buffer_panel = Panel(
            self._render_buffer_status(
                snapshot=snapshot,
                replay_n=replay_n,
                demo_n=demo_n,
                replay_growth_t=self._last_replay_growth_t,
                demo_growth_t=self._last_demo_growth_t,
                now=now,
            ),
            title="Buffers",
            border_style="green" if gate_started else "yellow",
        )
        data_panel = Panel(
            self._render_data_pipeline_status(
                disk=disk,
                ingestor=ingestor,
                rsync=rsync,
                disk_root=str(snapshot.get("disk_root", "")),
                replay_growth_t=self._last_replay_growth_t,
                demo_growth_t=self._last_demo_growth_t,
                now=now,
            ),
            title="Data Pipeline",
            border_style=self._sync_data_border_style(disk, ingestor, snapshot),
        )
        error_text = Text(str(last_error), overflow="fold") if last_error else Text("none")
        error_panel = Panel(
            error_text,
            title="Last Error",
            border_style="red" if last_error else "green",
        )
        action_range_panel = Panel(
            self._render_action_range_status(action_range),
            title="Action Range",
            border_style=self._action_range_border_style(action_range),
        )
        actor_action_panel = Panel(
            self._render_actor_action(last_actor_action, last_actor_action_t, now),
            title="Last Actor Action",
            border_style=actor_style,
        )

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(training_panel, Group(buffer_panel, action_range_panel))
        grid.add_row(data_panel, loss_panel)
        grid.add_row(error_panel, actor_action_panel)
        return Panel(
            Align.left(Group(grid)),
            title="PLD Learner TUI",
            border_style="bright_blue",
        )

    def _safe_snapshot(self) -> dict[str, Any]:
        try:
            return dict(self.snapshot_fn())
        except Exception as e:
            return {"snapshot_error": repr(e)}

    def _render_training_status(
        self,
        *,
        phase: str,
        step: int,
        gate_text: str,
        train_gate: dict[str, Any],
        mc_ready: bool,
        actor_status: str,
        actor_style: str,
        last_actor_stats_t: float | None,
        profile: dict[str, Any],
        sync_jax: Any,
        now: float,
    ) -> Table:
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        step_s = float(profile.get("last_step_duration_s", 0.0) or 0.0)
        updates = int(profile.get("updates", 0) or 0)
        table.add_row("phase", f"[bold]{phase}[/bold]")
        table.add_row("step", str(step))
        table.add_row("gate", gate_text)
        table.add_row(
            "replay gate",
            f"{train_gate.get('replay_ready', False)} ({train_gate.get('training_starts', '?')} start)",
        )
        table.add_row("valid MC", "ready" if mc_ready else "waiting")
        table.add_row("uptime", self._fmt_age(now - self._started_t))
        table.add_row("updates", str(updates))
        table.add_row("last iter", self._fmt_seconds(step_s))
        table.add_row("iter speed", f"{1.0 / step_s:.3f} Hz" if step_s > 0.0 else "?")
        table.add_row("sample / iter", self._fmt_seconds(profile.get("last_sample_duration_s")))
        table.add_row(
            "critic / iter", self._fmt_seconds(profile.get("last_critic_update_duration_s"))
        )
        table.add_row(
            "actor-temp / iter", self._fmt_seconds(profile.get("last_train_update_duration_s"))
        )
        table.add_row("JAX sync timing", str(bool(sync_jax)))
        table.add_row("actor", f"[{actor_style}]{actor_status}[/{actor_style}]")
        table.add_row("last actor stats", self._fmt_since(last_actor_stats_t, now))
        table.add_row("model publishes", str(profile.get("publish_count", 0)))
        table.add_row("last publish step", str(profile.get("last_publish_step", "?")))
        table.add_row("publish / iter", self._fmt_seconds(profile.get("last_publish_duration_s")))
        table.add_row("publish interval", self._fmt_seconds(profile.get("last_publish_interval_s")))
        last_publish_epoch = profile.get("last_publish_epoch")
        if isinstance(last_publish_epoch, (int, float)):
            table.add_row(
                "last publish age",
                self._fmt_seconds(time.time() - float(last_publish_epoch), suffix=" ago"),
            )
        return table

    def _render_buffer_status(
        self,
        *,
        snapshot: dict[str, Any],
        replay_n: int,
        demo_n: int,
        replay_growth_t: float | None,
        demo_growth_t: float | None,
        now: float,
    ) -> Table:
        table = Table(expand=True)
        table.add_column("buffer")
        table.add_column("size", justify="right")
        table.add_column("cap", justify="right")
        table.add_column("success", justify="right")
        table.add_column("valid MC", justify="right")
        table.add_column("last growth", justify="right")
        table.add_row(
            "replay",
            str(replay_n),
            str(snapshot.get("replay_capacity", "?")),
            str(snapshot.get("replay_success", "?")),
            str(snapshot.get("replay_valid_mc", "?")),
            self._fmt_since(replay_growth_t, now),
        )
        table.add_row(
            "demo",
            str(demo_n),
            str(snapshot.get("demo_capacity", "?")),
            str(snapshot.get("demo_success", "?")),
            str(snapshot.get("demo_valid_mc", "?")),
            self._fmt_since(demo_growth_t, now),
        )
        return table

    def _render_data_pipeline_status(
        self,
        *,
        disk: dict[str, Any],
        ingestor: dict[str, Any],
        rsync: dict[str, Any],
        disk_root: str,
        replay_growth_t: float | None,
        demo_growth_t: float | None,
        now: float,
    ) -> Table:
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        root = str(disk.get("root", disk_root) or "")
        sync_active = bool(disk.get("sync_in_progress", False))
        errors = int(ingestor.get("errors", 0) or disk.get("errors", 0) or 0)
        active_error = bool(ingestor.get("last_error") or disk.get("last_error"))
        if not root:
            status = "[cyan]not configured[/cyan]"
        elif not bool(disk.get("root_exists", False)):
            status = "[red]missing root[/red]"
        elif sync_active:
            status = "[yellow]upload active[/yellow]"
        elif active_error:
            status = "[red]error[/red]"
        else:
            status = "[green]watching[/green]"

        load_s = float(ingestor.get("last_load_duration_s", 0.0) or 0.0)
        insert_s = float(ingestor.get("last_insert_duration_s", 0.0) or 0.0)
        episode_s = float(ingestor.get("last_episode_duration_s", 0.0) or 0.0)
        scan_s = float(ingestor.get("last_scan_duration_s", 0.0) or 0.0)
        last_transitions = int(ingestor.get("last_episode_transitions", 0) or 0)
        last_buffer_growth = max(
            [t for t in (replay_growth_t, demo_growth_t) if t is not None],
            default=None,
        )
        total_transitions = int(ingestor.get("transitions", 0) or 0)
        total_pipeline_s = float(ingestor.get("total_load_duration_s", 0.0) or 0.0) + float(
            ingestor.get("total_insert_duration_s", 0.0) or 0.0
        )
        trans_per_s = total_transitions / total_pipeline_s if total_pipeline_s > 0 else None

        table.add_row("status", status)
        table.add_row("root", self._short_path(root))
        table.add_row(
            "last disk episode",
            self._fmt_seconds(disk.get("seconds_since_latest_complete"), suffix=" ago"),
        )
        table.add_row(
            "last pending episode",
            self._fmt_seconds(disk.get("seconds_since_latest_pending"), suffix=" ago"),
        )
        table.add_row("last buffer growth", self._fmt_since(last_buffer_growth, now))
        table.add_row(
            "upload/rsync age",
            self._fmt_seconds(rsync.get("seconds_since_last_end"), suffix=" ago"),
        )
        table.add_row("upload/rsync iter", self._fmt_seconds(rsync.get("last_duration_s")))
        table.add_row("upload/rsync period", self._fmt_seconds(rsync.get("interval_s")))
        table.add_row("pending episodes", str(disk.get("pending_episodes", 0)))
        table.add_row("pending transitions", str(disk.get("pending_estimated_transitions", 0)))
        table.add_row(
            "ingested / complete eps",
            f"{disk.get('ingested_episodes', 0)}/{disk.get('complete_episodes', 0)}",
        )
        table.add_row("loaded transitions", str(total_transitions))
        nonzero = int(ingestor.get("nonzero_reward_transitions", 0) or 0)
        if total_transitions > 0:
            pct = 100.0 * nonzero / total_transitions
            nonzero_text = f"{nonzero} ({pct:.1f}%)"
        else:
            nonzero_text = str(nonzero)
        table.add_row("nonzero reward trans", nonzero_text)
        table.add_row("scan / iter", self._fmt_seconds(scan_s))
        table.add_row("episode load / iter", self._fmt_seconds(load_s))
        table.add_row("buffer insert / iter", self._fmt_seconds(insert_s))
        table.add_row("total ingest / iter", self._fmt_seconds(episode_s))
        table.add_row("last ep transitions", str(last_transitions))
        table.add_row("avg load+insert speed", f"{trans_per_s:.1f} trans/s" if trans_per_s else "?")
        table.add_row("errors", str(errors))
        if errors and not active_error:
            table.add_row("last error", "[green]recovered[/green]")
        return table

    def _render_action_range_status(self, stats: dict[str, Any]) -> Table | Text:
        if not stats:
            return Text("not configured", style="dim")
        if not bool(stats.get("enabled", False)):
            return Text("not checked; no inverse action scaling", style="dim")

        checked = int(stats.get("checked_total", 0) or 0)
        demo_q01 = stats.get("demo_q01")
        demo_q99 = stats.get("demo_q99")
        has_demo_q = demo_q01 is not None
        if checked <= 0 and not has_demo_q:
            return Text("waiting for normalized disk actions", style="dim")

        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        if checked > 0:
            out_total = int(stats.get("out_of_range_total", 0) or 0)
            by_buffer = stats.get("out_of_range_by_buffer", {}) or {}
            checked_by_buffer = stats.get("checked_by_buffer", {}) or {}
            demo_out = int(by_buffer.get("demo", 0) or 0)
            replay_out = int(by_buffer.get("replay", 0) or 0)
            if out_total:
                table.add_row("status", "[bold red]OUT OF RANGE[/bold red]")
                table.add_row("demo violations", f"[red]{demo_out}[/red]")
                table.add_row("replay violations", f"[red]{replay_out}[/red]")
            else:
                table.add_row("status", "[green]ok[/green]")
                table.add_row("demo checked", str(checked_by_buffer.get("demo", 0)))
                table.add_row("replay checked", str(checked_by_buffer.get("replay", 0)))
            table.add_row("checked actions", str(checked))
            table.add_row("max |a| seen", self._fmt_float(stats.get("max_abs")))
            table.add_row("last ep max |a|", self._fmt_float(stats.get("last_max_abs")))
            last = stats.get("last")
            if isinstance(last, dict) and last:
                detail = (
                    f"{last.get('buffer', '?')}:{last.get('label', '?')} "
                    f"idx {last.get('transition_index', '?')} "
                    f"a[{last.get('max_index', '?')}]={self._fmt_float(last.get('value'))}"
                )
                table.add_row("last violation", f"[red]{detail}[/red]")
                table.add_row(
                    "episode", self._short_path(str(last.get("episode", "")), max_chars=36)
                )
        table.add_row("demo q01 |a|", self._fmt_float(demo_q01))
        table.add_row("demo q99 |a|", self._fmt_float(demo_q99))
        table.add_row("last ep q01 |a|", self._fmt_float(stats.get("last_demo_q01")))
        table.add_row("last ep q99 |a|", self._fmt_float(stats.get("last_demo_q99")))
        return table

    @staticmethod
    def _action_range_border_style(stats: dict[str, Any]) -> str:
        if not stats:
            return "cyan"
        if not bool(stats.get("enabled", False)):
            return "cyan"
        if int(stats.get("out_of_range_total", 0) or 0):
            return "red"
        if int(stats.get("checked_total", 0) or 0):
            return "green"
        return "yellow"

    def _recent(self, timestamp: float | None, now: float) -> bool:
        return timestamp is not None and (now - timestamp) <= self.actor_timeout_s

    def _render_actor_action(
        self,
        actor_action: dict[str, Any] | None,
        timestamp: float | None,
        now: float,
    ) -> Table | Text:
        if not actor_action:
            return Text("no right-arm action stats received yet", overflow="fold")
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_row("age", self._fmt_since(timestamp, now))
        table.add_row("step", str(actor_action.get("step", "?")))
        table.add_row("source", str(actor_action.get("source", "?")))
        table.add_row("action dim", str(actor_action.get("action_dim", "?")))
        table.add_row("dx sent", self._fmt_float(actor_action.get("right_dx")))
        table.add_row("dy sent", self._fmt_float(actor_action.get("right_dy")))
        table.add_row("dz sent", self._fmt_float(actor_action.get("right_dz")))
        table.add_row("dx norm", self._fmt_float(actor_action.get("right_dx_norm")))
        table.add_row("dy norm", self._fmt_float(actor_action.get("right_dy_norm")))
        table.add_row("dz norm", self._fmt_float(actor_action.get("right_dz_norm")))
        return table

    def _render_sync_data_status(
        self,
        *,
        disk: dict[str, Any],
        ingestor: dict[str, Any],
        disk_root: str,
        replay_recent: bool,
        demo_recent: bool,
        now: float,
        rsync: dict[str, Any] | None = None,
    ) -> Table:
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        root = str(disk.get("root", disk_root) or "")
        rsync = rsync or {}
        has_disk = bool(root)
        sync_active = bool(disk.get("sync_in_progress", False))
        errors = int(ingestor.get("errors", 0) or disk.get("errors", 0) or 0)
        active_error = bool(ingestor.get("last_error") or disk.get("last_error"))
        if not has_disk:
            status = "[cyan]not configured[/cyan]"
        elif sync_active:
            status = "[yellow]rsync active; scan paused[/yellow]"
        elif active_error:
            status = "[red]error[/red]"
        elif replay_recent or demo_recent:
            status = "[green]loading[/green]"
        else:
            status = "[green]watching[/green]"
        table.add_row("status", status)
        table.add_row("root", root)
        table.add_row("root exists", str(disk.get("root_exists", False) if has_disk else "n/a"))
        table.add_row("rsync lock", "active" if sync_active else ("clear" if has_disk else "n/a"))
        table.add_row("rsync status", str(rsync.get("last_status", "unknown")))
        table.add_row("rsync last duration", self._fmt_seconds(rsync.get("last_duration_s")))
        table.add_row("rsync last exit", str(rsync.get("last_exit_code", "?")))
        table.add_row(
            "rsync last end age",
            self._fmt_seconds(rsync.get("seconds_since_last_end"), suffix=" ago"),
        )
        table.add_row("rsync interval", self._fmt_seconds(rsync.get("interval_s")))
        table.add_row("complete episodes", str(disk.get("complete_episodes", 0)))
        table.add_row("pending episodes", str(disk.get("pending_episodes", 0)))
        table.add_row("ingested episodes", str(disk.get("ingested_episodes", 0)))
        table.add_row(
            "pending transitions",
            str(disk.get("pending_estimated_transitions", 0)),
        )
        table.add_row("scans", str(ingestor.get("scans", 0)))
        table.add_row("last scan", self._fmt_seconds(ingestor.get("last_scan_duration_s")))
        table.add_row("last discover", self._fmt_seconds(ingestor.get("last_discover_duration_s")))
        table.add_row("last episode load", self._fmt_seconds(ingestor.get("last_load_duration_s")))
        table.add_row(
            "last buffer insert", self._fmt_seconds(ingestor.get("last_insert_duration_s"))
        )
        table.add_row("last ep transitions", str(ingestor.get("last_episode_transitions", 0)))
        table.add_row("loaded episodes", str(ingestor.get("episodes", 0)))
        table.add_row("loaded transitions", str(ingestor.get("transitions", 0)))
        table.add_row("errors", str(errors))
        if errors and not active_error:
            table.add_row("last error", "[green]recovered[/green]")
        table.add_row("last replay growth", self._fmt_since(self._last_replay_growth_t, now))
        table.add_row("last demo growth", self._fmt_since(self._last_demo_growth_t, now))
        return table

    def _render_profile_info(self, profile: dict[str, Any], sync_jax: Any) -> Table:
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        updates = int(profile.get("updates", 0) or 0)
        step_s = float(profile.get("last_step_duration_s", 0.0) or 0.0)
        table.add_row("jax timing sync", str(bool(sync_jax)))
        table.add_row("updates", str(updates))
        table.add_row("critic updates", str(profile.get("critic_updates", 0)))
        table.add_row("last learner step", self._fmt_seconds(step_s))
        table.add_row("last sample", self._fmt_seconds(profile.get("last_sample_duration_s")))
        table.add_row(
            "last critic update",
            self._fmt_seconds(profile.get("last_critic_update_duration_s")),
        )
        table.add_row(
            "last actor/temp update",
            self._fmt_seconds(profile.get("last_train_update_duration_s")),
        )
        if step_s > 0.0:
            table.add_row("learner step freq", f"{1.0 / step_s:.3f} Hz")
        table.add_row("model publishes", str(profile.get("publish_count", 0)))
        table.add_row("last publish step", str(profile.get("last_publish_step", "?")))
        table.add_row("last publish", self._fmt_seconds(profile.get("last_publish_duration_s")))
        table.add_row("publish interval", self._fmt_seconds(profile.get("last_publish_interval_s")))
        last_publish_epoch = profile.get("last_publish_epoch")
        if isinstance(last_publish_epoch, (int, float)):
            table.add_row(
                "last publish age",
                self._fmt_seconds(time.time() - float(last_publish_epoch), suffix=" ago"),
            )
        return table

    @staticmethod
    def _sync_data_border_style(
        disk: dict[str, Any], ingestor: dict[str, Any], snapshot: dict[str, Any]
    ) -> str:
        disk_root = str(snapshot.get("disk_root", "") or disk.get("root", "") or "")
        if not disk_root:
            return "cyan"
        if ingestor.get("last_error") or disk.get("last_error"):
            return "red"
        if bool(disk.get("sync_in_progress", False)):
            return "yellow"
        if disk.get("root_exists", True) is False:
            return "red"
        return "green"

    def _render_buffer_action_stats(self, stats: dict[str, Any]) -> Table:
        table = Table(expand=True)
        table.add_column("buffer")
        table.add_column("n", justify="right")
        table.add_column("dim", justify="right")
        table.add_column("min", justify="right")
        table.add_column("mean", justify="right")
        table.add_column("max", justify="right")
        table.add_column("|mean|", justify="right")
        table.add_column("|max|", justify="right")
        for name, label in (("rollout", "rollout"), ("demo", "demo")):
            item = stats.get(name, {}) or {}
            table.add_row(
                label,
                str(item.get("count", 0)),
                str(item.get("dim", "?")),
                self._fmt_float(item.get("min")),
                self._fmt_float(item.get("mean")),
                self._fmt_float(item.get("max")),
                self._fmt_float(item.get("abs_mean")),
                self._fmt_float(item.get("abs_max")),
            )
        return table

    def _render_loss_info(
        self,
        metrics: dict[str, float],
        step: int | None,
        timestamp: float | None,
        now: float,
    ) -> Table | Text:
        if not metrics:
            return Text("no learner update metrics yet", overflow="fold")
        table = Table(expand=True)
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_row("age", self._fmt_since(timestamp, now))
        table.add_row("step", str(step if step is not None else "?"))

        def metric_key(item: tuple[str, float]) -> tuple[int, str]:
            name = item[0].lower()
            if "loss" in name:
                priority = 0
            elif any(token in name for token in ("alpha", "sigma", "entropy", "temperature")):
                priority = 1
            else:
                priority = 2
            return (priority, item[0])

        for name, value in sorted(metrics.items(), key=metric_key)[:16]:
            table.add_row(name, self._fmt_float(value))
        return table

    @classmethod
    def _numeric_metrics(cls, value: Any, prefix: str = "") -> dict[str, float]:
        metrics: dict[str, float] = {}
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}/{key}" if prefix else str(key)
                metrics.update(cls._numeric_metrics(child, child_prefix))
            return metrics
        try:
            if hasattr(value, "item"):
                value = value.item()
            number = float(value)
        except (TypeError, ValueError):
            return metrics
        if number == number and number not in (float("inf"), float("-inf")):
            metrics[prefix or "value"] = number
        return metrics

    @staticmethod
    def _fmt_since(timestamp: float | None, now: float) -> str:
        if timestamp is None:
            return "never"
        return f"{now - timestamp:.1f}s ago"

    @staticmethod
    def _fmt_float(value: Any) -> str:
        try:
            return f"{float(value):+.5f}"
        except (TypeError, ValueError):
            return "?"

    @staticmethod
    def _fmt_seconds(value: Any, suffix: str = "") -> str:
        try:
            return f"{float(value):.3f}s{suffix}"
        except (TypeError, ValueError):
            return "?"

    @staticmethod
    def _short_path(value: str, max_chars: int = 56) -> str:
        if len(value) <= max_chars:
            return value
        keep = max(8, max_chars - 3)
        return "..." + value[-keep:]

    @staticmethod
    def _fmt_age(seconds: float) -> str:
        seconds_i = max(0, int(seconds))
        minutes, sec = divmod(seconds_i, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m {sec:02d}s"
        return f"{minutes:02d}m {sec:02d}s"
