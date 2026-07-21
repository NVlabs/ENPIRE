"""Rich terminal dashboard for the PLD actor."""

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


class ActorTUI:
    """Read-only dashboard for actor-side robot, learner, and action flow."""

    def __init__(
        self,
        *,
        snapshot_fn: Callable[[], dict[str, Any]],
        refresh_hz: float = 5.0,
        timeout_s: float = 15.0,
        enabled: bool = True,
    ) -> None:
        self.snapshot_fn = snapshot_fn
        self.refresh_hz = max(0.2, float(refresh_hz))
        self.timeout_s = float(timeout_s)
        self.enabled = bool(enabled) and sys.stderr.isatty()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started_t = time.monotonic()
        self._phase = "starting"
        self._step = 0
        self._episode = 0
        self._last_action_t: float | None = None
        self._last_action: dict[str, Any] | None = None
        self._last_env_obs_t: float | None = None
        self._last_env_step_t: float | None = None
        self._last_learner_t: float | None = None
        self._last_network_t: float | None = None
        self._network_update_count = 0
        self._last_network_interval_s: float | None = None
        self._total_network_interval_s = 0.0
        self._learner_state = "starting"
        self._learner_error: str | None = None
        self._last_episode_info: dict[str, Any] | None = None

    @staticmethod
    def from_config(config: Any, snapshot_fn: Callable[[], dict[str, Any]]) -> "ActorTUI | None":
        if os.environ.get("PLD_ACTOR_TUI", "").strip() == "0":
            return None
        train_cfg = getattr(config, "train", None)
        enabled = bool(
            config.get("actor_tui", train_cfg.get("actor_tui", True) if train_cfg else True)
        )
        refresh_hz = float(
            config.get(
                "actor_tui_refresh_hz",
                train_cfg.get("actor_tui_refresh_hz", 5.0) if train_cfg else 5.0,
            )
        )
        timeout_s = float(
            config.get(
                "actor_tui_timeout_s",
                train_cfg.get("actor_tui_timeout_s", 15.0) if train_cfg else 15.0,
            )
        )
        tui = ActorTUI(
            snapshot_fn=snapshot_fn,
            refresh_hz=refresh_hz,
            timeout_s=timeout_s,
            enabled=enabled,
        )
        return tui if tui.enabled else None

    def start(self) -> None:
        if not self.enabled or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pld-actor-tui")
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

    def set_episode(self, episode: int) -> None:
        with self._lock:
            self._episode = int(episode)

    def record_action(self, action: dict[str, Any]) -> None:
        with self._lock:
            self._last_action_t = time.monotonic()
            self._last_action = dict(action)
            self._step = int(action.get("step", self._step))

    def record_env_observation(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._last_env_obs_t = now
            self._last_env_step_t = now

    def record_env_step(self, *, done: bool, truncated: bool, reward: float) -> None:
        with self._lock:
            self._last_env_step_t = time.monotonic()
            if done or truncated:
                self._last_episode_info = {
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "reward": float(reward),
                    "step": self._step,
                    "episode": self._episode,
                }

    def record_learner_connected(self) -> None:
        with self._lock:
            self._learner_state = "connected"
            self._learner_error = None
            self._last_learner_t = time.monotonic()

    def record_learner_disabled(self, reason: str) -> None:
        with self._lock:
            self._learner_state = "disabled"
            self._learner_error = str(reason)
            self._last_learner_t = None

    def record_learner_error(self, error: Any) -> None:
        with self._lock:
            self._learner_state = "error"
            self._learner_error = repr(error)
            self._last_learner_t = None

    def record_learner_request(self) -> None:
        with self._lock:
            self._last_learner_t = time.monotonic()

    def record_network_update(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_network_t is not None:
                self._last_network_interval_s = now - self._last_network_t
                self._total_network_interval_s += self._last_network_interval_s
            self._network_update_count += 1
            self._last_network_t = now
            self._last_learner_t = now
            if self._learner_state != "disabled":
                self._learner_state = "connected"

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
            episode = self._episode
            last_action_t = self._last_action_t
            last_action = dict(self._last_action) if self._last_action else None
            last_env_obs_t = self._last_env_obs_t
            last_env_step_t = self._last_env_step_t
            last_learner_t = self._last_learner_t
            last_network_t = self._last_network_t
            network_update_count = self._network_update_count
            last_network_interval_s = self._last_network_interval_s
            total_network_interval_s = self._total_network_interval_s
            learner_state = self._learner_state
            learner_error = self._learner_error
            last_episode_info = self._last_episode_info

        env_status = snapshot.get("env", {}) or {}
        actor_buffer = snapshot.get("actor_buffer", {}) or {}
        learner_recent = self._recent(last_learner_t, now) or self._recent(last_network_t, now)
        env_recent = self._recent(last_env_step_t, now)
        learner_style = (
            "green"
            if learner_state == "connected" and learner_recent
            else "yellow"
            if learner_state in ("starting", "disabled")
            else "red"
        )
        env_style = "green" if env_recent else "yellow"

        status_panel = Panel(
            self._render_actor_status(
                phase=phase,
                step=step,
                episode=episode,
                snapshot=snapshot,
                now=now,
            ),
            title="Actor",
            border_style="cyan",
        )
        network_panel = Panel(
            self._render_network_status(
                learner_state=learner_state,
                learner_style=learner_style,
                learner_error=learner_error,
                last_learner_t=last_learner_t,
                last_network_t=last_network_t,
                network_update_count=network_update_count,
                last_network_interval_s=last_network_interval_s,
                total_network_interval_s=total_network_interval_s,
                now=now,
            ),
            title="Network Sync",
            border_style=learner_style,
        )
        robot_panel = Panel(
            self._render_robot_status(
                env_status=env_status,
                actor_buffer=actor_buffer,
                env_style=env_style,
                env_recent=env_recent,
                last_env_obs_t=last_env_obs_t,
                last_env_step_t=last_env_step_t,
                last_episode_info=last_episode_info,
                now=now,
            ),
            title="Robot / Queues",
            border_style=env_style,
        )
        action_panel = Panel(
            self._render_action(last_action, last_action_t, now),
            title="Action",
            border_style="green" if self._recent(last_action_t, now) else "yellow",
        )
        rsync_status = snapshot.get("rsync", {}) or {}
        rsync_panel = Panel(
            self._render_rsync_status(rsync_status, bool(snapshot.get("send_online_data", True))),
            title="Local Rsync",
            border_style=self._rsync_border_style(rsync_status),
        )

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(status_panel, network_panel)
        grid.add_row(robot_panel, action_panel)
        grid.add_row(rsync_panel, Panel(""))
        return Panel(
            Align.left(Group(grid)),
            title="PLD Actor TUI",
            border_style="bright_blue",
        )

    def _safe_snapshot(self) -> dict[str, Any]:
        try:
            return dict(self.snapshot_fn())
        except Exception as e:
            return {"snapshot_error": repr(e)}

    def _render_actor_status(
        self,
        *,
        phase: str,
        step: int,
        episode: int,
        snapshot: dict[str, Any],
        now: float,
    ) -> Table:
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_row("phase", f"[bold]{phase}[/bold]")
        table.add_row("step", str(step))
        table.add_row("episode", str(episode))
        table.add_row("uptime", self._fmt_age(now - self._started_t))
        table.add_row("task", str(snapshot.get("instruction", "")))
        table.add_row("action repr", str(snapshot.get("action_repr", "?")))
        table.add_row("control mode", str(snapshot.get("control_mode", "?")))
        table.add_row(
            "portal address",
            str(snapshot.get("remote_server_address") or snapshot.get("remote_port", "?")),
        )
        return table

    def _render_network_status(
        self,
        *,
        learner_state: str,
        learner_style: str,
        learner_error: str | None,
        last_learner_t: float | None,
        last_network_t: float | None,
        network_update_count: int,
        last_network_interval_s: float | None,
        total_network_interval_s: float,
        now: float,
    ) -> Table:
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_row("learner", f"[{learner_style}]{learner_state}[/{learner_style}]")
        table.add_row("last learner contact", self._fmt_since(last_learner_t, now))
        table.add_row("last network update", self._fmt_since(last_network_t, now))
        table.add_row("network updates", str(network_update_count))
        table.add_row("last update interval", self._fmt_seconds(last_network_interval_s))
        if network_update_count > 1:
            avg_interval = total_network_interval_s / max(1, network_update_count - 1)
            table.add_row("avg update interval", self._fmt_seconds(avg_interval))
            table.add_row(
                "avg update freq",
                f"{1.0 / avg_interval:.3f} Hz" if avg_interval > 0 else "?",
            )
        table.add_row("error/mode", str(learner_error or "none"))
        return table

    def _render_robot_status(
        self,
        *,
        env_status: dict[str, Any],
        actor_buffer: dict[str, Any],
        env_style: str,
        env_recent: bool,
        last_env_obs_t: float | None,
        last_env_step_t: float | None,
        last_episode_info: dict[str, Any] | None,
        now: float,
    ) -> Table:
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_row(
            "robot env",
            f"[{env_style}]{'connected' if env_recent else 'waiting'}[/{env_style}]",
        )
        table.add_row("last robot step", self._fmt_since(last_env_step_t, now))
        table.add_row(
            "last robot obs",
            self._fmt_since(last_env_obs_t, now),
        )
        table.add_row(
            "last action sent",
            self._fmt_since(env_status.get("last_action_sent_t"), now),
        )
        table.add_row(
            "obs -> action",
            self._fmt_seconds(env_status.get("last_obs_to_action_s")),
        )
        table.add_row(
            "avg obs -> action",
            self._fmt_seconds(env_status.get("avg_obs_to_action_s")),
        )
        table.add_row("obs seen", str(env_status.get("last_obs_seen", last_env_obs_t is not None)))
        table.add_row("remote steps", str(env_status.get("step_count", "?")))
        table.add_row("obs queue", str(env_status.get("obs_queue_size", "?")))
        table.add_row("action queue", str(env_status.get("action_queue_size", "?")))
        table.add_row("actor_env queued", str(actor_buffer.get("actor_env", "?")))
        table.add_row("actor_env_intvn queued", str(actor_buffer.get("actor_env_intvn", "?")))
        if last_episode_info:
            table.add_row("last episode reward", self._fmt_float(last_episode_info.get("reward")))
            table.add_row("last episode step", str(last_episode_info.get("step", "?")))
        return table

    def _recent(self, timestamp: float | None, now: float) -> bool:
        return timestamp is not None and (now - timestamp) <= self.timeout_s

    def _render_action(
        self,
        action: dict[str, Any] | None,
        timestamp: float | None,
        now: float,
    ) -> Table | Text:
        if not action:
            return Text("no action generated yet", overflow="fold")
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_row("age", self._fmt_since(timestamp, now))
        table.add_row("step", str(action.get("step", "?")))
        table.add_row("source", str(action.get("source", "?")))
        table.add_row("action dim", str(action.get("action_dim", "?")))
        table.add_row("inference", self._fmt_seconds(action.get("inference_s")))
        table.add_row("x sent", self._fmt_float(action.get("right_dx")))
        table.add_row("y sent", self._fmt_float(action.get("right_dy")))
        table.add_row("z sent", self._fmt_float(action.get("right_dz")))
        table.add_row("x norm", self._fmt_float(action.get("right_dx_norm")))
        table.add_row("y norm", self._fmt_float(action.get("right_dy_norm")))
        table.add_row("z norm", self._fmt_float(action.get("right_dz_norm")))
        return table

    def _render_rsync_status(self, status: dict[str, Any], send_online_data: bool) -> Table:
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        configured = bool(status.get("configured", False))
        enabled = bool(status.get("enabled", False))
        running = status.get("running")
        if not configured:
            mode = "not launched by split wrapper"
        elif not enabled:
            mode = "disabled"
        elif running is True:
            mode = "[green]running[/green]"
        elif running is False:
            mode = "[red]stopped[/red]"
        else:
            mode = "[yellow]unknown[/yellow]"
        table.add_row("mode", mode)
        table.add_row(
            "data path",
            "agentlace online upload" if send_online_data else "disk sync",
        )
        table.add_row("remote", str(status.get("remote", "")))
        table.add_row("source", str(status.get("source", "")))
        table.add_row("dest", str(status.get("dest", "")))
        table.add_row(
            "interval", f"{status.get('interval', '')}s" if status.get("interval") else ""
        )
        table.add_row("pid", str(status.get("pid", "")))
        return table

    @staticmethod
    def _rsync_border_style(status: dict[str, Any]) -> str:
        if not status.get("configured", False) or not status.get("enabled", False):
            return "cyan"
        if status.get("running") is False:
            return "red"
        return "green" if status.get("running") is True else "yellow"

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
    def _fmt_seconds(value: Any) -> str:
        try:
            return f"{float(value):.3f}s"
        except (TypeError, ValueError):
            return "?"

    @staticmethod
    def _fmt_age(seconds: float) -> str:
        seconds_i = max(0, int(seconds))
        minutes, sec = divmod(seconds_i, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m {sec:02d}s"
        return f"{minutes:02d}m {sec:02d}s"
