"""Reporting/event adapters for PLD training scripts.

The training loops should emit structured events; presentation details live
here. This keeps SAC/SACMini and the 3-D action path unaware of TUI, tqdm, or
Agentlace stat payload formats.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

Vector3 = tuple[float, float, float]


def _vec3(value: Sequence[float]) -> Vector3:
    vals = tuple(float(x) for x in value)
    if len(vals) != 3:
        raise ValueError(f"expected 3-D vector, got {len(vals)} values")
    return vals  # type: ignore[return-value]


def _cfg_select(cfg: Any, path: str, default: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(cfg, path, default=default)
    except Exception:
        cur = cfg
        for part in path.split("."):
            if cur is None:
                return default
            if isinstance(cur, Mapping):
                cur = cur.get(part, default)
            else:
                cur = getattr(cur, part, default)
        return cur


@dataclass(frozen=True)
class ActorStepEvent:
    """One clean 3-D actor action event.

    ``stage`` is ``"selected"`` before env.step and ``"executed"`` after
    env.step returns. The action fields remain the same in both stages.
    """

    step: int
    episode: int
    phase: str
    source: str
    algo_action: Vector3
    exec_norm_action: Vector3
    sent_action: Vector3
    stage: str = "selected"
    reward: float | None = None
    done: bool = False
    truncated: bool = False
    running_return: float | None = None

    @classmethod
    def from_arrays(
        cls,
        *,
        step: int,
        episode: int,
        phase: str,
        source: str,
        algo_action: Sequence[float],
        exec_norm_action: Sequence[float],
        sent_action: Sequence[float],
        stage: str = "selected",
        reward: float | None = None,
        done: bool = False,
        truncated: bool = False,
        running_return: float | None = None,
    ) -> "ActorStepEvent":
        return cls(
            step=int(step),
            episode=int(episode),
            phase=str(phase),
            source=str(source),
            algo_action=_vec3(algo_action),
            exec_norm_action=_vec3(exec_norm_action),
            sent_action=_vec3(sent_action),
            stage=str(stage),
            reward=None if reward is None else float(reward),
            done=bool(done),
            truncated=bool(truncated),
            running_return=None if running_return is None else float(running_return),
        )

    @classmethod
    def from_stats(cls, stats: Mapping[str, Any]) -> "ActorStepEvent":
        return cls(
            step=int(stats.get("step", 0)),
            episode=int(stats.get("episode", 0)),
            phase=str(stats.get("phase", "actor stats")),
            source=str(stats.get("source", "?")),
            algo_action=_vec3(
                (
                    float(stats.get("right_dx_algo", stats.get("dx_algo", 0.0))),
                    float(stats.get("right_dy_algo", stats.get("dy_algo", 0.0))),
                    float(stats.get("right_dz_algo", stats.get("dz_algo", 0.0))),
                )
            ),
            exec_norm_action=_vec3(
                (
                    float(stats.get("right_dx_norm", stats.get("dx_exec_norm", 0.0))),
                    float(stats.get("right_dy_norm", stats.get("dy_exec_norm", 0.0))),
                    float(stats.get("right_dz_norm", stats.get("dz_exec_norm", 0.0))),
                )
            ),
            sent_action=_vec3(
                (
                    float(stats.get("right_dx", stats.get("dx_sent", stats.get("dx_m", 0.0)))),
                    float(stats.get("right_dy", stats.get("dy_sent", stats.get("dy_m", 0.0)))),
                    float(stats.get("right_dz", stats.get("dz_sent", stats.get("dz_m", 0.0)))),
                )
            ),
            stage=str(stats.get("stage", "selected")),
        )

    def with_outcome(
        self,
        *,
        reward: float,
        done: bool,
        truncated: bool,
        running_return: float,
    ) -> "ActorStepEvent":
        return ActorStepEvent(
            step=self.step,
            episode=self.episode,
            phase=self.phase,
            source=self.source,
            algo_action=self.algo_action,
            exec_norm_action=self.exec_norm_action,
            sent_action=self.sent_action,
            stage="executed",
            reward=float(reward),
            done=bool(done),
            truncated=bool(truncated),
            running_return=float(running_return),
        )

    def actor_action_stats(self) -> dict[str, Any]:
        sent = self.sent_action
        norm = self.exec_norm_action
        algo = self.algo_action
        return {
            "step": int(self.step),
            "episode": int(self.episode),
            "source": self.source,
            "stage": self.stage,
            "action_dim": 3,
            "right_dx": sent[0],
            "right_dy": sent[1],
            "right_dz": sent[2],
            "right_dx_norm": norm[0],
            "right_dy_norm": norm[1],
            "right_dz_norm": norm[2],
            "right_dx_algo": algo[0],
            "right_dy_algo": algo[1],
            "right_dz_algo": algo[2],
            "dx_sent": sent[0],
            "dy_sent": sent[1],
            "dz_sent": sent[2],
            "dx_exec_norm": norm[0],
            "dy_exec_norm": norm[1],
            "dz_exec_norm": norm[2],
            "dx_algo": algo[0],
            "dy_algo": algo[1],
            "dz_algo": algo[2],
        }

    def actor_stats_payload(self) -> dict[str, Any]:
        return {"actor_action": self.actor_action_stats()}

    def episode_payload(self) -> dict[str, Any] | None:
        if not (self.done or self.truncated):
            return None
        return {
            "environment": {
                "episode": {
                    "r": float(self.running_return if self.running_return is not None else 0.0),
                    "num_episodes": int(self.episode),
                    "num_actor_steps": int(self.step),
                }
            }
        }


@dataclass(frozen=True)
class LearnerUpdateEvent:
    step: int
    metrics: Mapping[str, Any]
    replay_size: int
    demo_size: int
    phase: str = "training"


@dataclass(frozen=True)
class BufferWaitEvent:
    replay_size: int
    demo_size: int
    target_size: int
    phase: str = "waiting for replay data"
    mc_required: bool = False
    mc_ready: bool = True


@dataclass(frozen=True)
class NetworkSyncEvent:
    role: str
    status: str
    step: int | None = None
    message: str | None = None
    error: Any | None = None


def _len_or_zero(value: Any) -> int:
    try:
        return int(len(value))
    except Exception:
        return 0


def _buffer_capacity(buffer: Any, default: int) -> int:
    for name in ("capacity", "_capacity", "max_size", "_max_size"):
        value = getattr(buffer, name, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return int(default)


def _buffer_valid_mc(buffer: Any) -> int | str:
    allow_idxs = getattr(buffer, "_allow_idxs", None)
    if allow_idxs is None:
        return "n/a"
    return _len_or_zero(allow_idxs)


def _buffer_success_count(buffer: Any) -> int | str:
    try:
        size = int(len(buffer))
        if size == 0:
            return 0
        dataset_dict = getattr(buffer, "dataset_dict", None)
        if dataset_dict is None:
            return "n/a"
        rewards = dataset_dict.get("rewards") if isinstance(dataset_dict, dict) else None
        if rewards is None:
            return "n/a"
        capacity = getattr(buffer, "_capacity", len(rewards))
        valid = rewards[:size] if size < capacity else rewards
        return int(np.sum(valid > 0))
    except Exception:
        return "n/a"


def _remote_env_status(env: Any) -> dict[str, Any]:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_fn = getattr(current, "get_remote_status", None)
        if callable(status_fn):
            try:
                return dict(status_fn())
            except Exception as exc:
                return {"last_error": repr(exc)}
        current = getattr(current, "env", None)
    return {}


@dataclass
class ActorSnapshotProvider:
    env: Any
    data_store: Any
    cfg: Any
    send_online_data: bool

    def snapshot(self) -> dict[str, Any]:
        env_status = _remote_env_status(self.env)
        host = _cfg_select(self.cfg, "env.remote_host", "")
        port = _cfg_select(self.cfg, "env.remote_port", "")
        address = env_status.get("server_address")
        if not address and host and port:
            address = f"{host}:{port}"
        return {
            "env": env_status,
            "actor_buffer": {
                "actor_env": _len_or_zero(self.data_store),
                "actor_env_intvn": 0,
            },
            "send_online_data": bool(self.send_online_data),
            "instruction": _cfg_select(self.cfg, "env.instruction", ""),
            "action_repr": _cfg_select(self.cfg, "env.action_repr", "delta_eef_pos"),
            "control_mode": _cfg_select(self.cfg, "env.control_mode", "right"),
            "remote_server_address": address,
            "remote_port": port,
            "rsync": {},
        }


@dataclass
class LearnerSnapshotProvider:
    replay_buffer: Any
    demo_buffer: Any
    disk_ingestor: Any
    cfg: Any
    profile_stats: Mapping[str, Any]
    train_gate: Mapping[str, Any]

    def snapshot(self) -> dict[str, Any]:
        disk_root = str(_cfg_select(self.cfg, "task.online_data_buffer_path", "") or "")
        if self.disk_ingestor is None:
            ingestor_stats: dict[str, Any] = {}
            disk_stats = {
                "root": disk_root,
                "root_exists": bool(disk_root and os.path.exists(os.path.expanduser(disk_root))),
            }
        else:
            ingestor_stats = self.disk_ingestor.stats()
            disk_stats = self.disk_ingestor.disk_inventory()
        return {
            "replay_size": _len_or_zero(self.replay_buffer),
            "demo_size": _len_or_zero(self.demo_buffer),
            "replay_capacity": _buffer_capacity(
                self.replay_buffer,
                int(_cfg_select(self.cfg, "train.replay_buffer_capacity", 0)),
            ),
            "demo_capacity": _buffer_capacity(
                self.demo_buffer,
                int(_cfg_select(self.cfg, "train.demo_buffer_capacity", 0)),
            ),
            "replay_valid_mc": _buffer_valid_mc(self.replay_buffer),
            "demo_valid_mc": _buffer_valid_mc(self.demo_buffer),
            "replay_success": _buffer_success_count(self.replay_buffer),
            "demo_success": _buffer_success_count(self.demo_buffer),
            "disk_root": disk_root,
            "disk": disk_stats,
            "ingestor": ingestor_stats,
            "rsync": {},
            "profile": dict(self.profile_stats),
            "profile_sync_jax": False,
            "train_gate": dict(self.train_gate),
        }


class Reporter:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def phase(self, role: str, phase: str) -> None:
        pass

    def actor_step(self, event: ActorStepEvent) -> None:
        pass

    def learner_update(self, event: LearnerUpdateEvent) -> None:
        pass

    def buffer_wait(self, event: BufferWaitEvent) -> None:
        pass

    def network_sync(self, event: NetworkSyncEvent) -> None:
        pass


class CompositeReporter(Reporter):
    def __init__(self, reporters: Sequence[Reporter | None] = ()) -> None:
        self._reporters = [r for r in reporters if r is not None]

    def start(self) -> None:
        for reporter in self._reporters:
            reporter.start()

    def stop(self) -> None:
        for reporter in reversed(self._reporters):
            reporter.stop()

    def phase(self, role: str, phase: str) -> None:
        for reporter in self._reporters:
            reporter.phase(role, phase)

    def actor_step(self, event: ActorStepEvent) -> None:
        for reporter in self._reporters:
            reporter.actor_step(event)

    def learner_update(self, event: LearnerUpdateEvent) -> None:
        for reporter in self._reporters:
            reporter.learner_update(event)

    def buffer_wait(self, event: BufferWaitEvent) -> None:
        for reporter in self._reporters:
            reporter.buffer_wait(event)

    def network_sync(self, event: NetworkSyncEvent) -> None:
        for reporter in self._reporters:
            reporter.network_sync(event)


class TUIReporter(Reporter):
    def __init__(self, *, actor_tui: Any = None, learner_tui: Any = None) -> None:
        self.actor_tui = actor_tui
        self.learner_tui = learner_tui

    @classmethod
    def for_actor(cls, cfg: Any, snapshot_fn: Callable[[], dict[str, Any]]) -> "TUIReporter":
        try:
            from actor_tui import ActorTUI
        except Exception as exc:
            print(f"[reporting] actor TUI disabled: {exc!r}")
            return cls()
        return cls(actor_tui=ActorTUI.from_config(cfg, snapshot_fn))

    @classmethod
    def for_learner(cls, cfg: Any, snapshot_fn: Callable[[], dict[str, Any]]) -> "TUIReporter":
        try:
            from learner_tui import LearnerTUI
        except Exception as exc:
            print(f"[reporting] learner TUI disabled: {exc!r}")
            return cls()
        return cls(learner_tui=LearnerTUI.from_config(cfg, snapshot_fn))

    def start(self) -> None:
        if self.actor_tui is not None:
            self.actor_tui.start()
        if self.learner_tui is not None:
            self.learner_tui.start()

    def stop(self) -> None:
        if self.actor_tui is not None:
            self.actor_tui.stop()
        if self.learner_tui is not None:
            self.learner_tui.stop()

    def phase(self, role: str, phase: str) -> None:
        if role == "actor" and self.actor_tui is not None:
            self.actor_tui.set_phase(phase)
        if role == "learner" and self.learner_tui is not None:
            self.learner_tui.set_phase(phase)

    def actor_step(self, event: ActorStepEvent) -> None:
        if self.actor_tui is not None:
            self.actor_tui.set_step(event.step)
            self.actor_tui.set_episode(event.episode)
            self.actor_tui.set_phase(event.phase)
            if event.stage == "selected":
                self.actor_tui.record_env_observation()
                self.actor_tui.record_action(event.actor_action_stats())
            elif event.stage == "executed" and event.reward is not None:
                self.actor_tui.record_env_step(
                    done=event.done,
                    truncated=event.truncated,
                    reward=event.reward,
                )
        if self.learner_tui is not None and event.stage == "selected":
            self.learner_tui.record_actor_stats(event.actor_stats_payload())

    def learner_update(self, event: LearnerUpdateEvent) -> None:
        if self.learner_tui is None:
            return
        self.learner_tui.set_phase(event.phase)
        self.learner_tui.set_step(event.step)
        self.learner_tui.record_update_info(event.step, dict(event.metrics))

    def buffer_wait(self, event: BufferWaitEvent) -> None:
        if self.learner_tui is not None:
            self.learner_tui.set_phase(event.phase)

    def network_sync(self, event: NetworkSyncEvent) -> None:
        if self.actor_tui is not None and event.role == "actor":
            if event.status == "connected":
                self.actor_tui.record_learner_connected()
            elif event.status == "disabled":
                self.actor_tui.record_learner_disabled(event.message or "disabled")
            elif event.status == "error":
                self.actor_tui.record_learner_error(event.error or event.message)
            elif event.status == "request":
                self.actor_tui.record_learner_request()
            elif event.status in ("received", "published"):
                self.actor_tui.record_network_update()
        if self.learner_tui is not None and event.role == "learner":
            if event.step is not None:
                self.learner_tui.set_step(event.step)


class TqdmActorReporter(Reporter):
    def __init__(self, pbar: Any, *, action_period: int = 10) -> None:
        self.pbar = pbar
        self.action_period = max(1, int(action_period))

    def actor_step(self, event: ActorStepEvent) -> None:
        if event.stage != "selected" or event.step % self.action_period != 0:
            return
        stats = event.actor_action_stats()
        self.pbar.set_postfix(
            dx=f"{stats['right_dx']:.4f}",
            dy=f"{stats['right_dy']:.4f}",
            dz=f"{stats['right_dz']:.4f}",
            src=event.source,
        )


class AgentlaceActorStatsReporter(Reporter):
    def __init__(self, client: Any, *, action_period: int = 10) -> None:
        self.client = client
        self.action_period = max(1, int(action_period))

    def actor_step(self, event: ActorStepEvent) -> None:
        if event.stage == "selected" and event.step % self.action_period == 0:
            self.client.request("send-stats", event.actor_stats_payload())
        episode_payload = event.episode_payload()
        if event.stage == "executed" and episode_payload is not None:
            self.client.request("send-stats", episode_payload)

    def network_sync(self, event: NetworkSyncEvent) -> None:
        if event.role == "actor" and event.status == "request":
            self.client.request(
                "send-stats",
                {"network_sync": {"status": event.status, "step": event.step}},
            )


def make_actor_reporter(
    cfg: Any,
    *,
    snapshot_fn: Callable[[], dict[str, Any]],
    client: Any = None,
    pbar: Any = None,
) -> Reporter:
    action_period = int(_cfg_select(cfg, "train.action_viz_period", 10))
    reporters: list[Reporter] = [TUIReporter.for_actor(cfg, snapshot_fn)]
    if pbar is not None:
        reporters.append(TqdmActorReporter(pbar, action_period=action_period))
    if client is not None:
        reporters.append(AgentlaceActorStatsReporter(client, action_period=action_period))
    return CompositeReporter(reporters)


def make_learner_reporter(
    cfg: Any,
    *,
    snapshot_fn: Callable[[], dict[str, Any]],
) -> Reporter:
    return CompositeReporter([TUIReporter.for_learner(cfg, snapshot_fn)])
