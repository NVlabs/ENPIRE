# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from enpire.policy.rl.context import RLContext


@dataclass
class AutoEvalState:
    active: bool = False
    completed: bool = False
    hole_index: int = 0
    episodes_for_hole: int = 0
    run_dirs: dict[str, list[str]] = field(default_factory=dict)
    seen_episode_dirs: set[str] = field(default_factory=set)
    output_yaml: Path | None = None


class AutoEvalController:
    """Keyboard-triggered automatic per-hole learn-mode evaluation."""

    def __init__(self) -> None:
        self.state = AutoEvalState()

    def start(self, ctx: RLContext) -> None:
        if self.state.active:
            print("[auto_eval] already running", flush=True)
            return
        if ctx.initial_pose_manager.count <= 0:
            print("[auto_eval] no initial positions loaded; ignoring start", flush=True)
            return

        self.state = AutoEvalState(active=True)
        self.state.output_yaml = self._resolve_output_yaml(ctx)
        ctx.initial_pose_manager.select_first()
        self._start_hole(ctx, 0)
        print(
            "[auto_eval] started: "
            f"{self._num_holes(ctx)}/{ctx.initial_pose_manager.count} holes, "
            f"{self._episodes_per_hole(ctx)} episodes/hole",
            flush=True,
        )

    def observe(self, ctx: RLContext) -> None:
        if not self.state.active:
            return

        last_episode_dir = getattr(ctx.env, "last_episode_dir", None)
        if last_episode_dir is None:
            return
        episode_key = str(Path(last_episode_dir))
        if episode_key in self.state.seen_episode_dirs:
            return

        self.state.seen_episode_dirs.add(episode_key)
        self.state.episodes_for_hole += 1
        print(
            "[auto_eval] "
            f"hole {self.state.hole_index + 1}/{self._num_holes(ctx)}: "
            f"{self.state.episodes_for_hole}/{self._episodes_per_hole(ctx)} episodes",
            flush=True,
        )

        if self.state.episodes_for_hole < self._episodes_per_hole(ctx):
            return

        next_hole_index = self.state.hole_index + 1
        limit = self._num_holes(ctx)
        if next_hole_index >= limit:
            self._finish(ctx)
            return
        self._start_hole(ctx, next_hole_index)

    def _start_hole(self, ctx: RLContext, hole_index: int) -> None:
        pause_pose = ctx.initial_pose_manager.build_center_pose()
        self.state.hole_index = hole_index
        self.state.episodes_for_hole = 0
        ctx.initial_pose_manager.select_initial_position_index(hole_index + 1)
        run_dir = self._new_run_dir(ctx, hole_index)
        self.state.run_dirs[f"hole_index_{hole_index + 1}"] = [str(run_dir)]
        self._rotate_output_dir(ctx, run_dir)
        self._reset_to_current_hole(ctx, pause_pose)
        ctx.state_machine.state = "hover"
        print(
            f"[auto_eval] hole_index_{hole_index + 1} recording to {run_dir}",
            flush=True,
        )

    def _finish(self, ctx: RLContext) -> None:
        self._write_yaml(ctx)
        self._discard_active_episode(ctx)
        ctx.state_machine.state = "home"
        self.state.active = False
        self.state.completed = True
        print("[auto_eval] completed; returning home", flush=True)

    def _reset_to_current_hole(self, ctx: RLContext, pause_pose: dict) -> None:
        ctx.policy_router.rl_policy.reset()
        ctx.obs, _ = ctx.env.reset(
            options={
                "target_ee_pose": pause_pose,
                "discard_episode": True,
            }
        )
        ctx.obs, _ = ctx.env.reset(options={"alias": "home", "discard_episode": True})
        ctx.obs, _ = ctx.env.reset(
            options={
                "target_ee_pose": ctx.initial_pose_manager.build_center_pose(),
                "discard_episode": True,
            }
        )
        ctx.terminal_event = None
        ctx.last_terminal_event = None
        ctx.event_router.reset_timer()

    def _rotate_output_dir(self, ctx: RLContext, run_dir: Path) -> None:
        ctx.env.set_output_dir(run_dir)
        if ctx.handshake_server is not None:
            ctx.handshake_server.set_path(run_dir)

    def _discard_active_episode(self, ctx: RLContext) -> None:
        finalize = getattr(ctx.env, "finalize_episode", None)
        if callable(finalize):
            finalize(discard_episode=True)

    def _write_yaml(self, ctx: RLContext) -> None:
        path = self.state.output_yaml or self._resolve_output_yaml(ctx)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"holes": self.state.run_dirs}
        with path.open("w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        print(f"[auto_eval] wrote data_stats config: {path}", flush=True)

    def _new_run_dir(self, ctx: RLContext, hole_index: int) -> Path:
        root = Path(getattr(ctx.env, "output_dir")).expanduser().resolve().parent
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = root / f"{timestamp}_auto_eval_hole_{hole_index + 1}"
        candidate = base
        suffix = 1
        while candidate.exists():
            candidate = root / f"{base.name}-{suffix}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _resolve_output_yaml(self, ctx: RLContext) -> Path:
        configured = ctx.cfg.auto_eval_output_yaml
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ctx.cfg.task_name or 'run'}_auto_eval_{timestamp}.yaml"
        if configured:
            path = Path(configured).expanduser()
            if path.suffix in {".yaml", ".yml"}:
                return path
            return path / filename
        root = Path(getattr(ctx.env, "output_dir")).expanduser().resolve().parent
        return root / filename

    @staticmethod
    def _episodes_per_hole(ctx: RLContext) -> int:
        return max(1, int(ctx.cfg.auto_eval_episodes_per_hole))

    @staticmethod
    def _num_holes(ctx: RLContext) -> int:
        cfg_limit = ctx.cfg.auto_eval_num_holes
        total = ctx.initial_pose_manager.count
        if cfg_limit is None or cfg_limit <= 0:
            return total
        return min(cfg_limit, total)

