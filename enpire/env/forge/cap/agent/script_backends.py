# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script runner backends — robot-specific exec context, recording, and post-exec.

The backend class is selected by the ``robot:`` Hydra config group:

    uv run python run_script.py robot=robocasa ...
    uv run python run_script.py robot=real_yam ...

Modern configs use ``robot.script_backend._target_``.  Older configs with
``robot._target_`` remain supported.  When ``robot:`` is absent,
``get_backend()`` auto-derives the class from ``env.name``.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from enpire.env.forge.cap.agent.recorder import ScriptRecorder


class ScriptRunnerBackend(ABC):
    """Per-robot exec lifecycle for run_script.py.

    Encapsulates the branches that differ between robot environments:
    dashboard choice, camera discovery, recording, and post-exec cleanup.
    """

    @abstractmethod
    def vis_cameras(self, env: Any) -> list[str]:
        """Camera names used for before/after frame capture."""

    @property
    @abstractmethod
    def is_quiet(self) -> bool:
        """True when a dashboard owns the terminal (suppress direct prints)."""

    @abstractmethod
    def setup_recorder(
        self, env: Any, log_dir: Path, cfg: Any
    ) -> "ScriptRecorder | None":
        """Create and start a recorder. Return None when not applicable."""

    @contextmanager
    @abstractmethod
    def exec_context(
        self, namespace: dict, script_path: Path
    ) -> Iterator[None]:
        """Context manager wrapping exec(). Handles dashboard + tool hooks."""

    @abstractmethod
    def post_exec(self, exec_error: str | None, namespace: dict) -> None:
        """Called after exec finishes — await_exit, go_home, etc."""


# ---------------------------------------------------------------------------
# RoboCasa backend
# ---------------------------------------------------------------------------


class RobocasaBackend(ScriptRunnerBackend):
    """LiveDashboard + ScriptRecorder for RoboCasa / sim envs."""

    def __init__(self, env: Any, cfg: Any, **_: Any):
        self._env = env
        self._cfg = cfg
        from enpire.env.forge.cap.agent.live_dashboard import dashboard_enabled
        self._use_live_dashboard = dashboard_enabled(sys.__stdout__)

    def vis_cameras(self, env: Any) -> list[str]:
        return list(env._camera_map.keys()) if hasattr(env, "_camera_map") else ["top"]

    @property
    def is_quiet(self) -> bool:
        return self._use_live_dashboard

    def setup_recorder(
        self, env: Any, log_dir: Path, cfg: Any
    ) -> "ScriptRecorder | None":
        if not cfg.recording.enabled:
            return None
        from enpire.env.forge.cap.agent.recorder import ScriptRecorder
        cam_names = (
            list(env._camera_map.keys()) if hasattr(env, "_camera_map") else ["top"]
        )
        recorder = ScriptRecorder.from_env(camera_names=cam_names, output_dir=log_dir)
        env.set_recorder(recorder)
        recorder.start()
        return recorder

    @contextmanager
    def exec_context(self, namespace: dict, script_path: Path) -> Iterator[None]:
        if not self._use_live_dashboard:
            yield
            return
        from enpire.env.forge.cap.agent.live_dashboard import LiveDashboard
        from enpire.env.forge.cap.agent.profiler import set_tool_event_hooks
        cfg = self._cfg
        dash = LiveDashboard(
            title=script_path.name,
            header={
                "env":    str(getattr(getattr(cfg, "env", None), "name", "-")),
                "seed":   str(cfg.env.seed) if getattr(cfg.env, "seed", None) is not None else "-",
                "layout": str(getattr(cfg.env, "layout_id", "-")),
                "style":  str(getattr(cfg.env, "style_id", "-")),
            },
        )
        with dash:
            set_tool_event_hooks(on_start=dash.on_tool_start, on_end=dash.on_tool_end)
            try:
                yield
            finally:
                set_tool_event_hooks(on_start=None, on_end=None)

    def post_exec(self, exec_error: str | None, namespace: dict) -> None:
        pass  # nothing to do for sim


# ---------------------------------------------------------------------------
# Real YAM backend
# ---------------------------------------------------------------------------


class RealYamBackend(ScriptRunnerBackend):
    """YamDashboard + await_exit + go_home for real bimanual YAM hardware."""

    def __init__(
        self,
        env: Any,
        cfg: Any,
        dashboard: bool = True,
        await_exit: bool = True,
        go_home_on_exit: bool = True,
        **_: Any,
    ):
        self._env = env
        self._cfg = cfg
        self._dashboard = bool(dashboard)
        self._await_exit = await_exit
        self._go_home_on_exit = go_home_on_exit
        self._started_dashboard = False

    def vis_cameras(self, env: Any) -> list[str]:
        # Real hardware scripts should not block on best-effort before/after
        # visualization snapshots.  The real-YAM env already starts continuous
        # camera reader threads for live tool access; grabbing extra snapshots
        # here can starve or stall script startup on RGB-D camera pipelines
        # (especially ZED NEURAL depth + two RealSense depth streams).
        #
        # Opt back in only when explicitly debugging visual snapshots.
        import os

        enabled = os.environ.get("CAP_REAL_YAM_RUN_SCRIPT_CAPTURE_FRAMES", "")
        if enabled.strip().lower() in {"1", "true", "yes", "on"}:
            return ["top", "left", "right"]
        return []

    @property
    def is_quiet(self) -> bool:
        return self._dashboard  # When enabled, YamDashboard owns the terminal.

    def setup_recorder(
        self, env: Any, log_dir: Path, cfg: Any
    ) -> "ScriptRecorder | None":
        if not cfg.recording.enabled:
            return None

        from enpire.env.forge.cap.agent.recorder import ScriptRecorder

        cameras = list(getattr(cfg.recording, "cameras", None) or [])
        if not cameras:
            cameras = ["top", "left", "right"]

        class _RealYamRecorderSource:
            def __init__(self, real_env: Any, camera_names: list[str]):
                self._env = real_env
                # ScriptRecorder's polling mode expects a CapServer-like
                # ``_cameras`` mapping and ``get_camera_image`` method.
                self._cameras = {name: None for name in camera_names}

            def get_camera_image(self, camera: str):
                return self._env.render_rgb(camera)

        recorder = ScriptRecorder(_RealYamRecorderSource(env, cameras), log_dir)
        recorder.start()
        return recorder

    @contextmanager
    def exec_context(self, namespace: dict, script_path: Path) -> Iterator[None]:
        if self._dashboard:
            dash = getattr(self._env, "_dashboard", None)
            if dash is None:
                get_robot_state = namespace.get("get_robot_state")
                if get_robot_state is None:
                    print(
                        "[run_script] YAM dashboard not started: "
                        "get_robot_state missing"
                    )
                else:
                    try:
                        from enpire.env.forge.cap.env.real_bimanual_yam.dashboard import YamDashboard

                        dash = YamDashboard(self._env, get_robot_state)
                        dash.start()
                        # Attach to env so run_script's stdout tee can forward
                        # script prints into the dashboard, and env.close() can
                        # clean up if the process exits through another path.
                        self._env._dashboard = dash
                        self._started_dashboard = True
                    except Exception as exc:
                        print(f"[run_script] YAM dashboard failed to start: {exc}")
                        dash = None
        else:
            print("[run_script] YAM dashboard disabled (robot.dashboard=false)")

        dash = getattr(self._env, "_dashboard", None)
        if dash is None:
            yield
            return

        from enpire.env.forge.cap.agent.profiler import set_tool_event_hooks

        set_tool_event_hooks(on_start=dash.on_tool_start, on_end=dash.on_tool_end)
        try:
            yield
        finally:
            set_tool_event_hooks(on_start=None, on_end=None)

    def post_exec(self, exec_error: str | None, namespace: dict) -> None:
        try:
            # Default for non-interactive agent-loop execution: go home before
            # the child run_script process exits, so the parent cannot begin the
            # next iteration while the real arm is still in the workspace.
            should_home = True

            if exec_error == "KeyboardInterrupt":
                # Mid-run X/ Ctrl+C sets the real-YAM stop flag. Direct motion
                # helpers turn that into KeyboardInterrupt after they stop
                # sending the active trajectory. Treat that as the operator's
                # "stop current thing, go home, and quit" request.
                should_home = True
            elif self._await_exit:
                dash = getattr(self._env, "_dashboard", None)
                if dash is not None:
                    should_home = dash.await_exit()
                else:
                    # Dashboard can be disabled (robot.dashboard=false) to avoid
                    # its background get_robot_state polling contending with
                    # short scripts.  Still honor robot.await_exit by falling
                    # back to a simple terminal prompt, matching
                    # YamDashboard.await_exit semantics:
                    #   ENTER/anything/X/Q -> go_home, S -> skip home.
                    import sys as _sys

                    _sys.stderr.write(
                        "\n[YAM] Script done — ENTER go home & exit | "
                        "S skip home | X go home & exit: "
                    )
                    _sys.stderr.flush()
                    try:
                        line = input().strip().lower()
                    except (EOFError, OSError):
                        line = ""
                    if line == "s":
                        should_home = False

            if should_home and self._go_home_on_exit:
                go_home_fn = namespace.get("go_home")
                if go_home_fn is not None:
                    try:
                        print("[run_script] Going home...")
                        go_home_fn()
                    except Exception as exc:
                        print(f"[run_script] go_home failed: {exc}")
        finally:
            if self._started_dashboard:
                dash = getattr(self._env, "_dashboard", None)
                if dash is not None:
                    dash.stop()
                try:
                    delattr(self._env, "_dashboard")
                except AttributeError:
                    pass


class StudyBackend(ScriptRunnerBackend):
    """No-op backend for offline study-mode scripts."""

    def __init__(self, env: Any, cfg: Any, **_: Any):
        self._env = env
        self._cfg = cfg

    def vis_cameras(self, env: Any) -> list[str]:
        _ = env
        return []

    @property
    def is_quiet(self) -> bool:
        return False

    def setup_recorder(
        self, env: Any, log_dir: Path, cfg: Any
    ) -> "ScriptRecorder | None":
        _ = (env, log_dir, cfg)
        return None

    @contextmanager
    def exec_context(
        self, namespace: dict, script_path: Path
    ) -> Iterator[None]:
        _ = (namespace, script_path)
        yield

    def post_exec(self, exec_error: str | None, namespace: dict) -> None:
        _ = (exec_error, namespace)
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_backend(env_name: str, cfg: Any, env: Any) -> ScriptRunnerBackend:
    """Return the appropriate backend.

    Instantiates from ``cfg.robot.script_backend._target_`` when present.
    Older configs with ``cfg.robot._target_`` are still supported, otherwise
    this auto-derives from ``env_name`` for backward compatibility.
    """
    robot_cfg = getattr(cfg, "robot", None)
    if robot_cfg is not None:
        backend_cfg = None
        target = None
        try:
            from omegaconf import OmegaConf

            backend_cfg = OmegaConf.select(robot_cfg, "script_backend", default=None)
            target = OmegaConf.select(backend_cfg, "_target_", default=None)
        except Exception:
            backend_cfg = None
            target = None
        if target:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf

            backend_kwargs = {}
            for field in ("dashboard", "await_exit", "go_home_on_exit"):
                value = OmegaConf.select(robot_cfg, field, default=None)
                if value is not None:
                    backend_kwargs[field] = value
            return instantiate(
                backend_cfg,
                env=env,
                cfg=cfg,
                **backend_kwargs,
                _recursive_=False,
            )

    if robot_cfg is not None and hasattr(robot_cfg, "_target_"):
        from hydra.utils import instantiate

        return instantiate(robot_cfg, env=env, cfg=cfg, _recursive_=False)

    # Auto-derive — keeps backward compat with experiment configs that omit robot:
    if env_name.startswith("yam-real"):
        return RealYamBackend(env=env, cfg=cfg)
    if env_name == "study" or env_name.startswith("study:"):
        return StudyBackend(env=env, cfg=cfg)
    return RobocasaBackend(env=env, cfg=cfg)
