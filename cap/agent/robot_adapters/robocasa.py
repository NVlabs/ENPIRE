"""RoboCasa robot adapter."""

from __future__ import annotations

from typing import Any

from cap.agent.robot_adapters.base import cfg_runtime_kwargs


class RobocasaAdapter:
    """Create RoboCasa env/tool namespace and child run_script overrides."""

    def __init__(self, config_group: str = "robocasa") -> None:
        self.config_group = config_group

    def create_runtime(
        self,
        *,
        cfg: Any | None = None,
        runtime_role: str = "script",
        env_name: str | None = None,
        viewer: bool = False,
        vlm_backend: str = "gemini",
        seed: int | None = None,
        layout_id: int | None = None,
        style_id: int | None = None,
        camera_height: int | None = None,
        camera_width: int | None = None,
        curobo_host: str = "127.0.0.1",
        curobo_port: int = 0,
        mppi_host: str = "127.0.0.1",
        mppi_port: int = 0,
    ) -> tuple[Any, dict[str, Any]]:
        kwargs = cfg_runtime_kwargs(
            cfg,
            runtime_role=runtime_role,
            env_name=env_name,
            viewer=viewer,
            vlm_backend=vlm_backend,
            seed=seed,
            layout_id=layout_id,
            style_id=style_id,
            camera_height=camera_height,
            camera_width=camera_width,
            curobo_host=curobo_host,
            curobo_port=curobo_port,
            mppi_host=mppi_host,
            mppi_port=mppi_port,
        )

        import os
        import random

        import numpy as np

        from cap.agent.tool_handle import make_tool_runner_namespace
        from cap.env import create_env
        from cap.env.robocasa.skills import make_namespace
        from cap.policy.robocasa_runner import make_policy_runner

        resolved_env_name = (
            kwargs["env_name"] or "robocasa:PickPlaceSinkToCounter"
        )

        # Resolve seed: explicit config/arg > env var > None.
        effective_seed = kwargs["seed"]
        if effective_seed is None:
            seed_str = os.environ.get("ROBOCASA_SEED", "")
            if seed_str:
                effective_seed = int(seed_str)

        # Seed after imports, right before env creation.
        if effective_seed is not None:
            np.random.seed(effective_seed)
            random.seed(effective_seed)

        env_kwargs: dict[str, Any] = {}
        if kwargs["layout_id"] is not None:
            env_kwargs["layout_ids"] = kwargs["layout_id"]
        if kwargs["style_id"] is not None:
            env_kwargs["style_ids"] = kwargs["style_id"]
        if effective_seed is not None:
            env_kwargs["seed"] = effective_seed
        if kwargs["camera_height"] is not None:
            env_kwargs["camera_height"] = kwargs["camera_height"]
        if kwargs["camera_width"] is not None:
            env_kwargs["camera_width"] = kwargs["camera_width"]

        env = create_env(resolved_env_name, viewer=kwargs["viewer"], **env_kwargs)
        namespace = make_namespace(
            env,
            vlm_backend=kwargs["vlm_backend"],
            curobo_host=kwargs["curobo_host"],
            curobo_port=kwargs["curobo_port"],
            policy_runner=make_policy_runner(cfg),
            cfg=cfg,
            runtime_role=kwargs["runtime_role"],
        )
        namespace.update(make_tool_runner_namespace())
        return env, namespace

    def run_script_overrides(self, cfg: Any | None = None) -> list[str]:
        _ = cfg
        return [f"robot={self.config_group}"]

    def child_env(
        self,
        cfg: Any | None = None,
        *,
        seed: int | None = None,
        slot: int = 0,
        n_seeds: int = 1,
    ) -> dict[str, str]:
        """No adapter-level child env overrides; agent_step handles shared ports."""
        _ = (cfg, seed, slot, n_seeds)
        return {}
