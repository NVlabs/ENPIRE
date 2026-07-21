"""Shared runtime setup for direct-mode execution (no CapServer).

``create_runtime`` is kept as a backward-compatible convenience wrapper.
Robot-specific runtime construction now lives in Hydra-configurable robot
adapters under ``cap.agent.robot_adapters``.
"""

from __future__ import annotations

from typing import Any, Optional


def create_runtime(
    env_name: str,
    *,
    viewer: bool = False,
    vlm_backend: str = "gemini",
    seed: Optional[int] = None,
    layout_id: Optional[int] = None,
    style_id: Optional[int] = None,
    camera_height: Optional[int] = None,
    camera_width: Optional[int] = None,
    curobo_host: str = "127.0.0.1",
    curobo_port: int = 0,
    mppi_host: str = "127.0.0.1",
    mppi_port: int = 0,
    cfg: Any = None,
    runtime_role: str = "script",
) -> tuple[Any, dict[str, Any]]:
    """Create env + tool namespace using the configured robot adapter."""
    if cfg is not None:
        from enpire.env.forge.cap.agent.robot_adapters import get_robot_adapter

        adapter = get_robot_adapter(cfg)
    else:
        from enpire.env.forge.cap.agent.robot_adapters import adapter_for_env

        adapter = adapter_for_env(env_name)

    return adapter.create_runtime(
        cfg=cfg,
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
