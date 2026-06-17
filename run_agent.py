"""Run an LLM agent loop.

The agent generates Python code, executes it via run_script.py subprocess,
observes the result, and optionally re-generates in a loop.

Usage (Hydra)::

    uv run python run_agent.py experiment=pick_place_sink_to_counter
    uv run python run_agent.py experiment=pick_place_sink_to_counter env.seed=99
    uv run python run_agent.py experiment=dry_run
    uv run python run_agent.py experiment=pick_place_sink_to_counter --cfg job
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_ROOT_PATH = Path(__file__).resolve().parent
_ROOT = str(_ROOT_PATH)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cap.agent.agent_config import register_configs, save_config_to_log

register_configs()


def _build_namespace(cfg: DictConfig) -> dict:
    """Build env + tool namespace from config."""
    from cap.agent.robot_adapters import get_robot_adapter

    env_name = cfg.env.name
    if not env_name:
        sys.exit("env.name is required (set in experiment config or CLI)")

    print(f"[run_agent] Creating {env_name} env (direct mode)")
    adapter = get_robot_adapter(cfg)
    _env, namespace = adapter.create_runtime(cfg=cfg, runtime_role="agent")
    print(f"[run_agent] Namespace: {len(namespace)} tools\n")
    return namespace


@hydra.main(version_base="1.3", config_path="experiments", config_name="config")
def main(cfg: DictConfig) -> None:
    from cap.agent.agent_context import AgentContext
    from cap.agent.agent_pipeline import AgentPipeline
    from cap.agent.agent_session import AgentRunSession
    from cap.agent.tools._artifact_log import set_artifact_dir
    from cap.agent.wandb_logger import WandbLogger
    from cap.prompt.loader import PromptMemory

    # --- Validate ---
    if not cfg.task:
        sys.exit("task is required (set in experiment config or CLI: task='...')")
    if not cfg.runtime.agent_name:
        sys.exit(
            "runtime.agent_name is required "
            "(set in experiment config, CLI, or CAP_AGENT_NAME env var)"
        )
    print(f"[run_agent] Config: {cfg.name} (task={cfg.task[:50]})")

    # --- Namespace ---
    namespace = _build_namespace(cfg)

    # --- Prompt memory ---
    prompt_memory = PromptMemory()
    prompt_memory.scan()

    # --- Session ---
    session = AgentRunSession.create(
        task=cfg.task,
        env_name=cfg.env.name,
        log_dir=Path(cfg.runtime.log_dir),
    )
    save_config_to_log(cfg, session.run_dir)
    get_task_info_fn = (
        namespace.get("get_task_info") if cfg.evaluation_method == "reward" else None
    )
    session.enable_profiling(
        state_fn=namespace.get("get_robot_state"),
        get_task_info_fn=get_task_info_fn,
    )
    set_artifact_dir(session.run_dir)

    # --- Skill library ---
    if cfg.skill_library.enabled:
        seed_dir = getattr(cfg.skill_library, "seed_dir", "") or ""
        session.init_skill_library(namespace, seed_dir=seed_dir or None)

    # --- Wandb ---
    wb_logger = WandbLogger.from_config(cfg, session)
    if wb_logger.active:
        print(f"[run_agent] wandb: {wb_logger.run_url}")
        wb_logger.log_namespace_tools(namespace)

    # --- Pipeline (config-driven) ---
    pipeline = AgentPipeline.from_config(cfg, prompt_memory)
    pipeline._wandb = wb_logger

    # --- Context ---
    ctx = AgentContext(
        task=cfg.task,
        namespace=namespace,
        tool_schemas=[],
        env_spec=None,
        config=cfg,
        session=session,
        max_iterations=cfg.max_iterations,
        prompt_memory=prompt_memory,
    )

    # --- Run ---
    step_names = " \u2192 ".join(s.name for s in pipeline.steps)
    print(f"{'─' * 60}")
    print(f"Task:   {cfg.task}")
    print(f"Env:    {cfg.env.name}")
    print(f"LLM:    {cfg.llm.backend if not cfg.runtime.dry_run else '(dry-run)'}")
    print(f"Steps:  {step_names}")
    print(f"{'─' * 60}\n")

    t0 = time.time()
    try:
        ctx = pipeline.run(ctx)
    except KeyboardInterrupt:
        print("\n[run_agent] Interrupted")
    except Exception:
        traceback.print_exc()
    finally:
        duration = time.time() - t0
        result = pipeline.result(ctx, duration_s=duration)
        session.close(
            memory=ctx.memory,
            metadata=OmegaConf.to_container(cfg, resolve=True),
        )
        if wb_logger.active:
            wb_logger.log_summary(
                success=result.success,
                iterations=ctx.iteration,
                duration_s=duration,
                final_score=result.final_score,
                stop_reason=result.stop_reason,
            )
            wb_logger.finish(exit_code=0 if result.success else 1)

    print(f"\n{'─' * 60}")
    print(
        f"[run_agent] Done \u2014 {ctx.iteration} iteration(s), "
        f"success={result.success}, score={result.final_score:.3f}, {duration:.1f}s"
    )
    print(f"[run_agent] Logs \u2192 {session.run_dir}")


if __name__ == "__main__":
    main()
