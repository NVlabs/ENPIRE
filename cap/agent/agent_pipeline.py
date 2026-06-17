"""AgentPipeline — composes AgentSteps into an iterated generate→execute→evaluate loop.

Usage::

    from cap.agent.agent_pipeline import AgentPipeline
    from cap.agent.agent_step import (
        ObserverStep, CodeGeneratorStep, ExecutorStep,
        RewardEvaluatorStep, SelfReflectionStep,
    )
    from cap.agent.agent_context import AgentContext
    from cap.agent.llm.cloud import CloudLLM

    llm = CloudLLM()
    pipeline = AgentPipeline([
        ObserverStep(),
        CodeGeneratorStep(llm),
        ExecutorStep(),
        RewardEvaluatorStep(),
        SelfReflectionStep(llm),
    ], max_iterations=5)

    ctx = AgentContext(task="Pick up the red object", namespace=ns, ...)
    ctx = pipeline.run(ctx)
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cap.agent.agent_context import AgentContext
    from cap.agent.agent_step import AgentStep
    from cap.agent.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


@dataclass
class AgentRunResult:
    """Summary of a completed pipeline run."""

    success: bool
    iterations: int
    stop_reason: str
    final_score: float = 0.0
    duration_s: float = 0.0


class AgentPipeline:
    """Composes AgentSteps and iterates them until success or max_iterations.

    Each iteration:
    1. Run each step in order, passing ctx through.
    2. If any step sets ctx.should_stop, exit immediately.
    3. Otherwise archive the iteration to ctx.history and increment ctx.iteration.
    """

    def __init__(
        self,
        steps: list[AgentStep],
        max_iterations: int = 5,
        server: object = None,
        record_env: object = None,
        wandb_logger: WandbLogger | None = None,
        *,
        code_backend: str | None = None,
        code_model: str | None = None,
        reflect_backend: str | None = None,
        reflect_model: str | None = None,
    ) -> None:
        self.steps = steps
        self.max_iterations = max_iterations
        self._server = server  # CapServer mode recording
        self._record_env = record_env  # direct mode recording (env with render_rgb)
        self._wandb = wandb_logger
        # Surfaced to the dashboard so the header shows which models are
        # doing the code-gen / reflection work on this run.
        self._code_backend = code_backend
        self._code_model = code_model
        self._reflect_backend = reflect_backend
        self._reflect_model = reflect_model
        logger.info(
            "AgentPipeline: %d steps, max_iterations=%d  [%s]",
            len(steps),
            max_iterations,
            " → ".join(s.name for s in steps),
        )

    # ------------------------------------------------------------------
    # Config-driven factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg,
        prompt_memory=None,
        *,
        server: object = None,
        record_env: object = None,
    ) -> AgentPipeline:
        """Build a full pipeline from Hydra config.

        Reads ``cfg.steps`` to decide which steps to include, then creates
        each step from the config (LLMs, reflection strategy, etc.).

        Args:
            cfg: Hydra DictConfig (AgentConfig schema).
            prompt_memory: PromptMemory for building system prompts.
            server: CapServer instance for recording (CapServer mode).
            record_env: Env instance for recording (direct mode).
        """
        from omegaconf import open_dict

        from cap.agent.llm import make_llm
        from cap.agent.agent_step import (
            CodeGeneratorStep,
            CodeReviewerStep,
            DryRunCodeStep,
            ObserverStep,
            OracleCodeStep,
            SelfReflectionStep,
            SubprocessExecutorStep,
        )

        oracle_mode = bool(getattr(cfg, "oracle", None))

        # Derive reflect flag from steps list (skipped in oracle mode — no LLM)
        with open_dict(cfg):
            if "self_reflection" in cfg.steps and not oracle_mode:
                cfg.reflect = True
            if oracle_mode:
                cfg.reflect = False
                cfg.review = False
                # reflect/review disabled — no LLM in oracle mode.
                # max_iterations is left to the user; multiple iterations are
                # useful for statistical sampling of stochastic policies.

        # --- System prompt ---
        cg_prompt_entries = list(cfg.code_generator.prompt)
        system_prompt = ""
        if prompt_memory and cg_prompt_entries:
            system_prompt = prompt_memory.build_prompt(cg_prompt_entries)

        # --- LLMs ---
        llm = None
        reflect_llm = None
        if not cfg.runtime.dry_run and not oracle_mode:
            llm = make_llm(cfg, system_prompt=system_prompt)
            if cfg.reflect:
                reflect_prompt_entries = list(cfg.reflection.prompt)
                reflect_system_prompt = (
                    prompt_memory.build_prompt(reflect_prompt_entries)
                    if prompt_memory and reflect_prompt_entries
                    else system_prompt
                )
                reflect_llm = make_llm(
                    cfg,
                    system_prompt=reflect_system_prompt,
                    model_override=cfg.reflect_model,
                    replace_system_prompt=True,
                )

        # --- Reflection strategy ---
        reflect_strategy = None
        if reflect_llm is not None:
            from cap.agent.reflection import build_reflection_strategy

            reflect_strategy = build_reflection_strategy(cfg, reflect_llm)

        # --- Observer cameras (for visual diff) ---
        observer_cameras: list[str] = []
        if reflect_strategy is not None and cfg.reflection.strategy in (
            "vision",
            "composite",
        ):
            observer_cameras = list(cfg.reflection.cameras)

        # --- Build step list from cfg.steps ---
        step_map: dict[str, AgentStep] = {}

        # Oracle loads a fixed script — there's no code to author or assemble,
        # and cfg.reflect / cfg.review are already force-disabled above (no
        # LLM is constructed). Collapse the two-stage code-gen slots into
        # code_generator and drop the reflect/review slots so the assembly
        # loop doesn't complain about steps we deliberately didn't register.
        resolved_steps: list[str] = list(cfg.steps)
        if oracle_mode:
            _DROP_IN_ORACLE = {"self_reflection", "code_reviewer"}
            collapsed: list[str] = []
            for s in resolved_steps:
                if s in _DROP_IN_ORACLE:
                    continue
                if s in ("skill_author", "assembly_generator"):
                    s = "code_generator"
                if collapsed and collapsed[-1] == s:
                    continue  # drop back-to-back duplicates from the collapse
                collapsed.append(s)
            resolved_steps = collapsed

        # Code generator: oracle, dry-run, or LLM
        if oracle_mode:
            step_map["code_generator"] = OracleCodeStep(cfg.oracle)
        elif cfg.runtime.dry_run:
            step_map["code_generator"] = DryRunCodeStep()
        elif llm is not None:
            step_map["code_generator"] = CodeGeneratorStep(llm)

        # Two-stage code generator (see docs/plan/two_stage_code_generator.md).
        # Only wire these up if the experiment lists them in cfg.steps — so
        # experiments still using `code_generator` are unaffected.
        if (
            not oracle_mode
            and not cfg.runtime.dry_run
            and (
                "skill_author" in resolved_steps
                or "assembly_generator" in resolved_steps
            )
        ):
            from cap.agent.assembly_generator_step import AssemblyGeneratorStep
            from cap.agent.skill_author_step import SkillAuthorStep

            skill_author_cfg = getattr(cfg, "skill_author", None)
            sa_prompt_entries = (
                list(skill_author_cfg.prompt) if skill_author_cfg else []
            )
            sa_system_prompt = (
                prompt_memory.build_prompt(sa_prompt_entries)
                if prompt_memory and sa_prompt_entries
                else system_prompt
            )
            sa_llm = make_llm(
                cfg,
                system_prompt=sa_system_prompt,
                replace_system_prompt=True,
            )
            step_map["skill_author"] = SkillAuthorStep(
                sa_llm,
                max_retries=skill_author_cfg.max_retries if skill_author_cfg else 2,
                allow_no_new_skills=(
                    skill_author_cfg.allow_no_new_skills if skill_author_cfg else True
                ),
            )

            asm_cfg = getattr(cfg, "assembly_generator", None)
            asm_prompt_entries = list(asm_cfg.prompt) if asm_cfg else []
            asm_system_prompt = (
                prompt_memory.build_prompt(asm_prompt_entries)
                if prompt_memory and asm_prompt_entries
                else system_prompt
            )
            asm_llm = make_llm(
                cfg,
                system_prompt=asm_system_prompt,
                replace_system_prompt=True,
            )
            step_map["assembly_generator"] = AssemblyGeneratorStep(
                asm_llm,
                max_retries=asm_cfg.max_retries if asm_cfg else 2,
                forbid_toplevel_def=(asm_cfg.forbid_toplevel_def if asm_cfg else True),
                require_skill_library_imports=(
                    asm_cfg.require_skill_library_imports if asm_cfg else False
                ),
            )

        step_map["observer"] = ObserverStep(cameras=observer_cameras)
        step_map["executor"] = SubprocessExecutorStep()

        if cfg.review and not cfg.runtime.dry_run and llm is not None:
            review_llm = make_llm(
                cfg,
                system_prompt=system_prompt,
                model_override=cfg.review_model,
            )
            step_map["code_reviewer"] = CodeReviewerStep(review_llm)

        if reflect_strategy is not None:
            step_map["self_reflection"] = SelfReflectionStep(reflect_strategy)

        # Steps folded into subprocess execution (no separate step needed)
        _FOLDED = {"reward_evaluator"}

        # Assemble in cfg.steps order
        steps: list[AgentStep] = []
        seen: set[str] = set()
        for name in resolved_steps:
            if name in _FOLDED:
                continue
            if name not in step_map:
                raise ValueError(
                    f"Unknown step {name!r} in cfg.steps. Known: "
                    f"{sorted(step_map)}. "
                    "Check experiment YAML + flags (e.g. oracle mode collapses "
                    "skill_author/assembly_generator into code_generator)."
                )
            if name not in seen:
                steps.append(step_map[name])
                seen.add(name)

        # --- Wandb ---

        wb_logger = None
        if cfg.wandb.enabled:
            # Session not created yet at this point — defer to caller.
            # WandbLogger is passed in via the pipeline constructor by main().
            pass

        # Surface the model names we'll actually use this run — the
        # dashboard displays them in its header. Code-gen = main cfg.llm
        # (reused by skill_author / assembly_generator / self_reflection
        # text); reflect = the VLM used for visual analysis.
        code_backend = cfg.llm.backend if cfg.llm else None
        code_model = cfg.llm.model if cfg.llm else None
        reflect_backend = (
            cfg.reflection.vlm_backend
            if cfg.reflect and getattr(cfg, "reflection", None)
            else None
        )
        reflect_model = (
            cfg.reflection.vlm_model
            if cfg.reflect and getattr(cfg, "reflection", None)
            else None
        )

        return cls(
            steps=steps,
            max_iterations=cfg.max_iterations,
            server=server,
            record_env=record_env,
            wandb_logger=wb_logger,
            code_backend=code_backend,
            code_model=code_model,
            reflect_backend=reflect_backend,
            reflect_model=reflect_model,
        )

    def run(self, ctx: AgentContext) -> AgentContext:
        """Run the pipeline loop. Returns the final context."""
        import sys as _sys

        from cap.agent.agent_dashboard import AgentDashboard, dashboard_enabled
        from cap.agent.providers.nvidia import (
            PROVIDER_URL_ENV,
            SCHEDULER_DB_ENV,
            TELEMETRY_FILE_ENV,
            apply_nvidia_health_results,
            health_check_nvidia_keys,
            init_nvidia_scheduler,
            list_nvidia_keys,
        )

        ctx.max_iterations = self.max_iterations
        t0 = time.time()
        previous_nvidia_telemetry_file = os.environ.get(TELEMETRY_FILE_ENV)
        previous_nvidia_scheduler_db = os.environ.get(SCHEDULER_DB_ENV)
        if ctx.session is not None:
            os.environ[TELEMETRY_FILE_ENV] = str(
                ctx.session.run_dir / "nvidia_requests.jsonl"
            )
            nvidia_in_use = "nvidia" in {
                str(self._code_backend or "").lower(),
                str(self._reflect_backend or "").lower(),
            }
            nvidia_provider_url = os.environ.get(PROVIDER_URL_ENV, "").strip()
            nvidia_keys = (
                list_nvidia_keys() if nvidia_in_use and not nvidia_provider_url else []
            )
            if nvidia_keys:
                scheduler_db = (
                    f"{tempfile.gettempdir()}/"
                    f"cap_nvidia_{ctx.session.run_dir.name}_{os.getpid()}.sqlite"
                )
                init_nvidia_scheduler(scheduler_db, keys=nvidia_keys)
                health_model = (
                    self._reflect_model
                    or self._code_model
                    or "gcp/google/gemini-3-flash-preview"
                )
                try:
                    health = health_check_nvidia_keys(
                        model=health_model,
                        max_workers=min(4, len(nvidia_keys)),
                    )
                    apply_nvidia_health_results(health)
                    healthy = sum(1 for row in health if row["status"] == "healthy")
                    logger.info(
                        "NVIDIA key health: healthy=%d/%d model=%s",
                        healthy,
                        len(health),
                        health_model,
                    )
                except Exception:
                    logger.warning("NVIDIA key health check failed", exc_info=True)

        # Activate parent dashboard iff stdout is a TTY + rich is importable.
        use_dashboard = dashboard_enabled(_sys.__stdout__)
        dash: AgentDashboard | None = (
            AgentDashboard(
                task=ctx.task,
                max_iterations=self.max_iterations,
                session=ctx.session,
                code_backend=self._code_backend,
                code_model=self._code_model,
                reflect_backend=self._reflect_backend,
                reflect_model=self._reflect_model,
            )
            if use_dashboard
            else None
        )

        import contextlib

        try:
            with dash if dash is not None else contextlib.nullcontext():
                self._publish_startup_service_health(ctx, dash)
                return self._run_loop(ctx, t0, dash)
        finally:
            if previous_nvidia_telemetry_file is None:
                os.environ.pop(TELEMETRY_FILE_ENV, None)
            else:
                os.environ[TELEMETRY_FILE_ENV] = previous_nvidia_telemetry_file
            if previous_nvidia_scheduler_db is None:
                os.environ.pop(SCHEDULER_DB_ENV, None)
            else:
                os.environ[SCHEDULER_DB_ENV] = previous_nvidia_scheduler_db

    def _publish_startup_service_health(self, ctx: AgentContext, dash: Any) -> None:
        """Probe configured runtime services once at run start."""
        try:
            from cap.agent.service_health import (
                check_runtime_services,
                format_service_health_lines,
                save_service_health,
            )

            rows = check_runtime_services(ctx.config, timeout_s=1.0)
            if ctx.session is not None:
                save_service_health(rows, ctx.session.run_dir / "service_health.json")
            if dash is not None:
                dash.on_service_health(rows)
            elif rows:
                print("[run_agent] Startup service health:")
                for line in format_service_health_lines(rows):
                    print(f"  {line}")
            unhealthy = [row for row in rows if row.get("status") != "healthy"]
            if unhealthy:
                logger.warning("Startup service health has %d unhealthy rows", len(unhealthy))
        except Exception:
            logger.warning("Startup service health check failed", exc_info=True)

    def _run_loop(
        self,
        ctx: AgentContext,
        t0: float,
        dash: Any,
    ) -> AgentContext:
        while not ctx.should_stop and ctx.iteration < self.max_iterations:
            iter_t0 = time.time()
            logger.info("─── Iteration %d / %d ───", ctx.iteration, self.max_iterations)
            if dash is None:
                print(f"\n{'═' * 60}")
                print(f"  Iteration {ctx.iteration} / {self.max_iterations}")
                print(f"{'═' * 60}")
            else:
                dash.on_iteration_start(ctx.iteration)
            ctx.reset_iteration_state()

            _recording_started = False
            step_timings: dict[str, float] = {}
            for step in self.steps:
                logger.debug("  step: %s", step.name)
                # Notify dashboard — for executor, also pass n_seeds so seed
                # panel can show progress against the expected total.
                if dash is not None:
                    n_seeds_hint = 0
                    if step.name == "executor":
                        exec_cfg = getattr(
                            getattr(ctx, "config", None), "execution", None
                        )
                        n_seeds_hint = exec_cfg.n_seeds if exec_cfg else 1
                        cg_cfg = getattr(
                            getattr(ctx, "config", None), "code_generator", None
                        )
                        n_candidates = int(
                            getattr(cg_cfg, "num_candidates", 1) or 1
                        )
                        n_seeds_hint *= max(1, n_candidates)
                    dash.on_step_start(step.name, n_seeds=n_seeds_hint)
                step_t0 = time.time()
                try:
                    ctx = step.run(ctx)
                except Exception as e:
                    import traceback as _tb
                    logger.exception("Step %s raised an exception", step.name)
                    print(f"[pipeline] EXCEPTION in step {step.name}: {e}")
                    print(_tb.format_exc())
                    ctx.should_stop = True
                    ctx.stop_reason = f"step_error:{step.name}:{e}"
                    step_ms = (time.time() - step_t0) * 1000
                    step_timings[step.name] = step_ms
                    if dash is not None:
                        dash.on_step_end(step.name, step_ms)
                    else:
                        print(f"  [step] {step.name} -> ERROR {step_ms:.0f}ms: {e}")
                    break

                step_ms = (time.time() - step_t0) * 1000
                step_timings[step.name] = step_ms
                if dash is not None:
                    dash.on_step_end(step.name, step_ms)
                else:
                    print(f"  [step] {step.name} -> {step_ms:.0f}ms")

                # Start per-iteration recording AFTER the observer step
                # (which resets the env) so the video doesn't contain
                # stale frames from the previous iteration.
                if not _recording_started and step.name == "observer":
                    if ctx.session is not None:
                        if self._server is not None:
                            ctx.session.start_iter_recording(
                                self._server, ctx.iteration
                            )
                        elif self._record_env is not None:
                            ctx.session.start_iter_recording_env(
                                self._record_env, ctx.iteration
                            )
                    _recording_started = True

                if ctx.should_stop:
                    break

            # Stop recorder for this iteration before archiving
            if ctx.session is not None:
                ctx.session.stop_iter_recording()

            iter_ms = (time.time() - iter_t0) * 1000
            score = ctx.evaluation.score if ctx.evaluation else 0.0
            success = (
                bool(ctx.evaluation and ctx.evaluation.success)
                if ctx.evaluation
                else False
            )
            if dash is None:
                print(
                    f"  [iter {ctx.iteration}] total={iter_ms:.0f}ms  score={score:.3f}  success={success}"
                )

            # wandb: log per-iteration metrics
            if self._wandb is not None:
                tool_calls = ctx.memory.get_iteration(ctx.iteration)
                exec_err = None
                if ctx.execution_result and ctx.execution_result.error:
                    exec_err = ctx.execution_result.error
                eval_details = (
                    ctx.evaluation.details
                    if ctx.evaluation and ctx.evaluation.details
                    else {}
                )

                # Collect profiling paths + seed values from the selected
                # candidate's seeds. Candidate-search runs keep non-selected
                # candidates under candidate_search.all_per_seed.
                prof_entries = []
                if ctx.session is not None:
                    exec_cfg = getattr(getattr(ctx, "config", None), "execution", None)
                    n_seeds = exec_cfg.n_seeds if exec_cfg else 1
                    seed_start = exec_cfg.seed_start if exec_cfg else 0
                    per_seed_rows = eval_details.get("per_seed") or []
                    if per_seed_rows:
                        iterator = [
                            (
                                int(row.get("seed", seed_start + idx)),
                                int(row.get("exec_id", idx)),
                            )
                            for idx, row in enumerate(per_seed_rows)
                        ]
                    else:
                        iterator = [(seed_start + i, i) for i in range(n_seeds)]
                    for seed_value, exec_id in iterator:
                        prof_entries.append(
                            (
                                seed_value,
                                ctx.session.exec_dir(ctx.iteration, exec_id)
                                / "profiling.json",
                            )
                        )

                # Per-seed aggregate stats for parallel executor
                exec_cfg = getattr(getattr(ctx, "config", None), "execution", None)
                cfg_n_seeds = exec_cfg.n_seeds if exec_cfg else 1
                n_seeds_val = int(eval_details.get("n_seeds", cfg_n_seeds) or 1)
                successes_val = int(eval_details.get("successes", int(success)))
                success_rate_val = float(
                    eval_details.get(
                        "success_rate",
                        (successes_val / n_seeds_val) if n_seeds_val else 0.0,
                    )
                )
                avg_score_val = float(eval_details.get("avg_score", score))

                iter_records = [
                    r for r in ctx.llm_usage if r.iteration == ctx.iteration
                ]
                iter_token_in = sum(r.input_tokens for r in iter_records)
                iter_token_out = sum(r.output_tokens for r in iter_records)
                self._wandb.log_iteration(
                    iteration=ctx.iteration,
                    step_timings=step_timings,
                    total_ms=iter_ms,
                    score=score,
                    success=success,
                    code_length=len(ctx.code or ""),
                    tool_call_count=len(tool_calls),
                    execution_error=exec_err,
                    profiling_entries=prof_entries,
                    n_seeds=n_seeds_val,
                    successes=successes_val,
                    success_rate=success_rate_val,
                    avg_score=avg_score_val,
                    token_input=iter_token_in,
                    token_output=iter_token_out,
                )

            # Aggregate per-tool logs under tool_log/ (independent of wandb)
            if ctx.session is not None:
                from cap.agent.tool_log import update_tool_logs

                exec_cfg = getattr(getattr(ctx, "config", None), "execution", None)
                n_seeds = exec_cfg.n_seeds if exec_cfg else 1
                seed_start = exec_cfg.seed_start if exec_cfg else 0
                eval_details = (
                    ctx.evaluation.details
                    if ctx.evaluation and ctx.evaluation.details
                    else {}
                )
                per_seed_rows = eval_details.get("per_seed") or []
                if per_seed_rows:
                    tool_entries = [
                        (
                            int(row.get("seed", seed_start + idx)),
                            ctx.session.exec_dir(
                                ctx.iteration,
                                int(row.get("exec_id", idx)),
                            )
                            / "profiling.json",
                        )
                        for idx, row in enumerate(per_seed_rows)
                    ]
                else:
                    tool_entries = [
                        (
                            seed_start + i,
                            ctx.session.exec_dir(ctx.iteration, i) / "profiling.json",
                        )
                        for i in range(n_seeds)
                    ]
                update_tool_logs(ctx.session.run_dir, ctx.iteration, tool_entries)

            # Notify dashboard about completed iteration
            if dash is not None:
                eval_details = (
                    ctx.evaluation.details
                    if ctx.evaluation and ctx.evaluation.details
                    else {}
                )
                exec_cfg = getattr(getattr(ctx, "config", None), "execution", None)
                cfg_n_seeds = exec_cfg.n_seeds if exec_cfg else 1
                n_seeds_val = int(eval_details.get("n_seeds", cfg_n_seeds) or 1)
                successes_val = int(eval_details.get("successes", int(success)))
                success_rate_val = float(
                    eval_details.get(
                        "success_rate",
                        (successes_val / n_seeds_val) if n_seeds_val else 0.0,
                    )
                )
                avg_score_val = float(eval_details.get("avg_score", score))
                dash.on_iteration_end(
                    iteration=ctx.iteration,
                    success=success,
                    n_seeds=n_seeds_val,
                    successes=successes_val,
                    success_rate=success_rate_val,
                    avg_score=avg_score_val,
                )

            # Snapshot the trial skill library before any rollback so we can
            # archive discarded attempts while restoring the last kept state.
            decision = "keep"
            if ctx.session is not None:
                try:
                    decision = ctx.session.record_skill_library_trial(
                        ctx.iteration,
                        score=score,
                        success=success,
                    )
                except Exception:
                    logger.exception(
                        "Failed to snapshot skill library state for iter %d",
                        ctx.iteration,
                    )

            # Generate one-paragraph history entry and append to history.md.
            # The LLM writes the paragraph knowing the actual score that resulted,
            # so the entry is retrospective ("tried X, got Y") not predictive.
            if ctx.session is not None and ctx.code:
                try:
                    _write_history_entry(
                        ctx=ctx,
                        steps=self.steps,
                        score=score,
                        decision=decision,
                    )
                except Exception:
                    logger.warning(
                        "Failed to write history entry for iter %d",
                        ctx.iteration,
                        exc_info=True,
                    )

            # Update dashboard with current skill count
            if dash is not None and ctx.session is not None:
                try:
                    sl = getattr(ctx.session, "_skill_library", None)
                    if sl is not None:
                        dash.on_skills_updated(sl.total_skills, sl.total_families)
                except Exception:
                    pass

            # Archive this iteration before looping
            ctx.snapshot_iteration()
            ctx.iteration += 1

        duration = time.time() - t0
        success = (
            bool(ctx.evaluation and ctx.evaluation.success) if ctx.evaluation else False
        )
        score = ctx.evaluation.score if ctx.evaluation else 0.0
        if dash is None:
            print(f"\n{'═' * 60}")
            print(
                f"  Pipeline done: {ctx.iteration} iter, success={success}, score={score:.3f}, {duration:.1f}s"
            )
            print(f"{'═' * 60}")
        logger.info(
            "AgentPipeline done: %d iteration(s), success=%s, score=%.3f, stop_reason=%r, %.1fs",
            ctx.iteration,
            success,
            score,
            ctx.stop_reason or "max_iterations",
            duration,
        )
        return ctx

    def result(self, ctx: AgentContext, duration_s: float = 0.0) -> AgentRunResult:
        """Build a summary result from the final context."""
        success = (
            bool(ctx.evaluation and ctx.evaluation.success) if ctx.evaluation else False
        )
        return AgentRunResult(
            success=success,
            iterations=ctx.iteration,
            stop_reason=ctx.stop_reason or ("success" if success else "max_iterations"),
            final_score=ctx.evaluation.score if ctx.evaluation else 0.0,
            duration_s=duration_s,
        )


def _write_history_entry(
    ctx: "AgentContext",
    steps: list,
    score: float,
    decision: str,
) -> None:
    """Generate a one-paragraph history entry via the assembly LLM and append
    it to {run_dir}/history.md.

    The paragraph is written AFTER the score is known so it can describe
    what was tried and what it achieved — retrospective, not predictive.
    """
    import re as _re

    if ctx.session is None or not ctx.code:
        return

    # Find the assembly LLM from the pipeline steps.
    # Fall back to any step that has a generate_text-capable LLM.
    asm_step = next((s for s in steps if s.name == "assembly_generator"), None)
    if asm_step is None:
        asm_step = next(
            (s for s in steps if getattr(s, "_llm", None) is not None), None
        )
    if asm_step is None:
        logger.warning("_write_history_entry: no LLM step found, skipping history")
        return
    llm = getattr(asm_step, "_llm", None)
    if llm is None:
        logger.warning("_write_history_entry: step %s has no _llm, skipping history", asm_step.name)
        return

    # Build succinct context for the LLM
    eval_details = (
        ctx.evaluation.details if ctx.evaluation and ctx.evaluation.details else {}
    )
    n_seeds = int(eval_details.get("n_seeds", 1) or 1)
    successes = int(eval_details.get("successes", int(score >= 1.0)))

    # Extract just the authored skill names and deployed imports — keep the
    # context minimal so the LLM produces a tight 1-2 sentence summary.
    authored_skills = ""
    deployed_skills = ""
    if ctx.session is not None:
        sa_path = ctx.session.iterations_dir(ctx.iteration) / "skill_author_response.md"
        if sa_path.exists():
            import re as _re2
            authored_skills = ", ".join(
                _re2.findall(
                    r"^### (?:new|refine|replace): (\S+)",
                    sa_path.read_text(encoding="utf-8"),
                    _re2.MULTILINE,
                )
            ) or "(none)"
    if ctx.code:
        deployed_skills = ", ".join(
            line.split("import ")[-1].strip()
            for line in ctx.code.splitlines()
            if line.startswith("from skill_library")
        ) or "(none)"

    # Include baseline score so the LLM can state whether this iteration
    # improved or regressed — making the history paragraph useful for
    # future iterations to know which deployed skills helped or hurt.
    prev_baseline = getattr(ctx.session, "_baseline_score", 0.0) if ctx.session else 0.0
    direction = (
        f"IMPROVED from {prev_baseline:.3f}"
        if score > prev_baseline
        else f"REGRESSED from {prev_baseline:.3f}"
        if score < prev_baseline
        else f"TIED baseline {prev_baseline:.3f}"
    )

    instruction_path = (
        Path(__file__).resolve().parents[2]
        / "cap/prompt/system/iteration_history.md"
    )
    try:
        instruction = instruction_path.read_text(encoding="utf-8").strip()
    except OSError:
        instruction = "Write 1-2 sentences in <history></history> tags naming deployed skills and whether score improved or regressed."

    prompt = (
        f"Robot agent iteration summary:\n"
        f"- Skills authored: {authored_skills}\n"
        f"- Skills deployed: {deployed_skills}\n"
        f"- Result: {successes}/{n_seeds} seeds succeeded (score={score:.3f}), {direction}\n"
        f"- Decision: {decision}\n\n"
        f"{instruction}"
    )

    try:
        raw = llm.generate_text(prompt)
        from cap.agent.llm.usage import record_llm_usage
        record_llm_usage(ctx, llm, "history_writer")
        m = _re.search(r"<history>(.*?)</history>", raw, _re.DOTALL)
        paragraph = m.group(1).strip() if m else raw.strip()
    except Exception:
        paragraph = (
            f"iter_{ctx.iteration}: score={score:.3f} ({successes}/{n_seeds}), "
            f"decision={decision}."
        )

    ctx.history_paragraph = paragraph

    history_path = ctx.session.run_dir / "history.md"
    entry = (
        f"\n## iter_{ctx.iteration:03d} — score={score:.3f} — {decision}\n\n"
        f"<history>{paragraph}</history>\n"
    )
    try:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(entry)
        print(f"[history] wrote iter_{ctx.iteration:03d} → {history_path}")
    except Exception as exc:
        print(f"[history] FAILED to write iter_{ctx.iteration:03d}: {exc}")
