# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in pipeline steps for the CAP agent system.

Each step reads from and writes to an AgentContext.  Compose them into an
AgentPipeline for the full generate → execute → evaluate loop.

Built-in steps:
    ObserverStep        — snapshot robot state and task info
    CodeGeneratorStep   — LLM generates code from task + context
    CodeReviewerStep    — second LLM reviews code before execution
    ExecutorStep        — sandboxed code execution
    RewardEvaluatorStep — get_task_info() success/reward signal
    SelfReflectionStep  — LLM analyzes failure and suggests improvements
"""

from __future__ import annotations

import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from enpire.env.forge.cap.agent.agent_context import AgentContext
    from enpire.env.forge.cap.agent.llm.base import LLMBackend
    from enpire.env.forge.cap.agent.reflection import ReflectionStrategy

logger = logging.getLogger(__name__)


def _dash_print(msg: str) -> None:
    """Print that silently no-ops while a parent AgentDashboard owns the terminal."""
    try:
        from enpire.env.forge.cap.agent.agent_dashboard import is_active

        if is_active():
            return
    except ImportError:
        pass
    print(msg)


def _read_run_log_tail(exec_dir: Any, *, max_chars: int = 2000) -> str:
    """Return the last ``max_chars`` of the seed's run_profiling_*.txt log.

    That log captures the script's own stdout (skill calls + their
    status-bearing results) and is the richest signal we have for the
    skill-author / reflection loop. Without this, per-seed feedback is
    just ``reward=0.0, success=False`` — which told iter_N+1 authoring
    nothing about *why* a seed failed.
    """
    try:
        logs = sorted(exec_dir.glob("run_profiling_*.txt"))
        if not logs:
            return ""
        txt = logs[-1].read_text(encoding="utf-8", errors="replace")
        return txt[-max_chars:] if len(txt) > max_chars else txt
    except Exception:
        return ""


def _save_llm_input(
    ctx: AgentContext,
    llm: LLMBackend,
    step_name: str,
) -> None:
    """Save the full LLM exchange (system + user + response) to disk.

    Writes ``conversations/iter_NNN_<step>_<timestamp>.md`` with three
    sections: System Prompt, User Message, Assistant Response. Reads from
    the backend's ``last_system_prompt`` / ``last_user_prompt`` /
    ``last_response`` attributes, all of which every LLMBackend populates
    on each generate_* call.

    Also records token usage via ``record_llm_usage`` if the backend
    populated ``last_usage``.
    """
    from enpire.env.forge.cap.agent.llm.usage import record_llm_usage
    record_llm_usage(ctx, llm, step_name)

    if ctx.session is None:
        return
    import time as _time

    sys_prompt = getattr(llm, "last_system_prompt", None)
    user_prompt = getattr(llm, "last_user_prompt", None)
    response = getattr(llm, "last_response", None)
    if not sys_prompt and not user_prompt and not response:
        return

    parts: list[str] = []
    if sys_prompt:
        parts.append(f"# System Prompt\n\n{sys_prompt}\n")
    if user_prompt:
        parts.append(f"# User Message\n\n{user_prompt}\n")
    if response:
        parts.append(f"# Assistant Response\n\n{response}\n")

    stamp = _time.strftime("%H%M%S")
    conv_dir = ctx.session.conversations_dir()
    filename = f"iter_{ctx.iteration:03d}_{step_name}_{stamp}.md"
    (conv_dir / filename).write_text("\n---\n\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class AgentStep(ABC):
    """A single step in an AgentPipeline.

    Implement ``run(ctx)`` to read from and write to the shared AgentContext.
    Return the (possibly modified) context.
    """

    name: str = "unnamed_step"

    @abstractmethod
    def run(self, ctx: AgentContext) -> AgentContext:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# ObserverStep
# ---------------------------------------------------------------------------


class ObserverStep(AgentStep):
    """Snapshot robot state and task info at the start of each iteration.

    On iteration > 0, resets the environment so each attempt starts from
    the same initial state.

    Writes: ctx.robot_state, ctx.task_info
    """

    name = "observer"

    def __init__(
        self,
        reset_between_iterations: bool = True,
        cameras: list[str] | None = None,
    ) -> None:
        self._reset = reset_between_iterations
        self._cameras = cameras or []

    def run(self, ctx: AgentContext) -> AgentContext:
        # Reset env on subsequent iterations so each attempt starts from the
        # same initial state.  Prefers reset_to_initial (legacy, deterministic
        # state restore), falls back to reset_env (teardown + recreate with
        # same seed for RoboCasa, or full re-randomization otherwise).
        if self._reset and ctx.iteration > 0:
            reset_fn = ctx.namespace.get(
                "reset_to_initial", ctx.namespace.get("reset_env")
            )
            if reset_fn is not None:
                try:
                    reset_fn()
                    logger.info(
                        "ObserverStep: env reset for iteration %d", ctx.iteration
                    )
                except Exception as e:
                    logger.warning("ObserverStep: reset failed: %s", e)

        get_state = ctx.namespace.get("get_robot_state")
        if get_state is not None:
            try:
                ctx.robot_state = get_state()
            except Exception as e:
                logger.warning("ObserverStep: get_robot_state failed: %s", e)

        get_task = ctx.namespace.get("get_task_info")
        if get_task is not None:
            try:
                ctx.task_info = get_task()
            except Exception as e:
                logger.warning("ObserverStep: get_task_info failed: %s", e)

        # Capture before-frames for visual differencing
        if self._cameras:
            get_camera_image = ctx.namespace.get("get_camera_image")
            if get_camera_image is not None:
                frames: dict[str, Any] = {}
                for cam in self._cameras:
                    try:
                        img = get_camera_image(cam)
                        if img is not None and img.shape[0] > 1 and img.shape[1] > 1:
                            frames[cam] = img
                    except Exception as e:
                        logger.warning(
                            "ObserverStep: get_camera_image(%r) failed: %s", cam, e
                        )
                if frames:
                    ctx.frames_before = frames

        return ctx


# ---------------------------------------------------------------------------
# CodeGeneratorStep
# ---------------------------------------------------------------------------


class CodeGeneratorStep(AgentStep):
    """Generate code for the task using an LLMBackend.

    Reads:  ctx.task, ctx.robot_state, ctx.memory, ctx.evaluation (feedback),
            ctx.review_feedback
    Writes: ctx.code
    Side effects: session.save_code()
    """

    name = "code_generator"

    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def run(self, ctx: AgentContext) -> AgentContext:
        robot_state_dict = _serialize_robot_state(ctx.robot_state)
        config = getattr(ctx, "config", None)
        th_max = config.code_generator.tool_history_max_chars if config else 3000
        history_summary = ctx.memory.summarize_for_llm(max_chars=th_max)

        # Build failure history from ALL prior iterations (not just last one)
        failure_history: list[dict[str, Any]] = []
        for rec in ctx.history:
            stdout = ""
            if rec.execution_result is not None:
                stdout = getattr(rec.execution_result, "stdout", "") or ""
            failure_history.append(
                {
                    "iteration": rec.iteration,
                    "code": rec.code,
                    "thoughts": rec.thoughts,
                    "feedback": rec.evaluation.feedback
                    if rec.evaluation
                    else "(no feedback)",
                    "stdout": stdout,
                    "score": float(rec.evaluation.score) if rec.evaluation else 0.0,
                }
            )

        # Champion: the highest-scoring attempt so far. Surfaced separately so
        # the LLM prompt can frame subsequent iterations as "improve from the
        # best" rather than "fix the most recent failure".
        champion_rec = _best_iteration(ctx.history)
        champion: dict[str, Any] | None = None
        if champion_rec is not None:
            champion = {
                "iteration": champion_rec.iteration,
                "code": champion_rec.code,
                "thoughts": champion_rec.thoughts,
                "score": float(champion_rec.evaluation.score)
                if champion_rec.evaluation
                else 0.0,
                "feedback": champion_rec.evaluation.feedback
                if champion_rec.evaluation
                else "",
            }

        # Pass config to LLM so it can read prompt templates
        config = getattr(ctx, "config", None)

        llm_context: dict[str, Any] = {
            "tools": ctx.tool_schemas,
            "robot_state": robot_state_dict,
            "history": history_summary,
            "iteration": ctx.iteration,
            "failure_history": failure_history,
            "champion": champion,
            "config": config,
        }
        if ctx.env_spec:
            llm_context["env_spec"] = ctx.env_spec
        if ctx.task_info:
            llm_context["task_info"] = ctx.task_info
        if ctx.session is not None:
            sl = getattr(ctx.session, "_skill_library", None)
            if sl is not None:
                # Always set the key (even empty string) so bridge_llm shows the section
                llm_context["skill_library_index"] = sl.index_for_prompt()

        logger.info(
            "CodeGeneratorStep: generating code (iter %d, %d prior failures)",
            ctx.iteration,
            len(failure_history),
        )

        cg_cfg = getattr(config, "code_generator", None)
        num_candidates = max(1, int(getattr(cg_cfg, "num_candidates", 1) or 1))
        candidate_parallelism = max(
            1, int(getattr(cg_cfg, "candidate_parallelism", 1) or 1)
        )
        backend_name = str(getattr(getattr(config, "llm", None), "backend", "")).lower()
        if backend_name.startswith("bridge"):
            # Bridge backends keep conversational session state; run candidate
            # requests serially to avoid interleaving turns.
            candidate_parallelism = 1

        def _candidate_task(candidate_index: int) -> str:
            if num_candidates <= 1:
                return ctx.task
            return (
                f"{ctx.task}\n\n"
                "CANDIDATE SEARCH MODE:\n"
                f"You are candidate {candidate_index + 1} of {num_candidates}. "
                "Write a complete standalone solution, but deliberately choose a "
                "somewhat different deterministic strategy, thresholds, fallback "
                "logic, or use of available tools than the other candidates are "
                "likely to choose. Do not mention that this is a candidate in the "
                "Python program output."
            )

        def _generate_one(candidate_index: int) -> tuple[int, str]:
            cand_context = dict(llm_context)
            cand_context["candidate_index"] = candidate_index
            cand_context["candidate_count"] = num_candidates
            raw = self._llm.generate_code(_candidate_task(candidate_index), cand_context)
            # Accurate per-candidate prompt logging is only safe when calls are
            # serial; parallel backends share ``last_*`` attributes.
            if candidate_parallelism == 1:
                step_suffix = (
                    "generator"
                    if num_candidates == 1
                    else f"generator_candidate_{candidate_index:03d}"
                )
                _save_llm_input(ctx, self._llm, step_suffix)
            return candidate_index, raw

        if num_candidates == 1 or candidate_parallelism == 1:
            raw_results = [_generate_one(i) for i in range(num_candidates)]
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            workers = min(num_candidates, candidate_parallelism)
            _dash_print(
                f"  [code_generator] generating {num_candidates} candidates "
                f"(parallel × {workers})"
            )
            raw_results = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_generate_one, i) for i in range(num_candidates)]
                for fut in as_completed(futures):
                    raw_results.append(fut.result())
            raw_results.sort(key=lambda item: item[0])

        candidates: list[dict[str, Any]] = []
        for candidate_index, raw_response in raw_results:
            thoughts, code = _split_thoughts_and_code(raw_response)
            candidates.append(
                {
                    "index": int(candidate_index),
                    "thoughts": thoughts,
                    "code": code,
                    "raw_response": raw_response,
                }
            )

        ctx.code_candidates = candidates
        ctx.thoughts = candidates[0]["thoughts"] if candidates else ""
        ctx.code = candidates[0]["code"] if candidates else ""

        if ctx.session is not None:
            # Keep the legacy top-level code.py populated with candidate 0
            # until the executor overwrites it with the best candidate.
            ctx.session.save_code(ctx.iteration, ctx.code or "")
            if ctx.thoughts:
                ctx.session.thoughts_path(ctx.iteration).write_text(
                    ctx.thoughts, encoding="utf-8"
                )
            if len(candidates) > 1:
                import json as _json

                cand_root = ctx.session.iterations_dir(ctx.iteration) / "candidates"
                cand_root.mkdir(parents=True, exist_ok=True)
                manifest = []
                for cand in candidates:
                    cdir = cand_root / f"cand_{cand['index']:03d}"
                    cdir.mkdir(parents=True, exist_ok=True)
                    (cdir / "code.py").write_text(cand["code"], encoding="utf-8")
                    (cdir / "thoughts.md").write_text(
                        cand.get("thoughts", "") or "", encoding="utf-8"
                    )
                    manifest.append(
                        {
                            "index": cand["index"],
                            "code_path": str(cdir / "code.py"),
                            "thoughts_path": str(cdir / "thoughts.md"),
                        }
                    )
                (cand_root / "manifest.json").write_text(
                    _json.dumps(manifest, indent=2) + "\n",
                    encoding="utf-8",
                )

        # Extract new atomic skill functions and save to skill_library/ BEFORE execution
        # so that imports from skill_library/ in ctx.code resolve when the executor runs.
        if ctx.session is not None:
            sl = getattr(ctx.session, "_skill_library", None)
            if sl is not None:
                try:
                    from enpire.env.forge.cap.agent.reflection import SkillPromoter
                    n_saved = SkillPromoter(sl).promote(ctx, [])
                    if n_saved > 0:
                        logger.info(
                            "CodeGeneratorStep: saved %d new skill(s) to library (iter %d)",
                            n_saved,
                            ctx.iteration,
                        )
                except Exception:
                    logger.warning(
                        "CodeGeneratorStep: skill extraction failed (iter %d)",
                        ctx.iteration,
                        exc_info=True,
                    )

        return ctx


# ---------------------------------------------------------------------------
# CodeReviewerStep
# ---------------------------------------------------------------------------


class CodeReviewerStep(AgentStep):
    """Review generated code using a second LLM before execution.

    Reads:  ctx.code, ctx.task
    Writes: ctx.review_feedback
    If the reviewer rejects the code, sets ctx.code = None (triggers re-generation
    on the next pass through the pipeline).
    """

    name = "code_reviewer"

    def __init__(self, llm: LLMBackend, max_retries: int = 1) -> None:
        self._llm = llm
        self._max_retries = max_retries

    def run(self, ctx: AgentContext) -> AgentContext:
        if not ctx.code:
            return ctx

        pm = getattr(ctx, "prompt_memory", None)
        if pm is not None:
            try:
                review_prompt = pm.load(
                    "system", "code_review", task=ctx.task, code=ctx.code
                )
            except FileNotFoundError:
                pm = None
        if pm is None:
            review_prompt = (
                f"Review this robot control code for task: {ctx.task}\n\n"
                f"```python\n{ctx.code}\n```\n\n"
                "Check for: safety issues, logic errors, incorrect tool usage, missing steps.\n"
                "If the code is acceptable, reply 'APPROVED'.\n"
                "If not, reply 'REJECTED: <brief reason>' and suggest a fix."
            )
        review_context: dict[str, Any] = {"tools": ctx.tool_schemas}
        feedback = self._llm.generate_code(review_prompt, review_context)
        _save_llm_input(ctx, self._llm, "reviewer")
        ctx.review_feedback = feedback

        if ctx.session is not None:
            ctx.session.save_review(ctx.iteration, feedback)

        if "REJECTED" in feedback.upper():
            logger.info("CodeReviewerStep: code rejected — %s", feedback[:100])
            ctx.code = None  # force re-generation

        return ctx


# ---------------------------------------------------------------------------
# ExecutorStep
# ---------------------------------------------------------------------------


_DRY_RUN_CODE = """\
import time, numpy as np
state = get_robot_state()
arm = list(state.arms.keys())[0]
info = get_task_info()
print(f"[dry-run] arm={arm}  obj={info.get('obj_name','?')}")
print(f"[dry-run] obj_pos={[round(float(x),3) for x in info.get('obj_pos',[])]}")
print(f"[dry-run] ee_pos={[round(float(x),4) for x in state.arms[arm].ee_pos]}")
print("[dry-run] opening gripper...")
set_gripper(arm, 1.0)
time.sleep(0.3)
print("[dry-run] closing gripper...")
set_gripper(arm, 0.0)
time.sleep(0.3)
info2 = get_task_info()
print(f"[dry-run] success={info2.get('success')} reward={info2.get('reward')}")
"""


class DryRunCodeStep(AgentStep):
    """Skip LLM — emit a fixed open/close gripper script (for testing reset/eval)."""

    name = "code_generator"

    def run(self, ctx: AgentContext) -> AgentContext:
        ctx.code = _DRY_RUN_CODE
        ctx.thoughts = "(dry-run: fixed open/close gripper script)"
        if ctx.session is not None:
            ctx.session.save_code(ctx.iteration, ctx.code)
        return ctx


class OracleCodeStep(AgentStep):
    """Skip the LLM — load a fixed script from disk as ``ctx.code``.

    Used when ``cfg.oracle=<path>`` to benchmark a known-good / hand-written
    script against N seeds. Script path resolution mirrors run_script.py:
    direct / absolute / ~-expanded → repo-relative → ``{saved_scripts_dir}/``.
    """

    name = "code_generator"

    def __init__(self, script_path: str) -> None:
        from pathlib import Path

        self._raw = script_path
        self._cached: Path | None = None

    def _resolve(self, ctx: AgentContext):
        from pathlib import Path

        if self._cached is not None:
            return self._cached
        raw = self._raw.strip().replace("\\", "/")
        if not raw:
            raise FileNotFoundError("Oracle script_file is empty")
        repo_root = _ROOT_PATH
        saved_scripts_dir = None
        cfg = getattr(ctx, "config", None)
        rt = getattr(cfg, "runtime", None)
        if rt is not None:
            ssd = getattr(rt, "saved_scripts_dir", None)
            if ssd:
                saved_scripts_dir = (repo_root / ssd).resolve()
        attempts: list[Path] = []

        def _try(p: Path):
            attempts.append(p)
            return p.resolve() if p.exists() else None

        result = _try(Path(raw).expanduser())
        if result is None and not Path(raw).is_absolute():
            result = _try(repo_root / raw)
        if result is None and saved_scripts_dir is not None:
            candidate = saved_scripts_dir / raw
            result = _try(candidate)
            if result is not None and not result.is_relative_to(saved_scripts_dir):
                result = None
        if result is None:
            tried = "\n".join(f"  Tried: {c}" for c in attempts)
            raise FileNotFoundError(f"Oracle script not found: '{self._raw}'\n{tried}")
        self._cached = result
        return result

    def run(self, ctx: AgentContext) -> AgentContext:
        path = self._resolve(ctx)
        ctx.code = path.read_text(encoding="utf-8")
        ctx.thoughts = f"(oracle mode: executing {path})"
        if ctx.session is not None:
            ctx.session.save_code(ctx.iteration, ctx.code)
        return ctx


# ---------------------------------------------------------------------------


class ExecutorStep(AgentStep):
    """Execute generated code in the sandboxed Executor.

    Reads:  ctx.code, ctx.namespace, ctx.paused_namespace
    Writes: ctx.execution_result, ctx.paused_namespace (if wait_for_agent)
    Side effects: session.save_execution_log()
    """

    name = "executor"

    def run(self, ctx: AgentContext) -> AgentContext:
        if not ctx.code:
            logger.warning("ExecutorStep: no code to execute")
            return ctx

        from enpire.env.forge.cap.agent.executor import Executor
        from enpire.env.forge.cap.agent.profiler import set_tool_event_hooks

        # Wire execution memory into profiler hooks
        memory = ctx.memory
        iteration = ctx.iteration

        def _on_tool_end(
            name: str, call_id: int, result: Any, error: Any, elapsed_ms: float
        ) -> None:
            import time as _time

            from enpire.env.forge.cap.agent.agent_context import ToolCallRecord

            memory.record(
                ToolCallRecord(
                    call_id=call_id,
                    tool_name=name,
                    result_repr=_fmt(result),
                    error=str(error) if error else None,
                    elapsed_ms=elapsed_ms,
                    timestamp=_time.strftime("%H:%M:%S"),
                    iteration=iteration,
                )
            )

        set_tool_event_hooks(on_end=_on_tool_end)

        # Pass raw callables — Executor wraps them with timing internally
        tool_callables = {
            k: v
            for k, v in ctx.namespace.items()
            if callable(v) and not k.startswith("_")
        }

        executor = Executor(tool_callables=tool_callables)
        logger.info("ExecutorStep: executing iter %d", ctx.iteration)

        result = executor.execute(ctx.code, extra_namespace=ctx.paused_namespace)
        ctx.execution_result = result

        # Capture after-frames for visual differencing (auto-detect cameras from before)
        if ctx.frames_before:
            get_camera_image = ctx.namespace.get("get_camera_image")
            if get_camera_image is not None:
                frames_after: dict[str, Any] = {}
                for cam in ctx.frames_before:
                    try:
                        img = get_camera_image(cam)
                        if img is not None and img.shape[0] > 1 and img.shape[1] > 1:
                            frames_after[cam] = img
                    except Exception as e:
                        logger.warning(
                            "ExecutorStep: after-frame capture(%r) failed: %s", cam, e
                        )
                if frames_after:
                    ctx.frames_after = frames_after

        if result.paused:
            ctx.paused_namespace = result.user_namespace
        else:
            ctx.paused_namespace = None

        if ctx.session is not None:
            ctx.session.save_execution_log(ctx.iteration, result)

        return ctx


# ---------------------------------------------------------------------------
# SubprocessExecutorStep
# ---------------------------------------------------------------------------


class SubprocessExecutorStep(AgentStep):
    """Execute code via ``run_script.py`` subprocess, N seeds for statistical eval.

    For each seed, spawns::

        uv run python run_script.py file=<code.py> env=<env> seed=<seed> ...

    Collects ``result.json`` from each execution, aggregates into
    ``eval_summary.json``, and stores the summary in ``ctx.evaluation``.

    Reads:  ctx.code, ctx.config.execution (n_seeds, seed_start, ...)
    Writes: ctx.execution_result (aggregated), ctx.evaluation
    """

    name = "executor"

    def run(self, ctx: AgentContext) -> AgentContext:
        print(f"[executor] iter={ctx.iteration} code_len={len(ctx.code or '')} should_stop={ctx.should_stop}")
        if not ctx.code:
            logger.warning("SubprocessExecutorStep: no code to execute")
            print(f"[executor] SKIP — no code (ctx.code={ctx.code!r})")
            return ctx

        import json
        import subprocess
        import time as _time

        from enpire.env.forge.cap.agent.agent_context import AgentEvaluation

        session = ctx.session
        config = getattr(ctx, "config", None)
        exec_cfg = getattr(config, "execution", None) if config else None

        n_seeds = exec_cfg.n_seeds if exec_cfg else 1
        seed_start = exec_cfg.seed_start if exec_cfg else 0
        timeout_s = exec_cfg.timeout_s if exec_cfg else 300
        do_record = exec_cfg.record if exec_cfg else False
        # config.env is an EnvConfig object (Hydra); read .name for the env string
        env_obj = getattr(config, "env", None) if config else None
        if env_obj is not None and hasattr(env_obj, "name"):
            env_name = env_obj.name or "robocasa:PickPlaceSinkToCounter"
        else:
            env_name = str(env_obj) if env_obj else "robocasa:PickPlaceSinkToCounter"
        robot_adapter = None
        if config is not None:
            from enpire.env.forge.cap.agent.robot_adapters import get_robot_adapter

            robot_adapter = get_robot_adapter(config)

        raw_candidates = list(getattr(ctx, "code_candidates", None) or [])
        if not raw_candidates:
            raw_candidates = [
                {
                    "index": 0,
                    "code": ctx.code or "",
                    "thoughts": ctx.thoughts or "",
                }
            ]
        candidate_count = len(raw_candidates)
        is_study_env = str(env_name).startswith("study")
        cg_cfg = getattr(config, "code_generator", None) if config is not None else None
        allow_candidate_search_on_robot = bool(
            getattr(cg_cfg, "allow_candidate_search_on_robot", False)
        )
        if candidate_count > 1 and not is_study_env and not allow_candidate_search_on_robot:
            ctx.evaluation = AgentEvaluation(
                success=False,
                score=0.0,
                feedback=(
                    "code_generator.num_candidates > 1 is only enabled by default "
                    "for study/offline envs. Set "
                    "code_generator.allow_candidate_search_on_robot=true to run "
                    "multiple generated programs on a non-study environment."
                ),
                method="candidate_search_guard",
                details={"num_candidates": candidate_count, "env_name": env_name},
            )
            return ctx

        # Write code candidate(s) to the iteration directory.
        candidates: list[dict[str, Any]] = []
        if session is not None:
            for raw in raw_candidates:
                idx = int(raw.get("index", len(candidates)))
                code = str(raw.get("code", "") or "")
                thoughts = str(raw.get("thoughts", "") or "")
                if candidate_count == 1:
                    code_path = session.save_code(ctx.iteration, code)
                else:
                    cdir = (
                        session.iterations_dir(ctx.iteration)
                        / "candidates"
                        / f"cand_{idx:03d}"
                    )
                    cdir.mkdir(parents=True, exist_ok=True)
                    code_path = cdir / "code.py"
                    code_path.write_text(code, encoding="utf-8")
                    (cdir / "thoughts.md").write_text(thoughts, encoding="utf-8")
                candidates.append(
                    {
                        "index": idx,
                        "code": code,
                        "thoughts": thoughts,
                        "code_path": code_path,
                    }
                )
        else:
            # Fallback: temp files
            import tempfile
            from pathlib import Path

            for raw in raw_candidates:
                idx = int(raw.get("index", len(candidates)))
                code = str(raw.get("code", "") or "")
                tmp = tempfile.NamedTemporaryFile(
                    suffix=f"_cand_{idx:03d}.py",
                    prefix="code_",
                    delete=False,
                    mode="w",
                )
                tmp.write(code)
                tmp.close()
                candidates.append(
                    {
                        "index": idx,
                        "code": code,
                        "thoughts": str(raw.get("thoughts", "") or ""),
                        "code_path": Path(tmp.name),
                    }
                )

        logger.info(
            "SubprocessExecutorStep: iter %d, %d candidate(s), %d seed(s), env=%s",
            ctx.iteration,
            candidate_count,
            n_seeds,
            env_name,
        )
        if candidate_count > 1:
            _dash_print(
                f"  [executor] Running {candidate_count} candidate(s) × "
                f"{n_seeds} seed(s) via run_script.py "
                f"(seeds {seed_start}..{seed_start + n_seeds - 1})"
            )
        else:
            _dash_print(
                f"  [executor] Running {n_seeds} seed(s) via run_script.py "
                f"(seeds {seed_start}..{seed_start + n_seeds - 1})"
            )

        exec_mode = str(exec_cfg.mode if exec_cfg else "sequential").lower()
        offset_service_ports = bool(
            getattr(robot_adapter, "offset_service_ports", True)
            if robot_adapter is not None
            else True
        )

        # Load fixed evaluation seeds from cap/assets/robocasa_40_seeds.txt.
        # Each line: "<seed> <layout_id> <style_id>"
        # Job i uses row i (mod file length) so the same seeds are used
        # regardless of n_seeds.
        _eval_seeds: list[tuple[int, int, int]] = []
        _seeds_file = _ROOT_PATH / "cap" / "assets" / "robocasa_40_seeds.txt"
        if _seeds_file.exists():
            for line in _seeds_file.read_text().splitlines():
                parts = line.split()
                if len(parts) == 3:
                    _eval_seeds.append((int(parts[0]), int(parts[1]), int(parts[2])))

        _n_gpus: int = 0
        if n_seeds > 1 and os.environ.get("MUJOCO_GL", "egl") == "egl":
            _cfg_gpus = exec_cfg.n_gpus if exec_cfg else 0
            if _cfg_gpus > 0:
                _n_gpus = _cfg_gpus
            else:
                try:
                    import subprocess as _subp

                    _n_gpus = (
                        _subp.check_output(
                            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            timeout=3,
                        )
                        .decode()
                        .strip()
                        .count("\n")
                        + 1
                    )
                except Exception:
                    _n_gpus = 0

        # Build command and per-seed env overrides for each seed
        def _build_cmd(
            seed: int,
            i: int,
            gpu_slot: int | None = None,
            *,
            candidate: dict[str, Any] | None = None,
            exec_id: int | None = None,
        ) -> tuple[list[str], Any, dict[str, str]]:
            # gpu_slot is the round-robin GPU index (i % n_render).
            # Use it for port assignment so seeds sharing a GPU reuse the same
            # service ports (safe because the semaphore serialises them).
            slot = gpu_slot if gpu_slot is not None else i
            port_slot = slot if offset_service_ports else 0
            job_exec_id = i if exec_id is None else int(exec_id)
            exec_dir = session.exec_dir(ctx.iteration, job_exec_id) if session else None
            job_code_path = (
                candidate["code_path"] if candidate is not None else candidates[0]["code_path"]
            )
            runtime_cfg = getattr(config, "runtime", None)
            env_cfg = getattr(config, "env", None)
            # Invoke the parent's interpreter directly instead of nesting
            # `uv run`. `uv run` consumes UV_PROJECT_ENVIRONMENT during
            # activation, so a child `uv run --no-sync` can't see the outer
            # venv and would fall back to `.venv/` (missing robocasa etc.).
            # sys.executable points at the already-activated venv's Python,
            # so the child reuses the exact same environment.
            # Use fixed evaluation seed row for this job index.
            if _eval_seeds:
                row = _eval_seeds[i % len(_eval_seeds)]
                job_seed, job_layout, job_style = row
            else:
                job_seed, job_layout, job_style = seed, None, None

            cmd = [
                sys.executable,
                "-u",
                str(_ROOT_PATH / "run_script.py"),
                f"script_file={job_code_path}",
                f"env.name={env_name}",
                f"env.seed={job_seed}",
            ]
            # Per-seed env var overrides (for services read via os.environ)
            seed_env: dict[str, str] = {}
            for passthrough_env_name in (
                "CAP_NVIDIA_PROVIDER_URL",
                "CAP_NVIDIA_TELEMETRY_FILE",
                "CAP_NVIDIA_SCHEDULER_DB",
            ):
                env_value = os.environ.get(passthrough_env_name)
                if env_value:
                    seed_env[passthrough_env_name] = env_value

            if _n_gpus > 0:
                seed_env["MUJOCO_EGL_DEVICE_ID"] = str(port_slot % _n_gpus)

            if runtime_cfg is not None:
                cmd.append(f"runtime.curobo_host={runtime_cfg.curobo_host}")
                base_port = runtime_cfg.curobo_port
                curobo_port = (
                    base_port + port_slot
                    if (offset_service_ports and base_port > 0 and n_seeds > 1)
                    else base_port
                )
                cmd.append(f"runtime.curobo_port={curobo_port}")
                sam3_host = str(getattr(runtime_cfg, "sam3_host", "127.0.0.1"))
                sam3_base = int(getattr(runtime_cfg, "sam3_port", 0) or 0)
                sam3_port = (
                    sam3_base + port_slot
                    if (offset_service_ports and sam3_base > 0 and n_seeds > 1)
                    else sam3_base
                )
                if sam3_port > 0:
                    cmd.append(f"runtime.sam3_host={sam3_host}")
                    cmd.append(f"runtime.sam3_port={sam3_port}")
                    seed_env["SAM3_SERVER_HOST"] = sam3_host
                    seed_env["SAM3_SERVER_PORT"] = str(sam3_port)
                anygrasp_base = getattr(runtime_cfg, "anygrasp_port", 0)
                anygrasp_host = str(getattr(runtime_cfg, "anygrasp_host", "127.0.0.1"))
                if anygrasp_base and anygrasp_base > 0:
                    anygrasp_port = (
                        anygrasp_base + port_slot
                        if (offset_service_ports and n_seeds > 1)
                        else anygrasp_base
                    )
                    cmd.append(f"runtime.anygrasp_host={anygrasp_host}")
                    cmd.append(f"runtime.anygrasp_port={anygrasp_port}")
                    seed_env["ANYGRASP_SERVER_PORT"] = str(anygrasp_port)
                    seed_env["ANYGRASP_SERVER_HOST"] = anygrasp_host
                    seed_env["ANYGRASP_SERVICE_URL"] = (
                        f"http://{seed_env['ANYGRASP_SERVER_HOST']}:"
                        f"{seed_env['ANYGRASP_SERVER_PORT']}"
                    )
                bundlesdf_base = int(getattr(runtime_cfg, "bundlesdf_port", 0) or 0)
                if bundlesdf_base > 0:
                    bundlesdf_host = str(
                        getattr(runtime_cfg, "bundlesdf_host", "127.0.0.1")
                    )
                    bundlesdf_port = (
                        bundlesdf_base + port_slot
                        if (offset_service_ports and n_seeds > 1)
                        else bundlesdf_base
                    )
                    cmd.append(f"runtime.bundlesdf_host={bundlesdf_host}")
                    cmd.append(f"runtime.bundlesdf_port={bundlesdf_port}")
                    seed_env["BUNDLESDF_SERVER_HOST"] = bundlesdf_host
                    seed_env["BUNDLESDF_SERVER_PORT"] = str(bundlesdf_port)
            if env_cfg is not None:
                layout = job_layout if job_layout is not None else env_cfg.layout_id
                style = job_style if job_style is not None else env_cfg.style_id
                cmd.append(f"env.layout_id={layout}")
                cmd.append(f"env.style_id={style}")
                # Camera resolution matters for VLA policies (GR00T 256×256).
                # Propagate so the subprocess matches the outer experiment.
                cam_h = getattr(env_cfg, "camera_height", None)
                cam_w = getattr(env_cfg, "camera_width", None)
                if cam_h is not None:
                    cmd.append(f"env.camera_height={cam_h}")
                if cam_w is not None:
                    cmd.append(f"env.camera_width={cam_w}")
            if exec_dir is not None:
                cmd.append(f"script_output_dir={exec_dir}")
            if do_record:
                cmd.append("recording.enabled=true")
            # Robot adapter owns robot-specific child Hydra overrides.
            if robot_adapter is not None:
                cmd.extend(robot_adapter.run_script_overrides(config))
                seed_env.update(
                    robot_adapter.child_env(
                        config,
                        seed=job_seed,
                        slot=port_slot,
                        n_seeds=n_seeds if offset_service_ports else 1,
                    )
                )
            # Forward task — needed by skill_reflection VLM prompt.
            # Use Hydra quoted-string syntax so spaces in the value are preserved.
            if config is not None and config.task:
                cmd.append(f'task="{config.task}"')
            # Forward VLM reward settings.  Real-hardware tasks expose
            # get_task_info() as a VLM reward inside run_script.py, so the
            # child process must see the same reward.* provider/model/camera
            # settings as the parent agent config.
            reward_cfg = getattr(config, "reward", None) if config is not None else None
            if reward_cfg is not None:
                if getattr(reward_cfg, "evaluator", None) is not None:
                    cmd.append(f"reward.evaluator={reward_cfg.evaluator}")
                if getattr(reward_cfg, "task", None):
                    cmd.append(f"reward.task={reward_cfg.task}")
                if getattr(reward_cfg, "inject_into_per_seed_vlm", None) is not None:
                    inject = str(bool(reward_cfg.inject_into_per_seed_vlm)).lower()
                    cmd.append(f"reward.inject_into_per_seed_vlm={inject}")
                if getattr(reward_cfg, "vlm_backend", None):
                    cmd.append(f"reward.vlm_backend={reward_cfg.vlm_backend}")
                if getattr(reward_cfg, "vlm_model", None):
                    cmd.append(f"reward.vlm_model={reward_cfg.vlm_model}")
                if getattr(reward_cfg, "vlm_camera", None):
                    cmd.append(f"reward.vlm_camera={reward_cfg.vlm_camera}")
                if getattr(reward_cfg, "vlm_reasoning_effort", None):
                    cmd.append(
                        f"reward.vlm_reasoning_effort={reward_cfg.vlm_reasoning_effort}"
                    )
            # Forward oracle_api flag so run_script.py registers get_oracle_targets.
            if config is not None and getattr(config, "oracle_api", False):
                cmd.append("oracle_api=true")
            # Forward skill_reflection config so the subprocess enables the hook.
            sr_cfg = getattr(config, "skill_reflection", None)
            if sr_cfg is not None and getattr(sr_cfg, "enabled", False):
                cmd.append("skill_reflection.enabled=true")
                cmd.append(f"skill_reflection.vlm_backend={getattr(sr_cfg, 'vlm_backend', 'nvidia')}")
                if getattr(sr_cfg, "vlm_model", None):
                    cmd.append(f"skill_reflection.vlm_model={sr_cfg.vlm_model}")
                cmd.append(f"skill_reflection.max_workers={getattr(sr_cfg, 'max_workers', 4)}")
                cameras = list(getattr(sr_cfg, "cameras", ["top", "wrist"]))
                cmd.append(f"skill_reflection.cameras=[{','.join(cameras)}]")
            # Pass skill library path so run_script.py can add it to sys.path.
            # Priority: session skill_library (seeded+agent-authored) >
            #           bundled skill_library/ sibling of oracle script
            if session is not None:
                skill_lib_dir = session.run_dir / "skill_library"
                if not any(skill_lib_dir.glob("*.py")):
                    oracle_cfg = getattr(getattr(ctx, "config", None), "oracle", None)
                    if oracle_cfg:
                        oracle_path = _ROOT_PATH / oracle_cfg
                        rt = getattr(getattr(ctx, "config", None), "runtime", None)
                        ssd = getattr(rt, "saved_scripts_dir", None) if rt else None
                        if ssd:
                            candidate = _ROOT_PATH / ssd / oracle_cfg
                            if candidate.exists():
                                oracle_path = candidate
                        sibling = (oracle_path.parent / "skill_library").resolve()
                        if sibling.exists() and any(sibling.glob("*.py")):
                            skill_lib_dir = sibling
                cmd.append(f"skill_library_path={skill_lib_dir}")
            return cmd, exec_dir, seed_env

        # Run a single seed and return result dict.
        #
        # We use Popen with ``start_new_session=True`` (each seed becomes its
        # own process group leader) + ``os.killpg`` on timeout/interrupt so
        # grandchildren (cuRobo / SAM3 / AnyGrasp clients, mujoco renderer
        # threads that fork, …) are terminated too — ``subprocess.run``'s
        # built-in timeout only SIGKILLs the direct child, which leaves
        # orphaned grandchildren adopted by init.
        import signal as _signal

        def _kill_process_group(proc: subprocess.Popen) -> None:
            try:
                pgid = os.getpgid(proc.pid)
            except (ProcessLookupError, PermissionError):
                return
            try:
                os.killpg(pgid, _signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                return
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass

        def _run_seed(
            seed: int,
            i: int,
            gpu_slot: int | None = None,
            *,
            candidate: dict[str, Any] | None = None,
            exec_id: int | None = None,
        ) -> dict[str, Any]:
            cmd, exec_dir, seed_env = _build_cmd(
                seed,
                i,
                gpu_slot,
                candidate=candidate,
                exec_id=exec_id,
            )
            proc_env = {**os.environ, **seed_env} if seed_env else None
            t0 = _time.time()
            cand_idx = int((candidate or candidates[0]).get("index", 0))
            prefix = f"cand={cand_idx} " if candidate_count > 1 else ""
            _dash_print(f"  [executor] {prefix}seed={seed} starting...")

            proc: subprocess.Popen | None = None
            stdout = ""
            stderr = ""
            timed_out = False
            exc: Exception | None = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=proc_env,
                    cwd=str(_ROOT_PATH),
                    start_new_session=True,
                )
                try:
                    stdout, stderr = proc.communicate(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_process_group(proc)
                    try:
                        stdout, stderr = proc.communicate(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        stderr = ""  # stdout won't be used after timeout
            except KeyboardInterrupt:
                # Propagate after killing the group so Ctrl-C in the main
                # thread takes down the entire subprocess tree for this seed.
                if proc is not None:
                    _kill_process_group(proc)
                raise
            except Exception as e:
                exc = e
                if proc is not None:
                    _kill_process_group(proc)

            elapsed = _time.time() - t0

            if timed_out:
                _dash_print(f"  [executor] {prefix}seed={seed} TIMEOUT ({timeout_s}s)")
                return {
                    "seed": seed,
                    "exec_id": i if exec_id is None else int(exec_id),
                    "seed_index": i,
                    "candidate_index": cand_idx,
                    "success": False,
                    "score": 0.0,
                    "feedback": f"Timed out after {timeout_s}s (process group killed)",
                    "elapsed_s": round(elapsed, 1),
                }
            if exc is not None:
                _dash_print(f"  [executor] {prefix}seed={seed} ERROR: {exc}")
                return {
                    "seed": seed,
                    "exec_id": i if exec_id is None else int(exec_id),
                    "seed_index": i,
                    "candidate_index": cand_idx,
                    "success": False,
                    "score": 0.0,
                    "feedback": f"Subprocess error: {exc}",
                    "elapsed_s": round(elapsed, 1),
                }

            returncode = proc.returncode if proc is not None else -1
            success_proc = returncode == 0
            seed_result: dict[str, Any] = {
                "seed": seed,
                "exec_id": i if exec_id is None else int(exec_id),
                "seed_index": i,
                "candidate_index": cand_idx,
                "returncode": returncode,
                "elapsed_s": round(elapsed, 1),
            }
            if exec_dir is not None:
                result_path = exec_dir / "result.json"
                if result_path.exists():
                    data = json.loads(result_path.read_text())
                    seed_result["success"] = data.get("success", False)
                    seed_result["score"] = data.get("score", 0.0)
                    base_feedback = data.get("feedback", "")
                    # Append a tail of the run's stdout (run_profiling_*.txt)
                    # so the skill author / reflection can see concrete
                    # failure symptoms — IK_Failed, specific skill errors,
                    # etc. — instead of just `reward=0, success=False`.
                    stdout_tail = _read_run_log_tail(exec_dir, max_chars=2000)
                    if stdout_tail:
                        seed_result["feedback"] = (
                            f"{base_feedback}\n[stdout tail]\n{stdout_tail}"
                        )
                    else:
                        seed_result["feedback"] = base_feedback
                else:
                    seed_result["success"] = False
                    seed_result["score"] = 0.0
                    stderr_tail = (stderr or "")[-500:]
                    seed_result["feedback"] = (
                        f"SCRIPT CRASHED (returncode={returncode}). "
                        f"Code never executed.\n{stderr_tail}"
                    )
            else:
                seed_result["success"] = success_proc
                seed_result["score"] = 0.0
            if not success_proc and stderr:
                seed_result["stderr"] = stderr[-500:]
            status = "OK" if seed_result.get("success") else "FAIL"
            _dash_print(
                f"  [executor] {prefix}seed={seed} {status} "
                f"score={seed_result.get('score', 0):.3f} "
                f"({elapsed:.1f}s)"
            )
            return seed_result

        def _run_seed_in_tmux_window(
            seed: int,
            i: int,
            gpu_slot: int | None = None,
            *,
            candidate: dict[str, Any] | None = None,
            exec_id: int | None = None,
        ) -> dict[str, Any]:
            """Launch run_script.py in its own tmux window so rich live UI renders.

            Waits by polling ``{exec_dir}/result.json`` + a ``.done`` sentinel.
            """
            cmd, exec_dir, seed_env = _build_cmd(
                seed,
                i,
                gpu_slot,
                candidate=candidate,
                exec_id=exec_id,
            )
            assert exec_dir is not None
            cand_idx = int((candidate or candidates[0]).get("index", 0))
            prefix = f"cand={cand_idx} " if candidate_count > 1 else ""
            done_marker = exec_dir / ".tmux_done"
            if done_marker.exists():
                done_marker.unlink()

            # Shell-escape each arg.  Pipe the child's exit status into the
            # sentinel so we can detect completion + returncode from outside.
            import shlex

            def _propagate_tmux_session_env() -> None:
                """Make parent-only provider env visible to tmux child windows.

                tmux starts new panes from the tmux server/session environment,
                not necessarily from the run_agent process environment.  That
                means secrets configured in the shell that launched run_agent
                (e.g. NVIDIA_API_KEY) can be visible to the parent LLM but absent
                in child run_script.py.  Do not put these values in the shell
                command line; set them in the tmux session environment instead.
                """

                names = [
                    "NVIDIA_API_KEY",
                    *[f"NVIDIA_API_KEY_{idx}" for idx in range(1, 101)],
                    "CAP_NVIDIA_TELEMETRY_FILE",
                    "CAP_NVIDIA_SCHEDULER_DB",
                    "CAP_NVIDIA_REQUEST_DELAY_S",
                    "CAP_NVIDIA_MAX_CONCURRENT_PER_KEY",
                    "CAP_NVIDIA_ACQUIRE_TIMEOUT_S",
                    "CAP_NVIDIA_429_COOLDOWN_S",
                    "OPENAI_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "GOOGLE_API_KEY",
                    "GEMINI_API_KEY",
                ]
                try:
                    tmux_session = (
                        subprocess.check_output(
                            ["tmux", "display-message", "-p", "#S"],
                            cwd=str(_ROOT_PATH),
                            text=True,
                        )
                        .strip()
                    )
                except Exception:
                    tmux_session = ""
                if not tmux_session:
                    return
                for name in names:
                    value = os.environ.get(name)
                    if not value:
                        continue
                    try:
                        subprocess.run(
                            ["tmux", "set-environment", "-t", tmux_session, name, value],
                            check=False,
                            cwd=str(_ROOT_PATH),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        pass

            _propagate_tmux_session_env()
            env_exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in seed_env.items())
            quoted = " ".join(shlex.quote(c) for c in cmd)
            stderr_log = shlex.quote(str(exec_dir / "stderr.txt"))
            shell_line = (
                f"cd {shlex.quote(str(_ROOT_PATH))}; "
                f"{env_exports + ' ' if env_exports else ''}{quoted} 2>{stderr_log}; "
                f"rc=$?; echo $rc > {shlex.quote(str(done_marker))}; "
                f"echo; echo '[seed={seed}] exit='$rc"
            )
            window_name = (
                f"exec_c{cand_idx}_s{seed}_iter{ctx.iteration:03d}"
                if candidate_count > 1
                else f"exec_s{seed}_iter{ctx.iteration:03d}"
            )
            tmux_cmd = [
                "tmux",
                "new-window",
                "-d",
                "-n",
                window_name,
                "sh",
                "-c",
                shell_line,
            ]
            t0 = _time.time()
            _dash_print(
                f"  [executor] {prefix}seed={seed} launching tmux window '{window_name}'"
            )
            try:
                subprocess.run(tmux_cmd, check=True, cwd=str(_ROOT_PATH))
            except Exception as e:
                elapsed = _time.time() - t0
                return {
                    "seed": seed,
                    "exec_id": i if exec_id is None else int(exec_id),
                    "seed_index": i,
                    "candidate_index": cand_idx,
                    "success": False,
                    "score": 0.0,
                    "feedback": f"tmux launch failed: {e}",
                    "elapsed_s": round(elapsed, 1),
                }

            # Poll for completion
            deadline = t0 + timeout_s
            while _time.time() < deadline:
                if done_marker.exists():
                    break
                _time.sleep(0.5)

            elapsed = _time.time() - t0
            seed_result: dict[str, Any] = {
                "seed": seed,
                "exec_id": i if exec_id is None else int(exec_id),
                "seed_index": i,
                "candidate_index": cand_idx,
                "elapsed_s": round(elapsed, 1),
            }
            try:
                returncode = (
                    int(done_marker.read_text().strip()) if done_marker.exists() else -1
                )
            except Exception:
                returncode = -1
            seed_result["returncode"] = returncode

            result_path = exec_dir / "result.json"
            if result_path.exists():
                data = json.loads(result_path.read_text())
                seed_result["success"] = data.get("success", False)
                seed_result["score"] = data.get("score", 0.0)
                base_feedback = data.get("feedback", "")
                stdout_tail = _read_run_log_tail(exec_dir, max_chars=2000)
                if stdout_tail:
                    seed_result["feedback"] = (
                        f"{base_feedback}\n[stdout tail]\n{stdout_tail}"
                    )
                else:
                    seed_result["feedback"] = base_feedback
            else:
                seed_result["success"] = False
                seed_result["score"] = 0.0
                if not done_marker.exists():
                    seed_result["feedback"] = (
                        f"Timed out after {timeout_s}s (tmux window)"
                    )
                else:
                    stderr_path = exec_dir / "stderr.txt"
                    stderr_tail = ""
                    if stderr_path.exists():
                        txt = stderr_path.read_text(errors="replace")
                        stderr_tail = txt[-1000:] if len(txt) > 1000 else txt
                    seed_result["feedback"] = (
                        f"SCRIPT CRASHED (returncode={returncode}) in tmux window '{window_name}'."
                        + (f"\n[stderr]\n{stderr_tail}" if stderr_tail else "")
                    )
                    if stderr_tail:
                        seed_result["stderr"] = stderr_tail
            status = "OK" if seed_result.get("success") else "FAIL"
            _dash_print(
                f"  [executor] {prefix}seed={seed} {status} "
                f"score={seed_result.get('score', 0):.3f} ({elapsed:.1f}s)"
            )
            return seed_result

        # Dispatch based on execution mode.  With candidate search enabled, a
        # "job" is (candidate, seed); every candidate sees the same seed set so
        # rankings compare averaged performance fairly.
        seeds = [(seed_start + i, i) for i in range(n_seeds)]
        jobs: list[dict[str, Any]] = []
        for candidate_pos, candidate in enumerate(candidates):
            for seed, seed_i in seeds:
                exec_id = (
                    seed_i
                    if candidate_count == 1
                    else candidate_pos * n_seeds + seed_i
                )
                jobs.append(
                    {
                        "seed": seed,
                        "seed_index": seed_i,
                        "candidate": candidate,
                        "candidate_pos": candidate_pos,
                        "exec_id": exec_id,
                    }
                )
        total_jobs = len(jobs)

        tmux_available = bool(os.environ.get("TMUX") and session is not None)
        # Study/offline candidate sweeps can launch many tiny subprocesses; keep
        # them captured instead of opening K×N tmux windows.
        use_tmux = exec_mode == "parallel" and tmux_available and not is_study_env

        if exec_mode == "tmux":
            if tmux_available:
                _dash_print(
                    f"  [executor] mode=tmux, {total_jobs} job(s), "
                    "launching run_script.py in tmux window(s)"
                )
                per_seed = [
                    _run_seed_in_tmux_window(
                        job["seed"],
                        job["seed_index"],
                        candidate=job["candidate"],
                        exec_id=job["exec_id"],
                    )
                    for job in jobs
                ]
            else:
                _dash_print(
                    "  [executor] mode=tmux requested but no active tmux/session; "
                    "falling back to captured subprocess"
                )
                per_seed = [
                    _run_seed(
                        job["seed"],
                        job["seed_index"],
                        candidate=job["candidate"],
                        exec_id=job["exec_id"],
                    )
                    for job in jobs
                ]
        elif exec_mode == "parallel" and total_jobs > 1:
            import threading
            from concurrent.futures import ThreadPoolExecutor

            from enpire.env.forge.cap.agent.agent_dashboard import current as _dash_current

            # Per-GPU semaphore — limits concurrent seeds per render GPU.
            # seeds_per_gpu=1 (default) = one seed at a time per GPU.
            # Increase to run more seeds concurrently on the same GPU.
            _cfg_gpus = exec_cfg.n_gpus if exec_cfg else 0
            _n_render = _cfg_gpus if _cfg_gpus > 0 else max(1, _n_gpus)
            _seeds_per_gpu = int(getattr(exec_cfg, "seeds_per_gpu", 1) or 1)
            _gpu_sem = [threading.Semaphore(_seeds_per_gpu) for _ in range(_n_render)]

            def _run_with_gpu_queue(run_fn, job: dict[str, Any]) -> dict:
                gpu_slot = int(job["seed_index"]) % _n_render
                actual_gpu = gpu_slot
                _dash = _dash_current()
                if _dash is not None:
                    _dash.on_seed_queued(int(job["exec_id"]), actual_gpu)
                with _gpu_sem[gpu_slot]:
                    if _dash is not None:
                        _dash.on_seed_running(int(job["exec_id"]), actual_gpu)
                    return run_fn(
                        job["seed"],
                        job["seed_index"],
                        gpu_slot,
                        candidate=job["candidate"],
                        exec_id=job["exec_id"],
                    )

            if use_tmux:
                _dash_print(
                    f"  [executor] mode=parallel via tmux windows, {total_jobs} jobs, "
                    f"{_n_render} render GPU(s), max {_seeds_per_gpu} task(s)/GPU"
                )
                with ThreadPoolExecutor(max_workers=total_jobs) as pool:
                    per_seed = list(pool.map(
                        lambda job: _run_with_gpu_queue(_run_seed_in_tmux_window, job),
                        jobs,
                    ))
            else:
                _dash_print(
                    f"  [executor] mode=parallel (captured), {total_jobs} jobs, "
                    f"{_n_render} render GPU(s), max {_seeds_per_gpu} task(s)/GPU"
                )
                pool = ThreadPoolExecutor(max_workers=total_jobs)
                futures = [
                    pool.submit(_run_with_gpu_queue, _run_seed, job) for job in jobs
                ]
                try:
                    per_seed = [f.result() for f in futures]
                except KeyboardInterrupt:
                    _dash_print("  [executor] Ctrl-C received — cancelling pending seeds")
                    for f in futures:
                        f.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                finally:
                    pool.shutdown(wait=True)
        else:
            per_seed: list[dict[str, Any]] = []
            for job in jobs:
                per_seed.append(
                    _run_seed(
                        job["seed"],
                        job["seed_index"],
                        candidate=job["candidate"],
                        exec_id=job["exec_id"],
                    )
                )

        all_candidate_results = list(per_seed)
        candidate_search_details: dict[str, Any] | None = None
        if candidate_count > 1:
            candidate_summaries: list[dict[str, Any]] = []
            candidate_by_index = {int(c["index"]): c for c in candidates}
            for candidate_pos, candidate in enumerate(candidates):
                cand_idx = int(candidate["index"])
                rows = [
                    r
                    for r in all_candidate_results
                    if int(r.get("candidate_index", -1)) == cand_idx
                ]
                rows.sort(key=lambda r: int(r.get("seed_index", r.get("exec_id", 0))))
                n_rows = len(rows)
                successes_c = sum(1 for r in rows if r.get("success"))
                scores_c = [float(r.get("score", 0.0) or 0.0) for r in rows]
                avg_score_c = sum(scores_c) / n_rows if n_rows else 0.0
                summary = {
                    "candidate_index": cand_idx,
                    "candidate_pos": candidate_pos,
                    "n_seeds": n_rows,
                    "successes": successes_c,
                    "success_rate": successes_c / n_rows if n_rows else 0.0,
                    "avg_score": avg_score_c,
                    "max_score": max(scores_c) if scores_c else 0.0,
                    "exec_ids": [int(r.get("exec_id", 0)) for r in rows],
                    "per_seed": rows,
                    "code_path": str(candidate.get("code_path", "")),
                }
                candidate_summaries.append(summary)

            candidate_summaries.sort(
                key=lambda row: (
                    float(row["avg_score"]),
                    float(row["success_rate"]),
                    int(row["successes"]),
                    -int(row["candidate_pos"]),
                ),
                reverse=True,
            )
            best_summary = candidate_summaries[0] if candidate_summaries else {}
            best_candidate_index = int(best_summary.get("candidate_index", 0))
            best_candidate = candidate_by_index.get(best_candidate_index, candidates[0])
            per_seed = list(best_summary.get("per_seed", []))
            ctx.code = str(best_candidate.get("code", "") or "")
            ctx.thoughts = str(best_candidate.get("thoughts", "") or "")
            if session is not None:
                session.save_code(ctx.iteration, ctx.code)
                if ctx.thoughts:
                    session.thoughts_path(ctx.iteration).write_text(
                        ctx.thoughts,
                        encoding="utf-8",
                    )
                candidate_summary_path = (
                    session.iterations_dir(ctx.iteration) / "candidate_summary.json"
                )
                import json as _json

                candidate_summary_path.write_text(
                    _json.dumps(
                        {
                            "num_candidates": candidate_count,
                            "n_seeds_per_candidate": n_seeds,
                            "best_candidate_index": best_candidate_index,
                            "candidates": candidate_summaries,
                        },
                        indent=2,
                        default=str,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            candidate_search_details = {
                "num_candidates": candidate_count,
                "n_seeds_per_candidate": n_seeds,
                "best_candidate_index": best_candidate_index,
                "candidates": candidate_summaries,
                "all_per_seed": all_candidate_results,
            }
            _dash_print(
                f"  [executor] candidate search best=cand {best_candidate_index} "
                f"avg_score={float(best_summary.get('avg_score', 0.0)):.3f} "
                f"success_rate={float(best_summary.get('success_rate', 0.0)):.2f}"
            )

        # ---- Post-exec profiling ---------------------------------------
        # Each phase below is timed so the dashboard can report where the
        # "executor" step spends time AFTER all seeds finish. The VLM
        # reflection phase in particular can dominate on many-seed runs.
        post_phase_ms: dict[str, float] = {}
        _phase_t0 = _time.perf_counter()

        # ---- Phase 1: Aggregate ----------------------------------------
        n = len(per_seed)
        successes = sum(1 for r in per_seed if r.get("success"))
        scores = [r.get("score", 0.0) for r in per_seed]
        success_rate = successes / n if n else 0.0
        avg_score = sum(scores) / n if n else 0.0

        _dash_print(
            f"  [executor] Summary: {successes}/{n} succeeded, "
            f"avg_score={avg_score:.3f}, success_rate={success_rate:.2f}"
        )
        post_phase_ms["aggregate"] = (_time.perf_counter() - _phase_t0) * 1000.0

        # ---- Phase 2: (deferred) eval_summary.json is written at the end
        # so that oracle reward fields stashed by Phase 4.5 are included.
        post_phase_ms["save_summary"] = 0.0

        # ---- Phase 3: Aggregate per-seed skill logs --------------------
        # Merge per-seed skill profiles into iterations/iter_NNN/skill_log.json
        # AND feed them back into the SkillLibrary index so the next
        # iteration's prompt sees accurate success_rate/calls per skill.
        # Without this, the index always shows calls=0 success_rate=0.00 and
        # the LLM has no signal to prefer working skills over broken ones.
        _phase_t0 = _time.perf_counter()
        if session is not None:
            _aggregate_skill_logs(
                session,
                ctx.iteration,
                n_seeds,
                seed_start,
                exec_ids=[int(r.get("exec_id", i)) for i, r in enumerate(per_seed)],
            )
            sl = getattr(session, "_skill_library", None)
            if sl is not None:
                import json as _json

                iter_log_path = (
                    session.iterations_dir(ctx.iteration) / "skill_log.json"
                )
                if iter_log_path.exists():
                    try:
                        iter_logs = _json.loads(
                            iter_log_path.read_text(encoding="utf-8")
                        )
                        if isinstance(iter_logs, list) and iter_logs:
                            sl.update_stats_from_logs(iter_logs)
                            sl.append_skill_logs(iter_logs)
                            logger.info(
                                "SubprocessExecutorStep: updated skill stats from "
                                "%d profile(s) in iter_%03d/skill_log.json",
                                len(iter_logs),
                                ctx.iteration,
                            )
                    except (_json.JSONDecodeError, OSError) as e:
                        logger.warning(
                            "SubprocessExecutorStep: failed to read "
                            "iter_%03d/skill_log.json: %s",
                            ctx.iteration,
                            e,
                        )
        post_phase_ms["skill_logs"] = (_time.perf_counter() - _phase_t0) * 1000.0

        # ---- Phase 4: Build feedback text ------------------------------
        _phase_t0 = _time.perf_counter()
        feedback_parts = [
            f"success_rate={success_rate:.2f} ({successes}/{n})",
            f"avg_score={avg_score:.3f}",
        ]
        if candidate_search_details:
            best_idx = candidate_search_details["best_candidate_index"]
            feedback_parts.append(
                "\n[candidate search] "
                f"generated {candidate_search_details['num_candidates']} candidates; "
                f"each evaluated on {candidate_search_details['n_seeds_per_candidate']} "
                f"seed(s); selected candidate {best_idx}."
            )
            for summary in candidate_search_details["candidates"][:5]:
                feedback_parts.append(
                    "  cand={candidate_index}: avg_score={avg_score:.3f}, "
                    "success_rate={success_rate:.2f} ({successes}/{n_seeds})".format(
                        **summary
                    )
                )
        # Per-seed feedback now carries a stdout tail (skill calls +
        # status-bearing results); bumped from 200 to 2500 so the author
        # actually sees IK_Failed / gripper info / freespace_move reasons
        # rather than just "reward=0, success=False".
        for r in per_seed:
            fb = r.get("feedback", "")
            if fb:
                feedback_parts.append(f"  seed={r['seed']}: {fb[:2500]}")
        if session is not None:
            skill_feedback = _format_skill_log_feedback(
                session.iterations_dir(ctx.iteration) / "skill_log.json"
            )
            if skill_feedback:
                feedback_parts.append(skill_feedback)
        post_phase_ms["build_feedback"] = (_time.perf_counter() - _phase_t0) * 1000.0

        # ---- Phase 4.5: Oracle reward diagnostics ----------------------
        # Reconstruct each sub-predicate of the task's _check_success from
        # result.json.details (no sim access). Output is stashed on each
        # per_seed entry and injected into the per-seed VLM prompt in Phase 5
        # and into the cross-seed synthesis prompt in Phase B (reflection.py).
        _phase_t0 = _time.perf_counter()
        reward_diagnostics: list[dict[str, Any]] = []
        try:
            from enpire.env.forge.cap.reward import build_reward_evaluator

            reward_cfg = getattr(config, "reward", None)
            env_name = getattr(getattr(config, "env", None), "name", "")
            evaluator = build_reward_evaluator(reward_cfg, env_name=env_name)
            for r in per_seed:
                exec_dir = (
                    session.exec_dir(ctx.iteration, r.get("exec_id", r["seed"]))
                    if session is not None
                    else None
                )
                details: dict[str, Any] = {}
                if exec_dir is not None:
                    result_path = exec_dir / "result.json"
                    if result_path.exists():
                        try:
                            import json as _json

                            details = (
                                _json.loads(result_path.read_text()).get(
                                    "details", {}
                                )
                                or {}
                            )
                        except Exception as e:
                            logger.warning(
                                "Phase 4.5: failed to read result.json for "
                                "seed=%s: %s",
                                r.get("seed"),
                                e,
                            )
                diagnosis = evaluator.evaluate(
                    details,
                    seed=int(r.get("seed", 0)),
                    exec_id=int(r.get("exec_id", 0)),
                )
                r["oracle_reward"] = diagnosis.to_dict()
                r["oracle_reward_markdown"] = evaluator.as_markdown(diagnosis)
                reward_diagnostics.append(diagnosis.to_dict())

            # Persist raw oracle diagnostics so they survive overwrites.
            if session is not None and reward_diagnostics:
                import json as _json

                diag_path = (
                    session.iterations_dir(ctx.iteration) / "reward_diagnostics.json"
                )
                diag_path.write_text(
                    _json.dumps(
                        {
                            "evaluator": getattr(evaluator, "name", "reward"),
                            "task": getattr(evaluator, "task", ""),
                            "per_seed": reward_diagnostics,
                        },
                        indent=2,
                        default=str,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        except Exception as e:
            logger.warning(
                "Phase 4.5: reward evaluator failed (iter=%d): %s",
                ctx.iteration,
                e,
            )
        post_phase_ms["oracle_reward"] = (_time.perf_counter() - _phase_t0) * 1000.0

        # Deferred eval_summary.json — now that Phase 4.5 has stashed
        # ``oracle_reward`` + ``oracle_reward_markdown`` on each per_seed
        # entry, the saved summary carries the full diagnostics instead of
        # just success/score/feedback.
        _phase_t0 = _time.perf_counter()
        if session is not None:
            try:
                session.save_eval_summary(ctx.iteration, per_seed)
            except Exception as e:
                logger.warning(
                    "save_eval_summary failed (iter=%d): %s", ctx.iteration, e
                )
        post_phase_ms["save_summary"] = (_time.perf_counter() - _phase_t0) * 1000.0

        # ---- Phase 5: Per-seed VLM reflection (parallel) ---------------
        # Each seed does one Gemini call with its before/after image. The
        # loop used to be serial (~7s × N seeds), which dominated the
        # "executor" tail on parallel runs. We parallelize with a
        # ThreadPool since the calls are network-bound. Cap workers at 10
        # to stay under typical Gemini-flash rate limits.
        _phase_t0 = _time.perf_counter()
        per_seed_reflections: list[str] = []
        reflections_by_seed: dict[int, str] = {}
        vlm_query = ctx.namespace.get("vlm_query")
        reflect_enabled = bool(getattr(config, "reflect", False))
        max_iters = int(getattr(config, "max_iterations", 1) or 1)
        oracle_mode = bool(getattr(config, "oracle", None))
        reflection_cfg = getattr(config, "reflection", None)
        vlm_backend = (
            getattr(reflection_cfg, "vlm_backend", "gemini")
            if reflection_cfg is not None
            else "gemini"
        )
        vlm_model = (
            getattr(reflection_cfg, "vlm_model", None)
            if reflection_cfg is not None
            else None
        )
        want_per_seed_vlm = (
            reflect_enabled
            and max_iters > 1
            and not oracle_mode
            and ctx.iteration + 1 < max_iters  # no benefit on the final iter
        )
        if (
            n > 1
            and vlm_query is not None
            and session is not None
            and want_per_seed_vlm
        ):
            reward_cfg_snapshot = getattr(config, "reward", None)
            inject_oracle = bool(
                getattr(reward_cfg_snapshot, "inject_into_per_seed_vlm", True)
            )

            # Visual evidence mode — dispatched inside _reflect_one.
            ve_cfg = getattr(reflection_cfg, "visual_evidence", None)
            ve_mode = getattr(ve_cfg, "mode", "before_after") if ve_cfg else "before_after"
            ve_n_frames = int(getattr(ve_cfg, "n_frames", 8) or 8)
            ve_cameras = list(getattr(ve_cfg, "cameras", None) or getattr(reflection_cfg, "cameras", ["top"]))

            def _reflect_one(r: dict[str, Any]) -> str | None:
                exec_dir = session.exec_dir(ctx.iteration, r["exec_id"])
                if exec_dir is None:
                    return None
                try:
                    import numpy as np
                    from PIL import Image

                    seed_status = "SUCCESS" if r.get("success") else "FAIL"
                    oracle_block = (
                        r.get("oracle_reward_markdown", "") if inject_oracle else ""
                    )

                    if ve_mode == "uniform_sample":
                        # Subsample trajectory video — gives the VLM a timeline
                        # of what happened rather than just start/end state.
                        images, labels = _sample_trajectory_frames(
                            exec_dir, ve_cameras, ve_n_frames
                        )
                        if not images:
                            return None
                        query = (
                            f"Task: {ctx.task}\n"
                            f"Seed {r['seed']} result: {seed_status}, score={r.get('score', 0):.3f}.\n"
                            + (f"\n{oracle_block}\n\n" if oracle_block else "")
                            + f"The {len(images)} images are uniformly sampled frames from the full execution trajectory "
                            f"(first to last). Describe what the robot attempted, what stage it reached, "
                            f"and what caused the final outcome. Under 100 words."
                        )
                    else:
                        # Default: before/after images
                        vis_dir = exec_dir / "vis"
                        if not vis_dir.exists():
                            return None
                        vis_images = sorted(vis_dir.glob("*_before_after.png"))
                        if not vis_images:
                            return None
                        images = [np.array(Image.open(p)) for p in vis_images]
                        labels = [p.stem.replace("_before_after", "") for p in vis_images]
                        query = (
                            f"This shows before (left) and after (right) images from a robot manipulation attempt. "
                            f"Seed {r['seed']} result: {seed_status}, score={r.get('score', 0):.3f}.\n"
                            f"Task: {ctx.task}\n"
                            + (
                                f"\n{oracle_block}\n\n"
                                "Use the ground-truth predicates above as authoritative — "
                                "the images should confirm, not contradict, which predicates passed/failed. "
                                if oracle_block
                                else ""
                            )
                            + "What changed? Did the robot make progress toward the goal? "
                            "What went wrong or right? Be specific, under 100 words."
                        )

                    vlm_kwargs: dict[str, Any] = {
                        "text": query,
                        "backend": vlm_backend,
                        "image": images,
                        "image_labels": labels,
                    }
                    if vlm_model:
                        vlm_kwargs["model"] = vlm_model
                    if vlm_backend == "nvidia":
                        vlm_kwargs["telemetry_source"] = "phase5_reflection"
                    refl = vlm_query(**vlm_kwargs)
                    return f"[seed={r['seed']} {seed_status}] {refl}"
                except Exception as e:
                    logger.warning(
                        "Per-seed reflection failed for seed=%s: %s", r["seed"], e
                    )
                    return None

            from concurrent.futures import ThreadPoolExecutor, as_completed

            # NVIDIA concurrency is governed in the shared provider scheduler.
            # Keep the thread fanout near the healthy key count so blocked
            # workers do not pile up in this parent phase.
            if vlm_backend == "nvidia":
                from enpire.env.forge.cap.agent.providers.nvidia import (
                    list_nvidia_keys,
                    read_nvidia_scheduler_stats,
                )

                healthy = [
                    row
                    for row in read_nvidia_scheduler_stats()
                    if row.get("status") == "healthy"
                ]
                n_keys = len(healthy) or len(list_nvidia_keys()) or 1
                max_workers = min(n, n_keys)
            else:
                max_workers = min(n, 10)
            _dash_print(
                f"  [executor] reflecting on {n} seeds (backend={vlm_backend}, "
                f"parallel × {max_workers})"
            )

            # Publish progress to the live dashboard if one is active.
            from enpire.env.forge.cap.agent.agent_dashboard import current as _dash_current

            _dash = _dash_current()
            if _dash is not None:
                _dash.on_reflect_start(n, backend=vlm_backend)

            raw_by_seed: dict[int, str | None] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_seed = {
                    pool.submit(_reflect_one, r): r["seed"] for r in per_seed
                }
                for fut in as_completed(future_to_seed):
                    seed = future_to_seed[fut]
                    try:
                        refl = fut.result()
                    except Exception:
                        refl = None
                    raw_by_seed[seed] = refl
                    if _dash is not None:
                        _dash.on_reflect_tick(failed=(refl is None))

            if _dash is not None:
                _dash.on_reflect_end()

            # Preserve per_seed ordering in the final list / logs, and strip
            # the ``[seed=N STATUS] `` prefix for the per-seed disk artifact
            # (Phase A markdown file) where the status is already in the
            # heading.
            import re as _re

            for r in per_seed:
                refl = raw_by_seed.get(r["seed"])
                if refl:
                    per_seed_reflections.append(refl)
                    cleaned = _re.sub(r"^\[seed=\d+\s+\w+\]\s*", "", refl)
                    reflections_by_seed[int(r["seed"])] = cleaned
                    _dash_print(
                        f"  [executor] seed={r['seed']} reflection: {refl[:100]}..."
                    )
        post_phase_ms["vlm_reflect"] = (_time.perf_counter() - _phase_t0) * 1000.0

        # ---- Phase 6: Finalize evaluation ------------------------------
        _phase_t0 = _time.perf_counter()
        if per_seed_reflections:
            feedback_parts.append("\n=== PER-SEED VISUAL ANALYSIS ===")
            feedback_parts.extend(per_seed_reflections)
            feedback_parts.append(
                f"\n=== SUMMARY: {successes}/{n} seeds succeeded. ==="
            )

        # Persist Phase A (raw per-seed reflections) to disk so it survives
        # the SelfReflectionStep overwrite of ``ctx.evaluation.feedback`` and
        # remains human-inspectable alongside the Phase B synthesis.
        if session is not None and (per_seed or reflections_by_seed):
            try:
                session.save_per_seed_reflections(
                    ctx.iteration,
                    task=ctx.task,
                    per_seed=per_seed,
                    reflections_by_seed=reflections_by_seed,
                    backend=vlm_backend,
                )
            except Exception as e:
                logger.warning(
                    "save_per_seed_reflections failed (iter=%d): %s",
                    ctx.iteration,
                    e,
                )

        ctx.evaluation = AgentEvaluation(
            success=success_rate >= 1.0,  # all seeds must succeed
            score=avg_score,
            feedback="\n".join(feedback_parts),
            method="subprocess_reward",
            details={
                "n_seeds": n,
                "success_rate": success_rate,
                "successes": successes,
                "avg_score": avg_score,
                "per_seed": per_seed,
                "per_seed_reflections": per_seed_reflections,
                "per_seed_reflections_by_seed": reflections_by_seed,
                "candidate_search": candidate_search_details,
                "post_exec_profile_ms": post_phase_ms,
            },
        )
        if success_rate >= 1.0:
            ctx.should_stop = True
            ctx.stop_reason = "task_success"
        post_phase_ms["finalize"] = (_time.perf_counter() - _phase_t0) * 1000.0

        # One-line profile summary for the dashboard log.
        def _fmt_phase(ms: float) -> str:
            return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s"

        total_ms = sum(post_phase_ms.values())
        _dash_print(
            "  [executor] post-exec profile ("
            + _fmt_phase(total_ms)
            + " total): "
            + ", ".join(f"{k}={_fmt_phase(v)}" for k, v in post_phase_ms.items())
        )

        return ctx


# Convenience: project root for building subprocess commands
_ROOT_PATH = __import__("pathlib").Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# RewardEvaluatorStep
# ---------------------------------------------------------------------------


class RewardEvaluatorStep(AgentStep):
    """Evaluate task success using get_task_info() reward/success signal.

    Reads:  ctx.namespace["get_task_info"]
    Writes: ctx.evaluation
    Sets ctx.should_stop = True on success.
    """

    name = "reward_evaluator"

    def run(self, ctx: AgentContext) -> AgentContext:
        from enpire.env.forge.cap.agent.agent_context import AgentEvaluation

        get_task_info = ctx.namespace.get("get_task_info")
        if get_task_info is None:
            return ctx

        try:
            info = get_task_info()
        except Exception as e:
            logger.warning("RewardEvaluatorStep: get_task_info failed: %s", e)
            return ctx

        reward = float(info.get("reward", 0.0))
        success = bool(info.get("success", False))
        obj_pos = info.get("obj_pos")

        feedback = f"reward={reward:.3f}, success={success}" + (
            f", obj_pos={[round(x, 3) for x in obj_pos]}" if obj_pos else ""
        )
        if not success and ctx.execution_result and ctx.execution_result.error:
            feedback += f"\nexecution error: {ctx.execution_result.error}"

        ctx.evaluation = AgentEvaluation(
            success=success,
            score=reward,
            feedback=feedback,
            method="reward",
            details=info,
        )

        if ctx.session is not None:
            ctx.session.save_evaluation(ctx.iteration, ctx.evaluation)

        if success:
            ctx.should_stop = True
            ctx.stop_reason = "task_success"
            logger.info("RewardEvaluatorStep: task succeeded (reward=%.3f)", reward)

        return ctx


# ---------------------------------------------------------------------------
# SelfReflectionStep
# ---------------------------------------------------------------------------


class SelfReflectionStep(AgentStep):
    """Produce improvement feedback for the next iteration using a ReflectionStrategy.

    Accepts either a ReflectionStrategy (preferred) or an LLMBackend (convenience
    shorthand — wraps it in TextReflectionStrategy automatically).

    Examples::

        # Text-based (default)
        SelfReflectionStep(TextReflectionStrategy(llm))
        SelfReflectionStep(llm)  # shorthand

        # Vision-based (Gemini describes scene)
        SelfReflectionStep(VisionReflectionStrategy(llm, cameras=["top", "wrist"]))

        # Both combined
        SelfReflectionStep(CompositeReflectionStrategy([
            TextReflectionStrategy(llm),
            VisionReflectionStrategy(llm),
        ]))
    """

    name = "self_reflection"

    def __init__(self, strategy: ReflectionStrategy | LLMBackend) -> None:
        from enpire.env.forge.cap.agent.reflection import (
            ReflectionStrategy as _RS,
        )
        from enpire.env.forge.cap.agent.reflection import (
            TextReflectionStrategy,
        )

        if isinstance(strategy, _RS):
            self._strategy = strategy
        else:
            # Convenience: bare LLMBackend → TextReflectionStrategy
            self._strategy = TextReflectionStrategy(strategy)

    def run(self, ctx: AgentContext) -> AgentContext:
        from enpire.env.forge.cap.agent.agent_context import AgentEvaluation

        if ctx.should_stop:
            return ctx

        try:
            feedback = self._strategy.reflect(ctx)
        except Exception as e:
            logger.warning("SelfReflectionStep: strategy failed: %s", e)
            feedback = f"(reflection failed: {e})"

        # Save raw LLM input from reflection
        llm = getattr(self._strategy, "_llm", None)
        if llm is not None:
            _save_llm_input(ctx, llm, "reflection")

        logger.info("SelfReflectionStep feedback:\n%s", feedback)

        if ctx.session is not None:
            ctx.session.reflection_path(ctx.iteration).write_text(
                feedback, encoding="utf-8"
            )

        if ctx.evaluation is not None:
            ctx.evaluation.feedback = feedback
            ctx.evaluation.method = self._strategy.__class__.__name__
        else:
            ctx.evaluation = AgentEvaluation(
                success=False,
                score=0.0,
                feedback=feedback,
                method=self._strategy.__class__.__name__,
            )

        return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aggregate_skill_logs(
    session: Any,
    iteration: int,
    n_seeds: int,
    seed_start: int,
    exec_ids: list[int] | None = None,
) -> None:
    """Merge exec_NNN/skill_log.json files into iter_NNN/skill_log.json."""
    import json as _json

    merged: dict[str, Any] = {}  # name -> entry with accumulated logs

    ids = exec_ids if exec_ids is not None else list(range(n_seeds))
    for i, exec_id in enumerate(ids):
        seed = seed_start + i
        exec_dir = session.exec_dir(iteration, int(exec_id))
        skill_log_path = exec_dir / "skill_log.json"
        if not skill_log_path.exists():
            continue
        try:
            entries = _json.loads(skill_log_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError):
            continue
        for entry in entries:
            name = entry.get("name", "")
            if name not in merged:
                merged[name] = {
                    "name": name,
                    "base_name": entry.get("base_name", name),
                    "version": entry.get("version", 1),
                    "docstring": entry.get("docstring", ""),
                    "calls": 0,
                    "logs": [],
                }
            # Tag each log with seed
            for log in entry.get("logs", []):
                log["seed"] = seed
            merged[name]["logs"].extend(entry.get("logs", []))
            merged[name]["calls"] += entry.get("calls", 0)

    if not merged:
        return

    # Compute success_rate
    for entry in merged.values():
        logs = entry["logs"]
        if logs:
            entry["success_rate"] = round(
                sum(1 for lg in logs if lg.get("success")) / len(logs), 3
            )
        else:
            entry["success_rate"] = 0.0

    iter_dir = session.iterations_dir(iteration)
    out_path = iter_dir / "skill_log.json"
    out_path.write_text(
        _json.dumps(list(merged.values()), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _format_skill_log_feedback(skill_log_path: Any, *, max_chars: int = 3500) -> str:
    """Compact skill execution summary for the next LLM iteration.

    The stdout tail is often dominated by dashboard/get_robot_state polling on
    real hardware, so the important per-skill return logs can be pushed out of
    context.  This summary keeps the next skill author/assembly prompt focused
    on which skills actually ran and what their last structured log said.
    """
    import json as _json

    try:
        entries = _json.loads(skill_log_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(entries, list) or not entries:
        return ""

    lines = ["[skill log summary]"]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        calls = int(entry.get("calls", 0) or 0)
        if calls <= 0:
            continue
        logs = entry.get("logs") or []
        last_log = logs[-1] if isinstance(logs, list) and logs else {}
        last_text = _json.dumps(last_log, default=str, ensure_ascii=False)
        if len(last_text) > 700:
            last_text = last_text[:700] + "..."
        lines.append(
            f"- {entry.get('name', '?')}: calls={calls}, "
            f"success_rate={entry.get('success_rate', 0.0)}, last={last_text}"
        )
    text = "\n".join(lines)
    return text[:max_chars]


def _best_iteration(history):
    """Return the ``IterationRecord`` with the highest ``evaluation.score``.

    Ties break to the earlier iteration (more stable). Returns ``None`` if
    no history entry has a populated evaluation.

    Used by ``SkillAuthorStep`` and ``AssemblyGeneratorStep`` to surface a
    CHAMPION ATTEMPT in their prior-attempts prompt so a run of failed
    iterations doesn't cascade-poison the feedback history.
    """
    best = None
    best_score = -1.0
    for rec in history:
        if rec.evaluation is None:
            continue
        score = float(getattr(rec.evaluation, "score", 0.0) or 0.0)
        if score > best_score:
            best = rec
            best_score = score
    return best


def _serialize_robot_state(state: Any) -> dict | None:
    if state is None:
        return None
    try:
        arms = {}
        for side, arm in state.arms.items():
            arms[side] = {
                "ee_pos": [round(float(x), 4) for x in arm.ee_pos],
                "ee_rpy": [round(float(x), 4) for x in arm.ee_rpy],
                "gripper_pos": round(float(arm.gripper_pos), 4),
            }
        return {"arms": arms}
    except Exception:
        return None


def _fmt(v: Any, max_len: int = 120) -> str:
    s = repr(v)
    return s[:max_len] + "..." if len(s) > max_len else s


def _split_thoughts_and_code(response: str) -> tuple[str, str]:
    """Split an LLM response into (thoughts, code).

    Expects the response to contain a ``THOUGHTS:`` section followed by a
    `````python`` code block. If no explicit thoughts section is found,
    everything before the first code block is treated as thoughts.
    """
    import re

    # Look for ```python ... ``` block
    code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)

    if code_match:
        code = code_match.group(1).strip()
        # Everything before the code block is thoughts
        thoughts = response[: code_match.start()].strip()
        # Strip "THOUGHTS:" prefix if present
        for prefix in ("THOUGHTS:", "**THOUGHTS:**", "## THOUGHTS", "Thoughts:"):
            if thoughts.upper().startswith(prefix.upper()):
                thoughts = thoughts[len(prefix) :].strip()
                break
        return thoughts, code

    # No code block — maybe raw code with thoughts as comments at top
    lines = response.strip().splitlines()
    thought_lines: list[str] = []
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("#")
            or stripped == ""
            or not any(
                stripped.startswith(kw)
                for kw in (
                    "import ",
                    "from ",
                    "def ",
                    "class ",
                    "state ",
                    "arm ",
                    "np.",
                    "set_",
                    "get_",
                    "move_",
                    "open_",
                    "close_",
                )
            )
        ):
            thought_lines.append(line)
        else:
            code_start = i
            break
    else:
        # All lines look like thoughts/comments — treat whole thing as code
        return "", response.strip()

    thoughts = "\n".join(thought_lines).strip()
    code = "\n".join(lines[code_start:]).strip()
    return thoughts, code


def _sample_trajectory_frames(
    exec_dir: Any,
    cameras: list[str],
    n_frames: int,
) -> tuple[list[Any], list[str]]:
    """Subsample ``n_frames`` from saved trajectory videos in exec_dir.

    Always includes the first and last frame. Returns (images, labels) where
    each label is ``"{camera}_frame{i:03d}"``.

    Falls back gracefully: skips cameras whose video is missing, returns empty
    lists if no video is readable.
    """
    from pathlib import Path as _Path

    images: list[Any] = []
    labels: list[str] = []

    for cam in cameras:
        video_path = _Path(exec_dir) / f"{cam}.mp4"
        if not video_path.exists():
            continue
        try:
            import numpy as _np

            # Try imageio first (lighter dep); fall back to cv2.
            try:
                import imageio.v3 as _iio

                all_frames = list(_iio.imiter(str(video_path)))
            except Exception:
                import cv2 as _cv2

                cap = _cv2.VideoCapture(str(video_path))
                all_frames = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    all_frames.append(_cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB))
                cap.release()

            total = len(all_frames)
            if total == 0:
                continue

            # Pick indices: always include 0 and total-1, fill with evenly
            # spaced interior indices up to n_frames total.
            if total <= n_frames:
                indices = list(range(total))
            else:
                step = (total - 1) / (n_frames - 1)
                indices = sorted(set(
                    [0]
                    + [round(step * i) for i in range(1, n_frames - 1)]
                    + [total - 1]
                ))

            for idx in indices:
                frame = _np.array(all_frames[idx])
                images.append(frame)
                labels.append(f"{cam}_frame{idx:03d}")

        except Exception as e:
            logger.warning("_sample_trajectory_frames: %s failed: %s", video_path, e)

    return images, labels
