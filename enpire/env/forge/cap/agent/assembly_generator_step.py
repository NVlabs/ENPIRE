# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 of the two-stage code generator: orchestrate skills into code.py.

Reads:  ctx.task, ctx.session._skill_library, ctx.task_info, ctx.robot_state,
        ctx.history (prior iterations' feedback)
Writes: ctx.code, ctx.thoughts (optional)

Validation: rejects any top-level ``def`` / ``class`` / ``lambda`` /
``async def`` in the generated code — helpers must live in the library.

See ``docs/plan/two_stage_code_generator.md`` for the full design.
"""

from __future__ import annotations

import ast
import json
import logging
from typing import TYPE_CHECKING, Any

from enpire.env.forge.cap.agent.agent_step import (
    AgentStep,
    _best_iteration,
    _save_llm_input,
    _serialize_robot_state,
    _split_thoughts_and_code,
)

if TYPE_CHECKING:
    from enpire.env.forge.cap.agent.agent_context import AgentContext
    from enpire.env.forge.cap.agent.llm.base import LLMBackend

logger = logging.getLogger(__name__)


class AssemblyGeneratorStep(AgentStep):
    """Generate the assembly ``code.py`` that imports from skill_library and orchestrates."""

    name = "assembly_generator"

    def __init__(
        self,
        llm: LLMBackend,
        max_retries: int = 2,
        forbid_toplevel_def: bool = True,
        require_skill_library_imports: bool = False,
    ) -> None:
        self._llm = llm
        self._max_retries = max(0, int(max_retries))
        self._forbid_def = bool(forbid_toplevel_def)
        self._require_imports = bool(require_skill_library_imports)

    def run(self, ctx: AgentContext) -> AgentContext:
        prompt = self._build_user_prompt(ctx)
        logger.info(
            "AssemblyGeneratorStep: requesting assembly code (iter %d)",
            ctx.iteration,
        )

        last_errors: list[str] = []
        code = ""
        thoughts = ""
        for attempt in range(self._max_retries + 1):
            if last_errors:
                retry_prompt = (
                    prompt
                    + "\n\n⚠ The previous assembly code had validation errors "
                    "— emit a corrected code.py:\n"
                    + "\n".join(f"  - {e}" for e in last_errors)
                )
            else:
                retry_prompt = prompt

            raw = self._llm.generate_code(retry_prompt, ctx_context(ctx)) or ""
            _save_llm_input(self._safe_session(ctx), self._llm, f"assembly_attempt{attempt}")
            thoughts, code = _split_thoughts_and_code(raw)

            errors = _validate_assembly(
                code,
                forbid_toplevel_def=self._forbid_def,
                known_skill_names=_known_skill_names(ctx),
                require_skill_imports=self._require_imports,
                skill_dir=_skill_dir(ctx),
            )
            if not errors:
                logger.info(
                    "AssemblyGeneratorStep: assembly accepted on attempt %d", attempt
                )
                break
            logger.warning(
                "AssemblyGeneratorStep: attempt %d validation errors: %s",
                attempt,
                errors,
            )
            last_errors = errors
        else:
            # Exhausted retries — save what we have and let executor / reflection
            # see the failure mode. Calling this an error is appropriate since
            # the whole point of stage 2 is to emit valid orchestration.
            logger.error(
                "AssemblyGeneratorStep: exhausted %d retries, saving best-effort code. "
                "Errors: %s",
                self._max_retries + 1,
                last_errors,
            )

        ctx.thoughts = thoughts
        ctx.code = code

        if ctx.session is not None:
            ctx.session.save_code(ctx.iteration, code)
            if thoughts:
                ctx.session.thoughts_path(ctx.iteration).write_text(
                    thoughts, encoding="utf-8"
                )
        return ctx

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_user_prompt(self, ctx: AgentContext) -> str:
        parts: list[str] = []
        parts.append(f"=== TASK ===\n{ctx.task}\n")

        session = ctx.session
        sl = getattr(session, "_skill_library", None) if session is not None else None
        if sl is not None:
            index_text = sl.index_for_prompt() or "(empty — no skills authored yet)"
            parts.append(
                f"\n=== CURRENT SKILL LIBRARY ===\n{index_text}\n"
                f"=== END SKILL LIBRARY ===\n"
            )

        robot_state_dict = _serialize_robot_state(ctx.robot_state)
        if robot_state_dict:
            parts.append(
                f"\nCurrent robot state:\n"
                f"{json.dumps(robot_state_dict, default=str, indent=2)}\n"
            )
        if ctx.task_info:
            parts.append(
                f"\nTask info:\n{json.dumps(ctx.task_info, default=str, indent=2)}\n"
            )

        if ctx.iteration > 0 and ctx.history:
            ag_cfg = getattr(getattr(ctx, "config", None), "assembly_generator", None)
            max_fb = int(getattr(ag_cfg, "max_feedback_chars", 12000) or 12000)
            max_code = int(getattr(ag_cfg, "max_prior_code_chars", 4000) or 4000)
            max_stdout = int(getattr(ag_cfg, "max_prior_stdout_chars", 2000) or 2000)
            max_prior = int(getattr(ag_cfg, "max_prior_attempts", 3) or 3)

            def _cap(s: str, n: int, label: str) -> str:
                if len(s) <= n:
                    return s
                return (
                    s[:n]
                    + f"\n[... truncated {len(s) - n} chars "
                    f"(max_{label}_chars={n}) ...]"
                )

            # Champion anchor — always shown first regardless of recency so the
            # LLM's goal is unambiguous: write assembly that beats the best score,
            # not assembly that fixes the most recent crash.
            champion = _best_iteration(ctx.history)
            recent = ctx.history[-max_prior:]
            if champion is not None and champion.evaluation and champion.evaluation.score > 0.0:
                fb = champion.evaluation.feedback
                score = champion.evaluation.score
                # Prefer the snapshot code.py (complete, untruncated) over the
                # in-memory history record which is capped at max_prior_code_chars.
                prior_code = champion.code or ""
                if ctx.session is not None:
                    snap_code = ctx.session.baseline_code_path()
                    if snap_code.exists():
                        prior_code = snap_code.read_text(encoding="utf-8")
                parts.append(
                    f"\n=== CHAMPION ASSEMBLY (iter {champion.iteration + 1}, "
                    f"score={score:.3f}) ===\n"
                    "This is the best-performing assembly. Start from this code. "
                    "Only change what the failure feedback explicitly identifies as wrong. "
                    "Preserve all parameter values that are not mentioned in the feedback.\n"
                    f"```python\n{prior_code}\n```\n"
                    f"feedback:\n{_cap(fb, max_fb, 'feedback')}\n"
                    "=== END CHAMPION ===\n"
                )

            # Show recent attempts, marking the champion inline if it falls here.
            parts.append("\n=== PRIOR ATTEMPTS ===")
            for rec in recent:
                fb = rec.evaluation.feedback if rec.evaluation else "(no feedback)"
                prior_code = rec.code or ""
                prior_stdout = ""
                if rec.execution_result is not None:
                    prior_stdout = (
                        getattr(rec.execution_result, "stdout", "") or ""
                    )
                score = rec.evaluation.score if rec.evaluation else 0.0
                is_champ = champion is not None and rec.iteration == champion.iteration
                label = f"[CHAMPION, score={score:.3f}]" if is_champ else f"score={score:.3f}"
                parts.append(
                    f"\n--- attempt {rec.iteration + 1} ({label}) ---\n"
                    f"code:\n```python\n{_cap(prior_code, max_code, 'prior_code')}\n```\n"
                    f"stdout:\n{_cap(prior_stdout, max_stdout, 'prior_stdout')}\n"
                    f"feedback:\n{_cap(fb, max_fb, 'feedback')}\n"
                )
            parts.append("=== END PRIOR ATTEMPTS ===\n")

        parts.append(
            "\n=== INSTRUCTIONS ===\n"
            "Write the assembly code.py that solves the task by calling the "
            "library skills above. First write a THOUGHTS: section explaining "
            "your orchestration plan, then a single ```python block with the "
            "assembly code.\n\n"
            "HARD RULES (validator will reject and retry if violated):\n"
            "  • No top-level `def`, `class`, `async def`, or `lambda`.\n"
            "  • No `try`/`except` anywhere.\n"
            "  • Every skill you call must come from a `from skill_library.<base> "
            "import <name>` at the top, unless it is a bare namespace tool.\n"
            "  • End with `go_home(SIDE)` and `print(f\"Success: ...\")`."
        )
        return "\n".join(parts)

    def _safe_session(self, ctx: AgentContext) -> AgentContext:
        # Helper for _save_llm_input which expects an AgentContext.
        return ctx


def ctx_context(ctx: AgentContext) -> dict[str, Any]:
    """Build the minimal dict passed to ``LLMBackend.generate_code``.

    Assembly-generator uses its own user prompt, so this dict is small —
    only the fields the LLM backend itself consults (e.g. ``iteration``,
    ``config`` for retry-template access).
    """
    return {
        "iteration": ctx.iteration,
        "config": getattr(ctx, "config", None),
        # Keep the other fields so bridge_llm._build_prompt can still run in
        # legacy mode if ever reached — but our user prompt is already built.
        "tools": ctx.tool_schemas,
        "robot_state": None,
        "history": "",
        "failure_history": [],
    }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _validate_assembly(
    code: str,
    *,
    forbid_toplevel_def: bool = True,
    known_skill_names: set[str] | None = None,
    require_skill_imports: bool = False,
    skill_dir=None,
) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    errors: list[str] = []
    if not code.strip():
        errors.append("empty code.py")
        return errors

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"syntax error: {e.msg} at line {e.lineno}")
        return errors

    if forbid_toplevel_def:
        forbidden_types = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        )
        for node in tree.body:
            if isinstance(node, forbidden_types):
                kind = type(node).__name__
                errors.append(
                    f"top-level {kind} `{node.name}` at line {node.lineno} "
                    f"not allowed — move to skill_library via stage-1 authoring"
                )
        # Detect lambdas anywhere (including inside assignments).
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                errors.append(
                    f"lambda at line {node.lineno} not allowed — move to "
                    f"skill_library via stage-1 authoring"
                )

    # Forbid try/except (same policy as coding_style.md).
    for node in ast.walk(tree):
        if isinstance(node, (ast.Try, ast.TryStar if hasattr(ast, "TryStar") else ast.Try)):
            errors.append(
                f"try/except at line {node.lineno} not allowed — check "
                f"status-bearing return values instead"
            )

    # Optionally require imports from skill_library (when the library is
    # expected to be non-empty). Off by default — iter 0 with empty library
    # may legitimately have no library imports.
    if require_skill_imports and known_skill_names:
        imported_from_lib: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "skill_library."
            ):
                for alias in node.names:
                    imported_from_lib.add(alias.asname or alias.name)
        if not imported_from_lib:
            errors.append(
                "no imports from skill_library — if the library has skills "
                "that fit this task, use them instead of writing raw namespace "
                "calls"
            )

    # If the code imports from skill_library.<x>, those imports must map to
    # entries that actually exist. Catches copy-paste drift.
    if known_skill_names is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "skill_library."
            ):
                for alias in node.names:
                    imported = alias.asname or alias.name
                    base = alias.name
                    if base not in known_skill_names:
                        errors.append(
                            f"import `from {node.module} import {imported}` "
                            f"references a skill that does not exist in the "
                            f"library — known skills: {sorted(known_skill_names)}"
                        )

    # Cross-stage signature contract: every skill_library.* call must match
    # the freshly-authored skill's signature. Catches the skill_author ↔
    # assembly_generator API drift that otherwise crashes all 30 seeds with
    # a TypeError on the first call.
    if skill_dir is not None:
        errors.extend(_validate_skill_call_signatures(code, skill_dir))

    return errors


def _known_skill_names(ctx: AgentContext) -> set[str] | None:
    session = ctx.session
    sl = getattr(session, "_skill_library", None) if session is not None else None
    if sl is None:
        return None
    return set(sl._index.keys())


def _skill_dir(ctx: AgentContext):
    session = ctx.session
    sl = getattr(session, "_skill_library", None) if session is not None else None
    if sl is None:
        return None
    return getattr(sl, "skill_dir", None)


def _skill_param_names(skill_dir, base_name: str, skill_name: str) -> set[str] | None:
    """Return the parameter names of ``skill_name`` inside
    ``skill_library/{base_name}.py``, or ``None`` if the skill file or
    function can't be located.

    Multiple versions can coexist in the same base file (``hover_above.py``
    holds ``hover_above_v1``, ``hover_above_v2`` …), so we walk the AST and
    match by function name.
    """
    from pathlib import Path

    path = Path(skill_dir) / f"{base_name}.py"
    if not path.exists():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != skill_name:
                continue
            a = node.args
            names: set[str] = set()
            names.update(arg.arg for arg in a.args)
            names.update(arg.arg for arg in a.kwonlyargs)
            names.update(arg.arg for arg in a.posonlyargs)
            if a.vararg is not None:
                names.add("*")  # sentinel — consumer treats as "any positional"
            if a.kwarg is not None:
                names.add("**")  # sentinel — consumer treats as "any kwarg ok"
            return names
    return None


def _validate_skill_call_signatures(
    code: str,
    skill_dir,
) -> list[str]:
    """Cross-stage contract check: every call site of a ``skill_library.*``
    function must match that function's actual ``def`` signature.

    This catches the most destructive class of assembly failures — the
    ``nudge_lift_v1(delta_z=...)``-type crash where skill_author emitted
    ``def nudge_lift_v1(side, target_z=...)`` but the assembly called it
    with a different kwarg. A TypeError in seed 0 kills all 30 seeds.

    Signature checks:
      1. Every keyword argument at the call site must be a parameter name
         (or ``**kwargs`` is declared on the skill).
      2. Positional arg count must fit the callable (or ``*args`` is
         declared).

    Returns a list of human-readable error strings (empty = ok). Skips
    unknown-skill calls; other validators handle those.
    """
    if skill_dir is None:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    # Build ``local_name → (base_module, original_skill_name)`` so aliased
    # imports (``import foo_v1 as f``) still resolve to the real AST node.
    imported: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        mod = node.module or ""
        if not mod.startswith("skill_library."):
            continue
        base_name = mod.split(".", 1)[1]
        for alias in node.names:
            local = alias.asname or alias.name
            original = alias.name
            imported[local] = (base_name, original)

    if not imported:
        return []

    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        fn_name: str | None = None
        if isinstance(fn, ast.Name):
            fn_name = fn.id
        # Skip attribute / subscripted calls — they're not direct library calls.
        if fn_name is None or fn_name not in imported:
            continue
        base_name, original = imported[fn_name]
        param_names = _skill_param_names(skill_dir, base_name, original)
        if param_names is None:
            continue  # unknown file; other validators handle this

        has_var_kw = "**" in param_names
        has_var_pos = "*" in param_names
        real_params = {p for p in param_names if p not in ("*", "**")}

        # Keyword-arg check
        for kw in node.keywords:
            if kw.arg is None:
                # **kwargs splat at call site — can't statically verify.
                continue
            if kw.arg not in real_params and not has_var_kw:
                errors.append(
                    f"line {node.lineno}: {fn_name}(...) called with keyword "
                    f"argument {kw.arg!r}, but the skill's signature accepts "
                    f"only: {sorted(real_params)}"
                )

        # Positional-arg count check
        pos_count = sum(1 for a in node.args if not isinstance(a, ast.Starred))
        if not has_var_pos:
            # Some positional args may have been passed as kwargs; worst case
            # is ``pos_count > len(real_params)``, which is an overflow.
            if pos_count > len(real_params):
                errors.append(
                    f"line {node.lineno}: {fn_name}(...) called with "
                    f"{pos_count} positional args, but the skill accepts "
                    f"at most {len(real_params)}"
                )

    return errors
