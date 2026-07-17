"""Stage-1 of the two-stage code generator: author new skills to the library.

Reads:  ctx.task, ctx.session._skill_library, ctx.task_info, ctx.robot_state,
        ctx.history (prior iterations' feedback)
Writes: ctx.session._skill_library (new entries), optionally ctx.thoughts
        (free-form authoring plan the LLM emits before its code blocks).

Does NOT write ``ctx.code`` — that is the responsibility of the stage-2
``AssemblyGeneratorStep``. Stage-1's output is skill files on disk plus
updates to the skill-library index.

See ``docs/plan/two_stage_code_generator.md`` for the full design.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from enpire.env.forge.cap.agent.agent_step import (
    AgentStep,
    _best_iteration,
    _save_llm_input,
    _serialize_robot_state,
)

if TYPE_CHECKING:
    from enpire.env.forge.cap.agent.agent_context import AgentContext
    from enpire.env.forge.cap.agent.llm.base import LLMBackend

logger = logging.getLogger(__name__)


# Grammar (see plan §5):
#   ### (new|refine|replace): <base_name>_v<N>
#   rationale: <one sentence>
#   parent: <existing_name>        # required for refine/replace
#   ```python
#   @skill
#   def <name>(...):
#       ...
#   ```
_HEADER_RE = re.compile(
    r"^###\s+(?P<cls>new|refine|replace)\s*:\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$",
    re.MULTILINE,
)
_RATIONALE_RE = re.compile(r"^rationale\s*:\s*(?P<text>.+?)\s*$", re.MULTILINE)
_PARENT_RE = re.compile(r"^parent\s*:\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(?P<code>.*?)```", re.DOTALL)


class SkillAuthorStep(AgentStep):
    """Author new or refined skills directly into the library before assembly runs."""

    name = "skill_author"

    def __init__(
        self,
        llm: LLMBackend,
        max_retries: int = 2,
        allow_no_new_skills: bool = True,
    ) -> None:
        self._llm = llm
        self._max_retries = max(0, int(max_retries))
        self._allow_empty = bool(allow_no_new_skills)

    def run(self, ctx: AgentContext) -> AgentContext:
        session = ctx.session
        sl = getattr(session, "_skill_library", None) if session is not None else None
        if sl is None:
            logger.info(
                "SkillAuthorStep: no skill library attached, skipping "
                "(set skill_library.enabled=true to author skills)"
            )
            return ctx

        prompt = self._build_user_prompt(ctx, sl)
        logger.info(
            "SkillAuthorStep: requesting skill plan (iter %d, library=%d entries)",
            ctx.iteration,
            sl.total_skills,
        )

        last_errors: list[str] = []
        response = ""
        for attempt in range(self._max_retries + 1):
            if last_errors:
                # Include validator feedback from the previous attempt
                retry_prompt = (
                    prompt
                    + "\n\n⚠ The previous authoring plan had validation errors "
                    "— fix them and re-emit the full plan:\n"
                    + "\n".join(f"  - {e}" for e in last_errors)
                )
            else:
                retry_prompt = prompt
            response = self._llm.generate_text(retry_prompt)
            _save_llm_input(ctx, self._llm, f"skill_author_attempt{attempt}")

            parsed_blocks, parse_errors = _parse_authoring_response(response)
            if parse_errors:
                last_errors = parse_errors
                logger.warning(
                    "SkillAuthorStep: attempt %d parse errors: %s", attempt, parse_errors
                )
                continue

            # Attempt to commit each block via author_skill
            commit_errors: list[str] = []
            committed: list[str] = []
            for block in parsed_blocks:
                try:
                    entry = sl.author_skill(
                        base_name=block["base_name"],
                        source=block["source"],
                        classification=block["classification"],
                        rationale=block["rationale"],
                        iteration=ctx.iteration,
                        parent_name=block.get("parent"),
                    )
                    committed.append(entry.get("name", block["base_name"]))
                except (ValueError, SyntaxError) as e:
                    commit_errors.append(
                        f"{block['classification']} {block['base_name']}: {e}"
                    )

            if commit_errors and not committed:
                # Nothing landed — retry with errors surfaced
                last_errors = commit_errors
                logger.warning(
                    "SkillAuthorStep: attempt %d commit errors: %s",
                    attempt,
                    commit_errors,
                )
                continue

            # Partial success or full success — accept and break out.
            if commit_errors:
                logger.warning(
                    "SkillAuthorStep: committed %d skill(s) but %d failed: %s",
                    len(committed),
                    len(commit_errors),
                    commit_errors,
                )
            logger.info(
                "SkillAuthorStep: committed %d skill(s) in iter %d: %s",
                len(committed),
                ctx.iteration,
                committed,
            )
            break
        else:
            # Exhausted retries. Graceful degradation: log, keep going.
            logger.warning(
                "SkillAuthorStep: exhausted %d retries without valid output — "
                "proceeding with existing library. Last errors: %s",
                self._max_retries + 1,
                last_errors,
            )
            if not self._allow_empty:
                raise RuntimeError(
                    f"SkillAuthorStep failed after {self._max_retries + 1} "
                    f"attempts: {last_errors}"
                )

        # Save the raw response for debugging (pristine — before any mutation)
        if session is not None:
            iter_dir = session.iterations_dir(ctx.iteration)
            (iter_dir / "skill_author_response.md").write_text(
                response, encoding="utf-8"
            )

        return ctx

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_user_prompt(self, ctx: AgentContext, sl: Any) -> str:
        parts: list[str] = []

        parts.append(f"=== TASK ===\n{ctx.task}\n")

        # Current library index (so LLM can decide reuse vs. author)
        index_text = sl.index_for_prompt() or "(empty — no skills authored yet)"
        parts.append(f"\n=== CURRENT SKILL LIBRARY ===\n{index_text}\n=== END SKILL LIBRARY ===\n")

        # Robot state / task info — geometry hints for parameter defaults.
        robot_state_dict = _serialize_robot_state(ctx.robot_state)
        if robot_state_dict:
            parts.append(
                f"\nCurrent robot state:\n{json.dumps(robot_state_dict, default=str, indent=2)}\n"
            )
        if ctx.task_info:
            parts.append(
                f"\nTask info:\n{json.dumps(ctx.task_info, default=str, indent=2)}\n"
            )

        # Full experiment history from history.md (all iterations, one paragraph each).
        # Gives the LLM the big picture and enables self-detecting stagnation.
        if ctx.session is not None and ctx.iteration > 0:
            history_path = ctx.session.run_dir / "history.md"
            if history_path.exists():
                history_text = history_path.read_text(encoding="utf-8").strip()
                if history_text:
                    parts.append(
                        f"\n=== EXPERIMENT HISTORY ===\n{history_text}\n=== END HISTORY ===\n"
                    )
                    # Escalation: count trailing discard entries
                    sa_cfg = getattr(getattr(ctx, "config", None), "skill_author", None)
                    threshold = int(getattr(sa_cfg, "escalation_threshold", 3) or 3)
                    stagnant = 0
                    for line in reversed(history_text.splitlines()):
                        if "— discard" in line and line.startswith("## iter_"):
                            stagnant += 1
                        elif line.startswith("## iter_"):
                            break  # hit a keep entry
                    if stagnant >= threshold:
                        escalation_path = (
                            Path(__file__).resolve().parents[2]
                            / "cap/prompt/system/skill_author_escalation.md"
                        )
                        try:
                            escalation_text = escalation_path.read_text(encoding="utf-8")
                            escalation_text = escalation_text.replace(
                                "{stagnant}", str(stagnant)
                            )
                        except OSError:
                            escalation_text = (
                                f"⚠ ESCALATION: baseline unchanged for {stagnant} "
                                f"iterations. Propose new: or replace: only."
                            )
                        parts.append(f"\n{escalation_text}\n")

        # Prior-iteration feedback (filtered: emphasis on skill-authoring concerns)
        if ctx.iteration > 0 and ctx.history:
            sa_cfg = getattr(getattr(ctx, "config", None), "skill_author", None)
            max_fb = int(getattr(sa_cfg, "max_feedback_chars", 12000) or 12000)
            max_prior = int(getattr(sa_cfg, "max_prior_attempts", 3) or 3)

            def _cap_fb(s: str) -> str:
                if len(s) <= max_fb:
                    return s
                return (
                    s[:max_fb]
                    + f"\n\n[... truncated {len(s) - max_fb} chars "
                    f"(max_feedback_chars={max_fb}) ...]"
                )

            # Champion anchor — always shown first regardless of recency so
            # the LLM's goal is clear: improve on the best score, not fix the
            # most recent crash.
            champion = _best_iteration(ctx.history)
            recent = ctx.history[-max_prior:]
            if champion is not None and champion.evaluation and champion.evaluation.score > 0.0:
                fb = champion.evaluation.feedback
                score = champion.evaluation.score
                parts.append(
                    f"\n=== CHAMPION ATTEMPT (iter {champion.iteration + 1}, "
                    f"score={score:.3f}) ===\n"
                    "This is the best-performing iteration. Your goal: author skills "
                    "that IMPROVE on this score. DO NOT regress — refine from this "
                    "baseline even if more recent attempts crashed.\n"
                    f"feedback:\n{_cap_fb(fb)}\n"
                    "=== END CHAMPION ===\n"
                )

            # Show recent attempts, marking the champion inline if it falls here.
            parts.append("\n=== PRIOR ATTEMPTS ===")
            for rec in recent:
                fb = rec.evaluation.feedback if rec.evaluation else "(no feedback)"
                score = rec.evaluation.score if rec.evaluation else 0.0
                is_champ = champion is not None and rec.iteration == champion.iteration
                label = f"[CHAMPION, score={score:.3f}]" if is_champ else f"score={score:.3f}"
                parts.append(
                    f"\n--- attempt {rec.iteration + 1} ({label}) ---\n"
                    f"feedback:\n{_cap_fb(fb)}\n"
                )
            parts.append("=== END PRIOR ATTEMPTS ===\n")

        # Instructions
        parts.append(
            "\n=== INSTRUCTIONS ===\n"
            "Decide which (if any) new or refined skills you need in order to "
            "solve this task. Follow the authoring grammar from the system "
            "prompt. If the library already contains what you need, output:\n\n"
            "## Authoring plan\n\n"
            "No new skills needed — proceed to assembly.\n\n"
            "Otherwise, emit one `### classification: base_name_vN` section per "
            "new/refined/replaced skill, each followed by a `rationale:` line "
            "(and `parent:` for refine/replace) and a ```python code block "
            "containing exactly one `@skill`-decorated function."
        )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_authoring_response(text: str) -> tuple[list[dict], list[str]]:
    """Parse the stage-1 LLM response into a list of authoring blocks.

    Returns:
        (blocks, errors) — blocks is a list of dicts each shaped like::

            {
              "classification": "new|refine|replace",
              "base_name": "hover_above",
              "source": "@skill\\ndef hover_above_v1(...): ...",
              "rationale": "...",
              "parent": "hover_above_v1",   # optional, only for refine/replace
            }

        errors is a list of human-readable error strings describing any
        sections that failed to parse. Empty errors + empty blocks is a valid
        "no new skills needed" response.
    """
    errors: list[str] = []
    blocks: list[dict] = []

    # Split into sections on `### ` headers. Prefix (if any) is discarded —
    # it contains the LLM's authoring plan prose.
    header_matches = list(_HEADER_RE.finditer(text))
    if not header_matches:
        # No skills authored. Check for the expected empty-plan phrasing to
        # avoid rejecting valid "nothing to do" responses.
        if re.search(r"no new skills? (needed|required)", text, re.IGNORECASE):
            return [], []
        # Tolerant: empty response also means no new skills.
        stripped = text.strip()
        if not stripped or stripped.lower().startswith("## authoring plan"):
            return [], []
        # Fallback: we don't recognize it, but don't fail — just author nothing.
        return [], []

    # Slice the response at each header boundary.
    for i, m in enumerate(header_matches):
        start = m.start()
        end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(text)
        section = text[start:end]

        classification = m.group("cls")
        full_name = m.group("name")  # e.g. "hover_above_v1"

        # Pull out rationale + parent + code block from the section.
        rat_m = _RATIONALE_RE.search(section)
        if not rat_m:
            errors.append(
                f"section {classification} {full_name}: missing 'rationale:' line"
            )
            continue
        rationale = rat_m.group("text").strip()

        parent_name: str | None = None
        par_m = _PARENT_RE.search(section)
        if par_m:
            parent_name = par_m.group("name").strip()
        if classification in ("refine", "replace") and not parent_name:
            errors.append(
                f"section {classification} {full_name}: "
                f"'parent:' line required for {classification}"
            )
            continue
        if classification == "new" and parent_name:
            errors.append(
                f"section new {full_name}: 'parent:' line is not allowed for "
                f"'new' classification"
            )
            continue

        code_m = _CODE_BLOCK_RE.search(section)
        if not code_m:
            errors.append(
                f"section {classification} {full_name}: missing ```python block"
            )
            continue
        source = textwrap.dedent(code_m.group("code")).strip() + "\n"

        # Derive base_name from the header's name field (strip _vN suffix).
        from enpire.env.forge.cap.agent.skill_registry import _parse_name

        base_name, _ = _parse_name(full_name)

        blocks.append(
            {
                "classification": classification,
                "base_name": base_name,
                "source": source,
                "rationale": rationale,
                "parent": parent_name,
            }
        )

    return blocks, errors
