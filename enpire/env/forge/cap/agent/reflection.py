# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable reflection strategies for SelfReflectionStep.

A ReflectionStrategy takes an AgentContext after a failed execution and
returns a plain-text feedback string for the next code generation iteration.

Built-in strategies:
    TextReflectionStrategy   — LLM analyzes code + stdout + error + tool history
    VisionReflectionStrategy — VLM describes last camera frame(s), LLM reflects
    CompositeReflectionStrategy — runs multiple strategies, merges feedback

Usage::

    from enpire.env.forge.cap.agent.reflection import TextReflectionStrategy, VisionReflectionStrategy
    from enpire.env.forge.cap.agent.agent_step import SelfReflectionStep

    # Text only
    step = SelfReflectionStep(TextReflectionStrategy(llm))

    # Vision only (Gemini describes scene, then reflects)
    step = SelfReflectionStep(VisionReflectionStrategy(llm, cameras=["top", "wrist"]))

    # Both combined
    step = SelfReflectionStep(CompositeReflectionStrategy([
        TextReflectionStrategy(llm),
        VisionReflectionStrategy(llm, cameras=["top"]),
    ]))
"""

from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from enpire.env.forge.cap.agent.agent_context import AgentContext
    from enpire.env.forge.cap.agent.llm.base import LLMBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ReflectionStrategy(ABC):
    """Interface for a reflection strategy.

    Receives the AgentContext after a failed iteration and returns a
    plain-text feedback string (no code) for the next code_generator pass.
    """

    @abstractmethod
    def reflect(self, ctx: AgentContext) -> str:
        """Analyze the failed iteration and return actionable feedback."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Text reflection (original behaviour)
# ---------------------------------------------------------------------------


class TextReflectionStrategy(ReflectionStrategy):
    """Reflect using code, stdout, error, reward info, and tool call history.

    Calls llm.generate_text() with a structured diagnosis prompt.
    """

    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def reflect(self, ctx: AgentContext) -> str:
        cross_seed = _reflect_cross_seed(ctx, self._llm)
        if cross_seed is not None:
            return cross_seed

        stdout, error = _load_execution_evidence(ctx)
        code = ctx.code or ""
        eval_info = _format_eval(ctx)
        thoughts = ctx.thoughts or "(no thoughts recorded)"

        pm = getattr(ctx, "prompt_memory", None)
        if pm is not None:
            try:
                prompt = pm.load(
                    "system",
                    "text_reflection",
                    task=ctx.task,
                    thoughts=thoughts,
                    iteration=str(ctx.iteration),
                    code=code,
                    stdout=stdout,
                    error=error,
                    eval_info=eval_info,
                )
            except FileNotFoundError:
                pm = None  # fall through to inline

        if pm is None:
            prompt = (
                "You are analyzing a failed robot manipulation attempt to guide the next try.\n\n"
                f"Task: {ctx.task}\n\n"
                f"=== PLANNED APPROACH (what the agent expected to happen) ===\n{thoughts}\n\n"
                f"=== Code (iteration {ctx.iteration}) ===\n```python\n{code}\n```\n\n"
                f"=== ACTUAL EXECUTION ===\n"
                f"Stdout:\n{stdout}\n\n"
                f"Error: {error}\n\n"
                f"Task evaluation: {eval_info}\n\n"
                "Compare the PLANNED APPROACH vs ACTUAL EXECUTION:\n"
                "1. Where did reality diverge from the plan?\n"
                "2. What was the root cause of the divergence?\n"
                "3. Give 2-4 specific, actionable fixes for the NEXT attempt.\n"
                "Be concrete — include corrected values, positions, or sequences. "
                "Do NOT rewrite the code. Plain text only, under 250 words."
            )
        return self._llm.generate_text(prompt)


# ---------------------------------------------------------------------------
# Vision reflection
# ---------------------------------------------------------------------------


class VisionReflectionStrategy(ReflectionStrategy):
    """Reflect using VLM visual scene understanding via the unified vlm_query tool.

    Uses vlm_query() from the execution namespace — the same tool available in
    generated robot code. Supports all backends: gemini (default), qwen, gpt, etc.
    Reads GEMINI_API_KEY from the environment automatically.

    Steps:
    1. Call vlm_query() with media=["camera:top", ...] to describe the end-of-iteration scene.
       Uses a single multi-image call so the VLM sees all cameras at once.
    2. Call llm.generate_text() combining the visual description with execution info.

    Falls back to TextReflectionStrategy if vlm_query is unavailable.
    """

    def __init__(
        self,
        llm: LLMBackend,
        cameras: list[str] | None = None,
        vlm_backend: str = "gemini",
    ) -> None:
        self._llm = llm
        self._cameras = cameras or ["top"]
        self._vlm_backend = vlm_backend
        self._text_fallback = TextReflectionStrategy(llm)

    def reflect(self, ctx: AgentContext) -> str:
        cross_seed = _reflect_cross_seed(ctx, self._llm)
        if cross_seed is not None:
            return cross_seed

        vlm_query = ctx.namespace.get("vlm_query")
        if vlm_query is None:
            print("[reflect:vision] vlm_query not in namespace — falling back to text")
            logger.warning(
                "VisionReflectionStrategy: vlm_query not in namespace, falling back to text"
            )
            return self._text_fallback.reflect(ctx)

        # Capture all camera frames upfront
        images, cam_labels = _capture_camera_frames(
            ctx, self._cameras, "reflect:vision"
        )
        if not images:
            print(
                "[reflect:vision] No valid camera frames — falling back to text reflection"
            )
            return self._text_fallback.reflect(ctx)

        pm = getattr(ctx, "prompt_memory", None)
        if pm is not None:
            try:
                scene_query = pm.load_section(
                    "system", "vision_reflection", "Scene Query", task=ctx.task
                )
            except (FileNotFoundError, ValueError):
                pm = None
        if pm is None:
            scene_query = (
                f"The robot was attempting this task: {ctx.task}\n\n"
                "Describe the robot's current state at the END of this attempt. "
                "Focus on: (1) where is the gripper/fingertip relative to the target, "
                "(2) did the robot accomplish the task goal (e.g. press a button, pick up an object, etc.), "
                "(3) what visually went wrong — be specific about positions and orientations."
            )

        # Single multi-image VLM call with all cameras
        print(
            f"[reflect:vision] Querying {self._vlm_backend} with {len(images)} camera(s): {cam_labels}"
        )
        try:
            scene_description = vlm_query(
                text=scene_query,
                backend=self._vlm_backend,
                image=images,
                image_labels=cam_labels,
            )
            print(f"[reflect:vision] scene: {scene_description[:200]}")
        except Exception as e:
            print(f"[reflect:vision] vlm_query FAILED: {e}")
            return self._text_fallback.reflect(ctx)

        logger.info("VisionReflectionStrategy: got description for %s", cam_labels)

        # Save raw VLM response for inspection
        if ctx.session is not None:
            ctx.session.vision_scene_path(ctx.iteration).write_text(
                f"# Visual Scene — Iteration {ctx.iteration}\n\n"
                f"**Cameras:** {', '.join(cam_labels)}\n"
                f"**Backend:** {self._vlm_backend}\n\n"
                f"{scene_description}",
                encoding="utf-8",
            )

        # Visual diff: compare before/after frames
        visual_diff = _compute_visual_diff(ctx, self._cameras, self._vlm_backend)

        eval_info = _format_eval(ctx)
        stdout, error = _load_execution_evidence(ctx)
        thoughts = ctx.thoughts or "(no thoughts recorded)"

        prompt_parts = [
            "You are analyzing a failed robot manipulation attempt to guide the next try.\n",
            f"Task: {ctx.task}\n",
            f"=== PLANNED APPROACH (what the agent expected to happen) ===\n{thoughts}\n",
            f"=== VISUAL SCENE (what actually happened — cameras: {', '.join(cam_labels)}) ===\n"
            f"{scene_description}\n",
        ]
        if visual_diff:
            prompt_parts.append(
                f"=== VISUAL CHANGES (before -> after execution) ===\n{visual_diff}\n"
            )
        eval_note = ""
        if ctx.evaluation and not ctx.evaluation.details.get("success", False):
            eval_note = (
                "(Note: evaluation is ground-truth from the simulator. "
                "If visual descriptions above claim success but evaluation says otherwise, "
                "trust the evaluation numbers.)\n"
            )
        prompt_parts.extend(
            [
                f"=== Task evaluation (ground truth) ===\n{eval_info}\n{eval_note}",
                f"=== Execution stdout ===\n{stdout}\n",
                f"=== Execution error ===\n{error}\n",
                "Compare the PLANNED APPROACH vs what the VISUAL SCENE and VISUAL CHANGES show:\n"
                "1. Where did reality diverge from the plan?\n"
                "2. What does the camera show that contradicts the expected behavior?\n"
                "3. What specific changes occurred between the start and end of execution?\n"
                "4. Give 2-4 specific, actionable fixes for the NEXT attempt.\n"
                "Be concrete — include corrected positions or sequences. "
                "Plain text only, under 300 words.",
            ]
        )
        return self._llm.generate_text("\n".join(prompt_parts))


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


class CompositeReflectionStrategy(ReflectionStrategy):
    """Unified reflection combining text analysis + visual observation in ONE LLM call.

    Instead of running strategies independently and concatenating, this:
    1. Captures visual scene descriptions from cameras (via vlm_query)
    2. Builds a single prompt with ALL evidence (thoughts, code, stdout, errors,
       visual scene, eval, tool history)
    3. Makes ONE LLM call for a joint analysis

    Falls back to TextReflectionStrategy if cameras are unavailable.
    """

    def __init__(
        self,
        strategies: list[ReflectionStrategy],
    ) -> None:
        # Extract LLM and camera config from the strategies
        self._llm: LLMBackend | None = None
        self._cameras: list[str] = ["top"]
        self._vlm_backend: str = "gemini"
        self._text_fallback: TextReflectionStrategy | None = None

        for s in strategies:
            if isinstance(s, TextReflectionStrategy):
                self._llm = s._llm
                self._text_fallback = s
            elif isinstance(s, VisionReflectionStrategy):
                self._llm = self._llm or s._llm
                self._cameras = s._cameras
                self._vlm_backend = s._vlm_backend

        if self._llm is None and strategies:
            # Fallback: use first strategy's LLM if available
            self._llm = getattr(strategies[0], "_llm", None)
        if self._text_fallback is None and self._llm is not None:
            self._text_fallback = TextReflectionStrategy(self._llm)

    def reflect(self, ctx: AgentContext) -> str:
        if self._llm is None:
            return "(no LLM available for reflection)"

        cross_seed = _reflect_cross_seed(ctx, self._llm)
        if cross_seed is not None:
            return cross_seed

        # --- Gather visual scene descriptions ---
        visual_section = ""
        cam_labels: list[str] = []
        vlm_query = ctx.namespace.get("vlm_query")

        if vlm_query is not None:
            images, cam_labels = _capture_camera_frames(
                ctx, self._cameras, "reflect:composite"
            )
            if images:
                # Priority: config override > markdown > inline fallback
                refl_cfg = getattr(getattr(ctx, "config", None), "reflection", None)
                pm = getattr(ctx, "prompt_memory", None)
                if refl_cfg and refl_cfg.scene_query:
                    scene_query_template = refl_cfg.scene_query
                elif pm is not None:
                    try:
                        scene_query_template = pm.load_section(
                            "system", "composite_reflection", "Scene Query"
                        )
                    except (FileNotFoundError, ValueError):
                        scene_query_template = None
                else:
                    scene_query_template = None

                if scene_query_template is None:
                    scene_query_template = (
                        "The robot was attempting this task: {task}\n\n"
                        "Describe the robot's current state at the END of this attempt. "
                        "Focus on: where is the gripper/fingertip relative to the target, "
                        "did the robot accomplish the task goal, and what visually went wrong."
                    )
                scene_query = scene_query_template.replace("{task}", ctx.task)

                # Single multi-image VLM call
                print(
                    f"[reflect:composite] Querying {self._vlm_backend} with "
                    f"{len(images)} camera(s): {cam_labels}"
                )
                try:
                    visual_section = vlm_query(
                        text=scene_query,
                        backend=self._vlm_backend,
                        image=images,
                        image_labels=cam_labels,
                    )
                    print(f"[reflect:composite] scene: {visual_section[:200]}")
                except Exception as e:
                    print(f"[reflect:composite] vlm_query failed: {e}")
                    visual_section = ""

                if visual_section and ctx.session is not None:
                    ctx.session.vision_scene_path(ctx.iteration).write_text(
                        f"# Visual Scene — Iteration {ctx.iteration}\n\n"
                        f"**Cameras:** {', '.join(cam_labels)}\n"
                        f"**Backend:** {self._vlm_backend}\n\n"
                        f"{visual_section}",
                        encoding="utf-8",
                    )

        # --- Gather text evidence from on-disk artifacts ---
        stdout, error = _load_execution_evidence(ctx)
        code = ctx.code or ""
        thoughts = ctx.thoughts or "(no thoughts recorded)"
        eval_info = _format_eval(ctx)

        # --- Build ONE unified prompt ---
        prompt_parts = [
            "You are analyzing a failed robot manipulation attempt. "
            "Consider ALL evidence below jointly — text execution data AND visual observations — "
            "to produce a single, unified analysis.\n",
            f"Task: {ctx.task}\n",
            f"=== PLANNED APPROACH ===\n{thoughts}\n",
            f"=== CODE (iteration {ctx.iteration}) ===\n```python\n{code}\n```\n",
        ]

        if visual_section:
            prompt_parts.append(
                f"=== VISUAL SCENE (cameras: {', '.join(cam_labels or self._cameras)}) ===\n{visual_section}\n"
            )

        # Visual diff: compare before/after frames
        visual_diff = _compute_visual_diff(ctx, self._cameras, self._vlm_backend)
        if visual_diff:
            prompt_parts.append(
                f"=== VISUAL CHANGES (before -> after execution) ===\n{visual_diff}\n"
            )

        # Priority: config override > markdown > inline fallback
        refl_cfg = getattr(getattr(ctx, "config", None), "reflection", None)
        pm = getattr(ctx, "prompt_memory", None)
        if refl_cfg and refl_cfg.analysis_prompt:
            analysis_instruction = refl_cfg.analysis_prompt
        elif pm is not None:
            try:
                analysis_instruction = pm.load_section(
                    "system", "composite_reflection", "Analysis Prompt"
                )
            except (FileNotFoundError, ValueError):
                analysis_instruction = None
        else:
            analysis_instruction = None

        if analysis_instruction is None:
            analysis_instruction = (
                "Jointly consider the planned approach, the code, the visual evidence, and the "
                "execution results. In a single unified analysis:\n"
                "1. Where did reality diverge from the plan? What does the visual evidence confirm?\n"
                "2. What is the root cause?\n"
                "3. Give 2-4 specific, actionable fixes for the NEXT attempt.\n"
                "Be concrete. Do NOT rewrite the code. Under 250 words."
            )

        # Ground-truth eval immediately after visual sections so the LLM
        # can calibrate VLM descriptions against simulator numbers.
        eval_note = ""
        if ctx.evaluation and not ctx.evaluation.details.get("success", False):
            eval_note = (
                "(Note: evaluation is ground-truth from the simulator. "
                "If visual descriptions above claim success but evaluation says otherwise, "
                "trust the evaluation numbers.)\n"
            )
        prompt_parts.append(
            f"=== TASK EVALUATION (ground truth) ===\n{eval_info}\n{eval_note}"
        )

        prompt_parts.extend(
            [
                f"=== EXECUTION OUTPUT ===\nStdout:\n{stdout}\nError: {error}\n",
                f"=== YOUR ANALYSIS ===\n{analysis_instruction}",
            ]
        )

        prompt = "\n".join(prompt_parts)

        try:
            return self._llm.generate_text(prompt)
        except Exception as e:
            logger.warning("CompositeReflectionStrategy: LLM failed: %s", e)
            if self._text_fallback is not None:
                return self._text_fallback.reflect(ctx)
            return f"(reflection failed: {e})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_camera_frames(
    ctx: AgentContext,
    cameras: list[str],
    label: str = "reflect",
) -> tuple[list, list[str]]:
    """Capture valid frames from multiple cameras.

    Returns (images, labels) — parallel lists of numpy arrays and camera names.
    Skips cameras that fail or return dummy frames.
    """
    get_camera_image = ctx.namespace.get("get_camera_image")
    if get_camera_image is None:
        return [], []

    images = []
    labels = []
    for cam in cameras:
        try:
            img = get_camera_image(cam)
        except Exception as e:
            print(f"[{label}] get_camera_image({cam!r}) failed: {e} — skipping")
            continue
        if img is None or img.shape[0] <= 1 or img.shape[1] <= 1:
            print(f"[{label}] {cam!r} returned a dummy/empty frame — skipping")
            continue
        images.append(img)
        labels.append(cam)
    return images, labels


def _compute_visual_diff(
    ctx: AgentContext,
    cameras: list[str],
    vlm_backend: str,
) -> str:
    """Compare before/after frames via VLM and return a visual diff description.

    Creates a side-by-side (np.hstack) comparison image per camera,
    sends all comparison images to vlm_query in a single multi-image call,
    and returns the VLM's description.  Returns empty string if frames are
    unavailable or the query fails.
    """
    if not ctx.frames_before or not ctx.frames_after:
        return ""

    vlm_query = ctx.namespace.get("vlm_query")
    if vlm_query is None:
        return ""

    import numpy as np

    from enpire.env.forge.cap.agent.tools._artifact_log import log_image

    pm = getattr(ctx, "prompt_memory", None)
    if pm is not None:
        try:
            diff_prompt = pm.load("system", "visual_diff", task=ctx.task)
        except FileNotFoundError:
            pm = None
    if pm is None:
        diff_prompt = (
            f"The robot was attempting this task: {ctx.task}\n\n"
            "These two images show the scene BEFORE (left half) and AFTER (right half) "
            "the robot executed its code.\n\n"
            "Describe specifically what changed between the two images. Focus on:\n"
            "1. Did any object move? Where was it before and where is it now?\n"
            "2. Did the robot gripper/arm position change?\n"
            "3. Did the task goal get closer to being achieved?\n"
            "Be specific about positions and spatial relationships."
        )

    diff_images: list = []
    diff_labels: list[str] = []
    for cam in cameras:
        before = ctx.frames_before.get(cam)
        after = ctx.frames_after.get(cam)
        if before is None or after is None:
            continue

        # Resize to match heights if needed for hstack
        h_b, h_a = before.shape[0], after.shape[0]
        if h_b != h_a:
            import cv2

            target_h = min(h_b, h_a)
            if h_b != target_h:
                before = cv2.resize(
                    before, (int(before.shape[1] * target_h / h_b), target_h)
                )
            if h_a != target_h:
                after = cv2.resize(
                    after, (int(after.shape[1] * target_h / h_a), target_h)
                )

        comparison = np.hstack([before, after])
        log_image(comparison, tag=f"visual_diff_iter{ctx.iteration:03d}", label=cam)
        diff_images.append(comparison)
        diff_labels.append(f"{cam} (before|after)")

    if not diff_images:
        return ""

    try:
        diff_text = vlm_query(
            text=diff_prompt,
            backend=vlm_backend,
            image=diff_images,
            image_labels=diff_labels,
        )
        print(f"[reflect:visual_diff] {diff_text[:200]}")
    except Exception as e:
        print(f"[reflect:visual_diff] vlm_query failed: {e}")
        logger.warning("visual_diff vlm_query failed: %s", e)
        return ""

    # Save diff description to session
    if ctx.session is not None:
        ctx.session.visual_diff_path(ctx.iteration).write_text(
            f"# Visual Diff — Iteration {ctx.iteration}\n\n"
            f"**Cameras:** {', '.join(cameras)}\n"
            f"**Backend:** {vlm_backend}\n\n"
            f"{diff_text}",
            encoding="utf-8",
        )

    return diff_text


def _load_execution_evidence(ctx: AgentContext) -> tuple[str, str]:
    """Read (stdout, error) for this iteration from on-disk artifacts.

    Single source of truth regardless of whether the executor was in-process
    or subprocess — both write ``exec_NNN/exec.log`` via the session.
    Returns empty strings if no session is attached or no artifacts exist.
    """
    session = getattr(ctx, "session", None)
    if session is None:
        return "", ""
    try:
        artifacts = session.load_run_artifacts(ctx.iteration, exec_id=0)
    except Exception as e:
        logger.warning("load_run_artifacts failed: %s", e)
        return "", ""
    return artifacts.stdout, artifacts.error


# ---------------------------------------------------------------------------
# Cross-seed helpers (Phase B reads Phase A + per-seed exec logs)
# ---------------------------------------------------------------------------


def _build_per_seed_evidence(
    ctx: AgentContext,
    stdout_tail_chars: int = 1200,
    error_tail_chars: int = 600,
) -> str | None:
    """Build the per-seed evidence block for the cross-seed reflection prompt.

    Returns ``None`` when no per-seed data is available (e.g. inline
    executor, n_seeds=1, or the executor did not produce
    ``ctx.evaluation.details['per_seed']``). Callers should fall back to
    single-seed reflection in that case.
    """
    evaluation = getattr(ctx, "evaluation", None)
    details = getattr(evaluation, "details", None) if evaluation is not None else None
    if not details:
        return None
    per_seed = details.get("per_seed") or []
    if not per_seed:
        return None

    refl_by_seed: dict[int, str] = {
        int(k): v
        for k, v in (details.get("per_seed_reflections_by_seed") or {}).items()
    }
    session = getattr(ctx, "session", None)

    blocks: list[str] = []
    for r in per_seed:
        seed = r.get("seed")
        exec_id = r.get("exec_id", seed)
        status = "SUCCESS" if r.get("success") else "FAIL"
        score = r.get("score", 0.0)

        stdout_text = ""
        error_text = ""
        if session is not None and exec_id is not None:
            try:
                art = session.load_run_artifacts(ctx.iteration, exec_id=int(exec_id))
                stdout_text = art.stdout
                error_text = art.error
            except Exception as e:
                logger.warning(
                    "load_run_artifacts(exec_id=%s) failed: %s", exec_id, e
                )

        # Prefer the richer per_seed["feedback"] (includes a deliberate
        # stdout tail) when the exec.log is missing or empty.
        if not stdout_text:
            stdout_text = r.get("feedback", "") or ""

        stdout_tail = (stdout_text or "")[-stdout_tail_chars:].rstrip()
        error_tail = (error_text or "")[-error_tail_chars:].rstrip()
        vlm_refl = (refl_by_seed.get(int(seed)) if seed is not None else "") or ""

        oracle = r.get("oracle_reward") or {}
        obj_name = oracle.get("obj_name") or ""
        head = f"### Seed {seed} — {status} (score={score:.3f})"
        if obj_name:
            head += f" — obj_name={obj_name!r}"
        lines = [head]
        oracle_md = r.get("oracle_reward_markdown") or ""
        if oracle_md:
            lines.append(oracle_md.strip())
        if vlm_refl:
            lines.append("**VLM reflection (Phase A):**")
            lines.append(vlm_refl.strip())
        if stdout_tail:
            lines.append("**stdout tail:**")
            lines.append("```")
            lines.append(stdout_tail)
            lines.append("```")
        if error_tail:
            lines.append("**error:**")
            lines.append("```")
            lines.append(error_tail)
            lines.append("```")
        if not vlm_refl and not stdout_tail and not error_tail:
            lines.append("_(no evidence captured)_")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _cross_seed_stats(ctx: AgentContext) -> dict[str, Any]:
    """Extract headline stats (n_seeds / successes / avg_score) from ctx."""
    details = (
        ctx.evaluation.details if ctx.evaluation and ctx.evaluation.details else {}
    )
    n = int(details.get("n_seeds", 0))
    # Aggregate oracle failure-category histogram + per-object breakdown
    # across seeds (if oracle data is available). Use ``failure_category``
    # (stable: "(a)"/"(b)"/"(c)"/"(d)") for aggregation rather than
    # ``failure_cause`` (which contains per-seed metrics and fragments the
    # histogram). Legacy oracle outputs without the category field fall
    # back to the first two tokens of failure_cause.
    cause_counts: dict[str, int] = {}
    per_obj: dict[str, dict[str, int]] = {}  # obj_name -> {"s": successes, "n": total}
    for r in details.get("per_seed") or []:
        oracle = r.get("oracle_reward") or {}
        category = oracle.get("failure_category") or ""
        if not category:
            # Legacy fallback: extract "(x)" prefix from failure_cause.
            fc = oracle.get("failure_cause") or ""
            if fc.startswith("("):
                category = fc.split(" ", 1)[0]
        if category and category != "success":
            cause_counts[category] = cause_counts.get(category, 0) + 1
        obj_name = oracle.get("obj_name") or ""
        if obj_name:
            slot = per_obj.setdefault(obj_name, {"s": 0, "n": 0})
            slot["n"] += 1
            if r.get("success"):
                slot["s"] += 1

    if cause_counts:
        causes_line = "; ".join(
            f"{k} × {v}"
            for k, v in sorted(cause_counts.items(), key=lambda kv: -kv[1])
        )
    else:
        causes_line = "(no oracle diagnostics available)"

    if per_obj:
        # Sort: ascending success_rate (worst performers first), then by count.
        def _sort_key(kv: tuple[str, dict[str, int]]) -> tuple[float, int]:
            v = kv[1]
            sr = (v["s"] / v["n"]) if v["n"] else 0.0
            return (sr, -v["n"])

        obj_line = "; ".join(
            f"{name}: {v['s']}/{v['n']}"
            for name, v in sorted(per_obj.items(), key=_sort_key)
        )
    else:
        obj_line = "(no obj_name data)"

    return {
        "n_seeds": n,
        "successes": int(details.get("successes", 0)),
        "success_rate": f"{float(details.get('success_rate', 0.0)):.2f}",
        "avg_score": f"{float(details.get('avg_score', 0.0)):.3f}",
        "failure_histogram": causes_line,
        "obj_breakdown": obj_line,
    }


def _render_cross_seed_prompt(ctx: AgentContext, per_seed_evidence: str) -> str:
    """Render the cross-seed reflection prompt from markdown (or inline fallback)."""
    stats = _cross_seed_stats(ctx)
    variables = {
        "task": ctx.task,
        "n_seeds": str(stats["n_seeds"]),
        "successes": str(stats["successes"]),
        "success_rate": stats["success_rate"],
        "avg_score": stats["avg_score"],
        "failure_histogram": stats["failure_histogram"],
        "obj_breakdown": stats["obj_breakdown"],
        "iteration": str(ctx.iteration),
        "thoughts": ctx.thoughts or "(no thoughts recorded)",
        "code": ctx.code or "",
        "per_seed_evidence": per_seed_evidence,
    }

    pm = getattr(ctx, "prompt_memory", None)
    if pm is not None:
        try:
            return pm.load("system", "cross_seed_reflection", **variables)
        except FileNotFoundError:
            pass

    # Inline fallback (kept tight — canonical copy lives in
    # cap/prompt/system/cross_seed_reflection.md).
    return (
        f"You are analyzing a parallel evaluation: the same code ran on "
        f"{variables['n_seeds']} seeds. "
        f"successes={variables['successes']}/{variables['n_seeds']}, "
        f"avg_score={variables['avg_score']}.\n"
        f"failure histogram: {variables['failure_histogram']}\n"
        f"obj breakdown:     {variables['obj_breakdown']}\n\n"
        f"Task: {variables['task']}\n\n"
        f"Planned approach:\n{variables['thoughts']}\n\n"
        f"Code (iteration {variables['iteration']}):\n"
        f"```python\n{variables['code']}\n```\n\n"
        f"=== Per-seed evidence ===\n{variables['per_seed_evidence']}\n\n"
        "Produce a thorough cross-seed post-mortem with these sections: "
        "(1) Task, (2) Successful seeds (list each), "
        "(3) Failed seeds categorized by cause — "
        "(a) not picked up, (b) placed at wrong position, "
        "(c) arm too close after placement, (d) other — list each failed seed "
        "individually, (4) Dominant failure pattern, "
        "(5) Actionable fixes. When failures cluster by obj_name, recommend "
        "branching on info['obj_name'] with per-object offsets. "
        "Cover every seed; do not abbreviate."
    )


def _reflect_cross_seed(ctx: AgentContext, llm: LLMBackend) -> str | None:
    """Run the cross-seed Phase B synthesis if per-seed data is available.

    Returns the LLM-generated feedback, or ``None`` if no per-seed data is
    present and the caller should fall back to single-seed reflection.
    """
    evidence = _build_per_seed_evidence(ctx)
    if evidence is None:
        return None
    prompt = _render_cross_seed_prompt(ctx, evidence)
    try:
        return llm.generate_text(prompt)
    except Exception as e:
        logger.warning("cross-seed reflection LLM call failed: %s", e)
        return None


def _format_eval(ctx: AgentContext) -> str:
    """Format evaluation details into a compact string."""
    if ctx.evaluation is None:
        return "(no evaluation)"
    d = ctx.evaluation.details
    reward = d.get("reward", 0.0)
    success = d.get("success", False)
    obj_pos = d.get("obj_pos")
    obj_to_eef = d.get("obj_to_robot0_eef_pos")
    lines = [f"reward={reward:.3f}, success={success}"]
    if obj_pos:
        lines.append(f"obj_pos={[round(x, 4) for x in obj_pos]}")
    if obj_to_eef:
        lines.append(f"obj_to_eef_offset={[round(x, 4) for x in obj_to_eef]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_reflection_strategy(
    cfg,
    llm: LLMBackend,
) -> ReflectionStrategy:
    """Build a ReflectionStrategy from Hydra config.

    Reads ``cfg.reflection.strategy``, ``cfg.reflection.cameras``,
    ``cfg.reflection.vlm_backend`` and returns the appropriate strategy.
    """
    cameras = list(cfg.reflection.cameras)
    strategy = cfg.reflection.strategy

    if strategy == "text":
        return TextReflectionStrategy(llm)
    elif strategy == "vision":
        return VisionReflectionStrategy(
            llm,
            cameras=cameras,
            vlm_backend=cfg.reflection.vlm_backend,
        )
    elif strategy == "composite":
        return CompositeReflectionStrategy(
            [
                TextReflectionStrategy(llm),
                VisionReflectionStrategy(
                    llm,
                    cameras=cameras,
                    vlm_backend=cfg.reflection.vlm_backend,
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown reflection strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Skill promotion
# ---------------------------------------------------------------------------


class SkillPromoter:
    """Identifies promotable atomic functions in agent code and appends them to the skill library.

    Called by SelfReflectionStep after the reflection LLM runs.
    A function is promotable if:
    - It's a top-level function definition (not nested)
    - It only calls names that are NOT other agent-defined functions (i.e. calls tools from namespace)
    - It returns a 2-tuple (val, dict) — checked by heuristic: last return has a dict literal or name
    - It was actually called during execution (present in skill logs from this iteration)

    Promotion = append the function source (with @skill decorator added) to skill_library/<base_name>.py
    """

    def __init__(self, skill_library: Any) -> None:
        self._sl = skill_library

    def promote(self, ctx: AgentContext, iteration_skill_logs: list[dict]) -> int:
        """Identify and promote new atomic skills from ctx.code.

        Args:
            ctx: AgentContext with ctx.code = agent-generated code this iteration
            iteration_skill_logs: merged skill_log entries from iter_NNN/skill_log.json
                (list of {name, base_name, version, logs, ...})

        Returns: number of new skills promoted.
        """
        if not ctx.code:
            return 0

        # Names of skills called this iteration (from logs)
        called_names: set[str] = {e.get("name", "") for e in iteration_skill_logs}

        # Names of skills already in library
        existing_names: set[str] = set(self._sl._index.keys())

        # Parse agent code
        try:
            tree = ast.parse(ctx.code)
        except SyntaxError:
            return 0

        # Collect all top-level function names defined in agent code
        agent_fn_names: set[str] = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and _is_top_level(node, tree)
        }

        # Collect names imported from skill_library.* — these are also forbidden
        # to call inside a new skill (no skill calls skill rule)
        imported_skill_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("skill_library"):
                    for alias in node.names:
                        imported_skill_names.add(alias.asname or alias.name)

        forbidden_calls = agent_fn_names | imported_skill_names

        promoted = 0
        source_lines = ctx.code.splitlines()

        logger.info(
            "SkillPromoter: found %d top-level function(s): %s",
            len(agent_fn_names),
            sorted(agent_fn_names),
        )

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not _is_top_level(node, tree):
                continue
            name = node.name
            if name in existing_names:
                logger.info("SkillPromoter: skip %s — already in library", name)
                continue
            from enpire.env.forge.cap.agent.skill_registry import _parse_name

            base_name, _version = _parse_name(name)
            if not _is_atomic(node, forbidden_calls):
                logger.info(
                    "SkillPromoter: skip %s — calls other skill or agent function", name
                )
                continue
            has_return = _has_skill_return(node)
            in_called = name in called_names
            if not in_called and not has_return:
                logger.info(
                    "SkillPromoter: skip %s — not called this iter and no (val, dict) return", name
                )
                continue

            import textwrap

            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            fn_source = textwrap.dedent("\n".join(source_lines[start:end]))
            promoted_source = f"@skill\n{fn_source}"

            self._sl.append_skill(base_name, promoted_source, ctx.iteration)
            promoted += 1
            logger.info(
                "SkillPromoter: promoted %s to skill_library/%s.py", name, base_name
            )

        # Update stats from this iteration's logs
        if iteration_skill_logs:
            self._sl.update_stats_from_logs(iteration_skill_logs)
            self._sl.append_skill_logs(iteration_skill_logs)

        return promoted


def _is_top_level(node: ast.FunctionDef, tree: ast.Module) -> bool:
    """True if *node* binds a name at module scope, even when defensively wrapped.

    Accepts ``def`` at the module root, but also inside module-level ``try/except``,
    ``if/else``, or ``with`` blocks — all of which still execute at import time and
    create a module-level name binding. Rejects functions nested inside another
    ``FunctionDef``, ``AsyncFunctionDef``, ``ClassDef``, or ``Lambda``.
    """
    parent_of: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_of[id(child)] = parent

    cur = parent_of.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.Module):
            return True
        if isinstance(
            cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return False
        cur = parent_of.get(id(cur))
    return False


def _is_atomic(node: ast.FunctionDef, forbidden_calls: set[str]) -> bool:
    """True if the function only calls namespace tools — not other skills or agent functions."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            fn = child.func
            called = None
            if isinstance(fn, ast.Name):
                called = fn.id
            elif isinstance(fn, ast.Attribute):
                called = fn.attr
            if called and called in forbidden_calls and called != node.name:
                return False
    return True


def _has_skill_return(node: ast.FunctionDef) -> bool:
    """Heuristic: function ends with 'return val, {...}' pattern."""
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            v = child.value
            if isinstance(v, ast.Tuple) and len(v.elts) == 2:
                if isinstance(v.elts[1], (ast.Dict, ast.Name)):
                    return True
    return False
