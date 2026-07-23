# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cloud LLM backend using the Anthropic (Claude) SDK."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

from enpire.env.forge.cap.agent.llm.base import LLMBackend
from enpire.env.forge.cap.bridge.system_prompt import build_system_prompt


def _format_tools(tool_schemas: list[dict[str, Any]]) -> str:
    lines = []
    for s in tool_schemas:
        params = s["parameters"]["properties"]
        sig_parts = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "Any")
            if "default" in pinfo:
                sig_parts.append(f"{pname}: {ptype} = {pinfo['default']!r}")
            elif pname not in s["parameters"].get("required", []):
                sig_parts.append(f"{pname}: {ptype} = None")
            else:
                sig_parts.append(f"{pname}: {ptype}")
        sig = ", ".join(sig_parts)
        lines.append(f"def {s['name']}({sig}):")
        lines.append(f'    """{s["description"]}"""')
        lines.append("")
    return "\n".join(lines)


class CloudLLM(LLMBackend):
    """Claude API backend using the ``anthropic`` SDK.

    Accepts an optional ``system_prompt`` — when supplied (via
    :func:`cap.agent.llm.make_llm`), that prompt is sent as the Anthropic
    request's ``system`` field verbatim, so stage-specific prompts
    (``skill_author.md``, ``assembly_generator.md``, etc.) are respected.
    When omitted, the backend falls back to the auto-generated tool-doc
    system prompt built by :func:`cap.bridge.system_prompt.build_system_prompt`.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 8192,
        system_prompt: str | None = None,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or None  # "" → None
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.last_response: str | None = None
        self.last_usage: dict | None = None

    def generate_code(self, task: str, context: dict[str, Any]) -> str:
        import anthropic

        tool_schemas = context.get("tools", [])
        robot_state = context.get("robot_state")
        env_spec = context.get("env_spec")

        iteration = context.get("iteration", 0)
        failure_history = context.get("failure_history", [])

        user_parts: list[str] = []

        # Retry: champion-first layout
        champion = context.get("champion")
        if iteration > 0 and failure_history:
            if champion is not None and champion["score"] > 0.0:
                champ_score = champion["score"]
                champ_iter = champion["iteration"]
                user_parts.append(
                    f"⚠ {iteration} attempt(s) made. BEST SO FAR: iteration {champ_iter + 1} "
                    f"(score={champ_score:.3f}). Your goal: beat that score.\n"
                )
                user_parts.append(f"=== CHAMPION (iteration {champ_iter + 1}, score={champ_score:.3f}) ===")
                if champion.get("code"):
                    user_parts.append(f"```python\n{champion['code']}\n```")
                user_parts.append(f"What still failed:\n{champion['feedback']}\n")
                non_champ = [a for a in failure_history if a["iteration"] != champ_iter]
                if non_champ:
                    user_parts.append("=== SUBSEQUENT ATTEMPTS (all scored WORSE, showing last 3) ===")
                    for attempt in non_champ[-3:]:
                        n = attempt["iteration"]
                        score = attempt.get("score", 0.0)
                        user_parts.append(f"--- Attempt {n + 1} (score={score:.3f}) ---")
                        if attempt.get("code"):
                            snippet = attempt["code"][:800]
                            user_parts.append(f"```python\n{snippet}\n```")
                        user_parts.append(f"Feedback: {attempt.get('feedback', '')[:400]}\n")
            else:
                user_parts.append(
                    f"⚠ CRITICAL: ALL {len(failure_history)} previous attempt(s) FAILED.\n"
                    "You MUST fix the issues below. Do NOT repeat the same mistakes.\n"
                )
                for attempt in failure_history:
                    n = attempt["iteration"]
                    user_parts.append(f"=== ATTEMPT {n + 1} (FAILED) ===")
                    if attempt.get("code"):
                        user_parts.append(f"```python\n{attempt['code']}\n```")
                    user_parts.append(f"Failure analysis:\n{attempt.get('feedback', '(no feedback)')}\n")
            user_parts.append(f"=== TASK ===\n{task}\n")
        else:
            user_parts.append(f"Task: {task}\n")

        if robot_state is not None:
            user_parts.append(
                f"\nCurrent robot state:\n{json.dumps(robot_state, default=str, indent=2)}"
            )

        if iteration > 0 and failure_history:
            if champion is not None and champion["score"] > 0.0:
                user_parts.append(
                    f"\nWrite a Python program (attempt {iteration + 1}) that IMPROVES on the champion "
                    f"(score={champion['score']:.3f}). Address the champion's remaining failures directly. "
                    "Output only executable code, no markdown or explanations."
                )
            else:
                user_parts.append(
                    f"\nWrite a CORRECTED Python program (attempt {iteration + 1}). "
                    "Fix ALL issues from ALL attempts above. "
                    "Output only executable code, no markdown or explanations."
                )
        else:
            user_parts.append(
                "\nWrite a Python program to accomplish the task. "
                "Output only executable code, no markdown or explanations."
            )

        # Prefer an externally-supplied system prompt (e.g. the stage-1 /
        # stage-2 prompt the pipeline built from cap/prompt/system/*.md).
        if self._system_prompt:
            system_prompt = self._system_prompt
        else:
            # Use curated env spec when available, otherwise auto-generate from schemas
            if env_spec:
                tool_docs = env_spec["tool_docs"]
                env_notes = env_spec.get("env_notes", "")
            else:
                tool_docs = (
                    _format_tools(tool_schemas)
                    if tool_schemas
                    else "(No tool documentation provided)"
                )
                env_notes = ""

            system_prompt = build_system_prompt(
                tool_docs,
                env_notes=env_notes,
                raw_code_only=True,
            )

        user_prompt = "\n".join(user_parts)
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        self.last_usage = {
            "model": self._model,
            "input_tokens": getattr(message.usage, "input_tokens", 0),
            "output_tokens": getattr(message.usage, "output_tokens", 0),
        }
        if getattr(message, "stop_reason", None) == "max_tokens":
            log.warning(
                "CloudLLM generate_code: output truncated at max_tokens=%d "
                "(stop_reason=max_tokens). Code may be incomplete.",
                self._max_tokens,
            )
        # Extract text from response
        text = ""
        for block in message.content:
            if block.type == "text":
                text += block.text

        # Record the raw response BEFORE fence stripping so the
        # conversation log preserves exactly what the LLM emitted.
        self.last_response = text.strip()

        # Strip markdown fences if the model wraps the code
        text = text.strip()
        if text.startswith("```python"):
            text = text[len("```python") :].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        return text

    def generate_text(self, prompt: str) -> str:
        """Generate free-form text (not code extraction) via the Anthropic SDK.

        Used by ``SkillAuthorStep`` (which needs the raw multi-section
        response) and by reflection. Passes the configured system prompt
        so the LLM sees stage-specific authoring / reflection rules.
        """
        import anthropic

        self.last_system_prompt = self._system_prompt
        self.last_user_prompt = prompt

        client = anthropic.Anthropic(api_key=self._api_key)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        message = client.messages.create(**kwargs)
        self.last_usage = {
            "model": self._model,
            "input_tokens": getattr(message.usage, "input_tokens", 0),
            "output_tokens": getattr(message.usage, "output_tokens", 0),
        }
        if getattr(message, "stop_reason", None) == "max_tokens":
            log.warning(
                "CloudLLM generate_text: output truncated at max_tokens=%d.",
                self._max_tokens,
            )
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        self.last_response = text
        return text
