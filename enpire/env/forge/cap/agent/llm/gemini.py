"""Gemini LLM backend using the google-genai SDK.

Direct access to Google's Gemini models via ``generativelanguage.googleapis.com``.
Auth: ``GEMINI_API_KEY`` env var (or ``api_key=`` kwarg).

Mirrors :class:`cap.agent.llm.cloud.CloudLLM`'s prompting pipeline so
stage-specific system prompts (skill_author, assembly_generator, …) are
passed through identically.
"""

from __future__ import annotations

import json
import os
from typing import Any

from enpire.env.forge.cap.agent.llm.base import LLMBackend
from enpire.env.forge.cap.bridge.system_prompt import build_system_prompt

DEFAULT_MODEL = "gemini-2.5-pro"


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


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python") :].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


class GeminiLLM(LLMBackend):
    """LLM backend that talks directly to Google Gemini via ``google-genai``.

    Uses the same ``system_instruction`` / user-message pattern as the VLM
    backend in ``cap.agent.tools.vlm.backends.gemini``.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or None
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.last_response: str | None = None
        self.last_usage: dict | None = None
        self._client_instance = None

    def _client(self):
        if self._client_instance is None:
            from google import genai

            self._client_instance = genai.Client(api_key=self._api_key)
        return self._client_instance

    def generate_code(self, task: str, context: dict[str, Any]) -> str:
        from google.genai import types

        tool_schemas = context.get("tools", [])
        robot_state = context.get("robot_state")
        env_spec = context.get("env_spec")
        iteration = context.get("iteration", 0)
        failure_history = context.get("failure_history", [])

        user_parts: list[str] = []
        champion = context.get("champion")
        if iteration > 0 and failure_history:
            if champion is not None and champion["score"] > 0.0:
                champ_score = champion["score"]
                champ_iter = champion["iteration"]
                user_parts.append(
                    f"⚠ {iteration} attempt(s) made. BEST SO FAR: iteration {champ_iter + 1} "
                    f"(score={champ_score:.3f}). Your goal: beat that score.\n"
                )
                user_parts.append(
                    f"=== CHAMPION (iteration {champ_iter + 1}, score={champ_score:.3f}) ==="
                )
                if champion.get("code"):
                    user_parts.append(f"```python\n{champion['code']}\n```")
                user_parts.append(f"What still failed:\n{champion['feedback']}\n")
                non_champ = [a for a in failure_history if a["iteration"] != champ_iter]
                if non_champ:
                    user_parts.append(
                        "=== SUBSEQUENT ATTEMPTS (all scored WORSE, showing last 3) ==="
                    )
                    for attempt in non_champ[-3:]:
                        n = attempt["iteration"]
                        score = attempt.get("score", 0.0)
                        user_parts.append(
                            f"--- Attempt {n + 1} (score={score:.3f}) ---"
                        )
                        if attempt.get("code"):
                            user_parts.append(
                                f"```python\n{attempt['code'][:800]}\n```"
                            )
                        user_parts.append(
                            f"Feedback: {attempt.get('feedback', '')[:400]}\n"
                        )
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
                    user_parts.append(
                        f"Failure analysis:\n{attempt.get('feedback', '(no feedback)')}\n"
                    )
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

        if self._system_prompt:
            system_prompt = self._system_prompt
        else:
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
                tool_docs, env_notes=env_notes, raw_code_only=True
            )

        user_prompt = "\n".join(user_parts)
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt

        response = self._client().models.generate_content(
            model=self._model,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            ),
        )
        um = getattr(response, "usage_metadata", None)
        self.last_usage = {
            "model": self._model,
            "input_tokens": int(getattr(um, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
        }
        raw = (response.text or "").strip()
        self.last_response = raw
        return _strip_code_fences(raw)

    def generate_text(self, prompt: str) -> str:
        from google.genai import types

        self.last_system_prompt = self._system_prompt
        self.last_user_prompt = prompt

        config = types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )
        if self._system_prompt:
            config.system_instruction = self._system_prompt

        response = self._client().models.generate_content(
            model=self._model,
            contents=[prompt],
            config=config,
        )
        um = getattr(response, "usage_metadata", None)
        self.last_usage = {
            "model": self._model,
            "input_tokens": int(getattr(um, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
        }
        text = (response.text or "").strip()
        self.last_response = text
        return text
