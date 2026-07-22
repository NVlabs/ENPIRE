# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA inference-gateway LLM backend.

Thin ``LLMBackend`` over NVIDIA's multi-provider inference gateway
(``https://inference-api.nvidia.com/v1/``). Uses direct HTTPS requests to the
chat-completions endpoint — OSMO containers can reach the gateway, so no
proxy is involved.

The gateway multiplexes providers via the ``model`` field, e.g.::

    gcp/google/gemini-3.1-pro-preview
    azure/openai/gpt-5.1
    azure/openai/gpt-5.1-codex
    aws/anthropic/claude-opus-4-5

Auth: ``NVIDIA_API_KEY`` env var (or ``api_key=`` kwarg).

Mirrors :class:`cap.agent.llm.cloud.CloudLLM`'s prompting pipeline — the
stage-specific system prompt (skill_author, assembly_generator, reflection…)
is passed straight through so downstream behavior is identical.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from enpire.env.forge.cap.agent.llm.base import LLMBackend
from enpire.env.forge.cap.agent.providers.nvidia import (
    DEFAULT_BASE_URL,
    NvidiaRateLimitError,
    list_nvidia_keys,
    nvidia_model_supports_temperature,
    post_chat_completions,
    response_text,
    usage_dict,
)
from enpire.env.forge.cap.bridge.system_prompt import build_system_prompt

log = logging.getLogger(__name__)

DEFAULT_MODEL = "aws/anthropic/bedrock-claude-opus-4-6"
DEFAULT_TRANSIENT_RETRIES = 3


def _transient_retry_count() -> int:
    raw = os.environ.get("CAP_NVIDIA_TRANSIENT_RETRIES")
    if not raw:
        return DEFAULT_TRANSIENT_RETRIES
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_TRANSIENT_RETRIES


def _is_transient_nvidia_exception(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    text = str(exc).lower()
    transient_markers = (
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "remote end closed",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    )
    return any(marker in text for marker in transient_markers)


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


class NvidiaLLM(LLMBackend):
    """LLM backend that targets NVIDIA's inference gateway directly.

    Requests are sent as direct HTTPS ``POST`` calls to NVIDIA's
    ``/chat/completions`` endpoint. Model strings follow NVIDIA's
    ``<cloud>/<provider>/<model>`` convention (see module docstring).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model
        self._explicit_api_key = api_key  # if set, always use this key (no rotation)
        self._base_url = base_url
        self._max_tokens = max_tokens  # None → let the gateway use its default
        self._temperature = temperature
        self._system_prompt = system_prompt or None
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.last_response: str | None = None
        self.last_usage: dict | None = None

    def _post_chat_completions(
        self, payload: dict[str, Any], api_key: str | None = None
    ) -> dict[str, Any]:
        return post_chat_completions(
            payload,
            api_key=api_key or self._explicit_api_key,
            base_url=self._base_url,
            telemetry_source="llm",
        )

    def _create_with_retry(self, messages: list[dict[str, Any]]) -> Any:
        """Call chat completions, rotating keys on 429 and retrying transients."""

        kwargs = self._create_kwargs(messages)
        n_keys = 1 if self._explicit_api_key else max(len(list_nvidia_keys()), 1)
        max_transient_retries = _transient_retry_count()
        last_exc: Exception | None = None
        rate_limited = 0
        transient_errors = 0
        while rate_limited < n_keys and transient_errors < max_transient_retries:
            try:
                resp = self._post_chat_completions(kwargs)
                choices = resp.get("choices")
                finish_reason = None
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    finish_reason = choices[0].get("finish_reason")
                if finish_reason == "length":
                    log.warning(
                        "NvidiaLLM: output truncated (finish_reason=length). "
                        "Set max_tokens higher or the code may be incomplete."
                    )
                return resp
            except NvidiaRateLimitError as exc:
                last_exc = exc
                rate_limited += 1
                if rate_limited >= n_keys:
                    break
                if n_keys > 1:
                    log.warning(
                        "NVIDIA 429 on attempt %d/%d; rotating key",
                        rate_limited,
                        n_keys,
                    )
                else:
                    log.warning("NVIDIA 429 on attempt %d/%d", rate_limited, n_keys)
            except Exception as exc:
                if not _is_transient_nvidia_exception(exc):
                    raise
                last_exc = exc
                transient_errors += 1
                if transient_errors >= max_transient_retries:
                    raise RuntimeError(
                        f"NVIDIA chat completions failed after "
                        f"{transient_errors} transient attempt(s): {exc}"
                    ) from exc
                sleep_s = min(2 ** (transient_errors - 1), 8)
                log.warning(
                    "NVIDIA transient error on attempt %d/%d: %s; retrying in %.1fs",
                    transient_errors,
                    max_transient_retries,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)
        if rate_limited >= n_keys:
            raise RuntimeError(f"All {n_keys} NVIDIA key(s) rate-limited") from last_exc
        raise RuntimeError(
            f"NVIDIA chat completions failed after {transient_errors} transient attempt(s)"
        ) from last_exc

    def _create_kwargs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if nvidia_model_supports_temperature(self._model):
            kwargs["temperature"] = self._temperature
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        return kwargs

    def generate_code(self, task: str, context: dict[str, Any]) -> str:
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
                            user_parts.append(f"```python\n{attempt['code'][:800]}\n```")
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

        response = self._create_with_retry(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        self.last_usage = usage_dict(response, model=self._model)
        raw = response_text(response).strip()
        # Record raw response BEFORE fence-stripping so the conversation log
        # preserves exactly what the LLM emitted.
        self.last_response = raw
        return _strip_code_fences(raw)

    def generate_text(self, prompt: str) -> str:
        self.last_system_prompt = self._system_prompt
        self.last_user_prompt = prompt

        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._create_with_retry(messages)
        self.last_usage = usage_dict(response, model=self._model)
        text = response_text(response).strip()
        self.last_response = text
        return text
