# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BridgeLLMBackend — adapts bridge AgentBackend to synchronous LLMBackend.

Wraps ClaudeCodeBackend or OpenAICodexBackend (which spawn CLI subprocesses
and have access to MCP tools) to produce Python code via generate_code().

Usage::

    from enpire.env.forge.cap.bridge.providers import ClaudeCodeBackend
    from enpire.env.forge.cap.agent.llm.bridge_llm import BridgeLLMBackend
    from enpire.env.forge.cap.bridge.providers.base import AgentBridgeConfig, ProviderContext

    llm = BridgeLLMBackend(
        backend=ClaudeCodeBackend(),
        config=AgentBridgeConfig(backend="claude_code", model="claude-opus-4-6"),
        context=ProviderContext(...),
        system_prompt=build_system_prompt(tool_docs),
    )
    code = llm.generate_code(task, context)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from enpire.env.forge.cap.agent.llm.base import LLMBackend
from enpire.env.forge.cap.bridge.providers.base import AgentBackend, AgentBridgeConfig, ChatSession, ProviderContext
from enpire.env.forge.cap.chat.parsing import extract_code_blocks

logger = logging.getLogger(__name__)

_DUMMY_CONV_DIR_NAME = "/tmp/cap_bridge_llm_conv"


class BridgeLLMBackend(LLMBackend):
    """Wraps a bridge AgentBackend (ClaudeCode / OpenAICodex) as a synchronous LLMBackend.

    The bridge backends spawn CLI subprocesses with MCP tool access (camera
    images, robot state), so generated code benefits from richer perception
    during planning.
    """

    def __init__(
        self,
        backend: AgentBackend,
        config: AgentBridgeConfig,
        context: ProviderContext,
        system_prompt: str = "",
        replace_system_prompt: bool = False,
    ) -> None:
        self._backend = backend
        self._config = config
        self._context = context
        self._system_prompt = system_prompt
        self._replace_system_prompt = replace_system_prompt
        import tempfile
        from pathlib import Path
        self._conv_dir = Path(tempfile.mkdtemp(prefix="cap_bridge_llm_"))
        self._session: ChatSession | None = None
        # Populated on every generate_* call so _save_llm_input can archive
        # the exchange to conversations/ for debugging.
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None
        self.last_response: str | None = None

    def _get_session(self) -> ChatSession:
        if self._session is None:
            self._session = ChatSession(conversation_dir=self._conv_dir)
        return self._session

    def generate_text(self, prompt: str) -> str:
        """Run one bridge turn and return the full text response (no code extraction)."""
        # Expose for logging by agent_step / reflection
        self.last_user_prompt = prompt
        self.last_system_prompt = self._system_prompt
        session = self._get_session()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, self._run_turn(prompt, session)).result()
            else:
                result = loop.run_until_complete(self._run_turn(prompt, session))
        except RuntimeError:
            result = asyncio.run(self._run_turn(prompt, session))
        text = result.full_text.strip()
        self.last_response = text
        return text

    def generate_code(self, task: str, context: dict[str, Any]) -> str:
        """Run one bridge turn and extract Python code from the response."""
        prompt = self._build_prompt(task, context)
        # Expose for logging by agent_step (raw message sent to LLM)
        self.last_user_prompt = prompt
        self.last_system_prompt = self._system_prompt
        session = self._get_session()

        # Run async bridge turn synchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an event loop (e.g. FastAPI) — run in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, self._run_turn(prompt, session)).result()
            else:
                result = loop.run_until_complete(self._run_turn(prompt, session))
        except RuntimeError:
            result = asyncio.run(self._run_turn(prompt, session))

        # Always archive the raw response for debugging / log replay.
        self.last_response = result.full_text.strip()

        # Extract first Python code block
        blocks = [b for b in extract_code_blocks(result.full_text) if b.language in ("python", "")]
        if blocks:
            return blocks[0].code.strip()

        # Fallback: if no fences, return the whole response (raw mode)
        logger.warning("BridgeLLMBackend: no python code block found, returning raw response")
        return self.last_response

    async def _run_turn(self, prompt: str, session: ChatSession) -> Any:
        ws_manager = _NullWSManager()
        return await self._backend.run_turn(
            message=prompt,
            session=session,
            ws_manager=ws_manager,
            system_prompt=self._system_prompt,
            config=self._config,
            context=self._context,
            replace_system_prompt=self._replace_system_prompt,
        )

    def _build_prompt(self, task: str, context: dict[str, Any]) -> str:
        import json
        iteration = context.get("iteration", 0)
        failure_history = context.get("failure_history", [])
        config = context.get("config")  # AgentConfig or None

        # Read config templates (or use defaults)
        if config is not None:
            cg = config.code_generator
            retry_cfg = cg.retry
            failure_prefix = retry_cfg.failure_prefix
            retry_thoughts = retry_cfg.thoughts_instruction
            retry_code = retry_cfg.code_instruction
            initial_thoughts = cg.initial_thoughts_instruction
            initial_code = cg.initial_code_instruction
            include_thoughts = retry_cfg.include_previous_thoughts
        else:
            failure_prefix = (
                "⚠ CRITICAL: ALL previous attempts FAILED. "
                "You MUST fix the issues below. Do NOT repeat the same mistakes.\n"
            )
            retry_thoughts = (
                "First, write THOUGHTS: explaining your strategy, what you're changing "
                "from previous attempts, and what you expect to happen step by step."
            )
            retry_code = "Then write the code in a single ```python block."
            initial_thoughts = (
                "First, write THOUGHTS: explaining your strategy to solve this task — "
                "what steps you'll take, what positions you'll target, and what you expect to happen."
            )
            initial_code = "Then write the executable Python code in a single ```python block."
            include_thoughts = True

        # --- Retry prompt (iteration > 0): champion-first layout ---
        if iteration > 0 and failure_history:
            max_chars = (
                config.code_generator.retry.max_history_chars if config else 12000
            )
            skill_library_index = context.get("skill_library_index", "")
            champion = context.get("champion")  # best-scoring attempt so far

            parts: list[str] = []
            if skill_library_index is not None:
                index_body = skill_library_index if skill_library_index else "(empty — no skills promoted yet)"
                parts.append(f"=== CURRENT SKILL LIBRARY ===\n{index_body}\n=== END SKILL LIBRARY ===\n")

            if champion is not None and champion["score"] > 0.0:
                # Champion-first: show the best attempt as the baseline to improve,
                # then show what was tried since (briefly) so the LLM avoids
                # repeating those directions.
                champ_score = champion["score"]
                champ_iter = champion["iteration"]
                parts.append(
                    f"⚠ {iteration} attempt(s) made. BEST SO FAR: iteration {champ_iter + 1} "
                    f"(score={champ_score:.3f}). Your goal: beat that score.\n"
                )
                parts.append(f"=== CHAMPION (iteration {champ_iter + 1}, score={champ_score:.3f}) ===")
                if champion.get("thoughts") and include_thoughts:
                    parts.append(f"Strategy:\n{champion['thoughts']}\n")
                if champion.get("code"):
                    parts.append(f"```python\n{champion['code']}\n```")
                parts.append(f"What still failed (champion feedback):\n{champion['feedback']}\n")

                # Show the most recent non-champion attempts briefly (last 3)
                non_champ = [a for a in failure_history if a["iteration"] != champ_iter]
                if non_champ:
                    shown = non_champ[-3:]
                    parts.append(
                        f"=== SUBSEQUENT ATTEMPTS (all scored WORSE than champion, showing last {len(shown)}) ==="
                    )
                    remaining_budget = max_chars - sum(len(p) for p in parts)
                    per_attempt = max(600, remaining_budget // len(shown))
                    for attempt in shown:
                        n = attempt["iteration"]
                        score = attempt.get("score", 0.0)
                        entry = [f"--- Attempt {n + 1} (score={score:.3f}) ---"]
                        code = attempt.get("code", "")
                        if code:
                            snippet = code[:800] + ("\n... (truncated)" if len(code) > 800 else "")
                            entry.append(f"```python\n{snippet}\n```")
                        fb = attempt.get("feedback", "")
                        if fb:
                            entry.append(f"Feedback: {fb[:400]}")
                        block = "\n".join(entry)
                        parts.append(block[:per_attempt])
            else:
                # No champion with score > 0 yet — use the standard failure-list layout.
                parts.append(f"{failure_prefix}\n")
                history_entries: list[str] = []
                for attempt in failure_history:
                    n = attempt["iteration"]
                    entry_parts = [f"=== ATTEMPT {n + 1} (FAILED) ==="]
                    thoughts = attempt.get("thoughts", "")
                    if thoughts and include_thoughts:
                        entry_parts.append(f"Planned approach:\n{thoughts}\n")
                    code = attempt.get("code", "")
                    if code:
                        entry_parts.append(f"```python\n{code}\n```")
                    stdout = attempt.get("stdout", "")
                    if stdout:
                        stdout_lines = [
                            l for l in stdout.splitlines()
                            if not l.strip().startswith("[profile")
                            and not l.strip().startswith("[Client]")
                        ]
                        trimmed = "\n".join(stdout_lines).strip()
                        if trimmed:
                            entry_parts.append(f"Execution output:\n{trimmed[:1000]}\n")
                    feedback = attempt.get("feedback", "(no feedback)")
                    entry_parts.append(f"Failure analysis (planned vs actual):\n{feedback}\n")
                    history_entries.append("\n".join(entry_parts))

                total = sum(len(e) for e in history_entries)
                while total > max_chars and len(history_entries) > 1:
                    total -= len(history_entries[0])
                    history_entries.pop(0)
                if total > max_chars and history_entries:
                    parts.append(f"(earlier {len(failure_history) - len(history_entries)} attempt(s) omitted for brevity)\n")
                for entry in history_entries:
                    parts.append(entry)

            parts.append(f"=== TASK ===\n{task}")

            robot_state = context.get("robot_state")
            if robot_state:
                parts.append(f"\nCurrent robot state:\n{json.dumps(robot_state, default=str, indent=2)}")

            task_info = context.get("task_info")
            if task_info:
                parts.append(f"\nTask info:\n{json.dumps(task_info, default=str, indent=2)}")

            if champion is not None and champion["score"] > 0.0:
                champ_score = champion["score"]
                parts.append(
                    f"\nWrite a Python program (attempt {iteration + 1}) that IMPROVES on the champion "
                    f"(score={champ_score:.3f}). Address the champion's remaining failures directly.\n\n"
                    f"{retry_thoughts}\n{retry_code}"
                )
            else:
                parts.append(
                    f"\nWrite a CORRECTED Python program (attempt {iteration + 1}) "
                    f"that fixes ALL issues from ALL attempts above.\n\n"
                    f"{retry_thoughts}\n{retry_code}"
                )
            return "\n".join(parts)

        # --- First iteration: normal prompt ---
        parts = [f"Task: {task}"]

        skill_library_index = context.get("skill_library_index", "")
        if skill_library_index is not None:  # None means no library; "" means empty library
            index_body = skill_library_index if skill_library_index else "(empty — no skills promoted yet)"
            parts.append(f"\n=== CURRENT SKILL LIBRARY ===\n{index_body}\n=== END SKILL LIBRARY ===")

        robot_state = context.get("robot_state")
        if robot_state:
            parts.append(f"\nCurrent robot state:\n{json.dumps(robot_state, default=str, indent=2)}")

        task_info = context.get("task_info")
        if task_info:
            parts.append(f"\nTask info:\n{json.dumps(task_info, default=str, indent=2)}")

        parts.append(f"\n{initial_thoughts}\n{initial_code}")
        return "\n".join(parts)


class _NullWSManager:
    """No-op WebSocket manager for headless bridge turns."""

    async def broadcast(self, event_type: str, data: Any) -> None:
        pass
