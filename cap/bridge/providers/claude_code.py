"""Claude Code backend for the generic CAP agent bridge."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .base import (
    AgentBackend,
    AgentBackendSpec,
    AgentBridgeConfig,
    ChatSession,
    ProviderContext,
    TurnResult,
)

logger = logging.getLogger(__name__)

_ALLOWED_TOOLS = (
    "mcp__cap-robot__get_robot_state,"
    "mcp__cap-robot__get_camera_image,"
    "mcp__cap-robot__list_saved_scripts,"
    "mcp__cap-robot__read_saved_script,"
    "mcp__cap-robot__vlm_query"
)


class ClaudeCodeBackend(AgentBackend):
    spec = AgentBackendSpec(
        backend="claude_code",
        label="Claude Code",
        models=("claude-opus-4-6", "claude-sonnet-4-6"),
        default_model="claude-opus-4-6",
        reasoning_options=("low", "medium", "high", "max"),
        default_reasoning="medium",
    )

    def build_command(
        self,
        *,
        message: str,
        session_id: str | None,
        system_prompt: str,
        config: AgentBridgeConfig,
        context: ProviderContext,
        replace_system_prompt: bool = False,
    ) -> list[str]:
        model, reasoning = self.spec.normalize(
            model=config.model, reasoning=config.reasoning
        )
        cmd = ["claude", "-p", message, "--output-format", "stream-json", "--verbose"]
        if context.mcp_config_path.exists():
            cmd.extend(["--mcp-config", str(context.mcp_config_path)])
        # --system-prompt replaces the default (skips CLAUDE.md);
        # --append-system-prompt adds to it (inherits CLAUDE.md rules).
        if replace_system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        else:
            cmd.extend(["--append-system-prompt", system_prompt])
        if model:
            cmd.extend(["--model", model])
        if reasoning:
            cmd.extend(["--effort", reasoning])
        if session_id:
            cmd.extend(["--resume", session_id])
        cmd.extend(["--allowedTools", _ALLOWED_TOOLS])
        return cmd

    async def run_turn(
        self,
        *,
        message: str,
        session: ChatSession,
        ws_manager: Any,
        system_prompt: str,
        config: AgentBridgeConfig,
        context: ProviderContext,
        replace_system_prompt: bool = False,
    ) -> TurnResult:
        cmd = self.build_command(
            message=message,
            session_id=session.session_id,
            system_prompt=system_prompt,
            config=config,
            context=context,
            replace_system_prompt=replace_system_prompt,
        )

        total_arg_bytes = sum(len(a.encode()) for a in cmd)
        logger.info("Spawning claude: %s... (arg bytes: %d)", " ".join(cmd[:6]), total_arg_bytes)
        if total_arg_bytes > 1_500_000:
            logger.warning("claude arg list is %d bytes — approaching ARG_MAX", total_arg_bytes)
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(context.project_root),
            )
        except OSError as exc:
            import errno
            if exc.errno == errno.E2BIG:
                await ws_manager.broadcast(
                    "chat_error",
                    {"error": f"System prompt too large ({total_arg_bytes // 1024}KB) — exceeds OS ARG_MAX. "
                               "Reduce the number of saved scripts or prompt files."},
                )
                return TurnResult(full_text="", session_id=session.session_id)
            raise
        except FileNotFoundError:
            await ws_manager.broadcast(
                "chat_error",
                {
                    "error": "claude CLI not found. Install it: npm install -g @anthropic-ai/claude-code"
                },
            )
            return TurnResult(full_text="", session_id=session.session_id)

        full_text = ""
        if process.stdout is None:
            raise RuntimeError(
                "claude subprocess stdout is None — check Popen configuration"
            )

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue
            try:
                event = json.loads(line_str)
            except json.JSONDecodeError:
                logger.debug("Non-JSON line: %s", line_str[:200])
                continue

            event_type = event.get("type", "")
            if event_type == "assistant":
                message_data = event.get("message", {})
                session_id = message_data.get("session_id") or event.get("session_id")
                if session_id:
                    session.session_id = session_id
                for block in message_data.get("content", []):
                    if block.get("type") == "tool_use":
                        tool_name = block.get("name", "")
                        tool_input = block.get("input", {})
                        session.log_tool_use(tool_name, tool_input)
                        await ws_manager.broadcast(
                            "chat_tool_use",
                            {"tool": tool_name, "input": tool_input},
                        )
            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    full_text += text
                    await ws_manager.broadcast("chat_text_delta", {"text": text})
            elif event_type == "result":
                session_id = event.get("session_id")
                if session_id:
                    session.session_id = session_id
                result_text = event.get("result", "")
                if result_text and not full_text:
                    full_text = result_text
                    await ws_manager.broadcast("chat_text_delta", {"text": result_text})

        await process.wait()
        logger.info("Claude process exited with code %s", process.returncode)
        if process.stderr:
            stderr = await process.stderr.read()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()
            if stderr_str:
                logger.info("claude stderr: %s", stderr_str[:500])
                if process.returncode != 0:
                    print(f"  [bridge:claude_code] ERROR (exit={process.returncode}): {stderr_str[:200]}")
                    await ws_manager.broadcast("chat_error", {"error": stderr_str})

        return TurnResult(full_text=full_text, session_id=session.session_id)
