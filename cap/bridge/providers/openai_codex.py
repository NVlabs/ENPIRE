"""OpenAI Codex backend for the generic CAP agent bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
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

_SESSION_KEYS = ("session_id", "conversation_id", "thread_id")
_TOOL_TYPES = {"tool_use", "tool_call", "function_call", "mcp_tool_call"}


def _iter_nodes(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_nodes(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_nodes(item)


def _extract_session_id(event: dict[str, Any]) -> str | None:
    for node in _iter_nodes(event):
        for key in _SESSION_KEYS:
            value = node.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _extract_text_delta(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", "")).lower()
    if event_type == "response.output_text.delta" and isinstance(
        event.get("delta"), str
    ):
        return event["delta"]
    if event_type == "content_block_delta":
        delta = event.get("delta", {})
        if isinstance(delta, dict) and delta.get("type") == "text_delta":
            return str(delta.get("text", ""))
    if "text" in event_type and isinstance(event.get("text"), str):
        return event["text"]
    delta = event.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("text"), str):
        return delta["text"]
    return ""


def _extract_final_text(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", "")).lower()
    if event_type == "item.completed":
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str):
                return text
    return ""


def _extract_tool_uses(event: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    results: list[tuple[str, dict[str, Any]]] = []
    for node in _iter_nodes(event):
        node_type = str(node.get("type", "")).lower()
        if node_type not in _TOOL_TYPES:
            continue
        name = node.get("name")
        tool_input = node.get("input", {})
        if isinstance(name, str):
            results.append((name, tool_input if isinstance(tool_input, dict) else {}))
    return results


class OpenAICodexBackend(AgentBackend):
    spec = AgentBackendSpec(
        backend="openai_codex",
        label="OpenAI Codex",
        models=(
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex",
            "gpt-5.2",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex-mini",
        ),
        default_model="gpt-5.4",
        reasoning_options=("low", "medium", "high", "xhigh"),
        default_reasoning="low",
    )

    def build_command(
        self,
        *,
        message: str,
        session_id: str | None,
        config: AgentBridgeConfig,
        context: ProviderContext,
        output_file: Path,
    ) -> list[str]:
        model, reasoning = self.spec.normalize(
            model=config.model, reasoning=config.reasoning
        )
        common_config = [
            "-c",
            'mcp_servers.cap-robot.command="uv"',
            "-c",
            'mcp_servers.cap-robot.args=["run","cap/bridge/cap_mcp_server.py"]',
            "-c",
            f'mcp_servers.cap-robot.env.CAP_AGENT_URL="{context.cap_agent_url}"',
        ]
        if model:
            common_config.extend(["-m", model])
        if reasoning:
            common_config.extend(["-c", f'model_reasoning_effort="{reasoning}"'])

        if session_id:
            return [
                "codex",
                "exec",
                "resume",
                "--json",
                "-o",
                str(output_file),
                *common_config,
                session_id,
                message,
            ]
        else:
            return [
                "codex",
                "exec",
                "--json",
                "--color",
                "never",
                "-C",
                str(context.project_root),
                "--output-last-message",
                str(output_file),
                "-s",
                "read-only",
                *common_config,
                message,
            ]

    async def run_turn(
        self,
        *,
        message: str,
        session: ChatSession,
        ws_manager: Any,
        system_prompt: str,
        config: AgentBridgeConfig,
        context: ProviderContext,
    ) -> TurnResult:
        wrapped_message = (
            f"[System instructions]\n\n{system_prompt}\n\n---\n\n"
            f"[User request]\n\n{message}"
        )
        full_text = ""
        saw_text_delta = False
        fallback_lines: list[str] = []

        with tempfile.TemporaryDirectory(prefix="cap-codex-bridge-") as tmpdir:
            output_file = Path(tmpdir) / "last_message.txt"
            cmd = self.build_command(
                message=wrapped_message,
                session_id=session.session_id,
                config=config,
                context=context,
                output_file=output_file,
            )
            logger.info("Spawning codex: %s...", " ".join(cmd[:8]))
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(context.project_root),
                )
            except FileNotFoundError:
                await ws_manager.broadcast(
                    "chat_error",
                    {
                        "error": "codex CLI not found. Install it: npm install -g @openai/codex"
                    },
                )
                return TurnResult(full_text="", session_id=session.session_id)

            if process.stdout is None:
                raise RuntimeError(
                    "codex subprocess stdout is None — check Popen configuration"
                )
            if process.stderr is None:
                raise RuntimeError(
                    "codex subprocess stderr is None — check Popen configuration"
                )

            stderr_lines: list[str] = []

            async def _drain_stderr() -> None:
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue
                    stderr_lines.append(line_str)
                    logger.info("codex stderr: %s", line_str[:500])

            stderr_task = asyncio.create_task(_drain_stderr())

            try:
                while True:
                    try:
                        line = await asyncio.wait_for(
                            process.stdout.readline(), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        if process.returncode is not None:
                            break
                        continue

                    if not line:
                        break
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue
                    try:
                        event = json.loads(line_str)
                    except json.JSONDecodeError:
                        fallback_lines.append(line_str)
                        continue

                    maybe_session_id = _extract_session_id(event)
                    if maybe_session_id:
                        session.session_id = maybe_session_id

                    for tool_name, tool_input in _extract_tool_uses(event):
                        session.log_tool_use(tool_name, tool_input)
                        await ws_manager.broadcast(
                            "chat_tool_use",
                            {"tool": tool_name, "input": tool_input},
                        )

                    delta_text = _extract_text_delta(event)
                    if delta_text:
                        saw_text_delta = True
                        full_text += delta_text
                        await ws_manager.broadcast("chat_text_delta", {"text": delta_text})
                        continue

                    final_text = _extract_final_text(event)
                    if final_text and not saw_text_delta and not full_text:
                        full_text = final_text
                        await ws_manager.broadcast(
                            "chat_text_delta", {"text": final_text}
                        )
            finally:
                await process.wait()
                await stderr_task

            logger.info("Codex process exited with code %s", process.returncode)

            if not full_text and output_file.exists():
                full_text = output_file.read_text(encoding="utf-8").strip()
                if full_text:
                    await ws_manager.broadcast("chat_text_delta", {"text": full_text})

            if not full_text and fallback_lines:
                full_text = "\n".join(fallback_lines).strip()
                if full_text:
                    await ws_manager.broadcast("chat_text_delta", {"text": full_text})

            stderr_str = "\n".join(stderr_lines).strip()
            if stderr_str and process.returncode != 0 and not full_text:
                event = "chat_error" if process.returncode != 0 else "chat_warning"
                await ws_manager.broadcast(event, {"error": stderr_str})

        return TurnResult(full_text=full_text, session_id=session.session_id)
