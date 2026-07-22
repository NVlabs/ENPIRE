# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from enpire.env.forge.cap.voice import ChatVoiceController

from .parsing import extract_code_blocks

logger = logging.getLogger(__name__)

MAX_CHAT_SESSION_MESSAGES = 40


class ChatRequest(BaseModel):
    message: str


class OkResponse(BaseModel):
    ok: bool


class VoiceRequest(BaseModel):
    text: str


class VoiceEnabledRequest(BaseModel):
    enabled: bool


@dataclass
class ChatSession:
    session_id: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    log_path: Path | None = None

    def append_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > MAX_CHAT_SESSION_MESSAGES:
            del self.messages[:-MAX_CHAT_SESSION_MESSAGES]

    def _ensure_log(self, conversation_dir: Path) -> Path:
        if self.log_path is None:
            conversation_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.log_path = conversation_dir / f"{ts}.md"
            self.log_path.write_text(f"# Conversation {ts}\n\n", encoding="utf-8")
            logger.info("Conversation log: %s", self.log_path)
        return self.log_path

    def log_user(self, conversation_dir: Path, message: str) -> None:
        path = self._ensure_log(conversation_dir)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"## User\n\n{message}\n\n")

    def log_assistant(self, conversation_dir: Path, text: str) -> None:
        path = self._ensure_log(conversation_dir)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"## Assistant\n\n{text}\n\n---\n\n")

    def log_tool_use(self, conversation_dir: Path, tool: str, tool_input: dict[str, Any]) -> None:
        path = self._ensure_log(conversation_dir)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"> **MCP Tool:** `{tool}({json.dumps(tool_input, default=str)})`\n\n")


class BridgeWSManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, event_type: str, data: Any) -> None:
        payload = json.dumps(
            {
                "type": event_type,
                "data": data,
                "timestamp": time.strftime("%H:%M:%S"),
            }
        )
        stale: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._connections.remove(ws)


class ChatTurnSink(Protocol):
    async def set_session_id(self, session_id: str | None) -> None: ...

    async def emit_text(self, text: str) -> None: ...

    async def emit_tool_use(self, tool: str, tool_input: dict[str, Any]) -> None: ...


class ChatBackend(Protocol):
    name: str

    async def run_turn(self, message: str, session: ChatSession, sink: ChatTurnSink) -> str: ...


class _BridgeTurnSink:
    def __init__(
        self,
        *,
        session: ChatSession,
        ws_manager: BridgeWSManager,
        conversation_dir: Path,
        chat_voice: ChatVoiceController | None,
    ) -> None:
        self._session = session
        self._ws_manager = ws_manager
        self._conversation_dir = conversation_dir
        self._chat_voice = chat_voice
        self.full_text = ""

    async def set_session_id(self, session_id: str | None) -> None:
        if session_id:
            self._session.session_id = session_id

    async def emit_text(self, text: str) -> None:
        if not text:
            return
        self.full_text += text
        if self._chat_voice is not None:
            self._chat_voice.on_text_delta(text)
        await self._ws_manager.broadcast("chat_text_delta", {"text": text})

    async def emit_tool_use(self, tool: str, tool_input: dict[str, Any]) -> None:
        self._session.log_tool_use(self._conversation_dir, tool, tool_input)
        await self._ws_manager.broadcast(
            "chat_tool_use",
            {"tool": tool, "input": tool_input},
        )


async def _run_backend_turn(
    *,
    backend: ChatBackend,
    message: str,
    session: ChatSession,
    ws_manager: BridgeWSManager,
    conversation_dir: Path,
    chat_voice: ChatVoiceController | None,
) -> None:
    sink = _BridgeTurnSink(
        session=session,
        ws_manager=ws_manager,
        conversation_dir=conversation_dir,
        chat_voice=chat_voice,
    )
    full_text = await backend.run_turn(message, session, sink)
    if full_text and not sink.full_text:
        await sink.emit_text(full_text)
    final_text = sink.full_text or full_text

    for block in extract_code_blocks(final_text):
        await ws_manager.broadcast(
            "chat_code_block",
            {
                "code": block.code,
                "language": block.language,
            },
        )

    logger.info(
        "%s response (%s chars):\n%s%s",
        backend.name,
        len(final_text),
        final_text[:500],
        "..." if len(final_text) > 500 else "",
    )

    session.log_user(conversation_dir, message)
    session.log_assistant(conversation_dir, final_text)
    session.append_message("user", message)
    session.append_message("assistant", final_text)

    if chat_voice is not None:
        try:
            chat_voice.on_turn_complete(final_text)
        except Exception:
            logger.exception("Failed to queue assistant text for voice output")

    await ws_manager.broadcast(
        "chat_turn_complete",
        {"session_id": session.session_id},
    )


def _render_prompt_sections(prompt_files: list[Path]) -> str:
    sections = []
    for path in prompt_files:
        content = path.read_text(encoding="utf-8").strip()
        sections.append(f"### {path.stem}\n\n{content}")
    return "\n\n".join(sections)


def _build_full_message(message: str, prompt_dir: Path, per_message_dir: Path) -> str:
    full_message = message
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_files = sorted(prompt_dir.glob("*.md"))
    if prompt_files:
        all_prompts = _render_prompt_sections(prompt_files)
        full_message = (
            "[Task Context — all prompts]\n\n"
            f"{all_prompts}\n\n---\n\nUser request: {message}"
        )

    per_message_files = sorted(per_message_dir.glob("*.md")) if per_message_dir.exists() else []
    if per_message_files:
        full_message += "\n\n---\n\n" + _render_prompt_sections(per_message_files)
    return full_message


def create_chat_app(
    backend: ChatBackend,
    *,
    title: str,
    prompt_dir: Path,
    per_message_dir: Path,
    conversation_dir: Path,
    chat_voice: ChatVoiceController | None = None,
) -> FastAPI:
    chat_voice = chat_voice or ChatVoiceController()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            chat_voice.shutdown()

    app = FastAPI(title=title, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws_manager = BridgeWSManager()
    session = ChatSession()
    _generating = False

    @app.post("/api/chat", response_model=OkResponse)
    async def chat(req: ChatRequest) -> OkResponse:
        nonlocal _generating
        if _generating:
            return OkResponse(ok=False)

        _generating = True
        full_message = _build_full_message(req.message, prompt_dir, per_message_dir)
        logger.info(
            "Chat message for %s: %s%s",
            backend.name,
            full_message[:200],
            "..." if len(full_message) > 200 else "",
        )

        async def _run() -> None:
            nonlocal _generating
            try:
                await _run_backend_turn(
                    backend=backend,
                    message=full_message,
                    session=session,
                    ws_manager=ws_manager,
                    conversation_dir=conversation_dir,
                    chat_voice=chat_voice,
                )
            except Exception as exc:
                logger.exception("%s turn failed", backend.name)
                chat_voice.on_turn_error()
                await ws_manager.broadcast("chat_error", {"error": str(exc)})
            finally:
                _generating = False

        asyncio.create_task(_run())
        return OkResponse(ok=True)

    @app.post("/api/chat/reset", response_model=OkResponse)
    async def reset_chat() -> OkResponse:
        nonlocal _generating
        session.session_id = None
        session.messages.clear()
        session.log_path = None
        _generating = False
        await ws_manager.broadcast("chat_turn_complete", {"session_id": None})
        return OkResponse(ok=True)

    @app.get("/api/chat/status")
    async def chat_status() -> dict[str, Any]:
        return {
            "backend": backend.name,
            "generating": _generating,
            "session_id": session.session_id,
            "message_count": len(session.messages),
        }

    @app.get("/api/voice/status")
    async def voice_status() -> dict[str, Any]:
        status = chat_voice.get_status()
        return {
            "enabled": status.enabled,
            "speaking": status.speaking,
            "last_spoken_text": status.last_spoken_text,
            "last_error": status.last_error,
        }

    @app.post("/api/voice/enabled", response_model=OkResponse)
    async def set_voice_enabled(req: VoiceEnabledRequest) -> OkResponse:
        chat_voice.set_enabled(req.enabled)
        return OkResponse(ok=True)

    @app.post("/api/voice/test", response_model=OkResponse)
    async def voice_test(req: VoiceRequest | None = None) -> OkResponse:
        text = req.text if req is not None else "Hello World"
        chat_voice.speak_text(text)
        return OkResponse(ok=True)

    @app.post("/api/voice/speak", response_model=OkResponse)
    async def voice_speak(req: VoiceRequest) -> OkResponse:
        if not chat_voice.is_speakable(req.text):
            return OkResponse(ok=False)
        chat_voice.speak_text(req.text)
        return OkResponse(ok=True)

    @app.get("/api/prompts")
    async def list_prompts() -> list[dict[str, str]]:
        prompt_dir.mkdir(parents=True, exist_ok=True)
        return [{"name": f.stem, "filename": f.name} for f in sorted(prompt_dir.glob("*.md"))]

    @app.get("/api/prompts/{name}")
    async def get_prompt(name: str) -> dict[str, Any]:
        path = prompt_dir / f"{name}.md"
        if not path.exists():
            return {"ok": False, "error": "not found", "content": ""}
        return {"ok": True, "content": path.read_text(encoding="utf-8")}

    @app.post("/api/evolve", response_model=OkResponse)
    async def evolve() -> OkResponse:
        nonlocal _generating
        if _generating:
            return OkResponse(ok=False)

        conversation_dir.mkdir(parents=True, exist_ok=True)
        conv_files = sorted(conversation_dir.glob("*.md"))
        if not conv_files:
            return OkResponse(ok=False)

        latest = conv_files[-1]
        conv_content = latest.read_text(encoding="utf-8")
        if len(conv_content) > 20000:
            conv_content = conv_content[:20000] + "\n\n... (truncated)"

        evolve_prompt = (
            "[EVOLVE] Analyze the past conversation below and extract useful "
            "task strategies, patterns, failure modes, or lessons that would "
            "help in future robot programming tasks.\n\n"
            "Output a concise markdown document (under 500 words) with the key insights. "
            "Use headings and bullet points. Focus on actionable robot programming "
            "strategies specific to the YAM station.\n\n"
            "After your analysis, output the final prompt document in a "
            "```markdown\\n...\\n``` fenced block. I will save this to cap/prompt/.\n\n"
            f"---\n\nConversation from {latest.name}:\n\n{conv_content}"
        )

        _generating = True

        async def _run() -> None:
            nonlocal _generating
            try:
                await _run_backend_turn(
                    backend=backend,
                    message=evolve_prompt,
                    session=session,
                    ws_manager=ws_manager,
                    conversation_dir=conversation_dir,
                    chat_voice=chat_voice,
                )
                if session.messages and session.messages[-1].get("role") == "assistant":
                    text = session.messages[-1].get("content", "")
                    md_blocks = re.findall(r"```markdown\s*\n(.*?)```", text, re.DOTALL)
                    if md_blocks:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        save_path = prompt_dir / f"evolved_{ts}.md"
                        save_path.write_text(md_blocks[-1].strip(), encoding="utf-8")
                        logger.info("Evolved prompt saved: %s", save_path)
                        await ws_manager.broadcast(
                            "chat_text_delta",
                            {"text": f"\n\n*Saved to `{save_path.name}`*"},
                        )
            except Exception as exc:
                logger.exception("Evolve turn failed for %s", backend.name)
                chat_voice.on_turn_error()
                await ws_manager.broadcast("chat_error", {"error": str(exc)})
            finally:
                _generating = False

        asyncio.create_task(_run())
        return OkResponse(ok=True)

    @app.websocket("/ws/chat")
    async def chat_ws(ws: WebSocket) -> None:
        await ws_manager.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(ws)

    return app
