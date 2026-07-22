# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from enpire.env.forge.cap.chat.runtime import ChatSession, ChatTurnSink, create_chat_app
from enpire.env.forge.cap.voice.output import VoiceStatus


class StubChatVoice:
    def __init__(self) -> None:
        self.enabled = True
        self._chunks: list[str] = []
        self.spoken: list[str] = []

    def on_text_delta(self, text: str) -> None:
        if text:
            self._chunks.append(text)

    def on_turn_complete(self, fallback_text: str = "") -> bool:
        text = "".join(self._chunks) or fallback_text
        self._chunks.clear()
        self.spoken.append(text)
        return True

    def on_turn_error(self) -> None:
        self._chunks.clear()

    def speak_text(self, text: str) -> bool:
        self.spoken.append(text)
        return True

    def is_speakable(self, text: str) -> bool:
        return bool(text.strip())

    def get_status(self) -> VoiceStatus:
        return VoiceStatus(
            enabled=self.enabled,
            speaking=False,
            last_spoken_text=self.spoken[-1] if self.spoken else "",
        )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def shutdown(self) -> None:
        return None


class FakeBackend:
    name = "fake"

    async def run_turn(self, message: str, session: ChatSession, sink: ChatTurnSink) -> str:
        await sink.set_session_id("fake-session")
        await sink.emit_tool_use("echo", {"message": message})
        await sink.emit_text("Hello\n```python\nprint('hi')\n```\nWorld")
        return ""


def test_runtime_supports_non_claude_backend(tmp_path) -> None:
    voice = StubChatVoice()
    app = create_chat_app(
        FakeBackend(),
        title="Fake Bridge",
        prompt_dir=tmp_path / "prompt",
        per_message_dir=tmp_path / "prompt" / "per_message",
        conversation_dir=tmp_path / "conversation",
        chat_voice=voice,
    )
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        response = client.post("/api/chat", json={"message": "say hello"})
        assert response.status_code == 200
        assert response.json() == {"ok": True}

        events = []
        while True:
            payload = json.loads(ws.receive_text())
            events.append(payload["type"])
            if payload["type"] == "chat_turn_complete":
                break

    assert events == [
        "chat_tool_use",
        "chat_text_delta",
        "chat_code_block",
        "chat_turn_complete",
    ]
    assert voice.spoken == ["Hello\n```python\nprint('hi')\n```\nWorld"]
