# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from enpire.env.forge.cap.bridge import claude_bridge
from enpire.env.forge.cap.chat import extract_code_blocks
from enpire.env.forge.cap.chat import runtime as chat_runtime
from enpire.env.forge.cap.voice.backends import ElevenLabsConfig


def test_voice_test_endpoint(monkeypatch) -> None:
    spoken: list[str] = []

    def fake_speak_text(self, text: str) -> bool:  # noqa: ANN001
        spoken.append(text)
        return True

    monkeypatch.setattr(chat_runtime.ChatVoiceController, "speak_text", fake_speak_text)

    app = claude_bridge.create_app()
    client = TestClient(app)

    response = client.post("/api/voice/test", json={"text": "Hello World"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert spoken == ["Hello World"]


def test_voice_stop_endpoint(monkeypatch) -> None:
    stop_calls: list[bool] = []

    def fake_stop(self) -> bool:  # noqa: ANN001
        stop_calls.append(True)
        return True

    monkeypatch.setattr(chat_runtime.ChatVoiceController, "stop", fake_stop)

    app = claude_bridge.create_app()
    client = TestClient(app)

    response = client.post("/api/voice/stop")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert stop_calls == [True]


def test_agent_bridge_voice_test_offline_uses_elevenlabs_backend(monkeypatch) -> None:
    from enpire.env.forge.cap.bridge import agent_bridge

    synth_calls: list[tuple[str, ElevenLabsConfig]] = []
    played_audio: list[bytes] = []

    class FakeSynthesizer:
        def synthesize(self, text: str, config: ElevenLabsConfig) -> bytes:
            synth_calls.append((text, config))
            return b"fake-elevenlabs-audio"

    class FakePlayer:
        def play(self, audio_bytes: bytes) -> None:
            played_audio.append(audio_bytes)

        def stop(self) -> bool:
            return False

        def shutdown(self) -> None:
            return None

    monkeypatch.setenv("CAP_VOICE_OUTPUT_PROVIDER", "elevenlabs")
    monkeypatch.setenv("CAP_BRIDGE_VOICE_ENABLED", "1")
    monkeypatch.setenv("ELVENSLAB_API_KEY", "offline-test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "female-conversational-en")
    monkeypatch.setenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    monkeypatch.setenv("ELEVENLABS_LANGUAGE_CODE", "en")
    monkeypatch.setattr(
        "enpire.env.forge.cap.voice.backends.UrllibElevenLabsSynthesizer",
        lambda: FakeSynthesizer(),
    )
    monkeypatch.setattr(
        "enpire.env.forge.cap.voice.backends.PygameMP3AudioPlayer",
        lambda: FakePlayer(),
    )

    app = agent_bridge.create_app()
    client = TestClient(app)

    response = client.post("/api/voice/test", json={"text": "Hello World"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    deadline = time.time() + 2
    while not synth_calls and time.time() < deadline:
        time.sleep(0.01)
    assert synth_calls
    assert synth_calls[0][0] == "Hello World"
    assert played_audio == [b"fake-elevenlabs-audio"]


def test_shared_code_block_extraction_supports_multiple_backends() -> None:
    blocks = extract_code_blocks(
        "Intro\n```python\nprint('hi')\n```\n"
        "and\n```json\n{\"ok\": true}\n```"
    )
    assert [(block.language, block.code) for block in blocks] == [
        ("python", "print('hi')"),
        ("json", '{"ok": true}'),
    ]
