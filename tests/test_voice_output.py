from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from cap.voice import VoiceOutputManager, extract_speakable_text
from cap.voice.backends import (
    ElevenLabsConfig,
    ElevenLabsVoiceOutputBackend,
    ElevenLabsVoiceSettings,
)


def test_extract_speakable_text_removes_markdown_code() -> None:
    raw = "Here is text.\n```python\nprint('hi')\n```\nAnd `inline_code` too."
    cleaned = extract_speakable_text(raw)
    assert "print('hi')" not in cleaned
    assert "inline_code" in cleaned
    assert "Here is text." in cleaned
    assert "And" in cleaned


def test_voice_output_runs_pre_speak_hook_before_playback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeStream:
        def __init__(self, _engine) -> None:  # noqa: ANN001
            pass

        def feed(self, text: str) -> "FakeStream":
            calls.append(f"feed:{text}")
            return self

        def play(self) -> None:
            calls.append("play")

    fake_module = types.SimpleNamespace(
        SystemEngine=lambda: object(),
        TextToAudioStream=lambda engine: FakeStream(engine),
    )
    monkeypatch.setitem(sys.modules, "RealtimeTTS", fake_module)

    manager = VoiceOutputManager(
        enabled=True,
        before_speak=lambda: calls.append("before"),
    )
    try:
        assert manager.speak_blocking("Hello World")
        assert calls[:3] == ["before", "feed:Hello World", "play"]
    finally:
        manager.shutdown()


def test_voice_output_stop_interrupts_active_playback(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    stop_called = threading.Event()
    allow_play_exit = threading.Event()

    class FakeStream:
        def __init__(self, _engine) -> None:  # noqa: ANN001
            pass

        def feed(self, text: str) -> "FakeStream":
            events.append(f"feed:{text}")
            return self

        def play(self) -> None:
            events.append("play")
            allow_play_exit.wait(timeout=2)

        def stop(self) -> None:
            events.append("stop")
            stop_called.set()
            allow_play_exit.set()

    fake_module = types.SimpleNamespace(
        SystemEngine=lambda: object(),
        TextToAudioStream=lambda engine: FakeStream(engine),
    )
    monkeypatch.setitem(sys.modules, "RealtimeTTS", fake_module)

    manager = VoiceOutputManager(enabled=True)
    try:
        assert manager.speak_async("Interrupt me")
        deadline = time.time() + 2
        while not manager.get_status().speaking and time.time() < deadline:
            time.sleep(0.01)
        assert manager.get_status().speaking is True
        assert manager.stop() is True
        assert stop_called.wait(timeout=2)
    finally:
        allow_play_exit.set()
        manager.shutdown()


def test_elevenlabs_voice_output_backend_speaks_hello_world_offline() -> None:
    requests: list[tuple[str, ElevenLabsConfig]] = []
    played_audio: list[bytes] = []

    class FakeSynthesizer:
        def synthesize(self, text: str, config: ElevenLabsConfig) -> bytes:
            requests.append((text, config))
            return b"fake-elevenlabs-audio"

    class FakePlayer:
        def play(self, audio_bytes: bytes) -> None:
            played_audio.append(audio_bytes)

        def stop(self) -> bool:
            return False

        def shutdown(self) -> None:
            return None

    config = ElevenLabsConfig(
        api_key="test-key",
        voice_id="female-conversational-en",
        model_id="eleven_flash_v2_5",
        language_code="en",
        voice_preset="conversational",
        voice_settings=ElevenLabsVoiceSettings(
            stability=0.45,
            similarity_boost=0.80,
            style=0.20,
            speed=1.00,
            use_speaker_boost=True,
        ),
    )
    backend = ElevenLabsVoiceOutputBackend(
        config,
        synthesizer=FakeSynthesizer(),
        player=FakePlayer(),
    )
    manager = VoiceOutputManager(enabled=True, backend=backend)
    try:
        assert manager.speak_blocking("Hello World") is True
        assert requests
        assert requests[0][0] == "Hello World"
        assert requests[0][1].model_id == "eleven_flash_v2_5"
        assert requests[0][1].language_code == "en"
        assert requests[0][1].voice_preset == "conversational"
        assert played_audio == [b"fake-elevenlabs-audio"]
        assert manager.get_status().last_spoken_text == "Hello World"
    finally:
        manager.shutdown()


def test_elevenlabs_voice_output_stop_interrupts_offline_playback() -> None:
    stop_called = threading.Event()
    allow_exit = threading.Event()

    class FakeSynthesizer:
        def synthesize(self, text: str, config: ElevenLabsConfig) -> bytes:  # noqa: ARG002
            return b"fake-elevenlabs-audio"

    class FakePlayer:
        def play(self, audio_bytes: bytes) -> None:  # noqa: ARG002
            allow_exit.wait(timeout=2)

        def stop(self) -> bool:
            stop_called.set()
            allow_exit.set()
            return True

        def shutdown(self) -> None:
            allow_exit.set()

    backend = ElevenLabsVoiceOutputBackend(
        ElevenLabsConfig(
            api_key="test-key",
            voice_id="female-conversational-en",
        ),
        synthesizer=FakeSynthesizer(),
        player=FakePlayer(),
    )
    manager = VoiceOutputManager(enabled=True, backend=backend)
    try:
        assert manager.speak_async("Hello World") is True
        deadline = time.time() + 2
        while not manager.get_status().speaking and time.time() < deadline:
            time.sleep(0.01)
        assert manager.get_status().speaking is True
        assert manager.stop() is True
        assert stop_called.wait(timeout=2)
    finally:
        allow_exit.set()
        manager.shutdown()


@pytest.mark.audio
def test_voice_output_hello_world_smoke() -> None:
    pytest.importorskip("RealtimeTTS")
    manager = VoiceOutputManager(enabled=True)
    try:
        assert manager.speak_blocking("Hello World")
        status = manager.get_status()
        assert status.last_spoken_text == "Hello World"
        assert status.last_error is None
    finally:
        manager.shutdown()
