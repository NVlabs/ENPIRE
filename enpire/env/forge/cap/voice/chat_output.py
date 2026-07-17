from __future__ import annotations

from .output import VoiceOutputManager, VoiceStatus
from .speakable_text import extract_speakable_text


class ChatVoiceController:
    """Backend-agnostic voice sink for chat-style assistant output."""

    def __init__(self, voice_output: VoiceOutputManager | None = None) -> None:
        self._voice_output = voice_output or VoiceOutputManager(enabled=True)
        self._assistant_chunks: list[str] = []

    def on_text_delta(self, text: str) -> None:
        if text:
            self._assistant_chunks.append(text)

    def on_turn_complete(self, fallback_text: str = "") -> bool:
        text = "".join(self._assistant_chunks) or fallback_text
        self._assistant_chunks.clear()
        return self._voice_output.speak_async(text)

    def on_turn_error(self) -> None:
        self._assistant_chunks.clear()

    def speak_text(self, text: str) -> bool:
        return self._voice_output.speak_async(text)

    def is_speakable(self, text: str) -> bool:
        return bool(extract_speakable_text(text))

    def get_status(self) -> VoiceStatus:
        return self._voice_output.get_status()

    def set_enabled(self, enabled: bool) -> None:
        self._voice_output.set_enabled(enabled)

    def stop(self) -> bool:
        self._assistant_chunks.clear()
        stop = getattr(self._voice_output, "stop", None)
        if stop is None:
            return False
        return bool(stop())

    def shutdown(self) -> None:
        self._voice_output.shutdown()
