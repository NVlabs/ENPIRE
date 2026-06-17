from __future__ import annotations

from cap.voice import ChatVoiceController


class StubVoiceOutput:
    def __init__(self) -> None:
        self.enabled = True
        self.spoken: list[str] = []
        self.stop_calls = 0

    def speak_async(self, text: str) -> bool:
        self.spoken.append(text)
        return True

    def stop(self) -> bool:
        self.stop_calls += 1
        return True

    def get_status(self):  # noqa: ANN201
        from cap.voice.output import VoiceStatus

        return VoiceStatus(enabled=self.enabled, speaking=False, last_spoken_text=self.spoken[-1] if self.spoken else "")

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def shutdown(self) -> None:
        return None


def test_chat_voice_controller_accumulates_turn_text() -> None:
    stub = StubVoiceOutput()
    controller = ChatVoiceController(voice_output=stub)

    controller.on_text_delta("Hello ")
    controller.on_text_delta("World")

    assert controller.on_turn_complete() is True
    assert stub.spoken == ["Hello World"]


def test_chat_voice_controller_clears_failed_turn() -> None:
    stub = StubVoiceOutput()
    controller = ChatVoiceController(voice_output=stub)

    controller.on_text_delta("partial")
    controller.on_turn_error()

    assert controller.on_turn_complete("fallback") is True
    assert stub.spoken == ["fallback"]


def test_chat_voice_controller_stop_forwards_to_voice_output() -> None:
    stub = StubVoiceOutput()
    controller = ChatVoiceController(voice_output=stub)

    controller.on_text_delta("partial")

    assert controller.stop() is True
    assert stub.stop_calls == 1
    assert controller.on_turn_complete("fallback") is True
    assert stub.spoken == ["fallback"]
