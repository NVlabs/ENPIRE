from __future__ import annotations

import argparse
import os

import pytest

from cap.voice import VoiceOutputManager

DEFAULT_TEXT = "Hey, how are you?"
DEFAULT_VOICE_ID = "yM93hbw8Qtvdma2wCnJG"
LIVE_TEST_ENV = "RUN_ELEVENLABS_API_VOICE_TEST"


def run_live_voice_test(*, text: str = DEFAULT_TEXT, voice_id: str = DEFAULT_VOICE_ID) -> bool:
    os.environ.setdefault("CAP_VOICE_OUTPUT_PROVIDER", "elevenlabs")
    os.environ.setdefault("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    os.environ.setdefault("ELEVENLABS_LANGUAGE_CODE", "en")
    os.environ["ELEVENLABS_VOICE_ID"] = voice_id

    manager = VoiceOutputManager(enabled=True)
    try:
        return manager.speak_blocking(text)
    finally:
        manager.shutdown()


@pytest.mark.audio
@pytest.mark.skipif(
    os.environ.get(LIVE_TEST_ENV) != "1",
    reason=(
        "Optional live ElevenLabs voice test. "
        f"Set {LIVE_TEST_ENV}=1 to run under pytest."
    ),
)
def test_elevenslab_api_voice_live() -> None:
    assert run_live_voice_test(
        text=os.environ.get("ELEVENLABS_TEST_TEXT", DEFAULT_TEXT),
        voice_id=os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional live ElevenLabs API voice test.")
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help='Text to speak. Defaults to "Hey, how are you?"',
    )
    parser.add_argument(
        "--voice-id",
        default=os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID),
        help="ElevenLabs voice ID to use for the live test.",
    )
    args = parser.parse_args()

    ok = run_live_voice_test(text=args.text, voice_id=args.voice_id)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
