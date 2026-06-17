from __future__ import annotations

import argparse
import os

from cap.voice import VoiceOutputManager
from cap.voice.elevenlabs_account import DEFAULT_FEMALE_ENGLISH_VOICE_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speak a hello-world sample with ElevenLabs TTS.")
    parser.add_argument(
        "--voice-id",
        default=os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_FEMALE_ENGLISH_VOICE_ID,
        help="ElevenLabs voice_id to use. Defaults to Rachel.",
    )
    parser.add_argument(
        "--text",
        default="Hello world. This is a conversational female English voice test from ElevenLabs.",
        help="Text to speak.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("CAP_VOICE_OUTPUT_PROVIDER", "elevenlabs")
    os.environ["ELEVENLABS_VOICE_ID"] = args.voice_id
    manager = VoiceOutputManager(enabled=True)
    try:
        ok = manager.speak_blocking(args.text)
    finally:
        manager.shutdown()
    if not ok:
        raise RuntimeError("Voice output did not accept the text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
