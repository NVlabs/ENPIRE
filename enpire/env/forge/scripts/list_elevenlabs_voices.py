# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enpire.env.forge.cap.voice.elevenlabs_account import (
    format_voice_summary,
    get_elevenlabs_api_key,
    list_elevenlabs_voices,
    pick_recommended_female_english_voice,
)


def main() -> int:
    api_key = get_elevenlabs_api_key()
    voices = list_elevenlabs_voices(api_key)
    print(f"Found {len(voices)} ElevenLabs voices")
    print()
    recommended = pick_recommended_female_english_voice(voices)
    if recommended is not None:
        print("Recommended female English voice:")
        print(f"  {format_voice_summary(recommended)}")
        print()
    for voice in voices:
        print(format_voice_summary(voice))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
