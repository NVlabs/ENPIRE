# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .backends import _get_env

logger = logging.getLogger(__name__)

DEFAULT_FEMALE_ENGLISH_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel


@dataclass(frozen=True)
class ElevenLabsVoiceSummary:
    voice_id: str
    name: str
    category: str | None = None
    gender: str | None = None
    age: str | None = None
    accent: str | None = None
    description: str | None = None
    use_case: str | None = None
    preview_url: str | None = None


def get_elevenlabs_api_key() -> str:
    api_key = _get_env("ELEVENLABS_API_KEY", "ELVENSLAB_API_KEY", "ELVENSLABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ElevenLabs API key. Set ELEVENLABS_API_KEY "
            "(ELVENSLAB_API_KEY also accepted as a fallback alias)."
        )
    return api_key


def list_elevenlabs_voices(api_key: str, api_base_url: str = "https://api.elevenlabs.io") -> list[ElevenLabsVoiceSummary]:
    request = urllib.request.Request(
        f"{api_base_url.rstrip('/')}/v1/voices",
        headers={"xi-api-key": api_key},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Failed to list ElevenLabs voices with HTTP {exc.code}: {body[:300]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to list ElevenLabs voices: {exc.reason}") from exc

    voices = payload.get("voices", [])
    results: list[ElevenLabsVoiceSummary] = []
    for item in voices:
        labels = item.get("labels") or {}
        results.append(
            ElevenLabsVoiceSummary(
                voice_id=str(item.get("voice_id", "")),
                name=str(item.get("name", "")),
                category=item.get("category"),
                gender=labels.get("gender"),
                age=labels.get("age"),
                accent=labels.get("accent") or labels.get("language"),
                description=item.get("description"),
                use_case=labels.get("use case") or labels.get("use_case"),
                preview_url=item.get("preview_url"),
            )
        )
    return results


def pick_recommended_female_english_voice(
    voices: list[ElevenLabsVoiceSummary],
) -> ElevenLabsVoiceSummary | None:
    if not voices:
        return None

    def _score(voice: ElevenLabsVoiceSummary) -> tuple[int, int, int, int]:
        name = voice.name.lower()
        gender = (voice.gender or "").lower()
        accent = (voice.accent or "").lower()
        use_case = (voice.use_case or "").lower()
        category = (voice.category or "").lower()
        return (
            1 if name == "rachel" else 0,
            1 if gender == "female" else 0,
            1 if ("english" in accent or accent in {"en", "us", "american"}) else 0,
            1 if ("convers" in use_case or category == "premade") else 0,
        )

    ranked = sorted(voices, key=_score, reverse=True)
    return ranked[0]


def format_voice_summary(voice: ElevenLabsVoiceSummary) -> str:
    details = [
        f"name={voice.name}",
        f"voice_id={voice.voice_id}",
    ]
    if voice.gender:
        details.append(f"gender={voice.gender}")
    if voice.accent:
        details.append(f"accent={voice.accent}")
    if voice.category:
        details.append(f"category={voice.category}")
    if voice.use_case:
        details.append(f"use_case={voice.use_case}")
    return " | ".join(details)
