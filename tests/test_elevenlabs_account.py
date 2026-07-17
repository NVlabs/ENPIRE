from __future__ import annotations

from enpire.env.forge.cap.voice.elevenlabs_account import (
    ElevenLabsVoiceSummary,
    format_voice_summary,
    pick_recommended_female_english_voice,
)


def test_pick_recommended_female_english_voice_prefers_rachel() -> None:
    voices = [
        ElevenLabsVoiceSummary(
            voice_id="male-1",
            name="Adam",
            gender="male",
            accent="American",
            category="premade",
        ),
        ElevenLabsVoiceSummary(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            name="Rachel",
            gender="female",
            accent="American",
            category="premade",
            use_case="conversational",
        ),
    ]
    recommended = pick_recommended_female_english_voice(voices)
    assert recommended is not None
    assert recommended.voice_id == "21m00Tcm4TlvDq8ikWAM"


def test_format_voice_summary_includes_key_fields() -> None:
    voice = ElevenLabsVoiceSummary(
        voice_id="voice-123",
        name="Rachel",
        gender="female",
        accent="American",
        category="premade",
        use_case="conversational",
    )
    formatted = format_voice_summary(voice)
    assert "name=Rachel" in formatted
    assert "voice_id=voice-123" in formatted
    assert "gender=female" in formatted
