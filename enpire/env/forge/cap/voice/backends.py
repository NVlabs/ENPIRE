from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def _ensure_audio_env() -> None:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    os.environ["XDG_RUNTIME_DIR"] = runtime_dir
    os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime_dir}/bus")


class VoiceOutputBackend(Protocol):
    def speak(self, text: str) -> None: ...

    def stop(self) -> bool: ...

    def shutdown(self) -> None: ...


class AudioPlayer(Protocol):
    def play(self, audio_bytes: bytes) -> None: ...

    def stop(self) -> bool: ...

    def shutdown(self) -> None: ...


class ElevenLabsSynthesizer(Protocol):
    def synthesize(self, text: str, config: "ElevenLabsConfig") -> bytes: ...


@dataclass(frozen=True)
class ElevenLabsVoiceSettings:
    stability: float | None = None
    similarity_boost: float | None = None
    style: float | None = None
    speed: float | None = None
    use_speaker_boost: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.stability is not None:
            payload["stability"] = self.stability
        if self.similarity_boost is not None:
            payload["similarity_boost"] = self.similarity_boost
        if self.style is not None:
            payload["style"] = self.style
        if self.speed is not None:
            payload["speed"] = self.speed
        if self.use_speaker_boost is not None:
            payload["use_speaker_boost"] = self.use_speaker_boost
        return payload


_ELEVENLABS_PRESETS: dict[str, ElevenLabsVoiceSettings] = {
    "default": ElevenLabsVoiceSettings(),
    "conversational": ElevenLabsVoiceSettings(
        stability=0.45,
        similarity_boost=0.80,
        style=0.20,
        speed=1.00,
        use_speaker_boost=True,
    ),
}


def _get_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def _get_env_bool(*names: str, default: bool | None = None) -> bool | None:
    raw = _get_env(*names)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _get_env_float(*names: str, default: float | None = None) -> float | None:
    raw = _get_env(*names)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float environment value for %s: %s", names[0], raw)
        return default


@dataclass(frozen=True)
class ElevenLabsConfig:
    api_key: str
    voice_id: str
    model_id: str = "eleven_flash_v2_5"
    language_code: str | None = "en"
    output_format: str = "mp3_44100_128"
    voice_preset: str = "conversational"
    voice_settings: ElevenLabsVoiceSettings = ElevenLabsVoiceSettings()
    api_base_url: str = "https://api.elevenlabs.io"
    request_timeout_s: float = 30.0

    @classmethod
    def from_env(cls) -> "ElevenLabsConfig":
        api_key = _get_env("ELEVENLABS_API_KEY", "ELVENSLAB_API_KEY", "ELVENSLABS_API_KEY")
        voice_id = _get_env("ELEVENLABS_VOICE_ID")
        if not api_key:
            raise RuntimeError(
                "ElevenLabs voice output requires ELEVENLABS_API_KEY "
                "(ELVENSLAB_API_KEY also accepted as a fallback alias)."
            )
        if not voice_id:
            raise RuntimeError("ElevenLabs voice output requires ELEVENLABS_VOICE_ID.")

        preset_name = (_get_env("ELEVENLABS_VOICE_PRESET", default="conversational") or "conversational").lower()
        preset = _ELEVENLABS_PRESETS.get(preset_name, _ELEVENLABS_PRESETS["conversational"])
        voice_settings = ElevenLabsVoiceSettings(
            stability=_get_env_float("ELEVENLABS_STABILITY", default=preset.stability),
            similarity_boost=_get_env_float(
                "ELEVENLABS_SIMILARITY_BOOST",
                default=preset.similarity_boost,
            ),
            style=_get_env_float("ELEVENLABS_STYLE", default=preset.style),
            speed=_get_env_float("ELEVENLABS_SPEED", default=preset.speed),
            use_speaker_boost=_get_env_bool(
                "ELEVENLABS_USE_SPEAKER_BOOST",
                default=preset.use_speaker_boost,
            ),
        )
        return cls(
            api_key=api_key,
            voice_id=voice_id,
            model_id=_get_env("ELEVENLABS_MODEL_ID", default="eleven_flash_v2_5") or "eleven_flash_v2_5",
            language_code=_get_env("ELEVENLABS_LANGUAGE_CODE", default="en"),
            output_format=_get_env("ELEVENLABS_OUTPUT_FORMAT", default="mp3_44100_128") or "mp3_44100_128",
            voice_preset=preset_name,
            voice_settings=voice_settings,
            api_base_url=_get_env("ELEVENLABS_API_BASE_URL", default="https://api.elevenlabs.io")
            or "https://api.elevenlabs.io",
            request_timeout_s=_get_env_float("ELEVENLABS_TIMEOUT_S", default=30.0) or 30.0,
        )


class SystemVoiceOutputBackend:
    def __init__(self) -> None:
        self._stream_lock = threading.RLock()
        self._current_stream: Any | None = None

    def speak(self, text: str) -> None:
        _ensure_audio_env()
        from RealtimeTTS import SystemEngine, TextToAudioStream

        stream = TextToAudioStream(SystemEngine())
        with self._stream_lock:
            self._current_stream = stream
        try:
            stream.feed(text).play()
        finally:
            with self._stream_lock:
                self._current_stream = None

    def stop(self) -> bool:
        with self._stream_lock:
            stream = self._current_stream
        if stream is None:
            return False
        try:
            stream.stop()
            return True
        except Exception:  # pragma: no cover - defensive logging path
            logger.exception("Failed to stop active voice output stream")
            return False

    def shutdown(self) -> None:
        self.stop()


class UrllibElevenLabsSynthesizer:
    def synthesize(self, text: str, config: ElevenLabsConfig) -> bytes:
        voice_id = urllib.parse.quote(config.voice_id, safe="")
        url = f"{config.api_base_url.rstrip('/')}/v1/text-to-speech/{voice_id}"
        payload: dict[str, Any] = {
            "text": text,
            "model_id": config.model_id,
            "output_format": config.output_format,
        }
        if config.language_code:
            payload["language_code"] = config.language_code
        voice_settings = config.voice_settings.to_dict()
        if voice_settings:
            payload["voice_settings"] = voice_settings
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": config.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.request_timeout_s) as response:
                audio = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ElevenLabs TTS request failed with HTTP {exc.code}: {body[:300]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ElevenLabs TTS request failed: {exc.reason}") from exc
        if not audio:
            raise RuntimeError("ElevenLabs TTS request returned empty audio.")
        return audio


class PygameMP3AudioPlayer:
    def __init__(self, poll_interval_s: float = 0.05) -> None:
        self._poll_interval_s = poll_interval_s
        self._lock = threading.RLock()
        self._pygame: Any | None = None
        self._stop_requested = False

    def _ensure_pygame(self) -> Any:
        if self._pygame is None:
            import pygame

            self._pygame = pygame
        if self._pygame.mixer.get_init() is None:
            _ensure_audio_env()
            self._pygame.mixer.init()
        return self._pygame

    def play(self, audio_bytes: bytes) -> None:
        if not audio_bytes:
            raise RuntimeError("Audio player received empty audio bytes.")

        with self._lock:
            pygame = self._ensure_pygame()
            self._stop_requested = False
            audio_stream = io.BytesIO(audio_bytes)
            try:
                pygame.mixer.music.load(audio_stream, "mp3")
            except TypeError:  # pragma: no cover - older pygame compatibility
                pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()

        try:
            while True:
                with self._lock:
                    busy = bool(pygame.mixer.music.get_busy())
                    stop_requested = self._stop_requested
                if not busy or stop_requested:
                    break
                time.sleep(self._poll_interval_s)
        finally:
            with self._lock:
                self._stop_requested = False

    def stop(self) -> bool:
        with self._lock:
            pygame = self._pygame
            if pygame is None or pygame.mixer.get_init() is None:
                return False
            was_busy = bool(pygame.mixer.music.get_busy())
            self._stop_requested = True
            pygame.mixer.music.stop()
            return was_busy

    def shutdown(self) -> None:
        with self._lock:
            pygame = self._pygame
            if pygame is None or pygame.mixer.get_init() is None:
                return
            try:
                pygame.mixer.music.stop()
            finally:
                pygame.mixer.quit()


class ElevenLabsVoiceOutputBackend:
    def __init__(
        self,
        config: ElevenLabsConfig,
        *,
        synthesizer: ElevenLabsSynthesizer | None = None,
        player: AudioPlayer | None = None,
    ) -> None:
        self._config = config
        self._synthesizer = synthesizer or UrllibElevenLabsSynthesizer()
        self._player = player or PygameMP3AudioPlayer()

    def speak(self, text: str) -> None:
        audio = self._synthesizer.synthesize(text, self._config)
        self._player.play(audio)

    def stop(self) -> bool:
        return self._player.stop()

    def shutdown(self) -> None:
        self._player.shutdown()


def build_voice_output_backend_from_env() -> VoiceOutputBackend:
    provider = (
        _get_env("CAP_VOICE_OUTPUT_PROVIDER", "CAP_VOICE_PROVIDER", default="system") or "system"
    ).strip().lower()
    if provider in {"system", "local_system", "realtimetts", "realtime_tts"}:
        return SystemVoiceOutputBackend()
    if provider in {"elevenlabs", "eleven_labs", "elvenslab", "elevenslab"}:
        return ElevenLabsVoiceOutputBackend(ElevenLabsConfig.from_env())
    raise RuntimeError(f"Unsupported voice output provider: {provider}")
