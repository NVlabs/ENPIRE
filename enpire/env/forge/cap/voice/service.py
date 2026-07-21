"""Host-side microphone voice input service."""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class RecorderProtocol(Protocol):
    def text(self, on_transcription_finished=None): ...
    def abort(self) -> None: ...
    def shutdown(self) -> None: ...


class VoiceInputService:
    """Manage a single RealtimeSTT recorder instance for voice input."""

    def __init__(
        self,
        *,
        project_root: Path,
        emit,
        recorder_factory=None,
    ) -> None:
        self._project_root = Path(project_root)
        self._emit = emit
        self._recorder_factory = recorder_factory
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._preload_thread: threading.Thread | None = None
        self._recorder: RecorderProtocol | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._phase = "idle"
        self._partial_text = ""
        self._final_text = ""
        self._last_error = ""
        self._preloaded = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            listening = self._thread is not None and self._thread.is_alive()
            return {
                "listening": listening,
                "preloading": self._preload_thread is not None
                and self._preload_thread.is_alive(),
                "preloaded": self._preloaded,
                "phase": self._phase,
                "partial_text": self._partial_text,
                "final_text": self._final_text,
                "error": self._last_error,
            }

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self._partial_text = ""
            self._final_text = ""
            self._last_error = ""
        self._emit_status()
        return self.snapshot()

    def start(self, loop: asyncio.AbstractEventLoop) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.snapshot()
            self._loop = loop
            self._stop_event = threading.Event()
            self._partial_text = ""
            self._final_text = ""
            self._last_error = ""
            self._phase = "starting"
            self._thread = threading.Thread(
                target=self._run, name="cap-voice-input", daemon=True
            )
            self._thread.start()
        self._emit_status()
        return self.snapshot()

    def preload(self) -> dict[str, Any]:
        with self._lock:
            if self._preloaded:
                return self.snapshot()
            if self._preload_thread is not None and self._preload_thread.is_alive():
                return self.snapshot()
            self._phase = "preloading"
            self._preload_thread = threading.Thread(
                target=self._preload_recorder,
                name="cap-voice-preload",
                daemon=True,
            )
            self._preload_thread.start()
        self._emit_status()
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        thread: threading.Thread | None
        recorder: RecorderProtocol | None
        with self._lock:
            self._stop_event.set()
            self._phase = "stopping"
            thread = self._thread
            recorder = self._recorder
        if recorder is not None:
            try:
                recorder.abort()
            except Exception:
                logger.exception("Failed to abort voice recorder cleanly")
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        self._emit_status()
        return self.snapshot()

    def shutdown(self) -> None:
        self.stop()
        with self._lock:
            recorder = self._recorder
            self._recorder = None
            preload_thread = self._preload_thread
            self._preloaded = False
        if preload_thread is not None and preload_thread.is_alive():
            preload_thread.join(timeout=5)
        if recorder is not None:
            try:
                recorder.shutdown()
            except Exception:
                logger.exception("Failed to shutdown preloaded voice recorder cleanly")

    def _emit_threadsafe(self, event_type: str, data: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._emit(event_type, data), loop)
        except Exception:
            logger.exception("Failed to emit voice event %s", event_type)

    def _emit_status(self) -> None:
        self._emit_threadsafe("voice_status", self.snapshot())

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
        self._emit_status()

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").strip()
        return str(value).strip()

    def _handle_partial(self, text: Any, *_args: Any, **_kwargs: Any) -> None:
        cleaned = self._coerce_text(text)
        with self._lock:
            if cleaned == self._partial_text:
                return
            self._partial_text = cleaned
        self._emit_threadsafe("voice_partial", {"text": cleaned})
        self._emit_status()

    def _handle_final(self, text: Any, *_args: Any, **_kwargs: Any) -> None:
        cleaned = self._coerce_text(text)
        if not cleaned:
            return
        with self._lock:
            self._partial_text = cleaned
            self._final_text = cleaned
            self._phase = "finalized"
        self._emit_threadsafe("voice_final", {"text": cleaned})
        self._emit_status()

    def _handle_recording_start(self, *_args: Any, **_kwargs: Any) -> None:
        self._set_phase("recording")

    def _handle_recording_stop(self, *_args: Any, **_kwargs: Any) -> None:
        self._set_phase("processing")

    def _handle_transcription_start(self, *_args: Any, **_kwargs: Any) -> None:
        self._set_phase("transcribing")

    def _load_recorder_factory(self):
        if self._recorder_factory is not None:
            return self._recorder_factory
        try:
            from RealtimeSTT import AudioToTextRecorder

            return AudioToTextRecorder
        except Exception:
            vendored_root = self._project_root / "third_party" / "ui" / "RealtimeSTT"
            if vendored_root.exists():
                sys.path.insert(0, str(vendored_root))
                from RealtimeSTT import AudioToTextRecorder

                return AudioToTextRecorder
            raise

    def _build_recorder(self) -> RecorderProtocol:
        recorder_factory = self._load_recorder_factory()
        return recorder_factory(
            model="tiny.en",
            language="en",
            spinner=False,
            level=logging.ERROR,
            no_log_file=True,
            use_extended_logging=False,
            enable_realtime_transcription=True,
            use_main_model_for_realtime=True,
            realtime_model_type="tiny.en",
            realtime_processing_pause=0.05,
            init_realtime_after_seconds=0.0,
            on_realtime_transcription_update=self._handle_partial,
            on_realtime_transcription_stabilized=self._handle_partial,
            on_recording_start=self._handle_recording_start,
            on_recording_stop=self._handle_recording_stop,
            on_transcription_start=self._handle_transcription_start,
            silero_sensitivity=0.05,
            silero_use_onnx=True,
            webrtc_sensitivity=3,
            faster_whisper_vad_filter=False,
            post_speech_silence_duration=0.25,
            min_length_of_recording=0.15,
            min_gap_between_recordings=0.0,
            beam_size=1,
            beam_size_realtime=1,
            batch_size=0,
            realtime_batch_size=0,
            ensure_sentence_starting_uppercase=False,
            ensure_sentence_ends_with_period=False,
        )

    def _preload_recorder(self) -> None:
        recorder: RecorderProtocol | None = None
        try:
            logger.info("Preloading voice recorder in background")
            recorder = self._build_recorder()
            with self._lock:
                self._preloaded = True
                if self._phase == "preloading":
                    self._phase = "ready"
        except Exception as exc:
            logger.exception("Voice recorder preload failed")
            with self._lock:
                self._preloaded = False
                self._last_error = str(exc)
                self._phase = "error"
        finally:
            if recorder is not None:
                try:
                    recorder.shutdown()
                except Exception:
                    logger.exception(
                        "Failed to shutdown preloaded voice recorder cleanly"
                    )
            with self._lock:
                self._preload_thread = None
            self._emit_status()

    def _run(self) -> None:
        recorder = None
        try:
            recorder = self._build_recorder()
            with self._lock:
                self._recorder = recorder
                self._phase = "listening"
            self._emit_status()
            while not self._stop_event.is_set():
                recorder.text(self._handle_final)
                if not self._stop_event.is_set():
                    self._set_phase("listening")
        except Exception as exc:
            logger.exception("Voice input failed")
            with self._lock:
                self._last_error = str(exc)
                self._phase = "error"
            self._emit_threadsafe("voice_error", {"error": str(exc)})
            self._emit_status()
        finally:
            if recorder is not None:
                try:
                    recorder.shutdown()
                except Exception:
                    logger.exception("Failed to shutdown voice recorder cleanly")
            with self._lock:
                self._recorder = None
                self._thread = None
                if self._phase != "error":
                    self._phase = "ready" if self._preloaded else "idle"
            self._emit_status()
