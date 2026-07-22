# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .backends import VoiceOutputBackend, build_voice_output_backend_from_env
from .speakable_text import extract_speakable_text

logger = logging.getLogger(__name__)


@dataclass
class VoiceStatus:
    enabled: bool = True
    speaking: bool = False
    last_spoken_text: str = ""
    last_error: str | None = None


class VoiceOutputManager:
    """Simple queued TTS manager for bridge-side voice output."""

    def __init__(
        self,
        enabled: bool = True,
        before_speak: Callable[[], None] | None = None,
        backend: VoiceOutputBackend | None = None,
    ) -> None:
        self._status = VoiceStatus(enabled=enabled)
        self._before_speak = before_speak
        self._backend = backend or build_voice_output_backend_from_env()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def get_status(self) -> VoiceStatus:
        return VoiceStatus(
            enabled=self._status.enabled,
            speaking=self._status.speaking,
            last_spoken_text=self._status.last_spoken_text,
            last_error=self._status.last_error,
        )

    def set_enabled(self, enabled: bool) -> None:
        self._status.enabled = enabled

    def speak_async(self, text: str) -> bool:
        speakable = extract_speakable_text(text)
        if not speakable:
            return False
        if not self._status.enabled:
            logger.info("Voice output disabled; skipping text: %s", speakable[:120])
            return False
        self._queue.put(speakable)
        return True

    def speak_blocking(self, text: str) -> bool:
        speakable = extract_speakable_text(text)
        if not speakable:
            return False
        if not self._status.enabled:
            return False
        self._speak(speakable)
        return True

    def stop(self) -> bool:
        stopped = False
        sentinel_found = False
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                sentinel_found = True
                continue
            stopped = True
        if sentinel_found:
            self._queue.put(None)
        try:
            stopped = self._backend.stop() or stopped
        except Exception:  # pragma: no cover - defensive logging path
            logger.exception("Failed to stop active voice output backend")
        self._status.speaking = False
        return stopped

    def shutdown(self) -> None:
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=2)
        try:
            self._backend.shutdown()
        except Exception:  # pragma: no cover - defensive logging path
            logger.exception("Failed to shutdown voice output backend")

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._speak(item)
            except Exception as exc:  # pragma: no cover - defensive logging path
                self._status.last_error = str(exc)
                logger.exception("Voice output failed")

    def _speak(self, text: str) -> None:
        if self._before_speak is not None:
            try:
                self._before_speak()
            except Exception:  # pragma: no cover - defensive logging path
                logger.exception("Voice pre-speak hook failed")
        self._status.speaking = True
        self._status.last_error = None
        try:
            self._backend.speak(text)
            self._status.last_spoken_text = text
        finally:
            self._status.speaking = False
