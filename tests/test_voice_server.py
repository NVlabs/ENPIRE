from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enpire.env.forge.cap.voice.voice_server import create_app


class FakeRecorder:
    def __init__(self, **kwargs: Any) -> None:
        self._on_partial = kwargs["on_realtime_transcription_update"]
        self._on_stable = kwargs["on_realtime_transcription_stabilized"]
        self._on_recording_start = kwargs["on_recording_start"]
        self._on_recording_stop = kwargs["on_recording_stop"]
        self._on_transcription_start = kwargs["on_transcription_start"]
        self._abort = threading.Event()
        self._called = False

    def text(self, on_transcription_finished=None):
        if self._called:
            self._abort.wait(timeout=0.2)
            return ""
        self._called = True
        self._on_recording_start({"event": "start"})
        self._on_partial("pick up")
        self._on_stable("pick up the red cup")
        self._on_recording_stop({"event": "stop"})
        self._on_transcription_start(b"fake-audio")
        if on_transcription_finished is not None:
            on_transcription_finished("pick up the red cup")
        self._abort.wait(timeout=0.2)
        return "pick up the red cup"

    def abort(self) -> None:
        self._abort.set()

    def shutdown(self) -> None:
        self._abort.set()


def test_voice_server_streams_partial_and_final_events() -> None:
    app = create_app(recorder_factory=FakeRecorder)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            resp = client.post("/api/start")
            assert resp.status_code == 200
            assert resp.json() == {"ok": True}

            seen_partial = False
            seen_final = False
            for _ in range(12):
                payload = json.loads(websocket.receive_text())
                if payload["type"] == "voice_partial":
                    seen_partial = seen_partial or payload["data"]["text"].startswith("pick up")
                if payload["type"] == "voice_final":
                    seen_final = payload["data"]["text"] == "pick up the red cup"
                    if seen_final:
                        break

            assert seen_partial
            assert seen_final

            status = client.get("/api/status")
            assert status.status_code == 200
            assert status.json()["final_text"] == "pick up the red cup"

            stop_resp = client.post("/api/stop")
            assert stop_resp.status_code == 200
            assert stop_resp.json() == {"ok": True}


def test_voice_service_callbacks_tolerate_extra_args() -> None:
    from enpire.env.forge.cap.voice.service import VoiceInputService

    service = VoiceInputService(project_root=Path("."), emit=lambda *_args, **_kwargs: None)

    service._handle_recording_start({"unexpected": True})
    assert service.snapshot()["phase"] == "recording"

    service._handle_transcription_start(b"audio-bytes")
    assert service.snapshot()["phase"] == "transcribing"

    service._handle_partial(b"hello world")
    assert service.snapshot()["partial_text"] == "hello world"

    service._handle_final(b"done now")
    snap = service.snapshot()
    assert snap["final_text"] == "done now"
    assert snap["phase"] == "finalized"
