# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone FastAPI voice server for host microphone capture."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from enpire.env.forge.cap.voice.service import VoiceInputService

logger = logging.getLogger(__name__)

VOICE_PORT = int(os.environ.get("CAP_VOICE_PORT", "8202"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class OkResponse(BaseModel):
    ok: bool


class VoiceWSManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, event_type: str, data: Any) -> None:
        payload = json.dumps(
            {"type": event_type, "data": data, "timestamp": time.strftime("%H:%M:%S")}
        )
        stale: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._connections.remove(ws)


def create_app(*, recorder_factory=None) -> FastAPI:
    app = FastAPI(title="CAP Voice Input")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws_manager = VoiceWSManager()
    voice_input = VoiceInputService(
        project_root=PROJECT_ROOT,
        emit=ws_manager.broadcast,
        recorder_factory=recorder_factory,
    )

    @app.post("/api/start", response_model=OkResponse)
    async def start_voice() -> OkResponse:
        import asyncio

        voice_input.start(asyncio.get_running_loop())
        return OkResponse(ok=True)

    @app.post("/api/stop", response_model=OkResponse)
    async def stop_voice() -> OkResponse:
        voice_input.stop()
        return OkResponse(ok=True)

    @app.post("/api/clear", response_model=OkResponse)
    async def clear_voice() -> OkResponse:
        voice_input.clear()
        return OkResponse(ok=True)

    @app.get("/api/status")
    async def voice_status() -> dict[str, Any]:
        return voice_input.snapshot()

    @app.websocket("/ws")
    async def voice_ws(ws: WebSocket) -> None:
        await ws_manager.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(ws)

    @app.on_event("startup")
    async def preload_voice_input() -> None:
        voice_input.preload()

    @app.on_event("shutdown")
    async def shutdown_voice_input() -> None:
        voice_input.shutdown()

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    logger.info("CAP Voice Input server starting on port %s", VOICE_PORT)
    uvicorn.run(app, host="0.0.0.0", port=VOICE_PORT)


if __name__ == "__main__":
    main()
