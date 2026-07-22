# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI admin endpoint — GET /status."""

from __future__ import annotations

import asyncio
import logging

from enpire.env.forge.cap.policy.grootpool.session_manager import RequestDispatcher
from enpire.env.forge.cap.policy.grootpool.supervisor import WorkerSupervisor

log = logging.getLogger("grootpool.admin")


async def run_admin_server(
    supervisor: WorkerSupervisor,
    dispatcher: RequestDispatcher,
    port: int,
    shutdown: asyncio.Event,
) -> None:
    try:
        from fastapi import FastAPI
        import uvicorn
    except ImportError:
        log.warning("fastapi/uvicorn not installed; admin endpoint disabled")
        await shutdown.wait()
        return

    app = FastAPI(title="grootpool admin", docs_url=None, redoc_url=None)

    @app.get("/status")
    def status():
        snap = dispatcher.status_snapshot()
        return {
            "workers": supervisor.status_snapshot(),
            **snap,
        }

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

    server_task = asyncio.create_task(server.serve())
    try:
        await shutdown.wait()
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=5.0)
        except asyncio.TimeoutError:
            log.warning("admin server did not shut down cleanly")
