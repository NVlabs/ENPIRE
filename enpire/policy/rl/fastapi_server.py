# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP control surface for the RL runner.

POST ``/home``, ``/pause``, ``/restart``, and ``/resume`` let an autonomous research driver
(or curl/UI client) inject events into the runner's state machine without
sitting at the keyboard. Events are dropped onto a ``queue.Queue`` that the
runner's main loop drains every tick through the RL event router.

``/pause`` follows the keyboard ``KEY_P`` path by enqueueing ``parking``.

``/restart`` is the only endpoint with a side effect: it computes the next
data-buffer folder path, creates it on disk (and copies YAML config files in
for provenance), then enqueues a ``("restart", {"path": <abs>})`` event. The
runner picks that up and swaps ``RecordEpisodeWrapper.output_dir``.
"""

from __future__ import annotations

import queue
import shutil
import threading
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request


def build_app(
    event_queue: "queue.Queue[tuple[str, dict]]",
    data_root: Path,
    config_files: list[Path] | None,
) -> FastAPI:
    """Build the FastAPI app bound to a runner's event queue + data root.

    Args:
        event_queue: thread-safe queue the runner drains every tick.
        data_root: parent directory under which timestamped run folders are
            created on ``/restart`` (typically ``<data_saving_path>/<task_name>``).
        config_files: optional source YAML paths that are copied into each new
            folder so every run is self-describing.
    """
    app = FastAPI(title="RL runner control")
    data_root = Path(data_root)

    help_payload = {
        "endpoints": {
            "POST /home": "enqueue 'home' event — returns runner to home (keyboard 'h')",
            "POST /resume": (
                "enqueue 'start' event — transitions idle/home -> hover (keyboard 's')"
            ),
            "POST /pause": "enqueue 'parking' event — stop rollout and park (keyboard 'p')",
            "POST /restart": (
                "rotate output dir to a fresh timestamp and enqueue 'restart' "
                "(keyboard F5); returns {'path': <abs>}"
            ),
            "GET  /healthz": "liveness probe — returns {'ok': True}",
            "GET|POST /help": "this message",
        },
        "examples": [
            "curl -X POST http://127.0.0.1:8203/home",
            "curl -X POST http://127.0.0.1:8203/pause",
            "curl -X POST http://127.0.0.1:8203/resume",
            "curl -X POST http://127.0.0.1:8203/restart",
            "curl http://127.0.0.1:8203/healthz",
        ],
    }

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/help")
    @app.post("/help")
    def help_() -> dict:
        return help_payload

    @app.post("/home")
    def home(request: Request) -> dict:
        client = request.client.host if request.client else "unknown"
        event_queue.put(("home", {"source": "fastapi", "client": client}))
        print(
            f"\033[1;36m[FastAPI] /home -> enqueued 'home' "
            f"client={client}\033[0m",
            flush=True,
        )
        return {"message": "Request Received"}

    @app.post("/restart")
    def restart() -> dict:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        new_dir = data_root / timestamp
        # If two restarts land in the same second, give the second one a suffix.
        suffix = 1
        candidate = new_dir
        while candidate.exists():
            candidate = data_root / f"{timestamp}-{suffix}"
            suffix += 1
        new_dir = candidate
        new_dir.mkdir(parents=True, exist_ok=True)

        for src in config_files or []:
            if src.exists():
                shutil.copy(src, new_dir / src.name)

        event_queue.put(("restart", {"path": str(new_dir), "source": "fastapi"}))
        print(f"\n\033[1;36m[FastAPI] /restart -> {new_dir}\033[0m", flush=True)
        return {"path": str(new_dir)}

    @app.post("/pause")
    def pause(request: Request) -> dict:
        client = request.client.host if request.client else "unknown"
        event_queue.put(("parking", {"source": "fastapi", "client": client}))
        print(
            f"\033[1;36m[FastAPI] /pause -> enqueued 'parking' "
            f"client={client}\033[0m",
            flush=True,
        )
        return {"message": "Request Received"}

    @app.post("/resume")
    def resume(request: Request) -> dict:
        client = request.client.host if request.client else "unknown"
        event_queue.put(("start", {"source": "fastapi", "client": client}))
        print(
            f"\033[1;36m[FastAPI] /resume -> enqueued 'start' "
            f"client={client}\033[0m",
            flush=True,
        )
        return {"message": "Request Received"}

    return app


def start_in_thread(
    host: str,
    port: int,
    event_queue: "queue.Queue[tuple[str, dict]]",
    data_root: Path,
    config_files: list[Path] | None = None,
) -> threading.Thread:
    """Launch uvicorn on a daemon thread so the runner's sync loop is unaffected.

    Disables uvicorn's signal handlers (they only work on the main thread) so
    Ctrl-C still propagates through to the runner.
    """
    app = build_app(
        event_queue=event_queue,
        data_root=data_root,
        config_files=config_files,
    )
    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None

    thread = threading.Thread(target=server.run, name="rl-fastapi-server", daemon=True)
    thread.start()
    print(f"[INFO] RL FastAPI server listening on http://{host}:{port}", flush=True)
    return thread
