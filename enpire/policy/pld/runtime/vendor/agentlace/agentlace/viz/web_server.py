# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal web server for the VizDashboard.

- GET /         → serves frontend.html
- GET /events   → SSE stream (JSON snapshot every 1s)
- GET /api/snapshot → one-shot JSON snapshot

Uses only stdlib: http.server, threading. Zero pip dependencies.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

_FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "frontend.html")


class _Handler(BaseHTTPRequestHandler):
    """HTTP request handler for the viz dashboard."""

    # Suppress per-request log lines
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/events":
            self._serve_sse()
        elif self.path == "/api/snapshot":
            self._serve_snapshot()
        else:
            self.send_error(404)

    def _serve_html(self):
        try:
            with open(_FRONTEND_PATH, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(500, "frontend.html not found")

    def _serve_snapshot(self):
        snapshot_fn = self.server.snapshot_fn
        data = snapshot_fn()
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        snapshot_fn = self.server.snapshot_fn
        stop_event = self.server.stop_event

        try:
            while not stop_event.is_set():
                data = snapshot_fn()
                msg = f"data: {json.dumps(data)}\n\n"
                self.wfile.write(msg.encode("utf-8"))
                self.wfile.flush()
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Client disconnected


class VizWebServer:
    """Thin wrapper that runs ThreadingHTTPServer in a daemon thread."""

    def __init__(self, port: int, snapshot_fn: Callable[[], dict]):
        self._port = port
        self._stop_event = threading.Event()
        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
        self._httpd.snapshot_fn = snapshot_fn
        self._httpd.stop_event = self._stop_event
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._httpd.shutdown()

