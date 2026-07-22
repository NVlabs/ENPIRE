# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import threading
from pathlib import Path


class OnlineBufferHandshakeServer:
    """ZMQ REP server that serves the current online-buffer directory on request.

    The forge runner owns the recording wrapper and is the source of truth for
    the active online-data-buffer path. Learner processes connect a REQ socket
    on launch and pull the current path. Restarts on the forge side update the
    served value in memory; nothing is pushed to the learner.
    """

    def __init__(self, bind_address: str, initial_path: Path | str):
        import zmq

        self._lock = threading.Lock()
        self._path: str = str(initial_path)
        self._stop = threading.Event()
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(bind_address)
        print(f"[INFO] Online-buffer handshake server bound: {bind_address}")
        self._thread = threading.Thread(
            target=self._serve, name="OnlineBufferHandshake", daemon=True
        )
        self._thread.start()

    def set_path(self, new_path: Path | str) -> None:
        text = str(new_path)
        with self._lock:
            self._path = text
        print(f"[INFO] Online-buffer handshake path updated: {text}", flush=True)

    def stop(self) -> None:
        self._stop.set()

    def _serve(self) -> None:
        import lz4.frame
        import zmq

        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=500)).get(self._sock):
                continue
            try:
                msg = pickle.loads(lz4.frame.decompress(self._sock.recv()))
            except Exception as exc:
                print(f"[HANDSHAKE] decode error: {exc!r}", flush=True)
                self._reply({"ok": False, "error": f"decode error: {exc!s}"})
                continue
            request_type = msg.get("type") if isinstance(msg, dict) else None
            if request_type == "get-online-data-buffer":
                with self._lock:
                    path = self._path
                print(f"[HANDSHAKE] served online buffer path: {path}", flush=True)
                self._reply({"ok": True, "online_data_buffer_path": path})
            else:
                self._reply({"ok": False, "error": f"unknown request: {request_type!r}"})

    def _reply(self, response: dict) -> None:
        import lz4.frame

        try:
            self._sock.send(lz4.frame.compress(pickle.dumps(response)))
        except Exception as exc:
            print(f"[HANDSHAKE] reply failed: {exc!r}", flush=True)

