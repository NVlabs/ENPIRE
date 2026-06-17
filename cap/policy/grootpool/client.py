"""Stateless client for the grootpool middleware.

N1.5 with panda_omron uses observation_indices=[0] — one frame in,
16-step action chunk out, no history. No session needed.

Usage:
    client = GrootPoolClient()
    action = client.predict(obs)
    client.close()

Or as a context manager:
    with GrootPoolClient() as client:
        action = client.predict(obs)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np
import zmq

from cap.policy.grootpool import protocol as P

log = logging.getLogger(__name__)


class GrootPoolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def _default_endpoint() -> str:
    return os.environ.get("GROOTPOOL_ENDPOINT", "tcp://127.0.0.1:7070")


class GrootPoolClient:
    """Stateless msgpack-over-ZMQ client. One predict() = one worker round-trip."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        recv_timeout_ms: int = 3_900_000,
    ) -> None:
        self._endpoint = endpoint or _default_endpoint()
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self._sock.connect(self._endpoint)
        self._lock = threading.Lock()

    def predict(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Send obs → get action. Blocks until worker responds."""
        with self._lock:
            self._sock.send_multipart([P.pack({"op": "step", "obs": obs})])
            parts = self._sock.recv_multipart()
        resp = P.unpack(parts[0])
        if resp.get("op") == "error":
            raise GrootPoolError(resp["code"], resp["message"])
        if resp.get("op") != "step_ok":
            raise GrootPoolError(P.ERR_INTERNAL, f"unexpected response: {resp}")
        return resp.get("action", {})

    def ping(self) -> dict:
        with self._lock:
            self._sock.send_multipart([P.pack({"op": "ping"})])
            parts = self._sock.recv_multipart()
        return P.unpack(parts[0])

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:
            pass

    def __enter__(self) -> "GrootPoolClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()


__all__ = ["GrootPoolClient", "GrootPoolError"]
