# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZMQ policy backend — connects to an Isaac-GR00T–style ZMQ server.

The *server* (Gr00tSimPolicyWrapper) handles all model-specific logic:
  - flat-key → nested-format conversion
  - state normalization / action denormalization
  - diffusion model forward pass
  - relative → absolute action conversion

This client only handles ZMQ transport + batch-dimension bookkeeping.
"""

from __future__ import annotations

import io
from typing import Any

import msgpack
import numpy as np
import zmq

from enpire.env.forge.cap.policy.backend import PolicyBackend

# -- msgpack serialization (matches Isaac-GR00T protocol) -------------------

def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        # msgpack raw=False → str keys; raw=True → bytes keys
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        if b"__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj[b"as_npy"]), allow_pickle=False)
    return obj


class ZMQPolicyBackend(PolicyBackend):
    """Connects to an Isaac-GR00T–style ZMQ REQ/REP policy server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        self._host = host
        self._port = port
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(f"tcp://{host}:{port}")

    # -- PolicyBackend interface ---------------------------------------------

    def predict(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        batched = self._add_batch(obs)
        action, _info = self._rpc("get_action", observation=batched)
        return self._remove_batch(action)

    def predict_batch(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        # obs from MultiStepWrapper + AsyncVectorEnv already has (B, T, H, W, C).
        # No dimension manipulation needed — pass through directly.
        action, _info = self._rpc("get_action", observation=obs)
        return action  # {key: (B, H, D)}

    def reset(self) -> None:
        self._rpc("reset", options=None)

    def close(self) -> None:
        self._sock.close()
        self._ctx.term()

    # -- extra ---------------------------------------------------------------

    def ping(self) -> bool:
        try:
            resp = self._send({"endpoint": "ping"})
            return resp.get("status") == "ok"
        except Exception:
            return False

    # -- transport internals -------------------------------------------------

    def _rpc(self, endpoint: str, **data: Any) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data:
            request["data"] = data
        resp = self._send(request)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return tuple(resp) if isinstance(resp, list) else resp

    def _send(self, request: dict) -> Any:
        self._sock.send(msgpack.packb(request, default=_encode))
        return msgpack.unpackb(self._sock.recv(), object_hook=_decode, raw=False)

    # -- batch helpers -------------------------------------------------------

    @staticmethod
    def _add_batch(obs: dict) -> dict:
        """Add B=1 and T=1 dims for the GR00T server.

        The server (Gr00tSimPolicyWrapper) expects:
          video keys:    (B, T, H, W, C)
          state keys:    (B, T, D)
          language keys: tuple[str] of length B
        """
        out: dict[str, Any] = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                out[k] = v[np.newaxis, np.newaxis]   # (…) → (1, 1, …)
            elif isinstance(v, str):
                out[k] = (v,)                         # B=1 tuple
            else:
                out[k] = v
        return out

    @staticmethod
    def _add_temporal(obs: dict) -> dict:
        """Add T=1 dim only (B already present from AsyncVectorEnv).

        (B, H, W, C) → (B, 1, H, W, C)
        (B, D)       → (B, 1, D)
        tuple[str]   → unchanged (language has no T dim)
        """
        out: dict[str, Any] = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                out[k] = v[:, np.newaxis]     # (B, …) → (B, 1, …)
            else:
                out[k] = v                     # language tuple, pass through
        return out

    @staticmethod
    def _remove_batch(action: dict) -> dict[str, np.ndarray]:
        return {k: v[0] for k, v in action.items()}
