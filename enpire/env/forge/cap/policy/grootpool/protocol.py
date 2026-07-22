# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire protocol — grootpool middleware (stateless request/response).

N1.5 panda_omron uses observation_indices=[0] (current frame only, no
history), so sessions and stickiness add zero value. The protocol is
now a simple request/response pair:

    step : {op: "step", obs}  →  {op: "step_ok", action}
    ping : {op: "ping"}       →  {op: "pong", inflight, idle_workers,
                                             step_p50_ms, step_p95_ms}

Error response (any op):
    → {op: "error", code, message}

Obs values: numpy ndarrays + plain strings encoded with ndarray-aware
msgpack (np.save → bytes in a dict envelope).
"""

from __future__ import annotations

import io
from typing import Any

import msgpack
import numpy as np


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        if b"__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj[b"as_npy"]), allow_pickle=False)
    return obj


def pack(msg: dict) -> bytes:
    return msgpack.packb(msg, default=_encode, use_bin_type=True)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, object_hook=_decode, raw=False)


ERR_NO_WORKER = "no_worker"
ERR_WORKER_CRASH = "worker_crash"
ERR_WORKER_TIMEOUT = "worker_timeout"
ERR_BAD_REQUEST = "bad_request"
ERR_INTERNAL = "internal"


def error(code: str, message: str) -> dict:
    return {"op": "error", "code": code, "message": message}
