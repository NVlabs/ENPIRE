"""Zero-overhead UDP diagnostics emitter (bc_policy copy).

Fire-and-forget: if no dashboard is listening, packets are silently dropped.
Typical sendto() overhead is ~1μs. Disable entirely with DIAG_ENABLED=0.

Usage:
    from diag_emitter import emit
    emit("rl_policy_server", "step_recv", ep=3, step=142)
"""

from __future__ import annotations

import os
import socket
import time

import msgpack

_ENABLED = os.environ.get("DIAG_ENABLED", "1") != "0"
_DIAG_HOST = os.environ.get("DIAG_HOST", "127.0.0.1")
_DIAG_PORT = int(os.environ.get("DIAG_PORT", "9999"))
_ADDR = (_DIAG_HOST, _DIAG_PORT)

_sock: socket.socket | None = None
if _ENABLED:
    _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _sock.setblocking(False)


def emit(src: str, event: str, ep: int = 0, step: int = 0, **meta) -> None:
    if _sock is None:
        return
    try:
        _sock.sendto(
            msgpack.packb(
                {
                    "src": src,
                    "event": event,
                    "ep": ep,
                    "step": step,
                    "t": time.time(),
                    "meta": meta,
                },
                use_bin_type=True,
            ),
            _ADDR,
        )
    except Exception:
        pass
