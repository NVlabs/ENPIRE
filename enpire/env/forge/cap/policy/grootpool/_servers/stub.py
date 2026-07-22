# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub GR00T server — speaks the N1.5 torch-over-ZMQ protocol, returns zeros.

Only requires ``torch`` + ``pyzmq``; no ``gr00t`` / ``flash-attn`` / GPU. Used by
``cap.policy.grootpool.server --stub`` to validate the full subprocess + ZMQ
path end-to-end without building ``Isaac-GR00T-benchmark/model_server_venv``.

Wire protocol (matches ``gr00t.eval.robot.RobotInferenceServer``):

    Request:  torch.save({"endpoint": <str>, "data": <dict>}, buf).getvalue()
    Response:
      - ping:        torch.save({"status": "ok"})
      - get_action:  torch.save((action_dict, info_dict))
      - reset:       torch.save({"status": "ok"})
"""

from __future__ import annotations

import argparse
import io
import sys
import time

import numpy as np
import torch
import zmq


def _fake_action(batch_size: int = 1, horizon: int = 16) -> dict:
    return {
        "action.joint_position": np.zeros((batch_size, horizon, 7), dtype=np.float32),
        "action.gripper": np.zeros((batch_size, horizon, 1), dtype=np.float32),
    }


def _batch_size_of(obs: dict) -> int:
    """Infer the batch dim. Middleware client always sends B=1, but robust to B>1."""
    for v in obs.values():
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            return int(v.shape[0])
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description="grootpool stub GR00T worker")
    p.add_argument("--port", type=int, required=True)
    p.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Accepted for CLI compatibility; ignored.",
    )
    p.add_argument(
        "--embodiment-tag",
        type=str,
        default="new_embodiment",
        help="Accepted for CLI compatibility; ignored.",
    )
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument(
        "--latency-ms",
        type=int,
        default=10,
        help="Simulated inference latency per get_action call.",
    )
    args = p.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[stub] listening on tcp://0.0.0.0:{args.port}", flush=True)

    try:
        while True:
            raw = sock.recv()
            req = torch.load(io.BytesIO(raw), weights_only=False)
            endpoint = req.get("endpoint", "") if isinstance(req, dict) else ""

            if endpoint == "ping":
                resp: object = {"status": "ok"}
            elif endpoint == "reset":
                resp = {"status": "ok"}
            elif endpoint == "get_action":
                if args.latency_ms:
                    time.sleep(args.latency_ms / 1000.0)
                data = req.get("data") or {}
                obs = data.get("observation") or {}
                b = _batch_size_of(obs) if isinstance(obs, dict) else 1
                resp = (_fake_action(b, args.horizon), {})
            else:
                resp = {"error": f"unknown endpoint: {endpoint}"}

            buf = io.BytesIO()
            torch.save(resp, buf)
            sock.send(buf.getvalue())
    except KeyboardInterrupt:
        print("[stub] shutting down", flush=True)
    finally:
        sock.close(linger=0)
        ctx.term()
    return 0


if __name__ == "__main__":
    sys.exit(main())
