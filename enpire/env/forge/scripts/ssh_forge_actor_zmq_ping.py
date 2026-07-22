#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import subprocess
import time

import zmq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-target", default="forge-actor")
    parser.add_argument("--local-port", type=int, default=15555)
    parser.add_argument("--remote-port", type=int, default=5555)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    tunnel = subprocess.Popen(
        [
            "ssh",
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"{args.local_port}:127.0.0.1:{args.remote_port}",
            args.ssh_target,
        ]
    )

    try:
        time.sleep(1.0)
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, int(args.timeout * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(args.timeout * 1000))
        sock.connect(f"tcp://127.0.0.1:{args.local_port}")
        sock.send(b"ping")
        print(sock.recv().decode("utf-8", errors="replace"))
    finally:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=2)
        except subprocess.TimeoutExpired:
            tunnel.kill()


if __name__ == "__main__":
    main()
