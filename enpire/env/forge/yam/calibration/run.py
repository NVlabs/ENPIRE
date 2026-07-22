# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe launcher for one source-faithful YAM calibration pass."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
from contextlib import closing

from . import config


def _port_open(port: int) -> bool:
    with closing(socket.socket()) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def run_calibration(
    *,
    camera: str,
    resolution: str | None,
    launch_server: bool,
    confirm_motion: bool,
) -> int:
    if not confirm_motion:
        raise RuntimeError("Pass --confirm-motion after clearing the robot workspace.")
    side = "right" if camera == "right_wrist" else "left"
    port = config.ARM_SERVER_PORT_RIGHT if side == "right" else config.ARM_SERVER_PORT
    process: subprocess.Popen | None = None
    if launch_server and not _port_open(port):
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "enpire.env.forge.yam.calibration.server",
                "--side",
                side,
                "--confirm-motion",
            ],
            env=os.environ.copy(),
            start_new_session=True,
        )

    argv = ["--camera", camera, "--no-interactive", "--confirm-motion"]
    if resolution is not None:
        argv.extend(["--resolution", resolution])
    try:
        from .calibrator import main

        return int(main(argv))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=3.0)
