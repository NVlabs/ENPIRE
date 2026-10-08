# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portal server for an SO-101 arm.

Binds the same RPC names as the YAM follower server, so
``FollowerRobotClient`` can drive an SO-101 without changes.

    uv run python -m enpire.env.forge.robot.so101.arm_server --dry-run
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Literal

from enpire.env.forge.robot.so101.constants import (
    SO101_FOLLOWER_ID,
    SO101_FOLLOWER_PORT,
    SO101_FOLLOWER_SERIAL,
    SO101_LEADER_ID,
    SO101_LEADER_PORT,
    SO101_LEADER_SERIAL,
)

FOLLOWER_RPCS = (
    "get_joint_pos",
    "command_joint_pos",
    "command_joint_state",
    "get_observations",
    "get_motor_temperatures",
)
LEADER_RPCS = ("get_info", "get_motor_temperatures")

_DEFAULTS = {
    "follower": (SO101_FOLLOWER_SERIAL, SO101_FOLLOWER_ID, SO101_FOLLOWER_PORT),
    "leader": (SO101_LEADER_SERIAL, SO101_LEADER_ID, SO101_LEADER_PORT),
}


def _wrap_rpc(label: str, method_name: str, fn: Callable) -> Callable:
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logging.exception("%s RPC %s failed", label, method_name)
            raise

    return wrapped


class So101ArmServer:
    def __init__(self, robot: Any, port: int, rpcs: tuple[str, ...], label: str):
        import portal

        self._label = label
        self._stop = threading.Event()
        self._server = portal.Server(port, errors=False)
        for name in rpcs:
            self._server.bind(name, _wrap_rpc(label, name, getattr(robot, name)))
        if "get_motor_temperatures" in rpcs:
            self._server.bind(
                "get_motor_temps",
                _wrap_rpc(label, "get_motor_temps", robot.get_motor_temperatures),
            )

    def serve(self) -> None:
        self._server.start(block=False)
        try:
            while not self._stop.is_set():
                if not self._server.loop.running:
                    raise RuntimeError(
                        f"{self._label} portal loop exited with exitcode={self._server.loop.exitcode}"
                    )
                if not self._server.socket.thread.running:
                    error = self._server.socket.error
                    raise RuntimeError(f"{self._label} portal socket thread exited") from error
                self._stop.wait(0.2)
        finally:
            if self._server.running:
                self._server.close(timeout=1.0)

    def close(self) -> None:
        self._stop.set()


@dataclass
class Args:
    mode: Literal["follower", "leader"] = "follower"
    serial: str | None = None
    robot_id: str | None = None
    port: int | None = None
    calibration_dir: str | None = None
    max_relative_target: float | None = None
    """Per-step joint change limit in degrees, enforced by LeRobot. Follower only."""
    dry_run: bool = False
    """Print the resolved configuration and exit without opening the serial port."""

    def __post_init__(self) -> None:
        serial, robot_id, port = _DEFAULTS[self.mode]
        self.serial = self.serial or serial
        self.robot_id = self.robot_id or robot_id
        self.port = self.port or port


def describe(args: Args) -> str:
    return (
        f"SO-101 {args.mode} id={args.robot_id} serial={args.serial} "
        f"portal=localhost:{args.port} calibration_dir={args.calibration_dir or 'lerobot default'} "
        f"max_relative_target={args.max_relative_target}"
    )


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print(describe(args))
    if args.dry_run:
        return

    from enpire.env.forge.robot.so101.so101_controller import So101LeaderArm, So101Robot

    if args.mode == "follower":
        device = So101Robot(
            port=args.serial,
            robot_id=args.robot_id,
            calibration_dir=args.calibration_dir,
            max_relative_target=args.max_relative_target,
        )
        rpcs = FOLLOWER_RPCS
    else:
        device = So101LeaderArm(
            port=args.serial,
            robot_id=args.robot_id,
            calibration_dir=args.calibration_dir,
        )
        rpcs = LEADER_RPCS
    device.connect()
    try:
        So101ArmServer(device, args.port, rpcs, f"SO-101 {args.mode}").serve()
    finally:
        try:
            device.disconnect()
        except Exception:
            logging.exception("[%s] failed to release motors on shutdown", args.serial)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
