#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from typing import List

import numpy as np

from enpire.env.forge.robot.fello.fello import FelloRobot

DEFAULT_KP = [3.0, 10.0, 10.0, 3.0, 3.0, 3.0, 3.0]
DEFAULT_KD = [.5, 1.0, 1.0, .5, .5, .5, .5]


def _parse_gains(values: List[float], name: str) -> np.ndarray:
    if len(values) == 1:
        return np.full(7, values[0], dtype=np.float32)
    if len(values) == 7:
        return np.asarray(values, dtype=np.float32)
    raise ValueError(f"{name} must have 1 or 7 values, got {len(values)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Command joint positions on the Fello robot.")
    parser.add_argument(
        "--can-interface",
        type=str,
        default="can0",
        help="CAN interface name (default: can0)",
    )
    parser.add_argument(
        "--pos",
        type=float,
        nargs=7,
        help="Target joint positions in radians (7 values).",
    )
    parser.add_argument(
        "--sin-amplitude",
        type=float,
        default=None,
        help="Sine wave amplitude in radians (enables sine mode when set).",
    )
    parser.add_argument(
        "--sin-offset",
        type=float,
        default=0.0,
        help="Sine wave offset in radians (default: 0.0).",
    )
    parser.add_argument(
        "--sin-freq-hz",
        type=float,
        default=0.5,
        help="Sine wave frequency in Hz (default: 0.5).",
    )
    parser.add_argument(
        "--kp",
        type=float,
        nargs="+",
        default=DEFAULT_KP,
        help="Joint stiffness gains (1 value for all joints or 7 values).",
    )
    parser.add_argument(
        "--kd",
        type=float,
        nargs="+",
        default=DEFAULT_KD,
        help="Joint damping gains (1 value for all joints or 7 values).",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=100.0,
        help="Command rate in Hz (default: 100).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds to hold the command (default: run until Ctrl-C).",
    )
    parser.add_argument(
        "--gripper-mode",
        type=str,
        choices=["disabled", "linear_spring", "gas_spring"],
        default="disabled",
        help="Gripper torque helper mode (default: disabled).",
    )
    parser.add_argument(
        "--gripper-spring-constant",
        type=float,
        default=0.0,
        help="Spring constant for linear_spring mode in Nm/rad.",
    )
    parser.add_argument(
        "--gripper-spring-rest-position",
        type=float,
        default=0.0,
        help="Rest position for linear_spring mode in radians.",
    )
    parser.add_argument(
        "--gripper-gas-spring-torque",
        type=float,
        default=0.0,
        help="Constant torque for gas_spring mode in Nm.",
    )

    args = parser.parse_args()

    kp = _parse_gains(args.kp, "kp")
    kd = _parse_gains(args.kd, "kd")
    if args.sin_amplitude is None and args.pos is None:
        raise ValueError("Provide --pos (7 values) or set --sin-amplitude.")
    if args.sin_amplitude is not None and args.pos is not None:
        raise ValueError("Use either --pos or --sin-amplitude, not both.")

    target_pos = None
    if args.pos is not None:
        target_pos = np.asarray(args.pos, dtype=np.float32)

    robot = FelloRobot(
        can_interface=args.can_interface,
        gripper_mode=args.gripper_mode,
        gripper_spring_constant=args.gripper_spring_constant,
        gripper_spring_rest_position=args.gripper_spring_rest_position,
        gripper_gas_spring_torque=args.gripper_gas_spring_torque,
    )

    if not robot.connect():
        raise RuntimeError("Failed to connect to Fello robot.")

    period = 1.0 / float(args.rate_hz)
    start_time = time.time()

    try:
        while True:
            if args.sin_amplitude is not None:
                t = time.time() - start_time
                angle = args.sin_offset + args.sin_amplitude * np.sin(2.0 * np.pi * args.sin_freq_hz * t)
                target_pos = np.full(7, angle, dtype=np.float32)
            robot.command_joint_pos(target_pos, kp=kp, kd=kd, use_gravity_comp=True)
            if args.duration is not None and (time.time() - start_time) >= args.duration:
                break
            time.sleep(period)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()

