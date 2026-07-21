"""Six-axis calibration arm server, adapted from yam-calibration/arm_server.py."""

from __future__ import annotations

import argparse
import logging
import threading
import time

import portal

from enpire.env.forge.robot.yam.yam_controller import YamRobot

from . import config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--confirm-motion", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_motion:
        raise RuntimeError("Pass --confirm-motion after clearing the robot workspace.")

    can_interface = config.CAN_INTERFACE if args.side == "left" else config.CAN_INTERFACE_RIGHT
    port = config.ARM_SERVER_PORT if args.side == "left" else config.ARM_SERVER_PORT_RIGHT
    robot = YamRobot(
        can_interface=can_interface,
        motor_ids=config.MOTOR_IDS,
        motor_types=config.MOTOR_TYPES,
        default_kp=config.ARM_KP,
        default_kd=config.ARM_KD,
        bustype="socketcan",
        model_xml_path=config.GRAVITY_COMP_XML,
    )
    robot.connect()
    shutdown = threading.Event()
    server = portal.Server(port, errors=False)
    server.bind("get_joint_pos", robot.get_joint_pos)
    server.bind("command_joint_pos", robot.command_joint_pos)
    server.bind("get_observations", robot.get_observations)
    server.bind("shutdown", shutdown.set)
    server.start(block=False)
    print(f"READY side={args.side} port={port}", flush=True)
    try:
        while not shutdown.is_set():
            time.sleep(0.2)
    finally:
        server.close(timeout=1.0)
        robot.disconnect()
        logging.info("Calibration arm server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
