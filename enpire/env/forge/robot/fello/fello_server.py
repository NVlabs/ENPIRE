# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Server entrypoint for Fello leader arm.

This wraps the reusable robot logic in fello.py with a Portal RPC server and a
background gravity-compensation loop.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import dataclasses
import logging
import signal
import threading
import time
from dataclasses import field
from typing import Annotated, Literal, Optional, Sequence

import numpy as np
import portal
import tyro

from enpire.env.forge.robot.constants import (
    CAN_BUSTYPE,
    LEFT_FELLO_CAN_INTERFACE,
    LEFT_LEADER_PORT,
    RIGHT_FELLO_CAN_INTERFACE,
    RIGHT_LEADER_PORT,
)
from enpire.env.forge.robot.fello import fello as fello_module
from enpire.env.forge.robot.fello.fello import FelloLeaderRobot, FelloRobot
from enpire.env.forge.robot.fello.fello_config import (
    get_config_value,
    get_fello_xml_path,
    load_fello_config,
)


def _get_requested_side_from_argv(argv: Sequence[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--side" and i + 1 < len(argv):
            return argv[i + 1].lower()
        if arg.startswith("--side="):
            return arg.split("=", 1)[1].lower()
    return None

class FelloLeaderRobotServer:
    """Portal RPC server for Fello leader robot."""

    def __init__(
        self,
        robot: FelloLeaderRobot,
        port: int,
        mode_state: dict,
        feedback_state: dict,
        arm_sign: Sequence[float],
        joint_position_ratio: Sequence[float],
        gripper_sign: float,
        gripper_range_rad: float,
        gripper_close_pos_rad: float,
    ):
        self._robot = robot
        self._mode_state = mode_state
        self._feedback_state = feedback_state
        self._arm_sign = np.asarray(arm_sign, dtype=np.float32)
        self._joint_position_ratio = np.asarray(joint_position_ratio, dtype=np.float32)
        self._gripper_sign = gripper_sign
        self._gripper_range_rad = gripper_range_rad
        self._gripper_close_pos_rad = gripper_close_pos_rad
        self._server = portal.Server(port)
        self._server.bind("get_info", self._get_info)
        # Only expose command_joint_state for safety (requires explicit gains)
        self._server.bind("command_joint_state", self._command_joint_state)
        self._server.bind("get_target_pos", self._robot.get_target_pos)
        self._server.bind("command_joint_pos", self._command_joint_pos)
        self._server.bind("set_mode", self.set_mode)
        self._server.bind("set_force_feedback_torques", self._set_force_feedback_torques)
        self._server.bind("get_motor_temperatures", self._robot.get_motor_temperatures)
        self._server.bind("get_motor_temps", self._robot.get_motor_temperatures)

    def serve(self) -> None:
        """Start the Portal server."""
        self._server.start()

    def set_mode(self, mode: str) -> None:
        if mode not in ("gravity", "position"):
            raise ValueError(f"Unsupported mode: {mode}")
        with self._mode_state["lock"]:
            self._mode_state["mode"] = mode
        logging.info(f"[FelloServer] mode={mode}")

    def _map_real_to_sim(self, values: np.ndarray) -> np.ndarray:
        mapped = np.asarray(values, dtype=np.float32).copy()
        if mapped.shape[0] >= 6:
            mapped[:6] *= self._arm_sign
        if mapped.shape[0] >= 7:
            mapped[6] = (mapped[6] - self._gripper_close_pos_rad) * self._gripper_sign / self._gripper_range_rad
            mapped[6] = float(np.clip(mapped[6], 0.0, 1.0))
        mapped *= self._joint_position_ratio[: mapped.shape[0]]
        return mapped

    def _map_sim_to_real(self, values: np.ndarray) -> np.ndarray:
        mapped = np.asarray(values, dtype=np.float32).copy()
        mapped /= self._joint_position_ratio[: mapped.shape[0]]
        if mapped.shape[0] >= 6:
            mapped[:6] *= self._arm_sign
        mapped[6] = self._gripper_close_pos_rad + (mapped[6] * self._gripper_range_rad * self._gripper_sign)
        return mapped

    def _map_sim_to_real_torques(self, values: np.ndarray) -> np.ndarray:
        mapped = np.asarray(values, dtype=np.float32).copy()
        if mapped.shape[0] >= 6:
            mapped[:6] *= self._arm_sign
        if mapped.shape[0] >= 7:
            mapped[6] *= self._gripper_sign
        return mapped

    _get_info_count = 0

    def _get_info(self) -> tuple[np.ndarray, np.ndarray]:
        qpos, buttons = self._robot.get_info()
        self._get_info_count += 1
        mapped = self._map_real_to_sim(qpos)
        if self._get_info_count % 30 == 0:
            logging.info(
                f"[get_info] raw_grip={qpos[6]:.4f} mapped_grip={mapped[6]:.4f} buttons={buttons}"
            )
        return mapped, buttons

    def _command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        joint_state = {
            **joint_state,
            "pos": self._map_sim_to_real(joint_state["pos"]),
            "vel": self._map_sim_to_real(joint_state["vel"]),
        }
        self._robot.command_joint_state(joint_state)

    def _command_joint_pos(self, joint_state: dict[str, np.ndarray]) -> None:
        joint_state = {
            **joint_state,
            "pos": self._map_sim_to_real(joint_state["pos"]),
        }
        self._robot.set_target_pos(joint_state)

    def _set_force_feedback_torques(self, torques_nm: np.ndarray) -> None:
        torques_nm = np.asarray(torques_nm, dtype=np.float32).reshape(-1)
        if torques_nm.size != 7:
            raise ValueError(f"Expected 7 force-feedback torques, got {torques_nm.size}")
        mapped = self._map_sim_to_real_torques(torques_nm)
        with self._feedback_state["lock"]:
            self._feedback_state["torques"] = mapped
            self._feedback_state["active"] = True


_CAN_FAIL_THRESHOLD = 3  # consecutive ticks with failures before reconnect


def start_control_loop(
    robot: FelloRobot,
    running_evt: threading.Event,
    mode_state: dict,
    feedback_state: dict,
    freq_hz: int,
) -> threading.Thread:
    """Start a background thread that continuously sends gravity-comp commands."""

    def control_loop():
        period = 1.0 / freq_hz
        consec_fail_ticks = 0
        logging.info(f"Starting gravity compensation control loop at {freq_hz} Hz")
        while running_evt.is_set():
            try:
                with mode_state["lock"]:
                    mode = mode_state["mode"]
                with feedback_state["lock"]:
                    feedback_torques = feedback_state["torques"].copy()
                    feedback_active = bool(feedback_state["active"])
                failures = 0
                if mode == "gravity":
                    if feedback_active:
                        failures = robot.send_gravity_compensation_and_feedback(
                            feedback_torques
                        )
                    else:
                        failures = robot.send_gravity_compensation_only()
                elif mode == "position":
                    pos, kp, kd = robot.get_last_commanded_state()
                    if pos is not None and kp is not None and kd is not None:
                        failures = robot.command_joint_pos(pos, kp=kp, kd=kd, use_gravity_comp=True)

                if failures > 0:
                    consec_fail_ticks += 1
                    if consec_fail_ticks >= _CAN_FAIL_THRESHOLD:
                        logging.error(
                            f"[control_loop] {consec_fail_ticks} consecutive ticks with CAN failures — attempting reconnect"
                        )
                        if robot.reconnect():
                            consec_fail_ticks = 0
                        else:
                            logging.error("[control_loop] Reconnect failed — will retry next tick")
                            time.sleep(1.0)
                else:
                    consec_fail_ticks = 0

                time.sleep(period)
            except Exception as e:
                logging.error(f"Error in control loop: {e}")
                time.sleep(period)

    thread = threading.Thread(target=control_loop, daemon=True)
    thread.start()
    return thread


def start_viewer_loop(
    robot: FelloRobot,
    running_evt: threading.Event,
    freq_hz: int,
    fello_xml_path: Path,
) -> threading.Thread:
    """Start a background thread that mirrors real robot joints into a MuJoCo viewer."""

    def viewer_loop():
        import mujoco
        import mujoco.viewer

        if not fello_xml_path.exists():
            logging.error(f"Fello XML file not found at {fello_xml_path}")
            return

        model = mujoco.MjModel.from_xml_path(str(fello_xml_path))
        data = mujoco.MjData(model)
        joint_addrs = []
        for i in range(7):
            joint = model.joint(f"joint_{i+1}")
            joint_addrs.append(int(joint.qposadr))

        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            period = 1.0 / freq_hz
            logging.info(f"MuJoCo viewer started at {freq_hz} Hz")
            while running_evt.is_set() and viewer.is_running():
                joint_pos = robot.get_joint_pos()
                if joint_pos is not None:
                    for i, addr in enumerate(joint_addrs):
                        data.qpos[addr] = float(joint_pos[i])
                    mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(period)

    thread = threading.Thread(target=viewer_loop, daemon=True)
    thread.start()
    return thread


def main() -> None:
    requested_side = _get_requested_side_from_argv(sys.argv[1:])
    cfg = load_fello_config(side=requested_side)
    control_cfg = get_config_value(cfg, "control", default={}) or {}
    server_cfg = get_config_value(cfg, "server", default={}) or {}
    mapping_cfg = get_config_value(cfg, "mapping", default={}) or {}

    @dataclasses.dataclass
    class Args:
        can_interface: Annotated[
            Optional[str], tyro.conf.arg(help="CAN interface name")
        ] = get_config_value(cfg, "hardware", "can_interface")
        port: Annotated[Optional[int], tyro.conf.arg(help="Server port")] = None
        side: Annotated[Optional[str], tyro.conf.arg(help="Side of leader arm [left or right]")] = None
        motor_ids: Annotated[
            Optional[list[int]], tyro.conf.arg(help="Motor CAN IDs (7 values)")
        ] = field(default_factory=lambda: get_config_value(cfg, "hardware", "motor_ids"))
        feedback_ids: Annotated[
            Optional[list[int]],
            tyro.conf.arg(help="Feedback IDs (7 values; default: same as motor-ids)"),
        ] = field(default_factory=lambda: get_config_value(cfg, "hardware", "feedback_ids"))
        motor_types: Annotated[
            Optional[list[str]], tyro.conf.arg(help="Motor types (7 values)")
        ] = field(default_factory=lambda: get_config_value(cfg, "hardware", "motor_types"))
        gravity_coefficients: Annotated[
            Optional[list[float]],
            tyro.conf.arg(help="Gravity compensation coefficients (7 values)"),
        ] = field(default_factory=lambda: get_config_value(cfg, "hardware", "gravity_coefficients"))
        debug: Annotated[
            Optional[bool],
            tyro.conf.arg(help="Enable debug mode to print motor states in table format"),
        ] = server_cfg.get("debug")
        gripper_mode: Annotated[
            Optional[Literal["disabled", "linear_spring", "gas_spring"]],
            tyro.conf.arg(help="Gripper (7th motor) control mode"),
        ] = control_cfg.get("gripper_mode")
        gripper_spring_constant: Annotated[
            Optional[float],
            tyro.conf.arg(help="Spring constant for linear_spring mode in Nm/rad"),
        ] = control_cfg.get("gripper_spring_constant")
        gripper_spring_rest_position: Annotated[
            Optional[float],
            tyro.conf.arg(help="Rest position for linear_spring mode in radians"),
        ] = control_cfg.get("gripper_spring_rest_position")
        gripper_gas_spring_torque: Annotated[
            Optional[float],
            tyro.conf.arg(help="Constant torque for gas_spring mode in Nm"),
        ] = control_cfg.get("gripper_gas_spring_torque")
        gravity_mode_kd: Annotated[
            Optional[list[float]],
            tyro.conf.arg(help="Gravity-mode damping gains (7 values)"),
        ] = field(default_factory=lambda: get_config_value(cfg, "control", "gravity_mode_kd"))
        k_fric: Annotated[
            Optional[list[float]],
            tyro.conf.arg(help="Gravity-mode viscous friction compensation gains (7 values)"),
        ] = field(default_factory=lambda: get_config_value(cfg, "control", "k_fric"))
        viewer: Annotated[
            Optional[bool],
            tyro.conf.arg(help="Launch MuJoCo viewer that mirrors real robot joint states"),
        ] = server_cfg.get("viewer")
        viewer_freq: Annotated[
            Optional[int], tyro.conf.arg(help="Viewer update frequency in Hz")
        ] = server_cfg.get("viewer_freq")
        control_loop_hz: Annotated[
            Optional[int], tyro.conf.arg(help="Control loop frequency in Hz")
        ] = server_cfg.get("control_loop_hz")
        initial_mode: Annotated[
            Optional[str], tyro.conf.arg(help="Initial control loop mode")
        ] = server_cfg.get("initial_mode") or server_cfg.get("initial-mode")
        print_footswitch: Annotated[
            bool, tyro.conf.arg(help="Print footswitch button states to confirm detection")
        ] = False

    args = tyro.cli(Args)

    if args.side is None:
        raise ValueError("Missing side; expected 'left' or 'right'")

    args.side = args.side.lower()
    fello_xml_path = get_fello_xml_path(cfg, args.side)

    if args.port is None:
        if args.side == "right":
            args.port = RIGHT_LEADER_PORT
        elif args.side == "left":
            args.port = LEFT_LEADER_PORT
        else:
            raise ValueError(f"{args.side} not available")

    def _require_length(name: str, values: Optional[list[object]], length: int = 7) -> None:
        if values is None:
            return
        if len(values) != length:
            raise ValueError(f"{name} must have {length} values, got {len(values)}")

    _require_length("motor_ids", args.motor_ids)
    _require_length("feedback_ids", args.feedback_ids)
    _require_length("motor_types", args.motor_types)
    _require_length("gravity_coefficients", args.gravity_coefficients)
    _require_length("gravity_mode_kd", args.gravity_mode_kd)
    _require_length("k_fric", args.k_fric)
    arm_sign = mapping_cfg.get("arm_sign")
    joint_position_ratio = mapping_cfg.get("joint_position_ratio", [1.0] * 7)
    gripper_sign = mapping_cfg.get("gripper_sign")
    gripper_range_rad = mapping_cfg.get("gripper_range_rad")
    gripper_close_pos_rad = mapping_cfg.get("gripper_close_pos_rad")
    if arm_sign is None:
        raise ValueError("Missing mapping.arm_sign in fello_config.yaml")
    if len(arm_sign) != 6:
        raise ValueError(f"mapping.arm_sign must have 6 values, got {len(arm_sign)}")
    if len(joint_position_ratio) != 7:
        raise ValueError(
            f"mapping.joint_position_ratio must have 7 values, got {len(joint_position_ratio)}"
        )
    joint_position_ratio_arr = np.asarray(joint_position_ratio, dtype=np.float32)
    if np.any(~np.isfinite(joint_position_ratio_arr)):
        raise ValueError("mapping.joint_position_ratio must be finite")
    if np.any(joint_position_ratio_arr == 0.0):
        raise ValueError("mapping.joint_position_ratio values must be non-zero")
    if gripper_sign is None:
        raise ValueError("Missing mapping.gripper_sign in fello_config.yaml")
    if gripper_range_rad is None:
        raise ValueError("Missing mapping.gripper_range_rad in fello_config.yaml")
    if gripper_close_pos_rad is None:
        raise ValueError("Missing mapping.gripper_close_pos_rad in fello_config.yaml")
    if args.port is None:
        raise ValueError("Missing server.port in fello_config.yaml")
    if args.control_loop_hz is None:
        raise ValueError("Missing server.control_loop_hz in fello_config.yaml")

    if not fello_xml_path.exists():
        logging.error(f"Fello XML file not found at {fello_xml_path}")
        return

    fello_module.FELLO_XML_PATH = fello_xml_path

    # Resolve CAN channel: on macOS uses gs_usb serial number, on Linux uses socketcan name
    can_channel = args.can_interface
    if CAN_BUSTYPE == "gs_usb":
        fello_can = RIGHT_FELLO_CAN_INTERFACE if args.side == "right" else LEFT_FELLO_CAN_INTERFACE
        if fello_can is None:
            raise RuntimeError(
                f"CAN interface not configured for {args.side} fello — "
                f"set {args.side.upper()}_LEADER_CAN_INTERFACE in robot/constants.py"
            )
        can_channel = fello_can

    logging.info(f"Initializing Fello robot on CAN channel: {can_channel} (bustype={CAN_BUSTYPE})")
    logging.info(f"Using Fello MuJoCo model: {fello_xml_path}")
    if args.debug:
        logging.info("Debug mode enabled - motor states will be printed in table format at 10 Hz")

    robot = FelloRobot(
        can_interface=can_channel,
        motor_ids=args.motor_ids,
        feedback_ids=args.feedback_ids,
        motor_types=args.motor_types,
        gravity_coefficients=args.gravity_coefficients,
        debug=args.debug,
        gripper_mode=args.gripper_mode,
        gripper_spring_constant=args.gripper_spring_constant,
        gripper_spring_rest_position=args.gripper_spring_rest_position,
        gripper_gas_spring_torque=args.gripper_gas_spring_torque,
        gravity_mode_kd=args.gravity_mode_kd,
        k_fric=args.k_fric,
        bustype=CAN_BUSTYPE,
        side=args.side,
    )

    if not robot.connect():
        logging.error("Failed to connect to Fello robot")
        return

    leader_robot = FelloLeaderRobot(
        robot,
        side=args.side,
    )

    if args.initial_mode is None:
        raise ValueError("Missing server.initial_mode in fello_config.yaml")

    control_loop_running = threading.Event()
    control_loop_running.set()
    mode_state = {"mode": args.initial_mode, "lock": threading.Lock()}
    feedback_state = {
        "torques": np.zeros(7, dtype=np.float32),
        "active": False,
        "lock": threading.Lock(),
    }
    control_thread = start_control_loop(
        robot,
        control_loop_running,
        mode_state,
        feedback_state,
        freq_hz=args.control_loop_hz,
    )
    footswitch_thread = None
    if args.print_footswitch:
        def _footswitch_loop() -> None:
            logging.info("Footswitch debug enabled; printing button states.")
            while control_loop_running.is_set():
                try:
                    _, buttons = leader_robot.get_info()
                    logging.info(f"Footswitch buttons: {buttons}")
                except Exception as e:
                    logging.error(f"Footswitch read error: {e}")
                time.sleep(0.2)

        footswitch_thread = threading.Thread(target=_footswitch_loop, daemon=True)
        footswitch_thread.start()
    viewer_thread = None
    if args.viewer:
        viewer_thread = start_viewer_loop(
            robot,
            control_loop_running,
            freq_hz=args.viewer_freq,
            fello_xml_path=fello_xml_path,
        )

    logging.info(f"Starting Fello leader server on localhost:{args.port}")
    server = FelloLeaderRobotServer(
        leader_robot,
        args.port,
        mode_state,
        feedback_state,
        arm_sign=arm_sign,
        joint_position_ratio=joint_position_ratio_arr,
        gripper_sign=float(gripper_sign),
        gripper_range_rad=float(gripper_range_rad),
        gripper_close_pos_rad=float(gripper_close_pos_rad),
    )

    # Convert SIGTERM into SystemExit so the finally block runs cleanly
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

    try:
        server.serve()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        logging.info("Shutting down Fello server...")
        control_loop_running.clear()
        control_thread.join(timeout=1.0)
        if footswitch_thread is not None:
            footswitch_thread.join(timeout=1.0)
        if viewer_thread is not None:
            viewer_thread.join(timeout=1.0)
        robot.disconnect()


if __name__ == "__main__":
    main()

