"""
Core Fello robot definitions and utilities.

Split out of fello_server.py so the robot logic can be reused without the
server/CLI wrapper.
"""

import atexit
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Literal

import numpy as np

from damiao_motor import DaMiaoController, DaMiaoMotor
from enpire.env.forge.robot.systemid.motor_calibrator import MotorCalibrator
from enpire.env.forge.robot.yam.mujoco_utils import MuJoCoKDL
from enpire.env.forge.robot.fello.fello_config import (
    get_config_value,
    load_fello_config,
)
from enpire.env.forge.experimental.footswitch import FootSwitchMonitor, KeyboardFootSwitchMonitor

logging.basicConfig(level=logging.INFO)

# Path to Fello XML model
FELLO_XML_PATH = Path(__file__).parents[1] / "models" / "fello" / "fello.xml"


def _reset_gs_usb_device(channel: str = "can0") -> None:
    """Reset the gs_usb USB-CAN device to recover from unclean shutdowns on macOS."""
    if sys.platform != "darwin":
        return
    try:
        import re
        import time as _time
        import usb.core
        from usb.backend import libusb1
        from gs_usb.gs_usb import GsUsb

        m = re.search(r"(\d+)$", str(channel))
        target_index = int(m.group(1)) if m else 0

        # First stop the device cleanly via gs_usb protocol
        devices = GsUsb.scan()
        if target_index < len(devices):
            devices[target_index].stop()
        # Release all gs_usb handles
        del devices

        # Then do a USB-level reset to recover the read endpoint
        backend = libusb1.get_backend()
        all_devs = list(
            usb.core.find(
                find_all=True, custom_match=GsUsb.is_gs_usb_device, backend=backend
            )
        )
        if target_index < len(all_devs):
            all_devs[target_index].reset()
            logging.info(f"Reset USB-CAN device (index {target_index})")
            _time.sleep(0.5)
    except Exception as e:
        logging.warning(f"Failed to reset gs_usb device: {e}")


class FelloRobot:
    """
    Fello robot interface using damiao-motor for hardware control with optional
    gravity compensation and gripper spring helpers.
    """

    def __init__(
        self,
        can_interface: Optional[str] = None,
        motor_ids: Optional[List[int]] = None,
        feedback_ids: Optional[List[int]] = None,
        motor_types: Optional[List[str]] = None,
        gravity_coefficients: Optional[List[float]] = None,
        debug: Optional[bool] = None,
        gripper_mode: Optional[
            Literal["disabled", "linear_spring", "gas_spring"]
        ] = None,
        gripper_spring_constant: Optional[float] = None,
        gripper_spring_rest_position: Optional[float] = None,
        gripper_gas_spring_torque: Optional[float] = None,
        gravity_mode_kd: Optional[List[float]] = None,
        k_fric: Optional[List[float]] = None,
        bustype: str = "socketcan",
        torque_calibration_csvs: Optional[List[Optional[str]]] = None,
        torque_cal_method: str = "lut",
        side: Optional[str] = None,
    ):
        self._side = side
        cfg = load_fello_config(side=side)
        hardware_cfg = get_config_value(cfg, "hardware", default={}) or {}
        control_cfg = get_config_value(cfg, "control", default={}) or {}

        if can_interface is None:
            can_interface = hardware_cfg.get("can_interface")
        if motor_ids is None:
            motor_ids = hardware_cfg.get("motor_ids")
        if feedback_ids is None:
            feedback_ids = hardware_cfg.get("feedback_ids")
        if motor_types is None:
            motor_types = hardware_cfg.get("motor_types")
        if gravity_coefficients is None:
            gravity_coefficients = hardware_cfg.get("gravity_coefficients")
        server_cfg = get_config_value(cfg, "server", default={}) or {}

        if gripper_mode is None:
            gripper_mode = control_cfg.get("gripper_mode")
        if gripper_spring_constant is None:
            gripper_spring_constant = control_cfg.get("gripper_spring_constant")
        if gripper_spring_rest_position is None:
            gripper_spring_rest_position = control_cfg.get(
                "gripper_spring_rest_position"
            )
        if gripper_gas_spring_torque is None:
            gripper_gas_spring_torque = control_cfg.get("gripper_gas_spring_torque")
        if gravity_mode_kd is None:
            gravity_mode_kd = get_config_value(cfg, "control", "gravity_mode_kd")

        if can_interface is None:
            raise ValueError("Missing hardware.can_interface in fello_config.yaml")
        if motor_ids is None:
            raise ValueError("Missing hardware.motor_ids in fello_config.yaml")
        if motor_types is None:
            raise ValueError("Missing hardware.motor_types in fello_config.yaml")
        if gravity_coefficients is None:
            raise ValueError(
                "Missing hardware.gravity_coefficients in fello_config.yaml"
            )
        if gripper_mode is None:
            raise ValueError("Missing control.gripper_mode in fello_config.yaml")
        if gripper_spring_constant is None:
            raise ValueError(
                "Missing control.gripper_spring_constant in fello_config.yaml"
            )
        if gripper_spring_rest_position is None:
            raise ValueError(
                "Missing control.gripper_spring_rest_position in fello_config.yaml"
            )
        if gripper_gas_spring_torque is None:
            raise ValueError(
                "Missing control.gripper_gas_spring_torque in fello_config.yaml"
            )

        self.can_interface = can_interface
        if debug is None:
            debug = server_cfg.get("debug")
        if debug is None:
            raise ValueError("Missing server.debug in fello_config.yaml")
        self.debug = bool(debug)
        self.gripper_mode = gripper_mode
        self.gripper_spring_constant = gripper_spring_constant
        self.gripper_spring_rest_position = gripper_spring_rest_position
        self.gripper_gas_spring_torque = gripper_gas_spring_torque
        if gravity_mode_kd is None:
            raise ValueError("Missing control.gravity_mode_kd in fello_config.yaml")
        if isinstance(gravity_mode_kd, (int, float)):
            gravity_mode_kd = [float(gravity_mode_kd)] * 7
        if len(gravity_mode_kd) != 7:
            raise ValueError(
                f"control.gravity_mode_kd must have 7 values, got {len(gravity_mode_kd)}"
            )
        self.gravity_mode_kd = np.asarray(gravity_mode_kd, dtype=np.float32)
        self.motor_ids = motor_ids
        self.feedback_ids = feedback_ids or self.motor_ids.copy()
        self.motor_types = motor_types

        if len(self.motor_ids) != 7:
            raise ValueError(f"Expected 7 motor IDs, got {len(self.motor_ids)}")
        if len(self.feedback_ids) != 7:
            raise ValueError(f"Expected 7 feedback IDs, got {len(self.feedback_ids)}")
        if len(self.motor_types) != 7:
            raise ValueError(f"Expected 7 motor types, got {len(self.motor_types)}")

        if len(gravity_coefficients) != 7:
            raise ValueError(
                f"Expected 7 gravity coefficients, got {len(gravity_coefficients)}"
            )
        self.gravity_coefficients = np.array(gravity_coefficients)

        self.bustype = bustype
        _reset_gs_usb_device(self.can_interface)
        self.controller = DaMiaoController(
            channel=self.can_interface, bustype=bustype, bitrate=1000000
        )
        self.motors: List[Optional[DaMiaoMotor]] = [None] * 7
        self.dynamics = None
        self._connected = False
        self._last_debug_print_time = 0.0
        if server_cfg.get("debug_print_interval_s") is None:
            raise ValueError(
                "Missing server.debug_print_interval_s in fello_config.yaml"
            )
        self._debug_print_interval = float(server_cfg.get("debug_print_interval_s"))
        self._stiction_compensation = np.zeros(7, dtype=np.float32)
        self._stiction_deadband = np.zeros(7, dtype=np.float32)
        self._load_torque_compensation()
        self._last_commanded_pos = None
        self._last_commanded_kp = None
        self._last_commanded_kd = None

        # Per-motor torque calibrators (None = passthrough)
        if torque_calibration_csvs is None:
            torque_calibration_csvs = hardware_cfg.get(
                "torque_calibration_csvs", [None] * 7
            )
        torque_cal_method = hardware_cfg.get("torque_cal_method", torque_cal_method)
        self._torque_calibrators: List[Optional[MotorCalibrator]] = []
        for csv_path in torque_calibration_csvs:
            if csv_path is not None:
                try:
                    self._torque_calibrators.append(
                        MotorCalibrator(csv_path, method=torque_cal_method)
                    )
                    logging.info(
                        f"[FelloRobot] Loaded torque calibrator from {csv_path} ({torque_cal_method})"
                    )
                except Exception as e:
                    logging.warning(
                        f"[FelloRobot] Failed to load torque calibrator {csv_path}: {e}"
                    )
                    self._torque_calibrators.append(None)
            else:
                self._torque_calibrators.append(None)
        # Pad to 7 if list is short
        while len(self._torque_calibrators) < 7:
            self._torque_calibrators.append(None)

    def _apply_stiction_compensation(self, torques: np.ndarray) -> np.ndarray:
        # Add compensation in the direction of commanded torque.
        sign = np.sign(torques)
        sign = np.where(np.abs(torques) < self._stiction_deadband, 0.0, sign)
        return torques + sign * np.abs(self._stiction_compensation)

    def _load_torque_compensation(self) -> None:
        data = load_fello_config(side=self._side)
        comp = data["torque_compensation"]
        stiction = comp["stiction_compensation_nm"]
        deadband = comp["stiction_deadband_nm"]

        if isinstance(stiction, list):
            if len(stiction) != 7:
                raise ValueError("stiction_compensation_nm list must have 7 entries")
            self._stiction_compensation = np.asarray(stiction, dtype=np.float32)
        else:
            if not isinstance(stiction, dict):
                raise ValueError("stiction_compensation_nm must be a list or dict")
            values = []
            for i in range(7):
                key = f"joint_{i + 1}"
                if key not in stiction:
                    raise ValueError(f"stiction_compensation_nm missing {key}")
                values.append(float(stiction[key]))
            self._stiction_compensation = np.asarray(values, dtype=np.float32)

        if isinstance(deadband, list):
            if len(deadband) != 7:
                raise ValueError("stiction_deadband_nm list must have 7 entries")
            self._stiction_deadband = np.asarray(deadband, dtype=np.float32)
            return

        if not isinstance(deadband, dict):
            raise ValueError("stiction_deadband_nm must be a list or dict")

        values = []
        for i in range(7):
            key = f"joint_{i + 1}"
            if key not in deadband:
                raise ValueError(f"stiction_deadband_nm missing {key}")
            values.append(float(deadband[key]))
        self._stiction_deadband = np.asarray(values, dtype=np.float32)

    # Connection and state helpers -------------------------------------------------
    def connect(self) -> bool:
        try:
            logging.info(f"Connecting to CAN interface: {self.can_interface}")
            logging.info("Initializing motors...")
            for i, (motor_id, feedback_id) in enumerate(
                zip(self.motor_ids, self.feedback_ids)
            ):
                try:
                    motor = self.controller.add_motor(
                        motor_id,
                        feedback_id,
                        motor_type=self.motor_types[i],
                    )
                    motor.enable()
                    self.motors[i] = motor
                    logging.info(
                        f"  Motor {i + 1}: ID={motor_id}, Feedback ID={feedback_id} - Connected"
                    )
                except Exception as e:
                    logging.error(f"  Motor {i + 1}: Failed to connect - {e}")
                    return False

            logging.info("All motors connected successfully!")
            self._connected = True
            atexit.register(self.disconnect)

            if FELLO_XML_PATH.exists():
                logging.info("Initializing MuJoCo model for gravity compensation...")
                try:
                    self.dynamics = MuJoCoKDL(str(FELLO_XML_PATH))
                    logging.info(f"Loaded MuJoCo model from: {FELLO_XML_PATH}")
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize MuJoCo KDL: {e}") from e
            else:
                raise FileNotFoundError(f"Fello XML file not found at {FELLO_XML_PATH}")

            return True
        except Exception as e:
            logging.error(f"Failed to connect to CAN bus: {e}")
            return False

    def disconnect(self) -> None:
        if not self._connected:
            return
        logging.info("\nDisconnecting motors...")
        self._connected = False
        try:
            self.controller.shutdown()
        except Exception:
            pass
        self.motors = [None] * 7

    def reconnect(self) -> bool:
        """Tear down CAN bus, reset USB device, and re-connect all motors."""
        logging.warning("[FelloRobot] Attempting CAN reconnect...")
        # Tear down old connection
        self._connected = False
        try:
            self.controller.shutdown()
        except Exception:
            pass
        self.motors = [None] * 7

        # Reset USB-CAN adapter and create fresh controller
        _reset_gs_usb_device(self.can_interface)
        from damiao_motor import DaMiaoController

        self.controller = DaMiaoController(
            channel=self.can_interface, bustype=self.bustype, bitrate=1000000
        )

        # Re-init motors
        for i, (motor_id, feedback_id) in enumerate(
            zip(self.motor_ids, self.feedback_ids)
        ):
            try:
                motor = self.controller.add_motor(
                    motor_id,
                    feedback_id,
                    motor_type=self.motor_types[i],
                )
                motor.enable()
                self.motors[i] = motor
            except Exception as e:
                logging.error(f"[FelloRobot] Reconnect: Motor {i + 1} failed - {e}")
                return False

        self._connected = True
        logging.warning("[FelloRobot] CAN reconnect successful — all motors re-enabled")
        return True

    # Readouts --------------------------------------------------------------------
    def read_motor_states(self) -> Optional[List[dict]]:
        if not self._connected:
            logging.warning("Robot not connected. Call connect() first.")
            return None
        states = []
        for i, motor in enumerate(self.motors):
            if motor is None:
                logging.error(f"Motor {i + 1} is not initialized")
                return None
            try:
                state = motor.get_states()
                states.append(state)
            except Exception as e:
                logging.error(f"Motor {i + 1}: Failed to read state - {e}")
                return None
        return states

    @staticmethod
    def _extract_temp_values(state: dict, keys: list[str]) -> list[float]:
        numeric: list[float] = []
        for key in keys:
            v = state.get(key)
            if v is None:
                continue
            fv = float(v)
            if np.isfinite(fv):
                numeric.append(fv)
        return numeric

    @classmethod
    def _extract_temp_mos_rotor(cls, state: dict) -> tuple[float, float]:
        mos_vals = cls._extract_temp_values(
            state, ["t_mos", "temp_mos", "temperature_mos"]
        )
        rotor_vals = cls._extract_temp_values(
            state, ["t_rotor", "temp_rotor", "temperature_rotor"]
        )
        mos = max(mos_vals) if mos_vals else float("nan")
        rotor = max(rotor_vals) if rotor_vals else float("nan")
        return mos, rotor

    @classmethod
    def _extract_temp(cls, state: dict) -> float:
        mos, rotor = cls._extract_temp_mos_rotor(state)
        fallback_vals = cls._extract_temp_values(state, ["temp", "temperature"])
        candidates: list[float] = []
        if np.isfinite(mos):
            candidates.append(float(mos))
        if np.isfinite(rotor):
            candidates.append(float(rotor))
        candidates.extend(fallback_vals)
        return max(candidates) if candidates else float("nan")

    @staticmethod
    def _colorize(text: str, color_code: str) -> str:
        return f"\033[{color_code}m{text}\033[0m"

    @classmethod
    def _format_temp(cls, value: float) -> str:
        if not np.isfinite(value):
            return cls._colorize("N/A", "2")
        if value >= 75.0:
            return cls._colorize(f"{value:>6.1f}", "1;31")
        if value >= 60.0:
            return cls._colorize(f"{value:>6.1f}", "33")
        return cls._colorize(f"{value:>6.1f}", "32")

    @classmethod
    def _format_temp_used(cls, mos: float, rotor: float, used: float) -> str:
        if not np.isfinite(used):
            return cls._colorize("N/A (unknown)", "2")
        if np.isfinite(mos) and np.isclose(used, mos):
            source = "MOS"
        elif np.isfinite(rotor) and np.isclose(used, rotor):
            source = "ROTOR"
        else:
            source = "GENERIC"
        return f"{cls._format_temp(used)} ({source})"

    def print_motor_states_table(self) -> None:
        if not self._connected:
            return
        states = self.read_motor_states()
        if states is None:
            return
        gravity_torques = self.compute_gravity_compensation(self.get_joint_pos())
        width = 148
        title = self._colorize("FELLO MOTOR STATE TABLE", "1;36")
        print("\n" + "=" * width)
        print(
            f"{title}  {self._colorize('(Temp Used = max(MOS, ROTOR, generic))', '2')}"
        )
        print(
            f"{'Motor':<7} {'ID':<5} {'Pos (rad)':>10} {'Vel (rad/s)':>12} "
            f"{'Torque (Nm)':>12} {'Grav (Nm)':>10} {'Temp MOS':>10} {'Temp Rotor':>11} "
            f"{'Temp Used':>18} {'Voltage (V)':>12}"
        )
        print("-" * width)
        for i, (motor, state) in enumerate(zip(self.motors, states)):
            if motor is None:
                print(
                    f"{i + 1:<7} {'N/A':<5} {'N/A':>10} {'N/A':>12} "
                    f"{'N/A':>12} {'N/A':>10} {'N/A':>10} {'N/A':>11} "
                    f"{'N/A':>18} {'N/A':>12}"
                )
                continue
            motor_id = self.motor_ids[i]
            pos = state.get("pos", 0.0)
            vel = state.get("vel", 0.0)
            torque = state.get("torq", 0.0)
            grav = gravity_torques[i] if i < len(gravity_torques) else 0.0
            temp_mos, temp_rotor = self._extract_temp_mos_rotor(state)
            temp_used = self._extract_temp(state)
            voltage = state.get("voltage", 0.0)
            motor_str = self._colorize(f"{i + 1:<7}", "36")
            id_str = self._colorize(f"{motor_id:<5}", "34")
            print(
                f"{motor_str} {id_str} "
                f"{pos:>10.4f} {vel:>12.4f} {torque:>12.4f} {grav:>10.4f} "
                f"{self._format_temp(temp_mos):>10} {self._format_temp(temp_rotor):>11} "
                f"{self._format_temp_used(temp_mos, temp_rotor, temp_used):>18} {voltage:>12.2f}"
            )
        print("=" * width + "\n")

    def get_joint_pos(self) -> np.ndarray:
        states = self.read_motor_states()
        if states is None:
            return np.zeros(7)
        return np.array([state.get("pos", 0.0) for state in states])

    def get_joint_vel(self) -> np.ndarray:
        states = self.read_motor_states()
        if states is None:
            return np.zeros(7)
        return np.array([state.get("vel", 0.0) for state in states])

    def get_observations(self) -> dict[str, np.ndarray]:
        states = self.read_motor_states()
        if states is None:
            return {
                "joint_pos": np.zeros(7),
                "joint_vel": np.zeros(7),
                "joint_torque": np.zeros(7),
            }
        return {
            "joint_pos": np.array([state.get("pos", 0.0) for state in states]),
            "joint_vel": np.array([state.get("vel", 0.0) for state in states]),
            "joint_torque": np.array([state.get("torq", 0.0) for state in states]),
        }

    def get_motor_temperatures(self) -> np.ndarray:
        states = self.read_motor_states()
        if states is None:
            return np.full(7, np.nan, dtype=np.float32)
        temps = [self._extract_temp(state) for state in states]
        return np.asarray(temps, dtype=np.float32)

    # Torque helpers --------------------------------------------------------------
    def compute_gravity_compensation(
        self, joint_positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self.dynamics is None:
            return np.zeros(7)
        if joint_positions is None:
            joint_positions = self.get_joint_pos()
        try:
            qdot = np.zeros(7)
            qdotdot = np.zeros(7)
            raw = self.dynamics.compute_inverse_dynamics(joint_positions, qdot, qdotdot)
            return raw * self.gravity_coefficients
        except Exception as e:
            logging.warning(f"Failed to calculate gravity torques: {e}")
            return np.zeros(7)

    def compute_gripper_torque(
        self, joint_positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        torques = np.zeros(7)
        if self.gripper_mode == "disabled":
            return torques
        if self.gripper_mode == "gas_spring":
            torques[6] = self.gripper_gas_spring_torque
            return torques
        if joint_positions is None:
            joint_positions = self.get_joint_pos()
        if self.gripper_mode == "linear_spring":
            torques[6] = -self.gripper_spring_constant * (
                joint_positions[6] - self.gripper_spring_rest_position
            )
        else:
            logging.warning(f"Unknown gripper mode: {self.gripper_mode}")
        return torques

    @staticmethod
    def _sanitize_feedback_torques(feedback_torques: np.ndarray) -> np.ndarray:
        torques = np.asarray(feedback_torques, dtype=np.float32).reshape(-1)
        if torques.size != 7:
            raise ValueError(f"Expected 7 feedback torques, got {torques.size}")
        torques = np.nan_to_num(torques, nan=0.0, posinf=0.0, neginf=0.0)
        return torques

    def _calibrate_torque(self, i: int, torque: float) -> float:
        cal = self._torque_calibrators[i]
        return cal.get_cmd_from_torque(torque) if cal is not None else torque

    # Commands --------------------------------------------------------------------
    # Inter-motor delay (seconds) to let CAN feedback drain between commands.
    # On macOS gs_usb, each bus.recv() takes >=1ms, so without spacing the
    # read pipeline falls behind for higher-numbered motors.
    _INTER_MOTOR_DELAY_S = 0.0005 if sys.platform == "darwin" else 0.0

    def _send_gravity_compensation(
        self, feedback_torques: Optional[np.ndarray] = None
    ) -> int:
        """Send gravity-comp commands with optional extra feedforward torques."""
        if not self._connected:
            logging.warning("Robot not connected. Call connect() first.")
            return 7
        current_pos = self.get_joint_pos()
        gravity_torques = self.compute_gravity_compensation(current_pos)
        gripper_torques = self.compute_gripper_torque(current_pos)
        if feedback_torques is None:
            feedback_torques = np.zeros(7, dtype=np.float32)
        feedback_torques = self._sanitize_feedback_torques(feedback_torques)
        total_torques = self._apply_stiction_compensation(
            gravity_torques + gripper_torques + feedback_torques
        )
        if self.debug:
            now = time.time()
            if now - self._last_debug_print_time >= self._debug_print_interval:
                self.print_motor_states_table()
                self._last_debug_print_time = now
        failures = 0
        delay = self._INTER_MOTOR_DELAY_S
        for i, (motor, torque) in enumerate(zip(self.motors, total_torques)):
            if motor is None:
                continue
            try:
                motor.send_cmd(
                    target_position=0.0,
                    target_velocity=0.0,
                    stiffness=0.0,
                    damping=float(self.gravity_mode_kd[i]),
                    feedforward_torque=self._calibrate_torque(i, torque),
                    control_mode="MIT",
                )
                if delay > 0 and i < len(self.motors) - 1:
                    time.sleep(delay)
            except Exception as e:
                failures += 1
                logging.warning(f"Motor {i + 1} failed to send command: {e}")
        return failures

    def send_gravity_compensation_only(self) -> int:
        """Send gravity compensation with no external feedback torques."""
        return self._send_gravity_compensation()

    def send_gravity_compensation_and_feedback(
        self, feedback_torques: np.ndarray
    ) -> int:
        """Send gravity compensation and add a 7D feedforward torque vector."""
        return self._send_gravity_compensation(feedback_torques)

    def command_joint_pos(
        self,
        joint_pos: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        use_gravity_comp: bool = True,
    ) -> int:
        """Send position commands. Returns number of motor send failures."""
        if not self._connected:
            logging.warning("Robot not connected. Call connect() first.")
            return 7
        if len(joint_pos) != 7:
            raise ValueError(f"Expected 7 joint positions, got {len(joint_pos)}")
        if kp is None:
            kp = np.zeros(7)
        if kd is None:
            kd = np.zeros(7)
        self._last_commanded_pos = joint_pos.copy()
        self._last_commanded_kp = kp.copy()
        self._last_commanded_kd = kd.copy()
        current_pos = self.get_joint_pos()
        gravity_torques = (
            self.compute_gravity_compensation(current_pos)
            if use_gravity_comp
            else np.zeros(7)
        )
        gripper_torques = self.compute_gripper_torque(current_pos)
        total_torques = gravity_torques + gripper_torques
        failures = 0
        delay = self._INTER_MOTOR_DELAY_S
        for i, (motor, torque) in enumerate(zip(self.motors, total_torques)):
            if motor is None:
                continue
            try:
                motor.send_cmd(
                    target_position=joint_pos[i],
                    target_velocity=0.0,
                    stiffness=kp[i],
                    damping=kd[i],
                    feedforward_torque=self._calibrate_torque(i, torque),
                    control_mode="MIT",
                )
                if delay > 0 and i < len(self.motors) - 1:
                    time.sleep(delay)
            except Exception as e:
                failures += 1
                logging.warning(f"Motor {i + 1} failed to send command: {e}")
        return failures

    def get_last_commanded_state(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        return (
            self._last_commanded_pos,
            self._last_commanded_kp,
            self._last_commanded_kd,
        )

    def set_last_commanded_state(
        self, pos: np.ndarray, kp: np.ndarray, kd: np.ndarray
    ) -> None:
        self._last_commanded_pos = pos.copy()
        self._last_commanded_kp = kp.copy()
        self._last_commanded_kd = kd.copy()


class FelloLeaderRobot:
    """Wrapper for FelloRobot to provide leader arm interface compatible with teleop."""

    def __init__(self, robot: FelloRobot, side: Optional[str] = None):
        self._robot = robot
        cfg = load_fello_config(side=side)
        footswitch_cfg = (
            get_config_value(cfg, "hardware", "footswitch", default={}) or {}
        )
        fs_type = footswitch_cfg.get("type", "").strip().lower()

        if fs_type == "serial":
            from enpire.env.forge.experimental.footswitch import SerialButtonHub, SerialButtonMonitor
            serial_port = footswitch_cfg.get("serial_port")
            button_map = footswitch_cfg.get("button_map", [0, 1, 2])
            baudrate = footswitch_cfg.get("serial_baudrate", 115200)
            hub = SerialButtonHub(port=serial_port, baudrate=baudrate, num_buttons=3)
            logging.info("Using serial footswitch on %s (button_map=%s)", serial_port, button_map)
            self._footswitches = tuple(
                SerialButtonMonitor(hub, index=button_map[i]) for i in range(3)
            )
        else:
            device_paths = [
                footswitch_cfg.get("device_0"),
                footswitch_cfg.get("device_1"),
                footswitch_cfg.get("device_2"),
            ]
            keyboard_keys = footswitch_cfg.get("keyboard_keys")
            have_keyboard = bool(keyboard_keys and len(keyboard_keys) == 3)
            display_available = bool(
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            )

            usable_device_paths = [
                str(path) for path in device_paths if isinstance(path, str) and path.strip()
            ]
            have_all_devices_configured = len(usable_device_paths) == 3
            existing_device_paths = [
                path for path in usable_device_paths if Path(path).exists()
            ]

            if have_all_devices_configured and len(existing_device_paths) == 3:
                logging.info(
                    "Using hardware footswitch devices: %s",
                    ", ".join(existing_device_paths),
                )
                self._footswitches = tuple(
                    FootSwitchMonitor(device_path=path) for path in usable_device_paths
                )
            elif have_keyboard and display_available:
                logging.info("Using keyboard footswitch fallback via pynput")
                self._footswitches = (
                    KeyboardFootSwitchMonitor(key=keyboard_keys[0]),
                    KeyboardFootSwitchMonitor(key=keyboard_keys[1]),
                    KeyboardFootSwitchMonitor(key=keyboard_keys[2]),
                )
            else:
                missing = [
                    path for path in usable_device_paths if not Path(path).exists()
                ] if have_all_devices_configured else []
                if missing:
                    raise FileNotFoundError(
                        "Configured footswitch devices were not found: "
                        + ", ".join(missing)
                        + ". Plug in the footswitch or update robot/models/fello/fello_config.yaml."
                    )
                if have_keyboard and not display_available:
                    raise RuntimeError(
                        "keyboard_keys is configured for the Fello footswitch, but no DISPLAY/"
                        "WAYLAND_DISPLAY is available. Use the real /dev/input/footswitch_* "
                        "devices or disable keyboard_keys in robot/models/fello/fello_config.yaml."
                    )
                raise ValueError(
                    "Missing usable Fello footswitch configuration. Expected either existing "
                    "hardware.footswitch.device_0/device_1/device_2 paths or 3 keyboard_keys "
                    "with a working GUI session."
                )

    def get_info(self) -> tuple[np.ndarray, np.ndarray]:
        qpos = self._robot.get_joint_pos()
        buttons = np.array(
            [
                1.0 if self._footswitches[0].is_pressed() else 0.0,
                1.0 if self._footswitches[1].is_pressed() else 0.0,
                1.0 if self._footswitches[2].is_pressed() else 0.0,
            ],
            dtype=np.float32,
        )
        return qpos, buttons

    def get_motor_temperatures(self) -> np.ndarray:
        return self._robot.get_motor_temperatures()

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        required_keys = {"pos", "vel", "kp", "kd"}
        if not required_keys.issubset(joint_state.keys()):
            missing = required_keys - joint_state.keys()
            raise ValueError(f"Missing required keys in joint_state: {missing}")

        pos = joint_state["pos"].copy()
        vel = joint_state["vel"]
        kp = joint_state["kp"]
        kd = joint_state["kd"]

        for key, value in joint_state.items():
            if len(value) != 7:
                raise ValueError(f"{key} must have 7 elements, got {len(value)}")

        pos = pos

        self._robot.command_joint_pos(
            joint_pos=pos,
            kp=kp,
            kd=kd,
            use_gravity_comp=True,
        )

    def get_target_pos(self) -> np.ndarray | None:
        pos, _, _ = self._robot.get_last_commanded_state()
        return None if pos is None else pos.copy()

    def set_target_pos(self, joint_state: dict[str, np.ndarray]) -> None:
        required_keys = {"pos", "vel", "kp", "kd"}
        if not required_keys.issubset(joint_state.keys()):
            missing = required_keys - joint_state.keys()
            raise ValueError(f"Missing required keys in joint_state: {missing}")

        pos = joint_state["pos"].copy()
        kp = joint_state["kp"].copy()
        kd = joint_state["kd"].copy()

        for key, value in joint_state.items():
            if len(value) != 7:
                raise ValueError(f"{key} must have 7 elements, got {len(value)}")

        pos = pos
        self._robot.set_last_commanded_state(pos, kp, kd)


__all__ = [
    "FelloRobot",
    "FelloLeaderRobot",
    "FELLO_XML_PATH",
]
