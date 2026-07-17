"""YAM arm controller using DaMiao motors directly."""

import logging
import os
import threading
import time
from typing import List, Optional

import numpy as np
from damiao_motor import MOTOR_TYPE_PRESETS, DaMiaoController, DaMiaoMotor

from enpire.env.forge.paths import third_party_path
from enpire.env.forge.robot.yam.mujoco_utils import MuJoCoKDL

# Default XML for gravity compensation (LINEAR_4310 gripper variant)
_YAM_XML_PATH = str(
    third_party_path("i2rt", "i2rt", "robot_models", "yam", "yam_4310_linear.xml")
)

# Global scale factor matching i2rt's get_yam_robot (gravity_comp_factor=1.3)
_GRAVITY_COMP_FACTOR = 1.3
_EEF_SITE_NAME = "grasp_site"
_EEF_FORCE_DAMPING = 1e-4


class YamRobot:
    """YAM robot interface using damiao-motor for hardware control."""

    def __init__(
        self,
        can_interface: str,
        motor_ids: List[int],
        motor_types: List[str],
        feedback_ids: Optional[List[int]] = None,
        default_kp: Optional[List[float]] = None,
        default_kd: Optional[List[float]] = None,
        gripper_index: Optional[int] = None,
        gripper_sign: int = -1,
        gripper_vel_limit: float = 30.0,
        gripper_torque_limit_nm: float = 1.0,
        send_rate_hz: float = 100.0,
        bustype: str = "socketcan",
        model_xml_path: str | None = None,
    ):
        if len(motor_types) != len(motor_ids):
            raise ValueError(
                f"motor_types length {len(motor_types)} != motor_ids length {len(motor_ids)}"
            )

        self.NUM_JOINTS = len(motor_ids)
        self.gripper_index = gripper_index
        self.gripper_sign = gripper_sign
        self.gripper_vel_limit = gripper_vel_limit
        self.gripper_torque_limit_nm = gripper_torque_limit_nm
        self.send_rate_hz = send_rate_hz
        self.model_xml_path = model_xml_path or _YAM_XML_PATH

        self.can_interface = can_interface
        self.motor_ids = motor_ids
        self.motor_types = motor_types
        self.feedback_ids = feedback_ids or motor_ids.copy()
        self.default_kp = (
            np.asarray(default_kp) if default_kp is not None else np.zeros(self.NUM_JOINTS)
        )
        self.default_kd = (
            np.asarray(default_kd) if default_kd is not None else np.zeros(self.NUM_JOINTS)
        )

        self.bustype = bustype
        self.controller = DaMiaoController(channel=can_interface, bustype=bustype, bitrate=1000000)
        self.motors: List[Optional[DaMiaoMotor]] = [None] * self.NUM_JOINTS
        self._connected = False

        # Set by calibration in connect()
        self.gripper_close_pos: Optional[float] = None
        self.gripper_open_pos: Optional[float] = None

        # Gravity compensation
        self._kdl: Optional[MuJoCoKDL] = None
        self._kdl_lock = threading.Lock()
        self._last_gravity_comp = np.zeros(self.NUM_JOINTS)

        # Background send loop
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._send_thread: Optional[threading.Thread] = None
        self._cmd_pos: Optional[np.ndarray] = None
        self._cmd_kp: Optional[np.ndarray] = None
        self._cmd_kd: Optional[np.ndarray] = None
        self._cmd_gripper_vel_limit: Optional[float] = None
        self._cmd_gripper_torque_limit_nm: Optional[float] = None
        self._background_error: Optional[BaseException] = None

    def _raise_if_background_error(self) -> None:
        if self._background_error is not None:
            raise RuntimeError(
                f"[{self.can_interface}] background send loop failed"
            ) from self._background_error

    def _motor_to_normalized(self, motor_pos: float) -> float:
        """Convert raw motor position → normalized [0=close, 1=open]."""
        env_pos = self.gripper_sign * motor_pos
        t = (env_pos - self.gripper_close_pos) / (self.gripper_open_pos - self.gripper_close_pos)
        return float(np.clip(t, 0.0, 1.0))

    def _normalized_to_motor(self, t: float) -> float:
        """Convert normalized [0=close, 1=open] → raw motor position."""
        env_pos = self.gripper_close_pos + t * (self.gripper_open_pos - self.gripper_close_pos)
        return self.gripper_sign * env_pos  # sign=±1, so 1/sign == sign

    def _calibrate_gripper(self, motor: DaMiaoMotor) -> None:
        """Move gripper to both physical stops to determine close/open positions."""
        import time

        CAL_VEL = 5.0  # rad/s, slow for safety
        CAL_TORQUE = 0.3  # torque limit ratio
        SETTLE_TIME = 2.5  # seconds to wait at each stop
        MOTOR_RANGE = motor._p_max  # position limit from motor type preset

        print(f"  Calibrating gripper on {self.can_interface}...")
        env_positions = []
        for cmd in [-MOTOR_RANGE, MOTOR_RANGE]:
            deadline = time.monotonic() + SETTLE_TIME
            while time.monotonic() < deadline:
                motor.send_cmd_force_pos(cmd, CAL_VEL, CAL_TORQUE)
                time.sleep(0.02)
            env_positions.append(self.gripper_sign * motor.get_states()["pos"])

        self.gripper_close_pos = min(env_positions)
        self.gripper_open_pos = max(env_positions)
        print(
            f"  Gripper calibrated: close={self.gripper_close_pos:.3f}, open={self.gripper_open_pos:.3f} (env rad)"
        )

    def _compute_gravity_compensation(
        self, joint_positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute gravity torques for arm joints from current positions."""
        grav = np.zeros(self.NUM_JOINTS)
        if self._kdl is None:
            return grav
        try:
            if joint_positions is None:
                # Read current raw motor positions (arm joints only, exclude gripper).
                arm_pos = []
                for i, motor in enumerate(self.motors):
                    if motor is None:
                        return grav
                    if self.gripper_index is not None and i == self.gripper_index:
                        continue
                    arm_pos.append(motor.get_states()["pos"])
                q = np.array(arm_pos)
            else:
                joint_positions = np.asarray(joint_positions, dtype=float).reshape(-1)
                if joint_positions.size != self.NUM_JOINTS:
                    raise ValueError(
                        f"Expected {self.NUM_JOINTS} joint positions, got {joint_positions.size}"
                    )
                if self.gripper_index is not None:
                    q = np.delete(joint_positions, self.gripper_index)
                else:
                    q = joint_positions
            with self._kdl_lock:
                raw = self._kdl.compute_inverse_dynamics(q, np.zeros_like(q), np.zeros_like(q))
            if np.max(np.abs(raw)) > 20.0:
                logging.warning(f"Gravity torques too large: {raw}, skipping")
                return grav
            j = 0
            for i in range(self.NUM_JOINTS):
                if self.gripper_index is not None and i == self.gripper_index:
                    continue
                grav[i] = raw[j] * _GRAVITY_COMP_FACTOR
                j += 1
        except Exception:
            logging.exception(
                "[%s] Gravity compensation failed; disabling compensation for this cycle",
                self.can_interface,
            )
        return grav

    def _arm_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != self.NUM_JOINTS:
            raise ValueError(f"Expected {self.NUM_JOINTS} values, got {values.size}")
        if self.gripper_index is None:
            return values
        return np.delete(values, self.gripper_index)

    def _compute_eef_force_estimate(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        residual_joint_torque: np.ndarray,
    ) -> np.ndarray:
        """Estimate Cartesian EEF force from gravity-subtracted joint torque.

        The local MuJoCo helper owns the grasp-site Jacobian solve; this method
        only converts full motor arrays into arm-only arrays.
        """
        eef_force = np.zeros(3, dtype=np.float32)
        if self._kdl is None:
            return eef_force

        try:
            q = self._arm_values(joint_positions)
            qvel = self._arm_values(joint_velocities)
            tau = self._arm_values(residual_joint_torque)

            with self._kdl_lock:
                return self._kdl.compute_site_force_from_joint_torque(
                    q,
                    qvel,
                    tau,
                    _EEF_SITE_NAME,
                    damping=_EEF_FORCE_DAMPING,
                )
        except Exception:
            logging.exception(
                "[%s] EEF force estimation failed; returning zeros",
                self.can_interface,
            )
            return eef_force

    def _gripper_torque_ratio_from_nm(self, motor_index: int, torque_limit_nm: float) -> float:
        """Convert torque limit in Nm to damiao FORCE_POS torque_limit_ratio."""
        motor_type = self.motor_types[motor_index]
        preset = MOTOR_TYPE_PRESETS.get(motor_type, {})
        t_max = float(preset.get("t_max", 0.0))
        if t_max <= 0.0:
            motor = self.motors[motor_index]
            t_max = float(getattr(motor, "_t_max", 0.0)) if motor is not None else 0.0
        if t_max <= 0.0:
            logging.warning(
                f"[{self.can_interface}] invalid T_max for gripper motor type '{motor_type}'"
            )
            return 0.0
        tau = float(np.clip(torque_limit_nm, 0.0, t_max))
        return tau / t_max

    @staticmethod
    def _invalidate_register_cache(motor: DaMiaoMotor, rid: int) -> None:
        lock = getattr(motor, "registers_lock", None)
        registers = getattr(motor, "registers", None)
        if registers is None:
            return
        if lock is None:
            registers.pop(rid, None)
            return
        with lock:
            registers.pop(rid, None)

    def _ensure_control_mode(self, motor: DaMiaoMotor, mode: str) -> None:
        """Set register 10 and force verification to read fresh motor state."""
        mode_to_register = {"MIT": 1, "POS_VEL": 2, "VEL": 3, "FORCE_POS": 4}
        desired = mode_to_register[mode]
        self._invalidate_register_cache(motor, 10)
        current = int(motor.get_register(10, timeout=1.0))
        if current == desired:
            return
        print(f"⚠ Control mode mismatch: register 10 = {current}, required = {desired}")
        print(f"  Setting control mode to {mode} (register value: {desired})...")
        motor.write_register(10, desired)
        time.sleep(0.2)
        self._invalidate_register_cache(motor, 10)
        verify = int(motor.get_register(10, timeout=1.0))
        if verify != desired:
            raise RuntimeError(
                f"Control mode verification failed after write: expected {desired}, got {verify}"
            )
        print(f"✓ Control mode set to {mode}")

    def _do_send(
        self,
        joint_pos: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        gripper_vel_limit: Optional[float],
        gripper_torque_limit_nm: Optional[float],
    ) -> None:
        """Send current command to all motors. Called from background loop."""
        if joint_pos.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected joint_pos shape {(self.NUM_JOINTS,)}, "
                f"got {joint_pos.shape}"
            )
        if kp.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected kp shape {(self.NUM_JOINTS,)}, got {kp.shape}"
            )
        if kd.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected kd shape {(self.NUM_JOINTS,)}, got {kd.shape}"
            )
        grav = self._compute_gravity_compensation()
        self._last_gravity_comp = grav.copy()
        for i, motor in enumerate(self.motors):
            if motor is None:
                raise RuntimeError(f"[{self.can_interface}] motor slot {i} is not connected")
            if self.gripper_index is not None and i == self.gripper_index:
                vel_limit = (
                    self.gripper_vel_limit
                    if gripper_vel_limit is None
                    else float(gripper_vel_limit)
                )
                torque_limit_nm = (
                    self.gripper_torque_limit_nm
                    if gripper_torque_limit_nm is None
                    else float(gripper_torque_limit_nm)
                )
                motor.send_cmd_force_pos(
                    target_position=self._normalized_to_motor(joint_pos[i]),
                    velocity_limit=vel_limit,
                    torque_limit_ratio=self._gripper_torque_ratio_from_nm(i, torque_limit_nm),
                )
            else:
                motor.send_cmd_mit(
                    target_position=joint_pos[i],
                    target_velocity=0.0,
                    stiffness=kp[i],
                    damping=kd[i],
                    feedforward_torque=grav[i],
                )

    def _send_loop(self) -> None:
        import time

        dt = 1.0 / self.send_rate_hz
        while not self._stop_event.is_set():
            with self._lock:
                pos = self._cmd_pos
                kp = self._cmd_kp
                kd = self._cmd_kd
                gripper_vel_limit = self._cmd_gripper_vel_limit
                gripper_torque_limit_nm = self._cmd_gripper_torque_limit_nm
            if pos is not None:
                try:
                    self._do_send(pos, kp, kd, gripper_vel_limit, gripper_torque_limit_nm)
                except Exception as e:
                    self._background_error = e
                    logging.exception("[%s] background send loop crashed", self.can_interface)
                    self._stop_event.set()
                    raise
            time.sleep(dt)

    def connect(self) -> None:
        if len(self.feedback_ids) != self.NUM_JOINTS:
            raise ValueError(
                f"feedback_ids length {len(self.feedback_ids)} != motor_ids length {self.NUM_JOINTS}"
            )
        for i, (motor_id, feedback_id) in enumerate(zip(self.motor_ids, self.feedback_ids)):
            motor = self.controller.add_motor(motor_id, feedback_id, motor_type=self.motor_types[i])
            is_gripper = self.gripper_index is not None and i == self.gripper_index
            mode = "FORCE_POS" if is_gripper else "MIT"
            self._ensure_control_mode(motor, mode)
            motor.enable()
            self.motors[i] = motor
            if is_gripper:
                self._calibrate_gripper(motor)
        self._connected = True

        # Load MuJoCo model for gravity compensation
        xml_path = os.path.normpath(self.model_xml_path)
        if os.path.exists(xml_path):
            self._kdl = MuJoCoKDL(xml_path)
            if not self._kdl.has_site(_EEF_SITE_NAME):
                logging.warning(
                    "MuJoCo site '%s' not found; EEF force estimation disabled",
                    _EEF_SITE_NAME,
                )
            logging.info(f"Loaded MuJoCo model for gravity comp: {xml_path}")
        else:
            logging.warning(f"MuJoCo XML not found at {xml_path}, gravity comp disabled")

        self._background_error = None
        self._stop_event.clear()
        self._send_thread = threading.Thread(
            target=self._send_loop,
            daemon=True,
            name=f"yam-send-{self.can_interface}",
        )
        self._send_thread.start()

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._send_thread is not None:
            self._send_thread.join(timeout=1.0)
            self._send_thread = None
        disable_errors = []
        for motor in self.motors:
            if motor is not None:
                try:
                    motor.disable()
                except Exception as e:
                    disable_errors.append(e)
        self.motors = [None] * self.NUM_JOINTS
        self._connected = False
        if disable_errors:
            raise RuntimeError(
                f"[{self.can_interface}] failed to disable {len(disable_errors)} motor(s)"
            ) from disable_errors[0]
        self._raise_if_background_error()

    def _read_states(self) -> Optional[List[dict]]:
        self._raise_if_background_error()
        if not self._connected:
            raise RuntimeError(f"[{self.can_interface}] robot is not connected")
        states = []
        for i, motor in enumerate(self.motors):
            if motor is None:
                raise RuntimeError(f"[{self.can_interface}] motor slot {i} is not connected")
            states.append(motor.get_states())
        return states

    def get_joint_pos(self) -> np.ndarray:
        """Returns arm positions (rad) + gripper normalized [0=close, 1=open]."""
        states = self._read_states()
        pos = np.array([s["pos"] for s in states])
        if self.gripper_index is not None:
            pos[self.gripper_index] = self._motor_to_normalized(pos[self.gripper_index])
        return pos

    def get_joint_vel(self) -> np.ndarray:
        states = self._read_states()
        return np.array([s["vel"] for s in states])

    def get_observations(self) -> dict[str, np.ndarray]:
        """Return joint/gripper state plus force estimates."""
        states = self._read_states()
        raw_pos = np.array([s["pos"] for s in states])
        pos = raw_pos.copy()
        if self.gripper_index is not None:
            pos[self.gripper_index] = self._motor_to_normalized(pos[self.gripper_index])
        joint_eff = np.array([s["torq"] for s in states])
        joint_vel = np.array([s["vel"] for s in states])
        gravity_comp = self._compute_gravity_compensation(raw_pos)
        self._last_gravity_comp = gravity_comp.copy()
        force_feedback_torque = joint_eff - gravity_comp
        eef_force = self._compute_eef_force_estimate(raw_pos, joint_vel, force_feedback_torque)
        obs = {
            "joint_pos": pos,
            "joint_vel": joint_vel,
            "joint_eff": joint_eff,
            "gravity_comp": gravity_comp,
            "force_feedback_torque": force_feedback_torque,
            "eef_force": eef_force,
        }
        if self.gripper_index is not None:
            obs["joint_pos"] = pos[: self.gripper_index]
            obs["gripper_pos"] = pos[self.gripper_index : self.gripper_index + 1]
        return obs

    def get_motor_temperatures(self) -> np.ndarray:
        """Returns per-motor temperature in Celsius (NaN when unavailable)."""
        states = self._read_states()

        def _extract_temp(state: dict) -> float:
            vals = [
                state.get("t_mos"),
                state.get("t_rotor"),
                state.get("temp_mos"),
                state.get("temp_rotor"),
                state.get("temperature_mos"),
                state.get("temperature_rotor"),
                state.get("temp"),
                state.get("temperature"),
            ]
            numeric: list[float] = []
            for v in vals:
                if v is None:
                    continue
                fv = float(v)
                if np.isfinite(fv):
                    numeric.append(fv)
            return max(numeric) if numeric else float("nan")

        temps = [_extract_temp(s) for s in states]
        return np.asarray(temps, dtype=np.float32)

    def command_joint_pos(
        self,
        joint_pos: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        gripper_vel_limit: Optional[float] = None,
        gripper_torque_limit_nm: Optional[float] = None,
    ) -> None:
        """Update target position. Background loop sends continuously at send_rate_hz."""
        self._raise_if_background_error()
        if not self._connected:
            raise RuntimeError(f"[{self.can_interface}] cannot command joints before connect()")
        if kp is None:
            kp = self.default_kp
        if kd is None:
            kd = self.default_kd
        joint_pos = np.asarray(joint_pos, dtype=float)
        kp = np.asarray(kp, dtype=float)
        kd = np.asarray(kd, dtype=float)
        if joint_pos.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected joint_pos shape {(self.NUM_JOINTS,)}, "
                f"got {joint_pos.shape}"
            )
        if kp.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected kp shape {(self.NUM_JOINTS,)}, got {kp.shape}"
            )
        if kd.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected kd shape {(self.NUM_JOINTS,)}, got {kd.shape}"
            )
        with self._lock:
            self._cmd_pos = joint_pos
            self._cmd_kp = kp
            self._cmd_kd = kd
            self._cmd_gripper_vel_limit = (
                None if gripper_vel_limit is None else float(gripper_vel_limit)
            )
            self._cmd_gripper_torque_limit_nm = (
                None if gripper_torque_limit_nm is None else float(gripper_torque_limit_nm)
            )
            print(f"[{self.can_interface}] target pos: {joint_pos[0:6]}")
            print(
                f"[{self.can_interface}] current observation: {self.get_observations()['joint_pos'][0:6]}"
            )
            print(
                f"[{self.can_interface}] err: {joint_pos[0:6] - self.get_observations()['joint_pos'][0:6]}"
            )

    def command_joint_state(self, joint_state: dict) -> None:
        self.command_joint_pos(
            joint_state["pos"],
            joint_state["kp"],
            joint_state["kd"],
            joint_state.get("gripper_vel_limit"),
            joint_state.get("gripper_torque_limit_nm"),
        )
