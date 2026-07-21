"""Portal client wrapper for the arm server."""

from __future__ import annotations

import time

import numpy as np
import portal

from . import config

_DT = 1.0 / config.INTERP_HZ


class ArmClient:
    def __init__(self, port: int = config.ARM_SERVER_PORT):
        self._client = portal.Client(f"127.0.0.1:{port}")

    def get_joint_pos(self) -> np.ndarray:
        return np.asarray(self._client.get_joint_pos().result(), dtype=np.float64)

    def command_joint_pos(
        self,
        q: np.ndarray,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> None:
        self._client.command_joint_pos(
            np.asarray(q, dtype=np.float64),
            np.asarray(kp, dtype=np.float64) if kp is not None else None,
            np.asarray(kd, dtype=np.float64) if kd is not None else None,
        ).result()

    def gravity_mode(self) -> None:
        """Send current pos with zero stiffness and damping-only — call in a loop."""
        q = self.get_joint_pos()[:6]
        self.command_joint_pos(
            q,
            kp=np.array(config.GRAVITY_KP),
            kd=np.array(config.GRAVITY_KD),
        )

    def smooth_move_to(
        self,
        q_target: np.ndarray,
        max_vel: float = config.MAX_VEL,
        settle: float = config.SETTLE_TIME,
        home_event=None,
    ) -> bool:
        """Interpolate to q_target at max_vel rad/s. Returns False if homed instead."""
        q_target = np.asarray(q_target, dtype=np.float64)
        max_step = max_vel * _DT
        kp = np.array(config.ARM_KP)
        kd = np.array(config.ARM_KD)

        q_setpoint = self.get_joint_pos()[:6].copy()
        deadline = time.time() + 15.0
        while True:
            if home_event is not None and home_event.is_set():
                return False

            q = self.get_joint_pos()[:6]
            if np.max(np.abs(q_target - q)) < 0.02:
                break
            if time.time() > deadline:
                break
            step = np.clip(q_target - q_setpoint, -max_step, max_step)
            q_setpoint = q_setpoint + step
            self.command_joint_pos(q_setpoint, kp=kp, kd=kd)
            time.sleep(_DT)

        time.sleep(settle)
        return True

    def home(self, q_home: np.ndarray, max_vel: float = config.HOME_VEL) -> None:
        """Interpolate joints back to q_home."""
        self.smooth_move_to(np.asarray(q_home), max_vel=max_vel, settle=0.5)

    def shutdown_server(self) -> None:
        """Tell the arm server to disconnect motors and exit."""
        try:
            self._client.shutdown().result(timeout=3.0)
        except Exception:
            pass
