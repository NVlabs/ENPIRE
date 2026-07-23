# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAM bimanual station environment for CAP.

Wraps the MuJoCo SimBackend and implements EefControlProtocol using
pinocchio/pink IK — the same kinematics the physical YAM uses.

Used standalone or via ``create_env("yam")``. For real hardware,
cap_server still uses its internal control loop (to be migrated).
"""

from __future__ import annotations

import time

import numpy as np
import pink
import pinocchio as pin

from enpire.env.forge.cap.config import (
    CONTROL_PERIOD_S,
    GRIPPER_MAX,
    GRIPPER_MIN,
    GRIPPER_SETTLE_THRESH,
    HOME_JOINT_STATE,
    MOVE_EEF_MAX_DURATION_S,
    MOVE_EEF_MAX_VEL,
)
from enpire.env.forge.cap.env.profile import yam_profile
from enpire.env.forge.cap.server.sim_backend import SimBackend


def _make_se3(pos: np.ndarray, quat_xyzw: np.ndarray):
    """Build pinocchio SE3 from position + quaternion (xyzw)."""
    return pin.SE3(
        pin.Quaternion(quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]).toRotationMatrix(),
        pos,
    )


class YamEnv:
    """YAM sim environment — EnvProtocol + EefControlProtocol + SceneProtocol.

    Delegates physics/rendering to SimBackend. Adds pinocchio IK for
    _ik_servo using the YAM station URDF.
    """

    # Expose SimBackend constants for SimCameraClient compatibility
    CAMERA_HEIGHT = SimBackend.CAMERA_HEIGHT
    CAMERA_WIDTH = SimBackend.CAMERA_WIDTH

    def __init__(self, viewer: bool = False):
        self._backend = SimBackend(viewer=viewer)
        self._profile = yam_profile()

        # Pinocchio model for FK/IK
        from enpire.env.forge.robot.models.station.paths import get_station_urdf

        _urdf = str(get_station_urdf())
        self._pin_model = pin.buildModelFromUrdf(_urdf)
        self._pin_cfg = pink.Configuration(
            self._pin_model,
            self._pin_model.createData(),
            pin.neutral(self._pin_model),
        )
        self._pin_q_lower = self._pin_model.lowerPositionLimit.copy()
        self._pin_q_upper = self._pin_model.upperPositionLimit.copy()

        # FK data (separate from IK config to avoid contention)
        self._fk_data = self._pin_model.createData()

        # Frame IDs
        self._frame_ids = {
            "left": self._pin_model.getFrameId("left_grasp"),
            "right": self._pin_model.getFrameId("right_grasp"),
        }

        # IK tasks
        self._pink_tasks = {
            "left": pink.FrameTask("left_grasp", position_cost=1.0, orientation_cost=1.0),
            "right": pink.FrameTask("right_grasp", position_cost=1.0, orientation_cost=1.0),
        }
        self._posture = pink.PostureTask(cost=1e-3)

        self._dt = CONTROL_PERIOD_S

    # ------------------------------------------------------------------
    # EnvProtocol
    # ------------------------------------------------------------------

    def step(self) -> None:
        self._backend.step()

    def get_arm_observation(self, side: str) -> dict[str, np.ndarray]:
        obs = self._backend.get_arm_observation(side)
        # Add FK-derived EE pose
        jp = obs["joint_pos"]
        pos, quat = self._fk_single(side, jp)
        obs["ee_pos"] = pos
        obs["ee_quat"] = quat
        return obs

    def command_arm(self, side: str, cmd: dict) -> None:
        self._backend.command_arm(side, cmd)

    def render_rgb(self, camera_name: str) -> np.ndarray:
        return self._backend.render_rgb(camera_name)

    def render_depth(self, camera_name: str) -> np.ndarray:
        return self._backend.render_depth(camera_name)

    def get_camera_intrinsics(self, camera_name: str) -> list[float]:
        return self._backend.get_camera_intrinsics(camera_name)

    def get_camera_extrinsics(self, camera_name: str) -> dict:
        return self._backend.get_camera_extrinsics(camera_name)

    def close(self) -> None:
        self._backend.close()

    # ------------------------------------------------------------------
    # SceneProtocol
    # ------------------------------------------------------------------

    def setup_scene(self, name: str) -> dict:
        return self._backend.setup_scene(name)

    def clear_table(self) -> dict:
        return self._backend.clear_table()

    def get_object_positions(self) -> dict:
        return self._backend.get_object_positions()

    def get_scenes(self) -> dict:
        return self._backend.get_scenes()

    def set_body_pose(self, name: str, pos: list[float], quat_wxyz: list[float], gravity_comp: bool = True) -> dict:
        return self._backend.set_body_pose(name, pos, quat_wxyz, gravity_comp)

    # ------------------------------------------------------------------
    # EefControlProtocol
    # ------------------------------------------------------------------

    def _ik_servo(
        self,
        side: str,
        pos: np.ndarray,
        quat_xyzw: np.ndarray,
        gripper: float | None = None,
        max_duration: float = MOVE_EEF_MAX_DURATION_S,
        tol: float = 0.02,
        max_vel: float = MOVE_EEF_MAX_VEL,
    ) -> dict:
        """Move EE to target pose using pinocchio/pink IK."""
        pos = np.asarray(pos, dtype=np.float64)
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)

        if side not in ("left", "right"):
            return {"success": False, "reason": f"invalid side: {side}"}

        # Seed pink from current joint state
        q = self._read_q()
        self._pin_cfg.update(self._clamp_q(q))

        # Hold non-moving sides, move target side
        for s, task in self._pink_tasks.items():
            task.set_target_from_configuration(self._pin_cfg)
        self._pink_tasks[side].set_target(_make_se3(pos, quat_xyzw))
        self._posture.set_target_from_configuration(self._pin_cfg)

        tasks = list(self._pink_tasks.values()) + [self._posture]
        moving_task = self._pink_tasks[side]
        max_iters = int(max_duration / self._dt)

        for i in range(max_iters):
            pos_err = np.linalg.norm(moving_task.compute_error(self._pin_cfg)[:3])
            moving_task.gain = min(1.0, max_vel * self._dt / (pos_err + 1e-6))

            vel = pink.solve_ik(self._pin_cfg, tasks, self._dt, solver="quadprog")
            self._pin_cfg.integrate_inplace(vel, self._dt)

            # Write joint targets to backend and step
            q_solved = self._pin_cfg.q
            for s in ("left", "right"):
                arm = self._profile.arms[s]
                jp = q_solved[arm.q_slice].copy()
                gp = gripper if (gripper is not None and s == side) else None
                obs = self._backend.get_arm_observation(s)
                cur_gp = obs["gripper_pos"][0]
                self._backend.command_arm(s, {
                    "pos": np.concatenate([jp, [gp if gp is not None else cur_gp]])
                })
            self._backend.step()

            ik_err = np.linalg.norm(moving_task.compute_error(self._pin_cfg))
            if ik_err < 1e-3:
                # Settle: keep commanding until robot reaches target
                for _ in range(20):
                    self._backend.step()
                break

            time.sleep(self._dt)

        final_err = np.linalg.norm(moving_task.compute_error(self._pin_cfg)[:3])
        return {
            "success": True,
            "reason": "ok",
            "feasible": bool(final_err < tol),
            "moved": True,
            "pos_err": float(final_err),
        }

    def go_home(self, max_joint_vel: float = 1.0) -> dict:
        """Return to home configuration via joint interpolation."""
        starts = {}
        for side in ("left", "right"):
            obs = self._backend.get_arm_observation(side)
            starts[side] = (obs["joint_pos"].copy(), obs["gripper_pos"].copy())

        targets = {}
        for side in ("left", "right"):
            targets[side] = (
                HOME_JOINT_STATE[f"{side}_joint_pos"].copy(),
                HOME_JOINT_STATE[f"{side}_gripper_pos"].copy(),
            )

        max_disp = max(
            np.max(np.abs(targets[s][0] - starts[s][0])) for s in ("left", "right")
        )
        n_steps = max(1, int(max_disp / max_joint_vel / self._dt))

        for i in range(n_steps):
            t = (i + 1) / n_steps
            for side in ("left", "right"):
                jp = starts[side][0] + t * (targets[side][0] - starts[side][0])
                gp = starts[side][1] + t * (targets[side][1] - starts[side][1])
                self._backend.command_arm(side, {"pos": np.concatenate([jp, gp])})
            self._backend.step()

        return {"success": True, "reason": "ok"}

    def set_gripper(self, side: str, value: float, timeout: float = 1.5) -> dict:
        """Set gripper. 0=closed, 1=open."""
        if side not in ("left", "right"):
            return {"success": False, "reason": f"invalid side: {side}"}

        value = float(np.clip(value, GRIPPER_MIN, GRIPPER_MAX))
        start = time.time()

        while time.time() - start < timeout:
            # Read current arm state, set gripper target
            for s in ("left", "right"):
                obs = self._backend.get_arm_observation(s)
                gp = value if s == side else obs["gripper_pos"][0]
                self._backend.command_arm(s, {"pos": np.concatenate([obs["joint_pos"], [gp]])})
            self._backend.step()

            # Check convergence
            obs = self._backend.get_arm_observation(side)
            if abs(obs["gripper_pos"][0] - value) < GRIPPER_SETTLE_THRESH:
                break

        return {"success": True, "reason": "ok"}

    # ------------------------------------------------------------------
    # FK helpers
    # ------------------------------------------------------------------

    def _fk_single(self, side: str, jp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """FK for one arm. Returns (pos, quat_xyzw)."""
        q = pin.neutral(self._pin_model)
        arm = self._profile.arms[side]
        q[arm.q_slice] = jp
        pin.forwardKinematics(self._pin_model, self._fk_data, q)
        pin.updateFramePlacements(self._pin_model, self._fk_data)
        T = self._fk_data.oMf[self._frame_ids[side]]
        return T.translation.copy(), pin.Quaternion(T.rotation).coeffs()

    def _read_q(self) -> np.ndarray:
        """Build pinocchio q-vector from current backend state."""
        q = pin.neutral(self._pin_model)
        for side in ("left", "right"):
            arm = self._profile.arms[side]
            obs = self._backend.get_arm_observation(side)
            q[arm.q_slice] = obs["joint_pos"]
        return q

    def _clamp_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self._pin_q_lower, self._pin_q_upper)
