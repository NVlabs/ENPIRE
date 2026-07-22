# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import mujoco
import numpy as np


class MuJoCoKDL:
    """A simple class for computing inverse dynamics using MuJoCo."""

    def __init__(self, path: str):
        self.model = mujoco.MjModel.from_xml_path(os.path.expanduser(path))
        self.set_gravity(np.array([0, 0, -9.81]))

        # Disable all collisions
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        # Disable all joint limit
        self.model.jnt_limited[:] = 0
        # Data must be created after model edits to avoid constraint under-allocation.
        self.data = mujoco.MjData(self.model)
        self._site_ids: dict[str, int] = {}

    @property
    def joint_limits(self):
        return self.model.jnt_range

    def compute_inverse_dynamics(
        self, q: np.ndarray, qdot: np.ndarray, qdotdot: np.ndarray
    ) -> np.ndarray:
        assert len(q) == len(qdot) == len(qdotdot)
        length = len(q)
        self.data.qpos[:length] = q
        self.data.qvel[:length] = qdot
        self.data.qacc[:length] = qdotdot
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[:length]

    def has_site(self, site_name: str) -> bool:
        """Returns True when the MuJoCo model contains site_name."""
        return self._site_id(site_name) is not None

    def compute_site_force_from_joint_torque(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        joint_torque: np.ndarray,
        site_name: str,
        damping: float = 1e-4,
    ) -> np.ndarray:
        """Estimate Cartesian site force from joint torque.

        Solves J_pos.T * force ~= joint_torque with damped least squares.
        """
        assert len(q) == len(qdot) == len(joint_torque)
        site_id = self._site_id(site_name)
        if site_id is None:
            raise RuntimeError(f"MuJoCo site '{site_name}' not found")

        length = len(q)
        self.data.qpos[:length] = q
        self.data.qvel[:length] = qdot
        self.data.qacc[:length] = 0.0
        mujoco.mj_forward(self.model, self.data)

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        jac_pos = jacp[:, :length]
        return self._damped_least_squares(
            jac_pos.T,
            joint_torque,
            damping,
        ).astype(np.float32)

    def _site_id(self, site_name: str) -> int | None:
        if site_name not in self._site_ids:
            site_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_name,
            )
            if site_id < 0:
                return None
            self._site_ids[site_name] = int(site_id)
        return self._site_ids[site_name]

    @staticmethod
    def _damped_least_squares(
        matrix: np.ndarray,
        target: np.ndarray,
        damping: float,
    ) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float).reshape(-1)
        lhs = matrix.T @ matrix
        rhs = matrix.T @ target
        lhs += (float(damping) ** 2) * np.eye(lhs.shape[0], dtype=float)
        return np.linalg.solve(lhs, rhs)

    def set_gravity(self, gravity: np.ndarray) -> None:
        """Sets the gravity vector for the robot.

        Args:
            gravity (np.ndarray): The gravity vector as a NumPy array.

        """
        assert gravity.shape == (3,)
        self.model.opt.gravity = gravity
