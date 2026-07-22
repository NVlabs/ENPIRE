# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from enpire.env.forge.robot.fello.fello_config import get_fello_xml_path, load_fello_config


class FelloKinematics:
    """Forward kinematics for the Fello leader model."""

    def __init__(
        self,
        side: Literal["left", "right"],
        *,
        xml_path: Path | None = None,
        eef_body: str = "arm_link_7",
        gripper_joint_value: float = 0.0,
    ):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side
        self.xml_path = (
            xml_path
            if xml_path is not None
            else get_fello_xml_path(load_fello_config(side=side), side)
        )
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Missing Fello XML for {side}: {self.xml_path}")

        self._model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self._data = mujoco.MjData(self._model)
        self._eef_body = eef_body
        self._eef_body_id = self._model.body(eef_body).id
        self._gripper_joint_value = float(gripper_joint_value)

    def forward_kinematics(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return Fello EEF position and xyzw quaternion.

        The first six joints drive the arm. Joint 7 is the normalized gripper
        value in teleop and is held fixed here so gripper motion does not alter
        the reported EEF orientation.
        """
        qpos_arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos_arr.size not in (6, 7):
            raise ValueError(
                f"Expected 6D or 7D Fello qpos, got shape {qpos_arr.shape}"
            )

        model_qpos = np.zeros(self._model.nq, dtype=np.float64)
        model_qpos[:6] = qpos_arr[:6]
        if self._model.nq >= 7:
            model_qpos[6] = self._gripper_joint_value

        self._data.qpos[:] = model_qpos
        mujoco.mj_forward(self._model, self._data)

        pos = self._data.xpos[self._eef_body_id].copy()
        quat_wxyz = self._data.xquat[self._eef_body_id].copy()
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        return pos.astype(np.float32), quat_xyzw.astype(np.float32)

