# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import mujoco
import numpy as np

from enpire.env.forge.experimental.embodiment_tags import EmbodimentTag
from enpire.env.forge.experimental.key_remapping_utils import map_observation
from enpire.env.forge.robot.yam.mujoco_utils import MuJoCoKDL
from enpire.env.forge.robot.yam.yam_controller import _EEF_SITE_NAME, _YAM_XML_PATH, YamRobot


def _estimator_only_robot() -> YamRobot:
    robot = YamRobot.__new__(YamRobot)
    robot.NUM_JOINTS = 7
    robot.gripper_index = 6
    robot.can_interface = "test"
    robot._kdl = MuJoCoKDL(_YAM_XML_PATH)
    robot._kdl_lock = threading.Lock()
    return robot


def test_yam_eef_force_estimate_maps_joint_torque_through_site_jacobian() -> None:
    robot = _estimator_only_robot()
    q_arm = np.array([0.2, 1.0, 1.2, 0.1, -0.2, 0.3], dtype=np.float64)
    expected_force = np.array([1.2, -0.4, 2.0], dtype=np.float64)

    model = robot._kdl.model
    data = robot._kdl.data
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _EEF_SITE_NAME)
    data.qpos[:6] = q_arm
    data.qvel[:6] = 0.0
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    residual_joint_torque = np.zeros(7, dtype=np.float64)
    residual_joint_torque[:6] = jacp[:, :6].T @ expected_force
    joint_positions = np.zeros(7, dtype=np.float64)
    joint_positions[:6] = q_arm

    estimated_force = robot._compute_eef_force_estimate(
        joint_positions,
        np.zeros(7, dtype=np.float64),
        residual_joint_torque,
    )

    np.testing.assert_allclose(estimated_force, expected_force, atol=1e-5)


def test_map_observation_passes_optional_eef_force_states() -> None:
    observation = {
        "left_camera_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "right_camera_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "left_joint_pos": np.zeros(6, dtype=np.float32),
        "left_gripper_pos": np.zeros(1, dtype=np.float32),
        "right_joint_pos": np.ones(6, dtype=np.float32),
        "right_gripper_pos": np.ones(1, dtype=np.float32),
        "left_eef_force": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "right_eef_force": np.array([4.0, 5.0, 6.0], dtype=np.float32),
    }

    _, proprio = map_observation(
        observation, EmbodimentTag.XDOF_WRISTONLY, resolution=256
    )

    np.testing.assert_array_equal(
        proprio["eef_force_obs_left"], observation["left_eef_force"]
    )
    np.testing.assert_array_equal(
        proprio["eef_force_obs_right"], observation["right_eef_force"]
    )
