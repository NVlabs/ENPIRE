# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SO-101 arm constants.

Joint angles are radians with the LeRobot degrees calibration, where each
joint reads zero at the middle of its range. The gripper is normalized to
[0, 1] with 0 closed and 1 open, matching the YAM convention.
"""

import os

import numpy as np

SO101_ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
SO101_GRIPPER_NAME = "gripper"
SO101_MOTOR_NAMES = SO101_ARM_JOINT_NAMES + (SO101_GRIPPER_NAME,)
SO101_DOF = len(SO101_ARM_JOINT_NAMES)

SO101_JOINT_LIMITS_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385])
SO101_JOINT_LIMITS_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121])

SO101_HOME_JOINT_POS = np.array([0.0, -1.41, 0.81, 1.17, 0.0])
SO101_HOME_GRIPPER_POS = np.array([1.0])

SO101_GRIPPER_MIN = 0.0
SO101_GRIPPER_MAX = 1.0

SO101_EE_SITE = "gripper_site"

SO101_FOLLOWER_PORT = 11337
SO101_LEADER_PORT = 11338

SO101_FOLLOWER_SERIAL = os.environ.get("ENPIRE_SO101_FOLLOWER_SERIAL", "/dev/ttyACM0")
SO101_LEADER_SERIAL = os.environ.get("ENPIRE_SO101_LEADER_SERIAL", "/dev/ttyACM1")
SO101_FOLLOWER_ID = os.environ.get("ENPIRE_SO101_FOLLOWER_ID", "enpire_so101_follower")
SO101_LEADER_ID = os.environ.get("ENPIRE_SO101_LEADER_ID", "enpire_so101_leader")
