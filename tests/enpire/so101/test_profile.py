# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from enpire.env.forge.cap.env.base.profile import so101_profile
from enpire.env.forge.robot.so101.constants import SO101_HOME_JOINT_POS


def test_so101_profile_is_single_five_dof_arm():
    profile = so101_profile()

    assert profile.name == "so101"
    assert profile.arm_names == ("left",)
    assert not profile.is_bimanual
    assert profile.total_arm_dof == 5
    assert profile.camera_names == ()
    assert profile.xml_path is not None and profile.xml_path.is_file()


def test_so101_profile_home_is_within_limits():
    arm = so101_profile().arms["left"]

    np.testing.assert_array_equal(arm.home_joint_pos, SO101_HOME_JOINT_POS)
    assert np.all(arm.home_joint_pos >= arm.joint_limits_low)
    assert np.all(arm.home_joint_pos <= arm.joint_limits_high)
    assert arm.gripper_min == 0.0 and arm.gripper_max == 1.0


def test_so101_profile_side_is_configurable():
    profile = so101_profile(side="right")

    assert profile.arm_names == ("right",)
    assert profile.arms["right"].name == "right"


def test_so101_profile_home_is_a_copy():
    arm = so101_profile().arms["left"]
    arm.home_joint_pos[0] = 99.0

    assert SO101_HOME_JOINT_POS[0] == 0.0
