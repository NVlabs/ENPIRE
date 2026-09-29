# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mink")

from enpire.env.forge.robot.so101.constants import (  # noqa: E402
    SO101_HOME_JOINT_POS,
    SO101_JOINT_LIMITS_HIGH,
    SO101_JOINT_LIMITS_LOW,
)
from enpire.env.forge.robot.so101.kinematics import So101Kinematics  # noqa: E402


@pytest.fixture(scope="module")
def kin():
    return So101Kinematics()


def test_home_pose_is_in_front_of_and_above_the_base(kin):
    pos, quat = kin.forward_kinematics(SO101_HOME_JOINT_POS)

    np.testing.assert_allclose(pos, [0.2, 0.0, 0.15], atol=5e-3)
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_position_only_ik_reaches_fk_target(kin):
    q_true = np.array([0.4, -1.1, 0.9, 0.6, 0.3])
    target, _ = kin.forward_kinematics(q_true)

    q, pos_err, rot_err = kin.inverse_kinematics(target, seed=SO101_HOME_JOINT_POS)

    assert pos_err < 2e-3
    assert rot_err == 0.0
    np.testing.assert_allclose(kin.forward_kinematics(q)[0], target, atol=2e-3)


def test_pose_ik_recovers_reachable_orientation(kin):
    q_true = np.array([-0.3, -1.2, 1.0, 0.9, 0.5])
    target_pos, target_quat = kin.forward_kinematics(q_true)

    _, pos_err, rot_err = kin.inverse_kinematics(
        target_pos, target_quat, seed=SO101_HOME_JOINT_POS, max_iters=500
    )

    assert pos_err < 5e-3
    assert rot_err < 0.05


def test_ik_respects_joint_limits_for_unreachable_target(kin):
    q, pos_err, _ = kin.inverse_kinematics(np.array([1.0, 0.0, 0.2]), seed=SO101_HOME_JOINT_POS)

    assert pos_err > 0.3
    assert np.all(q >= SO101_JOINT_LIMITS_LOW - 1e-6)
    assert np.all(q <= SO101_JOINT_LIMITS_HIGH + 1e-6)
