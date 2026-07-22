# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from enpire.env.forge.yam.calibration import arm_client


class _Result:
    def __init__(self, value=None):
        self.value = value

    def result(self, timeout=None):
        return self.value


def test_calibration_arm_client_preserves_gain_aware_rpc(monkeypatch) -> None:
    calls: list[tuple] = []
    portal_client = SimpleNamespace(
        get_joint_pos=lambda: _Result(np.arange(6)),
        command_joint_pos=lambda *args: calls.append(args) or _Result(),
    )
    monkeypatch.setattr(arm_client.portal, "Client", lambda _: portal_client)
    client = arm_client.ArmClient(port=1234)

    np.testing.assert_array_equal(client.get_joint_pos(), np.arange(6))
    client.command_joint_pos(np.ones(6), kp=np.ones(6) * 2, kd=np.ones(6) * 3)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][1], np.ones(6) * 2)
    np.testing.assert_array_equal(calls[0][2], np.ones(6) * 3)
