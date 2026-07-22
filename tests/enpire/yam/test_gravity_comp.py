# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from enpire.cli import main as cli_main
from enpire.env.forge.tools.debug import yam_gravcomp_viewer as viewer


def test_gravity_comp_requires_explicit_motion_confirmation() -> None:
    with pytest.raises(RuntimeError, match="--confirm-motion"):
        viewer.main(["--camera", "none", "--no-launch-server"])


def test_gravity_comp_preserves_source_command_contract() -> None:
    commands: list[dict] = []
    client = SimpleNamespace(
        get_joint_pos=lambda: np.arange(7, dtype=np.float32),
        command_joint_state=commands.append,
    )

    qpos = viewer._enter_gravity_compensation(side="left", client=client)

    np.testing.assert_array_equal(qpos, np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(commands[0]["kp"], np.zeros(7, dtype=np.float32))
    assert commands[0]["gripper_torque_limit_nm"] == 0.0


def test_station_cli_forwards_safe_viewer_arguments(monkeypatch) -> None:
    received: list[list[str]] = []
    module_name = "enpire.env.forge.tools.debug.yam_gravcomp_viewer"
    fake = ModuleType(module_name)
    fake.main = lambda argv: received.append(argv) or 0
    monkeypatch.setitem(sys.modules, module_name, fake)

    result = cli_main(
        [
            "station",
            "gravcomp",
            "--station",
            "test-bench",
            "--both",
            "--camera",
            "none",
            "--no-gui",
            "--confirm-motion",
        ]
    )

    assert result == 0
    assert received == [["--both", "--camera", "none", "--no-gui", "--confirm-motion"]]
