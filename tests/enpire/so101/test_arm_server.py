# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import socket
import sys
import threading

import numpy as np
import pytest

from enpire.env.forge.robot.so101 import arm_server
from enpire.env.forge.robot.so101.arm_server import FOLLOWER_RPCS, Args, So101ArmServer


class FakeRobot:
    def __init__(self):
        self.commands = []

    def get_joint_pos(self):
        return np.arange(6, dtype=float)

    def get_observations(self):
        return {"joint_pos": np.zeros(5), "gripper_pos": np.ones(1)}

    def command_joint_pos(self, joint_pos):
        self.commands.append(np.asarray(joint_pos))

    def command_joint_state(self, joint_state):
        self.commands.append(np.asarray(joint_state["pos"]))

    def get_motor_temperatures(self):
        return np.full(6, 30.0, dtype=np.float32)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_dry_run_does_not_touch_hardware(capsys, monkeypatch):
    monkeypatch.delitem(sys.modules, "enpire.env.forge.robot.so101.so101_controller", raising=False)

    arm_server.main(Args(serial="/dev/ttyTEST", robot_id="bench", dry_run=True))

    out = capsys.readouterr().out
    assert "serial=/dev/ttyTEST" in out
    assert "id=bench" in out
    assert "enpire.env.forge.robot.so101.so101_controller" not in sys.modules


def test_follower_client_round_trip():
    pytest.importorskip("portal")
    from enpire.env.forge.robot.yam.arm_client import FollowerRobotClient

    robot = FakeRobot()
    port = _free_port()
    server = So101ArmServer(robot, port, FOLLOWER_RPCS, "test")
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()

    client = FollowerRobotClient(port=port, label="so101 test")
    np.testing.assert_array_equal(client.get_joint_pos(), np.arange(6))
    np.testing.assert_array_equal(client.get_observations()["gripper_pos"], [1.0])
    client.command_joint_state({"pos": np.zeros(6)})
    temps = client._client.get_motor_temps().result()

    np.testing.assert_array_equal(temps, np.full(6, 30.0))
    assert len(robot.commands) == 1
    server.close()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
