# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Client interface for the follower arms. We use this in real-world teleoperation/human takeover/policy deployment
"""

import numpy as np
import portal

from enpire.env.forge.robot.constants import LEFT_FOLLOWER_PORT


class FollowerRobotClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = LEFT_FOLLOWER_PORT,
        *,
        label: str = "follower arm",
    ):
        self._host = host
        self._port = int(port)
        self._label = label
        print(
            f"[YAM] Connecting {label} portal client at {host}:{port}",
            flush=True,
        )
        self._client = portal.Client(f"{host}:{port}")
        print(
            f"[YAM] Connected {label} portal client at {host}:{port}",
            flush=True,
        )

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> dict[str, np.ndarray]:
        return self._client.get_observations().result()
