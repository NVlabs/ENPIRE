# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dummy RL policy server for sim testing.

Returns small random delta joint actions so learn_skill can run.
Replace with a real SAC agent for actual training.

Usage:
    uv run scripts/serve_rl_policy.py
    uv run scripts/serve_rl_policy.py --port 8965 --control-mode left
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import portal


class RandomRLPolicyServer:
    """Minimal RL policy server that returns random delta actions."""

    def __init__(self, port: int, control_mode: str, action_scale: float):
        self._port = port
        self._control_mode = control_mode
        self._action_scale = action_scale
        self._episode = 0
        self._step = 0

        self._server = portal.Server(port)
        self._server.bind("reset", self.reset)
        self._server.bind("step", self.step)
        self._server.bind("get_stats", self.get_stats)

    def reset(self, obs: dict) -> dict:
        self._episode += 1
        self._step = 0
        print(f"[RLPolicy] reset — episode {self._episode}")
        return {
            "action": self._sample_action(),
            "action_type": "delta_joint_angle",
        }

    def step(self, transition: dict) -> dict:
        self._step += 1
        reward = transition.get("reward", 0.0)
        source = transition.get("action_source", "rl")
        done = transition.get("done", False)
        if self._step % 50 == 0 or done:
            print(
                f"[RLPolicy] ep={self._episode} step={self._step} "
                f"r={reward:.3f} src={source} done={done}"
            )
        return {
            "action": self._sample_action(),
            "action_type": "delta_joint_angle",
        }

    def get_stats(self) -> dict:
        return {"episode": self._episode, "step": self._step}

    def _sample_action(self) -> dict:
        s = self._action_scale
        action = {}
        if self._control_mode in ("left", "both"):
            action["left_joint_pos"] = np.random.uniform(-s, s, size=6).astype(np.float64)
            action["left_gripper_pos"] = np.zeros(1, dtype=np.float64)
        if self._control_mode in ("right", "both"):
            action["right_joint_pos"] = np.random.uniform(-s, s, size=6).astype(np.float64)
            action["right_gripper_pos"] = np.zeros(1, dtype=np.float64)
        return action

    def serve(self):
        print(f"[RLPolicy] Random delta server on port {self._port} "
              f"(control_mode={self._control_mode}, scale={self._action_scale})")
        self._server.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8965)
    parser.add_argument("--control-mode", default="left", choices=["left", "right", "both"])
    parser.add_argument("--action-scale", type=float, default=0.02,
                        help="Max delta per joint per step (radians)")
    args = parser.parse_args()

    server = RandomRLPolicyServer(args.port, args.control_mode, args.action_scale)
    server.serve()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[RLPolicy] Shutting down")


if __name__ == "__main__":
    main()
