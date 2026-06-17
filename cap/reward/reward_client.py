"""Reward client — connects to the reward server via Portal RPC."""

from __future__ import annotations

import portal

from cap.config import REWARD_SERVER_PORT


class RewardClient:
    """Portal RPC client for querying the reward server.

    Args:
        host: Hostname of the reward server.
        port: Portal RPC port (default: REWARD_SERVER_PORT).
    """

    def __init__(self, host: str = "localhost", port: int = REWARD_SERVER_PORT) -> None:
        self._client = portal.Client(f"{host}:{port}")

    def get_reward(self, obs: dict) -> float:
        """Query the reward server for the current reward given an observation."""
        return float(self._client.get_reward(obs).result())
