# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reward server — Portal RPC server that computes per-step rewards.

Usage:
    python -m cap.reward.reward_server --mode insert_usb
    python -m cap.reward.reward_server --mode constant-1 --port 8500

Modes:
    constant-0   Always returns 0.0
    constant-1   Always returns 1.0
    random       Randomly returns 0.0 or 1.0 each call
    insert_usb   Returns 1.0 if USB drive is mounted, 0.0 otherwise
    gemini       VLM reward via Google Gemini API (requires GEMINI_API_KEY)
    smolvlm      VLM reward via local SmolVLM vLLM server (no API key needed)
"""

from __future__ import annotations

import argparse
import logging
import random
import threading
import time
from typing import Callable

import portal

from enpire.env.forge.cap.config import REWARD_SERVER_PORT
from enpire.env.forge.cap.diag.emitter import emit
from enpire.env.forge.cap.reward.insert_usb.reward import insert_usb_reward

logger = logging.getLogger(__name__)

_PRINT_INTERVAL = 0.1  # seconds between continuous reward prints


class RewardServer:
    """Portal RPC server wrapping a pluggable reward function.

    The server exposes a single method:
        get_reward(obs: dict) -> float

    It also continuously prints the reward at ~10 Hz regardless of RPC calls.

    Args:
        reward_fn: Callable that accepts an observation dict and returns a float reward.
        port: Portal RPC port to listen on (default: REWARD_SERVER_PORT).
    """

    def __init__(
        self, reward_fn: Callable[[dict], float], port: int = REWARD_SERVER_PORT
    ) -> None:
        self._reward_fn = reward_fn
        self._server = portal.Server(port, workers=4)
        self._server.bind("get_reward", self._get_reward)
        logger.info("[RewardServer] Bound on port %d", port)

    def _get_reward(self, obs: dict) -> float:
        emit("reward_server", "reward_recv")
        _t = time.time()
        result = float(self._reward_fn(obs))
        emit(
            "reward_server",
            "reward_compute",
            ms=(time.time() - _t) * 1000,
            reward=result,
        )
        emit("reward_server", "reward_send")
        return result

    def _print_loop(self) -> None:
        while True:
            try:
                reward = float(self._reward_fn({}))
                print(f"[RewardServer] reward={reward:.3f}", flush=True)
            except Exception as exc:
                logger.warning("[RewardServer] print_loop error: %s", exc)
            time.sleep(_PRINT_INTERVAL)

    def serve(self) -> None:
        """Start the server (blocks until stopped)."""
        t = threading.Thread(target=self._print_loop, daemon=True, name="reward-print")
        t.start()
        self._server.start()


# ---------------------------------------------------------------------------
# Built-in reward functions
# ---------------------------------------------------------------------------


def _constant_zero(_obs: dict) -> float:
    return 0.0


def _constant_one(_obs: dict) -> float:
    return 1.0


def _random_reward(_obs: dict) -> float:
    return float(random.randint(0, 1))


MODES: dict[str, Callable] = {
    "constant-0": _constant_zero,
    "constant-1": _constant_one,
    "random": _random_reward,
    "insert_usb": insert_usb_reward,
}


def _get_vlm_modes() -> dict[str, Callable]:
    """Lazily import VLM reward functions to avoid import-time file I/O failures."""
    from enpire.env.forge.cap.reward.gemini_reward import vlm_reward
    from enpire.env.forge.cap.reward.smolvlm_reward import smolvlm_reward

    return {"gemini": vlm_reward, "smolvlm": smolvlm_reward}


def get_modes() -> dict[str, Callable]:
    """Return all available reward modes, including lazily loaded VLM modes."""
    modes = dict(MODES)
    modes.update(_get_vlm_modes())
    return modes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward server")
    all_modes = get_modes()
    parser.add_argument(
        "--mode",
        choices=list(all_modes),
        default="insert_usb",
        help="Reward mode (default: insert_usb)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=REWARD_SERVER_PORT,
        help=f"Portal RPC port (default: {REWARD_SERVER_PORT})",
    )
    args = parser.parse_args()

    reward_fn = all_modes[args.mode]
    logger.info("Starting reward server — mode=%s, port=%d", args.mode, args.port)
    RewardServer(reward_fn, port=args.port).serve()


if __name__ == "__main__":
    main()
