"""Launch the reward server with the specified reward function.

Usage:
    python cap/reward/serve_reward.py [reward_name]

Available rewards:
    insert_usb  — returns 1.0 when USB_DRIVE_NAME is mounted, else 0.0 (default)
"""

from __future__ import annotations

import importlib
import logging
import sys

from cap.config import REWARD_SERVER_PORT
from cap.reward.reward_server import RewardServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REWARDS: dict[str, str] = {
    "insert_usb": "cap.reward.insert_usb.reward.insert_usb_reward",
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "insert_usb"
    if name not in REWARDS:
        print(f"Unknown reward: {name!r}. Available: {list(REWARDS)}", file=sys.stderr)
        sys.exit(1)

    module_path, fn_name = REWARDS[name].rsplit(".", 1)
    mod = importlib.import_module(module_path)
    reward_fn = getattr(mod, fn_name)

    print(f"Starting reward server '{name}' on port {REWARD_SERVER_PORT}")
    RewardServer(reward_fn).serve()
