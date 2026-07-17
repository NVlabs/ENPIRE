#!/usr/bin/env python3
"""Watch button states published by the Fello leader servers.

This sits one layer above the raw serial watcher:
- `watch_serial_buttons.py` checks the RP2040 boards directly.
- `watch_fello_leader_buttons.py` checks what each running Fello leader server
  publishes via `get_info()`.

Usage:
    uv run scripts/setup/watch_fello_leader_buttons.py
    uv run scripts/setup/watch_fello_leader_buttons.py --host localhost
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import portal

from enpire.env.forge.robot.constants import LEFT_LEADER_PORT, RIGHT_LEADER_PORT


def format_edge(previous: tuple[int, int, int], current: tuple[int, int, int]) -> str:
    events: list[str] = []
    for idx, (prev, cur) in enumerate(zip(previous, current, strict=True)):
        if prev == cur:
            continue
        state = "PRESSED" if cur else "RELEASED"
        events.append(f"button {idx} {state}")
    return ", ".join(events) if events else "state changed"


@dataclass
class LeaderWatcher:
    label: str
    endpoint: str
    client: portal.Client
    last_buttons: tuple[int, int, int] | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watch button states from running Fello leader servers"
    )
    parser.add_argument("--host", default="localhost", help="Leader server host")
    parser.add_argument("--left-port", type=int, default=LEFT_LEADER_PORT)
    parser.add_argument("--right-port", type=int, default=RIGHT_LEADER_PORT)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Polling interval in seconds",
    )
    args = parser.parse_args()

    watchers = [
        LeaderWatcher(
            label="left",
            endpoint=f"{args.host}:{args.left_port}",
            client=portal.Client(f"{args.host}:{args.left_port}"),
        ),
        LeaderWatcher(
            label="right",
            endpoint=f"{args.host}:{args.right_port}",
            client=portal.Client(f"{args.host}:{args.right_port}"),
        ),
    ]

    print()
    print("=" * 72)
    print("  Fello Leader Button Watcher")
    print("=" * 72)
    print()
    print("Watching these leader servers:")
    for watcher in watchers:
        print(f"  - {watcher.label}: {watcher.endpoint}")
    print()
    print("Press a left or right handle button and watch which leader server reports it.")
    print("Use Ctrl+C to stop.")
    print()

    try:
        while True:
            for watcher in watchers:
                try:
                    _, buttons = watcher.client.get_info().result(timeout=1.0)
                except Exception as exc:
                    stamp = time.strftime("%H:%M:%S")
                    print(f"[{stamp}] {watcher.label}: read error from {watcher.endpoint}: {exc}")
                    time.sleep(args.poll_interval)
                    continue
                current = tuple(
                    int(v > 0.5) for v in np.asarray(buttons, dtype=np.float32).reshape(-1)[:3]
                )
                previous = watcher.last_buttons
                watcher.last_buttons = current
                if previous is None or current == previous:
                    continue
                stamp = time.strftime("%H:%M:%S")
                edge = format_edge(previous, current)
                print(f"[{stamp}] {watcher.label}: {edge} -> {list(current)}")
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
