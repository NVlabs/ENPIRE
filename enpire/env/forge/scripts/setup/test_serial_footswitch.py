#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive test for the serial 3-button footswitch (Waveshare RP2040-Zero).

Reads one serial button board and shows which raw button index fires on each press.
After identifying all 3 buttons, prints the button_map for that specific device.

Usage:
    uv run scripts/setup/test_serial_footswitch.py
    uv run scripts/setup/test_serial_footswitch.py --port /dev/ttyACM2
    uv run scripts/setup/test_serial_footswitch.py --port /dev/ttyACM2 --prompt-order start,takeover,save
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROLES = ["save", "takeover", "start"]
ROLE_DESCRIPTIONS = {
    "save": "SAVE (save the current episode)",
    "takeover": "TAKEOVER (hold to enable teleoperation)",
    "start": "START (start recording a new episode)",
}
ROLE_ALIASES = {
    "save": "save",
    "save-recording": "save",
    "save_recording": "save",
    "takeover": "takeover",
    "start": "start",
    "start-recording": "start",
    "start_recording": "start",
}


def find_serial_device() -> str | None:
    by_id = Path("/dev/serial/by-id")
    if by_id.exists():
        for entry in sorted(by_id.iterdir()):
            name = entry.name.lower()
            if "2e8a" in name or "rp2040" in name:
                return str(entry.resolve())

    for tty in sorted(Path("/sys/class/tty").glob("ttyACM*")):
        device_dir = tty / "device"
        if not device_dir.exists():
            continue
        try:
            uevent = (device_dir / "../uevent").read_text()
        except (FileNotFoundError, OSError):
            continue
        if "2e8a" in uevent.lower() and "101f" in uevent.lower():
            return f"/dev/{tty.name}"
    return None


def read_buttons(ser) -> list[int] | None:
    raw = ser.readline()
    line = raw.decode(errors="ignore").strip()
    if not line:
        return None
    try:
        values = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(values, list) and len(values) >= 3:
        return values
    return None


def wait_for_press(ser) -> int | None:
    """Wait until exactly one button is pressed and return its index."""
    while True:
        values = read_buttons(ser)
        if values is None:
            continue
        pressed = [i for i, v in enumerate(values) if v]
        if len(pressed) == 1:
            return pressed[0]


def wait_for_release(ser) -> None:
    """Wait until all buttons are released."""
    while True:
        values = read_buttons(ser)
        if values is None:
            continue
        if not any(values):
            return


def parse_prompt_order(order: str) -> list[str]:
    tokens = [token.strip().lower().replace(" ", "-") for token in order.split(",")]
    if len(tokens) != 3:
        raise ValueError("prompt order must contain exactly 3 comma-separated roles")

    parsed: list[str] = []
    for token in tokens:
        role = ROLE_ALIASES.get(token)
        if role is None:
            valid = ", ".join(sorted(ROLE_ALIASES))
            raise ValueError(f"invalid role '{token}'. Valid names: {valid}")
        parsed.append(role)

    if len(set(parsed)) != 3 or set(parsed) != set(ROLES):
        raise ValueError(
            "prompt order must contain each role exactly once: save, takeover, start"
        )
    return parsed


def main() -> int:
    import serial as _serial

    parser = argparse.ArgumentParser(description="Test serial footswitch buttons")
    parser.add_argument("--port", type=str, default=None, help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument(
        "--prompt-order",
        type=str,
        default="save,takeover,start",
        help=(
            "Comma-separated prompt order using roles save,takeover,start. "
            "Example for left-handle workflow: start,takeover,save"
        ),
    )
    args = parser.parse_args()

    try:
        prompt_roles = parse_prompt_order(args.prompt_order)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    port = args.port or find_serial_device()
    if port is None:
        symlink = Path("/dev/serial-footswitch")
        if symlink.exists():
            port = str(symlink.resolve())
    if port is None:
        print("Error: serial footswitch not found. Pass --port or plug in the device.", file=sys.stderr)
        return 1

    print(f"Opening {port} @ {args.baudrate}...")
    ser = _serial.Serial(port, args.baudrate, timeout=1)
    time.sleep(0.3)
    # Drain any stale data
    while ser.in_waiting:
        ser.readline()

    print()
    print("=" * 60)
    print("  Serial Footswitch Button Mapping Tool")
    print("=" * 60)
    print()
    print("This will ask you to press each button one at a time.")
    print("Make sure only ONE button is pressed when prompted.")
    print(f"Prompt order: {' -> '.join(prompt_roles)}")
    print()

    button_map = [0, 0, 0]
    detected_by_role: dict[str, int] = {}
    assigned: set[int] = set()

    for prompt_idx, role in enumerate(prompt_roles):
        desc = ROLE_DESCRIPTIONS[role]
        while True:
            print(f"  [{prompt_idx + 1}/3] Press the button you want for: {desc}")
            print(f"         (press and hold...)")
            btn = wait_for_press(ser)
            if btn is None:
                continue
            if btn in assigned:
                prev_role = next((name for name, value in detected_by_role.items() if value == btn), "unknown")
                print(f"         Button {btn} is already assigned to '{prev_role}'. Try a different button.\n")
                wait_for_release(ser)
                continue
            print(f"         -> Detected button index: {btn}")
            detected_by_role[role] = btn
            button_map[ROLES.index(role)] = btn
            assigned.add(btn)
            wait_for_release(ser)
            print()
            break

    ser.close()

    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print()
    print("Detected raw button indices by requested role order:")
    for role in prompt_roles:
        print(f"  {role:8s} -> button {detected_by_role[role]}")
    print()
    print("Derived config mapping (`button_map` always stays in [save, takeover, start] order):")
    print()
    for role_idx, role in enumerate(ROLES):
        print(f"  Button {button_map[role_idx]}  →  {role} ({ROLE_DESCRIPTIONS[role]})")
    print()
    print("Add this to your fello_config.yaml under hardware.footswitch:")
    print()
    print(f"    button_map: {button_map}  # [{ROLES[0]}, {ROLES[1]}, {ROLES[2]}]")
    print()

    if button_map == [0, 1, 2]:
        print("  (This is the default — you can omit button_map entirely for this device.)")
    else:
        print("  Copy the line above into the config that uses this serial device.")
        print("  Left and right button boards may have different button_map values.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
