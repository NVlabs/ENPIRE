#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive footswitch button mapper.

Guides you through pressing each button one at a time, then prints
the config to paste into robot/device/{station}.yaml.

Usage:
    uv run python -m experimental.test_footswitch
"""
import json
import sys
import time
from pathlib import Path

import serial


def find_rp2040_ports() -> list[str]:
    by_id = Path("/dev/serial/by-id")
    ports = []
    if by_id.exists():
        for entry in sorted(by_id.iterdir()):
            name = entry.name.lower()
            if ("rp2040" in name or "2e8a" in name) and "if00" in name:
                ports.append(str(entry))
    return ports


def wait_for_press(connections: list[tuple[str, serial.Serial]], timeout: float = 30.0) -> tuple[int, int] | None:
    """Wait for a single button press. Returns (device_index, button_index) or None on timeout."""
    # Drain residual presses
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        for _, ser in connections:
            try:
                ser.readline()
            except Exception:
                pass

    # Wait for a fresh press
    deadline = time.monotonic() + timeout
    prev = [None] * len(connections)
    while time.monotonic() < deadline:
        for idx, (port, ser) in enumerate(connections):
            try:
                raw = ser.readline()
                line = raw.decode(errors="ignore").strip()
                if not line:
                    continue
                values = json.loads(line)
                if not isinstance(values, list) or len(values) < 3:
                    continue
            except Exception:
                continue

            if prev[idx] is not None:
                for btn in range(len(values)):
                    if values[btn] and not prev[idx][btn]:
                        return (idx, btn)
            prev[idx] = values[:]
    return None


def main():
    ports = find_rp2040_ports()
    if not ports:
        print("No RP2040 serial devices found!")
        sys.exit(1)

    print(f"\nFound {len(ports)} RP2040 device(s):")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")

    connections: list[tuple[str, serial.Serial]] = []
    for p in ports:
        try:
            s = serial.Serial(p, 115200, timeout=0.1)
            time.sleep(0.3)
            while s.in_waiting:
                s.readline()
            connections.append((p, s))
        except Exception as e:
            print(f"  Failed to open {p}: {e}")

    if not connections:
        print("No connections established!")
        sys.exit(1)

    # Define the guided sequence
    # Left pedal: [save, takeover, start]  (recording buttons + arm enable)
    # Right pedal: [home, pause, start]    (UI control buttons)
    steps = [
        ("LEFT",  "SAVE — stop/save recording"),
        ("LEFT",  "TAKEOVER — hold to enable LEFT arm"),
        ("LEFT",  "START — start recording"),
        ("RIGHT", "HOME — go to home position"),
        ("RIGHT", "PAUSE — pause policy"),
        ("RIGHT", "START — start/resume policy"),
    ]

    print("\n" + "=" * 60)
    print("  FOOTSWITCH BUTTON MAPPER")
    print("=" * 60)
    print()
    print("I will ask you to press 6 buttons, one at a time.")
    print("Wait for the prompt, then press and release.")
    print()

    raw_results: list[tuple[int, int]] = []

    for i, (side, label) in enumerate(steps):
        if i == 0:
            print(f"\n{'─' * 60}")
            print("  LEFT HANDLE footswitch (3 buttons)")
            print(f"{'─' * 60}")
        elif i == 3:
            print(f"\n{'─' * 60}")
            print("  RIGHT HANDLE footswitch (3 buttons)")
            print(f"{'─' * 60}")

        print(f"\n  [{i+1}/6] Press the {label} button now...")
        sys.stdout.flush()

        result = wait_for_press(connections)
        if result is None:
            print("      TIMEOUT — no button detected!")
            raw_results.append((-1, -1))
            continue

        dev_idx, btn_idx = result
        port = connections[dev_idx][0]
        print(f"      OK: device={Path(port).name}, raw_index={btn_idx}")
        raw_results.append((dev_idx, btn_idx))

    # Close connections
    for _, ser in connections:
        ser.close()

    # Build config
    # Left: steps 0,1,2 = save, takeover, start -> button_map[0]=save, [1]=takeover, [2]=start
    left_dev = raw_results[0][0] if raw_results[0][0] >= 0 else raw_results[1][0]
    left_port = connections[left_dev][0] if left_dev >= 0 else "UNKNOWN"
    left_map = [raw_results[0][1], raw_results[1][1], raw_results[2][1]]

    # Right: steps 3,4,5 = home, pause, start -> button_map = [home, pause, start]
    right_dev = raw_results[3][0] if raw_results[3][0] >= 0 else raw_results[4][0]
    right_port = connections[right_dev][0] if right_dev >= 0 else "UNKNOWN"
    right_map = [raw_results[3][1], raw_results[4][1], raw_results[5][1]]

    # Print results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print("\n  left footswitch:")
    print(f"    device: {left_port}")
    print(f"    button_map: {left_map}  (save={left_map[0]}, takeover={left_map[1]}, start={left_map[2]})")
    print("\n  right footswitch:")
    print(f"    device: {right_port}")
    print(f"    button_map: {right_map}  (home={right_map[0]}, pause={right_map[1]}, start={right_map[2]})")

    print("\n" + "=" * 60)
    print("  COPY EVERYTHING BELOW INTO THE CONVERSATION:")
    print("=" * 60)
    print()
    print(f"left_serial_port: {left_port}")
    print(f"left_button_map: {left_map}")
    print(f"right_serial_port: {right_port}")
    print(f"right_button_map: {right_map}")
    print()


if __name__ == "__main__":
    main()
