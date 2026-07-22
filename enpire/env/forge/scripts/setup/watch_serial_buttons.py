#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Watch one or more serial button boards simultaneously.

Useful for answering the hardware question "which physical handle is this
serial device?" without bringing up CAN, Fello servers, or the control loop.

Usage:
    uv run scripts/setup/watch_serial_buttons.py
    uv run scripts/setup/watch_serial_buttons.py --port /dev/ttyACM2 --port /dev/ttyACM4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _is_rp2040_serial_name(name: str) -> bool:
    lowered = name.lower()
    return "2e8a" in lowered or "rp2040" in lowered


def discover_serial_devices() -> list[str]:
    by_id = Path("/dev/serial/by-id")
    ports: list[str] = []
    seen: set[str] = set()

    if by_id.exists():
        for entry in sorted(by_id.iterdir()):
            if not _is_rp2040_serial_name(entry.name):
                continue
            if not entry.name.endswith("-if00"):
                continue
            resolved = str(entry.resolve())
            if resolved not in seen:
                ports.append(resolved)
                seen.add(resolved)
        if ports:
            return ports

    for tty in sorted(Path("/sys/class/tty").glob("ttyACM*")):
        device_dir = tty / "device"
        if not device_dir.exists():
            continue
        try:
            uevent = (device_dir / "../uevent").read_text()
        except (FileNotFoundError, OSError):
            continue
        if "2e8a" not in uevent.lower() or "101f" not in uevent.lower():
            continue
        port = f"/dev/{tty.name}"
        if port not in seen:
            ports.append(port)
            seen.add(port)
    return ports


def find_by_id_alias(port: str) -> str | None:
    by_id = Path("/dev/serial/by-id")
    if not by_id.exists():
        return None
    resolved_target = Path(port).resolve()
    for entry in sorted(by_id.iterdir()):
        try:
            if entry.resolve() == resolved_target and entry.name.endswith("-if00"):
                return entry.name
        except OSError:
            continue
    return None


def read_buttons(ser: Any) -> list[int] | None:
    raw = ser.readline()
    line = raw.decode(errors="ignore").strip()
    if not line:
        return None
    try:
        values = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(values, list) or len(values) < 3:
        return None
    return [1 if bool(v) else 0 for v in values[:3]]


def format_edge(previous: tuple[int, int, int], current: tuple[int, int, int]) -> str:
    events: list[str] = []
    for idx, (prev, cur) in enumerate(zip(previous, current, strict=True)):
        if prev == cur:
            continue
        state = "PRESSED" if cur else "RELEASED"
        events.append(f"button {idx} {state}")
    return ", ".join(events) if events else "state changed"


@dataclass
class WatchTarget:
    port: str
    alias: str | None
    serial: Any
    last_buttons: tuple[int, int, int] | None = None

    @property
    def label(self) -> str:
        if self.alias:
            return f"{self.port} [{self.alias}]"
        return self.port


def main() -> int:
    import serial as _serial

    parser = argparse.ArgumentParser(
        description="Watch one or more serial button boards and print button edges"
    )
    parser.add_argument(
        "--port",
        action="append",
        default=None,
        help="Serial port to watch. Repeat to watch multiple ports.",
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    args = parser.parse_args()

    requested_ports = args.port or discover_serial_devices()
    if not requested_ports:
        print(
            "Error: no RP2040 serial button boards found. Pass --port explicitly.",
            file=sys.stderr,
        )
        return 1

    ports: list[str] = []
    seen: set[str] = set()
    for port in requested_ports:
        resolved = str(Path(port).resolve()) if Path(port).exists() else port
        if resolved not in seen:
            ports.append(resolved)
            seen.add(resolved)

    watchers: list[WatchTarget] = []
    try:
        for port in ports:
            ser = _serial.Serial(port, args.baudrate, timeout=0.1)
            time.sleep(0.2)
            while ser.in_waiting:
                ser.readline()
            watchers.append(
                WatchTarget(
                    port=port,
                    alias=find_by_id_alias(port),
                    serial=ser,
                )
            )
    except Exception as exc:
        print(f"Error opening serial port: {exc}", file=sys.stderr)
        for watcher in watchers:
            watcher.serial.close()
        return 1

    print()
    print("=" * 72)
    print("  Serial Button Watcher")
    print("=" * 72)
    print()
    print("Watching these ports:")
    for watcher in watchers:
        print(f"  - {watcher.label}")
    print()
    print("Press a physical handle button. The port that prints the edge is that handle.")
    print("Use Ctrl+C to stop.")
    print()

    try:
        while True:
            for watcher in watchers:
                values = read_buttons(watcher.serial)
                if values is None:
                    continue
                current = tuple(values)
                previous = watcher.last_buttons
                watcher.last_buttons = current
                if previous is None or current == previous:
                    continue
                stamp = time.strftime("%H:%M:%S")
                edge = format_edge(previous, current)
                print(f"[{stamp}] {watcher.label}: {edge} -> {list(current)}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for watcher in watchers:
            watcher.serial.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
