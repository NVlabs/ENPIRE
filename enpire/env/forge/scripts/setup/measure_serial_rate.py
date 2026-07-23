#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the update rate of the serial footswitch device.

Usage:
    uv run scripts/setup/measure_serial_rate.py
    uv run scripts/setup/measure_serial_rate.py --port /dev/ttyACM2 --duration 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


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


def main() -> int:
    import serial as _serial

    parser = argparse.ArgumentParser(description="Measure serial footswitch update rate")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=5.0, help="Measurement window (seconds)")
    args = parser.parse_args()

    port = args.port or find_serial_device()
    if port is None:
        symlink = Path("/dev/serial-footswitch")
        if symlink.exists():
            port = str(symlink.resolve())
    if port is None:
        print("Error: device not found", file=sys.stderr)
        return 1

    print(f"Opening {port} @ {args.baudrate}...")
    ser = _serial.Serial(port, args.baudrate, timeout=1)
    time.sleep(0.3)
    while ser.in_waiting:
        ser.readline()

    print(f"Measuring for {args.duration}s — press buttons or leave idle...\n")

    timestamps: list[float] = []
    start = time.monotonic()
    while time.monotonic() - start < args.duration:
        raw = ser.readline()
        line = raw.decode(errors="ignore").strip()
        if line:
            timestamps.append(time.monotonic())

    ser.close()

    if len(timestamps) < 2:
        print("Not enough samples received.")
        return 1

    intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    avg_interval = sum(intervals) / len(intervals)
    min_interval = min(intervals)
    max_interval = max(intervals)
    avg_hz = 1.0 / avg_interval if avg_interval > 0 else 0
    elapsed = timestamps[-1] - timestamps[0]

    print(f"  Samples received : {len(timestamps)}")
    print(f"  Elapsed          : {elapsed:.2f}s")
    print(f"  Average rate     : {avg_hz:.1f} Hz")
    print(f"  Avg interval     : {avg_interval * 1000:.1f} ms")
    print(f"  Min interval     : {min_interval * 1000:.1f} ms")
    print(f"  Max interval     : {max_interval * 1000:.1f} ms")
    print()

    if avg_hz < 15:
        print(f"  WARNING: {avg_hz:.0f} Hz is slow. Brief taps may be missed.")
        print("  The min_hold_seconds latch (150ms) helps, but consider")
        print("  updating the RP2040 firmware to stream faster (50-100 Hz).")
    elif avg_hz < 30:
        print("  OK for 30 Hz control loop. The 150ms latch covers brief taps.")
    else:
        print("  Good — faster than the 30 Hz control loop.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
