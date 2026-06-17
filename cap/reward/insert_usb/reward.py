"""Reward function: detects whether the configured USB drive is inserted.

Returns 1.0 if USB_DRIVE_NAME is currently mounted, 0.0 otherwise.
Detection is done by scanning /proc/mounts for the drive label — no external
dependencies required.
"""

from __future__ import annotations

from cap.config import USB_DRIVE_NAME


def insert_usb_reward(obs: dict) -> float:
    """Return 1.0 if USB_DRIVE_NAME is mounted, 0.0 otherwise."""
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                # /proc/mounts encodes spaces as \040 (octal 040 = ASCII space)
                mountpoint = parts[1].replace("\\040", " ")
                if USB_DRIVE_NAME in mountpoint:
                    return 1.0
    except OSError:
        pass
    return 0.0
