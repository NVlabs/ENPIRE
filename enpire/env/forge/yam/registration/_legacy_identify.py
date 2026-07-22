#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive serial-number finder for the forge RL station.

Walks the operator through unplugging each physical device one at a time and
records the USB serial of the device that disappeared, so it can be assigned to
a logical role (can_leader_l, video_left_third, serial-right-buttons, ...). The
summary at the end is ready-to-paste content for ``forge_rl_station.rules`` and
``forge_rl_camera_aliases.json``.

USB-attached devices (CAN adapters, RP2040 button boards, the top webcam)
expose ``iSerial`` via sysfs. RealSense D405 firmware on this bench does not,
so RealSense detection goes through ``pyrealsense2`` instead.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

USB_ROOT = Path("/sys/bus/usb/devices")

CAN_VID_PID = ("1d50", "606f")
SERIAL_BUTTON_VID_PID = ("2e8a", "101f")
TOP_CAMERA_VID_PID = ("2b03", "f880")

CAN_ROLES = ["can_leader_l", "can_leader_r", "can_follow_l", "can_follow_r"]
BUTTON_ROLES = ["serial-left-buttons", "serial-right-buttons"]
REALSENSE_ROLES = ["video_left_third", "video_right", "video_left"]
TOP_CAMERA_ROLE = "video_top"
REALSENSE_TOP_ROLE = "video_top"

# Wait long enough for the kernel + pyrealsense2 to notice unplug/replug.
SETTLE_SECONDS = 1.5


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def scan_usb_serials(vendor_id: str, product_id: str) -> set[str]:
    serials: set[str] = set()
    if not USB_ROOT.exists():
        return serials
    for dev in USB_ROOT.iterdir():
        if _read(dev / "idVendor") != vendor_id:
            continue
        if _read(dev / "idProduct") != product_id:
            continue
        serial = _read(dev / "serial")
        if serial:
            serials.add(serial)
    return serials


def scan_realsense_serials() -> set[str]:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return set()
    serials: set[str] = set()
    for dev in rs.context().query_devices():
        try:
            serials.add(dev.get_info(rs.camera_info.serial_number))
        except Exception:
            continue
    return serials


def _prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        sys.exit(1)


def yes_no(prompt: str, *, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        answer = _prompt(prompt + suffix).strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def identify_unplug(scan: Callable[[], set[str]], kind: str, role: str) -> str | None:
    print(f"\n--- {kind}: {role} ---")
    while True:
        before = scan()
        print(f"  Connected {kind}(s): {sorted(before) if before else 'none'}")
        if not before:
            if yes_no(f"  No {kind} devices detected. Skip {role}?", default=True):
                return None
            continue

        choice = (
            _prompt(
                f"  Unplug the device you want to assign to '{role}', "
                "then press Enter (or type 'skip'): "
            )
            .strip()
            .lower()
        )
        if choice == "skip":
            return None

        time.sleep(SETTLE_SECONDS)
        after = scan()
        removed = before - after
        added = after - before

        if added:
            print(f"  Unexpected: new device(s) appeared: {sorted(added)}. Try again.")
            continue
        if not removed:
            print("  No device disappeared. Try again.")
            continue
        if len(removed) > 1:
            print(
                f"  More than one device disappeared: {sorted(removed)}. "
                "Plug everything back and try again."
            )
            continue

        serial = next(iter(removed))
        print(f"  Identified {role} -> serial {serial}")
        _prompt("  Plug it back in and press Enter to continue...")
        time.sleep(SETTLE_SECONDS)
        if serial not in scan():
            print(f"  Warning: serial {serial} not seen after replug.")
        return serial


def yes_no_top() -> bool:
    print("\n=== Top zed camera (2b03:f880, optional) ===")
    return yes_no("Identify the top zed camera?", default=True)


def yes_no_top_realsense() -> bool:
    print("\n=== RealSense top camera (optional, additional RealSense) ===")
    return yes_no("Identify a RealSense top camera?", default=False)


def emit_can_rule(role: str, serial: str) -> str:
    return (
        f'SUBSYSTEM=="net", ACTION=="add", ATTRS{{idVendor}}=="1d50", '
        f'ATTRS{{idProduct}}=="606f", ATTRS{{serial}}=="{serial}", '
        f'NAME="{role}", '
        f'RUN+="/sbin/ip link set {role} down", '
        f'RUN+="/sbin/ip link set {role} up type can bitrate 1000000"'
    )


def emit_button_rule(role: str, serial: str) -> str:
    return (
        f'ACTION=="add", SUBSYSTEM=="tty", KERNEL=="ttyACM*", '
        f'ATTRS{{idVendor}}=="2e8a", ATTRS{{idProduct}}=="101f", '
        f'ATTRS{{serial}}=="{serial}", ENV{{ID_USB_INTERFACE_NUM}}=="00", '
        f'SYMLINK+="{role}", MODE="0666", ENV{{ID_MM_DEVICE_IGNORE}}="1"'
    )


def emit_top_rule(role: str, serial: str) -> str:
    return (
        f'SUBSYSTEM=="video4linux", KERNEL=="video*", ATTR{{index}}=="0", '
        f'ATTRS{{idVendor}}=="2b03", ATTRS{{idProduct}}=="f880", '
        f'ATTRS{{serial}}=="{serial}", SYMLINK+="{role}", MODE="0666"'
    )


_REALSENSE_RULE_PLACEHOLDER = (
    'SUBSYSTEM=="video4linux", KERNEL=="video*", '
    'ENV{ID_VENDOR_ID}=="8086", ENV{ID_MODEL_ID}=="0b3a", MODE="0666", \\\n'
    '  RUN+="forge_rl_realsense_alias %k"\n'
    'SUBSYSTEM=="video4linux", KERNEL=="video*", '
    'ENV{ID_VENDOR_ID}=="8086", ENV{ID_MODEL_ID}=="0b5b", MODE="0666", \\\n'
    '  RUN+="forge_rl_realsense_alias %k"\n'
    'SUBSYSTEM=="video4linux", KERNEL=="video*", '
    'ENV{ID_VENDOR_ID}=="8086", ENV{ID_MODEL_ID}=="0b07", MODE="0666", \\\n'
    '  RUN+="forge_rl_realsense_alias %k"'
)


def _build_rules_text(results: dict[str, str]) -> str:
    sections: list[str] = []
    can_lines = [emit_can_rule(r, results[r]) for r in CAN_ROLES if r in results]
    if can_lines:
        sections.append("\n".join(can_lines))
    sections.append(_REALSENSE_RULE_PLACEHOLDER)
    if TOP_CAMERA_ROLE in results:
        sections.append(emit_top_rule(TOP_CAMERA_ROLE, results[TOP_CAMERA_ROLE]))
    btn_lines = [emit_button_rule(r, results[r]) for r in BUTTON_ROLES if r in results]
    if btn_lines:
        sections.append("\n".join(btn_lines))
    return "\n\n".join(sections) + "\n"


def _install_camera_aliases_only(aliases_path: Path) -> None:
    """Install only the RealSense serial alias map, preserving existing udev rules."""
    system_dir = Path(os.environ.get("FORGE_RL_SYSTEM_DIR", "/usr/local/lib/forge-rl-station"))
    aliases_dst = system_dir / "forge_rl_camera_aliases.json"
    subprocess.run(
        ["sudo", "install", "-d", "-m", "0755", str(system_dir)],
        check=True,
    )
    subprocess.run(
        ["sudo", "install", "-m", "0644", str(aliases_path), str(aliases_dst)],
        check=True,
    )
    subprocess.run(["sudo", "udevadm", "control", "--reload-rules"], check=False)
    subprocess.run(
        ["sudo", "udevadm", "trigger", "--action=add", "--subsystem-match=video4linux"],
        check=False,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify forge RL station devices by USB serial via the unplug-and-diff method.",
    )
    parser.add_argument("--can", action="store_true", help="Identify CAN adapters.")
    parser.add_argument("--buttons", action="store_true", help="Identify serial button boards.")
    parser.add_argument(
        "--cameras",
        action="store_true",
        help="Identify RealSense cameras (left_third/right/left wrist as video_left).",
    )
    parser.add_argument(
        "--top",
        action="store_true",
        help="Identify the optional top camera (USB webcam, 2b03:f880).",
    )
    args = parser.parse_args(argv)
    if not (args.can or args.buttons or args.cameras or args.top):
        # Default: walk every category, asking about the optional cameras.
        args.can = True
        args.buttons = True
        args.cameras = True
        args.top = None  # tri-state: ask the operator
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    print("Forge RL station device identification")
    print("======================================")
    print("Plug in every device you want to identify before starting.")
    print("You'll be prompted to unplug each one so we can record its serial.")

    try:
        import pyrealsense2  # noqa: F401

        realsense_ok = True
    except ImportError:
        realsense_ok = False
        if args.cameras:
            print("\nNote: pyrealsense2 not importable; RealSense identification will be skipped.")
            print(
                "      Run hardware/sync-forge-rl-station-udev.sh first, or set "
                "FORGE_RL_HELPER_PYTHON to a venv that has it."
            )

    results: dict[str, str] = {}

    if args.can:
        print("\n=== CAN adapters (1d50:606f) ===")
        for role in CAN_ROLES:
            serial = identify_unplug(lambda: scan_usb_serials(*CAN_VID_PID), "CAN adapter", role)
            if serial:
                results[role] = serial

    if args.buttons:
        print("\n=== Serial buttons (2e8a:101f) ===")
        for role in BUTTON_ROLES:
            serial = identify_unplug(
                lambda: scan_usb_serials(*SERIAL_BUTTON_VID_PID), "serial button", role
            )
            if serial:
                results[role] = serial

    if args.cameras and realsense_ok:
        print("\n=== RealSense cameras (left_third/right/left wrist as video_left) ===")
        for role in REALSENSE_ROLES:
            serial = identify_unplug(scan_realsense_serials, "RealSense camera", role)
            if serial:
                results[role] = serial

    do_top = args.top is True or (args.top is None and yes_no_top())
    if do_top:
        print("\n=== Top camera (2b03:f880) ===")
        serial = identify_unplug(
            lambda: scan_usb_serials(*TOP_CAMERA_VID_PID), "top camera", TOP_CAMERA_ROLE
        )
        if serial:
            results[TOP_CAMERA_ROLE] = serial

    if yes_no_top_realsense():
        if realsense_ok:
            print("\n=== RealSense top camera ===")
            serial = identify_unplug(scan_realsense_serials, "RealSense camera", REALSENSE_TOP_ROLE)
            if serial:
                results[REALSENSE_TOP_ROLE] = serial
        else:
            print("\n  Skipping RealSense top camera: pyrealsense2 not available.")

    print("\n======================================")
    print("Mapping summary")
    print("======================================")
    all_roles = CAN_ROLES + BUTTON_ROLES + REALSENSE_ROLES + [REALSENSE_TOP_ROLE, TOP_CAMERA_ROLE]
    for role in all_roles:
        if role in results:
            print(f"  {role:<22} -> {results[role]}")
        else:
            print(f"  {role:<22} -> (skipped)")

    udev_lines: list[str] = []
    for role in CAN_ROLES:
        if role in results:
            udev_lines.append(emit_can_rule(role, results[role]))
    for role in BUTTON_ROLES:
        if role in results:
            udev_lines.append(emit_button_rule(role, results[role]))
    if TOP_CAMERA_ROLE in results:
        udev_lines.append(emit_top_rule(TOP_CAMERA_ROLE, results[TOP_CAMERA_ROLE]))

    if udev_lines:
        print("\n--- Suggested forge_rl_station.rules entries ---")
        for line in udev_lines:
            print(line)

    all_realsense_roles = REALSENSE_ROLES + [REALSENSE_TOP_ROLE]
    json_text = ""
    aliases_path: Path | None = None
    if any(r in results for r in all_realsense_roles):
        items = [(results[r], r) for r in all_realsense_roles if r in results]
        lines = ["{"]
        lines.append(
            '  "_comment": "Map RealSense camera serial -> /dev/<alias> symlink for '
            'the forge RL station.",'
        )
        for i, (serial, alias) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            lines.append(f'  "{serial}": "{alias}"{comma}')
        lines.append("}")
        json_text = "\n".join(lines) + "\n"

        print("\n--- Suggested forge_rl_camera_aliases.json ---")
        print(json_text, end="")

    # --- Save to station directory ---
    print()
    station_name = _prompt("Station name (e.g. station_21): ").strip()
    while not station_name:
        station_name = _prompt("  Station name cannot be empty: ").strip()

    hardware_dir = Path(__file__).resolve().parent
    station_dir = hardware_dir / station_name
    station_dir.mkdir(exist_ok=True)

    rule_roles = set(CAN_ROLES + BUTTON_ROLES + [TOP_CAMERA_ROLE])
    wrote_rules = any(role in results for role in rule_roles)
    rules_path = station_dir / "forge_rl_station.rules"
    if wrote_rules:
        rules_path.write_text(_build_rules_text(results))
        print(f"Wrote {rules_path}")
    else:
        print(
            "Camera-only identification: not writing forge_rl_station.rules, "
            "so existing CAN/button udev rules are preserved."
        )

    if json_text:
        aliases_path = station_dir / "forge_rl_camera_aliases.json"
        aliases_path.write_text(json_text)
        print(f"Wrote {aliases_path}")

    sync_script = hardware_dir / "sync-forge-rl-station-udev.sh"
    sync_cmd = f"sudo bash hardware/sync-forge-rl-station-udev.sh --station {station_name}"

    print()
    if yes_no("Install now?"):
        if wrote_rules:
            subprocess.run(
                ["sudo", "bash", str(sync_script), "--station", station_name], check=True
            )
        elif aliases_path is not None:
            _install_camera_aliases_only(aliases_path)
        else:
            print("Nothing to install.")
    else:
        if wrote_rules:
            print(f"\nTo install later:\n  {sync_cmd}")
        elif aliases_path is not None:
            print(
                "\nTo install camera aliases later:\n"
                f"  sudo install -d -m 0755 /usr/local/lib/forge-rl-station\n"
                f"  sudo install -m 0644 {aliases_path} "
                "/usr/local/lib/forge-rl-station/forge_rl_camera_aliases.json\n"
                "  sudo udevadm control --reload-rules\n"
                "  sudo udevadm trigger --action=add --subsystem-match=video4linux"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
