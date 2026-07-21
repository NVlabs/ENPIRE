"""Launch arm servers and run all three camera calibrations in sequence.

Faithful port of yam-calibration/launch.py into the ENPIRE repo.

Usage:
    uv run enpire station calibrate-all --station my-yam --confirm-motion
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SESSION_ROBOT = "robots"
SESSION_CAL = "calibrator"


def _tmux(*args: str) -> None:
    subprocess.run(["tmux", *args], check=True)


def _has_session(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name], capture_output=True
    ).returncode == 0


def _reset_can_interface(iface: str) -> None:
    """Flush a stuck CAN TX queue by cycling the interface down/up.

    A previous failed server run can leave queued frames in the kernel qdisc
    (qlen=10 by default). If those frames were never ACKed they block new
    sends with ENOBUFS even after the process exits and motors are powered.
    Cycling the interface flushes the queue without changing bitrate config.
    """
    result = subprocess.run(["ip", "link", "show", iface], capture_output=True)
    if result.returncode != 0:
        return  # interface not present, skip silently
    down = subprocess.run(["sudo", "ip", "link", "set", iface, "down"], capture_output=True)
    up = subprocess.run(["sudo", "ip", "link", "set", iface, "up"], capture_output=True)
    if down.returncode != 0 or up.returncode != 0:
        print(f"  [warn] could not reset {iface} — if CAN sends fail with ENOBUFS, run:")
        print(f"         sudo ip link set {iface} down && sudo ip link set {iface} up")


_STEPS = [
    (
        "top",
        "TOP CAMERA",
        "Attach the ChArUco board to the gripper (or hold it near it).\n"
        "  The board will move with the arm during calibration.",
    ),
    (
        "left_wrist",
        "LEFT WRIST CAMERA",
        "Remove the board from the gripper.\n"
        "  Fix the ChArUco board stationary in the world at a location\n"
        "  the left wrist camera can see. The arm moves around the fixed board.",
    ),
    (
        "right_wrist",
        "RIGHT WRIST CAMERA",
        "Keep the board fixed in the world (or reposition it for the right arm).\n"
        "  The right arm will move around the fixed board.",
    ),
]


def _calibration_dirs(out_root: Path) -> set[Path]:
    return {p for p in out_root.glob("*") if p.is_dir()} if out_root.exists() else set()


def _newest_calibration_json(out_root: Path, before: set[Path]) -> Path | None:
    after = _calibration_dirs(out_root)
    candidates = (after - before) or after
    jsons = [d / "calibration.json" for d in candidates if (d / "calibration.json").exists()]
    if not jsons:
        return None
    return max(jsons, key=lambda p: p.stat().st_mtime)


def run_sequence(resolution: str | None = None) -> None:
    """Run all three calibrations in order. Called inside the calibrator tmux session."""
    import json

    from . import config
    from .calibrator import main as calibrator_main

    out_root = config.OUTPUT_ROOT
    json_paths: dict[str, str] = {}
    tw, th = config.TOP_CALIBRATION_RESOLUTION

    total = len(_STEPS)
    for i, (camera, title, setup) in enumerate(_STEPS, 1):
        if camera == "top":
            cam_resolution = resolution or f"{tw}x{th}"
        else:
            cam_resolution = resolution

        print("\n" + "=" * 60)
        print(f"  Step {i}/{total}: {title}")
        print()
        print(f"  {setup}")
        print("=" * 60)
        input("\n  Press Enter when ready to start calibration...\n")

        before = _calibration_dirs(out_root)
        argv = ["--camera", camera, "--no-interactive", "--confirm-motion"]
        if cam_resolution:
            argv.extend(["--resolution", cam_resolution])

        rc = int(calibrator_main(argv))
        if rc != 0:
            print(f"\n  [ERROR] {title} calibration failed (exit {rc}). Stopping.")
            sys.exit(1)

        jp = _newest_calibration_json(out_root, before)
        if jp is not None:
            cam_name = json.loads(jp.read_text()).get("camera_name", camera)
            json_paths[cam_name] = str(jp)

    print("\n" + "=" * 60)
    print("  All 3 calibrations complete.")
    print(f"  Final XML: {config.OUTPUT_XML}")
    print("=" * 60 + "\n")


def _env_prefix(station: str) -> str:
    """Build 'env KEY=val ...' prefix that works in bash, zsh, and fish."""
    pairs = [f"ENPIRE_STATION={station}"]
    for key in ("ENPIRE_YAM_MODEL_ROOT", "ENPIRE_YAM_CALIBRATED_XML_OUTPUT"):
        val = os.environ.get(key)
        if val:
            pairs.append(f"{key}={val}")
    return "env " + " ".join(pairs)


def launch(*, confirm_motion: bool, station: str, resolution: str | None = None) -> int:
    """Start arm servers in tmux, then run the full calibration sequence."""
    if not confirm_motion:
        raise RuntimeError("Pass --confirm-motion after clearing the robot workspace.")

    for session in (SESSION_ROBOT, SESSION_CAL):
        if _has_session(session):
            subprocess.run(["tmux", "kill-session", "-t", session])

    # Flush any stuck CAN TX queues from previous failed runs.
    from enpire.env.forge.robot.constants import (
        LEFT_FOLLOWER_CAN_INTERFACE,
        RIGHT_FOLLOWER_CAN_INTERFACE,
    )
    for iface in (LEFT_FOLLOWER_CAN_INTERFACE, RIGHT_FOLLOWER_CAN_INTERFACE):
        _reset_can_interface(iface)

    pfx = _env_prefix(station)
    server_cmd = (
        f"{sys.executable} -m enpire.env.forge.yam.calibration.server"
        f" --confirm-motion"
    )

    # Both arm servers in the 'robots' session (two windows)
    _tmux("new-session", "-d", "-s", SESSION_ROBOT)
    _tmux("send-keys", "-t", SESSION_ROBOT, f"{pfx} {server_cmd} --side left", "Enter")
    _tmux("new-window", "-t", SESSION_ROBOT)
    _tmux("send-keys", "-t", SESSION_ROBOT, f"{pfx} {server_cmd} --side right", "Enter")

    # Calibration sequence in 'calibrator' session
    sequence_cmd = (
        f"{pfx} {sys.executable} -m enpire.env.forge.yam.calibration.launch --sequence"
        + (f" --resolution {resolution}" if resolution else "")
    )
    _tmux("new-session", "-d", "-s", SESSION_CAL)
    _tmux("send-keys", "-t", SESSION_CAL, sequence_cmd, "Enter")

    print(f"  Arm servers starting in tmux session '{SESSION_ROBOT}'.")
    print(f"  Wait for motors to enable (green LED), then switch to '{SESSION_CAL}'.")
    print()

    if os.environ.get("TMUX"):
        os.execvp("tmux", ["tmux", "switch-client", "-t", SESSION_CAL])
    else:
        os.execvp("tmux", ["tmux", "attach-session", "-t", SESSION_CAL])

    return 0


def _sequence_entry() -> None:
    """Entry point when running as __main__ with --sequence."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--resolution", default=None)
    args = parser.parse_args()
    if args.sequence:
        run_sequence(resolution=args.resolution)
    else:
        print("Use 'enpire station calibrate-all' to launch.")
        sys.exit(1)


if __name__ == "__main__":
    _sequence_entry()
