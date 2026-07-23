# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import time
from dataclasses import dataclass
from typing import Literal

import tyro

from enpire.env.forge.robot.constants import (
    LEFT_LEADER_CAN_INTERFACE,
    LEFT_LEADER_PORT,
    RIGHT_LEADER_CAN_INTERFACE,
    RIGHT_LEADER_PORT,
)
from enpire.env.forge.robot.fello.fello_config import get_config_value, load_fello_config


def _load_default_force_feedback_ratios() -> tuple[
    float, float, float, float, float, float, float
]:
    cfg = load_fello_config()
    ratios = get_config_value(
        cfg,
        "mapping",
        "force_feedback_ratio",
        default=(1.0 / 3.0,) * 7,
    )
    values = tuple(float(v) for v in ratios)
    if len(values) != 7:
        raise ValueError(
            f"mapping.force_feedback_ratio must have 7 values, got {len(values)}"
        )
    if any(v < 0.0 or v > 1.0 / 3.0 for v in values):
        raise ValueError("mapping.force_feedback_ratio values must be in [0, 1/3]")
    return values


DEFAULT_FORCE_FEEDBACK_RATIOS = _load_default_force_feedback_ratios()


class TmuxSession:
    def __init__(self, session: str):
        self.session = session
        self._first_window_done = False

        # Kill old session if it exists
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Start new session
        subprocess.run(["tmux", "new-session", "-d", "-s", session], check=True)

        # Bind Ctrl-\ to kill tmux server
        subprocess.run(["tmux", "bind-key", "-n", "C-\\", "kill-server"], check=True)

        # Show exit instructions in status bar
        subprocess.run(
            [
                "tmux",
                "set-option",
                "-t",
                session,
                "status-right",
                "Press Ctrl+\\ to exit",
            ],
            check=True,
        )

    def new_window(self, name=None, command=None):
        # Create window
        args = ["tmux", "new-window", "-P", "-F", "#{window_id}", "-t", self.session]
        if name:
            args += ["-n", name]
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        window_id = result.stdout.strip()

        # Replace the default :0 window
        if not self._first_window_done:
            subprocess.run(
                ["tmux", "kill-window", "-t", f"{self.session}:0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["tmux", "move-window", "-s", window_id, "-t", f"{self.session}:0"],
                check=True,
            )
            self._first_window_done = True

        # If a command is given, send it to the window
        if command:
            send_args = ["tmux", "send-keys", "-t", window_id, command, "C-m"]
            last_err = ""
            for attempt in range(5):
                proc = subprocess.run(send_args, capture_output=True, text=True)
                if proc.returncode == 0:
                    break
                last_err = (proc.stderr or proc.stdout).strip()
                if attempt < 4:
                    time.sleep(0.05)
                    continue
                raise RuntimeError(
                    f"tmux send-keys failed for window '{name}': {last_err}"
                )

    def attach(self):
        subprocess.run(["tmux", "attach", "-t", self.session])


@dataclass
class Args:
    mode: Literal["dev", "data_collection", "a5_data_collection", "evaluation"] = "dev"
    use_fello: bool = False  # beta WIP
    use_fello_right: bool = False
    """Launch only the right Fello leader server."""
    force_feedback: bool = False
    """Mirror YAM torque-minus-gravity back to Fello during data collection."""
    force_feedback_ratios: tuple[float, float, float, float, float, float, float] = (
        DEFAULT_FORCE_FEEDBACK_RATIOS
    )
    """Per-joint force-feedback intensity ratios. Each value must be <= 1/3."""
    fello_only: bool = False
    """Launch only Fello leader servers (no YAM leaders). For HIL in sim evaluation."""
    use_voice: bool = False
    """Enable microphone voice annotations during data collection."""

    no_translation_mode: bool = False
    """Disable translation-only mode during data collection (left-pedal buttons unbound)."""

    data_saving_path: str | None = None
    """Override for the directory where data_collection episodes are saved. Defaults to $YAM_RAW_PATH."""

    attach: bool = True
    """Whether to attach to the main session."""

    monitor_motor_temperature: bool = False
    """Whether to monitor the motor temperatures of all arms."""


def main(args: Args):
    if args.force_feedback and not (args.use_fello or args.use_fello_right):
        raise ValueError("--force-feedback requires --use-fello or --use-fello-right")

    # Start robot servers
    robots_session = TmuxSession("robots")

    if not args.fello_only:
        robots_session.new_window(
            "follow_l", "uv run robot/yam/arm_server.py --mode follower --side left"
        )
        robots_session.new_window(
            "follow_r", "uv run robot/yam/arm_server.py --mode follower --side right"
        )

    if args.use_fello or args.fello_only:
        robots_session.new_window(
            "leader_l",
            f"uv run robot/fello/fello_server.py --side left --can-interface {LEFT_LEADER_CAN_INTERFACE} --port {LEFT_LEADER_PORT}",
        )
        robots_session.new_window(
            "leader_r",
            f"uv run robot/fello/fello_server.py --side right --can-interface {RIGHT_LEADER_CAN_INTERFACE} --port {RIGHT_LEADER_PORT}",
        )
    elif args.use_fello_right:
        robots_session.new_window(
            "leader_r",
            f"uv run robot/fello/fello_server.py --side right --can-interface {RIGHT_LEADER_CAN_INTERFACE} --port {RIGHT_LEADER_PORT}",
        )
    elif args.mode != "evaluation":
        robots_session.new_window(
            "leader_l", "uv run robot/yam/arm_server.py --mode leader --side left"
        )
        robots_session.new_window(
            "leader_r", "uv run robot/yam/arm_server.py --mode leader --side right"
        )

    # Live motor temperature table (followers + leaders)
    # Be careful: this procedure burns like 1.5 CPU core and could cause fluctuating latency issues in data collection.
    # Turn it off if its save to reserve more CPU power.
    if args.monitor_motor_temperature:
        robots_session.new_window("motor_temps", "uv run robot/monitor_motor_temps.py")

    # Start camera servers
    cameras_session = TmuxSession("cameras")
    cameras_session.new_window("top", "")
    cameras_session.new_window("left", "")
    cameras_session.new_window("right", "")

    # Start main loop
    main_session = TmuxSession("main")
    if args.mode == "dev":
        fello_flag = " --use-fello" if args.use_fello or args.use_fello_right else ""
        main_session.new_window("main", f"uv run teleop_policy.py{fello_flag}")
    elif args.mode == "data_collection":
        dc_parts = [
            "uv run python tools/data_collection/run_data_collection.py",
            "--station=1",
            "--display-image",
            "--data-saving-path ./data/BC"
        ]
        if args.use_fello or args.use_fello_right:
            dc_parts.append("--use-fello")
        if args.force_feedback:
            dc_parts.append("--force-feedback")
            dc_parts.append("--force-feedback-ratios")
            dc_parts.extend(str(ratio) for ratio in args.force_feedback_ratios)
        if args.use_voice:
            dc_parts.append("--use-voice")
        if args.no_translation_mode:
            dc_parts.append("--no-translation-mode")
        if args.data_saving_path:
            dc_parts.append(f"--data-saving-path={args.data_saving_path}")
        main_session.new_window("main", " ".join(dc_parts))
    elif args.mode == "a5_data_collection":
        main_session.new_window(
            "main",
            "uv run python tools/data_collection/run_data_collection.py --task-list-path a5-tasks.txt",
        )
    elif args.mode == "evaluation":
        main_session.new_window("main", "echo 'Run your evaluation script here:'")

    # Attach to main session
    if args.attach:
        main_session.attach()


if __name__ == "__main__":
    main(tyro.cli(Args))
