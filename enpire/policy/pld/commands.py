from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from enpire.policy.autoresearch import build_control_request, send_control_request

from .launcher import build_pld_launch, build_score_launch


def _launch(args: argparse.Namespace) -> int:
    launch = build_pld_launch(
        args.rl_command,
        args.task,
        device=args.device,
        overrides=args.override,
    )
    print(launch.display_command)
    return 0 if args.dry_run else launch.run()


def _score(args: argparse.Namespace) -> int:
    launch = build_score_launch(
        args.data_dir,
        window=args.window,
        output_dir=args.output_dir,
        plot=args.plot,
    )
    print(launch.display_command)
    return 0 if args.dry_run else launch.run()


def _control(args: argparse.Namespace) -> int:
    base_url = args.url or os.environ.get("FORGE_CONTROL_URL", "http://127.0.0.1:8203")
    request = build_control_request(args.action, base_url)
    print(f"{request.method} {request.url}")
    if args.dry_run:
        return 0
    if request.method == "POST" and not args.confirm_control:
        raise SystemExit("Refusing to mutate the robot/RL loop without --confirm-control")
    print(json.dumps(send_control_request(request, timeout_s=args.timeout), indent=2))
    return 0


def add_rl_parser(commands: argparse._SubParsersAction) -> None:
    rl = commands.add_parser("rl", help="Launch the isolated real-world PLD runtime")
    roles = rl.add_subparsers(dest="rl_command", required=True)
    for role in ("actor", "learner"):
        parser = roles.add_parser(role, help=f"Launch the PLD {role}")
        parser.add_argument(
            "--task", choices=("pin_insertion", "gpu_insertion", "ziptie"), required=True
        )
        parser.add_argument("--device", type=int, default=None)
        parser.add_argument("--override", action="append", default=[])
        parser.add_argument("--dry-run", action="store_true")
        parser.set_defaults(handler=_launch)

    score = roles.add_parser("score", help="Compute source-faithful rolling success metrics")
    score.add_argument("--data-dir", type=Path, required=True)
    score.add_argument("--window", type=int, default=50)
    score.add_argument("--output-dir", type=Path)
    score.add_argument("--plot", action="store_true")
    score.add_argument("--dry-run", action="store_true")
    score.set_defaults(handler=_score)

    control = roles.add_parser("control", help="Inspect or control the robot-side RL loop")
    control.add_argument(
        "action", choices=("health", "help", "home", "pause", "restart", "resume")
    )
    control.add_argument("--url", default=None)
    control.add_argument("--timeout", type=float, default=3.0)
    control.add_argument("--confirm-control", action="store_true")
    control.add_argument("--dry-run", action="store_true")
    control.set_defaults(handler=_control)
