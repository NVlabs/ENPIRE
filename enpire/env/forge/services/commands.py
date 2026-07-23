# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import subprocess

from .launcher import build_service_suite


def _start(args: argparse.Namespace) -> int:
    import os

    if getattr(args, "station", None):
        os.environ.setdefault("ENPIRE_STATION", args.station)
    selected = tuple(part.strip() for part in args.services.split(",") if part.strip())
    suite = build_service_suite(profile=args.profile, session=args.session, services=selected)
    for service in suite.services:
        print(f"{service.name:<10} {service.display_command}")
    if args.dry_run:
        return 0
    if any(service.moves_hardware for service in suite.services) and not args.confirm_motion:
        raise SystemExit("The selected profile starts robot hardware; pass --confirm-motion")
    return suite.start()


def _status(args: argparse.Namespace) -> int:
    return subprocess.run(["tmux", "list-windows", "-t", args.session], check=False).returncode


def add_services_parser(commands: argparse._SubParsersAction) -> None:
    services = commands.add_parser("services", help="Launch local perception, planning, and robot servers")
    sub = services.add_subparsers(dest="services_command", required=True)
    start = sub.add_parser("start", help="Start a tmux service suite")
    start.add_argument("--profile", choices=("perception", "cap-real", "robot", "all"), default="cap-real")
    start.add_argument("--services", default="", help="Comma-separated services overriding the profile")
    start.add_argument("--session", default="enpire")
    start.add_argument("--station", default=None, help="Station name; sets ENPIRE_STATION env var")
    start.add_argument("--confirm-motion", action="store_true")
    start.add_argument("--dry-run", action="store_true")
    start.set_defaults(handler=_start)
    status = sub.add_parser("status", help="Show windows in a service session")
    status.add_argument("--session", default="enpire")
    status.set_defaults(handler=_status)

