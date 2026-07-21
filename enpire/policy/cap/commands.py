from __future__ import annotations

import argparse
from pathlib import Path

from .launcher import build_cap_launch, list_tasks


def _list(_: argparse.Namespace) -> int:
    for task in list_tasks():
        print(f"{task.name:<22} {task.description}")
        print(f"  {task.script}")
    return 0


def _run(args: argparse.Namespace) -> int:
    launch = build_cap_launch(
        args.task,
        station=args.station,
        output=args.output,
        record=not args.no_record,
        debug_ui=args.debug_ui,
        overrides=args.override,
    )
    print(launch.display_command)
    if args.dry_run:
        return 0
    if not args.confirm_motion:
        raise SystemExit("Refusing to move hardware without --confirm-motion")
    return launch.run()


def add_cap_parser(commands: argparse._SubParsersAction) -> None:
    cap = commands.add_parser("cap", help="Run source-faithful code-as-policy tasks")
    sub = cap.add_subparsers(dest="cap_command", required=True)
    listing = sub.add_parser("list", help="List migrated task scripts")
    listing.set_defaults(handler=_list)
    run = sub.add_parser("run", help="Run a task against a configured real YAM station")
    run.add_argument("task", choices=tuple(task.name for task in list_tasks()))
    run.add_argument("--station", required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--no-record", action="store_true")
    run.add_argument("--debug-ui", action="store_true")
    run.add_argument("--override", action="append", default=[])
    run.add_argument("--confirm-motion", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(handler=_run)

