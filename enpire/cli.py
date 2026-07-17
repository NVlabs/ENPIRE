"""Dependency-light ENPIRE command-line interface."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from enpire import __version__
from enpire.policy.cap.commands import add_cap_parser
from enpire.env.forge.registry import default_skill_registry, default_tool_registry
from enpire.env.forge.yam.commands import add_station_parser
from enpire.policy.pld.commands import add_rl_parser
from enpire.env.forge.services.commands import add_services_parser


def _tools_list(args: argparse.Namespace) -> int:
    definitions = default_tool_registry().list(category=args.category)
    if args.json:
        print(json.dumps([asdict(item) | {"available": item.available} for item in definitions]))
        return 0
    for item in definitions:
        status = "ready" if item.available else f"install extra: {item.extra}"
        print(f"{item.name:<26} {item.category:<10} {status}")
        print(f"  {item.description}")
    return 0


def _examples_list(_: argparse.Namespace) -> int:
    root = Path(__file__).parent / "env" / "examples"
    examples = sorted(path.parent for path in root.glob("**/example.yaml"))
    for example in examples:
        print(example.relative_to(root))
    return 0


def _examples_run(args: argparse.Namespace) -> int:
    root = Path(__file__).parent / "env" / "examples"
    directory = (root / args.example).resolve()
    if root.resolve() not in directory.parents or not directory.is_dir():
        raise SystemExit(f"Unknown example: {args.example}")
    script = directory / "example.py"
    if not script.is_file():
        raise SystemExit(f"Example is not runnable yet: {args.example}")
    namespace = runpy.run_path(str(script))
    return int(namespace["main"](output=args.output))


def _skills_list(args: argparse.Namespace) -> int:
    definitions = default_skill_registry().list(category=args.category)
    if args.json:
        print(json.dumps([asdict(item) | {"available": item.available} for item in definitions]))
        return 0
    for item in definitions:
        status = "ready" if item.available else f"install extra: {item.extra}"
        print(f"{item.name:<32} {status}")
        print(f"  {item.description}")
    return 0


def _doctor(_: argparse.Namespace) -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"enpire: {__version__}")
    definitions = (*default_tool_registry().list(), *default_skill_registry().list())
    ready = sum(definition.available for definition in definitions)
    print(f"capabilities: {ready}/{len(definitions)} Python clients importable")
    print("hardware: not probed (use a station-specific doctor)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="enpire", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    tools = commands.add_parser("tools", help="Discover robot tools without importing them")
    tool_commands = tools.add_subparsers(dest="tools_command", required=True)
    tools_list = tool_commands.add_parser("list", help="List registered tools")
    tools_list.add_argument("--category", choices=("vision", "planning", "control", "vlm"))
    tools_list.add_argument("--json", action="store_true")
    tools_list.set_defaults(handler=_tools_list)

    examples = commands.add_parser("examples", help="Discover runnable examples")
    example_commands = examples.add_subparsers(dest="examples_command", required=True)
    examples_list = example_commands.add_parser("list", help="List examples")
    examples_list.set_defaults(handler=_examples_list)
    examples_run = example_commands.add_parser("run", help="Run a hardware-free example")
    examples_run.add_argument("example")
    examples_run.add_argument("--output", type=Path, default=Path("outputs/hello-environment"))
    examples_run.set_defaults(handler=_examples_run)

    skills = commands.add_parser("skills", help="Discover reusable code-as-policy skills")
    skill_commands = skills.add_subparsers(dest="skills_command", required=True)
    skills_list = skill_commands.add_parser("list", help="List registered skills")
    skills_list.add_argument("--category", default=None)
    skills_list.add_argument("--json", action="store_true")
    skills_list.set_defaults(handler=_skills_list)

    doctor = commands.add_parser("doctor", help="Check the dependency-light core install")
    doctor.set_defaults(handler=_doctor)
    add_station_parser(commands)
    add_cap_parser(commands)
    add_services_parser(commands)
    add_rl_parser(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
