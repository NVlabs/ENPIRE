# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one-line launches around Forge's original ``run_script.py``."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from enpire.env.forge.paths import FORGE_ROOT


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    script: str
    description: str


_TASKS = {
    item.name: item
    for item in (
        TaskDefinition(
            "pickup",
            "cap/saved_scripts/skill_library/pick_object.py",
            "Pick the visible object described by a text prompt.",
        ),
    )
}


def repository_root() -> Path:
    """Return the canonical, self-contained Forge runtime root."""

    return FORGE_ROOT


def list_tasks() -> tuple[TaskDefinition, ...]:
    return tuple(_TASKS[name] for name in sorted(_TASKS))


@dataclass(frozen=True)
class CapLaunch:
    task: TaskDefinition
    command: tuple[str, ...]
    env: dict[str, str]
    cwd: Path

    @property
    def display_command(self) -> str:
        station = self.env.get("ENPIRE_YAM_STATION")
        prompt = self.env.get("ENPIRE_PICK_PROMPT")
        prefix = []
        if station:
            prefix.append(f"ENPIRE_YAM_STATION={shlex.quote(station)}")
        if prompt:
            prefix.append(f"ENPIRE_PICK_PROMPT={shlex.quote(prompt)}")
        return " ".join([*prefix, *(shlex.quote(part) for part in self.command)])

    def run(self) -> int:
        return subprocess.run(self.command, cwd=self.cwd, env=self.env, check=False).returncode


def build_cap_launch(
    task: str,
    *,
    station: str,
    prompt: str,
    output: Path | None = None,
    record: bool = True,
    debug_ui: bool = False,
    overrides: Sequence[str] = (),
) -> CapLaunch:
    try:
        definition = _TASKS[task]
    except KeyError as exc:
        raise ValueError(f"Unknown CaP task {task!r}; choose one of {', '.join(sorted(_TASKS))}") from exc
    root = repository_root()
    runner = root / "run_script.py"
    script = root / definition.script
    if not runner.is_file() or not script.is_file():
        raise FileNotFoundError(f"Incomplete source checkout: expected {runner} and {script}")
    command = [
        sys.executable,
        str(runner),
        "robot=real_yam",
        f"script_file={definition.script}",
        "env.name=yam-real",
        "skill_library_path=cap/saved_scripts/skill_library",
        f"execution.record={'true' if record else 'false'}",
        f"debug_ui.enabled={'true' if debug_ui else 'false'}",
        "runtime.exit_on_error=true",
    ]
    if output is not None:
        command.append(f"script_output_dir={output.expanduser().resolve()}")
    command.extend(overrides)
    env = os.environ.copy()
    env["ENPIRE_YAM_STATION"] = station
    env["ENPIRE_STATION"] = station
    object_prompt = prompt.strip()
    if not object_prompt:
        raise ValueError("Pickup prompt must not be empty")
    if len(object_prompt) > 256 or any(ord(char) < 32 for char in object_prompt):
        raise ValueError("Pickup prompt must be a single printable line of at most 256 characters")
    env["ENPIRE_PICK_PROMPT"] = object_prompt
    return CapLaunch(definition, tuple(command), env, root)
