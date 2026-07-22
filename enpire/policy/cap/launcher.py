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
        TaskDefinition("cube-pick", "cap/saved_scripts/examples/pick_cube.py", "Pick a visible cube."),
        TaskDefinition("gpu-handover", "cap/saved_scripts/gpu/gpu_handover.py", "Pick and hand over a GPU, then hover above its socket."),
        TaskDefinition("gpu-reset", "cap/saved_scripts/gpu/gpu_reset.py", "Unplug and reset one GPU."),
        TaskDefinition("gpu-reset-dual", "cap/saved_scripts/gpu/gpu_reset_dual.py", "Reset both GPU slots."),
        TaskDefinition("gpu-nudge", "cap/saved_scripts/gpu/gpu_nudge_motherboard_parallel.py", "Reorient the motherboard fixture."),
        TaskDefinition("gpu-unplug", "cap/saved_scripts/gpu/gpu_press_then_unplug.py", "Press and unplug the active GPU."),
        TaskDefinition("ziptie-reset", "cap/saved_scripts/ziptie/reset_ziptie_v2.py", "Run the canonical zip-tie reset."),
        TaskDefinition("ziptie-reward", "cap/saved_scripts/ziptie/get_rew_rgb.py", "Stream the RGB zip-tie reward."),
        TaskDefinition("ziptie-reward-trt", "cap/saved_scripts/ziptie/get_rew_rgb_trt.py", "Stream the TensorRT zip-tie reward."),
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
        prefix = [f"ENPIRE_YAM_STATION={shlex.quote(station)}"] if station else []
        return " ".join([*prefix, *(shlex.quote(part) for part in self.command)])

    def run(self) -> int:
        return subprocess.run(self.command, cwd=self.cwd, env=self.env, check=False).returncode


def build_cap_launch(
    task: str,
    *,
    station: str,
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
    return CapLaunch(definition, tuple(command), env, root)
