"""Dependency-light launcher for the isolated PLD runtime."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

PldRole = Literal["actor", "learner"]

_TASK_CONFIGS = {
    "pin_insertion": ("remote_yam_right_pos3", "pin_insertion_delta_eef"),
    "gpu_insertion": ("remote_yam_left_gpu_pos3", "gpu_insertion_delta_eef"),
    "ziptie": ("remote_yam_right_pos3", "ziptie_delta_eef"),
}


def runtime_root() -> Path:
    return Path(__file__).resolve().parent / "runtime"


@dataclass(frozen=True)
class PldLaunch:
    role: PldRole
    task: str
    command: tuple[str, ...]
    env: dict[str, str]

    @property
    def display_command(self) -> str:
        prefixes = [
            f"{name}={shlex.quote(value)}"
            for name, value in sorted(self.env.items())
            if name in {"CUDA_VISIBLE_DEVICES", "WANDB_MODE"}
        ]
        return " ".join([*prefixes, *(shlex.quote(part) for part in self.command)])

    def run(self) -> int:
        return subprocess.run(self.command, env=self.env, check=False).returncode


@dataclass(frozen=True)
class PldScoreLaunch:
    command: tuple[str, ...]
    env: dict[str, str]

    @property
    def display_command(self) -> str:
        return shlex.join(self.command)

    def run(self) -> int:
        return subprocess.run(self.command, env=self.env, check=False).returncode


def build_pld_launch(
    role: PldRole,
    task: str,
    *,
    device: int | None = None,
    overrides: Sequence[str] = (),
) -> PldLaunch:
    try:
        task_config, experiment_config = _TASK_CONFIGS[task]
    except KeyError as exc:
        raise ValueError(
            f"Unknown PLD task {task!r}; choose one of {', '.join(sorted(_TASK_CONFIGS))}"
        ) from exc

    project = runtime_root()
    command = (
        "uv",
        "run",
        "--project",
        str(project),
        "python",
        "-m",
        "enpire_pld.train",
        "system=gear-yam-24",
        f"task={task_config}",
        f"+experiment={experiment_config}",
        f"train.{role}=true",
        *overrides,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device if device is not None else (0 if role == "actor" else 1))
    env.setdefault("WANDB_MODE", "offline" if role == "actor" else "online")
    return PldLaunch(role=role, task=task, command=command, env=env)


def build_score_launch(
    data_dir: Path,
    *,
    window: int = 50,
    output_dir: Path | None = None,
    plot: bool = False,
) -> PldScoreLaunch:
    if window <= 0:
        raise ValueError("window must be positive")
    project = runtime_root()
    command = [
        "uv",
        "run",
        "--project",
        str(project),
        "python",
        "-m",
        "enpire_pld.sr_rolling_window",
        "--data-dir",
        str(data_dir.expanduser().resolve()),
        "--window",
        str(window),
    ]
    if output_dir is not None:
        command.extend(("--output-dir", str(output_dir.expanduser().resolve())))
    if not plot:
        command.append("--no-plot")
    return PldScoreLaunch(tuple(command), os.environ.copy())
