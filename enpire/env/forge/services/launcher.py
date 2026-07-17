"""Launch supported robot services in a predictable tmux session."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from enpire.env.forge.paths import FORGE_ROOT


@dataclass(frozen=True)
class ServiceDefinition:
    name: str
    command: tuple[str, ...]
    port: int | None
    description: str
    moves_hardware: bool = False

    @property
    def display_command(self) -> str:
        return shlex.join(self.command)


@dataclass(frozen=True)
class ServiceSuite:
    session: str
    root: Path
    services: tuple[ServiceDefinition, ...]

    def start(self) -> int:
        if shutil.which("tmux") is None:
            raise RuntimeError("tmux is required to launch a service suite")
        exists = subprocess.run(
            ["tmux", "has-session", "-t", self.session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if exists:
            raise RuntimeError(f"tmux session {self.session!r} already exists")
        for index, service in enumerate(self.services):
            inner = (
                f"cd {shlex.quote(str(self.root))}; "
                "if [ -f .enpire_env ]; then set -a; source .enpire_env; set +a; fi; "
                f"exec {service.display_command}"
            )
            if index == 0:
                command = [
                    "tmux", "new-session", "-d", "-s", self.session,
                    "-n", service.name, "-c", str(self.root), "bash", "-lc", inner,
                ]
            else:
                command = [
                    "tmux", "new-window", "-d", "-t", self.session,
                    "-n", service.name, "-c", str(self.root), "bash", "-lc", inner,
                ]
            subprocess.run(command, check=True)
        return 0


_PROFILES = {
    "perception": ("sam3", "anygrasp"),
    "cap-real": ("sam3", "anygrasp", "curobo", "nvidia"),
    "robot": ("yam",),
    "all": ("sam3", "anygrasp", "curobo", "nvidia", "yam"),
}


def build_service_suite(
    *,
    profile: str = "cap-real",
    session: str = "enpire",
    services: Sequence[str] = (),
) -> ServiceSuite:
    root = FORGE_ROOT
    python = sys.executable
    catalog = {
        "sam3": ServiceDefinition(
            "sam3", (python, "-m", "enpire.env.forge.tools.vision.serve_sam3", "--port", "6767", "--preload"), 6767,
            "Text-prompted SAM3 segmentation server.",
        ),
        "anygrasp": ServiceDefinition(
            "anygrasp",
            (
                python,
                "-m",
                "enpire.env.forge.tools.vision.serve_anygrasp",
                "--host",
                "127.0.0.1",
                "--port",
                "8122",
            ),
            8122,
            "Licensed AnyGrasp grasp-proposal server.",
        ),
        "curobo": ServiceDefinition(
            "curobo", (python, "-m", "enpire.env.forge.experimental.serve_portal_motion_planner", "--port", "8611", "--solver-speed", "fast", "--robot-type", "yam"), 8611,
            "Portal RPC motion-planning server.",
        ),
        "nvidia": ServiceDefinition(
            "nvidia", (python, "-m", "enpire.env.forge.cap.agent.providers.nvidia_server", "serve", "--host", "127.0.0.1", "--port", "8765", "--fresh-start"), 8765,
            "Optional NVIDIA VLM provider.",
        ),
        "yam": ServiceDefinition(
            "yam", (python, str(root / "launch.py"), "--mode=evaluation", "--no-attach"), None,
            "Real YAM arm/camera servers.", moves_hardware=True,
        ),
    }
    names = tuple(services) if services else _PROFILES.get(profile)
    if names is None:
        raise ValueError(f"Unknown service profile {profile!r}; choose one of {', '.join(sorted(_PROFILES))}")
    unknown = sorted(set(names) - catalog.keys())
    if unknown:
        raise ValueError(f"Unknown services: {', '.join(unknown)}")
    return ServiceSuite(session, root, tuple(catalog[name] for name in names))
