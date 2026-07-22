# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External, serial-free-in-Git station profiles."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from enpire.env.forge.security import redact

_STATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def enpire_config_home() -> Path:
    configured = os.environ.get("ENPIRE_CONFIG_HOME")
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".config") / "enpire"


def enpire_data_home() -> Path:
    configured = os.environ.get("ENPIRE_DATA_HOME")
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share") / "enpire"


@dataclass(frozen=True)
class CameraProfile:
    device: str
    backend: str = "realsense"
    calibration: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if self.backend not in {"realsense", "zed"}:
            raise ValueError(f"Unsupported camera backend: {self.backend}")


@dataclass(frozen=True)
class ServiceProfile:
    host: str = "127.0.0.1"
    port: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ValueError(f"Invalid service port: {self.port}")


@dataclass(frozen=True)
class StationProfile:
    station_id: str
    devices: dict[str, str] = field(default_factory=dict)
    cameras: dict[str, CameraProfile] = field(default_factory=dict)
    services: dict[str, ServiceProfile] = field(default_factory=dict)
    calibration_bundle: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _STATION_ID.fullmatch(self.station_id):
            raise ValueError(
                "station_id must contain only letters, digits, dot, underscore, and dash"
            )

    @property
    def data_dir(self) -> Path:
        return enpire_data_home() / "stations" / self.station_id


def _profile_path(station_id: str, root: str | Path | None = None) -> Path:
    directory = Path(root).expanduser() if root is not None else enpire_config_home() / "stations"
    return directory / f"{station_id}.yaml"


def save_station(profile: StationProfile, root: str | Path | None = None) -> Path:
    """Save a station outside the repository with owner-only permissions."""

    target = _profile_path(profile.station_id, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(redact(asdict(profile)), sort_keys=False)
    target.write_text(payload, encoding="utf-8")
    target.chmod(0o600)
    return target


def load_station(station_id: str, root: str | Path | None = None) -> StationProfile:
    source = _profile_path(station_id, root)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    cameras = {name: CameraProfile(**value) for name, value in payload.pop("cameras", {}).items()}
    services = {
        name: ServiceProfile(**value) for name, value in payload.pop("services", {}).items()
    }
    if "station_id" not in payload:
        payload["station_id"] = station_id
    return StationProfile(cameras=cameras, services=services, **payload)
