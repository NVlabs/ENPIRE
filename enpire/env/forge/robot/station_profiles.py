# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility view of ENPIRE's external station profiles.

The original Forge runtime imports this module from several places.  ENPIRE
keeps that API while moving station identities, camera serials, and hostnames
out of the source tree and into ``~/.config/enpire/stations`` (or
``ENPIRE_CONFIG_HOME``).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

from enpire.env.forge.yam.station import CameraProfile, StationProfile, load_station


@dataclass(frozen=True)
class StationCAN:
    left_follower: str
    right_follower: str
    left_leader: str
    right_leader: str


@dataclass(frozen=True)
class CameraConfig:
    name: str
    type: Literal["realsense", "zed"]
    symlink: str | None = None
    device_id: str | None = None


@dataclass(frozen=True)
class StationCameras:
    cameras: tuple[CameraConfig, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(camera.name for camera in self.cameras)


_DEFAULT_CAMERAS = StationCameras(
    (
        CameraConfig("top", "realsense", "/dev/video_top"),
        CameraConfig("left_fixed", "realsense", "/dev/video_left_third"),
        CameraConfig("left", "realsense", "/dev/video_left"),
        CameraConfig("right", "realsense", "/dev/video_right"),
    )
)

_profile_cache: StationProfile | None = None
_profile_cache_key: str | None = None


def resolve_station_key() -> str:
    """Resolve the active public station ID without hostname inference."""

    return os.environ.get("ENPIRE_STATION", os.environ.get("LECAR_STATION", "default")).strip()


def _active_profile() -> StationProfile | None:
    global _profile_cache, _profile_cache_key
    key = resolve_station_key()
    if key == "default":
        return None
    if _profile_cache_key != key:
        try:
            _profile_cache = load_station(key)
        except FileNotFoundError:
            _profile_cache = None
        _profile_cache_key = key
    return _profile_cache


def active_station_can() -> StationCAN:
    """Return CAN aliases from the external profile, then environment defaults."""

    profile = _active_profile()
    devices = profile.devices if profile is not None else {}
    defaults = {
        "left_follower": "can_follow_l",
        "right_follower": "can_follow_r",
        "left_leader": "can_leader_l",
        "right_leader": "can_leader_r",
    }
    values = {
        name: devices.get(
            name,
            os.environ.get(f"ENPIRE_{name.upper()}_CAN", default),
        )
        for name, default in defaults.items()
    }
    if sys.platform == "darwin" and profile is None:
        raise RuntimeError(
            "macOS CAN adapters require an external station profile; run "
            "`enpire station init <station>` and set ENPIRE_STATION."
        )
    return StationCAN(**values)


def _camera_config(name: str, profile: CameraProfile) -> CameraConfig:
    backend = profile.backend
    if backend not in {"realsense", "zed"}:
        raise ValueError(f"Unsupported camera backend {backend!r} for {name!r}")
    device = profile.device.strip()
    return CameraConfig(
        name=name,
        type=backend,
        symlink=device if device.startswith("/dev/") else None,
        device_id=None if device.startswith("/dev/") else device,
    )


def active_station_cameras() -> StationCameras:
    profile = _active_profile()
    if profile is None or not profile.cameras:
        return _DEFAULT_CAMERAS
    return StationCameras(
        tuple(_camera_config(name, camera) for name, camera in profile.cameras.items())
    )
