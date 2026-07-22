# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

import pytest

from enpire.env.forge.yam.station import CameraProfile, StationProfile, save_station


def _reload_profiles():
    import enpire.env.forge.robot.station_profiles as station_profiles

    return importlib.reload(station_profiles)


def test_external_profile_drives_legacy_camera_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENPIRE_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("ENPIRE_STATION", "bench-a")
    save_station(
        StationProfile(
            station_id="bench-a",
            cameras={
                "top": CameraProfile(device="/dev/video_top", backend="zed"),
                "left": CameraProfile(device="serial-from-external-profile"),
            },
        )
    )

    cameras = _reload_profiles().active_station_cameras()

    assert cameras.names == ("top", "left")
    assert cameras.cameras[0].symlink == "/dev/video_top"
    assert cameras.cameras[0].type == "zed"
    assert cameras.cameras[1].device_id == "serial-from-external-profile"


def test_default_profile_contains_no_station_serials(monkeypatch) -> None:
    monkeypatch.delenv("ENPIRE_STATION", raising=False)
    monkeypatch.delenv("LECAR_STATION", raising=False)

    cameras = _reload_profiles().active_station_cameras()

    assert all(camera.device_id is None for camera in cameras.cameras)
    assert all(camera.symlink.startswith("/dev/") for camera in cameras.cameras)


def test_macos_requires_external_can_profile(monkeypatch) -> None:
    profiles = _reload_profiles()
    monkeypatch.setattr(profiles.sys, "platform", "darwin")

    with pytest.raises(RuntimeError, match="external station profile"):
        profiles.active_station_can()
