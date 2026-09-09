# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import stat

import pytest

from enpire.env.forge.security import REDACTED
from enpire.env.forge.yam.doctor import check_station, station_ready
from enpire.env.forge.yam.station import (
    CameraProfile,
    ServiceProfile,
    StationProfile,
    load_station,
    save_station,
)


def test_station_round_trip_is_external_and_redacts_metadata(tmp_path):
    profile = StationProfile(
        station_id="yam-test",
        devices={"can_follow_l": "TEST-SERIAL"},
        cameras={"left_fixed": CameraProfile(device="video_left_third")},
        services={"actor": ServiceProfile(port=8964)},
        metadata={"api_key": "must-not-be-written", "owner": "test"},
    )

    path = save_station(profile, tmp_path)
    loaded = load_station("yam-test", tmp_path)

    assert loaded.devices == {"can_follow_l": "TEST-SERIAL"}
    assert loaded.cameras["left_fixed"].device == "video_left_third"
    assert loaded.services["actor"].port == 8964
    assert loaded.metadata == {"api_key": REDACTED, "owner": "test"}
    assert "must-not-be-written" not in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("station_id", ["../station", "spaces are unsafe", ""])
def test_station_id_is_safe_for_paths(station_id):
    with pytest.raises(ValueError, match="station_id"):
        StationProfile(station_id)


def test_service_port_validation():
    with pytest.raises(ValueError, match="port"):
        ServiceProfile(port=70000)


def test_station_doctor_uses_registered_aliases_and_calibration(tmp_path):
    dev = tmp_path / "dev"
    net = tmp_path / "net"
    calibration = tmp_path / "calibration"
    dev.mkdir()
    net.mkdir()
    calibration.mkdir()
    (dev / "video_left").touch()
    (net / "can_follow_l").mkdir()
    profile = StationProfile(
        station_id="test",
        devices={"can_follow_l": "CAN-SERIAL"},
        cameras={"left_wrist": CameraProfile("video_left")},
        calibration_bundle=str(calibration),
    )

    checks = check_station(profile, dev_root=dev, net_root=net)

    assert station_ready(checks) is True
    assert {check.name for check in checks} == {
        "device:can_follow_l",
        "camera:left_wrist",
        "calibration",
    }


def test_station_doctor_reports_missing_required_camera(tmp_path):
    profile = StationProfile(
        station_id="test",
        cameras={"left_wrist": CameraProfile("video_left")},
    )
    checks = check_station(profile, dev_root=tmp_path / "dev", net_root=tmp_path / "net")
    assert station_ready(checks) is False
    assert [check.name for check in checks if not check.passed] == [
        "camera:left_wrist",
        "calibration",
    ]


def test_allow_untested_d405_firmware_defaults_off(monkeypatch):
    """The firmware guard stays on unless a station explicitly opts out."""
    from enpire.env.forge.robot.realsense import allow_untested_d405_firmware

    monkeypatch.delenv("ENPIRE_ALLOW_D405_FIRMWARE", raising=False)
    assert allow_untested_d405_firmware() is False
    for value in ("0", "false", "no", ""):
        monkeypatch.setenv("ENPIRE_ALLOW_D405_FIRMWARE", value)
        assert allow_untested_d405_firmware() is False
    for value in ("1", "true", "YES", "On"):
        monkeypatch.setenv("ENPIRE_ALLOW_D405_FIRMWARE", value)
        assert allow_untested_d405_firmware() is True


def test_yam_gripper_sign_is_per_side_and_defaults_to_global(monkeypatch):
    """Mirrored gripper motors need opposite signs; default must not change."""
    from enpire.env.forge.robot.constants import YAM_GRIPPER_SIGN, yam_gripper_sign

    for side in ("left", "right"):
        monkeypatch.delenv(f"ENPIRE_YAM_GRIPPER_SIGN_{side.upper()}", raising=False)
        assert yam_gripper_sign(side) == YAM_GRIPPER_SIGN

    monkeypatch.setenv("ENPIRE_YAM_GRIPPER_SIGN_LEFT", "1")
    assert yam_gripper_sign("left") == 1
    assert yam_gripper_sign("right") == YAM_GRIPPER_SIGN  # untouched side unaffected

    monkeypatch.setenv("ENPIRE_YAM_GRIPPER_SIGN_RIGHT", "-1")
    assert yam_gripper_sign("right") == -1


def test_yam_gripper_sign_rejects_bad_values(monkeypatch):
    from enpire.env.forge.robot.constants import yam_gripper_sign

    for bad in ("0", "2", "left"):
        monkeypatch.setenv("ENPIRE_YAM_GRIPPER_SIGN_LEFT", bad)
        with pytest.raises(ValueError, match="must be 1 or -1"):
            yam_gripper_sign("left")


def test_arm_server_disconnects_motors_even_when_serve_raises():
    """The follower server must disable motors on every exit path.

    A FORCE_POS gripper latches its target and keeps drawing current until it is
    explicitly disabled, so leaving motors energised on shutdown can cook one.
    """
    from enpire.env.forge.robot.yam import arm_server

    class _Robot:
        def __init__(self) -> None:
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    robot = _Robot()
    arm_server._disconnect_quietly(robot, "can_test")
    assert robot.disconnected is True


def test_disconnect_quietly_does_not_mask_the_original_failure(caplog):
    from enpire.env.forge.robot.yam import arm_server

    class _Robot:
        def disconnect(self) -> None:
            raise RuntimeError("CAN bus already gone")

    # Must not raise: a shutdown error would otherwise replace whatever
    # exception actually stopped the server.
    arm_server._disconnect_quietly(_Robot(), "can_test")
