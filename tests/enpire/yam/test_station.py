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
