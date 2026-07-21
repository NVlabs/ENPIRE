"""Read-only YAM station readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from enpire.env.forge.yam.station import StationProfile


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def check_station(
    station: StationProfile,
    *,
    dev_root: str | Path = "/dev",
    net_root: str | Path = "/sys/class/net",
) -> tuple[Check, ...]:
    """Check registered aliases and calibration paths without opening hardware."""

    dev_root = Path(dev_root)
    net_root = Path(net_root)
    checks: list[Check] = []
    for role in sorted(station.devices):
        if role.startswith("can_"):
            path = net_root / role
        else:
            path = dev_root / role
        checks.append(Check(f"device:{role}", path.exists(), str(path)))
    for name, camera in sorted(station.cameras.items()):
        device = camera.device.removeprefix("/dev/")
        path = dev_root / device
        passed = path.exists() or not camera.required
        checks.append(Check(f"camera:{name}", passed, str(path)))
    if station.calibration_bundle:
        path = Path(station.calibration_bundle).expanduser()
        checks.append(Check("calibration", path.exists(), str(path)))
    else:
        checks.append(Check("calibration", False, "no active calibration bundle"))
    return tuple(checks)


def station_ready(checks: tuple[Check, ...]) -> bool:
    return bool(checks) and all(check.passed for check in checks)
