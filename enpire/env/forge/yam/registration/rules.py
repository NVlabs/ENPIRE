# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure rendering functions for YAM device registration.

The actual rule syntax remains implemented by the characterized Forge source in
``_legacy_identify.py``. This module removes its interactive and privileged side
effects from the reusable API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from enpire.env.forge.yam.registration import _legacy_identify as legacy


@dataclass(frozen=True)
class RegistrationResult:
    """Logical hardware roles mapped to stable device serials."""

    station_id: str
    roles: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.roles) - set(supported_roles())
        if unknown:
            raise ValueError(f"Unknown YAM registration roles: {sorted(unknown)}")
        if len(set(self.roles.values())) != len(self.roles):
            raise ValueError("A physical serial cannot be assigned to multiple roles")


def supported_roles() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                *legacy.CAN_ROLES,
                *legacy.BUTTON_ROLES,
                *legacy.REALSENSE_ROLES,
                legacy.TOP_CAMERA_ROLE,
            ]
        )
    )


def render_udev_rules(registration: RegistrationResult) -> str:
    """Render the original Forge udev syntax without installing it."""

    return legacy._build_rules_text(registration.roles)


def render_camera_aliases(registration: RegistrationResult) -> str:
    """Render the serial-to-device-role map consumed by the udev helper."""

    roles = [*legacy.REALSENSE_ROLES, legacy.REALSENSE_TOP_ROLE]
    mapping = {
        serial: role for role in roles if (serial := registration.roles.get(role)) is not None
    }
    return json.dumps(mapping, indent=2, sort_keys=True) + "\n"


def discover_registration(
    station_id: str,
    *,
    can: bool = True,
    buttons: bool = True,
    cameras: bool = True,
    top_camera: bool = False,
) -> RegistrationResult:
    """Run the original unplug-and-diff identification without installing files."""

    roles: dict[str, str] = {}
    if can:
        for role in legacy.CAN_ROLES:
            serial = legacy.identify_unplug(
                lambda: legacy.scan_usb_serials(*legacy.CAN_VID_PID), "CAN adapter", role
            )
            if serial:
                roles[role] = serial
    if buttons:
        for role in legacy.BUTTON_ROLES:
            serial = legacy.identify_unplug(
                lambda: legacy.scan_usb_serials(*legacy.SERIAL_BUTTON_VID_PID),
                "serial button",
                role,
            )
            if serial:
                roles[role] = serial
    if cameras:
        for role in legacy.REALSENSE_ROLES:
            serial = legacy.identify_unplug(legacy.scan_realsense_serials, "RealSense", role)
            if serial:
                roles[role] = serial
    if top_camera:
        serial = legacy.identify_unplug(
            lambda: legacy.scan_usb_serials(*legacy.TOP_CAMERA_VID_PID),
            "top camera",
            legacy.TOP_CAMERA_ROLE,
        )
        if serial:
            roles[legacy.TOP_CAMERA_ROLE] = serial
    return RegistrationResult(station_id=station_id, roles=roles)


def write_registration_files(
    registration: RegistrationResult, directory: str | Path
) -> tuple[Path, Path]:
    """Write generated rules under external station data, never into source."""

    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rules = root / "enpire_yam_station.rules"
    aliases = root / "camera_aliases.json"
    rules.write_text(render_udev_rules(registration), encoding="utf-8")
    aliases.write_text(render_camera_aliases(registration), encoding="utf-8")
    rules.chmod(0o600)
    aliases.chmod(0o600)
    return rules, aliases
