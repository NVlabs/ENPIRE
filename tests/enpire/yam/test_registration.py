# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import stat

import pytest

from enpire.env.forge.yam.registration import (
    RegistrationResult,
    render_camera_aliases,
    render_udev_rules,
)
from enpire.env.forge.yam.registration import _legacy_identify as legacy
from enpire.env.forge.yam.registration.rules import write_registration_files


def test_rule_renderer_preserves_original_forge_rule_syntax():
    registration = RegistrationResult(
        station_id="test-station",
        roles={
            "can_follow_l": "CAN-TEST-LEFT",
            "serial-right-buttons": "BUTTON-TEST-RIGHT",
            "video_top": "CAMERA-TEST-TOP",
        },
    )

    rendered = render_udev_rules(registration)

    assert legacy.emit_can_rule("can_follow_l", "CAN-TEST-LEFT") in rendered
    assert legacy.emit_button_rule("serial-right-buttons", "BUTTON-TEST-RIGHT") in rendered
    assert legacy.emit_top_rule("video_top", "CAMERA-TEST-TOP") in rendered
    assert "forge_rl_realsense_alias %k" in rendered


def test_camera_aliases_use_original_gpu_branch_roles():
    registration = RegistrationResult(
        station_id="test-station",
        roles={
            "video_left_third": "FIXED-CAMERA",
            "video_left": "LEFT-WRIST-CAMERA",
            "video_right": "RIGHT-WRIST-CAMERA",
        },
    )

    aliases = json.loads(render_camera_aliases(registration))

    assert aliases == {
        "FIXED-CAMERA": "video_left_third",
        "LEFT-WRIST-CAMERA": "video_left",
        "RIGHT-WRIST-CAMERA": "video_right",
    }


def test_registration_rejects_unknown_and_duplicate_roles():
    with pytest.raises(ValueError, match="Unknown"):
        RegistrationResult("test", {"private_camera_role": "serial"})
    with pytest.raises(ValueError, match="multiple roles"):
        RegistrationResult("test", {"video_left": "serial", "video_right": "serial"})


def test_registration_files_are_written_owner_only(tmp_path):
    registration = RegistrationResult("test", {"video_left": "TEST-CAMERA"})
    rules, aliases = write_registration_files(registration, tmp_path / "registration")

    assert "forge_rl_realsense_alias" in rules.read_text()
    assert json.loads(aliases.read_text()) == {"TEST-CAMERA": "video_left"}
    assert stat.S_IMODE(rules.stat().st_mode) == 0o600
    assert stat.S_IMODE(aliases.stat().st_mode) == 0o600
