# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enpire.env.forge.services.launcher import build_service_suite


def test_cap_real_service_profile_is_local_and_non_motion() -> None:
    suite = build_service_suite(profile="cap-real", session="test")
    assert [service.name for service in suite.services] == [
        "sam3",
        "anygrasp",
        "curobo",
        "nvidia",
    ]
    assert [service.port for service in suite.services] == [6767, 8122, 8611, 8765]
    assert not any(service.moves_hardware for service in suite.services)
    assert all("API_KEY" not in service.display_command for service in suite.services)


def test_robot_profile_is_explicitly_motion_capable() -> None:
    suite = build_service_suite(profile="robot")
    assert len(suite.services) == 1
    assert suite.services[0].moves_hardware is True
