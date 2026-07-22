# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

from enpire.policy.cap.launcher import build_cap_launch, list_tasks


def test_every_task_points_to_a_migrated_script() -> None:
    launch = build_cap_launch("cube-pick", station="test-station")
    root = launch.cwd
    for task in list_tasks():
        assert (root / task.script).is_file(), task


def test_cap_launch_is_station_external_and_dry_run_friendly() -> None:
    launch = build_cap_launch(
        "gpu-handover",
        station="test-station",
        output=Path("outputs/test"),
        overrides=("robot.dashboard=false",),
    )
    assert launch.env["ENPIRE_YAM_STATION"] == "test-station"
    assert "robot=real_yam" in launch.command
    assert "runtime.exit_on_error=true" in launch.command
    assert "robot.dashboard=false" in launch.command
    assert "test-station" not in " ".join(launch.command)


def test_cube_script_uses_existing_pick_skill(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []
    package = ModuleType("skill_library")
    package.__path__ = []  # type: ignore[attr-defined]
    pick = ModuleType("skill_library.pick")

    def pick_object(name: str, *, camera: str, grasp_mode: str) -> str:
        calls.append((name, camera, grasp_mode))
        return "left"

    pick.pick_object = pick_object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "skill_library", package)
    monkeypatch.setitem(sys.modules, "skill_library.pick", pick)
    script = build_cap_launch("cube-pick", station="test-station").cwd / next(
        task.script for task in list_tasks() if task.name == "cube-pick"
    )
    namespace = runpy.run_path(str(script))
    assert calls == [("cube", "top", "anygrasp")]
    assert namespace["get_task_info"]()["success"] is True

