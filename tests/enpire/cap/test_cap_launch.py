# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

from enpire.policy.cap.launcher import build_cap_launch, list_tasks


def test_release_registers_the_moved_pickup_script() -> None:
    launch = build_cap_launch("pickup", station="test-station", prompt="blue cube")
    assert launch.task.script == "cap/saved_scripts/skill_library/pick_object.py"
    assert [task.name for task in list_tasks()] == ["pickup"]


def test_every_task_points_to_a_migrated_script() -> None:
    launch = build_cap_launch("pickup", station="test-station", prompt="blue cube")
    root = launch.cwd
    for task in list_tasks():
        assert (root / task.script).is_file(), task


def test_cap_launch_is_station_external_and_dry_run_friendly() -> None:
    launch = build_cap_launch(
        "pickup",
        station="test-station",
        prompt="blue cube",
        output=Path("outputs/test"),
        overrides=("robot.dashboard=false",),
    )
    assert launch.env["ENPIRE_YAM_STATION"] == "test-station"
    assert launch.env["ENPIRE_PICK_PROMPT"] == "blue cube"
    assert "robot=real_yam" in launch.command
    assert "runtime.exit_on_error=true" in launch.command
    assert "robot.dashboard=false" in launch.command
    assert "test-station" not in " ".join(launch.command)


def test_pickup_script_passes_prompt_to_existing_pick_skill(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []
    package = ModuleType("skill_library")
    package.__path__ = []  # type: ignore[attr-defined]
    pick = ModuleType("skill_library.pick")

    def pick_object(name: str, *, camera: str, grasp_mode: str) -> str:
        calls.append((name, camera, grasp_mode))
        return "left"

    def lift_grasped_object(side: str) -> bool:
        calls.append(("lift", side, "retained"))
        return True

    pick.pick_object = pick_object  # type: ignore[attr-defined]
    pick.lift_grasped_object = lift_grasped_object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "skill_library", package)
    monkeypatch.setitem(sys.modules, "skill_library.pick", pick)
    launch = build_cap_launch("pickup", station="test-station", prompt="blue cube")
    monkeypatch.setenv("ENPIRE_PICK_PROMPT", launch.env["ENPIRE_PICK_PROMPT"])
    # The script reads these from the ambient environment, so an operator shell
    # that exports them (station.env does) must not change what this asserts.
    monkeypatch.delenv("ENPIRE_PICK_GRASP_MODE", raising=False)
    monkeypatch.delenv("ENPIRE_PICK_CAMERA", raising=False)
    script = launch.cwd / next(
        task.script for task in list_tasks() if task.name == "pickup"
    )
    namespace = runpy.run_path(str(script))
    assert calls == [
        ("blue cube", "top", "anygrasp"),
        ("lift", "left", "retained"),
    ]
    assert namespace["get_task_info"]()["success"] is True
    assert namespace["get_task_info"]()["object_prompt"] == "blue cube"
    assert namespace["get_task_info"]()["lifted_and_retained"] is True


def test_pickup_prompt_rejects_empty_or_multiline_input() -> None:
    for prompt in ("", "   ", "blue\ncube"):
        try:
            build_cap_launch("pickup", station="test-station", prompt=prompt)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected prompt {prompt!r} to be rejected")
