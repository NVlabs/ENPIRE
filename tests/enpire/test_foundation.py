# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from enpire.env.forge.artifacts import ArtifactStore
from enpire.env.forge.interface import StepResult, VerificationResult
from enpire.env.forge.loop import TrialRunner
from enpire.env.forge.registry import (
    Registry,
    ToolDefinition,
    default_skill_registry,
    default_tool_registry,
)
from enpire.env.forge.security import REDACTED, redact
from enpire.policy.interface import FunctionPolicy


class CountingEnvironment:
    def __init__(self, target: int = 3):
        self.target = target
        self.value = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> dict[str, int | None]:
        self.value = 0
        return {"value": self.value, "seed": seed}

    def observe(self) -> dict[str, int]:
        return {"value": self.value}

    def step(self, action: int) -> StepResult:
        self.value += action
        return StepResult(
            observation={"value": self.value},
            reward=float(action),
            terminated=self.value >= self.target,
        )

    def verify(self) -> VerificationResult:
        return VerificationResult(
            success=self.value == self.target,
            score=float(self.value),
            metrics={"value": self.value},
        )

    def close(self) -> None:
        self.closed = True


def test_trial_runner_reset_execute_verify_and_artifacts(tmp_path):
    environment = CountingEnvironment()
    policy = FunctionPolicy(lambda observation: 1)
    artifacts = ArtifactStore(tmp_path / "run")
    runner = TrialRunner(environment, policy, max_steps=5, artifacts=artifacts)

    result = runner.run(seed=7)
    runner.close()

    assert result.success is True
    assert result.steps == 3
    assert result.total_reward == 3.0
    assert result.termination == "terminated"
    assert environment.closed is True

    written = json.loads((tmp_path / "run" / "result.json").read_text())
    assert written["verification"]["metrics"] == {"value": 3}
    events = (tmp_path / "run" / "events.jsonl").read_text().splitlines()
    assert [json.loads(line)["event"] for line in events] == [
        "reset",
        "step",
        "step",
        "step",
        "verify",
    ]


def test_trial_runner_requires_positive_step_budget():
    with pytest.raises(ValueError, match="max_steps"):
        TrialRunner(CountingEnvironment(), FunctionPolicy(lambda observation: 1), max_steps=0)


def test_artifact_store_redacts_credentials_and_blocks_escape(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    target = store.write_json(
        "config.json",
        {
            "api_key": "never-write-me",
            "nested": {"access_token": "never-write-me-either", "mode": "offline"},
        },
    )
    payload = json.loads(target.read_text())

    assert payload == {
        "api_key": REDACTED,
        "nested": {"access_token": REDACTED, "mode": "offline"},
    }
    assert "never-write" not in target.read_text()
    with pytest.raises(ValueError, match="escapes"):
        store.path("../outside.json")


def test_redact_does_not_hide_non_secret_configuration():
    assert redact({"keyboard_key": "enter", "monkey": "value"}) == {
        "keyboard_key": "enter",
        "monkey": "value",
    }


def test_registry_rejects_duplicates_and_is_sorted():
    registry = Registry()
    later = ToolDefinition("z.tool", "vision", "z", "builtins:dict")
    earlier = ToolDefinition("a.tool", "control", "a", "builtins:list")
    registry.register(later)
    registry.register(earlier)

    assert [entry.name for entry in registry.list()] == ["a.tool", "z.tool"]
    assert [entry.name for entry in registry.list(category="vision")] == ["z.tool"]
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(later)


def test_registry_loads_lazily_and_reports_missing_extra():
    available = ToolDefinition("core.dict", "core", "dict", "builtins:dict")
    assert available.load() is dict

    missing = ToolDefinition(
        "vision.fake",
        "vision",
        "fake",
        "missing_enpire_module:Tool",
        extra="vision",
        required_modules=("missing_enpire_module",),
    )
    with pytest.raises(RuntimeError, match="uv sync --extra vision"):
        missing.load()


def test_default_registry_has_practitioner_categories():
    definitions = default_tool_registry().list()
    assert {item.category for item in definitions} == {"vision", "planning", "control", "vlm"}
    assert all(item.extra for item in definitions)


def test_default_skill_registry_points_to_original_cap_skills():
    definitions = default_skill_registry().list()
    assert {item.name for item in definitions} == {
        "manipulation.pick",
        "manipulation.pick_and_place",
        "manipulation.vertical_grasp",
    }
    assert all(item.target.startswith("enpire.env.forge.cap.saved_scripts.skill_library.") for item in definitions)
