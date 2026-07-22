# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy registry for optional robot tools and reusable skills."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    category: str
    description: str
    target: str
    extra: str | None = None
    required_modules: tuple[str, ...] = ()
    modes: tuple[str, ...] = ("sim", "real")
    safety: str = "read_only"
    metadata: dict[str, Any] = field(default_factory=dict)

    def missing_modules(self) -> tuple[str, ...]:
        return tuple(
            module for module in self.required_modules if importlib.util.find_spec(module) is None
        )

    @property
    def available(self) -> bool:
        return not self.missing_modules()

    def load(self) -> type[Any]:
        missing = self.missing_modules()
        if missing:
            install = f" Install with `uv sync --extra {self.extra}`." if self.extra else ""
            raise RuntimeError(
                f"Tool {self.name!r} is missing optional modules: {', '.join(missing)}.{install}"
            )
        module_name, separator, attribute = self.target.partition(":")
        if not separator:
            raise ValueError(f"Invalid tool target {self.target!r}; expected module:attribute")
        return getattr(importlib.import_module(module_name), attribute)


@dataclass(frozen=True)
class SkillDefinition(ToolDefinition):
    """Lazy reference to an existing code-as-policy skill function."""


class Registry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Duplicate registry entry: {definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self._definitions))
            raise KeyError(f"Unknown tool {name!r}. Available: {choices}") from exc

    def list(self, *, category: str | None = None) -> tuple[ToolDefinition, ...]:
        values = self._definitions.values()
        if category is not None:
            values = (definition for definition in values if definition.category == category)
        return tuple(sorted(values, key=lambda definition: definition.name))


def default_tool_registry() -> Registry:
    registry = Registry()
    for definition in (
        ToolDefinition(
            name="vision.segment",
            category="vision",
            description="Segment a named object through the original SAM service client.",
            target="enpire.env.forge.cap.agent.tools.segmentation:SegmentObjectTool",
            extra="vision",
            required_modules=("requests", "PIL"),
        ),
        ToolDefinition(
            name="vision.detect",
            category="vision",
            description="Run the original one-shot multi-view object detector.",
            target="enpire.env.forge.cap.agent.tools.detection:DetectObjectsOneshotTool",
            extra="vision",
            required_modules=("requests", "PIL"),
        ),
        ToolDefinition(
            name="planning.freespace",
            category="planning",
            description="Plan and execute collision-aware end-effector motion.",
            target="enpire.env.forge.cap.agent.tools.freespace_move:FreespaceMoveTool",
            extra="planning",
            required_modules=("portal", "mujoco", "mink"),
            safety="motion",
        ),
        ToolDefinition(
            name="control.get_state",
            category="control",
            description="Read current arm and gripper state.",
            target="enpire.env.forge.cap.agent.tools.native:GetRobotStateTool",
            extra="control-yam",
            required_modules=("portal",),
        ),
        ToolDefinition(
            name="control.set_gripper",
            category="control",
            description="Command one YAM gripper.",
            target="enpire.env.forge.cap.agent.tools.native:SetGripperTool",
            extra="control-yam",
            required_modules=("portal",),
            safety="motion",
        ),
        ToolDefinition(
            name="vlm.query",
            category="vlm",
            description="Query a configured local or hosted vision-language backend.",
            target="enpire.env.forge.cap.agent.tools.vlm_query:VlmQueryTool",
            extra="vlm",
            required_modules=("requests", "PIL"),
        ),
    ):
        registry.register(definition)
    return registry


def default_skill_registry() -> Registry:
    registry = Registry()
    for definition in (
        SkillDefinition(
            name="manipulation.pick",
            category="manipulation",
            description="Detect, rank, approach, and grasp an object.",
            target="enpire.env.forge.cap.saved_scripts.skill_library.pick:pick_object",
            extra="planning",
            required_modules=("scipy",),
            safety="motion",
        ),
        SkillDefinition(
            name="manipulation.pick_and_place",
            category="manipulation",
            description="Run the original reusable pick-and-place CaP skill.",
            target="enpire.env.forge.cap.saved_scripts.skill_library.pick_place:pick_and_place",
            extra="planning",
            required_modules=("scipy",),
            safety="motion",
        ),
        SkillDefinition(
            name="manipulation.vertical_grasp",
            category="manipulation",
            description="Execute the original straight-down grasp helper.",
            target="enpire.env.forge.cap.saved_scripts.skill_library.vertical_grasp:vertical_grasp_v1",
            extra="control-yam",
            required_modules=("numpy",),
            safety="motion",
        ),
    ):
        registry.register(definition)
    return registry
