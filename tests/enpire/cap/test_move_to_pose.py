"""Hardware-free characterization and tests for generic CaP motion scripts."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from enpire.env.forge.cap.agent.tools import ToolRegistry
from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool


def test_cap_callable_returns_data_and_raises_on_tool_failure():
    registry = ToolRegistry()
    tool = SimpleNamespace(
        name="freespace_move",
        parameters=[],
        description="fake",
        execute=lambda **_: SimpleNamespace(success=True, data="planned"),
    )
    registry.register(tool)
    assert registry.callable_dict()["freespace_move"]() == "planned"
    tool.execute = lambda **_: SimpleNamespace(success=False, error="collision")
    with pytest.raises(RuntimeError, match="collision"):
        registry.callable_dict()["freespace_move"]()


def test_cap_display_rpy_and_quaternion_sign_contract():
    quat = FreespaceMoveTool._display_rpy_to_quat([0.0, 180.0, 0.0])
    np.testing.assert_allclose(np.abs(quat), [2**-0.5, 2**-0.5, 0, 0], atol=1e-12)
    assert FreespaceMoveTool._quat_error_deg(quat, -quat) == pytest.approx(0.0)


@pytest.fixture
def harness(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    library = root / "enpire/env/forge/cap/saved_scripts/skill_library"

    def load(name):
        spec = importlib.util.spec_from_file_location(name, library / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    motion, gripper = load("move_to_pose"), load("set_gripper")
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(motion.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(
        motion.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds)
    )
    arm = SimpleNamespace(ee_pos=[0.4, 0.1, 0.9], ee_quat=[0, 0, 0, 1])
    tools = ModuleType("skill_library.namespace")
    tools.get_robot_state = Mock(return_value=SimpleNamespace(arms={"left": arm, "right": arm}))
    tools.freespace_move = Mock(return_value=SimpleNamespace(status="Success"))
    tools.set_gripper = Mock(return_value={"success": True})
    package = ModuleType("skill_library")
    package.__path__ = []
    package.namespace = tools
    monkeypatch.setitem(sys.modules, "skill_library", package)
    monkeypatch.setitem(sys.modules, "skill_library.namespace", tools)
    return SimpleNamespace(motion=motion, gripper=gripper, tools=tools, arm=arm, clock=clock)


@pytest.mark.parametrize("side", ["left", "right"])
def test_exact_pose_uses_registered_tool_and_preserves_gripper(harness, side):
    report = harness.motion.move_to_pose(side, [0.4, 0.1, 0.9], planning_speed=0.25)
    assert report["pose_verified"] is True
    harness.tools.freespace_move.assert_called_once_with(
        **{
            f"{side}_target_pos": [0.4, 0.1, 0.9],
            f"{side}_target_quat": [0.0, 0.0, 0.0, 1.0],
            "preview_only": False,
            "planning_speed": 0.25,
        }
    )
    harness.tools.set_gripper.assert_not_called()
    assert harness.tools.get_robot_state.call_count == 4


def test_planner_success_does_not_hide_measured_position_error(harness):
    with pytest.raises(harness.motion.PoseNotReachedError) as error:
        harness.motion.move_to_pose("left", [0.4, 0.1, 0.86], settle_timeout_s=0.2)
    assert error.value.report["position_error_m"] == pytest.approx(0.04)
    assert error.value.report["pose_verified"] is False
    harness.tools.freespace_move.assert_called_once()
    harness.tools.set_gripper.assert_not_called()


def test_orientation_miss_fails_even_when_position_matches(harness):
    with pytest.raises(harness.motion.PoseNotReachedError) as error:
        harness.motion.move_to_pose("left", target_quat=[1, 0, 0, 0], settle_timeout_s=0.2)
    assert error.value.report["position_error_m"] == pytest.approx(0)
    assert error.value.report["rotation_error_deg"] == pytest.approx(180)


def test_cap_display_rpy_is_converted_with_existing_tool_helper(harness):
    quat = FreespaceMoveTool._display_rpy_to_quat([0, 125, 20])
    harness.arm.ee_quat = quat
    report = harness.motion.move_to_pose("left", target_rpy=[0, 125, 20])
    assert report["pose_verified"]
    np.testing.assert_allclose(
        harness.tools.freespace_move.call_args.kwargs["left_target_quat"], quat
    )


def test_quaternion_normalization_and_sign_equivalence(harness):
    report = harness.motion.move_to_pose("left", target_quat=[0, 0, 0, -2])
    assert report["pose_verified"]
    assert report["rotation_error_deg"] == pytest.approx(0)


def test_preview_never_claims_measured_completion(harness):
    report = harness.motion.move_to_pose("right", [0.8, 0, 1], preview_only=True)
    assert report["preview_only"] and not report["pose_verified"]
    assert harness.tools.get_robot_state.call_count == 1
    assert harness.tools.freespace_move.call_args.kwargs["preview_only"] is True


def test_tool_exception_propagates_without_retry(harness):
    harness.tools.freespace_move.side_effect = RuntimeError("collision")
    with pytest.raises(RuntimeError, match="collision"):
        harness.motion.move_to_pose("left", [0.4, 0.1, 0.9])
    assert harness.tools.get_robot_state.call_count == 1
    harness.tools.freespace_move.assert_called_once()


def test_failed_status_is_not_treated_as_success(harness):
    harness.tools.freespace_move.return_value = {"status": "IK_Failed"}
    with pytest.raises(RuntimeError, match="IK_Failed"):
        harness.motion.move_to_pose("left", [0.4, 0.1, 0.9])


def test_settling_requires_consecutive_in_tolerance_readings(harness):
    def state(z):
        return {"arms": {"left": {"ee_pos": [0.4, 0.1, z], "ee_quat": [0, 0, 0, 1]}}}

    harness.tools.get_robot_state.side_effect = [state(z) for z in (0.91, 0.9, 0.91, 0.9, 0.9, 0.9)]
    assert harness.motion.move_to_pose("left", [0.4, 0.1, 0.9])["pose_verified"]
    assert harness.tools.get_robot_state.call_count == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_pos": [float("nan"), 0, 0]},
        {"target_pos": [0, 0]},
        {"target_quat": [0, 0, 0, 0]},
        {"target_rpy": [0, 0, 0], "target_quat": [0, 0, 0, 1]},
        {"planning_speed": -1},
        {"settle_timeout_s": float("inf")},
        {"stable_samples": True},
        {"settle_timeout_s": 0.01},
        {"position_tolerance_m": -1},
        {"rotation_tolerance_deg": 0},
        {"preview_only": "false"},
    ],
)
def test_invalid_inputs_do_not_call_hardware(harness, kwargs):
    inputs = {"target_pos": [0.4, 0.1, 0.9], **kwargs}
    with pytest.raises(ValueError):
        harness.motion.move_to_pose("left", **inputs)
    harness.tools.get_robot_state.assert_not_called()
    harness.tools.freespace_move.assert_not_called()


def test_gripper_is_only_a_thin_tool_adapter(harness):
    assert harness.gripper.set_gripper("right", 0.25) == {"success": True}
    harness.tools.set_gripper.assert_called_once_with(side="right", pos=0.25)
    harness.tools.freespace_move.assert_not_called()


@pytest.mark.parametrize("pos", [-1, 2, float("nan")])
def test_invalid_gripper_opening_never_reaches_tool(harness, pos):
    with pytest.raises(ValueError):
        harness.gripper.set_gripper("left", pos)
    harness.tools.set_gripper.assert_not_called()
