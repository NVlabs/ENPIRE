from __future__ import annotations

import numpy as np
import pytest
import requests

from cap.agent.tools.grasp_2d import Grasp2DPlanResult, GraspCandidate
from cap.agent.tools.grasp_anygrasp import SampleGraspPoseAnyGraspTool


def _make_tool(monkeypatch: pytest.MonkeyPatch) -> SampleGraspPoseAnyGraspTool:
    tool = SampleGraspPoseAnyGraspTool()
    rgb = np.zeros((6, 6, 3), dtype=np.uint8)
    depth = np.ones((6, 6), dtype=np.float32)
    cam_K = np.eye(3, dtype=np.float64)
    T_cam_world = np.eye(4, dtype=np.float64)
    mask = np.ones((6, 6), dtype=np.uint8)

    monkeypatch.setattr(tool, "_get_rgb_depth_intrinsics", lambda _camera: (rgb, depth, cam_K))
    monkeypatch.setattr(tool, "_get_extrinsics", lambda _camera: T_cam_world)
    monkeypatch.setattr(tool, "_segment_object", lambda _rgb, _object_name: mask)
    return tool


def _fallback_plan() -> Grasp2DPlanResult:
    candidate = GraspCandidate(
        position=[0.12, -0.04, 0.76],
        rpy=[0.0, 180.0, -90.0],
        score=1.0,
        width=0.08,
    )
    grasp_rows = [
        {
            "rank": 1,
            "score": 1.0,
            "width": 0.08,
            "raw_xyz": candidate.position,
            "raw_rpy": candidate.rpy,
            "planner_xyz": candidate.position,
            "planner_rpy": candidate.rpy,
            "status": "returned",
            "status_reason": "2D fallback candidate",
            "thumbnail_b64": None,
        }
    ]
    return Grasp2DPlanResult(
        candidates=[candidate],
        grasp_rows=grasp_rows,
        overlay_jpeg=b"overlay",
        debug={
            "camera": "top",
            "object_name": "pliers",
            "n_grasps": 1,
            "status": "ok",
            "backend": "2dgrasp",
            "best_score": 1.0,
            "grasps": grasp_rows,
            "major_axis_ratio": 1.0,
            "axis_guided": False,
        },
    )


def test_anygrasp_success_does_not_fallback_even_for_short_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _make_tool(monkeypatch)
    fallback_calls = {"count": 0}

    monkeypatch.setattr(tool, "_compute_segmented_cloud_stats", lambda **_kwargs: (0, 0.0))
    monkeypatch.setattr(
        tool,
        "_plan_2d_fallback",
        lambda **_kwargs: fallback_calls.__setitem__("count", fallback_calls["count"] + 1),
    )

    raw_grasp = np.eye(4, dtype=np.float64)[None, ...]
    raw_grasp[0, :3, 3] = np.array([0.10, 0.20, 0.50], dtype=np.float64)
    monkeypatch.setattr(
        tool,
        "_call_anygrasp_viz",
        lambda *_args, **_kwargs: (
            raw_grasp,
            np.array([0.9], dtype=np.float64),
            np.array([0.04], dtype=np.float64),
            b"overlay",
            [],
        ),
    )

    result = tool.execute(object_name="pliers", camera="top", max_grasps=4)

    assert result.success is True
    assert len(result.data) == 1
    assert fallback_calls["count"] == 0
    assert tool.last_graspnet_debug is not None
    assert tool.last_graspnet_debug["backend"] == "anygrasp"
    assert tool.last_graspnet_debug["segmented_cloud_point_count"] == 0


def test_anygrasp_no_grasps_and_empty_cloud_falls_back_to_2d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _make_tool(monkeypatch)
    fallback_calls = {"count": 0}

    monkeypatch.setattr(tool, "_compute_segmented_cloud_stats", lambda **_kwargs: (0, 0.0))
    monkeypatch.setattr(
        tool,
        "_call_anygrasp_viz",
        lambda *_args, **_kwargs: (
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            b"",
            [],
        ),
    )

    def _fake_fallback(**_kwargs):
        fallback_calls["count"] += 1
        return _fallback_plan()

    monkeypatch.setattr(tool, "_plan_2d_fallback", _fake_fallback)

    result = tool.execute(object_name="pliers", camera="top", max_grasps=4)

    assert result.success is True
    assert len(result.data) == 1
    assert fallback_calls["count"] == 1
    assert tool.last_graspnet_debug is not None
    assert tool.last_graspnet_debug["backend"] == "2dgrasp"
    assert tool.last_graspnet_debug["fallback_from"] == "anygrasp"
    assert tool.last_graspnet_debug["fallback_reason"] == "short_or_empty_segmented_cloud"


def test_anygrasp_no_grasps_and_short_cloud_falls_back_to_2d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _make_tool(monkeypatch)
    fallback_calls = {"count": 0}

    monkeypatch.setattr(tool, "_compute_segmented_cloud_stats", lambda **_kwargs: (8, 0.01))
    monkeypatch.setattr(
        tool,
        "_call_anygrasp_viz",
        lambda *_args, **_kwargs: (
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            b"",
            [],
        ),
    )

    def _fake_fallback(**_kwargs):
        fallback_calls["count"] += 1
        return _fallback_plan()

    monkeypatch.setattr(tool, "_plan_2d_fallback", _fake_fallback)

    result = tool.execute(object_name="pliers", camera="top", max_grasps=4)

    assert result.success is True
    assert len(result.data) == 1
    assert fallback_calls["count"] == 1
    assert tool.last_graspnet_debug is not None
    assert tool.last_graspnet_debug["segmented_cloud_height_m"] == pytest.approx(0.01, abs=1e-9)


def test_anygrasp_no_grasps_and_tall_enough_cloud_preserves_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _make_tool(monkeypatch)
    fallback_calls = {"count": 0}

    monkeypatch.setattr(tool, "_compute_segmented_cloud_stats", lambda **_kwargs: (8, 0.02))
    monkeypatch.setattr(
        tool,
        "_call_anygrasp_viz",
        lambda *_args, **_kwargs: (
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            b"",
            [],
        ),
    )
    monkeypatch.setattr(
        tool,
        "_plan_2d_fallback",
        lambda **_kwargs: fallback_calls.__setitem__("count", fallback_calls["count"] + 1),
    )

    result = tool.execute(object_name="pliers", camera="top", max_grasps=4)

    assert result.success is False
    assert result.error == "AnyGrasp found no grasps for 'pliers'"
    assert fallback_calls["count"] == 0
    assert tool.last_graspnet_debug is not None
    assert tool.last_graspnet_debug["backend"] == "anygrasp"
    assert tool.last_graspnet_debug["status"] == "no_grasps"


def test_anygrasp_connection_error_does_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _make_tool(monkeypatch)
    fallback_calls = {"count": 0}

    monkeypatch.setattr(tool, "_compute_segmented_cloud_stats", lambda **_kwargs: (0, 0.0))
    monkeypatch.setattr(
        tool,
        "_call_anygrasp_viz",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.ConnectionError()),
    )
    monkeypatch.setattr(
        tool,
        "_plan_2d_fallback",
        lambda **_kwargs: fallback_calls.__setitem__("count", fallback_calls["count"] + 1),
    )

    result = tool.execute(object_name="pliers", camera="top", max_grasps=4)

    assert result.success is False
    assert "Cannot reach AnyGrasp server" in (result.error or "")
    assert fallback_calls["count"] == 0
