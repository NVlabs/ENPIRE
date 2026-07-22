# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import enpire.env.forge.tools.vision.serve_anygrasp_debug as anygrasp_debug
from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool
from enpire.env.forge.cap.agent.tools.grasp_2d import Grasp2DPlanResult, GraspCandidate


class _FakeCloudViser:
    def __init__(self, url: str = "http://cloud-preview") -> None:
        self.url = url
        self.calls: list[dict] = []

    def update(self, **kwargs):
        self.calls.append(kwargs)
        return self.url


def _setup_common_pipeline(monkeypatch: pytest.MonkeyPatch) -> _FakeCloudViser:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    cam_K = np.eye(3, dtype=np.float64)
    T_cam_world = np.eye(4, dtype=np.float64)
    mask = np.ones((2, 2), dtype=np.int32)
    scene_mask = np.ones((2, 2), dtype=bool)
    scene_points = np.array(
        [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.0, 0.1, 0.5], [0.1, 0.1, 0.5]],
        dtype=np.float32,
    )
    scene_colors = np.full((4, 3), 0.5, dtype=np.float32)
    cloud = _FakeCloudViser()

    monkeypatch.setattr(anygrasp_debug, "_get_camera_data", lambda _camera: (rgb, depth, cam_K, T_cam_world))
    monkeypatch.setattr(anygrasp_debug, "_segment_object", lambda _rgb, _prompt, **_kw: mask)
    monkeypatch.setattr(
        anygrasp_debug,
        "_frame_to_scene",
        lambda *_args, **_kwargs: (np.zeros((2, 2, 3), dtype=np.float32), scene_mask, scene_points, scene_colors),
    )
    monkeypatch.setattr(anygrasp_debug, "_get_cloud_viser", lambda: cloud)
    return cloud


def _sample_2d_plan() -> Grasp2DPlanResult:
    candidate = GraspCandidate(
        position=[0.12, -0.03, 0.76],
        rpy=[0.0, 180.0, -90.0],
        score=1.0,
        width=0.08,
    )
    return Grasp2DPlanResult(
        candidates=[candidate],
        grasp_rows=[],
        overlay_jpeg=b"2d-overlay",
        debug={
            "status": "ok",
            "best_score": 1.0,
            "major_axis_ratio": 1.4,
            "axis_guided": True,
        },
    )


def test_run_pipeline_supports_pure_2dgrasp_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    cloud = _setup_common_pipeline(monkeypatch)
    monkeypatch.setattr(anygrasp_debug, "_compute_segmented_cloud_stats", lambda *args, **kwargs: (0, 0.0))
    monkeypatch.setattr(anygrasp_debug, "plan_top_down_grasps_from_mask", lambda **_kwargs: _sample_2d_plan())

    result = anygrasp_debug._run_pipeline(
        anygrasp_debug.RunRequest(prompt="pliers", camera="top", grasp_backend="2dgrasp")
    )

    assert result.status == "ok"
    assert result.backend == "2dgrasp"
    assert result.n_grasps == 1
    assert result.object_input_mode == "mask_only"
    assert result.fallback_from is None
    assert result.cloud_preview_url == cloud.url
    assert result.major_axis_ratio == pytest.approx(1.4)
    assert result.axis_guided is True


def test_run_pipeline_falls_back_to_2dgrasp_when_anygrasp_returns_no_grasps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_common_pipeline(monkeypatch)
    monkeypatch.setattr(anygrasp_debug, "_compute_segmented_cloud_stats", lambda *args, **kwargs: (0, 0.0))
    monkeypatch.setattr(
        anygrasp_debug,
        "_call_anygrasp_viz",
        lambda *_args, **_kwargs: (
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            b"",
            None,
        ),
    )
    monkeypatch.setattr(anygrasp_debug, "plan_top_down_grasps_from_mask", lambda **_kwargs: _sample_2d_plan())

    result = anygrasp_debug._run_pipeline(
        anygrasp_debug.RunRequest(prompt="pliers", camera="top", grasp_backend="anygrasp+2d-fallback")
    )

    assert result.status == "ok"
    assert result.backend == "2dgrasp"
    assert result.fallback_from == "anygrasp"
    assert result.fallback_reason == "short_or_empty_segmented_cloud"


def test_run_pipeline_preserves_anygrasp_mode_when_grasps_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_common_pipeline(monkeypatch)
    monkeypatch.setattr(anygrasp_debug, "_compute_segmented_cloud_stats", lambda *args, **kwargs: (5, 0.02))
    raw_grasp = np.eye(4, dtype=np.float64)[None, ...]
    raw_grasp[0, :3, 3] = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    monkeypatch.setattr(
        anygrasp_debug,
        "_call_anygrasp_viz",
        lambda *_args, **_kwargs: (
            raw_grasp,
            np.array([0.9], dtype=np.float64),
            np.array([0.04], dtype=np.float64),
            b"anygrasp-overlay",
            0.9,
        ),
    )

    result = anygrasp_debug._run_pipeline(
        anygrasp_debug.RunRequest(prompt="mug", camera="top", grasp_backend="anygrasp")
    )

    assert result.status == "ok"
    assert result.backend == "anygrasp"
    assert result.n_grasps == 1
    assert result.fallback_from is None
    assert len(result.grasps) == 1


def test_run_pipeline_supports_side_camera_for_2dgrasp(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_common_pipeline(monkeypatch)
    monkeypatch.setattr(anygrasp_debug, "_compute_segmented_cloud_stats", lambda *args, **kwargs: (0, 0.0))
    monkeypatch.setattr(anygrasp_debug, "plan_top_down_grasps_from_mask", lambda **_kwargs: _sample_2d_plan())

    result = anygrasp_debug._run_pipeline(
        anygrasp_debug.RunRequest(prompt="pliers", camera="left", grasp_backend="2dgrasp")
    )

    assert result.status == "ok"
    assert result.backend == "2dgrasp"
    assert result.n_grasps == 1


def test_run_pipeline_passes_2d_grasp_z_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_common_pipeline(monkeypatch)
    monkeypatch.setattr(anygrasp_debug, "_compute_segmented_cloud_stats", lambda *args, **kwargs: (0, 0.0))
    monkeypatch.setattr(anygrasp_debug, "plan_top_down_grasps_from_mask", lambda **_kwargs: _sample_2d_plan())

    result = anygrasp_debug._run_pipeline(
        anygrasp_debug.RunRequest(
            prompt="pliers",
            camera="top",
            grasp_backend="2dgrasp",
            grasp_z_m=0.91,
        )
    )

    assert result.backend == "2dgrasp"
    assert result.grasp_z_m == pytest.approx(0.91)


def test_preview_ik_failure_uses_curobo_reported_residuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = FreespaceMoveTool()

    class _FakeRpc:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeClient:
        def get_state(self):
            return _FakeRpc(
                {
                    "left_joint_pos": np.zeros(6),
                    "right_joint_pos": np.zeros(6),
                    "left_gripper_pos": np.array([0.6]),
                    "right_gripper_pos": np.array([0.4]),
                }
            )

    class _Kin:
        def forward_kinematics(self, left_jp, right_jp):
            quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            return (
                np.asarray([0.4, 0.2, 0.8], dtype=np.float64),
                quat.copy(),
                np.asarray([0.4, -0.2, 0.8], dtype=np.float64),
                quat.copy(),
            )

    class _Planner:
        def __init__(self) -> None:
            self._kin = _Kin()

        def plan_to_pose(self, **kwargs):
            return {
                "status": "IK_Failed",
                "position_error_m": 0.0033,
                "rotation_error_deg": 40.91,
            }

    class _DiagKin:
        def forward_kinematics(self, left_jp, right_jp):
            quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            return (
                np.asarray([0.4, 0.2, 0.8], dtype=np.float64),
                quat.copy(),
                np.asarray([0.4, -0.2, 0.8], dtype=np.float64),
                quat.copy(),
            )

        def inverse_kinematics(self, *args, **kwargs):
            raise AssertionError("preview IK failure should not recompute fallback IK")

    planner = _Planner()
    monkeypatch.setattr(tool, "_get_client", lambda: _FakeClient())
    monkeypatch.setattr(tool, "_get_diagnostic_kinematics", lambda **kwargs: _DiagKin())
    monkeypatch.setattr(
        tool,
        "_compute_plan_diagnostics",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("preview IK failure should not recompute diagnostics")
        ),
    )
    monkeypatch.setattr(anygrasp_debug, "_get_freespace_tool", lambda: tool)
    monkeypatch.setattr(
        anygrasp_debug,
        "_tool_get_planner_with_solver_speed",
        lambda *args, **kwargs: planner,
    )

    result = anygrasp_debug._preview_pose_with_path(
        anygrasp_debug.PlannerRequest(
            side="left",
            xyz=[0.5, 0.1, 0.8],
            rpy=[0.0, 90.0, 0.0],
            pose_mode="planner",
            ik_error_threshold=0.005,
        )
    )

    assert result.ok is False
    assert result.status == "IK_Failed"
    assert result.final_pos_error_m == pytest.approx(0.0033)
    assert result.final_rot_error_deg == pytest.approx(40.91)
    assert "reported by cuRobo" in (result.reason or "")


def test_preview_request_cache_key_includes_rotation_threshold() -> None:
    req_a = anygrasp_debug.PlannerRequest(
        side="left",
        xyz=[0.5, 0.1, 0.8],
        rpy=[0.0, 90.0, 0.0],
        ik_error_threshold=0.005,
        ik_rot_threshold_deg=2.0,
    )
    req_b = anygrasp_debug.PlannerRequest(
        side="left",
        xyz=[0.5, 0.1, 0.8],
        rpy=[0.0, 90.0, 0.0],
        ik_error_threshold=0.005,
        ik_rot_threshold_deg=5.0,
    )

    assert anygrasp_debug._planner_request_cache_key(req_a) != anygrasp_debug._planner_request_cache_key(req_b)


def test_sort_request_propagates_rotation_threshold_to_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    class _FakeTool:
        def execute(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                success=True,
                data=SimpleNamespace(
                    batch_candidates=[],
                    evaluated_candidate_count=0,
                    input_candidate_count=1,
                    truncated_input_count=0,
                    batch_attempted=True,
                    batch_error=None,
                    curobo_solve_time_ms=0.0,
                    curobo_total_time_ms=0.0,
                    curobo_graph_time_ms=0.0,
                    curobo_ik_time_ms=0.0,
                ),
            )

    monkeypatch.setattr(anygrasp_debug, "_get_freespace_tool", lambda: _FakeTool())

    result = anygrasp_debug._run_sort_grasps_by_ik_error(
        anygrasp_debug.SortIkRequest(
            side="left",
            pose_mode="planner",
            ik_error_threshold=0.005,
            ik_rot_threshold_deg=7.5,
            grasps=[
                anygrasp_debug.GraspPoseRow(
                    rank=1,
                    score=0.8,
                    width=0.04,
                    raw_xyz=[0.5, 0.1, 0.8],
                    raw_rpy=[0.0, 90.0, 0.0],
                    planner_xyz=[0.5, 0.1, 0.8],
                    planner_rpy=[0.0, 90.0, 0.0],
                )
            ],
        )
    )

    assert result.ok is True
    assert result.ik_rot_threshold_deg == pytest.approx(7.5)
    assert captured["ik_error_threshold"] == pytest.approx(0.005)
    assert captured["ik_rot_threshold_deg"] == pytest.approx(7.5)
