from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import os

import numpy as np
import pytest

os.environ.setdefault("CAP_AGENT_NAME", "Mochi")

from enpire.env.forge.cap.agent.tools.freespace_move import FreespaceMoveTool


def test_infer_effective_side_freezes_other_arm_when_target_matches_current_pose() -> (
    None
):
    cur_l_pos = np.array([0.4, 0.2, 0.8], dtype=np.float64)
    cur_r_pos = np.array([0.4, -0.2, 0.8], dtype=np.float64)
    cur_l_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    cur_r_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    side = FreespaceMoveTool._infer_effective_side(
        has_left=True,
        has_right=True,
        tgt_l_pos=cur_l_pos + np.array([0.0, 0.0, 0.1], dtype=np.float64),
        tgt_l_q=cur_l_q,
        tgt_r_pos=cur_r_pos.copy(),
        tgt_r_q=cur_r_q.copy(),
        cur_l_pos=cur_l_pos,
        cur_l_q=cur_l_q,
        cur_r_pos=cur_r_pos,
        cur_r_q=cur_r_q,
        left_gripper_target_width=None,
        right_gripper_target_width=None,
    )

    assert side == "left"


def test_infer_effective_side_keeps_bimanual_when_other_gripper_target_is_requested() -> (
    None
):
    cur_l_pos = np.array([0.4, 0.2, 0.8], dtype=np.float64)
    cur_r_pos = np.array([0.4, -0.2, 0.8], dtype=np.float64)
    cur_l_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    cur_r_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    side = FreespaceMoveTool._infer_effective_side(
        has_left=True,
        has_right=True,
        tgt_l_pos=cur_l_pos + np.array([0.0, 0.0, 0.1], dtype=np.float64),
        tgt_l_q=cur_l_q,
        tgt_r_pos=cur_r_pos.copy(),
        tgt_r_q=cur_r_q.copy(),
        cur_l_pos=cur_l_pos,
        cur_l_q=cur_l_q,
        cur_r_pos=cur_r_pos,
        cur_r_q=cur_r_q,
        left_gripper_target_width=None,
        right_gripper_target_width=0.5,
    )

    assert side == "both"


def test_infer_effective_side_returns_none_for_true_noop() -> None:
    cur_l_pos = np.array([0.4, 0.2, 0.8], dtype=np.float64)
    cur_r_pos = np.array([0.4, -0.2, 0.8], dtype=np.float64)
    cur_l_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    cur_r_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    side = FreespaceMoveTool._infer_effective_side(
        has_left=True,
        has_right=True,
        tgt_l_pos=cur_l_pos.copy(),
        tgt_l_q=cur_l_q.copy(),
        tgt_r_pos=cur_r_pos.copy(),
        tgt_r_q=cur_r_q.copy(),
        cur_l_pos=cur_l_pos,
        cur_l_q=cur_l_q,
        cur_r_pos=cur_r_pos,
        cur_r_q=cur_r_q,
        left_gripper_target_width=None,
        right_gripper_target_width=None,
    )

    assert side is None


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


@dataclass
class _Candidate:
    position: list[float]
    rpy: list[float]
    score: float
    width: float


class _TrajectoryRpc:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _TrajectoryExecClient:
    def __init__(self):
        self.moves: list[tuple[str, tuple[float, ...], tuple[float, ...]]] = []

    def get_state(self):
        return _TrajectoryRpc(
            {
                "left_joint_pos": np.zeros(6),
                "right_joint_pos": np.zeros(6),
                "left_gripper_pos": np.array([0.6]),
                "right_gripper_pos": np.array([0.4]),
            }
        )

    def move_joint_keypoints(self, arm_side, timestamps, positions, gripper_positions):
        self.moves.append(
            (
                arm_side,
                tuple(np.asarray(timestamps, dtype=np.float64).tolist()),
                tuple(np.asarray(positions, dtype=np.float64).reshape(-1).tolist()),
            )
        )
        return _TrajectoryRpc({"success": True})

    def move_bimanual_joint_keypoints(
        self,
        timestamps,
        left_positions,
        right_positions,
        left_gripper_positions,
        right_gripper_positions,
    ):
        self.moves.append(
            (
                "both",
                tuple(np.asarray(timestamps, dtype=np.float64).tolist()),
                tuple(np.asarray(left_positions, dtype=np.float64).reshape(-1).tolist())
                + tuple(
                    np.asarray(right_positions, dtype=np.float64).reshape(-1).tolist()
                ),
            )
        )
        return _TrajectoryRpc({"success": True})


def test_batch_mode_exposes_cache_metadata_for_feasible_candidates(monkeypatch) -> None:
    tool = FreespaceMoveTool()
    monkeypatch.setattr(tool, "_get_client", lambda: _FakeClient())

    captured: dict[str, object] = {}

    class _Planner:
        def plan_batch_to_pose(self, **kwargs):
            captured["plan_kwargs"] = kwargs
            n = len(kwargs["target_left_pos"])
            left_positions_by_index = []
            right_positions_by_index = []
            for i in range(n):
                left_positions_by_index.append(
                    np.asarray(
                        [
                            np.zeros(6, dtype=np.float64) + i * 0.00,
                            np.zeros(6, dtype=np.float64) + i * 0.01,
                            np.zeros(6, dtype=np.float64) + i * 0.02,
                        ],
                        dtype=np.float64,
                    )
                )
                right_positions_by_index.append(
                    np.asarray(
                        [
                            np.zeros(6, dtype=np.float64) + i * 0.00,
                            np.zeros(6, dtype=np.float64) + i * 0.01,
                            np.zeros(6, dtype=np.float64) + i * 0.02,
                        ],
                        dtype=np.float64,
                    )
                )
            return {
                "success_mask": np.ones((n,), dtype=bool),
                "status_detail_by_index": [None] * n,
                "position_error_m": np.linspace(0.001, 0.016, n),
                "rotation_error_deg": np.linspace(0.1, 1.6, n),
                "left_positions_by_index": left_positions_by_index,
                "right_positions_by_index": right_positions_by_index,
                "curobo_solve_time_ms": 12.5,
                "curobo_total_time_ms": 18.0,
                "curobo_graph_time_ms": 4.0,
                "curobo_ik_time_ms": 3.0,
            }

    monkeypatch.setattr(
        tool,
        "_get_planner",
        lambda **kwargs: _Planner(),
    )

    candidates = [
        _Candidate(
            position=[0.5 + i * 0.001, 0.1, 0.8],
            rpy=[0.0, 90.0, 0.0],
            score=float(i),
            width=0.04,
        )
        for i in range(4)
    ]

    result = tool.execute(grasp_candidates=candidates, batch_side="left")

    assert result.success is True
    assert result.data.best_candidate is not None
    assert result.data.best_candidate.trajectory_cache_key is not None
    assert result.data.best_candidate.trajectory_steps > 0
    assert all(
        candidate.trajectory_cache_key is not None and candidate.trajectory_steps > 0
        for candidate in result.data.batch_candidates
        if candidate.motion_plan_error is False
    )
    plan_kwargs = captured["plan_kwargs"]
    assert plan_kwargs["target_left_pos"].shape == (4, 3)


def test_execute_by_trajectory_cache_key_reuses_cached_waypoints_without_replanning(
    monkeypatch,
) -> None:
    tool = FreespaceMoveTool()
    exec_client = _TrajectoryExecClient()
    monkeypatch.setattr(tool, "_get_client", lambda: exec_client)

    cached_left = np.asarray([[0.0] * 6, [0.1] * 6, [0.2] * 6], dtype=np.float64)
    cached_right = np.asarray([[0.0] * 6, [0.0] * 6, [0.0] * 6], dtype=np.float64)
    cache_key = tool._store_cached_trajectory(
        side="left",
        current_left_jp=np.zeros(6, dtype=np.float64),
        current_right_jp=np.zeros(6, dtype=np.float64),
        current_left_gp=0.6,
        current_right_gp=0.4,
        left_positions=cached_left,
        right_positions=cached_right,
        timestamps=[0.0, 0.5, 1.0],
        final_pos_error_m=0.001,
        final_rot_error_deg=0.1,
        final_pose_error=0.01,
    )

    called = {"plan": 0}

    class _Planner:
        def plan_to_pose(self, **kwargs):
            called["plan"] += 1
            raise AssertionError(
                "execute should not call planner when trajectory_cache_key is supplied"
            )

    monkeypatch.setattr(
        tool,
        "_get_planner",
        lambda **kwargs: _Planner(),
    )

    result = tool.execute(
        trajectory_cache_key=cache_key,
        side="left",
        preview_only=False,
    )

    assert result.success is True
    assert result.data.executed is True
    assert called["plan"] == 0
    assert exec_client.moves, "cached trajectory should have been executed"


def test_execute_rejects_missing_or_stale_trajectory_cache(monkeypatch) -> None:
    tool = FreespaceMoveTool()
    monkeypatch.setattr(tool, "_get_client", lambda: _FakeClient())

    missing = tool.execute(
        trajectory_cache_key="missing-cache-key",
        side="left",
        preview_only=False,
    )
    assert missing.success is False
    assert missing.data.status in {
        "Invalid",
        "Error",
        "Planning_Failed",
        "Preview_Missing",
    }
    assert missing.data.executed is False

    cache_key = tool._store_cached_trajectory(
        side="left",
        current_left_jp=np.zeros(6, dtype=np.float64),
        current_right_jp=np.zeros(6, dtype=np.float64),
        current_left_gp=0.6,
        current_right_gp=0.4,
        left_positions=np.asarray([[0.0] * 6, [0.1] * 6], dtype=np.float64),
        right_positions=np.asarray([[0.0] * 6, [0.0] * 6], dtype=np.float64),
        timestamps=[0.0, 1.0],
        final_pos_error_m=0.001,
        final_rot_error_deg=0.1,
        final_pose_error=0.01,
    )
    with tool._trajectory_cache_lock:
        tool._trajectory_cache[cache_key]["current_left_jp"] = np.ones(
            6, dtype=np.float64
        )

    stale = tool.execute(
        trajectory_cache_key=cache_key,
        side="left",
        preview_only=False,
    )
    assert stale.success is False
    assert stale.data.status in {"Preview_Stale", "Execution_Failed", "Error"}
    assert stale.data.executed is False


def test_single_preview_returns_cache_key_and_later_execute_reuses_it_without_replanning(
    monkeypatch,
) -> None:
    tool = FreespaceMoveTool()
    exec_client = _TrajectoryExecClient()
    monkeypatch.setattr(tool, "_get_client", lambda: exec_client)

    target_pos = np.asarray([0.52, 0.1, 0.82], dtype=np.float64)
    target_quat = tool._display_rpy_to_quat([0.0, 90.0, 0.0])
    current_pos = np.asarray([0.42, 0.1, 0.82], dtype=np.float64)

    class _Planner:
        ik_position_cost = 1.0
        ik_orientation_cost = 0.3

        def __init__(self):
            self.calls = 0

        def plan_to_pose(self, **kwargs):
            self.calls += 1
            return {
                "status": "Success",
                "left_positions": np.asarray(
                    [[0.0] * 6, [0.1] * 6, [0.2] * 6], dtype=np.float64
                ),
                "right_positions": np.asarray(
                    [[0.0] * 6, [0.0] * 6, [0.0] * 6], dtype=np.float64
                ),
            }

    planner = _Planner()
    monkeypatch.setattr(tool, "_get_planner", lambda **kwargs: planner)

    class _DiagKin:
        def forward_kinematics(self, left_jp, right_jp):
            left_jp = np.asarray(left_jp, dtype=np.float64)
            if np.allclose(left_jp, np.zeros(6, dtype=np.float64)):
                return (
                    current_pos.copy(),
                    target_quat.copy(),
                    np.asarray([0.4, -0.2, 0.8], dtype=np.float64),
                    target_quat.copy(),
                )
            return (
                target_pos.copy(),
                target_quat.copy(),
                np.asarray([0.4, -0.2, 0.8], dtype=np.float64),
                target_quat.copy(),
            )

        def inverse_kinematics(self, *args, **kwargs):
            raise AssertionError(
                "single preview success path should not call inverse_kinematics"
            )

    monkeypatch.setattr(
        tool,
        "_get_diagnostic_kinematics",
        lambda **kwargs: _DiagKin(),
    )

    preview = tool.execute(
        left_target_pos=target_pos.tolist(),
        left_target_rpy=[0.0, 90.0, 0.0],
        preview_only=True,
    )

    assert preview.success is True
    assert preview.data.executed is False
    assert preview.data.trajectory_cache_key is not None
    assert planner.calls == 1

    execute = tool.execute(
        trajectory_cache_key=preview.data.trajectory_cache_key,
        preview_only=False,
    )

    assert execute.success is True
    assert execute.data.executed is True
    assert execute.data.trajectory_cache_key == preview.data.trajectory_cache_key
    assert planner.calls == 1
    assert exec_client.moves, (
        "cached single-target trajectory should have been executed"
    )


def test_curobo_planner_cache_ignores_diagnostic_weights(monkeypatch) -> None:
    tool = FreespaceMoveTool()

    import enpire.env.forge.experimental.portal_motion_planner as portal_motion_planner

    created: list[dict[str, object]] = []

    class _FakePortalMotionPlanner:
        def __init__(self, backend, *, solver_speed, **kwargs):
            created.append(
                {
                    "backend": str(backend),
                    "solver_speed": str(solver_speed),
                    "extra_kwargs": dict(kwargs),
                }
            )

    monkeypatch.setattr(
        portal_motion_planner,
        "PortalMotionPlanner",
        _FakePortalMotionPlanner,
    )

    planner_a = tool._get_planner(
        planner_backend="curobo",
        solver_speed="slow",
        ik_xyz_weight=1.0,
        ik_rpy_weight=0.3,
    )
    planner_b = tool._get_planner(
        planner_backend="curobo",
        solver_speed="slow",
        ik_xyz_weight=9.0,
        ik_rpy_weight=4.0,
    )

    assert planner_a is planner_b
    assert len(created) == 1
    assert created[0]["backend"] == "curobo"
    assert created[0]["solver_speed"] == "slow"
    assert created[0]["extra_kwargs"]["position_threshold"] == pytest.approx(0.005)
    assert created[0]["extra_kwargs"]["rotation_threshold"] == pytest.approx(0.05)


def test_curobo_planner_cache_respects_thresholds(monkeypatch) -> None:
    tool = FreespaceMoveTool()

    import enpire.env.forge.experimental.portal_motion_planner as portal_motion_planner

    created: list[dict[str, object]] = []

    class _FakePortalMotionPlanner:
        def __init__(self, backend, *, solver_speed, **kwargs):
            created.append(
                {
                    "backend": str(backend),
                    "solver_speed": str(solver_speed),
                    "extra_kwargs": dict(kwargs),
                }
            )

    monkeypatch.setattr(
        portal_motion_planner,
        "PortalMotionPlanner",
        _FakePortalMotionPlanner,
    )

    planner_a = tool._get_planner(
        planner_backend="curobo",
        solver_speed="slow",
        ik_error_threshold=0.005,
        ik_rot_threshold_deg=2.0,
    )
    planner_b = tool._get_planner(
        planner_backend="curobo",
        solver_speed="slow",
        ik_error_threshold=0.010,
        ik_rot_threshold_deg=2.0,
    )

    assert planner_a is not planner_b
    assert len(created) == 2
    assert created[0]["extra_kwargs"]["position_threshold"] == pytest.approx(0.005)
    assert created[0]["extra_kwargs"]["rotation_threshold"] == pytest.approx(
        np.deg2rad(2.0)
    )
    assert created[1]["extra_kwargs"]["position_threshold"] == pytest.approx(0.010)


def test_curobo_planner_uses_remote_portal_env(monkeypatch) -> None:
    tool = FreespaceMoveTool()

    import enpire.env.forge.experimental.portal_motion_planner as portal_motion_planner

    created: list[dict[str, object]] = []

    class _FakePortalMotionPlanner:
        def __init__(
            self, backend, *, solver_speed, host, port, start_server, **kwargs
        ):
            created.append(
                {
                    "backend": str(backend),
                    "solver_speed": str(solver_speed),
                    "host": str(host),
                    "port": int(port),
                    "start_server": bool(start_server),
                    "extra_kwargs": dict(kwargs),
                }
            )

    monkeypatch.setattr(
        portal_motion_planner,
        "PortalMotionPlanner",
        _FakePortalMotionPlanner,
    )
    monkeypatch.setenv("CAP_CUROBO_HOST", "127.0.0.1")
    monkeypatch.setenv("CAP_CUROBO_PORT", "8611")
    monkeypatch.setenv("CAP_CUROBO_START_SERVER", "0")

    tool._get_planner(
        planner_backend="curobo",
        solver_speed="fast",
        ik_xyz_weight=1.0,
        ik_rpy_weight=0.3,
    )

    assert created == [
        {
            "backend": "curobo",
            "solver_speed": "fast",
            "host": "127.0.0.1",
            "port": 8611,
            "start_server": False,
            "extra_kwargs": {
                "position_threshold": pytest.approx(0.005),
                "rotation_threshold": pytest.approx(0.05),
                "robot_type": "yam",
            },
        }
    ]


def test_batch_mode_truncates_to_top_16_and_defaults_to_fast(monkeypatch) -> None:
    tool = FreespaceMoveTool()
    monkeypatch.setattr(tool, "_get_client", lambda: _FakeClient())

    captured: dict[str, object] = {}

    class _Planner:
        def plan_batch_to_pose(self, **kwargs):
            captured["plan_kwargs"] = kwargs
            n = len(kwargs["target_left_pos"])
            left_positions_by_index = [
                np.asarray(
                    [
                        np.zeros(6, dtype=np.float64) + i * 0.00,
                        np.zeros(6, dtype=np.float64) + i * 0.01,
                        np.zeros(6, dtype=np.float64) + i * 0.02,
                    ],
                    dtype=np.float64,
                )
                for i in range(n)
            ]
            right_positions_by_index = [
                np.asarray(
                    [
                        np.zeros(6, dtype=np.float64) + i * 0.00,
                        np.zeros(6, dtype=np.float64) + i * 0.01,
                        np.zeros(6, dtype=np.float64) + i * 0.02,
                    ],
                    dtype=np.float64,
                )
                for i in range(n)
            ]
            return {
                "success_mask": np.ones((n,), dtype=bool),
                "status_detail_by_index": [None] * n,
                "position_error_m": np.linspace(0.001, 0.016, n),
                "rotation_error_deg": np.linspace(0.1, 1.6, n),
                "left_positions_by_index": left_positions_by_index,
                "right_positions_by_index": right_positions_by_index,
                "curobo_solve_time_ms": 12.5,
                "curobo_total_time_ms": 18.0,
                "curobo_graph_time_ms": 4.0,
                "curobo_ik_time_ms": 3.0,
            }

    def _fake_get_planner(
        *,
        planner_backend,
        solver_speed,
        ik_error_threshold,
        ik_rot_threshold_deg,
        ik_xyz_weight,
        ik_rpy_weight,
    ):
        captured["planner_backend"] = planner_backend
        captured["solver_speed"] = solver_speed
        captured["ik_error_threshold"] = ik_error_threshold
        captured["ik_rot_threshold_deg"] = ik_rot_threshold_deg
        captured["ik_xyz_weight"] = ik_xyz_weight
        captured["ik_rpy_weight"] = ik_rpy_weight
        return _Planner()

    monkeypatch.setattr(tool, "_get_planner", _fake_get_planner)

    candidates = [
        _Candidate(
            position=[0.5 + i * 0.001, 0.1, 0.8],
            rpy=[0.0, 90.0, 0.0],
            score=float(i),
            width=0.04,
        )
        for i in range(20)
    ]

    result = tool.execute(grasp_candidates=candidates, batch_side="left")

    assert result.success is True
    assert result.data.planning_mode == "batch"
    assert result.data.input_candidate_count == 20
    assert result.data.evaluated_candidate_count == 16
    assert result.data.truncated_input_count == 4
    assert len(result.data.batch_candidates) == 16
    assert captured["solver_speed"] == "fast"
    assert captured["planner_backend"] == "curobo"
    assert captured["ik_error_threshold"] == pytest.approx(0.005)
    assert captured["ik_rot_threshold_deg"] == pytest.approx(np.degrees(0.05))
    plan_kwargs = captured["plan_kwargs"]
    assert plan_kwargs["target_left_pos"].shape == (16, 3)
    # top-16 by score => keep scores 19..4, so largest x should be first before IK rerank
    assert np.isclose(np.max(plan_kwargs["target_left_pos"][:, 0]), 0.519)
    assert result.data.best_candidate is not None


def test_batch_mode_accepts_dict_or_dataclass_candidates_and_marks_failures(
    monkeypatch,
) -> None:
    tool = FreespaceMoveTool()
    monkeypatch.setattr(tool, "_get_client", lambda: _FakeClient())

    class _Planner:
        def plan_batch_to_pose(self, **kwargs):
            n = len(kwargs["target_left_pos"])
            left_positions = [
                np.asarray(
                    [
                        np.zeros(6, dtype=np.float64) + i * 0.00,
                        np.zeros(6, dtype=np.float64) + i * 0.01,
                        np.zeros(6, dtype=np.float64) + i * 0.02,
                    ],
                    dtype=np.float64,
                )
                for i in range(n)
            ]
            right_positions = [
                np.asarray(
                    [
                        np.zeros(6, dtype=np.float64) + i * 0.00,
                        np.zeros(6, dtype=np.float64) + i * 0.01,
                        np.zeros(6, dtype=np.float64) + i * 0.02,
                    ],
                    dtype=np.float64,
                )
                for i in range(n)
            ]
            success_mask = np.asarray([True, False, True], dtype=bool)
            return {
                "success_mask": success_mask,
                "status_by_index": ["Success", "Planning_Failed", "Success"],
                "status_detail_by_index": [None, "collision", None],
                "position_error_m": np.asarray(
                    [0.004, np.nan, 0.002], dtype=np.float64
                ),
                "rotation_error_deg": np.asarray([1.0, np.nan, 0.5], dtype=np.float64),
                "left_positions_by_index": [left_positions[0], None, left_positions[2]],
                "right_positions_by_index": [
                    right_positions[0],
                    None,
                    right_positions[2],
                ],
                "curobo_solve_time_ms": 8.0,
                "curobo_total_time_ms": 10.0,
                "curobo_graph_time_ms": 2.0,
                "curobo_ik_time_ms": 1.0,
            }

    monkeypatch.setattr(
        tool,
        "_get_planner",
        lambda **kwargs: _Planner(),
    )

    candidates = [
        {
            "position": [0.5, 0.1, 0.8],
            "rpy": [0.0, 90.0, 0.0],
            "score": 0.8,
            "width": 0.04,
        },
        SimpleNamespace(
            position=[0.51, 0.1, 0.8], rpy=[0.0, 90.0, 10.0], score=0.9, width=0.05
        ),
        _Candidate(
            position=[0.52, 0.1, 0.8], rpy=[0.0, 90.0, -10.0], score=0.7, width=0.03
        ),
    ]

    result = tool.execute(
        grasp_candidates=candidates,
        batch_side="left",
        solver_speed="fast",
        ik_error_threshold=0.005,
    )

    assert result.success is True
    assert result.data.best_candidate is not None
    assert result.data.best_candidate.position == [0.52, 0.1, 0.8]
    assert result.data.best_candidate.ik_error_m == 0.002
    failed = [
        candidate
        for candidate in result.data.batch_candidates
        if candidate.motion_plan_error
    ]
    assert len(failed) == 1
    assert failed[0].motion_plan_reason == "collision"


def test_batch_mode_preserves_finite_ik_error_for_ik_failed_rows(
    monkeypatch,
) -> None:
    tool = FreespaceMoveTool()
    monkeypatch.setattr(tool, "_get_client", lambda: _FakeClient())

    class _Planner:
        def plan_batch_to_pose(self, **kwargs):
            n = len(kwargs["target_left_pos"])
            left_positions = [
                np.asarray(
                    [
                        np.zeros(6, dtype=np.float64) + i * 0.00,
                        np.zeros(6, dtype=np.float64) + i * 0.01,
                    ],
                    dtype=np.float64,
                )
                for i in range(n)
            ]
            right_positions = [
                np.asarray(
                    [
                        np.zeros(6, dtype=np.float64) + i * 0.00,
                        np.zeros(6, dtype=np.float64) + i * 0.01,
                    ],
                    dtype=np.float64,
                )
                for i in range(n)
            ]
            return {
                "success_mask": np.asarray([True, False, False], dtype=bool),
                "status_by_index": ["Success", "IK_Failed", "Planning_Failed"],
                "status_detail_by_index": [None, "ik_fail", "collision"],
                "position_error_m": np.asarray(
                    [0.004, 0.0362, np.nan], dtype=np.float64
                ),
                "rotation_error_deg": np.asarray([1.0, 9.26, np.nan], dtype=np.float64),
                "left_positions_by_index": [left_positions[0], None, None],
                "right_positions_by_index": [right_positions[0], None, None],
                "curobo_solve_time_ms": 8.0,
                "curobo_total_time_ms": 10.0,
                "curobo_graph_time_ms": 2.0,
                "curobo_ik_time_ms": 1.0,
            }

    monkeypatch.setattr(tool, "_get_planner", lambda **kwargs: _Planner())

    result = tool.execute(
        grasp_candidates=[
            {
                "position": [0.5, 0.1, 0.8],
                "rpy": [0.0, 90.0, 0.0],
                "score": 0.8,
                "width": 0.04,
            },
            {
                "position": [0.51, 0.1, 0.8],
                "rpy": [0.0, 90.0, 10.0],
                "score": 0.9,
                "width": 0.05,
            },
            {
                "position": [0.52, 0.1, 0.8],
                "rpy": [0.0, 90.0, -10.0],
                "score": 0.7,
                "width": 0.03,
            },
        ],
        batch_side="left",
        solver_speed="fast",
        ik_error_threshold=0.005,
    )

    assert result.success is True
    assert result.data.best_candidate is not None
    assert result.data.best_candidate.source_index == 0

    ranked = result.data.batch_candidates
    assert [candidate.planner_status for candidate in ranked] == [
        "Success",
        "IK_Failed",
        "Planning_Failed",
    ]
    assert ranked[0].is_executable is True
    assert ranked[0].is_ik_failed is False
    assert ranked[0].is_planning_failed is False

    ik_failed = ranked[1]
    assert ik_failed.ik_error_m == pytest.approx(0.0362)
    assert ik_failed.ik_rot_error_deg == pytest.approx(9.26)
    assert ik_failed.within_ik_threshold is False
    assert ik_failed.motion_plan_error is None
    assert ik_failed.trajectory_cache_key is None
    assert ik_failed.is_executable is False
    assert ik_failed.is_ik_failed is True
    assert ik_failed.is_planning_failed is False

    planning_failed = ranked[2]
    assert planning_failed.motion_plan_error is True
    assert planning_failed.motion_plan_reason == "collision"
    assert planning_failed.is_executable is False
    assert planning_failed.is_ik_failed is False
    assert planning_failed.is_planning_failed is True


def test_single_planning_failure_includes_planner_detail(monkeypatch) -> None:
    tool = FreespaceMoveTool()
    monkeypatch.setattr(tool, "_get_client", lambda: _FakeClient())

    class _Planner:
        def plan_to_pose(self, **kwargs):
            return {
                "status": "Planning_Failed",
                "status_detail": "self-collision near start state",
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

    monkeypatch.setattr(tool, "_get_planner", lambda **kwargs: _Planner())
    monkeypatch.setattr(
        tool,
        "_get_diagnostic_kinematics",
        lambda **kwargs: _DiagKin(),
    )

    result = tool.execute(
        left_target_pos=[0.5, 0.1, 0.8],
        left_target_rpy=[0.0, 90.0, 0.0],
    )

    assert result.success is False
    assert result.data.status == "Planning_Failed"
    assert "self-collision near start state" in result.data.reason
    assert "self-collision near start state" in str(result.error)


def test_batch_sort_breaks_display_ties_by_score(monkeypatch) -> None:
    tool = FreespaceMoveTool()
    monkeypatch.setattr(tool, "_get_client", lambda: _FakeClient())

    class _Planner:
        def plan_batch_to_pose(self, **kwargs):
            n = len(kwargs["target_left_pos"])
            left_positions = [
                np.asarray(
                    [
                        np.zeros(6, dtype=np.float64),
                        np.zeros(6, dtype=np.float64) + 0.01,
                    ],
                    dtype=np.float64,
                )
                for _ in range(n)
            ]
            right_positions = [
                np.asarray(
                    [
                        np.zeros(6, dtype=np.float64),
                        np.zeros(6, dtype=np.float64) + 0.01,
                    ],
                    dtype=np.float64,
                )
                for _ in range(n)
            ]
            return {
                "success_mask": np.asarray([True, True, True], dtype=bool),
                "status_by_index": ["Success", "Success", "Success"],
                "status_detail_by_index": [None, None, None],
                "position_error_m": np.asarray(
                    [0.00004, 0.00003, 0.00002], dtype=np.float64
                ),
                "rotation_error_deg": np.asarray(
                    [0.004, 0.003, 0.002], dtype=np.float64
                ),
                "left_positions_by_index": left_positions,
                "right_positions_by_index": right_positions,
            }

    monkeypatch.setattr(tool, "_get_planner", lambda **kwargs: _Planner())

    result = tool.execute(
        grasp_candidates=[
            {
                "position": [0.5, 0.1, 0.8],
                "rpy": [0.0, 90.0, 0.0],
                "score": 0.7,
                "width": 0.04,
            },
            {
                "position": [0.51, 0.1, 0.8],
                "rpy": [0.0, 90.0, 10.0],
                "score": 0.3,
                "width": 0.05,
            },
            {
                "position": [0.52, 0.1, 0.8],
                "rpy": [0.0, 90.0, -10.0],
                "score": 1.0,
                "width": 0.03,
            },
        ],
        batch_side="left",
        solver_speed="fast",
        ik_error_threshold=0.005,
    )

    assert result.success is True
    ranked = result.data.batch_candidates
    assert [candidate.source_index for candidate in ranked] == [2, 0, 1]
    assert [candidate.rank for candidate in ranked] == [1, 2, 3]
    assert result.data.best_candidate is not None
    assert result.data.best_candidate.source_index == 2


def test_batch_within_threshold_requires_both_position_and_rotation(
    monkeypatch,
) -> None:
    tool = FreespaceMoveTool()
    monkeypatch.setattr(tool, "_get_client", lambda: _FakeClient())

    class _Planner:
        def plan_batch_to_pose(self, **kwargs):
            left_positions = [
                np.asarray(
                    [
                        np.zeros(6, dtype=np.float64),
                        np.zeros(6, dtype=np.float64) + 0.01,
                    ],
                    dtype=np.float64,
                )
            ]
            right_positions = [
                np.asarray(
                    [
                        np.zeros(6, dtype=np.float64),
                        np.zeros(6, dtype=np.float64) + 0.01,
                    ],
                    dtype=np.float64,
                )
            ]
            return {
                "success_mask": np.asarray([True], dtype=bool),
                "status_by_index": ["Success"],
                "status_detail_by_index": [None],
                "position_error_m": np.asarray([0.004], dtype=np.float64),
                "rotation_error_deg": np.asarray([5.0], dtype=np.float64),
                "left_positions_by_index": left_positions,
                "right_positions_by_index": right_positions,
            }

    monkeypatch.setattr(tool, "_get_planner", lambda **kwargs: _Planner())

    result = tool.execute(
        grasp_candidates=[
            {
                "position": [0.5, 0.1, 0.8],
                "rpy": [0.0, 90.0, 0.0],
                "score": 0.8,
                "width": 0.04,
            }
        ],
        batch_side="left",
        ik_error_threshold=0.005,
        ik_rot_threshold_deg=2.0,
    )

    assert result.success is True
    assert len(result.data.batch_candidates) == 1
    assert result.data.batch_candidates[0].within_ik_threshold is False


def test_single_ik_failure_uses_curobo_reported_residuals(monkeypatch) -> None:
    tool = FreespaceMoveTool()
    monkeypatch.setattr(tool, "_get_client", lambda: _FakeClient())

    class _Planner:
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
            raise AssertionError("single IK failure should not recompute fallback IK")

    monkeypatch.setattr(tool, "_get_planner", lambda **kwargs: _Planner())
    monkeypatch.setattr(
        tool,
        "_get_diagnostic_kinematics",
        lambda **kwargs: _DiagKin(),
    )
    monkeypatch.setattr(
        tool,
        "_compute_plan_diagnostics",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("single IK failure should not recompute diagnostics")
        ),
    )

    result = tool.execute(
        left_target_pos=[0.5, 0.1, 0.8],
        left_target_rpy=[0.0, 90.0, 0.0],
        ik_error_threshold=0.005,
    )

    assert result.success is False
    assert result.data.status == "IK_Failed"
    assert result.data.ik_error_m == pytest.approx(0.0033)
    assert result.data.final_pos_error_m == pytest.approx(0.0033)
    assert result.data.final_rot_error_deg == pytest.approx(40.91)
    assert "reported by cuRobo" in result.data.reason
    assert "0.0033" in str(result.error)
    assert "40.91" in str(result.error)


def test_batch_mode_rejects_mixing_candidates_with_single_pose_inputs() -> None:
    tool = FreespaceMoveTool()
    result = tool.execute(
        grasp_candidates=[{"position": [0.5, 0.1, 0.8], "rpy": [0.0, 90.0, 0.0]}],
        batch_side="left",
        left_target_pos=[0.5, 0.1, 0.8],
    )

    assert result.success is False
    assert result.data.status == "Invalid"
