from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experimental.motion_planner_curobo import YamMotionPlannerCurobo


def test_plan_to_pose_routes_through_batch_interface() -> None:
    planner = YamMotionPlannerCurobo.__new__(YamMotionPlannerCurobo)
    captured: dict[str, object] = {}

    def _fake_plan_batch_to_pose(**kwargs):
        captured.update(kwargs)
        return {
            "status": "Success",
            "status_detail": None,
            "success_mask": np.asarray([True], dtype=bool),
            "status_by_index": ["Success"],
            "status_detail_by_index": [None],
            "position_error_m": np.asarray([0.001], dtype=np.float64),
            "rotation_error_deg": np.asarray([0.2], dtype=np.float64),
            "left_positions_by_index": [
                np.asarray([[0.0] * 6, [0.1] * 6], dtype=np.float64)
            ],
            "right_positions_by_index": [
                np.asarray([[0.0] * 6, [0.0] * 6], dtype=np.float64)
            ],
            "curobo_total_time_ms": 12.0,
        }

    planner.plan_batch_to_pose = _fake_plan_batch_to_pose  # type: ignore[attr-defined]

    result = YamMotionPlannerCurobo.plan_to_pose(
        planner,
        current_left_jp=np.zeros(6, dtype=np.float64),
        current_right_jp=np.zeros(6, dtype=np.float64),
        target_left_pos=np.asarray([0.5, 0.1, 0.8], dtype=np.float64),
        target_left_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        side="left",
    )

    assert np.asarray(captured["target_left_pos"]).shape == (1, 3)
    assert np.asarray(captured["target_left_quat_xyzw"]).shape == (1, 4)
    assert result["status"] == "Success"
    assert result["left_positions"].shape == (2, 6)
    assert result["right_positions"].shape == (2, 6)


# ---------------------------------------------------------------------------
# Helpers for batch planning tests
# ---------------------------------------------------------------------------


def _make_batch_planner(batch_capacity: int = 8) -> YamMotionPlannerCurobo:
    """Create a bare ``YamMotionPlannerCurobo`` with just enough state for
    ``plan_batch_to_pose`` to run without touching GPU or MuJoCo."""
    planner = YamMotionPlannerCurobo.__new__(YamMotionPlannerCurobo)
    planner._left_gripper = 0.0  # type: ignore[attr-defined]
    planner._right_gripper = 0.0  # type: ignore[attr-defined]
    planner._validator = None  # type: ignore[attr-defined]
    planner._joint_limit_lower = -np.pi * np.ones(12, dtype=np.float64)  # type: ignore[attr-defined]
    planner._joint_limit_upper = np.pi * np.ones(12, dtype=np.float64)  # type: ignore[attr-defined]
    planner._batch_planner_capacity = batch_capacity  # type: ignore[attr-defined]
    # Minimal kinematics stub returning identity-ish FK for current joints.
    planner._kin = SimpleNamespace(  # type: ignore[attr-defined]
        forward_kinematics=lambda left_jp, right_jp: (
            np.zeros(3, dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        ),
    )
    return planner


def _make_chunk_result(
    *,
    success_flags: list[bool],
    failed_status: str = "IK_Failed",
    traj_steps: int = 5,
) -> dict:
    """Build a dict shaped like ``_plan_batch_to_pose_chunk`` output."""
    n = len(success_flags)
    success_mask = np.asarray(success_flags, dtype=bool)
    status_by_index = ["Success" if ok else failed_status for ok in success_flags]
    status_detail_by_index: list[str | None] = [
        None if ok else "mock detail" for ok in success_flags
    ]
    left_positions_by_index: list[np.ndarray | None] = [
        np.random.default_rng(idx).random((traj_steps, 6)).astype(np.float64)
        if ok
        else None
        for idx, ok in enumerate(success_flags)
    ]
    right_positions_by_index: list[np.ndarray | None] = [
        np.random.default_rng(idx + 100).random((traj_steps, 6)).astype(np.float64)
        if ok
        else None
        for idx, ok in enumerate(success_flags)
    ]
    if np.all(success_mask):
        status = "Success"
    elif np.any(success_mask):
        status = "Partial_Success"
    else:
        status = "Planning_Failed"
    return {
        "status": status,
        "status_detail": None if np.all(success_mask) else "mock detail",
        "success_mask": success_mask,
        "status_by_index": status_by_index,
        "status_detail_by_index": status_detail_by_index,
        "position_error_m": np.full(n, 0.005, dtype=np.float64),
        "rotation_error_deg": np.full(n, 1.2, dtype=np.float64),
        "left_positions_by_index": left_positions_by_index,
        "right_positions_by_index": right_positions_by_index,
        "curobo_solve_time_ms": 10.0,
        "curobo_total_time_ms": 12.0,
        "curobo_ik_time_ms": 3.0,
        "curobo_graph_time_ms": 2.0,
        "curobo_trajopt_time_ms": 4.0,
        "curobo_finetune_time_ms": 1.0,
        "curobo_attempts": 1,
        "curobo_trajopt_attempts": 1,
        "curobo_used_graph": False,
    }


# ---------------------------------------------------------------------------
# Batch planning: partial success (indices 0,2 pass; indices 1,3 fail)
# ---------------------------------------------------------------------------


def test_plan_batch_to_pose_with_partial_success() -> None:
    planner = _make_batch_planner(batch_capacity=8)

    chunk_flags = [True, False, True, False]

    def _fake_chunk(**kwargs):
        return _make_chunk_result(success_flags=chunk_flags)

    planner._plan_batch_to_pose_chunk = _fake_chunk  # type: ignore[attr-defined]

    batch_size = 4
    result = YamMotionPlannerCurobo.plan_batch_to_pose(
        planner,
        current_left_jp=np.zeros(6, dtype=np.float64),
        current_right_jp=np.zeros(6, dtype=np.float64),
        target_left_pos=np.random.default_rng(0)
        .random((batch_size, 3))
        .astype(np.float64),
        target_left_quat_xyzw=np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (batch_size, 1)
        ),
        side="left",
    )

    # -- success_mask ---
    sm = np.asarray(result["success_mask"], dtype=bool).reshape(-1)
    assert sm.shape == (4,)
    assert sm.tolist() == [True, False, True, False]

    # -- per-index trajectories ---
    left_by_idx = result["left_positions_by_index"]
    right_by_idx = result["right_positions_by_index"]
    assert left_by_idx[0] is not None
    assert left_by_idx[2] is not None
    assert left_by_idx[1] is None
    assert left_by_idx[3] is None
    assert right_by_idx[0] is not None
    assert right_by_idx[2] is not None
    assert right_by_idx[1] is None
    assert right_by_idx[3] is None

    # -- per-index status ---
    assert result["status_by_index"][0] == "Success"
    assert result["status_by_index"][2] == "Success"
    assert result["status_by_index"][1] != "Success"
    assert result["status_by_index"][3] != "Success"

    # -- position_error_m finite for all indices ---
    pe = np.asarray(result["position_error_m"], dtype=np.float64).reshape(-1)
    assert pe.shape == (4,)
    assert np.all(np.isfinite(pe))

    # -- overall status ---
    assert result["status"] == "Partial_Success"


# ---------------------------------------------------------------------------
# Batch planning: all targets fail
# ---------------------------------------------------------------------------


def test_plan_batch_to_pose_all_fail() -> None:
    planner = _make_batch_planner(batch_capacity=8)

    chunk_flags = [False, False, False, False]

    def _fake_chunk(**kwargs):
        return _make_chunk_result(success_flags=chunk_flags)

    planner._plan_batch_to_pose_chunk = _fake_chunk  # type: ignore[attr-defined]

    batch_size = 4
    result = YamMotionPlannerCurobo.plan_batch_to_pose(
        planner,
        current_left_jp=np.zeros(6, dtype=np.float64),
        current_right_jp=np.zeros(6, dtype=np.float64),
        target_left_pos=np.random.default_rng(1)
        .random((batch_size, 3))
        .astype(np.float64),
        target_left_quat_xyzw=np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (batch_size, 1)
        ),
        side="left",
    )

    # -- overall status ---
    assert result["status"] == "Planning_Failed"

    # -- success_mask ---
    sm = np.asarray(result["success_mask"], dtype=bool).reshape(-1)
    assert sm.shape == (4,)
    assert not np.any(sm)

    # -- all per-index trajectories are None ---
    for idx in range(4):
        assert result["left_positions_by_index"][idx] is None
        assert result["right_positions_by_index"][idx] is None

    # -- all per-index statuses are failures ---
    for idx in range(4):
        assert result["status_by_index"][idx] != "Success"

    # -- position_error_m still has finite values ---
    pe = np.asarray(result["position_error_m"], dtype=np.float64).reshape(-1)
    assert pe.shape == (4,)
    assert np.all(np.isfinite(pe))
