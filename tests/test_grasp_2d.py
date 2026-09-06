# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from enpire.env.forge.cap.agent.tools.grasp_2d import (
    _TWO_D_GRASP_PLANNER_Z_M,
    SampleGraspPose2DTool,
    compute_segmented_cloud_height_m,
    estimate_local_tangent_from_mask,
    extract_segmented_object_world_points,
    plan_top_down_grasps_from_mask,
    project_pixel_to_plane_world,
    project_world_to_pixel,
    top_down_yaw_from_world_axis,
    world_axis_from_top_down_yaw,
)
from enpire.env.forge.cap.config import TABLE_SURFACE_Z_M


def _top_down_camera_transform() -> np.ndarray:
    T_cam_world = np.eye(4, dtype=np.float64)
    T_cam_world[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_cam_world[:3, 3] = np.array([0.2, -0.1, 1.0], dtype=np.float64)
    return T_cam_world


def test_extract_segmented_object_world_points_and_height_filters_invalid_depth() -> None:
    depth = np.array([[0.40, 0.50], [0.52, 2.00]], dtype=np.float32)
    cam_K = np.eye(3, dtype=np.float64)
    T_cam_world = np.eye(4, dtype=np.float64)
    mask = np.ones((2, 2), dtype=np.uint8)

    points_world = extract_segmented_object_world_points(depth, cam_K, T_cam_world, mask)

    assert points_world.shape == (3, 3)
    assert compute_segmented_cloud_height_m(points_world) == pytest.approx(0.12, abs=1e-6)
    assert compute_segmented_cloud_height_m(np.empty((0, 3), dtype=np.float64)) == 0.0


def test_project_pixel_to_plane_world_intersects_table_plane() -> None:
    cam_K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    T_cam_world = _top_down_camera_transform()

    world = project_pixel_to_plane_world(
        [50.0, 40.0],
        cam_K,
        T_cam_world,
        plane_z_m=float(TABLE_SURFACE_Z_M),
    )

    assert world is not None
    assert world == pytest.approx([0.2, -0.1, TABLE_SURFACE_Z_M], abs=1e-6)


def test_top_down_yaw_from_world_axis_matches_repo_convention() -> None:
    assert top_down_yaw_from_world_axis(np.array([0.0, -1.0])) == pytest.approx(0.0, abs=1e-6)
    assert top_down_yaw_from_world_axis(np.array([1.0, 0.0])) == pytest.approx(-90.0, abs=1e-6)
    assert top_down_yaw_from_world_axis(np.array([-1.0, 0.0])) == pytest.approx(90.0, abs=1e-6)


def test_world_axis_from_top_down_yaw_is_inverse_of_repo_convention() -> None:
    axis = world_axis_from_top_down_yaw(0.0)
    assert axis == pytest.approx([0.0, -1.0], abs=1e-6)
    assert top_down_yaw_from_world_axis(axis) == pytest.approx(0.0, abs=1e-6)

    axis = world_axis_from_top_down_yaw(-90.0)
    assert axis == pytest.approx([1.0, 0.0], abs=1e-6)
    assert top_down_yaw_from_world_axis(axis) == pytest.approx(-90.0, abs=1e-6)


def test_project_world_to_pixel_inverts_top_down_center_projection() -> None:
    cam_K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    T_cam_world = _top_down_camera_transform()

    pixel = project_world_to_pixel([0.2, -0.1, TABLE_SURFACE_Z_M], cam_K, T_cam_world)

    assert pixel == (50, 40)


def test_estimate_local_tangent_from_mask_changes_across_bent_shape() -> None:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[8:11, 2:13] = 1
    mask[10:21, 10:13] = 1

    tangent_left, _endpoints_left, _ratio_left = estimate_local_tangent_from_mask(
        mask,
        [4.0, 9.0],
        reference_axis_px=np.array([1.0, 0.0]),
    )
    tangent_lower, _endpoints_lower, _ratio_lower = estimate_local_tangent_from_mask(
        mask,
        [11.0, 18.0],
        reference_axis_px=np.array([0.0, 1.0]),
    )

    assert tangent_left is not None
    assert tangent_lower is not None
    assert abs(float(np.dot(tangent_left, np.array([1.0, 0.0])))) > 0.9
    assert abs(float(np.dot(tangent_lower, np.array([0.0, 1.0])))) > 0.9


def _canonical_opening_axis(yaw_deg: float) -> tuple[float, float]:
    axis = np.asarray(world_axis_from_top_down_yaw(yaw_deg), dtype=np.float64)
    if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
        axis = -axis
    return (round(float(axis[0]), 3), round(float(axis[1]), 3))


def test_plan_top_down_grasps_from_mask_uses_local_angles_for_bent_shape() -> None:
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[8:11, 2:13] = 1
    mask[10:21, 10:13] = 1
    cam_K = np.array([[100.0, 0.0, 12.0], [0.0, 100.0, 12.0], [0.0, 0.0, 1.0]])
    T_cam_world = _top_down_camera_transform()

    result = plan_top_down_grasps_from_mask(
        rgb=rgb,
        mask=mask,
        cam_K=cam_K,
        T_cam_world=T_cam_world,
        object_name="pliers",
        camera="top",
        max_grasps=12,
    )

    assert result.debug["axis_guided"] is True
    assert result.debug["yaw_mode"] == "local_cross_section"
    primary_axes = [_canonical_opening_axis(result.candidates[idx].rpy[2]) for idx in range(0, len(result.candidates), 4)]
    assert len(primary_axes) >= 2
    assert len(set(primary_axes)) > 1
    assert result.overlay_jpeg is not None
    assert len(result.overlay_jpeg) > 0


def test_plan_top_down_grasps_from_mask_isotropic_returns_center_only_yaws() -> None:
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    cam_K = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])
    T_cam_world = _top_down_camera_transform()

    result = plan_top_down_grasps_from_mask(
        rgb=rgb,
        mask=mask,
        cam_K=cam_K,
        T_cam_world=T_cam_world,
        object_name="pliers",
        camera="top",
        max_grasps=10,
    )

    assert result.debug["backend"] == "2dgrasp"
    assert result.debug["axis_guided"] is False
    assert result.debug["yaw_mode"] == "center_only"
    assert result.debug["major_axis_ratio"] < 1.25
    assert len(result.candidates) == 4
    assert [cand.rpy[2] for cand in result.candidates] == [0.0, -180.0, 90.0, -90.0]
    xs = {cand.position[0] for cand in result.candidates}
    ys = {cand.position[1] for cand in result.candidates}
    zs = {cand.position[2] for cand in result.candidates}
    assert len(xs) == 1
    assert len(ys) == 1
    # Grasp z is the table plane plus ENPIRE_2D_GRASP_Z_OFFSET_M (default 0.0),
    # resolved at import time. Assert against the resolved constant rather than a
    # literal so the test tracks the configured offset instead of pinning one.
    assert zs == {round(float(_TWO_D_GRASP_PLANNER_Z_M), 5)}
    assert result.overlay_jpeg is not None
    assert len(result.overlay_jpeg) > 0


def test_two_d_grasp_z_offset_defaults_to_table_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset ENPIRE_2D_GRASP_Z_OFFSET_M grasps at the table plane, not above it.

    The offset IS the grasp z, not a clearance floor, so a non-zero default would
    silently lift every 2D grasp off flat objects. It is read at import time, so
    reload the module to exercise the environment contract.
    """
    import importlib

    from enpire.env.forge.cap.agent.tools import grasp_2d

    monkeypatch.delenv("ENPIRE_2D_GRASP_Z_OFFSET_M", raising=False)
    reloaded = importlib.reload(grasp_2d)
    assert reloaded._TWO_D_GRASP_Z_OFFSET_M == 0.0
    assert reloaded._TWO_D_GRASP_PLANNER_Z_M == pytest.approx(float(TABLE_SURFACE_Z_M))

    monkeypatch.setenv("ENPIRE_2D_GRASP_Z_OFFSET_M", "0.03")
    reloaded = importlib.reload(grasp_2d)
    assert reloaded._TWO_D_GRASP_Z_OFFSET_M == pytest.approx(0.03)
    assert reloaded._TWO_D_GRASP_PLANNER_Z_M == pytest.approx(
        float(TABLE_SURFACE_Z_M) + 0.03
    )

    # Leave the module in its default state for tests that import it afterwards.
    monkeypatch.delenv("ENPIRE_2D_GRASP_Z_OFFSET_M", raising=False)
    importlib.reload(grasp_2d)


def test_sample_grasp_pose_2d_accepts_side_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = SampleGraspPose2DTool()
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    depth = np.ones((50, 50), dtype=np.float32) * 0.5
    cam_K = np.array([[100.0, 0, 25.0], [0, 100.0, 25.0], [0, 0, 1.0]], dtype=np.float64)
    T = _top_down_camera_transform()
    mask = np.zeros((50, 50), dtype=np.int32)
    mask[20:30, 15:35] = 1

    monkeypatch.setattr(tool, "_get_rgb_depth_intrinsics", lambda _cam: (rgb, depth, cam_K))
    monkeypatch.setattr(tool, "_get_extrinsics", lambda _cam: T)
    monkeypatch.setattr(tool, "_segment_object", lambda _rgb, _name: mask)

    result = tool.execute(object_name="pliers", camera="left")

    assert result.success is True
    assert len(result.data) > 0
