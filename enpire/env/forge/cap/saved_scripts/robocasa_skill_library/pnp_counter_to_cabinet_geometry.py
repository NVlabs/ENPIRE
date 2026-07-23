# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pnp_counter_to_cabinet_geometry.py — PickPlaceCounterToCabinet geometry helpers
import numpy as np
from scipy.spatial.transform import Rotation as R
from skill_library.namespace import *  # noqa: F401, F403


def camera_surface_normal_from_cameras_v1(
    anchor_pos,
    *,
    side,
    label="target",
    snap_axis=True,
    face_robot=True,
):
    top_pos = np.asarray(get_camera_extrinsics("top")["position"], dtype=float)
    right_pos = np.asarray(get_camera_extrinsics("right")["position"], dtype=float)
    camera_vector_xy = right_pos - top_pos
    camera_vector_xy[2] = 0.0
    raw_normal = xy_cw_vertical_v1(camera_vector_xy)
    if float(np.linalg.norm(raw_normal[:2])) < 1e-6:
        raise RuntimeError("camera-derived surface normal is degenerate")

    surface_normal = snap_xy_axis_v1(raw_normal) if snap_axis else raw_normal
    if face_robot:
        ee_now = np.asarray(get_robot_state().arms[side].ee_pos, dtype=float)
        to_robot = ee_now - np.asarray(anchor_pos, dtype=float)
        to_robot[2] = 0.0
        if float(np.dot(surface_normal, to_robot)) < 0.0:
            surface_normal = -surface_normal

    print(f"  {label}_top_camera_pos={fmt_xyz_v1(top_pos)}")
    print(f"  {label}_right_camera_pos={fmt_xyz_v1(right_pos)}")
    print(f"  {label}_camera_vector_xy={fmt_xyz_v1(camera_vector_xy)}")
    print(f"  {label}_raw_surface_normal={fmt_xyz_v1(raw_normal)}")
    print(f"  {label}_axis_surface_normal={fmt_xyz_v1(surface_normal)}")
    return surface_normal


def cabinet_surface_normal_from_cameras_v1(cabinet_pos, *, side):
    return camera_surface_normal_from_cameras_v1(
        cabinet_pos,
        side=side,
        label="cabinet",
    )


def make_cabinet_press_quat_v1(surface_normal, roll_deg=0.0):
    z_axis = normalize_v1(-np.asarray(surface_normal, dtype=float))
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    x_axis = normalize_v1(world_up - z_axis * np.dot(world_up, z_axis))
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        x_axis = normalize_v1(x_axis - z_axis * np.dot(x_axis, z_axis))
    y_axis = normalize_v1(np.cross(z_axis, x_axis))
    x_axis = normalize_v1(np.cross(y_axis, z_axis))
    rot = R.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))
    if abs(float(roll_deg)) > 1e-6:
        rot = R.from_rotvec(z_axis * np.radians(float(roll_deg))) * rot
    return rot.as_quat()


def topdown_grasp_quat_candidates_from_axes_v1(long_axis_world, short_axis_world):
    long_axis = normalize_v1(np.asarray(long_axis_world, dtype=float).copy())
    short_axis = normalize_v1(np.asarray(short_axis_world, dtype=float).copy())
    long_axis[2] = 0.0
    short_axis[2] = 0.0
    long_axis = normalize_v1(long_axis)
    short_axis = normalize_v1(short_axis)
    if float(np.linalg.norm(long_axis[:2])) < 1e-6:
        raise RuntimeError("mask long axis is degenerate in XY")
    if float(np.linalg.norm(short_axis[:2])) < 1e-6:
        short_axis = normalize_v1(np.array([-long_axis[1], long_axis[0], 0.0]))

    z_down = np.array([0.0, 0.0, -1.0], dtype=float)
    candidates = []

    quat = quat_from_gripper_axes_v1(
        x_axis_world=long_axis,
        y_axis_world=short_axis,
        z_axis_world=z_down,
        prefer_y_axis=True,
    )
    candidates.append(("mask_long_x_short_y_down", quat))

    quat = quat_from_gripper_axes_v1(
        x_axis_world=short_axis,
        y_axis_world=long_axis,
        z_axis_world=z_down,
        prefer_y_axis=True,
    )
    candidates.append(("mask_short_x_long_y_down", quat))

    candidates.append(
        (
            "mask_long_x_short_y_down_flip",
            (R.from_rotvec(z_down * np.pi) * R.from_quat(candidates[0][1])).as_quat(),
        )
    )
    candidates.append(
        (
            "mask_short_x_long_y_down_flip",
            (R.from_rotvec(z_down * np.pi) * R.from_quat(candidates[1][1])).as_quat(),
        )
    )
    return candidates


def quat_from_gripper_axes_v1(
    *,
    x_axis_world,
    y_axis_world=None,
    z_axis_world,
    prefer_y_axis=False,
):
    z_axis = normalize_v1(np.asarray(z_axis_world, dtype=float))
    if float(np.linalg.norm(z_axis)) < 1e-6:
        raise RuntimeError("cannot build quat from zero z axis")

    if prefer_y_axis and y_axis_world is not None:
        y_axis = np.asarray(y_axis_world, dtype=float)
        y_axis = normalize_v1(y_axis - z_axis * np.dot(y_axis, z_axis))
        if float(np.linalg.norm(y_axis)) < 1e-6:
            y_axis = None
        else:
            x_axis = normalize_v1(np.cross(y_axis, z_axis))
            y_axis = normalize_v1(np.cross(z_axis, x_axis))
            return R.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat()

    x_axis = np.asarray(x_axis_world, dtype=float)
    x_axis = normalize_v1(x_axis - z_axis * np.dot(x_axis, z_axis))
    if float(np.linalg.norm(x_axis)) < 1e-6:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(helper, z_axis))) > 0.95:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = normalize_v1(helper - z_axis * np.dot(helper, z_axis))
    y_axis = normalize_v1(np.cross(z_axis, x_axis))
    x_axis = normalize_v1(np.cross(y_axis, z_axis))
    return R.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat()


def quat_from_gripper_direction_v1(direction_world):
    z_axis = normalize_v1(np.asarray(direction_world, dtype=float))
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(z_axis, helper))) > 0.95:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = normalize_v1(np.cross(helper, z_axis))
    y_axis = normalize_v1(np.cross(z_axis, x_axis))
    return R.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat()


def xy_cw_vertical_v1(v):
    v = np.asarray(v, dtype=float).copy()
    v[2] = 0.0
    return normalize_v1(np.array([v[1], -v[0], 0.0], dtype=float))


def snap_xy_axis_v1(v):
    v = np.asarray(v, dtype=float).copy()
    v[2] = 0.0
    if float(np.linalg.norm(v[:2])) < 1e-6:
        raise RuntimeError("cannot snap zero XY vector to axis")
    if abs(float(v[0])) >= abs(float(v[1])):
        return np.array([1.0 if v[0] >= 0.0 else -1.0, 0.0, 0.0], dtype=float)
    return np.array([0.0, 1.0 if v[1] >= 0.0 else -1.0, 0.0], dtype=float)


def normalize_v1(v, eps=1e-8):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros_like(v)


def fmt_xyz_v1(v, ndigits=3):
    return [round(float(x), ndigits) for x in np.asarray(v, dtype=float)]
