# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# stove_knob_common.py - shared TurnOffStove constants and geometry helpers
from skill_library.namespace import *  # noqa: F401, F403

import numpy as np
from scipy.spatial.transform import Rotation as R


STOVE_KNOB_QUERIES_V1 = (
    "stove knobs",
    "all stove knobs",
    "stove control knobs",
    "burner control knobs",
    "stove knob",
    "burner knob",
    "stove burner knob",
)
STOVE_KNOB_CAMERAS_V1 = ("top", "right")
STOVE_FLAT_KNOB_Z_P90_P10_M_V1 = 0.020
STOVE_GAP_INSERT_RATIO_V1 = 1.8
STOVE_HOVER_POS_TOL_M_V1 = 0.040
STOVE_HOVER_ROT_TOL_DEG_V1 = 15.0
STOVE_HOVER_POSE_RETRIES_V1 = 2
STOVE_APPROACH_ROLL_CANDIDATES_DEG_V1 = (
    0.0,
    45.0,
    -45.0,
    90.0,
    -90.0,
    135.0,
    -135.0,
    180.0,
)
STOVE_INDEX_ORDERS_V1 = {
    "4_standard": ("front_left", "rear_left", "rear_right", "front_right"),
    "4_center": ("front_left", "front_center", "rear_center", "front_right"),
    "5_standard": ("front_left", "rear_left", "center", "rear_right", "front_right"),
    "5_center": (
        "front_left",
        "front_center",
        "rear_center",
        "rear_right",
        "front_right",
    ),
    "6_standard": (
        "front_left",
        "rear_left",
        "front_center",
        "rear_center",
        "rear_right",
        "front_right",
    ),
    "6_aux_left": (
        "temp",
        "time",
        "front_left",
        "rear_left",
        "rear_right",
        "front_right",
    ),
    "7_aux_left": (
        "temp",
        "time",
        "front_left",
        "rear_left",
        "center",
        "rear_right",
        "front_right",
    ),
}
STOVE_REAL_ORDERS_V1 = (
    ("front_left", "rear_left", "rear_right", "front_right"),
    ("front_left", "rear_left", "rear_right", "front_right", "time", "temp"),
    ("front_left", "rear_left", "time", "temp", "rear_right", "front_right"),
    ("front_left", "rear_left", "temp", "time", "rear_right", "front_right"),
    (
        "front_left",
        "rear_left",
        "center",
        "time",
        "temp",
        "rear_right",
        "front_right",
    ),
    (
        "temp",
        "time",
        "front_left",
        "rear_left",
        "center",
        "rear_right",
        "front_right",
    ),
    (
        "front_left",
        "rear_left",
        "front_center",
        "temp",
        "time",
        "rear_center",
        "rear_right",
        "front_right",
    ),
    (
        "front_left",
        "rear_left",
        "front_center",
        "time",
        "temp",
        "rear_center",
        "rear_right",
        "front_right",
    ),
    (
        "front_left",
        "rear_left",
        "temp",
        "time",
        "front_center",
        "rear_center",
        "rear_right",
        "front_right",
    ),
    (
        "front_left",
        "rear_left",
        "time",
        "temp",
        "front_center",
        "rear_center",
        "rear_right",
        "front_right",
    ),
)
STOVE_TARGET_SIGNED_RANK_CANDIDATES_V1 = {
    "front_left": (1, 3),
    "rear_left": (2, 4),
    "front_center": (3, -4),
    "center": (3, -3),
    "rear_center": (-3,),
    "rear_right": (-2, 3),
    "front_right": (-1, -3),
}


def camera_transform_v1(camera):
    fx, fy, cx, cy = [float(x) for x in get_camera_intrinsics(camera)]
    extr = get_camera_extrinsics(camera)
    rot = np.asarray(extr["rotation"], dtype=float).reshape(3, 3)
    pos = np.asarray(extr["position"], dtype=float)
    transform = np.eye(4, dtype=float)
    if extr.get("needs_optical_flip", True):
        transform[:3, :3] = rot @ np.diag([-1.0, -1.0, 1.0])
    else:
        transform[:3, :3] = rot
    transform[:3, 3] = pos
    return transform, fx, fy, cx, cy


def quat_angle_error_deg_v1(q_a, q_b):
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    norm_a = float(np.linalg.norm(q_a))
    norm_b = float(np.linalg.norm(q_b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return float("inf")
    q_a = q_a / norm_a
    q_b = q_b / norm_b
    dot = abs(float(np.dot(q_a, q_b)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def backproject_mask_with_pixel_v1(
    mask,
    depth,
    transform,
    fx,
    fy,
    cx,
    cy,
    *,
    lower_middle_z=False,
):
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0.10)
    n_valid = int(valid.sum())
    if n_valid < 20:
        return None
    vs, us = np.where(valid)
    zs = depth[valid].astype(float)
    xs = (us.astype(float) - float(cx)) * zs / float(fx)
    ys = (vs.astype(float) - float(cy)) * zs / float(fy)
    pts_cam = np.stack([xs, ys, zs], axis=1)
    centroid_cam = np.median(pts_cam, axis=0)
    pts_world = (transform[:3, :3] @ pts_cam.T).T + transform[:3, 3]
    world_z = pts_world[:, 2]
    world_z_p90_p10 = float(
        np.percentile(world_z, 90.0) - np.percentile(world_z, 10.0)
    )
    pos = transform[:3, :3] @ centroid_cam + transform[:3, 3]
    pos_mode = "mask_median"
    z_band_center_m = None
    z_band_n = 0
    if lower_middle_z and world_z_p90_p10 > STOVE_FLAT_KNOB_Z_P90_P10_M_V1:
        z_band_center_m = float(np.percentile(world_z, 35.0))
        band_mask = np.abs(world_z - z_band_center_m) <= 0.010
        if int(band_mask.sum()) < 10:
            z_low = float(np.percentile(world_z, 25.0))
            z_high = float(np.percentile(world_z, 50.0))
            band_mask = (world_z >= z_low) & (world_z <= z_high)
        if int(band_mask.sum()) >= 10:
            pos = np.array(
                [
                    float(np.median(pts_world[:, 0])),
                    float(np.median(pts_world[:, 1])),
                    float(np.median(world_z[band_mask])),
                ],
                dtype=float,
            )
            pos_mode = "lower_middle_z_band"
            z_band_n = int(band_mask.sum())
    return {
        "pos": np.asarray(pos, dtype=float),
        "pixel_uv": [
            float(np.median(us.astype(float))),
            float(np.median(vs.astype(float))),
        ],
        "median_depth": float(np.median(zs)),
        "n_valid": n_valid,
        "world_z_range_m": float(np.max(world_z) - np.min(world_z)),
        "world_z_p90_p10_m": world_z_p90_p10,
        "world_z_std_m": float(np.std(world_z)),
        "pos_mode": pos_mode,
        "z_band_center_m": z_band_center_m,
        "z_band_n": z_band_n,
    }


def make_gripper_z_quat_v1(z_axis, roll_deg=0.0):
    z_axis = normalize_v1(np.asarray(z_axis, dtype=float))
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


def xy_cw_vertical_v1(v):
    v = np.asarray(v, dtype=float).copy()
    v[2] = 0.0
    return normalize_v1(np.array([v[1], -v[0], 0.0], dtype=float))


def normalize_v1(v, eps=1e-8):
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > eps else np.zeros_like(v)


def fmt_xyz_v1(v, ndigits=3):
    return [round(float(x), ndigits) for x in np.asarray(v, dtype=float)]
