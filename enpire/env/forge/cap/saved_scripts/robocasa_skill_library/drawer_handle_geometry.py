# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# drawer_handle_geometry.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403

from enpire.env.forge.cap.agent.skill_registry import skill


def _normalize(v, eps=1e-8):
    import numpy as np

    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros_like(v)


def _make_press_quat(surface_normal, roll_deg=0.0):
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    z_axis = _normalize(-np.asarray(surface_normal, dtype=float))
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    x_axis = _normalize(world_up - z_axis * np.dot(world_up, z_axis))
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        x_axis = _normalize(x_axis - z_axis * np.dot(x_axis, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    rot = R.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))
    if abs(roll_deg) > 1e-6:
        rot = R.from_rotvec(z_axis * np.radians(roll_deg)) * rot
    return rot.as_quat()


def _make_surface_vertical_plane_quat(surface_normal):
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    z_axis = _normalize(-np.asarray(surface_normal, dtype=float))
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    y_axis = _normalize(world_up - z_axis * np.dot(world_up, z_axis))
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        y_axis = _normalize(y_axis - z_axis * np.dot(y_axis, z_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        x_axis = _normalize(x_axis - z_axis * np.dot(x_axis, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return R.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat()


def _xy_cw_vertical(v):
    import numpy as np

    v = np.asarray(v, dtype=float).copy()
    v[2] = 0.0
    return _normalize(np.array([v[1], -v[0], 0.0], dtype=float))


@skill
def drawer_handle_geometry_v1(target, rectangular_handle_z_drop_m=0.10):
    import numpy as np

    target_pos = np.asarray(target["pos"], dtype=float)
    target_size = np.asarray(target.get("size", [0.0, 0.0, 0.0]), dtype=float)
    target_geom_name = str(target.get("geom_name", ""))
    target_geom_type_name = str(target.get("geom_type_name", ""))
    surface_normal_hint = _normalize(np.asarray(target["surface_normal"], dtype=float))
    axis_world = _normalize(np.asarray(target["axis_world"], dtype=float))
    anchor_world = np.asarray(target["anchor_world"], dtype=float)
    desired_fraction = float(target.get("desired_fraction", 1.0))
    current_fraction = float(target.get("normalized_qpos", 0.0))
    standoff = max(float(target.get("recommended_standoff", 0.05)), 0.15)
    contact_offset = float(target.get("recommended_contact_offset", 0.015))
    travel_distance = float(target.get("recommended_travel_distance", 0.18))
    retreat_distance = float(target.get("recommended_retreat_distance", 0.10))

    surface_normal = surface_normal_hint.copy()
    surface_normal_source = "target_surface_normal"
    camera_vector_xy = np.zeros(3, dtype=float)
    try:
        top_extr = get_camera_extrinsics("top")
        right_extr = get_camera_extrinsics("right")
        top_pos = np.asarray(top_extr["position"], dtype=float)
        right_pos = np.asarray(right_extr["position"], dtype=float)
        camera_vector_xy = right_pos - top_pos
        camera_vector_xy[2] = 0.0
        camera_surface_normal = _xy_cw_vertical(camera_vector_xy)
        if np.linalg.norm(camera_surface_normal) > 1e-6:
            surface_normal = camera_surface_normal
            surface_normal_source = "top_to_right_xy_cw_vertical"
    except Exception:
        pass

    is_round_handle = target_geom_type_name in {
        "cylinder",
        "capsule",
        "sphere",
        "ellipsoid",
    }
    if not target_geom_type_name:
        is_round_handle = bool(
            target_size.size >= 3 and abs(float(target_size[2])) < 1e-6
        )

    handle_pos = np.asarray(target_pos, dtype=float).copy()

    approach_dir = _normalize(surface_normal)
    sign = 1.0 if desired_fraction >= current_fraction else -1.0
    pull_dir = _normalize(axis_world) * sign
    horizontal_pull = np.asarray(pull_dir, dtype=float).copy()
    horizontal_pull[2] = 0.0
    if np.linalg.norm(horizontal_pull) < 1e-6:
        horizontal_pull = np.asarray(axis_world, dtype=float).copy()
        horizontal_pull[2] = 0.0
    if np.linalg.norm(horizontal_pull) < 1e-6:
        horizontal_pull = np.array([1.0, 0.0, 0.0], dtype=float)
    horizontal_pull = _normalize(horizontal_pull)
    grasp_quat = _make_surface_vertical_plane_quat(surface_normal)

    return True, {
        "target_pos": target_pos,
        "target_size": target_size,
        "target_geom_name": target_geom_name,
        "target_geom_type_name": target_geom_type_name,
        "surface_normal": surface_normal,
        "surface_normal_source": surface_normal_source,
        "camera_vector_xy": camera_vector_xy,
        "surface_normal_hint": surface_normal_hint,
        "axis_world": axis_world,
        "anchor_world": anchor_world,
        "desired_fraction": desired_fraction,
        "current_fraction": current_fraction,
        "pull_sign_hint": sign,
        "handle_pos": handle_pos,
        "is_round_handle": is_round_handle,
        "grasp_quat": grasp_quat,
        "approach_dir": approach_dir,
        "horizontal_pull": horizontal_pull,
        "pull_dir": pull_dir,
        "standoff": standoff,
        "contact_offset": contact_offset,
        "travel_distance": travel_distance,
        "retreat_distance": retreat_distance,
    }
