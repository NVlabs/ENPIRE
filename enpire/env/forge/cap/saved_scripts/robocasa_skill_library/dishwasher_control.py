# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# dishwasher_control.py — push top rack in + close dishwasher door
from skill_library.namespace import *  # noqa: F401, F403

import time

import numpy as np

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.detection import (
    debug_marker_v1,
    detect_object_v1,
    show_debug_markers_v1,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_geometry import (
    fmt_xyz_v1,
)


DISHWASHER_QUERIES_V1 = ("dishwasher", "open dishwasher", "dishwasher with door open")
TOP_RACK_QUERIES_V1 = (
    "dishwasher top rack",
    "dishwasher rack",
    "top rack of dishwasher",
    "extended dishwasher rack",
)
DISHWASHER_DOOR_QUERIES_V1 = (
    "dishwasher door",
    "open dishwasher door",
    "dishwasher front panel",
)
DISHWASHER_CAMERAS_V1 = ("top", "right", "wrist")


def detect_dishwasher_v1(cameras=DISHWASHER_CAMERAS_V1):
    return detect_object_v1(list(DISHWASHER_QUERIES_V1), cameras)


def detect_top_rack_v1(cameras=DISHWASHER_CAMERAS_V1):
    """Top rack is typically the upper of the two horizontal slatted surfaces."""
    return detect_object_v1(list(TOP_RACK_QUERIES_V1), cameras)


def detect_dishwasher_door_v1(cameras=DISHWASHER_CAMERAS_V1):
    return detect_object_v1(list(DISHWASHER_DOOR_QUERIES_V1), cameras)


def push_top_rack_in_v1(
    side,
    rack_pos,
    dishwasher_pos,
    *,
    hover_clearance_m=0.10,
    push_distance_m=0.30,
    n_push_steps=18,
    retreat_distance_m=0.20,
):
    """Push the top rack horizontally toward the dishwasher body to retract it."""
    rack_pos = np.asarray(rack_pos, dtype=float)
    dishwasher_pos = np.asarray(dishwasher_pos, dtype=float)
    push_dir = dishwasher_pos - rack_pos
    push_dir[2] = 0.0
    norm = float(np.linalg.norm(push_dir))
    push_dir = push_dir / norm if norm > 1e-3 else np.array([0.0, 1.0, 0.0])

    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)
    hover = rack_pos.copy()
    hover[2] += float(hover_clearance_m)
    result = freespace_move(
        right_target_pos=hover.tolist(),
        right_target_quat=current_quat.tolist(),
        side=side,
        gripper=0.0,
        auto_update_world=True,
    )
    hover_status = str(getattr(result, "status", result))
    print(f"  rack hover status={hover_status}")
    if hover_status != "Success":
        return False, {"phase": "hover", "status": hover_status}

    push = nudge_brutal(
        side=side,
        delta_pos=(push_dir * float(push_distance_m)).tolist(),
        n_steps=int(n_push_steps),
    )
    push_ok = bool(getattr(push, "success", False))
    print(f"  rack push: dir={fmt_xyz_v1(push_dir)} success={push_ok}")

    retreat = nudge_brutal(
        side=side,
        delta_pos=[0.0, 0.0, float(retreat_distance_m)],
        n_steps=8,
    )
    return True, {
        "phase": "done",
        "push_success": push_ok,
        "retreat_success": bool(getattr(retreat, "success", False)),
    }


def close_dishwasher_door_v1(
    side,
    door_pos,
    dishwasher_pos,
    *,
    hover_clearance_m=0.12,
    push_distance_m=0.40,
    n_push_steps=25,
    retreat_distance_m=0.20,
):
    """Push the dishwasher door upward + inward to close it."""
    door_pos = np.asarray(door_pos, dtype=float)
    dishwasher_pos = np.asarray(dishwasher_pos, dtype=float)
    # Door rotates about its lower hinge. Push direction is dish_pos - door_pos
    # in xy plus a strong upward component.
    push_dir_xy = dishwasher_pos - door_pos
    push_dir_xy[2] = 0.0
    norm = float(np.linalg.norm(push_dir_xy))
    push_dir_xy = push_dir_xy / norm if norm > 1e-3 else np.array([0.0, 1.0, 0.0])
    push_vec = push_dir_xy * float(push_distance_m) + np.array([0.0, 0.0, 0.10])

    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)
    hover = door_pos.copy()
    hover[2] += float(hover_clearance_m)
    result = freespace_move(
        right_target_pos=hover.tolist(),
        right_target_quat=current_quat.tolist(),
        side=side,
        gripper=0.0,
        auto_update_world=True,
    )
    hover_status = str(getattr(result, "status", result))
    print(f"  door hover status={hover_status}")
    if hover_status != "Success":
        return False, {"phase": "hover", "status": hover_status}

    push = nudge_brutal(
        side=side,
        delta_pos=push_vec.tolist(),
        n_steps=int(n_push_steps),
    )
    push_ok = bool(getattr(push, "success", False))
    print(f"  door push: vec={fmt_xyz_v1(push_vec)} success={push_ok}")

    retreat = nudge_brutal(
        side=side,
        delta_pos=[0.0, 0.0, float(retreat_distance_m)],
        n_steps=8,
    )
    return True, {
        "phase": "done",
        "push_success": push_ok,
        "retreat_success": bool(getattr(retreat, "success", False)),
    }


def show_dishwasher_debug_markers_v1(*, dishwasher_pos=None, rack_pos=None, door_pos=None):
    markers = []
    if dishwasher_pos is not None:
        markers.append(debug_marker_v1("dishwasher", np.asarray(dishwasher_pos, dtype=float), (180, 96, 255), radius=0.04))
    if rack_pos is not None:
        markers.append(debug_marker_v1("dishwasher_top_rack", np.asarray(rack_pos, dtype=float), (96, 192, 255), radius=0.025))
    if door_pos is not None:
        markers.append(debug_marker_v1("dishwasher_door", np.asarray(door_pos, dtype=float), (255, 180, 64), radius=0.022))
    show_debug_markers_v1(markers)
