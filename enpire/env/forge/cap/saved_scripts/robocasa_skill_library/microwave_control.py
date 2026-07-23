# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# microwave_control.py — close microwave door + press start button
import time

import numpy as np
from skill_library.namespace import *  # noqa: F401, F403

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.detection import (
    debug_marker_v1,
    detect_object_v1,
    detect_sam3_depth_candidates_v1,
    filter_candidates_near_v1,
    show_debug_markers_v1,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_geometry import (
    fmt_xyz_v1,
)

MICROWAVE_QUERIES_V1 = ("microwave", "microwave oven", "open microwave")
MICROWAVE_DOOR_QUERIES_V1 = (
    "microwave door",
    "microwave open door",
    "open microwave door",
    "microwave door panel",
)
MICROWAVE_BUTTON_QUERIES_V1 = (
    "microwave start button",
    "microwave button",
    "start button on microwave",
    "microwave control panel button",
)
MICROWAVE_CAMERAS_V1 = ("top", "right", "wrist")


def detect_microwave_v1(cameras=MICROWAVE_CAMERAS_V1):
    return detect_object_v1(list(MICROWAVE_QUERIES_V1), cameras)


def detect_microwave_door_v1(microwave_pos, *, cameras=MICROWAVE_CAMERAS_V1):
    """Detect the microwave door panel position."""
    raw = detect_sam3_depth_candidates_v1(
        list(MICROWAVE_DOOR_QUERIES_V1),
        cameras,
        score_threshold=0.20,
        max_results=12,
        min_depth_m=0.10,
        min_valid_points=20,
    )
    near = filter_candidates_near_v1(raw, microwave_pos, 0.50, label="microwave")
    near.sort(key=lambda d: float(d["score"]), reverse=True)
    chosen = dict(near[0])
    print(
        f"  microwave door: {chosen['query']!r} from {chosen['camera']} "
        f"score={chosen['score']:.3f} pos={fmt_xyz_v1(chosen['pos'])}"
    )
    return chosen


def detect_microwave_start_button_v1(microwave_pos, *, cameras=("right", "wrist")):
    """Detect the microwave start button. Filter to right side of the panel."""
    try:
        raw = detect_sam3_depth_candidates_v1(
            list(MICROWAVE_BUTTON_QUERIES_V1),
            cameras,
            score_threshold=0.15,
            max_results=15,
            min_depth_m=0.10,
            min_valid_points=10,
        )
    except RuntimeError:
        # Fallback: heuristic — start button is typically to the right of the door
        # at roughly the same z as the microwave centroid.
        microwave_pos = np.asarray(microwave_pos, dtype=float)
        guess = microwave_pos.copy()
        guess[0] += 0.18  # right-side panel offset (x in camera frame)
        return {
            "pos": guess, "camera": "heuristic",
            "score": 0.0, "query": "heuristic_button",
        }
    near = filter_candidates_near_v1(raw, microwave_pos, 0.50, label="microwave")
    near.sort(key=lambda d: float(d["score"]), reverse=True)
    chosen = dict(near[0])
    chosen["candidates"] = near
    print(
        f"  microwave button: {chosen['query']!r} from {chosen['camera']} "
        f"score={chosen['score']:.3f} pos={fmt_xyz_v1(chosen['pos'])}"
    )
    return chosen


def close_microwave_door_v1(
    side,
    door_pos,
    microwave_pos,
    *,
    hover_clearance_m=0.12,
    push_distance_m=0.30,
    n_push_steps=20,
    retreat_distance_m=0.20,
):
    """Close the microwave door by pushing from the open door position toward
    the microwave body. Returns ``(success, info)``.
    """
    door_pos = np.asarray(door_pos, dtype=float)
    microwave_pos = np.asarray(microwave_pos, dtype=float)
    push_dir = microwave_pos - door_pos
    push_dir[2] = 0.0
    norm = float(np.linalg.norm(push_dir))
    if norm < 1e-3:
        push_dir = np.array([0.0, 1.0, 0.0])
    else:
        push_dir = push_dir / norm
    print(
        f"  push direction (door->microwave): {fmt_xyz_v1(push_dir)} "
        f"distance={float(push_distance_m):.2f}m"
    )

    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)
    # Hover near door front face, slightly above
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
        # Try slightly higher hover
        hover[2] += 0.05
        result = freespace_move(
            right_target_pos=hover.tolist(),
            right_target_quat=current_quat.tolist(),
            side=side,
            gripper=0.0,
            auto_update_world=True,
        )
        hover_status = str(getattr(result, "status", result))
        print(f"  door hover (higher) status={hover_status}")
        if hover_status != "Success":
            return False, {"phase": "hover", "status": hover_status}

    push = nudge_brutal(
        side=side,
        delta_pos=(push_dir * float(push_distance_m)).tolist(),
        n_steps=int(n_push_steps),
    )
    push_ok = bool(getattr(push, "success", False))
    print(f"  door push: success={push_ok}")

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


def press_microwave_start_v1(
    side,
    button_pos,
    *,
    hover_clearance_m=0.06,
    press_depth_m=0.04,
    n_repeats=2,
    n_press_steps=10,
    xy_offsets_m=(
        (0.0, 0.0),
        (0.005, 0.0),
        (-0.005, 0.0),
        (0.0, 0.005),
        (0.0, -0.005),
        (0.01, 0.0),
        (-0.01, 0.0),
        (0.015, 0.015),
        (-0.015, -0.015),
    ),
    retreat_distance_m=0.20,
):
    """Press the microwave start button. Same pattern as coffee/kettle press."""
    button_pos = np.asarray(button_pos, dtype=float)
    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)

    hover = button_pos.copy()
    hover[2] += float(hover_clearance_m)
    hover_status = "Skipped"
    for dx, dy in xy_offsets_m:
        candidate = hover.copy()
        candidate[0] += float(dx)
        candidate[1] += float(dy)
        result = freespace_move(
            right_target_pos=candidate.tolist(),
            right_target_quat=current_quat.tolist(),
            side=side,
            gripper=0.0,
            auto_update_world=True,
        )
        hover_status = str(getattr(result, "status", result))
        print(f"  mw button hover xy=({dx:+.3f},{dy:+.3f}): status={hover_status}")
        if hover_status == "Success":
            break
    if hover_status != "Success":
        return False, {"phase": "hover", "status": hover_status}

    close_gripper(side)
    time.sleep(0.2)

    press_results = []
    for press_idx in range(int(n_repeats)):
        result = nudge_brutal(
            side=side,
            delta_pos=[0.0, 0.0, -float(press_depth_m)],
            n_steps=int(n_press_steps),
        )
        ok = bool(getattr(result, "success", False))
        press_results.append(ok)
        print(f"  mw press[{press_idx + 1}/{int(n_repeats)}]: success={ok}")
        time.sleep(0.15)

    retreat = nudge_brutal(
        side=side,
        delta_pos=[0.0, 0.0, float(retreat_distance_m)],
        n_steps=8,
    )
    return True, {
        "phase": "done",
        "press_results": press_results,
        "retreat_success": bool(getattr(retreat, "success", False)),
    }


def show_microwave_debug_markers_v1(*, microwave_pos=None, door_pos=None, button_pos=None):
    markers = []
    if microwave_pos is not None:
        markers.append(debug_marker_v1("microwave", np.asarray(microwave_pos, dtype=float), (180, 96, 255), radius=0.04))
    if door_pos is not None:
        markers.append(debug_marker_v1("microwave_door", np.asarray(door_pos, dtype=float), (96, 192, 255), radius=0.025))
    if button_pos is not None:
        markers.append(debug_marker_v1("microwave_button", np.asarray(button_pos, dtype=float), (255, 180, 64), radius=0.020))
    show_debug_markers_v1(markers)
