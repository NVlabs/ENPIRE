# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# coffee_machine_control.py — PrepareCoffee dispenser placement and start button press
from skill_library.namespace import *  # noqa: F401, F403

import time

import numpy as np

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.detection import (
    debug_marker_v1,
    detect_object_v1,
    detect_sam3_depth_candidates_v1,
    filter_candidates_near_v1,
    show_debug_markers_v1,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_geometry import (
    fmt_xyz_v1,
    normalize_v1,
)


COFFEE_MACHINE_QUERIES_V1 = (
    "coffee machine",
    "espresso machine",
    "coffee maker",
    "coffee brewer",
)
COFFEE_DISPENSER_QUERIES_V1 = (
    "coffee machine dispenser",
    "coffee dispenser",
    "coffee machine spout",
    "coffee machine nozzle",
    "coffee machine outlet",
)
COFFEE_BUTTON_QUERIES_V1 = (
    "coffee machine start button",
    "coffee machine button",
    "start button on coffee machine",
    "coffee machine power button",
    "coffee machine on button",
)
COFFEE_CAMERAS_V1 = ("top", "right", "wrist")
COFFEE_BUTTON_CAMERAS_V1 = ("right", "wrist", "top")


def detect_coffee_machine_v1(cameras=COFFEE_CAMERAS_V1):
    """Detect the coffee machine bulk position from external cameras."""
    return detect_object_v1(list(COFFEE_MACHINE_QUERIES_V1), cameras)


def detect_coffee_dispenser_v1(
    coffee_pos,
    *,
    cameras=COFFEE_CAMERAS_V1,
    max_dist_from_machine_m=0.30,
    score_threshold=0.20,
    cradle_drop_m=0.06,
):
    """Detect the coffee dispenser cradle (where the mug sits).

    Strategy: try SAM3 for dispenser-class queries near the coffee machine, then
    fall back to a heuristic offset below the machine center if nothing matches.
    Returns a dict with 'pos', 'camera', 'score', 'mode', 'query'.
    """
    coffee_pos = np.asarray(coffee_pos, dtype=float)
    try:
        raw = detect_sam3_depth_candidates_v1(
            list(COFFEE_DISPENSER_QUERIES_V1),
            cameras,
            score_threshold=score_threshold,
            max_results=12,
            min_depth_m=0.10,
            min_valid_points=20,
        )
        near = filter_candidates_near_v1(
            raw, coffee_pos, max_dist_from_machine_m, label="coffee_machine"
        )
        # Cradle is the lowest-Z dispenser-like object near the coffee machine.
        near.sort(key=lambda d: float(d["pos"][2]))
        chosen = near[0]
        chosen = dict(chosen)
        chosen["mode"] = "sam3_dispenser"
        print(
            f"  coffee dispenser via SAM3: {chosen['query']!r} from "
            f"{chosen['camera']} score={chosen['score']:.3f} "
            f"pos={fmt_xyz_v1(chosen['pos'])}"
        )
        return chosen
    except RuntimeError as exc:
        print(f"  SAM3 dispenser detection failed ({exc}); using heuristic")

    cradle = coffee_pos.copy()
    cradle[2] -= float(cradle_drop_m)
    chosen = {
        "pos": cradle,
        "camera": "heuristic",
        "score": 0.0,
        "query": "coffee_machine_center_minus_z",
        "mode": "heuristic",
    }
    print(
        f"  coffee dispenser heuristic: drop {float(cradle_drop_m):.3f}m below "
        f"machine center -> pos={fmt_xyz_v1(cradle)}"
    )
    return chosen


def detect_coffee_start_button_v1(
    coffee_pos,
    *,
    cameras=COFFEE_BUTTON_CAMERAS_V1,
    max_dist_from_machine_m=0.30,
    score_threshold=0.15,
    max_results=20,
    button_above_dispenser_m=None,
):
    """Detect the start button near the coffee machine.

    Returns a dict with 'pos', 'camera', 'score', 'query', plus a 'candidates'
    list ordered by descending score for fallback retries.
    """
    coffee_pos = np.asarray(coffee_pos, dtype=float)
    try:
        raw = detect_sam3_depth_candidates_v1(
            list(COFFEE_BUTTON_QUERIES_V1),
            cameras,
            score_threshold=score_threshold,
            max_results=max_results,
            min_depth_m=0.10,
            min_valid_points=15,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not detect coffee start button via SAM3: {exc}"
        )
    near = filter_candidates_near_v1(
        raw,
        coffee_pos,
        max_dist_from_machine_m,
        label="coffee_machine",
    )
    if button_above_dispenser_m is not None:
        z_floor = float(coffee_pos[2]) + float(button_above_dispenser_m)
        higher = [d for d in near if float(d["pos"][2]) >= z_floor]
        if higher:
            near = higher
    near.sort(key=lambda d: float(d["score"]), reverse=True)
    chosen = dict(near[0])
    chosen["candidates"] = near
    print(
        f"  coffee start button: {chosen['query']!r} from {chosen['camera']} "
        f"score={chosen['score']:.3f} pos={fmt_xyz_v1(chosen['pos'])} "
        f"({len(near)} candidates)"
    )
    return chosen


def place_mug_under_dispenser_v1(
    side,
    dispenser_pos,
    *,
    hover_clearance_m=0.10,
    lower_offset_m=0.02,
    xy_offsets_m=(
        (0.0, 0.0),
        (0.02, 0.0),
        (-0.02, 0.0),
        (0.0, 0.02),
        (0.0, -0.02),
        (0.03, 0.03),
        (-0.03, -0.03),
    ),
    retreat_clearance_m=0.20,
):
    """Place a grasped mug under the dispenser cradle.

    Hover above the cradle, lower with XY-candidate offsets, open gripper,
    retreat upward to satisfy gripper-far check (default 0.15 m).
    Returns ``(success, info)``.
    """
    dispenser_pos = np.asarray(dispenser_pos, dtype=float)
    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)

    hover_pos = dispenser_pos.copy()
    hover_pos[2] += float(hover_clearance_m)
    print(
        "\n--- Coffee place: hover above dispenser cradle ---\n"
        f"  dispenser={fmt_xyz_v1(dispenser_pos)} hover={fmt_xyz_v1(hover_pos)}"
    )
    hover_result = freespace_move(
        right_target_pos=hover_pos.tolist(),
        right_target_quat=current_quat.tolist(),
        side=side,
        gripper=0.1,
        auto_update_world=True,
    )
    hover_status = str(getattr(hover_result, "status", hover_result))
    print(f"  hover status={hover_status}")
    if hover_status != "Success":
        return False, {
            "phase": "hover",
            "status": hover_status,
            "dispenser_pos": fmt_xyz_v1(dispenser_pos),
            "hover_pos": fmt_xyz_v1(hover_pos),
        }

    lower_status = "Skipped"
    used_offset_xy = (0.0, 0.0)
    used_z = float(dispenser_pos[2]) + float(lower_offset_m)
    for dx, dy in xy_offsets_m:
        candidate = dispenser_pos.copy()
        candidate[0] += float(dx)
        candidate[1] += float(dy)
        candidate[2] = used_z
        result = freespace_move(
            right_target_pos=candidate.tolist(),
            side=side,
            gripper=0.1,
            auto_update_world=True,
        )
        lower_status = str(getattr(result, "status", result))
        print(
            f"  lower xy=({dx:+.3f},{dy:+.3f}): status={lower_status} "
            f"target={fmt_xyz_v1(candidate)}"
        )
        if lower_status == "Success":
            used_offset_xy = (float(dx), float(dy))
            break

    if lower_status != "Success":
        # Final attempt: brutal nudge straight down from current EE pos.
        nudge_result = nudge_brutal(
            side=side,
            delta_pos=[0.0, 0.0, -float(hover_clearance_m) + float(lower_offset_m)],
            n_steps=10,
        )
        nudge_ok = bool(getattr(nudge_result, "success", False))
        print(f"  nudge-down fallback: success={nudge_ok}")
        if not nudge_ok:
            return False, {
                "phase": "lower",
                "status": lower_status,
                "used_offset_xy": list(used_offset_xy),
                "used_z": float(used_z),
            }

    print("  open gripper to release mug")
    open_gripper(side)
    time.sleep(0.4)

    print("  retreat upward to clear mug")
    retreat = nudge_brutal(
        side=side,
        delta_pos=[0.0, 0.0, float(retreat_clearance_m)],
        n_steps=8,
    )
    retreat_ok = bool(getattr(retreat, "success", False))
    print(f"  retreat success={retreat_ok}")

    return True, {
        "phase": "done",
        "used_offset_xy": list(used_offset_xy),
        "used_z": float(used_z),
        "retreat_success": retreat_ok,
    }


def press_coffee_start_v1(
    side,
    button_pos,
    *,
    hover_clearance_m=0.06,
    press_depth_m=0.03,
    n_repeats=2,
    n_press_steps=8,
    xy_offsets_m=(
        (0.0, 0.0),
        (0.005, 0.0),
        (-0.005, 0.0),
        (0.0, 0.005),
        (0.0, -0.005),
        (0.01, 0.0),
        (-0.01, 0.0),
        (0.0, 0.01),
        (0.0, -0.01),
    ),
    retreat_distance_m=0.20,
):
    """Press the coffee machine start button by hovering, closing gripper,
    nudging straight down, and retreating clear.

    Returns ``(pressed, info)``.
    """
    button_pos = np.asarray(button_pos, dtype=float)
    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)

    hover_pos = button_pos.copy()
    hover_pos[2] += float(hover_clearance_m)
    print(
        "\n--- Coffee press: hover above button ---\n"
        f"  button={fmt_xyz_v1(button_pos)} hover={fmt_xyz_v1(hover_pos)}"
    )
    hover_status = "Skipped"
    used_offset_xy = (0.0, 0.0)
    for dx, dy in xy_offsets_m:
        candidate = hover_pos.copy()
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
        print(
            f"  press hover xy=({dx:+.3f},{dy:+.3f}): status={hover_status} "
            f"target={fmt_xyz_v1(candidate)}"
        )
        if hover_status == "Success":
            used_offset_xy = (float(dx), float(dy))
            break
    if hover_status != "Success":
        return False, {"phase": "hover", "status": hover_status}

    print("  close gripper for press")
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
        print(
            f"  press[{press_idx + 1}/{int(n_repeats)}]: success={ok} "
            f"depth={float(press_depth_m):.3f}m"
        )
        time.sleep(0.15)

    print("  retreat clear of button")
    retreat = nudge_brutal(
        side=side,
        delta_pos=[0.0, 0.0, float(retreat_distance_m)],
        n_steps=8,
    )
    retreat_ok = bool(getattr(retreat, "success", False))
    print(f"  retreat success={retreat_ok}")

    return True, {
        "phase": "done",
        "used_offset_xy": list(used_offset_xy),
        "press_results": press_results,
        "retreat_success": retreat_ok,
    }


def show_coffee_debug_markers_v1(
    *,
    coffee_pos=None,
    dispenser_pos=None,
    button_pos=None,
):
    markers = []
    if coffee_pos is not None:
        markers.append(
            debug_marker_v1(
                "coffee_machine",
                np.asarray(coffee_pos, dtype=float),
                (180, 96, 255),
                radius=0.04,
            )
        )
    if dispenser_pos is not None:
        markers.append(
            debug_marker_v1(
                "coffee_dispenser",
                np.asarray(dispenser_pos, dtype=float),
                (96, 192, 255),
                radius=0.025,
            )
        )
    if button_pos is not None:
        markers.append(
            debug_marker_v1(
                "coffee_start_button",
                np.asarray(button_pos, dtype=float),
                (255, 180, 64),
                radius=0.02,
            )
        )
    show_debug_markers_v1(markers)


_ = normalize_v1  # keep import alive for future helpers
