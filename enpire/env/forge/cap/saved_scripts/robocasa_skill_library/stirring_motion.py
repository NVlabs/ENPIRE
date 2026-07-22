# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# stirring_motion.py — circular stirring motion for spatula inside pot
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


POT_QUERIES_V1 = (
    "pot",
    "pot on stove",
    "cooking pot",
    "saucepan",
    "stockpot",
)
POT_CAMERAS_V1 = ("top", "wrist")


def detect_pot_v1(cameras=POT_CAMERAS_V1):
    """Detect pot centroid + interior anchor for stirring motion."""
    det = detect_object_v1(list(POT_QUERIES_V1), cameras)
    interior = np.asarray(det["pos"], dtype=float).copy()
    # Pot centroid is roughly the rim center; pot interior is slightly below.
    interior[2] -= 0.02
    out = dict(det)
    out["interior_pos"] = interior
    print(f"  pot interior (heuristic): {fmt_xyz_v1(interior)}")
    return out


def plan_stir_circle_v1(
    pot_interior_pos,
    *,
    radius_m=0.05,
    n_points=8,
    n_revolutions=3,
    descent_offset_m=0.04,
):
    """Generate XY waypoints around a circle inside the pot, depth = interior_z - descent_offset."""
    interior = np.asarray(pot_interior_pos, dtype=float)
    cx, cy, base_z = float(interior[0]), float(interior[1]), float(interior[2])
    z = base_z - float(descent_offset_m)
    waypoints = []
    total = int(n_points) * int(n_revolutions)
    for i in range(total):
        theta = 2.0 * np.pi * float(i) / float(n_points)
        x = cx + float(radius_m) * np.cos(theta)
        y = cy + float(radius_m) * np.sin(theta)
        waypoints.append(np.array([x, y, z], dtype=float))
    return waypoints


def execute_stir_v1(
    side,
    waypoints,
    *,
    n_steps_per_segment=3,
    poll_seconds=0.10,
    success_ticks_required=5,
    early_exit_on_success=True,
):
    """Drive EE through stir waypoints. Polls task_info each segment."""
    success_ticks = 0
    for idx, target in enumerate(waypoints):
        target = np.asarray(target, dtype=float)
        current = np.asarray(get_robot_state().arms[side].ee_pos, dtype=float)
        delta = target - current
        result = nudge_brutal(side=side, delta_pos=delta.tolist(), n_steps=int(n_steps_per_segment))
        ok = bool(getattr(result, "success", False))
        if idx % 4 == 0:
            print(
                f"  stir wp[{idx + 1}/{len(waypoints)}] target={fmt_xyz_v1(target)} "
                f"delta={fmt_xyz_v1(delta)} success={ok}"
            )
        time.sleep(float(poll_seconds))
        info = get_task_info()
        if info.get("success", False):
            success_ticks += 1
            if early_exit_on_success and success_ticks >= int(success_ticks_required):
                print(f"  task_success after {success_ticks} success ticks (wp {idx + 1})")
                return True, {"stir_steps": idx + 1, "success_ticks": success_ticks}

    info = get_task_info()
    return bool(info.get("success", False)), {
        "stir_steps": len(waypoints),
        "success_ticks": success_ticks,
    }


def show_stir_debug_markers_v1(pot_interior_pos, waypoints=None):
    markers = [
        debug_marker_v1(
            "pot_interior",
            np.asarray(pot_interior_pos, dtype=float),
            (255, 140, 64),
            radius=0.030,
        )
    ]
    for i, wp in enumerate(waypoints or []):
        if i % 3 != 0:
            continue
        markers.append(
            debug_marker_v1(
                f"stir_wp_{i}",
                np.asarray(wp, dtype=float),
                (96, 255, 96),
                radius=0.010,
            )
        )
    show_debug_markers_v1(markers)
