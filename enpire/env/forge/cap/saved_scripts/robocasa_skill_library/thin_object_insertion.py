# thin_object_insertion.py — insert a thin grasped object (straw) into a narrow opening (cup mouth)
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


CUP_QUERIES_V1 = (
    "glass cup",
    "drinking glass",
    "glass tumbler",
    "cup",
    "tall cup",
)
CUP_CAMERAS_V1 = ("top", "right", "wrist")


def detect_cup_mouth_v1(cameras=CUP_CAMERAS_V1):
    """Detect a cup. Cup mouth is approximated as cup_pos with z bumped to top.

    The detected pos is the mask centroid; we add a small upward offset so the
    insertion target is above the cup mouth rather than its centroid.
    """
    det = detect_object_v1(list(CUP_QUERIES_V1), cameras)
    pos = np.asarray(det["pos"], dtype=float).copy()
    # Heuristic: cup mouth is ~0.05m above mask centroid (depends on cup height).
    pos[2] += 0.04
    det = dict(det)
    det["mouth_pos"] = pos
    print(f"  cup mouth (heuristic): {fmt_xyz_v1(pos)}")
    return det


def insert_thin_object_v1(
    side,
    cup_mouth_pos,
    *,
    hover_clearance_m=0.18,
    insert_depth_m=0.04,
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
        (0.015, 0.015),
        (-0.015, -0.015),
    ),
    release_lift_m=0.20,
):
    """Insert the held object into the cup mouth.

    Hover above with extra Z, then descend with XY-candidate offsets, open
    gripper to release, retreat upward.
    Returns ``(success, info)``.
    """
    cup_mouth_pos = np.asarray(cup_mouth_pos, dtype=float)
    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)

    hover = cup_mouth_pos.copy()
    hover[2] += float(hover_clearance_m)
    print(
        "\n--- Thin insert: hover above cup ---\n"
        f"  cup_mouth={fmt_xyz_v1(cup_mouth_pos)} hover={fmt_xyz_v1(hover)}"
    )
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
        print(
            f"  hover xy=({dx:+.3f},{dy:+.3f}): status={hover_status} "
            f"target={fmt_xyz_v1(candidate)}"
        )
        if hover_status == "Success":
            break
    if hover_status != "Success":
        return False, {"phase": "hover", "status": hover_status}

    descend = cup_mouth_pos.copy()
    descend[2] -= float(insert_depth_m)
    descend_status = "Skipped"
    used_offset = (0.0, 0.0)
    for dx, dy in xy_offsets_m:
        candidate = descend.copy()
        candidate[0] += float(dx)
        candidate[1] += float(dy)
        result = freespace_move(
            right_target_pos=candidate.tolist(),
            side=side,
            gripper=0.0,
            auto_update_world=True,
        )
        descend_status = str(getattr(result, "status", result))
        print(
            f"  descend xy=({dx:+.3f},{dy:+.3f}): status={descend_status} "
            f"target={fmt_xyz_v1(candidate)}"
        )
        if descend_status == "Success":
            used_offset = (float(dx), float(dy))
            break

    if descend_status != "Success":
        # Brutal nudge straight down to seat the object as a fallback.
        nudge_brutal(
            side=side,
            delta_pos=[0.0, 0.0, -(float(hover_clearance_m) + float(insert_depth_m))],
            n_steps=10,
        )

    print("  open gripper to release")
    open_gripper(side)
    time.sleep(0.4)

    print(f"  retreat upward {float(release_lift_m):.2f}m")
    retreat = nudge_brutal(
        side=side,
        delta_pos=[0.0, 0.0, float(release_lift_m)],
        n_steps=8,
    )
    retreat_ok = bool(getattr(retreat, "success", False))

    return True, {
        "phase": "done",
        "used_offset_xy": list(used_offset),
        "retreat_success": retreat_ok,
    }


def show_cup_debug_markers_v1(cup_mouth_pos):
    show_debug_markers_v1(
        [
            debug_marker_v1(
                "cup_mouth",
                np.asarray(cup_mouth_pos, dtype=float),
                (96, 255, 192),
                radius=0.018,
            ),
        ]
    )
