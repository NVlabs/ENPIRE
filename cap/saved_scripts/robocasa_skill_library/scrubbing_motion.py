# scrubbing_motion.py — sweep a grasped sponge across a cutting board surface
from skill_library.namespace import *  # noqa: F401, F403

import time

import numpy as np

from cap.saved_scripts.robocasa_skill_library.detection import (
    debug_marker_v1,
    detect_object_v1,
    show_debug_markers_v1,
)
from cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_geometry import (
    fmt_xyz_v1,
)


CUTTING_BOARD_QUERIES_V1 = (
    "cutting board",
    "wooden cutting board",
    "chopping board",
    "cutting board on counter",
)
CUTTING_BOARD_CAMERAS_V1 = ("top", "wrist")


def detect_cutting_board_v1(cameras=CUTTING_BOARD_CAMERAS_V1):
    """Detect cutting board centroid for sweep planning."""
    return detect_object_v1(list(CUTTING_BOARD_QUERIES_V1), cameras)


def plan_scrub_waypoints_v1(
    board_pos,
    *,
    n_contacts=6,
    span_m=0.12,
    contact_dx_m=0.025,
    contact_dy_m=0.025,
    descend_z_m=0.04,
    lift_z_m=0.015,
):
    """Generate (down, hold, lift, translate) waypoints in a zigzag pattern.

    Returns a list of dicts: {"action": "descend"|"lift"|"translate", "pos": ndarray}
    Each contact step descends descend_z_m below board_pos.z, then lifts lift_z_m
    before translating to the next contact location. The XY pattern alternates
    Y between -span/2 and +span/2 to maximize sweep_range while keeping each
    contact >= contact_dx_m apart from the previous (so success counter ticks).
    """
    board_pos = np.asarray(board_pos, dtype=float)
    base_z = float(board_pos[2])
    cx, cy = float(board_pos[0]), float(board_pos[1])

    # X positions evenly along [cx - span/2, cx + span/2]
    half = float(span_m) * 0.5
    n = max(2, int(n_contacts))
    xs = np.linspace(cx - half, cx + half, n)
    # Alternate Y for zigzag
    ys = np.array([cy - half * 0.4 if i % 2 == 0 else cy + half * 0.4 for i in range(n)])

    contact_z = base_z + float(lift_z_m) - float(descend_z_m)  # below board surface
    above_z = base_z + float(lift_z_m) + float(descend_z_m)

    waypoints = []
    for i in range(n):
        # 1) hover above contact i
        waypoints.append({
            "action": "hover",
            "pos": np.array([xs[i], ys[i], above_z], dtype=float),
            "label": f"hover_{i}",
        })
        # 2) descend to contact_z
        waypoints.append({
            "action": "descend",
            "pos": np.array([xs[i], ys[i], contact_z], dtype=float),
            "label": f"descend_{i}",
        })
        # 3) lift slightly
        waypoints.append({
            "action": "lift",
            "pos": np.array([xs[i], ys[i], above_z], dtype=float),
            "label": f"lift_{i}",
        })
    return waypoints


def execute_scrub_sweep_v1(
    side,
    waypoints,
    *,
    contacts_required=5,
    poll_seconds=0.10,
    early_exit_on_success=True,
    descend_n_steps=4,
    lift_n_steps=3,
    translate_n_steps=4,
):
    """Execute a sequence of scrub waypoints. Polls task_info between contacts
    and early-exits when success is reached.
    """
    contacts_made = 0
    for idx, wp in enumerate(waypoints):
        action = wp["action"]
        target = np.asarray(wp["pos"], dtype=float)
        label = wp.get("label", f"wp_{idx}")
        current = np.asarray(get_robot_state().arms[side].ee_pos, dtype=float)
        delta = target - current
        n_steps = (
            descend_n_steps if action == "descend"
            else lift_n_steps if action == "lift"
            else translate_n_steps
        )
        result = nudge_brutal(side=side, delta_pos=delta.tolist(), n_steps=int(n_steps))
        ok = bool(getattr(result, "success", False))
        print(
            f"  scrub {label}: action={action} delta={fmt_xyz_v1(delta)} "
            f"target={fmt_xyz_v1(target)} success={ok}"
        )
        if action == "descend":
            contacts_made += 1
            time.sleep(float(poll_seconds))
            info = get_task_info()
            print(
                f"    after contact #{contacts_made}: "
                f"reward={float(info.get('reward', 0.0)):.3f} "
                f"success={bool(info.get('success', False))}"
            )
            if early_exit_on_success and info.get("success", False):
                return True, {
                    "contacts_made": contacts_made,
                    "stopped_at": label,
                }
            if contacts_made >= int(contacts_required) and info.get("success", False):
                return True, {
                    "contacts_made": contacts_made,
                    "stopped_at": label,
                }

    info = get_task_info()
    return bool(info.get("success", False)), {
        "contacts_made": contacts_made,
        "stopped_at": None,
    }


def show_scrub_debug_markers_v1(board_pos, waypoints=None):
    markers = [
        debug_marker_v1(
            "cutting_board",
            np.asarray(board_pos, dtype=float),
            (180, 130, 64),
            radius=0.030,
        )
    ]
    for i, wp in enumerate(waypoints or []):
        if wp["action"] != "descend":
            continue
        markers.append(
            debug_marker_v1(
                f"scrub_contact_{i}",
                np.asarray(wp["pos"], dtype=float),
                (96, 255, 96),
                radius=0.012,
            )
        )
    show_debug_markers_v1(markers)
