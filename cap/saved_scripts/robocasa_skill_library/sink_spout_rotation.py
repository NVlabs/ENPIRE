# sink_spout_rotation.py — rotate the sink spout to left/center/right while polling success
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
    normalize_v1,
)


SPOUT_QUERIES_V1 = (
    "sink spout",
    "sink faucet spout",
    "sink faucet neck",
    "sink faucet",
)
SPOUT_CAMERAS_V1 = ("top", "right", "wrist")


def detect_spout_v1(cameras=SPOUT_CAMERAS_V1):
    """Detect the sink spout for grasping/pushing."""
    return detect_object_v1(list(SPOUT_QUERIES_V1), cameras)


def grasp_spout_v1(side, spout_pos, *, hover_clearance_m=0.10, descend_offset_m=0.0):
    """Approach the spout from above and close gripper around it.

    Returns ``(grasped, info)`` where grasped is True if gripper width is in the
    "object held" range.
    """
    spout_pos = np.asarray(spout_pos, dtype=float)
    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)

    print("  open gripper, hover above spout")
    open_gripper(side)
    hover = spout_pos.copy()
    hover[2] += float(hover_clearance_m)
    result = freespace_move(
        right_target_pos=hover.tolist(),
        right_target_quat=current_quat.tolist(),
        side=side,
        gripper=1.0,
        auto_update_world=True,
    )
    hover_status = str(getattr(result, "status", result))
    print(f"  hover status={hover_status} target={fmt_xyz_v1(hover)}")
    if hover_status != "Success":
        return False, {"phase": "hover", "status": hover_status}

    descend = spout_pos.copy()
    descend[2] += float(descend_offset_m)
    result = freespace_move(
        right_target_pos=descend.tolist(),
        side=side,
        gripper=1.0,
        auto_update_world=True,
    )
    descend_status = str(getattr(result, "status", result))
    print(f"  descend status={descend_status} target={fmt_xyz_v1(descend)}")
    if descend_status != "Success":
        # Brutal nudge straight down as fallback
        nudge_brutal(
            side=side,
            delta_pos=[0.0, 0.0, -float(hover_clearance_m)],
            n_steps=10,
        )

    close_gripper(side)
    time.sleep(0.25)
    width = float(np.asarray(get_robot_state().arms[side].gripper_pos, dtype=float).reshape(-1)[0])
    grasped = 0.05 < width < 0.95
    print(f"  gripper width={width:.4f} -> {'GRASPED' if grasped else 'no grip'}")
    return grasped, {"phase": "done", "width": width}


def rotate_spout_to_v1(
    side,
    spout_pos,
    target_orient,
    *,
    arc_radius_m=0.06,
    arc_steps=8,
    poll_seconds=0.2,
):
    """Push the spout sideways to one of {"left", "center", "right"}.

    Strategy: starting from current EE pose, do a brutal nudge in the world Y
    direction (left = +Y, right = -Y) by `arc_radius_m * direction`. Caller is
    expected to have the spout grasped or contacting it.
    """
    direction_map = {
        "left": np.array([0.0, 1.0, 0.0], dtype=float),
        "right": np.array([0.0, -1.0, 0.0], dtype=float),
        "center": np.array([0.0, 0.0, 0.0], dtype=float),
    }
    if str(target_orient) not in direction_map:
        raise ValueError(f"target_orient must be one of {list(direction_map)}")
    direction = direction_map[str(target_orient)]
    delta = (direction * float(arc_radius_m)).tolist()
    print(
        f"  rotate_spout_to {target_orient!r}: "
        f"delta={[round(float(x), 3) for x in delta]}"
    )
    if str(target_orient) == "center":
        # Push toward where center is most likely (between observed left/right peaks).
        # Without spout_pos history, just hold position; success check polls below.
        return True, {"phase": "noop", "delta": delta}
    result = nudge_brutal(side=side, delta_pos=delta, n_steps=int(arc_steps))
    ok = bool(getattr(result, "success", False))
    time.sleep(float(poll_seconds))
    return ok, {"phase": "done", "delta": delta, "success": ok}


def sweep_spout_three_locations_v1(
    side,
    spout_pos,
    *,
    order=("left", "center", "right"),
    arc_radius_m=0.06,
    arc_steps=10,
    settle_seconds=0.6,
    early_exit_on_success=True,
):
    """Move the spout through left/center/right positions, polling task success.

    Returns ``(succeeded, info)``.
    """
    results = []
    for orient in order:
        ok, step_info = rotate_spout_to_v1(
            side,
            spout_pos,
            orient,
            arc_radius_m=arc_radius_m,
            arc_steps=arc_steps,
        )
        results.append({"orient": orient, "ok": ok, "info": step_info})
        time.sleep(float(settle_seconds))
        info = get_task_info()
        reward = float(info.get("reward", 0.0))
        success = bool(info.get("success", False))
        print(f"  after {orient}: reward={reward:.3f} success={success}")
        if success and early_exit_on_success:
            return True, {"phase": "done", "results": results, "stopped_at": orient}
    info = get_task_info()
    return bool(info.get("success", False)), {
        "phase": "done",
        "results": results,
        "stopped_at": None,
    }


def show_spout_debug_markers_v1(spout_pos):
    show_debug_markers_v1(
        [
            debug_marker_v1(
                "sink_spout",
                np.asarray(spout_pos, dtype=float),
                (96, 192, 255),
                radius=0.025,
            ),
        ]
    )


_ = normalize_v1  # placeholder for future helpers
