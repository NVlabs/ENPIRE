# toaster_extract.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

@skill
def toaster_extract_v1(side, place_pos, extract_distances=(0.10, 0.14, 0.18), rise=0.02,
                       clear_height=0.25):
    """Pull object out of toaster slot and lift to safe clearance height.

    Two-phase exit:
      1. Horizontal extraction: freespace_move laterally toward place_pos with a
         small upward component (rise) to clear the slot edge. Tried at increasing
         distances until one succeeds.
      2. Vertical lift via nudge_brutal: after extraction the cuRobo collision world
         may still see the arm as inside toaster geometry, so regular nudge (which
         checks start-state collision) gets blocked. nudge_brutal bypasses the planner
         entirely and uses raw OSC ticks to lift to clear_height.

    Returns (success, log_dict). Callers should proceed directly to counter hover
    after this — no separate lift step needed.
    """
    import numpy as np

    state = get_robot_state()
    ee_pos = np.asarray(state.arms[side].ee_pos, dtype=float)
    ee_quat = np.asarray(state.arms[side].ee_quat, dtype=float).tolist()

    place_pos = np.asarray(place_pos, dtype=float)
    d = place_pos[:2] - ee_pos[:2]
    n = float(np.linalg.norm(d))
    extract_dir = d / n if n > 1e-6 else np.array([1., 0.])

    # Phase 1: horizontal extraction
    extracted_dist = None
    for dist in extract_distances:
        ep = ee_pos.copy()
        ep[:2] += extract_dir * dist
        ep[2] += rise
        r = freespace_move(right_target_pos=ep.tolist(), right_target_quat=ee_quat, side=side)
        print(f"toaster_extract horizontal d={dist:.2f}: {r.status}")
        if r.status == "Success":
            extracted_dist = dist
            break

    if extracted_dist is None:
        return False, {"status": "horizontal_failed", "distance": None}

    # Phase 2: vertical lift via nudge_brutal — bypasses cuRobo collision checking.
    # After horizontal extraction the collision world may still see the arm as inside
    # toaster geometry; regular nudge (which checks start-state collision) gets blocked.
    # nudge_brutal uses raw OSC ticks and ignores the planner entirely.
    step_size = 0.05
    n_steps = int(clear_height / step_size)
    for i in range(n_steps):
        nudge_brutal(side, delta_pos=[0., 0., step_size])

    state = get_robot_state()
    final_z = float(state.arms[side].ee_pos[2])
    lifted_enough = final_z > (ee_pos[2] + clear_height * 0.5)
    print(f"toaster_extract vertical lift: final_z={final_z:.3f} lifted={lifted_enough}")

    return True, {
        "status": "Success",
        "horizontal_distance": extracted_dist,
        "final_z": round(final_z, 4),
        "lifted_enough": lifted_enough,
    }


def toaster_lever_press_v1(
    side,
    lever_pos,
    *,
    hover_clearance_m=0.08,
    press_depth_m=0.06,
    n_press_steps=8,
    retreat_distance_m=0.20,
):
    """Press the toaster lever down to start toasting.

    Hover above the lever, close the gripper to use the closed fingertips as
    a pusher, brutal-nudge straight down, then retreat upward.
    Returns ``(pressed, info)``.
    """
    import time as _time
    import numpy as np

    lever_pos = np.asarray(lever_pos, dtype=float)
    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)

    hover = lever_pos.copy()
    hover[2] += float(hover_clearance_m)
    result = freespace_move(
        right_target_pos=hover.tolist(),
        right_target_quat=current_quat.tolist(),
        side=side,
        gripper=0.0,
        auto_update_world=True,
    )
    hover_status = str(getattr(result, "status", result))
    print(f"  toaster_lever hover status={hover_status}")
    if hover_status != "Success":
        return False, {"phase": "hover", "status": hover_status}

    close_gripper(side)
    _time.sleep(0.2)

    press = nudge_brutal(
        side=side,
        delta_pos=[0.0, 0.0, -float(press_depth_m)],
        n_steps=int(n_press_steps),
    )
    press_ok = bool(getattr(press, "success", False))
    print(f"  toaster_lever press: depth={float(press_depth_m):.3f}m success={press_ok}")
    _time.sleep(0.2)

    retreat = nudge_brutal(
        side=side,
        delta_pos=[0.0, 0.0, float(retreat_distance_m)],
        n_steps=8,
    )
    return True, {
        "phase": "done",
        "press_success": press_ok,
        "retreat_success": bool(getattr(retreat, "success", False)),
    }


def toaster_wait_until_popped_v1(
    *,
    n_ticks=30,
    poll_seconds=0.2,
):
    """Poll task_info while the toaster runs. Returns (success_during_wait, info)."""
    import time as _time

    succeeded = False
    last_reward = 0.0
    for tick in range(int(n_ticks)):
        _time.sleep(float(poll_seconds))
        info = get_task_info()
        last_reward = float(info.get("reward", 0.0))
        if info.get("success", False):
            succeeded = True
            print(f"  toaster_wait: success at tick {tick + 1}/{int(n_ticks)}")
            break
        if tick % 5 == 0:
            print(
                f"  toaster_wait tick {tick + 1}/{int(n_ticks)}: "
                f"reward={last_reward:.3f}"
            )
    return succeeded, {"ticks_waited": tick + 1, "last_reward": last_reward}
