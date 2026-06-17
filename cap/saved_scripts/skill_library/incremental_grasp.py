# incremental_grasp.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from cap.agent.skill_registry import skill

@skill
def incremental_grasp_v1(side, obj_pos, hover_clearance=0.15, step_size=0.025, hold_strength=0.4):
    """Hover above obj_pos then nudge down incrementally to grasp.
    After closing, escapes sink collision with nudge_brutal then verifies
    grasp by checking whether the object rose with the arm (obj_pos[2] delta).
    Ground-truth position check is more reliable than visual or force heuristics.
    """
    import numpy as np

    # Step 1: hover above the object
    hover_target = np.array(obj_pos, dtype=float).copy()
    hover_target[2] += hover_clearance
    r_hover = freespace_move(right_target_pos=hover_target.tolist(), side=side)
    print(f"incremental_grasp hover: status={r_hover.status} target={hover_target.tolist()}")

    if r_hover.status != "Success":
        hover_target[2] += 0.05
        r_hover = freespace_move(right_target_pos=hover_target.tolist(), side=side)
        print(f"incremental_grasp hover retry: status={r_hover.status}")
        if r_hover.status != "Success":
            return False, {"success": False, "reason": "hover_failed", "status": r_hover.status}

    # Step 2: nudge down incrementally toward the object
    target_z = obj_pos[2]
    state = get_robot_state()
    current_z = state.arms[side].ee_pos[2]
    max_steps = min(int((current_z - target_z) / step_size) + 3, 15)

    descent_ok = True
    for i in range(max_steps):
        state = get_robot_state()
        current_z = state.arms[side].ee_pos[2]
        if current_z <= target_z + 0.005:
            print(f"incremental_grasp reached target z={current_z:.4f}")
            break
        r_nudge = nudge(side=side, delta_pos=[0.0, 0.0, -step_size])
        if not r_nudge.success:
            print(f"incremental_grasp nudge {i} failed at z={current_z:.4f}")
            descent_ok = False
            break

    # Step 3: record object position before closing
    info_before = get_task_info()
    obj_z_before = info_before["obj_pos"][2]

    # Step 4: close gripper
    close_gripper(side, compliant=True, hold_strength=hold_strength)

    # Step 5: nudge_brutal upward to escape sink-wall collision before verifying
    nudge_brutal(side=side, delta_pos=[0.0, 0.0, 0.05])

    # Step 6: check if object rose with the arm — ground-truth position delta
    info_after = get_task_info()
    obj_z_after = info_after["obj_pos"][2]
    obj_rose = obj_z_after - obj_z_before
    grasped = obj_rose > 0.015
    print(f"incremental_grasp: obj_z before={obj_z_before:.4f} after={obj_z_after:.4f} rose={obj_rose:.4f} grasped={grasped}")

    final_state = get_robot_state()
    return grasped, {
        "success": grasped,
        "descent_ok": descent_ok,
        "obj_z_before": obj_z_before,
        "obj_z_after": obj_z_after,
        "obj_rose": round(obj_rose, 4),
        "final_z": final_state.arms[side].ee_pos[2],
        "target_z": target_z,
    }
