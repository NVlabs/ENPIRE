# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# incremental_grasp.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

@skill
def incremental_grasp_v1(side, obj_pos, hover_clearance=0.15, step_size=0.025, hold_strength=0.4):
    """Hover above obj_pos then nudge down incrementally to grasp.
    After closing, escapes sink collision with nudge_brutal, then verifies
    grasp using gripper state instead of simulator ground truth.
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

    # Step 3: close gripper
    close_gripper(side, compliant=True, hold_strength=hold_strength)
    state_after_close = get_robot_state()
    grip_val_after_close = float(np.asarray(
        state_after_close.arms[side].gripper_pos, dtype=float
    ).reshape(-1)[0])

    # Step 4: nudge_brutal upward to escape sink-wall collision before verifying
    nudge_brutal(side=side, delta_pos=[0.0, 0.0, 0.05])

    final_state = get_robot_state()
    grip_val_final = float(np.asarray(
        final_state.arms[side].gripper_pos, dtype=float
    ).reshape(-1)[0])
    grasped = 0.05 < grip_val_final < 0.95
    print(
        f"incremental_grasp: grip_after_close={grip_val_after_close:.4f} "
        f"grip_final={grip_val_final:.4f} grasped={grasped}"
    )

    return grasped, {
        "success": grasped,
        "descent_ok": descent_ok,
        "grip_after_close": grip_val_after_close,
        "grip_final": grip_val_final,
        "final_z": final_state.arms[side].ee_pos[2],
        "target_z": target_z,
    }
