# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# post_grasp_lift.py — skill library, append-only
from enpire.env.forge.cap.agent.skill_registry import skill
from skill_library.namespace import *  # noqa: F401, F403


@skill
def post_grasp_lift_v1(
    side,
    retreat_height=0.08,
    lift_height=0.20,
    hold_gripper=0.1,
):
    """After a successful grasp: retreat upward, restore collision world, lift out of sink.

    Steps:
    1. Retreat upward by retreat_height (collision-avoidance still disabled)
    2. Restore collision world (excluding robot, gripper, object)
    3. Lift by lift_height via freespace_move

    Returns: (success, {"retreat_status": str, "lift_status": str, "final_z": float})
    """
    import numpy as np

    # Retreat upward before restoring collisions
    state = get_robot_state()
    retreat_pos = np.array(state.arms[side].ee_pos, dtype=float)
    retreat_pos[2] += retreat_height
    r_retreat = freespace_move(right_target_pos=retreat_pos.tolist(), side=side,
                               gripper=hold_gripper)
    print(f"post_grasp_lift retreat: {r_retreat.status}")

    # Restore collision world (exclude robot + object so grasped object doesn't block)
    world = update_planner_world(
        exclude_body_prefixes=["robot0", "gripper", "mobilebase", "obj"]
    )
    print(f"post_grasp_lift: collision world restored ({world.get('n_obstacles', '?')} obstacles)")

    # Lift
    state = get_robot_state()
    lift_pos = np.array(state.arms[side].ee_pos, dtype=float)
    lift_pos[2] += lift_height
    r_lift = freespace_move(right_target_pos=lift_pos.tolist(), side=side,
                            gripper=hold_gripper)
    print(f"post_grasp_lift lift: {r_lift.status}")

    final_state = get_robot_state()
    success = r_lift.status == "Success"
    return success, {
        "retreat_status": r_retreat.status,
        "lift_status": r_lift.status,
        "final_z": round(float(final_state.arms[side].ee_pos[2]), 4),
    }
