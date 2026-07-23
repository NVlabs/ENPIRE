# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# descend_and_grasp.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403

from enpire.env.forge.cap.agent.skill_registry import skill


@skill
def descend_and_grasp_v1(
    side,
    obj_pos,
    selected_quat,
    pre_descend_offset=0.010,
    descend_offsets=(0.020, 0.015, 0.010, 0.005, 0.000, -0.005),
    hold_strength=0.3,
    contact_pos_thresh=0.12,
    contact_z_thresh=0.08,
    nudge_retry_deltas=(-0.005, -0.010, -0.015, -0.020, -0.025, -0.030),
):
    """Pre-descend then final descend to obj_pos with IK-retry offsets, then grasp.

    Steps:
    1. Pre-descend to obj_pos + pre_descend_offset (fine-position above object)
    2. Final descend trying each z-offset until IK succeeds
    3. Detect contact via pose error check
    4. Close gripper + nudge-down retry loop

    Disables collision avoidance (exclude_body_prefixes=[""]) before final
    descend and restores after, matching the human script pattern.

    Returns: (grasped, {"grip_val": float, "descend_status": str})
    """
    import numpy as np

    selected_quat = list(selected_quat)
    obj_pos = np.array(obj_pos, dtype=float)

    # Pre-descend
    pre_target = obj_pos.copy()
    pre_target[2] += pre_descend_offset
    r_pre = freespace_move(right_target_pos=pre_target.tolist(),
                           right_target_quat=selected_quat, side=side)
    print(f"descend_and_grasp pre-descend z=+{pre_descend_offset:.3f}: {r_pre.status}")

    # Disable collision avoidance for grasp phase
    update_planner_world(exclude_body_prefixes=[""])
    print("descend_and_grasp: collision world cleared for grasp phase")

    # Final descend — try each z-offset
    descend_result = None
    attempt_target = obj_pos.copy()
    for dz in descend_offsets:
        candidate = obj_pos.copy()
        candidate[2] -= dz
        r = freespace_move(right_target_pos=candidate.tolist(),
                           right_target_quat=selected_quat, side=side)
        label = "Descend" if dz == descend_offsets[0] else f"Retry z=-{dz:.3f}"
        print(f"descend_and_grasp {label}: {r.status}")
        descend_result = r
        attempt_target = candidate
        if r.status == "Success":
            break
        if r.status != "IK_Failed":
            break

    # Contact check
    state = get_robot_state()
    actual_pos = np.array(state.arms[side].ee_pos, dtype=float)
    pos_err = float(np.linalg.norm(actual_pos - attempt_target))
    z_err = float(actual_pos[2] - attempt_target[2])
    likely_contact = (descend_result.status == "Success" and
                      (pos_err > contact_pos_thresh or z_err > contact_z_thresh))
    print(f"descend_and_grasp pose check: pos_err={pos_err:.3f} z_err={z_err:+.3f} "
          f"-> {'CONTACT' if likely_contact else 'OK'}")

    if likely_contact or descend_result.status != "Success":
        print("descend_and_grasp: descend failed or contact detected")
        return False, {"grip_val": 0.0, "descend_status": descend_result.status,
                       "reason": "descend_failed"}

    # Close gripper
    close_gripper(side, compliant=True, hold_strength=hold_strength)
    state = get_robot_state()
    grip_val = float(np.asarray(state.arms[side].gripper_pos, dtype=float).reshape(-1)[0])
    grasped = 0.05 < grip_val < 0.95
    print(f"descend_and_grasp close: grip={grip_val:.4f} -> {'GRASPED' if grasped else 'MISSED'}")

    # Nudge-down retry loop
    for retry_idx, dz in enumerate(nudge_retry_deltas, 1):
        if grasped:
            break
        open_gripper(side)
        r_nudge = nudge(side=side, delta_pos=[0.0, 0.0, float(dz)])
        if not r_nudge.success:
            print(f"descend_and_grasp retry {retry_idx} nudge failed")
            continue
        close_gripper(side, compliant=True, hold_strength=hold_strength)
        state = get_robot_state()
        grip_val = float(np.asarray(state.arms[side].gripper_pos, dtype=float).reshape(-1)[0])
        grasped = 0.05 < grip_val < 0.90
        print(f"descend_and_grasp retry {retry_idx}: grip={grip_val:.4f} -> "
              f"{'GRASPED' if grasped else 'MISSED'}")

    return grasped, {
        "grip_val": grip_val,
        "descend_status": descend_result.status,
        "grasped_by_sensor": grasped,
    }
