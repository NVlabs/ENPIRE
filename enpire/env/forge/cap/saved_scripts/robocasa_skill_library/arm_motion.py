# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# arm_motion.py — shared arm motion helpers for RoboCasa skill library
from skill_library.namespace import *  # noqa: F401, F403


def vec(x):
    import numpy as np

    return np.asarray(x, dtype=float)


def current_arm_pose():
    state = get_robot_state()
    arm = list(state.arms.keys())[0]
    arm_state = state.arms[arm]
    return arm, vec(arm_state.ee_pos), vec(arm_state.ee_quat)


def move_with_orientation(arm, target_pos, target_quat, exclude_body_prefixes=None):
    kwargs = dict(
        right_target_pos=vec(target_pos).tolist(),
        right_target_quat=vec(target_quat).tolist(),
        side=arm,
    )
    if exclude_body_prefixes is not None:
        kwargs["exclude_body_prefixes"] = exclude_body_prefixes
    return freespace_move(**kwargs)


def move_checked(
    arm,
    label,
    target_pos,
    target_quat,
    max_err,
    exclude_body_prefixes=None,
):
    result = move_with_orientation(
        arm,
        target_pos,
        target_quat,
        exclude_body_prefixes=exclude_body_prefixes,
    )
    err = float(getattr(result, "final_pos_error_m", 1e9) or 1e9)
    status = getattr(result, "status", "Unknown")
    print(
        f"{label}: status={status} err={err:.4f} "
        f"pos={[round(float(x), 3) for x in vec(target_pos)]}"
    )
    return status == "Success" and err <= max_err


def go_home_checked(arm, label, exclude_body_prefixes=None):
    result = go_home(arm, exclude_body_prefixes=exclude_body_prefixes)
    status = (
        result.get("status", "Unknown")
        if isinstance(result, dict)
        else getattr(result, "status", "Unknown")
    )
    print(f"{label}: status={status}")
    return status == "Success"


def release_with_clearance_v1(side, *, clearance_m=0.20, n_steps=10, settle_s=0.4):
    """Open the gripper, retreat the EE upward by `clearance_m`, return retreat result.

    Useful for satisfying `gripper_obj_far` checks (default threshold 0.15 m).
    """
    import time as _time

    open_gripper(side)
    _time.sleep(float(settle_s))
    retreat = nudge_brutal(
        side=side,
        delta_pos=[0.0, 0.0, float(clearance_m)],
        n_steps=int(n_steps),
    )
    ok = bool(getattr(retreat, "success", False))
    final_pos = getattr(retreat, "final_pos", None)
    print(
        f"release_with_clearance: clearance_m={float(clearance_m):.3f} "
        f"success={ok} final_pos={final_pos}"
    )
    return ok, {"clearance_m": float(clearance_m), "success": ok}
