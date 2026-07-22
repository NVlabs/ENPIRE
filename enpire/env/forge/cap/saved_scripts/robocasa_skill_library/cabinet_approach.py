# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cabinet_approach.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

@skill
def cabinet_approach_v1(side, approach_pos, front_normal, hold_gripper=0.0):
    """Move to the cabinet opening with gripper oriented inward (facing -front_normal).

    front_normal: outward normal of cabinet front face (pointing away from cabinet).
    Returns: (success, {"status": str, "orientation_used": str})
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    def normalize(v):
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v

    def quat_from_direction(d):
        z = normalize(d)
        h = np.array([0., 0., 1.])
        if abs(np.dot(z, h)) > 0.95:
            h = np.array([1., 0., 0.])
        x = normalize(np.cross(h, z))
        y = normalize(np.cross(z, x))
        return R.from_matrix(np.column_stack([x, y, z])).as_quat()

    fn = normalize(front_normal)
    orient_candidates = [
        ("cab-in", quat_from_direction(-fn)),
        ("cab-out", quat_from_direction(fn)),
        ("down", quat_from_direction([0., 0., -1.])),
        ("current", np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)),
    ]
    approach_pos = np.array(approach_pos, dtype=float)

    for label, quat in orient_candidates:
        r = freespace_move(
            right_target_pos=approach_pos.tolist(),
            right_target_quat=np.asarray(quat).tolist(),
            side=side,
        )
        print(f"cabinet_approach: {label} -> {r.status}")
        if r.status == "Success":
            return True, {"status": r.status, "orientation_used": label}

    return False, {"status": "all_failed", "orientation_used": None}
