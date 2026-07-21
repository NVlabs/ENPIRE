# robust_grasp.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

@skill
def robust_grasp_v1(
    side,
    obj_pos,
    hover_clearance=0.12,
    grasp_z_offset=0.0,
    clearance_lift=0.03,
    hold_strength=0.4,
):
    """Hover + descend with multi-orientation candidate search, then verify via gripper width.

    Unlike incremental_grasp_v1 (nudge-based descent), this uses freespace_move
    with multiple gripper orientations so IK has more solutions to choose from.
    Grasp is confirmed by checking gripper width (0.02 < width < 0.98) rather
    than object position delta. A small clearance lift follows to escape any
    collision zone before the main lift.

    Returns (success, log_dict).
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    obj_pos = np.array(obj_pos, dtype=float)

    # --- Build orientation candidates (gripper +Z pointing in 6 cardinal directions) ---
    def _quat_from_dir(d):
        z = np.array(d, dtype=float)
        z /= max(float(np.linalg.norm(z)), 1e-8)
        up = np.array([0., 0., 1.])
        if abs(float(np.dot(z, up))) > 0.95:
            up = np.array([1., 0., 0.])
        x = np.cross(up, z); x /= max(float(np.linalg.norm(x)), 1e-8)
        y = np.cross(z, x)
        return R.from_matrix(np.column_stack([x, y, z])).as_quat()

    state = get_robot_state()
    cur_quat = np.array(state.arms[side].ee_quat, dtype=float)

    orient_candidates = [
        ("current", cur_quat),
        ("dir+z",   _quat_from_dir([0., 0.,  1.])),
        ("dir-z",   _quat_from_dir([0., 0., -1.])),
        ("dir+x",   _quat_from_dir([1., 0.,  0.])),
        ("dir-x",   _quat_from_dir([-1., 0., 0.])),
        ("dir+y",   _quat_from_dir([0.,  1., 0.])),
        ("dir-y",   _quat_from_dir([0., -1., 0.])),
    ]

    # --- Step 1: Hover above object ---
    hover_target = obj_pos.copy(); hover_target[2] += hover_clearance
    hover_ok = False
    for label, quat in orient_candidates:
        r = freespace_move(
            right_target_pos=hover_target.tolist(),
            right_target_quat=quat.tolist(),
            side=side,
        )
        print(f"robust_grasp hover [{label}]: {r.status}")
        if r.status == "Success":
            hover_ok = True
            break

    if not hover_ok:
        return False, {"success": False, "reason": "hover_failed"}

    # --- Step 2: Descend to object with position + orientation offsets ---
    state = get_robot_state()
    cur_quat = np.array(state.arms[side].ee_quat, dtype=float)

    grasp_target = obj_pos.copy(); grasp_target[2] += grasp_z_offset

    # Small position offsets: nominal first, then lateral/z fallbacks
    pos_offsets = [
        [0.,     0.,    0.  ],
        [0.,     0.,    0.01],
        [0.,     0.,   -0.01],
        [0.,     0.,    0.02],
        [0.01,   0.,    0.  ],
        [-0.01,  0.,    0.  ],
        [0.,     0.01,  0.  ],
        [0.,    -0.01,  0.  ],
    ]
    # Orientation tweaks: identity, then small roll/pitch/yaw
    orient_offsets_deg = [
        [0.,   0.,   0. ],
        [0.,   0., -25. ],
        [0.,   0.,  25. ],
        [-20., 0.,   0. ],
        [ 20., 0.,   0. ],
    ]

    grasp_ok = False
    for off in pos_offsets:
        candidate_pos = grasp_target + np.array(off, dtype=float)
        for rpy in orient_offsets_deg:
            candidate_quat = (
                R.from_euler("xyz", rpy, degrees=True) * R.from_quat(cur_quat)
            ).as_quat()
            r = freespace_move(
                right_target_pos=candidate_pos.tolist(),
                right_target_quat=candidate_quat.tolist(),
                side=side,
            )
            if r.status == "Success":
                grasp_ok = True
                break
        if grasp_ok:
            break

    print(f"robust_grasp descend: ok={grasp_ok}")
    if not grasp_ok:
        return False, {"success": False, "reason": "descend_failed"}

    # --- Step 3: Close gripper and check width ---
    close_gripper(side, compliant=True, hold_strength=hold_strength)

    state = get_robot_state()
    grip_vals = state.arms[side].gripper_pos
    grip_w = float(np.asarray(grip_vals, dtype=float).reshape(-1)[0])
    grasped = 0.02 < grip_w < 0.98
    print(f"robust_grasp gripper width={grip_w:.4f} -> {'GRASPED' if grasped else 'MISSED'}")

    if not grasped:
        open_gripper(side)
        return False, {"success": False, "reason": "gripper_miss", "width": grip_w}

    # --- Step 4: Small clearance lift to escape collision zone ---
    set_gripper(side, 0.0)
    if clearance_lift > 0:
        nudge(side, delta_pos=[0., 0., clearance_lift])

    return True, {
        "success": True,
        "width": grip_w,
        "clearance_lift": clearance_lift,
    }
