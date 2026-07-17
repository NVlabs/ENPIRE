# place_with_orientation.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

@skill
def place_with_orientation_v1(
    side,
    container_pos,
    hold_gripper=0.1,
    hover_height=0.12,
    place_z_offset=0.03,
):
    """Move to home, hover above container trying multiple orientations, lower, release.

    Tries 6 place-quaternion candidates (current EE, vertical, 4 horizontal directions)
    for the hover move. Uses whatever succeeded for the lower move.

    Returns: (success, {"hover_status": str, "lower_status": str})
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    def normalize(v):
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v

    def quat_from_direction(d):
        z = normalize(d)
        x_hint = np.array([1., 0., 0.], dtype=float)
        x = x_hint - z * float(np.dot(x_hint, z))
        if float(np.linalg.norm(x)) < 1e-6:
            x = np.cross(np.array([0., 1., 0.]), z)
        x = normalize(x)
        y = normalize(np.cross(z, x))
        x = normalize(np.cross(y, z))
        return R.from_matrix(np.column_stack([x, y, z])).as_quat()

    container_pos = np.array(container_pos, dtype=float)

    state = get_robot_state()
    cur_quat = np.asarray(state.arms[side].ee_quat, dtype=float).tolist()

    place_candidates = [
        ("current", cur_quat),
        ("vertical", quat_from_direction([0., 0., -1.]).tolist()),
        ("horizontal_+x", quat_from_direction([1., 0., 0.]).tolist()),
        ("horizontal_-x", quat_from_direction([-1., 0., 0.]).tolist()),
        ("horizontal_+y", quat_from_direction([0., 1., 0.]).tolist()),
        ("horizontal_-y", quat_from_direction([0., -1., 0.]).tolist()),
    ]

    # Hover above container
    hover_target = container_pos.copy()
    hover_target[2] += hover_height
    selected_quat = None
    hover_status = "all_failed"
    for label, quat in place_candidates:
        r = freespace_move(right_target_pos=hover_target.tolist(),
                           right_target_quat=quat, side=side, gripper=hold_gripper)
        print(f"place_with_orientation hover {label}: {r.status}")
        if r.status == "Success":
            selected_quat = quat
            hover_status = r.status
            break

    # Lower to place
    place_target = container_pos.copy()
    place_target[2] += place_z_offset
    lower_candidates = [("selected", selected_quat)] if selected_quat else place_candidates
    lower_status = "skipped"
    for label, quat in lower_candidates:
        r = freespace_move(right_target_pos=place_target.tolist(),
                           right_target_quat=quat, side=side, gripper=hold_gripper)
        print(f"place_with_orientation lower {label}: {r.status}")
        lower_status = r.status
        if r.status == "Success":
            break

    # Release and retract
    open_gripper(side)
    go_home(side)
    print("place_with_orientation: released and homed")

    success = lower_status == "Success"
    return success, {"hover_status": hover_status, "lower_status": lower_status}
