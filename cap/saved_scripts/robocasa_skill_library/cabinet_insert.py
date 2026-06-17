# cabinet_insert.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from cap.agent.skill_registry import skill

@skill
def cabinet_insert_v1(side, insert_pos, settle_pos=None, hold_gripper=0.0):
    """Move to insert_pos then optionally to settle_pos inside the cabinet.

    Returns: (success, {"insert_status": str, "settle_status": str})
    """
    import numpy as np

    insert_pos = np.array(insert_pos, dtype=float)
    state = get_robot_state()
    cur_quat = np.asarray(state.arms[side].ee_quat, dtype=float).tolist()

    r_insert = freespace_move(
        right_target_pos=insert_pos.tolist(),
        right_target_quat=cur_quat,
        side=side,
    )
    print(f"cabinet_insert: insert -> {r_insert.status}")

    settle_status = "skipped"
    if settle_pos is not None:
        settle_arr = np.array(settle_pos, dtype=float)
        r_settle = freespace_move(
            right_target_pos=settle_arr.tolist(),
            right_target_quat=cur_quat,
            side=side,
        )
        settle_status = r_settle.status
        print(f"cabinet_insert: settle -> {r_settle.status}")

    success = r_insert.status == "Success"
    return success, {"insert_status": r_insert.status, "settle_status": settle_status}
