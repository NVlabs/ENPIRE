# vertical_grasp.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

@skill
def vertical_grasp_v1(side, obj_pos, z_offset=0.0, hold_strength=0.3):
    """Descend onto obj_pos with optional z_offset, compliant-close, verify grasp.

    Grasp orientation is whatever the EE currently holds — no quaternion set.
    Suitable for small flat/wedge objects where top-down grip is safe.
    """
    import numpy as np
    target = np.array(obj_pos, dtype=float).copy()
    target[2] += z_offset
    r = freespace_move(right_target_pos=target.tolist(), side=side)
    descend_ok = r.status == "Success"
    print(f"vertical_grasp_v1: target={target.tolist()} status={r.status}")

    close_gripper(side, compliant=True, hold_strength=hold_strength)
    info = get_gripper_info(side)
    grasped = bool(
        info.get("has_object")
        and not info.get("is_fully_closed", False)
        and (info.get("actuator_force_N") or 0.0) > 1.0
    )
    print(f"vertical_grasp_v1: grasped={grasped} gripper_info={info}")
    return grasped, {
        "success": grasped,
        "descend_status": r.status,
        "descend_ok": descend_ok,
        "gripper_info": info,
        "target": target.tolist(),
        "z_offset": z_offset,
    }
