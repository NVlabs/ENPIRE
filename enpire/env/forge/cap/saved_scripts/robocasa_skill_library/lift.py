# lift.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

@skill
def lift_v1(side, delta_z=0.20):
    """Lift EE by +delta_z from current position. Preserves XY and orientation."""
    import numpy as np
    state = get_robot_state()
    cur = np.array(state.arms[side].ee_pos)
    target = cur.copy()
    target[2] += delta_z
    r = freespace_move(right_target_pos=target.tolist(), side=side)
    success = r.status == "Success"
    print(f"lift_v1: from={cur.tolist()} to={target.tolist()} status={r.status}")
    return success, {
        "success": success,
        "status": r.status,
        "delta_z": delta_z,
        "start_pos": cur.tolist(),
    }

@skill
def lift_v2(side, delta_z=0.20, step_size=0.04):
    """Lift EE by +delta_z using incremental nudge steps. More reliable than single freespace_move from constrained positions."""
    import numpy as np

    state = get_robot_state()
    start_z = state.arms[side].ee_pos[2]
    target_z = start_z + delta_z

    n_steps = int(delta_z / step_size) + 1
    n_steps = min(n_steps, 10)

    all_ok = True
    for i in range(n_steps):
        state = get_robot_state()
        current_z = state.arms[side].ee_pos[2]
        if current_z >= target_z - 0.005:
            break
        remaining = target_z - current_z
        this_step = min(step_size, remaining)
        r = nudge(side=side, delta_pos=[0.0, 0.0, this_step])
        if not r.success:
            print(f"lift_v2 nudge {i} failed at z={current_z:.4f}")
            all_ok = False
            break

    final_state = get_robot_state()
    final_z = final_state.arms[side].ee_pos[2]
    achieved = final_z - start_z
    success = achieved > delta_z * 0.5

    return success, {
        "success": success,
        "all_nudges_ok": all_ok,
        "start_z": start_z,
        "final_z": final_z,
        "achieved_delta": achieved,
        "target_delta": delta_z,
    }
