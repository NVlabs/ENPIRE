# under_water_holding.py — hold a grasped object under the faucet stream until success
from skill_library.namespace import *  # noqa: F401, F403

import time

import numpy as np

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_geometry import (
    fmt_xyz_v1,
)


def hold_under_faucet_v1(
    side,
    faucet_pos,
    *,
    hold_offset_m=(0.0, 0.0, -0.10),
    n_timesteps=40,
    poll_seconds=0.15,
    success_check=True,
    fall_back_offsets_m=(
        (0.0, 0.0, -0.05),
        (0.05, 0.0, -0.10),
        (-0.05, 0.0, -0.10),
        (0.0, 0.05, -0.10),
        (0.0, -0.05, -0.10),
    ),
):
    """Hold a grasped object below the faucet position and wait for task success.

    Strategy:
      1. compute hold target = faucet_pos + hold_offset_m (default 10cm below)
      2. freespace_move to hold pose with current EE quaternion
      3. poll get_task_info() for `n_timesteps` ticks; early-exit on success
      4. if move planning fails, try alternate offsets

    Returns ``(success, info)``.
    """
    faucet_pos = np.asarray(faucet_pos, dtype=float)
    hold_offset_m = np.asarray(hold_offset_m, dtype=float)
    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)

    candidate_offsets = [tuple(float(x) for x in hold_offset_m)]
    for off in fall_back_offsets_m:
        if tuple(off) not in candidate_offsets:
            candidate_offsets.append(tuple(float(x) for x in off))

    move_status = "Skipped"
    used_offset = None
    for off in candidate_offsets:
        target = faucet_pos + np.asarray(off, dtype=float)
        result = freespace_move(
            right_target_pos=target.tolist(),
            right_target_quat=current_quat.tolist(),
            side=side,
            gripper=0.0,
            auto_update_world=True,
        )
        move_status = str(getattr(result, "status", result))
        print(
            f"  hold offset={tuple(round(float(v), 3) for v in off)}: "
            f"status={move_status} target={fmt_xyz_v1(target)}"
        )
        if move_status == "Success":
            used_offset = tuple(float(x) for x in off)
            break
    if move_status != "Success":
        return False, {"phase": "move_to_hold", "status": move_status}

    print(f"  holding for up to {int(n_timesteps)} ticks (poll={poll_seconds:.2f}s)")
    succeeded = False
    elapsed_ticks = 0
    for tick in range(int(n_timesteps)):
        elapsed_ticks = tick + 1
        time.sleep(float(poll_seconds))
        if not success_check:
            continue
        info = get_task_info()
        if info.get("success", False):
            succeeded = True
            print(f"  task_success=True after {elapsed_ticks} ticks")
            break
        if tick % 5 == 0:
            print(
                f"  tick {elapsed_ticks}/{int(n_timesteps)}: "
                f"reward={float(info.get('reward', 0.0)):.3f}"
            )

    return succeeded, {
        "phase": "done",
        "used_offset": list(used_offset) if used_offset else None,
        "ticks_held": elapsed_ticks,
        "succeeded_during_hold": bool(succeeded),
    }
