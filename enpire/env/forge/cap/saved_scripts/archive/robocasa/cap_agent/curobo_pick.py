# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pick up an object in RoboCasa using cuRobo motion planning.
#
# Uses ground-truth object position from get_task_info() — no vision needed.
#
# Tools used:
#   - create_motion_planner      — cuRobo Panda planner (sandbox API)
#   - update_planner_world       — load kitchen collision geometry into planner
#   - get_robot_state            — EE pose, joint state, gripper
#   - get_task_info              — reward, success, object ground truth
#   - set_gripper                — gripper control
#   - move_joint_keypoints       — joint trajectory execution
#
# Run from web UI or via run_script.py:
#   ROBOCASA_CONTROLLER_TYPE=joint_position CAP_ROBOT_TYPE=panda \
#   CAP_AGENT_NAME=demo CAP_CUROBO_PORT=8611 \
#   uv run python -u run_script.py \
#   --file robocasa_curobo_pick.py \
#   --env "robocasa:PickPlaceCounterToCabinet" --cap-port 18600 --no-log

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# 1. Initialize cuRobo planner (via sandbox API — connects to remote server)
# ---------------------------------------------------------------------------
print("Initialising cuRobo Panda planner...")
planner = create_motion_planner(solver_speed="slow")
print("cuRobo ready.\n")

# ---------------------------------------------------------------------------
# 2. Load collision world + get base frame transform
# ---------------------------------------------------------------------------
print("Loading kitchen obstacles into cuRobo...")
world_info = update_planner_world(planner)
base_pos = np.array(world_info["base_pos"], dtype=np.float64)
base_quat = np.array(world_info["base_quat_xyzw"], dtype=np.float64)
R_base = R.from_quat(base_quat)
print(f"Base position: {[round(x, 3) for x in base_pos]}")
print(f"Loaded {world_info['n_obstacles']} obstacles")

# ---------------------------------------------------------------------------
# 3. Get robot state and task info
# ---------------------------------------------------------------------------
state = get_robot_state()
arm = list(state.arms.keys())[0]
ee_pos = np.array(state.arms[arm].ee_pos)
ee_quat = np.array(state.arms[arm].ee_quat)
joint_pos = np.array(state.arms[arm].joint_pos)
print(f"Arm: {arm}")
print(f"EE position: {[round(x, 3) for x in ee_pos]}")
print(f"Joint positions: {[round(x, 3) for x in joint_pos]}")

task_info = get_task_info()
obj_pos = np.array(task_info.get("obj_pos", ee_pos))
print(f"Object position: {[round(float(x), 3) for x in obj_pos]}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def world_to_base(pos, quat_xyzw):
    """Transform a world-frame pose to the arm base frame."""
    pos_base = R_base.inv().apply(np.asarray(pos) - base_pos)
    quat_base = (R_base.inv() * R.from_quat(quat_xyzw)).as_quat()
    return pos_base, quat_base


def plan_and_move(target_pos_world, target_quat_world, gripper=None, label="move"):
    """Plan with cuRobo and execute via move_joint_keypoints."""
    cur_state = get_robot_state()
    cur_jp = np.array(cur_state.arms[arm].joint_pos)

    target_pos_base, target_quat_base = world_to_base(target_pos_world, target_quat_world)
    print(f"  [{label}] World target: {[round(x, 3) for x in target_pos_world]}")
    print(f"  [{label}] Base target:  {[round(x, 3) for x in target_pos_base]}")

    t0 = time.time()
    result = planner.plan_to_pose(
        current_left_jp=np.zeros(0),
        current_right_jp=cur_jp,
        target_right_pos=target_pos_base,
        target_right_quat_xyzw=target_quat_base,
        side="right",
    )
    dt = time.time() - t0

    status = result["status"]
    print(f"  [{label}] Status: {status} ({dt*1000:.0f}ms)")
    if status != "Success":
        detail = result.get("status_detail", "")
        print(f"  [{label}] FAILED: {detail}")
        return False

    positions = result["right_positions"]  # (N, 7)
    n_steps = positions.shape[0]
    print(f"  [{label}] Trajectory: {n_steps} waypoints")

    # Build timestamps (constant velocity)
    timestamps = [0.0]
    speed = 1.0  # rad/s max joint speed
    for i in range(n_steps - 1):
        max_delta = float(np.max(np.abs(positions[i + 1] - positions[i])))
        dt_seg = max(0.01, max_delta / speed)
        timestamps.append(timestamps[-1] + dt_seg)

    # Build gripper trajectory
    gripper_positions = None
    if gripper is not None:
        gp = cur_state.arms[arm].gripper_pos
        cur_gp = float(gp[0]) if hasattr(gp, "__getitem__") else float(gp)
        progress = np.array(timestamps) / max(timestamps[-1], 1e-6)
        gripper_positions = (cur_gp + (gripper - cur_gp) * progress).tolist()

    # Execute
    print(f"  [{label}] Executing ({timestamps[-1]:.2f}s)...")
    try:
        move_joint_keypoints(
            side=arm,
            timestamps=timestamps,
            joint_positions=positions.tolist(),
            gripper_positions=gripper_positions,
        )
    except RuntimeError as e:
        print(f"  [{label}] Execution failed: {e}")
        return False
    print(f"  [{label}] Done.")
    return True


# ---------------------------------------------------------------------------
# 4. Open gripper
# ---------------------------------------------------------------------------
print("\nStep 1: Opening gripper...")
set_gripper(arm, 1.0)
time.sleep(0.5)

# ---------------------------------------------------------------------------
# 5. Move above object (10cm hover)
# ---------------------------------------------------------------------------
print("\nStep 2: Moving above object...")
hover = obj_pos.copy()
hover[2] += 0.10
plan_and_move(hover, ee_quat, gripper=1.0, label="hover")

# ---------------------------------------------------------------------------
# 6. Lower to object
# ---------------------------------------------------------------------------
print("\nStep 3: Lowering to object...")
state2 = get_robot_state()
task_info2 = get_task_info()
obj_now = np.array(task_info2.get("obj_pos", obj_pos))
plan_and_move(obj_now, state2.arms[arm].ee_quat, label="lower")

# ---------------------------------------------------------------------------
# 7. Close gripper
# ---------------------------------------------------------------------------
print("\nStep 4: Closing gripper...")
set_gripper(arm, 0.0)
time.sleep(0.5)

# ---------------------------------------------------------------------------
# 8. Lift
# ---------------------------------------------------------------------------
print("\nStep 5: Lifting...")
state3 = get_robot_state()
lift_pos = np.array(state3.arms[arm].ee_pos)
lift_pos[2] += 0.20
plan_and_move(lift_pos, state3.arms[arm].ee_quat, gripper=0.0, label="lift")

# ---------------------------------------------------------------------------
# 9. Check result
# ---------------------------------------------------------------------------
task_final = get_task_info()
obj_final = np.array(task_final.get("obj_pos", obj_pos))
lifted = obj_final[2] - obj_pos[2]
print(f"\nObject lifted: {lifted:.3f}m")
print(f"{'PICK SUCCESS!' if lifted > 0.05 else 'PICK FAILED'}")
