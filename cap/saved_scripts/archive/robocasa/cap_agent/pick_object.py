# Pick up an object in RoboCasa using CAP tools.
# Run with: cap_server --env robocasa:PickPlaceCounterToCabinet
#
# Uses: get_robot_state(), freespace_move(), set_gripper(), get_task_info()

import numpy as np
from scipy.spatial.transform import Rotation

# Get initial state
state = get_robot_state()
arm = list(state.arms.keys())[0]  # use first available arm
ee_pos = state.arms[arm].ee_pos
ee_quat = state.arms[arm].ee_quat
print(f"Arm: {arm}")
print(f"EE position: {[round(x, 3) for x in ee_pos]}")

# Get object position from task info
task_info = get_task_info()
obj_pos = np.array(task_info.get("obj_pos", ee_pos))
print(f"Object position: {[round(float(x), 3) for x in obj_pos]}")

# Step 1: Open gripper
print("Opening gripper...")
set_gripper(arm, 1.0)

# Step 2: Move above object (10cm hover)
print("Moving above object...")
hover = obj_pos.copy()
hover[2] += 0.10
ee_rpy = list(Rotation.from_quat(ee_quat).as_euler('xyz', degrees=True))
freespace_move(**{f"{arm}_target_pos": list(hover), f"{arm}_target_rpy": ee_rpy})

# Step 3: Lower to object center
print("Lowering to object...")
state2 = get_robot_state()
task_info2 = get_task_info()
obj_now = np.array(task_info2.get("obj_pos", obj_pos))
ee_rpy2 = list(Rotation.from_quat(state2.arms[arm].ee_quat).as_euler('xyz', degrees=True))
freespace_move(**{f"{arm}_target_pos": list(obj_now), f"{arm}_target_rpy": ee_rpy2})

# Step 4: Close gripper
print("Closing gripper...")
set_gripper(arm, 0.0)

# Step 5: Lift
print("Lifting...")
state3 = get_robot_state()
lift_pos = np.array(state3.arms[arm].ee_pos)
lift_pos[2] += 0.20
ee_rpy3 = list(Rotation.from_quat(state3.arms[arm].ee_quat).as_euler('xyz', degrees=True))
freespace_move(**{f"{arm}_target_pos": list(lift_pos), f"{arm}_target_rpy": ee_rpy3})

# Check result
task_final = get_task_info()
obj_final = np.array(task_final.get("obj_pos", obj_pos))
lifted = obj_final[2] - obj_pos[2]
print(f"Object lifted: {lifted:.3f}m")
print(f"{'PICK SUCCESS!' if lifted > 0.05 else 'PICK FAILED'}")
