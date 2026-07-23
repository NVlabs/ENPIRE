# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Turn on a microwave in RoboCasa.
# Run with: uv run run_script.py --file robocasa_turn_on_microwave.py --env robocasa:TurnOnMicrowave
#
# Strategy: close gripper → approach button from front → press into it → retract.
# The microwave button/handle is at obj_pos from get_task_info().

import numpy as np
from scipy.spatial.transform import Rotation

# Get state and target
state = get_robot_state()
arm = list(state.arms.keys())[0]
ee_pos = np.array(state.arms[arm].ee_pos)
ee_quat = state.arms[arm].ee_quat
print(f"Arm: {arm}")
print(f"EE: {[round(float(x), 3) for x in ee_pos]}")

task_info = get_task_info()
obj_pos = np.array(task_info.get("obj_pos", ee_pos))
print(f"Button/handle pos: {[round(float(x), 3) for x in obj_pos]}")

# Step 1: Close gripper (use fingertip to press button)
print("Closing gripper...")
set_gripper(arm, 0.0)

# Step 2: Compute approach direction — from EEF toward button
# obj_to_eef tells us which direction the arm is relative to the button
eef_to_obj = obj_pos - ee_pos
direction = eef_to_obj / (np.linalg.norm(eef_to_obj) + 1e-6)  # unit vector EEF→button
print(f"EEF→button direction: {[round(float(x), 3) for x in direction]}")

# Move to 6cm before the button along approach direction
approach = obj_pos - direction * 0.06
print(f"Approach: {[round(float(x), 3) for x in approach]}")
try:
    ee_rpy = list(Rotation.from_quat(get_robot_state().arms[arm].ee_quat).as_euler('xyz', degrees=True))
    freespace_move(**{f"{arm}_target_pos": list(approach), f"{arm}_target_rpy": ee_rpy})
except Exception as e:
    print(f"Approach warn: {e}")

# Step 3: Press — push 3cm past the button along approach direction
print("Pressing button...")
press_pos = obj_pos + direction * 0.03
try:
    ee_rpy2 = list(Rotation.from_quat(get_robot_state().arms[arm].ee_quat).as_euler('xyz', degrees=True))
    freespace_move(**{f"{arm}_target_pos": list(press_pos), f"{arm}_target_rpy": ee_rpy2})
except Exception as e:
    print(f"Press warn: {e}")

# Step 4: Hold briefly (let sim register the press)
import time

time.sleep(0.5)

# Step 5: Retract — pull back along approach direction
print("Retracting...")
retract_state = get_robot_state()
retract = np.array(retract_state.arms[arm].ee_pos)
retract -= direction * 0.12  # pull back from microwave
try:
    ee_rpy3 = list(Rotation.from_quat(retract_state.arms[arm].ee_quat).as_euler('xyz', degrees=True))
    freespace_move(**{f"{arm}_target_pos": list(retract), f"{arm}_target_rpy": ee_rpy3})
except Exception as e:
    print(f"Retract warn: {e}")

# Check result
result = get_task_info()
print(f"\nSuccess: {result.get('success', False)}")
print(f"Reward: {result.get('reward', 0.0)}")
