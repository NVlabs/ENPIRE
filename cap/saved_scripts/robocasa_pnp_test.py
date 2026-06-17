import numpy as np

SIDE = "right"

# Get task info
info = get_task_info()
obj_pos = np.array(info["obj_pos"])
container_pos = np.array(info["container_pos"])
print(f"Object (lemon_wedge): {[round(float(x), 3) for x in obj_pos]}")
print(f"Container (plate):    {[round(float(x), 3) for x in container_pos]}")

# Step 1: Open gripper fully
print("\n--- Step 1: Open gripper ---")
open_gripper(SIDE)

# Step 2: Hover above object (12cm clearance for sink)
print("\n--- Step 2: Hover above object ---")
hover = obj_pos.copy()
hover[2] += 0.12
result = freespace_move(right_target_pos=hover.tolist(), side=SIDE)
print(f"Hover status: {result.status}")

# Step 3: Descend INTO the object — no Z offset for this small flat wedge
print("\n--- Step 3: Descend to object ---")
obj_now = np.array(get_task_info()["obj_pos"])
grasp_target = obj_now.copy()
# No Z offset — go right to the object center for a small wedge
result = freespace_move(right_target_pos=grasp_target.tolist(), side=SIDE)
print(f"Descend status: {result.status}")

# Step 4: Close gripper
print("\n--- Step 4: Close gripper ---")
close_gripper(SIDE, compliant=True, hold_strength=0.2)

# Verify grasp
state2 = get_robot_state()
grip_val = float(state2.arms[SIDE].gripper_pos)
grasped = grip_val > 0.005
print(f"Gripper width: {grip_val:.4f} -> {'GRASPED' if grasped else 'MISSED'}")

# If missed, nudge down and retry
if not grasped:
    print("\n--- Retry: nudge down 2cm and re-grasp ---")
    open_gripper(SIDE)
    nudge(side=SIDE, delta_pos=[0.0, 0.0, -0.02])
    close_gripper(SIDE, compliant=True, hold_strength=0.2)
    state3 = get_robot_state()
    grip_val = float(state3.arms[SIDE].gripper_pos)
    grasped = grip_val > 0.005
    print(
        f"Retry gripper width: {grip_val:.4f} -> {'GRASPED' if grasped else 'MISSED again'}"
    )

if not grasped:
    # Second retry — nudge down another 2cm
    print("\n--- Retry 2: nudge down another 2cm ---")
    open_gripper(SIDE)
    nudge(side=SIDE, delta_pos=[0.0, 0.0, -0.02])
    close_gripper(SIDE, compliant=True, hold_strength=0.2)
    state4 = get_robot_state()
    grip_val = float(state4.arms[SIDE].gripper_pos)
    grasped = grip_val > 0.005
    print(
        f"Retry 2 gripper width: {grip_val:.4f} -> {'GRASPED' if grasped else 'GIVING UP'}"
    )

if not grasped:
    print("All grasp attempts failed. Going home.")
    open_gripper(SIDE)
    go_home(side=SIDE)
else:
    # Step 5: Lift out of sink
    print("\n--- Step 5: Lift ---")
    lift_pos = np.array(get_robot_state().arms[SIDE].ee_pos)
    lift_pos[2] += 0.20
    result = freespace_move(right_target_pos=lift_pos.tolist(), side=SIDE)
    print(f"Lift status: {result.status}")

    # Step 6: Move above plate
    print("\n--- Step 6: Move above plate ---")
    place_hover = container_pos.copy()
    place_hover[2] += 0.12
    result = freespace_move(right_target_pos=place_hover.tolist(), side=SIDE)
    print(f"Place hover status: {result.status}")

    # Step 7: Lower to plate surface
    print("\n--- Step 7: Lower to plate ---")
    place_target = container_pos.copy()
    place_target[2] += 0.03  # just above plate
    result = freespace_move(right_target_pos=place_target.tolist(), side=SIDE)
    print(f"Place lower status: {result.status}")

    # Step 8: Release
    print("\n--- Step 8: Release ---")
    open_gripper(SIDE)

    # Step 9: Retract (conservative height)
    print("\n--- Step 9: Retract ---")
    go_home(SIDE)

    # Check success
    final = get_task_info()
    print(f"\nSuccess: {final.get('success', False)}")
    print(f"Reward:  {final.get('reward', 0.0)}")
