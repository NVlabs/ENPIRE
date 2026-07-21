# RoboCasa Pick and Place

Strategy guide for `PickPlaceCounterToCabinet`, `PickPlaceSinkToCounter`, and similar pick-and-place tasks.

## Key facts

- **Single arm** — use `list(state.arms.keys())[0]` to get the arm name (always `"right"` for PandaOmron).
- **Ground-truth positions** — `get_task_info()` provides oracle XYZ for objects and targets:
  - `obj_pos` — object to pick
  - `obj_name` — object name (for vision/grasp queries)
  - `container_pos` — target container (preferred place target)
  - `distr_counter_pos` — target counter surface (fallback place target)
  - `distr_cab_pos` — target cabinet (for counter-to-cabinet tasks)
- **Place target resolution** — use `container_pos` if present, else `distr_counter_pos` or `distr_cab_pos` depending on the task variant. The `distr_` prefix is RoboCasa naming — it IS the correct destination.
- **Success signal** — `get_task_info()["success"]` is `True` when the task is complete.
- **Quaternion** — always reuse the arm's current `ee_quat` from `get_robot_state()`. Never compute quaternions manually.
- **Motion** — use `freespace_move()` for ALL arm movement. It plans collision-free trajectories via cuRobo automatically. No external planner setup needed.
- **`set_gripper` works** — call it as a separate step before or after motion.

## AnyGrasp: learned grasp pose generation

When the object shape is non-trivial (not a simple cube/cylinder), use AnyGrasp to find feasible grasp poses instead of going straight to the oracle object position.

- **Call**: `sample_grasp_pose_anygrasp(object_name, camera, max_grasps=8)`
- **Camera fallback** — try wrist camera first (closer, better depth), fall back to top camera if wrist returns 0 candidates.
- **Grasp orientation** — AnyGrasp returns RPY in display convention. Convert to quaternion with: `euler_xyz = [-pitch, roll, -yaw - 90]` then `Rotation.from_euler("xyz", radians).as_quat()`.

## Pick and Place strategy

```python
import numpy as np

state = get_robot_state()
arm = list(state.arms.keys())[0]

task_info = get_task_info()
obj_pos = np.array(task_info["obj_pos"])

# 1. Open gripper
set_gripper(arm, 1.0)

# 2. Hover 12cm above object
hover = obj_pos.copy(); hover[2] += 0.12
freespace_move(right_target_pos=hover.tolist(), side=arm)

# 3. Descend to object (refresh position in case it moved)
obj_pos = np.array(get_task_info()["obj_pos"])
freespace_move(right_target_pos=obj_pos.tolist(), side=arm)

# 4. Close gripper — prefer compliant=True for stall-detect + force cap.
#    hold_strength controls sustained clamp force (0.4 ≈ 10 N balanced).
close_gripper(arm, compliant=True, hold_strength=0.4)

# 5. Verify grasp via structured gripper info (not raw gripper_pos).
info = get_gripper_info(arm)
grasped = (
    info["has_object"] is True
    and not info["is_fully_closed"]
    and (info["actuator_force_N"] or 0) > 1.0
)
if not grasped:
    print(f"Grasp missed — {info}")

# 6. Lift clear
lift = np.array(get_robot_state().arms[arm].ee_pos); lift[2] += 0.25
freespace_move(right_target_pos=lift.tolist(), side=arm)

# 7. Resolve place target (priority: container > counter > cabinet)
info = get_task_info()
if "container_pos" in info:
    place_pos = np.array(info["container_pos"])
elif "distr_counter_pos" in info:
    place_pos = np.array(info["distr_counter_pos"])
else:
    place_pos = np.array(info["distr_cab_pos"])

# 8. Move above place target
approach = place_pos.copy(); approach[2] += 0.15
result = freespace_move(right_target_pos=approach.tolist(), side=arm)

# 9. Lower to place
freespace_move(right_target_pos=place_pos.tolist(), side=arm)

# 10. Open gripper to release
set_gripper(arm, 1.0)

# 11. Retract — go_home is the most reliable retract (RoboCasa requires
#     gripper >25cm from object for success, so a small upward move is not enough)
go_home(arm)

# 12. Check success
print(f"Success: {get_task_info()['success']}")
```

## Reference: oracle-position pick-place with nudge-based retry (`robocasa_pnp_test.py`)

Tested end-to-end on `PickPlaceSinkToCounter`. Uses oracle object/container
positions (no vision), the compliant close with low clamp force, and a
two-tier nudge-down retry loop for missed grasps. Good starting template for
simple reach-and-grasp tasks where the oracle pose is known and the object
shape is cube-ish or wedge-ish.

```python
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

# Step 4: Close gripper (compliant: stall-detect + capped clamp force so
# tapered objects don't squirt out of a max-force parallel grip)
print("\n--- Step 4: Close gripper ---")
close_gripper(SIDE, compliant=True, hold_strength=0.2)

# Verify grasp
state2 = get_robot_state()
grip_val = float(state2.arms[SIDE].gripper_pos)
grasped = grip_val > 0.005
print(f"Gripper width: {grip_val:.4f} -> {'GRASPED' if grasped else 'MISSED'}")

# If missed, nudge down 2cm and retry (up to two re-grasps)
if not grasped:
    print("\n--- Retry: nudge down 2cm and re-grasp ---")
    open_gripper(SIDE)
    nudge(side=SIDE, delta_pos=[0.0, 0.0, -0.02])
    close_gripper(SIDE, compliant=True, hold_strength=0.2)
    state3 = get_robot_state()
    grip_val = float(state3.arms[SIDE].gripper_pos)
    grasped = grip_val > 0.005
    print(f"Retry gripper width: {grip_val:.4f} -> {'GRASPED' if grasped else 'MISSED again'}")

if not grasped:
    print("\n--- Retry 2: nudge down another 2cm ---")
    open_gripper(SIDE)
    nudge(side=SIDE, delta_pos=[0.0, 0.0, -0.02])
    close_gripper(SIDE, compliant=True, hold_strength=0.2)
    state4 = get_robot_state()
    grip_val = float(state4.arms[SIDE].gripper_pos)
    grasped = grip_val > 0.005
    print(f"Retry 2 gripper width: {grip_val:.4f} -> {'GRASPED' if grasped else 'GIVING UP'}")

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

    # Step 9: Retract via go_home (RoboCasa requires gripper >25cm from object)
    print("\n--- Step 9: Retract ---")
    go_home(SIDE)

    # Check success
    final = get_task_info()
    print(f"\nSuccess: {final.get('success', False)}")
    print(f"Reward:  {final.get('reward', 0.0)}")
```

Tuning knobs if this script doesn't land on the first try:

- **Grasp clearance**: the script descends to the object centre with no Z
  offset. Fine for small flat wedges in a sink basin; add `grasp_target[2] += 0.005`
  for objects that sit on a surface the gripper tips would otherwise hit.
- **Clamp force** (`hold_strength`): raise toward `0.6` if the object slips
  during the lift / carry; lower toward `0.1` if it squirts out of the grip
  at close time (wedge geometry).
- **Retry strategy**: the 2 cm nudge-down is tuned for objects partially
  obscured by surface geometry. Swap for a lateral nudge (`delta_pos=[0.0,
  0.02, 0.0]`) or a fresh hover + detect if the miss direction is lateral.

## Reference: full pick-and-place with AnyGrasp (may need adaptation)

This is a tested script that uses oracle positions + AnyGrasp for grasp planning.
It may not run as-is in the agent — adapt the patterns to available tools.

```python
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

SIDE = "right"
CAMERAS = ["wrist", "top"]  # try wrist first, fallback to top
MAX_GRASPS = 8
HOVER_HEIGHT = 0.12  # metres above grasp/place point

def rpy_deg_to_quat(rpy_deg):
    """Convert AnyGrasp display RPY (degrees) to quaternion xyzw."""
    roll, pitch, yaw = [float(x) for x in rpy_deg]
    euler_xyz_deg = [-pitch, roll, -yaw - 90.0]
    return R.from_euler("xyz", np.deg2rad(euler_xyz_deg)).as_quat().tolist()

# --- Step 1: Get task info ---
info = get_task_info()
obj_pos_oracle = np.array([float(x) for x in info["obj_pos"]])
pick_name = info.get("obj_name", "object")

if "container_pos" in info:
    place_pos = np.array([float(x) for x in info["container_pos"]])
elif "distr_counter_pos" in info:
    place_pos = np.array([float(x) for x in info["distr_counter_pos"]])
else:
    place_pos = obj_pos_oracle + np.array([0.15, 0.0, 0.05])

# --- Step 2: AnyGrasp — find feasible grasp (wrist → top fallback) ---
grasp = None
used_camera = None
for cam in CAMERAS:
    grasps = sample_grasp_pose_anygrasp(
        object_name=pick_name, camera=cam, max_grasps=MAX_GRASPS,
    )
    if not grasps:
        continue
    # Pick best-scoring grasp — check freespace_move feasibility if needed
    grasp = grasps[0]
    used_camera = cam
    break

if grasp is None:
    print("No feasible grasp found — going home.")
    go_home()
    raise RuntimeError("ABORT")

grasp_pos = np.array(grasp.position)
grasp_quat = rpy_deg_to_quat(grasp.rpy)

# --- Step 3: Pick ---
set_gripper(SIDE, 1.0)
time.sleep(0.3)

# Move to grasp pose
result = freespace_move(right_target_pos=grasp_pos.tolist(),
                        right_target_quat=grasp_quat, side=SIDE)
if result.status != "Success":
    raise RuntimeError(f"Cannot reach grasp: {result.status}")

# Close gripper
set_gripper(SIDE, 0.0)
time.sleep(0.5)

# Verify grasp
state2 = get_robot_state()
grip_val = float(state2.arms[SIDE].gripper_pos)
grasped = grip_val > 0.005
print(f"Gripper width: {grip_val:.4f} -> {'GRASPED' if grasped else 'MISSED'}")

if not grasped:
    set_gripper(SIDE, 1.0)
    go_home()
    raise RuntimeError("Grasp failed")

# --- Step 4: Lift ---
state3 = get_robot_state()
lift_pos = np.array(state3.arms[SIDE].ee_pos)
lift_quat = state3.arms[SIDE].ee_quat
lift_pos[2] += HOVER_HEIGHT
freespace_move(right_target_pos=lift_pos.tolist(),
               right_target_quat=np.array(lift_quat).tolist(), side=SIDE)

# --- Step 5: Place ---
place_hover = place_pos.copy()
place_hover[2] += HOVER_HEIGHT
freespace_move(right_target_pos=place_hover.tolist(),
               right_target_quat=np.array(lift_quat).tolist(), side=SIDE)

freespace_move(right_target_pos=place_pos.tolist(),
               right_target_quat=np.array(lift_quat).tolist(), side=SIDE)

# Release
set_gripper(SIDE, 1.0)
time.sleep(0.3)

# Retract — go_home moves gripper >25cm from object (required for success)
go_home(SIDE)

# --- Step 6: Check ---
print(f"Success: {get_task_info()['success']}")
print(f"Reward: {get_task_info()['reward']}")
```

## Common failure modes

- **Success=False despite object on plate** — RoboCasa requires gripper **>25cm from object** for success. Always call `go_home()` after releasing — a small upward retract is not enough.
- **Gripper misses object** — object moved between hover and descend; always re-call `get_task_info()["obj_pos"]` just before the final descent.
- **Arm collides with counter** — add at least 10–12cm hover before descending.
- **Object drops on retract** — call `set_gripper(arm, 0.0)` again after lifting if `get_task_info()["success"]` is still False.
- **freespace_move reports "Success" but EE didn't reach target** — in constrained spaces (sinks, cabinets), `freespace_move` may report `status="Success"` with low `final_pos_error_m` even when the EE is far from the target. Always verify reach by checking `get_task_info()["obj_to_robot0_eef_pos"]` — if the Z component is large (>0.05m), the arm didn't actually get there. Use `nudge()` for the final descent or try a different approach orientation.
- **freespace_move returns status != "Success"** — target may be in collision or unreachable. Try a higher hover height, or `go_home()` first to reset the arm.
- **AnyGrasp returns 0 candidates** — try a different camera (wrist vs top), or fall back to the direct oracle position approach.
- **Object detection fails due to language mismatch** — perception tools (AnyGrasp, `detect_object`) rely on text queries. Try rephrasing with different descriptions. For example, for `lime_wedge`, try `green slice`, `citrus wedge`, etc.
- **All AnyGrasp grasps fail cuRobo IK** — the object may be in a pose where no kinematically reachable grasp exists. Try `go_home()` first to reset the arm, then replan.
