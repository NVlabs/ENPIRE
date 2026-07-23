# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa pick-and-place demo with full vision tool stack.

Tools used:
  - vlm_query                  → Gemini scene understanding from camera images
  - sample_grasp_pose_anygrasp → SAM3 segmentation + AnyGrasp grasp planning
  - detect_objects_oneshot     → BundleSDF 6-DOF pose estimation
  - get_robot_state            → EE pose and gripper state
  - get_task_info              → ground-truth object positions, reward, success
  - freespace_move              → motion via env's native controller
  - open_gripper / close_gripper
  - go_home
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIDE = "right"
VLM_BACKEND = "gemini"
VLM_MODEL = "gemini-2.5-flash"
CAMERA = "top"
MOVE_TOL = 0.02
MOVE_TIMEOUT = 10.0
HOVER_Z = 0.12
LIFT_Z = 0.15
MAX_GRASPS = 8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vlm(prompt, cameras="top"):
    media = [f"camera:{c.strip()}" for c in cameras.split(",")]
    return vlm_query(text=prompt, backend=VLM_BACKEND, model=VLM_MODEL, media=media)


def ee_state():
    s = get_robot_state()
    pos = list(getattr(s, f"{SIDE}_ee_pos"))
    rpy = list(getattr(s, f"{SIDE}_ee_rpy"))
    grip = float(getattr(s, f"{SIDE}_gripper_pos"))
    return pos, rpy, grip


def move(pos, rpy=None, tol=MOVE_TOL, timeout=MOVE_TIMEOUT):
    if rpy is None:
        _, rpy, _ = ee_state()
    return freespace_move(
        right_target_pos=[float(x) for x in pos],
        right_target_rpy=[float(x) for x in rpy],
        planning_speed=1.5,
    )


# ===================================================================
print("=" * 60)
print("STEP 1 — VLM: What's in the scene?")
print("=" * 60)

scene = vlm("Briefly list every object you see on the kitchen counter.", cameras=CAMERA)
print(f"  {scene}\n")

info = get_task_info()
print(f"  Task: {info.get('env_name')}")
gt_keys = [k for k in info if k.endswith("_pos") and "robot" not in k and "eef" not in k]
for k in gt_keys:
    print(f"    {k}: {[round(float(x), 3) for x in info[k]]}")

# ===================================================================
print("\n" + "=" * 60)
print("STEP 2 — VLM: What should the robot pick?")
print("=" * 60)

pick_name = vlm(
    "You are controlling a robot arm in a kitchen. "
    "What is the single main object to pick up from the counter? "
    "Reply ONLY with the object name, nothing else."
).strip().strip('"\'')
print(f"  Pick target: '{pick_name}'")

# ===================================================================
print("\n" + "=" * 60)
print("STEP 3 — AnyGrasp: Plan grasps")
print("=" * 60)

grasps = sample_grasp_pose_anygrasp(
    object_name=pick_name, camera=CAMERA, max_grasps=MAX_GRASPS,
)
print(f"  {len(grasps)} grasp candidates:")
for i, g in enumerate(grasps[:5]):
    print(f"    [{i}] xyz={[round(float(x), 3) for x in g.position]} "
          f"rpy={[round(float(x), 1) for x in g.rpy]} score={g.score:.3f}")

grasp = grasps[0]
grasp_pos = [float(x) for x in grasp.position]
grasp_rpy = [float(x) for x in grasp.rpy]

# ===================================================================
print("\n" + "=" * 60)
print("STEP 4 — BundleSDF: Where to place?")
print("=" * 60)

place_name = vlm(
    "The robot picked up an object from the counter. "
    "Where is the target location to place it? "
    "Reply ONLY with the target name (e.g. 'cabinet', 'shelf')."
).strip().strip('"\'')
print(f"  VLM place target: '{place_name}'")

place_pos = None
try:
    dets = detect_objects_oneshot(place_name, camera=CAMERA)
    det_list = dets.get(place_name, [])
    if det_list and det_list[0].position_3d:
        place_pos = [float(x) for x in det_list[0].position_3d]
        print(f"  BundleSDF: '{place_name}' at {[round(x, 3) for x in place_pos]}")
except Exception as e:
    print(f"  BundleSDF failed: {e}")

if place_pos is None:
    place_pos = [grasp_pos[0] + 0.15, grasp_pos[1], grasp_pos[2] + 0.05]
    print(f"  Fallback place pos: {[round(x, 3) for x in place_pos]}")

# ===================================================================
print("\n" + "=" * 60)
print("STEP 5 — Pick")
print("=" * 60)

pos, rpy, grip = ee_state()
print(f"  EE now: xyz={[round(float(x), 3) for x in pos]} grip={grip:.2f}")

open_gripper(SIDE)

hover = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + HOVER_Z]
print(f"  → Hover: {[round(x, 3) for x in hover]}")
move(hover, grasp_rpy)

print(f"  → Descend: {[round(x, 3) for x in grasp_pos]}")
move(grasp_pos, grasp_rpy, tol=0.01, timeout=12.0)

print("  → Close gripper")
close_gripper(SIDE)

_, _, grip_after = ee_state()
grasped = grip_after > 0.005
print(f"  Gripper: {grip_after:.4f} → {'GRASPED' if grasped else 'MISSED'}")

if not grasped:
    print("  Grasp failed.")
    open_gripper(SIDE)
    go_home()
else:
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 6 — Lift and Place")
    print("=" * 60)

    cur, cur_rpy, _ = ee_state()
    lift = [float(cur[0]), float(cur[1]), float(cur[2]) + LIFT_Z]
    print(f"  → Lift: {[round(x, 3) for x in lift]}")
    move(lift, cur_rpy)

    place_hover = [place_pos[0], place_pos[1], place_pos[2] + HOVER_Z]
    print(f"  → Place hover: {[round(float(x), 3) for x in place_hover]}")
    move(place_hover)

    print(f"  → Lower: {[round(float(x), 3) for x in place_pos]}")
    move(place_pos, tol=0.03, timeout=12.0)

    print("  → Release")
    open_gripper(SIDE)

    retract = [place_pos[0], place_pos[1], place_pos[2] + HOVER_Z]
    move(retract)

    go_home()

    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 7 — Result")
    print("=" * 60)

    final = get_task_info()
    print(f"  Done:    {final.get('done')}")
    print(f"  Success: {final.get('success')}")
    print(f"  Reward:  {final.get('reward')}")

    verdict = vlm("Did the robot successfully move the object? What do you see now?")
    print(f"  VLM: {verdict}")

print("\n" + "=" * 60)
print("Demo complete!")
print("=" * 60)
