# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pick and place in RoboCasa using the full vision + cuRobo tool pipeline.
#
# Tools used:
#   - create_motion_planner      — cuRobo Panda planner (sandbox API)
#   - update_planner_world       — load kitchen collision geometry into planner
#   - vlm_query                  — Gemini scene understanding
#   - detect_objects_oneshot     — BundleSDF 6-DOF pose estimation
#   - sample_grasp_pose_anygrasp — SAM3 segmentation + AnyGrasp grasp planning
#   - get_robot_state            — EE pose, joint state, gripper
#   - get_task_info              — reward, success, object ground truth
#   - set_gripper                — gripper control
#   - go_home                    — return to start pose
#   - move_joint_keypoints       — joint trajectory execution
#   - get_camera_image           — camera RGB frames
#   - get_camera_intrinsics      — camera [fx, fy, cx, cy]
#   - get_camera_extrinsics      — camera rotation, position
#
# Run from web UI or via run_script.py:
#   ROBOCASA_CONTROLLER_TYPE=joint_position CAP_ROBOT_TYPE=panda \
#   ROBOCASA_LAYOUT_ID=3 ROBOCASA_STYLE_ID=5 CAP_AGENT_NAME=demo \
#   uv run python -u run_script.py \
#   --file robocasa_vision_curobo_pick.py \
#   --env "robocasa:PickPlaceCounterToCabinet" --cap-port 18600 --no-log --record
#
# Requires:
#   - CUDA GPU for cuRobo (local or via portal)
#   - Remote vision servers: SAM3 (:6767), AnyGrasp (:8122), BundleSDF (:8119)
#   - GEMINI_API_KEY for VLM

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIDE = "right"
VLM_BACKEND = "gemini"
VLM_MODEL = "gemini-2.5-flash"
CAMERA = "top"
MAX_GRASPS = 8
HOVER_HEIGHT = 0.12       # metres above grasp/place point
CUROBO_SPEED = "slow"     # "slow" = more seeds, better quality
JOINT_SPEED = 1.0         # rad/s max joint velocity for timestamp generation

# ---------------------------------------------------------------------------
# Init cuRobo planner (via sandbox API — no direct imports needed)
# ---------------------------------------------------------------------------

print("Initialising cuRobo Panda planner...")
planner = create_motion_planner(solver_speed=CUROBO_SPEED)
print("cuRobo ready.\n")

# ---------------------------------------------------------------------------
# Load collision world + get base frame transform
# ---------------------------------------------------------------------------

print("Loading kitchen obstacles into cuRobo...")
world_info = update_planner_world(planner)
base_pos = np.array(world_info["base_pos"], dtype=np.float64)
base_quat = np.array(world_info["base_quat_xyzw"], dtype=np.float64)
R_base = R.from_quat(base_quat)
print(f"Base pos: {[round(x, 3) for x in base_pos]}")
print(f"Loaded {world_info['n_obstacles']} obstacles")


def world_to_base(pos, quat_xyzw):
    pos_b = R_base.inv().apply(np.asarray(pos) - base_pos)
    quat_b = (R_base.inv() * R.from_quat(quat_xyzw)).as_quat()
    return pos_b, quat_b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vlm(prompt, camera=CAMERA):
    return vlm_query(
        text=prompt, backend=VLM_BACKEND, model=VLM_MODEL,
        media=[f"camera:{camera}"],
    )


def rpy_deg_to_quat(rpy_deg):
    """Convert display RPY (degrees) from AnyGrasp to quaternion xyzw.

    Display RPY uses a non-standard convention matching freespace_move:
      euler_xyz = [-pitch, roll, -yaw - 90]
    """
    roll, pitch, yaw = [float(x) for x in rpy_deg]
    euler_xyz_deg = [-pitch, roll, -yaw - 90.0]
    return R.from_euler("xyz", np.deg2rad(euler_xyz_deg)).as_quat().tolist()


def plan_and_move(target_pos_world, target_quat_world, gripper=None, label="move"):
    """Plan with cuRobo in base frame, execute via move_joint_keypoints."""
    cur = get_robot_state()
    cur_jp = np.array(cur.arms[SIDE].joint_pos)

    target_pos_b, target_quat_b = world_to_base(target_pos_world, target_quat_world)
    print(f"  [{label}] target(world): {[round(float(x), 3) for x in target_pos_world]}")

    result = planner.plan_to_pose(
        current_left_jp=np.zeros(0),
        current_right_jp=cur_jp,
        target_right_pos=target_pos_b,
        target_right_quat_xyzw=target_quat_b,
        side="right",
    )
    status = result["status"]
    if status != "Success":
        print(f"  [{label}] cuRobo FAILED: {status} — {result.get('status_detail', '')}")
        return False

    positions = result["right_positions"]  # (N, 7)
    n_steps = positions.shape[0]
    print(f"  [{label}] planned {n_steps} waypoints")

    # Constant-velocity timestamps
    timestamps = [0.0]
    for i in range(n_steps - 1):
        dt = max(0.01, float(np.max(np.abs(positions[i + 1] - positions[i]))) / JOINT_SPEED)
        timestamps.append(timestamps[-1] + dt)

    # Gripper trajectory (linear interpolation)
    gripper_positions = None
    if gripper is not None:
        gp = cur.arms[SIDE].gripper_pos
        cur_gp = float(gp[0]) if hasattr(gp, "__getitem__") else float(gp)
        progress = np.array(timestamps) / max(timestamps[-1], 1e-6)
        gripper_positions = (cur_gp + (gripper - cur_gp) * progress).tolist()

    try:
        move_joint_keypoints(
            side=SIDE,
            timestamps=timestamps,
            joint_positions=positions.tolist(),
            gripper_positions=gripper_positions,
        )
    except RuntimeError as e:
        print(f"  [{label}] execution failed: {e}")
        return False
    print(f"  [{label}] done ({timestamps[-1]:.1f}s)")
    return True


# ===================================================================
print("=" * 60)
print("STEP 1 — VLM: Understand the scene")
print("=" * 60)

scene = vlm("Briefly list every object you see on the kitchen counter.")
print(f"  VLM: {scene[:200]}\n")

info = get_task_info()
print(f"  Task: {info.get('env_name')}")
gt_keys = [k for k in info if k.endswith("_pos") and "robot" not in k and "eef" not in k]
for k in gt_keys:
    print(f"    {k}: {[round(float(x), 3) for x in info[k]]}")

# ===================================================================
print("\n" + "=" * 60)
print("STEP 2 — VLM: Identify pick target")
print("=" * 60)

pick_name = vlm(
    "You are controlling a robot arm in a kitchen. "
    "What is the single main object to pick up from the counter? "
    "Reply ONLY with the object name, nothing else."
).strip().strip('"\'')

pick_name = "yellow mustard bottle"
print(f"  Pick target: '{pick_name}'")

# ===================================================================
print("\n" + "=" * 60)
print("STEP 3 — AnyGrasp: Plan grasps on '{}'".format(pick_name))
print("=" * 60)

grasps = sample_grasp_pose_anygrasp(
    object_name=pick_name, camera=CAMERA, max_grasps=MAX_GRASPS,
)
print(f"  {len(grasps)} grasp candidates:")
for i, g in enumerate(grasps[:5]):
    print(f"    [{i}] xyz={[round(float(x), 3) for x in g.position]} "
          f"rpy={[round(float(x), 1) for x in g.rpy]} score={g.score:.3f}")

# ---------------------------------------------------------------------------
# STEP 3b — Rank grasps through cuRobo (test each candidate for reachability)
# ---------------------------------------------------------------------------
print(f"\n  Ranking {len(grasps)} grasps through cuRobo...")

cur_state = get_robot_state()
cur_jp = np.array(cur_state.arms[SIDE].joint_pos)

grasp = None
grasp_traj = None
feasible = []
for i, g in enumerate(grasps):
    g_pos_b, g_quat_b = world_to_base(g.position, rpy_deg_to_quat(g.rpy))
    result = planner.plan_to_pose(
        current_left_jp=np.zeros(0),
        current_right_jp=cur_jp,
        target_right_pos=g_pos_b,
        target_right_quat_xyzw=g_quat_b,
        side="right",
    )
    ok = result["status"] == "Success"
    feasible.append(ok)
    n_wp = result["right_positions"].shape[0] if ok else 0
    print(f"    [{i}] {'OK' if ok else 'FAIL':4s}  score={g.score:.3f}  "
          f"xyz={[round(float(x), 3) for x in g.position]}  waypoints={n_wp}")
    if ok and grasp is None:
        grasp = g
        grasp_traj = result["right_positions"]

# ---------------------------------------------------------------------------
# Visualize grasps on camera image (green=feasible, red=failed) → saved to vis/
# ---------------------------------------------------------------------------
import cv2
from enpire.env.forge.cap.agent.tools._artifact_log import log_image

rgb = get_camera_image(CAMERA)
K = get_camera_intrinsics(CAMERA)
extr = get_camera_extrinsics(CAMERA)
R_cam = np.array(extr["rotation"]).reshape(3, 3)
t_cam = np.array(extr["position"])
if extr.get("needs_optical_flip", True):
    R_cam = R_cam @ np.diag([-1.0, -1.0, 1.0])
K_mat = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]])

vis = rgb.copy()
selected_idx = None
for i, g in enumerate(grasps):
    ok = feasible[i] if i < len(feasible) else False
    p_world = np.array(g.position)
    p_cam = R_cam.T @ (p_world - t_cam)
    if p_cam[2] <= 0:
        continue
    px = K_mat @ (p_cam / p_cam[2])
    u, v = int(px[0]), int(px[1])
    if ok:
        color = (0, 255, 0)
        if selected_idx is None:
            selected_idx = i
    else:
        color = (0, 0, 255)
    cv2.circle(vis, (u, v), 8, color, -1)
    cv2.circle(vis, (u, v), 9, (255, 255, 255), 1)
    label = f"[{i}] {g.score:.3f}"
    cv2.putText(vis, label, (u + 12, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if ok and i == selected_idx:
        cv2.circle(vis, (u, v), 14, (0, 255, 255), 2)

vis_path = log_image(vis, tag="grasp_ranking", label=pick_name)
print(f"  Grasp visualization saved to {vis_path}")

if grasp is None:
    print("  No feasible grasp found — going home.")
    go_home()
    raise RuntimeError("ABORT")

grasp_pos = np.array(grasp.position)
grasp_quat = rpy_deg_to_quat(grasp.rpy)
print(f"  Selected: xyz={[round(float(x), 3) for x in grasp_pos]} "
      f"quat={[round(float(x), 3) for x in grasp_quat]}")

# ===================================================================
print("\n" + "=" * 60)
print("STEP 4 — BundleSDF: Locate place target")
print("=" * 60)

place_name = vlm(
    "The robot is about to pick up an object from the counter. "
    "Where should it place the object according to this kitchen task? "
    "Reply ONLY with the target name (e.g. 'cabinet', 'shelf', 'sink')."
).strip().strip('"\'')
print(f"  VLM place target: '{place_name}'")

place_pos = None
try:
    dets = detect_objects_oneshot(place_name, camera=CAMERA)
    det_list = dets.get(place_name, [])
    if det_list and det_list[0].position_3d:
        place_pos = np.array(det_list[0].position_3d)
        print(f"  BundleSDF: '{place_name}' at {[round(float(x), 3) for x in place_pos]}")
except Exception as e:
    print(f"  BundleSDF failed: {e}")

if place_pos is None:
    # Fallback: offset from grasp position
    place_pos = grasp_pos + np.array([0.15, 0.0, 0.05])
    print(f"  Fallback place pos: {[round(float(x), 3) for x in place_pos]}")

# ===================================================================
print("\n" + "=" * 60)
print("STEP 5 — Pick")
print("=" * 60)

state = get_robot_state()
ee_pos = np.array(state.arms[SIDE].ee_pos)
ee_quat = np.array(state.arms[SIDE].ee_quat)
print(f"  EE start: {[round(float(x), 3) for x in ee_pos]}")

# Open gripper
print("  Opening gripper...")
set_gripper(SIDE, 1.0)
time.sleep(0.3)

# Hover above grasp
hover_pos = grasp_pos.copy()
hover_pos[2] += HOVER_HEIGHT
print(f"  Moving to hover above grasp...")
if not plan_and_move(hover_pos, grasp_quat, gripper=1.0, label="hover"):
    print("  ABORT: cannot reach hover pose. Trying next grasp or going home.")
    go_home()
    raise RuntimeError("ABORT")

# Descend to grasp
print(f"  Descending to grasp...")
if not plan_and_move(grasp_pos, grasp_quat, label="descend"):
    print("  ABORT: cannot reach grasp pose.")
    go_home()
    raise RuntimeError("ABORT")

# Close gripper
print("  Closing gripper...")
set_gripper(SIDE, 0.0)
time.sleep(0.5)

# Check if grasped
state2 = get_robot_state()
grip_val = state2.arms[SIDE].gripper_pos
grip_val = float(grip_val[0]) if hasattr(grip_val, "__getitem__") else float(grip_val)
grasped = grip_val > 0.005
print(f"  Gripper width: {grip_val:.4f} → {'GRASPED' if grasped else 'MISSED'}")

if not grasped:
    print("  Grasp failed — returning home.")
    set_gripper(SIDE, 1.0)
    go_home()
else:
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 6 — Lift")
    print("=" * 60)

    state3 = get_robot_state()
    lift_pos = np.array(state3.arms[SIDE].ee_pos)
    lift_quat = np.array(state3.arms[SIDE].ee_quat)
    lift_pos[2] += HOVER_HEIGHT
    if not plan_and_move(lift_pos, lift_quat, gripper=0.0, label="lift"):
        print("  ABORT: cannot lift. Going home.")
        set_gripper(SIDE, 1.0)
        go_home()
        raise RuntimeError("ABORT")

    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 7 — Place")
    print("=" * 60)

    # Hover above place target
    place_hover = place_pos.copy()
    place_hover[2] += HOVER_HEIGHT
    print(f"  Moving to place hover...")
    if not plan_and_move(place_hover, lift_quat, gripper=0.0, label="place-hover"):
        print("  Cannot reach place hover — releasing here.")

    # Lower to place
    print(f"  Lowering to place...")
    plan_and_move(place_pos, lift_quat, gripper=0.0, label="place-lower")

    # Release
    print("  Releasing...")
    set_gripper(SIDE, 1.0)
    time.sleep(0.3)

    # Retract
    retract_pos = place_pos.copy()
    retract_pos[2] += HOVER_HEIGHT
    plan_and_move(retract_pos, lift_quat, gripper=1.0, label="retract")

    # Home
    go_home()

    # ===================================================================
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    final = get_task_info()
    print(f"  Done:    {final.get('done')}")
    print(f"  Success: {final.get('success')}")
    print(f"  Reward:  {final.get('reward')}")

    verdict = vlm("Did the robot successfully pick and place the object? What do you see?")
    print(f"  VLM: {verdict[:200]}")

print("\nDone!")
