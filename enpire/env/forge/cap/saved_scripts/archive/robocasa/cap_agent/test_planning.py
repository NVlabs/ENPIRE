# Test planning: pick and place using oracle object/place positions + AnyGrasp.
#
# Uses get_task_info() for oracle obj_pos and container_pos (no VLM for detection).
# Still uses AnyGrasp for grasp pose generation near the oracle object position.
#
# Tools used:
#   - create_motion_planner      — cuRobo Panda planner (sandbox API)
#   - update_planner_world       — load kitchen collision geometry into planner
#   - sample_grasp_pose_anygrasp — SAM3 segmentation + AnyGrasp grasp planning
#   - get_robot_state            — EE pose, joint state, gripper
#   - get_task_info              — reward, success, object ground truth
#   - set_gripper                — gripper control
#   - go_home                    — return to start pose
#   - move_joint_keypoints       — joint trajectory execution
#   - get_camera_image           — camera RGB frames
#   - get_camera_intrinsics      — camera [fx, fy, cx, cy]
#   - get_camera_extrinsics      — camera rotation, position

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIDE = "right"
CAMERAS = ["wrist", "top"]  # try wrist first, fallback to top
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
print("STEP 1 — Oracle: Get task info")
print("=" * 60)

info = get_task_info()
print(f"  Task: {info.get('env_name')}")
obj_pos_oracle = np.array([float(x) for x in info["obj_pos"]])
print(f"  Object pos (oracle): {[round(x, 3) for x in obj_pos_oracle]}")

# Place target: use container_pos if available, else distr_counter_pos
if "container_pos" in info:
    place_pos_oracle = np.array([float(x) for x in info["container_pos"]])
    print(f"  Place pos (oracle, container): {[round(x, 3) for x in place_pos_oracle]}")
elif "distr_counter_pos" in info:
    place_pos_oracle = np.array([float(x) for x in info["distr_counter_pos"]])
    print(f"  Place pos (oracle, counter): {[round(x, 3) for x in place_pos_oracle]}")
else:
    place_pos_oracle = obj_pos_oracle + np.array([0.15, 0.0, 0.05])
    print(f"  Place pos (fallback): {[round(x, 3) for x in place_pos_oracle]}")

gt_keys = [k for k in info if k.endswith("_pos") and "robot" not in k and "eef" not in k]
for k in gt_keys:
    print(f"    {k}: {[round(float(x), 3) for x in info[k]]}")

# Get object name from oracle (extracted from model path)
pick_name = info.get("obj_name", "object")
print(f"\n  AnyGrasp target name (oracle): '{pick_name}'")

# ---------------------------------------------------------------------------
# Grasp planning helper (AnyGrasp + cuRobo feasibility + visualization)
# ---------------------------------------------------------------------------
import cv2
from enpire.env.forge.cap.agent.tools._artifact_log import log_image


def find_feasible_grasp(camera, object_name):
    """Run AnyGrasp on *camera*, rank by cuRobo feasibility.

    Returns (grasp, grasp_traj) if a feasible grasp is found, else (None, None).
    """
    print(f"\n  [{camera}] Sampling grasps for '{object_name}'...")
    grasps = sample_grasp_pose_anygrasp(
        object_name=object_name, camera=camera, max_grasps=MAX_GRASPS,
        disable_planner_z_clipping=True,
    )
    if not grasps:
        print(f"  [{camera}] AnyGrasp returned 0 candidates")
        return None, None

    print(f"  [{camera}] {len(grasps)} grasp candidates:")
    for i, g in enumerate(grasps[:5]):
        print(f"    [{i}] xyz={[round(float(x), 3) for x in g.position]} "
              f"rpy={[round(float(x), 1) for x in g.rpy]} score={g.score:.3f}")

    # Rank through cuRobo
    print(f"  [{camera}] Ranking {len(grasps)} grasps through cuRobo...")
    cur_state = get_robot_state()
    cur_jp = np.array(cur_state.arms[SIDE].joint_pos)

    best_grasp = None
    best_traj = None
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
        if ok and best_grasp is None:
            best_grasp = g
            best_traj = result["right_positions"]

    # Visualize grasps on camera image with 3D axes (matching CAP UI style)
    rgb = get_camera_image(camera)
    K = get_camera_intrinsics(camera)
    extr = get_camera_extrinsics(camera)
    cam_rot = np.array(extr["rotation"]).reshape(3, 3)
    cam_pos = np.array(extr["position"])
    fx, fy, cx, cy = [float(x) for x in K]
    AXIS_LEN = 0.04  # 4cm axes

    def _project(pos_3d):
        """Project world-frame 3D point to pixel (same as CAP UI)."""
        p_cam = np.linalg.inv(cam_rot) @ (np.asarray(pos_3d) - cam_pos)
        p_cam[0] = -p_cam[0]
        p_cam[1] = -p_cam[1]
        if p_cam[2] <= 0:
            return None
        u = fx * p_cam[0] / p_cam[2] + cx
        v = fy * p_cam[1] / p_cam[2] + cy
        return int(u), int(v)

    vis = rgb.copy()
    selected_idx = None
    # Axis colors: X=red, Y=green, Z=blue (BGR for cv2)
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    for i, g in enumerate(grasps):
        ok = feasible[i] if i < len(feasible) else False
        pos = np.array(g.position)
        origin_px = _project(pos)
        if origin_px is None:
            continue

        # Draw 3D axes from grasp rotation matrix
        rot_mat = R.from_euler(
            "xyz", [-g.rpy[1], g.rpy[0], -g.rpy[2] - 90.0], degrees=True
        ).as_matrix()
        for col in range(3):
            tip = pos + AXIS_LEN * rot_mat[:, col]
            tip_px = _project(tip.tolist())
            if tip_px is not None:
                thickness = 3 if (ok and selected_idx is None) else 2
                cv2.line(vis, origin_px, tip_px, axis_colors[col], thickness)

        # Circle at origin: green=feasible, red=failed, yellow ring=selected
        if ok:
            circle_color = (0, 255, 0)
            if selected_idx is None:
                selected_idx = i
        else:
            circle_color = (0, 0, 255)
        cv2.circle(vis, origin_px, 5, circle_color, -1)
        cv2.circle(vis, origin_px, 6, (255, 255, 255), 1)

        # Label
        label = f"[{i}] {g.score:.3f}"
        cv2.putText(vis, label, (origin_px[0] + 12, origin_px[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, circle_color, 1)

        # Highlight selected with yellow ring
        if ok and i == selected_idx:
            cv2.circle(vis, origin_px, 12, (0, 255, 255), 2)

    vis_path = log_image(vis, tag=f"grasp_ranking_{camera}", label=object_name)
    print(f"  [{camera}] Grasp visualization saved to {vis_path}")

    n_ok = sum(feasible)
    print(f"  [{camera}] {n_ok}/{len(grasps)} feasible")
    return best_grasp, best_traj


# ===================================================================
print("\n" + "=" * 60)
print("STEP 3 — AnyGrasp: Plan grasps on '{}' (wrist → top fallback)".format(pick_name))
print("=" * 60)

grasp = None
grasp_traj = None
used_camera = None
for cam in CAMERAS:
    grasp, grasp_traj = find_feasible_grasp(cam, pick_name)
    if grasp is not None:
        used_camera = cam
        break
    print(f"  [{cam}] No feasible grasp — trying next camera...")

if grasp is None:
    print("  No feasible grasp found on any camera — going home.")
    go_home()
    raise RuntimeError("ABORT")

grasp_pos = np.array(grasp.position)
grasp_quat = rpy_deg_to_quat(grasp.rpy)
print(f"  Selected ({used_camera}): xyz={[round(float(x), 3) for x in grasp_pos]} "
      f"quat={[round(float(x), 3) for x in grasp_quat]}")

# ===================================================================
print("\n" + "=" * 60)
print("STEP 4 — Oracle: Place target")
print("=" * 60)

place_pos = place_pos_oracle
print(f"  Place target (oracle): {[round(float(x), 3) for x in place_pos]}")

# ===================================================================
print("\n" + "=" * 60)
print("STEP 5 — Pick")
print("=" * 60)

state = get_robot_state()
ee_pos = np.array(state.arms[SIDE].ee_pos)
ee_quat = np.array(state.arms[SIDE].ee_quat)
print(f"  EE start: {[round(float(x), 3) for x in ee_pos]}")

# Open gripper, move to grasp, close
set_gripper(SIDE, 1.0)
time.sleep(0.3)

if not plan_and_move(grasp_pos, grasp_quat, gripper=1.0, label="grasp"):
    raise RuntimeError("ABORT: cannot reach grasp pose")

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

print("\nDone!")
