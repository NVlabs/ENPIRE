# Test remote vision stack (SAM3 + AnyGrasp) + remote cuRobo for RoboCasa pick-and-place.
#
# Uses AnyGrasp (via remote SAM3 segmentation) to get grasp candidates,
# ranks them with single-target cuRobo IK (preview_only), executes best grasp,
# then places at the target container position from get_task_info() ground truth.
#
# All planning goes through remote cuRobo on S1 via SSH tunnel.
# No collision world loaded (matches working single-target freespace_move path).
#
# Prereqs:
#   SSH tunnels: -L 6767:127.0.0.1:6767 -L 8122:127.0.0.1:8122 -L 8612:127.0.0.1:8612
#
# Run:
#   CAP_CUROBO_HOST=127.0.0.1 CAP_CUROBO_PORT=8612 CAP_CUROBO_START_SERVER=0 \
#   ROBOCASA_CONTROLLER_TYPE=joint_position CAP_ROBOT_TYPE=panda \
#   uv run python -u run_script.py \
#     --file robocasa/test_remote_vision_cuRobo.py \
#     --env "robocasa:PickPlaceCounterToCabinet" --cap-port 18600 --no-log

import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOVER_HEIGHT = 0.10
LIFT_HEIGHT = 0.15
PLACE_Z_OFFSET = 0.12
MAX_GRASP_ATTEMPTS = 3
MAX_GRASP_CANDIDATES = 16
PLANNING_SPEED = 0.8
IK_ERROR_THRESHOLD = 0.01
VIDEO_FPS = 15
VIDEO_CAMERAS = ["top", "wrist"]

# ---------------------------------------------------------------------------
# Video recorder
# ---------------------------------------------------------------------------
_video_frames: dict[str, list[np.ndarray]] = {c: [] for c in VIDEO_CAMERAS}
_recording = threading.Event()
_stop = threading.Event()


def _record_loop():
    dt = 1.0 / VIDEO_FPS
    while not _stop.is_set():
        if _recording.is_set():
            for cam in VIDEO_CAMERAS:
                frame = server.get_camera_image(cam)
                if frame.shape[0] > 1:
                    _video_frames[cam].append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        _stop.wait(dt)


threading.Thread(target=_record_loop, daemon=True).start()


def save_videos(output_dir: str):
    _stop.set()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cam, frames in _video_frames.items():
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        raw = str(out / f"{cam}.raw.mp4")
        final = str(out / f"{cam}.mp4")
        wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))
        for f in frames:
            wr.write(f)
        wr.release()
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-preset", "fast",
                 "-crf", "23", "-pix_fmt", "yuv420p", final],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            os.unlink(raw)
        except (subprocess.CalledProcessError, FileNotFoundError):
            os.rename(raw, final)
        print(f"[video] {len(frames)} frames -> {final}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_obj_pos(info: dict) -> tuple[np.ndarray, str]:
    """Return (position, key_used) for the pick object."""
    if "obj_pos" in info:
        return np.array(info["obj_pos"]), "obj_pos"
    for k, v in info.items():
        if k.endswith("_pos") and not k.startswith("robot") and "container" not in k and "to_" not in k:
            return np.array(v), k
    raise RuntimeError(f"No object position in task_info: {list(info.keys())}")


def find_target_pos(info: dict) -> tuple[np.ndarray, str]:
    """Return (position, key_used) for the place target (container/cabinet/plate)."""
    obj_pos_key = "obj_pos"
    # Try common target keys in priority order
    for pattern in ["container", "cab", "plate", "box", "bin", "target"]:
        for k, v in info.items():
            if pattern in k and k.endswith("_pos") and "to_" not in k:
                return np.array(v), k
    # Fallback: any _pos that isn't the pick object or robot
    for k, v in info.items():
        if k.endswith("_pos") and k != obj_pos_key and not k.startswith("robot") and "to_" not in k:
            return np.array(v), k
    raise RuntimeError(f"No target position in task_info: {list(info.keys())}")


def do_freespace_move(target_pos, gripper=None, speed=PLANNING_SPEED, label="move"):
    """Single-target freespace_move for the right arm (Panda single-arm)."""
    kw = {
        "right_target_pos": target_pos.tolist() if isinstance(target_pos, np.ndarray) else list(target_pos),
        "solver_speed": "fast",
        "planning_speed": speed,
        "ik_error_threshold": IK_ERROR_THRESHOLD,
    }
    if gripper is not None:
        kw["right_gripper"] = gripper
    print(f"  [{label}] target={[round(x, 3) for x in kw['right_target_pos']]}")
    t0 = time.time()
    result = freespace_move(**kw)
    dt = time.time() - t0
    print(f"  [{label}] status={result.status} ({dt*1000:.0f}ms, {result.trajectory_steps} steps)")
    return result


def rank_grasps_single_target(grasps, max_try=8):
    """Rank grasp candidates using single-target preview_only freespace_move.

    The batch path in freespace_move has YAM-specific RPY conversion that
    doesn't work for Panda. This loops single-target calls instead.
    """
    ranked = []
    for i, g in enumerate(grasps[:max_try]):
        try:
            result = freespace_move(
                right_target_pos=list(g.position),
                right_target_rpy=list(g.rpy),
                right_gripper=1.0,
                solver_speed="fast",
                planning_speed=PLANNING_SPEED,
                ik_error_threshold=IK_ERROR_THRESHOLD,
                preview_only=True,
            )
            if result.status == "Success":
                err = result.final_pos_error_m or 0.0
                ranked.append((err, i, g, result.trajectory_cache_key))
                print(f"    grasp[{i}] OK pos_err={err:.4f}m score={g.score:.3f}")
            else:
                print(f"    grasp[{i}] FAIL: {result.status}")
        except Exception as e:
            print(f"    grasp[{i}] ERROR: {e}")
    ranked.sort(key=lambda x: x[0])
    return ranked


# ===========================================================================
# Main
# ===========================================================================
print("=" * 60)
print("Remote Vision + cuRobo Pick-and-Place Test")
print("=" * 60)
print(f"  CAP_CUROBO_HOST={os.environ.get('CAP_CUROBO_HOST', '(not set)')}")
print(f"  CAP_CUROBO_PORT={os.environ.get('CAP_CUROBO_PORT', '(not set)')}")

# ---------------------------------------------------------------------------
# 1. Read state and task info
# ---------------------------------------------------------------------------
state = get_robot_state()
arm = "right"
ee_pos = np.array(state.arms[arm].ee_pos)
home_pos = ee_pos.copy()
print(f"Arm: {arm}, EE: {[round(x, 3) for x in ee_pos]}")

task_info = get_task_info()
env_name = task_info.get("env_name", "unknown")
obj_pos, obj_key = find_obj_pos(task_info)
obj_name = task_info.get("obj_name", obj_key.replace("_pos", ""))
print(f"Task: {env_name}")
print(f"Object: '{obj_name}' at {[round(float(x), 3) for x in obj_pos]} (key={obj_key})")

try:
    target_pos, target_key = find_target_pos(task_info)
    target_name = target_key.replace("_pos", "").replace("_", " ")
    print(f"Target: '{target_name}' at {[round(float(x), 3) for x in target_pos]} (key={target_key})")
except RuntimeError:
    target_pos = None
    target_name = None
    print("No place target found — will do pick-only")

# Start recording
_recording.set()
time.sleep(0.5)

# ---------------------------------------------------------------------------
# 2. Test remote services
# ---------------------------------------------------------------------------
print("\n--- Testing remote services ---")

# Test SAM3 + AnyGrasp
# Use a generic prompt — SAM3 often can't find specific object names like
# "marshmallow" in rendered sim images. "object on the counter" works reliably.
grasp_prompt = "object on the counter"
print(f"  Calling sample_grasp_pose_anygrasp('{grasp_prompt}') via SAM3 + AnyGrasp tunnel...")
grasps = None
for attempt in range(MAX_GRASP_ATTEMPTS):
    try:
        raw_grasps = sample_grasp_pose_anygrasp(
            object_name=grasp_prompt,
            camera="top",
            max_grasps=MAX_GRASP_CANDIDATES,
        )
        # Filter grasps to be near the ground-truth object position
        # (AnyGrasp may return grasps on other objects in the scene)
        GRASP_DIST_THRESH = 0.20  # 20cm
        grasps = [
            g for g in raw_grasps
            if np.linalg.norm(np.array(g.position) - obj_pos) < GRASP_DIST_THRESH
        ]
        print(f"  AnyGrasp: {len(raw_grasps)} raw, {len(grasps)} near object (<{GRASP_DIST_THRESH}m)")
        if grasps:
            g = grasps[0]
            print(f"    best: pos={[round(float(x), 3) for x in g.position]}, "
                  f"rpy={[round(float(x), 1) for x in g.rpy]}, score={g.score:.3f}")
        break
    except Exception as e:
        print(f"  AnyGrasp attempt {attempt+1} failed: {e}")
        if attempt == MAX_GRASP_ATTEMPTS - 1:
            print("  AnyGrasp exhausted — falling back to ground-truth position")

# ---------------------------------------------------------------------------
# 3. Open gripper
# ---------------------------------------------------------------------------
print("\n--- Open gripper ---")
set_gripper(arm, 1.0)
time.sleep(0.5)

# ---------------------------------------------------------------------------
# 4. Pick: AnyGrasp + cuRobo ranking, or fallback to ground truth
# ---------------------------------------------------------------------------
picked = False

if grasps and len(grasps) > 0:
    print(f"\n--- Ranking {min(len(grasps), 8)} grasp candidates via cuRobo preview ---")
    ranked = rank_grasps_single_target(grasps, max_try=8)

    if ranked:
        best_err, best_idx, best_grasp, cache_key = ranked[0]
        print(f"\n--- Executing best grasp (idx={best_idx}, err={best_err:.4f}m) ---")
        do_freespace_move(np.array(best_grasp.position), gripper=1.0, label="grasp")

        # Close gripper
        print("\n--- Close gripper ---")
        set_gripper(arm, 0.0)
        time.sleep(1.5)

        # Check grasp
        state2 = get_robot_state()
        gp = state2.arms[arm].gripper_pos
        gripper_val = float(gp[0]) if hasattr(gp, "__getitem__") else float(gp)
        print(f"  Gripper pos after close: {gripper_val:.4f}")
        picked = gripper_val > 0.001
        if picked:
            print("  GRASP OK")
        else:
            print("  GRASP FAILED (gripper fully closed — nothing grasped)")
    else:
        print("  No feasible grasps found via cuRobo ranking")

if not picked:
    # Fallback: ground-truth pick
    print("\n--- Fallback: ground-truth pick ---")
    hover_pos = obj_pos.copy()
    hover_pos[2] += HOVER_HEIGHT
    do_freespace_move(hover_pos, gripper=1.0, label="hover")

    task_info2 = get_task_info()
    obj_now, _ = find_obj_pos(task_info2)
    do_freespace_move(obj_now, gripper=1.0, label="lower")

    print("\n--- Close gripper ---")
    set_gripper(arm, 0.0)
    time.sleep(1.5)
    picked = True

# ---------------------------------------------------------------------------
# 5. Lift
# ---------------------------------------------------------------------------
print("\n--- Lift ---")
state3 = get_robot_state()
lift_pos = np.array(state3.arms[arm].ee_pos)
lift_pos[2] += LIFT_HEIGHT
do_freespace_move(lift_pos, gripper=0.0, label="lift")

# ---------------------------------------------------------------------------
# 6. Place at target (ground-truth position)
# ---------------------------------------------------------------------------
if target_pos is not None:
    print(f"\n--- Place at '{target_name}' ---")
    place_pos = target_pos.copy()
    place_pos[2] += PLACE_Z_OFFSET
    try:
        do_freespace_move(place_pos, gripper=0.0, label="place-above")
    except RuntimeError as e:
        print(f"  Place move failed (target may be out of reach): {e}")
        print("  Releasing at current position instead")

    print("\n--- Release ---")
    set_gripper(arm, 1.0)
    time.sleep(0.5)

# ---------------------------------------------------------------------------
# 7. Return to start
# ---------------------------------------------------------------------------
print("\n--- Return home ---")
do_freespace_move(home_pos, gripper=1.0, label="home")
time.sleep(1.0)

# ---------------------------------------------------------------------------
# 8. Results
# ---------------------------------------------------------------------------
task_final = get_task_info()
reward = task_final.get("reward", 0.0)
success = task_final.get("success", False)
obj_final, _ = find_obj_pos(task_final)
lifted = obj_final[2] - obj_pos[2]

print(f"\n{'=' * 60}")
print(f"Task: {env_name}")
print(f"Reward: {reward}")
print(f"Success: {success}")
print(f"Object Z delta: {lifted:.3f}m")
print(f"{'=' * 60}")

timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
save_videos(f"logs/robocasa_freespace_pick/{env_name}/{timestamp}")
