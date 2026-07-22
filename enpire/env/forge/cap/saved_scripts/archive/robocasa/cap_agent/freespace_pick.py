# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pick up an object in RoboCasa using freespace_move with remote cuRobo
# (batch_size=1 single-target planning, fast solver config).
#
# Uses ground-truth object position from get_task_info() — no vision.
# freespace_move connects to a REMOTE cuRobo server via CAP_CUROBO_HOST/PORT.
#
# Run:
#   CAP_CUROBO_HOST=<host> CAP_CUROBO_PORT=<port> \
#   ROBOCASA_CONTROLLER_TYPE=joint_position CAP_ROBOT_TYPE=panda \
#   uv run python -u run_script.py \
#     --file robocasa/robocasa_freespace_pick.py \
#     --env "robocasa:PickPlaceCounterToCabinet" --cap-port 18600 --no-log

import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOVER_HEIGHT = 0.10   # metres above object for approach
LIFT_HEIGHT = 0.20    # metres to lift after grasp
VIDEO_FPS = 15
VIDEO_CAMERAS = ["top", "wrist"]

# ---------------------------------------------------------------------------
# Video recorder (background thread, per-camera MP4s)
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
        print(f"[video] {len(frames)} frames → {final}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_obj_pos(info: dict) -> np.ndarray:
    """Find object position from task_info, handling different task naming."""
    if "obj_pos" in info:
        return np.array(info["obj_pos"])
    for k, v in info.items():
        if k.endswith("_pos") and not k.startswith("robot") and "container" not in k:
            print(f"  Using '{k}' as object target")
            return np.array(v)
    raise RuntimeError(f"No object position in task_info: {list(info.keys())}")


# ---------------------------------------------------------------------------
# 1. Print cuRobo config
# ---------------------------------------------------------------------------
print("cuRobo config:")
print(f"  CAP_CUROBO_HOST={os.environ.get('CAP_CUROBO_HOST', '(not set — will auto-start local)')}")
print(f"  CAP_CUROBO_PORT={os.environ.get('CAP_CUROBO_PORT', '(not set — will auto-start local)')}")

# Derive task name from env for video path
env_name = os.environ.get("ROBOCASA_ENV_NAME", "")
if not env_name:
    # Fallback: parse from server's env_name
    try:
        info = get_task_info()
        env_name = info.get("env_name", "unknown")
    except Exception:
        env_name = "unknown"

# ---------------------------------------------------------------------------
# 2. Read state
# ---------------------------------------------------------------------------
state = get_robot_state()
arm = list(state.arms.keys())[0]
ee_pos = np.array(state.arms[arm].ee_pos)
ee_quat = np.array(state.arms[arm].ee_quat)
print(f"Arm: {arm}")
print(f"EE pos: {[round(x, 3) for x in ee_pos]}")

task_info = get_task_info()
if not env_name or env_name == "unknown":
    env_name = task_info.get("env_name", "unknown")
obj_pos = find_obj_pos(task_info)
print(f"Task: {env_name}")
print(f"Obj pos: {[round(float(x), 3) for x in obj_pos]}")

ee_rpy_deg = R.from_quat(ee_quat).as_euler("xyz", degrees=True).tolist()
print(f"EE RPY (deg): {[round(x, 1) for x in ee_rpy_deg]}")

# Start recording
_recording.set()
time.sleep(0.5)

# ---------------------------------------------------------------------------
# 3. Open gripper
# ---------------------------------------------------------------------------
print("\n--- Open gripper ---")
set_gripper(arm, 1.0)
time.sleep(0.5)

# ---------------------------------------------------------------------------
# 4. Hover above object
# ---------------------------------------------------------------------------
print("\n--- Hover ---")
hover_pos = obj_pos.copy()
hover_pos[2] += HOVER_HEIGHT

result = freespace_move(
    right_target_pos=hover_pos.tolist(),
    right_gripper=1.0,
    solver_speed="fast",
    planning_speed=0.8,
    ik_error_threshold=0.01,
)
print(f"  status={result.status}, steps={result.trajectory_steps}")

# ---------------------------------------------------------------------------
# 5. Lower to object
# ---------------------------------------------------------------------------
print("\n--- Lower ---")
task_info2 = get_task_info()
obj_now = find_obj_pos(task_info2)

result = freespace_move(
    right_target_pos=obj_now.tolist(),
    right_gripper=1.0,
    solver_speed="fast",
    planning_speed=0.8,
    ik_error_threshold=0.01,
)
print(f"  status={result.status}, steps={result.trajectory_steps}")

# ---------------------------------------------------------------------------
# 6. Close gripper
# ---------------------------------------------------------------------------
print("\n--- Close gripper ---")
set_gripper(arm, 0.0)
time.sleep(1.5)

# ---------------------------------------------------------------------------
# 7. Lift
# ---------------------------------------------------------------------------
print("\n--- Lift ---")
state3 = get_robot_state()
lift_pos = np.array(state3.arms[arm].ee_pos)
lift_pos[2] += LIFT_HEIGHT

result = freespace_move(
    right_target_pos=lift_pos.tolist(),
    right_gripper=0.0,
    solver_speed="fast",
    planning_speed=0.8,
    ik_error_threshold=0.01,
)
print(f"  status={result.status}, steps={result.trajectory_steps}")

time.sleep(1.0)

# ---------------------------------------------------------------------------
# 8. Result + save video
# ---------------------------------------------------------------------------
task_final = get_task_info()
obj_final_pos = find_obj_pos(task_final)
lifted = obj_final_pos[2] - obj_pos[2]
print(f"\nLifted: {lifted:.3f}m  {'PICK SUCCESS!' if lifted > 0.05 else 'PICK FAILED'}")

timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
save_videos(f"logs/robocasa_freespace_pick/{env_name}/{timestamp}")
