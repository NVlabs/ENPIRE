# Turn on microwave using vision tools + VLM visual servoing.
#
# Run with:
#   ROBOCASA_LAYOUT_ID=3 ROBOCASA_STYLE_ID=5 CAP_AGENT_NAME=demo \
#   uv run python -u run_script.py --file robocasa_turn_on_microwave_vision.py \
#   --env "robocasa:TurnOnMicrowave" --cap-port 18600 --no-log --record
#
# Vision tools used:
#   - vlm_query (Gemini)          — scene understanding + visual servoing
#   - detect_objects_oneshot       — BundleSDF 3D pose of microwave
#   - get_camera_image             — camera frames for debug
#   - get_robot_state / get_task_info — state + ground truth
#   - freespace_move / set_gripper  — motion
#
# Strategy:
#   1. Use BundleSDF to get microwave 3D position (or fall back to task_info)
#   2. Use VLM (top cam) to understand scene layout
#   3. Close gripper for fingertip press
#   4. Approach from the front of the microwave
#   5. Use VLM (wrist cam) to servo toward the start button
#   6. Press and check result

import numpy as np
import time
from scipy.spatial.transform import Rotation

SIDE = "right"
VLM_BACKEND = "gemini"
VLM_MODEL = "gemini-2.5-flash"
SERVO_STEPS = 3        # VLM servo iterations
SERVO_STEP_M = 0.015   # meters per nudge
APPROACH_OFFSET = 0.15  # how far in front of button to start
PRESS_DEPTH = 0.08      # how far to push past button surface
MOVE_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vlm(prompt, camera="top"):
    return vlm_query(
        text=prompt, backend=VLM_BACKEND, model=VLM_MODEL,
        media=[f"camera:{camera}"],
    )


def ee():
    s = get_robot_state()
    arm = s.arms[SIDE]
    return np.array(arm.ee_pos), np.array(arm.ee_quat)


def move(pos, quat=None, tol=0.02, timeout=MOVE_TIMEOUT):
    if quat is None:
        _, quat = ee()
    rpy = list(Rotation.from_quat(quat).as_euler('xyz', degrees=True))
    result = freespace_move(
        right_target_pos=[float(x) for x in pos],
        right_target_rpy=[float(x) for x in rpy],
        planning_speed=1.5,
    )
    return result


def parse_nudge(response):
    """Parse VLM response into a 3D nudge vector.

    Wrist camera convention (approximate):
      left/right  → world Y (robot's lateral axis)
      up/down     → world Z
      closer      → world X (toward microwave)
    """
    resp = response.lower()
    nudge = np.zeros(3)
    if "left" in resp:
        nudge[1] += SERVO_STEP_M
    if "right" in resp:
        nudge[1] -= SERVO_STEP_M
    if "up" in resp or "above" in resp or "higher" in resp:
        nudge[2] += SERVO_STEP_M
    if "down" in resp or "below" in resp or "lower" in resp:
        nudge[2] -= SERVO_STEP_M
    if "closer" in resp or "forward" in resp:
        nudge[0] += SERVO_STEP_M
    if "farther" in resp or "back" in resp or "away" in resp:
        nudge[0] -= SERVO_STEP_M
    return nudge


# ===========================================================================
print("=" * 60)
print("STEP 1 — Locate microwave")
print("=" * 60)

task_info = get_task_info()
obj_pos = np.array(task_info.get("obj_pos", [0, 0, 0]))
print(f"  Ground truth obj_pos: {[round(float(x), 3) for x in obj_pos]}")

# BundleSDF for vision-based 3D pose
dets = detect_objects_oneshot("microwave", camera="top")
det_list = dets.get("microwave", [])
if not det_list or not det_list[0].position_3d:
    raise RuntimeError("BundleSDF failed to detect microwave")
target_pos = np.array(det_list[0].position_3d)
print(f"  BundleSDF detection: {[round(float(x), 3) for x in target_pos]}")

# VLM scene understanding
scene = vlm("Describe the kitchen scene briefly. Where is the microwave?")
print(f"  VLM: {scene[:150]}")

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 2 — Prepare for button press")
print("=" * 60)

ee_pos, ee_quat = ee()
print(f"  EE pos: {[round(float(x), 3) for x in ee_pos]}")

# Close gripper for fingertip press
print("  Closing gripper...")
set_gripper(SIDE, 0.0)
time.sleep(0.3)

# Compute approach: the microwave button faces roughly toward the robot.
# We approach along the EE→button direction, stopping APPROACH_OFFSET in front.
eef_to_target = target_pos - ee_pos
approach_dir = eef_to_target / (np.linalg.norm(eef_to_target) + 1e-8)
approach_pos = target_pos - approach_dir * APPROACH_OFFSET

print(f"  Approach direction: {[round(float(x), 3) for x in approach_dir]}")
print(f"  Approach position: {[round(float(x), 3) for x in approach_pos]}")

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 3 — Approach microwave")
print("=" * 60)

move(approach_pos, ee_quat)
time.sleep(0.5)

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 4 — VLM visual servoing (wrist camera)")
print("=" * 60)

for step in range(1, SERVO_STEPS + 1):
    print(f"\n  --- Servo step {step}/{SERVO_STEPS} ---")

    guidance = vlm(
        "You are looking at a microwave panel through a robot wrist camera. "
        "The robot needs to press the START or power button. "
        "Where is the button relative to the CENTER of this image? "
        "Answer with direction words: left, right, up, down, closer, farther. "
        "If the button is centered, say 'centered'. Be concise.",
        camera="wrist",
    )
    print(f"  VLM: {guidance[:100]}")

    if "centered" in guidance.lower() or "aligned" in guidance.lower():
        print("  Button is centered — ready to press")
        break

    nudge = parse_nudge(guidance)
    if np.linalg.norm(nudge) < 1e-4:
        print("  No nudge needed")
        break

    cur_pos, cur_quat = ee()
    new_pos = cur_pos + nudge
    print(f"  Nudge: {[round(float(x), 3) for x in nudge]} → {[round(float(x), 3) for x in new_pos]}")
    move(new_pos, cur_quat, tol=0.01, timeout=6.0)
    time.sleep(0.3)

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 5 — Press button")
print("=" * 60)

cur_pos, cur_quat = ee()

# Push forward along approach direction
press_pos = cur_pos + approach_dir * PRESS_DEPTH
print(f"  Pressing: {[round(float(x), 3) for x in cur_pos]} → {[round(float(x), 3) for x in press_pos]}")
move(press_pos, cur_quat, tol=0.01, timeout=8.0)

# Hold press
print("  Holding...")
time.sleep(0.5)

# Check mid-press
mid_info = get_task_info()
print(f"  Mid-press reward: {mid_info.get('reward', 0)}")

# If not yet successful, push a bit more
if not mid_info.get("success", False):
    print("  Pushing deeper...")
    deeper = np.array(get_robot_state().arms[SIDE].ee_pos) + approach_dir * 0.04
    move(deeper, cur_quat, tol=0.01, timeout=6.0)
    time.sleep(0.5)

# ===========================================================================
print("\n" + "=" * 60)
print("STEP 6 — Retract")
print("=" * 60)

cur_pos, _ = ee()
retract_pos = cur_pos - approach_dir * 0.15
print(f"  Retracting to: {[round(float(x), 3) for x in retract_pos]}")
move(retract_pos, cur_quat, timeout=8.0)

go_home()

# ===========================================================================
print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

result = get_task_info()
print(f"  Success: {result.get('success', False)}")
print(f"  Reward:  {result.get('reward', 0.0)}")
print(f"  Done:    {result.get('done', False)}")

verdict = vlm("Did the robot successfully press the microwave button? What do you see?")
print(f"  VLM: {verdict[:200]}")
print("\nDone!")
