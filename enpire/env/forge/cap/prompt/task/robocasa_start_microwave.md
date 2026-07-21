# RoboCasa Start Microwave

Strategy for `TurnOnMicrowave` tasks. **This is a button-press task, NOT pick-and-place.**

## ⚠ KNOWN FAILURE — READ THIS FIRST

Every previous attempt at this task has failed because the code used `ee_quat` (gripper pointing DOWN)
instead of computing a face-forward quaternion. The OSC controller CANNOT reach a vertical microwave
button with a downward-facing gripper — every `freespace_move` will timeout.

**YOU MUST:**
- Import `scipy.spatial.transform.Rotation` at the top of your code
- Compute a look-at quaternion BEFORE any `freespace_move` call
- Try multiple roll angles (0°, ±30°, ±60°, ..., 180°) to find an IK-feasible orientation
- Use the working quaternion for ALL `freespace_move` calls

**YOU MUST NOT:**
- Use `ee_quat` from `get_robot_state()` — it points DOWN and will always fail
- Use `get_robot_state().arms[arm].ee_quat` as the quaternion argument to `freespace_move`
- Skip the orientation step — there is no shortcut

## REQUIRED CODE TEMPLATE — Start your code with this:

```python
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

state = get_robot_state()
arm = list(state.arms.keys())[0]
ee_pos = np.array(state.arms[arm].ee_pos)
obj_pos = np.array(get_task_info()["obj_pos"])
direction = (obj_pos - ee_pos)
direction = direction / (np.linalg.norm(direction) + 1e-6)

set_gripper(arm, 0.0)  # close gripper for fingertip press

def make_face_quat(fwd, roll_deg=0.0):
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, fwd)
    right = right / (np.linalg.norm(right) + 1e-6)
    up = np.cross(fwd, right)
    base_rot = R.from_matrix(np.column_stack([right, up, fwd]))
    if abs(roll_deg) > 0.1:
        base_rot = R.from_rotvec(fwd * np.radians(roll_deg)) * base_rot
    return base_rot.as_quat().tolist()

# Find an IK-feasible orientation by trying multiple roll angles
face_quat = None
test_pos = obj_pos - direction * 0.12
for roll in [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]:
    candidate = make_face_quat(direction, roll_deg=roll)
    try:
        freespace_move(arm, test_pos, candidate, max_duration_sec=8.0)
        face_quat = candidate
        print(f"Roll={roll}° works!")
        break
    except Exception:
        print(f"Roll={roll}° failed, trying next...")

if face_quat is None:
    print("No feasible orientation found!")
else:
    # ... continue with pressing using face_quat ...
    pass
```

After this template, add your button-pressing logic using `face_quat` for all moves.

## After finding face_quat — pressing steps:

### Use VLM to verify (optional but recommended)
```python
view_pos = obj_pos - direction * 0.15
freespace_move(arm, view_pos, face_quat, max_duration_sec=10.0)
```

### Step 4: Use VLM to locate the start button
```python
scene = vlm_query(
    "Describe the microwave control panel. Where is the START/power button "
    "relative to the center of the image? Answer: left/right/up/down/center.",
    backend="gemini", camera="wrist")
```
Parse the response and nudge ±2cm in Y (left/right) or Z (up/down).

### Step 5: Press the button
Push forward 10cm along `direction`:
```python
press = np.array(get_robot_state().arms[arm].ee_pos) + direction * 0.10
freespace_move(arm, press, face_quat, max_duration_sec=8.0)
time.sleep(0.5)  # hold press
```

### Step 6: Retract
```python
retract = np.array(get_robot_state().arms[arm].ee_pos) - direction * 0.12
freespace_move(arm, retract, face_quat, max_duration_sec=8.0)
```

## Key rules

- **NEVER use `ee_quat` from get_robot_state() directly** — it points the gripper down. Always compute `face_quat`.
- **ALWAYS use `face_quat`** in every `freespace_move` call.
- **ALWAYS use `vlm_query(camera="wrist")`** after positioning to verify button location.
- **ALWAYS wrap `freespace_move` in try/except** — timeouts are common.
- **ALWAYS use `max_duration_sec=10.0`** or higher for approach moves.
- `get_task_info()["obj_pos"]` = approximate button position. `get_task_info()["success"]` = task complete.

## Common failures

- **Gripper faces down instead of at microwave** — didn't compute `face_quat`. This is the #1 failure mode.
- **Press doesn't register** — didn't push deep enough. Push 10cm past approach position.
- **Timeout on approach** — increase `max_duration_sec` to 15.
- **Wrong button** — use VLM with wrist camera to verify before pressing.
