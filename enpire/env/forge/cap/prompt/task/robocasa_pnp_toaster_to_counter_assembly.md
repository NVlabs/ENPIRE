# RoboCasa Pick-and-Place Toaster→Counter — Assembly Guide

Orchestration guide for `PickPlaceToasterToCounter`.

## Key facts

- Grasp with `+0.06 z offset` above object centroid (grips through slot opening)
- Must **extract horizontally** before lifting — cannot lift straight out of toaster
- `toaster_extract_v1` handles **both** horizontal extraction and vertical lift to clear height
- Do NOT add a nudge-lift loop after `toaster_extract_v1` — it already lifts internally
- Normal place on counter after extraction

## Why no separate lift step

After horizontal extraction the arm is near joint limits from lateral extension.
OSC nudge stalls in this configuration (observed: arm moved only +3 mm instead of +50 mm
on 2nd nudge step). `toaster_extract_v1` uses `freespace_move` for the vertical phase,
which re-plans from current joints and is reliable once outside the slot.

## Reference assembly

```python
THOUGHTS:
Toaster-to-counter: grasp with +0.06z offset (clearance_lift=0), call toaster_extract_v1
which handles horizontal pull + vertical lift in one step, then hover above counter, lower,
release, retract, go_home.

```python
import numpy as np
import time
from skill_library.robust_grasp import robust_grasp_v1
from skill_library.toaster_extract import toaster_extract_v1
from skill_library.hover_above import hover_above_v1
from skill_library.vertical_place import vertical_place_v1

SIDE = "right"
TOASTER_GRASP_Z_OFFSET = 0.06

info = get_task_info()
obj_pos = np.array(info["obj_pos"])
obj_name = info.get("obj_name", "unknown")

place_pos = np.array(
    info["container_pos"] if "container_pos" in info and info["container_pos"] is not None
    else info.get("distr_counter_pos", info.get("distr_cab_pos", obj_pos + np.array([0.5, 0., 0.])))
)
print(f"Task: toaster pick '{obj_name}' -> counter at {place_pos.tolist()}")

open_gripper(SIDE)

# Step 1+2: Hover + descend; clearance_lift=0 because arm is still inside the slot
info = get_task_info()
obj_now = np.array(info["obj_pos"]); obj_now[2] += TOASTER_GRASP_Z_OFFSET
s_grasp, log = robust_grasp_v1(SIDE, obj_now.tolist(), hover_clearance=0.15, clearance_lift=0.0)
print(f"grasp: success={s_grasp}, log={log}")

if not s_grasp:
    print("Grasp failed — going home.")
    open_gripper(SIDE)
    go_home(SIDE)
else:
    # Step 3: Extract horizontally + lift to clear height (one call, no nudge loop needed)
    s_extract, log = toaster_extract_v1(SIDE, place_pos.tolist())
    print(f"extract+lift: success={s_extract}, log={log}")
    if not s_extract:
        print("Extraction failed — going home.")
        open_gripper(SIDE); go_home(SIDE)
    else:
        # Step 4: Move above counter target
        s_ph, log = hover_above_v1(SIDE, place_pos.tolist(), clearance=0.12)
        print(f"place-hover: success={s_ph}, log={log}")

        # Step 5: Lower and place
        s_place, log = vertical_place_v1(SIDE, place_pos.tolist(), z_offset=0.03)
        print(f"place: success={s_place}, log={log}")
        if not s_place:
            s_place, log = vertical_place_v1(SIDE, place_pos.tolist(), z_offset=0.05)
            print(f"place-shallow: success={s_place}, log={log}")

        open_gripper(SIDE); time.sleep(0.25)
        nudge(SIDE, delta_pos=[0.0, 0.0, 0.15])
        go_home(SIDE)

final = get_task_info()
print(f"Success: {final.get('success', False)}   Reward: {final.get('reward', 0.0)}")
```
