# RoboCasa Pick-and-Place Counter→Cabinet — Assembly Guide

Orchestration guide for `PickPlaceCounterToCabinet` under the two-stage code generator.
The skill-author step has already written the needed skills; your job is to import and
sequence them correctly.

## Key facts

- Object is on a **counter** (not a sink) — no collision-avoidance escape needed for lift.
- Target is **inside a cabinet** — must approach from the front opening, insert, settle,
  then retract back out before go_home.
- Use `cab_approach_pos`, `cab_insert_pos`, `cab_settle_pos`, `cab_front_normal` from
  `get_task_info()` if available; fall back to `distr_cab_pos` if not.
- Retract requirement: EE must be > 25 cm from object after release — always retract
  to `cab_approach_pos` before calling `go_home`.

## Reference assembly (validated pattern)

```python
THOUGHTS:
Counter-to-cabinet: hover above object, grasp with obj-rise verification, lift,
approach cabinet opening with inward orientation, insert to insert_pos, settle,
release, retract to approach_pos, go_home.

```python
import numpy as np
from skill_library.hover_above import hover_above_v1
from skill_library.incremental_grasp import incremental_grasp_v1
from skill_library.lift import lift_v1
from skill_library.cabinet_approach import cabinet_approach_v1
from skill_library.cabinet_insert import cabinet_insert_v1

SIDE = "right"

info = get_task_info()
obj_pos = np.array(info["obj_pos"])
obj_name = info.get("obj_name", "unknown")

# Cabinet waypoints — use oracle positions if available
cab_approach = np.array(info["cab_approach_pos"]) if "cab_approach_pos" in info else None
cab_insert = np.array(info["cab_insert_pos"]) if "cab_insert_pos" in info else None
cab_settle = np.array(info["cab_settle_pos"]) if "cab_settle_pos" in info else None
cab_front_normal = np.array(info["cab_front_normal"]) if "cab_front_normal" in info else None
distr_cab = np.array(info["distr_cab_pos"]) if "distr_cab_pos" in info else None

# Estimate front_normal from current EE position if not provided
if cab_front_normal is None and distr_cab is not None:
    state = get_robot_state()
    ee_pos = np.array(state.arms[SIDE].ee_pos)
    front = ee_pos[:2] - distr_cab[:2]
    n = np.linalg.norm(front)
    cab_front_normal = np.array([front[0]/n, front[1]/n, 0.0]) if n > 1e-6 else np.array([1., 0., 0.])

# Determine insert target
insert_target = cab_insert if cab_insert is not None else (distr_cab if distr_cab is not None else obj_pos + np.array([0.5, 0., 0.]))
settle_target = cab_settle  # None is fine — cabinet_insert_v1 skips settle if None

# Determine approach waypoint
if cab_approach is not None:
    approach_target = cab_approach
elif cab_front_normal is not None and insert_target is not None:
    approach_target = insert_target + cab_front_normal * 0.20
    approach_target[2] = insert_target[2]
else:
    approach_target = insert_target

print(f"Task: pick '{obj_name}' from counter -> place in cabinet")
print(f"obj_pos={obj_pos.tolist()} insert={insert_target.tolist()} approach={approach_target.tolist()}")

# Step 1: Open gripper
open_gripper(SIDE)
print("Gripper opened")

original_obj_z = obj_pos[2]

# Step 2: Hover above object
s_hover, log = hover_above_v1(SIDE, obj_pos.tolist(), clearance=0.12)
print(f"hover: success={s_hover}, log={log}")

# Refresh position before grasp
info = get_task_info()
obj_pos = np.array(info["obj_pos"])

# Step 3: Grasp with obj-rise verification
s_grasp, log = incremental_grasp_v1(SIDE, obj_pos.tolist())
print(f"grasp: success={s_grasp}, log={log}")

retries = 0
while not s_grasp and retries < 3:
    nudge(SIDE, delta_pos=[0.0, 0.0, -0.04])
    open_gripper(SIDE)
    nudge(SIDE, delta_pos=[0.0, 0.0, -0.02])
    close_gripper(SIDE, compliant=True, hold_strength=0.4)
    r_up = nudge(SIDE, delta_pos=[0.0, 0.0, 0.04])
    info_retry = get_task_info()
    s_grasp = (info_retry["obj_pos"][2] - original_obj_z) > 0.015
    print(f"retry {retries}: grasped={s_grasp}")
    retries += 1

if not s_grasp:
    print("All grasp attempts failed — going home.")
    open_gripper(SIDE)
    go_home(SIDE)
else:
    # Step 4: Lift off counter
    s_lift, log = lift_v1(SIDE, delta_z=0.20)
    print(f"lift: success={s_lift}, log={log}")

    # Step 5: Approach cabinet opening
    s_approach, log = cabinet_approach_v1(SIDE, approach_target.tolist(),
                                           cab_front_normal.tolist() if cab_front_normal is not None else [1.,0.,0.])
    print(f"cabinet_approach: success={s_approach}, log={log}")

    # Step 6: Insert into cabinet and settle
    s_insert, log = cabinet_insert_v1(SIDE, insert_target.tolist(),
                                       settle_pos=settle_target.tolist() if settle_target is not None else None)
    print(f"cabinet_insert: success={s_insert}, log={log}")

    # Step 7: Release
    open_gripper(SIDE)
    print("Object released inside cabinet")

    # Step 8: Retract to approach position (critical for > 25 cm retract requirement)
    r_retract = freespace_move(right_target_pos=approach_target.tolist(), side=SIDE, gripper=1.0)
    print(f"retract to approach: status={r_retract.status}")

    # Step 9: Go home
    go_home(SIDE)

final = get_task_info()
print(f"Success: {final.get('success', False)}   Reward: {final.get('reward', 0.0)}")
```

## Common failure patterns and fixes

- **Cabinet approach IK fails**: try `hover_above_v1` over `approach_pos` first to get arm
  into a good configuration, then move to `approach_pos` with orientation.
- **Insert IK fails**: the arm is not aligned with the cabinet opening. Add a pre-insert
  nudge along `+front_normal` direction to back out slightly before trying again.
- **Object drops**: place_clearance in `insert_target` should be at or slightly above the
  cabinet floor (`z = cab_settle_pos[2] + 0.01`).
- **Retract fails**: arm is inside cabinet in collision. Nudge backward along
  `+front_normal` before calling `freespace_move` to approach_pos.
