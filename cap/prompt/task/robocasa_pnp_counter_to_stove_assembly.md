# RoboCasa Pick-and-Place Counter→Stove — Assembly Guide

Orchestration guide for `PickPlaceCounterToStove`.

## Key facts

- Object on counter, target on stove surface — no sink or cabinet complexity
- Place slightly **below** stove surface (`z_offset = -0.005`) so object rests flat
- Shallow-release fallback at `+0.05 m` if primary place descent fails
- Longer release wait: `0.40 s`

## Reference assembly

```python
THOUGHTS:
Counter-to-stove: hover, grasp with obj-rise check, lift, hover above stove,
lower with negative clearance, release with long wait, retract, go_home.

```python
import numpy as np
import time
from skill_library.robust_grasp import robust_grasp_v1
from skill_library.lift import lift_v1
from skill_library.hover_above import hover_above_v1
from skill_library.vertical_place import vertical_place_v1

SIDE = "right"
PLACE_CLEARANCE = -0.005
STOVE_RELEASE_HEIGHT = 0.05

info = get_task_info()
obj_pos = np.array(info["obj_pos"])
obj_name = info.get("obj_name", "unknown")

place_pos = np.array(
    info["container_pos"] if "container_pos" in info and info["container_pos"] is not None
    else info.get("distr_counter_pos", info.get("distr_cab_pos", obj_pos))
)
print(f"Task: pick '{obj_name}' -> stove at {place_pos.tolist()}")

open_gripper(SIDE)

info = get_task_info()
obj_pos = np.array(info["obj_pos"])
s_grasp, log = robust_grasp_v1(SIDE, obj_pos.tolist(), hover_clearance=0.12)
print(f"grasp: success={s_grasp}, log={log}")

if not s_grasp:
    print("Grasp failed — going home.")
    open_gripper(SIDE)
    go_home(SIDE)
else:
    s_lift, log = lift_v1(SIDE, delta_z=0.20)
    print(f"lift: success={s_lift}, log={log}")

    s_ph, log = hover_above_v1(SIDE, place_pos.tolist(), clearance=0.12)
    print(f"place-hover: success={s_ph}, log={log}")

    s_place, log = vertical_place_v1(SIDE, place_pos.tolist(), z_offset=PLACE_CLEARANCE)
    print(f"place: success={s_place}, log={log}")

    if not s_place:
        print("Primary place failed — shallow release fallback")
        s_place, log = vertical_place_v1(SIDE, place_pos.tolist(), z_offset=STOVE_RELEASE_HEIGHT)
        print(f"place-shallow: success={s_place}, log={log}")

    time.sleep(0.40)
    nudge(SIDE, delta_pos=[0.0, 0.0, 0.15])
    go_home(SIDE)

final = get_task_info()
print(f"Success: {final.get('success', False)}   Reward: {final.get('reward', 0.0)}")
```

---

## Vision-based assembly (PickPlaceCounterToStove)

Use this structure when the skill library contains **vision perception skills**.
Key structural difference from sink→counter: the arm hovers wrist above the
object before grasp planning, and the post-place retract is a `nudge([0,0,0.15])`
before `go_home`.

Typical skill library imports after vision skill authoring:

| Skill | What it does |
|---|---|
| `detect_stove_target_v1(target_query)` | SAM3 on top+right; queries `["pan on stove","pan","pot",target_query]`; returns XYZ |
| `detect_object_top_v1(obj_query)` | SAM3 on top+wrist cameras (object on counter); returns XYZ |
| `hover_wrist_above_v1(obj_pos)` | Moves wrist to `obj_pos + [0,0,0.15]` with `down_quat` for close-up view |
| `plan_and_rank_grasps_v1(obj_query, planner, camera)` | AnyGrasp (wrist first, top fallback) + cuRobo batch rank; returns `list[Candidate]` |
| `execute_grasp_v1(candidates, planner, approach_dir_mode)` | Descend + close + width-check + nudge-along-approach retries |
| `lift_and_transit_home_v1(home_pos, home_quat, planner)` | Retreat + lift + transit to home |
| `stove_place_v1(tgt_pos, planner)` | Hover + lower to `-0.005` (fallback `+0.05`) + open + nudge up + go_home |

**Reference assembly structure:**

```python
import time, re
import numpy as np
from scipy.spatial.transform import Rotation as R
from skill_library.detect_stove_target  import detect_stove_target_v1
from skill_library.detect_object_top    import detect_object_top_v1
from skill_library.hover_wrist_above    import hover_wrist_above_v1
from skill_library.plan_and_rank_grasps import plan_and_rank_grasps_v1
from skill_library.execute_grasp        import execute_grasp_v1
from skill_library.lift_and_transit     import lift_and_transit_home_v1
from skill_library.stove_place          import stove_place_v1

SIDE = "right"
MAX_ATTEMPTS = 4

state0 = get_robot_state().arms[SIDE]
home_pos = np.array(state0.ee_pos, dtype=float)
home_quat = np.array(state0.ee_quat, dtype=float)
_planner = create_motion_planner()

desc = get_task_description()
m = re.search(r"[Pp]ick (?:up )?(?:the )?(.+?) from .+ place (?:it )?(?:on |in |into )?(?:the )?(.+?)(?:\.|$)", desc)
obj_query, target_query = (m.group(1).strip(), m.group(2).strip()) if m else ("object", "stove")

_, tgt_log = detect_stove_target_v1(target_query)
tgt_pos = tgt_log["position"]

for attempt in range(1, MAX_ATTEMPTS + 1):
    open_gripper(SIDE); go_home(SIDE)

    _, obj_log = detect_object_top_v1(obj_query)
    if not obj_log["success"]:
        print(f"Attempt {attempt}: detection failed"); continue

    hover_wrist_above_v1(obj_log["position"])

    _, grasp_log = plan_and_rank_grasps_v1(obj_query, _planner, camera="wrist")
    if not grasp_log["feasible"]:
        go_home(SIDE); continue

    s_grasp, glog = execute_grasp_v1(grasp_log["candidates"], _planner)
    if not s_grasp:
        go_home(SIDE); continue

    lift_and_transit_home_v1(home_pos, home_quat, _planner)
    stove_place_v1(tgt_pos, _planner)
    break

final = get_task_info()
print(f"Success: {final.get('success', False)}  Reward: {final.get('reward', 0.0)}")
```

**If the skill library doesn't yet have these vision skills** (e.g. iteration 0),
fall back to the oracle assembly above — `get_task_info()["obj_pos"]` is always
available. The vision skills will be authored in a later iteration.
