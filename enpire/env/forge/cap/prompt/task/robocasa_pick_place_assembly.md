# RoboCasa Pick and Place — Assembly Form (stage 2)

Task-specific orchestration guide for `PickPlaceCounterToCabinet`,
`PickPlaceSinkToCounter`, and similar pick-and-place tasks **under the
two-stage code generator**. The skill-author step runs before you and has
already written any new skills the task needs; your job is to import them
and orchestrate them into `code.py`.

See `cap/prompt/task/robocasa_pick_place_skills.md` for the complementary
stage-1 catalogue of what skills a pick-place task typically needs.

> **PickPlaceSinkToCounter — vision pipeline required.**
> Use the Vision-based assembly at the bottom of this file as your starting
> point. Do **not** use `get_task_info()["obj_pos"]` or
> `get_task_info()["container_pos"]` for localization — use
> `detect_objects_oneshot`, `sample_grasp_pose_anygrasp`, and
> `select_best_grasp` instead.

## Key facts (same as the legacy guide)

- **Single arm** — `list(state.arms.keys())[0]` → `"right"` for PandaOmron.
- **Ground-truth positions** from `get_task_info()`:
  - `obj_pos` — object to pick.
  - `obj_name` — display name.
  - `container_pos` — target container (preferred place target).
  - `distr_counter_pos` / `distr_cab_pos` — fallback place targets.
- **Place target priority**: `container_pos` > `distr_counter_pos` > `distr_cab_pos`.
- **Success signal**: `get_task_info()["success"]`.
- **Retract requirement**: RoboCasa only marks success when the EE is
  > 25 cm from the released object. After release, the EE is still right
  above the object — a single "return to neutral" call can fail if it
  plans from an in-collision start state, or finish at a pose that still
  sits within the 25 cm radius. End every attempt with an explicit retract
  that moves the EE clearly away from the released object, then verify
  with `get_task_info()["obj_to_robot0_eef_pos"]`. How to achieve the
  retract is your call — pick from the available skills.

## Reference assembly (validated on PickPlaceSinkToCounter)

This assembly has been validated in a real run and achieves fair performance.
Use it as the starting baseline — start from this code and only change what
the failure feedback explicitly identifies as wrong.

```python
THOUGHTS:
Incremental-descent grasp for sink pick-place. hover_above_v1 to approach,
incremental_grasp_v1 to descend and close, object-rise verification via nudge,
lift_v2 (incremental nudge lift) to escape sink, hover + place + retract.

```python
import numpy as np
from skill_library.hover_above    import hover_above_v1
from skill_library.incremental_grasp import incremental_grasp_v1
from skill_library.lift           import lift_v2
from skill_library.vertical_place import vertical_place_v1

SIDE = "right"

info = get_task_info()
obj_pos = np.array(info["obj_pos"])
obj_name = info.get("obj_name", "unknown")

if "container_pos" in info and info["container_pos"] is not None:
    place_pos = np.array(info["container_pos"])
    place_name = info.get("container_name", "container")
elif "distr_counter_pos" in info and info["distr_counter_pos"] is not None:
    place_pos = np.array(info["distr_counter_pos"])
    place_name = "counter"
else:
    place_pos = np.array(info["distr_cab_pos"])
    place_name = "cabinet"

print(f"Task: pick '{obj_name}' at {obj_pos.tolist()} -> '{place_name}' at {place_pos.tolist()}")

# Object-specific grasp parameters
small_objects = ("egg", "garlic", "mushroom", "lemon_wedge", "lemon", "lime", "cherry_tomato")
round_medium = ("apple", "orange", "peach")
if obj_name in small_objects:
    step_size, hover_cl, hold_str, max_retries, nudge_delta = 0.015, 0.12, 0.6, 5, -0.02
elif obj_name in round_medium:
    step_size, hover_cl, hold_str, max_retries, nudge_delta = 0.02, 0.12, 0.5, 4, -0.02
else:
    step_size, hover_cl, hold_str, max_retries, nudge_delta = 0.025, 0.15, 0.4, 3, -0.015

open_gripper(SIDE)
original_obj_z = obj_pos[2]

# Grasp: hover then incremental descent
s_grasp, log = incremental_grasp_v1(SIDE, obj_pos.tolist(),
                                     hover_clearance=hover_cl,
                                     step_size=step_size,
                                     hold_strength=hold_str)
print(f"incremental_grasp: success={s_grasp}, log={log}")

# Verify grasp by checking if object rises with a small test nudge
r_test = nudge(SIDE, delta_pos=[0.0, 0.0, 0.04])
info_check = get_task_info()
actually_grasped = (info_check["obj_pos"][2] - original_obj_z) > 0.015
print(f"grasp verified: {actually_grasped}")

# Retry loop with nudge-down regrasp
retries = 0
while not actually_grasped and retries < max_retries:
    nudge(SIDE, delta_pos=[0.0, 0.0, -0.04])
    open_gripper(SIDE)
    nudge(SIDE, delta_pos=[0.0, 0.0, nudge_delta])
    close_gripper(SIDE, compliant=True, hold_strength=hold_str)
    r_up = nudge(SIDE, delta_pos=[0.0, 0.0, 0.04])
    info_retry = get_task_info()
    actually_grasped = (info_retry["obj_pos"][2] - original_obj_z) > 0.015
    print(f"retry {retries}: grasped={actually_grasped}")
    retries += 1

if not actually_grasped:
    print("All grasp attempts failed — going home.")
    open_gripper(SIDE)
    go_home(SIDE)
else:
    # Lift using incremental nudge (more reliable from constrained sink positions)
    s_lift, log = lift_v2(SIDE, delta_z=0.25)
    print(f"lift: success={s_lift}, log={log}")

    s_ph, log = hover_above_v1(SIDE, place_pos.tolist(), clearance=0.12)
    print(f"place-hover: success={s_ph}, log={log}")

    s_place, log = vertical_place_v1(SIDE, place_pos.tolist(), z_offset=0.03)
    print(f"place: success={s_place}, log={log}")

    nudge(SIDE, delta_pos=[0.0, 0.0, 0.15])
    go_home(SIDE)

final = get_task_info()
print(f"Success: {final.get('success', False)}   Reward: {final.get('reward', 0.0)}")
```

## Advanced reference assembly (multi-orientation, multi-attempt)

When the skill library contains `hover_orientation_search_v1`,
`descend_and_grasp_v1`, `post_grasp_lift_v1`, and `place_with_orientation_v1`,
use this more capable assembly. It tries multiple gripper orientations for
hover and place, handles sink collisions, and runs up to 4 attempts with a
fallback orientation mode — matching the human-expert strategy.

```python
THOUGHTS:
Multi-orientation multi-attempt pick-place. Hover with z180/tilt search,
pre-descend+descend with IK retries, obj-rise verification, post-grasp
collision restore + lift, place with orientation candidates.

```python
import numpy as np
from skill_library.hover_orientation_search import hover_orientation_search_v1
from skill_library.descend_and_grasp import descend_and_grasp_v1
from skill_library.post_grasp_lift import post_grasp_lift_v1
from skill_library.place_with_orientation import place_with_orientation_v1

SIDE = "right"
PRIMARY_ATTEMPTS = 3
TOTAL_ATTEMPTS = PRIMARY_ATTEMPTS + 1

info = get_task_info()
obj_pos = np.array(info["obj_pos"])
container_pos = np.array(info["container_pos"] if info.get("container_pos") else
                          info.get("distr_counter_pos", info.get("distr_cab_pos")))
print(f"obj={obj_pos.tolist()} container={container_pos.tolist()}")

success = False
last_prefer_z180 = True

for attempt_idx in range(1, TOTAL_ATTEMPTS + 1):
    prefer_z180 = (last_prefer_z180 if attempt_idx < TOTAL_ATTEMPTS
                   else not last_prefer_z180)
    print(f"\n===== Attempt {attempt_idx}/{TOTAL_ATTEMPTS} prefer_z180={prefer_z180} =====")
    open_gripper(SIDE)

    # Refresh obj position
    info = get_task_info()
    obj_pos = np.array(info["obj_pos"])

    # Step 1: Hover with orientation search
    s_hover, hlog = hover_orientation_search_v1(SIDE, obj_pos.tolist(),
                                                 prefer_z180=prefer_z180)
    print(f"hover: success={s_hover} label={hlog['label']}")
    selected_quat = hlog["quat"]
    last_prefer_z180 = prefer_z180

    # Step 2: Descend and grasp (handles pre-descend, IK retries, collision disable)
    info = get_task_info()
    obj_pos = np.array(info["obj_pos"])
    s_grasp, glog = descend_and_grasp_v1(SIDE, obj_pos.tolist(), selected_quat)
    print(f"grasp: success={s_grasp} obj_rose={glog['obj_rose']:.4f}")

    if not s_grasp:
        print(f"Attempt {attempt_idx} grasp failed — going home")
        open_gripper(SIDE)
        update_planner_world()  # restore collision world
        go_home(SIDE)
        continue

    # Step 3: Post-grasp retreat + restore collisions + lift
    s_lift, llog = post_grasp_lift_v1(SIDE)
    print(f"lift: success={s_lift} final_z={llog['final_z']:.4f}")

    if not s_lift:
        print(f"Attempt {attempt_idx} lift failed — going home")
        open_gripper(SIDE)
        go_home(SIDE)
        continue

    # Step 4: Place with orientation candidates
    s_place, plog = place_with_orientation_v1(SIDE, container_pos.tolist())
    print(f"place: success={s_place}")

    final = get_task_info()
    success = bool(final.get("success", False))
    print(f"Attempt {attempt_idx} task success: {success}")
    if success:
        break

final = get_task_info()
print(f"Success: {final.get('success', False)}   Reward: {final.get('reward', 0.0)}")
```

## Variants and tuning — no new helpers needed

If reflection from the prior iteration asks for changes, adjust **arguments
to existing skills**, not by writing new helpers:

- Higher hover clearance for sink depth: `hover_above_v1(..., clearance=0.15)`.
- Firmer grip for slippery objects: `descend_and_grasp_v1(..., hold_strength=0.4)`.
- More descent retries: `descend_and_grasp_v1(..., descend_offsets=(0.025, 0.020, ...))`.
- Disable z180 preference: `hover_orientation_search_v1(..., prefer_z180=False)`.

If you reach for a helper the library doesn't have, don't write it inline.
Finish the assembly with what's available and rely on reflection to request
it from the skill-author on the next iter.

## Common failure modes — diagnostic steps in assembly

- **Success=False despite object on plate** — insufficient retract. The
  EE is still within 25 cm of the object, or the "return to neutral"
  skill failed with `Start state is colliding with world` because the
  post-release pose was in collision. Make the retract an explicit step,
  not a side effect of the final skill.
- **Gripper misses object** — object moved. Re-query
  `get_task_info()["obj_pos"]` immediately before `vertical_grasp_v1`.
- **Arm collides with counter** — bump `clearance=0.15`.
- **freespace_move returns "Success" but EE didn't reach** — cross-check
  `get_task_info()["obj_to_robot0_eef_pos"]`; fall through to
  `nudge_down_and_regrasp_v1` if z-distance is large.

---

## Vision-based assembly (PickPlaceSinkToCounter)

Use this structure when the skill library contains **vision perception skills**
(authored by the skill-author step) rather than oracle-position skills.
The assembly wraps everything in a `MAX_ATTEMPTS` retry loop; each attempt
goes home → detect → plan grasps → execute → lift → place.

Typical skill library imports after vision skill authoring:

| Skill | What it does |
|---|---|
| `detect_target_v1(target_query)` | SAM3 on top+right cameras; returns `np.ndarray` XYZ |
| `detect_object_wrist_v1(obj_query)` | SAM3 on wrist camera (from home, into sink); returns XYZ |
| `plan_and_rank_grasps_v1(obj_query, planner)` | AnyGrasp + cuRobo batch rank; returns `list[Candidate]` |
| `execute_grasp_v1(candidate, planner)` | Descend + close + width-check + nudge retries; returns `(bool, approach_dir)` |
| `lift_and_transit_home_v1(home_pos, home_quat, planner)` | Retreat + lift + transit to home |
| `vision_place_v1(tgt_pos, planner)` | Hover + lower + open + go_home |

**Reference assembly structure:**

```python
import time, re
import numpy as np
from scipy.spatial.transform import Rotation as R
from skill_library.detect_target       import detect_target_v1
from skill_library.detect_object_wrist import detect_object_wrist_v1
from skill_library.plan_and_rank_grasps import plan_and_rank_grasps_v1
from skill_library.execute_grasp       import execute_grasp_v1
from skill_library.lift_and_transit    import lift_and_transit_home_v1
from skill_library.vision_place        import vision_place_v1

SIDE = "right"
MAX_ATTEMPTS = 4

state0 = get_robot_state().arms[SIDE]
home_pos = np.array(state0.ee_pos, dtype=float)
home_quat = np.array(state0.ee_quat, dtype=float)
_planner = create_motion_planner()

desc = get_task_description()
m = re.search(r"[Pp]ick (?:up )?(?:the )?(.+?) from .+ place (?:it )?(?:on |in |into )?(?:the )?(.+?)(?:\.|$)", desc)
obj_query, target_query = (m.group(1).strip(), m.group(2).strip()) if m else ("object", "counter")

_, tgt_log = detect_target_v1(target_query)
tgt_pos = tgt_log["position"]

for attempt in range(1, MAX_ATTEMPTS + 1):
    go_home(SIDE)

    _, obj_log = detect_object_wrist_v1(obj_query)
    if not obj_log["success"]:
        print(f"Attempt {attempt}: detection failed"); continue

    _, grasp_log = plan_and_rank_grasps_v1(obj_query, _planner)
    if not grasp_log["feasible"]:
        go_home(SIDE); continue

    s_grasp, glog = execute_grasp_v1(grasp_log["candidates"], _planner)
    if not s_grasp:
        go_home(SIDE); continue

    lift_and_transit_home_v1(home_pos, home_quat, _planner)
    vision_place_v1(tgt_pos, _planner)
    break

final = get_task_info()
print(f"Success: {final.get('success', False)}  Reward: {final.get('reward', 0.0)}")
```

**If the skill library doesn't yet have these vision skills** (e.g. iteration 0),
fall back to the oracle assembly above — `get_task_info()["obj_pos"]` is always
available as a position source. The vision skills will be authored in a later
iteration once the oracle assembly has established a baseline.
