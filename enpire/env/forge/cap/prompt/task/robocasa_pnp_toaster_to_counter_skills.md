# RoboCasa Pick-and-Place Toaster→Counter — Skill Guide

Strategy guide for `PickPlaceToasterToCounter` when the skill library is active.
The object is inside a toaster slot; extraction requires pulling along the
toaster-exit direction before lifting normally.

## Key facts

- `obj_pos` — object inside toaster (z ≈ 0.85 m, approach with +0.06 z offset)
- `container_pos` / `distr_counter_pos` — counter target position
- Object is **inside a toaster slot** — cannot lift straight up; must pull out horizontally first
- `TOASTER_GRASP_Z_OFFSET = 0.06` — grasp 6 cm above object centroid (fingers close around slot)
- **`toaster_extract_v1` handles the complete exit** — horizontal pull + vertical lift to clear height.
  Do NOT add a separate nudge-lift loop after calling it.

## Why nudge-lift fails after extraction

After horizontal extraction, the cuRobo collision world still sees the arm as inside
toaster geometry (BVH update is not instant). Regular `nudge` checks start-state
collision before issuing OSC commands — it gets blocked and barely moves (+3 mm instead
of +50 mm). `nudge_brutal` bypasses the planner entirely (raw OSC ticks) and lifts the
arm regardless of the stale collision state. `toaster_extract_v1` uses `nudge_brutal`
internally for the vertical phase.

## Grasp strategy

1. Hover above the toaster opening (`robust_grasp_v1` with `clearance_lift=0.0`)
2. Descend with **+0.06 z offset** (grasps the upper part of the object through the slot)
3. Close gripper (no clearance lift — arm is still inside the slot)
4. Call `toaster_extract_v1` — it handles horizontal pull + vertical lift in one call
5. Proceed directly to counter hover + place

## Skill: toaster_extract_v1 (updated)

```python
from skill_library.robust_grasp import robust_grasp_v1
from skill_library.toaster_extract import toaster_extract_v1
from skill_library.hover_above import hover_above_v1
from skill_library.vertical_place import vertical_place_v1

info = get_task_info()
obj_now = np.array(info["obj_pos"]); obj_now[2] += 0.06
s_grasp, log = robust_grasp_v1(SIDE, obj_now.tolist(), hover_clearance=0.15, clearance_lift=0.0)

if s_grasp:
    s_extract, log = toaster_extract_v1(SIDE, place_pos.tolist())
    # arm is now outside slot and lifted to clear height — go straight to counter
    if s_extract:
        hover_above_v1(SIDE, place_pos.tolist(), clearance=0.12)
        vertical_place_v1(SIDE, place_pos.tolist(), z_offset=0.03)
```

## Common failure modes

- **Grasp IK fails at toaster depth**: use `TOASTER_GRASP_Z_OFFSET=0.06` — grasp above centroid
- **Object doesn't come out**: extraction distance too short — `toaster_extract_v1` tries 0.10, 0.14, 0.18 m
- **Nudge stalls during lift** (old pattern): do NOT use incremental nudge after extraction — use `toaster_extract_v1` which lifts via freespace_move
