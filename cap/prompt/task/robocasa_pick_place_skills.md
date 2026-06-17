# RoboCasa Pick and Place — Skill-Library Form

Strategy guide for `PickPlaceCounterToCabinet`, `PickPlaceSinkToCounter`, and similar tasks, **when the skill library is active**. This prompt replaces `task/robocasa_pick_place` when `agent: skill_library` is used. All code is organized as atomic `*_v1` helpers so the library can accumulate reusable skills across iterations.

> **PickPlaceSinkToCounter — vision pipeline required.**
> For this task, localize objects and targets using **SAM3 + AnyGrasp + cuRobo**
> (see the Vision-based perception strategy section at the bottom of this file).
> Do **not** use `get_task_info()["obj_pos"]` or `get_task_info()["container_pos"]`
> as position sources — treat those as unavailable for this task.
> `get_task_info()["success"]` and `get_task_info()["obj_name"]` are still valid.

## Key facts (unchanged from base guide)

- **Single arm** — use `list(state.arms.keys())[0]` to get the arm name (always `"right"` for PandaOmron).
- **Ground-truth positions** — `get_task_info()` provides oracle XYZ for objects and targets:
  - `obj_pos` — object to pick
  - `obj_name` — object name (for vision/grasp queries)
  - `container_pos` — target container (preferred place target)
  - `distr_counter_pos` — target counter surface (fallback place target)
  - `distr_cab_pos` — target cabinet (for counter-to-cabinet tasks)
- **Place target resolution** — use `container_pos` if present, else `distr_counter_pos` or `distr_cab_pos`.
- **Success signal** — `get_task_info()["success"]` is `True` when the task is complete.
- **Motion** — use `freespace_move(right_target_pos=...)`; it plans collision-free via cuRobo automatically.
- **Retract** — RoboCasa requires the gripper >25 cm from the object for success; always end with `go_home(side)`.

## Helper naming convention — MANDATORY

Every atomic helper gets a `_v1` suffix (the library auto-bumps later versions). The base name must **encode the assumption the helper bakes in** so a future run's LLM can pick the right variant from the index:

| Assumption baked in | Example name |
|---|---|
| EE orientation fixed to vertical (no orientation arg) | `vertical_grasp_v1`, `vertical_place_v1` |
| Orientation comes from AnyGrasp / learned pose | `anygrasp_grasp_v1` |
| Orientation reused from current EE state | `reuse_orientation_place_v1` |
| Pure z-offset hover above XY | `hover_above_v1` |
| Small delta move in EE frame | `nudge_down_and_regrasp_v1` |
| Lift in +z by a fixed delta | `lift_v1` |

**Rule**: if you hardcode a vertical grasp (no orientation argument, no quaternion, EE drops straight down), the base name **must** contain `vertical_grasp`. If you compute orientation from some tool, pick a name that says so. Never hide a hardcoded assumption behind a neutral name like `grasp_v1`.

## Three-layer code structure — MANDATORY

Every script you write has exactly these three layers, in order:

```python
# Layer 1 — import existing skills from the library. ONLY import names that
# appear in the "Available skills" index you were given. Do NOT wrap these
# imports in try/except. If a skill is not in the index (e.g. iter 0 when
# the library is empty), leave Layer 1 empty and define the helper in Layer 2.
from skill_library.vertical_grasp import vertical_grasp_v1
from skill_library.hover_above   import hover_above_v1

# Layer 2 — define NEW atomic helpers (plain def, no decorator)
#   - each returns (success_bool, {"success": bool, "time_s": float, ...})
#   - never calls another top-level function in this file
#   - only calls tools from the namespace (freespace_move, set_gripper, ...)
def lift_v1(side, delta_z=0.20):
    ...
    return success, {"success": success, "time_s": ..., "delta_z": delta_z}

# Layer 3 — assembly: orchestrate the helpers at module top level (not inside a function)
info = get_task_info()
obj_pos = np.array(info["obj_pos"])
...
s, log = lift_v1(SIDE, delta_z=0.20)
...
```

**Hard rules** (violating any of these blocks promotion to the library):

1. Helpers live at **module top level** — not nested inside another function.
2. Helpers **return `(val, {...})`** — the second element is a **dict literal** or a local variable bound to a dict. Never a bare value, never a function call.
3. Helpers **do not call other helpers** defined in this file, and **do not call** anything imported from `skill_library.*`. They only call namespace tools (`freespace_move`, `close_gripper`, `get_task_info`, `np.*`, etc.).
4. Assembly (Layer 3) lives at **module top level**, not wrapped in `def run_task()`. Wrapping it in a function is fine but the function will never be promoted (it calls helpers) — which is OK, but the main flow must execute at import time so the executor sees its side effects.

## Advanced skill set (multi-orientation, collision-aware)

The following skills encode the human-expert strategy for PickPlaceSinkToCounter.
Prefer these when the simpler skills are failing:

| Skill | What it does |
|-------|-------------|
| `hover_orientation_search_v1` | Tries z180 + tilt variants of the current EE orientation; returns the label+quat that succeeded so descend can reuse it |
| `descend_and_grasp_v1` | Pre-descend → final descent with IK-retry z-offsets → disable collision world → close gripper → nudge-down retry loop → obj-rise verification |
| `post_grasp_lift_v1` | Retreat upward → restore collision world (excluding robot+object) → lift to clear sink |
| `place_with_orientation_v1` | Tries 6 place orientations for hover; lowers and releases; calls go_home |

These four skills together replace `hover_above_v1` + `incremental_grasp_v1` + `lift_v2` + `vertical_place_v1` with a more robust strategy. The assembly wraps them in a 3+1 attempt loop — first 3 prefer z180 hover orientation, fallback attempt flips to default.

Key improvements over the basic stack:
- **Multi-orientation hover** avoids IK failures from a fixed vertical approach
- **Collision disable during grasp** prevents the planner from refusing descent into the sink
- **Retreat + collision restore** before lift ensures the lift path is collision-free
- **Multiple attempts** with orientation fallback handles diverse object positions

## Grasp success verification

After closing the gripper and escaping any sink-wall collision with
`nudge_brutal`, verify the grasp by checking whether the object rose with
the arm using `get_task_info()["obj_pos"][2]`. This is ground-truth —
no sensor threshold or camera interpretation needed.

**Pattern:**
```python
info_before = get_task_info()
obj_z_before = info_before["obj_pos"][2]
close_gripper(side, compliant=True, hold_strength=hold_strength)
nudge_brutal(side=side, delta_pos=[0.0, 0.0, 0.05])   # escape collision
info_after = get_task_info()
grasped = (info_after["obj_pos"][2] - obj_z_before) > 0.015
```

Use this inside any grasp `@skill`. Do not use gripper force/has_object
alone — these produce false positives when fingers close on air or walls.
Do not use VLM camera checks for grasp verification — the wrist-camera view
after a nudge_brutal escape is ambiguous and also produces false positives.

## Proven skill implementations (validated on PickPlaceSinkToCounter)

These skills have been validated in a real run and achieve fair performance. Reuse them directly via the skill library — do not reinvent them. Only refine a skill if the library index shows its `success_rate` is below 0.8.

```python
@skill
def hover_above_v1(side, xyz, clearance=0.12):
    """Move EE above xyz by clearance metres. No orientation change."""
    import numpy as np
    target = np.array(xyz, dtype=float).copy()
    target[2] += clearance
    r = freespace_move(right_target_pos=target.tolist(), side=side)
    success = r.status == "Success"
    return success, {"success": success, "status": r.status, "target": target.tolist()}

@skill
def incremental_grasp_v1(side, obj_pos, hover_clearance=0.15, step_size=0.025, hold_strength=0.4):
    """Hover above obj_pos then nudge down incrementally to grasp.
    After closing, escapes sink collision with nudge_brutal then verifies
    via VLM wrist-camera check — more reliable than sensor heuristics alone.
    """
    import numpy as np
    hover_target = np.array(obj_pos, dtype=float).copy()
    hover_target[2] += hover_clearance
    r_hover = freespace_move(right_target_pos=hover_target.tolist(), side=side)
    if r_hover.status != "Success":
        hover_target[2] += 0.05
        r_hover = freespace_move(right_target_pos=hover_target.tolist(), side=side)
        if r_hover.status != "Success":
            return False, {"success": False, "reason": "hover_failed"}
    target_z = obj_pos[2]
    state = get_robot_state()
    current_z = state.arms[side].ee_pos[2]
    max_steps = min(int((current_z - target_z) / step_size) + 3, 15)
    descent_ok = True
    for i in range(max_steps):
        state = get_robot_state()
        current_z = state.arms[side].ee_pos[2]
        if current_z <= target_z + 0.005:
            break
        r_nudge = nudge(side=side, delta_pos=[0.0, 0.0, -step_size])
        if not r_nudge.success:
            descent_ok = False
            break
    info_before = get_task_info()
    obj_z_before = info_before["obj_pos"][2]
    close_gripper(side, compliant=True, hold_strength=hold_strength)
    # Escape collision before verifying
    nudge_brutal(side=side, delta_pos=[0.0, 0.0, 0.05])
    # Ground-truth: did the object rise with the arm?
    info_after = get_task_info()
    obj_rose = info_after["obj_pos"][2] - obj_z_before
    grasped = obj_rose > 0.015
    final_state = get_robot_state()
    return grasped, {"success": grasped, "descent_ok": descent_ok, "obj_rose": round(obj_rose, 4),
                     "final_z": final_state.arms[side].ee_pos[2], "target_z": target_z}

@skill
def lift_v2(side, delta_z=0.20, step_size=0.04):
    """Lift EE by +delta_z using incremental nudge steps.
    More reliable than single freespace_move from constrained sink positions
    where the arm may be in light contact with the basin walls.
    """
    import numpy as np
    state = get_robot_state()
    start_z = state.arms[side].ee_pos[2]
    target_z = start_z + delta_z
    n_steps = min(int(delta_z / step_size) + 1, 10)
    all_ok = True
    for i in range(n_steps):
        state = get_robot_state()
        current_z = state.arms[side].ee_pos[2]
        if current_z >= target_z - 0.005:
            break
        remaining = target_z - current_z
        r = nudge(side=side, delta_pos=[0.0, 0.0, min(step_size, remaining)])
        if not r.success:
            all_ok = False
            break
    final_state = get_robot_state()
    final_z = final_state.arms[side].ee_pos[2]
    achieved = final_z - start_z
    success = achieved > delta_z * 0.5
    return success, {"success": success, "all_nudges_ok": all_ok,
                     "achieved_delta": achieved, "target_delta": delta_z}

@skill
def vertical_place_v1(side, target_pos, z_offset=0.03):
    """Lower EE to target_pos + [0,0,z_offset] and open gripper. No orientation change."""
    import numpy as np
    target = np.array(target_pos, dtype=float).copy()
    target[2] += z_offset
    r = freespace_move(right_target_pos=target.tolist(), side=side)
    lowered = r.status == "Success"
    open_gripper(side)
    return lowered, {"success": lowered, "status": r.status, "target": target.tolist()}
```

## Reference: vertical-grasp pick-place (decomposed form)

Oracle positions, hardcoded vertical EE orientation, two-tier nudge-down retry. Good starting template for small flat / wedge-shaped objects in a sink or on a counter.

```python
import numpy as np

SIDE = "right"

# ---------- Layer 2: atomic helpers ----------
# IMPORTANT: each helper imports its own dependencies (numpy, scipy) INSIDE the
# function body. Module-level imports are not extracted when the SkillPromoter
# saves the helper to skill_library/, so later iterations that re-import the
# skill would hit NameError.
# Note: do NOT measure time yourself — the @skill decorator applied at save
# time injects "time_s" into the log automatically.

def hover_above_v1(side, xyz, clearance=0.12):
    """Move EE to (xyz_x, xyz_y, xyz_z + clearance). No orientation change."""
    import numpy as np
    target = np.array(xyz, dtype=float).copy()
    target[2] += clearance
    r = freespace_move(right_target_pos=target.tolist(), side=side)
    success = r.status == "Success"
    return success, {
        "success": success,
        "status": r.status,
        "target": target.tolist(),
    }

def vertical_grasp_v1(side, obj_pos, z_offset=0.0, hold_strength=0.2):
    """Descend straight down onto obj_pos (no orientation change), compliant-close, verify grasp.

    Grasp orientation is whatever the EE currently holds — we do NOT set a quaternion.
    Suitable for small flat objects where a top-down grip is safe.
    """
    import numpy as np
    target = np.array(obj_pos, dtype=float).copy()
    target[2] += z_offset
    r = freespace_move(right_target_pos=target.tolist(), side=side)
    descend_ok = r.status == "Success"

    close_gripper(side, compliant=True, hold_strength=hold_strength)
    info = get_gripper_info(side)
    grasped = bool(
        info.get("has_object")
        and not info.get("is_fully_closed", False)
        and (info.get("actuator_force_N") or 0.0) > 1.0
    )
    return grasped, {
        "success": grasped,
        "descend_status": r.status,
        "descend_ok": descend_ok,
        "gripper_info": info,
        "target": target.tolist(),
        "z_offset": z_offset,
    }

def nudge_down_and_regrasp_v1(side, delta_z=-0.02, hold_strength=0.2):
    """Re-open gripper, nudge down delta_z in EE frame, compliant-close, verify."""
    open_gripper(side)
    nudge(side=side, delta_pos=[0.0, 0.0, delta_z])
    close_gripper(side, compliant=True, hold_strength=hold_strength)
    info = get_gripper_info(side)
    grasped = bool(
        info.get("has_object")
        and not info.get("is_fully_closed", False)
        and (info.get("actuator_force_N") or 0.0) > 1.0
    )
    return grasped, {
        "success": grasped,
        "delta_z": delta_z,
        "gripper_info": info,
    }

def lift_v1(side, delta_z=0.20):
    """Lift EE by +delta_z from current position. Preserves XY and orientation."""
    import numpy as np
    state = get_robot_state()
    cur = np.array(state.arms[side].ee_pos)
    target = cur.copy()
    target[2] += delta_z
    r = freespace_move(right_target_pos=target.tolist(), side=side)
    success = r.status == "Success"
    return success, {
        "success": success,
        "status": r.status,
        "delta_z": delta_z,
        "start_pos": cur.tolist(),
    }

def vertical_place_v1(side, target_pos, z_offset=0.03):
    """Lower EE to (target_pos + [0,0,z_offset]) and fully open gripper to release.

    No orientation change — the EE keeps whatever it held after the grasp.
    """
    import numpy as np
    target = np.array(target_pos, dtype=float).copy()
    target[2] += z_offset
    r = freespace_move(right_target_pos=target.tolist(), side=side)
    lowered = r.status == "Success"
    open_gripper(side)
    return lowered, {
        "success": lowered,
        "status": r.status,
        "target": target.tolist(),
        "z_offset": z_offset,
    }

# ---------- Layer 3: assembly + main loop ----------

info = get_task_info()
obj_pos = np.array(info["obj_pos"])
if "container_pos" in info:
    place_pos = np.array(info["container_pos"])
elif "distr_counter_pos" in info:
    place_pos = np.array(info["distr_counter_pos"])
else:
    place_pos = np.array(info["distr_cab_pos"])

print(f"obj={obj_pos.tolist()} place={place_pos.tolist()}")

open_gripper(SIDE)

s_hover, log = hover_above_v1(SIDE, obj_pos.tolist(), clearance=0.12)
print(f"hover: {log}")

# Refresh — object may have moved during hover
obj_pos = np.array(get_task_info()["obj_pos"])
s_grasp, log = vertical_grasp_v1(SIDE, obj_pos.tolist(), z_offset=0.0)
print(f"grasp: {log}")

retries = 0
while not s_grasp and retries < 2:
    s_grasp, log = nudge_down_and_regrasp_v1(SIDE, delta_z=-0.02)
    print(f"retry {retries}: {log}")
    retries += 1

if not s_grasp:
    print("All grasp attempts failed — going home.")
    open_gripper(SIDE)
    go_home(SIDE)
else:
    s_lift, log = lift_v1(SIDE, delta_z=0.20)
    print(f"lift: {log}")

    s_ph, log = hover_above_v1(SIDE, place_pos.tolist(), clearance=0.12)
    print(f"place-hover: {log}")

    s_place, log = vertical_place_v1(SIDE, place_pos.tolist(), z_offset=0.03)
    print(f"place: {log}")

    go_home(SIDE)

final = get_task_info()
print(f"Success: {final.get('success', False)}   Reward: {final.get('reward', 0.0)}")
```

## Tuning knobs (mapped to helper parameters)

- **Grasp clearance**: `vertical_grasp_v1(..., z_offset=0.005)` for objects that sit on a surface the gripper tips would otherwise hit.
- **Clamp force**: `vertical_grasp_v1(..., hold_strength=0.6)` if the object slips during lift; `0.1` if wedge-shaped objects squirt out.
- **Retry strategy**: if misses are lateral rather than vertical, write a new helper `nudge_lateral_and_regrasp_v1(side, dy)` — **do not** overload `nudge_down_and_regrasp_v1` with a lateral delta; that would hide the assumption and break the naming convention.
- **Hover clearance**: `hover_above_v1(..., clearance=0.15)` in cabinet/shelf scenes.

## When to write a non-vertical grasp

If the object is tall / has non-trivial shape and you need AnyGrasp, write a **separate** `anygrasp_grasp_v1(side, object_name, camera, max_grasps)` helper — do **not** extend `vertical_grasp_v1`. The library relies on the name telling the next iteration's LLM which assumptions hold.

Skeleton (fill in based on AnyGrasp output):

```python
def anygrasp_grasp_v1(side, object_name, camera="wrist", max_grasps=8, hold_strength=0.2):
    """Sample grasp poses via AnyGrasp, try the top-scoring one, compliant-close, verify."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    grasps = sample_grasp_pose_anygrasp(object_name=object_name, camera=camera, max_grasps=max_grasps)
    if not grasps:
        return False, {"success": False, "reason": "no_grasps"}
    g = grasps[0]
    # ... convert g.rpy to quat, call freespace_move with right_target_pos + right_target_quat,
    # ... close_gripper, verify via get_gripper_info
    return grasped, {"success": grasped, "grasp": {"pos": g.position, "rpy": g.rpy}}
```

## Common failure modes (same diagnoses, helper-mapped fixes)

- **Success=False despite object on plate** — gripper within 25 cm. Fix: always call `go_home(SIDE)` at the end of assembly.
- **Gripper misses object** — object moved. Fix: re-query `get_task_info()["obj_pos"]` just before calling `vertical_grasp_v1`.
- **Arm collides with counter** — `hover_above_v1(..., clearance=0.12)` minimum; try `0.15` if still hitting.
- **Object drops on retract** — insufficient clamp force. Tune `vertical_grasp_v1(..., hold_strength=0.3~0.6)`.
- **freespace_move reports "Success" but EE didn't reach** — `hover_above_v1` / `vertical_grasp_v1` will still return `success=True` from the planner's perspective. Cross-check `get_task_info()["obj_to_robot0_eef_pos"]` in assembly and fall through to `nudge_down_and_regrasp_v1` if z-distance is large.
- **All AnyGrasp grasps fail cuRobo IK** — try `go_home(SIDE)` first, then retry. Or fall back to `vertical_grasp_v1` for cube-ish objects.
- **Lift reports `Planning_Failed: Start state is colliding with world`** — the arm is physically stuck in contact with the sink wall or basin after grasping. cuRobo refuses to plan from a colliding start state, so `freespace_move` and `nudge` both fail immediately. Use `nudge_brutal` first to escape the contact before the main lift. `nudge_brutal` bypasses cuRobo's collision check and directly drives the OSC controller. A small upward escape (e.g. `[0, 0, 0.05]`) is usually enough to clear the wall, after which normal `nudge` or `freespace_move` can proceed. Author a lift skill like `escape_and_lift_v1` that calls `nudge_brutal` as the first step.

---

## Vision-based perception strategy (PickPlaceSinkToCounter)

Use this when writing code that does **not** rely on `get_task_info()` oracle
positions. Localizes objects via SAM3 depth, plans grasps via AnyGrasp, and
ranks them via cuRobo batch IK.

### Vision tools

| Tool | Key arguments | Returns |
|---|---|---|
| `get_task_description()` | — | `str` natural language task instruction |
| `detect_objects_oneshot(query, camera)` | `query: str`, `camera: str` | `dict[str, list[Detection]]`; each `det` has `.position_3d`, `.score` |
| `sample_grasp_pose_anygrasp(query, camera, max_grasps, top_down_only, disable_planner_z_clipping)` | all keyword | `list[Grasp]`; each `g` has `.position`, `.rpy`, `.score` |
| `select_best_grasp(grasps, side, batch_top_k, augment_yaw_flip)` | all keyword | `BatchGraspResult`; `.batch_candidates: list[Candidate]` |
| `create_motion_planner()` | — | planner handle |
| `update_planner_world(planner, exclude_body_prefixes)` | `planner`, `exclude: list[str]\|None` | `dict` with `n_obstacles` |
| `display_rpy_to_quat(rpy)` | `rpy: list[float]` | `list[float]` quaternion |

`Candidate` fields: `is_executable`, `is_ik_failed`, `is_planning_failed`,
`position`, `rpy`, `score`, `rank`, `source_index`, `ik_error_m`, `planner_status`.

### Camera strategy — sink → counter

- **Object detection**: wrist camera (arm at home, wrist looks down into sink)
- **Target detection**: top and right cameras (plate/counter, unobstructed)
- Detect target **once before** the attempt loop — it doesn't move
- Detect object **inside each attempt** after going home

### Key parameters (validated in human oracle)

```python
HOVER_CLEARANCE_M    = 0.12
LIFT_HEIGHT_M        = 0.20
PLACE_HOVER_M        = 0.12
PLACE_LOWER_M        = 0.03
POST_GRASP_RETREAT_M = 0.08
GRASP_RECHECK_NUDGES = [-0.01, -0.02, -0.03]  # downward z nudges on miss
TCP_OFFSET_M         = 0.02   # offset grasp_pos along approach_dir
MAX_ATTEMPTS         = 4
```

### Grasp verification via gripper width

```python
w = float(np.asarray(get_robot_state().arms[SIDE].gripper_pos, dtype=float).reshape(-1)[0])
grasped = 0.05 < w < 0.95   # fully open ≈ 1.0, fully closed ≈ 0.0
```

On miss, nudge downward and retry:

```python
for dz in GRASP_RECHECK_NUDGES:
    open_gripper(SIDE)
    nudge(side=SIDE, delta_pos=[0.0, 0.0, dz])
    close_gripper(SIDE)
    # re-check width
```

### Reference assembly (proven vision pipeline)

```python
import time, re
import numpy as np
from scipy.spatial.transform import Rotation as R

SIDE = "right"
MAX_ATTEMPTS, TCP_OFFSET_M = 4, 0.02
LIFT_HEIGHT_M, POST_GRASP_RETREAT_M = 0.20, 0.08
PLACE_HOVER_M, PLACE_LOWER_M = 0.12, 0.03
GRASP_RECHECK_NUDGES = [-0.01, -0.02, -0.03]

state0 = get_robot_state().arms[SIDE]
home_pos = np.array(state0.ee_pos, dtype=float)
home_quat = np.array(state0.ee_quat, dtype=float)
_planner = create_motion_planner()

def refresh_planner(reason, exclude=None):
    info = update_planner_world(_planner, exclude_body_prefixes=exclude)
    print(f"[curobo] {reason}: {info['n_obstacles']} obstacles")

def check_grasp():
    w = float(np.asarray(get_robot_state().arms[SIDE].gripper_pos, dtype=float).reshape(-1)[0])
    return 0.05 < w < 0.95, w

def detect_target(target_query):
    best_score, best_pos = -1.0, None
    for cam in ("top", "right"):
        for q in ["round plate", target_query]:
            dets = detect_objects_oneshot(q, camera=cam).get(q, [])
            if dets and dets[0].score > best_score:
                best_score, best_pos = dets[0].score, np.array(dets[0].position_3d, dtype=float)
    if best_pos is None:
        raise RuntimeError("Could not detect target")
    return best_pos

def detect_object_wrist(obj_query):
    for q in [obj_query] + obj_query.split() + ["food", "object in sink", "object"]:
        dets = detect_objects_oneshot(q, camera="wrist").get(q, [])
        if dets:
            return np.array(dets[0].position_3d, dtype=float)
    raise RuntimeError("Could not detect object")

desc = get_task_description()
m = re.search(r"[Pp]ick (?:up )?(?:the )?(.+?) from .+ place (?:it )?(?:on |in |into )?(?:the )?(.+?)(?:\.|$)", desc)
obj_query, target_query = (m.group(1).strip(), m.group(2).strip()) if m else ("object", "counter")
tgt_pos = detect_target(target_query)
print(f"Target: {tgt_pos.tolist()}")

for attempt in range(1, MAX_ATTEMPTS + 1):
    go_home(SIDE)
    try:
        obj_pos = detect_object_wrist(obj_query)
    except RuntimeError as e:
        print(f"Detection failed: {e}"); continue

    open_gripper(SIDE)
    refresh_planner("run start")

    grasps = []
    for gq in [obj_query] + obj_query.split() + ["food", "object"]:
        if grasps: break
        for tdonly in (True, False):
            try:
                grasps = sample_grasp_pose_anygrasp(gq, camera="wrist", max_grasps=10,
                                                    top_down_only=tdonly,
                                                    disable_planner_z_clipping=True)
            except RuntimeError:
                continue
            if grasps: break

    if not grasps:
        open_gripper(SIDE); refresh_planner("restore"); go_home(SIDE); continue

    refresh_planner("grasp approach (collisions off)", exclude=[""])
    result = select_best_grasp(grasps, side=SIDE, batch_top_k=16, augment_yaw_flip=True)
    feasible = [c for c in (result.batch_candidates or []) if c.is_executable]
    if not feasible:
        open_gripper(SIDE); refresh_planner("restore"); go_home(SIDE); continue

    grasped = False
    for candidate in feasible[:3]:
        grasp_pos = np.array(candidate.position, dtype=float)
        grasp_quat = display_rpy_to_quat(candidate.rpy)
        approach_dir = R.from_quat(grasp_quat).apply([0, 0, 1])
        r = freespace_move(right_target_pos=(grasp_pos + TCP_OFFSET_M * approach_dir).tolist(),
                           right_target_quat=grasp_quat.tolist(), side=SIDE)
        if r.status != "Success": continue
        close_gripper(SIDE); time.sleep(0.25)
        grasped, _ = check_grasp()
        if not grasped:
            for dz in GRASP_RECHECK_NUDGES:
                open_gripper(SIDE); nudge(side=SIDE, delta_pos=[0.0, 0.0, dz])
                close_gripper(SIDE); time.sleep(0.25)
                grasped, _ = check_grasp()
                if grasped: break
        if grasped: break
        open_gripper(SIDE)
        rp = np.array(get_robot_state().arms[SIDE].ee_pos, dtype=float)
        rp[2] += POST_GRASP_RETREAT_M
        freespace_move(right_target_pos=rp.tolist(), side=SIDE, gripper=1.0)

    if not grasped:
        open_gripper(SIDE); refresh_planner("restore"); go_home(SIDE); continue

    rp = np.array(get_robot_state().arms[SIDE].ee_pos, dtype=float)
    rp[2] += POST_GRASP_RETREAT_M
    freespace_move(right_target_pos=rp.tolist(), side=SIDE, gripper=0.1)
    refresh_planner("post grasp", exclude=["robot0", "gripper", "mobilebase", "obj"])
    lp = np.array(get_robot_state().arms[SIDE].ee_pos, dtype=float)
    lp[2] += LIFT_HEIGHT_M
    freespace_move(right_target_pos=lp.tolist(), side=SIDE, gripper=0.1)

    freespace_move(right_target_pos=home_pos.tolist(), right_target_quat=home_quat.tolist(),
                   side=SIDE, gripper=0.1)

    ph = tgt_pos.copy(); ph[2] += PLACE_HOVER_M
    down_quat = R.from_euler("xyz", [0, 180, 0], degrees=True).as_quat()
    for _, quat in [("vertical", down_quat),
                    ("current", np.array(get_robot_state().arms[SIDE].ee_quat, dtype=float))]:
        r = freespace_move(right_target_pos=ph.tolist(), right_target_quat=quat.tolist(),
                           side=SIDE, gripper=0.1)
        if r.status == "Success": break
    pt = tgt_pos.copy(); pt[2] += PLACE_LOWER_M
    freespace_move(right_target_pos=pt.tolist(), side=SIDE, gripper=0.1)
    open_gripper(SIDE); time.sleep(0.40); go_home(SIDE)
    break

final = get_task_info()
print(f"Success: {final.get('success', False)}  Reward: {final.get('reward', 0.0)}")
```

### Vision failure modes

| Symptom | Fix |
|---|---|
| Detection returns empty for `obj_query` | Try simpler fallbacks: individual words → `"food"` → `"object in sink"` |
| `sample_grasp_pose_anygrasp` raises | Try `top_down_only=True` first, then `False`; try simpler query strings |
| No `is_executable` candidates | `go_home(SIDE)` first to reset arm pose, then retry |
| Gripper closes on air (width ≈ 0 or ≈ 1) | Apply `GRASP_RECHECK_NUDGES` (-0.01, -0.02, -0.03 m downward) |
| Place hover IK fails with vertical orientation | Fall back to current EE orientation |
