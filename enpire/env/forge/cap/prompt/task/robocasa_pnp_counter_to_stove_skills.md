# RoboCasa Pick-and-Place Counter→Stove — Skill Guide

Strategy guide for `PickPlaceCounterToStove` when the skill library is active.
The object sits on a counter; the target is a stove burner or surface.

## Key facts

- `obj_pos` — object to pick (on counter, z ≈ 0.85–1.0 m)
- `container_pos` / `distr_counter_pos` — stove target position
- **Place target is at or slightly below stove surface** — use `PLACE_CLEARANCE = -0.005`
  (negative means gripper stops just below the surface so the object rests cleanly)
- Shallow-release fallback at `+0.05 m` if the primary descent fails IK
- No sink or cabinet geometry — grasp and lift are straightforward

## Preferred grasp skill: `robust_grasp_v1`

Use `robust_grasp_v1` from the skill library. It tries multiple gripper
orientations during hover and descent (7 cardinal directions), verifies via
gripper-width check (`0.02 < width < 0.98`), and applies a small clearance
lift after closing. More reliable than `incremental_grasp_v1` on open surfaces.

```python
from skill_library.robust_grasp import robust_grasp_v1
s_grasp, log = robust_grasp_v1(SIDE, obj_pos.tolist(), hover_clearance=0.12)
```

## Key difference from sink-to-counter

- `PLACE_CLEARANCE = -0.005` — place slightly below surface so object rests flat
- Fallback to `STOVE_RELEASE_HEIGHT = +0.05` if primary place descent fails
- Release wait is longer (`0.40 s`) to let the object settle on the stove grating

## Common failure modes

- **Object bounces off stove**: `PLACE_CLEARANCE` too low — try `0.0` or `+0.01`
- **Place IK fails**: stove has different geometry than counter; try higher clearance first
  then descend, or use the shallow-release fallback at `+0.05`
- **Success=False after placing**: EE not retracted far enough — `go_home` must move
  arm > 25 cm from placed object

---

## Vision-based perception strategy (PickPlaceCounterToStove)

Use this when writing code that does **not** rely on `get_task_info()` oracle
positions. Localizes objects via SAM3 depth, plans grasps via AnyGrasp, ranks
them via cuRobo batch IK.

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

### Camera strategy — counter → stove

- **Object detection**: top camera first (object on counter, visible from above), fallback wrist
- **Wrist hover**: move wrist directly above object at +0.15 m with `down_quat` before AnyGrasp —
  gives a close-up view for better grasp quality
- **Grasp planning**: wrist camera (after hover), fallback to top camera
- **Target detection**: top and right cameras; queries `["pan on stove", "pan", "pot", target_query]`
- Detect target **once before** the attempt loop

### Key differences vs sink→counter

| | sink→counter | counter→stove |
|---|---|---|
| Object camera | wrist (from home) | top (from home), hover wrist |
| Grasp camera | wrist (from home) | wrist (after hover above object) |
| Target queries | `["round plate", target_query]` | `["pan on stove", "pan", "pot", target_query]` |
| `PLACE_LOWER_M` | `+0.03` | `-0.005` (slightly below stove surface) |
| Place fallback | vertical → current quat | primary → shallow `+0.05` |
| Post-place retract | `open → go_home` | `open → nudge([0,0,0.15]) → go_home` |
| Recheck nudge direction | downward `-z` | along approach_dir (upward `+d`) |

### Key parameters (validated in human oracle)

```python
LIFT_HEIGHT_M        = 0.20
PLACE_HOVER_M        = 0.12
PLACE_LOWER_M        = -0.005   # slightly below stove surface
PLACE_SHALLOW_M      = 0.05     # fallback if primary descent fails IK
POST_GRASP_RETREAT_M = 0.08
GRASP_RECHECK_NUDGES = [0.01, 0.02, 0.03]  # along approach_dir (not fixed -z)
TCP_OFFSET_M         = 0.02
MAX_ATTEMPTS         = 4
```

### Grasp verification via gripper width

```python
w = float(np.asarray(get_robot_state().arms[SIDE].gripper_pos, dtype=float).reshape(-1)[0])
grasped = 0.05 < w < 0.95
```

On miss, nudge **along approach direction** (not fixed downward) and retry:

```python
for nudge_d in GRASP_RECHECK_NUDGES:
    open_gripper(SIDE)
    nudge_vec = (nudge_d * approach_dir).tolist()
    nudge(side=SIDE, delta_pos=nudge_vec)
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
PLACE_HOVER_M, PLACE_LOWER_M, PLACE_SHALLOW_M = 0.12, -0.005, 0.05
GRASP_RECHECK_NUDGES = [0.01, 0.02, 0.03]

state0 = get_robot_state().arms[SIDE]
home_pos = np.array(state0.ee_pos, dtype=float)
home_quat = np.array(state0.ee_quat, dtype=float)
_planner = create_motion_planner()
down_quat = R.from_euler("xyz", [0, 180, 0], degrees=True).as_quat()

def refresh_planner(reason, exclude=None):
    info = update_planner_world(_planner, exclude_body_prefixes=exclude)
    print(f"[curobo] {reason}: {info['n_obstacles']} obstacles")

def check_grasp():
    w = float(np.asarray(get_robot_state().arms[SIDE].gripper_pos, dtype=float).reshape(-1)[0])
    return 0.05 < w < 0.95, w

def detect_target(target_query):
    best_score, best_pos = -1.0, None
    for cam in ("top", "right"):
        for q in ["pan on stove", "pan", "pot", target_query]:
            dets = detect_objects_oneshot(q, camera=cam).get(q, [])
            if dets and dets[0].score > best_score:
                best_score, best_pos = dets[0].score, np.array(dets[0].position_3d, dtype=float)
    if best_pos is None:
        raise RuntimeError("Could not detect target")
    return best_pos

def detect_object(obj_query):
    for cam in ("top", "wrist"):
        for q in [obj_query] + obj_query.split() + ["food", "object"]:
            dets = detect_objects_oneshot(q, camera=cam).get(q, [])
            if dets:
                return np.array(dets[0].position_3d, dtype=float)
    raise RuntimeError("Could not detect object")

def plan_grasps(obj_query, camera="wrist"):
    for gq in [obj_query] + obj_query.split() + ["food", "object"]:
        for tdonly in (True, False):
            try:
                grasps = sample_grasp_pose_anygrasp(gq, camera=camera, max_grasps=10,
                                                    top_down_only=tdonly,
                                                    disable_planner_z_clipping=True)
            except RuntimeError:
                continue
            if grasps:
                return grasps
    return []

desc = get_task_description()
m = re.search(r"[Pp]ick (?:up )?(?:the )?(.+?) from .+ place (?:it )?(?:on |in |into )?(?:the )?(.+?)(?:\.|$)", desc)
obj_query, target_query = (m.group(1).strip(), m.group(2).strip()) if m else ("object", "stove")
tgt_pos = detect_target(target_query)
print(f"Target: {tgt_pos.tolist()}")

for attempt in range(1, MAX_ATTEMPTS + 1):
    open_gripper(SIDE); refresh_planner("start of attempt"); go_home(SIDE)
    try:
        obj_pos = detect_object(obj_query)
    except RuntimeError as e:
        print(f"Detection failed: {e}"); continue

    # Hover wrist above object for close-up grasp planning
    hover_pos = obj_pos.copy(); hover_pos[2] += 0.15
    r = freespace_move(right_target_pos=hover_pos.tolist(),
                       right_target_quat=down_quat.tolist(), side=SIDE)
    print(f"Hover above object: {r.status}")

    grasps = plan_grasps(obj_query, camera="wrist")
    if not grasps:
        grasps = plan_grasps(obj_query, camera="top")
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
            for d in GRASP_RECHECK_NUDGES:
                open_gripper(SIDE); nudge(side=SIDE, delta_pos=(d * approach_dir).tolist())
                close_gripper(SIDE); time.sleep(0.25)
                grasped, _ = check_grasp()
                if grasped: break
        if grasped: break
        open_gripper(SIDE)
        rp = np.array(get_robot_state().arms[SIDE].ee_pos, dtype=float)
        rp[2] += POST_GRASP_RETREAT_M
        freespace_move(right_target_pos=rp.tolist(), side=SIDE, gripper=1.0)

    if not grasped:
        refresh_planner("restore"); go_home(SIDE); continue

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
    for _, quat in [("vertical", down_quat),
                    ("current", np.array(get_robot_state().arms[SIDE].ee_quat, dtype=float))]:
        r = freespace_move(right_target_pos=ph.tolist(), right_target_quat=quat.tolist(),
                           side=SIDE, gripper=0.1)
        if r.status == "Success": break

    pt = tgt_pos.copy(); pt[2] += PLACE_LOWER_M
    r = freespace_move(right_target_pos=pt.tolist(), side=SIDE, gripper=0.1)
    if r.status != "Success":
        pt[2] = tgt_pos[2] + PLACE_SHALLOW_M
        freespace_move(right_target_pos=pt.tolist(), side=SIDE, gripper=0.1)

    open_gripper(SIDE); time.sleep(0.40)
    nudge(SIDE, delta_pos=[0.0, 0.0, 0.15])
    go_home(SIDE)
    break

final = get_task_info()
print(f"Success: {final.get('success', False)}  Reward: {final.get('reward', 0.0)}")
```

### Vision failure modes

| Symptom | Fix |
|---|---|
| Detection returns empty for `obj_query` | Try simpler fallbacks: individual words → `"food"` → `"object"` |
| `sample_grasp_pose_anygrasp` raises | Try `top_down_only=True` first, then `False`; try `camera="top"` fallback |
| No `is_executable` candidates | `go_home(SIDE)` first to reset arm pose, then retry |
| Gripper closes on air (width ≈ 0 or ≈ 1) | Apply `GRASP_RECHECK_NUDGES` along `approach_dir` (not fixed -z) |
| Object slides off stove | `PLACE_LOWER_M` too low — try `0.0` then `+0.01` |
| Place hover IK fails with vertical orientation | Fall back to current EE orientation |
