# RoboCasa OpenCabinet — Skill-Library Strategy

Strategy guide for `OpenCabinet` tasks when the skill library is active.
The three library skills below encode the complete human-expert solution.
**Always import and use them directly** — do not re-implement cabinet
manipulation from scratch.

## `get_oracle_targets()` schema for OpenCabinet

```python
oracle = get_oracle_targets()
# oracle keys:
#   task_name       str   "OpenCabinet"
#   supported       bool  False when the fixture layout isn't supported
#   reason          str   explanation when supported=False
#   active_control  str   name of the primary handle (e.g. "handle" or "left_handle")
#   controls        dict  {control_name: target_dict, ...}  — may have 1+ handles
#   fixture_state   dict  {door_joint_name: float}  — current open fraction (0→1)
#   target          dict  target_dict for active_control (legacy single-handle path)
```

**Always validate before proceeding:**

```python
oracle = get_oracle_targets()
if oracle.get("task_name") != "OpenCabinet":
    raise RuntimeError(f"wrong task: {oracle.get('task_name')}")
if not oracle.get("supported"):
    raise RuntimeError(f"oracle target unsupported: {oracle.get('reason')}")
```

## Helper: `get_control_target(oracle, control_name)`

Returns the `target` dict for a named control, or `None` if unavailable.
Imported from `cap.saved_scripts.robocasa_skill_library.cabinet_control_state`.

```python
from cap.saved_scripts.robocasa_skill_library.cabinet_control_state import get_control_target
target = get_control_target(oracle, control_name)  # None → skip this control
```

## Helper: `door_progress(oracle, control_name)`

Returns the current open fraction (float 0→1) for a named control, or `None`.
Use to monitor progress and decide when the door is open enough.

```python
from cap.saved_scripts.robocasa_skill_library.cabinet_control_state import door_progress
progress = door_progress(oracle, control_name)  # None or 0.0–1.0
```

## The three library skills

### 1. `cabinet_handle_targets_v1(oracle)` — prioritise controls by progress gap

```python
from cap.saved_scripts.robocasa_skill_library.cabinet_handle_targets import cabinet_handle_targets_v1

_, target_log = cabinet_handle_targets_v1(oracle, set_markers=False)
control_names = target_log["ordered_controls"]   # list[str], highest-gap first
```

Returns `(True, {"ordered_controls": [...], "progress_by_control": {...}})`.

### 2. `cabinet_handle_grasp_v1(control_name, target)` — approach and grasp the handle

```python
from cap.saved_scripts.robocasa_skill_library.cabinet_handle_grasp import cabinet_handle_grasp_v1

target = get_control_target(oracle, control_name)
s_grasp, grasp_log = cabinet_handle_grasp_v1(control_name, target)
# grasp_log keys when successful:
#   arm, control_name, grasp_quat, pre_grasp, radial_xy,
#   selected_grasp_sign, travel_distance
```

Returns `(True, grasp_log)` on success, `(False, {"reason": str})` on failure.
Handles: home pose → staged approach → deep grasp → gripper close.

### 3. `cabinet_handle_finish_v1(control_name, grasp_log)` — pull the door open

```python
from cap.saved_scripts.robocasa_skill_library.cabinet_handle_finish import cabinet_handle_finish_v1

s_finish, finish_log = cabinet_handle_finish_v1(control_name, grasp_log)
# finish_log keys: best_progress (float, final door fraction)
```

Returns `(True, {"best_progress": float})` on success, `(False, {"reason": str})` on failure.
Handles: incremental pull steps → sign probe → release at 45% → vertical finish at 50%.

## Reference assembly (proven pattern from human oracle)

```python
from cap.saved_scripts.robocasa_skill_library.cabinet_handle_targets import cabinet_handle_targets_v1
from cap.saved_scripts.robocasa_skill_library.cabinet_handle_grasp import cabinet_handle_grasp_v1
from cap.saved_scripts.robocasa_skill_library.cabinet_handle_finish import cabinet_handle_finish_v1
from cap.saved_scripts.robocasa_skill_library.cabinet_control_state import get_control_target

oracle = get_oracle_targets()
if oracle.get("task_name") != "OpenCabinet":
    raise RuntimeError(f"wrong task: {oracle.get('task_name')}")
if not oracle.get("supported"):
    raise RuntimeError(f"oracle target unsupported: {oracle.get('reason')}")

_, target_log = cabinet_handle_targets_v1(oracle, set_markers=False)
control_names = target_log["ordered_controls"]
last_error = ""

for control_name in control_names:
    if get_task_info().get("success", False):
        print(f"Task already successful before {control_name}; stopping")
        break

    latest_oracle = get_oracle_targets()
    target = get_control_target(latest_oracle, control_name)
    if target is None:
        print(f"Skipping {control_name}: target unavailable")
        continue

    s_grasp, grasp_log = cabinet_handle_grasp_v1(control_name, target)
    print(f"{control_name} grasp: success={s_grasp}, log={grasp_log}")
    if not s_grasp:
        last_error = grasp_log.get("reason", "grasp failed")
        continue

    s_finish, finish_log = cabinet_handle_finish_v1(control_name, grasp_log)
    print(f"{control_name} finish: success={s_finish}, log={finish_log}")
    if not s_finish:
        last_error = finish_log.get("reason", "handle finish failed")
        continue

final = get_task_info()
print(f"Final state: {get_oracle_targets().get('fixture_state')}")
print(f"Success: {final.get('success', False)}")
print(f"Reward: {final.get('reward', 0.0)}")
if not final.get("success", False):
    raise RuntimeError(last_error or "cabinet not opened successfully")
```

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `grasp_log["reason"] == "failed to reach cabinet handle"` | Handle occluded or IK infeasible from current pose | `cabinet_handle_grasp_v1` retries via staged approach; if still failing, check `supported=True` and `target` non-None |
| `finish_log["reason"] == "pull step N failed"` | Pull direction wrong or collision | skill probes both pull signs on step 1 automatically |
| `best_progress` stalls below 0.5 | Door partially open but stuck | skill switches to vertical finish path automatically above 0.3 progress |
| `supported=False` | Fixture layout not recognised by oracle API | task cannot run oracle-assisted — check `oracle.get("reason")` |
| Multiple controls (`controls` dict has 2+ keys) | Double-door cabinet | `cabinet_handle_targets_v1` orders them by progress gap; loop handles each |
