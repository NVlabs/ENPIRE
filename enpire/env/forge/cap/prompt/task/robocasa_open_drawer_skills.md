# RoboCasa OpenDrawer — Skill-Library Strategy

Strategy guide for `OpenDrawer` tasks when the skill library is active.
The three library skills below encode the complete human-expert solution.
**Always import and use them directly** — do not re-implement drawer
manipulation from scratch.

## `get_oracle_targets()` schema for OpenDrawer

```python
oracle = get_oracle_targets()
# oracle keys:
#   task_name       str   "OpenDrawer"
#   supported       bool  False when the fixture layout isn't supported
#   reason          str   explanation when supported=False
#   active_control  str   name of the primary handle (e.g. "handle")
#   controls        dict  {control_name: target_dict, ...}  — may have 1+ handles
#   fixture_state   dict  {drawer_joint_name: float}  — current open fraction (0→1)
#   target          dict  target_dict for active_control (legacy single-handle path)
```

**Always validate before proceeding:**

```python
oracle = get_oracle_targets()
if oracle.get("task_name") != "OpenDrawer":
    raise RuntimeError(f"wrong task: {oracle.get('task_name')}")
if not oracle.get("supported"):
    raise RuntimeError(f"oracle target unsupported: {oracle.get('reason')}")
```

## Helper: `get_control_target(oracle, control_name)`

Returns the `target` dict for a named control, or `None` if unavailable.
Imported from `cap.saved_scripts.robocasa_skill_library.drawer_control_state`.

```python
from cap.saved_scripts.robocasa_skill_library.drawer_control_state import get_control_target
target = get_control_target(oracle, control_name)  # None → skip this control
```

## Helper: `drawer_progress(oracle, control_name)`

Returns the current open fraction (float 0→1) for a named control, or `None`.

```python
from cap.saved_scripts.robocasa_skill_library.drawer_control_state import drawer_progress
progress = drawer_progress(oracle, control_name)  # None or 0.0–1.0
```

## The three library skills

### 1. `drawer_handle_targets_v1(oracle)` — prioritise controls by progress gap

```python
from cap.saved_scripts.robocasa_skill_library.drawer_handle_targets import drawer_handle_targets_v1

_, target_log = drawer_handle_targets_v1(oracle, set_markers=False)
control_names = target_log["ordered_controls"]   # list[str], highest-gap first
```

Returns `(True, {"ordered_controls": [...], "progress_by_control": {...}})`.

### 2. `drawer_handle_grasp_v1(control_name, target)` — approach and grasp the handle

```python
from cap.saved_scripts.robocasa_skill_library.drawer_handle_grasp import drawer_handle_grasp_v1

target = get_control_target(oracle, control_name)
s_grasp, grasp_log = drawer_handle_grasp_v1(control_name, target)
# grasp_log keys when successful:
#   arm, control_name, grasp_quat, pre_grasp, surface_normal,
#   selected_grasp_sign, selected_tilt_deg, travel_distance, retreat_distance
```

Returns `(True, grasp_log)` on success, `(False, {"reason": str})` on failure.
Handles: home pose → pre-grasp approach (tilt candidates) → deep grasp steps → gripper close.

### 3. `drawer_handle_finish_v1(control_name, grasp_log)` — pull the drawer open

```python
from cap.saved_scripts.robocasa_skill_library.drawer_handle_finish import drawer_handle_finish_v1

s_finish, finish_log = drawer_handle_finish_v1(control_name, grasp_log)
# finish_log keys: best_progress (float, final drawer fraction)
```

Returns `(True, {"best_progress": float})` on success, `(False, {"reason": str})` on failure.
Handles: incremental pull steps → pull-sign probe on step 1 → sign flip on stall → stops at progress ≥ 0.95.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `grasp_log["reason"] == "failed to reach pre_grasp"` | All tilt candidates infeasible | Check `supported=True` and `target` non-None; skill tries 5 tilt angles (0°, ±8°, ±15°) |
| `finish_log["reason"] == "pull step N failed"` | Pull direction wrong | Skill probes both pull signs on step 1 automatically; sign-flip probe also applied after low progress |
| `best_progress` stalls before 0.95 | Drawer stuck or collision | Skill applies `max_stalled_pull_steps_in_a_row=6` guard then gives up |
| `supported=False` | Fixture layout not recognised by oracle API | Task cannot run oracle-assisted — check `oracle.get("reason")` |
| Multiple controls (`controls` dict has 2+ keys) | Multi-handle drawer | `drawer_handle_targets_v1` orders them by progress gap; loop handles each |
