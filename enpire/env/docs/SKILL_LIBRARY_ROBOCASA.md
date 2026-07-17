# Skill Library — RoboCasa

> **Cross-references**: [SKILL_LIBRARY](SKILL_LIBRARY.md) | [ROBOCASA_INTEGRATION](ROBOCASA_INTEGRATION.md) | [ROBOCASA_RANDOMNESS](ROBOCASA_RANDOMNESS.md) | [CUROBO_UPDATE_SUMMARY](CUROBO_UPDATE_SUMMARY.md)

---

## Overview

RoboCasa provides 365 household robot tasks across kitchen environments. Two robot embodiments are supported: **PandaOmron** (single-arm) and **GR1ArmsOnly** (bimanual). Two execution modes exist:

| Mode | Controller | Primary motion tool | IK method | Execution path |
|------|-----------|---------------------|-----------|----------------|
| **CapServer** | `osc_pose` (pinned) | `freespace_move` | cuRobo on remote cloud GPU | Portal RPC → CapServer → RoboCasaEnv |
| **Direct (run_agent.py default)** | `osc_pose` (pinned) | `freespace_move` | cuRobo on remote cloud GPU | In-process → RoboCasaEnv |

**Controller pinning**: RoboCasaEnv is always OSC_POSE. cuRobo joint trajectories are
tracked via per-waypoint FK + OSC tick-stepping (`_execute_osc_trajectory` in
`cap/env/robocasa/skills.py`). The `controller_type` kwarg is silently dropped
(`cap/env/__init__.py:85-87`) and `ROBOCASA_CONTROLLER_TYPE` is no longer read.

## Embodiments

### PandaOmron (Single-Arm)

- **Arms**: `right` only — always pass `side='right'`
- **Cameras**: `top`, `wrist`
- **DOF**: 7 joints + 1 gripper
- **Control freq**: 20 Hz
- **Mobile base**: Yes (but not controlled by CAP tools)
- **Embodiment spec**: `cap/prompt/embodiment/robocasa_panda.md`

### GR1ArmsOnly (Bimanual)

- **Arms**: `left`, `right`
- **Cameras**: `top`, `wrist`
- **DOF**: 7 joints + 1 gripper per arm
- **Control freq**: 20 Hz
- **Mobile base**: No
- **Embodiment spec**: `cap/prompt/embodiment/robocasa_gr1.md`

---

## Mode A: CapServer + cuRobo (Default)

### Available Tools

The agent prompt exposes a focused subset. All tools from the full registry are technically callable but the following are recommended:

| Tool | Notes |
|------|-------|
| `get_robot_state()` | Returns `RobotState` with per-arm `ArmState` |
| `freespace_move(right_target_pos, right_target_quat, side)` | **Primary motion** — cuRobo collision-free planning on remote GPU, joint trajectory executed locally. Quaternion format `[x,y,z,w]`. |
| `go_home(side)` | Plans trajectory back to initial EE pose via cuRobo |
| `nudge(side, delta_pos, delta_rpy)` | Small delta move via cuRobo planning |
| `set_gripper(side, pos)` | 0.0=closed, 1.0=open |
| `open_gripper(side)` | Shortcut for `set_gripper(side, 1.0)` |
| `close_gripper(side)` | Shortcut for `set_gripper(side, 0.0)` |
| `get_camera_image(camera)` | `"top"` or `"wrist"` |
| `detect_object(query, backend='oracle')` | Ground-truth positions from task info |
| `get_task_info()` | Task state, object positions, success flag |
| `vlm_query(text, camera)` | Vision-language queries (Gemini, etc.) |

### Key Behaviors

- **Always use `freespace_move`** for all arm motion — it plans collision-free trajectories via cuRobo. No IK setup needed.
- **Always reuse `ee_quat`** from `get_robot_state()`. Never compute quaternions manually (exception: tasks requiring non-standard orientations like microwave button pressing).
- **Oracle detection is preferred** — `detect_object(query, backend='oracle')` fuzzy-matches query against object names in `get_task_info()` keys ending in `_pos`. No vision server needed.
- **Check `FreespaceResult.status`** — `freespace_move` returns a `FreespaceResult`. Check `.status == "Success"` before proceeding to the next step.
- **Gripper**: Call `set_gripper` (or `open_gripper`/`close_gripper`) as a separate step after motion completes.

### Task Info Keys (from `get_task_info()`)

| Key | Type | Description |
|-----|------|-------------|
| `done` | bool | Episode finished |
| `reward` | float | Current reward |
| `success` | bool | Task completed successfully |
| `env_name` | str | RoboCasa task name |
| `obj_pos` | list[float] | Object-to-pick XYZ |
| `obj_name` | str | Object display name |
| `container_pos` | list[float] | Target container XYZ (if applicable) |
| `distr_counter_pos` | list[float] | Target counter surface XYZ |
| `distr_cab_pos` | list[float] | Target cabinet XYZ |

### Pick Pattern

```python
import numpy as np

state = get_robot_state()
arm = list(state.arms.keys())[0]  # "right" for PandaOmron
ee_quat = state.arms[arm].ee_quat

task_info = get_task_info()
obj_pos = np.array(task_info["obj_pos"])

# 1. Open gripper
open_gripper(arm)

# 2. Hover 12cm above
hover = obj_pos.copy(); hover[2] += 0.12
result = freespace_move(right_target_pos=hover.tolist(), right_target_quat=ee_quat, side=arm)
if result.status != "Success":
    print(f"Hover failed: {result.reason}")

# 3. Descend (re-read position — object may have moved)
obj_pos = np.array(get_task_info()["obj_pos"])
freespace_move(right_target_pos=obj_pos.tolist(),
               right_target_quat=get_robot_state().arms[arm].ee_quat, side=arm)

# 4. Close gripper
close_gripper(arm)

# 5. Verify grasp
if get_robot_state().arms[arm].gripper_pos < 0.005:
    print("Gripper closed on nothing — grasp missed")

# 6. Lift
lift = np.array(get_robot_state().arms[arm].ee_pos); lift[2] += 0.25
freespace_move(right_target_pos=lift.tolist(),
               right_target_quat=get_robot_state().arms[arm].ee_quat, side=arm)
```

### Place Pattern

```python
# Resolve target (priority: container > counter > cabinet)
info = get_task_info()
if "container_pos" in info:
    place_pos = np.array(info["container_pos"])
elif "distr_counter_pos" in info:
    place_pos = np.array(info["distr_counter_pos"])
else:
    place_pos = np.array(info["distr_cab_pos"])

# Approach above target
approach = place_pos.copy(); approach[2] += 0.15
freespace_move(right_target_pos=approach.tolist(),
               right_target_quat=get_robot_state().arms[arm].ee_quat, side=arm)

# Lower, release, retract
freespace_move(right_target_pos=place_pos.tolist(),
               right_target_quat=get_robot_state().arms[arm].ee_quat, side=arm)
open_gripper(arm)
retract = np.array(get_robot_state().arms[arm].ee_pos); retract[2] += 0.20
freespace_move(right_target_pos=retract.tolist(),
               right_target_quat=get_robot_state().arms[arm].ee_quat, side=arm)
```

---

## Mode B: Direct (in-process, no CapServer)

This is the default mode for `run_agent.py` (Hydra-configured, no CLI flag). The env
bypasses CapServer entirely — tool callables are built by
`cap/env/robocasa/skills.py:make_namespace()` via `cap/env/setup.py:create_runtime`.
Same cuRobo backend as CapServer mode; difference is execution path (in-process vs Portal RPC).

### Available Tools (Direct Mode)

| Tool | Notes |
|------|-------|
| `get_robot_state()` | Same `RobotState` format |
| `freespace_move(right_target_pos, right_target_quat, side, planning_speed)` | **Primary motion** — cuRobo collision-free planning on remote GPU, joint trajectory executed locally |
| `set_gripper(side, pos)` | Steps env synchronously until gripper settles |
| `open_gripper(side)` | Shortcut |
| `close_gripper(side)` | Shortcut |
| `go_home(side)` | Plans trajectory to initial EE pose via cuRobo (no scene teleport) |
| `nudge(side, delta_pos, delta_rpy)` | Small delta move via cuRobo planning |
| `get_camera_image(camera)` | Direct env render |
| `get_task_info()` | Same task info dict |
| `reset_env()` | Deterministic reset (restores initial sim state snapshot) |
| `reset_to_initial()` | Deterministic reset (same scene, same positions) |
| `vlm_query(text, camera, ...)` | Same multi-backend VLM queries |
| `detect_object(query, backend='oracle')` | Oracle detection from task info |

### Key Differences from CapServer Mode

- **Same `freespace_move` API** — `right_target_pos` / `right_target_quat`, quaternion `[x, y, z, w]`.
- **cuRobo runs on a remote GPU** — connect via `CAP_CUROBO_HOST` / `CAP_CUROBO_PORT` env vars.
- **`go_home` plans a trajectory** instead of teleporting the scene.
- **No `grasp` / `place` high-level tools** — compose from `freespace_move` + `set_gripper`.
- **No BundleSDF / tracking tools** — only oracle detection available.
- **Gripper steps env synchronously** — `set_gripper` drives the gripper by stepping the simulation.

### Pick Pattern (cuRobo)

```python
import numpy as np

state = get_robot_state()
arm = list(state.arms.keys())[0]
ee_quat = state.arms[arm].ee_quat

task_info = get_task_info()
obj_pos = np.array(task_info["obj_pos"])

# 1. Open gripper
open_gripper(arm)

# 2. Hover above object
hover = obj_pos.copy(); hover[2] += 0.12
result = freespace_move(right_target_pos=hover.tolist(), right_target_quat=ee_quat, side=arm)
if result.status != "Success":
    print(f"Hover failed: {result.reason}")

# 3. Descend
obj_pos = np.array(get_task_info()["obj_pos"])
result = freespace_move(right_target_pos=obj_pos.tolist(),
                        right_target_quat=get_robot_state().arms[arm].ee_quat, side=arm)

# 4. Close gripper
close_gripper(arm)

# 5. Lift
lift = np.array(get_robot_state().arms[arm].ee_pos); lift[2] += 0.20
freespace_move(right_target_pos=lift.tolist(),
               right_target_quat=get_robot_state().arms[arm].ee_quat, side=arm)
```

### Environment Variables (Direct Mode)

| Variable | Purpose | Default |
|----------|---------|---------|
| `CAP_CUROBO_HOST` | cuRobo server hostname/IP | `127.0.0.1` |
| `CAP_CUROBO_PORT` | cuRobo Portal RPC port | `0` (auto-start local) |
| `CAP_ROBOT_TYPE` | Robot model (consumed by freespace_move/run_script; not by RoboCasa skills — planner is hardcoded to `panda` in `skills.py:317`) | `yam` |
| `ROBOCASA_LAYOUT_ID` | Pin kitchen layout (1–60, or group shortcut -1 to -6) | random |
| `ROBOCASA_STYLE_ID` | Pin kitchen style (1–60, or group shortcut -1 to -6) | random |
| `ROBOCASA_SEED` | Random seed for determinism | random |

---

## AnyGrasp Integration

For objects with non-trivial geometry, use AnyGrasp to generate feasible grasp poses instead of going straight to oracle positions.

1. **Sample grasps**: `sample_grasp_pose_anygrasp(object_name, camera, max_grasps=8)`
2. **Camera fallback**: Try wrist camera first (closer, better depth), fall back to top.
3. **Feasibility ranking**: Plan each candidate through cuRobo — pick the first IK-feasible one.
4. **Orientation conversion**: AnyGrasp returns RPY in display convention. Convert: `euler_xyz = [-pitch, roll, -yaw - 90]` then `Rotation.from_euler("xyz", radians).as_quat()`.

---

## Task-Specific Strategy Guides

Strategy guides live in `cap/prompt/task/` and are injected into agent prompts based on the active task:

| Guide | File | Tasks |
|-------|------|-------|
| Pick & Place | `cap/prompt/task/robocasa_pick_place.md` | PickPlaceCounterToCabinet, PickPlaceSinkToCounter, etc. (non-skill-library mode) |
| Pick & Place (skill library) | `cap/prompt/task/robocasa_pick_place_skills.md` | Same tasks when `agent: skill_library` is active — three-layer decomposed form with `*_v1` helpers |
| Start Microwave | `cap/prompt/task/robocasa_start_microwave.md` | TurnOnMicrowave |
| Table Bussing | `cap/prompt/task/table_bussing.md` | Table clearing (YAM bimanual) |

### Task-Specific Notes

**Pick & Place**: Straightforward — detect object, hover, descend, grasp, lift, move to target, place. Uses `freespace_move()` exclusively for all arm motion. See `robocasa_pick_place.md` for the full strategy.

**Pick & Place (skill library)**: When `agent: skill_library` is active the experiment YAML should swap `task/robocasa_pick_place` → `task/robocasa_pick_place_skills`. The skills variant shows the same motion plan decomposed into named `*_v1` atomic helpers (`vertical_grasp_v1`, `hover_above_v1`, `lift_v1`, `vertical_place_v1`, `nudge_down_and_regrasp_v1`) so the `SkillPromoter` can accumulate them into `{run_dir}/skill_library/` across iterations. Helper names **must** encode hardcoded assumptions (e.g. `vertical_grasp_v1` for a top-down grasp with no orientation argument, `anygrasp_grasp_v1` for a learned-pose grasp).

**Start Microwave**: Requires computing a face-forward quaternion — the default downward-facing `ee_quat` cannot reach vertical microwave buttons. Must use `scipy.spatial.transform.Rotation` to compute orientation, then try multiple roll angles. See `robocasa_start_microwave.md` for full template.

---

## Common Failure Modes

| Failure | Cause | Fix |
|---------|-------|-----|
| Gripper misses object | Object moved between hover and descend | Re-read `get_task_info()["obj_pos"]` just before final descent |
| Arm collides with counter | Insufficient hover height | Add 10–12cm hover. Use cuRobo for collision-free planning |
| Object drops on retract | Weak grasp | Call `set_gripper(arm, 0.0)` again after lifting |
| `freespace_move` returns `IK_Failed` | Target out of reach or beyond joint limits | Reduce offset, lift higher first, check `result.reason` |
| `freespace_move` returns `Planning_Failed` | Collision in path | Try `go_home()` first to clear, then replan |
| AnyGrasp returns 0 candidates | Camera angle bad | Try different camera (wrist vs top) |
| All cuRobo plans fail | Target in collision or beyond joint limits | Check `result["status_detail"]`, try `go_home()` first |
| OSC can't reach vertical surface | Downward-facing gripper | Compute face-forward quaternion (see microwave task) |

## Key Files

- `cap/env/robocasa/env.py` — `RoboCasaEnv` (EnvProtocol + EefControlProtocol + TaskProtocol)
- `cap/env/robocasa/skills.py` — `make_namespace()` for direct mode
- `cap/prompt/embodiment/robocasa_panda.md` — PandaOmron tool spec for LLM prompts
- `cap/prompt/embodiment/robocasa_gr1.md` — GR1 tool spec for LLM prompts
- `cap/prompt/task/robocasa_pick_place.md` — Pick & place strategy guide (flat inline form)
- `cap/prompt/task/robocasa_pick_place_skills.md` — Pick & place strategy guide (three-layer decomposed form, used with `agent: skill_library`)
- `cap/prompt/task/robocasa_start_microwave.md` — Microwave button-press strategy
