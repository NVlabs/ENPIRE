# Design Doc: Viser 6D Gizmo IK Teleoperation

> **Status**: Implemented.
> **Last updated**: 2026-04-08.
> **Primary files**: `experimental/scripted_policy.py`, `experimental/start_stop_play_policy.py`, `experimental/yam_control_loop.py`.

## Goal

Replace Tonghe's discrete button-nudge scripted policy (PR #17) with **draggable 6D transform gizmos** in the Viser 3D viewer. Instead of clicking "+X / -Z / +Roll" buttons one nudge at a time, the operator drags a gizmo to a target EE pose in 3D space. The system continuously solves IK and streams joint commands through the existing `ScriptedPolicy` -> `get_action()` -> control loop pipeline.

**Scope**: Only `experimental/` files. No changes to `cap/` (cap_server, cap_agent, visualizer, tools).

### What Tonghe Built (PR #17)

- **`experimental/scripted_policy.py`** -- `ScriptedPolicy` class: holds an internal joint buffer. On each `apply_nudge(side, delta_pos, delta_quat)`, it does FK -> apply delta -> IK -> smooth interpolation over 15 steps. Between nudges, `get_action()` returns the same joint command (motors locked).
- **`experimental/start_stop_play_policy.py`** -- Viser UI: per-arm folder with 6 translation buttons (+X/-X/+Y/-Y/+Z/-Z), 6 rotation buttons (+Roll/-Roll/+Pitch/-Pitch/+Yaw/-Yaw), step-size sliders, and gripper controls.
- Commands flow via Portal RPC: ViserUI -> `scripted_nudge` -> `StartStopPlayPolicyWrapper._apply_scripted_command` -> `ScriptedPolicy.apply_nudge()`.

**The limitation**: clicking axis-aligned buttons is clunky -- no diagonal, no freeform 6D dragging, no spatial intuition from the 3D scene.

### Inspiration

NVIDIA GR00T's `viser_policy.py` implements this pattern: two `server.scene.add_transform_controls()` gizmos with IK solving. We adapt the same idea for lecar-tbd's experimental control loop. See also `experimental/viser_policy.py` for the local ViserPolicy implementation.

---

## Architecture

```
Browser (Viser, port 8080)
  |
  |  User drags 6D gizmo in 3D scene
  |  gizmo.position / gizmo.wxyz update in real-time
  |
  v
+-----------------------------------------------------------------------+
|  ViserUI (start_stop_play_policy.py:1431, port 8080)                  |
|                                                                       |
|  Existing:                                                            |
|    +-- URDF robot model (ViserUrdf)                                   |
|    +-- Camera image panes                                             |
|    +-- Policy control buttons (Start/Pause/Home)                      |
|    +-- [OLD] Scripted policy folder (nudge buttons) -- REPLACED       |
|                                                                       |
|  6D Gizmo Control (line ~1797):                                       |
|    +-- ik_left:  TransformControls ("/ik_target_left")  [line 1781]   |
|    +-- ik_right: TransformControls ("/ik_target_right") [line 1788]   |
|    +-- GUI: Show gizmos toggle, Live teleop toggle,                   |
|            Snap button, gripper sliders, Move-To panel                |
|                                                                       |
|  Two operating modes:                                                 |
|    1. LIVE TELEOP: arm follows gizmo in real-time (~20 Hz)            |
|       _poll_and_send_gizmo_targets() [line 4994]                      |
|       -> Portal RPC "scripted_set_target"                             |
|    2. PASSIVE WAYPOINT: gizmos mark targets for Move-To buttons       |
|       _read_gizmo_pose() [line 1911]                                  |
|       -> Portal RPC "scripted_move_to"                                |
+-----------------------------------------------------------------------+
            |
            |  Portal RPC (configurable, default 8009/8010)
            v
+-----------------------------------------------------------------------+
|  StartStopPlayPolicyWrapper (start_stop_play_policy.py:305)           |
|    Portal RPC handlers [lines 366-381]:                               |
|      scripted_set_target  -> _portal_scripted_set_target  [line 533]  |
|      scripted_get_ee_poses -> _portal_scripted_get_ee_poses [line 538]|
|      scripted_move_to     -> _portal_scripted_move_to     [line 546]  |
|      scripted_execute_trajectory -> [line 551]                        |
|      scripted_nudge       -> _portal_scripted_nudge       [line 523]  |
|      scripted_set_gripper -> _portal_scripted_set_gripper [line 528]  |
|      sync_to_init         -> _portal_sync_to_init         [line 556]  |
|                                                                       |
|    _apply_scripted_command() [line 959]:                              |
|      "set_target"          -> ScriptedPolicy.set_target_pose_bimanual |
|      "move_to"             -> ScriptedPolicy.move_to                  |
|      "execute_trajectory"  -> ScriptedPolicy.execute_trajectory       |
|      "nudge"               -> ScriptedPolicy.apply_nudge (legacy)     |
|      "gripper"             -> ScriptedPolicy.set_gripper              |
|    _apply_planned_move_to() [line 1120]:                             |
|      -> YamMotionPlanner / PortalMotionPlanner -> execute_trajectory  |
|                                                                       |
|  ScriptedPolicy (scripted_policy.py:68)                               |
|    Three control modes:                                               |
|    1. Gizmo: set_target_pose / set_target_pose_bimanual               |
|    2. Move-To: move_to / execute_trajectory                           |
|    3. Nudge (legacy): apply_nudge                                     |
|                                                                       |
|    get_action() [line 669]:                                           |
|      - Consumes trajectory queue (if any)                             |
|      - OR interpolates _command_joints toward _target_joints          |
|      - Safety clamp: per-step delta against observation               |
|      - Returns absolute joint_position action                         |
+-----------------------------------------------------------------------+
            |
            |  Action dict returned to control loop
            v
     yam_control_loop.py [line ~1512] --> env.step(action) --> hardware
```

### Why This Approach (Not a Separate IK Thread)

The control loop already calls `ScriptedPolicy.get_action()` at `policy_control_freq` Hz (default 30, `yam_control_loop.py:166`). Instead of creating a parallel IK thread that races with the control loop, we:
1. **ViserUI polls gizmo poses** and sends them via Portal RPC (same path as existing nudge commands).
2. **ScriptedPolicy receives absolute EE targets** and solves IK internally (it already has `YamKinematics`, `scripted_policy.py:111`).
3. **`get_action()` returns smoothly interpolated joint commands** (existing mechanism).

This is simpler than a standalone IK loop and avoids any concurrency issues -- everything flows through the existing `get_action()` call in the control loop.

---

## ScriptedPolicy (scripted_policy.py)

The policy module at `experimental/scripted_policy.py` operates in `joint_position` mode. It keeps an internal absolute joint-position buffer updated when new targets arrive (IK -> new joint target). Between updates, `get_action()` returns the exact same joint command, so motor servos lock the robot firmly in place.

### Class Structure

```
ScriptedPolicy (line 68)
  __init__(action_space, control_hz, safety)  [line 100]
    _kin: YamKinematics                        [line 111]
    _lock: threading.Lock                      [line 123]
    _target_joints: dict | None                [line 126]
    _command_joints: dict | None               [line 129]
    _steps_left_left: int                      [line 132]
    _steps_left_right: int                     [line 133]
    _trajectory_queue: list | None             [line 140]
    _joint_record: list                        [line 136]

  # 6D Gizmo teleop
  set_target_pose(side, pos, quat_xyzw)                     [line 239]
  set_target_pose_bimanual(l_pos, l_q, r_pos, r_q)          [line 290]
  get_current_ee_poses() -> dict | None                      [line 320]
  get_current_state() -> dict | None                         [line 336]

  # Move-To (absolute target with per-arm speed)
  move_to(left_target_pos, ..., ik_error_threshold)          [line 404]
  execute_trajectory(left_positions, right_positions, ...)    [line 551]
  _solve_ik_multi_seed(...)                                  [line 356]
  _retime_joint_trajectory(...)                              [line 515]

  # Legacy nudge
  apply_nudge(side, delta_pos, delta_quat_xyzw)              [line 147]
  set_gripper(side, value)                                    [line 228]

  # Policy interface
  ensure_initialized(observation)                             [line 647]
  get_action(observation) -> (action, info)                   [line 669]
  reset()                                                     [line 840]
  is_moving: bool (property)                                  [line 507]

  # Recording
  _record_joint_positions(action)                             [line 775]
  record_scripted_trajectory_to_parquet(output_path, ...)     [line 791]
```

### Constants (scripted_policy.py:85-98)

| Constant | Value | Purpose |
|----------|-------|---------|
| `HOME_GRIPPER` | `1.0` | Open gripper on reset |
| `SMOOTHING_STEPS` | `15` | Interpolation steps for nudge (~0.5 s @ 30 Hz) |
| `GIZMO_SMOOTHING_STEPS` | `3` | Fewer steps for continuous gizmo tracking |
| `DEFAULT_DISTANCE_PER_STEP` | `0.01` | 1 cm/step for move_to auto-scaling (~0.3 m/s @ 30 Hz) |
| `MIN_SMOOTHING_STEPS` | `5` | Floor for move_to interpolation |
| `MAX_SMOOTHING_STEPS` | `150` | Ceiling for move_to interpolation |

### Safety (scripted_policy.py:48-61, robot/constants.py:71)

`SafetyLimits` dataclass wraps `MAX_JOINT_VELOCITY_RAD_S = 6 rad/s` from `robot/constants.py:71`.
Per-step delta is computed in `get_action()` (line 741):
```
max_step = max_joint_velocity_rad_s / control_hz
```
Example at 30 Hz: `6 / 30 = 0.2 rad/step`. If any joint delta exceeds `max_step`, the command is clamped (line 762) and a warning is logged.

### IK Configuration (scripted_policy.py:40-41)

```python
SCRIPTED_IK_POSITION_COST = 1.0
SCRIPTED_IK_ORIENTATION_COST = 0.05
```

These weights are passed to `YamKinematics` (at `robot/yam/kinematics.py:8`) at construction time.

### Method: set_target_pose (line 239)

Sets an absolute EE target for one arm. IK is solved immediately using `YamKinematics.inverse_kinematics(seeded=True)`. The other arm is held in place via FK from `_target_joints`.

Per-arm smoothing steps are set independently:
```python
self._steps_left_left = max(self._steps_left_left, self.GIZMO_SMOOTHING_STEPS)
self._steps_left_right = max(self._steps_left_right, self.GIZMO_SMOOTHING_STEPS)
```

### Method: set_target_pose_bimanual (line 290)

For dragging both gizmos simultaneously -- solves bimanual IK in one call. Same per-arm smoothing as `set_target_pose`.

### Method: get_current_ee_poses (line 320)

FK on current `_target_joints` to return EE poses. Returns dict with `.tolist()` values for Portal RPC serialization. Used by ViserUI for gizmo snapping (note: ViserUI now prefers local FK from cached joint positions instead; see `_snap_gizmos_to_current`).

### Method: get_current_state (line 336)

Returns the best available commanded state (preferring `_command_joints` over `_target_joints`). Used by the motion planner to get the starting configuration.

### Method: move_to (line 404)

Absolute EE target with per-arm speed control. Uses `_solve_ik_multi_seed()` (line 356) which tries seeded IK first, then up to 8 random seeds for better convergence. Computes per-arm smoothing steps from Cartesian distance: `steps = distance / distance_per_step`, clamped to `[MIN_SMOOTHING_STEPS, MAX_SMOOTHING_STEPS]`.

### Method: execute_trajectory (line 551)

Queues a pre-planned joint trajectory for execution. Trajectory is retimed by `_retime_joint_trajectory()` (line 515) to satisfy `MAX_JOINT_VELOCITY_RAD_S`. If the first waypoint differs from current position, a bridging segment is prepended. Trajectory waypoints are consumed one-per-step in `get_action()`.

### Method: get_action (line 669)

The core policy interface. Within the lock:
1. Auto-initializes from observation if needed.
2. Consumes trajectory queue waypoints (one per step) if a trajectory is active.
3. Otherwise interpolates `_command_joints` toward `_target_joints` per-arm: `frac = 1.0 / steps_left`.
4. After interpolation, applies safety clamp against observed joint positions.
5. Records the action to `_joint_record`.

### Per-Arm Smoothing

The original design doc showed a single `_steps_left` field. The implementation uses separate counters:
- `_steps_left_left: int` (line 132)
- `_steps_left_right: int` (line 133)

This allows independent interpolation speeds per arm (e.g., during `move_to` where arms may travel different distances).

### Joint Recording

`_record_joint_positions()` (line 775) appends each action as a 14D float32 array to `_joint_record`. `record_scripted_trajectory_to_parquet()` (line 791) writes the buffer to a parquet file, with optional append-to-existing support.

---

## ViserUI (start_stop_play_policy.py)

The `ViserUI` class starts at `start_stop_play_policy.py:1431`. It runs in a subprocess launched by `run_viser_subprocess()` (line 5149) via `portal.Process`.

### Gizmo Setup (lines 1781-1793)

Two `add_transform_controls` gizmos are added to the Viser server scene alongside the existing URDF visualization:

```python
# start_stop_play_policy.py:1781-1793
self.ik_left = self.server.scene.add_transform_controls(
    "/ik_target_left",
    scale=0.12,
    position=(0.0, 0.0, 0.0),
    wxyz=(1.0, 0.0, 0.0, 0.0),
    visible=False,
)
self.ik_right = self.server.scene.add_transform_controls(
    "/ik_target_right",
    scale=0.12,
    position=(0.0, 0.0, 0.0),
    wxyz=(1.0, 0.0, 0.0, 0.0),
    visible=False,
)
```

**Note**: Initial positions are `(0, 0, 0)` (not a workspace position). Gizmos are always snapped to FK-computed EE poses before first use.

### Gizmo State Flags (lines 1772-1780)

Three boolean flags control gizmo behavior:

| Flag | Purpose |
|------|---------|
| `_gizmo_visible` | Gizmos shown in 3D scene (for both live teleop and Move-To waypoints) |
| `_gizmo_live_teleop` | When True, arms follow gizmos in real-time (~30 Hz). When False, gizmos are passive waypoint markers. |
| `_gizmos_snapped` | True only after gizmo positions have been set from FK. **Safety gate** that prevents sending stale/default positions to the robot. |

### GUI Controls: "6D Gizmo Control" Folder (line 1797)

Only shown when `show_scripted_controls=True` (which maps to `--use-scripted-policy` CLI flag).

| Control | Behavior |
|---------|----------|
| **Show gizmos** checkbox (line 1798) | Show/hide 6D gizmos. Auto-snaps on show. Hiding also disables live teleop. |
| **Live teleop** checkbox (line 1799) | Arms follow gizmos in real-time. Auto-shows and auto-snaps gizmos. Refuses to enable if snap fails (no joint data yet). |
| **Snap gizmos to current EE** button (line 1802) | Manually snap gizmo poses to FK-computed EE positions. |
| **Left/Right arm** folders (line 1878) | Per-arm gripper slider (0=close, 1=open), Open/Close buttons. |
| **Pose readouts** (line 1848) | Per-arm XYZ position, RPY (global), RPY (gripper-frame) -- updated in `_update_gizmo_pose_display()`. |
| **Move-To panel** (line 1909+) | Reads gizmo positions as waypoints. "Move LEFT/RIGHT/BOTH to gizmo" buttons with per-arm speed sliders, optional motion planner (RRTConnect or cuRobo). |

### _snap_gizmos_to_current (line 4925)

Computes FK from cached joint positions (updated every observation frame in `update_ui()`) rather than querying ScriptedPolicy via RPC. This avoids `None` returns after homing.

```python
# start_stop_play_policy.py:4944-4948
if not hasattr(self, "_kinematics"):
    self._kinematics = YamKinematics()
left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw = (
    self._kinematics.forward_kinematics(left_jp[:6], right_jp[:6])
)
```

Sets `_gizmos_snapped = True` on success (line 4968).

### _poll_and_send_gizmo_targets (line 4994)

Called every UI update tick from `update_ui()` (line 4332). Only sends targets when both `_gizmo_live_teleop` and `_gizmos_snapped` are True. Converts `wxyz` -> `xyzw` and sends via `scripted_set_target` Portal RPC.

```python
# start_stop_play_policy.py:5012-5029
left_pos = np.array(self.ik_left.position)
left_wxyz = np.array(self.ik_left.wxyz)
left_xyzw = np.array([left_wxyz[1], left_wxyz[2], left_wxyz[3], left_wxyz[0]])
# ... same for right ...
payload = {
    "left_pos": left_pos.tolist(),
    "left_quat_xyzw": left_xyzw.tolist(),
    "right_pos": right_pos.tolist(),
    "right_quat_xyzw": right_xyzw.tolist(),
}
self._client.scripted_set_target(payload).result()
```

### _read_gizmo_pose (line 1911)

Helper for Move-To panel -- reads gizmo position as a waypoint target (not live teleop). Returns `(pos, quat_xyzw)` from the gizmo handle.

### _update_gizmo_pose_display (line 4971)

Updates the GUI text readouts (XYZ position, RPY in global and gripper frames) from current gizmo positions. Only runs when `_gizmo_visible` is True.

### Move-To Panel (lines 1909-2840+)

The Move-To panel uses gizmo positions as waypoint targets. Three button groups:
- **Move LEFT to gizmo** (line 2763): Reads left gizmo, sends `scripted_move_to` via Portal RPC.
- **Move RIGHT to gizmo** (line 2765): Same for right.
- **Move BOTH to gizmos** (line 2768): Reads both gizmos, sends bimanual move.

Each supports optional motion planning (RRTConnect or cuRobo), configurable via UI toggles. When planner is enabled, trajectory is visualized in 3D before execution, and sent via `scripted_execute_trajectory`.

### Exact Text Input (Move-To, line 2311+)

In addition to gizmo-based targets, the Move-To panel also supports exact numeric input via text fields for precise Cartesian positioning.

---

## StartStopPlayPolicyWrapper (start_stop_play_policy.py:305)

### Portal RPC Bindings (lines 366-381)

```python
self._server.bind("scripted_nudge",              self._portal_scripted_nudge)
self._server.bind("scripted_set_gripper",         self._portal_scripted_set_gripper)
self._server.bind("scripted_set_target",          self._portal_scripted_set_target)
self._server.bind("scripted_get_ee_poses",        self._portal_scripted_get_ee_poses)
self._server.bind("scripted_move_to",             self._portal_scripted_move_to)
self._server.bind("scripted_execute_trajectory",  self._portal_scripted_execute_trajectory)
self._server.bind("sync_to_init",                 self._portal_sync_to_init)
```

### RPC Handler: scripted_set_target (line 533)

Enqueues `{"type": "set_target", ...}` payload. Processed in `_apply_scripted_command()` (line 1104):
```python
self.policy.set_target_pose_bimanual(
    left_pos=np.asarray(payload["left_pos"]),
    left_quat_xyzw=np.asarray(payload["left_quat_xyzw"]),
    right_pos=np.asarray(payload["right_pos"]),
    right_quat_xyzw=np.asarray(payload["right_quat_xyzw"]),
)
```

### RPC Handler: scripted_get_ee_poses (line 538)

Directly calls `self.policy.get_current_ee_poses()` and returns the result. Does NOT go through the command queue (synchronous read). Returns `{}` if policy does not support it.

### RPC Handler: scripted_move_to (line 546)

Enqueues `{"type": "move_to", ...}`. Supports optional motion planner when `use_planner=True`:
- `_apply_planned_move_to()` (line 1120) uses `PortalMotionPlanner` or `YamMotionPlanner` based on `motion_planner_backend` config.
- Planner generates a collision-free joint trajectory, which is executed via `execute_trajectory()`.

### Command Processing (lines 729-738)

All enqueued scripted commands are drained at the top of each `get_action()` call under the wrapper lock:
```python
if self._pending_scripted_commands:
    pending = self._pending_scripted_commands
    self._pending_scripted_commands = []
    if hasattr(self.policy, "ensure_initialized"):
        self.policy.ensure_initialized(observation)
    for cmd in pending:
        self._apply_scripted_command(cmd, observation)
```

---

## yam_control_loop.py

### EvalConfig Flags (lines 150-295)

| Flag | Default | Purpose |
|------|---------|---------|
| `use_scripted_policy` | `False` | Enables gizmo teleop / scripted policy (line 238) |
| `policy_control_freq` | `30` | Control loop Hz (line 166) |
| `scripted_use_rrt` | `True` | Default Move-To uses motion planner (line 240) |
| `scripted_planner_max_joint_vel` | `2.0` | RRT execution speed (line 242) |
| `scripted_planner_solver_speed` | `"fast"` | cuRobo solver preset (line 244) |
| `motion_planner_backend` | `"curobo"` | Backend: `"rrtconnect"` or `"curobo"` (line 252) |
| `show_mp_feasible_region` | `False` | Show planner reachability point cloud (line 246) |
| `policy_port` / `viser_port` / `viser_web_port` | `8009` / `8010` / `8080` | IPC and web ports (lines 226-230) |
| `reclaim_control_ports` | `True` | Auto-reclaim busy ports from older processes (line 232) |

### Policy Instantiation (lines 758-764)

When `use_scripted_policy=True`:
```python
safety = SafetyLimits()
policy = ScriptedPolicy(
    action_space=env.action_space,
    control_hz=cfg.policy_control_freq,
    safety=safety,
)
```

Control mode is forced to `"joint_position"` (line 683). Cameras are disabled (`enable_cameras=False`, lines 702, 710) since the scripted policy does not need vision.

### ViserUI Subprocess Launch (lines 1065-1095)

```python
# line 1082
show_scripted_controls=cfg.use_scripted_policy,
```

`portal.Process(start_viser, start=True)` (line 1095) launches `run_viser_subprocess()` in a separate process. The subprocess runs `ViserUI.update_ui()` in a loop.

---

## Data Flow (Single Tick -- Live Gizmo Teleop)

```
1. User drags gizmo in browser
   ik_left.position  -> (x, y, z)     [viser auto-updates]
   ik_left.wxyz      -> (w, x, y, z)  [viser auto-updates]

2. ViserUI.update_ui() calls _poll_and_send_gizmo_targets() [line 4332]
   SAFETY CHECK: _gizmo_live_teleop AND _gizmos_snapped must both be True
   Convert wxyz -> xyzw
   Portal RPC "scripted_set_target" -> StartStopPlayPolicyWrapper [line 5029]

3. Wrapper enqueues {"type": "set_target", "left_pos": ..., ...} [line 535]

4. On next get_action() call (policy_control_freq Hz, default 30):
   a. Wrapper drains pending commands [line 729-738]
   b. Calls ScriptedPolicy.set_target_pose_bimanual(l_pos, l_q, r_pos, r_q)
   c. ScriptedPolicy solves bimanual IK via YamKinematics (mink-based, ~0.37ms seeded)
   d. Sets new _target_joints; per-arm _steps_left = max(current, GIZMO_SMOOTHING_STEPS=3)

5. get_action() returns interpolated joint command [line 669]
   Safety clamp vs. observed joints (MAX_JOINT_VELOCITY_RAD_S / control_hz)

6. yam_control_loop: env.step(action) -> hardware [line ~1512]
```

Total latency: ~1ms Portal RPC + ~0.4ms IK = ~1.4ms. Well within 33ms budget.

## Data Flow (Single Tick -- Move-To via Gizmo Waypoint)

```
1. User positions gizmo in browser (passive waypoint, live teleop OFF)

2. User clicks "Move LEFT to gizmo" button [line 2763]
   _read_gizmo_pose("left") -> (pos, quat_xyzw) [line 1911]

3. If planner enabled:
   a. ViserUI calls _plan_and_visualize() -> trajectory shown in 3D
   b. Portal RPC "scripted_move_to" with use_planner=True
   c. Wrapper._apply_planned_move_to() [line 1120]
      -> PortalMotionPlanner.plan_to_pose() or YamMotionPlanner.plan_to_pose()
      -> ScriptedPolicy.execute_trajectory()
   d. Trajectory retimed by _retime_joint_trajectory() [line 515]
   e. Waypoints consumed one-per-step in get_action()

4. If planner disabled:
   a. Portal RPC "scripted_move_to"
   b. Wrapper._apply_scripted_command() -> ScriptedPolicy.move_to()
   c. Multi-seed IK solve, per-arm smoothing based on Cartesian distance
```

---

## Quaternion Convention Reference

| System | Convention | Example |
|--------|-----------|---------|
| Viser (gizmos, URDF) | `wxyz` | `handle.wxyz = (1.0, 0.0, 0.0, 0.0)` |
| YamKinematics / scipy | `xyzw` | `R.from_quat([0, 0, 0, 1])` |
| mink `SO3` | `wxyz` | `mink.SO3(wxyz=[1, 0, 0, 0])` |

Conversions happen at two boundaries:
- **ViserUI -> Portal RPC**: `wxyz` -> `xyzw` (before sending to ScriptedPolicy; `start_stop_play_policy.py:5014`)
- **ScriptedPolicy -> ViserUI** (snap): `xyzw` -> `wxyz` (in `_snap_gizmos_to_current`; `start_stop_play_policy.py:4958-4962`)

---

## Files

### Modified

| File | Key Lines | Change |
|------|-----------|--------|
| `experimental/scripted_policy.py` | 68-851 | Full `ScriptedPolicy` class: `set_target_pose()` (239), `set_target_pose_bimanual()` (290), `get_current_ee_poses()` (320), `get_current_state()` (336), `move_to()` (404), `execute_trajectory()` (551), `_solve_ik_multi_seed()` (356), `_retime_joint_trajectory()` (515), safety clamp in `get_action()` (741), joint recording (775+), `SafetyLimits` (49). Keeps `apply_nudge()` (147) for backward compat. |
| `experimental/start_stop_play_policy.py` | 305-5230 | **ViserUI** (1431): Two `add_transform_controls` gizmos (1781, 1788). "6D Gizmo Control" folder (1797) with show/live-teleop/snap controls, gripper sliders, pose readouts, Move-To panel with gizmo-based and exact-input targets, optional motion planner (RRTConnect/cuRobo). `_snap_gizmos_to_current()` (4925) uses local FK. `_poll_and_send_gizmo_targets()` (4994) with `_gizmos_snapped` safety gate. **Wrapper** (305): Portal RPC handlers for `scripted_set_target` (533), `scripted_get_ee_poses` (538), `scripted_move_to` (546), `scripted_execute_trajectory` (551), plus `_apply_planned_move_to()` (1120) for planner-based motion. |
| `experimental/yam_control_loop.py` | 150-1095 | `EvalConfig` with `use_scripted_policy` (238), planner config (240-252), port config (226-234). ScriptedPolicy instantiation (758-764) forces `joint_position` mode, disables cameras. ViserUI subprocess launch with `show_scripted_controls=cfg.use_scripted_policy` (1082). |
| `docs/VISER_IK_TELEOP_DESIGN.md` | -- | This document. |

### Supporting Files

| File | Role |
|------|------|
| `robot/yam/kinematics.py` | `YamKinematics` (line 8): FK/IK using mink. Instantiated by both ScriptedPolicy and ViserUI. |
| `robot/constants.py` | `MAX_JOINT_VELOCITY_RAD_S = 6` (line 71): safety velocity limit. |
| `experimental/motion_planner.py` | `YamMotionPlanner` (line 74): RRTConnect-based collision-free path planner. |
| `experimental/portal_motion_planner.py` | `PortalMotionPlanner` (line 196): Portal-RPC wrapper for cuRobo-based motion planner server. |
| `experimental/viser_policy.py` | `ViserPolicy` (line 94): Older standalone Viser+IK policy (inspiration for this design). Uses pyroki IK. |
| `viser_env_wrapper.py` | `ViserEnvWrapper`: Viser-based environment wrapper for data collection. |
| `record_episode_wrapper.py` | `RecordEpisodeWrapper`: Records episodes to disk (joint commands flow through `env.step()` regardless of policy type). |

### Not Modified

| File | Why |
|------|-----|
| `cap/server/cap_server.py` | No new RPC endpoints -- joint commands flow through existing `env.step()` |
| `cap/agent/cap_agent.py` | Not involved -- gizmos live in experimental ViserUI |
| `cap/agent/visualizer.py` | Separate Viser server for the CAP agent; not connected to scripted policy |

---

## Edge Cases

| Scenario | Handling |
|----------|----------|
| Gizmo dragged outside workspace | IK saturates at joint limits. Arm reaches as far as possible. No crash. `_solve_ik_multi_seed` tries random seeds if seeded IK fails. |
| Gizmo at a singularity | mink's damped IK handles singularities. `seeded=True` provides continuity. |
| Both gizmos dragged simultaneously | `set_target_pose_bimanual()` solves both in one IK call. |
| Gizmo not dragged (idle) | `_poll_and_send_gizmo_targets()` still sends current gizmo pose. IK returns same joints. No unnecessary motion. |
| Live teleop enabled before first observation | SAFETY: `_snap_gizmos_to_current()` requires `_cached_left_jp` / `_cached_right_jp` to be populated. Live teleop checkbox refuses to enable if snap fails (`_gizmos_snapped=False`). |
| Gizmos never snapped | SAFETY: `_poll_and_send_gizmo_targets()` checks `_gizmos_snapped` and blocks target sends with a printed warning. |
| Policy mode switched while gizmos enabled | Gizmos are only active when `use_scripted_policy=True`. The `show_scripted_controls` flag controls UI visibility. |
| Multiple browser clients | Viser handles multi-client. All see gizmos. Last dragger wins. |
| Move-To with planner fails | `_apply_planned_move_to()` catches exceptions and prints errors. Falls back gracefully. |
| move_to IK infeasible | `_solve_ik_multi_seed()` tries 8+ seeds. If all fail, `move_to()` returns `False` and prints a red error message. Target unchanged. |
| Joint velocity exceeded | `get_action()` safety clamp (line 741) clips per-step delta to `max_step = MAX_JOINT_VELOCITY_RAD_S / control_hz`. Logged as a warning. |
| Trajectory execution while gizmo live teleop on | Both can coexist technically: `execute_trajectory` sets `_trajectory_queue` which takes priority over interpolation in `get_action()`. Gizmo targets arriving during trajectory execution update `_target_joints` but the trajectory queue is consumed first. |
| Homing during gizmo teleop | `ScriptedPolicy.reset()` clears `_target_joints`, `_command_joints`, and all step counters. Gizmos remain visible but `get_current_ee_poses()` returns `None` until re-initialized. |

---

## Dependencies

No new dependencies beyond what the project already uses:
- `viser` (>=1.0.6) -- `add_transform_controls` available
- `mink` (0.0.12) -- IK via `YamKinematics`
- `mujoco` -- FK/IK model
- `portal` -- RPC communication
- `scipy` -- quaternion conversions
- `trimesh` -- 3D mesh geometry for Move-To dot visualization

---

## Comparison with Other Approaches

| Aspect | Tonghe's Button Nudge (PR #17) | 6D Gizmo Teleop (this design) | Move-To Panel (this design) | GR00T `viser_policy.py` |
|--------|-------------------------------|-------------------------------|-------------------------------|------------------------|
| Input | Axis-aligned buttons | Freeform 6D drag | Gizmo position as waypoint, or exact text input | Freeform 6D drag |
| Granularity | Discrete steps | Continuous, any direction | Continuous, auto-scaled speed | Continuous |
| Spatial intuition | None -- must mentally map axes | Direct -- see target in 3D | Direct -- see target in 3D | Direct |
| IK library | mink (YamKinematics) | mink (YamKinematics) | mink (multi-seed) + optional motion planner | pyroki |
| Architecture | Portal RPC -> apply_nudge -> FK -> delta -> IK | Portal RPC -> set_target_pose -> IK | Portal RPC -> move_to or execute_trajectory | Direct gizmo read -> IK -> env.step |
| Smoothing | 15 steps per nudge | 3 steps per target (continuous) | Distance-proportional (5-150 steps) | Per-frame IK (no smoothing) |
| Both arms | Independent nudge per arm | Bimanual IK in one call | Bimanual IK or per-arm | Bimanual IK in one call |
| Motion planning | None | None (direct IK) | Optional RRTConnect or cuRobo | None |
| Safety | None | `_gizmos_snapped` gate + velocity clamp | Multi-seed IK + velocity clamp | None |

---

## Cross-References

| Doc | Relevance |
|-----|-----------|
| [`docs/CAP_DESIGN.md`](CAP_DESIGN.md) | Overall CAP architecture. Scripted policy is in `experimental/`, outside the CAP agent pipeline. |
| [`docs/SAFETY_ZONE_DESIGN.md`](SAFETY_ZONE_DESIGN.md) | Task-aware EE safety zones. The `SafetyLimits` in scripted policy enforces velocity limits but does not implement workspace safety zones. |
| [`docs/TABLE_BUSSING_SKILLS.md`](TABLE_BUSSING_SKILLS.md) | Table bussing skill tools (freespace move, nudge, gripper). Move-To panel provides similar functionality via the Viser UI. |
| [`docs/CAP_UI_DESIGN.md`](CAP_UI_DESIGN.md) | CAP web UI (React). The Viser UI for scripted policy is separate (Python/Viser, port 8080). |
| [`docs/CUROBO_UPDATE_SUMMARY.md`](CUROBO_UPDATE_SUMMARY.md) | cuRobo integration. The Move-To panel supports cuRobo as a motion planner backend via `PortalMotionPlanner`. |
| [`docs/SERIAL_FOOTSWITCH.md`](SERIAL_FOOTSWITCH.md) | Footswitch for recording control. Works alongside scripted policy -- footswitch events pass through `StartStopPlayPolicyWrapper`. |
| [`docs/VOICE_INPUT.md`](VOICE_INPUT.md) | Voice input for task commands. Voice prompt runs in parallel with scripted policy; task command changes trigger policy reset. |
| [`docs/GRASPNET_DEBUG_UI.md`](GRASPNET_DEBUG_UI.md) | GraspNet debug UI (port 8120). Separate from the Viser scripted-policy UI (port 8080). |
