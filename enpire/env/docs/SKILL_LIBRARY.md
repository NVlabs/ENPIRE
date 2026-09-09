# Skill library — the tools a CaP script can call

This is the reference for writing a task script. It covers the two tool
surfaces, the world-frame and RPY contract that every tool shares, which tools
work on real hardware, and the camera data contract.

Paths below are relative to `enpire/env/forge/`. Line numbers are deliberately
omitted — they rot faster than the code.

---

## 1. Two surfaces, one set of tools

**Script namespace** — what a CaP task script sees as bare functions. Built by
`cap/env/real_bimanual_yam/skills.py: make_namespace(env)` and injected into the
script's globals by `run_script.py`, which is why saved scripts call
`freespace_move(...)` with no import:

| Group | Callables |
|---|---|
| Motion | `freespace_move`, `nudge`, `nudge_world_pose_preserving`, `go_home`, `go_home_fast` |
| Gripper | `set_gripper`, `set_gripper_fast`, `open_gripper`, `open_gripper_fast`, `close_gripper` |
| Grasping | `sample_grasp_pose_anygrasp`, `sample_grasp_pose_2d`, `sample_grasp_pose_3d_bb`, `select_best_grasp` |
| Perception | `segment_object`, `segment_all_objects`, `detect_object`, `detect_objects_oneshot`, `end_detection`, `list_detections`, `vlm_query` |
| Cameras | `get_camera_image`, `get_camera_intrinsics`, `get_camera_extrinsics`, `render_rgb`, `render_depth` |
| State | `get_robot_state`, `get_task_info`, `display_rpy_to_quat` |

**Registry** — the stable, named surface exposed by `uv run enpire tools list`,
used by the agent and the CLI. It is deliberately much smaller:

```
control.get_state     control.set_gripper     planning.freespace
vision.detect         vision.segment          vlm.query
```

Both dispatch into the same implementations under `cap/agent/tools/`. Tools
branch on `self._env`: in direct mode they call env methods in-process; in the
legacy server mode they go over Portal RPC.

You can also call them from plain Python without a script: build a
`RealYamEnv`, pass it to `make_namespace(env)`, and call the returned
callables directly. That path has no dashboard, so no pause and no manual
gripper release.

---

## 2. Coordinate frame contract

**Every tool input and output is in the robot world frame** (URDF `base_link`):

```
    +X  forward  (toward the work table)
    +Y  left     (toward the left arm)
    +Z  up
    Origin: floor level, centred between the arm bases (left y=+0.31, right y=-0.31)
```

Reference points: floor `z=0`, table surface `z≈0.75`, arm bases at
`(0.2525, ±0.31, 0.75)`.

| Data | Source | Frame |
|---|---|---|
| `get_robot_state()` EE poses | Pinocchio FK | World |
| `freespace_move()` targets | user input | World |
| `nudge()` deltas | user input | World, applied to the FK-sourced current pose |
| `sample_grasp_pose_*()` candidates | remapped to planner convention | World (display RPY) |

### Display RPY — the thing that catches people

The RPY exposed to scripts is **not** conventional XYZ Euler. It is a Viser
display convention:

```
roll = euler_xyz[1]      pitch = -euler_xyz[0]      yaw = -(euler_xyz[2] + 90)
```

`freespace_move`, `nudge`, and `grasp_anygrasp` all use it internally and
consistently. Do not compose rotations by hand — use `display_rpy_to_quat`, or
use `nudge`, which takes a world-frame **delta** and sidesteps the convention
entirely.

Camera→world uses a mount flip `F = diag(-1, -1, 1)` for D405 cameras to convert
OpenCV to Pinocchio convention; ZED 2i uses none. This is controlled by
`robot/models/station/paths.py: needs_optical_flip()` and applied inside the
tool, not trusted from a model server.

---

## 3. Sim-only vs real-transferable

Some perception calls return **simulator ground truth** and cannot run on
hardware. Mixing them into a policy silently prevents sim→real transfer.

**Ground truth — sim only:**

| Call | Why |
|---|---|
| `detect_object(..., backend="oracle")` | reads simulator body poses, then fuzzy-matches the query name |
| `get_task_info()` | task state from the simulator |
| `get_object_positions()` / `get_oracle_targets()` | raw GT scene state |

**Sensor-based — runs unchanged in sim and on real YAM:**
`segment_object` (SAM3), `sample_grasp_pose_anygrasp`, `sample_grasp_pose_2d`,
`detect_objects_oneshot`, `vlm_query`, and the camera calls.

**Rule of thumb:** if a call needs `backend="oracle"` or `get_task_info`, it is
sim-only. Autoresearch evaluation strips oracle tools from generated scripts
(`runtime_role="script"`, `CAP_DISABLE_TASK_INFO_IN_SCRIPT=1`) so vision-only
policies cannot cheat.

---

## 4. Camera data contract

`render_rgb` / `render_depth` / `get_camera_intrinsics` / `get_camera_extrinsics`
return:

| Field | Type | Notes |
|---|---|---|
| `rgb` | `np.ndarray (H, W, 3) uint8` | cameras named `top`, `left`, `right` |
| `depth` | `np.ndarray (H, W) float32` | metres |
| `intrinsics` | `[fx, fy, cx, cy]` | pinhole |
| `extrinsics` | `{position[3], rotation[9 row-major], needs_optical_flip}` | cam→world |

For HTTP model servers, `rgb` is PNG→base64 (`image_base64`) and `depth` is
`np.save` float32 bytes→base64 (`depth_base64`), with the optical flip
**pre-applied** so it arrives in OpenCV convention.

Cameras are addressed by **role**, not device path. See the "Binding cameras to
roles" section of the top-level `README.md`.

---

## 5. Grasping: pick the right backend

| Backend | Needs | Best for |
|---|---|---|
| `sample_grasp_pose_anygrasp` | machine-locked licence ([`ANYGRASP_SETUP.md`](ANYGRASP_SETUP.md)) | 6-DoF grasps on arbitrary geometry |
| `sample_grasp_pose_2d` | segmentation + a known table plane | **flat or short objects on a table** — often better than AnyGrasp, and needs no licence |
| `sample_grasp_pose_3d_bb` | point cloud | coarse bounding-box grasps |

`sample_grasp_pose_2d` never uses depth, which matters on a short-baseline
camera such as the D405. Its height is `TABLE_SURFACE_Z_M +
ENPIRE_2D_GRASP_Z_OFFSET_M`, written verbatim into every candidate with **no
clearance floor** — unlike the AnyGrasp path, which clamps to
`ANYGRASP_MIN_PLANNER_Z_M` (default `0.80`). Measure your table plane. See the
"2D top-down grasps" section of `README.md`.

`select_best_grasp` ranks candidates; `freespace_move(grasp_candidates=...,
batch_side=..., preview_only=True)` ranks them by plannability without moving.

---

## 6. Compliant gripper close

`set_gripper(side, pos, vel_limit=None, torque_limit=None)` — `pos` is `1.0`
open, `0.0` closed.

When `torque_limit` is set, the settle loop performs **stall detection**: if the
gripper stops moving for `GRIPPER_TORQUE_LIMIT_HOLD_S`, it concludes it has
clamped an object and returns early rather than forcing the position target.
That is the compliant close — grip firmly without crushing. `vel_limit` slows
the close for gentleness.

`open_gripper` / `close_gripper` are thin wrappers in `cap/agent/tools/native.py`.

> The operator TUI also binds **`O`** to open both grippers mid-script; it
> pauses first so the script thread cannot overwrite the open command.

---

## 7. Bimanual notes

Arms are `"left"` and `"right"`. `freespace_move` takes targets for one arm or
both; **omit the inactive arm entirely** rather than passing its current pose —
targets not supplied hold position. Supplying both plans a synchronized
bimanual motion.

Gripper motor direction is per-arm (`yam_gripper_sign`), overridable with
`ENPIRE_YAM_GRIPPER_SIGN_LEFT` / `_RIGHT` for stations whose motors are mounted
mirrored.

---

## 8. A real-transferable example

```python
# 1. PERCEIVE — sensor-based only, so this runs in sim and on real YAM
mask  = segment_object("red block", camera="top")
cands = sample_grasp_pose_2d("red block", camera="top")

# 2. PLAN — rank candidates by plannability without moving
ranked = freespace_move(grasp_candidates=cands, batch_side="right", preview_only=True)
best   = ranked.best_candidate

# 3. APPROACH — collision-free move (RPY in degrees, world frame, display convention)
freespace_move(right_target_pos=best.position, right_target_rpy=best.rpy)

# 4. GRASP — compliant, force-limited close
close_gripper("right", vel_limit=2.0, torque_limit=0.4)

# 5. LIFT, then descend with small world-frame deltas
freespace_move(right_target_pos=hover_pos, right_target_rpy=best.rpy)
nudge("right", delta_pos=[0, 0, -0.01])

# 6. VERIFY
state = get_robot_state()
```

`freespace_move` requires the cuRobo service on port 8611. Until it is
listening the call blocks silently and the arm never moves — see
[`CUROBO_SETUP.md`](CUROBO_SETUP.md).

---

## Related

- [`NEW_TASK.md`](NEW_TASK.md) — authoring and running a task
- [`grasp_orientation.md`](grasp_orientation.md) — grasp frame conventions in depth
- [`CAP_DESIGN.md`](CAP_DESIGN.md) — how the layers fit together
- `README.md` — camera role binding, services, running a pick
