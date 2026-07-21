# Tools → Skills Migration Plan

## Runtime compatibility guarantee

After this refactor, every tool must work correctly in **both** modes with identical behavior:

### Mode A — cap_server + cap_agent + cap_ui (Portal RPC)

```
cap_ui  →  cap_agent  →  Executor  →  tool(env=None)  →  Portal RPC  →  cap_server  →  robot
```

- Tools instantiated with `env=None` (default) — existing Portal RPC path, unchanged.
- `cap_agent.py` builds namespace exactly as today: `FreespaceMoveTool()`, `SegmentObjectTool()`, etc.
- cap_server owns the hardware control loop, camera streams, gripper drivers.
- No changes to cap_agent, cap_server, or cap_ui required.

### Mode B — run_script (direct, no cap_server)

```
run_script.py  →  skills.make_namespace(env)  →  tool(env=env)  →  env  →  robot
```

- Tools instantiated with `env=<RealYamEnv>` — all Portal RPC calls bypassed.
- `skills.py` is the thin shim that instantiates tools with `env=env`.
- No cap_server process required. No Portal RPC sockets opened.
- cuRobo planning (`PortalMotionPlanner`) still connects to its own separate cuRobo server — this is NOT cap_server and is kept in both modes.

### What changes per mode

| Component | Mode A (Portal RPC) | Mode B (Direct) |
|---|---|---|
| Camera images | `client.get_camera_image(cam)` | `env.render_rgb(cam)` |
| Camera depth | `client.get_camera_depth(cam)` | `env.render_depth(cam)` |
| Camera intrinsics/extrinsics | `client.get_camera_intrinsics/extrinsics()` | `env.get_camera_intrinsics/extrinsics()` |
| Robot state | `client.get_robot_state()` | `env.get_observations("left/right")` |
| Trajectory execution | `client.move_bimanual_joint_keypoints()` | `env._move_bimanual_joint_keypoints()` |
| Gripper | `client.set_gripper()` | `env.set_gripper()` |
| cuRobo planning | `PortalMotionPlanner` (separate server) | same — unchanged |
| SAM3 / AnyGrasp / BundleSDF | HTTP to external servers | same — unchanged |
| VLM calls | HTTP to NVIDIA/Gemini/Claude | same — unchanged |

### Debug output routing

Tools must never directly reference `ws_manager`, WebSocket, or cap_ui internals.
Debug output is routed differently per mode, but tools are unaware of the difference:

**Image saving — already dual-mode via `_artifact_log.py`:**

```
cap_agent:   set_artifact_dir(session.run_dir)  →  tools call log_detection/log_mask/log_grasp/log_vlm_query
run_script:  set_artifact_dir(log_dir)          →  same tool calls, same files on disk
```

`_artifact_log.py` is a global singleton. Both `cap_agent.py` and `run_script.py` call
`set_artifact_dir(path)` before execution. Tools just call `log_*()` — they never know which
mode they're in. No changes needed here.

**Event streaming — dual-mode via hook injection:**

```
Mode A (cap_agent):  profiler wraps tool calls → WebSocket broadcast → cap_ui live display
Mode B (run_script): set_tool_event_hooks(on_start=debug_writer.tool_start,
                                          on_end=debug_writer.tool_end)
                     → DebugEventWriter → debug_events.jsonl → run_script debug_ui polls file → browser
```

`run_script.py` already wires this up. Tools are completely unaware — they just execute.
The event hooks fire around tool calls at the profiler/executor layer, not inside tools.

**Invariant enforced by this migration:**

> Tools in `cap/agent/tools/` must never import `ws_manager`, `asyncio`, WebSocket, or any
> cap_ui reference. All debug output goes through `_artifact_log.py` (images) or standard
> `logging` (text). Routing to cap_ui vs log file is the caller's concern, not the tool's.

### Testing checklist (per tool after migration)

- [ ] Tool instantiated with `env=None` passes existing cap_agent integration tests
- [ ] Tool instantiated with `env=<mock or real env>` produces same result as Portal RPC path
- [ ] `run_script.sh` executes `nclass_sorting_nvidiagemini.py` end-to-end without cap_server
- [ ] cap_ui agent mode runs the same script with identical tool behavior

---

## Design

### Role of `skills.py` (after migration)

`skills.py` is **only** a lightweight customization shim. Its entire job:

1. Import tool classes from `cap/agent/tools/`
2. Instantiate each with `env=env` — this single parameter switches off all Portal RPC
3. Build and return the `make_namespace()` dict

It contains **no tool logic**. No Portal RPC. No cap_server dependency whatsoever.
If a line in `skills.py` imports `portal` or references `CAP_SERVER_PORT`, that is a bug.

```python
# cap/env/real_bimanual_yam/skills.py — the entire file after migration

def make_namespace(env, vlm_backend="gemini", cfg=None):
    from cap.agent.tools.freespace_move import FreespaceMoveTool
    from cap.agent.tools.nudge import NudgeTool
    from cap.agent.tools.segmentation import SegmentObjectTool, SegmentAllObjectsTool
    from cap.agent.tools.detection import DetectObjectsOneshotTool
    from cap.agent.tools.grasp_anygrasp import SampleGraspPoseAnyGraspTool
    from cap.agent.tools.grasp_2d import SampleGraspPose2DTool
    from cap.agent.tools.vlm_query import VlmQueryTool
    from cap.agent.tools.camera import (
        GetCameraIntrinsicsTool, GetCameraExtrinsicsTool, RenderRgbTool, RenderDepthTool
    )
    from cap.agent.tools.native import (
        GoHomeTool, SetGripperTool, OpenGripperTool, CloseGripperTool,
        GetRobotStateTool, GetCameraImageTool
    )

    # Every tool gets env=env — no Portal RPC, no cap_server
    _fm   = FreespaceMoveTool(env=env)
    _seg  = SegmentObjectTool(env=env)
    _segA = SegmentAllObjectsTool(env=env)
    _det  = DetectObjectsOneshotTool(env=env)
    _ag   = SampleGraspPoseAnyGraspTool(env=env)
    _2d   = SampleGraspPose2DTool(env=env)
    _vlm  = VlmQueryTool(env=env, default_backend=vlm_backend)
    _cam  = GetCameraImageTool(env=env)
    _intr = GetCameraIntrinsicsTool(env=env)
    _extr = GetCameraExtrinsicsTool(env=env)
    _rgb  = RenderRgbTool(env=env)
    _dep  = RenderDepthTool(env=env)
    _home = GoHomeTool(env=env)
    _grip = SetGripperTool(env=env)
    _open = OpenGripperTool(env=env)
    _clos = CloseGripperTool(env=env)
    _state = GetRobotStateTool(env=env)

    # Thin wrappers + the handful of truly unique direct-env functions
    def nudge_world_pose_preserving(...): ...   # no tool equivalent, keep here
    def display_rpy_to_quat(...): ...           # pure math, keep here
    def get_task_info(): ...                    # direct env call, keep here

    return {
        "freespace_move":             lambda **kw: _fm.execute(**kw).data,
        "select_best_grasp":          lambda **kw: _fm.execute(**kw).data,
        "nudge":                      lambda **kw: NudgeTool(env=env).execute(**kw).data,
        "segment_object":             lambda **kw: _seg.execute(**kw).data,
        "segment_all_objects":        lambda **kw: _segA.execute(**kw).data,
        "detect_objects_oneshot":     lambda **kw: _det.execute(**kw).data,
        "sample_grasp_pose_anygrasp": lambda **kw: _ag.execute(**kw).data,
        "sample_grasp_pose_2d":       lambda **kw: _2d.execute(**kw).data,
        "vlm_query":                  lambda **kw: _vlm.execute(**kw).data,
        "get_camera_image":           lambda **kw: _cam.execute(**kw).data,
        "get_camera_intrinsics":      lambda **kw: _intr.execute(**kw).data,
        "get_camera_extrinsics":      lambda **kw: _extr.execute(**kw).data,
        "render_rgb":                 lambda **kw: _rgb.execute(**kw).data,
        "render_depth":               lambda **kw: _dep.execute(**kw).data,
        "go_home":                    lambda **kw: _home.execute(**kw).data,
        "set_gripper":                lambda **kw: _grip.execute(**kw).data,
        "open_gripper":               lambda **kw: _open.execute(**kw).data,
        "close_gripper":              lambda **kw: _clos.execute(**kw).data,
        "get_robot_state":            lambda **kw: _state.execute(**kw).data,
        "nudge_world_pose_preserving": nudge_world_pose_preserving,
        "display_rpy_to_quat":        display_rpy_to_quat,
        "get_task_info":              get_task_info,
    }
```

### Role of `cap/agent/tools/` (after migration)

Each tool class contains all logic and supports two transport modes via `env=`:

- `env=None` (default) → Portal RPC → cap_server (Mode A, cap_agent unchanged)
- `env=<RealYamEnv>` → direct env calls, zero Portal RPC (Mode B, run_script)

All tool logic (cuRobo planning, SAM3, AnyGrasp, BundleSDF, VLM, debug image saving) lives
here and is shared by both modes.

---

## Portal RPC → Direct Env substitution table

| Portal RPC call in tool | Direct env replacement |
|---|---|
| `client.get_robot_state()` | `env.get_observations("left/right")` |
| `client.get_camera_image(cam)` | `env.render_rgb(cam)` |
| `client.get_camera_depth(cam)` | `env.render_depth(cam)` |
| `client.get_camera_intrinsics(cam)` | `env.get_camera_intrinsics(cam)` → build K (3×3) |
| `client.get_camera_extrinsics(cam)` | `env.get_camera_extrinsics(cam)` → build T_cam_world (4×4) |
| `client.move_joint_keypoints(side, ts, pos, grip)` | `env.move_joint_keypoints(ts, lp, rp, lg, rg)` |
| `client.move_bimanual_joint_keypoints(ts, lp, rp, lg, rg)` | same |
| `client.set_gripper(side, width)` | `env.set_gripper(side, width)` |

---

## Tools to migrate (conflicts — add `env=` mode)

### 1. `FreespaceMoveTool` (freespace_move.py) — HIGH PRIORITY

**Why:** skills.py freespace_move is Cartesian mink-IK only (single arm, no cuRobo, no bimanual,
no trajectory caching). The tool version has cuRobo planning, bimanual simultaneous execution,
batch grasp ranking, trajectory caching, `preview_only`, full `FreespaceResult` diagnostics.

**Portal RPC sites to gate on `env is None`:**
- `_get_client().get_robot_state()` — 3 call sites
- `_execute_trajectory(client, ...)` — 1 call site (the only place motion is sent)

**cuRobo planning (`PortalMotionPlanner`)** connects to a separate cuRobo server — NOT cap_server.
Keep exactly as-is in both modes.

**`_execute_trajectory` direct-env implementation:**
```python
def _execute_trajectory_direct(self, side, timestamps, left_positions, right_positions,
                                left_gripper_positions, right_gripper_positions):
    # env exposes move_bimanual_joint_keypoints (already in skills.py today)
    return self._env.move_bimanual_joint_keypoints(
        np.asarray(timestamps),
        left_positions, right_positions,
        left_gripper_positions, right_gripper_positions,
    )
```

Helper methods `_densify_joint_waypoints`, `_timestamps_from_waypoints`,
`_store_cached_trajectory`, `_compute_plan_diagnostics` are pure logic — unchanged in both modes.

---

### 2. `select_best_grasp` (batch path of FreespaceMoveTool) — HIGH PRIORITY

The `grasp_candidates is not None` branch already lives inside `FreespaceMoveTool.execute()`.
When the tool gains `env=` support, `select_best_grasp` in skills.py becomes:
```python
namespace["select_best_grasp"] = lambda **kw: _fm.execute(grasp_candidates=..., **kw).data
```
No separate migration needed — falls out of `freespace_move` migration.

---

### 3. `SegmentObjectTool` / `SegmentAllObjectsTool` (segmentation.py) — MEDIUM PRIORITY

**Why:** skills.py segment_object lacks debug JPEG overlay saving, structured `SegmentResult`,
retry on empty mask.

**Portal RPC sites:** Only `_get_portal().get_camera_image(cam)` → `env.render_rgb(cam)`.
SAM3 HTTP call and artifact saving are pure — unchanged.

**Direct env constructor:**
```python
class SegmentObjectTool:
    def __init__(self, env=None, cap_server_host="localhost", cap_server_port=CAP_SERVER_PORT):
        self._env = env
        ...
    def _fetch_image(self, camera: str) -> np.ndarray:
        if self._env:
            return self._env.render_rgb(camera)
        return np.asarray(self._get_portal().get_camera_image(camera).result())
```

---

### 4. `DetectObjectsOneshotTool` (detection.py) — MEDIUM PRIORITY

**Why:** skills.py detect_objects_oneshot lacks debug saving, richer `Detection` result,
multi-view aggregation.

**Portal RPC sites:** `client.get_camera_image/depth/intrinsics/extrinsics()` → direct env.
BundleSDF HTTP call is pure — unchanged.

---

### 5. `SampleGraspPoseAnyGraspTool` (grasp_anygrasp.py) — MEDIUM PRIORITY

**Why:** skills.py version (already fixed K/T_cam_world) lacks debug JPEG saving per attempt,
richer `GraspCandidate` metadata, `disable_planner_z_clipping`.

**Portal RPC sites:** `_get_portal().get_camera_image/depth()` → direct env.
AnyGrasp HTTP call and debug saving are pure — unchanged.

---

### 6. `SampleGraspPose2DTool` (grasp_2d.py) — MEDIUM PRIORITY

**Why:** skills.py delegates to a basic stateless function. The tool version has full mask-based
planning — RANSAC OBB, principal axis, multi-yaw candidates, rendered overlay debug image.

**Portal RPC sites:** `_get_portal().get_camera_image/depth()` → direct env.
All grasp geometry (1100+ lines of pure numpy) — unchanged.

---

### 7. `VlmQueryTool` (vlm_query.py) — MEDIUM PRIORITY

**Why:** skills.py vlm_query lacks debug JPEG saving per query, artifact dir integration,
structured result with token counts.

**Portal RPC sites:** `_get_portal().get_camera_image(cam)` → `env.render_rgb(cam)`.
VLM HTTP calls and artifact saving are pure — unchanged.

---

### 8. `NudgeTool` (nudge.py) — LOW PRIORITY

**Why:** skills.py nudge is functional. Missing: structured `NudgeResult`, `preview_only`.
nudge.py itself uses Portal RPC for both state read and execution — skills.py direct
implementation is the better base here.

**Migration:** Additive only — add `NudgeResult` return type and `preview_only` to existing
skills.py implementation. No tool-class refactor needed.

---

## Functions to DELETE from skills.py

### `move_eef_pose_precise` — DELETE

Pure Mink IK Cartesian servo loop with tolerance checking. No cuRobo. Duplicates the old
`freespace_move` mechanism at a stricter tolerance. With the migrated `FreespaceMoveTool`
(cuRobo-backed, full diagnostics), there is no reason for this to exist.

**Action:** Remove from skills.py and remove from the `make_namespace()` export dict.
Any script calling `move_eef_pose_precise` should use `freespace_move` instead.

### `move_bimanual_joint_keypoints` — MAKE INTERNAL, NOT EXPORTED

This is a joint trajectory **replay** primitive — takes pre-computed `(timestamps,
left_positions, right_positions)` from cuRobo and replays them on both arms from one clock.
It does no planning. It is the execution backend that the migrated `freespace_move` calls
after cuRobo returns a trajectory (replacing `_execute_trajectory(portal_client, ...)`).

**Action:** Keep the implementation but rename to `_move_bimanual_joint_keypoints` (private)
and remove from the `make_namespace()` export dict. Not a user-facing tool.

### Motion planning policy (enforced by this migration)

> **Only two motion tools are exposed to scripts:**
> - `freespace_move` — all EE motion planning via cuRobo
> - `nudge` — small-scale delta moves only
>
> No other motion primitives in the namespace.

---

## Functions NOT to migrate (tool equivalent exists in native.py — add `env=` there)

`GoHomeTool`, `SetGripperTool`, `OpenGripperTool`, `CloseGripperTool`, `MoveJointKeypointsTool`,
and `GetRobotStateTool` / `GetCameraImageTool` are all already in `native.py`. Once those tools
gain `env=` support, their skills.py implementations are replaced by tool instantiation in
`make_namespace()` — same as the perception/motion tools above.

| Function | Action |
|---|---|
| `go_home` | Replace with `GoHomeTool(env=env)` |
| `set_gripper` / `open_gripper` / `close_gripper` | Replace with `SetGripperTool(env=env)` etc. |
| `get_robot_state` | Replace with `GetRobotStateTool(env=env)` |
| `get_camera_image` | Replace with `GetCameraImageTool(env=env)` |
| `move_eef_pose_precise` | **Deleted** — replaced by `freespace_move` (cuRobo) |
| `move_bimanual_joint_keypoints` | **Made private** (`_move_bimanual_joint_keypoints`) — internal execution primitive for `freespace_move`, not exported |

## New tool file: `cap/agent/tools/camera.py`

`get_camera_intrinsics`, `get_camera_extrinsics`, `render_rgb`, `render_depth` are not unique to
skills.py — they're used inline via Portal RPC in at least 5 existing tools
(`grasp_anygrasp.py`, `detection.py`, `grasp_2d.py`, `grasp_3d_bb.py`, `object_tracking.py`).
They should be a proper tool file so the logic lives in one place.

**New tools in `cap/agent/tools/camera.py`:**

| Tool class | Function exposed | Returns |
|---|---|---|
| `GetCameraIntrinsicsTool` | `get_camera_intrinsics(camera)` | `[fx, fy, cx, cy]` + K (3×3) |
| `GetCameraExtrinsicsTool` | `get_camera_extrinsics(camera)` | `{position, rotation}` + T_cam_world (4×4) |
| `RenderRgbTool` | `render_rgb(camera)` | `np.ndarray` (H,W,3) uint8 |
| `RenderDepthTool` | `render_depth(camera)` | `np.ndarray` (H,W) float32 metres |

All four get `env=` support. Portal RPC path: `client.get_camera_intrinsics/extrinsics/image/depth()`.
Direct env path: `env.get_camera_intrinsics/extrinsics/render_rgb/render_depth()`.

**Side effect:** All tools that currently do inline Portal RPC camera fetches
(`grasp_anygrasp.py`, `detection.py`, `grasp_2d.py`, `grasp_3d_bb.py`, `object_tracking.py`)
are refactored to call `self._camera.get_intrinsics(cam)` / `self._camera.render_rgb(cam)`
via a shared `CameraAccessor` helper — no more duplicated Portal RPC calls scattered across files.

## Truly unique to skills.py (no cap/agent/tools equivalent — keep as-is)

| Function | Reason |
|---|---|
| `nudge_world_pose_preserving` | Specific variant not in nudge.py |
| `display_rpy_to_quat` | Pure math utility |
| `get_task_info` | Direct env call, no tool equivalent |

## Expected skills.py size after migration

**Before:** ~2200 lines (full reimplementation of every tool)

**After:** ~300 lines total:
- `make_namespace()` shim instantiating all tools with `env=env` (~100 lines)
- The 5 truly unique functions above (~150 lines)
- `_move_bimanual_joint_keypoints` private execution primitive (~80 lines)

---

## Migration order

1. `FreespaceMoveTool` + `select_best_grasp` — fixes bimanual birdseye view, proper grasp ranking
2. `SegmentObjectTool` / `SegmentAllObjectsTool` — debug images for segmentation
3. `SampleGraspPoseAnyGraspTool` — grasp debug visibility
4. `DetectObjectsOneshotTool` — richer detection results
5. `SampleGraspPose2DTool` — full 2D grasp planning
6. `VlmQueryTool` — debug image saving per query
7. `NudgeTool` — additive improvements only

---

## skills.py after migration (thin shim)

```python
# cap/env/real_bimanual_yam/skills.py

def make_namespace(env, vlm_backend="gemini", cfg=None):
    from cap.agent.tools.freespace_move import FreespaceMoveTool
    from cap.agent.tools.segmentation import SegmentObjectTool, SegmentAllObjectsTool
    from cap.agent.tools.detection import DetectObjectsOneshotTool
    from cap.agent.tools.grasp_anygrasp import SampleGraspPoseAnyGraspTool
    from cap.agent.tools.grasp_2d import SampleGraspPose2DTool
    from cap.agent.tools.vlm_query import VlmQueryTool

    _fm    = FreespaceMoveTool(env=env)
    _seg   = SegmentObjectTool(env=env)
    _segA  = SegmentAllObjectsTool(env=env)
    _det   = DetectObjectsOneshotTool(env=env)
    _ag    = SampleGraspPoseAnyGraspTool(env=env)
    _2d    = SampleGraspPose2DTool(env=env)
    _vlm   = VlmQueryTool(env=env, default_backend=vlm_backend)

    # Direct-env-only tools (unchanged from today)
    def get_robot_state(): ...
    def go_home(): ...
    def set_gripper(...): ...
    ...

    return {
        "freespace_move":           lambda **kw: _fm.execute(**kw).data,
        "select_best_grasp":        lambda **kw: _fm.execute(**kw).data,
        "segment_object":           lambda **kw: _seg.execute(**kw).data,
        "segment_all_objects":      lambda **kw: _segA.execute(**kw).data,
        "detect_objects_oneshot":   lambda **kw: _det.execute(**kw).data,
        "sample_grasp_pose_anygrasp": lambda **kw: _ag.execute(**kw).data,
        "sample_grasp_pose_2d":     lambda **kw: _2d.execute(**kw).data,
        "vlm_query":                lambda **kw: _vlm.execute(**kw).data,
        "get_robot_state":          get_robot_state,
        "go_home":                  go_home,
        "set_gripper":              set_gripper,
        ...
    }
```
