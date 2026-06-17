# Table Bussing Skill Tools

CAP agent tool abstractions for ultra long-horizon table bussing tasks.

> **Last verified against codebase**: 2026-04-08
>
> **Cross-references**:
> - [CAP_DESIGN.md](CAP_DESIGN.md) -- CAP system architecture, Portal RPC layer, agent orchestrator
> - [RL_PIPELINE_DESIGN.md](RL_PIPELINE_DESIGN.md) -- `execute_skill`, `learn_skill`, reward setup, policy output tools
> - [SAFETY_ZONE_DESIGN.md](SAFETY_ZONE_DESIGN.md) -- `set_safety_zone` / `clear_safety_zone` for RL exploration
> - [BUNDLESDF_OBJECT_DETECTION.md](BUNDLESDF_OBJECT_DETECTION.md) -- BundleSDF multi-object 6-DOF tracking internals
> - [VLM_QUERY.md](VLM_QUERY.md) -- Multi-backend VLM architecture (smol_vlm, gemini, qwen, gpt)
> - [grasp_orientation.md](grasp_orientation.md) -- Grasp orientation and AnyGrasp debug UI
> - [MULTI_CAMERA_CONFIG.md](MULTI_CAMERA_CONFIG.md) -- Camera naming (`top`, `left`, `right`) and extrinsics


## Overview

The CAP agent exposes all tools through a dynamic `ToolRegistry` (`cap/agent/tools/__init__.py:29`).
LLM-generated code or direct tool calls receive plain-function wrappers via `registry.callable_dict()` (`cap/agent/tools/__init__.py:57`), so tools can be called as `result = freespace_move(...)` in generated scripts.

Every tool returns `ToolResult(success, data, error)` (`cap/agent/tools/base.py:11`).

### Core Table Bussing Tools

These are the primary tools used in table bussing pick-and-place sequences:

| Tool | File | Line | Purpose |
|------|------|------|---------|
| `track_object` | `cap/agent/tools/object_tracking.py` | 96 | Start real-time 6-DOF tracking via BundleSDF |
| `get_object_pose` | `cap/agent/tools/object_tracking.py` | 192 | Read latest tracked object pose (world frame) |
| `stop_tracking` | `cap/agent/tools/object_tracking.py` | ~305 | Stop tracking session, free GPU |
| `freespace_move` | `cap/agent/tools/freespace_move.py` | 79 | Collision-free motion planning + execution (cuRobo default, RRT-Connect optional) |
| `nudge` | `cap/agent/tools/nudge.py` | 41 | Small delta EE adjustments (position + orientation) in world frame |
| `grasp` | `cap/agent/tools/grasp.py` | 31 | High-level grasp primitive (approach + close gripper + lift) |
| `place` | `cap/agent/tools/grasp.py` | 194 | High-level place primitive (hover + descend + open gripper + retreat) |
| `list_scene_objects` | `cap/agent/tools/scene_objects.py` | 46 | Detect objects in scene via Qwen3-VL |
| `vlm_query` | `cap/agent/tools/vlm_query.py` | ~70 | Multi-backend VLM queries (qwen / smol_vlm / gemini / gemini_pro / gpt) |
| `segment_object` | `cap/agent/tools/segmentation.py` | 26 | SAM3 text-prompted segmentation returning binary mask |
| `sample_grasp_pose_anygrasp` | `cap/agent/tools/grasp_anygrasp.py` | 109 | AnyGrasp 6-DOF grasp candidate planning (SAM3 + AnyGrasp pipeline) |
| `save_image` | `cap/agent/tools/save_image.py` | 39 | Save camera/local/web image to disk |

### Gripper Tools

| Tool | File | Line | Purpose |
|------|------|------|---------|
| `set_gripper` | `cap/agent/tools/native.py` | 216 | Set gripper to specific position with optional vel/torque limits |
| `open_gripper` | `cap/agent/tools/native.py` | 263 | Open gripper (pos=1.0) with optional vel/torque limits |
| `close_gripper` | `cap/agent/tools/native.py` | 303 | Close gripper (pos=0.0) with optional vel/torque limits |

### Motion Primitives (Low-Level)

| Tool | File | Line | Purpose |
|------|------|------|---------|
| `get_robot_state` | `cap/agent/tools/native.py` | 45 | Joint positions, gripper states, EE poses (both arms) |
| `freespace_move` | `cap/agent/tools/freespace_move.py` | 79 | Collision-free EE motion planning (cuRobo default) |
| `nudge` | `cap/agent/tools/nudge.py` | 41 | Small delta EE adjustments (calls `_ik_servo` internally) |
| `_ik_servo_keypoints` | `cap/agent/tools/native.py` | 371 | Timestamped EEF waypoints via Pink IK (internal) |
| `move_joint_keypoints` | `cap/agent/tools/native.py` | 426 | Joint-space trajectory via timestamped waypoints |
| `go_home` | `cap/agent/tools/native.py` | 348 | Move both arms to zero/home configuration |
| `get_camera_image` | `cap/agent/tools/native.py` | 472 | Get latest RGB image from `top`, `left`, or `right` camera |

### Detection and Tracking (Multi-Object)

| Tool | File | Line | Purpose |
|------|------|------|---------|
| `detect_object` | `cap/agent/tools/detection.py` | 39 | Detect objects via BundleSDF or sim oracle (single-shot or tracked) |
| `detect_objects_oneshot` | `cap/agent/tools/detection.py` | ~120 | One-shot multi-object detection wrapper |
| `add_detection` | `cap/agent/tools/bundlesdf_track.py` | 42 | Start 6-DOF multi-object tracking session (async) |
| `get_detection` | `cap/agent/tools/bundlesdf_track.py` | ~80 | Poll latest pose for a named tracking session |
| `end_detection` | `cap/agent/tools/bundlesdf_track.py` | ~110 | End a named tracking session |
| `list_detections` | `cap/agent/tools/bundlesdf_track.py` | ~130 | List all active tracking sessions |

### RL and Policy Tools

| Tool | File | Line | Purpose |
|------|------|------|---------|
| `execute_skill` | `cap/agent/tools/skill.py` | 22 | Run a learned flow-matching policy |
| `learn_skill` | `cap/agent/tools/skill.py` | 73 | RL training episode with human-in-the-loop takeover |
| `start_policy_output` | `cap/agent/tools/policy_output.py` | 61 | Prepare an external policy model session |
| `step_policy_output` | `cap/agent/tools/policy_output.py` | 117 | Execute bounded burst from active policy session |
| `stop_policy_output` | `cap/agent/tools/policy_output.py` | 144 | Stop active policy-output session |
| `use_policy_output` | `cap/agent/tools/policy_output.py` | 161 | One-shot start+step+stop convenience wrapper |
| `setup_reward` | `cap/agent/tools/native.py` | 624 | Set reward function for RL training loop |

### Safety and Scene Management

| Tool | File | Line | Purpose |
|------|------|------|---------|
| `set_safety_zone` | `cap/agent/tools/safety.py` | 50 | Set task-aware EE safety zone (convex hull of keyposes + margins) |
| `clear_safety_zone` | `cap/agent/tools/safety.py` | 108 | Clear safety zone for one or both arms |
| `get_safety_zone` | `cap/agent/tools/safety.py` | 141 | Query current safety zone configuration |
| `setup_scene` | `cap/agent/tools/native.py` | 499 | Load a named scene into simulation |
| `clear_table` | `cap/agent/tools/native.py` | 522 | Remove all scene objects (sim-only) |
| `list_scenes` | `cap/agent/tools/native.py` | 542 | List available scene files |
| `get_object_positions` | `cap/agent/tools/native.py` | 562 | Get positions/orientations of all scene objects (sim-only) |
| `set_body_pose` | `cap/agent/tools/native.py` | 584 | Set a scene body's pose in simulation |


## AnyGrasp Object-Input Default

Where table-bussing flows use the SAM3 + AnyGrasp pipeline (`cap/agent/tools/grasp_anygrasp.py:109`),
the default AnyGrasp input is the **SAM3-segmented object point cloud**
(`object_input_mode="segmented_object_cloud"`, line 164). This is the repo-wide default
because it reduces grasp ambiguity compared with sending a larger
ROI/workspace-cropped cloud that may still contain neighboring clutter.

`roi_workspace` is retained only as an explicit override for debugging,
ablation, or comparison workflows. The parameter docstring marks it as **DEPRECATED** (line 161).

The AnyGrasp server runs at port 8122 (default, configurable via `ANYGRASP_SERVER_HOST` / `ANYGRASP_SERVER_PORT` env vars in `cap/config.py:285`).


## Coordinate Frame Contract

**All tool inputs and outputs use the robot world frame (URDF `base_link`):**

```
    +X  forward  (toward the work table)
    +Y  left     (toward the left arm, y=+0.31)
    +Z  up       (sky)
    Origin: floor level, centred between arm bases (left at y=+0.31, right at y=-0.31)
```

Reference points: floor z=0, table surface z~0.745, arm bases at (0.2525, +/-0.31, 0.75).

**Frame consistency across tools:**

| Data | Source | Frame | Code Reference |
|------|--------|-------|----------------|
| `get_robot_state()` EE poses | Pinocchio FK | World | `cap/agent/tools/native.py:56` |
| `get_object_pose()` position/quat | BundleSDF `ob_in_cam` **transformed tool-side** via cap_server extrinsics | World | `cap/agent/tools/object_tracking.py:266` |
| `freespace_move()` targets | User input | World (same as FK) | `cap/agent/tools/freespace_move.py:104` |
| `nudge()` deltas | User input | World (applied to FK-sourced current pose) | `cap/agent/tools/nudge.py:53` |
| `_ik_servo()` targets | User input | World (Pinocchio IK) | `cap/server/cap_server.py:1485` |
| `sample_grasp_pose_anygrasp()` candidates | AnyGrasp, remapped to planner convention | World (display RPY) | `cap/agent/tools/grasp_anygrasp.py:109` |

**Camera-to-world transform** (inside `get_object_pose`, `cap/agent/tools/object_tracking.py:61`):
1. Read `ob_in_cam` (4x4 SE3, OpenCV convention: +x right, +y down, +z forward) from BundleSDF
2. Get camera extrinsics from cap_server (Pinocchio FK via Portal RPC)
3. Apply mount flip `F=diag(-1,-1,1)` for D405 cameras to convert OpenCV to Pinocchio frame
   (ZED 2i uses no flip -- controlled by `robot/models/station/paths.py:needs_optical_flip()`)
4. Compose: `ob_in_world = T_cam_to_world @ ob_in_cam`

This transform is done **inside the tool**, not relying on `tools/vision/serve_bundlesdf.py`'s own `ob_in_world` (which has a silent fallback to identity on extrinsics failure).

**Display RPY convention** (`cap/agent/tools/native.py:59`): The RPY values exposed to LLM/agent code
use a Viser UI display convention where `roll = euler_xyz[1]`, `pitch = -euler_xyz[0]`,
`yaw = -(euler_xyz[2] + 90)`. The inverse mapping is in `nudge` / `freespace_move` tool internals.
`freespace_move` and `grasp_anygrasp` use the same convention internally.


## Architecture

```
CAP Agent (LLM generates code or tool calls)
    |
    |-- ToolRegistry.callable_dict()           --> plain-function wrappers (cap/agent/tools/__init__.py:57)
    |
    |-- track_object("red cup")                --> tools/vision/serve_bundlesdf.py (HTTP :8119)
    |                                               '-- SAM3 detect --> SAM2 track --> BundleSDF 6-DOF
    |-- get_object_pose()                      --> tools/vision/serve_bundlesdf.py GET /get_detection/{name}
    |                                          --> cap_server get_camera_extrinsics (Portal RPC :8300)
    |                                          --> tool applies cam-->world transform internally
    |
    |-- list_scene_objects()                   --> Qwen3-VL via vLLM OpenAI API (:8402)
    |-- vlm_query("Is the cup upright?")       --> qwen (:8402) / smol_vlm (:8401) / gemini / gpt
    |
    |-- segment_object("red cup")              --> tools/vision/serve_sam3.py (HTTP :6767)
    |-- sample_grasp_pose_anygrasp("red cup")  --> SAM3 (:6767) + AnyGrasp (:8122)
    |                                               '-- returns ranked GraspCandidate list
    |
    |-- freespace_move(                        --> PortalMotionPlanner (cuRobo) or YamMotionPlanner (RRT)
    |     left_target_pos=...,                      '-- cap_server move_joint_keypoints (Portal RPC :8300)
    |     left_target_rpy=...,
    |   )
    |   (batch mode)                           --> cuRobo CUDA-graph batched ranking (up to 16 candidates)
    |   (cached trajectory)                    --> replay cached trajectory directly, no replanning
    |
    |-- grasp("left", position=..., rpy=...)   --> freespace_move + nudge + set_gripper internally
    |-- place("left", position=..., rpy=...)   --> freespace_move + open_gripper internally
    |
    |-- nudge("left",                          --> cap_server _ik_servo (Portal RPC :8300)
    |     delta_pos=[0,0,0.02])
    |
    |-- set_gripper("left", 0.5)               --> cap_server set_gripper (Portal RPC :8300)
    |
    |-- set_safety_zone("left", keyposes=...)  --> cap_server set_safety_zone (Portal RPC :8300)
    |
    |-- execute_skill("grasp", params={})      --> cap_server execute_skill (flow-matching policy)
    |-- learn_skill("pick", params={})         --> cap_server learn_skill (RL training loop)
    |-- use_policy_output("pi05", ...)         --> cap_server start/step/stop_policy_output
    |
    '-- save_image(media="camera:top", ...)    --> cap_server get_camera_image + local disk
```

### External Services Required

| Service | Port | Config | Purpose |
|---------|------|--------|---------|
| `cap_server` | 8300 | `cap/config.py:28` | All motion/gripper/state RPCs via Portal |
| `serve_bundlesdf.py` | 8119 | `cap/config.py:37` | 6-DOF pose tracking (BundleSDF + SAM2) |
| `serve_sam3.py` | 6767 | `cap/config.py:43` | SAM3 text-prompted segmentation |
| AnyGrasp server | 8122 | `cap/config.py:286` | Grasp pose planning |
| Qwen3-VL (vLLM) | 8402 | `cap/config.py:251` | VLM for scene understanding + `list_scene_objects` |
| SmolVLM (vLLM) | 8401 | `cap/config.py:240` | Lightweight VLM backend |
| cuRobo planner | auto | env `CAP_CUROBO_PORT` | GPU motion planner (started by `freespace_move` unless `CAP_CUROBO_START_SERVER=0`) |


## Tool Details

### 1. Object Tracking (`track_object`, `get_object_pose`, `stop_tracking`)

**File**: `cap/agent/tools/object_tracking.py`

Wraps the BundleSDF server HTTP API. The server runs in a separate process, pulling RGB+depth from cap_server and continuously updating the latest 6-DOF pose.

A shared `_TrackingContext` (line 48) carries the active camera name and session key between the three tools so `get_object_pose` can fetch the correct extrinsics.

```python
# Start tracking
track_object("red cup", camera="top")

# Read pose (call repeatedly -- always returns latest)
pose = get_object_pose()  # Detection3D with position_3d, quaternion_xyzw

# Stop when done
stop_tracking()
```

**Returns** `Detection3D` (`cap/agent/tools/base.py:97`):
- `position_3d`: `[x, y, z]` in world frame (metres)
- `quaternion_xyzw`: `[x, y, z, w]` in world frame
- `label`: the text query used for tracking
- `score`: tracker confidence
- `half_extents`: object bounding box half-extents (if available)

**Note**: `rpy` field on `Detection3D` is NOT populated by `get_object_pose` (it returns quaternion only). Convert with scipy if you need RPY.


### 2. Free-space Movement (`freespace_move`)

**File**: `cap/agent/tools/freespace_move.py:79`

Replaces the former linear-interpolation EE motion tool for free-space motions. Uses cuRobo by default for GPU-accelerated collision-free planning, with RRT-Connect available as an alternate backend.

**All orientations are RPY in degrees** (display RPY convention).

```python
# Single arm
result = freespace_move(
    left_target_pos=[0.3, 0.1, 0.8],
    left_target_rpy=[0, 90, 0],
    ik_error_threshold=0.005,  # 5 mm (default)
)

# Bimanual
result = freespace_move(
    left_target_pos=[0.3, 0.1, 0.8],
    left_target_rpy=[0, 90, 0],
    right_target_pos=[-0.3, 0.1, 0.8],
    right_target_rpy=[0, 90, 0],
)

# Batched grasp candidate ranking (AnyGrasp integration)
result = freespace_move(
    grasp_candidates=[
        {"position": [0.3, 0.1, 0.78], "rpy": [0, 90, 0], "score": 0.9, "width": 0.04},
        {"position": [0.3, 0.0, 0.78], "rpy": [0, 85, 0], "score": 0.8, "width": 0.05},
    ],
    batch_side="right",
    batch_top_k=16,
    preview_only=True,  # rank without executing
)
# Then execute the best candidate's cached trajectory:
freespace_move(trajectory_cache_key=result.best_candidate.trajectory_cache_key)
```

**Returns** `FreespaceResult` (`cap/agent/tools/base.py:148`):
- `status`: `"Success"`, `"IK_Failed"`, `"Planning_Failed"`, or `"Execution_Failed"`
- `ik_error_m`: maximum IK position error in metres
- `final_pos_error_m`: maximum final Cartesian position error
- `final_rot_error_deg`: maximum final orientation error in degrees
- `trajectory_steps`: number of waypoints in the planned trajectory
- `trajectory_cache_key`: key for replaying via `trajectory_cache_key` parameter
- `executed`: whether the trajectory was actually sent to the robot
- `reason`: human-readable failure explanation
- `planning_mode`: `"single"` | `"batch"` | `"cached"`
- `batch_candidates`: ranked `FreespaceBatchCandidate` list (batch mode only)
- `best_candidate`: top-ranked candidate (batch mode only)
- Timing fields: `timing_total_ms`, `timing_plan_eval_ms`, `curobo_solve_time_ms`, etc.

**Key parameters** (all optional except targets):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `ik_error_threshold` | 0.005 m (5 mm) | Max acceptable IK position error |
| `ik_rot_threshold_deg` | ~2.86 deg | Max acceptable IK rotation error |
| `ik_xyz_weight` | 1.0 | IK translation cost weight |
| `ik_rpy_weight` | 0.3 | IK orientation cost weight |
| `planning_speed` | 1.5 rad/s | Planner speed (clamped to [0.05, 3.0]) |
| `backend` | `"curobo"` | `"curobo"` or `"rrt-connect"` |
| `solver_speed` | `"fast"` | cuRobo solver preset: `"fast"` or `"slow"` |
| `preview_only` | `False` | Plan but do not execute |
| `left_gripper` / `right_gripper` | current | Gripper state for collision checking |
| `left_gripper_target_width` / `right_gripper_target_width` | None | Synchronized gripper command during trajectory |
| `grasp_candidates` | None | Batch candidate list for ranked planning |
| `batch_side` | None | Which arm evaluates batch candidates |
| `batch_top_k` | 16 | Max candidates for CUDA-graph batch planner |
| `trajectory_cache_key` | None | Replay a cached trajectory directly |


### 3. Nudge (`nudge`)

**File**: `cap/agent/tools/nudge.py:41`

Small delta EE adjustment in world frame. Uses `_ik_servo` internally -- robust for nearby targets.

```python
nudge("left", delta_pos=[0, 0, 0.02])                        # 2 cm up
nudge("right", delta_rpy=[0, 0, 5.0])                        # yaw +5 degrees
nudge("left", delta_pos=[0.01, 0, 0], delta_rpy=[0, 5.0, 0]) # combined
```

**Parameters**:
- `side` (required): `"left"` or `"right"`
- `delta_pos` (optional): `[dx, dy, dz]` in metres, world frame. Default `[0, 0, 0]`.
- `delta_rpy` (optional): `[droll, dpitch, dyaw]` in **degrees**, extrinsic XYZ. Default `[0, 0, 0]`.
- `max_duration_sec` (optional): move timeout in seconds (default from `MOVE_EEF_MAX_DURATION_S` = 5.0 s)
- `max_vel` (optional): max EE speed in m/s (default from `MOVE_EEF_MAX_VEL` = 0.3 m/s)

At least one of `delta_pos` or `delta_rpy` must be provided.

**Returns** `NudgeResult` (`cap/agent/tools/base.py:197`):
- `success`: bool
- `final_pos`: `[x, y, z]` final EE position (world frame, rounded to 4 decimals)
- `final_quat`: `[x, y, z, w]` final EE quaternion (world frame, rounded to 4 decimals)

**Implementation**: reads current EE pose via Portal `get_state()`, computes new pose as
`new_pos = cur_pos + delta_pos` and `new_rot = delta_rot @ cur_rot` (world-frame rotation composition),
then calls `_ik_servo()` to reach the target.


### 4. Grasp and Place (`grasp`, `place`)

**File**: `cap/agent/tools/grasp.py`

High-level pick-and-place primitives that combine freespace motion + gripper control.
Both tools internally instantiate `FreespaceMoveTool` and `NudgeTool` (lazy, line 82-91).

**`grasp`** (line 31) -- Approach, close gripper, lift sequence:
1. Open gripper
2. `freespace_move` to hover position (`pre_height` above grasp Z)
3. Descend to grasp height -- uses `freespace_move` if above table, `nudge` if near table surface
4. Close gripper
5. Lift back to hover (falls back to `nudge` upward if `freespace_move` fails)

```python
grasp("left", position=[0.3, 0.1, 0.78], rpy=[0, 90, 0])
grasp("right", position=[0.3, -0.1, 0.78], pre_height=0.15, z_offset=0.03)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `side` | required | `"left"` or `"right"` |
| `position` | required | Object position `[x, y, z]` in metres (world frame) |
| `rpy` | current arm RPY | Grasp orientation `[roll, pitch, yaw]` in degrees |
| `pre_height` | 0.10 m | Hover height above object before descending |
| `z_offset` | 0.05 m | Extra Z offset added to grasp position for safety |

**`place`** (line 194) -- Move to target and release:
1. `freespace_move` to hover above target (`pre_height` above target Z)
2. Descend to place height (target Z + 0.05 m)
3. Open gripper
4. Lift back to hover

```python
place("left", position=[0.3, -0.1, 0.85], rpy=[0, 90, 0])
place("right", position=[0.5, 0.0, 0.80], pre_height=0.20)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `side` | required | `"left"` or `"right"` |
| `position` | required | Target position `[x, y, z]` in metres (world frame) |
| `rpy` | current arm RPY | Orientation `[roll, pitch, yaw]` in degrees |
| `pre_height` | 0.15 m | Hover height above target |

**Table surface clamping**: both tools clamp hover Z to at least `_TABLE_Z + 0.10` (0.85 m) if the
computed hover would be too close to the table surface (`_TABLE_Z = 0.75`, line 28).


### 5. Gripper (`set_gripper`, `open_gripper`, `close_gripper`)

**File**: `cap/agent/tools/native.py`

Pre-existing tools. Support `vel_limit` (speed) and `torque_limit` (grip force).
All use `GRIPPER_SETTLE_TIMEOUT_S` = 1.5 s (`cap/config.py:185`) to block until the gripper settles.

```python
set_gripper("left", pos=0.5, vel_limit=0.5)    # half-open, slow
close_gripper("left", torque_limit=0.3)          # gentle grasp
open_gripper("right")                            # full open, default speed
```


### 6. Scene Understanding (`list_scene_objects`, `vlm_query`)

**`list_scene_objects`** (`cap/agent/tools/scene_objects.py:46`):
Captures top camera image, sends to Qwen3-VL, parses structured JSON list of object names.

```python
objects = list_scene_objects()                           # auto-detect via Qwen3-VL
objects = list_scene_objects(camera="left")              # from wrist cam
objects = list_scene_objects(prompt="Count the plates")  # custom prompt
```

**`vlm_query`** (`cap/agent/tools/vlm_query.py`):
Multi-backend VLM queries with flexible media sources.

```python
result = vlm_query("Is the cup upright?")                     # default backend (qwen)
result = vlm_query("What does the gripper hold?", media=["camera:left"])
result = vlm_query("Describe the scene", backend="gemini")    # Google Gemini
result = vlm_query("Check alignment", backend="gpt")          # OpenAI GPT
```

**VLM backends** (configurable via `cap/config.py:240-258`):
| Backend | Model | URL/Config | API Key |
|---------|-------|------------|---------|
| `qwen` (default) | Qwen3-VL-8B-Instruct | localhost:8402 | None (vLLM) |
| `smol_vlm` | SmolVLM-256M-Instruct | 172.26.34.251:8401 | None (vLLM) |
| `gemini` | gemini-2.5-flash | Google API | `GEMINI_API_KEY` |
| `gemini_pro` | gemini-3.1-pro-preview | Google API | `GEMINI_API_KEY` |
| `gpt` | gpt-5.4 | OpenAI API | `OPENAI_API_KEY` |

**Media source prefixes**: `camera:top`, `camera:left`, `camera:right` (live capture),
`local:~/path/to/image.png` (local file), `web:https://...` (URL download).


### 7. Segmentation (`segment_object`)

**File**: `cap/agent/tools/segmentation.py:26`

Text-prompted segmentation via SAM3 server. Returns a binary mask for the described object.

```python
seg = segment_object("red cup")                          # from top camera
seg = segment_object("plate", media="camera:left")       # from wrist cam
seg = segment_object("block", media="local:snapshot.png") # from saved image
```

**Returns** `SegmentationResult` (`cap/agent/tools/base.py:110`):
- `mask`: numpy uint8 HxW (0/1 binary mask)
- `bbox_xywh`: `[x, y, w, h]` bounding box
- `score`: segmentation confidence
- `mask_area`: number of pixels in the mask


### 8. AnyGrasp Grasp Planning (`sample_grasp_pose_anygrasp`)

**File**: `cap/agent/tools/grasp_anygrasp.py:109`

Plans 6-DOF grasp candidates using the SAM3 + AnyGrasp pipeline. Returned poses use the same
world-frame display-RPY convention as `freespace_move`, so they can be passed directly.

```python
grasps = sample_grasp_pose_anygrasp("red cup")
grasps = sample_grasp_pose_anygrasp(
    "plate",
    camera="top",
    max_grasps=10,
    top_down_only=True,
    vertical_threshold=0.8,
)
# Each candidate: GraspCandidate(position, rpy, score, width)
best = grasps[0]
freespace_move(right_target_pos=best.position, right_target_rpy=best.rpy)
```

**Parameters**:
- `object_name` (required): natural-language object description
- `camera`: `"top"` (default), `"left"`, `"right"`
- `max_grasps`: max candidates to return (default 10)
- `top_down_only`: filter for top-down grasps only (default False)
- `vertical_threshold`: dot product threshold for top-down filtering (default 0.8)
- `object_input_mode`: `"segmented_object_cloud"` (default) or `"roi_workspace"` (deprecated)
- `tcp_offset_z_m`: TCP offset along local +Z before returning grasps
- `disable_planner_z_clipping`: skip table-safe floor clamping

**Grasp frame remapping** (line 49): AnyGrasp uses a vendor gripper frame (+X approach, +Y finger opening,
+Z height). The tool remaps to the planner convention (+X opening, +Y height, +Z approach)
via `_ANYGRASP_TO_GRIPPER` rotation matrix, then converts to display RPY.


### 9. Safety Zones (`set_safety_zone`, `clear_safety_zone`, `get_safety_zone`)

**File**: `cap/agent/tools/safety.py`

Task-aware EE safety zones for RL exploration. Set before `learn_skill` to restrict the
end-effector to a task-relevant region.

```python
set_safety_zone(
    side="right",
    keyposes=[
        [0.3, -0.1, 0.8, 0, 0, 0, 1],   # pre-grasp hover
        [0.3, -0.1, 0.75, 0, 0, 0, 1],   # grasp position
        [0.3, 0.0, 0.85, 0, 0, 0, 1],    # place position
    ],
    pos_margin=0.08,    # 8 cm around convex hull
    ori_margin=0.3,     # ~17 degrees orientation margin
)
# ... RL training ...
clear_safety_zone()     # clear both arms
```

Zones persist across `learn_skill` episodes until explicitly cleared.
See [SAFETY_ZONE_DESIGN.md](SAFETY_ZONE_DESIGN.md) for the attenuation and clamping algorithm.


### 10. Policy Execution (`execute_skill`, `learn_skill`, policy output tools)

**File**: `cap/agent/tools/skill.py` and `cap/agent/tools/policy_output.py`

**`execute_skill`** (line 22): Run a learned flow-matching policy at `POLICY_FREQ_HZ` (30 Hz).
The policy inference loop runs inside `cap_server`.

**`learn_skill`** (line 73): RL training episode. Single-step actions come from a remote
`rl_policy_server` (SAC + base agent). Supports human Fello takeover via footswitch.

**Policy output tools** provide finer-grained control over external model servers:
- `start_policy_output`: prepare a session (model name, replan horizon, task description)
- `step_policy_output`: execute bounded burst of actions
- `stop_policy_output`: end session
- `use_policy_output`: one-shot convenience wrapper

Supported models: defined in `POLICY_MODEL_CONFIGS` (`cap/config.py:78`).


### 11. Save Image (`save_image`)

**File**: `cap/agent/tools/save_image.py:39`

Saves images from cameras, local files, or URLs to disk. Uses the same media prefix
convention as `vlm_query`.

```python
save_image(media="camera:top", path="cap/tasks/demo/")         # auto-timestamped
save_image(media="camera:left", path="~/snapshots/left.png")   # specific filename
```


## Intended Usage: Pick-and-Place Loop

```python
# 1. Survey the scene
objects = list_scene_objects()

# 2. Start tracking the target object
track_object("red cup")

# 3. Plan grasp candidates
grasps = sample_grasp_pose_anygrasp("red cup", top_down_only=True)

# 4. Evaluate candidates with batched motion planning (preview mode)
result = freespace_move(
    grasp_candidates=[
        {"position": g.position, "rpy": g.rpy, "score": g.score, "width": g.width}
        for g in grasps
    ],
    batch_side="right",
    preview_only=True,
)

# 5. Execute the best candidate's cached trajectory
if result.best_candidate:
    freespace_move(trajectory_cache_key=result.best_candidate.trajectory_cache_key)

    # 6. Fine-adjust and grasp
    nudge("right", delta_pos=[0, 0, -0.03])  # final descent
    close_gripper("right", torque_limit=0.3)

    # 7. Lift and place
    freespace_move(right_target_pos=[0.5, 0.0, 0.90], right_target_rpy=[0, 90, 0])
    place("right", position=[0.5, 0.0, 0.78])

stop_tracking()
```

### Simpler Pick-and-Place (using high-level grasp/place tools)

```python
track_object("red cup")
pose = get_object_pose()

grasp("right", position=pose.position_3d, rpy=[0, 90, 0])
place("right", position=[0.5, 0.0, 0.80], rpy=[0, 90, 0])

stop_tracking()
```

### Multi-Object Table Clearing

```python
objects = list_scene_objects()
for obj_name in objects:
    track_object(obj_name)
    pose = get_object_pose()

    # Decide which arm is closer
    state = get_robot_state()
    left_dist = sum((a - b) ** 2 for a, b in zip(state.left_ee_pos, pose.position_3d)) ** 0.5
    right_dist = sum((a - b) ** 2 for a, b in zip(state.right_ee_pos, pose.position_3d)) ** 0.5
    side = "left" if left_dist < right_dist else "right"

    grasp(side, position=pose.position_3d)
    place(side, position=BUS_BIN_POSITION)
    stop_tracking()
```


## Data Types Reference

All data types are defined in `cap/agent/tools/base.py`.

| Type | Line | Fields | Used By |
|------|------|--------|---------|
| `ToolResult` | 11 | `success`, `data`, `error` | All tools |
| `ToolParameter` | 20 | `name`, `type`, `description`, `required`, `default` | Tool schema |
| `RobotState` | 70 | `{left,right}_{joint_pos,gripper_pos,ee_pos,ee_quat,ee_rpy}` | `get_robot_state` |
| `MoveResult` | 86 | `reached`, `feasible`, `cmd_pos_err`, `final_pos`, `final_quat` | `freespace_move`, `nudge`, `move_*_keypoints` |
| `Detection3D` | 97 | `label`, `score`, `box_2d`, `position_3d`, `quaternion_xyzw`, `rpy`, `half_extents` | `detect_object`, `get_object_pose` |
| `SegmentationResult` | 110 | `mask`, `bbox_xywh`, `score`, `mask_area` | `segment_object` |
| `SkillResult` | 120 | `success`, `steps_executed`, `info` | `execute_skill`, `learn_skill`, policy output |
| `FreespaceBatchCandidate` | 129 | `rank`, `position`, `rpy`, `score`, `width`, IK/trajectory fields | `freespace_move` (batch mode) |
| `FreespaceResult` | 148 | `status`, errors, trajectory, timing, batch candidates | `freespace_move` |
| `NudgeResult` | 197 | `success`, `final_pos`, `final_quat` | `nudge` |


## Configuration Constants

Key constants from `cap/config.py`:

| Constant | Value | Line | Purpose |
|----------|-------|------|---------|
| `CAP_SERVER_PORT` | 8300 | 28 | Portal RPC for all motion/gripper tools |
| `BUNDLESDF_SERVER_PORT` | 8119 | 37 | BundleSDF HTTP tracking API |
| `SAM3_SERVER_PORT` | 6767 | 43 | SAM3 segmentation HTTP API |
| `ANYGRASP_SERVER_PORT` | 8122 | 286 | AnyGrasp grasp planning HTTP API |
| `CONTROL_FREQ_HZ` | 60.0 | 107 | cap_server control loop frequency |
| `POLICY_FREQ_HZ` | 30.0 | 110 | Policy execution frequency |
| `MOVE_EEF_MAX_DURATION_S` | 5.0 | 193 | Default _ik_servo / nudge timeout |
| `MOVE_EEF_MAX_VEL` | 0.3 m/s | 194 | Default max EE translation speed |
| `GRIPPER_SETTLE_TIMEOUT_S` | 1.5 | 185 | How long to wait for gripper to settle |
| `DEFAULT_VLM_BACKEND` | `"qwen"` | 248 | Default VLM backend for `vlm_query` |


## Saved Scripts

Historical table bussing scripts are preserved in `cap/saved_scripts/table_bussing/` organized by version:

| Version | Directory | Notable Scripts |
|---------|-----------|----------------|
| v0 | `cap/saved_scripts/table_bussing/v0/` | Initial table bussing scripts, prompts, and working logs |
| v1 | `cap/saved_scripts/table_bussing/v1/` | YOLO/VLM integration, BundleSDF one-shot, multithreaded grasp |
| v2 | `cap/saved_scripts/table_bussing/v2/` | AnyGrasp + new motion planner integration |
| v3 | `cap/saved_scripts/table_bussing/v3/` | ZED 2i camera, parallel execution, nudge optimization |
| v4 | `cap/saved_scripts/table_bussing/v4/` | Direct grasp with orientation preservation, clip thresholds |
| v5 | `cap/saved_scripts/table_bussing/v5/` | Keep-orientation strategies |
| v6 | `cap/saved_scripts/table_bussing/v6/` | BEV batched execution, loop guards, multi-plate sorting |
| v7 | `cap/saved_scripts/table_bussing/v7/` | N-class sorting |

These scripts demonstrate the evolution of the table bussing pipeline and serve as reference
implementations for LLM code generation.


## Dependencies

- `cap_server` must be running (port 8300) for all motion/gripper/state tools
- `tools/vision/serve_bundlesdf.py` must be running (port 8119) for object tracking
- `tools/vision/serve_sam3.py` must be running (port 6767) for segmentation and AnyGrasp
- AnyGrasp server must be running (port 8122) for `sample_grasp_pose_anygrasp`
- Qwen3-VL via vLLM must be running (port 8402) for `list_scene_objects` and default `vlm_query`
- `experimental.portal_motion_planner` / `experimental.motion_planner` loaded lazily by `freespace_move`
- cuRobo portal planner auto-started unless `CAP_CUROBO_START_SERVER=0` is set
