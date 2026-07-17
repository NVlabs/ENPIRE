# Grasp Orientation and Grasping Strategy

Complete reference for grasp planning, orientation selection, and pick-and-place
execution on the YAM bimanual robot. Covers both the **AnyGrasp 6-DOF neural
grasp planner** (primary) and the **heuristic RPY approach** (fallback).

> **Cross-references:**
> - [TABLE_BUSSING_SKILLS.md](TABLE_BUSSING_SKILLS.md) -- tool abstractions and AnyGrasp `object_input_mode` default
> - [BUNDLESDF_OBJECT_DETECTION.md](BUNDLESDF_OBJECT_DETECTION.md) -- 6-DOF pose tracking used by `detect_object()`
> - [CAP_DESIGN.md](CAP_DESIGN.md) -- overall CAP agent architecture
> - [SAFETY_ZONE_DESIGN.md](SAFETY_ZONE_DESIGN.md) -- task-aware EE safety zones for RL exploration

---

## 1. Coordinate Frame and RPY Convention

All poses in this repo use a **display RPY** convention (degrees) that is *not*
standard Euler XYZ. The conversion lives in `freespace_move`:

```
cap/agent/tools/freespace_move.py:635
```

```python
# Display RPY -> planner quaternion [x,y,z,w]
roll, pitch, yaw = rpy  # degrees
euler_xyz = [-pitch, roll, -yaw - 90.0]
quat_xyzw = Rotation.from_euler("xyz", euler_xyz, degrees=True).as_quat()
```

The inverse (quaternion -> display RPY) used by AnyGrasp output:

```
cap/agent/tools/grasp_anygrasp.py:61-70
```

```python
ex, ey, ez = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
display_rpy = [ey, -ex, -ez - 90.0]  # wrapped to [-180, 180)
```

**Key rule:** Always pass RPY in degrees using this display convention.
Never pass raw Euler angles or quaternions to `freespace_move`.

Home orientation is approximately `[0, 90, 0]` degrees (display RPY).

---

## 2. Grasping Strategies Overview

| Strategy | When to use | Orientation source | Tool |
|----------|-------------|-------------------|------|
| **AnyGrasp 6-DOF** | Primary for all picks | Neural network (SAM3 + AnyGrasp) | `sample_grasp_pose_anygrasp` |
| **detect_object RPY** | Fallback / simple grasps | BundleSDF pose tracker | `detect_object` + `freespace_move` |
| **Home RPY** | No detection available | Current arm orientation | `get_robot_state` + `freespace_move` |
| **`grasp()` tool** | Simple approach-grasp-lift | Current arm RPY or user-specified | `grasp` |

---

## 3. Primary Strategy: AnyGrasp 6-DOF Grasp Planning

### 3.1 Architecture

```
SAM3 segmentation -> AnyGrasp detection server -> pose transform -> freespace_move
```

**Services involved:**

| Service | Port | File | Purpose |
|---------|------|------|---------|
| AnyGrasp detection server | 8122 | `tools/vision/serve_anygrasp.py` | Neural 6-DOF grasp inference |
| AnyGrasp debug UI | 8121 | `tools/vision/serve_anygrasp_debug.py` | Visual debug dashboard |
| SAM3 segmentation server | 6767 | (external) | Object segmentation mask |
| CAP server | 8300 | `cap/server/cap_server.py` | Camera images, depth, extrinsics |

### 3.2 The `sample_grasp_pose_anygrasp` Tool

Defined at `cap/agent/tools/grasp_anygrasp.py:109-520`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `object_name` | str | required | Name of object to grasp (e.g. `"red block"`) |
| `camera` | str | `"top"` | Camera view: `"top"`, `"left"`, `"right"` |
| `max_grasps` | int | `10` | Max candidates to return |
| `top_down_only` | bool | `False` | Filter for vertical grasps only |
| `vertical_threshold` | float | `0.8` | Dot-product threshold for top-down filtering |
| `object_input_mode` | str | `"segmented_object_cloud"` | `"segmented_object_cloud"` (default) or `"roi_workspace"` (deprecated) |
| `tcp_offset_z_m` | float | `0.0` | TCP offset along local +Z before returning grasps |
| `disable_planner_z_clipping` | bool | `False` | Skip planner-Z safety floor (default 0.80 m) |

**Returns:** List of `GraspCandidate(position, rpy, score, width)` sorted by
score (best first). Position and RPY are in the same display-RPY convention
expected by `freespace_move`.

```
cap/agent/tools/grasp_anygrasp.py:89-97
```

```python
@dataclass
class GraspCandidate:
    position: list[float]   # [x, y, z] world frame (metres)
    rpy: list[float]        # display RPY [roll, pitch, yaw] (degrees)
    score: float            # AnyGrasp confidence score
    width: float = 0.08     # gripper opening width (metres)
```

### 3.3 Internal Pipeline

The tool executes this pipeline (`cap/agent/tools/grasp_anygrasp.py:326-502`):

1. **Capture** RGB, depth, intrinsics from the specified camera via Portal RPC
2. **Segment** the object using SAM3 (`/segment` endpoint on port 6767)
3. **Call AnyGrasp** `/plan_viz` with the segmented mask, depth, and intrinsics
4. **Frame transform** AnyGrasp vendor frame -> planner/gripper frame -> world frame:
   - AnyGrasp vendor frame: +X approach, +Y finger opening, +Z gripper height
   - Planner frame: +X opening, +Y height, +Z approach
   - The remapping matrix at `grasp_anygrasp.py:49-56`:
     ```python
     _ANYGRASP_TO_GRIPPER = np.array([
         [0.0, 0.0, 1.0],  # planner X = anygrasp Y
         [1.0, 0.0, 0.0],  # planner Y = anygrasp Z
         [0.0, 1.0, 0.0],  # planner Z = anygrasp X (approach)
     ])
     ```
   - Camera-to-world transform via `get_camera_extrinsics()`
5. **TCP offset** applied along local +Z (configurable, default 0.0 m)
6. **Planner Z clipping** to safety floor (default 0.80 m from `cap/config.py:52`)
7. **Top-down filtering** (optional) by dot product with world -Z
8. **Sort by score** and convert rotation to display RPY

### 3.4 AnyGrasp Detection Server

`tools/vision/serve_anygrasp.py` -- FastAPI server wrapping the AnyGrasp SDK.

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/plan` | POST | Return grasps as numpy arrays (no visualization) |
| `/plan_viz` | POST | Return grasps + overlay JPEG + per-grasp thumbnails |
| `/health` | GET | Model status and config |
| `/reset_state` | POST | Clear CUDA cache |

**Request schema** (`tools/vision/serve_anygrasp.py:79-90`):

```python
class PlanRequest(BaseModel):
    rgb_base64: str
    depth_base64: str
    cam_K_base64: str
    segmap_base64: str
    segmap_id: int = 1
    z_range: list[float] | None = None        # default [1e-6, 1.5]
    max_grasps: int = 20
    workspace_margin: float = 0.02
    collision_detection: bool = True
    object_input_mode: str = "segmented_object_cloud"
```

**Server defaults:**
- `--port 8122`
- `--top-down-grasp` enabled by default (pass `--no-top-down-grasp` to disable)
- `--max-gripper-width 0.1` (clamped to [0.0, 0.1])
- `--gripper-height 0.03`

**object_input_mode options** (`tools/vision/serve_anygrasp.py:105-127`):
- `"segmented_object_cloud"` (default): only SAM3-segmented object points sent to AnyGrasp -- reduces pose ambiguity
- `"roi_workspace"`: full scene cloud cropped by object workspace bounds -- deprecated, causes ambiguity

### 3.5 Configuration Constants

From `cap/config.py`:

| Constant | Value | Line | Purpose |
|----------|-------|------|---------|
| `GRIPPER_TCP_OFFSET_Z_M` | `0.0` | 47 | Default TCP offset along local +Z |
| `ANYGRASP_MIN_PLANNER_Z_M` | `0.80` | 52 | Safety floor for planner-facing grasp Z (env overridable) |
| `GRIPPER_DEFAULT_WIDTH_M` | `0.08` | 175 | Default grasp width |
| `ANYGRASP_SERVER_HOST` | `localhost` | 285 | AnyGrasp server hostname (env overridable) |
| `ANYGRASP_SERVER_PORT` | `8122` | 286 | AnyGrasp server port (env overridable) |
| `SAM3_SERVER_HOST` | `localhost` | 42 | SAM3 server hostname (env overridable) |
| `SAM3_SERVER_PORT` | `6767` | 43 | SAM3 server port (env overridable) |

### 3.6 Usage: Simple AnyGrasp Pick

```python
# 1. Plan grasps
grasps = sample_grasp_pose_anygrasp("orange", camera="top", max_grasps=10)
best = grasps[0]

# 2. Move directly to grasp pose (position + RPY already in planner convention)
freespace_move(left_target_pos=best.position, left_target_rpy=best.rpy)

# 3. Close gripper
close_gripper("left")
```

### 3.7 Usage: Production Pick-and-Place with Batch Ranking

The v7 table bussing scripts (`cap/saved_scripts/table_bussing/v7/nclass_sorting.py`)
demonstrate the full production flow:

```python
# 1. Get AnyGrasp candidates (SAM3 segmented cloud, no Z clipping)
#    nclass_sorting.py:388-416
grasps = sample_grasp_pose_anygrasp(
    object_name="plate",
    camera="top",
    max_grasps=16,
    object_input_mode="segmented_object_cloud",
    tcp_offset_z_m=0.0,
    disable_planner_z_clipping=True,
)

# 2. Pick arm closest to best grasp candidate
#    nclass_sorting.py:443-461
chosen_side = detect_which_arm_closer_to_obj(grasps)

# 3. Batch cuRobo grasp ranking -- sends candidates to freespace_move
#    which runs IK + collision checking for all at once
#    nclass_sorting.py:464-532
selected = select_best_grasp_for_target(grasps[:16], side=chosen_side)
# Returns SelectedGrasp with .trajectory_cache_key for replaying the plan

# 4. Execute: open gripper, move to grasp, close, verify
#    nclass_sorting.py:560-594
open_gripper(chosen_side)
freespace_move(preview_only=False, trajectory_cache_key=selected.trajectory_cache_key)
close_gripper(chosen_side)

# 5. Transport to birds-eye-view, then drop at target
#    nclass_sorting.py:695-717
safe_move(chosen_side, birdseye_pos, transport_rpy)
safe_move(chosen_side, drop_pos, transport_rpy)
open_gripper(chosen_side)
```

**Batch grasp ranking** is a feature of `freespace_move` (`cap/agent/tools/freespace_move.py:1011-1130`):

```python
freespace_move(
    grasp_candidates=list(grasps),  # list of GraspCandidate or dicts
    batch_side="left",              # which arm
    batch_top_k=16,                 # top-K candidates to evaluate
    solver_speed="fast",            # cuRobo solver speed
    batch_validate_trajectory=False, # skip full trajectory validation for speed
)
```

Returns a result with `.best_candidate` (position, rpy, score, width, trajectory_cache_key)
and `.batch_candidates` (all evaluated with IK error and feasibility).

---

## 4. Fallback Strategy: detect_object RPY

Use `detect_object()` RPY when AnyGrasp is unavailable or for simple known objects.

```python
dets = detect_object("orange")
pos = dets[0].position_3d
rpy = dets[0].rpy  # [roll, pitch, yaw] in degrees (display convention)

# Approach from above with detected orientation
freespace_move(left_target_pos=[pos[0], pos[1], pos[2] + 0.10], left_target_rpy=rpy)

# Descend to grasp height (add Z safety offset)
freespace_move(left_target_pos=[pos[0], pos[1], pos[2] + 0.05], left_target_rpy=rpy)

close_gripper("left")
```

---

## 5. Fallback Strategy: Home Orientation

When no detection RPY is available:

```python
state = get_robot_state()
home_rpy = state.left_ee_rpy  # or state.right_ee_rpy

freespace_move(left_target_pos=target_pos, left_target_rpy=home_rpy)
```

Home orientation is approximately `[0, 90, 0]` degrees.

---

## 6. High-Level `grasp()` and `place()` Tools

Defined at `cap/agent/tools/grasp.py:31-303`. These tools encapsulate the
full approach-grasp-lift and move-place-release sequences.

### `grasp(side, position, rpy=None, pre_height=0.10, z_offset=0.05)`

(`cap/agent/tools/grasp.py:31-191`)

1. Open gripper
2. Freespace move to hover position (object Z + z_offset + pre_height)
3. Descend to grasp height (freespace_move or nudge if below table surface)
4. Close gripper
5. Lift to hover

Table surface constant: `_TABLE_Z = 0.75` (`grasp.py:28`).

```python
grasp(side="left", position=[0.5, 0.2, 0.82], rpy=[0, 90, 0], pre_height=0.10, z_offset=0.05)
```

If `rpy` is omitted, the current arm orientation is used.

### `place(side, position, rpy=None, pre_height=0.15)`

(`cap/agent/tools/grasp.py:194-303`)

1. Move to hover above target
2. Descend to place height (position Z + 0.05)
3. Open gripper
4. Lift away

```python
place(side="left", position=[0.4, -0.1, 0.82], pre_height=0.15)
```

---

## 7. Top-Down vs Horizontal Approach (Heuristic Fallback)

For non-AnyGrasp grasps, select approach style by proximity:

### Top-down approach
- Object is close to the arm base in XY (xy_dist < 0.15 m)
- Use `rpy=[0, 180, 0]` for straight-down approach

```python
# Top-down: hover high, then descend
freespace_move(left_target_pos=[pos[0], pos[1], 0.9], left_target_rpy=[0, 180, 0])
nudge("left", delta_pos=[0, 0, -0.15])
close_gripper("left")
```

### Horizontal approach (default)
- Object is far enough from the arm base (xy_dist >= 0.15 m)
- Use home RPY or detected RPY

```python
state = get_robot_state()
ee_rpy = state.left_ee_rpy

freespace_move(left_target_pos=[pos[0], pos[1], pos[2] + 0.05], left_target_rpy=ee_rpy)
close_gripper("left")
```

### Selection rule

```python
import math

state = get_robot_state()
side = "left" if obj_pos[1] >= -0.05 else "right"
ee_pos = state.left_ee_pos if side == "left" else state.right_ee_pos
ee_rpy = state.left_ee_rpy if side == "left" else state.right_ee_rpy

xy_dist = math.sqrt((ee_pos[0] - obj_pos[0])**2 + (ee_pos[1] - obj_pos[1])**2)

if xy_dist < 0.15:
    grasp_rpy = [0, 180, 0]
    freespace_move(**{f"{side}_target_pos": [obj_pos[0], obj_pos[1], 0.9],
                      f"{side}_target_rpy": grasp_rpy})
    nudge(side, delta_pos=[0, 0, -0.15])
else:
    grasp_rpy = ee_rpy
    freespace_move(**{f"{side}_target_pos": [obj_pos[0], obj_pos[1], obj_pos[2] + 0.05],
                      f"{side}_target_rpy": grasp_rpy})

close_gripper(side)
```

---

## 8. AnyGrasp Debug UI

`tools/vision/serve_anygrasp_debug.py` (port 8121) provides a web dashboard that:

- Shows two pose views per grasp: raw AnyGrasp world-frame and planner-aligned
- Renders overlay images with gripper wireframes projected onto RGB
- Supports testing individual grasps with cuRobo IK preview
- Displays per-grasp thumbnails cropped around the object + gripper

The Viser 3D visualizer (`cap/agent/visualizer.py:477-580`) also includes a
"Grasp Planning" panel that calls `sample_grasp_pose_anygrasp` and renders
candidates as coordinate frames with score labels.

### Key constants used by the debug UI

(`tools/vision/serve_anygrasp_debug.py:66-80`):

| Constant | Default | Env var |
|----------|---------|---------|
| Debug port | 8121 | `DEBUG_PORT` |
| AnyGrasp URL | `http://localhost:8122` | `ANYGRASP_URL` or `ANYGRASP_SERVICE_URL` |
| SAM3 URL | `http://localhost:6767` | `SAM3_URL` |
| 2D top-down Z | 0.79 m | `ANYGRASP_2D_TOP_DOWN_Z_M` |

---

## 9. AnyGrasp Runtime Setup

The `cap/utils/anygrasp_runtime.py` module handles:

- Resolving Git LFS binary objects for native `.so` extensions
- Setting up MinkowskiEngine and PointNet2 prebuilt Python roots under `/tmp/anygrasp_sdk_runtime`
- Extracting license files from the provided zip
- Configuring `sys.path` for AnyGrasp SDK imports

```python
from cap.utils.anygrasp_runtime import prepare_anygrasp_runtime, configure_anygrasp_imports

runtime = prepare_anygrasp_runtime(license_zip="license_JalenLu.zip")
configure_anygrasp_imports(runtime)
```

The vendored AnyGrasp SDK lives at `third_party/anygrasp_sdk/` with:
- `grasp_detection/` -- detection-only inference (`gsnet.so`)
- `grasp_tracking/` -- tracking mode (`tracker.so`)
- `dependencies/MinkowskiEngine/` -- sparse convolution backend
- `pointnet2/` -- PointNet++ feature extraction

A warmup script is provided at `tools/vision/warmup_anygrasp.py` to prime the
first inference path (cold-start can take 60-300s).

---

## 10. Tool Registration

All grasp-related tools are registered in `cap/agent/tools/__init__.py:146-272`:

```python
from cap.agent.tools.grasp_anygrasp import SampleGraspPoseAnyGraspTool

registry.register(SampleGraspPoseAnyGraspTool(
    cap_server_host=cap_server_host,
    cap_server_port=srv_port,
    sam3_url=f"http://{s3_host}:{s3_port}",
))
```

The bridge layer at `cap/bridge/agent_bridge.py:102-154` defines the
script-facing function signatures that CAP-generated code calls directly:

| Function | Bridge line | Description |
|----------|-------------|-------------|
| `get_robot_state()` | 102 | Read arm state (positions, RPY, gripper) |
| `freespace_move(...)` | 106 | Collision-free motion planning + execution |
| `nudge(side, delta_pos, delta_rpy)` | 122 | Small delta EE adjustments |
| `open_gripper(side)` | 133 | Open gripper |
| `close_gripper(side)` | 136 | Close gripper |
| `detect_object(query, camera)` | 154 | BundleSDF 6-DOF pose detection |
| `grasp(side, position, rpy, ...)` | 139 (via cap_server) | High-level grasp primitive |
| `place(side, position, rpy, ...)` | 145 (via cap_server) | High-level place primitive |

---

## 11. Common Mistakes

1. **Using quaternions directly** -- always use display RPY in degrees. The display RPY convention is *not* standard Euler XYZ (see Section 1).

2. **Skipping the hover-then-descend pattern** -- for top-down grasps, always hover high first, then descend. The `grasp()` tool handles this automatically.

3. **Not re-detecting before each pick** -- object positions shift after robot moves. Re-call `sample_grasp_pose_anygrasp` or `detect_object` each attempt.

4. **Grasping too high** -- use `z_offset` for small objects. The AnyGrasp planner Z floor (0.80 m) can be too high for flat objects on the table; disable clipping with `disable_planner_z_clipping=True`.

5. **Using `roi_workspace` input mode** -- always prefer `segmented_object_cloud` (the default). The ROI mode sends the broader scene cloud and causes pose ambiguity from neighboring clutter.

6. **Ignoring batch grasp ranking** -- for production picks, always use `freespace_move(grasp_candidates=..., batch_side=...)` to find the kinematically feasible grasp before executing. Direct execution of the highest-score grasp often fails IK.

7. **Not handling cold-start timeout** -- the first AnyGrasp inference after server boot can take 60-300s. Use `tools/vision/warmup_anygrasp.py` or increase `ANYGRASP_PLAN_TIMEOUT_S` (env var, default 300s).

---

## 12. File Reference Index

| File | Description |
|------|-------------|
| `cap/agent/tools/grasp_anygrasp.py` | `sample_grasp_pose_anygrasp` tool (6-DOF AnyGrasp planning) |
| `cap/agent/tools/grasp.py` | `grasp` and `place` high-level primitives |
| `cap/agent/tools/freespace_move.py` | `freespace_move` with batch grasp ranking support |
| `cap/agent/tools/nudge.py` | `nudge` for small delta EE adjustments |
| `cap/agent/tools/__init__.py` | Tool registry (all tool registration) |
| `cap/bridge/agent_bridge.py` | Script-facing function signatures |
| `cap/config.py` | All grasp-related configuration constants |
| `cap/utils/anygrasp_runtime.py` | AnyGrasp SDK runtime preparation |
| `cap/agent/visualizer.py` | Viser 3D grasp visualization |
| `tools/vision/serve_anygrasp.py` | AnyGrasp detection server (port 8122) |
| `tools/vision/serve_anygrasp_debug.py` | AnyGrasp debug UI (port 8121) |
| `tools/vision/warmup_anygrasp.py` | Cold-start warmup script |
| `cap/saved_scripts/table_bussing/v7/nclass_sorting.py` | Latest production pick-and-place script |
| `cap/saved_scripts/rl/test_graspnet_pick.py` | Simple AnyGrasp pick test script |
| `third_party/anygrasp_sdk/` | Vendored AnyGrasp SDK (detection + tracking + deps) |
