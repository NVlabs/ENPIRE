# BundleSDF — Multi-Object 6-DOF Pose Tracking

> **Last updated**: 2026-04-08
> **Source of truth**: `tools/vision/serve_bundlesdf.py`, `cap/agent/tools/object_tracking.py`

## Overview

BundleSDF is a multi-object 6-DOF pose tracking system integrated into the CAP framework. It combines three neural network services -- SAM3 (text-prompted segmentation, running as a **separate server**), SAM2 (video mask propagation), and BundleTrack (pose optimization) -- to provide real-time object pose tracking from natural language descriptions.

`detect_object` now uses BundleSDF as its default backend (`backend="bundlesdf"`). It supports two backends: `bundlesdf` for 6-DOF tracking and `oracle` for simulation ground-truth poses. BundleSDF tracks **full 6-DOF pose** (position + orientation) continuously across frames, enabling tasks that require orientation-aware manipulation (pouring, insertion, handover).

**Cross-references**: [CAP_DESIGN.md](CAP_DESIGN.md) (system architecture), [TABLE_BUSSING_SKILLS.md](TABLE_BUSSING_SKILLS.md) (manipulation tools that consume poses), [SAFETY_ZONE_DESIGN.md](SAFETY_ZONE_DESIGN.md) (EE safety zones for RL exploration), [CAP_UI_DESIGN.md](CAP_UI_DESIGN.md) (UI detection debug panel).

## Architecture

```
Camera Feed (Portal RPC from cap_server:8300)
    |
    |  (or POST /push_frame for standalone mode)
    v
tools/vision/serve_bundlesdf.py (FastAPI :8119)
    |
    |-- _CameraTrackingLoop (one per camera)   [:579-900]
    |     |-- SharedSam2Tracker: SAM2.1 video mask propagation (shared model)
    |     |-- _ObjectState (one per tracked object)   [:474-575]
    |     |     |-- BundleSdf: 6-DOF pose tracker (C++ via pybind)
    |     |     |-- ReferenceModel: OBB center + half-extents builder
    |     |     |-- Occlusion state machine
    |     |     +-- MJPEG visualization buffer
    |     +-- ThreadPoolExecutor(max_workers=8): parallel BundleTrack per object
    |
    +-- HTTP Endpoints
          |-- GET  /health                -> liveness check
          |-- POST /push_frame            -> push RGB+depth (standalone mode)
          |-- POST /add_detection         -> SAM3 detect + SAM2 anchor + start tracking
          |-- GET  /get_detection/{name}  -> latest 6-DOF pose (camera + world frame)
          |-- POST /end_detection/{name}  -> stop tracking, free GPU
          |-- GET  /list_detections       -> all active sessions with poses
          |-- POST /single_frame_pose     -> one-shot pose, no persistent session
          |-- POST /segment               -> SAM3-only segmentation (no pose)
          |-- POST /reset_state           -> end all detections, clear buffers
          +-- MJPEG streams: /preview, /stream/{name}, /stream_composite

tools/vision/serve_sam3.py (FastAPI :6767)          <-- SEPARATE PROCESS
    |-- POST /segment        -> SAM3 text-prompted segmentation
    +-- serve_bundlesdf calls this over HTTP (not in-process)

CAP Agent Tools (HTTP clients -> serve_bundlesdf)
    |
    |-- High-level (preferred):                         [cap/agent/tools/object_tracking.py]
    |     |-- track_object    -> start tracking
    |     |-- get_object_pose -> poll pose (Detection3D, cam->world done tool-side)
    |     +-- stop_tracking   -> stop tracking
    |
    |-- Low-level (direct HTTP wrappers):               [cap/agent/tools/bundlesdf_track.py]
    |     |-- add_detection   -> POST /add_detection
    |     |-- get_detection   -> GET  /get_detection/{name}
    |     |-- end_detection   -> POST /end_detection/{name}
    |     +-- list_detections -> GET  /list_detections
    |
    |-- Detection (auto-start/poll):                    [cap/agent/tools/detection.py]
    |     |-- detect_object          -> BundleSDF tracking or oracle backend
    |     |-- detect_objects_oneshot -> one-shot pose, shared snapshot, no session
    |     +-- detect_object_realtime -> continuous detection loop (blocks until stop)
    |
    +-- Segmentation:                                   [cap/agent/tools/segmentation.py]
          +-- segment_object -> SAM3-only mask via serve_sam3
```

## Model Pipeline

### Phase 1: Single-Image Detection (SAM3) -- External Server

Text-prompted instance segmentation on the initial frame. **Runs in a separate process** (`tools/vision/serve_sam3.py` on port **6767**). `serve_bundlesdf` calls it over HTTP.

- **Input**: RGB frame (base64) + natural language text (e.g. "yellow mustard bottle")
- **Model**: Facebook SAM3 (Segment Anything 3), loaded via HuggingFace `transformers`
- **Output**: binary mask (0/1 uint8), bounding box `(x, y, w, h)`, confidence score
- **Memory**: ~1.2 GB VRAM on the SAM3 server process
- **Server**: `tools/vision/serve_sam3.py:1` (FastAPI on port 6767)

Called during `add_detection` (twice -- initial detect, then fresh-frame re-detect for accurate bbox) and during occlusion recovery. The HTTP client in serve_bundlesdf is at `serve_bundlesdf.py:88-119`.

`serve_bundlesdf` also supports in-process SAM3 via `bundlesdf/run_live_bundlesdf.py:text_to_mask()` (`run_live_bundlesdf.py:422-490`), but the external server is the default deployment.

### Phase 2: Video Mask Propagation (SAM2)

Tracks all objects on a camera in a single shared forward pass.

- **Model**: Facebook SAM2.1 (Hiera-Large) via HuggingFace `transformers` (`Sam2VideoProcessor`, `Sam2VideoModel`)
- **Precision**: `torch.bfloat16`
- **Memory**: ~900 MB VRAM (shared singleton, loaded once across all cameras)
- **Architecture**: ONE `SharedSam2Tracker` instance per camera; objects added/removed dynamically
- **Session refresh**: every 100 frames (`_SAM2_REFRESH_INTERVAL`), recreates session to clear accumulated per-frame KV-cache state
- **Thread safety**: `_model_lock` serialises all model.forward() and session mutation calls
- **Source**: `third_party/bundlesdf/run_live_bundlesdf.py:73-312` (`SharedSam2Tracker` class)

Key methods on `SharedSam2Tracker`:
- `add_object(rgb, bbox_xywh) -> obj_id` -- initialize tracking for a new object (`run_live_bundlesdf.py:143`)
- `propagate(rgb) -> {obj_id: (mask_255, score)}` -- single forward pass returns masks + scores for all active objects (`run_live_bundlesdf.py:173`)
- `re_anchor_object(rgb, obj_id, bbox_xywh)` -- re-anchor one object with a fresh bbox prompt (`run_live_bundlesdf.py:187`)
- `deactivate_object(obj_id)` -- exclude from output (stays in KV cache; no per-object removal API) (`run_live_bundlesdf.py:201`)
- `refresh_session(rgb, bboxes)` -- recreate session, re-anchor all active objects (`run_live_bundlesdf.py:258`)

Additionally, `Sam3Tracker` (`run_live_bundlesdf.py:316-412`) provides streaming SAM3 video segmentation for single-object tracking scenarios (not used in the default BundleSDF pipeline but available).

### Phase 3: 6-DOF Pose Estimation (BundleTrack)

Per-object C++ tracker that estimates SE(3) pose from masked RGB-D frames.

- **Engine**: BundleTrack (NVLabs), NeRF/GUI stripped, tracking-only
- **Source**: `third_party/bundlesdf/bundlesdf.py:96` (`BundleSdf` class)
- **Features**: LoFTR feature matching (~45 MB, singleton in `loftr_wrapper.py:20` `LoftrRunner`)
- **Algorithm**: feature correlation + depth-to-points + RANSAC + bundle adjustment over keyframe window
- **Output**: `ob_in_cam` (4x4 SE(3) in camera frame) -- returned by `BundleSdf.run()` at `bundlesdf.py:366`
- **Reference model**: After warm-up (`reference_model.py:59`, default 12 frames — `BundleSdf` overrides `ReferenceModel`'s default of 20 via `BUNDLESDF_CENTER_WARMUP_FRAMES` env var at `bundlesdf.py:130`), builds a denoised OBB point cloud for stable center/half-extents output

Robustness:
- Mask erosion (configurable kernel, default 3x3) removes SAM2 boundary artifacts (`serve_bundlesdf.py:669`)
- Depth validation via `has_valid_depth()`: requires 30+ valid pixels within mask, depth < `zfar` (3m) (`run_live_bundlesdf.py:576`)
- Per-object depth copy prevents cross-object corruption (in-place percentile denoise) (`serve_bundlesdf.py:681`)
- CUDA OOM caught and recovered (skip frame, empty cache) (`serve_bundlesdf.py:691`)
- GPU RANSAC guard: skips pairs with < 5 correspondences to avoid CUDA async failures (`bundlesdf.py:223`)

### Phase 4: 3D Position Refinement (ReferenceModel)

After the BundleTrack warm-up period, the `ReferenceModel` builds a denoised 3D point cloud of the tracked object and computes its oriented bounding box (OBB).

- **Source**: `third_party/bundlesdf/reference_model.py:1-154`
- **Warm-up**: 20 frames by default (configurable via `BUNDLESDF_CENTER_WARMUP_FRAMES` env var, read at `bundlesdf.py:131`)
- **Process**: Accumulates masked depth observations in object frame, voxel-downsamples (3mm), removes statistical outliers, computes OBB via Open3D
- **Output**:
  - `center_local` -- OBB center in object frame, used as the `position_3d` for stable centroid reporting
  - `half_extents` -- OBB half-sizes in meters, returned in the API response
- **Fallback**: Before the reference model is ready, `estimate_center_cam()` uses the masked-depth median as center (`bundlesdf.py:153-174`)

## CAP Agent Tool API

### Tool Tiers

The system provides three tiers of tools, all registered in `cap/agent/tools/__init__.py:86-273`:

#### Tier 1: High-Level Tracking (preferred for LLM-generated code)

Source: `cap/agent/tools/object_tracking.py:1-338`

These three tools share a `_TrackingContext` that tracks the active camera/query/session-name across calls.

##### track_object

Start real-time 6-DOF tracking. Internally stops any previous session, then calls `POST /add_detection`. Waits 1s for tracker convergence before returning.

```python
track_object("yellow mustard bottle", camera="top")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Natural-language object description |
| `camera` | str | `"top"` | Camera: `top`, `left`, or `right` |

Source: `object_tracking.py:96-189` (`TrackObjectTool`)

##### get_object_pose

Read the latest tracked 6-DOF pose. Internally calls `GET /get_detection/{name}`, then performs the **camera-to-world transform tool-side** using cap_server's Pinocchio FK extrinsics as the single source of truth. This guarantees the returned pose is in the robot world frame regardless of server-side extrinsic issues.

```python
pose = get_object_pose()
# Returns Detection3D:
#   .position_3d     [x, y, z] in world frame (meters)
#   .quaternion_xyzw [x, y, z, w] quaternion
#   .score           tracking confidence (0-1)
#   .half_extents    OBB half-sizes (meters), populated after warm-up
```

Coordinate frame contract (documented at `object_tracking.py:6-14`):
- `+X` forward (toward the work table)
- `+Y` left (toward the left arm)
- `+Z` up (sky)
- Origin: URDF `base_link` (floor level, centred between the two arm bases)

Camera convention handling: D405 cameras need an optical flip (`diag(-1,-1,1)`) applied via `needs_optical_flip()` from `robot/models/station/paths.py`; ZED 2i cameras do not. This logic is in `object_tracking.py:61-79` (`_build_cam_to_world`).

Source: `object_tracking.py:192-297` (`GetObjectPoseTool`)

##### stop_tracking

Stop the active tracking session and free GPU memory.

```python
stop_tracking()
```

Source: `object_tracking.py:299-338` (`StopTrackingTool`)

#### Tier 2: Low-Level Multi-Session Tools

Source: `cap/agent/tools/bundlesdf_track.py:1-219`

Direct HTTP wrappers for the multi-session BundleSDF API. Useful when tracking multiple objects simultaneously and managing sessions independently.

| Tool | HTTP Endpoint | Description |
|------|---------------|-------------|
| `add_detection` | `POST /add_detection` | Start tracking; returns session name |
| `get_detection` | `GET /get_detection/{name}` | Latest pose as `Detection3D` |
| `end_detection` | `POST /end_detection/{name}` | Stop one session, free GPU |
| `list_detections` | `GET /list_detections` | All active sessions with poses |

Note: `get_detection` returns the server-side `position_3d` and `quaternion_xyzw` directly (no tool-side cam-to-world re-computation like Tier 1).

#### Tier 3: Detection & One-Shot Tools

Source: `cap/agent/tools/detection.py:1-594`

##### detect_object

High-level detection with auto-start/poll and retry logic. Supports two backends:

```python
# BundleSDF backend (default) — starts tracking session, polls until pose ready
dets = detect_object("red cup", camera="top", backend="bundlesdf")

# Oracle backend (sim-only) — ground-truth pose from MuJoCo
dets = detect_object("red_block", backend="oracle")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Text description of the object |
| `camera` | str | `"top"` | Camera: `top`, `left`, or `right` |
| `backend` | str | `"bundlesdf"` | `"bundlesdf"` or `"oracle"` (sim-only) |
| `max_retries` | int | `3` | Retry attempts if detection fails |

BundleSDF backend behavior (`detection.py:313-448`):
- Auto-starts a tracking session if the query/camera changed
- Stops any previously active session first
- Polls `GET /get_detection/{name}` with 2s intervals for up to 60s (cold start) or 10s (warm)
- Requires both `position_3d` and `ob_in_cam` to be present (rejects depth-invalid poses)
- Reports diagnostic errors for reflective/shiny objects that SAM3 tracks but depth cannot reach

Oracle backend behavior (`detection.py:251-304`):
- Calls `cap_server.get_object_positions()` Portal RPC
- Fuzzy-matches query against MuJoCo scene body names (case-insensitive substring)
- Returns ground-truth position, quaternion, and geom half-extents

Source: `detection.py:39-448` (`DetectObjectTool`)

##### detect_objects_oneshot

One-shot BundleSDF pose inference on a shared snapshot. Never starts a real-time tracking session. Accepts a single query or a list of queries. Captures one RGB-D snapshot and runs `POST /single_frame_pose` for each query.

```python
results = detect_objects_oneshot(["red cup", "blue plate"], camera="top")
# Returns dict mapping each query to its Detection3D list
```

Source: `detection.py:450-533` (`DetectObjectsOneshotTool`)

##### detect_object_realtime

Continuously detects objects and updates visualization until stopped by Home/Stop/E-Stop. Blocks the executor thread.

```python
detect_object_realtime("red cup", camera="top")  # blocks until stop_event
```

Source: `detection.py:536-594` (`DetectObjectRealtimeTool`)

### Related Tools

- **segment_object** (`cap/agent/tools/segmentation.py`): SAM3-only text-prompted segmentation via `tools/vision/serve_sam3.py`. Returns binary mask, bbox, score. No pose estimation.
- **list_scene_objects** (`cap/agent/tools/scene_objects.py`): Uses Qwen3-VL to enumerate all objects visible in the scene. Returns a list of object names suitable for feeding into `detect_object`.

## Detection3D Data Type

All detection/tracking tools return `Detection3D` (defined at `cap/agent/tools/base.py:97-106`):

```python
@dataclass
class Detection3D:
    label: str                          # echoes the query text
    score: float                        # confidence (0-1)
    box_2d: list[float]                 # [x1, y1, x2, y2] pixels (may be empty)
    position_3d: list[float]            # [x, y, z] in world frame (meters)
    quaternion_xyzw: list[float] = []   # [x, y, z, w] quaternion
    rpy: list[float] = []              # [roll, pitch, yaw] in degrees
    half_extents: list[float] = []     # OBB half-sizes (meters)
```

**position_3d semantics**: After the ReferenceModel warms up (~20 frames), `position_3d` is the OBB center projected to world frame (stable, centered on the object). Before warm-up, it falls back to the masked-depth median. The raw SE(3) pose origin (may not be on the object centroid) is available as `pose_origin_3d` in the HTTP API response.

## Tracking State Management

### Per-Object State (`_ObjectState`)

Source: `serve_bundlesdf.py:474-575`

Each tracked object maintains:

| Field | Type | Description |
|-------|------|-------------|
| `ob_in_cam` | 4x4 ndarray | Latest SE(3) pose in camera frame |
| `ob_in_world` | 4x4 ndarray | Latest SE(3) pose in world frame (`T_cam_world @ ob_in_cam`) |
| `position_3d_world` | 3-vec ndarray | OBB center or masked-depth median in world frame |
| `pose_origin_3d_world` | 3-vec ndarray | Raw SE(3) translation in world frame |
| `position_3d_source` | str | `"reference_model_obb"` or `"masked_depth_median"` |
| `bbox` | list[int] | Latest SAM2 2D bbox `[x, y, w, h]` |
| `score` | float | Tracking confidence (0-1) |
| `frame_idx` | int | Latest processed frame number |
| `_bad_streak` | int | Consecutive frames with bad mask/score |
| `_occluded` | bool | Whether in occlusion recovery mode |
| `_sam3_check_frame` | int | Next frame index for SAM3 re-detection |
| `tracker` | BundleSdf | Per-object C++ tracker instance |
| `_tracker_lock` | Lock | Serialises `tracker.run()` vs `stop()` |

### Per-Camera Tracking Loop (`_CameraTrackingLoop`)

Source: `serve_bundlesdf.py:579-900`

One background thread per camera (daemon thread, `_loop()` at `:740`):

1. **Acquire frame**: push buffer (if standalone) first, Portal RPC fallback. Frame deduplication via sequence number (push) or identity check (portal). (`:750-773`)
2. **SAM2 propagate**: Run shared `SAM2.propagate()` once -- masks for all objects in one GPU call. (`:783-791`)
3. **Get extrinsics**: push buffer first, Portal RPC fallback. Build `T_cam_world` via `_build_SE3()`. (`:794-802`)
4. **Per-object BundleTrack** (parallel via `ThreadPoolExecutor`, single-object path skips pool overhead): (`:807-831`)
   - Check occlusion state machine
   - Erode mask, validate depth via `has_valid_depth()`
   - Copy depth (prevents cross-object percentile denoise corruption)
   - Run `BundleSdf.run()` -> `ob_in_cam`
   - Compute `ob_in_world = T_cam_world @ ob_in_cam`
   - Compute `position_3d` via `estimate_center_cam()` (OBB center or masked-depth median)
5. **Push visualization frame** for each object (`:844-849`)
6. **Every 100 frames**: refresh SAM2 session to clear accumulated state (`:852-881`)

### Occlusion Recovery

Source: `serve_bundlesdf.py:628-667` (inside `_run_one_object`)

When `score < score_thresh` or `mask_area < 200px`:
- Increment `_bad_streak`, skip BundleSdf to avoid poisoning pose graph
- After `_OCCLUSION_STREAK_THRESH` bad frames: mark `_occluded = True`
- Every `_SAM3_CHECK_INTERVAL` (45) frames while occluded: re-detect with SAM3
- If found: `re_anchor_object()` on SAM2, clear occlusion state, resume tracking

Default `_OCCLUSION_STREAK_THRESH = 999999` (effectively disabled; set to ~10 to enable).

Constants at `serve_bundlesdf.py:625-626`.

## HTTP Server Endpoints

Server: `tools/vision/serve_bundlesdf.py` on port **8119** (configurable via `--port`).
Created by `create_app()` at `serve_bundlesdf.py:1134`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check. Returns `{"status": "ok"}`. |
| `/` | GET | HTML management UI (inline at `:904-1129`). |
| `/preview` | GET | MJPEG stream of raw camera frames. |
| `/stream/{name}` | GET | MJPEG stream for specific detection (mask overlay + pose axes). |
| `/stream_composite` | GET | Tiled MJPEG view of all active detections. Falls back to raw preview when no sessions active. |
| `/push_frame` | POST | Push RGB+depth frame for a camera (standalone mode). Body: `PushFrameRequest` (`:129-134`). Returns `{ok, camera, seq}`. |
| `/add_detection` | POST | Start tracking (`:1294-1401`). Body: `AddDetectionRequest` (`:143-153`). Runs SAM3 detect (twice: initial + fresh-frame re-detect), loads BundleSdf, anchors SAM2. Returns `{name, bbox, first_score}`. **409** if session already active. |
| `/get_detection/{name}` | GET | Latest pose (`:1403-1413`). Returns `PoseResponse` (`:201-214`) with `tracking, ob_in_cam, ob_in_world, position_3d, pose_origin_3d, position_3d_source, quaternion_xyzw, rpy, half_extents, bbox, score, frame_idx`. **404** if not found. |
| `/end_detection/{name}` | POST | Stop tracking, free GPU (`:1415-1431`). Tears down camera loop if no objects remain. Returns `{ok}`. |
| `/list_detections` | GET | All active sessions with poses (`:1433-1461`). Returns `{detections: {name: DetectionEntry}}`. |
| `/single_frame_pose` | POST | One-shot pose estimation (`:1489-1615`). No persistent session. Body: `SingleFramePoseRequest` (`:178-188`). Loads a temporary BundleSdf, runs one frame, returns pose, then cleans up. |
| `/segment` | POST | SAM3-only segmentation (`:1617-1654`). Body: `{text, camera?, image_base64?}`. Returns `{mask_b64, bbox_xywh, score, mask_area, height, width}`. |
| `/reset_state` | POST | End all active detections, clear frame buffers, reset Portal client (`:1463-1487`). Returns `{ok, ended_detections, retained_camera_loops}`. |

### Standalone Mode (no cap_server)

BundleSDF can operate without cap_server by pushing frames via HTTP:

```bash
# 1. Start serve_sam3 (SAM3 segmentation server)
uv run python tools/vision/serve_sam3.py

# 2. Start serve_bundlesdf (no cap_server needed)
uv run python tools/vision/serve_bundlesdf.py --sam3_url http://localhost:6767

# 3. Push frames from any source (Python example)
import base64, io, requests
import numpy as np
from PIL import Image

rgb = ...  # HxWx3 uint8 numpy array
depth = ...  # HxW float32 numpy array (meters)

# Encode
buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG")
rgb_b64 = base64.b64encode(buf.getvalue()).decode()
buf = io.BytesIO(); np.save(buf, depth.astype(np.float32))
depth_b64 = base64.b64encode(buf.getvalue()).decode()

# Push frame + camera params (intrinsics/extrinsics only needed once)
requests.post("http://localhost:8119/push_frame", json={
    "camera": "top",
    "image_base64": rgb_b64,
    "depth_base64": depth_b64,
    "intrinsics": [fx, fy, cx, cy],
    "extrinsics": {"position": [x, y, z], "rotation": [r00, r01, ..., r22]},
})

# 4. Start tracking (can also pass image_base64 + intrinsics inline)
requests.post("http://localhost:8119/add_detection", json={"text": "blue plate"})

# 5. Continue pushing frames — tracking loop reads from buffer
```

Frame/intrinsics/extrinsics resolution priority (applies to all endpoints): **request body > push buffer > Portal RPC (cap_server)**.

## Configuration

### BundleTrack Parameters

Generated by `build_configs()` at `run_live_bundlesdf.py:596-629`. Written to a temporary YAML file per session.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth_processing.zfar` | 3.0 | Max tracking distance (meters) |
| `depth_processing.percentile` | 95 | Depth outlier filtering |
| `erode_mask` | 3 | Morphological erosion kernel size |
| `bundle.max_BA_frames` | 7 | Bundle adjustment window |
| `bundle.max_optimized_feature_loss` | 0.03 | BA optimisation loss threshold |
| `feature_corres.resize` | 320 | Feature map resolution |
| `feature_corres.max_dist_neighbor` | 0.02 | Feature correspondence distance threshold |
| `feature_corres.map_points` | true | Map points across keyframes |
| `keyframe.min_rot` | 5 | Min rotation (degrees) to add keyframe |
| `ransac.inlier_dist` | 0.01 | RANSAC inlier distance (meters) |
| `ransac.inlier_normal_angle` | 20 | RANSAC inlier normal angle (degrees) |
| `ransac.max_trans_neighbor` | 0.02 | Max translation between neighbors |
| `ransac.max_rot_deg_neighbor` | 30 | Max rotation between neighbors (degrees) |
| `p2p.max_dist` | 0.02 | Point-to-point ICP max distance |
| `p2p.max_normal_angle` | 45 | Point-to-point ICP max normal angle |

### ReferenceModel Parameters

Set in `reference_model.py:32-37` and `bundlesdf.py:130-131`:

| Parameter | Default | Env Var | Description |
|-----------|---------|---------|-------------|
| `voxel_size` | 0.003 | -- | Voxel downsample resolution (meters) |
| `warmup_frames` | 20 | `BUNDLESDF_CENTER_WARMUP_FRAMES` (default 12 in BundleSdf) | Frames before model freezes |
| `min_model_points` | 200 | -- | Minimum points for a valid model |

### Ports

Defined in `cap/config.py:27-43`:

| Service | Port | Protocol | Config Constant |
|---------|------|----------|-----------------|
| serve_bundlesdf | 8119 | HTTP | `BUNDLESDF_SERVER_PORT` |
| serve_sam3 | 9500 (cap/config.py default) / 6767 (serve_sam3 CLI default) | HTTP | `SAM3_SERVER_PORT` |
| cap_server (camera source) | 8300 | Portal RPC | `CAP_SERVER_PORT` |
| serve_pose (OWLv2, legacy) | 8118 | HTTP | `DETECTION_SERVER_PORT` |

### Camera Names

Resolved from the active station profile at `cap/config.py:213-222`. Default: `top`, `left`, `right`.

### Session Key Convention

All tools and the server agree on session keys via `cap.config.make_bundlesdf_name()` (`cap/config.py:55-62`):
URL-safe, lowercase, truncated to 80 chars. If you use the same object text, you get the same key everywhere.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Set at `serve_bundlesdf.py:45` |
| `BUNDLESDF_CENTER_WARMUP_FRAMES` | `12` | Frames before OBB reference model freezes |
| `BUNDLESDF_RUNTIME_LIB_DIR` | -- | Override path for native .so libs |
| `BUNDLESDF_SAM2_LOAD_RETRIES` | `3` | Max SAM2 model load attempts |
| `BUNDLESDF_SAM2_LOCAL_FILES_ONLY` | `1` | Use local HF cache only (no download) |
| `SAM3_SERVER_HOST` | `localhost` | SAM3 server host |
| `SAM3_SERVER_PORT` | `9500` (cap/config.py) / `6767` (serve_sam3.py CLI) | SAM3 server port — table-bussing launchers override to `6767` |
| `HF_HUB_ETAG_TIMEOUT` | `1200` | HuggingFace Hub download timeout |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `1200` | HuggingFace Hub download timeout |

## Performance

| Component | Latency | Notes |
|-----------|---------|-------|
| SAM3 initial detect (remote) | 2-10 s | First call loads model; subsequent ~2-5 s |
| SAM2 propagate (shared) | 50-100 ms | GPU inference, all objects in one call |
| BundleTrack per object | 30-200 ms | C++ GPU tracking, parallelized |
| **Total (1 object)** | **85-325 ms** | **3-11 fps** |
| **Total (3 objects)** | **165-525 ms** | **2-6 fps** |

| Model | VRAM | Process |
|-------|------|---------|
| LoFTR (preloaded) | ~45 MB | serve_bundlesdf |
| SAM2 (preloaded) | ~900 MB | serve_bundlesdf |
| SAM3 | ~1.2 GB | serve_sam3 (separate process) |
| BundleTrack (per object) | ~200-300 MB | serve_bundlesdf |
| **serve_bundlesdf idle** | **~1 GB** | |
| **serve_bundlesdf + 1 object** | **~1.3 GB** | |
| **serve_sam3** | **~1.2 GB** | separate process |

## Startup & Model Preloading

At startup (`serve_bundlesdf.py:1676-1708`), `serve_bundlesdf` preloads and warms up models:

1. **LoFTR**: `LoftrRunner()` loads weights from `BundleTrack/LoFTR/weights/outdoor_ds.ckpt` to CUDA (~45 MB)
2. **SAM2**: `SharedSam2Tracker.preload()` loads `facebook/sam2.1-hiera-large` to CUDA (~900 MB)
3. **SAM3**: Delegated to the external `serve_sam3` process -- not loaded by serve_bundlesdf
4. **CUDA warmup**: Runs dummy LoFTR inference and SAM2 add_object+propagate to JIT-compile kernels (important for Blackwell/sm_120 GPUs)

## cap_server Changes

The `--no-arms` flag on `cap_server.py` enables camera-only operation:

```bash
uv run cap/server/cap_server.py --no-arms
```

This uses `_StubArmClient` (returns zeros, no-op commands) so the server can run without arm hardware -- useful when only cameras are needed for detection development. Also adds a stub `get_object_positions` binding in non-sim mode to prevent Portal worker crashes.

See [CAP_DESIGN.md](CAP_DESIGN.md) for full cap_server documentation.

## Setup

### BundleTrack Native Libraries (Git LFS)

The BundleTrack C++ tracker depends on shared libraries in `third_party/bundlesdf/libs/` (OpenCV, PCL, Boost, etc.). These are stored via **Git LFS** and **not downloaded by default** (configured in `.lfsconfig`).

The `__init__.py` bootstrap (`third_party/bundlesdf/__init__.py:1-212`) handles:
1. Resolving Git LFS pointer files / broken symlinks to real LFS objects
2. Preloading shared libraries in dependency order via `ctypes.CDLL(..., RTLD_GLOBAL)`
3. Preloading `libpython` to satisfy pybind11 dependencies
4. Setting `LD_LIBRARY_PATH` for the `my_cpp` pybind module

On the robot machine (or any machine that will run BundleSDF):

```bash
# One-time: fetch the .so files (~346 MB)
git lfs pull --include="third_party/bundlesdf/libs/*"

# Verify
ls third_party/bundlesdf/libs/*.so | head -5  # should see real binaries, not pointer files
```

If the files are LFS pointers (small text files starting with `version https://git-lfs.github.com/spec/v1`), run the `git lfs pull` command above.

**Note:** These libraries are Linux x86_64 binaries. They are not needed on macOS dev machines -- only on the robot or GPU server that runs BundleSDF. See [lfs_setup.md](lfs_setup.md) for general LFS configuration.

## Running

### Startup

```bash
# 1. Start cap_server (camera source)
uv run cap/server/cap_server.py --no-arms  # or with arms

# 2. Start SAM3 segmentation server (separate process, port 6767)
uv run python tools/vision/serve_sam3.py --preload

# 3. Start BundleSDF tracking server (port 8119)
uv run python tools/vision/serve_bundlesdf.py --sam3_url http://localhost:6767
# Preloads LoFTR + SAM2 at startup, warms up CUDA kernels
# Ready when: "Uvicorn running on http://0.0.0.0:8119"

# 4. Start cap_agent (optional, for tool API)
uv run cap/agent/cap_agent.py
```

Environment:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### CLI Arguments (serve_bundlesdf)

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `8119` | HTTP port |
| `--host` | `0.0.0.0` | Bind address |
| `--cap_server_host` | `localhost` | cap_server host |
| `--cap_server_port` | `8300` | cap_server Portal RPC port |
| `--camera` | `top` | Default camera name |
| `--sam3_url` | `http://localhost:6767` | External SAM3 server URL |

### Which services do I need?

| Task | Required |
|------|----------|
| BundleSDF tracking only | cap_server + serve_sam3 + serve_bundlesdf |
| BundleSDF via CAP tools | cap_server + serve_sam3 + serve_bundlesdf + cap_agent |
| Standalone (no cap_server) | serve_sam3 + serve_bundlesdf (push frames via HTTP) |
| Full CAP + BundleSDF | All services from [CAP_DESIGN.md](CAP_DESIGN.md) + serve_sam3 + serve_bundlesdf |

### SSH Port Forwarding

Add ports 8119 and 6767 to your tunnel if accessing from a different machine:

```bash
ssh -N -L 8119:127.0.0.1:8119 -L 6767:127.0.0.1:6767 ... <robot-host>
```

See [remote_serving.md](remote_serving.md) for remote GPU serving configuration.

## File Structure

```
lecar-tbd/
|-- tools/vision/
|   |-- serve_bundlesdf.py                 # Main tracking server (FastAPI, ~1720 lines)
|   +-- serve_sam3.py                      # SAM3 segmentation server (FastAPI, port 6767)
|-- cap/
|   |-- config.py                          # BUNDLESDF_SERVER_PORT, make_bundlesdf_name(), etc.
|   |-- agent/tools/
|   |   |-- __init__.py                    # create_default_registry() — registers all tools
|   |   |-- base.py                        # Detection3D dataclass (line 97)
|   |   |-- object_tracking.py             # TrackObjectTool, GetObjectPoseTool, StopTrackingTool
|   |   |-- bundlesdf_track.py             # AddDetection, GetDetection, EndDetection, ListDetections
|   |   |-- detection.py                   # DetectObjectTool (bundlesdf + oracle backends),
|   |   |                                  #   DetectObjectsOneshotTool, DetectObjectRealtimeTool
|   |   |-- segmentation.py               # SegmentObjectTool (SAM3-only, via serve_sam3)
|   |   +-- scene_objects.py               # ListSceneObjectsTool (Qwen3-VL)
|   |-- prompt/tools/
|   |   |-- detect_object.md               # LLM prompt for detect_object usage
|   |   +-- bundlesdf_track.md             # LLM prompt for bundlesdf_track usage
|   |-- skills/
|   |   +-- serve_bundlesdf.py             # Skill server variant (alternative entry point)
|   |-- ui/src/components/
|   |   +-- DetectionDebug.tsx             # UI debug panel for detection 3D bounding boxes
|   +-- server/
|       +-- cap_server.py                  # --no-arms flag, _StubArmClient
+-- third_party/bundlesdf/
    |-- __init__.py                        # LFS resolution, LD_LIBRARY_PATH, .so preloading
    |-- bundlesdf.py                       # BundleSdf tracker class (C++ pybind wrapper)
    |-- run_live_bundlesdf.py              # SharedSam2Tracker, Sam3Tracker, text_to_mask,
    |                                      #   build_configs, has_valid_depth, draw_mask_overlay
    |-- reference_model.py                 # ReferenceModel: OBB point cloud builder
    |-- loftr_wrapper.py                   # LoftrRunner singleton (LoFTR feature matcher)
    |-- pose_utils.py                      # PoseSmoother (SE(3) EMA)
    |-- offscreen_renderer.py              # pyrender/pytinyrenderer offscreen rendering
    |-- render_compare.py                  # Render-and-compare utilities
    |-- tool.py                            # Utility functions
    |-- Utils.py                           # Geometry + point-cloud helpers (depth2xyzmap, etc.)
    |-- BundleTrack/                       # C++ tracker (pybind via my_cpp)
    |   |-- config_ho3d.yml                # Default BundleTrack config template
    |   |-- LoFTR/                         # LoFTR feature matching submodule
    |   |   +-- weights/outdoor_ds.ckpt    # Pretrained LoFTR weights
    |   +-- scripts/data_reader.py         # Data I/O utilities
    +-- libs/                              # Bundled shared libraries (OpenCV, PCL, etc.)
        |-- libBundleTrack.so              # Core C++ tracker
        |-- libMY_CUDA_LIB.so             # CUDA kernels
        +-- my_cpp.cpython-311-x86_64-linux-gnu.so  # pybind11 module
```

## Comparison: Detection Tools

| | `detect_object(backend="bundlesdf")` | `detect_object(backend="oracle")` | `detect_objects_oneshot` | `track_object` + `get_object_pose` |
|---|---|---|---|---|
| **Output** | Full 6-DOF pose | Ground-truth pose | Full 6-DOF pose | Full 6-DOF pose |
| **Tracking** | Auto-start session | None (sim query) | One-shot, no session | Persistent session |
| **Multi-object** | One at a time (auto-stops previous) | Multiple matches | Batch (shared snapshot) | One at a time |
| **Cam->world** | Server-side | N/A (world frame) | Server-side | Tool-side (authoritative) |
| **Retry/poll** | Built-in (60s cold, 10s warm) | Immediate | Built-in (3 attempts) | Manual poll |
| **Sim support** | No | Yes | No | No |
| **Latency** | 2-60s first call, ~2s warm | ~10ms | 2-10s per query | 1s startup, then instant |
| **Server** | serve_bundlesdf:8119 | cap_server:8300 | serve_bundlesdf:8119 | serve_bundlesdf:8119 |

## Example Usage

### Track object with high-level tools (preferred)

```python
# Start tracking a bottle on the top camera
track_object("yellow mustard bottle", camera="top")

import time
time.sleep(2)  # let tracker converge

# Get 6-DOF pose (world frame, cam->world transform done tool-side)
pose = get_object_pose()
pos = pose.position_3d      # [x, y, z] in world frame (OBB center after warmup)
quat = pose.quaternion_xyzw  # [x, y, z, w] quaternion

# Use pose for manipulation
freespace_move(left_target_pos=[pos[0], pos[1], pos[2] + 0.10])
nudge("left", delta_pos=[0, 0, -0.05])  # descend to grasp height
close_gripper("left")

# When done
stop_tracking()
```

### One-shot multi-object detection

```python
# Detect multiple objects from one shared snapshot (no persistent sessions)
results = detect_objects_oneshot(["red cup", "blue plate", "yellow bottle"], camera="top")
for query, dets in results.items():
    if dets:
        print(f"{query}: pos={dets[0].position_3d}")
```

### Auto-managed detection (simplest)

```python
# detect_object auto-starts/polls BundleSDF internally
dets = detect_object("red cup", camera="top")
pos = dets[0].position_3d
rpy = dets[0].rpy
half = dets[0].half_extents  # OBB half-sizes after warmup
```

### Simulation oracle

```python
# In sim mode, use ground-truth from MuJoCo
dets = detect_object("red_block", backend="oracle")
pos = dets[0].position_3d      # exact MuJoCo position
quat = dets[0].quaternion_xyzw  # exact MuJoCo quaternion
size = dets[0].half_extents     # geom half-extents
```
