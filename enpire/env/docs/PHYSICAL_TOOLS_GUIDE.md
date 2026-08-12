# CAP Physical Tools — Guide (perception · planning · contact)

> Companion to [`CAP_DESIGN.md`](CAP_DESIGN.md) (full tool API + ports).
> This doc focuses on three things that bite people: which tools secretly use
> **ground-truth** (sim-only), the **camera/visual-input path + formats**
> (incl. BundleSDF I/O), and the **compliant gripper close** — plus a
> real, end-to-end example.

The tool layer (`cap/agent/tools/`) is the bridge in **Agent → Tools → Server →
Env**. A tool call fetches RGB-D + state from **cap_server** (Portal RPC, `8300`)
or the in-process env, talks to a model server (HTTP) or the cuRobo planner
(Portal), transforms the result into the **world frame** (+X fwd, +Y left, +Z up,
origin at base_link), and returns a typed dataclass.

---

## 1. Ground-truth (sim-only) vs real-transferable tools

Some perception "tools" return **simulator ground-truth** and therefore **cannot
run on real hardware** — they must be clearly separated from sensor-based tools.

### ❌ Ground-truth / oracle — SIM ONLY (not real-transferable)
| Tool / call | Where | Why it's GT |
|---|---|---|
| `detect_object(..., backend="oracle")` | `detection.py:322` `_execute_oracle` | reads `client.get_object_positions()` — simulator body poses, then fuzzy-matches the query name |
| `get_object_positions()` / `get_oracle_targets()` | cap_server / sim | raw GT scene state |

> Autoresearch evals deliberately strip oracle tools from generated scripts (`runtime_role="script"`, `CAP_DISABLE_TASK_INFO_IN_SCRIPT=1`) so vision-only policies can't cheat. An oracle script must run with `runtime_role="agent"`.

### ✅ Sensor-based — real-transferable
`detect_object(backend="bundlesdf")`, `detect_objects_oneshot`,
`track_object`/`get_object_pose`/`stop_tracking` (BundleSDF), `segment_object`
(SAM3), `sample_grasp_pose_anygrasp` (AnyGrasp), `vlm_query`, `list_scene_objects`.
These consume camera RGB-D only, so the same code runs in sim and on real YAM.

**Rule of thumb:** if a call needs `backend="oracle"`, `get_task_info`, or any
`get_object_positions`/oracle helper, it is **sim-only** — keep it out of code
meant to transfer to real.

---

## 2. The camera / visual-input path and formats

### Files in the path
| File | Role |
|---|---|
| `cap/agent/tools/detection.py` | `_capture_snapshot` (`:130`) — the fetch + encode point |
| `cap/server/cap_server.py` | Portal RPC surface: `get_camera_image` (`:3019`), `get_camera_depth` (`:3024`), `get_camera_intrinsics` (`:3030`), `get_camera_extrinsics` (`:3036`) |
| `cap/env/adapters/sim.py` | `SimCameraAdapter`/`SimCameraClient` — sim frames to cap_server |
| `robot/camera_factory.py`, `robot/station_profiles.py` | real RealSense/ZED backends + per-station camera config |
| `cap/utils/image.py` | shared image helpers |

### Raw visual-input contract (what the tools receive)
`_capture_snapshot` returns, via either the in-process env (`env.render_rgb/…`)
or Portal RPC (`client.get_camera_image(camera).result()`):

| Field | Type / shape | Notes |
|---|---|---|
| `rgb` | `np.ndarray (H, W, 3) uint8` | RGB; cameras named `top` / `left` / `right` (`wrist` in sim) |
| `depth` | `np.ndarray (H, W) float32` | metres |
| `intrinsics` | `list[float] = [fx, fy, cx, cy]` | pinhole |
| `extrinsics` | `dict {position:[x,y,z], rotation:[9 floats row-major 3×3], needs_optical_flip:bool}` | cam→world |

### Wire encoding for HTTP model servers (`detection.py:20-48`)
- `rgb` → PNG → base64 → **`image_base64`**
- `depth` → `np.save` (float32 `.npy` bytes) → base64 → **`depth_base64`**
- `intrinsics` → `[fx, fy, cx, cy]`
- `extrinsics` → `{position[3], rotation[9], needs_optical_flip}` (optical flip
  **pre-applied** so it arrives in OpenCV convention, `_jsonify_extrinsics`)

### BundleSDF server I/O (`cap/skills/serve_bundlesdf.py`)
| Endpoint | Input (JSON) | Output (JSON) |
|---|---|---|
| `POST /add_detection` | `{text, camera, name?, image_base64, depth_base64, intrinsics[fx,fy,cx,cy], extrinsics{position,rotation}}` | starts a SAM3+SAM2+BundleSDF session; `{bbox[x,y,w,h], first_score, …}` |
| `POST /push_frame` | `{camera, image_base64, depth_base64, intrinsics, extrinsics}` | feeds a frame (direct-env mode) |
| `GET /get_detection/{name}` | — | `{tracking:bool, score, bbox[x,y,w,h], ob_in_cam:4×4, position_3d[x,y,z], position_3d_source, quaternion_xyzw[4], rpy[3]}` |
| `POST /single_frame_pose` | like `/add_detection` | one-shot `{bbox, score, ob_in_cam:4×4, position_3d, quaternion_xyzw, …}` (no tracking session) |
| `POST /segment` (SAM3) | `{text, image_base64, intrinsics?}` | `{mask_b64 (np.save), bbox_xywh, score}` |

`ob_in_cam` is the raw **camera-frame** 4×4 pose. The tool side
(`object_tracking.py:get_object_pose`) re-derives the **world-frame** pose using
cap_server extrinsics as the source of truth, then returns a `Detection3D`
(`position_3d`, `quaternion_xyzw`, `rpy`, `half_extents`). SAM3 masks are
`np.save`'d uint8/bool arrays, base64-encoded.

---

## 3. Compliant (force-limited) gripper close

`set_gripper(side, pos, vel_limit=None, torque_limit=None)` — `pos` 1.0 open /
0.0 closed. When **`torque_limit` is set**, `cap_server.set_gripper`
(`cap_server.py:2876`) runs a settle loop with **stall detection**: if the
gripper stops moving for `GRIPPER_TORQUE_LIMIT_HOLD_S`, it concludes it has
clamped on an object and **returns early** instead of forcing the position
target. That is the *compliant close* — grip firmly without crushing; `vel_limit`
also slows the close for gentleness. (`open_gripper`/`close_gripper` are thin
wrappers, `native.py`.)

The internal GPU handover script previously used this interface, but that
script is not distributed in the open-source release. Released tasks can use
the same `vel_limit` and `torque_limit` arguments without relying on that
internal example.

---

## 4. End-to-end example (real-transferable: detect → plan → grasp → insert)

```python
# 1. PERCEIVE — sensor-based only (no oracle), so this runs in sim AND on real YAM
det   = detect_object("red block", backend="bundlesdf")[0]   # 6-DOF world pose (Detection3D)
cands = sample_grasp_pose_anygrasp("red block")              # ranked GraspCandidates, planner-aligned

# 2. PLAN — batch cuRobo IK+motion-gen over candidates, rank without moving
ranked = freespace_move(grasp_candidates=cands, batch_side="right", preview_only=True)
best   = ranked.best_candidate                               # executable, highest score

# 3. APPROACH — collision-free move to the chosen grasp (RPY in degrees, world frame)
freespace_move(right_target_pos=best.position, right_target_rpy=best.rpy)

# 4. GRASP — compliant, force-limited close (won't crush)
close_gripper("right", vel_limit=2.0, torque_limit=0.4)

# 5. MOVE to insertion hover, then fine compliant descent via small world-frame deltas
freespace_move(right_target_pos=hover_above_slot, right_target_rpy=insert_rpy)
nudge("right", delta_pos=[0, 0, -0.01])                     # 1 cm down into the slot

# 6. VERIFY — read state back (real-transferable); use get_task_info() ONLY in sim
state = get_robot_state()
```

Flow: perception servers (BundleSDF/SAM3/AnyGrasp over HTTP) → world-frame
targets → cuRobo (Portal RPC, GPU) plans/ranks → cap_server (Portal RPC) executes
trajectories + compliant gripper → state read back. Swap step 1 for
`detect_object(backend="oracle")` only when in sim and you explicitly want GT.

---

## 5. Where to look
`cap/agent/tools/base.py` (result dataclasses) · `detection.py` (oracle vs
bundlesdf, `_capture_snapshot`) · `object_tracking.py` (cam→world) ·
`segmentation.py` · `freespace_move.py` (cuRobo, RPY convention `:710`, batch
`:1154`) · `grasp_anygrasp.py` · `native.py` + `cap_server.py:2876` (compliant
close) · `cap/skills/serve_bundlesdf.py` / `serve_sam3.py` (server I/O).
