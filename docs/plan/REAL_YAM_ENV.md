# Real Bimanual YAM Environment

> **Cross-references**: [CAP_DESIGN](../CAP_DESIGN.md) | [CUROBO_UPDATE_SUMMARY](../CUROBO_UPDATE_SUMMARY.md) | [MULTI_CAMERA_CONFIG](../MULTI_CAMERA_CONFIG.md)

---

## Problem

The CAP framework has two divergent code paths for YAM:

| Mode | How CapServer gets arms/cameras | IK path | Config source |
|------|--------------------------------|---------|---------------|
| **Sim** (`env_name="yam"`) | `SimArmClient` / `SimCameraClient` wrapping `YamEnv` | Env's mink IK | `RobotProfile` from env |
| **Real hardware** (`env_name=None`) | `_ArmClient` (Portal RPC to arm_server) / `_CameraClient` (background threads) directly | Legacy pinocchio in CapServer (~lines 980-1030, 1210-1290) | Hardcoded `cap.config` constants |

This split causes:
- Duplicate IK/FK code (pinocchio in CapServer vs mink in `YamKinematics`)
- Hardcoded `YamKinematics` in `visualizer.py` and `freespace_move.py`
- Real hardware can't use profile-driven features (dynamic arm configs, env-provided FK)
- `_ik_servo` on real hardware uses a completely different solver (pinocchio/Pink) than sim (mink)

## Solution

Create `cap/env/real_bimanual_yam/` implementing `EnvProtocol` + `EefControlProtocol`. CapServer treats real hardware identically to sim — wraps it with `SimArmClient`/`SimCameraClient`, reads `RobotProfile`, uses env-provided FK/IK.

```
Before:  Agent → Tools → CapServer → _ArmClient → arm_server
                                    → _CameraClient → USB cameras
                                    → legacy pinocchio IK

After:   Agent → Tools → CapServer → SimArmClient  → RealYamEnv → FollowerRobotClient → arm_server
                                    → SimCameraClient → RealYamEnv → create_camera (depth=True)
                                                                    → YamKinematics (FK/IK)
```

---

## Two-Phase Rollout

### Phase 1: Env Wrapper (safe, no legacy removal)

RealYamEnv wraps hardware. CapServer loads it as `env_name="yam-real"` → `_sim_mode=True`. The legacy pinocchio path still loads (because `yam_profile().urdf_path` is set) but is unused for FK/IK because:
- FK: `get_arm_observation()` returns `ee_pos`/`ee_quat` → CapServer's control loop reads them at line 1418-1422 **only when `_has_urdf_ik=False`**

**Problem:** `yam_profile()` sets `urdf_path` → CapServer sets `_has_urdf_ik=True` → control loop takes the pinocchio FK path (line 1398), ignoring env-provided EE poses.

**Fix:** `RealYamEnv` provides a **modified profile** with `urdf_path=None` and `q_slice=None` on both arms. This tells CapServer: "don't load pinocchio, read EE poses from env observations."

```python
def _real_yam_profile() -> RobotProfile:
    """Same as yam_profile() but urdf_path=None — env owns FK/IK."""
    profile = yam_profile()
    # Override: no URDF, no q_slice — CapServer delegates FK to env
    left = replace(profile.arms["left"], q_slice=None)
    right = replace(profile.arms["right"], q_slice=None)
    return replace(profile, urdf_path=None, arms={"left": left, "right": right})
```

This means:
- `_has_urdf_ik = False` → no pinocchio loaded
- Control loop reads `ee_pos`/`ee_quat` from `get_arm_observation()` (line 1412-1424)
- `_ik_servo` checks `EefControlProtocol` first (line 2146) → uses env's mink IK
- Joint clamping reads from profile (line 1523-1539)
- Home positions read from profile (line 2974)

### Phase 2: Legacy Cleanup (after Phase 1 proven working)

Remove:
- Legacy pinocchio IK in `cap_server.py` (~lines 980-1030, 1210-1290)
- Hardcoded `YamKinematics` in `visualizer.py` (~lines 75-77, 857)
- Hardcoded `YamKinematics` in `freespace_move.py` (~lines 53, 287, 935)
- `_ArmClient`, `_StubArmClient`, `_CameraClient` classes (replaced by env path)

---

## Validation Target

Must support **all 8 tools** called by `cap/saved_scripts/table_bussing/v7/nclass_sorting.py`:

| Tool | Env methods required |
|------|---------------------|
| `get_robot_state()` | `get_arm_observation(side)` — must return `ee_pos`/`ee_quat` (FK via `YamKinematics`) |
| `freespace_move(...)` | `get_arm_observation`, `command_arm`, `step()` |
| `open_gripper(side)` / `close_gripper(side)` | `command_arm(side, cmd)` |
| `go_home()` | `command_arm` (CapServer reads profile home positions) |
| `vlm_query(...)` | `render_rgb(camera)` for top/left/right |
| `sample_grasp_pose_anygrasp(...)` | `render_rgb`, `render_depth`, `get_camera_intrinsics`, `get_camera_extrinsics` |
| `detect_objects_oneshot(...)` | `render_rgb`, `render_depth`, `get_camera_intrinsics`, `get_camera_extrinsics` |

AnyGrasp and BundleSDF both require **depth + intrinsics + extrinsics** for point cloud construction. This is the critical requirement that rules out `NonBlockingCamera` (which hardcodes `enable_depth=False`).

---

## Design

### Constructor

Wraps existing hardware drivers — no new hardware code needed:

| Component | Source | Notes |
|-----------|--------|-------|
| `FollowerRobotClient` | `robot.yam.yam_real_env` | One per arm. Portal RPC to arm_server. |
| `create_camera(name, enable_depth=True)` | `robot.camera_factory` | Same factory as `_CameraClient` in `cap_server.py`. Provides RGB + depth + intrinsics from RealSense D405 (wrist) and ZED 2i (top). |
| `YamKinematics` | `robot.yam.kinematics` | Mink-based FK/IK using MuJoCo XML. Supports arbitrary-frame FK via `configuration.get_transform_frame_to_world()`. |
| `_real_yam_profile()` | Internal | Same as `yam_profile()` but `urdf_path=None`, `q_slice=None` — tells CapServer to delegate FK/IK to env. |

Each camera runs a background daemon thread (pattern from `_CameraClient._worker`) that continuously reads and caches the latest RGB, depth, and intrinsics.

### EnvProtocol Methods

| Method | Implementation |
|--------|---------------|
| `step()` | No-op — arm_server runs its own 200 Hz control loop. CapServer calls this at 60 Hz; harmless. |
| `get_arm_observation(side)` | `FollowerRobotClient.get_observations()` → `{joint_pos, gripper_pos}` + `YamKinematics.forward_kinematics()` → adds `{ee_pos, ee_quat}`. **Must return EE poses** since `_has_urdf_ik=False`. |
| `command_arm(side, cmd)` | `FollowerRobotClient.command_joint_state(cmd)` — `cmd["pos"]` is shape `(dof+1,)` (6 joints + 1 gripper). Passes through `vel`, `kp`, `kd`, `gripper_vel_limit`, `gripper_torque_limit_nm` from cmd dict. |
| `render_rgb(camera)` | Return cached RGB from background camera thread |
| `render_depth(camera)` | Return cached depth from background camera thread |
| `get_camera_intrinsics(camera)` | Return `[fx, fy, cx, cy]` from camera hardware (cached from `create_camera`) |
| `get_camera_extrinsics(camera)` | See [Camera Extrinsics Detail](#camera-extrinsics-detail) below |
| `close()` | Stop camera threads, disconnect arm clients |

### EefControlProtocol (Required)

**Must be implemented** — without it, `_ik_servo` falls to the pinocchio IK path (line 2203) which is unavailable when `_has_urdf_ik=False`.

| Method | Implementation |
|--------|---------------|
| `compute_eef_action(side, target_pos, target_quat, gripper)` | `YamKinematics.inverse_kinematics(seeded=True)` → `{"pos": concat([joint_pos, gripper])}`. Called once per control tick. Must NOT call `step()`. Uses `seeded=True` for 0.37ms per-tick IK (benchmarked). |

**Seeded IK workflow:**
1. CapServer's control loop calls `compute_eef_action()` every tick (~60 Hz)
2. `YamKinematics.inverse_kinematics(seeded=True)` starts from current joint config
3. Returns joint targets for this tick
4. CapServer calls `command_arm()` → sends to arm_server
5. CapServer calls `step()` → no-op

### Camera Extrinsics Detail

When CapServer runs with an env (`_sim_mode=True`), `get_camera_extrinsics` calls `env.get_camera_extrinsics()` at line 3036-3040. The env must compute extrinsics itself.

**Implementation using `YamKinematics.configuration`:**

`YamKinematics` wraps a `mink.Configuration` which loads the full station MuJoCo XML. It already supports arbitrary-frame FK via:
```python
T = self.configuration.get_transform_frame_to_world(frame_name, frame_type)
pos = T.translation()
rot = T.rotation()  # SO3 object
```

Camera frame computation:
- **Top camera (ZED 2i)**: Fixed mount. Extrinsics from ChArUco calibration stored in `robot/calibrate_cameras_legacy.py`. Alternatively, use FK to the top camera body in MuJoCo XML if the calibration is baked into the model.
- **Wrist cameras (D405)**: FK via `YamKinematics.configuration.get_transform_frame_to_world()` using current joint state. Frame names: `"left_camera_d405"` and `"right_camera_d405"` (defined in `robot/models/station/station.xml` lines 194, 303).

**Frame name resolution** (mirrors CapServer's real-hardware path at line 3044-3048):
```python
_CAM_FRAME_MAP = {
    "top": os.environ.get("CAP_TOP_CAMERA_FRAME", "top_camera_zed2i"),
    "left": os.environ.get("CAP_LEFT_CAMERA_FRAME", "left_camera_d405"),
    "right": os.environ.get("CAP_RIGHT_CAMERA_FRAME", "right_camera_d405"),
}
```

Respects the same env var overrides as the current real-hardware path.

**Return format** must match `cap_server.py:3061-3064`:
```python
{"position": [x, y, z], "rotation": [[...], [...], [...]], "needs_optical_flip": bool}
```

### Why Not `NonBlockingCamera`

`NonBlockingCamera` from `robot/yam/yam_real_env.py` hardcodes `enable_depth=False`. The camera thread pattern is the same — we just call `create_camera(name, enable_depth=True)` instead. This is what CapServer's `_CameraClient` already does for real hardware.

---

## CapServer Interaction — Branch-by-Branch Verification

With `env_name="yam-real"` and `_real_yam_profile()` (`urdf_path=None`):

| CapServer Branch | Line | What Happens | Correct? |
|---|---|---|---|
| Constructor: arms/cameras | 817-845 | `SimArmClient`/`SimCameraClient` wrapping `RealYamEnv` | ✓ |
| Profile reading | 827 | `env._profile` → `_real_yam_profile()` | ✓ |
| DOF from profile | 888-893 | Reads `arm.dof=6` per side | ✓ |
| URDF loading | 950-965 | `urdf_path=None` → `_has_urdf_ik=False` → **no pinocchio** | ✓ |
| FK in control loop | 1398-1424 | `_has_urdf_ik=False` → reads `ee_pos`/`ee_quat` from `get_observations()` | ✓ |
| Joint clamping | 1523-1539 | Profile present → uses `arm.joint_limits_*` | ✓ |
| EefControlProtocol check | 1554-1555 | `isinstance(env, EefControlProtocol)` → True | ✓ |
| `compute_eef_action()` | 1561-1564 | Called per tick for active EE targets | ✓ |
| `step()` | 1586-1587 | Called per tick → no-op | ✓ |
| `_ik_servo` routing | 2146-2147 | `EefControlProtocol` → sets target, control loop drives | ✓ |
| _ik_servo fallback | 2203-2209 | Not reached — EefControlProtocol path taken | ✓ |
| go_home | 2967-2981 | Profile has `home_joint_pos` | ✓ |
| Extrinsics | 3036-3040 | `_sim_mode=True` → calls `env.get_camera_extrinsics()` | ✓ |
| Intrinsics | 3028-3032 | Polymorphic `get_intrinsics()` on `SimCameraClient` | ✓ |
| Collision geoms | 1846-1847 | Returns empty (no sim geometry) — acceptable for real | ✓ |
| TaskProtocol RPCs | 1154-1162 | `RealYamEnv` doesn't implement `TaskProtocol` → not bound | ✓ |
| SceneProtocol RPCs | 1167-1187 | `RealYamEnv` doesn't implement `SceneProtocol` → stubs bound | ✓ |

---

## Files to Create

| File | Purpose |
|------|---------|
| `cap/env/real_bimanual_yam/__init__.py` | Export `RealYamEnv` |
| `cap/env/real_bimanual_yam/env.py` | Main env class: `EnvProtocol` + `EefControlProtocol` |

## Files to Modify

| File | Change |
|------|--------|
| `cap/env/__init__.py` | Add `"yam-real"` case to `create_env()` factory |
| `robot/yam/kinematics.py` | Add `frame_pose(frame_name, frame_type, left_jp, right_jp)` method exposing `configuration.get_transform_frame_to_world()` for camera extrinsics |
| `cap/prompt/embodiment/yam_real.md` | Embodiment spec for LLM prompts (same tools as YAM sim minus scene management) |

## Deferred Cleanup — Phase 2 (After Proven Working)

| File | What to remove |
|------|----------------|
| `cap/server/cap_server.py` | Legacy pinocchio IK path (~lines 980-1030, 1210-1290), `_ArmClient`, `_StubArmClient`, `_CameraClient` classes |
| `cap/agent/visualizer.py` | Hardcoded `YamKinematics` (~lines 75-77, 857) → read from env/profile |
| `cap/agent/tools/freespace_move.py` | Hardcoded `YamKinematics` (~lines 53, 287, 935) → read from env/profile |

## Existing Code to Reuse

| Component | Location | Notes |
|-----------|----------|-------|
| `FollowerRobotClient` | `robot/yam/yam_real_env.py` | Portal RPC. Has `get_observations()`, `command_joint_state()` |
| `create_camera` factory | `robot/camera_factory.py` | Accepts `enable_depth=True`. Returns `RealSenseCamera` or `ZedCamera`. |
| `YamKinematics` | `robot/yam/kinematics.py` | Mink FK/IK. `configuration.get_transform_frame_to_world()` for arbitrary frames. |
| `yam_profile()` | `cap/env/base/profile.py` | Base profile — wrap with `urdf_path=None` override. |
| Camera frame names | `robot/models/station/station.xml` | `"left_camera_d405"` (line 194), `"right_camera_d405"` (line 303), `"top_camera_zed2i"` |
| Camera thread pattern | `cap/server/cap_server.py:_CameraClient._worker` | Background daemon thread reading + caching RGB/depth/intrinsics |

---

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Mink IK ≠ pinocchio IK** — different solvers may produce different joint configs | Medium | Benchmark both on 50 random EE targets; verify max positional error < 1mm. Mink IK is 0.37ms seeded, pinocchio is ~0.5ms. |
| **Portal RPC latency** — FollowerRobotClient adds ~1-10ms per `get_observations()` call in 60 Hz loop (16.6ms budget) | Low | Monitor loop overrun warnings at line 1602 on first deploy. Current `_ArmClient` has identical latency. |
| **Gripper stall detection** — CapServer polls `gripper_pos` from observations (line 2912-2914). RealYamEnv must return actual motor feedback, not commanded value. | Medium | Verify `FollowerRobotClient.get_observations()["gripper_pos"]` returns encoder feedback. |
| **Camera extrinsics mismatch** — mink FK to camera frames vs pinocchio FK may differ slightly | Medium | Compare extrinsics from both paths for all 3 cameras; verify AnyGrasp point clouds align. |
| **Thread safety of `YamKinematics`** — `get_camera_extrinsics()` may be called concurrently with `compute_eef_action()` | Medium | Use a lock around `self.configuration` mutations, or use separate `mink.Configuration` instances for FK and IK. |

---

## Verification

### Phase 1 Gates (must pass before Phase 2)

1. **Unit**: Create `RealYamEnv`, verify `get_arm_observation()` returns `{joint_pos, gripper_pos, ee_pos, ee_quat}`
2. **FK parity**: Compare `YamKinematics.forward_kinematics()` vs pinocchio `_forward_kinematics()` on 20 random joint configs — max error < 0.5mm position, < 0.5° orientation
3. **IK parity**: Compare `YamKinematics.inverse_kinematics(seeded=True)` vs pinocchio `_inverse_kinematics()` on 20 EE targets — verify both converge to same joint config (< 1° per joint)
4. **Camera**: Verify `render_rgb` and `render_depth` return correct shapes/dtypes, intrinsics match hardware readout
5. **Extrinsics**: Compare `env.get_camera_extrinsics()` vs CapServer's pinocchio `_frame_pose()` for all 3 cameras — max position error < 2mm
6. **Integration**: Launch `cap_server.py --env yam-real`, verify `get_robot_state()`, camera image/depth/intrinsics/extrinsics all return valid data
7. **_ik_servo**: Verify `_ik_servo` uses `EefControlProtocol` path (not pinocchio fallback)
8. **Gripper**: Verify `set_gripper` + stall detection works (open, close with object, check timeout)
9. **End-to-end**: Run `nclass_sorting.py` through the agent pipeline with `--env yam-real`
10. **Requires**: Physical YAM hardware online (arm_server running, cameras connected)
