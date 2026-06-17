# Design Doc: Task-Aware EE Safety Zones

> **Last verified against code**: 2026-04-08
>
> **Related docs**:
> - [`docs/RL_PIPELINE_DESIGN.md`](RL_PIPELINE_DESIGN.md) — RL training pipeline (`learn_skill`, diagnostics, timing)
> - [`docs/CAP_DESIGN.md`](CAP_DESIGN.md) — CAP system architecture and layer stack (safety overview in "Safety" section)
> - [`docs/CAP_UI_DESIGN.md`](CAP_UI_DESIGN.md) — CAP web UI (Viser 3D viewer embedded as iframe)

## Goal

Restrict end-effector exploration during online RL (`learn_skill`) to a task-relevant region. For example, during stick insertion, the arm holding the stick should only explore near the hover pose and insertion hole — not wander across the entire workspace.

The safety zone is **defined by the LLM agent** via tool calls before launching RL, not hardcoded in config. This lets the agent inspect the scene (detect objects, read robot state) and set zones that adapt to each specific task instance.

---

## Architecture

```
LLM Agent (code editor / Claude)
  │
  │  set_safety_zone("left", keyposes=[...], pos_margin=0.08, ori_margin=0.3)
  │  learn_skill("insertion", {...})
  │  clear_safety_zone()
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  cap_agent (port 8200)                                  │
│    cap/agent/cap_agent.py                               │
│    ├─ SetSafetyZoneTool  ──Portal RPC──►  cap_server    │
│    ├─ ClearSafetyZoneTool                               │
│    ├─ GetSafetyZoneTool                                 │
│    └─ Viser visualizer (polls zone config at ~3Hz)      │
│       cap/agent/visualizer.py:987                       │
└─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────┐
│  cap_server (port 8300)                                 │
│    cap/server/cap_server.py                             │
│    SafetyChecker (cap/server/safety.py:269)             │
│      ├─ E-stop (existing)                               │
│      └─ TaskSafetyZone                                  │
│           ├─ left: ArmSafetyZone                        │
│           └─ right: ArmSafetyZone                       │
│                                                         │
│    learn_skill loop (POLICY_FREQ_HZ = 30Hz):            │
│      cap/server/cap_server.py:2776                      │
│      1. RL policy → proposed joint targets              │
│      2. Fello takeover handling                         │
│      3. ★ _enforce_safety_zone()                        │
│         cap/server/cap_server.py:3701                   │
│         a. Clamp joints to URDF limits for FK           │
│         b. FK proposed joints → EE pose (pinocchio)     │
│         c. Check position & orientation vs zone         │
│         d. If violated: interpolate back in joint space │
│      4. Write safe targets to CONTROL_FREQ_HZ loop      │
│         cap/server/cap_server.py:3190                   │
│                                                         │
│    CONTROL_FREQ_HZ control loop (60Hz):                 │
│      cap/server/cap_server.py:1240                      │
│      ★ Joint limit clamping before hardware command     │
│        (catches all sources: RL, Fello, _ik_servo)      │
└─────────────────────────────────────────────────────────┘
```

---

## Core Concept: Convex Hull + Margin + Elastic Boundary

The safe zone for each arm is defined by **keyposes** — task-relevant EE poses (7D: position + quaternion). The safe exploration region is:

```
Safe region = ConvexHull(keypose positions) ⊕ Sphere(pos_margin)
```

This is the Minkowski sum of the convex hull and a sphere. In practice, for 2-4 keyposes this creates shapes like:

| Keyposes | Shape | Example |
|----------|-------|---------|
| 1 | Sphere | Single grasp target |
| 2 | Capsule | Hover + insertion target |
| 3 | Triangular prism | Multi-waypoint approach |
| 4+ | 3D convex volume | Complex task regions |

### Why Not a Naive AABB

An axis-aligned bounding box wastes exploration budget. For a vertical insertion with hover pose at `(0.3, 0.1, 0.35)` and target at `(0.3, 0.1, 0.15)`, the AABB includes all of `(0.3±m, 0.1±m, 0.15 to 0.35)` — a box. The convex hull + margin creates a tight capsule along the vertical axis, excluding the irrelevant corners.

### Three-Zone Enforcement

```
┌─────────────────────────────────┐
│  Interior (inside hull + margin)│  factor = 1.0  (full freedom)
├─────────────────────────────────┤
│  Elastic band                   │  factor ∈ (0, 1)  (attenuated)
│  (margin → margin + band)       │
├─────────────────────────────────┤
│  Beyond hard boundary           │  factor = 0.0  (hold position)
└─────────────────────────────────┘
```

- **Interior**: No modification. Policy has full freedom.
- **Elastic band**: Actions pointing outward are progressively attenuated. The scale factor decreases linearly from 1.0 to 0.0 across the band width. This creates a smooth, learnable constraint — the policy receives gradient signal at the boundary.
- **Hard cutoff**: Clamp to current position. Absolute safety limit.

The elastic band prevents the discontinuity that a hard clamp would create, which destabilises RL training (same observation → different effective actions when clamped).

Implementation: `ArmSafetyZone.compute_scale_factor()` at `cap/server/safety.py:201`. The factor is `min(pos_factor, ori_factor)` — the stricter of position and orientation wins.

### Orientation Constraint

At each EE position, the "reference" orientation is the quaternion of the nearest keypose (`cap/server/safety.py:187`). The angular distance between the proposed orientation and reference must be within `ori_margin` (radians). The same elastic/hard-clamp logic applies to orientation violations.

Angular distance uses `2 * arccos(|q1 · q2|)` (`cap/server/safety.py:84`) — handles quaternion double-cover via `abs()`.

This prevents dangerous EE rotations even when position is safe — critical for tasks like insertion where the stick must remain vertical.

### Joint-Space Interpolation

When the scale factor `f < 1.0`, enforcement acts in joint space (`cap/server/cap_server.py:3749`):

```python
safe_jp = current_cmd_jp + f * (proposed_jp - current_cmd_jp)
```

No IK is needed — we interpolate between the current (known safe) commanded position and the proposed position. This is fast (no optimisation loop) and natural (the robot moves a fraction of the way toward the proposed target).

The gripper command is **not** interpolated — it passes through unmodified since gripper state is independent of EE safety zone (`cap/server/cap_server.py:3747`).

### Joint Limit Clamping

The `CONTROL_FREQ_HZ` (60Hz) control loop clamps all joint commands to URDF limits before sending to hardware (`cap/server/cap_server.py:1240`):

```python
ljp_cmd = np.clip(ljp_cmd, JOINT_LIMITS_LOW[:6], JOINT_LIMITS_HIGH[:6])
rjp_cmd = np.clip(rjp_cmd, JOINT_LIMITS_LOW[6:], JOINT_LIMITS_HIGH[6:])
lgp_cmd = np.clip(lgp_cmd, GRIPPER_MIN, GRIPPER_MAX)
rgp_cmd = np.clip(rgp_cmd, GRIPPER_MIN, GRIPPER_MAX)
```

Joint limits are defined in `cap/config.py:161-170` (from `station.xml` actuator ctrlrange). Gripper range is `[0.0, 1.0]` (`cap/config.py:173-174`).

This is a hard safety floor that catches all sources (RL policy, Fello takeover, _ik_servo, learn_skill) regardless of whether a safety zone is set. The `_enforce_safety_zone` FK call also clamps joints before pinocchio FK to avoid limit-violation exceptions (`cap/server/cap_server.py:3723`):

```python
q_lo = self._pink_model.lowerPositionLimit
q_hi = self._pink_model.upperPositionLimit
left_jp_fk = np.clip(left_jp, q_lo[:6], q_hi[:6])
right_jp_fk = np.clip(right_jp, q_lo[8:14], q_hi[8:14])
```

Note these use pinocchio's URDF limits (`_pink_model.lowerPositionLimit`) rather than the config `JOINT_LIMITS_LOW/HIGH`. Both derive from the same URDF, but the pinocchio model includes the 2 extra fixed joints (indices 6-7) between left and right arm, hence the `[:6]` / `[8:14]` indexing.

---

## Tool API

Three tools are exposed to the LLM agent via the tool registry (`cap/agent/tools/__init__.py:231`). All three inherit from `_PortalMixin` (`cap/agent/tools/safety.py:22`) which creates a **fresh Portal client per RPC call** to avoid the autoconn race condition in Portal's `Future.set_result`.

### `set_safety_zone(side, keyposes, pos_margin=0.08, ori_margin=0.3)`

**Tool class**: `SetSafetyZoneTool` (`cap/agent/tools/safety.py:50`)
**RPC endpoint**: `CapServer.set_safety_zone()` (`cap/server/cap_server.py:3664`)

Set the safety zone for one arm. Zones are cumulative — calling for "left" doesn't affect "right". Zones persist across `learn_skill` episodes until cleared.

The RPC endpoint creates an `ArmSafetyZone` with `elastic_band = pos_margin / 2` and `elastic_band_ori = ori_margin / 2` (hardcoded at `cap/server/cap_server.py:3673-3674`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `side` | `str` | `"left"` or `"right"` |
| `keyposes` | `list[list[float]]` | List of 7D arrays `[x, y, z, qx, qy, qz, qw]` |
| `pos_margin` | `float` | Metres around convex hull (default 0.08 = 8cm) |
| `ori_margin` | `float` | Radians max angular deviation (default 0.3 ~ 17deg) |

### `clear_safety_zone(side=None)`

**Tool class**: `ClearSafetyZoneTool` (`cap/agent/tools/safety.py:108`)
**RPC endpoint**: `CapServer.clear_safety_zone()` (`cap/server/cap_server.py:3682`)

Clear zones. No args = clear both. Pass `"left"` or `"right"` for one arm.

Portal quirk: empty strings are not passed (zero-length buffer fails Portal's `SendBuffer` assertion) — the tool sends no argument to mean "both" (`cap/agent/tools/safety.py:131`), and the RPC endpoint treats an empty string as `None` (`cap/server/cap_server.py:3685`).

### `get_safety_zone()`

**Tool class**: `GetSafetyZoneTool` (`cap/agent/tools/safety.py:141`)
**RPC endpoint**: `CapServer.get_safety_zone()` (`cap/server/cap_server.py:3691`)

Returns the current zone config dict via `SafetyChecker.get_zone_config()` (`cap/server/safety.py:332`). Returns `{"active": False}` if no zone is set, otherwise includes per-arm keyposes, margins, and elastic band widths.

### Example: Stick Insertion

When defining keyposes for a tool-holding task (e.g. stick insertion), the keyposes must account for the tool extending below the gripper. The lowest EE keypose should be `plate_z + STICK_BELOW_GRASP`, not the plate surface itself — otherwise the safe zone allows the gripper to crash into the table.

```python
STICK_BELOW_GRASP = 0.08  # ~8cm of stick below gripper grasp point

# Detect task geometry
state = get_robot_state()
plate_pos, _ = detect_obj("blue_plate")
hover_quat = Rotation.from_euler("ZX", [yaw, np.pi]).as_quat().tolist()

# Compute keyposes — account for stick length below gripper
regrasp_offset = GRASP_OFFSET * Rotation.from_quat(hover_quat).apply([0, -1, 0])
hover_ee = plate_pos + [0, 0, 0.15] + regrasp_offset

insert_ee = plate_pos + regrasp_offset
insert_ee[2] = plate_pos[2] + STICK_BELOW_GRASP  # EE stays above plate

# Set zone — 2 keyposes define vertical range (hover → insert)
set_safety_zone("left", keyposes=[
    list(hover_ee) + hover_quat,     # pre-insertion hover
    list(insert_ee) + hover_quat,    # insertion target (stick tip at plate)
], pos_margin=0.08, ori_margin=0.3)

# Run RL — enforcement is automatic inside learn_skill
for ep in range(100):
    freespace_move("left", hover_ee, hover_quat, max_vel=0.2)
    learn_skill("insertion", {"max_steps": 100, "control_mode": "left"})
    # Re-detect plate and update zone each episode
    plate_pos, _ = detect_obj("blue_plate")
    insert_ee = plate_pos + regrasp_offset
    insert_ee[2] = plate_pos[2] + STICK_BELOW_GRASP
    set_safety_zone("left", keyposes=[
        list(hover_ee) + hover_quat,
        list(insert_ee) + hover_quat,
    ], pos_margin=0.08, ori_margin=0.3)

clear_safety_zone()
```

See also: `cap/saved_scripts/rl/claude_plate_and_stick_learn_insert_safe.py` for a full working example.

---

## Enforcement Pipeline Detail

The `_enforce_safety_zone()` method (`cap/server/cap_server.py:3701`) runs inside the `learn_skill` loop at `POLICY_FREQ_HZ` (30Hz), after Fello takeover handling and before writing targets to the control loop (`cap/server/cap_server.py:3116`).

### Step-by-step

1. **Early exit**: If no task zone is set (`SafetyChecker.has_task_zone()` returns `False`), return immediately with no modification.

2. **Clamp for FK**: Clamp proposed joints to pinocchio URDF limits (`_pink_model.lowerPositionLimit` / `upperPositionLimit`) to prevent FK exceptions.

3. **Per-arm loop** (left, right):
   a. Run pinocchio FK via `_frame_pose()` (`cap/server/cap_server.py:1023`) on the `"left_grasp"` / `"right_grasp"` frame to get the proposed EE pose (SE3 → position + quaternion).
   b. Call `SafetyChecker.enforce_ee()` (`cap/server/safety.py:350`) which delegates to `ArmSafetyZone.compute_scale_factor()` (`cap/server/safety.py:201`).
   c. If `factor < 1.0`: read current commanded joint positions under `_state_lock`, then interpolate: `safe_jp = cur_jp + factor * (jp - cur_jp)`.
   d. Replace proposed joint positions with safe ones. Gripper passes through unmodified.

4. **Logging**: First 5 calls log unconditionally, then every 100th call (`cap/server/cap_server.py:3757`). Log prefix: `[safety_enforce]`.

5. **Diagnostics**: If any zone triggered (elastic or hard clamp), emit a `safety_enforced` UDP event (`cap/server/cap_server.py:3128`) with episode, step, side, and full enforcement info dict.

6. **Crash resilience**: The entire method is wrapped in try/except — on any exception it logs and returns the original unmodified targets (`cap/server/cap_server.py:3777`).

---

## Viser 3D Visualization

The safety zone is rendered in the Viser 3D viewer (port 8080) using boxes. Viser's `add_icosphere` and `add_line_segments` crash the Three.js WebGL renderer, so only `add_box` is used.

### Visualization Toggle

A "Show Safety Zone" checkbox in the Viser GUI sidebar controls visibility (`cap/agent/visualizer.py:93`). Initially unchecked (`initial_value=False`). Toggling on re-renders from cached config; toggling off removes handles but preserves cached config so re-enabling is instant.

### Per-Arm Elements

| Element | Geometry | Color | Opacity | Scene path | Code |
|---------|----------|-------|---------|------------|------|
| Keypose markers | Small solid box (2cm) per keypose | Yellow `(255,220,0)` | 1.0 | `/safety/{side}/keypose_{i}` | `visualizer.py:1051` |
| Safe zone | One box = AABB of keyposes + pos_margin | Cyan `(0,180,255)` | 0.12 | `/safety/{side}/safe` | `visualizer.py:1067` |
| Hard boundary | One box = AABB of keyposes + pos_margin + elastic_band | Red `(255,80,80)` | 0.06 | `/safety/{side}/hard` | `visualizer.py:1094` |

The safe zone and hard boundary boxes use `side="back"` (back-face only rendering) so they never occlude objects inside them — the robot and scene objects are always fully visible.

The AABB is computed per-axis from the keypose positions (`visualizer.py:1043-1044`), so for a vertical insertion task the Z extent is tight to the actual hover→insertion range rather than a uniform cube.

### Polling Mechanism

The `cap_agent.py` broadcast loop (`cap/agent/cap_agent.py:2403`) runs at ~30Hz. Safety zone config is polled every 10th iteration (~3Hz) via `_fetch_safety_zone()` (`cap/agent/cap_agent.py:2354`), which calls `cap_server.get_safety_zone()` over Portal RPC. The visualizer caches the config dict and only re-renders when it changes (`visualizer.py:999`).

Can be disabled via the `CAP_SAFETY_VIZ_ENABLED` environment variable (`cap/agent/cap_agent.py:67`):
```bash
CAP_SAFETY_VIZ_ENABLED=0  # or "false", "no", "off"
```

### Known Viser Limitations

- `add_icosphere` crashes Three.js: `Cannot assign to read only property 'scale'`
- `add_line_segments` crashes Three.js: `Cannot read properties of undefined (reading 'count')`
- Both cause `THREE.WebGLRenderer: Context Lost` — use `add_box` only

---

## Distance Computation

The `ArmSafetyZone.distance_to_hull()` method (`cap/server/safety.py:145`) computes exact distance from a point to the convex hull of keypose positions, handling all cases:

| N keyposes | Geometry | Method | Code |
|-----------|----------|--------|------|
| 1 | Point | Euclidean distance | `safety.py:153` |
| 2 | Line segment | Point-to-segment projection (`_dist_point_to_segment`) | `safety.py:157` |
| 3 | Triangle | Barycentric projection onto plane, fallback to edges (`_dist_point_to_triangle`) | `safety.py:160` |
| 4+ | 3D convex hull | `scipy.spatial.Delaunay` for inside check, face-distance for outside | `safety.py:163-178` |

Degenerate cases (collinear 3+ points, coplanar 4+ points) fall back gracefully to edge-based distances, which are conservative (slightly overestimate distance = slightly tighter zone = safer).

For 4+ keyposes, `ConvexHull` and `Delaunay` are precomputed in `__post_init__` (`safety.py:131-141`). If scipy raises (degenerate point set), the fallback is pairwise edge distances (`safety.py:176-178`).

Signed distance: `distance_to_hull(pos) - pos_margin` (`safety.py:183`). Negative means inside the safe zone.

---

## Files

### Core implementation

| File | Purpose | Key classes/functions |
|------|---------|----------------------|
| `cap/server/safety.py` | Safety zone math and thread-safe checker | `ArmSafetyZone` (`:94`), `TaskSafetyZone` (`:258`), `SafetyChecker` (`:269`) |
| `cap/server/cap_server.py` | RPC endpoints and enforcement in learn_skill loop | `set_safety_zone` (`:3664`), `clear_safety_zone` (`:3682`), `get_safety_zone` (`:3691`), `_enforce_safety_zone` (`:3701`) |
| `cap/agent/tools/safety.py` | LLM-facing tool wrappers (Portal RPC) | `SetSafetyZoneTool` (`:50`), `ClearSafetyZoneTool` (`:108`), `GetSafetyZoneTool` (`:141`), `_PortalMixin` (`:22`) |
| `cap/agent/tools/__init__.py` | Tool registration in `create_default_registry()` | Registration at `:231-233` |
| `cap/agent/visualizer.py` | Viser 3D box rendering of zones | `update_safety_zone` (`:987`), `clear_safety_zone` (`:1129`), `_clear_safety_handles` (`:1119`) |
| `cap/agent/cap_agent.py` | Polling loop and env var toggle | `_fetch_safety_zone` (`:2354`), `SAFETY_VIZ_ENABLED` (`:67`) |
| `cap/config.py` | Joint limits, frequencies, gripper range | `JOINT_LIMITS_LOW/HIGH` (`:161-169`), `CONTROL_FREQ_HZ` (`:107`), `POLICY_FREQ_HZ` (`:110`), `GRIPPER_MIN/MAX` (`:173-174`) |

### Example scripts

| File | Description |
|------|-------------|
| `cap/saved_scripts/rl/claude_plate_and_stick_learn_insert_safe.py` | Stick insertion with safety zones |
| `cap/saved_scripts/rl/claude_plate_and_stick_learn_insert_safe_regrasp.py` | Stick insertion with regrasp and safety zones |
| `cap/saved_scripts/rl/claude_test_safety_zone.py` | Safety zone test script |

---

## Configuration Defaults

| Parameter | Default | Derived from | Code |
|-----------|---------|--------------|------|
| `pos_margin` | 0.08m (8cm) | Set by LLM | `safety.py:117` |
| `ori_margin` | 0.3 rad (~17deg) | Set by LLM | `safety.py:118` |
| `elastic_band` | `pos_margin / 2` | Auto (cap_server.py:3673) | `safety.py:119` |
| `elastic_band_ori` | `ori_margin / 2` | Auto (cap_server.py:3674) | `safety.py:120` |

The elastic band widths are automatically set to half the margin. This means:
- Position: 8cm safe → 4cm elastic → hard clamp at 12cm from hull
- Orientation: 17deg safe → 8.5deg elastic → hard clamp at 25.5deg from reference

---

## Diagnostics

When enforcement triggers, the cap_server emits a `safety_enforced` UDP event (via `cap/diag/emitter.py:31`) with:

```python
emit("cap_server", "safety_enforced", ep=episode, step=steps,
     side="left",
     pos_signed_dist=0.02,      # how far outside pos boundary
     ori_violation=-0.1,        # negative = ori is fine
     pos_in_elastic=True,       # in the elastic attenuation band
     pos_hard_clamp=False,
     ori_in_elastic=False,
     ori_hard_clamp=False)
```

Emit call location: `cap/server/cap_server.py:3128-3135`. These events are generic msgpack-over-UDP packets — any UDP listener on the diagnostics port can consume them.

### Log prefixes

| Prefix | Location | What it logs |
|--------|----------|-------------|
| `[safety_enforce]` | `cap/server/cap_server.py:3763` | Enforcement calls (first 5, then every 100th) |
| `[safety_viz]` | `cap/agent/visualizer.py` (multiple) | Zone rendering, config changes, handle counts |
| `[Safety]` | `cap/server/safety.py:305,314,322` | Zone set/clear events (via SafetyChecker) |
| `[broadcast]` | `cap/agent/cap_agent.py:2471` | Polling failures in broadcast loop |

---

## Threading Model

The `SafetyChecker` class (`cap/server/safety.py:269`) is thread-safe. All zone access goes through a `threading.Lock` (`_lock`). The three threads that interact with it:

1. **Portal RPC thread** — calls `set_arm_zone()`, `clear_task_zone()`, `get_zone_config()` when the LLM agent invokes tools.
2. **learn_skill loop thread** — calls `has_task_zone()` and `enforce_ee()` at 30Hz during RL training.
3. **CONTROL_FREQ_HZ loop thread** — calls `is_estopped()` at 60Hz (e-stop only, not zone enforcement).

Zone enforcement itself (`_enforce_safety_zone` in cap_server) also reads `_cmd_left_jp` / `_cmd_right_jp` under `_state_lock` to get the current commanded position for interpolation (`cap/server/cap_server.py:3741-3745`).

---

## Frequency Reference

| Loop | Rate | Config | File:Line |
|------|------|--------|-----------|
| Control loop (hardware I/O) | 60Hz | `CONTROL_FREQ_HZ` | `cap/config.py:107` |
| learn_skill / Fello loop | 30Hz | `POLICY_FREQ_HZ` | `cap/config.py:110` |
| Safety zone enforcement | 30Hz (inside learn_skill) | Same as POLICY_FREQ_HZ | `cap/server/cap_server.py:3116` |
| Viser safety viz polling | ~3Hz | `_loop_count % 10` at ~30Hz base | `cap/agent/cap_agent.py:2431` |
| Broadcast loop base rate | ~30Hz | `asyncio.sleep(0.033)` | `cap/agent/cap_agent.py:2492` |

---

## Future Work

- **Bimanual collision avoidance**: Minimum EE-to-EE distance enforcement between arms
- **Convex hull interpolation for orientation**: Instead of nearest-keypose orientation, interpolate based on barycentric coordinates within the hull
- **Adaptive margins**: Shrink margins as RL training converges (curriculum-style)
- **Binary search enforcement**: For large joint steps, binary search on the interpolation factor to find the exact safe boundary rather than using linear approximation
- **Viser icosphere fix**: Investigate Viser version upgrade or patch for the Three.js `scale` property crash to enable sphere-based visualization
- **Dashboard integration**: The `safety_enforced` events are emitted but `cap/diag/dashboard.py` does not yet have a dedicated panel for visualizing enforcement frequency or zone violations over time
