# RoboCasa Integration

> **Status**: Active development on `jl/robocasa365` branch
> **Last verified**: 2026-04-13

**Cross-references**: [CAP_DESIGN](CAP_DESIGN.md) | [SAFETY_ZONE_DESIGN](SAFETY_ZONE_DESIGN.md) | [TABLE_BUSSING_SKILLS](TABLE_BUSSING_SKILLS.md) | [CUROBO_UPDATE_SUMMARY](CUROBO_UPDATE_SUMMARY.md) | [remote_serving](remote_serving.md)

---

## Overview

RoboCasa is a large-scale simulation framework for household robot tasks (365 tasks across kitchen environments). This integration connects RoboCasa to the CAP agent framework via a protocol-based environment abstraction, allowing the same CAP tools (freespace_move, nudge, set_gripper, VLM queries) to drive simulated robots in RoboCasa environments.

Key design principle: **no external IK needed** — RoboCasa's internal OSC_POSE controller handles inverse kinematics via robosuite's MuJoCo model. The CAP server just sends EE delta targets.

`RoboCasaEnv` is **pinned to `OSC_POSE`** (`cap/env/robocasa/env.py:38`, `cap/env/__init__.py:85-87` — the `controller_type` kwarg is silently dropped). cuRobo joint trajectories are tracked via per-waypoint FK + OSC tick-stepping in `_execute_osc_trajectory` (`cap/env/robocasa/skills.py`). `ROBOCASA_CONTROLLER_TYPE` is no longer read. cuRobo still runs on a remote cloud GPU server — the client connects with explicit `host:port`. See [cuRobo Cloud Serving](#curobo-cloud-serving-direct-mode) below.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  CAP Agent  (LLM + tools)                               │
│  cap/agent/cap_agent.py                                 │
├─────────────────────────────────────────────────────────┤
│  Portal RPC (port 18500)                                │
├─────────────────────────────────────────────────────────┤
│  CAP Server  cap/server/cap_server.py                   │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ SimArmClient │  │SimCameraClient│ │  TaskProtocol │ │
│  │ sim_backend  │  │ sim_backend   │ │  (reset/reward)│ │
│  │ .py:440      │  │ .py:457      │ │               │ │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘ │
├─────────┼──────────────────┼──────────────────┼─────────┤
│  RoboCasaEnv  cap/env/robocasa/env.py:35                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │  EnvProtocol      — physics step, observation    │   │
│  │  EefControlProtocol — per-tick EE delta control  │   │
│  │  TaskProtocol     — episode reset, reward, done  │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  robosuite (MuJoCo) + robocasa (kitchen assets/tasks)   │
└─────────────────────────────────────────────────────────┘
```

## Environment Naming Convention

Format: `robocasa:<TaskName>:<RobotName>`

| Example | Task | Robot |
|---------|------|-------|
| `robocasa` | PickPlaceCounterToCabinet (default) | PandaOmron (default) |
| `robocasa:PickPlaceCounterToCabinet` | PickPlaceCounterToCabinet | PandaOmron |
| `robocasa:PickPlaceCounterToCabinet:GR1ArmsOnly` | PickPlaceCounterToCabinet | GR1ArmsOnly |

Parsed by `cap/env/__init__.py:36-86`.

## Core Components

### RoboCasaEnv

**File**: `cap/env/robocasa/env.py:35`

Implements three protocols:

| Protocol | Purpose | Key Methods |
|----------|---------|-------------|
| `EnvProtocol` | Physics stepping, observation | `step()`, `get_arm_observation()`, `render_rgb()`, `render_depth()` |
| `EefControlProtocol` | Per-tick EE motion control | `compute_eef_action()` |
| `TaskProtocol` | Episode management | `reset_env()`, `load_task()`, `get_task_info()` |

#### Constructor Parameters (`robocasa.py:38-87`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env_name` | str | — | RoboCasa task name |
| `robot` | str | `"PandaOmron"` | Robot type |
| `profile` | RobotProfile | None | Robot configuration |
| `camera_height` | int | 480 | Render height |
| `camera_width` | int | 640 | Render width |
| `has_renderer` | bool | False | MuJoCo viewer |
| `control_freq` | int | 20 | Hz |

#### Action Space (OSC_POSE)

```
action = [dx, dy, dz, drx, dry, drz, gripper]  # 7-DOF per arm
```

| Component | Range | Scale | Reference |
|-----------|-------|-------|-----------|
| Position delta | ±1.0 (normalized) | ±0.05 m/action | `robocasa.py:101` |
| Orientation delta | ±1.0 (normalized) | ±0.5 rad/action | `robocasa.py:102` |
| Gripper | -1.0 (closed) to 1.0 (open) | — | `robocasa.py:288` |

#### Observation Space

```python
get_arm_observation(side) → {
    "joint_pos":   np.ndarray  # shape (7,), radians
    "gripper_pos": float       # 0.0 (closed) to 1.0 (open)
    "ee_pos":      np.ndarray  # shape (3,), world frame meters
    "ee_quat":     np.ndarray  # shape (4,), [x,y,z,w]
}
```

Reference: `robocasa.py:142-154`

#### Camera System (`robocasa.py:60-67, 163-227`)

Camera name mapping (CAP name → RoboCasa observation key):

| CAP Name | RoboCasa Key | Default |
|----------|-------------|---------|
| `"top"` | `"robot0_agentview_left_image"` | Yes |
| `"wrist"` | `"robot0_eye_in_hand_image"` | Yes |

Configurable via `profile.camera_obs_key_map` (`profile.py:266-269`).

Rendering runs on a dedicated `ThreadPoolExecutor` (`robocasa.py:114-119`) to respect EGL context affinity. Camera intrinsics computed from the MuJoCo model at `robocasa.py:204-217`.

#### EE Motion Control (`robocasa.py:233-292`)

`compute_eef_action(side, target_pos, target_quat_xyzw, gripper=None)`:

1. Compute position delta in world frame
2. Compute orientation delta as axis-angle (rotvec)
3. Transform to base frame if needed (`robocasa.py:267-270`)
4. Clip to ±1.0 normalized range (`robocasa.py:281-284`)
5. Encode gripper: `action[grip_start] = 1.0 - 2.0 * gripper_value` (`robocasa.py:288`)
6. Store as `_pending_eef_action` for next `step()` call

No external IK solver — robosuite's OSC_POSE controller handles it internally.

#### Task Management

| Method | Line | Description |
|--------|------|-------------|
| `reset_env()` | `robocasa.py:298-306` | Reset episode, returns `{"ok": True, "task": env_name}` |
| `reset_to_initial()` | `robocasa.py:468-` | Deterministic reset — restores `_post_reset_state` snapshot (same scene, same objects, same positions) |
| `load_task(task_name)` | `robocasa.py:308-347` | Tear down env, rebuild with new task |
| `get_task_info()` | `robocasa.py:352-364` | Returns `{done, reward, success, env_name, obj_pos}` |

#### Reset Mechanism: `_post_reset_state` Snapshot

RoboCasa's reset has a subtle problem: robosuite captures `sim_state_initial` before `_reset_internal()` runs, so restoring it yields an empty scene (no objects). Additionally, calling `env.reset()` again advances the internal RNG, producing different objects/positions even with the same seed.

The solution is a **post-reset sim state snapshot**:

1. On construction, `RoboCasaEnv.__init__()` calls `env.reset()` which runs the full reset pipeline (layout setup, object sampling, placement, settling).
2. After reset completes, it captures `self._post_reset_state = self._env.sim.get_state()` (`env.py:119`). This snapshot includes all placed and settled objects.
3. `reset_to_initial()` restores this snapshot via `sim.set_state()` + `sim.forward()`, giving an identical scene every time — no RNG advancement, no re-randomization.

**Why not just call `env.reset()` again?** RoboCasa's internal RNG (`env.rng`) advances on every `reset()` call, changing object instances and placements. Even recreating the env from scratch doesn't help because some fixture constructors call `np.random.default_rng()` without a seed (reads `/dev/urandom`).

**In direct mode** (via `run_agent.py`), the skills namespace exposes `reset_env()` which internally calls `reset_to_initial()` — NOT `env.reset()`. This ensures deterministic resets for the agent retry loop. See `cap/env/robocasa/skills.py:333-341`.

### Robot Profiles

**File**: `cap/env/profile.py`

#### PandaOmron (`profile.py:216-271`)

| Property | Value |
|----------|-------|
| DOF | 7 |
| Arms | `("right",)` — single arm |
| Control freq | 20 Hz |
| Joint limits | ±2.8973 to 3.7525 rad (joint-dependent) |
| Gripper | 0.0 (closed) to 1.0 (open) |
| Mobile base | Yes (`is_mobile_base=True`) |
| Cameras | `("top", "wrist")` |

#### GR1ArmsOnly (`profile.py:274-326`)

| Property | Value |
|----------|-------|
| DOF | 7 per arm (14 total) |
| Arms | `("left", "right")` — bimanual |
| Control freq | 20 Hz |
| Mobile base | No |
| Cameras | `("top", "wrist")` |

### SimArmClient & SimCameraClient

**File**: `cap/server/sim_backend.py:440-507`

Drop-in replacements for hardware arm/camera clients:

```
SimArmClient(backend, side)     → routes to RoboCasaEnv.get_arm_observation()
SimCameraClient(backend, name)  → background render thread at ~15 FPS
```

The CAP server creates these dynamically based on the profile's arm and camera names (`cap_server.py:815-839`).

## CAP Server Integration

### Environment Creation (`cap_server.py:815-839`)

```python
self._sim_backend = create_env(env_name, viewer=env_viewer)
self._robot_profile = getattr(self._sim_backend, "_profile", None)
self._arms = {side: SimArmClient(self._sim_backend, side) for side in arm_names}
self._cameras = {name: SimCameraClient(self._sim_backend, name) for name in camera_names}
```

Profile drives dynamic arm/camera discovery — single-arm robots (PandaOmron) only get `"right"`.

### EE Control in Control Loop (`cap_server.py:1378-1389`)

Backend provides EE poses directly (no pinocchio FK needed):

```python
obs = self._arms[side].get_observations()
if "ee_pos" in obs and "ee_quat" in obs:
    self._ee_pos[side][:] = obs["ee_pos"]
    self._ee_quat[side][:] = obs["ee_quat"]
```

### _ik_servo Integration (`cap_server.py:1791-1808`)

When `EefControlProtocol` is detected:
1. Sets EE target in `_eef_targets[side]`
2. Control loop calls `compute_eef_action()` every tick
3. Waits for convergence by polling cached `_ee_pos`

### Task RPC Endpoints (`cap_server.py:1120-1130`)

Registered when backend implements `TaskProtocol`:

| RPC | Handler | Description |
|-----|---------|-------------|
| `reset_env` | `_rpc_reset_env` (`cap_server.py:4115`) | Reset episode |
| `get_task_info` | `_rpc_get_task_info` (`cap_server.py:4130`) | Get done/reward/success |
| `get_last_reward` | `_rpc_get_last_reward` (`cap_server.py:4140`) | Reward value only |
| `load_task` | `_rpc_load_task` (`cap_server.py:4145`) | Switch to new task |

## Tools

### LoadTaskTool (`cap/agent/tools/native.py:585-604`)

```python
class LoadTaskTool(_PortalMixin, Tool):
    name = "load_task"
    description = "Load a new task/scene (e.g. RoboCasa task). Resets the environment."
```

Available when the backend supports `load_task()`. All other CAP tools (`freespace_move`, `nudge`, `set_gripper`, `get_robot_state`, `vlm_query`, etc.) work unchanged.

## cuRobo Cloud Serving (Direct Mode)

When running RoboCasa in **direct mode** (via `run_agent.py --direct`), `freespace_move` replaces `_ik_servo` as the primary motion tool. It uses cuRobo for collision-free trajectory planning, and cuRobo runs on a **remote cloud GPU server** rather than being started as a local subprocess.

### Architecture

```
┌──────────────────────────┐         ┌──────────────────────────┐
│  Local Machine           │         │  Cloud GPU Server         │
│                          │         │                           │
│  run_agent.py --direct   │         │  serve_portal_motion_     │
│   └─ RoboCasaEnv         │  Portal │  planner.py               │
│   └─ skills.make_namespace│  RPC   │   └─ cuRobo solver        │
│   └─ freespace_move ─────┼────────▶│   └─ GPU motion planning  │
│       (PortalMotionPlanner│         │                           │
│        start_server=False)│         │  Port: 8611 (default)     │
└──────────────────────────┘         └──────────────────────────┘
```

### How it works

1. **cuRobo planner server** runs on a cloud GPU machine (e.g. via `experimental/serve_portal_motion_planner.py`). It exposes a Portal RPC endpoint (default port `8611`).

2. **`freespace_move` tool** (`cap/agent/tools/freespace_move.py`) creates a `PortalMotionPlanner` client with `start_server=False` and explicit `host:port`, connecting to the remote server instead of spawning a local cuRobo subprocess.

3. **`go_home`** uses `freespace_move` (cuRobo) to plan a collision-free trajectory back to the initial EE pose captured at env reset. This replaces the previous approach of calling `reset_to_initial()` which teleported the entire scene (resetting all object positions).

4. **Controller pinning**: `RoboCasaEnv` is pinned to `OSC_POSE` (see above). cuRobo-planned joint trajectories are executed via per-waypoint FK + OSC tick-stepping in `_execute_osc_trajectory`, not by switching the underlying robosuite controller.

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CAP_CUROBO_HOST` | cuRobo server hostname/IP | `127.0.0.1` |
| `CAP_CUROBO_PORT` | cuRobo server Portal RPC port | (none — auto-starts if unset) |
| `CAP_ROBOT_TYPE` | Robot model for cuRobo/freespace_move (`panda`, `yam`); RoboCasa skills hardcode `panda` internally | `yam` |

### Running with remote cuRobo

`run_agent.py` is Hydra-configured; ports live in `experiments/infra/ports.yaml`
and env-specific overrides in the experiment YAML. Example:

```bash
# On the cloud GPU server:
uv run experimental/serve_portal_motion_planner.py --port 8611 --robot-type panda

# On the local machine:
CAP_CUROBO_HOST=<cloud-gpu-ip> \
uv run python run_agent.py experiment=pick_place_sink_to_counter runtime.curobo_port=8611

# Or with SSH tunnel:
ssh -N -L 8611:localhost:8611 user@cloud-gpu &
uv run python run_agent.py experiment=pick_place_sink_to_counter runtime.curobo_port=8611
```

### Tool behavior (OSC_POSE-pinned env)

| Tool | Behavior |
|------|----------|
| `_ik_servo` | YAM IK-servo path — not used in RoboCasa |
| `freespace_move` | **Primary motion tool** — cuRobo plans collision-free joint trajectories; executed via FK + OSC tick-stepping |
| `go_home` | Uses `freespace_move` to plan trajectory to initial EE pose |
| `reset_env` (direct-mode skills namespace) | Aliases to `env.reset_to_initial()` — sim-state restore, not `env.reset()` |
| `reset_to_initial` (CapServer RPC) | Direct sim-state restore call (separate from `reset_env`) |

### Cross-references

- [CUROBO_UPDATE_SUMMARY](CUROBO_UPDATE_SUMMARY.md) — Full cuRobo architecture, robot configs, collision system
- [remote_serving](remote_serving.md) — Remote GPU serving setup, SSH tunnels, port map
- [ROBOCASA_RANDOMNESS](ROBOCASA_RANDOMNESS.md) — Seed/layout/style determinism for reproducible experiments

## Running

### Start CAP server with RoboCasa

```bash
# Default task + robot
uv run cap/server/cap_server.py --env robocasa --viewer

# Specific task
uv run cap/server/cap_server.py --env robocasa:PickPlaceCounterToCabinet --viewer

# Specific task + robot
uv run cap/server/cap_server.py --env robocasa:PickPlaceCounterToCabinet:GR1ArmsOnly --viewer
```

### Run a saved script

```bash
uv run run_script.py --file robocasa_pick_object.py --env robocasa:PickPlaceCounterToCabinet --viewer
```

### Run the demo pick script

```bash
uv run cap/server/demo_robocasa_pick_tools.py
```

## GR00T Policy Evaluation

See [ROBOCASA_INTEGRATION_POLICY](ROBOCASA_INTEGRATION_POLICY.md) for full documentation on:
- PandaOmron24 evaluation (GR00T N1.6)
- RoboCasa365 benchmark evaluation (GR00T N1.5 / N1.6)
- Environment setup, model checkpoints, and CLI reference

## Dependencies

**pyproject.toml** (`pyproject.toml:99-103`):

```toml
[project.optional-dependencies]
robocasa = [
    "robosuite",
    "robocasa",
    "qpsolvers[quadprog]>=4.3.1",
]
```

Install:
```bash
uv sync --extra robocasa
```

Download kitchen assets:
```bash
download-robocasa-assets
```

Robosuite is vendored as a git submodule at `third_party/robosuite/` (`.gitmodules`). RoboCasa is vendored at `third_party/robocasa/` as a local editable install (`pyproject.toml:149: robocasa = { path = "third_party/robocasa", editable = true }`).

## Example: Scripted Pick Object

**File**: `cap/saved_scripts/robocasa_pick_object.py`

```python
# Get object position from task info
info = get_task_info()
obj_pos = info["obj_pos"]

# Hover above object
freespace_move("right", [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1])

# Open gripper
set_gripper("right", 1.0)

# Lower to grasp
freespace_move("right", obj_pos)

# Close gripper
set_gripper("right", 0.0)

# Lift
freespace_move("right", [obj_pos[0], obj_pos[1], obj_pos[2] + 0.2])
```

## Thread Safety

| Resource | Protection | Location |
|----------|-----------|----------|
| Physics state (`_obs`, `_reward`, `_done`) | `threading.Lock` | `robocasa.py:112` |
| EGL render context | `ThreadPoolExecutor` (single thread) | `robocasa.py:114-119` |
| Pending EE action | `_pending_eef_action` dict | `robocasa.py:110` |

Renderers are closed on the executor thread (`robocasa.py:388-399`) to respect EGL context affinity.

## Differences from YAM Hardware

| Aspect | YAM (real/MuJoCo sim) | RoboCasa |
|--------|----------------------|----------|
| IK | Pinocchio FK + cuRobo motion planning | OSC_POSE (pinned); cuRobo joint plans executed via FK + OSC tick-stepping |
| Arms | 2 (left + right, 7-DOF each) | 1 (PandaOmron) or 2 (GR1) |
| Control freq | 60 Hz | 20 Hz |
| Cameras | RealSense / ZED hardware | MuJoCo rendered (EGL) |
| Gripper encoding | 0.0–1.0 | -1.0 (closed) to 1.0 (open), remapped |
| Safety zones | Enforced via `_enforce_safety_zone()` | Not applicable |
| Mobile base | No | Yes (PandaOmron) |
| Task management | N/A (operator-driven) | `load_task()`, `reset_env()`, `get_task_info()` |

## File Reference

| File | Key Contents |
|------|-------------|
| `cap/env/__init__.py:36-86` | `create_env()` factory, env name parsing |
| `cap/env/base.py` | `EnvProtocol`, `EefControlProtocol`, `TaskProtocol` definitions |
| `cap/env/robocasa/env.py:35+` | `RoboCasaEnv` — main environment wrapper (`cap/env/robocasa.py` is now a 6-line compat shim) |
| `cap/env/robocasa/env.py` | `RoboCasaEnv` — refactored env with `_post_reset_state` snapshot and `reset_to_initial()` |
| `cap/env/robocasa/skills.py` | Direct-mode skills namespace (`reset_env` calls `reset_to_initial` internally) |
| `cap/env/profile.py:216-326` | `robocasa_panda_omron_profile()`, `GR1ArmsOnly` profile |
| `cap/env/yam.py` | YAM environment (for comparison) |
| `cap/server/cap_server.py:815-839` | Env creation, arm/camera binding |
| `cap/server/cap_server.py:1120-1130` | Task RPC registration |
| `cap/server/cap_server.py:1378-1389` | EE pose caching from backend |
| `cap/server/cap_server.py:1791-1808` | EefControlProtocol _ik_servo path |
| `cap/server/cap_server.py:4115-4149` | Task RPC implementations |
| `cap/server/sim_backend.py:440-507` | `SimArmClient`, `SimCameraClient` |
| `cap/server/demo_robocasa_pick_tools.py` | Demo pick script |
| `cap/saved_scripts/robocasa_pick_object.py` | Saved script example |
| `cap/agent/tools/native.py:585-604` | `LoadTaskTool` |
| `run_script.py:489-630` | Script runner with `--env` flag |
| `pyproject.toml:99-103` | RoboCasa optional dependency group |
| `.gitmodules` | robosuite submodule |
| `third_party/robocasa/` | RoboCasa source (vendored) |
| `third_party/robosuite/` | robosuite submodule |
