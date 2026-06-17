# cuRobo Integration Update Summary

This document summarizes the cuRobo integration into `lecar-tbd`, including:

- Architecture overview and file map
- Environment setup
- Robot model and config details
- Collision avoidance system
- Motion planning API and integration layers
- CAP agent tool integration
- Remote serving architecture
- How to run and compare planners
- Benchmark results and tuning
- Test coverage

> **Cross-references:**
> - [CUROBO_ISAACSIM_SETUP.md](CUROBO_ISAACSIM_SETUP.md) -- Isaac Sim USD generation, Lula sphere fitting, standalone cuRobo bringup
> - [remote_serving.md](remote_serving.md) -- Remote cuRobo Portal server on S1, SSH forwarding, port map
> - [TABLE_BUSSING_SKILLS.md](TABLE_BUSSING_SKILLS.md) -- Table bussing freespace_move skill tool usage
> - [CAP_DESIGN.md](CAP_DESIGN.md) -- CAP agent framework architecture

---

## 1. Architecture overview

cuRobo is the **default motion planner** for the YAM bimanual platform. It replaces the legacy RRT-Connect planner for both the Viser scripted-UI Move-To flow and the CAP agent `freespace_move` tool.

### Data flow

```
                          +-----------------------+
                          | CAP agent / Viser UI  |
                          +----------+------------+
                                     |
                   freespace_move()  |  plan_to_pose() / plan_batch_to_pose()
                                     v
                          +----------+------------+
                          | PortalMotionPlanner   |  (Portal RPC client)
                          | portal_motion_planner |
                          +----------+------------+
                                     | Portal RPC
                                     v
                 +-------------------+--------------------+
                 | PortalMotionPlannerServer              |
                 | (runs on GPU host, e.g. LeCAR-S1)     |
                 +-------------------+--------------------+
                                     |
                                     v
                 +-------------------+--------------------+
                 | YamMotionPlannerCurobo                 |
                 | motion_planner_curobo.py               |
                 |   - MotionGen (cuRobo core)            |
                 |   - MuJoCo trajectory validator        |
                 |   - Depth-scene collision (optional)   |
                 +-------------------+--------------------+
                                     |
                         +-----------+-----------+
                         | cuRobo MotionGen      |
                         | (IK + graph + trajopt)|
                         +-----------+-----------+
                                     |
                         +-----------+-----------+
                         | Robot YAML config     |
                         | URDF + collision      |
                         | spheres               |
                         +-----------+-----------+
```

### Key file map

| File | Purpose |
|------|---------|
| `experimental/motion_planner_curobo.py` | Core cuRobo planner class `YamMotionPlannerCurobo` |
| `experimental/portal_motion_planner.py` | Portal RPC server + client wrappers |
| `experimental/serve_portal_motion_planner.py` | CLI entrypoint for the remote Portal server |
| `experimental/curobo_depth_world.py` | Depth-image to collision world conversion, robot segmentation |
| `experimental/start_stop_play_policy.py` | Planner lifecycle in the Viser scripted-UI flow |
| `experimental/yam_control_loop.py` | Main control loop; configures planner backend |
| `cap/agent/tools/freespace_move.py` | CAP agent `freespace_move` tool (default backend: cuRobo) |
| `cap/agent/cap_agent.py` | CAP agent orchestrator; routes `freespace_move` |
| `robot/models/station/curobo/*.yml` | cuRobo robot YAML configs (collision spheres, joint space) |
| `robot/models/station/station_physics_fixed_fingers.urdf` | URDF used by cuRobo planner |
| `third_party/curobo/` | cuRobo source (git submodule, editable install) |
| `tmux/remote_serving/launch_lecar_s1.sh` | Launches cuRobo Portal on S1 (port 8611) |
| `tmux/remote_serving/install_curobo_s1.sh` | Verifies / installs cuRobo on S1 |

### Benchmark and visualization tools

| File | Purpose |
|------|---------|
| `experimental/benchmark_curobo.py` | Reachability and speed benchmarks |
| `experimental/full_benchmark_curobo.py` | Threshold vs time + single vs dual sweep |
| `experimental/benchmark_rrt_vs_curobo.py` | RRT-Connect vs cuRobo comparison |
| `experimental/plot_curobo_benchmark.py` | PNG plot generator for benchmark runs |
| `experimental/plot_curobo_benchmark_viser.py` | Viser 3D visualizer for reachability results |

### Test files

| File | Purpose |
|------|---------|
| `tests/test_curobo_single_arm_regression.py` | Single-arm trajectory validation logic |
| `tests/test_motion_planner_curobo_unified.py` | Batch planner routing and merge logic |
| `tests/manual/check_single_arm_curobo_control_loop.py` | Manual GPU integration check |

---

## 2. Environment setup

Use the **repo environment** and run commands from repo root:

```bash
cd /home/lecar/yiyang/lecar-tbd
```

### Sync repo dependencies

```bash
git submodule update --init --recursive
uv sync --extra cap
```

If additional optional dependencies are needed:

```bash
uv sync --extra experimental
```

### Install cuRobo into the same env used by lecar

```bash
uv pip install --python .venv/bin/python ninja
CUDA_HOME=/usr/local/cuda-12.8 \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0 \
uv pip install --python .venv/bin/python -e third_party/curobo --no-build-isolation
```

> **Note:** On LeCAR-S1, the script `tmux/remote_serving/install_curobo_s1.sh` automates this verification and install step.

### Quick verification

```bash
uv run python - <<'PY'
import torch, curobo
print("cuda available:", torch.cuda.is_available())
print("curobo:", curobo.__file__)
PY
```

> **See also:** [CUROBO_ISAACSIM_SETUP.md](CUROBO_ISAACSIM_SETUP.md) for Isaac Sim environment setup (separate Python 3.10/3.11 envs, USD generation, Lula sphere fitting).

---

## 3. Robot model and config files

### Active planner config (used at runtime)

The planner loads its robot config from:

- **YAML:** `robot/models/station/curobo/yam_dual_isaacsim_physics_fixed_fingers.yml`
  - Defined at `experimental/motion_planner_curobo.py:36-41` (`_ROBOT_CFG`)
- **URDF:** `robot/models/station/station_physics_fixed_fingers.urdf`
  - Defined at `experimental/motion_planner_curobo.py:51-53` (`_URDF_PATH`)

The YAML is loaded and patched at runtime to force URDF-based kinematics:

```python
# experimental/motion_planner_curobo.py:345-356
kin["use_usd_kinematics"] = False
kin["usd_path"] = ""
kin["isaac_usd_path"] = ""
kin["urdf_path"] = str(self._urdf_path)
kin["asset_root_path"] = str(self._asset_root)
```

### All cuRobo YAML configs in the repo

| Config file | Purpose |
|-------------|---------|
| `robot/models/station/curobo/yam_dual_isaacsim_physics_fixed_fingers.yml` | **Active** -- dual-arm, physics URDF, fixed fingers, collision spheres enabled |
| `robot/models/station/curobo/yam_dual_isaacsim_physics_fixed_fingers_no_collision.yml` | Dual-arm, no collision spheres (legacy testing fallback) |
| `robot/models/station/curobo/yam_dual_isaacsim_physics.yml` | Dual-arm, physics URDF (non-fixed fingers) |
| `robot/models/station/curobo/yam_dual_isaacsim_fixed_fingers.yml` | Dual-arm, non-physics URDF, fixed fingers |
| `robot/models/station/curobo/yam_dual.yml` | Base dual-arm config |
| `robot/models/station/curobo/spheres/` | Lula-exported collision sphere YAMLs |

> **See also:** [CUROBO_ISAACSIM_SETUP.md](CUROBO_ISAACSIM_SETUP.md) for config naming conventions, USD generation, and Lula sphere fitting workflow.

### YAML config structure

The active config (`yam_dual_isaacsim_physics_fixed_fingers.yml`) defines:

- **Kinematics:** `base_link`, `ee_link: left_grasp`, `link_names: [left_grasp, right_grasp]`
  - 12-DOF joint space: `left_joint1..6`, `right_joint1..6` (`yam_dual_isaacsim_physics_fixed_fingers.yml:165-177`)
  - Retract config and null-space weights (`yam_dual_isaacsim_physics_fixed_fingers.yml:178-192`)
  - `max_jerk: 500.0`, `max_acceleration: 15.0` (`yam_dual_isaacsim_physics_fixed_fingers.yml:193-194`)
- **Collision links:** 15 links with fitted Lula collision spheres (`yam_dual_isaacsim_physics_fixed_fingers.yml:21-112`)
  - Left arm: `left_arm`, `left_link_1` through `left_link_6`
  - Right arm: `right_arm`, `right_link_1` through `right_link_6`, `right_left_link_finger`
  - `collision_sphere_buffer: 0.005` (`yam_dual_isaacsim_physics_fixed_fingers.yml:113`)
- **Self-collision ignore:** Adjacent link pairs excluded from self-collision checks (`yam_dual_isaacsim_physics_fixed_fingers.yml:115-128`)

### World obstacle links

Static environment obstacles are extracted from the URDF at runtime:

```python
# experimental/motion_planner_curobo.py:55
_WORLD_OBSTACLE_LINKS = ("gate_collision", "play_table")
```

These URDF fixed-joint links are converted to cuRobo `Cuboid` obstacles in `_build_world_cfg()` (`experimental/motion_planner_curobo.py:358-428`).

---

## 4. Collision avoidance system

### Collision layers

The cuRobo planner uses three collision layers, composed at runtime:

1. **Static world** (`_static_world_cfg`): cuboid obstacles from URDF fixed links (`gate_collision`, `play_table`). Built once in `_setup_motion_gen()` at `experimental/motion_planner_curobo.py:487-490`.

2. **Depth scene** (`_depth_world_cfg`, optional): mesh obstacles from a live depth camera point cloud. Updated via `set_depth_collision_scene()` at `experimental/motion_planner_curobo.py:540-600`. Enabled when `enable_depth_collision=True` or `CUROBO_ENABLE_DEPTH_COLLISION=1`.

3. **Debug collision ball** (`_debug_world_cfg`, optional): a single sphere obstacle for interactive testing. Set via `set_debug_collision_ball()` at `experimental/motion_planner_curobo.py:609-635`.

All three are composed in `_compose_world_cfg()` (`experimental/motion_planner_curobo.py:520-530`) and pushed to the `MotionGen` instance via `_refresh_world()` (`experimental/motion_planner_curobo.py:532-538`).

### Collision checker type

```python
# experimental/motion_planner_curobo.py:439
collision_checker_type=self._CollisionCheckerType.MESH
```

Self-collision checking is **disabled** at the cuRobo level:

```python
# experimental/motion_planner_curobo.py:440-441
self_collision_check=False,
self_collision_opt=False,
```

### MuJoCo trajectory validation

After cuRobo generates a trajectory, it is optionally validated against the legacy MuJoCo collision checker for parity (`experimental/motion_planner_curobo.py:904-960`). This catches cases where:
- cuRobo plans through the inactive arm for single-arm moves
- The fixed-arm variant is tested first and used if collision-free

Controlled by `validate_with_mujoco` (default `True`) and `validate_trajectory` per call.

### Depth-image collision pipeline

`experimental/curobo_depth_world.py` provides:

| Function | Line | Purpose |
|----------|------|---------|
| `point_cloud_from_depth()` | `:26-58` | Depth image to 3D point cloud with clip/downsample |
| `transform_points()` | `:61-70` | Transform points by 4x4 matrix (camera to world) |
| `filter_depth_with_robot_mask()` | `:116-178` | Remove robot pixels from depth using cuRobo `RobotSegmenter` |
| `get_robot_spheres_world()` | `:182-209` | Get collision spheres in world frame for given joint state |
| `filter_points_near_robot()` | `:212-237` | Remove points too close to robot spheres |
| `create_world_config_from_points()` | `:240-258` | Convert filtered point cloud to cuRobo `WorldConfig` via marching cubes |

Robot segmentation uses `curobo.wrap.model.robot_segmenter.RobotSegmenter` with the same YAML config and URDF as the planner (`experimental/curobo_depth_world.py:74-113`).

---

## 5. YamMotionPlannerCurobo API

**File:** `experimental/motion_planner_curobo.py:212-1431`

### Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `robot_cfg_path` | `_ROBOT_CFG` (physics fixed fingers) | cuRobo robot YAML |
| `urdf_path` | `_URDF_PATH` (physics fixed fingers) | URDF for kinematics |
| `validate_with_mujoco` | `True` | Enable MuJoCo trajectory validator |
| `device` | `"cuda:0"` | CUDA device |
| `enable_finetune_trajopt` | `False` | Enable trajectory optimization finetune pass |
| `enable_depth_collision` | `False` | Enable depth-camera collision scene |
| `solver_speed` | `"fast"` | Solver preset (`"fast"` or `"slow"`) |
| `position_threshold` | `0.005` (5 mm) | IK position convergence threshold |
| `rotation_threshold` | `0.05` (rad) | IK rotation convergence threshold |
| `cspace_threshold` | `0.05` (rad) | Configuration space threshold |

### Compile-time constants

| Constant | Value | Line |
|----------|-------|------|
| `_USE_CUDA_GRAPH_BY_DEFAULT` | `True` | `:60` |
| `_ENABLE_GRAPH_SEARCH_BY_DEFAULT` | `True` | `:61` |
| `_CUROBO_TORCH_COMPILE_DISABLE_DEFAULT` | `"1"` (disabled) | `:62` |
| `_DEFAULT_BATCH_PLANNER_CAPACITY` | `16` | `:63` |
| `_DEFAULT_SOLVER_SPEED` | `"fast"` | `:64` |
| `_INTERPOLATION_DT` | `1/30 s` | `:56` |

### Solver speed presets

Defined at `experimental/motion_planner_curobo.py:65-100`:

| Setting | `fast` | `slow` |
|---------|--------|--------|
| `num_ik_seeds` | 8 | 32 |
| `num_graph_seeds` | 1 | 12 |
| `num_trajopt_seeds` | 2 | 12 |
| `trajopt_tsteps` | 32 | 48 |
| `ik_opt_iters` | 96 | 256 |
| `grad_trajopt_iters` | 96 | 256 |
| `enable_graph_attempt` | 1 | 4 |
| `max_attempts` | 2 | 10 |
| `timeout` | 2.5 s | 10.0 s |
| `time_dilation_factor_single` | 0.5 | 0.5 |

### Public methods

| Method | Line | Description |
|--------|------|-------------|
| `plan_to_pose()` | `:1206` | Single EE target planning (routes through batch internally) |
| `plan_batch_to_pose()` | `:1344` | Batched EE target planning with auto-chunking |
| `set_finetune_enabled()` | `:505` | Toggle finetune trajectory optimization at runtime |
| `set_gripper_qpos()` | `:962` | Set gripper positions for collision checking |
| `set_depth_collision_scene()` | `:540` | Update depth-camera collision world |
| `clear_depth_collision_scene()` | `:602` | Remove depth collision obstacles |
| `set_debug_collision_ball()` | `:609` | Add a sphere obstacle for debugging |
| `clear_debug_collision_ball()` | `:637` | Remove debug sphere |
| `check_collision()` | `:974` | Delegate to MuJoCo collision check |

### Batch planning internals

`plan_batch_to_pose()` at `:1344` auto-chunks queries exceeding the CUDA-graph batch capacity (default 16) into multiple `_plan_batch_to_pose_chunk()` calls at `:988`, then merges results via `_merge_batch_plan_results()` at `:789`.

Each chunk pads to fixed batch size using the **last real query** (not a no-op current-pose) to avoid cuRobo shape bugs with partially filled CUDA-graph batches (`experimental/motion_planner_curobo.py:1013-1045`).

### Return format

Both `plan_to_pose()` and `plan_batch_to_pose()` return a dict with:

```python
{
    "status": "Success" | "Partial_Success" | "Planning_Failed" | "IK_Failed",
    "status_detail": str | None,
    "left_positions": np.ndarray,     # (T, 6) trajectory
    "right_positions": np.ndarray,    # (T, 6) trajectory
    "position_error_m": float,
    "rotation_error_deg": float,
    "curobo_solve_time_ms": float,
    "curobo_total_time_ms": float,
    "curobo_ik_time_ms": float,
    "curobo_graph_time_ms": float,
    "curobo_trajopt_time_ms": float,
    "curobo_finetune_time_ms": float,
    "curobo_attempts": int,
    "curobo_trajopt_attempts": int,
    "curobo_used_graph": bool,
}
```

The alias `YamMotionPlanner = YamMotionPlannerCurobo` at `:1430` enables drop-in replacement.

---

## 6. Integration layers

### Portal RPC wrapper

**File:** `experimental/portal_motion_planner.py`

The planner runs in a dedicated GPU process and is accessed via Portal RPC.

- **Server:** `PortalMotionPlannerServer` (`:36`) wraps `YamMotionPlannerCurobo` and exposes RPC bindings for: `health_check`, `set_finetune_enabled`, `set_gripper_qpos`, `set_depth_collision_scene`, `clear_depth_collision_scene`, `set_debug_collision_ball`, `clear_debug_collision_ball`, `plan_to_pose`, `plan_batch_to_pose`.
- **Client:** `PortalMotionPlanner` (`:206`) provides the same public API as `YamMotionPlannerCurobo` but proxies all calls over Portal RPC to the server.
- **CLI:** `experimental/serve_portal_motion_planner.py` starts the server with `--port`, `--solver-speed`, `--enable-depth-collision` flags.

### Viser scripted-UI integration

**File:** `experimental/start_stop_play_policy.py`

The `StartStopPlayPolicyWrapper` manages the planner lifecycle:

- Backend selection: `_normalize_motion_planner_backend()` at `:237` (choices: `"rrtconnect"`, `"curobo"`)
- Solver speed: `_normalize_motion_planner_solver_speed()` at `:245` (choices: `"slow"`, `"fast"`)
- Lazy planner creation: `_create_motion_planner()` at `:255` creates either `PortalMotionPlanner` (cuRobo) or `YamMotionPlanner` (RRT-Connect)
- Preloading: when `preload_motion_planner=True` and backend is `curobo`, the planner is warmed up at init time

The Viser Move-To panel supports:
- **Use motion planner** toggle
- **Planner backend** dropdown: `rrtconnect` / `curobo`
- **cuRobo finetune** checkbox
- **Solver speed** selector: `fast` / `slow`

### CAP agent tool integration

**File:** `cap/agent/tools/freespace_move.py`

The `FreespaceMoveTool` (`:79`) is the CAP agent's primary free-space movement tool.

Key defaults at `:66-76`:

```python
_DEFAULT_BACKEND = "curobo"                    # line 69
_DEFAULT_SOLVER_SPEED = "fast"                 # line 66
_DEFAULT_BATCH_TOP_K = 16                      # line 72
_DEFAULT_BATCH_SOLVER_SPEED = "fast"           # line 73
_DEFAULT_BATCH_VALIDATE_TRAJECTORY = False      # line 74
```

Tool parameters include: `backend` (default `"curobo"`), `solver_speed`, `grasp_candidates` for batched ranking, `batch_side`, `batch_top_k`, `trajectory_cache_key` for replay.

The tool description visible to the LLM agent:

```
"Move robot arm(s) to target end-effector pose(s) via collision-free motion planning.
 Defaults to cuRobo; use backend='rrt-connect' to override."
```

Referenced in prompts:
- `cap/bridge/system_prompt.py:74` -- "motion planning (cuRobo by default)"
- `cap/bridge/agent_bridge.py:108` -- "Move arm(s) to target pose via collision-free motion planning (cuRobo by default)"

### Control loop configuration

**File:** `experimental/yam_control_loop.py`

Config dataclass fields at `:240-252`:

```python
scripted_planner_solver_speed: Literal["slow", "fast"] = "fast"   # :244
motion_planner_backend: Literal["rrtconnect", "curobo"] = "curobo"  # :252
```

Depth collision is enabled via environment variable:

```python
# experimental/yam_control_loop.py:697
_depth_env = os.environ.get("CUROBO_ENABLE_DEPTH_COLLISION", "").strip().lower()
```

---

## 7. Remote serving

cuRobo runs on LeCAR-S1 as a Portal RPC server. See [remote_serving.md](remote_serving.md) for full details.

### Port assignment

| Service | Port | Host |
|---------|------|------|
| cuRobo Portal | `8611` | LeCAR-S1 (forwarded to local via SSH) |

### Launch scripts

| Script | Purpose |
|--------|---------|
| `tmux/remote_serving/launch_lecar_s1.sh` | Starts cuRobo (+ SAM3, BundleSDF, AnyGrasp) on S1. Pass `curobo` as a service argument. |
| `tmux/remote_serving/install_curobo_s1.sh` | Verifies existing cuRobo install; rebuilds if needed |
| `experimental/serve_portal_motion_planner.py` | CLI entrypoint: `--port 8611 --solver-speed fast` |
| `tools/remote/check_motion_planner_portal.py` | Health / roundtrip probe for the Portal server |

### Health check

```bash
# From S1:
uv run python tools/remote/check_motion_planner_portal.py --port 8611

# From 4070 through SSH forwarding:
uv run python tools/remote/check_motion_planner_portal.py --host 127.0.0.1 --port 8611
```

---

## 8. How to run

### Start scripted sim UI

```bash
cd /home/lecar/yiyang/lecar-tbd
uv run python -m experimental.yam_control_loop --use-scripted-policy --motion-planner-backend curobo
```

Then in the website:

- enable **Use motion planner**
- `Planner backend` should already default to **curobo**
- optionally toggle **cuRobo finetune**

### Start remote serving (real robot)

```bash
# On S1:
bash tmux/remote_serving/launch_lecar_s1.sh curobo

# On 4070 (local client):
bash tmux/table_bussing/launch_table_bussing_remote.sh
```

---

## 9. How to compare with the existing motion planner

### Compare in the website

Use the same Move-To target in the UI and switch:

- `rrtconnect`
- `curobo`

This is the fastest online comparison.

### Compare with benchmark

Run:

```bash
uv run python -m experimental.benchmark_rrt_vs_curobo \
  --side both \
  --start-state zero \
  --speed-samples 30 \
  --speed-warmup-samples 3
```

By default this uses:

- **cuRobo finetune OFF**
- **CUDA graph ON**

To compare with cuRobo finetune enabled:

```bash
uv run python -m experimental.benchmark_rrt_vs_curobo \
  --side both \
  --start-state zero \
  --speed-samples 30 \
  --speed-warmup-samples 3 \
  --enable-finetune
```

---

## 10. Useful commands

### cuRobo reachability benchmark

```bash
uv run python -m experimental.benchmark_curobo \
  --mode reachability \
  --side both \
  --start-state zero
```

### Position-oriented reachability with orientation sweep

```bash
uv run python -m experimental.benchmark_curobo \
  --mode reachability \
  --side both \
  --start-state zero \
  --orientation any_roll \
  --orientation-samples 16
```

### Use two orientations only

Default orientation + `-180,0,-90`:

```bash
uv run python -m experimental.benchmark_curobo \
  --mode reachability \
  --side both \
  --start-state zero \
  --orientation any_roll \
  --orientation-samples 2 \
  --orientation-extra-rpy-deg=-180,0,-90
```

### Plot benchmark PNGs

```bash
uv run python -m experimental.plot_curobo_benchmark artifacts/curobo_benchmark/<run_dir>
```

### Show reachability in Viser

```bash
uv run python -m experimental.plot_curobo_benchmark_viser artifacts/curobo_benchmark/<run_dir>
```

### Full cuRobo benchmark

Threshold vs time + single vs dual:

```bash
uv run python -m experimental.full_benchmark_curobo \
  --start-state zero \
  --speed-samples 30 \
  --speed-warmup-samples 3
```

### RRT vs cuRobo benchmark

```bash
uv run python -m experimental.benchmark_rrt_vs_curobo \
  --side both \
  --start-state zero \
  --speed-samples 30 \
  --speed-warmup-samples 3
```

---

## 11. Benchmark results

## Before enabling CUDA graph

Typical dual-arm default-threshold cuRobo result:

- mean: about **570 ms**
- p50: about **470 ms**
- p95: about **1.2 s**

This was with:

- finetune OFF
- CUDA graph OFF

## After enabling CUDA graph

Run:

- `artifacts/curobo_full_benchmark/20260327_190355_full_cuda_graph_on`

### Dual-arm, default threshold

- mean: **57.4 ms**
- p50: **49.7 ms**
- p95: **106.8 ms**

### Single vs dual

- left: **49.8 ms**
- right: **58.3 ms**
- both: **57.4 ms**

### Threshold sweep (dual-arm)

- strict: **57.8 ms**
- medium: **58.3 ms**
- default: **57.6 ms**
- loose: **53.7 ms**

Main conclusion:

> The main speed issue was CUDA graph being disabled. Turning it on gave about a 10x speedup.

---

## 12. Timing breakdown

For successful cuRobo plans with:

- CUDA graph ON
- finetune OFF

the main time is spent in:

- **IK**
- **trajopt**

Finetune time is `0` when disabled.

Earlier, when finetune was enabled, finetune was the dominant cost.

So:

- **finetune ON** -- better path quality, slower
- **finetune OFF** -- much faster, good for interactive use

---

## 13. Current defaults and recommendation

For interactive website usage and CAP agent, the current defaults are:

| Setting | Value | Source |
|---------|-------|--------|
| CUDA graph | ON | `motion_planner_curobo.py:60` |
| Graph search | ON | `motion_planner_curobo.py:61` |
| Torch compile | OFF (disabled) | `motion_planner_curobo.py:62` |
| Solver speed | `fast` | `motion_planner_curobo.py:64` |
| Finetune | OFF | `YamMotionPlannerCurobo.__init__` default |
| Batch capacity | 16 | `motion_planner_curobo.py:63` |
| Position threshold | 5 mm | `motion_planner_curobo.py:57` |
| Rotation threshold | 0.05 rad | `motion_planner_curobo.py:58` |
| Cspace threshold | 0.05 rad | `motion_planner_curobo.py:59` |
| Self-collision check | OFF | `motion_planner_curobo.py:440-441` |
| Collision checker type | MESH | `motion_planner_curobo.py:439` |
| Depth collision | OFF (opt-in via env var) | `yam_control_loop.py:697` |
| MuJoCo validation | ON | `YamMotionPlannerCurobo.__init__` default |
| Interpolation dt | 1/30 s | `motion_planner_curobo.py:56` |

This is the best practical setup for responsiveness (~50 ms p50 dual-arm planning).

---

## 14. Glossary

| Term | Meaning |
|------|---------|
| **MotionGen** | cuRobo's core motion generation class (IK + graph search + trajectory optimization) |
| **trajopt** | Trajectory optimization phase within cuRobo |
| **finetune** | Optional second-pass trajectory optimization for smoother paths |
| **CUDA graph** | CUDA graph capture for faster repeated kernel launches |
| **graph search** | RRT-like roadmap search used as warm-start for trajectory optimization |
| **Portal RPC** | `portal` library used for inter-process RPC communication |
| **collision spheres** | Lula-fitted spheres approximating robot link geometry for fast collision checking |
| **batch planning** | Planning multiple EE targets in one GPU call (used for grasp candidate ranking) |
