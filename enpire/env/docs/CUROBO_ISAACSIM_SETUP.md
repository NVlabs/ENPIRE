# cuRobo + Isaac Sim Setup

This document covers the YAM bimanual robot cuRobo integration: environment setup,
robot config files, URDF variants, collision sphere fitting, Isaac Sim USD generation,
the runtime motion planner architecture, and depth-based collision avoidance.

**Related docs:**
- [`docs/CUROBO_UPDATE_SUMMARY.md`](CUROBO_UPDATE_SUMMARY.md) -- cuRobo integration summary, benchmarks, and recommended defaults
- [`docs/CAP_DESIGN.md`](CAP_DESIGN.md) -- CAP agent architecture (freespace_move tool uses the planner)
- [`docs/TABLE_BUSSING_SKILLS.md`](TABLE_BUSSING_SKILLS.md) -- Table bussing skills that invoke freespace_move

---

## 1. Environments

### Production environment (recommended)

Use the repo venv. All commands assume you are in the repo root.

```bash
cd /home/<user>/yiyang/lecar-tbd   # or wherever the repo lives

# Sync repo dependencies
git submodule update --init --recursive
uv sync --extra cap

# Install cuRobo into the repo venv
uv pip install --python .venv/bin/python ninja
CUDA_HOME=/usr/local/cuda-12.8 \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0 \
uv pip install --python .venv/bin/python -e third_party/curobo --no-build-isolation

# Quick verification
uv run python - <<'PY'
import torch, curobo
print("cuda available:", torch.cuda.is_available())
print("curobo:", curobo.__file__)
PY
```

The install script at `tmux/remote_serving/install_curobo_s1.sh` automates this for
the lecar S1 server (verifies existing install, only rebuilds when needed).

### Isaac Sim standalone environment (for USD generation / Lula fitting only)

Keep Isaac Sim in a **separate** Python 3.11 venv if you need the Isaac Sim GUI for
USD conversion or Lula sphere fitting. This is not needed for runtime planning.

```bash
python3.11 -m venv ~/env_isaacsim
source ~/env_isaacsim/bin/activate
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install -e /home/<user>/Workspace/lecar-tbd/third_party/curobo --no-build-isolation
python -m pip install packaging==23.0 wheel==0.45.1
python -m pip check
```

---

## 2. Robot Model Files (URDFs)

All URDF variants live under `robot/models/station/`.

| File | Purpose | Finger joints |
|------|---------|---------------|
| `station.urdf` | Base kinematic URDF (no inertials/collisions) | prismatic |
| `station_physics.urdf` | Physics-enriched (inertials + collision geom from `station.xml`) | prismatic |
| `station_physics_fixed_fingers.urdf` | Physics-enriched with finger joints converted to `fixed` | fixed |
| `station_zed2itop.urdf` | Variant with ZED2i top-mounted camera link | prismatic |

There is also a `robot/models/station/loose_limit_version/` directory with a
`station.urdf` that has wider joint limits (for exploration / RL).

### URDF builder scripts

Located in `robot/models/station/curobo/`:

- **`build_station_physics_urdf.py`** -- Reads `station.urdf` + `station.xml` (MuJoCo),
  merges inertial and collision data to produce `station_physics.urdf`.
  (`robot/models/station/curobo/build_station_physics_urdf.py:1`)

- **`build_fixed_finger_urdf.py`** -- Converts the four finger prismatic joints to
  `fixed` type. Input: any station URDF. Default output: `station_fixed_fingers.urdf`.
  (`robot/models/station/curobo/build_fixed_finger_urdf.py:1`)

---

## 3. cuRobo Config Files

All configs live under `robot/models/station/curobo/`.

### Primary configs (dual-arm only)

The codebase has evolved to use **only dual-arm** configs. The single-arm configs
(`yam_left.yml`, `yam_right.yml`) referenced in the original doc **no longer exist**
as files in the repo. Single-arm planning is handled by sending targets for one arm
while keeping the other arm's joints at their current positions within the dual-arm planner.

| Config file | URDF | Collision spheres | Purpose |
|------------|------|-------------------|---------|
| `yam_dual.yml` | `station_physics.urdf` | external file `spheres/yam_dual.yml` | Canonical dual-arm config with lock_joints for finger prismatic joints |
| `yam_dual_isaacsim_physics.yml` | `station_physics.urdf` (via `robot/yam_station/` symlink) | inline | Isaac Sim bringup with physics URDF |
| `yam_dual_isaacsim_fixed_fingers.yml` | fixed-finger URDF (via `robot/yam_station/` symlink) | none (empty) | Isaac Sim bringup with fixed fingers, no collision |
| `yam_dual_isaacsim_physics_fixed_fingers.yml` | `station_physics_fixed_fingers.urdf` (via `robot/yam_station/` symlink) | inline | **Runtime default** -- physics URDF + fixed fingers + collision spheres |
| `yam_dual_isaacsim_physics_fixed_fingers_no_collision.yml` | same as above | none (null) | Diagnostic -- disables self-collision checking |

### Which config is used at runtime?

The cuRobo motion planner (`experimental/motion_planner_curobo.py:36-53`) defaults to:

- **YAML**: `robot/models/station/curobo/yam_dual_isaacsim_physics_fixed_fingers.yml`
- **URDF**: `robot/models/station/station_physics_fixed_fingers.urdf`

These are set via `_ROBOT_CFG` and `_URDF_PATH` constants at
`experimental/motion_planner_curobo.py:36-52`.

The no-collision variant (`yam_dual_isaacsim_physics_fixed_fingers_no_collision.yml`)
was kept as a fallback for the zero-start self-collision issue during early testing
(see `docs/CUROBO_UPDATE_SUMMARY.md` section 2).

### Isaac Sim configs use symlinked asset paths

The Isaac Sim helper configs reference assets via `robot/yam_station/...` paths.
This requires the symlink described in section 5 below.

### Collision sphere files

Sphere data lives under `robot/models/station/curobo/spheres/`:

- `yam_dual.yml` -- Collision spheres for dual-arm configuration
  (`robot/models/station/curobo/spheres/yam_dual.yml:1`)
- `README.md` -- Notes on expected sphere file naming

The `yam_dual.yml` config references these via `collision_spheres: "spheres/yam_dual.yml"`.
The Isaac Sim variants (`yam_dual_isaacsim_physics*.yml`) embed the sphere data
inline instead of referencing the external file.

---

## 4. Collision Sphere Layout

Both arms share a symmetric sphere layout (15 collision links, ~26 spheres total).

**Collision links** (from `yam_dual_isaacsim_physics_fixed_fingers.yml:22-36`):

```
left_arm, left_link_1..6
right_arm, right_link_1..6
right_left_link_finger
```

**Self-collision ignore pairs** (`yam_dual_isaacsim_physics_fixed_fingers.yml:115-128`):
Adjacent links are ignored (e.g. `left_arm` ignores `left_link_1`, etc.).

**Self-collision buffer**: all set to `0.0` (no extra padding).

**Collision sphere buffer**: `0.005` (5 mm global buffer on all spheres).

**Sphere radii summary**:
- Arm base / link_1: `0.045 m`
- link_2 (4 spheres along length): `0.04 m`
- link_3 (3 spheres): `0.04-0.045 m`
- link_4: `0.04 m`
- link_5: `0.038 m`
- link_6 (wrist, 3 spheres): `0.02-0.035 m`
- right_left_link_finger (2 spheres): `0.015-0.018 m`

---

## 5. YAM Asset Symlink (for Isaac Sim / USD generation)

The upstream `convert_urdf_to_usd.py` helper expects robot assets under cuRobo's
internal content tree. The Isaac Sim configs reference `robot/yam_station/...` paths.

Create the symlink:

```bash
mkdir -p third_party/curobo/src/curobo/content/assets/robot
ln -sfn $(pwd)/robot/models/station \
  third_party/curobo/src/curobo/content/assets/robot/yam_station
```

This is only needed for Isaac Sim USD generation. The runtime planner resolves
paths directly via absolute paths (see `_load_robot_cfg` at
`experimental/motion_planner_curobo.py:345-356`).

---

## 6. USD Generation

Generate USDs with the helper configs (requires Isaac Sim environment from section 1):

```bash
python third_party/curobo/examples/isaac_sim/util/convert_urdf_to_usd.py \
  --robot robot/models/station/curobo/yam_dual_isaacsim_physics_fixed_fingers.yml \
  --save_usd
```

Repeat with other configs as needed. The generated `.usd` files are referenced by
the `usd_path` field in the YAML configs but are **not used at runtime** --
the planner sets `use_usd_kinematics: false` and loads the URDF directly
(`experimental/motion_planner_curobo.py:350`).

---

## 7. Lula Sphere Fitting

Lula sphere fitting is done interactively in the Isaac Sim GUI. This is the
process that produced the sphere data in `robot/models/station/curobo/spheres/yam_dual.yml`.

### Procedure

1. Open Isaac Sim with the generated USD (from section 6).
2. Click **Play**.
3. Open **Tools > Robotics > Lula Robot Description Editor**.
4. Fit collision spheres per link.
5. Export the YAML.
6. Copy the resulting sphere data into:
   - `robot/models/station/curobo/spheres/yam_dual.yml` (external sphere file)
   - Update the inline `collision_spheres:` block in whichever Isaac Sim config
     files embed spheres directly (e.g. `yam_dual_isaacsim_physics_fixed_fingers.yml`)

### Current status

The current spheres are **hand-tuned conservative starter models**, not a full Lula
fit. They cover the arm links and one finger link (`right_left_link_finger`).
Left finger links are not included in the collision model.

---

## 8. Runtime Motion Planner Architecture

### Core planner class

`experimental/motion_planner_curobo.py:212` -- `YamMotionPlannerCurobo`

This is the GPU-accelerated cuRobo motion planner. Key features:
- Dual-arm planning with single-arm mode (side="left"|"right"|"both")
- Batch planning support (capacity default: 16)
- Two solver speed presets: `"fast"` and `"slow"` (`motion_planner_curobo.py:65-100`)
- Optional MuJoCo trajectory validation (`motion_planner_curobo.py:288-299`)
- CUDA graph acceleration (on by default)
- Graph search (on by default)
- Optional finetune trajopt (off by default for speed)
- Depth-based collision scene injection

### Constructor defaults (`motion_planner_curobo.py:57-64`)

```python
_DEFAULT_POSITION_THRESHOLD_M = 0.005       # 5 mm
_DEFAULT_ROTATION_THRESHOLD_RAD = 0.05      # ~2.9 deg
_DEFAULT_CSPACE_THRESHOLD_RAD = 0.05
_USE_CUDA_GRAPH_BY_DEFAULT = True
_ENABLE_GRAPH_SEARCH_BY_DEFAULT = True
_DEFAULT_BATCH_PLANNER_CAPACITY = 16
_DEFAULT_SOLVER_SPEED = "fast"
```

### World configuration

The planner builds its world model from the URDF at init time
(`motion_planner_curobo.py:358-428`). It parses fixed joints to find
world-obstacle links defined in `_WORLD_OBSTACLE_LINKS`:

```python
_WORLD_OBSTACLE_LINKS = ("gate_collision", "play_table")
```

These are extracted as cuboid obstacles from URDF collision geometry and added
to the cuRobo `WorldConfig`.

### Portal RPC server

`experimental/serve_portal_motion_planner.py` -- Wraps the planner in a Portal
RPC server for remote access.

`experimental/portal_motion_planner.py:27-33` -- `PortalMotionPlannerConfig` dataclass:
```python
backend: str = "curobo"
solver_speed: str = "fast"
port: int = 0
position_threshold: float = 0.005
rotation_threshold: float = 0.05
enable_depth_collision: bool = False
```

Default port: `8611` (set via `CAP_CUROBO_PORT` or `CAP_CUROBO_REMOTE_PORT`
env vars in `tmux/remote_serving/launch_lecar_s1.sh:27`).

### CAP agent integration

`cap/agent/tools/freespace_move.py:1-30` -- The `freespace_move` tool defaults to
cuRobo via `experimental.portal_motion_planner.PortalMotionPlanner`. It accepts
single-arm and bimanual targets in robot world frame coordinates.

---

## 9. Depth-Based Collision Avoidance

The planner supports injecting depth-camera point clouds as collision obstacles.

### Depth world module

`experimental/curobo_depth_world.py` provides:
- `point_cloud_from_depth()` -- Converts depth image + intrinsics to 3D point cloud
- `filter_depth_with_robot_mask()` -- Segments out robot pixels from depth using
  cuRobo's robot segmenter
- `transform_points()` -- Transforms points from camera frame to world frame
- `create_world_config_from_points()` -- Builds cuRobo `WorldConfig` mesh from points

### Planner depth integration

`experimental/motion_planner_curobo.py:540-586` -- `set_depth_collision_scene()`:
1. Optionally segments robot out of depth image
2. Converts depth to point cloud
3. Transforms to world frame
4. Creates mesh world config
5. Composes with static world obstacles
6. Updates all motion generators via `_refresh_world()`

The Portal RPC server exposes `set_depth_collision_scene` and
`clear_depth_collision_scene` endpoints for remote depth updates.

---

## 10. Testing

### Automated tests (no GPU required)

| Test file | What it tests |
|-----------|---------------|
| `tests/test_curobo_single_arm_regression.py` | Trajectory validation prefers fixed inactive arm for single-arm moves |
| `tests/test_motion_planner_curobo_unified.py` | Batch planning interface routing, batch capacity, result format |
| `tests/test_freespace_move_tool.py` | freespace_move tool integration |

### Manual tests (GPU required)

| Test file | What it tests |
|-----------|---------------|
| `tests/manual/check_single_arm_curobo_control_loop.py` | Full sim loop: reset env, create planner, plan + execute a move |

### Benchmarks

See `docs/CUROBO_UPDATE_SUMMARY.md` sections 3-9 for full benchmark commands and results.

| Script | Purpose |
|--------|---------|
| `experimental/benchmark_curobo.py` | Reachability and speed benchmarks |
| `experimental/full_benchmark_curobo.py` | Threshold sweep + single vs dual comparison |
| `experimental/benchmark_rrt_vs_curobo.py` | Head-to-head RRT-Connect vs cuRobo |
| `experimental/plot_curobo_benchmark.py` | Generate PNG plots from benchmark artifacts |
| `experimental/plot_curobo_benchmark_viser.py` | Interactive Viser 3D visualization of results |

Benchmark artifacts are saved to `artifacts/curobo_benchmark/`,
`artifacts/curobo_full_benchmark/`, and `artifacts/rrt_vs_curobo_benchmark/`.

---

## 11. Remote Serving (lecar S1)

The cuRobo planner runs on the GPU-equipped lecar S1 server and is accessed
via Portal RPC over SSH tunnels.

### Install on S1

```bash
bash tmux/remote_serving/install_curobo_s1.sh
```

This script (`tmux/remote_serving/install_curobo_s1.sh:1-79`):
1. Checks for existing healthy cuRobo install (skips rebuild if OK)
2. Installs `ninja` build dependency
3. Builds cuRobo editable install with CUDA 12.8
4. Verifies the install

### Launch on S1

```bash
# Launch cuRobo as part of the remote serving stack
bash tmux/remote_serving/launch_lecar_s1.sh curobo --curobo-port 8611
```

Or launch the full stack (SAM3, AnyGrasp, BundleSDF, cuRobo):

```bash
bash tmux/remote_serving/launch_lecar_s1.sh
```

The planner process runs:
```bash
.venv/bin/python experimental/serve_portal_motion_planner.py --port 8611 --solver-speed fast
```

### SSH tunnel (from local machine)

```bash
ssh -L 8611:127.0.0.1:8611 user@192.0.2.104
```

---

## 12. File Tree Reference

```
robot/models/station/
  station.urdf                          # Base kinematic URDF
  station.xml                           # MuJoCo XML (source for physics data)
  station_physics.urdf                  # Physics-enriched URDF
  station_physics_fixed_fingers.urdf    # Physics + fixed finger joints (RUNTIME DEFAULT)
  station_zed2itop.urdf                 # ZED2i top camera variant
  loose_limit_version/
    station.urdf                        # Wider joint limits (RL exploration)
  curobo/
    yam_dual.yml                        # Canonical dual-arm config (lock_joints, external spheres)
    yam_dual_isaacsim_physics.yml       # Isaac Sim + physics URDF + inline spheres
    yam_dual_isaacsim_fixed_fingers.yml # Isaac Sim + fixed fingers, no collision
    yam_dual_isaacsim_physics_fixed_fingers.yml         # RUNTIME DEFAULT config
    yam_dual_isaacsim_physics_fixed_fingers_no_collision.yml  # Diagnostic (no self-collision)
    build_station_physics_urdf.py       # Generates station_physics.urdf
    build_fixed_finger_urdf.py          # Converts finger joints to fixed
    spheres/
      yam_dual.yml                      # External collision sphere definitions
      README.md                         # Sphere file naming notes

experimental/
  motion_planner_curobo.py              # YamMotionPlannerCurobo class (main planner)
  curobo_depth_world.py                 # Depth-to-collision-world utilities
  portal_motion_planner.py              # Portal RPC client/server wrapper
  serve_portal_motion_planner.py        # CLI to launch Portal RPC planner server
  benchmark_curobo.py                   # Reachability + speed benchmarks
  full_benchmark_curobo.py              # Threshold + single/dual sweep
  benchmark_rrt_vs_curobo.py            # RRT vs cuRobo comparison
  plot_curobo_benchmark.py              # PNG plot generation
  plot_curobo_benchmark_viser.py        # Viser 3D visualization

tests/
  test_curobo_single_arm_regression.py  # Single-arm trajectory validation
  test_motion_planner_curobo_unified.py # Batch planning interface
  manual/
    check_single_arm_curobo_control_loop.py  # Full sim + plan loop

tmux/remote_serving/
  install_curobo_s1.sh                  # Automated cuRobo install for S1
  launch_lecar_s1.sh                    # Remote serving stack launcher

third_party/curobo/                     # cuRobo git submodule (editable install)
```

---

## 13. Common Configuration Fields

All cuRobo YAML configs share this structure (reference: `yam_dual_isaacsim_physics_fixed_fingers.yml`):

```yaml
robot_cfg:
  kinematics:
    use_usd_kinematics: false           # Always false at runtime
    urdf_path: "..."                    # Relative to asset_root or overridden to absolute
    asset_root_path: "..."              # Mesh/asset root directory
    base_link: "base_link"
    ee_link: "left_grasp"              # Primary EE (left arm)
    link_names: ["left_grasp", "right_grasp"]  # Both EE links
    lock_joints: null                   # Or dict of finger joints to lock values
    collision_link_names: [...]         # Links with collision spheres
    collision_spheres: {...}            # Inline dict or "spheres/yam_dual.yml"
    collision_sphere_buffer: 0.005      # 5 mm global buffer
    self_collision_ignore: {...}        # Adjacent link pairs
    self_collision_buffer: {...}        # Per-link buffer (all 0.0)
    mesh_link_names: [...]              # Links for mesh visualization
    cspace:
      joint_names: [left_joint1..6, right_joint1..6]  # 12 DoF
      retract_config: [...]             # Safe retract pose (12 values)
      null_space_weight: [1.0 x 12]
      cspace_distance_weight: [1.0 x 12]
      max_jerk: 500.0
      max_acceleration: 15.0
```

### Retract config (shared across all configs)

```
left:  [-0.3, 1.35, 1.6, -0.8, 0.3, -0.25]
right: [ 0.3, 1.35, 1.6, -0.8, -0.3, 0.25]
```

### lock_joints behavior

- `yam_dual.yml`: Locks all 4 finger joints to `0.0` (for physics URDF with prismatic fingers)
- Isaac Sim fixed-finger configs: `lock_joints: null` (fingers are already `fixed` type in URDF)
