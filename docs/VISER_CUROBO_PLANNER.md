# Viser Interactive cuRobo Motion Planner

Standalone Viser 3D UI for interactive collision-free motion planning with cuRobo in RoboCasa.

**Related docs:**
- [`docs/CUROBO_ISAACSIM_SETUP.md`](CUROBO_ISAACSIM_SETUP.md) -- cuRobo setup, robot configs, collision spheres
- [`docs/ROBOCASA_INTEGRATION.md`](ROBOCASA_INTEGRATION.md) -- RoboCasa environment integration
- [`docs/VISER_IK_TELEOP_DESIGN.md`](VISER_IK_TELEOP_DESIGN.md) -- Viser gizmo teleop (YAM)

---

## Overview

`tools/viser_curobo_planner.py` provides a browser-based 3D interface for:

1. **Visualizing** the Panda arm (URDF), kitchen collision boxes, scene point cloud, and camera feed
2. **Setting targets** via a draggable 6-DOF transform gizmo
3. **Planning** collision-free trajectories via remote cuRobo (Portal RPC)
4. **Visualizing trajectories** as colored point clouds (blue -> red gradient)
5. **Executing** planned trajectories on the robot with live state polling

## Architecture

```
Browser (localhost:8890)
    |  Viser WebSocket
    v
tools/viser_curobo_planner.py
    |                       |
    | Portal RPC            | Portal RPC
    v                       v
cap_server.py            cuRobo server
(CAP_PORT, e.g. 18600)  (CAP_CUROBO_PORT, e.g. 8611)
```

The script is a **standalone client** -- it does not run inside the CAP sandbox. It connects to both the cap_server (for robot state, collision geometry, camera, FK) and the remote cuRobo planner (for motion planning).

## Usage

### Prerequisites

1. cap_server running with a RoboCasa environment
2. cuRobo remote server running (e.g. on LeCAR-S1 GPU box, port 8611)

### Launch

```bash
# Terminal 1: cap_server
CAP_ROBOT_TYPE=panda ROBOCASA_CONTROLLER_TYPE=joint_position \
uv run python -u cap/server/cap_server.py \
  --env robocasa:PickPlaceCounterToCabinet --port 18600

# Terminal 2: Viser planner
CAP_PORT=18600 CAP_CUROBO_PORT=8611 \
uv run python tools/viser_curobo_planner.py

# Open http://localhost:8890
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Viser server bind host |
| `--port` | `8890` | Viser server port |
| `--cap-port` | `$CAP_PORT` or `18600` | cap_server Portal RPC port |
| `--curobo-port` | `$CAP_CUROBO_PORT` or `8611` | cuRobo server Portal RPC port |
| `--cap-host` | `$CAP_HOST` or `127.0.0.1` | Host for both servers |
| `--side` | `right` | Arm side (PandaOmron: always `right`) |
| `--sam3-url` | `$SAM3_SERVICE_URL` or `http://127.0.0.1:6767` | SAM3 segmentation server URL |
| `--anygrasp-url` | `$ANYGRASP_SERVICE_URL` or `http://127.0.0.1:8122` | AnyGrasp grasp planning server URL |

## UI Elements

### 3D Scene

| Element | Scene Path | Description |
|---------|-----------|-------------|
| Floor grid | `/grid` | Reference grid |
| Collision boxes | `/collision/box_*` | Kitchen obstacles (semi-transparent blue) |
| Scene point cloud | `/scene/pointcloud` | Depth camera projection (colored) |
| Panda URDF | `/robot_base/panda/*` | Arm mesh at current joint config |
| EE sphere | `/robot/ee_sphere` | Green sphere at current EE position |
| EE frame | `/robot/ee_frame` | Axes at current EE pose |
| Target gizmo | `/target/gizmo` | 6-DOF draggable transform control |
| Trajectory | `/trajectory/points` | Planned path (blue->red point cloud) |
| Grasp candidates | `/grasps/candidate_*` | AnyGrasp grasp frames + colored spheres (green=feasible, red=failed, orange=selected) |

### GUI Panel

- **Camera**: Top + wrist camera feeds (320px, both update on refresh/execute)
- **Motion Planner**:
  - `Refresh State` -- re-fetch robot state, collision geoms, cameras, point cloud; sync cuRobo world
  - `Snap Gizmo to EE` -- move gizmo to current EE pose
  - `Plan` -- plan from current joints to gizmo target via cuRobo
  - `Execute` -- send planned trajectory with live 10Hz state polling
  - `Speed` slider -- execution speed (0.1-3.0 rad/s)
- **Target Pose**: Read-only position/quaternion of gizmo
- **Oracle Objects**: Queries `get_task_info()` from cap_server; shows task name, object names (from MJCF paths), and ground-truth positions
- **AnyGrasp Grasps**: Interactive grasp planning panel
  - Object Name text input + Camera dropdown (wrist/top)
  - `Get Grasp Poses` -- segments object via SAM3, plans 8 grasp candidates via AnyGrasp, tests each with cuRobo feasibility check, renders all in 3D
  - `Snap Gizmo to Grasp` -- moves the cuRobo target gizmo to the selected (best feasible) grasp pose

## Cap Server RPCs Used

| RPC | Purpose |
|-----|---------|
| `get_state(side)` | Current joint positions, EE pose, gripper |
| `get_collision_geoms()` | Kitchen obstacle boxes for cuRobo + Viser |
| `get_camera_image(camera)` | RGB image for GUI panel |
| `get_camera_depth(camera)` | Depth for scene point cloud |
| `get_camera_intrinsics(camera)` | `[fx, fy, cx, cy]` |
| `get_camera_extrinsics(camera)` | Rotation, position, optical flip flag |
| `forward_kinematics_batch(side, joint_positions)` | Batch FK for trajectory EE positions |
| `move_joint_keypoints(side, timestamps, positions, gripper)` | Trajectory execution |
| `get_task_info()` | Oracle task info: object names, positions, reward, success |

### External HTTP services used (AnyGrasp panel)

| Service | Endpoint | Purpose |
|---------|----------|---------|
| SAM3 | `POST /segment` | Text-prompted object segmentation |
| AnyGrasp | `POST /plan_viz` | 6-DOF grasp candidate planning |

### `forward_kinematics_batch` (new)

Added to cap_server for this tool. Computes FK for N joint configurations using a **separate `MjData` copy** to avoid racing with the 60Hz control loop.

```python
# Input:  side="right", joint_positions=(N, 7) array
# Output: {"ee_positions": (N,3), "ee_quats_xyzw": (N,4),
#          "base_pos": (3,), "base_quat_xyzw": (4,)}
```

## Coordinate Frames

- **Viser**: world frame, quaternions as **wxyz**
- **RoboCasa / scipy**: world frame, quaternions as **xyzw**
- **cuRobo**: arm-base frame, quaternions as **wxyz** (internally)
- **Conversion**: `pos_base = R_base_inv @ (pos_world - base_pos)`

The gizmo pose (Viser wxyz, world frame) must be converted to xyzw then to base frame before calling `plan_to_pose`.

## Key Files

| File | Role |
|------|------|
| `tools/viser_curobo_planner.py` | Main Viser UI script |
| `cap/server/cap_server.py` | `forward_kinematics_batch`, `get_collision_geoms`, `get_visual_meshes` RPCs |
| `experimental/portal_motion_planner.py` | `PortalMotionPlanner` client for cuRobo |
| `experimental/motion_planner_curobo_panda.py` | Panda cuRobo planner with `update_world_from_geoms` |
| `third_party/curobo/.../franka_description/panda.urdf` | Panda URDF for Viser visualization |
