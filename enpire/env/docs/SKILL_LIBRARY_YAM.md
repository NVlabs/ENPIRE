# Skill Library — YAM

> **Cross-references**: [SKILL_LIBRARY](SKILL_LIBRARY.md) | [CAP_DESIGN](CAP_DESIGN.md) | [TABLE_BUSSING_SKILLS](TABLE_BUSSING_SKILLS.md) | [SAFETY_ZONE_DESIGN](SAFETY_ZONE_DESIGN.md)

---

## Overview

YAM is a custom bimanual robot platform with two 6-DOF arms. Three env backends are available:

| Backend | Location | Physics | Rendering | IK | Use case |
|---------|----------|---------|-----------|-----|----------|
| **YAM MuJoCo** | `cap/env/yam_mujoco.py` | MuJoCo CPU | MuJoCo EGL | Pinocchio + pink | Default sim, development |
| **YAM Warp** | `cap/env/yam_warp.py` | MuJoCo Warp (GPU) | NVIDIA Warp BVH | Pinocchio + pink | Headless GPU machines, batch sim |
| **YAM Hardware** | `robot/yam/` | Real world | RealSense / ZED cameras | Pinocchio + pink | Physical robot |

All three expose the same tool interface to agent code.

## Robot Specifications

- **Arms**: 2 (left, right) — 6-DOF each (DaMiao motors via CAN bus)
- **Grippers**: Custom, 0.0 (closed) to 1.0 (open)
- **Cameras**: top (ZED 2i), left (D405 wrist), right (D405 wrist)
- **Control frequency**: 60 Hz
- **IK**: Pinocchio FK + pink differential IK for `_ik_servo`; cuRobo for `freespace_move`
- **Home configuration**: All joints at zero position

---

## Available Tools

All ~43 tools in the registry are available. The following are the primary tools for YAM tasks:

### Core Motion

| Tool | Notes |
|------|-------|
| `get_robot_state()` | Both arms always present in `state.arms["left"]` / `state.arms["right"]` |
| `_ik_servo(side, pos, rpy, ...)` | Uses Pinocchio + pink IK. RPY in degrees. Blocking. |
| `_ik_servo_keypoints(side, timestamps, keypoints, max_vel)` | Trajectory execution via pink IK. Keypoints are `[px,py,pz,r,p,y]` in radians. |
| `move_joint_keypoints(side, timestamps, joint_positions, gripper_positions)` | Direct joint-space trajectory. Linearly interpolated. |
| `set_gripper(side, pos)` | 0.0=closed, 1.0=open. Supports `vel_limit`, `torque_limit`. |
| `open_gripper(side)` / `close_gripper(side)` | Shortcuts. |
| `go_home()` | Moves both arms to zero configuration. |

### Collision-Free Planning

| Tool | Notes |
|------|-------|
| `freespace_move(left_target_pos, left_target_rpy, right_target_pos, right_target_rpy, ...)` | cuRobo (default) or RRT-Connect. Supports single-arm, synchronized bimanual, and batch grasp candidate evaluation. |
| `nudge(side, delta_pos, delta_rpy)` | Small delta move in world frame. `delta_rpy` in degrees (extrinsic XYZ). |

**Bimanual `freespace_move`**: Pass both `left_target_*` and `right_target_*` for synchronized bimanual motion. The planner considers self-collision between arms.

### Grasping

| Tool | Notes |
|------|-------|
| `grasp(side, position, rpy, pre_height, z_offset)` | Composite: open → approach → descend → close → lift. Uses `freespace_move` internally. |
| `place(side, position, rpy, pre_height)` | Inverse of grasp. |
| `sample_grasp_pose_anygrasp(object_points, grasp_from_top, k, timeout_s)` | AnyGrasp grasp pose generation from point cloud. Returns ranked candidates for `freespace_move` batch evaluation. |

### Perception

| Tool | Notes |
|------|-------|
| `detect_object(query, camera, backend)` | `backend='bundlesdf'` for 6-DOF pose tracking (stateful); `backend='oracle'` for sim ground-truth. |
| `track_object(query, camera, name)` | Start real-time BundleSDF tracking. Use `get_object_pose()` to poll. |
| `get_object_pose()` | Poll latest tracked pose (world frame). |
| `stop_tracking()` | Free GPU resources when done. |
| `segment_object(query, camera)` | SAM3 text-prompted segmentation. |
| `list_scene_objects(camera)` | Qwen3-VL scene object listing. |
| `vlm_query(text, media, ...)` | Multi-backend VLM queries. |

### Scene Management (Sim Only)

| Tool | Notes |
|------|-------|
| `setup_scene(name)` | Load YAML scene definition into MuJoCo sim. |
| `clear_table()` | Remove all objects. |
| `list_scenes()` | List available scenes + active scene. |
| `get_object_positions()` | Debug — all object poses. |
| `set_body_pose(name, pos, quat_wxyz, gravity_comp)` | Reposition objects. `gravity_comp=True` for floating. |

### Safety Zones (RL Training)

| Tool | Notes |
|------|-------|
| `set_safety_zone(side, keyposes, pos_margin, ori_margin)` | Define convex hull workspace constraint. Persists across `learn_skill` episodes. |
| `get_safety_zone()` | Query active zone config. |
| `clear_safety_zone(side)` | Remove zone. Omit `side` to clear both. |

### Skill Execution & RL

| Tool | Notes |
|------|-------|
| `execute_skill(skill_name, params)` | Run learned policy skill. |
| `learn_skill(skill_name, params)` | RL training episode with remote policy server. Supports Fello takeover (footswitch). |
| `start_policy_output` / `step_policy_output` / `stop_policy_output` | External policy model control loop. |
| `setup_reward(mode)` | Set reward mode before `learn_skill`. |

---

## Bimanual Coordination

YAM is bimanual — arm selection matters:

- **Y-axis heuristic**: `y > 0` → prefer left arm, `y < 0` → prefer right arm.
- **`freespace_move` bimanual**: Pass targets for both arms simultaneously. The cuRobo planner handles arm-arm collision avoidance.
- **Sequential pick-and-place**: Use one arm at a time, `go_home()` between picks.
- **Move the other arm out of the way** when working near center to avoid collisions.

## Table Bussing Pattern

The canonical YAM task — clear objects from a table into a container:

```python
go_home()
state = get_robot_state()
left_rpy = state.arms["left"].ee_rpy
right_rpy = state.arms["right"].ee_rpy

# Detect container
box_dets = detect_object("box")
box_pos = box_dets[0].position_3d

# For each object
dets = detect_object("orange")
obj = dets[0]
side = "left" if obj.position_3d[1] >= -0.05 else "right"
ee_rpy = left_rpy if side == "left" else right_rpy

open_gripper(side)

# Approach from above
hover = [obj.position_3d[0], obj.position_3d[1], obj.position_3d[2] + 0.15]
freespace_move(**{f"{side}_target_pos": hover, f"{side}_target_rpy": obj.rpy})

# Descend
grasp_pos = [obj.position_3d[0], obj.position_3d[1], obj.position_3d[2] + 0.05]
freespace_move(**{f"{side}_target_pos": grasp_pos, f"{side}_target_rpy": obj.rpy})

# Grasp, lift, drop
close_gripper(side)
freespace_move(**{f"{side}_target_pos": hover, f"{side}_target_rpy": obj.rpy})

box_dets = detect_object("box")  # re-detect in case it moved
drop = [box_dets[0].position_3d[0], box_dets[0].position_3d[1],
        box_dets[0].position_3d[2] + 0.25]
freespace_move(**{f"{side}_target_pos": drop, f"{side}_target_rpy": ee_rpy})

open_gripper(side)
go_home()
```

See `cap/prompt/task/table_bussing.md` for the full strategy with retry loops.

---

## RL Training Workflow

1. **Set safety zones** to constrain exploration:
   ```python
   set_safety_zone("right", keyposes=[...], pos_margin=0.05, ori_margin=0.3)
   ```
2. **Set reward function**:
   ```python
   setup_reward("gemini")  # or "constant-1", "insert_usb", etc.
   ```
3. **Run training episodes**:
   ```python
   result = learn_skill("insertion", params={
       "rl_host": "localhost", "rl_port": 5555,
       "max_steps": 200, "control_mode": "right",
   })
   ```
4. **Deploy learned skill**:
   ```python
   result = execute_skill("insertion", params={...})
   ```

---

## cuRobo Motion Planning

YAM uses cuRobo for collision-free trajectory planning (the `freespace_move` tool):

- **Default solver**: cuRobo with GPU acceleration (requires NVIDIA GPU)
- **Fallback**: `backend='rrt-connect'` for CPU-only planning
- **Solver speeds**: "fast" (fewer iterations, less accurate) or "slow" (more iterations, higher success)
- **MuJoCo validation**: Optional trajectory validation against MuJoCo physics model before execution
- **Batch mode**: Pass `grasp_candidates` to evaluate multiple grasp poses in one call — cuRobo plans all candidates and returns ranked results by IK feasibility

### World Collision Setup

cuRobo loads collision geometry from the sim:
- Static world: table, walls, shelves (loaded once at init)
- Dynamic obstacles: depth camera point clouds updated per-plan
- Self-collision: arm-arm avoidance for bimanual motions

---

## Hardware-Specific Notes

On physical YAM hardware (vs sim):

- **No scene management** — `setup_scene`, `clear_table`, etc. are sim-only.
- **No oracle detection** — must use BundleSDF or AnyGrasp for all perception.
- **Camera calibration required** — see `robot/calibrate_cameras_legacy.py` for ChArUco hand-eye calibration.
- **Fello leader arm for teleop** — `robot/fello/` provides teleoperation via DaMiao leader arms.
- **Safety zones recommended** — constrain RL exploration to safe workspace regions.
- **Control loop overruns** — physical hardware is sensitive to timing; see `docs/TODO_DEBUG_OVERRUN.md`.

## Key Files

- `cap/env/yam.py` — Base YAM env (delegates to SimBackend)
- `cap/env/yam_mujoco.py` — MuJoCo CPU backend
- `cap/env/yam_warp.py` — NVIDIA Warp GPU backend
- `cap/env/adapters/sim.py` — SimArmAdapter, SimCameraAdapter (CapServer glue)
- `cap/prompt/task/table_bussing.md` — Table bussing strategy guide
- `cap/agent/tools/freespace_move.py` — cuRobo motion planning tool (~90KB)
- `robot/yam/` — Hardware drivers (yam_controller, arm_server, kinematics)
- `robot/fello/` — Fello leader arm (teleop, gravity comp)
