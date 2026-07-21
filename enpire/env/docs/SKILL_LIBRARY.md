# Skill Library — Overview

> **Cross-references**: [SKILL_LIBRARY_ROBOCASA](SKILL_LIBRARY_ROBOCASA.md) | [SKILL_LIBRARY_YAM](SKILL_LIBRARY_YAM.md) | [CAP_DESIGN](CAP_DESIGN.md) | [AGENT_PIPELINE_DESIGN](AGENT_PIPELINE_DESIGN.md)

---

## Overview

The CAP agent framework provides ~43 tools (skills) that LLM-generated code can call to control robots, perceive the scene, plan motions, and execute learned policies. This document is the master catalog. Per-environment pages cover env-specific behavior and recommended tool subsets.

**Key design**: All tools are registered globally regardless of environment. There is no per-env filtering at the registry level — instead, embodiment prompt specs (`cap/prompt/embodiment/`) document a reduced tool set to keep LLM context focused.

## Architecture

Tools flow through two execution paths:

```
Path A: CapServer mode (Portal RPC)
  Agent code → Tool class → Portal RPC → CapServer → Env

Path B: Direct mode (in-process)
  Agent code → Direct callable → Env methods (no RPC)
```

- **Path A** (`cap/agent/tools/__init__.py`): `ToolRegistry` registers `Tool` subclasses that communicate via Portal RPC to `cap_server`. Used for YAM (real + sim) and RoboCasa via CapServer.
- **Path B** (`cap/env/robocasa/skills.py` + `cap/agent/tools/direct.py`): Direct callables bypass RPC and call env/server methods in-process. Used for `run_agent.py --direct` mode in RoboCasa.

Both paths return the same dataclass types (`RobotState`, `FreespaceResult`, `Detection3D`, etc.) so agent code is portable.

## Data Types

All tools return structured dataclasses defined in `cap/agent/tools/base.py`:

### RobotState / ArmState

```python
@dataclass
class ArmState:
    joint_pos: list[float]   # 7 joint positions in radians
    gripper_pos: float       # 0.0 (closed) to 1.0 (open)
    ee_pos: list[float]      # [x, y, z] in meters, world frame
    ee_quat: list[float]     # [x, y, z, w] quaternion
    ee_rpy: list[float]      # [roll, pitch, yaw] in degrees

@dataclass
class RobotState:
    arms: dict[str, ArmState]  # keyed by side name ("left", "right")
    # Backward-compatible properties: state.left_ee_pos, state.right_joint_pos, etc.
```

### MoveResult

```python
@dataclass
class MoveResult:
    reached: bool              # whether EE converged to target
    feasible: bool = True      # whether controller found a valid path
    cmd_pos_err: float | None  # final position error in meters
    final_pos: list[float]
    final_quat: list[float]
```

### FreespaceResult

```python
@dataclass
class FreespaceResult:
    status: str          # "Success", "IK_Failed", "Planning_Failed", "Execution_Failed"
    ik_error_m: float
    final_pos_error_m: float
    final_rot_error_deg: float
    trajectory_steps: int
    executed: bool
    reason: str          # human-readable explanation when not Success
    side: str | None
    batch_candidates: list[FreespaceBatchCandidate]  # when using grasp_candidates mode
```

### Detection3D

```python
@dataclass
class Detection3D:
    label: str                         # object name
    score: float                       # confidence (1.0 for oracle)
    box_2d: list[float]                # 2D bounding box [x1, y1, x2, y2]
    position_3d: list[float]           # [x, y, z] in meters, world frame
    quaternion_xyzw: list[float]       # [x, y, z, w]
    rpy: list[float]                   # [roll, pitch, yaw] in degrees
    half_extents: list[float]          # bounding box half-sizes [dx, dy, dz]
    vis_b64: str | None                # annotated visualization (base64 JPEG)
```

### Other Types

| Type | Returned by | Key fields |
|------|-------------|------------|
| `SegmentationResult` | `segment_object` | mask, bbox_xywh, score, mask_area |
| `SkillResult` | `execute_skill`, `learn_skill` | success, steps_executed, info |
| `NudgeResult` | `nudge` | success, final_pos, final_quat |
| `FreespaceBatchCandidate` | `freespace_move` (batch mode) | rank, position, rpy, score, ik_error_m, trajectory_steps |

---

## Tool Catalog

### A. Robot State & Motion Control

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `get_robot_state` | Get joint positions, gripper states, EE poses for all arms | (none) |
| `_ik_servo` | Move EE to target [x,y,z] + orientation via IK/OSC. Blocking. `rpy` accepts 3-value RPY (degrees) or 4-value quaternion [x,y,z,w]. **YAM only** — RoboCasa uses `freespace_move`. | side, pos, rpy, gripper, max_duration_sec, max_vel, tol, check_feasibility, dry_run_only |
| `_ik_servo_keypoints` | Execute EE trajectory through timestamped waypoints. Blocking. **YAM only.** | side, timestamps, keypoints (each [px,py,pz,r,p,y] radians), max_vel |
| `move_joint_keypoints` | Execute joint-space trajectory via timestamped waypoints. Blocking. | side, timestamps, joint_positions, gripper_positions |
| `set_gripper` | Set gripper position. Blocking until settled. | side, pos (0=closed, 1=open), vel_limit, torque_limit |
| `open_gripper` | Fully open gripper. Shortcut for `set_gripper(side, 1.0)`. | side |
| `close_gripper` | Fully close gripper. Shortcut for `set_gripper(side, 0.0)`. | side |
| `go_home` | Move all arms to home configuration. Blocking. | (none) |

### B. Collision-Free Motion Planning

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `freespace_move` | Collision-free planning via cuRobo (default) or RRT-Connect. Supports single-arm, bimanual, and batched grasp candidate evaluation. Blocking. | left_target_pos, left_target_rpy, right_target_pos, right_target_rpy, grasp_candidates, planning_speed, backend, solver_speed, validate_trajectory |
| `nudge` | Small delta EE movement (position + orientation). Uses `_ik_servo` (YAM) or cuRobo planning (RoboCasa). | side, delta_pos [dx,dy,dz], delta_rpy [dr,dp,dy] degrees |

### C. Grasping & Manipulation

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `grasp` | High-level grasp: open → approach → descend → close → lift. Blocking. | side, position, rpy, pre_height, z_offset |
| `place` | High-level place: move above → descend → open → lift. Blocking. | side, position, rpy, pre_height |
| `sample_grasp_pose_anygrasp` | Generate grasp poses from point cloud via AnyGrasp. Returns ranked candidates. | object_points, grasp_from_top, k, timeout_s |

### D. Perception & Object Detection

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `detect_object` | Detect objects via BundleSDF (6-DOF tracking) or oracle (sim ground-truth). | query, camera, backend ("oracle" or "bundlesdf"), max_retries |
| `detect_objects_oneshot` | One-shot detection without tracking state. | query, camera, backend |
| `track_object` | Start real-time 6-DOF tracking via BundleSDF. Returns immediately. | query, camera, name |
| `get_object_pose` | Poll latest pose from active tracking session. | (none) |
| `stop_tracking` | Stop active BundleSDF tracking. | (none) |
| `segment_object` | Segment object via SAM3 text-prompted segmentation. | query, media, camera, score_thresh |
| `list_scene_objects` | Use Qwen3-VL to list all visible objects. | camera, prompt |

### E. Vision-Language Models

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `vlm_query` | Unified VLM interface — SmolVLM, Gemini, Gemini Pro, Qwen, GPT. | text, media, model, temperature, reasoning_effort |
| `smol_vlm` | Legacy SmolVLM query (deprecated — use `vlm_query`). | text, camera, image |

### F. Skill Execution & Policy

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `execute_skill` | Run a learned flow-matching policy skill. Blocking. | skill_name, params |
| `learn_skill` | Run RL training episode with remote policy server. Blocking. | skill_name, params (rl_host, rl_port, max_steps, output_dir, control_mode) |
| `start_policy_output` | Initialize external policy model session. | model, replan_horizon, task_description |
| `step_policy_output` | Execute one step of external policy. Blocking. | (none) |
| `stop_policy_output` | Halt policy session. | (none) |
| `use_policy_output` | One-shot wrapper: start → step until done → stop. | model, max_steps, task_description |

### G. Scene Management (Sim Only)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `setup_scene` | Load named scene into simulation. | name |
| `clear_table` | Remove all scene objects. | (none) |
| `list_scenes` | List available scenes. | (none) |
| `get_object_positions` | Get positions/orientations of all scene objects. | (none) |
| `set_body_pose` | Set a body's pose in simulation. | name, pos, quat_wxyz, gravity_comp |

### H. Task Management

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `get_task_info` | Get task state: reward, success, done, object positions, env name. | (none) |
| `load_task` | Load new task/scene (e.g. RoboCasa task). Resets env. | task_name |

### I. Camera & Image

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `get_camera_image` | Get latest RGB image from camera. | camera |
| `save_image` | Save image from media source to local file. | media, path, filename |

### J. BundleSDF Multi-Object Tracking

Lower-level multi-session tracking (alternative to `track_object`/`detect_object`).

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `add_detection` | Start 6-DOF tracking for named object. Async. | object, camera, name |
| `get_detection` | Poll latest pose from named session. | name |
| `end_detection` | Stop tracking for named session. | name |
| `list_detections` | List all active sessions. | (none) |

### K. Safety Zones

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `set_safety_zone` | Define safe EE workspace (convex hull of keyposes ± margins). | side, keyposes (7-D each), pos_margin, ori_margin |
| `get_safety_zone` | Query active safety zone. | (none) |
| `clear_safety_zone` | Clear safety zone(s). | side (optional) |

### L. Reward

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `setup_reward` | Set reward function for RL training. | mode (constant-0, constant-1, random, gemini, etc.) |

---

## Environment Comparison

| Aspect | RoboCasa (PandaOmron) | RoboCasa (GR1) | YAM (MuJoCo/Warp) | YAM (Hardware) |
|--------|-----------------------|----------------|---------------------|----------------|
| Arms | 1 (right) | 2 (left, right) | 2 (left, right) | 2 (left, right) |
| Cameras | top, wrist | top, wrist | top, left, right | top, left, right |
| Control freq | 20 Hz | 20 Hz | 60 Hz | 60 Hz |
| IK method | OSC_POSE (default) or joint_position (cuRobo) | OSC_POSE or joint_position | Pinocchio + pink IK | Pinocchio + pink IK |
| Gripper encoding | 0–1 (remapped from robosuite -1 to +1) | 0–1 | 0–1 | 0–1 |
| `detect_object` oracle | Yes (ground-truth from task info) | Yes | Yes (scene object positions) | No (vision only) |
| `freespace_move` | Available (cuRobo via remote cloud GPU) | Available | Available (cuRobo local or remote) | Available |
| Scene management | `load_task`, `get_task_info` | `load_task`, `get_task_info` | `setup_scene`, `clear_table` | N/A |
| Safety zones | Not typically used | Not typically used | Available | Available |
| `learn_skill` / RL | Supported | Supported | Supported | Supported |

## Key Files

- `cap/agent/tools/__init__.py` — `ToolRegistry`, `create_default_registry()` factory
- `cap/agent/tools/base.py` — `Tool` ABC, all result dataclasses
- `cap/agent/tools/direct.py` — Direct callable wrappers (bypass Portal RPC)
- `cap/env/robocasa/skills.py` — RoboCasa direct-mode namespace (cuRobo + env)
- `cap/prompt/embodiment/` — Per-robot LLM prompt specs (tool subsets)
- `cap/prompt/task/` — Per-task strategy guides
