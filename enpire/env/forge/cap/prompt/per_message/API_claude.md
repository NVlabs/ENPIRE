# CAP Sandbox API

## Planning-Time Inspection Policy

For any **real robot motion** request, do the read-only inspection before generating code:

- First call the planning inspection tool `get_robot_state`.
- If scene context matters, also inspect the relevant camera view(s) with `get_camera_image`.
- These planning-time inspection calls are **pre-approved**. Do **not** ask the user to allow them.
- Do **not** skip this inspection step for real-arm motion, even if the user asks to skip it.

## Orientation Constants (RPY in degrees)

```python
# Common grasp orientations as [roll, pitch, yaw] in degrees
# Home orientation: rpy=[0, 90, 0]
HOME_RPY = [0, 90, 0]
```

## Available Imports

```python
import numpy as np
import math
import time
import copy
import json
import random
import collections
import itertools
import functools
import dataclasses
import typing
```

## Functions

```python
# --- State ---
state = get_robot_state()                     # -> RobotState

# --- Motion (all blocking) ---
freespace_move(left_target_pos=None, left_target_rpy=None,
               right_target_pos=None, right_target_rpy=None,
               left_gripper=None, right_gripper=None,
               left_gripper_target_width=None, right_gripper_target_width=None,
               planning_speed=1.5)
# pos=[x,y,z] metres, rpy=[roll,pitch,yaw] degrees. Collision-free motion planning (cuRobo by default).
# left/right_gripper are for collision checking; left/right_gripper_target_width optionally
# commands synchronized gripper opening/closing during the same planned move.
# planning_speed is clamped to [0.05, 2.0] rad/s and controls the commanded max joint speed.
# IMPORTANT: for a single-arm move, ONLY pass that arm's target fields.
# Do NOT redundantly pass the other arm's current pose/state.right_ee_pos/state.right_ee_rpy
# (or the left-arm equivalents). Omit the inactive arm entirely unless you intentionally
# want a synchronized bimanual plan.

nudge(side, delta_pos=None, delta_rpy=None)
# delta_pos=[dx,dy,dz] metres, delta_rpy=[droll,dpitch,dyaw] degrees. World frame.

# --- Gripper (all blocking) ---
# pos: 1.0 = fully open, 0.0 = fully closed
set_gripper(side, pos, vel_limit=None, torque_limit=None)
open_gripper(side,               vel_limit=None, torque_limit=None)
close_gripper(side,              vel_limit=None, torque_limit=None)
go_home()                        # both arms to zero config

# --- Grasp & Place (high-level, blocking) ---
grasp(side, position, rpy=None, pre_height=0.10, z_offset=0.05)
# Approach from above, close gripper, lift. Uses detect_object position/rpy.
place(side, position, rpy=None, pre_height=0.15)
# Move above target, descend, open gripper, lift away.

# --- Perception ---
image = get_camera_image(camera)             # camera: "top"|"left"|"right" -> numpy uint8 RGB
dets  = detect_object(query, camera="top", max_retries=3)  # -> list[Detection3D]
# Default backend is BundleSDF (6-DOF pose). Auto-retries on failure.

# --- Vision-Language Model ---
text = vlm_query("list objects in the scene")                  # top cam, qwen backend
text = vlm_query("is the cup upright?")                        # custom question (top cam)
text = vlm_query("what does the gripper hold?", camera="left") # wrist cam
text = vlm_query("describe the scene", camera="all")           # all cameras

# --- Tracking ---
stop_tracking()   # Stop active BundleSDF session, free GPU memory

# --- Skills ---
result = execute_skill(skill_name, params={})    # -> SkillResult
result = start_policy_output(model, replan_horizon=30, task_description="", policy_server="")  # -> SkillResult
# Starts a policy session but does not send observations or move the robot yet.
result = step_policy_output(max_steps=None)  # -> SkillResult
# Executes a bounded policy burst from the active session.
result = stop_policy_output()  # -> SkillResult
# Clears the active policy session and local queued policy state.
result = use_policy_output(model, replan_horizon=30, task_description="", policy_server="")  # -> SkillResult

# --- Agent ---
wait_for_agent(message="")
# Pauses execution here; all variables are preserved.
```

## Return Types

```python
# RobotState
state.left_joint_pos     # list[float] length 6
state.left_gripper_pos   # float
state.left_ee_pos        # [x, y, z]  metres
state.left_ee_rpy        # [roll, pitch, yaw]  degrees
state.right_joint_pos
state.right_gripper_pos
state.right_ee_pos
state.right_ee_rpy

# Detection3D
det.label        # str — echoes query
det.score        # float 0-1
det.position_3d  # [x, y, z]  world frame metres
det.rpy          # [roll, pitch, yaw]  degrees
det.half_extents # [hx, hy, hz]  bounding box half-sizes

# SkillResult
result.success         # bool
result.steps_executed  # int
result.info            # dict
```
