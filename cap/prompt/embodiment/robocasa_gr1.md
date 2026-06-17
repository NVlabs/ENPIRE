# RoboCasa — GR1ArmsOnly (bimanual, 7-DOF per arm)

## Tool API

```python
def get_robot_state() -> RobotState:
    """Get current joint positions, gripper states, and EE poses for both arms.
    """

def freespace_move(right_target_pos: list[float], right_target_quat: list[float] = None, side: str = "right") -> FreespaceResult:
    """Move arm to target EE pose via cuRobo collision-free motion planning. Blocking — plans a joint trajectory, then executes it.
    right_target_pos: Target [x, y, z] in meters, world frame.
    right_target_quat: Target orientation [x, y, z, w] quaternion. Omit to keep current orientation.
    side: Arm side.  (one of ['left', 'right'])
    """

def set_gripper(side: str, pos: float) -> None:
    """Set gripper position on the specified arm. Blocking — waits until settled.
    side: Arm side.  (one of ['left', 'right'])
    pos: 0.0 = fully closed, 1.0 = fully open.
    """

def open_gripper(side: str) -> None:
    """Fully open the gripper. Shortcut for set_gripper(side, 1.0).
    side: Arm side.  (one of ['left', 'right'])
    """

def close_gripper(side: str, compliant: bool = False, hold_strength: float = 0.4) -> None:
    """Close the gripper.
    side: Arm side.  (one of ['left', 'right'])
    compliant: True → stop on contact (stall-detect, ~0.5 s) and cap
      sustained clamp force for subsequent motion. Recommended for
      fragile / wedge-shaped objects.
    hold_strength: With compliant=True, controls clamp force during motion.
      Quantised: 0.0→~1N, 0.2→~6N, 0.4→~10N (default), 0.6→~14N, 0.8→~18N,
      1.0→20N (= non-compliant). Raise if object slips, lower if it
      squirts out of a tapered grip.
    """

def get_gripper_info(side: str = "right") -> dict:
    """Detailed gripper state — use after close_gripper to verify grasp.
    Returns {pos, qpos_m, is_fully_closed, is_fully_open, commanded,
             has_object, actuator_force_N, contact_bodies}.
    ``has_object`` is True if compliant close stalled on contact.
    ``actuator_force_N`` is the live clamp force per finger.
    ``contact_bodies`` lists bodies touching the fingers.
    Robust grasp check:
      info = get_gripper_info(side)
      grasped = (info["has_object"] is True
                 and not info["is_fully_closed"]
                 and (info["actuator_force_N"] or 0) > 1.0)
    """

def get_camera_image(camera: str) -> np.ndarray:
    """Get latest RGB image from a camera.
    camera: Camera name.  (one of ['top', 'wrist'])
    """

def detect_object(query: str, camera: str = 'top', backend: str = 'oracle') -> list[Detection3D]:
    """Detect objects in the scene. Use backend='oracle' for sim ground-truth positions (preferred in RoboCasa). Substring-matches query against scene object names.
    query: Object name or substring to search for.
    camera: Camera name (only used for vision backends, ignored by oracle).
    backend: 'oracle' for sim ground truth (recommended), 'bundlesdf' for vision.
    """

def get_task_info() -> dict  # {done: bool, reward: float, success: bool, env_name: str, obj_pos: list[float], ...}:
    """Get task state from the environment: reward, success flag, done flag, and object positions. Use this to check progress and get ground-truth object locations.
    """

def vlm_query(query: str, camera: str = 'top') -> str:
    """Ask a vision-language model about the scene. Useful for spatial reasoning, counting objects, verifying grasp success, reading labels.
    query: Natural language question about the scene.
    camera: Camera name, or 'all' for multi-camera composite.
    """

# Return types
# RobotState: Robot state container.
#   .arms — dict[str, ArmState] — keyed by 'left' and 'right'
# ArmState: Per-arm state.
#   .joint_pos — list[float] — 7 joint positions in radians
#   .gripper_pos — float — 0.0 (closed) to 1.0 (open)
#   .ee_pos — list[float] — [x, y, z] end-effector position in meters, world frame
#   .ee_quat — list[float] — [x, y, z, w] quaternion
#   .ee_rpy — list[float] — [roll, pitch, yaw] in degrees
# FreespaceResult: Result from freespace_move / go_home.
#   .status — str — "Success", "IK_Failed", "Planning_Failed", or "Execution_Failed"
#   .final_pos_error_m — float — final position error in metres
#   .trajectory_steps — int — number of waypoints executed
# Detection3D: 3D detection result.
#   .label — str — object name
#   .score — float — confidence (always 1.0 for oracle)
#   .position_3d — list[float] — [x, y, z] in meters, world frame
#   .quaternion_xyzw — list[float] — [x, y, z, w]
#   .half_extents — list[float] — bounding box half-sizes [dx, dy, dz]
```

## Environment Notes

**Environment**: robocasa — GR1ArmsOnly
**Arms**: left, right
**Cameras**: top, wrist
**Control frequency**: 20 Hz
**Gripper range**: 0.0 (closed) to 1.0 (open)

- Bimanual robot — 'left' and 'right' arms available.
- Use `freespace_move()` for all arm movement — cuRobo handles collision-free motion planning.
- Reuse ee_quat from `get_robot_state()` to keep current orientation. Never compute quaternions manually.
- get_task_info()['obj_pos'] gives ground-truth object XYZ. Prefer this over vision detection.
- Gripper: 0.0 = closed, 1.0 = open.
- Increase max_duration_sec for long-range moves (>0.3 m): use 10–15 s.
- Use `freespace_move()` for all arm movement — it plans collision-free trajectories via cuRobo. Check `.status == "Success"` before proceeding.
- For bimanual tasks, plan which arm handles which subtask before writing code.
- numpy is available as `np` in the execution namespace.
