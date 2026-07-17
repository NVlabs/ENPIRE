# RoboCasa — PandaOmron (single-arm, 7-DOF)

## Tool API

```python
def get_robot_state() -> RobotState:
    """Get current joint positions, gripper state, and EE pose.
    """

def freespace_move(right_target_pos: list[float], right_target_quat: list[float] = None, side: str = "right") -> FreespaceResult:
    """Move arm to target EE pose via cuRobo collision-free motion planning. Blocking — plans a joint trajectory, then executes it.
    right_target_pos: Target [x, y, z] in meters, world frame.
    right_target_quat: Target orientation [x, y, z, w] quaternion. Omit to keep current orientation.
    side: Arm side. Always 'right' for PandaOmron.
    """

def go_home(side: str = "right") -> FreespaceResult:
    """Drive arm back to its initial EE pose via cuRobo freespace planning.
    side: Arm side. Always 'right' for PandaOmron.
    """

def nudge(side: str, delta_pos: list[float] = None, delta_rpy: list[float] = None) -> NudgeResult:
    """Small delta EE move in world frame via cuRobo planning.
    side: Arm side. Always 'right' for PandaOmron.
    delta_pos: [dx, dy, dz] in metres. Omit for no translation.
    delta_rpy: [droll, dpitch, dyaw] in degrees. Omit for no rotation.
    """

def nudge_brutal(side: str, delta_pos: list[float], n_steps: int = 10) -> NudgeResult:
    """Collision-blind OSC nudge. Bypasses cuRobo entirely — no collision check.
    Use ONLY when the arm is in a colliding state and nudge/freespace_move refuse
    to plan ("Start state is colliding with world"). A small upward nudge_brutal
    can extract the arm before handing back to nudge or freespace_move.
    Do NOT use for large motions or in open space.
    side: Arm side. Always 'right' for PandaOmron.
    delta_pos: [dx, dy, dz] displacement in world frame, metres.
    n_steps: OSC ticks to spread the move over (default 10 ≈ 0.5 s).
    """

def set_gripper(side: str, pos: float) -> None:
    """Set gripper position. Blocking — waits until settled.
    side: Arm side. Always 'right' for PandaOmron.
    pos: 0.0 = fully closed, 1.0 = fully open.
    """

def open_gripper(side: str) -> None:
    """Fully open the gripper. Shortcut for set_gripper(side, 1.0).
    side: Arm side. Always 'right' for PandaOmron.
    """

def close_gripper(side: str, compliant: bool = False, hold_strength: float = 0.4) -> None:
    """Close the gripper.
    side: Arm side. Always 'right' for PandaOmron.
    compliant: When True, close stops as soon as fingers stall on an object
      (fast exit — ~0.5s instead of ~4.5s), then reduces sustained clamp
      force during subsequent arm motion. Use for fragile / wedge-shaped
      objects that can squirt out of a max-force parallel grip.
    hold_strength: Only meaningful with compliant=True. Clamp force level
      during subsequent motion. The internal decay quantises to 6 levels:
        0.0 → ~1 N  (object held only by static friction, may slip)
        0.2 → ~6 N  (gentle)
        0.4 → ~10 N (balanced, recommended default for most objects)
        0.6 → ~14 N (firmer grip)
        0.8 → ~18 N
        1.0 → 20 N  (same as compliant=False, max force)
      Start at 0.4; raise if object slips mid-motion, lower if the grip
      visibly squeezes the object out (wedge-squirt for tapered shapes).
    """

def get_gripper_info(side: str = "right") -> dict:
    """Detailed gripper state — use after close_gripper to check the grasp.
    Returns:
      {
        "pos":              float,         # [0, 1] normalised width; 0=closed
        "qpos_m":           float,         # raw finger qpos in metres
        "is_fully_closed":  bool,          # pos < 0.02 (empty close)
        "is_fully_open":    bool,          # pos > 0.98
        "commanded":        float | None,  # last set/open/close intent
        "has_object":       bool | None,   # compliant-close stall hit contact;
                                           #   True=grasped, False=empty close,
                                           #   None=non-compliant close (unknown)
        "actuator_force_N": float | None,  # current clamp force per finger
        "contact_bodies":   list[str],     # body names in contact with fingers
      }
    Reliable grasp check (preferred over `gripper_pos > 0.005`):
      info = get_gripper_info(side)
      grasped = (info["has_object"] is True
                 and not info["is_fully_closed"]
                 and (info["actuator_force_N"] or 0) > 1.0)
    """

def get_camera_image(camera: str) -> np.ndarray:
    """Get latest RGB image from a camera.
    camera: Camera name.  (one of ['top', 'wrist'])
    """

def get_camera_intrinsics(camera: str) -> list[float]:
    """Get camera intrinsic parameters [fx, fy, cx, cy].
    camera: Camera name.  (one of ['top', 'wrist'])
    """

def get_camera_extrinsics(camera: str) -> dict:
    """Get camera extrinsic parameters {rotation: 3x3, position: [x,y,z]}.
    camera: Camera name.  (one of ['top', 'wrist'])
    """

def detect_object(query: str, camera: str = 'top', backend: str = 'oracle') -> list[Detection3D]:
    """Detect objects in the scene. Use backend='oracle' for sim ground-truth positions (preferred in RoboCasa). Substring-matches query against scene object names.
    query: Object name or substring to search for.
    camera: Camera name (only used for vision backends, ignored by oracle).
    backend: 'oracle' for sim ground truth (recommended).
    """

def get_task_info() -> dict:
    """Get task state: {done, reward, success, env_name, obj_pos, container_pos, obj_name, ...}.
    Use this to check progress and get ground-truth object locations.
    """

def sample_grasp_pose_anygrasp(object_name: str, camera: str = "top", max_grasps: int = 10) -> list[GraspCandidate]:
    """Plan grasp poses using SAM3 segmentation + AnyGrasp (remote servers). Returns ranked grasp candidates.
    object_name: Name of object to grasp (used for segmentation).
    camera: Camera for RGB+depth capture.  (one of ['top', 'wrist'])
    max_grasps: Maximum number of grasp candidates to return.
    """

def update_planner_world(planner, exclude_body_prefixes: list[str] = None) -> dict:
    """Load collision geometry from the sim into a cuRobo planner. Returns {base_pos, base_quat_xyzw, n_obstacles}.
    planner: cuRobo planner instance.
    """

def vlm_query(text: str, camera: str = 'top') -> str:
    """Ask a vision-language model about the scene. Useful for spatial reasoning, counting objects, verifying grasp success.
    text: Natural language question about the scene.
    camera: Camera name, or 'all' for multi-camera composite.
    """

# Return types
# RobotState: Robot state container.
#   .arms — dict[str, ArmState] — keyed by arm name (just 'right' for PandaOmron)
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
# NudgeResult: Result from nudge.
#   .success — bool
#   .final_pos — list[float] — final EE [x, y, z]
#   .final_quat — list[float] — final EE [x, y, z, w]
# Detection3D: 3D detection result.
#   .label — str — object name
#   .score — float — confidence (always 1.0 for oracle)
#   .position_3d — list[float] — [x, y, z] in meters, world frame
# GraspCandidate: Grasp pose from AnyGrasp.
#   .position — list[float] — [x, y, z] in world frame
#   .rpy — list[float] — [roll, pitch, yaw] in degrees
#   .score — float — grasp quality score
#   .width — float — grasp width in metres
```

## Environment Notes

**Environment**: robocasa — PandaOmron
**Arms**: right
**Cameras**: top, wrist
**Control frequency**: 20 Hz
**Gripper range**: 0.0 (closed) to 1.0 (open)
**Motion planning**: cuRobo (cloud-served, collision-aware)

- Single arm — always use side='right'.
- Use `freespace_move()` for all arm movement — it plans collision-free trajectories via cuRobo. No external IK or motion planner setup needed.
- **Reuse `ee_quat` from `get_robot_state()` for casual moves** (tabletop repositioning, moving between open waypoints). For grasps and places, pick the orientation deliberately — see the approach-orientation table below.

## Approach orientations — pick deliberately for grasp / place

The Panda's reachable workspace depends heavily on wrist orientation. A top-down wrist cannot reach into a cabinet at z≈1.5m; a forward-pointing wrist can. When `freespace_move` returns `status=IK_Failed reason=IK Fail`, the target is almost always kinematically reachable with a different wrist orientation. The common approach archetypes (use `scipy.spatial.transform.Rotation` to build the quat — the URDF's precise reference frame varies, so don't hardcode quaternion components):

| Approach | Euler `[roll, pitch, yaw]` (deg, XYZ) | Reachable Z (world m) | When to use |
|---|---|---|---|
| Top-down | `[180, 0, 0]` | ~0.85 – 1.30 | Tabletop grasps, open counter placement |
| Forward-pointing (+X) | `[180, 0, 90]` or `[90, 0, 0]` | ~0.85 – 1.65 | **Cabinet / shelf / microwave placement, tall objects** |
| Side-approach (+Y) | `[180, 0, 180]` | ~0.80 – 1.55 | Objects against a wall, side-slotted cabinets |
| 45° tilt forward | `[180, 45, 0]` | ~0.80 – 1.50 | Objects leaning against a surface |

Convert to the quat `freespace_move` expects (xyzw):

```python
from scipy.spatial.transform import Rotation as R
quat = R.from_euler('xyz', [180, 0, 90], degrees=True).as_quat()  # (x, y, z, w)
r = freespace_move(right_target_pos=cabinet_pos, right_target_quat=quat.tolist(), side="right")
```

If IK fails with one orientation, perturb the Euler angles by 15-30° — not the target position. Rule of thumb: **vary orientation before you vary height**. The table's Reachable-Z ranges are empirical ballparks; trust the IK result over the table.

For authoring skills, prefer exposing a `target_quat` argument (default top-down), so callers can pick the approach per-task instead of inheriting whatever the prior motion left on the wrist.
- `get_task_info()['obj_pos']` gives ground-truth object XYZ. Prefer this over vision detection.
- Gripper: 0.0 = closed, 1.0 = open.
- **Prefer `close_gripper(side, compliant=True, hold_strength=0.4)`** for grasping in sim. It stall-detects contact (~0.5 s vs ~4.5 s for non-compliant) and caps clamp force so tapered/wedge-shaped objects don't squirt out. Raise `hold_strength` if the grip slips during motion; lower it if you visibly deform the object.
- **Verify grasps with `get_gripper_info(side)`**, not raw `gripper_pos`. The field `has_object` tells you whether compliant close stalled on contact; `actuator_force_N` tells you the current clamping force; `contact_bodies` lists what the fingers are touching. A robust check: `info["has_object"] is True and not info["is_fully_closed"] and (info["actuator_force_N"] or 0) > 1.0`.
- `freespace_move()` returns a `FreespaceResult` — check `.status == "Success"` before proceeding.
- numpy is available as `np` in the execution namespace.
