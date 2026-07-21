# RoboCasa PickPlaceSinkToCounter — Vision Skill Guide

Strategy guide for `PickPlaceSinkToCounter` when the skill library is active.

> **IMPORTANT — no oracle positions.**
> Do NOT use `get_task_info()["obj_pos"]` or `get_task_info()["container_pos"]`
> to locate the object or target. Use SAM3 + depth for localization and
> AnyGrasp for grasp planning. The reference pipeline below is the required approach.

## Vision tools

| Tool | Key arguments | Returns |
|---|---|---|
| `get_task_description()` | — | `str` natural language task instruction |
| `detect_objects_oneshot(query, camera)` | `query: str`, `camera: str` | `dict[str, list[Detection]]`; each `det` has `.position_3d`, `.score` |
| `sample_grasp_pose_anygrasp(query, camera, max_grasps, top_down_only, disable_planner_z_clipping)` | all keyword | `list[Grasp]`; each `g` has `.position`, `.rpy`, `.score` |
| `select_best_grasp(grasps, side, batch_top_k, augment_yaw_flip)` | all keyword | `BatchGraspResult`; `.batch_candidates: list[Candidate]` |
| `create_motion_planner()` | — | planner handle |
| `update_planner_world(planner, exclude_body_prefixes)` | `planner`, `exclude: list[str]\|None` | `dict` with `n_obstacles` |
| `display_rpy_to_quat(rpy)` | `rpy: list[float]` | `list[float]` quaternion |

`Candidate` fields: `is_executable`, `is_ik_failed`, `is_planning_failed`,
`position`, `rpy`, `score`, `rank`, `source_index`, `ik_error_m`, `planner_status`.

`get_task_info()["obj_name"]` and `get_task_info()["success"]` are still valid —
only the XYZ position fields (`obj_pos`, `container_pos`, etc.) must not be used
for localization.

## Camera strategy

- **Object**: wrist camera — arm at home, wrist looks down into the sink
- **Target** (plate/counter): top camera first, wrist camera only if a closer view is needed. In PandaOmron RoboCasa, valid camera names are `top` and `wrist`; `right` is the arm side, not a camera.
- Detect target **once before** the attempt loop; it doesn't move
- Detect object **inside each attempt** after going home

## Parse task description

```python
import re
desc = get_task_description()
m = re.search(r"[Pp]ick (?:up )?(?:the )?(.+?) from .+ place (?:it )?(?:on |in |into )?(?:the )?(.+?)(?:\.|$)", desc)
obj_query, target_query = (m.group(1).strip(), m.group(2).strip()) if m else ("object", "counter")
```

## Detection helpers

```python
def detect_target_v1(target_query):
    """Detect place target (plate/counter) from valid PandaOmron cameras."""
    best_score, best_pos = -1.0, None
    for cam in ("top", "wrist"):
        for q in ["round plate", target_query]:
            dets = detect_objects_oneshot(q, camera=cam).get(q, [])
            if dets and dets[0].score > best_score:
                best_score, best_pos = dets[0].score, np.array(dets[0].position_3d, dtype=float)
    if best_pos is None:
        return False, {"success": False, "reason": "target not detected"}
    return True, {"success": True, "position": best_pos, "score": best_score}

def detect_object_wrist_v1(obj_query):
    """Detect object from wrist camera (arm at home, looking into sink)."""
    for q in [obj_query] + obj_query.split() + ["food", "object in sink", "object"]:
        dets = detect_objects_oneshot(q, camera="wrist").get(q, [])
        if dets:
            return True, {"success": True, "position": np.array(dets[0].position_3d, dtype=float), "query": q}
    return False, {"success": False, "reason": "object not detected"}
```

## Grasp planning and ranking

```python
def plan_and_rank_grasps_v1(obj_query, planner):
    """AnyGrasp on wrist camera + cuRobo batch rank. Returns feasible candidates."""
    import numpy as np
    grasps = []
    for gq in [obj_query] + obj_query.split() + ["food", "object"]:
        if grasps: break
        for tdonly in (True, False):
            try:
                grasps = sample_grasp_pose_anygrasp(gq, camera="wrist", max_grasps=10,
                                                    top_down_only=tdonly,
                                                    disable_planner_z_clipping=True)
            except RuntimeError:
                continue
            if grasps: break
    if not grasps:
        return False, {"success": False, "feasible": [], "reason": "no grasps"}
    info = update_planner_world(planner, exclude_body_prefixes=[""])
    result = select_best_grasp(grasps, side="right", batch_top_k=16, augment_yaw_flip=True)
    feasible = [c for c in (result.batch_candidates or []) if c.is_executable]
    return bool(feasible), {"success": bool(feasible), "feasible": feasible}
```

## Grasp execution

Key parameters (validated in human oracle):

```python
TCP_OFFSET_M         = 0.02
POST_GRASP_RETREAT_M = 0.08
GRASP_RECHECK_NUDGES = [-0.01, -0.02, -0.03]   # downward z on miss
```

Gripper width check: `0.05 < width < 0.95` means object is held.

```python
def execute_grasp_v1(feasible_candidates, planner):
    """Try top-3 ranked grasps; nudge-retry downward on miss."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    SIDE = "right"
    TCP_OFFSET_M = 0.02
    GRASP_RECHECK_NUDGES = [-0.01, -0.02, -0.03]

    def _check():
        w = float(np.asarray(get_robot_state().arms[SIDE].gripper_pos, dtype=float).reshape(-1)[0])
        return 0.05 < w < 0.95, w

    for candidate in feasible_candidates[:3]:
        grasp_pos = np.array(candidate.position, dtype=float)
        grasp_quat = display_rpy_to_quat(candidate.rpy)
        approach_dir = R.from_quat(grasp_quat).apply([0, 0, 1])
        r = freespace_move(right_target_pos=(grasp_pos + TCP_OFFSET_M * approach_dir).tolist(),
                           right_target_quat=grasp_quat.tolist(), side=SIDE)
        if r.status != "Success": continue
        import time; close_gripper(SIDE); time.sleep(0.25)
        grasped, _ = _check()
        if not grasped:
            for dz in GRASP_RECHECK_NUDGES:
                open_gripper(SIDE); nudge(side=SIDE, delta_pos=[0.0, 0.0, dz])
                close_gripper(SIDE); time.sleep(0.25)
                grasped, _ = _check()
                if grasped: break
        if grasped:
            return True, {"success": True, "rank": candidate.rank}
        open_gripper(SIDE)
        rp = np.array(get_robot_state().arms[SIDE].ee_pos, dtype=float)
        rp[2] += 0.08
        freespace_move(right_target_pos=rp.tolist(), side=SIDE, gripper=1.0)
    return False, {"success": False, "reason": "all candidates failed"}
```

## Lift, transit, place

Key parameters:

```python
LIFT_HEIGHT_M = 0.20
PLACE_HOVER_M = 0.12
PLACE_LOWER_M = 0.03
```

```python
def lift_and_transit_home_v1(home_pos, home_quat, planner):
    """Post-grasp retreat → restore collisions → lift → transit to home."""
    import numpy as np
    SIDE = "right"
    rp = np.array(get_robot_state().arms[SIDE].ee_pos, dtype=float)
    rp[2] += 0.08
    freespace_move(right_target_pos=rp.tolist(), side=SIDE, gripper=0.1)
    update_planner_world(planner, exclude_body_prefixes=["robot0", "gripper", "mobilebase", "obj"])
    lp = np.array(get_robot_state().arms[SIDE].ee_pos, dtype=float)
    lp[2] += 0.20
    freespace_move(right_target_pos=lp.tolist(), side=SIDE, gripper=0.1)
    freespace_move(right_target_pos=home_pos.tolist(), right_target_quat=home_quat.tolist(),
                   side=SIDE, gripper=0.1)
    return True, {"success": True}

def vision_place_v1(tgt_pos, planner):
    """Hover above target → lower → release → go_home."""
    import numpy as np, time
    from scipy.spatial.transform import Rotation as R
    SIDE = "right"
    ph = np.array(tgt_pos, dtype=float); ph[2] += 0.12
    down_quat = R.from_euler("xyz", [0, 180, 0], degrees=True).as_quat()
    for _, quat in [("vertical", down_quat),
                    ("current", np.array(get_robot_state().arms[SIDE].ee_quat, dtype=float))]:
        r = freespace_move(right_target_pos=ph.tolist(), right_target_quat=quat.tolist(),
                           side=SIDE, gripper=0.1)
        if r.status == "Success": break
    pt = np.array(tgt_pos, dtype=float); pt[2] += 0.03
    freespace_move(right_target_pos=pt.tolist(), side=SIDE, gripper=0.1)
    open_gripper(SIDE); time.sleep(0.40); go_home(SIDE)
    return True, {"success": True}
```

## Common failure modes

| Symptom | Fix |
|---|---|
| Detection returns empty | Try simpler fallback queries: individual words → `"food"` → `"object in sink"` |
| `sample_grasp_pose_anygrasp` raises | Try `top_down_only=True` first, then `False`; simpler query strings |
| No `is_executable` candidates | `go_home(SIDE)` first to reset arm pose, then retry |
| Gripper closes on air | Apply `GRASP_RECHECK_NUDGES` (-0.01, -0.02, -0.03 m downward) |
| Lift fails with colliding start state | Use `nudge_brutal` upward before lift to escape sink walls |
| Place hover IK fails with vertical orientation | Fall back to current EE orientation |
