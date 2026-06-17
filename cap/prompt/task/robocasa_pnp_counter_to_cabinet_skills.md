# RoboCasa Pick-and-Place Counter→Cabinet — Skill Guide

Strategy guide for `PickPlaceCounterToCabinet` when the skill library is active.
The object sits on a counter surface; the target is inside a cabinet that must be
approached from the front opening, inserted, and retracted cleanly.

## Key facts

- **Single arm** — always `"right"` for PandaOmron.
- **Ground-truth positions** from `get_task_info()`:
  - `obj_pos` — object to pick (on counter, z ≈ 0.85–1.0 m)
  - `obj_name` — object name
  - `distr_cab_pos` — cabinet interior target position (preferred)
  - `cab_approach_pos` — cabinet front opening waypoint (use if available)
  - `cab_insert_pos` — position inside cabinet to place object (use if available)
  - `cab_settle_pos` — final resting position inside cabinet (use if available)
  - `cab_front_normal` — outward normal of cabinet front face (use if available)
- **Place target priority**: `cab_settle_pos` > `cab_insert_pos` > `distr_cab_pos`.
- **Success signal**: `get_task_info()["success"]`.
- **Retract requirement**: EE must be > 25 cm from placed object.

## Cabinet-specific strategy

Unlike sink-to-counter, the main challenge here is **cabinet insertion**:

1. The gripper must approach from the **front of the cabinet** (along `cab_front_normal`),
   not from directly above. Top-down approach will hit the cabinet ceiling.
2. After inserting, the arm must **retract back out** through the opening before going home.
3. Orientation matters — the gripper needs to face inward (`-cab_front_normal` direction).

Use `cab_front_normal` or estimate it as the vector from the cabinet interior to the
current EE position (flattened to XY, normalized).

## Grasp success verification

Object is on a flat counter — no collision issues during lift. Verify grasp by checking
whether the object rose with the arm using `get_task_info()["obj_pos"][2]` before and
after a small test nudge:

```python
info_before = get_task_info()
obj_z_before = info_before["obj_pos"][2]
close_gripper(side, compliant=True, hold_strength=hold_strength)
nudge(side=side, delta_pos=[0.0, 0.0, 0.04])   # test lift
info_after = get_task_info()
grasped = (info_after["obj_pos"][2] - obj_z_before) > 0.015
```

No `nudge_brutal` needed — counter objects don't have the sink-wall collision issue.

## Proven skill set for counter-to-cabinet

| Skill | Role |
|-------|------|
| `hover_above_v1` | Hover above object before grasp |
| `vertical_grasp_v1` or `incremental_grasp_v1` | Descend and close gripper |
| `lift_v1` or `lift_v2` | Lift object off counter |
| `cabinet_approach_v1` | Move to cabinet opening (approach_pos + orientation) |
| `cabinet_insert_v1` | Insert object along cabinet normal to insert_pos |
| `cabinet_settle_v1` | Nudge to settle_pos inside cabinet |
| `vertical_place_v1` | Fallback if no cab_insert_pos available |

## Skill: cabinet_approach_v1

```python
@skill
def cabinet_approach_v1(side, approach_pos, front_normal, hold_gripper=0.0):
    """Move to the cabinet opening with gripper oriented inward (facing -front_normal).

    front_normal: outward normal of cabinet front face (pointing away from cabinet).
    Returns: (success, {"status": str, "orientation_used": str})
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    def normalize(v):
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v

    def quat_from_direction(d):
        z = normalize(d)
        h = np.array([0., 0., 1.])
        if abs(np.dot(z, h)) > 0.95: h = np.array([1., 0., 0.])
        x = normalize(np.cross(h, z)); y = normalize(np.cross(z, x))
        return R.from_matrix(np.column_stack([x, y, z])).as_quat()

    fn = normalize(front_normal)
    # Gripper faces into cabinet (-front_normal), also try down and perpendiculars
    orient_candidates = [
        ("cab-in", quat_from_direction(-fn)),
        ("cab-out", quat_from_direction(fn)),
        ("down", quat_from_direction([0., 0., -1.])),
        ("current", np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)),
    ]
    approach_pos = np.array(approach_pos, dtype=float)

    for label, quat in orient_candidates:
        r = freespace_move(right_target_pos=approach_pos.tolist(),
                           right_target_quat=np.asarray(quat).tolist(), side=side)
        print(f"cabinet_approach: {label} -> {r.status}")
        if r.status == "Success":
            return True, {"status": r.status, "orientation_used": label}

    return False, {"status": "all_failed", "orientation_used": None}
```

## Skill: cabinet_insert_v1

```python
@skill
def cabinet_insert_v1(side, insert_pos, settle_pos=None, hold_gripper=0.0):
    """Move to insert_pos then optionally to settle_pos inside the cabinet.

    Returns: (success, {"insert_status": str, "settle_status": str})
    """
    import numpy as np

    insert_pos = np.array(insert_pos, dtype=float)
    state = get_robot_state()
    cur_quat = np.asarray(state.arms[side].ee_quat, dtype=float).tolist()

    r_insert = freespace_move(right_target_pos=insert_pos.tolist(),
                               right_target_quat=cur_quat, side=side)
    print(f"cabinet_insert: insert -> {r_insert.status}")

    settle_status = "skipped"
    if settle_pos is not None:
        settle_arr = np.array(settle_pos, dtype=float)
        r_settle = freespace_move(right_target_pos=settle_arr.tolist(),
                                   right_target_quat=cur_quat, side=side)
        settle_status = r_settle.status
        print(f"cabinet_insert: settle -> {r_settle.status}")

    success = r_insert.status == "Success"
    return success, {"insert_status": r_insert.status, "settle_status": settle_status}
```

## Common failure modes

- **Gripper hits cabinet ceiling**: approach orientation is wrong — must use `-front_normal`
  direction, not vertical. Use `cab_front_normal` from task_info.
- **Insert fails (IK)**: try offset positions slightly up/down along cabinet front_normal.
- **Object drops after release inside cabinet**: place_clearance needs to be 0 or slightly
  positive (object resting on cabinet floor, not floating).
- **Retract fails (Planning_Failed)**: arm is inside cabinet, can't plan back out.
  Use the same approach_pos as the waypoint and retract along `+front_normal` direction.
- **Success=False despite object in cabinet**: gripper must be > 25 cm away — must
  retract to approach_pos before go_home.
