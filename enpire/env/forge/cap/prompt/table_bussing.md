# Table Bussing

Clear all objects from the table and place them into a container (box, bin, plate).

## Strategy

1. **Survey first**: Call `get_robot_state()` and `get_camera_image("top")` to understand the scene.
2. **Pick the right arm**: Use whichever arm is closer based on y-position:
   - `y > 0` -> prefer `left` arm
   - `y < 0` -> prefer `right` arm
3. **One object at a time**: Pick, place, then next.
4. **Re-detect before every pick**: Object positions shift. Never reuse stale positions.
5. **Verify after every place**: Re-detect. If still found, retry (up to 3 attempts).
6. **Use `go_home()` between picks**.

## Pick-and-Place Pattern

```python
go_home()
state = get_robot_state()
left_rpy = state.left_ee_rpy
right_rpy = state.right_ee_rpy

# Detect container
box_dets = detect_object("box")
assert len(box_dets) > 0, "No container found!"
box_pos = box_dets[0].position_3d

# For each object...
dets = detect_object("orange")
obj_pos = dets[0].position_3d
obj_rpy = dets[0].rpy

# Choose arm
side = "left" if obj_pos[1] >= -0.05 else "right"
ee_rpy = left_rpy if side == "left" else right_rpy

open_gripper(side)

# Approach from above
hover_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15]
freespace_move(**{f"{side}_target_pos": hover_pos, f"{side}_target_rpy": obj_rpy})

# Descend to grasp (Z safety offset)
grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.05]
freespace_move(**{f"{side}_target_pos": grasp_pos, f"{side}_target_rpy": obj_rpy})

# Grasp and lift
close_gripper(side)
freespace_move(**{f"{side}_target_pos": hover_pos, f"{side}_target_rpy": obj_rpy})

# Drop above container
box_dets = detect_object("box")
drop_pos = [box_dets[0].position_3d[0], box_dets[0].position_3d[1], box_dets[0].position_3d[2] + 0.25]
freespace_move(**{f"{side}_target_pos": drop_pos, f"{side}_target_rpy": ee_rpy})

open_gripper(side)
go_home()

# Verify
dets = detect_object("orange")
if not dets:
    print("Success!")
```

## Retry Loop

```python
for name, z_extra in objects:
    for attempt in range(1, 4):
        print(f"Picking: {name} (attempt {attempt}/3)")
        go_home()
        if pick_and_place(name, z_extra):
            break
    else:
        print(f"Failed to pick {name} after 3 attempts")
```

## Common Pitfalls

- **Don't forget to open the gripper** before descending to grasp.
- **Add Z offset** (+0.05 m) to detected positions for safe approach.
- **Re-detect after each operation** — positions shift.
- **Re-detect the container** before placing.
- **Move the other arm out of the way** when picking near center.
- **Use `stop_tracking()`** after all detection is done to free GPU.
