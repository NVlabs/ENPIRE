# Coordinate System & EE Orientation Reference

All positions are in the **world frame** (metres). All orientations are **RPY [roll, pitch, yaw] in degrees**.

## World Frame Axes

- **+x**: Forward — from the robot base toward the play table
- **+y**: Left — from the robot's perspective (left arm side is +y, right arm side is -y)
- **+z**: Up — perpendicular to the floor

The floor is at z = 0. The play table surface is at z ~ 0.75.

## Key Landmarks (metres)

| Landmark | Position | Notes |
|----------|----------|-------|
| World origin | `[0, 0, 0]` | Center of the robot gate base, on the floor |
| Left arm base | `[0.2525, 0.31, 0.75]` | Mounted on gate, table height |
| Right arm base | `[0.2525, -0.31, 0.75]` | Mirror of left arm across y=0 |
| Play table surface | `x=0.6, z~0.75` | y spans +/-0.65 |
| Top camera | `[~0.086, ~-0.009, ~1.665]` | Mounted on top bar, looking down |
| Workspace bounds | `[-1, -1, -0.1]` to `[1, 1, 1.2]` | EE positions outside are rejected |

## Arm Sidedness

- **Left arm**: y > 0 side of workspace
- **Right arm**: y < 0 side of workspace
- **Center line**: y = 0 (both arms can reach)

Rule of thumb for arm selection:
- Object at `y >= -0.05` -> use left arm
- Object at `y < -0.05` -> use right arm

## RPY Orientation Reference

### Home orientation
At `go_home()`, the default RPY is `[0, 90, 0]` (degrees). Read via:
```python
state = get_robot_state()
state.left_ee_rpy    # left arm RPY in degrees
state.right_ee_rpy   # right arm RPY in degrees
```

**Best practice**: Read `state.left_ee_rpy` / `state.right_ee_rpy` at the start and reuse for standard grasps. Or use the object's detected RPY from `detect_object()`.

## Typical Z Heights

| Height (z) | What's there |
|------------|--------------|
| 0.0 | Floor — COLLISION ZONE, never target freespace_move here |
| 0.75 | Table surface / arm bases — COLLISION ZONE, freespace_move targets must be above this |
| 0.76-0.80 | Small objects on table (apple, orange, tape) |
| 0.80-0.95 | Tall objects on table (water bottle) |
| 0.95-1.00 | Hover above table for approach |
| 1.02-1.10 | Handover zone (arms meet at center) |
| 1.20 | Workspace ceiling (safety limit) |

**IMPORTANT**: `freespace_move` targets at or below z=0.75 will collide with the table and fail. Always keep targets above the table surface. Use `nudge()` for final descent to grasp height.

## Detection Coordinates

`detect_object()` returns `position_3d` in world frame and `rpy` in degrees. The z-coordinate is the **object surface**, not the center. Add +0.05 m to Z for safe approach targets.

## Camera Frame vs World Frame

`detect_object()` handles the camera-to-world transform automatically. You never need to do manual coordinate transforms.
