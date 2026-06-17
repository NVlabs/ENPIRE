# BundleSDF Tracking — Continuous 6-DOF Pose

BundleSDF provides continuous 6-DOF pose tracking. It is the **default backend** for `detect_object()`.

## Quick Reference

```python
# Detect + get pose (BundleSDF is the default, no need to specify backend)
dets = detect_object("red cup")
pos = dets[0].position_3d   # [x, y, z] world frame, metres
rpy = dets[0].rpy           # [roll, pitch, yaw] in degrees

# Use position + orientation for movement
freespace_move(left_target_pos=[pos[0], pos[1], pos[2] + 0.05], left_target_rpy=rpy)

# Stop tracking when done to free GPU
stop_tracking()
```

## Key Points

- **BundleSDF is the default** — just call `detect_object(query)` without specifying backend.
- **All orientations are RPY** [roll, pitch, yaw] in degrees. Use `dets[0].rpy` directly.
- **Auto-retry**: `max_retries=3` by default. Failed detections are retried automatically.
- **One object at a time**: Detecting a new object auto-stops the previous tracking session.
- **Always call `stop_tracking()`** when done to free GPU memory.
- **Z-offset**: Add +0.05 m to detected Z for safe movement targets.

## Movement Pattern

```python
dets = detect_object("blue plate")
pos = dets[0].position_3d
rpy = dets[0].rpy

# 1. Coarse: freespace_move to approach position
freespace_move(right_target_pos=[pos[0], pos[1], pos[2] + 0.10], right_target_rpy=rpy)

# 2. Fine: nudge down to grasp
nudge("right", delta_pos=[0, 0, -0.05])

# 3. Grasp
close_gripper("right")

# 4. Cleanup
stop_tracking()
```
