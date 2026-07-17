# detect_object — 6-DOF Object Detection

Detects objects via BundleSDF 6-DOF pose tracking (default). Returns position and orientation (RPY in degrees).

## API

```python
dets = detect_object(query, camera="top", max_retries=3)
```

**Parameters:**
- `query` (str): Text description of the object (e.g., `"red cup"`, `"orange"`).
- `camera` (str, optional): `"top"`, `"left"`, or `"right"`. Default: `"top"`.
- `max_retries` (int, optional): Number of retry attempts on failure. Default: 3.

**Returns:** List of `Detection3D` (typically 1 result):

```python
@dataclass
class Detection3D:
    label: str              # echoes the query
    score: float            # confidence (0–1)
    position_3d: list[float]  # [x, y, z] in world frame (metres)
    rpy: list[float]          # [roll, pitch, yaw] in degrees
    half_extents: list[float] # bounding box half-sizes (metres)
```

## Usage Patterns

### Basic detection
```python
dets = detect_object("orange")
pos = dets[0].position_3d
rpy = dets[0].rpy
print(f"Orange at x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
print(f"  RPY: roll={rpy[0]:.1f}, pitch={rpy[1]:.1f}, yaw={rpy[2]:.1f}")
```

### Detection + oriented grasp
```python
dets = detect_object("yellow mustard bottle")
pos = dets[0].position_3d
rpy = dets[0].rpy

# Approach from above, then descend
freespace_move(right_target_pos=[pos[0], pos[1], pos[2] + 0.10], right_target_rpy=rpy)
freespace_move(right_target_pos=[pos[0], pos[1], pos[2] + 0.05], right_target_rpy=rpy)
close_gripper("right")
```

### Retry with more attempts for hard-to-detect objects
```python
dets = detect_object("small screw", max_retries=5)
if not dets:
    print("Object not found after 5 attempts")
```

### Wait for stable pose
```python
import time

prev_pos = None
for _ in range(10):
    dets = detect_object("red cup")
    pos = dets[0].position_3d
    if prev_pos is not None:
        delta = sum((a - b) ** 2 for a, b in zip(pos, prev_pos)) ** 0.5
        if delta < 0.002:  # < 2mm
            break
    prev_pos = pos
    time.sleep(0.3)

freespace_move(left_target_pos=[pos[0], pos[1], pos[2] + 0.05], left_target_rpy=dets[0].rpy)
```

### Post-grasp verification
```python
dets = detect_object("apple")
if not dets:
    print("Apple picked successfully!")
else:
    print("Apple still on table — retry")
```

## Tips

- **Default is BundleSDF** — no need to specify `backend`. Just call `detect_object(query)`.
- **Always use RPY** from `dets[0].rpy` for orientation. Never use quaternions.
- **Z-offset safety**: Add +0.05 m to detected Z before using as a movement target.
- **Re-detect before acting**: Positions change after robot moves. Don't cache old positions.
- **First call is slow** (~5-15s model loading). Subsequent calls are faster (~2-5s).
- **Score threshold**: Below 0.3 means tracking may have lost the object.
- **`stop_tracking()`**: Call when done with a tracking session to free GPU memory.
- **`max_retries`**: Default is 3. Increase for unreliable detections.
- **Prefer `camera="top"`** for widest view and best depth.
