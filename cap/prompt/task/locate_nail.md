# Locate Nail — Offline Wrist-Camera Annotation Task

You are writing a Python script for an offline vision task. Do **not** move a robot. Do **not** call `go_home`, `freespace_move`, `get_camera_image`, or any live-camera/live-robot tool. You may use the study-mode offline perception helpers `segment_object(query, image=rgb, ...)` and `segment_all_objects(query, image=rgb, ...)`, which send the provided saved RGB array to the already-running SAM3 service. You may also call `vlm_query` only for the final Gemini reward critique.

## Goal

Implement a reusable function that locates the nail/pin and gripper tips in saved wrist-camera images. This function will later be reused to adjust a real pre-grasp pose.

The two fixed input images are loaded by:

```python
from cap.tasks.locate_nail import load_locate_nail_images

pairs = load_locate_nail_images()
```

The images are:

```text
/home/lecar/Project/lecar-tbd/logs/nail_bussing_20260426T170248/vis/170306_capture_wrist_cam_raw_left_round1_nail1.png
/home/lecar/Project/lecar-tbd/logs/nail_bussing_20260426T170248/vis/170417_capture_wrist_cam_raw_left_round2_nail1.png
```

Both are 640×480 RGB wrist-camera frames. The nail is the small bright metal pin near the image center. The two gripper tips are the inner black fingertip/pad tips on the left and right jaws.

## Required function

Define a top-level function with this exact signature:

```python
def locate_nail_annotation(label, rgb):
    ...
    return annotated_rgb, record
```

`rgb` is a `uint8` RGB numpy array. `annotated_rgb` must also be a `uint8` RGB numpy array. `record` must be a JSON-serializable dictionary containing at least:

```python
{
    "label": label,
    "bbox_xyxy": [x1, y1, x2, y2],
    "pin_midpoint_px": [cx, cy],
    "pin_lower_3_4_point_px": [gx, gy],
    "grasp_target_px": [gx, gy],
    "left_gripper_tip_px": [lx, ly],
    "right_gripper_tip_px": [rx, ry],
}
```

## Annotation requirements

On each annotated image:

1. Draw a tight bounding box around only the visible metal nail/pin. Do not include gripper fingers, table, or large extra margin.
2. Draw the lower 3/4 grasp target on the pin/nail body, i.e. the point 75% of the way from the upper/free visible end toward the lower/base end, so the robot grasps low and leaves the upper pin exposed for handover.
3. Draw a line connecting the two visible gripper-tip contact points.
4. Make marks visible with distinct colors and labels.

Use deterministic computer vision or geometry, optionally seeded by SAM3 masks from `segment_object(..., image=rgb)`. Do not ask Gemini to produce coordinates. Gemini is only for reward/critique after your annotation is complete.

## Recommended script structure

Use this harness pattern:

```python
import json
import cv2
import numpy as np

from cap.tasks.locate_nail import (
    evaluate_locate_nail_annotations,
    load_locate_nail_images,
    save_locate_nail_outputs,
)

TASK_INFO = {"success": False, "reward": 0.0, "status": "not_run"}

def locate_nail_annotation(label, rgb):
    # Your implementation here.
    return annotated_rgb, record

pairs = load_locate_nail_images()
annotated = {}
records = []
for label, rgb in pairs:
    ann, rec = locate_nail_annotation(label, rgb)
    annotated[label] = ann
    records.append(rec)

saved_paths = save_locate_nail_outputs(annotated, records)
TASK_INFO = evaluate_locate_nail_annotations(
    vlm_query,
    pairs,
    annotated,
    records,
    backend="nvidia",
    model="gcp/google/gemini-3.1-pro-preview",
)
TASK_INFO["saved_paths"] = saved_paths
print(json.dumps(TASK_INFO, indent=2))

def get_task_info():
    return TASK_INFO
```

## Implementation hints

The nail is the small high-contrast bright metal object near the center. It may be nearly vertical or slightly slanted. Good approaches include calling `segment_object("metal nail pin", image=rgb, score_thresh=0.05)` to get an initial mask, then refining the mask with thresholding for bright/low-saturation pixels near the image center, connected components, contours/min-area rectangles, and filtering by area/aspect ratio. The gripper tips are dark black pads/fingertips on the left and right sides; a good first approximation is to find dark connected components in left/right regions and choose the innermost points facing the nail.

Print the computed coordinates for each image. The next iteration will use Gemini's critique and your stdout to improve the function.
