# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick one blue cube using a conservative color mask and the standard CaP tools."""

import os
import time

import cv2
import numpy as np
from skill_library.namespace import (
    get_camera_extrinsics,
    get_camera_image,
    get_camera_intrinsics,
    get_robot_state,
    render_depth,
    sample_grasp_pose_2d,
)
from skill_library.pick import (
    _execute_grasp,
    _safe_move,
    _select_best_grasp,
    birdseye_pose,
    choose_arm,
)

from enpire.env.forge.cap.agent.tools.grasp_2d import (
    extract_segmented_object_world_points,
)


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _blue_components(rgb):
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    lower = np.array(
        [
            _env_int("ENPIRE_BLUE_H_MIN", 80),
            _env_int("ENPIRE_BLUE_S_MIN", 80),
            _env_int("ENPIRE_BLUE_V_MIN", 50),
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            _env_int("ENPIRE_BLUE_H_MAX", 100),
            _env_int("ENPIRE_BLUE_S_MAX", 255),
            _env_int("ENPIRE_BLUE_V_MAX", 255),
        ],
        dtype=np.uint8,
    )
    raw = cv2.inRange(hsv, lower, upper)

    height, width = raw.shape
    roi = np.zeros_like(raw)
    x0 = max(0, min(width, _env_int("ENPIRE_BLUE_ROI_X0", 220)))
    x1 = max(x0, min(width, _env_int("ENPIRE_BLUE_ROI_X1", 500)))
    y0 = max(0, min(height, _env_int("ENPIRE_BLUE_ROI_Y0", 0)))
    y1 = max(y0, min(height, _env_int("ENPIRE_BLUE_ROI_Y1", 330)))
    roi[y0:y1, x0:x1] = 255
    raw = cv2.bitwise_and(raw, roi)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    components = []
    for index in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[index]]
        if 300 <= area <= 2500 and 12 <= w <= 80 and 12 <= h <= 80:
            components.append((index, x, y, w, h, area))
    return labels, components


def _find_single_blue_cube(rgb):
    labels, components = _blue_components(rgb)
    if len(components) != 1:
        raise RuntimeError(
            "Blue-cube safety gate expected exactly one plausible component, "
            f"found {len(components)}"
        )

    index, x, y, w, h, area = components[0]
    print(f"  Blue component: bbox={[x, y, w, h]}, area={area}")
    return (labels == index).astype(np.uint8)


rgb = get_camera_image("top")
mask = _find_single_blue_cube(rgb)
depth = np.asarray(render_depth("top"), dtype=np.float32)
intrinsics = get_camera_intrinsics("top")
cam_k = np.asarray(intrinsics["K"], dtype=np.float64).reshape(3, 3)
extrinsics = get_camera_extrinsics("top")
t_cam_world = np.asarray(extrinsics["T_cam_world"], dtype=np.float64).reshape(4, 4)
object_points = extract_segmented_object_world_points(
    depth,
    cam_k,
    t_cam_world,
    mask,
)
if object_points.shape[0] < 100:
    raise RuntimeError(
        f"Blue-cube depth gate requires at least 100 points, got {object_points.shape[0]}"
    )
projection_z_m = float(np.median(object_points[:, 2]))
if not 0.75 <= projection_z_m <= 0.80:
    raise RuntimeError(
        f"Blue-cube depth gate rejected median world z={projection_z_m:.4f}"
    )
print(f"  Blue depth projection plane: z={projection_z_m:.4f} m")

grasps = sample_grasp_pose_2d(
    object_name="blue cube",
    camera="top",
    mask=mask,
    max_grasps=8,
    projection_z_m=projection_z_m,
    return_debug=False,
)
safe_grasps = [
    grasp
    for grasp in grasps
    if 0.42 <= float(grasp.position[0]) <= 0.78
    and -0.38 <= float(grasp.position[1]) <= 0.38
    and 0.76 <= float(grasp.position[2]) <= 0.85
]
if not safe_grasps:
    raise RuntimeError("No blue-cube grasp candidate passed the workspace safety gate")

picked_side = choose_arm(safe_grasps[0].position)
selected = _select_best_grasp(
    safe_grasps,
    picked_side,
    label="blue-cube color-2d",
    batch_top_k=8,
    solver_speed="fast",
    batch_validate_trajectory=False,
)
if selected is None:
    raise RuntimeError("cuRobo found no collision-free blue-cube grasp")
if not _execute_grasp(picked_side, selected, label="blue-cube color-2d"):
    raise RuntimeError("The gripper did not retain the blue cube")

lift_pos, lift_rpy = birdseye_pose(picked_side)
lifted = _safe_move(picked_side, lift_pos, lift_rpy)
if not lifted:
    raise RuntimeError("The blue cube was grasped but could not be lifted")

time.sleep(0.25)
post_lift_rgb = get_camera_image("top")
post_lift_depth = np.asarray(render_depth("top"), dtype=np.float32)
post_lift_labels, post_lift_components = _blue_components(post_lift_rgb)
post_lift_component_z = []
for component in post_lift_components:
    component_mask = (post_lift_labels == component[0]).astype(np.uint8)
    points = extract_segmented_object_world_points(
        post_lift_depth,
        cam_k,
        t_cam_world,
        component_mask,
    )
    if points.shape[0] >= 100:
        post_lift_component_z.append(float(np.median(points[:, 2])))

post_lift_state = get_robot_state()
post_lift_gripper = float(
    post_lift_state.left_gripper_pos
    if picked_side == "left"
    else post_lift_state.right_gripper_pos
)
if any(z < 0.85 for z in post_lift_component_z):
    raise RuntimeError(
        "Post-lift depth verification failed: a blue component remains at table height"
    )
if not any(z > 0.90 for z in post_lift_component_z):
    raise RuntimeError(
        "Post-lift depth verification failed: no blue component reached gripper height"
    )
if post_lift_gripper < 0.015:
    raise RuntimeError(
        "Post-lift gripper verification failed: the gripper closed without retaining the cube"
    )
print(
    "  Post-lift verification: "
    f"blue_z={[round(z, 4) for z in post_lift_component_z]}, "
    f"gripper={post_lift_gripper:.4f}"
)
verified = True


def get_task_info():
    """Result contract consumed by run_script.py."""
    return {
        "success": bool(verified),
        "reward": 1.0 if verified else 0.0,
        "picked_side": picked_side,
        "grasp_mode": "color-2d",
        "post_lift_gripper": post_lift_gripper,
        "post_lift_blue_z": post_lift_component_z,
    }
