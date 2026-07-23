# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import cv2
import numpy as np
from skill_library.constants.sorting import TABLE_SORT_RUN_CONFIG
from skill_library.namespace import (
    close_gripper,
    freespace_move,
    get_camera_extrinsics,
    get_camera_image,
    get_camera_intrinsics,
    get_robot_state,
    go_home,
    nudge,
    open_gripper,
    render_depth,
    rotate_joint,
    segment_all_objects,
)
from skill_library.pick import pick_object

from enpire.env.forge.cap.agent.tools._artifact_log import log_mask
from enpire.env.forge.cap.constants import (
    TOPDOWN_RPY,
)
from enpire.env.forge.cap.constants.planning import (  # per-move freespace speeds
    SPEED_MEDIUM,
    SPEED_MEDIUM_SLOW,
)

_move_to_rew_pose = load_module("ziptie/reward/_move_to_rew_pose.py")
move_to_rew_pose=_move_to_rew_pose["move_to_rew_pose"]
move_to_rew_pose_left=_move_to_rew_pose["move_to_rew_pose_left"]
move_to_rew_pose_right=_move_to_rew_pose["move_to_rew_pose_right"]
_rew = load_module("ziptie/reward/_compute_rew_rgb.py")
ZIPTIE_COLOR = _rew["ZIPTIE_COLOR"]
OBJECT_NAME, STRIP_LEN, SEG_TIMEOUT_S = f"{ZIPTIE_COLOR} strip", 0.20, 3.0
# Restrict the INITIAL top-camera SAM3 strip search to this region so it can't latch onto clutter
# outside the staging area. (x, y, w, h) on the top image -> [x0, y0, x1, y1] for image_bbox; the
# returned mask/world coords stay in full-image coordinates (only SAM3's input is masked).
STRIP_ROI_XYWH = (72, 114, 516, 185)
STRIP_ROI_BBOX = [STRIP_ROI_XYWH[0], STRIP_ROI_XYWH[1], STRIP_ROI_XYWH[0] + STRIP_ROI_XYWH[2], STRIP_ROI_XYWH[1] + STRIP_ROI_XYWH[3]]
# Wrist-camera ROI for the ziptie HEAD search (both wrists share this geometry): (x, y, w, h).
HEAD_ROI_XYWH = (177, 9, 284, 468)
HEAD_ROI_BBOX = [HEAD_ROI_XYWH[0], HEAD_ROI_XYWH[1], HEAD_ROI_XYWH[0] + HEAD_ROI_XYWH[2], HEAD_ROI_XYWH[1] + HEAD_ROI_XYWH[3]]
def _black_pad(rgb, bbox):
    """Copy of rgb with everything OUTSIDE [x0,y0,x1,y1] zeroed — exactly what SAM3 receives with
    image_bbox set. Used to LOG the cropped ROI (so the unselected region shows as black)."""
    x0, y0, x1, y1 = (int(v) for v in bbox)
    out = np.zeros_like(rgb); out[y0:y1, x0:x1] = rgb[y0:y1, x0:x1]; return out
MIN_GRASP_SITE=0.0 #0.1
MAX_GRASP_SITE=0.2 #0.2
RUN_CONFIG = dict(TABLE_SORT_RUN_CONFIG)
PLAN = {k: RUN_CONFIG[k] for k in ("planning_speed", "ik_error_threshold", "ik_xyz_weight", "ik_rpy_weight", "planner_backend") if k in RUN_CONFIG}
HEAD_FILTERS, HEAD_RANK = [{"metric": "C", "op": ">=", "value": 0.30}, {"metric": "P", "op": "<=", "value": 500}], {"metric": "C", "direction": "max"}
PITCH_FALLBACKS = (TOPDOWN_RPY[1], TOPDOWN_RPY[1] - 5.0, TOPDOWN_RPY[1] + 5.0)
def _retry(label, **kw):
    t = time.monotonic() + SEG_TIMEOUT_S
    while time.monotonic() < t:
        r = segment_all_objects(**kw)
        if r: return r[0]
        print(f"[transport] {label} not found, retrying...")
    raise RuntimeError(f"[transport] {label} not found after {SEG_TIMEOUT_S:.1f}s")
def _slide_topdown(side, xyz, yaw):
    for pitch in PITCH_FALLBACKS:
        try:
            freespace_move(planning_speed=SPEED_MEDIUM, **{f"{side}_target_pos": xyz, f"{side}_target_rpy": [TOPDOWN_RPY[0], pitch, yaw]}); print(f"[transport] {side} pitch={pitch:.0f}° yaw={yaw:.0f}° OK"); return
        except RuntimeError as e: print(f"[transport] pitch={pitch:.0f}° failed: {e}")
    raise RuntimeError(f"[transport] no top-down pose at {xyz}")

# 1. From top camera view, get the ziptie head's location and move closer arm to hover above it
_top0 = get_camera_image("top")  # log the black-padded strip ROI (exactly what SAM3 will search)
log_mask(_black_pad(_top0, STRIP_ROI_BBOX), np.zeros(_top0.shape[:2], bool), query=f"{ZIPTIE_COLOR} strip ROI (SAM3 input)", color="cyan", extra_text=f"image_bbox={STRIP_ROI_BBOX}", tag="strip_roi")
holding = pick_object(OBJECT_NAME, grasp_mode="2d", min_grasp_site=MIN_GRASP_SITE, max_grasp_site=MAX_GRASP_SITE, grasp_width=1.0, image_bbox=STRIP_ROI_BBOX, **RUN_CONFIG)
if holding is None: print("[transport] initial hover failed"); go_home(); raise SystemExit(1)

# 2. Detect and grab the ziptie head from table (# grab 0.5cm BELOW the head's LOWER bbox border (down the strip, under the head) instead of the centroid)
head = _retry(f"head of {ZIPTIE_COLOR} ziptie", query=f"head of {ZIPTIE_COLOR} ziptie", camera=holding, filters=HEAD_FILTERS, rank_by=HEAD_RANK, image_bbox=HEAD_ROI_BBOX)
hx, hy, hz = head.centroid_world_xyz
_st = get_robot_state(); _ap = getattr(_st, f"{holding}_ee_pos"); ax, ay, az = float(_ap[0]), float(_ap[1]), float(_ap[2]); _yaw_now = float(getattr(_st, f"{holding}_ee_rpy")[2])
_hm = np.asarray(head.mask, bool); _ys, _xs = np.nonzero(_hm); _vb = int(_ys.max()); _ub = int(np.median(_xs[_ys == _vb]))
_ix = get_camera_intrinsics(holding); _fx, _fy, _cx, _cy = ((_ix["fx"], _ix["fy"], _ix["cx"], _ix["cy"]) if isinstance(_ix, dict) else tuple(_ix[:4]))
_ex = get_camera_extrinsics(holding); _R = np.asarray(_ex["rotation"], float).reshape(3, 3); _t = np.asarray(_ex["position"], float)
if _ex.get("needs_optical_flip", True): _R = _R @ np.diag([-1.0, -1.0, 1.0])
_db = float(np.asarray(render_depth(holding), float)[_vb, _ub])
GRASP_SITE_BELOW_ZIPTIE_HEAD_BOUNDING_BOX=-0.008
if np.isfinite(_db) and _db > 0:                                                  # deproject border px, then push 0.5cm past it (away from centroid)
    _pb = np.array([(_ub - _cx) * _db / _fx, (_vb - _cy) * _db / _fy, _db]) @ _R.T + _t; _dir = _pb[:2] - np.array([hx, hy]); _dir = _dir / (np.linalg.norm(_dir) + 1e-9); hx, hy = float(_pb[0] + GRASP_SITE_BELOW_ZIPTIE_HEAD_BOUNDING_BOX * _dir[0]), float(_pb[1] + GRASP_SITE_BELOW_ZIPTIE_HEAD_BOUNDING_BOX * _dir[1])
_himg = _black_pad(get_camera_image(holding), HEAD_ROI_BBOX); cv2.drawMarker(_himg, (_ub, _vb), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2); cv2.circle(_himg, (_ub, _vb), 8, (0, 255, 0), 2)
log_mask(_himg, head.mask, query=f"head of {ZIPTIE_COLOR} ziptie (ROI)", color="red", extra_text=f"image_bbox={HEAD_ROI_BBOX}\ngrab 0.5cm below head lower border uv=({_ub},{_vb})\ngrab xy = ({hx:.3f}, {hy:.3f}) m  head z = {hz:.3f} m\n{holding} arm xyz = ({ax:.3f}, {ay:.3f}, {az:.3f}) m", tag="head_chosen")
_slide_topdown(holding, [hx-0.002, hy, az-0.006], yaw=_yaw_now); # !!!! offset for run_script to grab tightly & grab under the head.
close_gripper(holding, torque_limit=0.6)

# After the head grasp, directly move the LEFT arm to its final RL/handover pose.
print(f"[transport] {holding} arm grasped the ziptie head — moving LEFT arm to final pose", flush=True)

# Move down to the operator-dragged pose (recorded via gravity-comp, 2026-06-04) while closing gripper to 0.05.
open_gripper("right")
freespace_move(preview_only=False, **dict(PLAN, planning_speed=SPEED_MEDIUM_SLOW, ik_error_threshold=0.001, ik_rot_threshold_deg=0.1),
               left_target_pos=[0.452, 0.161, 1.166], 
               left_target_rpy=[27.1, 56.4, 54.9], 
               right_gripper_target_width=1.0)
freespace_move(preview_only=False, **dict(PLAN, planning_speed=SPEED_MEDIUM_SLOW, ik_error_threshold=0.001, ik_rot_threshold_deg=0.1),
               right_target_pos=[0.499, 0.048, 1.025], 
               right_target_rpy=[48.3, -75.6, 168.0], 
               right_gripper_target_width=1.0)
rotate_joint(rotations=[{"side": "left", "joint": 6, "delta_deg": -20}])
_rcs = get_robot_state()
freespace_move(preview_only=False, **dict(PLAN, planning_speed=SPEED_MEDIUM_SLOW, ik_error_threshold=0.001, ik_rot_threshold_deg=0.1),
               right_target_pos=list(getattr(_rcs, "right_ee_pos")), right_target_rpy=list(getattr(_rcs, "right_ee_rpy")), 
               right_gripper_target_width=0.0)

move_to_rew_pose_left(settle_s=1.5)
nudge("right", delta_pos=[0.0, -0.065, 0.07])
move_to_rew_pose_right(torque_limit=3.5)

