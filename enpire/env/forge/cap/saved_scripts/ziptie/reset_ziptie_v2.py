# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reset {ZIPTIE_COLOR}-ziptie head: loop grasp + classify (distinguish.py); middle-top → flat.py,
same-side → side.py, else lift LIFT_M + release + restart. Bounded by MAX_ATTEMPTS."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from skill_library.constants.robot import LEFT_HOME_XYZ
from skill_library.constants.sorting import TABLE_SORT_RUN_CONFIG
from skill_library.namespace import (
    close_gripper,
    freespace_move,
    get_camera_extrinsics,
    get_camera_image,
    get_camera_intrinsics,
    get_robot_state,
    go_home,
    open_gripper,
    render_depth,
    rotate_joint,
    segment_all_objects,
)
from skill_library.pick import go_birdeye, pick_object

from enpire.env.forge.cap.agent.tools._artifact_log import log_mask
from enpire.env.forge.cap.config import TABLE_SURFACE_Z_M
from enpire.env.forge.cap.constants import (
    TOPDOWN_RPY,
    handedness_transform_ee,
    handedness_transform_joint,
)

# `reward/` isn't a sandbox-whitelisted top-level package, so pull its definitions
# in via CAP's injected load_module (path relative to saved_scripts/, returns a
# namespace dict — same mechanism _compute_rew_rgb.py uses for its viz sibling).
_rew = load_module("ziptie/reward/_compute_rew_rgb.py")
ZIPTIE_COLOR = _rew["ZIPTIE_COLOR"]

FREESPACE_PLANNING_SPEED_RAD_S = 2.0  # freespace_move planning_speed (clamped to [0.05, 3.0] rad/s by cuRobo)
JOINT_ROTATION_SPEED_DEG_S = float(np.degrees(FREESPACE_PLANNING_SPEED_RAD_S))  # rotate_joint speed_deg_s — kept in lockstep with the freespace speed so the whole script has one motion-speed knob
RUN = dict(TABLE_SORT_RUN_CONFIG, planning_speed=FREESPACE_PLANNING_SPEED_RAD_S); 
PLAN = {k: RUN[k] for k in ("planning_speed", "ik_error_threshold", "ik_xyz_weight", "ik_rpy_weight", "planner_backend") if k in RUN}
HEAD_F, HEAD_R = [{"metric": "C", "op": ">=", "value": 0.30}, {"metric": "P", "op": "<=", "value": 500}], {"metric": "C", "direction": "max"}
STRAP_F, STRAP_R = [{"metric": "AR", "op": "<=", "value": 0.5}], {"metric": "AR", "direction": "min"}
SEG_T, SETTLE, GW, HD, ND, GW_F, GH_F, MID_T, DZ_T = 3.0, 0.1, 0.0, 5, 8, 0.45, 0.45, 0.60, 3.0  # SETTLE cut 1.0→0.1: caller has already stopped motion before classify_head
COMFORT_XYZ, LIFT_M, STRIP_LEN, MAX_ATTEMPTS = [0.50, 0.0, TABLE_SURFACE_Z_M + 0.03], 0.12, 0.20, 5  # comfort: centerline @ table+3cm; lift: +15cm in place for not-allowed-pose drop
PITCH_FALLBACKS = (TOPDOWN_RPY[1], TOPDOWN_RPY[1] - 5.0, TOPDOWN_RPY[1] + 5.0)
ALLOWED_SIDES = {"left": ("left", "middle-top"), "right": ("right", "middle-top")}  # other modes → reset
MAX_AUTORESET=999

def _retry(label, **kw):
    end = time.monotonic() + SEG_T
    while time.monotonic() < end:
        if (r := segment_all_objects(**kw)): return r
    raise RuntimeError(f"[reset] {label} not found after {SEG_T:.1f}s")
_dil = lambda m, n: cv2.dilate(m.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=n).astype(bool)  # ~15x faster than the np.roll-OR chain for HxW boolean masks


def _slide_topdown(side, xyz, yaw):
    for pitch in PITCH_FALLBACKS:
        try:
            freespace_move(**{f"{side}_target_pos": xyz, f"{side}_target_rpy": [TOPDOWN_RPY[0], pitch, yaw]}); print(f"[reset] {side} pitch={pitch:.0f}° yaw={yaw:.0f}° OK"); return
        except RuntimeError as e: print(f"[reset] pitch={pitch:.0f}° failed: {e}")
    raise RuntimeError(f"[reset] no top-down pose at {xyz}")


def classify_head(holding, axis_shift_m: float = 0.01):
    """Segment head + strap on wrist d405, classify by strap-touching head-boundary
    balance (left/middle/right), refine middle via head/strap depth diff."""
    time.sleep(SETTLE)
    with ThreadPoolExecutor(max_workers=3) as ex:  # fire head SAM3 + strap SAM3 + depth fetch concurrently; wall clock = max(...) not sum
        f_h = ex.submit(_retry, "head", query="head of f{ZIPTIE_COLOR} ziptie", camera=holding, filters=HEAD_F, rank_by=HEAD_R); f_s = ex.submit(_retry, "strap", query="f{ZIPTIE_COLOR} strip", camera=holding, filters=STRAP_F, rank_by=STRAP_R); f_d = ex.submit(render_depth, holding)
        head, strap_cs, depth = f_h.result()[0], f_s.result(), np.asarray(f_d.result(), dtype=np.float64)
    H, W = np.asarray(head.mask).shape[:2]; ys, xs = np.indices((H, W))
    gz = (ys >= (1 - GH_F) * H) & ((xs < GW_F * W) | (xs >= W - GW_F * W))
    head_b = np.asarray(head.mask, dtype=bool) & ~gz; head_d = _dil(head_b.copy(), HD) & ~gz
    strap_b = ([s for s in (np.asarray(c.mask, dtype=bool) & ~gz for c in strap_cs) if (s & head_d).any()] or [None])[0]
    if strap_b is None: raise RuntimeError(f"no strap candidate intersects the head boundary (head_b={int(head_b.sum())}px, strap_cs={len(strap_cs)})")
    strap_c = strap_b & ~head_b; touching = (head_d & ~head_b) & strap_b
    h_cu = float(np.nonzero(head_b)[1].mean()); t_xs = np.nonzero(touching)[1]
    n_l, n_r = int((t_xs < h_cu).sum()), int((t_xs > h_cu).sum()); balance = (n_r - n_l) / max(n_l + n_r, 1)
    side = "middle" if abs(balance) < MID_T else ("left" if balance > 0 else "right")
    print(f"[reset] head is {side.upper()}  (touching={n_l+n_r}px left={n_l} right={n_r} balance={balance:+.2f} thresh ±{MID_T:.2f})")
    st = get_robot_state(); ap = getattr(st, f"{holding}_ee_pos")
    ax, ay, az, yaw_now = float(ap[0]), float(ap[1]), float(ap[2]), float(getattr(st, f"{holding}_ee_rpy")[2])
    depth_diff_mm = head_up_mm = strap_mm = strap_neck = None
    if side == "middle":
        strap_neck = strap_c & (_dil(head_b.copy(), ND) & ~head_b & ~gz); h_d, s_d = depth[head_b], depth[strap_neck]  # `depth` is already fetched above in the parallel block; no inline render_depth needed
        h_d, s_d = h_d[np.isfinite(h_d) & (h_d > 0)], s_d[np.isfinite(s_d) & (s_d > 0)]
        if h_d.size == 0 or s_d.size == 0: raise RuntimeError(f"no valid depth under head/strap masks (head={h_d.size}px strap_neck={s_d.size}px) — wrist d435 likely missing the thin strap; outer loop will recover")  # promote IndexError → RuntimeError so the recovery handler catches it
        head_up_mm, strap_mm = float(np.percentile(h_d, 10)) * 1000.0, float(np.median(s_d)) * 1000.0
        depth_diff_mm = strap_mm - head_up_mm; side = "middle-top" if depth_diff_mm > DZ_T else "middle-bottom"
        print(f"[reset] depth: head_upper={head_up_mm:.1f}mm strap={strap_mm:.1f}mm diff={depth_diff_mm:+.1f}mm (thresh {DZ_T:.1f}mm) → {side.upper()}")
    # Shift grasp 2cm along the ziptie's axis (head ← strap direction), away from the strap-attachment neck → less slip on closing.
    _intr = get_camera_intrinsics(holding); _extr = get_camera_extrinsics(holding)
    _fx, _fy, _cx, _cy = ((float(_intr["fx"]), float(_intr["fy"]), float(_intr["cx"]), float(_intr["cy"])) if isinstance(_intr, dict) else tuple(float(v) for v in np.asarray(_intr).reshape(-1)[:4]))
    if isinstance(_extr, dict) and "T_cam_world" in _extr: _Tcw = np.asarray(_extr["T_cam_world"], dtype=np.float64).reshape(4, 4)
    else: _R = np.asarray(_extr["rotation"], dtype=np.float64).reshape(3, 3); _t = np.asarray(_extr["position"], dtype=np.float64).reshape(3); _Tcw = np.eye(4, dtype=np.float64); _Tcw[:3, :3] = _R @ np.diag([-1.0, -1.0, 1.0]) if bool(_extr.get("needs_optical_flip", True)) else _R; _Tcw[:3, 3] = _t
    _sp = np.nonzero(strap_b); _sd = depth[strap_b][np.isfinite(depth[strap_b]) & (depth[strap_b] > 0)]
    _sz = float(np.median(_sd)) if _sd.size > 0 else float(np.median(depth[head_b][np.isfinite(depth[head_b]) & (depth[head_b] > 0)]) if head_b.any() else 0.5)
    _strap_w = (_Tcw @ np.array([(float(_sp[1].mean()) - _cx) * _sz / _fx, (float(_sp[0].mean()) - _cy) * _sz / _fy, _sz, 1.0], dtype=np.float64))[:3]
    _head_w = np.asarray(head.centroid_world_xyz, dtype=np.float64)
    _axis = _head_w - _strap_w; _ln = float(np.linalg.norm(_axis))
    _shift = (_axis / _ln * float(axis_shift_m)) if _ln > 1e-6 else np.zeros(3)
    hx, hy, hz = float(_head_w[0] + _shift[0]), float(_head_w[1] + _shift[1]), float(_head_w[2] + _shift[2])
    # Project the shifted (hx, hy, hz) back to image for the overlay circle.
    _Twc = np.linalg.inv(_Tcw); _gc = (_Twc @ np.array([hx, hy, hz, 1.0]))[:3]
    _g_u, _g_v = int(round(_fx * _gc[0] / _gc[2] + _cx)) if _gc[2] > 1e-6 else int(head.centroid_uv[0]), int(round(_fy * _gc[1] / _gc[2] + _cy)) if _gc[2] > 1e-6 else int(head.centroid_uv[1])
    rgb = get_camera_image(holding).copy(); rgb[gz] = 0
    rgb[strap_c] = np.clip(rgb[strap_c].astype(np.float32) * 0.55 + np.array([60, 180, 255], dtype=np.float32) * 0.45, 0, 255).astype(np.uint8)
    if strap_neck is not None and strap_neck.any(): rgb[strap_neck] = [255, 0, 255]
    rgb[touching] = [255, 255, 0]; cv2.line(rgb, (int(h_cu), 0), (int(h_cu), H - 1), (255, 255, 255), 1)
    cv2.circle(rgb, (_g_u, _g_v), 7, (0, 255, 0), 2); cv2.circle(rgb, (int(head.centroid_uv[0]), int(head.centroid_uv[1])), 4, (200, 200, 200), 1)   # green = chosen grasp site (centroid + 2cm along ziptie axis), gray = original centroid
    ex = (f"touching = {n_l+n_r} px   left = {n_l}   right = {n_r}\nbalance = {balance:+.2f} (thresh ±{MID_T:.2f})" + (f"\ndepth: head_upper = {head_up_mm:.1f}mm   strap = {strap_mm:.1f}mm   diff = {depth_diff_mm:+.1f}mm (thresh {DZ_T:.1f}mm)" if depth_diff_mm is not None else ""))
    threading.Thread(target=log_mask, args=(rgb, head.mask), kwargs={"query": f"head: {side.upper()}", "color": "red", "extra_text": ex, "tag": "reset_classify"}, daemon=True).start()  # async PNG save, off the hot path
    return side, hx, hy, hz, ax, ay, az, yaw_now


def reset_failed_grasp(holding, mode="drag"):
    """mode='drag' → COMFORT_XYZ (workspace centerline; bootstrap / RuntimeError recovery).
    mode='lift' → current xy + LIFT_M z (drop in place; not-allowed-pose recovery). Then release + bev."""
    st = get_robot_state(); ap = getattr(st, f"{holding}_ee_pos"); yaw_now = float(getattr(st, f"{holding}_ee_rpy")[2])
    target = COMFORT_XYZ if mode == "drag" else [float(ap[0]), float(ap[1]), float(ap[2]) + LIFT_M]
    print(f"[reset] {mode} {holding} grip → {[round(v, 3) for v in target]}, releasing + bev")
    _slide_topdown(holding, target, yaw=yaw_now)
    open_gripper(holding); go_birdeye(side="both", **RUN)


def run_flat_routine(holding):  # middle-top handoff: re-localize head → close → captured dual-arm handover → separation + inward roll
    open_gripper(holding)  # the strap drifts a few mm when the fingers open; classify_head's SETTLE pauses for it before segmenting
    _, hx, hy, _hz, _ax, _ay, az, yaw_now = classify_head(holding, axis_shift_m=0.01)   # flat mode: 1cm along ziptie axis
    _slide_topdown(holding, [hx, hy, az], yaw=yaw_now); close_gripper(holding)
    other, sign = ("right", +1) if holding == "left" else ("left", -1)
    # Captured dual-arm handover pose (left=holding, right=other; via gravity_comp drag
    HOLD_XYZ, HOLD_RPY = [0.419, 0.217, 1.235], [2.0, 29.0, 45.6]
    OTHER_XYZ, OTHER_RPY = [0.54, 0.0565, 1.20], [60.0, -86.9, 134.2]
    if holding == "right":
        HOLD_XYZ, HOLD_RPY = handedness_transform_ee(HOLD_XYZ, HOLD_RPY)
        OTHER_XYZ, OTHER_RPY = handedness_transform_ee(OTHER_XYZ, OTHER_RPY)
    open_gripper(other)
    freespace_move(preview_only=False, **PLAN, **{f"{holding}_target_pos": HOLD_XYZ, f"{holding}_target_rpy": HOLD_RPY, f"{other}_target_pos": OTHER_XYZ, f"{other}_target_rpy": OTHER_RPY, f"{other}_gripper_target_width": 1.0})
    time.sleep(1.0) 
    close_gripper(holding, torque_limit=0.8); 
    close_gripper(other, torque_limit=1.0); 
    time.sleep(0.5)   # settle, re-tighten holding, close other hard on the tail
    _xyz_topcam=[0.395+0.05, 0.071, 1.178]; _rpy_topcam=[11.8, 92.0, 60.5]; _xyz_other_topcam=[0.443+0.055, -0.069-0.01, 1.175]; _rpy_other_topcam=[-79.7, 39.0, 15.4]
    if holding == "right":
        _xyz_topcam, _rpy_topcam = handedness_transform_ee(_xyz_topcam, _rpy_topcam); _xyz_other_topcam, _rpy_other_topcam = handedness_transform_ee(_xyz_other_topcam, _rpy_other_topcam)
    freespace_move(preview_only=False, **PLAN, **{f"{holding}_target_pos": _xyz_topcam, f"{holding}_target_rpy": _rpy_topcam, f"{other}_target_pos": _xyz_other_topcam, f"{other}_target_rpy": _rpy_other_topcam})
    close_gripper(other, torque_limit=0.75)
    return hx, hy, az, yaw_now


def run_side_routine(holding):  # same-side handoff: holding to MID, other coarse + refined approach + grasp + synchronized roll
    # Adjustment (same vision-based re-grasp as flat mode): open + classify_head (returns hx, hy already shifted along the ziptie axis via the head←strap direction inside classify_head) + slide to shifted xy at current arm Z + close. Side mode uses a smaller 0.005m shift than flat mode's 0.01m — the side handoff cares about grip position more conservatively.
    open_gripper(holding)
    _, hx, hy, _hz, _ax, _ay, az, yaw_now = classify_head(holding, axis_shift_m=0.002)   # side mode: 5mm along ziptie axis
    _slide_topdown(holding, [hx, hy, az], yaw=yaw_now); close_gripper(holding)
    other, sign = ("right", +1) if holding == "left" else ("left", -1)
    MID_XYZ = [LEFT_HOME_XYZ[0] + 0.15, 0.0, 1.10]
    MID_RPY, TAIL_RPY = [sign * -90.0, 90.0, sign * 90.0], [sign * -90.0, 90.0, sign * -90.0]
    TAIL_XYZ_C = [MID_XYZ[0] - 0.20, MID_XYZ[1] - sign * 0.06, MID_XYZ[2] - 0.05]
    TAIL_XYZ_R = [MID_XYZ[0] - 0.20, MID_XYZ[1] - sign * 0.09, MID_XYZ[2] - 0.05]
    print(f"[transport] holding={holding} sign={sign} MID_RPY={MID_RPY} TAIL_RPY={TAIL_RPY}")
    print(f"[transport] {holding} -> middle xyz={MID_XYZ}")
    freespace_move(preview_only=False, **PLAN, **{f"{holding}_target_pos": MID_XYZ, f"{holding}_target_rpy": MID_RPY})
    print("[transport] pausing 1s so strip settles"); time.sleep(1.0)
    print(f"[transport] {other} open + coarse approach tail xyz={[round(v,4) for v in TAIL_XYZ_C]} rpy={TAIL_RPY}")
    open_gripper(other)
    freespace_move(preview_only=False, **PLAN, **{f"{other}_target_pos": TAIL_XYZ_C, f"{other}_target_rpy": TAIL_RPY, f"{other}_gripper_target_width": 1.0})
    freespace_move(preview_only=False, **PLAN, **{f"{other}_target_pos": TAIL_XYZ_R, f"{other}_target_rpy": TAIL_RPY})
    close_gripper(other, torque_limit=1.5)
    rotate_joint(rotations=handedness_transform_joint([{"side": holding, "joint": 6, "delta_deg": 150.0}, {"side": other, "joint": 6, "delta_deg": 110.0}], mirror=(holding == "right")), speed_deg_s=JOINT_ROTATION_SPEED_DEG_S)
    _st = get_robot_state(); _xyz = list(getattr(_st, f"{holding}_ee_pos")); _xyz[0] -= 0.1; _xyz[1]-=sign *0.01; _xyz[2] += 0.03; freespace_move(preview_only=False, **PLAN, **{f"{holding}_target_pos": _xyz, f"{holding}_target_rpy": list(getattr(_st, f"{holding}_ee_rpy"))})   # holding: x-=2cm, z+=2cm, rpy unchanged
    close_gripper(other, torque_limit=0.75)
    return hx, hy, az, yaw_now


for auto_reset_itr in range(MAX_AUTORESET):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n[reset] ===== attempt {attempt}/{MAX_ATTEMPTS} =====")
        go_birdeye(side="both", **RUN)  # INSPECTION grasp: pick closer arm + classify orientation only
        holding = pick_object(f"{ZIPTIE_COLOR} strip", grasp_mode="2d", min_grasp_site=0.10, max_grasp_site=0.20, grasp_width=GW, **RUN)
        if holding is None:  # 10-20% site may exclude an off-table strip; bootstrap with any-site grasp + drag to comfort so next attempt's restricted pick can succeed
            print("[reset] 10-20% restricted pick failed; bootstrap unrestricted grasp + drag to comfort"); holding = pick_object(f"{ZIPTIE_COLOR} strip", grasp_mode="2d", grasp_width=GW, **RUN)
            if holding is None: print("[reset] unrestricted pick also failed, retrying"); continue
            reset_failed_grasp(holding); continue
        # Head not detected → comfort + rotate j6 + open + bev (strap settles in a different
        # orientation, hopefully exposing the head), then outer loop re-grasps fresh.
        try:
            side, *_ = classify_head(holding)
        except RuntimeError as e:
            print(f"[reset] head not detected: {e} — comfort + rotate j6 + open + bev, then outer retry")
            _yaw = float(getattr(get_robot_state(), f"{holding}_ee_rpy")[2]); _slide_topdown(holding, COMFORT_XYZ, yaw=_yaw)  # 1. comfort (keeps strap from tangling during rotation)
            _j6 = np.degrees(float(getattr(get_robot_state(), f"{holding}_joint_pos")[5]))
            rotate_joint(rotations=[{"side": holding, "joint": 6, "delta_deg": (120.0 - _j6) if abs(120.0 - _j6) > abs(120.0 + _j6) else -(120.0 + _j6)}], speed_deg_s=JOINT_ROTATION_SPEED_DEG_S)  # 2. rotate j6 toward the larger remaining range
            open_gripper(holding); go_birdeye(side=holding, **RUN); continue  # 3. release, 4. bev, 5. outer retry re-grasps
        try:
            if side not in ALLOWED_SIDES[holding]: print(f"[reset] inspection side {side.upper()} not allowed for {holding} hand — lift+release in place"); reset_failed_grasp(holding, "lift"); continue
            if side == "middle-top":
                print(f"[reset] running FLAT routine for {holding} hand"); hx, hy, az, yaw_now = run_flat_routine(holding)
            else:
                print(f"[reset] running SIDE routine ({side}) for {holding} hand — keeping inspection grip"); hx, hy, az, yaw_now = run_side_routine(holding)
            print(f"[reset] done after {attempt} attempt(s)"); break
        except RuntimeError as e:
            # IK/routine failure (e.g. cuRobo "IK did not converge"): keep grip closed, slide to
            # comfort first so the strap doesn't get torn at the failed pose, THEN go_birdeye
            # (still holding) so the strap is lifted clear, THEN release at bev, then outer retry.
            print(f"[reset] caught during routine: {e} — comfort + bev (keep grip) + release + outer retry")
            try: _slide_topdown(holding, COMFORT_XYZ, yaw=float(getattr(get_robot_state(), f"{holding}_ee_rpy")[2]))
            except RuntimeError as e2: print(f"[reset] comfort slide also failed: {e2} — proceeding to bev anyway")
            go_birdeye(side="both", **RUN); open_gripper(holding); continue
    else:
        print(f"[reset] gave up after {MAX_ATTEMPTS} attempts"); go_home(); raise RuntimeError(f"[reset] gave up after {MAX_ATTEMPTS} attempts")

    time.sleep(2)
    other = "right" if holding == "left" else "left"
    open_gripper(other)
    _slide_topdown(holding, [hx, hy, az+0.02], yaw_now)
    open_gripper(holding)
    time.sleep(2)
    go_birdeye()
