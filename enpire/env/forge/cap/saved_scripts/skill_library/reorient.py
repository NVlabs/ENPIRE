# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enpire.env.forge.cap.agent.skill_registry import skill
from skill_library.namespace import *  # noqa: F401, F403


def _tool(name):
    import builtins

    import skill_library.namespace as namespace

    if name in globals():
        return globals()[name]
    if hasattr(namespace, name):
        return getattr(namespace, name)
    if hasattr(builtins, name):
        return getattr(builtins, name)
    if name == "rotate_joint":
        from enpire.env.forge.cap.agent.tools.rotate_joint import RotateJointTool

        tool = RotateJointTool()
        return lambda **kwargs: tool.execute(**kwargs)
    raise NameError(f"tool {name!r} is not available in skill_library.namespace")


def _state_attr(state, side, suffix):
    direct = f"{side}_{suffix}"
    if hasattr(state, direct):
        return getattr(state, direct)
    arms = getattr(state, "arms", None)
    if arms is not None:
        arm = arms[side] if isinstance(arms, dict) else getattr(arms, side)
        return getattr(arm, suffix)
    raise AttributeError(f"robot state missing {direct}")


def _norm_angle(deg):
    return ((float(deg) + 180.0) % 360.0) - 180.0


def _fill_to_batch(candidates, batch_top_k):
    ranked = sorted(candidates, key=lambda g: -float(g["score"]))[:batch_top_k]
    return ranked + [ranked[0]] * (batch_top_k - len(ranked)) if ranked else []


def batch_best(
    candidates,
    side,
    label="",
    batch_top_k=16,
    planning_speed=1.5,
    ik_error_threshold_m=0.01,
    ik_xyz_weight=1.0,
    ik_rpy_weight=0.3,
    solver_speed="fast",
    batch_validate_trajectory=False,
    planner_backend="curobo",
    pad=True,
):
    top_k = int(batch_top_k)
    if top_k <= 0:
        top_k = 16
    ranked = sorted(candidates, key=lambda g: -float(g["score"]))[:top_k]
    if not ranked:
        print(f"  batch [{label}]: 0/0 executable")
        return None
    payload = _fill_to_batch(ranked, top_k) if pad else ranked
    batch = freespace_move(
        grasp_candidates=payload,
        batch_side=side,
        batch_top_k=len(payload),
        solver_speed=solver_speed,
        batch_validate_trajectory=batch_validate_trajectory,
        planning_speed=planning_speed,
        ik_error_threshold=ik_error_threshold_m,
        ik_xyz_weight=ik_xyz_weight,
        ik_rpy_weight=ik_rpy_weight,
        planner_backend=planner_backend,
    )
    feasible = [
        c
        for c in (getattr(batch, "batch_candidates", []) or [])
        if getattr(c, "motion_plan_error", True) is False
        and getattr(c, "trajectory_cache_key", None) is not None
    ]
    print(f"  batch [{label}]: {len(feasible)}/{len(payload)} executable")
    return feasible[0] if feasible else None


def batch_best_in_chunks(candidates, side, label="", chunk_size=64, **kwargs):
    ranked = sorted(candidates, key=lambda g: -float(g["score"]))
    if not ranked:
        print(f"  batch [{label}]: 0/0 feasible")
        return None
    chunk_n = max(1, int(chunk_size))
    for bi in range(0, len(ranked), chunk_n):
        chunk = ranked[bi : bi + chunk_n]
        best = batch_best(
            chunk,
            side,
            f"{label}-{bi // chunk_n}",
            batch_top_k=len(chunk),
            pad=False,
            **kwargs,
        )
        if best:
            return best
    return None


@skill
def rotate_joint_checked(
    side,
    joint,
    delta_deg,
    speed_deg_s=None,
    limit_rad=None,
    settle_s=0.25,
    min_delta_deg=1.0,
    keypoint_spacing_deg=1.0,
    rotate_joint_fn=None,
):
    kwargs = {
        "side": side,
        "joint": int(joint),
        "delta_deg": float(delta_deg),
        "settle_s": float(settle_s),
        "min_duration_s": 0.0,
        "min_delta_deg": float(min_delta_deg),
        "keypoint_spacing_deg": float(keypoint_spacing_deg),
    }
    if speed_deg_s is not None:
        kwargs["speed_deg_s"] = float(speed_deg_s)
    if limit_rad is not None:
        kwargs["limit_rad"] = float(limit_rad)

    raw = (rotate_joint_fn or _tool("rotate_joint"))(**kwargs)
    if isinstance(raw, dict):
        data = raw.get("data", raw)
        if raw.get("success", True) is False:
            raise RuntimeError(raw.get("error", "rotate_joint failed"))
    else:
        success = getattr(raw, "success", True)
        if not success:
            raise RuntimeError(getattr(raw, "error", "rotate_joint failed"))
        data = getattr(raw, "data", raw)
    if not isinstance(data, dict):
        raise RuntimeError(f"rotate_joint returned unsupported result: {data!r}")
    if data.get("skipped"):
        print(f"  j{joint}: already aligned / at limit")
    else:
        print(
            f"  j{joint} rotated {float(data.get('achieved_delta_deg', 0.0)):.1f}deg "
            f"(requested {float(data.get('actual_delta_deg', delta_deg)):.1f}deg)"
        )
    return data


@skill
def rotate_in_hand(side, delta_deg, joint=6):
    return rotate_joint_checked(side=side, joint=joint, delta_deg=delta_deg)


@skill
def gentle_close(side, torque_limit=None, grip_tight_m=0.005, poll_secs=0.6, poll_steps=6):
    import time

    close_gripper(side, torque_limit=torque_limit)
    samples = []
    for _ in range(int(poll_steps)):
        state = get_robot_state()
        samples.append(float(_state_attr(state, side, "gripper_pos")))
        time.sleep(float(poll_secs) / int(poll_steps))
    thickness = min(samples)
    tight = thickness > float(grip_tight_m)
    if tight:
        set_gripper(side, pos=max(0.0, thickness * 0.95), torque_limit=torque_limit)
    print(f"  {side} gentle close: thickness={thickness:.3f} tight={tight}")
    return thickness, tight


@skill
def grip_tight(side, threshold=None, poll_secs=0.6, poll_steps=6):
    import time

    thresh = 0.005 if threshold is None else float(threshold)
    samples = []
    for _ in range(int(poll_steps)):
        state = get_robot_state()
        samples.append(float(_state_attr(state, side, "gripper_pos")))
        time.sleep(float(poll_secs) / int(poll_steps))
    ok = min(samples) > thresh
    print(f"  {side} grip={[round(v, 3) for v in samples]} -> {'TIGHT' if ok else 'SLIPPED'}")
    return ok


@skill
def assign_roles(reference_xy):
    import numpy as np

    state = get_robot_state()
    ref = np.asarray(reference_xy, dtype=float)
    left_d = float(np.linalg.norm(np.asarray(_state_attr(state, "left", "ee_pos")[:2], float) - ref))
    right_d = float(np.linalg.norm(np.asarray(_state_attr(state, "right", "ee_pos")[:2], float) - ref))
    primary = "left" if left_d <= right_d else "right"
    secondary = "right" if primary == "left" else "left"
    print(f"  arm assignment: primary={primary} secondary={secondary}")
    return primary, secondary


def strip_pca_axis(strip_obj, camera="top", max_grasps=32, min_aspect_ratio=1.5):
    import numpy as np

    grasps = sample_grasp_pose_anygrasp(object_name=strip_obj, camera=camera, max_grasps=max_grasps)
    if not grasps:
        raise RuntimeError(f"AnyGrasp returned no candidates for {strip_obj!r}")
    pts = np.asarray([g.position[:2] for g in grasps], dtype=float)
    center = pts.mean(axis=0)
    centered = pts - center
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    axis = eigvecs[:, order[0]]
    aspect = float(np.sqrt(max(eigvals[0], 1e-12) / max(eigvals[1], 1e-12)))
    print(f"  strip PCA: center={center.round(3).tolist()} aspect={aspect:.2f} L_hat={axis.round(3).tolist()}")
    if aspect < float(min_aspect_ratio):
        raise RuntimeError(f"Strip aspect {aspect:.2f} too low")
    return center, axis


def cord_centroid_xy(cord_obj, camera="top", grasp_z_m=0.80):
    import numpy as np

    grasps = sample_grasp_pose_2d(
        object_name=cord_obj, camera=camera, max_grasps=8, grasp_z_m=grasp_z_m
    )
    if not grasps:
        raise RuntimeError(f"No cord detections for {cord_obj!r}")
    return np.asarray([g.position[:2] for g in grasps], dtype=float).mean(axis=0)


def cord_signed_strip_axis(strip_center_xy, strip_axis_xy, cord_xy):
    import numpy as np

    center = np.asarray(strip_center_xy, dtype=float)
    axis = np.asarray(strip_axis_xy, dtype=float)
    cord = np.asarray(cord_xy, dtype=float)
    return axis if (cord - center) @ axis > 0 else -axis


def perpendicular_axis(axis_xy):
    import numpy as np

    axis = np.asarray(axis_xy, dtype=float)
    return np.array([-axis[1], axis[0]], dtype=float)


def assign_unplug_arms_by_cord(cord_xy):
    import numpy as np

    state = get_robot_state()
    cord = np.asarray(cord_xy, dtype=float)
    left_d = float(np.linalg.norm(np.asarray(_state_attr(state, "left", "ee_pos")[:2], float) - cord))
    right_d = float(np.linalg.norm(np.asarray(_state_attr(state, "right", "ee_pos")[:2], float) - cord))
    strip_side = "left" if left_d <= right_d else "right"
    adapter_side = "right" if strip_side == "left" else "left"
    print(f"  arm assignment: strip={strip_side} adapter={adapter_side}")
    return strip_side, adapter_side


def _quat_to_display_rpy(quat_xyzw):
    import numpy as np
    from scipy.spatial.transform import Rotation

    ex, ey, ez = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
    disp = np.array([ey, -ex, -ez - 90.0])
    return ((disp + 180.0) % 360.0 - 180.0).tolist()


def _topdown_rotation(yaw_deg):
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", [-180.0, 0.0, -float(yaw_deg) - 90.0], degrees=True)


def _yaw_for_closing_across(axis_xy):
    import numpy as np

    axis = np.asarray(axis_xy, dtype=float)
    perp = np.array([-axis[1], axis[0]], dtype=float)
    yaw = -np.degrees(np.arctan2(perp[1], perp[0])) - 90.0
    return _norm_angle(yaw)


def place_rpy(yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    import numpy as np
    from scipy.spatial.transform import Rotation

    rot = _topdown_rotation(yaw_deg)
    if abs(float(pitch_deg)) > 0.1:
        rot = Rotation.from_rotvec([0.0, np.deg2rad(pitch_deg), 0.0]) * rot
    if abs(float(roll_deg)) > 0.1:
        rot = Rotation.from_rotvec([np.deg2rad(roll_deg), 0.0, 0.0]) * rot
    return _quat_to_display_rpy(rot.as_quat())


def tilted_grasp_candidates(
    ref_xy,
    axis_xy,
    half_len,
    fractions,
    tilt_dir,
    tilt_deg,
    score_fn,
    grasp_z,
    gripper_width_m=0.08,
):
    import numpy as np
    from scipy.spatial.transform import Rotation

    axis = np.asarray(axis_xy, dtype=float)
    yaw_deg = _yaw_for_closing_across(axis)
    c_world = np.array([-axis[1], axis[0], 0.0])
    r_tilt = Rotation.from_rotvec(np.deg2rad(float(tilt_dir) * float(tilt_deg)) * c_world)
    r_final = r_tilt * _topdown_rotation(yaw_deg)
    if r_final.apply([0.0, 1.0, 0.0])[2] > 0.0:
        yaw_deg = _norm_angle(yaw_deg + 360.0)
        r_final = r_tilt * _topdown_rotation(yaw_deg)
    rpy = _quat_to_display_rpy(r_final.as_quat())
    ref = np.asarray(ref_xy, dtype=float)
    return [
        {
            "position": [
                float(ref[0] + t * float(half_len) * axis[0]),
                float(ref[1] + t * float(half_len) * axis[1]),
                float(grasp_z),
            ],
            "rpy": list(rpy),
            "score": float(score_fn(t)),
            "width": float(gripper_width_m),
        }
        for t in fractions
    ]


def grasp_with_tilt_scan(
    ref_xy,
    axis_xy,
    half_len,
    fractions,
    tilt_dir,
    primary_tilt,
    fallback_tilts,
    score_fn,
    grasp_z,
    side,
    label,
    close_fn=None,
    batch_top_k=16,
    planning_speed=1.5,
    ik_error_threshold_m=0.01,
    ik_xyz_weight=1.0,
    ik_rpy_weight=0.3,
    gripper_width_m=0.08,
):
    close_fn = close_fn or (lambda arm: (None, grip_tight(arm)))
    for tilt in [primary_tilt] + [t for t in fallback_tilts if t != primary_tilt]:
        candidates = tilted_grasp_candidates(
            ref_xy,
            axis_xy,
            half_len,
            fractions,
            tilt_dir,
            tilt,
            score_fn,
            grasp_z,
            gripper_width_m=gripper_width_m,
        )
        best = batch_best(
            candidates,
            side,
            f"{label}@{float(tilt):.0f}",
            batch_top_k=batch_top_k,
            planning_speed=planning_speed,
            ik_error_threshold_m=ik_error_threshold_m,
            ik_xyz_weight=ik_xyz_weight,
            ik_rpy_weight=ik_rpy_weight,
        )
        if best is not None:
            open_gripper(side)
            freespace_move(trajectory_cache_key=best.trajectory_cache_key)
            _, tight = close_fn(side)
            return tight
    return False


def grasp_powerstrip_for_unplug(
    side,
    strip_center_xy,
    strip_axis_xy,
    grasp_profile,
    batch_profile,
    grip_profile,
    label="strip",
):
    def score_fn(t):
        return 1.0 - 2.0 * abs(t - float(grasp_profile["score_target"]))

    close_fn = lambda arm: gentle_close(arm, **grip_profile)
    return grasp_with_tilt_scan(
        strip_center_xy,
        strip_axis_xy,
        grasp_profile["half_len"],
        grasp_profile["fractions"],
        grasp_profile["tilt_dir"],
        grasp_profile["primary_tilt"],
        grasp_profile["fallback_tilts"],
        score_fn,
        grasp_profile["grasp_z"],
        side,
        label,
        close_fn=close_fn,
        gripper_width_m=grasp_profile["gripper_width_m"],
        **batch_profile,
    )


def reorient_strip_to_y(
    side,
    place_pos_left,
    place_pos_right,
    place_yaw_left,
    place_yaw_right,
    place_pitch,
    place_roll_left,
    place_roll_right,
    reorient_batch_k=32,
    batch_top_k=16,
    planning_speed=1.5,
    ik_error_threshold_m=0.01,
    ik_xyz_weight=1.0,
    ik_rpy_weight=0.3,
    gripper_width_m=0.08,
    nonfatal=False,
):
    base_pos = list(place_pos_left if side == "left" else place_pos_right)
    target_yaw = float(place_yaw_left if side == "left" else place_yaw_right)
    place_roll = float(place_roll_left if side == "left" else place_roll_right)
    y_sign = 1.0 if side == "left" else -1.0

    candidates = []
    for yaw_delta in [0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0]:
        rpy = place_rpy(target_yaw + yaw_delta, place_pitch, place_roll)
        for dx in [0.0, -0.03, 0.03, -0.06, 0.06]:
            for dy_raw in [0.0, 0.02, -0.02, 0.04]:
                dy = dy_raw * y_sign
                candidates.append(
                    {
                        "position": [base_pos[0] + dx, base_pos[1] + dy, base_pos[2]],
                        "rpy": list(rpy),
                        "score": 1.0 - 0.3 * abs(yaw_delta) / 30.0 - 5.0 * (dx**2 + dy**2) ** 0.5,
                        "width": float(gripper_width_m),
                    }
                )
    candidates = sorted(candidates, key=lambda g: -g["score"])[: int(reorient_batch_k)]
    best = batch_best(
        candidates,
        side,
        "strip-reorient",
        batch_top_k=batch_top_k,
        planning_speed=planning_speed,
        ik_error_threshold_m=ik_error_threshold_m,
        ik_xyz_weight=ik_xyz_weight,
        ik_rpy_weight=ik_rpy_weight,
    )
    if not best:
        print(f"  strip reorient failed{' (non-fatal)' if nonfatal else ''}")
        return False
    freespace_move(trajectory_cache_key=best.trajectory_cache_key)
    try:
        nudge(side, delta_pos=[0.0, 0.0, -0.015])
    except Exception:
        pass
    print(f"  strip reoriented by {side}")
    return True


def reorient_strip_for_unplug(side, reorient_profile, batch_profile):
    return reorient_strip_to_y(side, **reorient_profile, **batch_profile)


def grasp_adapter_3d_bb(adapter_side, adapter_profile, batch_profile, grip_profile):
    grasps = sample_grasp_pose_3d_bb(
        object_name=adapter_profile["adapter_obj"],
        camera="top",
        tcp_offset_z_m=adapter_profile["tcp_offset_z_m"],
    )
    if not grasps:
        raise RuntimeError(f"3D BB returned no candidates for {adapter_profile['adapter_obj']!r}")

    best = grasps[0]
    cx, cy, grasp_z = best.position
    obb_yaw = best.rpy[2]
    flipped_yaw = _norm_angle(obb_yaw + 180.0)
    print(f"  3D BB adapter: xy=[{cx:.4f}, {cy:.4f}] z={grasp_z:.4f} yaw={obb_yaw:.1f}")

    candidates = [
        {
            "position": [cx + dx, cy + dy, grasp_z],
            "rpy": [0.0, 180.0, yaw],
            "score": 1.0 - 100.0 * (dx**2 + dy**2) ** 0.5,
            "width": adapter_profile["gripper_width_m"],
        }
        for yaw in (obb_yaw, flipped_yaw)
        for dx, dy in adapter_profile["xy_offsets"]
    ]
    candidates = sorted(candidates, key=lambda g: -g["score"])[: int(adapter_profile["adapter_batch_k"])]
    print(f"  expanded: {len(candidates)} top-down candidates")

    best_c = batch_best(candidates, adapter_side, "adapter-3dbb", **batch_profile)
    if not best_c:
        return False
    open_gripper(adapter_side)
    freespace_move(trajectory_cache_key=best_c.trajectory_cache_key)
    _, tight = gentle_close(adapter_side, **grip_profile)
    return tight


def grip_tight_profile(side, grip_profile):
    return grip_tight(
        side,
        threshold=grip_profile["grip_tight_m"],
        poll_secs=grip_profile["poll_secs"],
        poll_steps=grip_profile["poll_steps"],
    )


@skill
def reorient_by_part_axis(object_query, part_query, target_yaw_range, camera_side=None, max_attempts=999):
    raise NotImplementedError("reset-powerstrip adapter prong loop remains task-specific")


@skill
def reorient_axis(object_query, desired_axis="y", camera="top", max_attempts=999):
    raise NotImplementedError("generic axis reorientation is not wired for this refactor pass")


@skill
def wobble_along_axis(
    side,
    axis_xy,
    duration_s=None,
    steps=None,
    cycles=None,
    amplitude_m=None,
    lift_m=None,
    pitch_deg=2.5,
    move_eef_keypoints_fn=None,
):
    import numpy as np
    from scipy.spatial.transform import Rotation

    duration = 4.0 if duration_s is None else float(duration_s)
    n_steps = 120 if steps is None else int(steps)
    n_cycles = 12 if cycles is None else int(cycles)
    amplitude = 0.010 if amplitude_m is None else float(amplitude_m)
    lift = 0.20 if lift_m is None else float(lift_m)

    state = get_robot_state()
    start_pos = np.asarray(_state_attr(state, side, "ee_pos"), dtype=float)
    start_quat = np.asarray(_state_attr(state, side, "ee_quat"), dtype=float)
    start_rot = Rotation.from_quat(start_quat)

    axis = np.asarray(axis_xy, dtype=float).reshape(2)
    pitch_axis = np.array([axis[0], axis[1], 0.0], dtype=float)
    pitch_peak_rad = np.deg2rad(float(pitch_deg))
    dt = duration / n_steps

    timestamps, keypoints = [], []
    for i in range(1, n_steps + 1):
        sin_p = float(np.sin(2.0 * np.pi * n_cycles * (i / n_steps)))
        target_pos = start_pos + np.array(
            [amplitude * sin_p * axis[0], amplitude * sin_p * axis[1], lift * (i / n_steps)]
        )
        target_rot = Rotation.from_rotvec(pitch_peak_rad * sin_p * pitch_axis) * start_rot
        rpy_rad = target_rot.as_euler("xyz", degrees=False)
        timestamps.append(float(i * dt))
        keypoints.append(
            [
                float(target_pos[0]),
                float(target_pos[1]),
                float(target_pos[2]),
                float(rpy_rad[0]),
                float(rpy_rad[1]),
                float(rpy_rad[2]),
            ]
        )

    (move_eef_keypoints_fn or _tool("move_eef_keypoints"))(
        side=side, timestamps=timestamps, keypoints=keypoints
    )
    return True


def unplug_succeeded(adapter_side, grasp_z, lift_threshold_m):
    state = get_robot_state()
    ee_z = float(_state_attr(state, adapter_side, "ee_pos")[2])
    lift = ee_z - float(grasp_z)
    ok = lift > float(lift_threshold_m)
    print(f"  unplug check: lift={lift:.3f}m {'UNPLUGGED' if ok else 'STILL STUCK'}")
    return ok


def retreat_along_axis(side, axis_xy, distance_m, freespace_profile):
    import numpy as np

    state = get_robot_state()
    ee_pos = np.asarray(_state_attr(state, side, "ee_pos"), dtype=float)
    ee_rpy = list(_state_attr(state, side, "ee_rpy"))
    axis = np.asarray(axis_xy, dtype=float)
    freespace_move(
        **{
            f"{side}_target_pos": [
                float(ee_pos[0] - float(distance_m) * axis[0]),
                float(ee_pos[1] - float(distance_m) * axis[1]),
                float(ee_pos[2]),
            ],
            f"{side}_target_rpy": ee_rpy,
            **freespace_profile,
        }
    )
