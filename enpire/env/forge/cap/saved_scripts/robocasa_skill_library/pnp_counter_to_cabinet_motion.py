# pnp_counter_to_cabinet_motion.py — PickPlaceCounterToCabinet motion helpers
from skill_library.namespace import *  # noqa: F401, F403

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.detection import detect_object_v1
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_detection import (
    estimate_segment_grasp_orientation_v1,
    refine_object_position_from_segment_v1,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_geometry import (
    camera_surface_normal_from_cameras_v1,
    fmt_xyz_v1,
    make_cabinet_press_quat_v1,
    quat_from_gripper_direction_v1,
    topdown_grasp_quat_candidates_from_axes_v1,
)


def refresh_counter_to_cabinet_planner_world_v1(reason):
    world = update_planner_world()
    print(f"[curobo] {reason}: {world['n_obstacles']} obstacles")
    return world


def position_grasp_object_v1(
    *,
    side,
    obj_pos,
    hover_clearance_m,
    grasp_z_offset_m,
    nudge_down_step_m,
    nudge_down_max_steps,
    nudge_down_min_z_change_m,
    object_query=None,
    grasp_axes=None,
    long_axis_offsets_m=(
        0.0,
        -0.01,
        0.01,
        -0.02,
        0.02,
        -0.03,
        0.03,
        -0.04,
        0.04,
    ),
):
    print("\n--- Step 3b: position grasp + close ---")
    open_gripper(side)
    time.sleep(0.1)

    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)
    hover_surface_normal = camera_surface_normal_from_cameras_v1(
        obj_pos,
        side=side,
        label="object_hover",
    )
    hover_pos = (
        np.asarray(obj_pos, dtype=float)
        + hover_surface_normal * float(hover_clearance_m)
    )
    print(
        "  Hover base: "
        f"object_pos={fmt_xyz_v1(obj_pos)} "
        f"surface_normal={fmt_xyz_v1(hover_surface_normal)} "
        f"offset_m={float(hover_clearance_m):.3f} "
        f"target={fmt_xyz_v1(hover_pos)}"
    )
    hover_pos_offsets = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.04],
        [0.0, 0.0, 0.08],
        [0.03, 0.0, 0.04],
        [-0.03, 0.0, 0.04],
        [0.0, 0.03, 0.04],
        [0.0, -0.03, 0.04],
        [0.03, 0.03, 0.08],
        [0.03, -0.03, 0.08],
        [-0.03, 0.03, 0.08],
        [-0.03, -0.03, 0.08],
    ]
    hover_pos_offsets = _with_long_axis_offsets(
        hover_pos_offsets,
        grasp_axes,
        long_axis_offsets_m,
    )
    grasp_pos_offsets = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.02],
        [0.02, 0.0, 0.02],
        [-0.02, 0.0, 0.02],
        [0.0, 0.02, 0.02],
        [0.0, -0.02, 0.02],
    ]
    orientation_offsets = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -25.0],
        [0.0, 0.0, 25.0],
        [0.0, 0.0, -45.0],
        [0.0, 0.0, 45.0],
        [-20.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [0.0, -20.0, 0.0],
        [0.0, 20.0, 0.0],
    ]

    base_quats = _position_grasp_base_quats(current_quat)
    hover_label, hover_used, hover_quat = _plan_execute_position_candidates(
        side,
        hover_pos,
        base_quat_candidates=base_quats,
        pos_offsets=hover_pos_offsets,
        orientation_offsets_deg=orientation_offsets,
        status_prefix="position hover",
    )
    print(
        f"  Hover used label={hover_label} "
        f"pos={fmt_xyz_v1(hover_used)} quat={fmt_xyz_v1(hover_quat, 4)}"
    )

    refined_obj_pos, refined_grasp_axes = _refine_object_from_wrist_after_hover(
        object_query,
        fallback_pos=obj_pos,
        fallback_grasp_axes=grasp_axes,
    )
    grasp_pos = np.asarray(refined_obj_pos, dtype=float).copy()
    grasp_pos[2] += float(grasp_z_offset_m)
    grasp_pos_offsets = _with_long_axis_offsets(
        grasp_pos_offsets,
        refined_grasp_axes,
        long_axis_offsets_m,
    )

    current_quat = np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)
    primary_quats, fallback_quats = _position_grasp_quat_groups(
        current_quat,
        grasp_axes=refined_grasp_axes,
    )
    try:
        grasp_label, grasp_used, grasp_quat = _plan_execute_position_candidates(
            side,
            grasp_pos,
            base_quat_candidates=primary_quats,
            pos_offsets=grasp_pos_offsets,
            orientation_offsets_deg=[[0.0, 0.0, 0.0]],
            status_prefix="position grasp wrist primary",
        )
    except Exception as exc:
        print(
            "  Primary wrist-mask grasp orientation failed: "
            f"{type(exc).__name__}: {exc}; trying fallbacks"
        )
        grasp_label, grasp_used, grasp_quat = _plan_execute_position_candidates(
            side,
            grasp_pos,
            base_quat_candidates=fallback_quats,
            pos_offsets=grasp_pos_offsets,
            orientation_offsets_deg=orientation_offsets,
            status_prefix="position grasp fallback",
        )
    print(
        f"  Grasp used label={grasp_label} "
        f"pos={fmt_xyz_v1(grasp_used)} quat={fmt_xyz_v1(grasp_quat, 4)}"
    )

    nudge_info = _nudge_down_before_close(
        side,
        step_m=nudge_down_step_m,
        max_steps=nudge_down_max_steps,
        min_z_change_m=nudge_down_min_z_change_m,
        object_z_m=float(refined_obj_pos[2]),
        stop_below_object_m=0.015,
    )
    print(
        "  Nudge down before close: "
        f"method={nudge_info['method']} steps={nudge_info['steps']} "
        f"actual={nudge_info['actual_down_m']:.4f} "
        f"stop={nudge_info['stop_reason']} "
        f"stop_z={nudge_info['stop_z_m']} trace={nudge_info['trace']}"
    )

    close_gripper(side)
    time.sleep(0.25)
    grasped, width = check_counter_to_cabinet_grasp_v1(side)
    print(f"  Gripper width={width:.4f} -> {'GRASPED' if grasped else 'MISSED'}")
    return {
        "grasped": bool(grasped),
        "width": float(width),
        "hover_label": hover_label,
        "hover_pos": hover_used,
        "hover_surface_normal": hover_surface_normal,
        "grasp_label": grasp_label,
        "grasp_pos": grasp_used,
        "nudge_down": nudge_info,
    }


def move_home_closed_v1(*, side, home_pos, home_quat):
    print("\n--- Step 4: go home closed ---")
    world = update_planner_world()
    print(f"  Collision avoidance enabled for home: {world['n_obstacles']} obstacles")
    result = _move_world_pose(
        side,
        home_pos,
        home_quat,
        label="go home closed",
        gripper=0.1,
        auto_update_world=False,
    )
    if not _success(result):
        raise RuntimeError(f"go home closed failed: {_status(result)}")
    return result


def move_to_cabinet_normal_offset_v1(
    *,
    side,
    cabinet_pos,
    surface_normal,
    offset_candidates_m,
    target_z_offset_m,
    extra_z_offsets_m=(),
    cross_axis_offsets_m=(0.0,),
):
    print("\n--- Step 5b: move to cabinet normal offset ---")
    quat = make_cabinet_press_quat_v1(surface_normal)
    errors = []

    surface_normal = np.asarray(surface_normal, dtype=float)
    cross_axis = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), surface_normal)
    cross_axis[2] = 0.0
    cross_norm = float(np.linalg.norm(cross_axis))
    if cross_norm < 1e-6:
        cross_axis = np.zeros(3, dtype=float)
        cross_axis_offsets = [0.0]
    else:
        cross_axis = cross_axis / cross_norm
        cross_axis_offsets = [float(v) for v in cross_axis_offsets_m]
    if not cross_axis_offsets:
        cross_axis_offsets = [0.0]

    positive_z_offsets = [float(z) for z in extra_z_offsets_m if float(z) > 0.0]
    negative_z_offsets = [float(z) for z in extra_z_offsets_m if float(z) < 0.0]
    z_offsets = [0.0] + positive_z_offsets + negative_z_offsets
    printed_negative_fallback = False
    printed_cross_fallback = False
    for cross_offset_idx, cross_offset_m in enumerate(cross_axis_offsets):
        if cross_offset_idx == 1:
            printed_cross_fallback = True
            print(
                "  Normal/z offset candidates failed; "
                f"trying cross(z, surface_normal) offsets={cross_axis_offsets[1:]}"
            )
        for z_offset_idx, extra_z_m in enumerate(z_offsets):
            if cross_offset_idx == 0 and z_offset_idx == 1:
                print(
                    "  Base cabinet normal offsets failed; "
                    f"trying z+ offsets={positive_z_offsets}"
                )
            if (
                cross_offset_idx == 0
                and extra_z_m < 0.0
                and not printed_negative_fallback
            ):
                printed_negative_fallback = True
                print(
                    "  Positive extra z offsets failed; "
                    f"trying z- offsets={negative_z_offsets}"
                )
            for offset_m in offset_candidates_m:
                target = (
                    np.asarray(cabinet_pos, dtype=float)
                    + surface_normal * float(offset_m)
                    + cross_axis * float(cross_offset_m)
                )
                target[2] += float(target_z_offset_m) + extra_z_m
                z_label = _format_signed_offset_label("z", extra_z_m)
                cross_label = _format_signed_offset_label("cross", cross_offset_m)
                result = _move_world_pose(
                    side,
                    target,
                    quat,
                    label=(
                        f"cabinet normal offset {offset_m:.2f}m"
                        f"{z_label}{cross_label}"
                    ),
                    gripper=0.1,
                )
                if _success(result):
                    print(
                        f"  Step 5b used offset={offset_m:.2f} "
                        f"extra_z={extra_z_m:.2f} "
                        f"cross_offset={float(cross_offset_m):.2f} "
                        f"target={fmt_xyz_v1(target)} "
                        f"surface_normal={fmt_xyz_v1(surface_normal)} "
                        f"cross_axis={fmt_xyz_v1(cross_axis)}"
                    )
                    return {
                        "target_pos": target,
                        "target_quat": quat,
                        "offset_m": float(offset_m),
                        "extra_z_m": extra_z_m,
                        "cross_offset_m": float(cross_offset_m),
                        "cross_axis": cross_axis,
                        "result": result,
                    }
                errors.append(
                    f"{offset_m:.2f}+z{extra_z_m:.2f}"
                    f"+cross{float(cross_offset_m):.2f}:{_status(result)}"
                )
    if not printed_cross_fallback and len(cross_axis_offsets) > 1:
        print("  Cross-axis fallback was not reached.")
    raise RuntimeError("all cabinet normal offsets failed; last=" + errors[-1])


def nudge_negative_cabinet_normal_v1(*, side, surface_normal, step_m, n_steps):
    print("\n--- Step 5c: nudge -surface normal ---")
    direction = -np.asarray(surface_normal, dtype=float)
    reached = []
    for step_idx in range(1, int(n_steps) + 1):
        state = get_robot_state().arms[side]
        pos = np.asarray(state.ee_pos, dtype=float)
        result = nudge_brutal(
            side=side,
            delta_pos=(direction * float(step_m)).tolist(),
            n_steps=10,
        )
        final_pos = np.asarray(getattr(result, "final_pos", pos), dtype=float)
        reached.append(final_pos.copy())
        print(
            f"  -normal nudge {step_idx}/{int(n_steps)}: "
            f"success={bool(getattr(result, 'success', False))} "
            f"target_delta={fmt_xyz_v1(direction * float(step_m))} "
            f"actual_delta={fmt_xyz_v1(final_pos - pos)}"
        )
        if not bool(getattr(result, "success", False)):
            break

    total = float(step_m) * len(reached)
    print(
        f"  Nudge result steps={len(reached)} "
        f"step_m={float(step_m):.2f} total_m={total:.2f} "
        f"direction={fmt_xyz_v1(direction)}"
    )
    return {
        "steps": len(reached),
        "step_m": float(step_m),
        "total_m": total,
        "direction": direction,
        "final_target": reached[-1] if reached else None,
    }


def _format_signed_offset_label(name, value):
    value = float(value)
    if abs(value) < 1e-9:
        return ""
    sign = "+" if value > 0.0 else "-"
    return f" {name}{sign}{abs(value):.2f}"


def check_counter_to_cabinet_grasp_v1(side):
    state = get_robot_state().arms[side]
    width = float(np.asarray(state.gripper_pos, dtype=float).reshape(-1)[0])
    return 0.05 < width < 0.95, width


def _status(result):
    return str(getattr(result, "status", result))


def _success(result):
    return _status(result) == "Success"


def _refine_object_from_wrist_after_hover(
    object_query,
    *,
    fallback_pos,
    fallback_grasp_axes,
):
    query = str(object_query or "").strip()
    if not query:
        print("  Wrist refine skipped: empty object query")
        return np.asarray(fallback_pos, dtype=float), fallback_grasp_axes

    refined_pos = np.asarray(fallback_pos, dtype=float)
    refined_axes = fallback_grasp_axes

    try:
        det = refine_object_position_from_segment_v1(query, "wrist")
        refined_pos = np.asarray(det["pos"], dtype=float)
        print(
            f"  Wrist refine position {query!r}: "
            f"pos={fmt_xyz_v1(refined_pos)} score={det['score']:.3f} "
            f"mode={det['mode']} z_range={det['z_range_m']:.3f} "
            f"z_upper={det['z_upper_m']} n_used={det['n_used']}/{det['n_valid']}"
        )
    except Exception as exc:
        print(f"  Wrist segment position refine failed: {type(exc).__name__}: {exc}")
        try:
            det = detect_object_v1(query, cameras="wrist", required=True)
            refined_pos = np.asarray(det["pos"], dtype=float)
            print(
                f"  Wrist refine detection fallback {query!r}: "
                f"pos={fmt_xyz_v1(refined_pos)} score={det['score']:.3f}"
            )
        except Exception as fallback_exc:
            print(
                f"  Wrist refine detection failed: "
                f"{type(fallback_exc).__name__}: {fallback_exc}; "
                "using hover target position"
            )

    wrist_axes = estimate_segment_grasp_orientation_v1(
        query,
        cameras=("wrist",),
        required=False,
    )
    if wrist_axes is not None:
        refined_axes = wrist_axes
        print("  Using wrist-mask grasp orientation for descent")
    elif refined_axes is not None:
        print("  Using fallback grasp orientation for descent")
    else:
        print("  No mask grasp orientation available; using geometric fallbacks")

    return refined_pos, refined_axes


def _position_grasp_base_quats(current_quat, grasp_axes=None):
    primary, fallback = _position_grasp_quat_groups(current_quat, grasp_axes=grasp_axes)
    return primary + fallback


def _position_grasp_quat_groups(current_quat, grasp_axes=None):
    mask_candidates = _mask_grasp_quat_candidates(grasp_axes)
    if mask_candidates:
        primary = [mask_candidates[0]]
        fallback = mask_candidates[1:] + _fallback_grasp_quats(current_quat)
    else:
        primary = [("current", np.asarray(current_quat, dtype=float))]
        fallback = _fallback_grasp_quats(current_quat, include_current=False)
    return primary, fallback


def _mask_grasp_quat_candidates(grasp_axes):
    candidates = []
    if grasp_axes is not None:
        try:
            mask_candidates = topdown_grasp_quat_candidates_from_axes_v1(
                grasp_axes["long_axis_world"],
                grasp_axes["short_axis_world"],
            )
            for label, quat in mask_candidates:
                candidates.append((label, np.asarray(quat, dtype=float)))
        except Exception as exc:
            print(
                "  Mask grasp orientation candidates skipped: "
                f"{type(exc).__name__}: {exc}"
            )
    return candidates


def _fallback_grasp_quats(current_quat, *, include_current=True):
    candidates = []
    if include_current:
        candidates.append(("current", np.asarray(current_quat, dtype=float)))
    for label, direction in (
        ("dir+z", [0.0, 0.0, 1.0]),
        ("dir-z", [0.0, 0.0, -1.0]),
        ("dir+x", [1.0, 0.0, 0.0]),
        ("dir-x", [-1.0, 0.0, 0.0]),
        ("dir+y", [0.0, 1.0, 0.0]),
        ("dir-y", [0.0, -1.0, 0.0]),
    ):
        candidates.append((label, quat_from_gripper_direction_v1(direction)))
    return candidates


def _with_long_axis_offsets(pos_offsets, grasp_axes, long_axis_offsets_m):
    offsets = [np.asarray(offset, dtype=float) for offset in pos_offsets]
    if grasp_axes is None:
        return _unique_position_offsets(offsets)

    try:
        long_axis = np.asarray(grasp_axes["long_axis_world"], dtype=float).copy()
        long_axis[2] = 0.0
        long_axis = long_axis / max(float(np.linalg.norm(long_axis)), 1e-8)
        if float(np.linalg.norm(long_axis[:2])) < 1e-6:
            return _unique_position_offsets(offsets)
        long_offsets = [
            long_axis * float(offset_m)
            for offset_m in long_axis_offsets_m
        ]
        print(
            "  Added long-axis position offsets: "
            f"axis={fmt_xyz_v1(long_axis)} offsets_m={list(long_axis_offsets_m)}"
        )
        return _unique_position_offsets(long_offsets + offsets)
    except Exception as exc:
        print(
            "  Long-axis position offsets skipped: "
            f"{type(exc).__name__}: {exc}"
        )
        return _unique_position_offsets(offsets)


def _unique_position_offsets(pos_offsets):
    unique = []
    seen = set()
    for offset in pos_offsets:
        arr = np.asarray(offset, dtype=float)
        key = tuple(np.round(arr, 4).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(arr.tolist())
    return unique


def _move_world_pose(
    side,
    target_pos,
    target_quat=None,
    label="move",
    gripper=None,
    auto_update_world=True,
):
    kwargs = {
        "right_target_pos": np.asarray(target_pos, dtype=float).tolist(),
        "side": side,
        "auto_update_world": bool(auto_update_world),
    }
    if target_quat is not None:
        kwargs["right_target_quat"] = np.asarray(target_quat, dtype=float).tolist()
    if gripper is not None:
        kwargs["gripper"] = float(gripper)
    result = freespace_move(**kwargs)
    print(f"  {label}: status={_status(result)} target={fmt_xyz_v1(target_pos)}")
    return result


def _plan_execute_position_candidates(
    side,
    target_pos,
    *,
    base_quat_candidates,
    pos_offsets,
    orientation_offsets_deg,
    status_prefix,
):
    errors = []
    seen = set()
    target_pos = np.asarray(target_pos, dtype=float)
    for base_name, base_quat in base_quat_candidates:
        base_rot = R.from_quat(np.asarray(base_quat, dtype=float))
        for pos_offset in pos_offsets:
            offset_arr = np.asarray(pos_offset, dtype=float)
            for rpy_offset in orientation_offsets_deg:
                rpy_arr = np.asarray(rpy_offset, dtype=float)
                key = (
                    str(base_name),
                    tuple(np.round(offset_arr, 4).tolist()),
                    tuple(np.round(rpy_arr, 4).tolist()),
                )
                if key in seen:
                    continue
                seen.add(key)

                candidate_pos = target_pos + offset_arr
                candidate_quat = (
                    base_rot * R.from_euler("xyz", rpy_arr, degrees=True)
                ).as_quat()
                label = status_prefix
                if base_name != "current":
                    label += f"+{base_name}"
                if np.linalg.norm(offset_arr) > 1e-6:
                    label += f"+d{np.round(offset_arr, 3).tolist()}"
                if np.linalg.norm(rpy_arr) > 1e-6:
                    label += f"+rpy{np.round(rpy_arr, 1).tolist()}"

                result = _move_world_pose(side, candidate_pos, candidate_quat, label)
                if _success(result):
                    return label, candidate_pos, candidate_quat
                errors.append(f"{label}: {_status(result)}")

    if errors:
        raise RuntimeError("all position candidates failed; last=" + errors[-1])
    raise RuntimeError("no position candidates")


def _nudge_down_before_close(
    side,
    *,
    step_m,
    max_steps,
    min_z_change_m,
    object_z_m=None,
    stop_below_object_m=0.015,
):
    step_m = abs(float(step_m))
    start_z = float(np.asarray(get_robot_state().arms[side].ee_pos, dtype=float)[2])
    prev_z = start_z
    final_z = start_z
    trace = []
    stop_reason = "max_steps"
    method = "nudge_brutal"
    stop_z = None
    nudge_steps = 0
    if object_z_m is not None:
        stop_z = float(object_z_m) - float(stop_below_object_m)

    for step_idx in range(1, int(max_steps) + 1):
        if stop_z is not None and prev_z <= stop_z:
            stop_reason = "below_object_z_threshold"
            trace.append(
                f"pre{step_idx}:z={prev_z:.4f},stop_z={stop_z:.4f}"
            )
            break

        result = nudge_brutal(
            side=side,
            delta_pos=[0.0, 0.0, -step_m],
            n_steps=5,
        )
        nudge_steps += 1

        cur_z = float(np.asarray(get_robot_state().arms[side].ee_pos, dtype=float)[2])
        z_step_down = prev_z - cur_z
        total_down = start_z - cur_z
        final_z = cur_z
        trace.append(
            f"step{step_idx}:success={bool(getattr(result, 'success', False))},"
            f"dz={z_step_down:.4f},total={total_down:.4f}"
        )
        prev_z = cur_z
        if stop_z is not None and cur_z <= stop_z:
            stop_reason = "below_object_z_threshold"
            break

    return {
        "method": method,
        "steps": nudge_steps,
        "actual_down_m": start_z - final_z,
        "stop_reason": stop_reason,
        "stop_z_m": stop_z,
        "trace": "; ".join(trace),
    }
