# grasp_object.py - reusable object grasp skill
from skill_library.namespace import *  # noqa: F401, F403

import time

import numpy as np

from cap.agent.skill_registry import skill
from cap.saved_scripts.robocasa_skill_library.detection import detect_object_v1
from cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_detection import (
    estimate_segment_grasp_orientation_v1,
    object_query_from_task_v1,
)
from cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_geometry import (
    fmt_xyz_v1,
)
from cap.saved_scripts.robocasa_skill_library.pnp_counter_to_cabinet_motion import (
    position_grasp_object_v1,
)


# deprecated: use pick_object_anygrasp_v1 instead
@skill
def grasp_object_v1(
    side,
    object_query=None,
    object_cameras=("top", "wrist"),
    hover_clearance_m=0.10,
    grasp_z_offset_m=0.01,
    nudge_down_step_m=0.010,
    nudge_down_max_steps=8,
    nudge_down_min_z_change_m=0.0030,
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
    estimate_initial_grasp_axes=True,
):
    """Detect an object, hover over it, refine from wrist, then grasp it.

    This wraps the PickPlace object-grasp sequence:
    1. detect the task object from the configured external cameras;
    2. move to a hover pose over that object;
    3. detect the object again from the wrist camera and estimate mask axes;
    4. descend with orientation and long-axis position candidates, nudge down,
       close the gripper, and verify by gripper width.

    Returns ``(grasped, log)`` for the skill registry.
    """
    query = object_query or object_query_from_task_v1()
    cameras = tuple(object_cameras)

    print("\n--- Grasp object: detect target object ---")
    object_det = detect_object_v1(
        query,
        cameras=cameras,
    )
    if estimate_initial_grasp_axes:
        object_det["grasp_axes"] = estimate_segment_grasp_orientation_v1(
            object_det["query"],
            cameras=_grasp_orientation_camera_order(object_det["camera"], cameras),
            required=False,
        )
    else:
        object_det["grasp_axes"] = None

    print("\n--- Grasp object: hover, wrist refine, grasp ---")
    grasp_info = position_grasp_object_v1(
        side=side,
        obj_pos=object_det["pos"],
        hover_clearance_m=hover_clearance_m,
        grasp_z_offset_m=grasp_z_offset_m,
        nudge_down_step_m=nudge_down_step_m,
        nudge_down_max_steps=nudge_down_max_steps,
        nudge_down_min_z_change_m=nudge_down_min_z_change_m,
        object_query=object_det["query"],
        grasp_axes=object_det.get("grasp_axes"),
        long_axis_offsets_m=long_axis_offsets_m,
    )
    grasped = bool(grasp_info["grasped"])

    log = {
        "success": grasped,
        "side": side,
        "object_query": object_det["query"],
        "object_camera": object_det["camera"],
        "object_score": float(object_det["score"]),
        "object_pos": _as_list(object_det["pos"]),
        "object_cameras": list(cameras),
        "hover_clearance_m": float(hover_clearance_m),
        "grasp_z_offset_m": float(grasp_z_offset_m),
        "long_axis_offsets_m": [float(v) for v in long_axis_offsets_m],
        "grasped": grasped,
        "gripper_width": float(grasp_info["width"]),
        "hover_label": str(grasp_info["hover_label"]),
        "hover_pos": _as_list(grasp_info["hover_pos"]),
        "hover_surface_normal": _as_list(grasp_info["hover_surface_normal"]),
        "grasp_label": str(grasp_info["grasp_label"]),
        "grasp_pos": _as_list(grasp_info["grasp_pos"]),
        "nudge_down": dict(grasp_info["nudge_down"]),
    }
    print(
        "  grasp_object result: "
        f"grasped={grasped} width={log['gripper_width']:.4f} "
        f"object={fmt_xyz_v1(object_det['pos'])}"
    )
    return grasped, log


def hover_above_place_target_v1(
    side,
    tgt_pos,
    hover_clearance_m=0.12,
    face_down_only=False,
):
    """Hover above a place target with yaw aligned to the home→target direction.

    Tries orientation candidates in order until one plans successfully.
    With face_down_only=True only the face-down (180° pitch) orientation is
    attempted — useful when the target surface is always horizontal (e.g. stove).
    """
    from scipy.spatial.transform import Rotation as R

    place_hover = np.asarray(tgt_pos, dtype=float).copy()
    place_hover[2] += float(hover_clearance_m)
    home_pos = np.asarray(get_robot_state().arms[side].ee_pos, dtype=float)
    dx, dy = place_hover[0] - home_pos[0], place_hover[1] - home_pos[1]
    yaw_deg = float(np.degrees(np.arctan2(dy, dx)))
    approach_quat = (
        R.from_euler("z", yaw_deg + 180, degrees=True) * R.from_euler("y", 180, degrees=True)
    ).as_quat()
    if face_down_only:
        candidates = [("approach", approach_quat)]
    else:
        horizontal_quat = (
            R.from_euler("z", yaw_deg + 180, degrees=True) * R.from_euler("y", 90, degrees=True)
        ).as_quat()
        candidates = [
            ("approach", approach_quat),
            ("horizontal", horizontal_quat),
            ("current", np.asarray(get_robot_state().arms[side].ee_quat, dtype=float)),
        ]
    print(f"  Place hover yaw={yaw_deg:.1f}° (home→target)")
    for label, quat in candidates:
        r = freespace_move(
            **{
                f"{side}_target_pos": place_hover.tolist(),
                f"{side}_target_quat": quat.tolist(),
                "side": side,
                "gripper": 0.1,
            }
        )
        print(f"  Place hover [{label}]: {r.status}")
        if _status(r) == "Success":
            return {"success": True, "pos": _as_list(place_hover), "label": label}
    return {"success": False, "pos": _as_list(place_hover), "label": None}


def hover_above_object_v1(
    side,
    object_query=None,
    cameras=("top", "right"),
    hover_clearance_m=0.15,
):
    """Detect an object with ranked cameras, then hover above it."""
    from scipy.spatial.transform import Rotation as R

    det = detect_object_v1(
        object_query or object_query_from_task_v1(),
        cameras=cameras,
    )
    hover_pos = np.asarray(det["pos"], dtype=float)
    hover_pos[2] += float(hover_clearance_m)
    down_quat = R.from_euler("xyz", [0, 180, 0], degrees=True).as_quat()
    result = freespace_move(
        **{
            f"{side}_target_pos": hover_pos.tolist(),
            f"{side}_target_quat": down_quat.tolist(),
            "side": side,
        }
    )
    print(f"  Hover status: {_status(result)} target={fmt_xyz_v1(hover_pos)}")
    return {
        "success": _status(result) == "Success",
        "detection": det,
        "hover_pos": _as_list(hover_pos),
        "status": _status(result),
    }


@skill
def pick_object_anygrasp_v1(
    side,
    object_query=None,
    fallback_cameras=("top", "right"),
    max_grasps=16,
    batch_top_k=16,
    pregrasp_offset_m=0.03,
    disable_planner_z_clipping=True,
    object_input_mode="segmented_object_cloud",
    use_wrist=True,
    **_ignored,
):
    """Pick an object with wrist-first detect+AnyGrasp and two-step ranking."""
    prompts = _specific_prompt_variants(object_query or object_query_from_task_v1())
    fallback_cameras = _camera_order(fallback_cameras)
    open_gripper(side)

    object_det = None
    grasps = []
    grasp_camera = None
    grasp_query = None
    detection_path = "wrist"

    print("\n--- Pick AnyGrasp: wrist detect + grasp ---")
    if not use_wrist:
        print("  Skipping wrist (hover did not succeed)")
    for prompt in prompts if use_wrist else []:
        try:
            det = detect_object_v1(
                prompt,
                cameras="wrist",
                required=False,
            )
        except Exception as exc:
            print(f"  wrist detect {prompt!r}: {type(exc).__name__}: {exc}")
            continue
        if det is None:
            print(f"  wrist detect {prompt!r}: no detection")
            continue
        print(
            f"  wrist detect {prompt!r}: pos={fmt_xyz_v1(det['pos'])} "
            f"score={det['score']:.3f}"
        )
        try:
            candidates = sample_grasp_pose_anygrasp(
                prompt,
                camera="wrist",
                max_grasps=int(max_grasps),
                top_down_only=False,
                object_input_mode=str(object_input_mode),
                disable_planner_z_clipping=bool(disable_planner_z_clipping),
            )
        except Exception as exc:
            print(f"  wrist AnyGrasp {prompt!r}: {type(exc).__name__}: {exc}")
            continue
        if candidates:
            object_det = det
            grasps = list(candidates)
            grasp_camera = "wrist"
            grasp_query = prompt
            _publish_live_object_detection(object_det)
            print(f"  Using {len(grasps)} wrist grasp(s) query={prompt!r}")
            break
        print(f"  wrist AnyGrasp {prompt!r}: no grasps")

    if not grasps:
        print("\n--- Pick AnyGrasp: fallback ranked detect + grasp ---")
        object_det = detect_object_v1(
            prompts,
            cameras=fallback_cameras,
        )
        detection_path = "fallback_rank"
        grasp_camera = str(object_det["camera"])
        grasp_query = str(object_det["query"])
        _publish_live_object_detection(object_det)
        try:
            grasps = list(
                sample_grasp_pose_anygrasp(
                    grasp_query,
                    camera=grasp_camera,
                    max_grasps=int(max_grasps),
                    top_down_only=False,
                    object_input_mode=str(object_input_mode),
                    disable_planner_z_clipping=bool(disable_planner_z_clipping),
                )
            )
        except Exception as exc:
            print(
                f"  fallback AnyGrasp camera={grasp_camera} "
                f"query={grasp_query!r}: {type(exc).__name__}: {exc}"
            )
            grasps = []

    if not grasps:
        print("  AnyGrasp found no candidates")
        return False, _pick_log(
            side=side,
            object_det=object_det,
            detection_path=detection_path,
            grasped=False,
            width=0.0,
            grasp_camera=grasp_camera,
            grasp_query=grasp_query,
            n_grasps=0,
        )

    print("\n--- Pick AnyGrasp: two-step batch-rank grasps ---")
    batch_result = select_best_grasp_two_step(
        grasps,
        side=side,
        batch_top_k=int(batch_top_k),
        pregrasp_offset_m=float(pregrasp_offset_m),
        augment_yaw_flip=True,
    )
    ranked = list(getattr(batch_result, "batch_candidates", None) or [])
    feasible = [c for c in ranked if getattr(c, "is_executable", False)]
    print(
        f"  Ranked {len(ranked)} candidates: "
        f"{len(feasible)} two-step executable, status={_status(batch_result)}"
    )
    for row in ranked[:5]:
        print(
            f"    #{row.rank} src={row.source_index} "
            f"status={row.planner_status} err={row.ik_error_m} "
            f"rot={row.ik_rot_error_deg} score={row.score}"
        )
    _publish_live_pick_poses(feasible, pregrasp_offset_m)

    grasped = False
    width = 0.0
    executed = None

    def _reset_after_failed_grasp(reason):
        print(f"  Reset after failed grasp ({reason}): opening gripper and going home")
        open_gripper(side)
        r = go_home(side)
        print(f"  Reset go_home status: {_status(r)}")

    for attempt_idx, candidate in enumerate(feasible):
        actual_pos = np.asarray(candidate.position, dtype=float)
        grasp_rpy = [float(x) for x in candidate.rpy]
        grasp_quat = display_rpy_to_quat(grasp_rpy)
        from scipy.spatial.transform import Rotation as R

        approach_dir = R.from_quat(grasp_quat).apply([0.0, 0.0, 1.0])
        pregrasp_pos = actual_pos - (2.0 * float(pregrasp_offset_m) * approach_dir)

        print(
            "\n--- Pick AnyGrasp: execute two-step grasp "
            f"rank={candidate.rank} src={candidate.source_index} ---"
        )
        pre = freespace_move(
            **{
                f"{side}_target_pos": pregrasp_pos.tolist(),
                f"{side}_target_quat": grasp_quat.tolist(),
                "side": side,
            }
        )
        print(f"  Pregrasp status: {_status(pre)} pos={fmt_xyz_v1(pregrasp_pos)}")
        if _status(pre) != "Success":
            _reset_after_failed_grasp("pregrasp failed")
            continue

        desc = freespace_move(
            **{
                f"{side}_target_pos": actual_pos.tolist(),
                f"{side}_target_quat": grasp_quat.tolist(),
                "side": side,
                "exclude_body_prefixes": [""],
            }
        )
        print(f"  Descend status: {_status(desc)} pos={fmt_xyz_v1(actual_pos)}")
        if _status(desc) != "Success":
            print("  Descend freespace_move failed -> nudge_brutal fallback")
            nudge_brutal(
                side=side, delta_pos=(actual_pos - pregrasp_pos).tolist(), n_steps=24
            )

        close_gripper(side)
        time.sleep(0.25)
        width = _gripper_width(side)
        print(f"  Closed width={width:.4f}")
        if _is_fully_closed(width):
            print("  Gripper fully closed at grasp pose -> MISSED")
            _reset_after_failed_grasp("empty close")
            continue

        retreat = freespace_move(
            **{
                f"{side}_target_pos": pregrasp_pos.tolist(),
                f"{side}_target_quat": grasp_quat.tolist(),
                "side": side,
                "gripper": 0.1,
                "exclude_body_prefixes": [""],
            }
        )
        print(
            f"  Retreat-to-hover status: {_status(retreat)} pos={fmt_xyz_v1(pregrasp_pos)}"
        )
        if _status(retreat) != "Success":
            print("  Retreat freespace_move failed -> nudge_brutal fallback")
            current_pos = np.asarray(get_robot_state().arms[side].ee_pos, dtype=float)
            nudge_brutal(
                side=side, delta_pos=(pregrasp_pos - current_pos).tolist(), n_steps=24
            )

        width = _gripper_width(side)
        grasped = not _is_fully_closed(width)
        print(f"  Lifted width={width:.4f} -> {'GRASPED' if grasped else 'MISSED'}")
        if grasped:
            executed = candidate
            break
        _reset_after_failed_grasp("lifted empty close")

    log = _pick_log(
        side=side,
        object_det=object_det,
        detection_path=detection_path,
        grasped=bool(grasped),
        width=float(width),
        grasp_camera=grasp_camera,
        grasp_query=grasp_query,
        n_grasps=len(grasps),
        n_ranked=len(ranked),
        n_feasible=len(feasible),
        executed=executed,
        pregrasp_offset_m=float(pregrasp_offset_m),
    )
    print(
        "  pick_object_anygrasp result: "
        f"grasped={grasped} width={width:.4f} "
        f"object={fmt_xyz_v1(object_det['pos']) if object_det else 'unknown'}"
    )
    return bool(grasped), log


def _as_list(value):
    return [float(x) for x in np.asarray(value, dtype=float).reshape(-1)]


def _gripper_width(side):
    state = get_robot_state().arms[side]
    return float(np.asarray(state.gripper_pos, dtype=float).reshape(-1)[0])


def _is_fully_closed(width):
    return float(width) < 0.02


def _specific_prompt_variants(query):
    variants = []
    vague = {"object", "item", "thing", "stuff", "food"}
    if isinstance(query, (list, tuple)):
        raw_prompts = query
    else:
        text = str(query or "").strip()
        words = [w for w in text.split() if len(w) >= 3]
        raw_prompts = [text, *words]
    for q in raw_prompts:
        q = str(q).strip()
        if q and q not in variants:
            variants.append(q)
    if not variants:
        raise RuntimeError(f"no specific object prompts from query {query!r}")
    return variants


def _prompt_list(query):
    vague = {"object", "item", "thing", "stuff", "food"}
    raw_prompts = query if isinstance(query, (list, tuple)) else [query]
    prompts = []
    for q in raw_prompts:
        q = str(q or "").strip()
        if q and q.lower() not in vague and q not in prompts:
            prompts.append(q)
    if not prompts:
        raise RuntimeError(f"no specific object prompts from query {query!r}")
    return prompts


def _camera_order(primary, fallback=()):
    ordered = []
    if isinstance(primary, str):
        primary_items = [primary]
    else:
        primary_items = list(primary or ())
    if isinstance(fallback, str):
        fallback_items = [fallback]
    else:
        fallback_items = list(fallback or ())
    for camera in [*primary_items, *fallback_items]:
        camera = str(camera).strip()
        if camera and camera not in ordered:
            ordered.append(camera)
    return tuple(ordered or ("top", "wrist", "right"))


def _grasp_orientation_camera_order(best_camera, cameras):
    ordered = []
    if "top" in cameras:
        ordered.append("top")
    if best_camera not in ordered:
        ordered.append(best_camera)
    for camera in cameras:
        if camera not in ordered:
            ordered.append(camera)
    return tuple(ordered)


def _pick_log(
    *,
    side,
    object_det,
    detection_path,
    grasped,
    width,
    grasp_camera,
    grasp_query,
    n_grasps,
    n_ranked=0,
    n_feasible=0,
    executed=None,
    pregrasp_offset_m=0.02,
):
    return {
        "success": bool(grasped),
        "side": side,
        "object_query": None if object_det is None else object_det.get("query"),
        "object_camera": None if object_det is None else object_det.get("camera"),
        "object_score": 0.0
        if object_det is None
        else float(object_det.get("score", 0.0)),
        "object_pos": [] if object_det is None else _as_list(object_det.get("pos", [])),
        "detection_path": detection_path,
        "grasped": bool(grasped),
        "gripper_width": float(width),
        "grasp_method": "anygrasp_two_step",
        "grasp_camera": grasp_camera,
        "grasp_query": grasp_query,
        "n_grasps": int(n_grasps),
        "n_ranked": int(n_ranked),
        "n_feasible": int(n_feasible),
        "pregrasp_offset_m": float(pregrasp_offset_m),
        "grasp_rank": getattr(executed, "rank", None) if executed is not None else None,
        "grasp_pos": _as_list(getattr(executed, "position", []))
        if executed is not None
        else [],
        "grasp_rpy": _as_list(getattr(executed, "rpy", []))
        if executed is not None
        else [],
    }


def _publish_live_pick_poses(candidates, pregrasp_offset_m):
    hover_publisher = globals().get("set_live_hover_poses")
    grasp_publisher = globals().get("set_live_grasp_poses")
    if not callable(hover_publisher) and not callable(grasp_publisher):
        return
    hover_poses, grasp_poses = _ranked_hover_and_grasp_poses(
        candidates,
        pregrasp_offset_m,
    )
    if callable(hover_publisher):
        try:
            hover_publisher(hover_poses)
        except Exception as exc:
            print(f"  live hover pose publish failed: {type(exc).__name__}: {exc}")
    if callable(grasp_publisher):
        try:
            grasp_publisher(grasp_poses)
        except Exception as exc:
            print(f"  live grasp pose publish failed: {type(exc).__name__}: {exc}")


def _publish_live_object_detection(object_det):
    publisher = globals().get("set_live_detections")
    if not callable(publisher) or not object_det:
        return
    try:
        publisher(
            [
                {
                    "label": f"object: {object_det.get('query', 'object')}",
                    "pos": _as_list(object_det.get("pos", [])),
                    "score": float(object_det.get("score", 0.0)),
                }
            ]
        )
    except Exception as exc:
        print(f"  live object marker publish failed: {type(exc).__name__}: {exc}")


def _ranked_hover_and_grasp_poses(candidates, pregrasp_offset_m):
    from scipy.spatial.transform import Rotation as R

    hover_poses = []
    grasp_poses = []
    for candidate in list(candidates or [])[:20]:
        rpy = [float(x) for x in candidate.rpy]
        quat = display_rpy_to_quat(rpy)
        approach_dir = R.from_quat(quat).apply([0.0, 0.0, 1.0])
        hover_pos = np.asarray(candidate.position, dtype=float) - (
            2.0 * float(pregrasp_offset_m) * approach_dir
        )
        hover_poses.append(
            {
                "pos": _as_list(hover_pos),
                "rpy": rpy,
                "score": float(candidate.score),
                "rank": int(candidate.rank),
                "source_index": int(candidate.source_index),
            }
        )
        grasp_poses.append(
            {
                "pos": _as_list(candidate.position),
                "rpy": rpy,
                "score": float(candidate.score),
                "rank": int(candidate.rank),
                "source_index": int(candidate.source_index),
            }
        )
    return hover_poses, grasp_poses


def _status(result):
    return str(getattr(result, "status", result))
