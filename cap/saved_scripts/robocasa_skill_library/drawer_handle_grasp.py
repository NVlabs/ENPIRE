# drawer_handle_grasp.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from cap.agent.skill_registry import skill

from cap.saved_scripts.robocasa_skill_library.arm_motion import (
    current_arm_pose,
    go_home_checked,
    move_checked,
)
from cap.saved_scripts.robocasa_skill_library.drawer_handle_geometry import (
    drawer_handle_geometry_v1,
)
from cap.saved_scripts.robocasa_skill_library.planner_world import (
    refresh_planner_world,
)
from cap.saved_scripts.robocasa_skill_library.show_debug_balls import (
    show_debug_balls_v1,
)


@skill
def drawer_handle_grasp_v1(
    control_name,
    target,
    stage_standoff_m=0.30,
    deep_grasp_extra_m=0.04,
    contact_depth_bias_m=0.0,
    grasp_plane_tilt_candidate_degs=(0.0, 8.0, -8.0, 15.0, -15.0),
    deep_grasp_step_m=0.02,
):
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    _, plan = drawer_handle_geometry_v1(target)
    target_pos = plan["target_pos"]
    target_size = plan["target_size"]
    target_geom_name = plan["target_geom_name"]
    target_geom_type_name = plan["target_geom_type_name"]
    surface_normal = plan["surface_normal"]
    surface_normal_source = plan.get("surface_normal_source", "unknown")
    camera_vector_xy = plan.get("camera_vector_xy")
    axis_world = plan["axis_world"]
    desired_fraction = plan["desired_fraction"]
    current_fraction = plan["current_fraction"]
    pull_sign_hint = plan["pull_sign_hint"]
    handle_pos = plan["handle_pos"]
    is_round_handle = plan["is_round_handle"]
    approach_dir = plan["approach_dir"]
    horizontal_pull = plan["horizontal_pull"]
    pull_dir = plan["pull_dir"]
    standoff = plan["standoff"]
    contact_offset = plan["contact_offset"]
    grasp_quat = plan["grasp_quat"]

    print("Task:OpenDrawer")
    print(f"Control:{control_name}")
    print(f"Target pos:{[round(float(x), 3) for x in target_pos]}")
    print(f"Target size:{[round(float(x), 4) for x in target_size]}")
    print(f"Target geom:{target_geom_name}")
    print(f"Target geom type:{target_geom_type_name}")
    print(f"Surface normal source:{surface_normal_source}")
    if camera_vector_xy is not None:
        print(f"Camera vector xy:{[round(float(x), 3) for x in camera_vector_xy]}")
    print(f"Surface normal:{[round(float(x), 3) for x in surface_normal]}")
    print(f"Axis world:{[round(float(x), 3) for x in axis_world]}")
    print(f"Current fraction:{current_fraction:.3f}")
    print(f"Desired fraction:{desired_fraction:.3f}")

    arm, ee_pos, travel_quat = current_arm_pose()
    print(f"Current ee pos:{[round(float(x), 3) for x in ee_pos]}")
    print(f"Current ee quat:{[round(float(x), 4) for x in travel_quat]}")
    print(f"Handle type:{'round' if is_round_handle else 'rectangular'}")
    print(f"Handle pos used:{[round(float(x), 3) for x in handle_pos]}")
    print(f"Approach dir:{[round(float(x), 3) for x in approach_dir]}")
    print(f"Horizontal pull:{[round(float(x), 3) for x in horizontal_pull]}")
    print(f"Pull sign hint:{pull_sign_hint:+.0f}")
    print(f"Pull dir:{[round(float(x), 3) for x in pull_dir]}")
    surface_normal_xy = np.asarray(surface_normal, dtype=float).copy()
    surface_normal_xy[2] = 0.0
    surface_normal_xy = surface_normal_xy / max(float(np.linalg.norm(surface_normal_xy)), 1e-8)
    pull_dir_xy = np.asarray(pull_dir, dtype=float).copy()
    pull_dir_xy[2] = 0.0
    pull_dir_xy = pull_dir_xy / max(float(np.linalg.norm(pull_dir_xy)), 1e-8)
    normal_pull_dot = float(np.clip(np.dot(surface_normal_xy, pull_dir_xy), -1.0, 1.0))
    normal_pull_angle_deg = float(np.degrees(np.arccos(normal_pull_dot)))
    print(
        "Normal vs pull:"
        f" normal_xy={[round(float(x), 3) for x in surface_normal_xy]}"
        f" pull_xy={[round(float(x), 3) for x in pull_dir_xy]}"
        f" dot={normal_pull_dot:.3f}"
        f" angle_deg={normal_pull_angle_deg:.1f}"
    )

    open_gripper(arm)
    refresh_planner_world(f"{control_name} run start")
    if not go_home_checked(arm, "go_home"):
        return False, {"reason": "failed to reach home pose"}

    pre_grasp = handle_pos + surface_normal * standoff
    print(f"Pre-grasp pos:{[round(float(x), 3) for x in pre_grasp]}")
    deep_grasp_depth = 0.015
    grasp_pos = handle_pos - approach_dir * deep_grasp_depth
    print(f"Deep grasp pos:{[round(float(x), 3) for x in grasp_pos]}")
    grasp_delta = grasp_pos - pre_grasp
    grasp_distance = float(np.linalg.norm(grasp_delta))
    n_grasp_steps = max(1, int(np.ceil(grasp_distance / max(float(deep_grasp_step_m), 1e-6))))
    print(
        f"Deep grasp stepping: distance={grasp_distance:.3f}m "
        f"step_m={float(deep_grasp_step_m):.3f} n_steps={n_grasp_steps}"
    )
    show_debug_balls_v1(
        [
            {
                "name": f"drawer_handle_target_{control_name}",
                "label": control_name,
                "position": handle_pos,
                "color": [64, 255, 255],
                "radius_m": 0.03,
                "alpha": 0.75,
            },
            {
                "name": f"drawer_pre_grasp_target_{control_name}",
                "label": f"{control_name}_pre",
                "position": pre_grasp,
                "color": [255, 128, 192],
                "radius_m": 0.03,
                "alpha": 0.75,
            },
            {
                "name": f"drawer_grasp_target_{control_name}",
                "label": f"{control_name}_grasp",
                "position": grasp_pos,
                "color": [128, 255, 128],
                "radius_m": 0.03,
                "alpha": 0.75,
            },
        ]
    )
    print(
        "Using Viser-aligned grasp orientation "
        f"{[round(float(x), 4) for x in grasp_quat]}"
    )

    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    plane_normal = np.cross(surface_normal, world_up)
    if np.linalg.norm(plane_normal) < 1e-6:
        plane_normal = np.cross(surface_normal, np.array([0.0, 1.0, 0.0], dtype=float))
    plane_normal = plane_normal / max(float(np.linalg.norm(plane_normal)), 1e-8)
    print(
        "Grasp plane tilt axis:"
        f" {[round(float(x), 4) for x in plane_normal]}"
        f" candidates_deg={list(grasp_plane_tilt_candidate_degs)}"
    )

    selected_tilt_deg = None
    selected_grasp_quat = None
    base_grasp_rot = R.from_quat(grasp_quat)
    for tilt_deg in grasp_plane_tilt_candidate_degs:
        candidate_grasp_quat = (
            R.from_rotvec(plane_normal * np.radians(float(tilt_deg))) * base_grasp_rot
        ).as_quat()
        print(
            f"Trying pre-grasp tilt {float(tilt_deg):+.1f} deg:"
            f" quat={[round(float(x), 4) for x in candidate_grasp_quat]}"
        )
        if move_checked(
            arm,
            f"pre_grasp_direct_tilt_{float(tilt_deg):+05.1f}",
            pre_grasp,
            candidate_grasp_quat,
            0.10,
        ):
            selected_tilt_deg = float(tilt_deg)
            selected_grasp_quat = candidate_grasp_quat
            break

    if selected_grasp_quat is None:
        return False, {"reason": "failed to reach pre_grasp"}

    refresh_planner_world("deep grasp collision-disabled", exclude_body_prefixes=[""])
    failed_grasp_step = None
    for step_idx in range(n_grasp_steps):
        alpha = float(step_idx + 1) / float(n_grasp_steps)
        step_target = pre_grasp + grasp_delta * alpha
        label = "grasp" if step_idx == n_grasp_steps - 1 else f"grasp_step_{step_idx + 1}"
        _, ee_pos_now, _ = current_arm_pose()
        delta_pos = (step_target - ee_pos_now).tolist()
        nudge_result = nudge_brutal(
            arm,
            delta_pos=delta_pos,
            n_steps=5,
        )
        final_pos = np.asarray(getattr(nudge_result, "final_pos", ee_pos_now), dtype=float)
        actual_delta = final_pos - ee_pos_now
        status = "Success" if bool(getattr(nudge_result, "success", False)) else "Failed"
        print(
            f"{label}: status={status} "
            f"target={[round(float(x), 3) for x in step_target]} "
            f"delta_pos={[round(float(x), 3) for x in delta_pos]} "
            f"actual_delta={[round(float(x), 3) for x in actual_delta]}"
        )
        if not bool(getattr(nudge_result, "success", False)):
            failed_grasp_step = step_idx + 1
            print(
                f"{control_name} deep grasp stalled at step {failed_grasp_step}/"
                f"{n_grasp_steps}; closing gripper anyway"
            )
            break

    close_gripper(arm)
    return True, {
        "arm": arm,
        "control_name": control_name,
        "grasp_quat": selected_grasp_quat,
        "pre_grasp": pre_grasp,
        "surface_normal": surface_normal,
        "selected_grasp_sign": 1.0,
        "selected_tilt_deg": selected_tilt_deg,
        "deep_grasp_completed": failed_grasp_step is None,
        "failed_grasp_step": failed_grasp_step,
        "n_grasp_steps": n_grasp_steps,
        "travel_distance": plan["travel_distance"],
        "retreat_distance": plan["retreat_distance"],
    }
