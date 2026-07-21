# cabinet_handle_grasp.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.arm_motion import (
    current_arm_pose,
    go_home_checked,
    move_checked,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.cabinet_handle_geometry import (
    cabinet_handle_geometry_v1,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.planner_world import (
    refresh_planner_world,
)


@skill
def cabinet_handle_grasp_v1(
    control_name,
    target,
    stage_standoff_m=0.30,
    pre_grasp_up_bias_m=0.03,
    deep_grasp_extra_m=0.04,
    contact_depth_bias_m=0.0,
):
    import numpy as np

    _, plan = cabinet_handle_geometry_v1(target)
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
    radial_xy = plan["radial_xy"]
    horizontal_tangent = plan["horizontal_tangent"]
    tangent = plan["tangent"]
    pull_dir = plan["pull_dir"]
    standoff = plan["standoff"]
    contact_offset = plan["contact_offset"]
    grasp_quat = plan["grasp_quat"]

    print("Task:OpenCabinet")
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
    print(f"Radial xy:{[round(float(x), 3) for x in radial_xy]}")
    print(f"Horizontal tangent:{[round(float(x), 3) for x in horizontal_tangent]}")
    print(f"Tangent:{[round(float(x), 3) for x in tangent]}")
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

    stage_pos = handle_pos + approach_dir * stage_standoff_m
    print(f"Stage pos:{[round(float(x), 3) for x in stage_pos]}")
    pre_grasp = handle_pos + surface_normal * standoff
    print(f"Pre-grasp pos:{[round(float(x), 3) for x in pre_grasp]}")
    print(
        "Using Viser-aligned grasp orientation "
        f"{[round(float(x), 4) for x in grasp_quat]}"
    )

    if move_checked(
        arm,
        "pre_grasp_direct",
        pre_grasp,
        grasp_quat,
        0.10,
    ):
        selected_grasp_sign = 1.0
    else:
        print("Direct pre-grasp failed; falling back to staged approach")
        if not move_checked(
            arm,
            "stage_normal",
            stage_pos,
            grasp_quat,
            0.10,
        ):
            return False, {"reason": "failed to reach pre_grasp"}
        if not move_checked(
            arm,
            "pre_grasp",
            pre_grasp,
            grasp_quat,
            0.10,
        ):
            return False, {"reason": "failed to reach pre_grasp"}
        selected_grasp_sign = 1.0

    if grasp_quat is None:
        return False, {"reason": "failed to reach pre_grasp"}

    deep_grasp_depth = 0.03
    grasp_pos = handle_pos - approach_dir * deep_grasp_depth
    print(f"Deep grasp pos:{[round(float(x), 3) for x in grasp_pos]}")
    refresh_planner_world("deep grasp collision-disabled", exclude_body_prefixes=[""])
    if not move_checked(
        arm,
        "grasp",
        grasp_pos,
        grasp_quat,
        0.08,
        exclude_body_prefixes=[""],
    ):
        return False, {"reason": "failed to reach cabinet handle"}

    close_gripper(arm)
    return True, {
        "arm": arm,
        "control_name": control_name,
        "grasp_quat": grasp_quat,
        "pre_grasp": pre_grasp,
        "radial_xy": radial_xy,
        "selected_grasp_sign": selected_grasp_sign,
        "travel_distance": plan["travel_distance"],
    }
