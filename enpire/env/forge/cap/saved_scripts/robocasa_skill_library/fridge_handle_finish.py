# fridge_handle_finish.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.arm_motion import (
    current_arm_pose,
    move_checked,
    vec,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.fridge_control_state import (
    fridge_progress,
    get_control_target,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.fridge_handle_geometry import (
    _make_press_quat,
    _normalize,
    fridge_handle_geometry_v1,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.planner_world import (
    refresh_planner_world,
)


@skill
def fridge_handle_finish_v1(
    control_name,
    grasp_log,
    min_close_steps=40,
    max_close_steps=40,
    max_close_step_m=0.02,
    close_sign_probe_steps=3,
    close_sign_probe_gap_reduction_threshold=0.005,
    close_progress_eps=1e-4,
    door_closed_progress_threshold=0.05,
    inward_bias_m=0.015,
    max_failed_close_steps_in_a_row=3,
    max_stalled_close_steps_in_a_row=6,
):
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    def _door_exclude_prefix(target):
        source_body_name = str(target.get("source_body_name") or "")
        if "_door" in source_body_name:
            return source_body_name.split("_door", 1)[0] + "_door"
        geom_name = str(target.get("geom_name") or "")
        if "_door" in geom_name:
            return geom_name.split("_door", 1)[0] + "_door"
        return None

    def _xy_yaw_deg(v):
        return float(np.degrees(np.arctan2(float(v[1]), float(v[0]))))

    def _quat_delta_deg(q_from, q_to):
        q_from = vec(q_from)
        q_to = vec(q_to)
        rot_delta = R.from_quat(q_to) * R.from_quat(q_from).inv()
        return float(np.degrees(np.linalg.norm(rot_delta.as_rotvec())))

    arm = grasp_log["arm"]
    pull_sign = float(grasp_log.get("selected_grasp_sign", 1.0))
    travel_distance = float(grasp_log["travel_distance"])

    num_close_steps = max(
        min_close_steps,
        int(np.ceil(travel_distance / max_close_step_m)),
    )
    num_close_steps = min(num_close_steps, max_close_steps)
    step_distance = travel_distance / float(num_close_steps)
    print(
        f"{control_name} close plan:"
        f" travel_distance={travel_distance:.3f}"
        f" planned_steps={num_close_steps}"
        f" step_distance={step_distance:.3f}"
        f" inward_bias_m={inward_bias_m:.3f}"
    )

    initial_oracle = get_oracle_targets()
    progress_before = fridge_progress(initial_oracle, control_name)
    latest_target = get_control_target(initial_oracle, control_name)
    desired_fraction = (
        float(latest_target.get("desired_fraction", 0.0))
        if latest_target is not None
        else 0.0
    )
    if progress_before is None:
        print(
            f"{control_name} door fraction before close: unavailable "
            f"fixture_state={initial_oracle.get('fixture_state')}"
        )
        best_gap = None
    else:
        print(f"{control_name} door fraction before close: {progress_before:.3f}")
        best_gap = abs(progress_before - desired_fraction)
        print(f"{control_name} door gap before close: {best_gap:.3f}")

    probe_flip_used = False
    post_step3_refresh_done = False
    failed_close_steps_in_a_row = 0
    stalled_close_steps_in_a_row = 0

    for step_idx in range(num_close_steps):
        latest_oracle = get_oracle_targets()
        latest_target = get_control_target(latest_oracle, control_name)
        if latest_target is None:
            refresh_planner_world("post-close restore")
            return False, {"reason": f"{control_name} target unavailable during close"}
        desired_fraction = float(latest_target.get("desired_fraction", desired_fraction))
        _, latest_plan = fridge_handle_geometry_v1(latest_target)
        _, ee_pos_now, ee_quat_now = current_arm_pose()
        surface_normal_xy = vec(latest_plan["surface_normal"]).copy()
        surface_normal_xy[2] = 0.0
        surface_normal_xy = _normalize(surface_normal_xy)
        base_close_dir_xy = vec(latest_plan["horizontal_tangent"]).copy()
        base_close_dir_xy[2] = 0.0
        base_close_dir_xy = _normalize(base_close_dir_xy)
        contact_bias_dir = vec(
            latest_plan.get("surface_normal_hint", latest_plan["surface_normal"])
        ).copy()
        contact_bias_dir = _normalize(contact_bias_dir)
        candidate_signs = [pull_sign, -pull_sign] if step_idx == 0 else [pull_sign]

        move_success = False
        selected_close_dir_xy = None
        for sign_idx, candidate_sign in enumerate(candidate_signs):
            current_close_dir_xy = _normalize(base_close_dir_xy * candidate_sign)
            close_quat = _make_press_quat(current_close_dir_xy)
            close_target = (
                ee_pos_now
                + current_close_dir_xy * step_distance
                - contact_bias_dir * inward_bias_m
            )
            close_dir_yaw_deg = _xy_yaw_deg(current_close_dir_xy)
            close_quat_delta_deg = _quat_delta_deg(ee_quat_now, close_quat)
            normal_close_dot = float(
                np.clip(
                    np.dot(surface_normal_xy[:2], current_close_dir_xy[:2]),
                    -1.0,
                    1.0,
                )
            )
            normal_close_angle_deg = float(np.degrees(np.arccos(normal_close_dot)))
            print(
                f"Close step {step_idx + 1}/{num_close_steps}: "
                f"delta={step_distance:.3f} "
                f"inward_bias={inward_bias_m:.3f} "
                f"normal_xy={[round(float(x), 3) for x in surface_normal_xy]} "
                f"raw_dir={[round(float(x), 3) for x in base_close_dir_xy]} "
                f"dir_xy={[round(float(x), 3) for x in current_close_dir_xy]} "
                f"normal_close_dot={normal_close_dot:.3f} "
                f"normal_close_angle_deg={normal_close_angle_deg:.1f} "
                f"dir_yaw_deg={close_dir_yaw_deg:.1f} "
                f"quat_delta_deg={close_quat_delta_deg:.1f} "
                f"target={[round(float(x), 3) for x in close_target]}"
            )
            step_name = f"close_step_{step_idx + 1}"
            if step_idx == 0 and sign_idx == 1:
                step_name = f"close_step_{step_idx + 1}_alt"
            if move_checked(
                arm,
                step_name,
                close_target,
                close_quat,
                0.18,
            ):
                if step_idx == 0 and candidate_sign != pull_sign:
                    pull_sign = candidate_sign
                    print(f"Close step 1 alt sign selected:{pull_sign:+.0f}")
                move_success = True
                selected_close_dir_xy = current_close_dir_xy.copy()
                break
            if step_idx == 0 and sign_idx == 0:
                print("Close step 1 failed; retrying with alternate sign")

        if not move_success:
            failed_close_steps_in_a_row += 1
            print(
                f"{control_name} close step {step_idx + 1} failed"
                f" ({failed_close_steps_in_a_row}/{max_failed_close_steps_in_a_row} in a row)"
            )
            if failed_close_steps_in_a_row >= max_failed_close_steps_in_a_row:
                refresh_planner_world("post-close restore")
                open_gripper(arm)
                return False, {"reason": f"close step {step_idx + 1} failed"}
            continue

        failed_close_steps_in_a_row = 0

        latest_oracle_after = get_oracle_targets()
        progress_after = fridge_progress(latest_oracle_after, control_name)
        if progress_after is not None:
            gap_after = abs(progress_after - desired_fraction)
            print(f"{control_name} door fraction after step {step_idx + 1}: {progress_after:.3f}")
            print(f"{control_name} door gap after step {step_idx + 1}: {gap_after:.3f}")

            gap_reduction = None
            if progress_before is not None:
                gap_before = abs(progress_before - desired_fraction)
                gap_reduction = float(gap_before - gap_after)
            if (
                not probe_flip_used
                and gap_reduction is not None
                and step_idx + 1 <= close_sign_probe_steps
                and gap_reduction < close_sign_probe_gap_reduction_threshold
            ):
                pull_sign = -pull_sign
                probe_flip_used = True
                print(
                    f"{control_name} probe gap reduction after step {step_idx + 1} "
                    f"is {gap_reduction:.3f} < "
                    f"{close_sign_probe_gap_reduction_threshold:.3f}; "
                    f"flipping close sign to {pull_sign:+.0f}"
                )
                continue

            if best_gap is None or gap_after < best_gap - close_progress_eps:
                best_gap = gap_after
                stalled_close_steps_in_a_row = 0
            else:
                stalled_close_steps_in_a_row += 1
                print(
                    f"{control_name} close progress stalled after step {step_idx + 1}"
                    f" ({stalled_close_steps_in_a_row}/{max_stalled_close_steps_in_a_row})"
                )

            task_info = get_task_info()
            if progress_after <= door_closed_progress_threshold or task_info.get("success", False):
                print(
                    f"{control_name} reached closed threshold "
                    f"{door_closed_progress_threshold:.2f} after close step "
                    f"{step_idx + 1}"
                )
                print(f"{control_name} close succeeded; stopping immediately")
                return True, {"best_gap": gap_after, "best_progress": progress_after}

            if stalled_close_steps_in_a_row >= max_stalled_close_steps_in_a_row:
                refresh_planner_world("post-close restore")
                open_gripper(arm)
                return False, {
                    "reason": "fridge close stalled before success",
                    "best_gap": best_gap,
                    "best_progress": progress_after,
                }
        else:
            print(
                f"{control_name} door fraction after step {step_idx + 1}: unavailable "
                f"fixture_state={latest_oracle_after.get('fixture_state')}"
            )

        if not post_step3_refresh_done and step_idx + 1 == 3:
            exclude_prefixes = ["robot0", "gripper", "mobilebase"]
            active_door_prefix = _door_exclude_prefix(latest_target)
            if active_door_prefix:
                exclude_prefixes.append(active_door_prefix)
            print(
                "Refreshing planner world after 3 close steps with exclusions:"
                f" {exclude_prefixes}"
            )
            refresh_planner_world(
                "post-step3 close collision refresh",
                exclude_body_prefixes=exclude_prefixes,
            )
            post_step3_refresh_done = True

    refresh_planner_world("post-close restore")
    open_gripper(arm)
    final = get_task_info()
    return False, {
        "reason": "fridge not closed successfully",
        "best_gap": best_gap,
        "success": final.get("success", False),
    }
