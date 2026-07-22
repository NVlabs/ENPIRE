# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# drawer_handle_finish.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.arm_motion import (
    current_arm_pose,
    go_home_checked,
    vec,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.drawer_control_state import (
    drawer_progress,
    get_control_target,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.drawer_handle_geometry import (
    _normalize,
    drawer_handle_geometry_v1,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.planner_world import (
    refresh_planner_world,
)


@skill
def drawer_handle_finish_v1(
    control_name,
    grasp_log,
    min_pull_steps=4,
    max_pull_steps=40,
    max_pull_step_m=0.05,
    max_total_pull_steps=60,
    pull_sign_probe_steps=3,
    pull_sign_probe_progress_threshold=0.005,
    max_failed_pull_steps_in_a_row=4,
    max_stalled_pull_steps_in_a_row=6,
    drawer_open_progress_threshold=0.95,
):
    import time
    import numpy as np

    arm = grasp_log["arm"]
    grasp_quat = vec(grasp_log["grasp_quat"])
    pull_sign = float(grasp_log.get("selected_grasp_sign", 1.0))
    travel_distance = float(grasp_log["travel_distance"])

    num_pull_steps = max(min_pull_steps, int(np.ceil(travel_distance / max_pull_step_m)))
    num_pull_steps = min(num_pull_steps, max_pull_steps)
    step_distance = travel_distance / float(num_pull_steps)
    print(
        f"{control_name} pull plan:"
        f" travel_distance={travel_distance:.3f}"
        f" planned_steps={num_pull_steps}"
        f" step_distance={step_distance:.3f}"
        f" max_total_pull_steps={max_total_pull_steps}"
    )

    initial_progress_oracle = get_oracle_targets()
    progress_before = drawer_progress(initial_progress_oracle, control_name)
    if progress_before is None:
        print(
            f"{control_name} drawer fraction before pull: unavailable "
            f"fixture_state={initial_progress_oracle.get('fixture_state')}"
        )
    else:
        print(f"{control_name} drawer fraction before pull: {progress_before:.3f}")

    best_progress = progress_before
    probe_flip_used = False
    post_step3_refresh_done = False
    failed_pull_steps_in_a_row = 0
    stalled_pull_steps_in_a_row = 0

    for step_idx in range(max_total_pull_steps):
        latest_oracle = get_oracle_targets()
        latest_target = get_control_target(latest_oracle, control_name)
        if latest_target is None:
            refresh_planner_world("post-pull restore")
            return False, {"reason": f"{control_name} target unavailable during pull"}
        _, latest_plan = drawer_handle_geometry_v1(latest_target)
        _, ee_pos_now, _ = current_arm_pose()
        surface_normal_xy = vec(latest_plan["surface_normal"]).copy()
        surface_normal_xy[2] = 0.0
        surface_normal_xy = _normalize(surface_normal_xy)
        base_pull_dir = vec(latest_plan["pull_dir"]).copy()
        base_pull_dir = _normalize(base_pull_dir)
        base_pull_dir_xy = base_pull_dir.copy()
        base_pull_dir_xy[2] = 0.0
        if np.linalg.norm(base_pull_dir_xy) < 1e-6:
            base_pull_dir_xy = vec(latest_plan["horizontal_pull"]).copy()
        base_pull_dir_xy = _normalize(base_pull_dir_xy)
        candidate_signs = [pull_sign]
        if step_idx == 0:
            candidate_signs = sorted(
                [pull_sign, -pull_sign],
                key=lambda sign: float(
                    np.dot(surface_normal_xy[:2], (base_pull_dir_xy * sign)[:2])
                ),
                reverse=True,
            )
            print(
                "Pull step 1 sign order from surface normal alignment:"
                f" {candidate_signs}"
            )

        move_success = False
        for sign_idx, candidate_sign in enumerate(candidate_signs):
            current_pull_dir = _normalize(base_pull_dir * candidate_sign)
            current_pull_dir_xy = current_pull_dir.copy()
            current_pull_dir_xy[2] = 0.0
            if np.linalg.norm(current_pull_dir_xy) < 1e-6:
                current_pull_dir_xy = _normalize(base_pull_dir_xy * candidate_sign)
            else:
                current_pull_dir_xy = _normalize(current_pull_dir_xy)
            pull_target = ee_pos_now + current_pull_dir * step_distance
            normal_pull_dot = float(
                np.clip(
                    np.dot(surface_normal_xy[:2], current_pull_dir_xy[:2]),
                    -1.0,
                    1.0,
                )
            )
            normal_pull_angle_deg = float(np.degrees(np.arccos(normal_pull_dot)))
            print(
                f"Pull step {step_idx + 1}/{num_pull_steps}: "
                f"delta={step_distance:.3f} "
                f"normal_xy={[round(float(x), 3) for x in surface_normal_xy]} "
                f"raw_dir={[round(float(x), 3) for x in base_pull_dir_xy]} "
                f"dir_xy={[round(float(x), 3) for x in current_pull_dir_xy]} "
                f"normal_pull_dot={normal_pull_dot:.3f} "
                f"normal_pull_angle_deg={normal_pull_angle_deg:.1f} "
                f"target={[round(float(x), 3) for x in pull_target]}"
            )
            step_name = f"pull_step_{step_idx + 1}"
            if step_idx == 0 and sign_idx == 1:
                step_name = f"pull_step_{step_idx + 1}_alt"
            delta_pos = (current_pull_dir * step_distance).tolist()
            nudge_result = nudge_brutal(
                arm,
                delta_pos=delta_pos,
                n_steps=5,
            )
            final_pos = np.asarray(getattr(nudge_result, "final_pos", ee_pos_now), dtype=float)
            actual_delta = final_pos - ee_pos_now
            actual_along_pull = float(np.dot(actual_delta, current_pull_dir))
            status = "Success" if bool(getattr(nudge_result, "success", False)) else "Failed"
            print(
                f"{step_name}: status={status} "
                f"delta_pos={[round(float(x), 3) for x in delta_pos]} "
                f"actual_delta={[round(float(x), 3) for x in actual_delta]} "
                f"along_pull={actual_along_pull:.4f}"
            )
            if bool(getattr(nudge_result, "success", False)):
                if step_idx == 0 and candidate_sign != pull_sign:
                    pull_sign = candidate_sign
                    print(f"Pull step 1 alt sign selected:{pull_sign:+.0f}")
                move_success = True
                break
            if step_idx > 0 or sign_idx == len(candidate_signs) - 1:
                continue
            print("Pull step 1 failed; retrying with alternate sign")

        if not move_success:
            failed_pull_steps_in_a_row += 1
            print(
                f"{control_name} pull step {step_idx + 1} failed"
                f" ({failed_pull_steps_in_a_row}/{max_failed_pull_steps_in_a_row} in a row)"
            )
            if failed_pull_steps_in_a_row >= max_failed_pull_steps_in_a_row:
                refresh_planner_world("post-pull restore")
                return False, {"reason": f"pull step {step_idx + 1} failed"}
            continue

        failed_pull_steps_in_a_row = 0

        latest_oracle_after = get_oracle_targets()
        progress_after = drawer_progress(latest_oracle_after, control_name)
        if progress_after is not None:
            print(f"{control_name} drawer fraction after step {step_idx + 1}: {progress_after:.3f}")
            progress_gain = None
            if progress_before is not None:
                progress_gain = float(progress_after - progress_before)
            if (
                not probe_flip_used
                and progress_gain is not None
                and step_idx + 1 <= pull_sign_probe_steps
                and progress_gain < pull_sign_probe_progress_threshold
            ):
                pull_sign = -pull_sign
                probe_flip_used = True
                print(
                    f"{control_name} probe progress after step {step_idx + 1} "
                    f"is {progress_gain:.3f} < "
                    f"{pull_sign_probe_progress_threshold:.3f}; "
                    f"flipping pull sign to {pull_sign:+.0f}"
                )
                continue
            if best_progress is None or progress_after > best_progress + 1e-4:
                best_progress = progress_after
                stalled_pull_steps_in_a_row = 0
            else:
                stalled_pull_steps_in_a_row += 1
                print(
                    f"{control_name} pull progress stalled after step {step_idx + 1}"
                    f" ({stalled_pull_steps_in_a_row}/{max_stalled_pull_steps_in_a_row})"
                )
            task_info = get_task_info()
            if (
                progress_after >= drawer_open_progress_threshold
                or task_info.get("success", False)
            ):
                print(
                    f"{control_name} reached open threshold "
                    f"{drawer_open_progress_threshold:.2f} after pull step "
                    f"{step_idx + 1}"
                )
                print(f"{control_name} pull succeeded; stopping immediately")
                return True, {"best_progress": progress_after}
            if stalled_pull_steps_in_a_row >= max_stalled_pull_steps_in_a_row:
                refresh_planner_world("post-pull restore")
                return False, {
                    "reason": "drawer pull stalled before success",
                    "best_progress": best_progress,
                }
        else:
            print(
                f"{control_name} drawer fraction after step {step_idx + 1}: unavailable "
                f"fixture_state={latest_oracle_after.get('fixture_state')}"
            )

        if not post_step3_refresh_done and step_idx + 1 == 3:
            exclude_prefixes = ["robot0", "gripper", "mobilebase"]
            source_body_name = str(latest_target.get("source_body_name") or "")
            if source_body_name:
                exclude_prefixes.append(source_body_name)
            print(
                "Refreshing planner world after 3 pull steps with exclusions:"
                f" {exclude_prefixes}"
            )
            refresh_planner_world(
                "post-step3 pull collision refresh",
                exclude_body_prefixes=exclude_prefixes,
            )
            post_step3_refresh_done = True

    refresh_planner_world("post-pull restore")
    open_gripper(arm)
    time.sleep(0.10)
    final = get_task_info()
    return False, {
        "reason": "drawer not opened successfully",
        "best_progress": best_progress,
        "success": final.get("success", False),
    }
