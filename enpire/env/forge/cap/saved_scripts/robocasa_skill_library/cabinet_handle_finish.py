# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cabinet_handle_finish.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.arm_motion import (
    current_arm_pose,
    go_home_checked,
    move_checked,
    move_with_orientation,
    vec,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.cabinet_control_state import (
    door_progress,
    get_control_target,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.cabinet_handle_geometry import (
    _make_press_quat,
    _normalize,
    cabinet_handle_geometry_v1,
)
from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.planner_world import (
    refresh_planner_world,
)


@skill
def cabinet_handle_finish_v1(
    control_name,
    grasp_log,
    min_pull_steps=40,
    max_pull_steps=40,
    max_pull_step_m=0.02,
    pull_sign_probe_steps=3,
    pull_sign_probe_progress_threshold=0.005,
    post_pull_release_trigger_progress=0.45,
    post_pull_release_extra_steps=2,
    post_pull_vertical_trigger_progress=0.5,
    post_pull_vertical_stall_or_fail_progress_threshold=0.3,
    post_pull_vertical_progress_eps=1e-4,
    door_open_progress_threshold=0.95,
    post_pull_vertical_steps=12,
    post_pull_vertical_step_m=0.15,
    post_pull_vertical_pre_offset_m=0.10,
):
    import time
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
    grasp_quat = vec(grasp_log["grasp_quat"])
    pre_grasp = vec(grasp_log["pre_grasp"])
    radial_xy = vec(grasp_log["radial_xy"])
    pull_sign = float(grasp_log.get("selected_grasp_sign", 1.0))
    travel_distance = float(grasp_log["travel_distance"])

    num_pull_steps = max(min_pull_steps, int(np.ceil(travel_distance / max_pull_step_m)))
    num_pull_steps = min(num_pull_steps, max_pull_steps)
    step_distance = travel_distance / float(num_pull_steps)

    initial_progress_oracle = get_oracle_targets()
    progress_before = door_progress(initial_progress_oracle, control_name)
    if progress_before is None:
        print(
            f"{control_name} door fraction before pull: unavailable "
            f"fixture_state={initial_progress_oracle.get('fixture_state')}"
        )
    else:
        print(f"{control_name} door fraction before pull: {progress_before:.3f}")

    best_progress = progress_before
    initial_pull_dir_xy = None
    should_run_vertical_finish = False
    released_for_vertical_finish = False
    probe_flip_used = False
    post_step3_refresh_done = False

    for step_idx in range(num_pull_steps):
        latest_oracle = get_oracle_targets()
        latest_target = get_control_target(latest_oracle, control_name)
        if latest_target is None:
            refresh_planner_world("post-pull restore")
            return False, {"reason": f"{control_name} target unavailable during pull"}
        _, latest_plan = cabinet_handle_geometry_v1(latest_target)
        _, ee_pos_now, ee_quat_now = current_arm_pose()
        surface_normal_xy = vec(latest_plan["surface_normal"]).copy()
        surface_normal_xy[2] = 0.0
        surface_normal_xy = _normalize(surface_normal_xy)
        base_pull_dir_xy = vec(latest_plan["horizontal_tangent"]).copy()
        base_pull_dir_xy[2] = 0.0
        base_pull_dir_xy = _normalize(base_pull_dir_xy)
        candidate_signs = [pull_sign]
        if step_idx == 0:
            candidate_signs = sorted(
                [pull_sign, -pull_sign],
                key=lambda sign: float(
                    np.dot(
                        surface_normal_xy[:2],
                        (base_pull_dir_xy * sign)[:2],
                    )
                ),
                reverse=True,
            )
            print(
                "Pull step 1 sign order from surface normal alignment:"
                f" {candidate_signs}"
            )

        move_success = False
        for sign_idx, candidate_sign in enumerate(candidate_signs):
            current_pull_dir_xy = base_pull_dir_xy * candidate_sign
            current_pull_quat = _make_press_quat(current_pull_dir_xy)
            pull_target = ee_pos_now + current_pull_dir_xy * step_distance
            pull_dir_yaw_deg = _xy_yaw_deg(current_pull_dir_xy)
            pull_quat_delta_deg = _quat_delta_deg(ee_quat_now, current_pull_quat)
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
                f"dir_yaw_deg={pull_dir_yaw_deg:.1f} "
                f"quat_delta_deg={pull_quat_delta_deg:.1f} "
                f"target={[round(float(x), 3) for x in pull_target]}"
            )
            step_name = f"pull_step_{step_idx + 1}"
            if step_idx == 0 and sign_idx == 1:
                step_name = f"pull_step_{step_idx + 1}_alt"
            if move_checked(
                arm,
                step_name,
                pull_target,
                current_pull_quat,
                0.18,
            ):
                if step_idx == 0 and candidate_sign != pull_sign:
                    pull_sign = candidate_sign
                    print(f"Pull step 1 alt sign selected:{pull_sign:+.0f}")
                move_success = True
                break
            if step_idx > 0 or sign_idx == len(candidate_signs) - 1:
                continue
            print("Pull step 1 failed; retrying with alternate sign")
        if not move_success:
            if (
                best_progress is not None
                and best_progress > post_pull_vertical_stall_or_fail_progress_threshold
            ):
                should_run_vertical_finish = True
                if initial_pull_dir_xy is None:
                    initial_pull_dir_xy = current_pull_dir_xy.copy()
                print(
                    f"{control_name} pull step {step_idx + 1} failed after reaching "
                    f"{best_progress:.3f}; switching to vertical finish because progress is above "
                    f"{post_pull_vertical_stall_or_fail_progress_threshold:.3f}"
                )
                break
            refresh_planner_world("post-pull restore")
            return False, {"reason": f"pull step {step_idx + 1} failed"}
        if initial_pull_dir_xy is None:
            initial_pull_dir_xy = current_pull_dir_xy.copy()
            print(
                "Initial pull dir used:"
                f"{[round(float(x), 3) for x in initial_pull_dir_xy]}"
            )
        latest_oracle_after = get_oracle_targets()
        progress_after = door_progress(latest_oracle_after, control_name)
        if progress_after is not None:
            print(f"{control_name} door fraction after step {step_idx + 1}: {progress_after:.3f}")
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
                initial_pull_dir_xy = None
                print(
                    f"{control_name} probe progress after step {step_idx + 1} "
                    f"is {progress_gain:.3f} < "
                    f"{pull_sign_probe_progress_threshold:.3f}; "
                    f"flipping pull sign to {pull_sign:+.0f}"
                )
                continue
            if best_progress is None or progress_after > best_progress + post_pull_vertical_progress_eps:
                best_progress = progress_after
            else:
                if best_progress > post_pull_vertical_stall_or_fail_progress_threshold:
                    should_run_vertical_finish = True
                    print(
                        f"{control_name} progress stopped increasing above "
                        f"{post_pull_vertical_stall_or_fail_progress_threshold:.3f} "
                        f"({best_progress:.3f} -> {progress_after:.3f}); "
                        "switching to vertical finish"
                    )
                else:
                    print(
                        f"{control_name} progress stopped increasing "
                        f"({best_progress:.3f} -> {progress_after:.3f}); "
                        "stopping pull and moving to the next step"
                    )
                break
            if (
                not released_for_vertical_finish
                and progress_after > post_pull_release_trigger_progress
            ):
                released_for_vertical_finish = True
                print(
                    f"{control_name} progress exceeded "
                    f"{post_pull_release_trigger_progress:.3f}; opening gripper and "
                    f"doing {post_pull_release_extra_steps} more pull steps before vertical finish"
                )
                open_gripper(arm)
                time.sleep(0.10)
                for extra_idx in range(post_pull_release_extra_steps):
                    extra_oracle = get_oracle_targets()
                    extra_target = get_control_target(extra_oracle, control_name)
                    if extra_target is None:
                        refresh_planner_world("post-pull restore")
                        return False, {"reason": f"{control_name} target unavailable during release pull"}
                    _, extra_plan = cabinet_handle_geometry_v1(extra_target)
                    _, ee_pos_now, ee_quat_now = current_arm_pose()
                    extra_pull_dir_xy = vec(extra_plan["horizontal_tangent"]).copy() * pull_sign
                    extra_pull_dir_xy[2] = 0.0
                    extra_pull_dir_xy = _normalize(extra_pull_dir_xy)
                    extra_pull_quat = _make_press_quat(extra_pull_dir_xy)
                    extra_pull_target = ee_pos_now + extra_pull_dir_xy * step_distance
                    extra_pull_dir_yaw_deg = _xy_yaw_deg(extra_pull_dir_xy)
                    extra_pull_quat_delta_deg = _quat_delta_deg(
                        ee_quat_now,
                        extra_pull_quat,
                    )
                    print(
                        f"Release pull step {extra_idx + 1}/{post_pull_release_extra_steps}: "
                        f"delta={step_distance:.3f} "
                        f"dir_xy={[round(float(x), 3) for x in extra_pull_dir_xy]} "
                        f"dir_yaw_deg={extra_pull_dir_yaw_deg:.1f} "
                        f"quat_delta_deg={extra_pull_quat_delta_deg:.1f} "
                        f"target={[round(float(x), 3) for x in extra_pull_target]}"
                    )
                    if not move_checked(
                        arm,
                        f"release_pull_step_{extra_idx + 1}",
                        extra_pull_target,
                        extra_pull_quat,
                        0.18,
                    ):
                        if (
                            best_progress is not None
                            and best_progress > post_pull_vertical_stall_or_fail_progress_threshold
                        ):
                            print(
                                f"{control_name} release pull step {extra_idx + 1} failed after reaching "
                                f"{best_progress:.3f}; continuing to vertical finish"
                            )
                            break
                        refresh_planner_world("post-pull restore")
                        return False, {"reason": "failed release pull step"}
                    extra_progress = door_progress(get_oracle_targets(), control_name)
                    if extra_progress is not None:
                        print(
                            f"{control_name} door fraction after release pull step "
                            f"{extra_idx + 1}: {extra_progress:.3f}"
                        )
                        if best_progress is None or extra_progress > best_progress + post_pull_vertical_progress_eps:
                            best_progress = extra_progress
                should_run_vertical_finish = True
                print(f"{control_name} switching to vertical finish after release pull steps")
                break
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
                "Refreshing planner world after 3 pull steps with exclusions:"
                f" {exclude_prefixes}"
            )
            refresh_planner_world(
                "post-step3 pull collision refresh",
                exclude_body_prefixes=exclude_prefixes,
            )
            post_step3_refresh_done = True

    if (
        not should_run_vertical_finish
        and best_progress is not None
        and best_progress > post_pull_vertical_trigger_progress
    ):
        should_run_vertical_finish = True
        print(
            f"{control_name} finished pull steps above "
            f"{post_pull_vertical_trigger_progress:.3f}; running vertical finish"
        )

    if should_run_vertical_finish:
        if initial_pull_dir_xy is None:
            refresh_planner_world("post-pull restore")
            return False, {"reason": "missing initial pull direction for vertical finish"}
        pull_dir_xy = vec(initial_pull_dir_xy).copy()
        pull_dir_xy[2] = 0.0
        pull_dir_xy = _normalize(pull_dir_xy)
        vertical_candidates = [
            _normalize(np.array([pull_dir_xy[1], -pull_dir_xy[0], 0.0], dtype=float)),
            _normalize(np.array([-pull_dir_xy[1], pull_dir_xy[0], 0.0], dtype=float)),
        ]
        radial_target_xy = -_normalize(vec(radial_xy).copy())
        candidate_scores = [
            float(np.dot(candidate[:2], radial_target_xy[:2]))
            for candidate in vertical_candidates
        ]
        best_idx = int(np.argmax(candidate_scores))
        vertical_move_dir_xy = vertical_candidates[best_idx]
        print(f"Vertical finish radial:{[round(float(x), 3) for x in radial_xy]}")
        print(f"Vertical finish initial pull dir:{[round(float(x), 3) for x in pull_dir_xy]}")
        print(f"Vertical finish radial target:{[round(float(x), 3) for x in radial_target_xy]}")
        print(f"Vertical finish candidate scores:{[round(float(x), 3) for x in candidate_scores]}")
        print(f"Vertical finish chosen candidate:{best_idx}")
        print(f"Vertical finish move dir:{[round(float(x), 3) for x in vertical_move_dir_xy]}")

        vertical_pre_candidates = [
            ("center", vec(pre_grasp)),
            ("minus", vec(pre_grasp) - vertical_move_dir_xy * post_pull_vertical_pre_offset_m),
            ("plus", vec(pre_grasp) + vertical_move_dir_xy * post_pull_vertical_pre_offset_m),
        ]
        open_gripper(arm)
        time.sleep(0.10)
        print(
            "Refreshing planner world for vertical-finish pre-grasp with full collision "
            "geometry (no exclusions)"
        )
        refresh_planner_world("post-pull vertical finish pre-grasp collision-enabled")
        vertical_pre_reached = False
        for candidate_name, vertical_pre_grasp in vertical_pre_candidates:
            print(
                f"Vertical finish pre pose ({candidate_name}):"
                f"{[round(float(x), 3) for x in vertical_pre_grasp]}"
            )
            vertical_pre_result = move_with_orientation(
                arm,
                vertical_pre_grasp,
                grasp_quat,
            )
            vertical_pre_err = float(
                getattr(vertical_pre_result, "final_pos_error_m", 1e9) or 1e9
            )
            vertical_pre_status = getattr(vertical_pre_result, "status", "Unknown")
            print(
                f"vertical_finish_pre_grasp_{candidate_name}: "
                f"status={vertical_pre_status} err={vertical_pre_err:.4f} "
                f"pos={[round(float(x), 3) for x in vertical_pre_grasp]}"
            )
            if vertical_pre_status == "Success" and vertical_pre_err <= 0.18:
                vertical_pre_reached = True
                break

        if not vertical_pre_reached:
            print(
                "Failed to return to pre_grasp for vertical finish with collision "
                "avoidance; trying go_home and pre_grasp recovery without collision avoidance"
            )
            if not go_home_checked(
                arm,
                "vertical_finish final recovery go_home no-collision",
                exclude_body_prefixes=[""],
            ):
                refresh_planner_world("post-pull restore")
                return False, {
                    "reason": "failed to return to pre_grasp for vertical finish"
                }
            for candidate_name, vertical_pre_grasp in vertical_pre_candidates:
                vertical_pre_result = move_with_orientation(
                    arm,
                    vertical_pre_grasp,
                    grasp_quat,
                    exclude_body_prefixes=[""],
                )
                vertical_pre_err = float(
                    getattr(vertical_pre_result, "final_pos_error_m", 1e9) or 1e9
                )
                vertical_pre_status = getattr(vertical_pre_result, "status", "Unknown")
                print(
                    f"vertical_finish_pre_grasp_{candidate_name}_no_collision: "
                    f"status={vertical_pre_status} err={vertical_pre_err:.4f} "
                    f"pos={[round(float(x), 3) for x in vertical_pre_grasp]}"
                )
                if vertical_pre_status == "Success" and vertical_pre_err <= 0.18:
                    vertical_pre_reached = True
                    break

        if not vertical_pre_reached:
            refresh_planner_world("post-pull restore")
            return False, {"reason": "failed to return to pre_grasp for vertical finish"}

        print(
            "Disabling collision avoidance for final vertical steps after reaching "
            "vertical-finish pre-grasp"
        )
        refresh_planner_world(
            "vertical finish steps collision-disabled",
            exclude_body_prefixes=[""],
        )
        for step_idx in range(post_pull_vertical_steps):
            _, ee_pos_now, _ = current_arm_pose()
            vertical_target = ee_pos_now + vertical_move_dir_xy * post_pull_vertical_step_m
            print(
                f"Vertical finish step {step_idx + 1}/{post_pull_vertical_steps}: "
                f"delta={post_pull_vertical_step_m:.3f} "
                f"dir_xy={[round(float(x), 3) for x in vertical_move_dir_xy]} "
                f"target={[round(float(x), 3) for x in vertical_target]}"
            )
            if not move_checked(
                arm,
                f"vertical_finish_step_{step_idx + 1}",
                vertical_target,
                grasp_quat,
                0.20,
                exclude_body_prefixes=[""],
            ):
                refresh_planner_world("post-pull restore")
                return False, {"reason": "failed vertical finish step"}
            vertical_progress_oracle = get_oracle_targets()
            vertical_progress = door_progress(vertical_progress_oracle, control_name)
            if vertical_progress is None:
                print(
                    f"{control_name} door fraction after vertical step {step_idx + 1}: unavailable "
                    f"fixture_state={vertical_progress_oracle.get('fixture_state')}"
                )
            else:
                print(
                    f"{control_name} door fraction after vertical step "
                    f"{step_idx + 1}: {vertical_progress:.3f}"
                )
            if (
                vertical_progress is not None
                and vertical_progress >= door_open_progress_threshold
            ):
                print(
                    f"{control_name} reached open threshold "
                    f"{door_open_progress_threshold:.2f} during vertical finish step "
                    f"{step_idx + 1}"
                )
                time.sleep(0.5)
                go_home_checked(
                    arm,
                    "vertical_finish go_home",
                    exclude_body_prefixes=[""],
                )
                return True, {"best_progress": vertical_progress}

    refresh_planner_world("post-pull restore")
    open_gripper(arm)
    time.sleep(0.10)
    return True, {"best_progress": best_progress}
