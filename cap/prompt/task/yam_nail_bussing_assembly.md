# YAM Nail Bussing — Assembly Guide

Write top-level assembly code that imports the flat pin-insert skill library and orchestrates it. Do not define helper functions. Do not use `try`/`except`. Do not write new `@skill`s in assembly.

## Exact skill contracts — do not guess

The curated YAM pin-insert skills are not uniformly `(success, log)`.
Use these exact call signatures and return values:

```python
detections, det_log = detect_all_nails_v*()
grasps, plan_log = plan_nail_grasps_v*(mask, detection_index)
ranked, rank_log = rank_grasps_by_ik_v*(grasps, side)
move_ok, move_log = move_to_target_v*(side, grasp_candidate)
grasp_ok, grasp_log = grasp_at_current_pose_v*(side, grasp_candidate)
transport_ok, transport_log = transport_via_birdseye_v*(side)
drop_pos, target_log = estimate_target_drop_pos_v*(TARGET_NAME)
drop_ok, drop_log = drop_nail_on_target_v*(side, drop_pos)
verify_ok, verify_log = verify_nails_on_plate_v*(TARGET_NAME)
```

Critical: `detect_all_nails_v*()` returns the detection list as the **first**
value. Never write `s_detect, detect_log = detect_all_nails_v*()` and never
read detections from `detect_log["detections"]`; that silently skips all
motion skills. Similarly, `plan_nail_grasps_v*()` returns the grasp list as
the first value, and `rank_grasps_by_ik_v*()` returns the ranked grasp list as
the first value.

## Required flow

Use this sequence as the baseline:

1. Import the highest visible versions of these skill families from `skill_library.*`:
   - `detect_all_nails`
   - `plan_nail_grasps`
   - `rank_grasps_by_ik`
   - `move_to_target`
   - `grasp_at_current_pose`
   - `transport_via_birdseye`
   - `estimate_target_drop_pos`
   - `drop_nail_on_target`
   - `verify_nails_on_plate`
2. Loop for at most five rounds.
3. Call `detections, det_log = detect_all_nails_v*()`.
4. If it returns no actionable detections, call `verify_nails_on_plate_v*()`. Stop only if the verifier succeeds or if `MAX_NO_PROGRESS_ROUNDS` consecutive perception rounds fail. If the verifier is not successful, print that SAM3 has no actionable masks but VLM did not confirm completion, increment the no-progress counter, and continue to the next round.
5. For each detection, extract its mask with `det["mask"] if isinstance(det, dict) else det.mask`.
6. Call `grasps, plan_log = plan_nail_grasps_v*(mask, detection_index)`; if no grasps are returned, call `go_home()` and continue.
7. Use `plan_log["side"]` as the chosen arm, then call `rank_grasps_by_ik_v*`.
8. Try ranked grasps with separate credit assignment: first call `move_to_target_v*`; only if it succeeds, call `grasp_at_current_pose_v*`. Use up to three attempts per candidate.
9. After grasp success, call `transport_via_birdseye_v*`, `estimate_target_drop_pos_v*`, and `drop_nail_on_target_v*`.
10. Call `go_home()` after every placed nail and at final exit.
11. End by printing `Success: {get_task_info().get("success")}` and the reward/status.

## Safety constraints

This is real hardware. Use one seed only. If a step returns failure or uncertainty, branch conservatively and print the log. Never command exploratory movements based only on a VLM guess.

If SAM3 calls fail with `Connection refused`, do not work around it by changing the assembly to a different detector. Print the infrastructure error and stop safely; the expected fix is to restore the tunnel/ports for `127.0.0.1:6767`, not to discard the curated SAM3 detection skill.

## Known-good assembly skeleton

```python
from skill_library.detect_all_nails import detect_all_nails_v1
from skill_library.plan_nail_grasps import plan_nail_grasps_v1
from skill_library.rank_grasps_by_ik import rank_grasps_by_ik_v1
from skill_library.move_to_target import move_to_target_v1
from skill_library.grasp_at_current_pose import grasp_at_current_pose_v1
from skill_library.transport_via_birdseye import transport_via_birdseye_v1
from skill_library.estimate_target_drop_pos import estimate_target_drop_pos_v1
from skill_library.drop_nail_on_target import drop_nail_on_target_v1
from skill_library.verify_nails_on_plate import verify_nails_on_plate_v1

TARGET_NAME = "blue plate"
MAX_ROUNDS = 5
MAX_GRASP_ATTEMPTS = 3
MAX_NO_PROGRESS_ROUNDS = 3

no_progress_rounds = 0
total_moved = 0
for round_num in range(1, MAX_ROUNDS + 1):
    print(f"\n{'=' * 60}")
    print(f"=== Round {round_num}/{MAX_ROUNDS} ===")
    print(f"{'=' * 60}")

    detections, det_log = detect_all_nails_v1()
    print(f"detect_all_nails_v1: log={det_log}")
    if not detections:
        all_on_plate, verify_log = verify_nails_on_plate_v1(TARGET_NAME)
        print(f"verify_nails_on_plate_v1: log={verify_log}")
        if all_on_plate:
            print("No more off-plate nails detected, and VLM confirms completion. Done!")
            break
        print("SAM3 returned no actionable off-plate nail masks, but VLM did not confirm completion.")
        no_progress_rounds += 1
        print(f"No detection progress ({no_progress_rounds}/{MAX_NO_PROGRESS_ROUNDS}).")
        if no_progress_rounds >= MAX_NO_PROGRESS_ROUNDS:
            print("Stopping: perception did not produce actionable masks for too many rounds.")
            break
        continue

    moved_any = False
    for i, det in enumerate(detections):
        print(f"\n{'=' * 50}")
        print(f"Picking nail #{i + 1} -> {TARGET_NAME}")
        print(f"{'=' * 50}")

        mask = det["mask"] if isinstance(det, dict) else det.mask
        grasps, plan_log = plan_nail_grasps_v1(mask, i)
        print(f"plan_nail_grasps_v1[{i}]: log={plan_log}")
        if not grasps:
            go_home()
            continue

        chosen_side = plan_log["side"]
        ranked, rank_log = rank_grasps_by_ik_v1(grasps, chosen_side)
        print(f"rank_grasps_by_ik_v1[{i}]: log={rank_log}")
        if not ranked:
            go_home()
            continue

        grasped = False
        selected_idx = None
        for grasp_idx, cand in enumerate(ranked, start=1):
            print(f"\n  --- Trying grasp {grasp_idx}/{len(ranked)}: {cand} ---")
            for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
                print(f"    attempt {attempt}/{MAX_GRASP_ATTEMPTS}")
                move_ok, move_log = move_to_target_v1(chosen_side, cand)
                print(f"move_to_target_v1: log={move_log}")
                if not move_ok:
                    print("    move-to-target failed")
                    go_home()
                    continue
                grasp_ok, grasp_log = grasp_at_current_pose_v1(chosen_side, cand)
                print(f"grasp_at_current_pose_v1: log={grasp_log}")
                if grasp_ok:
                    grasped = True
                    selected_idx = grasp_idx
                    break
                print("    grasp miss")
                go_home()
            if grasped:
                break

        if not grasped:
            print(f"  Giving up on nail #{i + 1} (no grasp worked)")
            go_home()
            continue

        _, transport_log = transport_via_birdseye_v1(chosen_side)
        print(f"transport_via_birdseye_v1: log={transport_log}")

        drop_pos, target_log = estimate_target_drop_pos_v1(TARGET_NAME)
        print(f"estimate_target_drop_pos_v1: log={target_log}")

        placed, drop_log = drop_nail_on_target_v1(chosen_side, drop_pos)
        print(f"drop_nail_on_target_v1: log={drop_log}")
        go_home()

        if placed:
            moved_any = True
            total_moved += 1
            print(f"  Nail #{i + 1} placed on {TARGET_NAME}! selected_grasp_index={selected_idx}")

    if moved_any:
        no_progress_rounds = 0
    else:
        no_progress_rounds += 1
        print(f"No nails picked this round ({no_progress_rounds}/{MAX_NO_PROGRESS_ROUNDS} consecutive no-progress).")
        if no_progress_rounds >= MAX_NO_PROGRESS_ROUNDS:
            print("Stopping: no progress for too many rounds.")
            break

go_home()
print(f"\nNail bussing complete! moved={total_moved}")
final_info = get_task_info()
print(f"Success: {final_info.get('success', False)}")
print(f"Reward: {final_info.get('reward', 0.0)}")
print(f"Status: {final_info.get('status', 'unknown')}")
```
