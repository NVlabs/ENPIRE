# YAM Nail Bussing — Skill Author Guide

Task: place all visible small black nails or screws on the table onto the blue plate.

The human-curated seed library is `cap/saved_scripts/yam_autorl/pin_insert`. It is a flat single-layer skill library. Each `@skill` must be atomic and must not call another `@skill`; composite behavior belongs only in assembly code.

## Existing skill families

The seed library should already provide these atomic mechanisms:

- `detect_all_nails_v1`: SAM3 top-camera segmentation for candidate nails, filtering nails whose centroid is already inside the filled blue-plate mask.
- `plan_nail_grasps_v1`: generate top-down 2D grasp candidates from one nail mask and select the closer arm.
- `rank_grasps_by_ik_v1`: batch-rank candidate grasps with cuRobo feasibility.
- `move_to_target_v1`: open gripper and move to one selected grasp target; does not close the gripper.
- `grasp_at_current_pose_v1`: close the gripper at the current pose; does not move the arm.
- `transport_via_birdseye_v1`: move the carrying arm through the high transport pose.
- `estimate_target_drop_pos_v1`: detect the blue plate target and return an above-plate drop position.
- `drop_nail_on_target_v1`: compensate fingertip offset, move to drop pose, and release.
- `verify_nails_on_plate_v1`: VLM verifier for whether all visible nails are on the blue plate.

## Authoring policy

On the first iteration, do not author new skills unless the index is missing one of the mechanisms above. The existing skills are human-tuned for real YAM; the assembly step should deploy them first.

SAM3 is expected to be reachable through the table-bussing SSH tunnel at `runtime.sam3_host:runtime.sam3_port` (normally `127.0.0.1:6767`). If a run reports `Connection refused`, `No RGB image`, or a failed `/segment_all` call, treat it as a runtime/tunnel/camera configuration problem unless the operator explicitly says to redesign perception. Do not replace the SAM3-based nail detection path with a lower-fidelity fallback solely because the service was unreachable.

After a failed run, refine exactly the failing atomic skill indicated by stdout or reward feedback. Examples:

- If SAM3 misses nails, refine `detect_all_nails` thresholds or query strings.
- If grasp candidates are poor, refine `plan_nail_grasps` grasp Z, mask handling, or candidate count.
- If IK rejects good grasps, refine `rank_grasps_by_ik` ranking/solver parameters.
- If the arm does not reach the selected grasp pose, refine `move_to_target` or the motion acceptance threshold/logging.
- If the arm reaches the pose but no nail is picked, refine `grasp_at_current_pose` or the grasp candidate geometry.
- If the nail is grasped but dropped in the wrong place, refine `estimate_target_drop_pos` or `drop_nail_on_target`.
- If final reward says unsure despite no SAM3 detections, refine `verify_nails_on_plate` prompt/model only.

Do not add a central constants file. Keep tunable values local to the skill whose behavior they affect, so future optimization is localized and attributable.
