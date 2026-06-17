# Saved Script Refactor Plan

## Goal

Refactor the provided saved scripts into a clear hierarchy that maximizes reuse while preserving task meaning:

- `cap/agent/tools`: low-level primitives, minimal policy.
- `cap/saved_scripts/skill_library`: reusable robotics APIs that wrap tools into compact behaviors.
- task scripts: short orchestration files containing object names, class/target mappings, retry/FSM sequencing, and only task-specific constants.

The refactor must be an encapsulation change, not a behavior change. Numerical constants and retry semantics from the existing scripts must be inherited unless explicitly overridden by a task script.

Target script sizes:

- `cap/saved_scripts/table_bussing/nclass_sorting.py`: under 100 lines.
- `cap/saved_scripts/small_object/small_obj_sort.py`: under 100 lines.
- reset powerstrip / adapter plug-unplug orchestration: under 200 lines per top-level task script where feasible.

## Tool Layer

Keep `/home/forge/Project/forge/cap/agent/tools` mostly unchanged.

Required tool-layer actions:

1. Reuse existing `cap/agent/tools/rotate_joint.py`.
   - Do not reimplement joint trajectory generation in saved scripts.
   - Ensure `rotate_joint` is available in the saved-script namespace so skills can call it.

2. Reuse existing motion and perception tools:
   - `freespace_move`
   - `move_joint_keypoints`
   - `sample_grasp_pose_anygrasp`
   - `sample_grasp_pose_2d`
   - `sample_grasp_pose_3d_bb`
   - `detect_objects_oneshot`
   - `vlm_query`

3. Only add tool changes if strictly required for runtime registration or structured debug data. Do not move task policy into tools.

## Skill Library Layout

Add:

```text
cap/saved_scripts/skill_library/
  constants/
    __init__.py
    planning.py
    vision.py
    robot.py
    manipulation.py

  trajopt.py
  grasp_geometry.py
  vlm_classification.py
  reorient.py
  pick_place.py
```

Use intuitive function names. Do not use `_v1` suffixes.

Each skill module should:

- import `from skill_library.namespace import *`
- import `skill` from `cap.agent.skill_registry`
- expose concise functions decorated with `@skill` when they are intended to be loaded/called directly
- keep comments focused on non-obvious policy boundaries and equivalence constraints

## Constants

Move duplicated defaults into constants modules. Task-specific overrides remain in task scripts or function arguments.

### `constants/planning.py`

Include:

- `PLANNING_SPEED`
- `IK_ERROR_THRESHOLD_M`
- `IK_XYZ_WEIGHT`
- `IK_RPY_WEIGHT`
- `BATCH_TOP_K`
- `BATCH_SOLVER_SPEED`
- `BATCH_VALIDATE_TRAJECTORY`
- `MOTION_PLANNER_BACKEND`
- `DEFAULT_XYZ_RELAXATIONS`
- `DEFAULT_RPY_RELAXATIONS`

### `constants/vision.py`

Include:

- `VLM_BACKEND`
- `VLM_MODEL`
- `DEFAULT_VLM_CAMERAS`
- `NUM_VOTES`
- `MAJORITY`
- `BUNDLESDF_CAMERA`
- `ANYGRASP_TCP_OFFSET_Z_M`
- `ANYGRASP_DISABLE_PLANNER_Z_CLIPPING`

### `constants/robot.py`

Include:

- `LEFT_HOME_XYZ`
- `RIGHT_HOME_XYZ`
- `HOME_VIEW_Z_OFFSET`
- `LEFT_BIRDEYE_VIEW_RPY`
- `RIGHT_BIRDEYE_VIEW_RPY`

### `constants/manipulation.py`

Include:

- `GRIPPER_WIDTH_M`
- `GRIP_TIGHT_M`
- `HOVER_CLEARANCE_M`
- `HOVER_CLEARANCE_CANDIDATES_M`
- `LIFT_HEIGHT_M`
- `J6_SPEED_DEG_S`
- `J6_LIMIT_RAD`
- `KEYPOINT_SPACING_DEG`

## Core Skills

### `trajopt.py`

Purpose: reusable trajectory optimization policy around `freespace_move`.

Required functions:

- `pose_candidates(target_pos, target_rpy=None, target_quat=None, score=1.0, width=None, label="target")`
- `relax_candidates(candidates, pos_offsets=None, rpy_offsets=None, score_decay=0.95)`
- `solve_candidates(candidates, side, batch_top_k=None, label="trajopt", pad=True)`
- `solve_groups(groups, side, relax=False, pos_offsets=None, rpy_offsets=None, group_labels=None)`
- `execute_solution(solution, side=None)`
- `safe_move(side, target_pos=None, target_rpy=None, target_quat=None, trajectory_cache_key=None)`

Important behavior:

- Preserve the existing batched cuRobo ranking semantics.
- Support solving with or without relaxation.
- Support candidate groups for parallel/fallback optimization.
- Return structured metadata: feasible count, selected pose, trajectory cache key, failure reasons.

### `grasp_geometry.py`

Purpose: generate grasp poses and object geometry. It should not execute robot motion.

Required functions:

- `sample_anygrasp(object_name, camera="top", max_grasps=None, tcp_offset_z_m=None, disable_planner_z_clipping=None, clip_min_z=None)`
- `sample_obb(object_name, camera="top", queries=None, tcp_offset_z_m=0.0)`
- `sample_2d(object_name, camera="top", grasp_z_m=None, max_grasps=10, return_debug=True, reuse_cached_frame=False)`
- `detect_part_yaw(part_query, body_query, camera, plane_z_m=None, publish_debug=True)`
- `detect_signed_axis(body_query, sign_query, camera="top")`
- `topdown_candidates(center_xyz, yaw_deg, width=None, xy_offsets=None, yaw_offsets=(0.0, 180.0), score=1.0)`
- `tilted_axis_candidates(ref_xy, axis_xy, half_len, fractions, tilt_dir, tilt_deg, grasp_z, score_fn=None, width=None)`
- `detect_obb_safe_state(object_name, camera, queries=None, rel_tol=None, abs_tol_m=None)`

Important behavior:

- Inherit AnyGrasp, 2D top-down, OBB, prong yaw, strip-axis, and debug-overlay behavior from the current scripts.
- Later migration from `origin/jl/2dgrasp` should land here, especially OBB and debug image logic.

### `vlm_classification.py`

Purpose: backend-selectable VLM multiclass classification. No motion behavior.

Required functions:

- `query(prompt, cameras=None, backend=None, model=None)`
- `build_assignments(classes, targets)`
- `build_prompt(assignments)`
- `parse_response(response, assignments)`
- `survey(classes, targets, cameras=None, backend=None, model=None, num_votes=None, majority=None, go_birdeye=True)`

Important behavior:

- Backend and model must be first-class parameters.
- Preserve robust JSON/bracket/list parsing from table and small-object scripts.
- Preserve vote aggregation and majority filtering.

### `reorient.py`

Purpose: reusable reorientation and bimanual manipulation policy.

Required functions:

- `rotate_joint_checked(side, joint, delta_deg, speed_deg_s=None, limit_rad=None, settle_s=0.25, min_delta_deg=1.0)`
- `rotate_in_hand(side, delta_deg, joint=6)`
- `gentle_close(side, torque_limit=None)`
- `grip_tight(side, threshold=None)`
- `assign_roles(reference_xy)`
- `reorient_by_part_axis(object_query, part_query, target_yaw_range, camera_side=None, max_attempts=999)`
- `reorient_axis(object_query, desired_axis="y", camera="top", max_attempts=999)`
- `wobble_along_axis(side, axis_xy, duration_s=None, steps=None, cycles=None, amplitude_m=None, lift_m=None)`

Important behavior:

- Wrap existing `rotate_joint` tool.
- Preserve adapter prong reorientation, strip reorientation, torque/gentle-close checks, bimanual role assignment, and unplug wobble semantics.

### `pick_place.py`

Purpose: compact object transport using VLM, grasp geometry, and trajopt.

Required functions:

- `birdseye_pose(side)`
- `go_birdeye(side="both", planning_speed=None)`
- `choose_arm(xyz)`
- `pick_object(object_name, camera="top", grasp_mode="anygrasp", max_attempts=None, **kwargs)`
- `estimate_drop(target_name, z_offset=None, camera=None)`
- `place_object(side, target_name=None, drop_pos=None, transport_rpy=None)`
- `pick_and_place(object_name, target_name, grasp_mode="anygrasp", **kwargs)`

Important behavior:

- Preserve table bussing pick/place semantics.
- Preserve small-object 2D top-down fallback semantics via `grasp_mode`.
- Keep task scripts mostly declarative.

## Task Refactors

### Table Bussing

Refactor `cap/saved_scripts/table_bussing/nclass_sorting.py` to:

- define `TARGETS`, `CLASSES`, task-specific drop offsets if needed
- keep the VLM survey, class-target loop, pick/place call, no-progress retry, and resurvey visible
- import a reusable run profile from constants and override only task-specific keys
- keep `nclass_sorting_nvidiagemini.py` equivalent except backend/model selection if the file is still required

Target: under 100 lines.

### Small Object Sorting

Refactor `cap/saved_scripts/small_object/small_obj_sort.py` to:

- define `TARGETS`, `CLASSES`, local pose/drop overrides if needed
- keep the VLM survey, class-target loop, wrist-2D pick/place call, no-progress retry, and resurvey visible
- import a reusable run profile from constants and override only task-specific keys

Target: under 100 lines.

### Reset Powerstrip / Adapter

Refactor only the provided reset-powerstrip scripts:

- `before_plug.py`
- `unplug.py`
- `adapterstate/1_move_to_comfort.py`
- `adapterstate/2_reorient_adapter_v2.py`
- `adapterstate/3_grasp_reorient_power_strip.py`
- `adapterstate/4_grasp_adapter_hover_power_strip.py`

Scripts should retain their FSM sequencing and object names but delegate geometry, trajopt, joint spin, role assignment, and wobble behavior to skills.

Target: under 200 lines for top-level plug/unplug orchestration where feasible.

## Equivalence Requirements

A judge pass must compare pre-refactor task scripts with refactored scripts and verify:

- object names are preserved
- class/target mappings are preserved
- numerical constants are either preserved in task files or moved into constants with the same values
- retry limits and stop conditions are preserved
- planner backend, batch size, solver speed, IK thresholds, gripper thresholds, torque values, lift/wobble values, and yaw tolerances are preserved
- VLM backend/model behavior is preserved unless explicitly parameterized
- no task behavior is silently changed while shrinking scripts

Allowed changes:

- moving code into skill library
- renaming helper functions
- replacing repeated local helper logic with shared skill calls
- adding explicit keyword arguments to preserve task-specific values

Disallowed changes:

- changing constants without noting it
- changing object queries silently
- removing fallback/retry behavior
- changing task sequencing
- replacing a specialized behavior with a simpler behavior that only works for one case
