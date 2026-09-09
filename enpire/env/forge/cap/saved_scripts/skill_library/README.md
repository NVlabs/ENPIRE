# Calling atomic CaP skills

These scripts use the live `skill_library.namespace` injected by ENPIRE's
`run_script.py`, exactly like the other saved CaP scripts. They neither start
their own robot controller nor implement another perception/planning backend.
Importing them does not connect to hardware.

From another saved script:

```python
from skill_library.sample_grasp_pose_3d_bb import sample_grasp_pose_3d_bb
from skill_library.move_to_pose import move_to_pose

observation = sample_grasp_pose_3d_bb("blue cube", camera="top")
object_pose = observation["object_pose"]
grasp = observation["grasps"][0]

# Inspect the observation, choose a collision-free approach and sufficient
# payload clearance, then pass the chosen pose to the motion skill.
report = move_to_pose(
    "left", target_pos=grasp["position"], target_rpy=grasp["rpy"],
    gripper_before=1.0, gripper_after=0.0, preview_only=True,
)
```

For CLI calls, run from the Forge directory in the configured environment:

```bash
cd enpire/env/forge
export ENPIRE_YAM_STATION=my-yam
export ENPIRE_STATION="$ENPIRE_YAM_STATION"

# Reuse the ordinary runner, station settings, planner, logging and error path.
cap_run=(../../../.venv/bin/python run_script.py
  robot=real_yam env.name=yam-real
  skill_library_path=cap/saved_scripts/skill_library
  robot.dashboard=false robot.await_exit=false robot.go_home_on_exit=false
  runtime.exit_on_error=true)

"${cap_run[@]}" \
  script_file=cap/saved_scripts/skill_library/sample_grasp_pose_3d_bb.py \
  script_function=sample_grasp_pose_3d_bb \
  '+script_kwargs={object_name:"blue cube",camera:top}' \
  script_output_dir=outputs/cube_pose

# Preview example coordinates; replace them with a reviewed world-frame target.
# Set preview_only:false to explicitly execute after station, calibration and
# collision-planning preflight. A preview issues NO gripper commands.
"${cap_run[@]}" \
  script_file=cap/saved_scripts/skill_library/move_to_pose.py \
  script_function=move_to_pose \
  '+script_kwargs={side:left,target_pos:[0.5,0.1,0.95],target_rpy:[0,180,90],gripper_before:1,gripper_after:0,preview_only:true}' \
  script_output_dir=outputs/move_pose
```

Hydra needs the leading `+` when filling the initially empty `script_kwargs`
mapping. Each call returns JSON at `result.json` → `details.return_value` and
records tool timing in `profiling.json`. Check the exit code and top-level
`success`; preview success means planning succeeded, not that motion occurred.
The `robot.go_home_on_exit=false` setting keeps the endpoint/gripper state for
the next atomic call, and prevents a perception-only call from homing the arm.

`move_to_pose` uses world-frame metres and XYZW quaternions. `target_rpy` uses
CaP display degrees, not standard XYZ Euler angles. Omitted pose components
preserve measured values. Grippers use normalized commands, 0 closed and 1
open. Both gripper arguments are optional; omit both to preserve the gripper.
Optional `gripper_vel_limit` and `gripper_torque_limit` pass through to the
registered tool; omit them to retain station settings.

The before-action runs before planning, so the planner sees its resulting
gripper state. A reported unfinished before-action stops the call. The
after-action runs only after stable measured arm convergence (defaults: 5 mm,
3 degrees, 3 consecutive samples). A planning/pose failure prevents it; the
before-action is not undone. Preview plans with current gripper geometry, so
execution replans after any before-action. Gripper feedback is included in
the report; it is not an object-retention test, especially for UMI grippers.

`sample_grasp_pose_3d_bb` wraps the registered SAM3 + calibrated RGB-D + RANSAC
tool. It returns an `object_pose` (OBB centre, XYZW quaternion and rotation
matrix), `extents_m`, `top_surface_z_m`, `n_points`, observation call timestamps,
and separate proposed `grasps`. Multiple grasps are alternatives for one object.
For repeated objects, supply a specific description and `image_bbox` in pixel
`[xmin,ymin,xmax,ymax]` coordinates. Optional `min_world_z`/`max_world_z` filter
depth points through the underlying tool.

OBB axes have arbitrary signs/order, particularly for symmetric cubes, and
occlusion/depth noise can bias dimensions and centre. Returned quality fields
do not claim measured accuracy or a calibrated confidence score. Grasp scores
are heuristic rankings. Review size, height, camera evidence and clearance
before executing a pick. The OBB centre is not itself an EEF target. This
script does not invoke BundleSDF or AnyGrasp.

The prompted pickup entrypoint is now `skill_library/pick_object.py` and remains
available through `enpire cap run pickup`. It uses the existing `pick.py` policy;
the atomic scripts do not certify that older policy's grasp-retention checks
or its complete pick-and-place behavior on a particular gripper.
