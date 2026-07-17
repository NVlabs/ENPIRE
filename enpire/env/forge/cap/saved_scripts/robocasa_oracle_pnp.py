# Oracle pick-and-place demo for RoboCasa, runnable directly via run_script.py.
#
# IMPORTANT: run_script.py builds the tool namespace with runtime_role="script",
# which DROPS the top-level get_task_info()/reset_env(). This script therefore
# reads object positions via detect_object(backend="oracle") (which works in
# script role — it calls the env oracle internally). The success SCORE is computed
# by the oracle reward module after the script finishes, so the script does not
# need get_task_info at all.
#
# Run:
#   MUJOCO_GL=egl ROBOCASA_SEED=42 ROBOCASA_LAYOUT_ID=3 ROBOCASA_STYLE_ID=5 \
#     uv run python run_script.py \
#       script_file=cap/saved_scripts/robocasa_oracle_pnp.py \
#       env.name=robocasa:PickPlaceSinkToCounter env.seed=42 \
#       runtime.curobo_port=0 recording.enabled=true

import numpy as np

SIDE = "right"


def _oracle_pos(*queries):
    """Return the first oracle object position matching any query, else None."""
    for q in queries:
        try:
            dets = detect_object(q, backend="oracle")  # noqa: F821
        except Exception as exc:  # no match raises with the available names
            print(f"[oracle_pnp] detect_object({q!r}) -> {exc}")
            continue
        if dets:
            print(f"[oracle_pnp] matched {q!r} -> {dets[0].label} @ {dets[0].position_3d}")
            return np.asarray(dets[0].position_3d, dtype=float)
    return None


print("[oracle_pnp] task:", get_task_description())  # noqa: F821

obj = _oracle_pos("obj", "object")
if obj is None:
    raise RuntimeError("could not locate the pick object via oracle detection")

place = _oracle_pos("counter", "container", "target")
if place is None:
    # Fallback: just lift and shift aside so the run still produces a result/video.
    place = obj + np.array([0.0, -0.25, 0.05])
    print("[oracle_pnp] no place target found; using fallback offset")

UP = np.array([0.0, 0.0, 0.15])

# --- pick ---
open_gripper(SIDE)  # noqa: F821
freespace_move(right_target_pos=(obj + UP).tolist(), side=SIDE)  # noqa: F821  hover
freespace_move(right_target_pos=(obj + np.array([0, 0, 0.02])).tolist(), side=SIDE)  # noqa: F821 descend
close_gripper(SIDE, vel_limit=2.0, torque_limit=0.4)  # noqa: F821  compliant close
freespace_move(right_target_pos=(obj + UP).tolist(), side=SIDE)  # noqa: F821  lift

# --- place ---
freespace_move(right_target_pos=(place + UP).tolist(), side=SIDE)  # noqa: F821  hover over target
freespace_move(right_target_pos=(place + np.array([0, 0, 0.05])).tolist(), side=SIDE)  # noqa: F821 lower
open_gripper(SIDE)  # noqa: F821  release
freespace_move(right_target_pos=(place + UP).tolist(), side=SIDE)  # noqa: F821  retreat

print("[oracle_pnp] done — oracle reward module will score this run")
