"""Pick one visible cube with the standard real-YAM vision/planning stack."""

import os

from skill_library.pick import pick_object

grasp_mode = os.environ.get("ENPIRE_CUBE_GRASP_MODE", "anygrasp")
picked_side = pick_object("cube", camera="top", grasp_mode=grasp_mode)
if picked_side is None:
    raise RuntimeError("No collision-free cube grasp was found")


def get_task_info():
    """Result contract consumed by run_script.py."""
    return {
        "success": picked_side in {"left", "right"},
        "reward": 1.0 if picked_side in {"left", "right"} else 0.0,
        "picked_side": picked_side,
        "grasp_mode": grasp_mode,
    }
