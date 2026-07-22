# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical right-arm hover pose for the GPU insertion task.

Single source of truth for where the right arm (the motherboard-fixture holder)
rests during GPU insertion. The SAME pose is applied at train and test time so
the policy's observation is identical in both:

  * data collection / learning:
        rl/handlers.py:do_hover -> _maybe_move_right_to_hover (imports this module)
  * full-loop inference:
        tmux/realworld_rl/gpu_insertion_dual_full_cycle.sh -> rl_gear.sh ->
        the same RL runner -> do_hover -> _maybe_move_right_to_hover

Because both data collection and the full-loop run the same RL runner, editing
the pose here updates both paths at once.

Pose captured with `uv run tools/debug/yam_gravcomp_viewer.py --both`.

Run standalone (moves the right arm to the hover pose and exits):
    uv run python run_script.py robot=real_yam env.name=yam-real \
        script_file=cap/saved_scripts/gpu/right_arm_hover.py \
        skill_library_path=cap/saved_scripts/skill_library
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Right wrist EEF pose in the world frame (captured via the gravcomp viewer).
RIGHT_HOVER_EEF_POSITION = (0.5526, 0.1032, 0.7923)
RIGHT_HOVER_EEF_QUAT_XYZW = (-0.7933, -0.0083, 0.0017, 0.6088)
# Informational only (equivalent orientation in degrees): roll/pitch/yaw.
RIGHT_HOVER_EEF_RPY_DEG = (-104.99, -0.43, 0.88)


def right_hover_target_ee_pose() -> dict[str, Any]:
    """The right-arm hover target in the env's ``target_ee_pose`` mapping form."""
    return {
        "right": {
            "position": list(RIGHT_HOVER_EEF_POSITION),
            "quat_xyzw": list(RIGHT_HOVER_EEF_QUAT_XYZW),
        }
    }


def move_right_arm_to_hover(env: Any, *, log=print) -> dict[str, np.ndarray]:
    """Drive ONLY the right arm to the canonical hover EEF pose via seeded IK.

    ``env`` is the unwrapped ``YamRealEnv``. The left side is seeded/held at its
    current joint state, so only the right arm moves. Uses the env's direct
    joint interpolation, which bypasses the recording wrapper (so it never
    finalizes/discards the active episode). ``_interpolate_to_target`` skips
    motion when the arm is already within tolerance.
    """
    current = env._read_current_joint_state()
    target_state = env._get_joint_position_target_from_ee_pose_target(
        right_hover_target_ee_pose(),
        seed_joint_state=current,
    )
    log(
        "[right_arm_hover] right -> hover "
        f"pos={list(RIGHT_HOVER_EEF_POSITION)} "
        f"quat_xyzw={list(RIGHT_HOVER_EEF_QUAT_XYZW)} "
        f"joints={np.round(target_state['right_joint_pos'], 4)}"
    )
    env._interpolate_to_target(target_state)
    return target_state


# --- Optional standalone execution via run_script.py ------------------------
# run_script.py exec's this file with skill-library callables injected into the
# module globals (e.g. ``get_robot_state``). Only drive the robot when that
# context is present; a plain ``import`` (from the RL runner) is a no-op.
def _direct_env_from_get_robot_state(get_robot_state_fn) -> Any:
    obj = get_robot_state_fn()
    for candidate in (obj, getattr(obj, "robot", None)):
        env = getattr(candidate, "_env", None)
        if env is not None and hasattr(env, "_interpolate_to_target"):
            return env
    raise RuntimeError("direct YAM env not available for right-arm hover")


_RIGHT_HOVER_COMPLETED = False

if "get_robot_state" in globals():
    move_right_arm_to_hover(
        _direct_env_from_get_robot_state(globals()["get_robot_state"])
    )
    _RIGHT_HOVER_COMPLETED = True

    def get_task_info() -> dict:
        return {
            "success": bool(_RIGHT_HOVER_COMPLETED),
            "reward": 1.0 if _RIGHT_HOVER_COMPLETED else 0.0,
        }

