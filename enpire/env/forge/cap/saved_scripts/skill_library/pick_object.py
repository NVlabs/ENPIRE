# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick one visible object described by the caller's text prompt."""

import os

from skill_library.pick import lift_grasped_object, pick_object

object_prompt = os.environ.get("ENPIRE_PICK_PROMPT", "").strip()
if not object_prompt:
    raise RuntimeError("ENPIRE_PICK_PROMPT must describe the visible object to pick")

camera = os.environ.get("ENPIRE_PICK_CAMERA", "top").strip() or "top"
grasp_mode = os.environ.get("ENPIRE_PICK_GRASP_MODE", "anygrasp").strip() or "anygrasp"
picked_side = pick_object(object_prompt, camera=camera, grasp_mode=grasp_mode)
if picked_side is None:
    raise RuntimeError(f"No collision-free grasp was found for {object_prompt!r}")
lifted_and_retained = lift_grasped_object(picked_side)
if not lifted_and_retained:
    raise RuntimeError(f"The {object_prompt!r} grasp was not retained after lifting")


def get_task_info():
    """Result contract consumed by run_script.py."""
    success = picked_side in {"left", "right"} and lifted_and_retained
    return {
        "success": success,
        "reward": 1.0 if success else 0.0,
        "object_prompt": object_prompt,
        "picked_side": picked_side,
        "lifted_and_retained": lifted_and_retained,
        "camera": camera,
        "grasp_mode": grasp_mode,
    }
