# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill


@skill
def print_camera_vector_v1(camera_a="top", camera_b="right"):
    import numpy as np

    try:
        extr_a = get_camera_extrinsics(camera_a)
        extr_b = get_camera_extrinsics(camera_b)
    except Exception as exc:
        print(
            f"camera pose debug unavailable: {camera_a}/{camera_b} "
            f"extrinsics lookup failed ({exc})"
        )
        return False, {
            "reason": "camera extrinsics lookup failed",
            "camera_a": camera_a,
            "camera_b": camera_b,
        }

    pos_a = np.asarray(extr_a["position"], dtype=float)
    pos_b = np.asarray(extr_b["position"], dtype=float)
    vec_ab = pos_b - pos_a

    print(f"{camera_a} camera pos:{[round(float(x), 4) for x in pos_a]}")
    print(f"{camera_b} camera pos:{[round(float(x), 4) for x in pos_b]}")
    print(
        f"{camera_a}_to_{camera_b} vector:"
        f"{[round(float(x), 4) for x in vec_ab]}"
    )

    return True, {
        "camera_a": camera_a,
        "camera_b": camera_b,
        "position_a": pos_a.tolist(),
        "position_b": pos_b.tolist(),
        "vector_ab": vec_ab.tolist(),
    }
