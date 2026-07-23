# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# show_debug_balls.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403

from enpire.env.forge.cap.agent.skill_registry import skill


def _debug_marker_tool(name):
    tool = globals().get(name)
    return tool if callable(tool) else None


@skill
def show_debug_ball_v1(
    position,
    *,
    name="debug_ball",
    label=None,
    color=(64, 255, 255),
    radius_m=0.03,
    alpha=0.75,
):
    marker = {
        "name": str(name),
        "label": str(label if label is not None else name),
        "position": list(position),
        "color": list(color),
        "radius_m": float(radius_m),
        "alpha": float(alpha),
    }
    return show_debug_balls_v1([marker])


@skill
def show_debug_balls_v1(markers):
    set_debug_markers_fn = _debug_marker_tool("set_debug_markers")
    if set_debug_markers_fn is None:
        print("debug_markers: set_debug_markers unavailable")
        return False, {"reason": "set_debug_markers unavailable", "count": len(markers)}

    normalized_markers = []
    for marker in markers:
        normalized_markers.append(
            {
                "name": str(marker["name"]),
                "label": str(marker.get("label", marker["name"])),
                "position": list(marker["position"]),
                "color": list(marker.get("color", [64, 255, 255])),
                "radius_m": float(marker.get("radius_m", 0.03)),
                "alpha": float(marker.get("alpha", 0.75)),
            }
        )

    try:
        set_debug_markers_fn(normalized_markers)
    except Exception as exc:
        print(f"debug_markers: failed to update ({exc})")
        return False, {"reason": str(exc), "count": len(normalized_markers)}

    print(f"debug_markers: updated {len(normalized_markers)} marker(s)")
    return True, {"count": len(normalized_markers)}
