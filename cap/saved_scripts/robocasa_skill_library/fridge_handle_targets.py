# fridge_handle_targets.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from cap.agent.skill_registry import skill

from cap.saved_scripts.robocasa_skill_library.fridge_control_state import (
    fridge_progress,
    get_control_target,
    handle_progress_gap,
)
from cap.saved_scripts.robocasa_skill_library.fridge_handle_geometry import (
    fridge_handle_geometry_v1,
)
from cap.saved_scripts.robocasa_skill_library.show_debug_balls import (
    show_debug_balls_v1,
)


@skill
def fridge_handle_targets_v1(oracle, set_markers=True):
    controls = oracle.get("controls") or {}
    if not controls:
        ordered_controls = [oracle.get("active_control", "handle")]
    else:
        ordered_controls = [
            name
            for name, _ in sorted(
                controls.items(),
                key=lambda item: handle_progress_gap(item[1]),
                reverse=True,
            )
        ]

    progress_by_control = {
        control_name: fridge_progress(oracle, control_name)
        for control_name in ordered_controls
    }

    if set_markers:
        markers = []
        palette = (
            [64, 255, 255],
            [255, 192, 64],
            [128, 255, 128],
            [255, 128, 192],
        )
        for idx, control_name in enumerate(ordered_controls):
            target = get_control_target(oracle, control_name)
            if target is None:
                continue
            _, plan = fridge_handle_geometry_v1(target)
            markers.append(
                {
                    "name": f"fridge_handle_target_{control_name}",
                    "label": control_name,
                    "position": plan["handle_pos"],
                    "color": palette[idx % len(palette)],
                    "radius_m": 0.03,
                    "alpha": 0.75,
                }
            )
        if markers:
            show_debug_balls_v1(markers)

    return True, {
        "ordered_controls": ordered_controls,
        "progress_by_control": progress_by_control,
    }
