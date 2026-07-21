# fridge_handle_geometry.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

from enpire.env.forge.cap.saved_scripts.robocasa_skill_library.cabinet_handle_geometry import (
    _make_press_quat,
    _normalize,
    cabinet_handle_geometry_v1,
)


@skill
def fridge_handle_geometry_v1(target, rectangular_handle_z_drop_m=0.10):
    return cabinet_handle_geometry_v1(
        target,
        rectangular_handle_z_drop_m=rectangular_handle_z_drop_m,
    )
