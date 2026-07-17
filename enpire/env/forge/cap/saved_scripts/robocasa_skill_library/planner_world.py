# planner_world.py — shared planner world refresh helpers
from skill_library.namespace import *  # noqa: F401, F403


def refresh_planner_world(reason, exclude_body_prefixes=None):
    print(f"[curobo] Refreshing collision world ({reason})")
    kwargs = {}
    if exclude_body_prefixes is not None:
        kwargs["exclude_body_prefixes"] = exclude_body_prefixes
    try:
        world_info = update_planner_world(**kwargs)
        print(f"[curobo] Loaded {world_info['n_obstacles']} obstacles")
    except Exception as exc:
        print(f"[curobo] Planner world refresh unavailable ({exc})")
