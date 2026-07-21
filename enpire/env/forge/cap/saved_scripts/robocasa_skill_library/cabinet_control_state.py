# cabinet_control_state.py — shared cabinet control lookup helpers
from skill_library.namespace import *  # noqa: F401, F403


def handle_progress_gap(target):
    return abs(
        float(target.get("desired_fraction", 0.0))
        - float(target.get("normalized_qpos", 0.0))
    )


def door_progress(oracle, control_name):
    fixture_state = oracle.get("fixture_state") or {}
    if not fixture_state:
        return None
    candidate_keys = []
    if control_name.endswith("_handle"):
        candidate_keys.append(control_name.replace("_handle", "_door"))
    candidate_keys.append(control_name)
    active_control = str(oracle.get("active_control", ""))
    if active_control.endswith("_handle"):
        candidate_keys.append(active_control.replace("_handle", "_door"))
    if len(fixture_state) == 1:
        candidate_keys.extend(fixture_state.keys())
    for key in candidate_keys:
        if key in fixture_state:
            return float(fixture_state[key])
    return None


def get_control_target(oracle, control_name):
    controls = oracle.get("controls") or {}
    if control_name in controls:
        return controls[control_name]
    if oracle.get("active_control") == control_name:
        return oracle.get("target")
    return None
