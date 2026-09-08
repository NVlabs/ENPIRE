"""Thin adapter to the registered CaP set_gripper tool; no arm motion logic."""

import math


def set_gripper(side, pos, *, vel_limit=None, torque_limit=None):
    """Command normalized opening (0=closed, 1=open), preserving tool behavior.

    This is not an object-retention verifier. A closed gripper can be empty.
    Omitted velocity/torque limits retain the configured station limits.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if not math.isfinite(pos) or not 0 <= pos <= 1:
        raise ValueError("pos must be finite and in [0, 1]")
    kwargs = {"side": side, "pos": float(pos)}
    for name, value in (("vel_limit", vel_limit), ("torque_limit", torque_limit)):
        if value is not None:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            kwargs[name] = float(value)
    import skill_library.namespace as tools

    return tools.set_gripper(**kwargs)
