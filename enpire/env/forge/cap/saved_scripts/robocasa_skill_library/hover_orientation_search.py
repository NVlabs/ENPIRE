# hover_orientation_search.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403
from enpire.env.forge.cap.agent.skill_registry import skill

@skill
def hover_orientation_search_v1(
    side,
    obj_pos,
    hover_height=0.09,
    hover_tilt_deg=21.0,
    prefer_z180=True,
):
    """Hover above obj_pos trying multiple gripper orientations.

    Builds a candidate set of hover quaternions (z180 rotations + tilt variants)
    and tries each until one succeeds. Returns the succeeded label and quat so the
    caller can use the same orientation for the subsequent descend.

    prefer_z180: if True, tries z180 candidates first (often more reliable in sinks).
    Returns: (success, {"label": str, "quat": list[float], "status": str})
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    def normalize(v):
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v

    def apply_world_z180(q):
        q = np.asarray(q, dtype=float)
        return (R.from_rotvec(np.pi * np.array([0., 0., 1.])) * R.from_quat(q)).as_quat()

    def apply_tilt(q, axis_deg):
        axis, deg = axis_deg
        return (R.from_rotvec(np.deg2rad(deg) * np.asarray(axis, dtype=float)) * R.from_quat(q)).as_quat()

    state = get_robot_state()
    cur_quat = np.asarray(state.arms[side].ee_quat, dtype=float)
    z180_quat = apply_world_z180(cur_quat)

    tilt_axes = [
        ([1., 0., 0.], hover_tilt_deg), ([-1., 0., 0.], hover_tilt_deg),
        ([0., 1., 0.], hover_tilt_deg), ([0., -1., 0.], hover_tilt_deg),
    ]

    def build_family(base_q, prefix):
        variants = [(prefix, base_q)]
        for ax in tilt_axes:
            variants.append((f"{prefix}_tilt", apply_tilt(base_q, ax)))
        return variants

    default_family = build_family(cur_quat, "current")
    z180_family = build_family(z180_quat, "z180")

    candidates = (z180_family + default_family) if prefer_z180 else (default_family + z180_family)

    # Deduplicate by quaternion (sign-normalised)
    seen = set()
    deduped = []
    for label, q in candidates:
        q = np.asarray(q, dtype=float)
        if q[3] < 0.:
            q = -q
        key = tuple(np.round(q, 4))
        if key not in seen:
            seen.add(key)
            deduped.append((label, q))

    target = np.array(obj_pos, dtype=float).copy()
    target[2] += hover_height

    for label, q in deduped:
        r = freespace_move(right_target_pos=target.tolist(),
                           right_target_quat=q.tolist(), side=side)
        print(f"hover_orientation_search: {label} -> {r.status}")
        if r.status == "Success":
            return True, {"label": label, "quat": q.tolist(), "status": r.status,
                          "target": target.tolist()}

    # All failed — return best-effort (first candidate, arm wherever it ended up)
    first_label, first_q = deduped[0]
    return False, {"label": first_label, "quat": first_q.tolist(), "status": "all_failed",
                   "target": target.tolist()}
