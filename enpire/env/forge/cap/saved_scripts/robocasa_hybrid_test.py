"""Hybrid pick-place: skill-based pre-positioning → GR00T VLA for the rest.

Phase 1 (skills): detect object, move arm above it, open gripper.
Phase 2 (GR00T):  hand off to grootpool/n15 from the current env state.
                  The policy sees the arm already positioned near the object
                  and completes grasp + place + retract.
"""

import numpy as np

SIDE = "right"
HOVER_Z = 0.12   # clearance above object to position before handing to VLA

# ── Phase 1: skill-based pre-positioning ─────────────────────────────────────

info = get_task_info()  # noqa: F821
obj_pos = np.array(info["obj_pos"])
print(f"[hybrid] Object pos: {[round(float(x), 3) for x in obj_pos]}")

print("\n[hybrid] Phase 1: positioning arm above object via skills")

open_gripper(SIDE)  # noqa: F821

hover = obj_pos.copy()
hover[2] += HOVER_Z
result = freespace_move(right_target_pos=hover.tolist(), side=SIDE)  # noqa: F821
print(f"[hybrid] Hover move: {result.status}  err={result.final_pos_error_m:.4f}m")

if result.status != "Success":
    print("[hybrid] Pre-positioning failed — handing off anyway")

# ── Phase 2: GR00T VLA from current state ────────────────────────────────────

print("\n[hybrid] Phase 2: GR00T rollout from current arm position")

result = use_policy_output(  # noqa: F821
    model="grootpool/n15",
    replan_horizon=16,
    max_steps=500,
)

print(f"\n[hybrid] success={result['success']}  steps={result['steps']}")
print(f"[hybrid] task   : {result['task_description']!r}")
