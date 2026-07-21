"""Move YAM home for PushT supervisor /home."""


PRE_HOME_LIFT_M = 0.08
PRE_HOME_LIFT_MAX_Z = 0.93


def _state_ee_pos(state, side):
    direct = getattr(state, f"{side}_ee_pos", None)
    if direct is not None:
        return [float(v) for v in direct[:3]]
    if isinstance(state, dict):
        direct = state.get(f"{side}_ee_pos")
        if direct is not None:
            return [float(v) for v in direct[:3]]
    arms = state.get("arms", {}) if isinstance(state, dict) else getattr(state, "arms", {})
    arm = arms.get(side) if isinstance(arms, dict) else getattr(arms, side, None)
    if arm is None:
        return None
    ee_pos = arm.get("ee_pos") if isinstance(arm, dict) else getattr(arm, "ee_pos", None)
    if ee_pos is None:
        return None
    return [float(v) for v in ee_pos[:3]]


def _lift_before_home():
    move = globals().get("freespace_move")
    read_state = globals().get("get_robot_state")
    if not callable(move) or not callable(read_state):
        print("[pusht_go_home] pre_home_lift skipped: missing freespace_move/get_robot_state")
        return {"success": False, "skipped": True, "reason": "missing_skill"}

    state = read_state()
    results = {}
    for side in ("left", "right"):
        pos = _state_ee_pos(state, side)
        if pos is None:
            results[side] = {"success": False, "skipped": True, "reason": "missing_ee_pos"}
            continue
        target = list(pos)
        target[2] = min(float(PRE_HOME_LIFT_MAX_Z), target[2] + float(PRE_HOME_LIFT_M))
        if target[2] <= pos[2] + 0.005:
            results[side] = {"success": True, "skipped": True, "reason": "already_high", "start": pos}
            continue
        print(f"[pusht_go_home] pre_home_lift {side}: {pos} -> {target}")
        kwargs = {
            "side": side,
            f"{side}_target_pos": target,
            "planning_speed": 4.0,
            "backend": "rrt-connect",
            "planner_backend": "rrtconnect",
            "ik_rpy_weight": 0.0,
            "ik_error_threshold": 0.03,
        }
        try:
            result = move(**kwargs)
            ok = getattr(result, "status", None) in {None, "Success", "success", "done"}
            results[side] = {"success": bool(ok), "status": getattr(result, "status", None), "result": str(result)}
        except Exception as exc:
            results[side] = {"success": False, "error": str(exc)}
            print(f"[pusht_go_home] pre_home_lift {side} failed: {exc}")
    return {"success": any(v.get("success") for v in results.values()), "results": results}


lift_result = _lift_before_home()
print(f"[pusht_go_home] pre_home_lift result={lift_result}")

fast_home = globals().get("go_home_fast")
if callable(fast_home):
    print("[pusht_go_home] go_home_fast start")
    result = fast_home()
    print(f"[pusht_go_home] go_home_fast result={result}")
else:
    home = globals().get("go_home")
    if not callable(home):
        raise RuntimeError("Neither go_home_fast nor go_home is available")
    print("[pusht_go_home] go_home fallback start")
    result = home()
    print(f"[pusht_go_home] go_home result={result}")

