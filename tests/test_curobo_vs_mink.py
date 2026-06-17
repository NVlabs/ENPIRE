"""Compare CuRobo (plan_to_pose) vs Mink (seeded IK) + cubic spline."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def make_circle(center, radius, z, n_points=64):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.zeros((n_points, 3))
    pts[:, 0] = center[0] + radius * np.cos(theta)
    pts[:, 1] = center[1] + radius * np.sin(theta)
    pts[:, 2] = z
    return np.vstack([pts, pts[:1]])


def make_figure8(center, radius, z, n_points=128):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.zeros((n_points, 3))
    pts[:, 0] = center[0] + radius * np.sin(theta)
    pts[:, 1] = center[1] + radius * np.sin(2 * theta) / 2.0
    pts[:, 2] = z
    return np.vstack([pts, pts[:1]])


def plot_comparison(desired, ee_curobo, ee_mink, res_curobo, res_mink, title, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  matplotlib not available, skipping {title}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # XY overlay — all three
    axes[0].plot(desired[:, 0], desired[:, 1], "b-", label="Ground Truth", lw=2.5)
    if len(ee_curobo) > 0:
        axes[0].plot(ee_curobo[:, 0], ee_curobo[:, 1], "r--", label="CuRobo IK + Spline", lw=1.8, dashes=(6, 3))
    axes[0].plot(ee_mink[:, 0], ee_mink[:, 1], color="green", marker=".", markersize=2, linestyle="none", label="Mink IK + Spline")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title("XY Overlay")
    axes[0].legend(fontsize=9)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    # XZ overlay — all three
    axes[1].plot(desired[:, 0], desired[:, 2], "b-", label="Ground Truth", lw=2.5)
    if len(ee_curobo) > 0:
        axes[1].plot(ee_curobo[:, 0], ee_curobo[:, 2], "r--", label="CuRobo IK + Spline", lw=1.8, dashes=(6, 3))
    axes[1].plot(ee_mink[:, 0], ee_mink[:, 2], color="green", marker=".", markersize=2, linestyle="none", label="Mink IK + Spline")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")
    axes[1].set_title("XZ Overlay")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Error comparison — both on same axes + mean lines
    errs_c = res_curobo.get("cart_errors", np.array([])) * 1000.0
    errs_m = res_mink.get("cart_errors", np.array([])) * 1000.0
    if len(errs_c) > 0:
        t_c = np.linspace(0, 1, len(errs_c))
        axes[2].plot(t_c, errs_c, "r-", label=f"CuRobo (max={errs_c.max():.1f}mm)", lw=0.8, alpha=0.7)
        axes[2].axhline(errs_c.mean(), color="r", linestyle="--", lw=1.5, alpha=0.9, label=f"CuRobo mean={errs_c.mean():.2f}mm")
    if len(errs_m) > 0:
        t_m = np.linspace(0, 1, len(errs_m))
        axes[2].plot(t_m, errs_m, "g-", label=f"Mink (max={errs_m.max():.1f}mm)", lw=0.8, alpha=0.7)
        axes[2].axhline(errs_m.mean(), color="g", linestyle=":", lw=1.5, alpha=0.9, label=f"Mink mean={errs_m.mean():.2f}mm")
    axes[2].set_xlabel("Normalized Path Progress")
    axes[2].set_ylabel("Error (mm)")
    axes[2].set_title("Tracking Error")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Summary bar
    ct = res_curobo.get("planning_ms", 0)
    mt = res_mink.get("planning_ms", 0)
    speedup = f"{ct/mt:.0f}x faster" if mt > 0 and ct > 0 else ""
    summary = (
        f"CuRobo IK: {ct:.0f}ms | max {res_curobo.get('max_cart_err_m',0)*1000:.1f}mm | mean {res_curobo.get('mean_cart_err_m',0)*1000:.1f}mm    "
        f"Mink IK: {mt:.0f}ms | max {res_mink['max_cart_err_m']*1000:.1f}mm | mean {res_mink['mean_cart_err_m']*1000:.1f}mm    "
        f"{speedup}"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")


def run_curobo(waypoints, rpy, side, subsample):
    """CuRobo plan_to_pose segment-by-segment (old approach)."""
    from experimental.motion_planner_curobo import YamMotionPlannerCurobo
    from experimental.curobo_waypoint_planner import (
        densify_waypoints, interpolate_orientations, display_rpy_to_quat_xyzw,
    )
    from scipy.interpolate import CubicSpline
    from robot.yam.kinematics import YamKinematics
    import time

    planner = YamMotionPlannerCurobo(
        device="cuda:0", solver_speed="fast",
        validate_with_mujoco=False, collision_checking=True,
        enable_finetune_trajopt=False,
    )
    kin = YamKinematics()

    retract_l = np.array([-0.3, 1.35, 1.6, -0.8, 0.3, -0.25])
    retract_r = np.array([0.3, 1.35, 1.6, -0.8, -0.3, 0.25])

    dense = densify_waypoints(waypoints, 0.005)
    n_dense = len(dense)
    start_q = display_rpy_to_quat_xyzw(rpy)
    quats = interpolate_orientations(start_q, start_q, n_dense)

    indices = list(range(0, n_dense, max(1, subsample)))
    if indices[-1] != n_dense - 1:
        indices.append(n_dense - 1)

    t0 = time.perf_counter()
    solutions = []
    cl, cr = retract_l.copy(), retract_r.copy()
    n_failed = 0
    for idx in indices:
        kw = dict(current_left_jp=cl, current_right_jp=cr, side=side, validate_trajectory=False)
        if side == "right":
            kw["target_right_pos"] = dense[idx]
            kw["target_right_quat_xyzw"] = quats[idx]
        else:
            kw["target_left_pos"] = dense[idx]
            kw["target_left_quat_xyzw"] = quats[idx]
        res = planner.plan_to_pose(**kw)
        if res.get("status") == "Success" and res["right_positions"].shape[0] > 0:
            sol = res["right_positions"][-1] if side == "right" else res["left_positions"][-1]
            solutions.append(sol)
            cl, cr = res["left_positions"][-1], res["right_positions"][-1]
        else:
            if solutions:
                solutions.append(solutions[-1])
            n_failed += 1

    if not solutions:
        return {"success": False, "planning_ms": (time.perf_counter()-t0)*1000,
                "ee_positions": np.empty((0,3)), "cart_errors": np.array([]),
                "max_cart_err_m": float("inf"), "mean_cart_err_m": float("inf"),
                "n_traj_points": 0, "desired_positions": dense}

    sols = np.array(solutions)
    arcs = np.zeros(len(sols))
    target_pos = dense[indices[:len(sols)]]
    for i in range(1, len(sols)):
        arcs[i] = arcs[i-1] + np.linalg.norm(target_pos[i] - target_pos[i-1])

    spline = CubicSpline(arcs, sols, bc_type="clamped", axis=0)
    n_sp = max(len(sols) * 10, 50)
    s = np.linspace(0, arcs[-1], n_sp)
    joints = spline(s)
    n_pts = len(joints)

    if side == "right":
        fl = np.tile(retract_l.reshape(1,6), (n_pts,1))
        fr = joints
    else:
        fl = joints
        fr = np.tile(retract_r.reshape(1,6), (n_pts,1))

    ee = np.zeros((n_pts, 3))
    for i in range(n_pts):
        lp, lq, rp, rq = kin.forward_kinematics(fl[i], fr[i])
        ee[i] = lp if side == "left" else rp

    errs = np.array([float(np.min(np.linalg.norm(dense - ee[i], axis=1))) for i in range(n_pts)])
    planning_ms = (time.perf_counter() - t0) * 1000

    return {
        "success": n_failed == 0,
        "planning_ms": planning_ms,
        "ee_positions": ee,
        "desired_positions": dense,
        "cart_errors": errs,
        "max_cart_err_m": float(np.max(errs)),
        "mean_cart_err_m": float(np.mean(errs)),
        "n_traj_points": n_pts,
    }


def run_mink(planner, waypoints, rpy, side, subsample):
    """Mink seeded IK + cubic spline (current approach)."""
    return planner.plan_waypoint_trajectory(
        waypoints_xyz=waypoints,
        start_rpy_deg=rpy,
        end_rpy_deg=rpy,
        side=side,
        arc_spacing=0.005,
        subsample=subsample,
        spline_points_per_segment=10,
    )


def main():
    from experimental.curobo_waypoint_planner import CuroboWaypointPlanner

    center = [0.55, -0.15]
    z = 0.82
    rpy = [0.0, 180.0, 0.0]
    sub = 10

    circle = make_circle(center, 0.10, z)
    fig8 = make_figure8(center, 0.08, z)

    # --- CuRobo runs ---
    print("=== CuRobo (plan_to_pose + spline) ===")
    print("  Circle...")
    rc_circle = run_curobo(circle, rpy, "right", sub)
    print(f"  {rc_circle['planning_ms']:.0f}ms  max={rc_circle['max_cart_err_m']*1000:.1f}mm  mean={rc_circle['mean_cart_err_m']*1000:.1f}mm")
    print("  Figure-8...")
    rc_fig8 = run_curobo(fig8, rpy, "right", sub)
    print(f"  {rc_fig8['planning_ms']:.0f}ms  max={rc_fig8['max_cart_err_m']*1000:.1f}mm  mean={rc_fig8['mean_cart_err_m']*1000:.1f}mm")

    # --- Mink runs ---
    print("\n=== Mink (seeded IK + spline) ===")
    mink_planner = CuroboWaypointPlanner(device="cuda:0")
    print("  Circle...")
    rm_circle = run_mink(mink_planner, circle, rpy, "right", sub)
    print(f"  {rm_circle['planning_ms']:.0f}ms  max={rm_circle['max_cart_err_m']*1000:.1f}mm  mean={rm_circle['mean_cart_err_m']*1000:.1f}mm")
    print("  Figure-8...")
    rm_fig8 = run_mink(mink_planner, fig8, rpy, "right", sub)
    print(f"  {rm_fig8['planning_ms']:.0f}ms  max={rm_fig8['max_cart_err_m']*1000:.1f}mm  mean={rm_fig8['mean_cart_err_m']*1000:.1f}mm")

    # --- Comparison plots ---
    print("\n=== Generating comparison plots ===")
    plot_comparison(
        rc_circle.get("desired_positions", make_circle(center, 0.10, z)),
        rc_circle.get("ee_positions", np.empty((0,3))),
        rm_circle["ee_positions"],
        rc_circle, rm_circle,
        "Circle — CuRobo vs Mink",
        str(_REPO / "tests" / "circle_curobo_vs_mink.png"),
    )
    plot_comparison(
        rc_fig8.get("desired_positions", make_figure8(center, 0.08, z)),
        rc_fig8.get("ee_positions", np.empty((0,3))),
        rm_fig8["ee_positions"],
        rc_fig8, rm_fig8,
        "Figure-8 — CuRobo vs Mink",
        str(_REPO / "tests" / "figure8_curobo_vs_mink.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
