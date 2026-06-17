"""Offline tests for CuRobo waypoint planner.

Test 1: Circle (z=0.8, center=[0.6, 0.2], r=0.10m)
Test 2: Figure-8 (z=0.8, center=[0.6, 0.2], r=0.08m)

Both use right arm with top-down grasp [0, 180, 0].
Plots planned EE overlay vs desired shape + prints metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experimental.curobo_waypoint_planner import CuroboWaypointPlanner


def make_circle(center, radius, z, n_points=64):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.zeros((n_points, 3), dtype=np.float64)
    pts[:, 0] = center[0] + radius * np.cos(theta)
    pts[:, 1] = center[1] + radius * np.sin(theta)
    pts[:, 2] = z
    return np.vstack([pts, pts[:1]])


def make_figure8(center, radius, z, n_points=128):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.zeros((n_points, 3), dtype=np.float64)
    pts[:, 0] = center[0] + radius * np.sin(theta)
    pts[:, 1] = center[1] + radius * np.sin(2 * theta) / 2.0
    pts[:, 2] = z
    return np.vstack([pts, pts[:1]])


def plot_overlay(desired, actual, title, save_path, result=None):
    if len(actual) == 0:
        print(f"  No planned points, skipping plot for {title}")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  matplotlib not available, skipping plot for {title}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    axes[0].plot(desired[:, 0], desired[:, 1], "b-", label="Desired", linewidth=2)
    axes[0].plot(actual[:, 0], actual[:, 1], "r--", label="Planned EE", linewidth=1.5)
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title("XY Overlay")
    axes[0].legend()
    axes[0].set_aspect("equal")
    axes[0].grid(True)

    axes[1].plot(desired[:, 0], desired[:, 2], "b-", label="Desired", linewidth=2)
    axes[1].plot(actual[:, 0], actual[:, 2], "r--", label="Planned EE", linewidth=1.5)
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")
    axes[1].set_title("XZ Overlay")
    axes[1].legend()
    axes[1].grid(True)

    cart_errors = result.get("cart_errors", np.array([])) * 1000.0
    if len(cart_errors) > 0:
        axes[2].plot(cart_errors, "g-")
        axes[2].set_xlabel("On-Path Point")
        axes[2].set_ylabel("Error (mm)")
        axes[2].set_title(f"Path Error (max={cart_errors.max():.1f}mm, mean={cart_errors.mean():.1f}mm)")
    axes[2].grid(True)
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved plot to {save_path}")


def print_metrics(result, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Success:          {result['success']}")
    print(f"  Dense waypoints:  {result['n_waypoints']}")
    print(f"  IK targets:       {result.get('n_ik_targets', '?')}")
    print(f"  Failed IK:        {result.get('n_failed_ik', '?')}")
    print(f"  Traj points:      {len(result['joints'])}")
    print(f"  Max cart error:   {result['max_cart_err_m']*1000:.2f} mm")
    print(f"  Mean cart error:  {result['mean_cart_err_m']*1000:.2f} mm")
    print(f"  Max joint jump:   {result['max_joint_jump_rad']:.4f} rad")
    print(f"  Max acceleration: {result.get('max_acceleration', 0):.2f} rad/s²")
    print(f"  Max jerk:         {result.get('max_jerk', 0):.2f} rad/s³")
    print(f"  Jump indices:     {result['jump_indices']}")
    print(f"  Planning time:    {result['planning_ms']:.1f} ms")
    duration = float(result['timestamps'][-1]) if len(result['timestamps']) > 0 else 0
    print(f"  Trajectory dur:   {duration:.2f} s")
    print(f"{'='*60}\n")


def main():
    print("Initializing CuRobo planner (one-time warmup)...")
    planner = CuroboWaypointPlanner(device="cuda:0", solver_speed="fast")

    top_down_rpy = [0.0, 180.0, 0.0]
    center = [0.55, -0.15]
    z = 0.82

    # --- Test 1: Circle ---
    print("\n--- Test 1: Circle ---")
    circle_pts = make_circle(center, 0.10, z, n_points=64)
    result_circle = planner.plan_waypoint_trajectory(
        waypoints_xyz=circle_pts,
        start_rpy_deg=top_down_rpy,
        end_rpy_deg=top_down_rpy,
        side="right",
        arc_spacing=0.005,
        subsample=10,
    )
    print_metrics(result_circle, "Circle (r=10cm)")
    plot_overlay(
        result_circle["desired_positions"],
        result_circle["ee_positions"],
        "Circle: Desired vs Planned EE (mink)",
        str(_REPO / "tests" / "circle_overlay_mink.png"),
        result_circle,
    )

    # --- Re-run to test speed after warmup ---
    print("--- Circle (re-run, warmed up) ---")
    result_circle2 = planner.plan_waypoint_trajectory(
        waypoints_xyz=circle_pts,
        start_rpy_deg=top_down_rpy,
        end_rpy_deg=top_down_rpy,
        side="right",
        arc_spacing=0.005,
        subsample=10,
    )
    print_metrics(result_circle2, "Circle (warmed up)")

    # --- Test 2: Figure-8 ---
    print("\n--- Test 2: Figure-8 ---")
    fig8_pts = make_figure8(center, 0.08, z, n_points=128)
    result_fig8 = planner.plan_waypoint_trajectory(
        waypoints_xyz=fig8_pts,
        start_rpy_deg=top_down_rpy,
        end_rpy_deg=top_down_rpy,
        side="right",
        arc_spacing=0.005,
        subsample=10,
    )
    print_metrics(result_fig8, "Figure-8 (r=8cm)")
    plot_overlay(
        result_fig8["desired_positions"],
        result_fig8["ee_positions"],
        "Figure-8: Desired vs Planned EE (mink)",
        str(_REPO / "tests" / "figure8_overlay_mink.png"),
        result_fig8,
    )

    # --- Sanity check: FK of a single top-down point ---
    print("\n--- Sanity check: single top-down FK ---")
    from experimental.curobo_waypoint_planner import display_rpy_to_quat_xyzw, quat_xyzw_to_display_rpy
    q_xyzw = display_rpy_to_quat_xyzw(top_down_rpy)
    rpy_back = quat_xyzw_to_display_rpy(q_xyzw)
    print(f"  Top-down RPY: {top_down_rpy}")
    print(f"  -> quat xyzw: [{q_xyzw[0]:.4f}, {q_xyzw[1]:.4f}, {q_xyzw[2]:.4f}, {q_xyzw[3]:.4f}]")
    print(f"  -> RPY back:  [{rpy_back[0]:.1f}, {rpy_back[1]:.1f}, {rpy_back[2]:.1f}]")

    if result_circle["n_waypoints"] > 0:
        first_ee = result_circle["ee_positions"][0]
        first_desired = result_circle["desired_positions"][0]
        print(f"  First desired pos: [{first_desired[0]:.4f}, {first_desired[1]:.4f}, {first_desired[2]:.4f}]")
        print(f"  First planned EE:  [{first_ee[0]:.4f}, {first_ee[1]:.4f}, {first_ee[2]:.4f}]")
        err_mm = np.linalg.norm(first_ee - first_desired) * 1000
        print(f"  Error: {err_mm:.2f} mm")

    all_pass = result_circle["success"] and result_fig8["success"]
    print(f"\n{'='*60}")
    print(f"  ALL TESTS {'PASSED' if all_pass else 'FAILED'}")
    print(f"{'='*60}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
