# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize cuRobo reachability results in Viser.

Examples:
    # Visualize a specific benchmark run:
    uv run python -m experimental.plot_curobo_benchmark_viser \
      artifacts/curobo_benchmark/20260326_212408_both_both_zero

    # Visualize the latest run under the default root:
    uv run python -m experimental.plot_curobo_benchmark_viser
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import viser
from viser.extras import ViserUrdf


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=Path("artifacts/curobo_benchmark"),
        type=Path,
        help="Benchmark run directory, or a root directory containing multiple run directories.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--point-size", type=float, default=0.025)
    return parser


def _resolve_run_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        if (path.parent / "summary.json").exists():
            return path.parent
        raise FileNotFoundError(
            f"Expected a benchmark run directory or a file inside one, got file: {path}"
        )
    if (path / "summary.json").exists():
        return path
    candidates = sorted(
        [p.parent for p in path.glob("**/summary.json")],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No benchmark run directories found under: {path}")
    return candidates[-1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _color(n: int, rgb: tuple[int, int, int]) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.tile(np.asarray(rgb, dtype=np.uint8).reshape(1, 3), (n, 1))


def _subset_for_z(df: pd.DataFrame, z_label: str) -> pd.DataFrame:
    if z_label == "all":
        return df
    z_val = float(z_label)
    return df[np.isclose(df["z"].to_numpy(dtype=float), z_val)]


def _summary_markdown(summary: dict, df: pd.DataFrame) -> str:
    args = summary.get("args", {})
    success_rate = float(df["success"].mean()) if not df.empty else 0.0
    status_counts = (
        df["status_detail"].fillna(df["status"]).value_counts().sort_index().to_dict()
        if not df.empty
        else {}
    )
    lines = [
        "## Reachability",
        f"- mode: `{args.get('mode', '?')}`",
        f"- side: `{args.get('side', '?')}`",
        f"- start_state: `{args.get('start_state', '?')}`",
        f"- orientation: `{args.get('orientation', '?')}`",
        f"- samples shown: `{len(df)}`",
        f"- success rate: `{100.0 * success_rate:.1f}%`",
        "",
        "### Status counts",
    ]
    if status_counts:
        for k, v in status_counts.items():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def _add_robot(
    server: viser.ViserServer,
    summary: dict,
) -> tuple[ViserUrdf | None, dict[str, float]]:
    urdf_path = summary.get("planner", {}).get("urdf_path")
    if not urdf_path:
        print("[ViserReachability] No URDF path found in summary; skipping robot render.")
        return None, {}
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        print(f"[ViserReachability] URDF path does not exist: {urdf_path}; skipping robot render.")
        return None, {}

    urdf_vis = ViserUrdf(server, urdf_or_path=urdf_path, load_meshes=True)
    joint_limits = dict(urdf_vis.get_actuated_joint_limits())  # type: ignore[arg-type]
    joint_names = list(joint_limits.keys())

    start_left = np.asarray(summary["start_state"]["left_joint_pos"], dtype=np.float64).reshape(-1)
    start_right = np.asarray(summary["start_state"]["right_joint_pos"], dtype=np.float64).reshape(-1)
    name_to_val: dict[str, float] = {}
    for i in range(min(6, len(start_left))):
        name_to_val[f"left_joint{i+1}"] = float(start_left[i])
    for i in range(min(6, len(start_right))):
        name_to_val[f"right_joint{i+1}"] = float(start_right[i])

    cfg = np.zeros(len(joint_names), dtype=np.float64)
    for i, name in enumerate(joint_names):
        cfg[i] = name_to_val.get(name, 0.0)
    urdf_vis.update_cfg(cfg)
    print(f"[ViserReachability] Loaded robot URDF: {urdf_path}")
    return urdf_vis, name_to_val


def _make_point_groups(summary: dict, df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    run_side = summary.get("args", {}).get("side", "left")
    success_df = df[df["success"].astype(bool)]
    failure_df = df[~df["success"].astype(bool)]
    groups: list[tuple[str, np.ndarray, np.ndarray]] = []

    if run_side == "both":
        succ_left = success_df[["x", "y_abs", "z"]].to_numpy(dtype=np.float32)
        succ_right = success_df.assign(y=-success_df["y_abs"])[["x", "y", "z"]].to_numpy(dtype=np.float32)
        fail_left = failure_df[["x", "y_abs", "z"]].to_numpy(dtype=np.float32)
        fail_right = failure_df.assign(y=-failure_df["y_abs"])[["x", "y", "z"]].to_numpy(dtype=np.float32)
        groups += [
            ("success_left", succ_left, _color(len(succ_left), (0, 200, 0))),
            ("success_right", succ_right, _color(len(succ_right), (80, 255, 120))),
            ("failure_left", fail_left, _color(len(fail_left), (220, 40, 40))),
            ("failure_right", fail_right, _color(len(fail_right), (255, 120, 120))),
        ]
    else:
        pts_success = success_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        pts_failure = failure_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        if run_side == "left":
            succ_rgb, fail_rgb = (0, 200, 0), (220, 40, 40)
        else:
            succ_rgb, fail_rgb = (80, 255, 120), (255, 120, 120)
        groups += [
            ("success", pts_success, _color(len(pts_success), succ_rgb)),
            ("failure", pts_failure, _color(len(pts_failure), fail_rgb)),
        ]
    return groups


def main() -> None:
    args = _make_parser().parse_args()
    run_dir = _resolve_run_dir(args.path)
    summary = _load_json(run_dir / "summary.json")
    reachability_csv = run_dir / "reachability_samples.csv"
    if not reachability_csv.exists():
        raise FileNotFoundError(f"Missing reachability_samples.csv in {run_dir}")
    df_all = pd.read_csv(reachability_csv)
    if df_all.empty:
        raise ValueError(f"No reachability samples found in {reachability_csv}")

    z_values = sorted(float(z) for z in df_all["z"].unique())
    z_options = ["all", *[f"{z:.4f}" for z in z_values]]

    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"[ViserReachability] Loaded run: {run_dir}")
    urdf_vis, _robot_joint_map = _add_robot(server, summary)

    with server.gui.add_folder("Reachability"):
        z_dropdown = server.gui.add_dropdown("Z slice", options=z_options, initial_value="all")
        show_success = server.gui.add_checkbox("Show success", True)
        show_failure = server.gui.add_checkbox("Show failure", True)
        show_start = server.gui.add_checkbox("Show start markers", True)
        show_robot = server.gui.add_checkbox("Show robot", urdf_vis is not None)
        point_size = server.gui.add_slider("Point size", 0.002, 0.08, 0.001, args.point_size)
        stats = server.gui.add_markdown("")

    current_handles: list = []

    def _clear_handles() -> None:
        nonlocal current_handles
        for h in current_handles:
            try:
                h.remove()
            except Exception:
                pass
        current_handles = []

    def _render() -> None:
        nonlocal current_handles
        _clear_handles()
        df = _subset_for_z(df_all, str(z_dropdown.value))
        stats.content = _summary_markdown(summary, df)
        if urdf_vis is not None:
            urdf_vis.show_visual = bool(show_robot.value)

        groups = _make_point_groups(summary, df)
        size = float(point_size.value)
        for name, pts, colors in groups:
            is_success = "success" in name
            if is_success and not bool(show_success.value):
                continue
            if (not is_success) and not bool(show_failure.value):
                continue
            if len(pts) == 0:
                continue
            current_handles.append(
                server.scene.add_point_cloud(
                    f"/reachability/{name}",
                    points=pts,
                    colors=colors,
                    point_size=size,
                    point_shape="circle",
                )
            )

        if bool(show_start.value):
            start_left = np.asarray(summary["start_pose"]["start_left_pose"]["position"], dtype=np.float32).reshape(1, 3)
            start_right = np.asarray(summary["start_pose"]["start_right_pose"]["position"], dtype=np.float32).reshape(1, 3)
            current_handles.append(
                server.scene.add_point_cloud(
                    "/reachability/start_left",
                    points=start_left,
                    colors=_color(1, (50, 120, 255)),
                    point_size=max(size * 1.6, 0.03),
                    point_shape="diamond",
                )
            )
            current_handles.append(
                server.scene.add_point_cloud(
                    "/reachability/start_right",
                    points=start_right,
                    colors=_color(1, (255, 170, 0)),
                    point_size=max(size * 1.6, 0.03),
                    point_shape="diamond",
                )
            )

    z_dropdown.on_update(lambda _: _render())
    show_success.on_update(lambda _: _render())
    show_failure.on_update(lambda _: _render())
    show_start.on_update(lambda _: _render())
    show_robot.on_update(lambda _: _render())
    point_size.on_update(lambda _: _render())

    _render()
    print("[ViserReachability] Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[ViserReachability] Stopping.")


if __name__ == "__main__":
    main()
