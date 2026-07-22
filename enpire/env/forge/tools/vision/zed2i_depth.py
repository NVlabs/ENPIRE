#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal standalone ZED 2i depth reader.

Examples:
  # Neural network depth estimation + realtime rgb streaming
  uv run python3 tools/vision/zed2i_depth.py --show --frames 0
  # Official pointcloud:
  uv run python3 "/usr/local/zed/samples/depth sensing/depth sensing/python/depth_sensing.py"

    python3 tools/vision/zed2i_depth.py
  python3 tools/vision/zed2i_depth.py --show --frames 0
  python3 tools/vision/zed2i_depth.py --show-pc --frames 0
  python3 tools/vision/zed2i_depth.py --show-pc --point-size 3 --frames 0
  python3 tools/vision/zed2i_depth.py --save-npy depth.npy --frames 1
  python3 tools/vision/zed2i_depth.py --save-ply pointcloud.ply --frames 1
  python3 tools/vision/zed2i_depth.py --kill-camera-blockers
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enpire.env.forge.tools._bootstrap import maybe_reexec_with_uv

maybe_reexec_with_uv(__file__, REPO_ROOT, required_modules=["pyzed"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read depth from a Stereolabs ZED 2i")
    parser.add_argument("--frames", type=int, default=200, help="number of frames to grab; 0 means run until q/Esc")
    parser.add_argument(
        "--depth-mode",
        choices=["neural_plus", "neural", "neural_light", "ultra", "quality", "performance"],
        # default="neural",
        default="neural_plus",
        help="depth mode",
    )
    parser.add_argument(
        "--resolution",
        choices=["hd2k", "hd1200", "hd1080", "hd720", "svga", "vga"],
        default="hd720",
        help="camera resolution",
    )
    parser.add_argument("--fps", type=int, default=30, help="camera fps")
    parser.add_argument(
        "--unit",
        choices=["meter", "centimeter", "millimeter"],
        default="meter",
        help="depth unit",
    )
    parser.add_argument("--min-depth", type=float, default=None, help="minimum depth distance in chosen units")
    parser.add_argument("--max-depth", type=float, default=None, help="maximum depth distance in chosen units")
    parser.add_argument("--confidence", type=int, default=50, help="confidence threshold 0-100")
    parser.add_argument("--texture-confidence", type=int, default=100, help="texture confidence threshold 0-100")
    parser.add_argument("--save-npy", type=Path, default=None, help="save the last depth map as .npy")
    parser.add_argument(
        "--save-ply",
        type=Path,
        default=None,
        help="save the last colored point cloud as .ply on exit",
    )
    parser.add_argument("--show", action="store_true", help="show left image and OpenCV depth heatmap")
    parser.add_argument(
        "--show-pc",
        action="store_true",
        help="show the RGB/depth dashboard and print the official ZED point-cloud command to run separately",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="point size for the point-cloud panel in the dashboard (default: 2.0)",
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=0,
        help="if > 0, relay the RGB/depth dashboard to http://127.0.0.1:PORT for viewing in Chrome",
    )
    parser.add_argument(
        "--kill-camera-blockers",
        action="store_true",
        help="if the ZED camera is busy, terminate local processes currently holding the ZED video nodes and retry once",
    )
    return parser.parse_args()


def _run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError:
        return None


def _detect_zed_video_nodes() -> list[str]:
    v4l2_ctl = shutil.which("v4l2-ctl")
    if v4l2_ctl:
        proc = _run_capture([v4l2_ctl, "--list-devices"])
        if proc is not None and proc.stdout:
            nodes: list[str] = []
            capture = False
            for line in proc.stdout.splitlines():
                stripped = line.strip()
                if not stripped:
                    capture = False
                    continue
                if not line[:1].isspace():
                    capture = "zed" in stripped.lower()
                    continue
                if capture and stripped.startswith("/dev/video"):
                    nodes.append(stripped)
            if nodes:
                return sorted(set(nodes))
    return [str(path) for path in sorted(Path("/dev").glob("video*"))]


def _read_proc_cmdline(pid: int) -> str:
    proc_dir = Path("/proc") / str(pid)
    try:
        cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\x00", b" ").decode().strip()
        if cmdline:
            return cmdline
    except Exception:
        pass
    try:
        return (proc_dir / "comm").read_text().strip()
    except Exception:
        return "<unknown>"


def _find_zed_camera_blockers() -> list[tuple[int, str]]:
    nodes = _detect_zed_video_nodes()
    pids: set[int] = set()

    if nodes:
        fuser = shutil.which("fuser")
        if fuser:
            proc = _run_capture([fuser, *nodes])
            if proc is not None:
                tokens = f"{proc.stdout} {proc.stderr}".replace(":", " ").split()
                pids.update(int(tok) for tok in tokens if tok.isdigit())

        if not pids:
            lsof = shutil.which("lsof")
            if lsof:
                proc = _run_capture([lsof, "-t", *nodes])
                if proc is not None:
                    pids.update(int(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdigit())

    this_pid = os.getpid()
    blockers = [(pid, _read_proc_cmdline(pid)) for pid in sorted(pids) if pid != this_pid]
    return blockers


def _terminate_camera_blockers(blockers: list[tuple[int, str]], *, grace_seconds: float = 2.0) -> None:
    remaining: list[int] = []
    for pid, cmd in blockers:
        try:
            print(f"[zed2i_depth] Terminating blocker PID {pid}: {cmd}")
            os.kill(pid, signal.SIGTERM)
            remaining.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            print(f"[zed2i_depth] Cannot terminate PID {pid}: {exc}", file=sys.stderr)

    if not remaining:
        return

    deadline = time.time() + grace_seconds
    while time.time() < deadline and remaining:
        survivors: list[int] = []
        for pid in remaining:
            if (Path("/proc") / str(pid)).exists():
                survivors.append(pid)
        remaining = survivors
        if remaining:
            time.sleep(0.1)

    for pid in remaining:
        try:
            print(f"[zed2i_depth] Force-killing blocker PID {pid}")
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            print(f"[zed2i_depth] Cannot force-kill PID {pid}: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()

    try:
        import numpy as np
        import pyzed.sl as sl
    except ImportError as e:
        print("Missing dependency:", e, file=sys.stderr)
        print(
            "Install / use the ZED Python API first. On this machine, pyzed is not visible to the current python3.",
            file=sys.stderr,
        )
        return 2

    default_view = {
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
    }
    point_size_state = {"size": max(1.0, float(args.point_size))}
    official_view_state = {
        "module": None,
        "viewer": None,
        "enabled": False,
        "initialized": False,
        "window_id": None,
        "render_mode": "direct-official-gl",
        "background_bgr": (233, 230, 223),
    }
    view_state = {
        "yaw_deg": float(default_view["yaw_deg"]),
        "pitch_deg": float(default_view["pitch_deg"]),
        "dragging": False,
        "last_xy": None,
    }
    render_state = {
        "center": None,
        "camera_distance": None,
    }
    panel_layout = {
        "panel_w": None,
        "panel_h": None,
        "pc_x0": None,
        "pc_x1": None,
        "pc_y0": None,
        "pc_y1": None,
    }

    has_display = bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    )
    auto_show_gui = (
        not args.show
        and not args.show_pc
        and args.serve_port <= 0
        and args.save_npy is None
        and args.save_ply is None
        and has_display
    )
    show_gui = args.show or args.show_pc or auto_show_gui
    if auto_show_gui and "--frames" not in sys.argv:
        args.frames = 0
    serve_dashboard = args.serve_port > 0
    show_2d = show_gui or serve_dashboard

    cv2 = None
    if show_2d:
        try:
            import cv2  # type: ignore
        except ImportError:
            print(
                "2D display requested but OpenCV is not installed; continuing without RGB/depth windows.",
                file=sys.stderr,
            )
        else:
            build_info = cv2.getBuildInformation()
            if "GUI:                           NONE" in build_info:
                if show_gui and not serve_dashboard:
                    print(
                        "2D display requested, but this OpenCV build has no GUI support (cv2.imshow is unavailable).\n"
                        "You are likely using opencv-python-headless in the uv environment.\n"
                        "Fix: remove opencv-python-headless and use opencv-python, or use --serve-port 1980.",
                        file=sys.stderr,
                    )
                    return 3
                if show_gui and serve_dashboard:
                    print(
                        "OpenCV GUI support is unavailable, so the desktop window is disabled.\n"
                        "Browser relay will still run on the requested --serve-port.",
                        file=sys.stderr,
                    )
                    show_gui = False

    official_viewer_path = Path("/usr/local/zed/samples/depth sensing/depth sensing/python/ogl_viewer/viewer.py")
    official_pointcloud_script = official_viewer_path.parent.parent / "depth_sensing.py"
    stream_state = {
        "condition": threading.Condition(),
        "jpeg": None,
        "seq": 0,
        "server": None,
        "thread": None,
    }

    def depth_to_heatmap(depth_np, finite_values):
        if cv2 is None:
            return None
        if finite_values.size == 0:
            return np.zeros((depth_np.shape[0], depth_np.shape[1], 3), dtype=np.uint8)

        lo = args.min_depth if args.min_depth is not None else float(np.percentile(finite_values, 5))
        hi = args.max_depth if args.max_depth is not None else float(np.percentile(finite_values, 95))
        if not math.isfinite(lo):
            lo = float(np.min(finite_values))
        if not math.isfinite(hi):
            hi = float(np.max(finite_values))
        if hi <= lo:
            hi = lo + 1e-6

        clipped = np.clip(depth_np, lo, hi)
        normalized = np.nan_to_num((clipped - lo) / (hi - lo) * 255.0, nan=0.0, posinf=255.0, neginf=0.0).astype(
            np.uint8
        )
        normalized[~np.isfinite(depth_np)] = 0
        colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        colored[~np.isfinite(depth_np)] = (0, 0, 0)
        return colored

    def fit_to_panel(image, panel_w: int, panel_h: int, fill_bgr=(12, 12, 12)):
        if cv2 is None:
            return image

        src_h, src_w = image.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return np.full((panel_h, panel_w, 3), fill_bgr, dtype=np.uint8)

        scale = min(float(panel_w) / float(src_w), float(panel_h) / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        if new_w == src_w and new_h == src_h:
            resized = image.copy()
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

        panel = np.full((panel_h, panel_w, 3), fill_bgr, dtype=np.uint8)
        x0 = (panel_w - new_w) // 2
        y0 = (panel_h - new_h) // 2
        panel[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return panel

    def wrap_text_lines(text: str | None, scale: float, thickness: int, max_width: int):
        if cv2 is None or not text:
            return []

        wrapped_lines = []
        for raw_line in text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            words = raw_line.split()
            if not words:
                continue

            current = words[0]
            for word in words[1:]:
                trial = f"{current} {word}"
                trial_w = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
                if trial_w <= max_width:
                    current = trial
                else:
                    wrapped_lines.append(current)
                    current = word
            wrapped_lines.append(current)

        return wrapped_lines

    def build_info_panel(
        panel_w: int,
        panel_h: int,
        title: str,
        stats_line_1: str,
        stats_line_2: str,
        controls_line: str | None = None,
    ):
        if cv2 is None:
            return None

        out = np.full((panel_h, panel_w, 3), (16, 16, 16), dtype=np.uint8)
        h, w = out.shape[:2]
        pad_x = max(16, int(round(w * 0.03)))
        top_pad = max(14, int(round(h * 0.08)))
        bottom_pad = max(14, int(round(h * 0.08)))
        max_text_w = max(32, w - 2 * pad_x)

        title_scale = max(0.95, min(1.35, h / 170.0))
        body_scale = max(0.78, min(1.0, h / 210.0))
        controls_scale = max(0.72, min(0.92, h / 230.0))
        title_thickness = max(2, int(round(title_scale * 1.7)))
        body_thickness = max(1, int(round(body_scale * 1.5)))
        controls_thickness = max(1, int(round(controls_scale * 1.4)))

        styled_lines = []
        for line in wrap_text_lines(title, title_scale, title_thickness, max_text_w):
            styled_lines.append((line, title_scale, title_thickness, (255, 255, 255)))
        for line in wrap_text_lines(stats_line_1, body_scale, body_thickness, max_text_w):
            styled_lines.append((line, body_scale, body_thickness, (230, 230, 230)))
        for line in wrap_text_lines(stats_line_2, body_scale, body_thickness, max_text_w):
            styled_lines.append((line, body_scale, body_thickness, (230, 230, 230)))
        if controls_line:
            for line in wrap_text_lines(controls_line, controls_scale, controls_thickness, max_text_w):
                styled_lines.append((line, controls_scale, controls_thickness, (180, 255, 180)))

        line_gap = max(6, int(round(h * 0.008)))
        total_text_h = 0
        measured_lines = []
        for line, scale, thickness, color in styled_lines:
            (_, text_h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            measured_lines.append((line, scale, thickness, color, text_h, baseline))
            total_text_h += text_h + baseline + line_gap

        text_block_h = top_pad + bottom_pad + max(0, total_text_h - line_gap)
        y0 = max(0, (h - text_block_h) // 2)

        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (55, 55, 55), 1)

        current_y = y0 + top_pad
        for line, scale, thickness, color, text_h, baseline in measured_lines:
            baseline_y = current_y + text_h
            cv2.putText(
                out,
                line,
                (pad_x, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            current_y = baseline_y + baseline + line_gap
        return out

    def ensure_official_viewer(point_cloud_mat):
        if (
            not official_view_state["enabled"]
            or official_view_state["module"] is None
            or official_view_state["initialized"]
        ):
            return

        if point_cloud_mat is None or point_cloud_mat.get_width() <= 0 or point_cloud_mat.get_height() <= 0:
            return

        module = official_view_state["module"]
        viewer = module.GLViewer()
        res = point_cloud_mat.get_resolution()
        viewer.init(len(sys.argv), sys.argv, res)
        official_view_state["window_id"] = module.glutGetWindow()
        try:
            module.glutHideWindow()
        except Exception:
            pass
        official_view_state["viewer"] = viewer
        official_view_state["initialized"] = True

    def reset_official_camera():
        if not official_view_state["enabled"] or not official_view_state["initialized"]:
            view_state["yaw_deg"] = float(default_view["yaw_deg"])
            view_state["pitch_deg"] = float(default_view["pitch_deg"])
            return

        module = official_view_state["module"]
        viewer = official_view_state["viewer"]
        viewer.camera = module.CameraGL()
        viewer.camera.update()
        viewer.mouse_button = [False, False]
        viewer.mouseMotion = [0.0, 0.0]
        viewer.previousMouseMotion = [0.0, 0.0]
        viewer.mouseCurrentPosition = [0.0, 0.0]
        viewer.wheelPosition = 0.0
        view_state["yaw_deg"] = float(default_view["yaw_deg"])
        view_state["pitch_deg"] = float(default_view["pitch_deg"])

    def render_point_cloud_panel_official(point_cloud_mat, panel_w: int, panel_h: int):
        if cv2 is None or not official_view_state["enabled"]:
            return None

        ensure_official_viewer(point_cloud_mat)
        if not official_view_state["initialized"] or official_view_state["viewer"] is None:
            return None

        viewer = official_view_state["viewer"]
        module = official_view_state["module"]

        try:
            if official_view_state["window_id"] is not None:
                module.glutSetWindow(int(official_view_state["window_id"]))
            viewer.updateData(point_cloud_mat)
            module.glViewport(0, 0, panel_w, panel_h)
            viewer.camera.setProjection(float(panel_h) / float(max(panel_w, 1)))
            viewer.camera.update()

            bg = viewer.bckgrnd_clr
            module.glClearColor(float(bg[0]), float(bg[1]), float(bg[2]), 1.0)
            module.glClear(module.GL_COLOR_BUFFER_BIT | module.GL_DEPTH_BUFFER_BIT)

            viewer.mutex.acquire()
            try:
                viewer.update()
                vp_matrix = viewer.camera.getViewProjectionMatrix()

                module.glUseProgram(viewer.shader_image.get_program_id())
                module.glUniformMatrix4fv(
                    viewer.shader_image_MVP,
                    1,
                    module.GL_TRUE,
                    (module.GLfloat * len(vp_matrix))(*vp_matrix),
                )
                module.glPolygonMode(module.GL_FRONT_AND_BACK, module.GL_FILL)
                viewer.zedModel.draw()
                module.glUseProgram(0)

                module.glUseProgram(viewer.shader_pc.get_program_id())
                module.glUniformMatrix4fv(
                    viewer.shader_pc_MVP,
                    1,
                    module.GL_TRUE,
                    (module.GLfloat * len(vp_matrix))(*vp_matrix),
                )
                module.glPointSize(float(max(1.0, point_size_state["size"])))
                viewer.point_cloud.draw()
                module.glUseProgram(0)
            finally:
                viewer.mutex.release()

            rgb = module.glReadPixels(0, 0, panel_w, panel_h, module.GL_RGB, module.GL_UNSIGNED_BYTE)
            if rgb is None:
                return np.full((panel_h, panel_w, 3), official_view_state["background_bgr"], dtype=np.uint8)

            frame = np.frombuffer(rgb, dtype=np.uint8).reshape(panel_h, panel_w, 3)
            frame = np.flipud(frame)
            return frame[:, :, ::-1].copy()
        except Exception as e:
            print(
                f"Official point-cloud renderer failed; falling back to internal renderer: {e}",
                file=sys.stderr,
            )
            official_view_state["enabled"] = False
            return None

    def decode_xyzrgba(point_cloud_np):
        xyz = np.ascontiguousarray(point_cloud_np[..., :3], dtype=np.float32).reshape(-1, 3)
        rgba_packed = np.ascontiguousarray(point_cloud_np[..., 3]).reshape(-1).view(np.uint32)
        # Official ZED sample shader interprets bytes as R, G, B in the low 24 bits.
        r = (rgba_packed & np.uint32(0x000000FF)).astype(np.uint8)
        g = ((rgba_packed & np.uint32(0x0000FF00)) >> 8).astype(np.uint8)
        b = ((rgba_packed & np.uint32(0x00FF0000)) >> 16).astype(np.uint8)
        bgr = np.stack([b, g, r], axis=1)
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-4) & (np.linalg.norm(xyz, axis=1) > 1e-6)
        return xyz[valid], bgr[valid]

    def sample_point_cloud_for_panel(point_cloud_np, panel_w: int, panel_h: int):
        total_points = int(point_cloud_np.shape[0] * point_cloud_np.shape[1])
        max_points = max(60_000, min(300_000, int(panel_w * panel_h * 0.45)))
        if total_points <= max_points:
            return point_cloud_np

        step = max(1, int(math.ceil(math.sqrt(float(total_points) / float(max_points)))))
        return point_cloud_np[::step, ::step]

    def render_point_cloud_panel(point_cloud_mat, panel_w: int, panel_h: int):
        if cv2 is None:
            return None

        official_canvas = render_point_cloud_panel_official(point_cloud_mat, panel_w, panel_h)
        if official_canvas is not None:
            return official_canvas

        point_cloud_np = point_cloud_mat.get_data() if point_cloud_mat is not None else None
        canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        if point_cloud_np is None or point_cloud_np.size == 0:
            return canvas

        sampled_pc = sample_point_cloud_for_panel(point_cloud_np, panel_w, panel_h)
        xyz, bgr = decode_xyzrgba(sampled_pc)
        if xyz.shape[0] == 0:
            return canvas

        stats_pts = xyz
        if stats_pts.shape[0] > 50_000:
            stats_step = max(1, stats_pts.shape[0] // 50_000)
            stats_pts = stats_pts[::stats_step]

        target_center = np.median(stats_pts, axis=0).astype(np.float32)
        if render_state["center"] is None:
            render_state["center"] = target_center
        else:
            render_state["center"] = (
                0.88 * render_state["center"] + 0.12 * target_center
            ).astype(np.float32)
        center = render_state["center"]

        yaw = math.radians(view_state["yaw_deg"])
        pitch = math.radians(view_state["pitch_deg"])
        cyaw, syaw = math.cos(yaw), math.sin(yaw)
        cpitch, spitch = math.cos(pitch), math.sin(pitch)
        r_y = np.array(
            [[cyaw, 0.0, syaw], [0.0, 1.0, 0.0], [-syaw, 0.0, cyaw]],
            dtype=np.float32,
        )
        r_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cpitch, -spitch], [0.0, spitch, cpitch]],
            dtype=np.float32,
        )
        rot = (r_x @ r_y).astype(np.float32)

        centered_stats = (stats_pts - center) @ rot.T
        hfov = math.radians(70.0)
        aspect = max(float(panel_w) / float(max(panel_h, 1)), 1e-6)
        vfov = 2.0 * math.atan(math.tan(hfov * 0.5) / aspect)
        tan_half_h = max(math.tan(hfov * 0.5), 1e-4)
        tan_half_v = max(math.tan(vfov * 0.5), 1e-4)

        x_extent = float(np.percentile(np.abs(centered_stats[:, 0]), 98.5))
        y_extent = float(np.percentile(np.abs(centered_stats[:, 1]), 98.5))
        z_near = float(np.percentile(centered_stats[:, 2], 1.0))
        radius = float(np.percentile(np.linalg.norm(centered_stats, axis=1), 95.0))
        target_camera_distance = max(
            x_extent / tan_half_h - z_near,
            y_extent / tan_half_v - z_near,
            radius * 1.6,
            0.35,
        )
        if render_state["camera_distance"] is None:
            render_state["camera_distance"] = float(target_camera_distance)
        else:
            render_state["camera_distance"] = 0.88 * float(render_state["camera_distance"]) + 0.12 * float(
                target_camera_distance
            )
        camera_distance = float(render_state["camera_distance"])

        view = (xyz - center) @ rot.T
        depth = view[:, 2] + camera_distance
        valid = np.isfinite(depth) & (depth > 1e-3)
        if not np.any(valid):
            return canvas

        view = view[valid]
        bgr = bgr[valid]
        depth = depth[valid]

        focal = 0.5 * float(panel_w) / tan_half_h
        u = np.round(panel_w * 0.5 + focal * view[:, 0] / depth).astype(np.int32)
        v = np.round(panel_h * 0.5 - focal * view[:, 1] / depth).astype(np.int32)

        in_bounds = (u >= 0) & (u < panel_w) & (v >= 0) & (v < panel_h)
        if not np.any(in_bounds):
            return canvas

        u = u[in_bounds]
        v = v[in_bounds]
        depth = depth[in_bounds]
        bgr = bgr[in_bounds]

        # Draw far -> near so nearer points overwrite farther ones.
        order = np.argsort(depth, kind="stable")[::-1]
        u = u[order]
        v = v[order]
        bgr = bgr[order]
        canvas[v, u] = bgr

        point_size = max(1, int(round(point_size_state["size"])))
        if point_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (point_size * 2 - 1, point_size * 2 - 1)
            )
            canvas = cv2.dilate(canvas, kernel, iterations=1)

        return canvas

    def on_dashboard_mouse(event, x, y, flags, param):
        if cv2 is None or not args.show_pc:
            return

        pc_x0 = panel_layout["pc_x0"]
        pc_x1 = panel_layout["pc_x1"]
        pc_y0 = panel_layout["pc_y0"]
        pc_y1 = panel_layout["pc_y1"]
        if pc_x0 is None or pc_x1 is None or pc_y0 is None or pc_y1 is None:
            return

        in_pc_panel = pc_x0 <= x < pc_x1 and pc_y0 <= y < pc_y1

        if event == cv2.EVENT_LBUTTONDOWN and in_pc_panel:
            view_state["dragging"] = True
            view_state["last_xy"] = (x, y)
            if official_view_state["initialized"] and official_view_state["viewer"] is not None:
                local_x = x - pc_x0
                local_y = y - pc_y0
                viewer = official_view_state["viewer"]
                viewer.mouse_button[0] = True
                viewer.mouseCurrentPosition = [float(local_x), float(local_y)]
                viewer.previousMouseMotion = [float(local_x), float(local_y)]
        elif event == cv2.EVENT_MOUSEMOVE and view_state["dragging"] and view_state["last_xy"] is not None:
            last_x, last_y = view_state["last_xy"]
            dx = x - last_x
            dy = y - last_y
            if official_view_state["initialized"] and official_view_state["viewer"] is not None:
                local_x = x - pc_x0
                local_y = y - pc_y0
                viewer = official_view_state["viewer"]
                viewer.mouseMotion = [float(dx), float(dy)]
                viewer.previousMouseMotion = [float(local_x), float(local_y)]
                viewer.mouseCurrentPosition = [float(local_x), float(local_y)]

            view_state["yaw_deg"] += dx * 0.35
            view_state["pitch_deg"] += dy * 0.25
            view_state["pitch_deg"] = max(-89.0, min(89.0, view_state["pitch_deg"]))
            view_state["last_xy"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            view_state["dragging"] = False
            view_state["last_xy"] = None
            if official_view_state["initialized"] and official_view_state["viewer"] is not None:
                official_view_state["viewer"].mouse_button[0] = False

    if serve_dashboard:
        class DashboardHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ZED Dashboard</title>
  <style>
    body {{
      margin: 0;
      background: #111;
      color: #eee;
      font-family: sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      padding: 16px;
    }}
    img {{
      max-width: min(96vw, 1800px);
      max-height: 90vh;
      width: auto;
      height: auto;
      border: 1px solid #333;
      background: #000;
    }}
    code {{
      color: #9fe870;
    }}
  </style>
</head>
<body>
  <div>ZED RGB/Depth dashboard</div>
  <img src="/stream.mjpg" alt="dashboard stream" />
  <div>Official point cloud: <code>python "{official_pointcloud_script}"</code></div>
</body>
</html>"""
                    payload = html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()

                    last_seq = -1
                    try:
                        while True:
                            with stream_state["condition"]:
                                if stream_state["jpeg"] is None or stream_state["seq"] == last_seq:
                                    stream_state["condition"].wait(timeout=5.0)
                                frame = stream_state["jpeg"]
                                seq = stream_state["seq"]

                            if frame is None:
                                continue

                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            last_seq = seq
                    except (BrokenPipeError, ConnectionResetError):
                        return

                self.send_error(404)

            def log_message(self, format, *args):
                return

        try:
            server = ThreadingHTTPServer(("127.0.0.1", int(args.serve_port)), DashboardHandler)
        except OSError as e:
            print(f"Failed to start browser relay on port {args.serve_port}: {e}", file=sys.stderr)
            return 5

        stream_state["server"] = server
        stream_state["thread"] = threading.Thread(target=server.serve_forever, daemon=True)
        stream_state["thread"].start()
        print(f"Browser dashboard: http://127.0.0.1:{args.serve_port}")

    depth_mode_map = {
        "neural_plus": sl.DEPTH_MODE.NEURAL_PLUS,
        "neural": sl.DEPTH_MODE.NEURAL,
        "neural_light": sl.DEPTH_MODE.NEURAL_LIGHT,
        "ultra": sl.DEPTH_MODE.ULTRA,
        "quality": sl.DEPTH_MODE.QUALITY,
        "performance": sl.DEPTH_MODE.PERFORMANCE,
    }
    resolution_map = {
        "hd2k": sl.RESOLUTION.HD2K,
        "hd1200": sl.RESOLUTION.HD1200,
        "hd1080": sl.RESOLUTION.HD1080,
        "hd720": sl.RESOLUTION.HD720,
        "svga": sl.RESOLUTION.SVGA,
        "vga": sl.RESOLUTION.VGA,
    }
    unit_map = {
        "meter": sl.UNIT.METER,
        "centimeter": sl.UNIT.CENTIMETER,
        "millimeter": sl.UNIT.MILLIMETER,
    }
    unit_suffix = {
        "meter": "m",
        "centimeter": "cm",
        "millimeter": "mm",
    }[args.unit]

    init = sl.InitParameters()
    init.camera_resolution = resolution_map[args.resolution]
    init.camera_fps = args.fps
    init.depth_mode = depth_mode_map[args.depth_mode]
    init.coordinate_units = unit_map[args.unit]
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    if args.min_depth is not None:
        init.depth_minimum_distance = args.min_depth
    if args.max_depth is not None:
        init.depth_maximum_distance = args.max_depth

    zed = sl.Camera()
    status = zed.open(init)
    retried_after_kill = False
    if status != sl.ERROR_CODE.SUCCESS:
        if str(status) == "CAMERA STREAM FAILED TO START":
            blockers = _find_zed_camera_blockers()
            if blockers:
                print("[zed2i_depth] Detected processes using the ZED camera nodes:", file=sys.stderr)
                for pid, cmd in blockers:
                    print(f"  PID {pid}: {cmd}", file=sys.stderr)
                if args.kill_camera_blockers:
                    _terminate_camera_blockers(blockers)
                    time.sleep(1.0)
                    try:
                        zed.close()
                    except Exception:
                        pass
                    zed = sl.Camera()
                    status = zed.open(init)
                    retried_after_kill = True
            if status != sl.ERROR_CODE.SUCCESS and not args.kill_camera_blockers:
                print(
                    "Hint: another app may already be using the ZED (for example ZED_Explorer).\n"
                    "Retry with --kill-camera-blockers to terminate the local blocker(s) automatically.",
                    file=sys.stderr,
                )
        print(f"Failed to open camera: {status}", file=sys.stderr)
        return 1
    if retried_after_kill:
        print("[zed2i_depth] Camera opened successfully after terminating blocker processes.")

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = args.confidence
    runtime.texture_confidence_threshold = args.texture_confidence

    left = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    last_point_cloud_cpu = sl.Mat()
    full_res = sl.Resolution()
    full_res.width = -1
    full_res.height = -1

    last_depth_np = None
    windows_initialized = False

    print(
        f"Streaming ZED depth: mode={args.depth_mode} resolution={args.resolution} fps={args.fps} unit={args.unit}"
    )
    if auto_show_gui:
        print("No output mode specified; opening the live dashboard window by default.")
    if args.show_pc or serve_dashboard:
        print(
            "UI: RGB | Depth dashboard.\n"
            "The official point-cloud viewer cannot be auto-launched alongside this dashboard because the ZED camera "
            "is opened exclusively by one process at a time.\n"
            f"Run this separately when you want the official point cloud:\n  python \"{official_pointcloud_script}\""
        )

    try:
        i = 0
        while args.frames <= 0 or i < args.frames:
            grab_status = zed.grab(runtime)
            if grab_status != sl.ERROR_CODE.SUCCESS:
                print(f"grab failed on frame {i}: {grab_status}", file=sys.stderr)
                time.sleep(0.01)
                continue

            zed.retrieve_image(left, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            if args.show_pc or args.save_ply is not None:
                zed.retrieve_measure(
                    last_point_cloud_cpu, sl.MEASURE.XYZRGBA, sl.MEM.CPU, full_res
                )

            width = depth.get_width()
            height = depth.get_height()
            cx, cy = width // 2, height // 2

            err, center_depth = depth.get_value(cx, cy)
            if err != sl.ERROR_CODE.SUCCESS:
                center_depth = float("nan")

            depth_np = depth.get_data()
            last_depth_np = depth_np.copy()
            finite = depth_np[np.isfinite(depth_np)]
            valid_ratio = float(finite.size) / float(depth_np.size) if depth_np.size else 0.0

            if finite.size:
                min_depth = float(np.min(finite))
                median_depth = float(np.median(finite))
                max_depth = float(np.max(finite))
            else:
                min_depth = median_depth = max_depth = float("nan")

            center_str = f"{center_depth:.3f} {unit_suffix}" if math.isfinite(center_depth) else "nan"
            print(
                f"frame={i:04d} center={center_str} valid={valid_ratio:.1%} "
                f"min={min_depth:.3f} median={median_depth:.3f} max={max_depth:.3f} {unit_suffix}"
            )

            if cv2 is not None:
                num_panels = 2
                panel_w = width
                image_panel_h = max(180, int(round(height * 0.86)))
                info_panel_h = max(140, int(round(height * 0.26)))
                dashboard_w = panel_w * num_panels
                dashboard_h = image_panel_h + info_panel_h

                panel_layout["panel_w"] = panel_w
                panel_layout["panel_h"] = image_panel_h
                panel_layout["pc_x0"] = None
                panel_layout["pc_x1"] = None
                panel_layout["pc_y0"] = None
                panel_layout["pc_y1"] = None

                if not windows_initialized:
                    if show_gui:
                        window_flags = cv2.WINDOW_NORMAL
                        if hasattr(cv2, "WINDOW_KEEPRATIO"):
                            window_flags |= cv2.WINDOW_KEEPRATIO
                        cv2.namedWindow("ZED Dashboard", window_flags)
                        cv2.resizeWindow("ZED Dashboard", dashboard_w, dashboard_h)
                        cv2.moveWindow("ZED Dashboard", 20, 40)
                    windows_initialized = True

                left_bgra = left.get_data()
                left_bgr = cv2.cvtColor(left_bgra, cv2.COLOR_BGRA2BGR)
                depth_bgr = depth_to_heatmap(depth_np, finite)
                cv2.drawMarker(
                    left_bgr,
                    (cx, cy),
                    (0, 255, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                )
                cv2.drawMarker(
                    depth_bgr,
                    (cx, cy),
                    (255, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                )
                left_bgr = fit_to_panel(left_bgr, panel_w, image_panel_h)
                depth_bgr = fit_to_panel(depth_bgr, panel_w, image_panel_h)

                line1 = f"frame {i:04d}   center {center_str}   valid {valid_ratio:.1%}"
                line2 = (
                    f"min {min_depth:.3f}   median {median_depth:.3f}   "
                    f"max {max_depth:.3f} {unit_suffix}"
                )
                left_info = build_info_panel(
                    panel_w,
                    info_panel_h,
                    "RGB",
                    line1,
                    line2,
                )
                depth_info = build_info_panel(
                    panel_w,
                    info_panel_h,
                    "DEPTH",
                    line1,
                    line2,
                )
                top_panels = [left_bgr, depth_bgr]
                bottom_panels = [left_info, depth_info]
                try:
                    top_row = cv2.hconcat(top_panels)
                    bottom_row = cv2.hconcat(bottom_panels)
                    dashboard = cv2.vconcat([top_row, bottom_row])
                    if serve_dashboard:
                        ok, enc = cv2.imencode(
                            ".jpg",
                            dashboard,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                        )
                        if ok:
                            with stream_state["condition"]:
                                stream_state["jpeg"] = enc.tobytes()
                                stream_state["seq"] += 1
                                stream_state["condition"].notify_all()
                    if show_gui:
                        cv2.imshow("ZED Dashboard", dashboard)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break
                        if key == ord("s") and (
                            args.show_pc or args.save_ply is not None
                        ):
                            err = last_point_cloud_cpu.write("Pointcloud.ply")
                            if err == sl.ERROR_CODE.SUCCESS:
                                print("Saved current point cloud to Pointcloud.ply")
                            else:
                                print(f"Pointcloud.ply save failed: {err}", file=sys.stderr)
                except cv2.error as e:
                    print(f"OpenCV display failed: {e}", file=sys.stderr)
                    print(
                        "Your OpenCV build likely lacks GUI support. Install a non-headless OpenCV build.",
                        file=sys.stderr,
                    )
                    return 4
            i += 1
    finally:
        zed.close()
        if stream_state["server"] is not None:
            try:
                stream_state["server"].shutdown()
                stream_state["server"].server_close()
            except Exception:
                pass
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if args.save_npy is not None and last_depth_np is not None:
        args.save_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_npy, last_depth_np)
        print(f"Saved depth map to {args.save_npy}")

    if args.save_ply is not None:
        args.save_ply.parent.mkdir(parents=True, exist_ok=True)
        if last_point_cloud_cpu.get_width() <= 0 or last_point_cloud_cpu.get_height() <= 0:
            print(
                "PLY export failed: no point cloud frame was captured before exit",
                file=sys.stderr,
            )
            return 7
        err = last_point_cloud_cpu.write(str(args.save_ply))
        if err == sl.ERROR_CODE.SUCCESS:
            print(f"Saved colored point cloud to {args.save_ply}")
        else:
            print(f"PLY export failed: {err}", file=sys.stderr)
            return 7

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
