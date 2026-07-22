# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import time

from enpire.policy.rl.config import DataCollectionConfig


_TERMINAL_LABEL = {
    "success": ("Success", (0, 255, 0)),
    "fail": ("Fail", (0, 0, 255)),
    "timeout": ("Timeout", (0, 165, 255)),
    "out-of-range": ("Out of Range", (0, 165, 255)),
}


def _display_side(cfg: DataCollectionConfig | None) -> str:
    if cfg is not None and cfg.enabled_sides in ("left", "right"):
        return cfg.enabled_sides
    return "right"


def _camera_frames_from_obs(
    obs: dict,
    cfg: DataCollectionConfig | None,
) -> list[tuple[str, np.ndarray]]:
    names: list[str] = []
    if cfg is not None:
        names.extend(str(name) for name in cfg.enabled_camera_names)
    names.extend(
        key[: -len("_camera_image")]
        for key in sorted(obs)
        if key.endswith("_camera_image")
    )

    frames: list[tuple[str, np.ndarray]] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        frame = obs.get(f"{name}_camera_image")
        if frame is None:
            continue
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            continue
        frames.append((name, arr))
    return frames


def _concat_camera_frames(frames: list[tuple[str, np.ndarray]]) -> np.ndarray:
    if not frames:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    target_h = min(frame.shape[0] for _, frame in frames)
    resized = []
    for name, frame in frames:
        if frame.shape[0] != target_h:
            width = max(1, int(round(frame.shape[1] * target_h / frame.shape[0])))
            frame = cv2.resize(frame, (width, target_h), interpolation=cv2.INTER_AREA)
        frame = frame.copy()
        cv2.putText(
            frame,
            name,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        resized.append(frame)
    return np.concatenate(resized, axis=1)


def draw_action_delta(
    img,
    action_trail: list[tuple[float, tuple[float, float, float]]],
    cfg: DataCollectionConfig | None,
) -> None:
    if not action_trail:
        return
    max_xyz = (
        np.asarray(cfg.delta_ee_translation_xyz_max, dtype=float)
        if cfg is not None
        else np.ones(3, dtype=float)
    )
    max_xyz = np.maximum(max_xyz, 1e-9)
    duration = max(float(cfg.action_trail_duration_s), 0.0) if cfg is not None else 0.5

    center = (img.shape[1] - 78, 92)
    radius = 24
    cv2.circle(img, center, radius, (80, 80, 80), 1, cv2.LINE_AA)
    z_x = center[0] + 42
    z_half_height = 18
    z_width = 8
    z_top = center[1] - z_half_height
    z_bottom = center[1] + z_half_height
    cv2.rectangle(
        img,
        (z_x - z_width // 2, z_top),
        (z_x + z_width // 2, z_bottom),
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        img,
        (z_x - z_width, center[1]),
        (z_x + z_width, center[1]),
        (110, 110, 110),
        1,
        cv2.LINE_AA,
    )

    now = time.perf_counter()
    newest = action_trail[-1][1]
    for t, xyz in action_trail:
        d = np.asarray(xyz, dtype=float)
        age = max(now - t, 0.0)
        fade = 1.0 if duration <= 0 else max(0.15, 1.0 - age / duration)
        color = tuple(int(c * fade) for c in (0, 220, 255))
        screen = np.array(
            [-d[1] / max_xyz[1] * radius, -d[0] / max_xyz[0] * radius],
            dtype=float,
        )
        length = float(np.linalg.norm(screen))
        if length > radius:
            screen *= radius / length
        tip = (int(center[0] + screen[0]), int(center[1] + screen[1]))
        cv2.arrowedLine(img, center, tip, color, 2, cv2.LINE_AA, tipLength=0.35)

        z = float(np.clip(d[2] / max_xyz[2], -1.0, 1.0))
        z_tip = int(round(center[1] - z * z_half_height))
        base_color = (0, 220, 0) if z >= 0 else (0, 80, 255)
        z_color = tuple(int(c * fade) for c in base_color)
        cv2.rectangle(
            img,
            (z_x - z_width // 2 + 1, min(center[1], z_tip)),
            (z_x + z_width // 2 - 1, max(center[1], z_tip)),
            z_color,
            -1,
            cv2.LINE_AA,
        )

    cv2.putText(
        img,
        f"A {newest[0]:+.4f} {newest[1]:+.4f} {newest[2]:+.4f}",
        (center[0] - radius, center[1] + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_oor_selector(img, x_idx, y_idx, z_idx, x_lim, y_lim, z_lim) -> None:
    """Two stacked 2×2 XY grids (z-hi on top, z-lo below) in the top-right corner."""
    CELL, CGAP, ZGAP, MARGIN = 16, 4, 10, 8
    x0 = img.shape[1] - 2 * (CELL + CGAP) - MARGIN
    y0 = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    DIM, SEL = (60, 60, 60), (0, 200, 255)
    for zi in range(2):
        z_val = 1 - zi
        gy = y0 + zi * (2 * (CELL + CGAP) + ZGAP)
        zlabel = f"z+:{z_lim[1]:.2f}" if z_val else f"z-:{z_lim[0]:.2f}"
        cv2.putText(
            img,
            zlabel,
            (x0 - 62, gy + CELL),
            font,
            0.3,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        for row in range(2):
            yi = 1 - row
            for col in range(2):
                xi = col
                cx = x0 + col * (CELL + CGAP)
                cy = gy + row * (CELL + CGAP)
                sel = xi == x_idx and yi == y_idx and z_val == z_idx
                cv2.rectangle(
                    img,
                    (cx, cy),
                    (cx + CELL, cy + CELL),
                    SEL if sel else DIM,
                    -1 if sel else 1,
                )
    bottom = y0 + 2 * (2 * (CELL + CGAP) + ZGAP)
    cv2.putText(
        img,
        f"-x:{x_lim[0]:.3f}",
        (x0, bottom + 2),
        font,
        0.28,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"+x:{x_lim[1]:.3f}",
        (x0 + CELL + CGAP, bottom + 2),
        font,
        0.28,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def render_frame(
    img_queue,
    state: str,
    obs: dict,
    terminal_event: str | None = None,
    last_terminal_event: str | None = None,
    action_clipped: bool = False,
    oor_corner: tuple | None = None,
    cfg: DataCollectionConfig | None = None,
    hover_pose_label: str | None = None,
    mode_label: str | None = None,
    action_trail: list[tuple[float, tuple[float, float, float]]] | None = None,
    demo_counter: tuple[int, int] | None = None,
    demo_rolling: tuple[int, int] | None = None,
) -> None:
    from enpire.env.forge.display_utils import put_latest_image

    concat = _concat_camera_frames(_camera_frames_from_obs(obs, cfg))
    padded = np.zeros((concat.shape[0] + 8, concat.shape[1] + 8, 3), dtype=concat.dtype)
    padded[4:-4, 4:-4] = concat

    display_side = _display_side(cfg)
    force_key = f"{display_side}_eef_force"
    if force_key not in obs and "right_eef_force" in obs:
        force_key = "right_eef_force"
    if force_key in obs:
        f = np.asarray(obs[force_key], dtype=float)
        axes = cfg.visualize_force_axis if cfg is not None else ("z",)
        axis_map = {"x": 0, "y": 1, "z": 2}
        x0 = padded.shape[1] // 2 + 24
        for row, ax in enumerate(axes):
            if ax not in axis_map:
                continue
            idx = axis_map[ax]
            text_y, bar_y = 18 + row * 22, 24 + row * 22
            cv2.putText(
                padded,
                f"{display_side[0].upper()}F{ax.upper()} {f[idx]:+.2f}",
                (x0, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            fval = float(np.clip(f[idx], -10, 10))
            x1 = x0 + int(fval * 12)
            padded[bar_y : bar_y + 8, min(x0, x1) : max(x0, x1) + 1] = (0, 80, 255)

    pos_key = f"{display_side}_ee_pos"
    if pos_key not in obs and "right_ee_pos" in obs:
        pos_key = "right_ee_pos"
    if pos_key in obs:
        p = np.asarray(obs[pos_key], dtype=float)
        axes = cfg.visualize_position_axis if cfg is not None else ("x", "y", "z")
        axis_map = {"x": 0, "y": 1, "z": 2}
        x0 = padded.shape[1] // 2 + 150
        for row, ax in enumerate(axes):
            if ax not in axis_map:
                continue
            idx = axis_map[ax]
            cv2.putText(
                padded,
                f"{display_side[0].upper()}{ax.upper()} {p[idx]:+.4f}",
                (x0, 18 + row * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    if action_trail is not None:
        draw_action_delta(padded, action_trail, cfg)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if hover_pose_label:
        cv2.putText(
            padded,
            hover_pose_label,
            (12, padded.shape[0] - 14),
            font,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if mode_label:
        cv2.putText(
            padded, mode_label, (12, 36), font, 0.8, (0, 200, 255), 2, cv2.LINE_AA
        )
    elif state == "learn" and terminal_event in _TERMINAL_LABEL:
        label, color = _TERMINAL_LABEL[terminal_event]
        cv2.putText(padded, label, (12, 36), font, 1.0, color, 3, cv2.LINE_AA)
    elif state == "hover":
        cv2.putText(
            padded, "Hovering", (12, 36), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA
        )
    elif state == "pulling_up":
        cv2.putText(
            padded, "Pulling Up", (12, 36), font, 0.8, (255, 200, 0), 2, cv2.LINE_AA
        )
    elif state == "going_to_next_pose":
        cv2.putText(
            padded,
            "Going to Next Pose",
            (12, 36),
            font,
            0.8,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )
    elif state == "oor_test":
        cv2.putText(
            padded, "OOR Test", (12, 36), font, 0.8, (0, 200, 255), 2, cv2.LINE_AA
        )
    elif state == "boundary_test":
        cv2.putText(
            padded,
            "Boundary Test",
            (12, 36),
            font,
            0.8,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    elif state == "learn":
        cv2.putText(
            padded,
            "Collecting Data",
            (12, 36),
            font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    elif state == "manual":
        cv2.putText(
            padded, "Manual Control", (12, 36), font, 0.8, (0, 255, 200), 2, cv2.LINE_AA
        )
        enabled = (
            {"left", "right"}
            if (cfg is not None and cfg.enabled_sides == "both")
            else {cfg.enabled_sides if cfg is not None else "right"}
        )
        for row, side in enumerate(sorted(enabled)):
            ee_pos = obs.get(f"{side}_ee_pos")
            if ee_pos is not None:
                p = np.asarray(ee_pos, dtype=float)
                cv2.putText(
                    padded,
                    f"{side[0].upper()} x:{p[0]:.4f} y:{p[1]:.4f} z:{p[2]:.4f}",
                    (12, 68 + row * 22),
                    font,
                    0.5,
                    (0, 255, 200),
                    1,
                    cv2.LINE_AA,
                )
    elif state == "tuning":
        cv2.putText(
            padded, "Tuning", (12, 36), font, 0.8, (0, 255, 200), 2, cv2.LINE_AA
        )
        if cfg is not None:
            cv2.putText(
                padded,
                f"Fz limit {cfg.right_arm_z_force_limit:+.2f}",
                (12, 68),
                font,
                0.5,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )

    if action_clipped:
        cv2.putText(
            padded,
            "COLLISION CLIPPED",
            (12, 68),
            font,
            0.7,
            (0, 80, 255),
            2,
            cv2.LINE_AA,
        )
    if demo_counter is not None:
        succ, total = demo_counter
        text = f"Demo {succ}/{total}"
        (tw, _), _ = cv2.getTextSize(text, font, 0.8, 2)
        cv2.putText(
            padded,
            text,
            (padded.shape[1] - tw - 12, padded.shape[0] - 14),
            font,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    if demo_rolling is not None:
        r_succ, r_window = demo_rolling
        r_text = f"Last {r_succ}/{r_window}"
        (rw, _), _ = cv2.getTextSize(r_text, font, 0.8, 2)
        cv2.putText(
            padded,
            r_text,
            (padded.shape[1] - rw - 12, padded.shape[0] - 44),
            font,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    if state != "init_pose" and last_terminal_event in _TERMINAL_LABEL:
        label, color = _TERMINAL_LABEL[last_terminal_event]
        cv2.putText(
            padded, f"Last: {label}", (12, 96), font, 0.7, color, 2, cv2.LINE_AA
        )
    if oor_corner is not None and cfg is not None:
        draw_oor_selector(
            padded,
            oor_corner[0],
            oor_corner[1],
            oor_corner[2],
            cfg.x_oor_lim,
            cfg.y_oor_lim,
            cfg.z_oor_lim,
        )

    put_latest_image(img_queue, padded)
