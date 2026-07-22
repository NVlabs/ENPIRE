#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start YAM follower arm(s) in gravity compensation and show live state.

This launches ``robot/yam/arm_server.py`` for the selected follower arm(s),
connects through the portal client, commands gravity-compensation mode, opens
an optional live camera feed, and prints live joint state plus EEF pose.
"""

from __future__ import annotations

import argparse
import os
import queue
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from enpire.env.forge.robot.constants import (
    LEFT_FOLLOWER_CAN_INTERFACE,
    LEFT_FOLLOWER_PORT,
    RIGHT_FOLLOWER_CAN_INTERFACE,
    RIGHT_FOLLOWER_PORT,
)
from enpire.env.forge.robot.yam.arm_client import FollowerRobotClient
from enpire.env.forge.robot.yam.kinematics import YamKinematics
from enpire.env.forge.robot.yam.non_blocking_camera import NonBlockingCamera

REPO_ROOT = Path(__file__).resolve().parents[2]
Side = Literal["left", "right"]


@dataclass(frozen=True)
class SideConfig:
    port: int
    can_interface: str
    expected_can_interface: str


@dataclass
class ArmRuntime:
    side: Side
    port: int
    client: FollowerRobotClient
    server_proc: subprocess.Popen | None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch selected YAM follower arm server(s), put them in gravity "
            "compensation, show an optional camera feed, and print live state."
        )
    )
    side_group = parser.add_mutually_exclusive_group()
    side_group.add_argument(
        "--left",
        dest="side_selection",
        action="store_const",
        const="left",
        default="left",
        help="use the left follower arm (default)",
    )
    side_group.add_argument(
        "--right",
        dest="side_selection",
        action="store_const",
        const="right",
        help="use the right follower arm",
    )
    side_group.add_argument(
        "--both",
        dest="side_selection",
        action="store_const",
        const="both",
        help="use both follower arms",
    )
    parser.add_argument(
        "--camera",
        choices=("left", "right", "top", "both", "none"),
        default="both",
        help="camera feed to display; 'both' shows left+right side-by-side; 'none' disables the window",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="override portal port for single-arm mode only",
    )
    parser.add_argument("--left-port", type=int, default=LEFT_FOLLOWER_PORT)
    parser.add_argument("--right-port", type=int, default=RIGHT_FOLLOWER_PORT)
    parser.add_argument(
        "--expected-left-can-interface",
        default="can_follow_l",
        help="fail fast if robot.constants maps the left follower elsewhere",
    )
    parser.add_argument(
        "--expected-right-can-interface",
        default="can_follow_r",
        help="fail fast if robot.constants maps the right follower elsewhere",
    )
    parser.add_argument(
        "--no-launch-server",
        action="store_true",
        help="connect to already-running follower arm_server processes instead",
    )
    parser.add_argument(
        "--force-launch-server",
        action="store_true",
        help="try launching arm_server even if the portal port already accepts connections",
    )
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--print-hz", type=float, default=0.5)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to run; 0 means until q/ESC/Ctrl-C",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="do not open an OpenCV window; still print live robot state",
    )
    parser.add_argument(
        "--confirm-motion",
        action="store_true",
        help="confirm the workspace is clear and enable gravity compensation",
    )
    return parser


def _selected_sides(args: argparse.Namespace) -> tuple[Side, ...]:
    if args.side_selection == "both":
        return ("left", "right")
    return (args.side_selection,)


def _side_configs(args: argparse.Namespace) -> dict[Side, SideConfig]:
    return {
        "left": SideConfig(
            port=args.left_port,
            can_interface=LEFT_FOLLOWER_CAN_INTERFACE,
            expected_can_interface=args.expected_left_can_interface,
        ),
        "right": SideConfig(
            port=args.right_port,
            can_interface=RIGHT_FOLLOWER_CAN_INTERFACE,
            expected_can_interface=args.expected_right_can_interface,
        ),
    }


def _port_accepts_connection(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _default_port(side: Side) -> int:
    return LEFT_FOLLOWER_PORT if side == "left" else RIGHT_FOLLOWER_PORT


def _validate_side_config(
    *,
    side: Side,
    config: SideConfig,
    no_launch_server: bool,
) -> None:
    if config.can_interface != config.expected_can_interface:
        raise RuntimeError(
            f"{side.title()} follower CAN interface mismatch: "
            f"robot.constants.{side.upper()}_FOLLOWER_CAN_INTERFACE="
            f"{config.can_interface!r}, expected {config.expected_can_interface!r}"
        )
    if not no_launch_server and config.port != _default_port(side):
        raise RuntimeError(
            "robot/yam/arm_server.py serves follower arms on fixed constants; "
            f"{side} follower uses port {_default_port(side)}, got {config.port}. "
            "Use --no-launch-server when connecting to a separately managed "
            "server on a custom port."
        )


def _launch_arm_server(
    *,
    args: argparse.Namespace,
    side: Side,
    config: SideConfig,
) -> subprocess.Popen | None:
    _validate_side_config(
        side=side,
        config=config,
        no_launch_server=args.no_launch_server,
    )

    label = f"{side}-yam"
    if args.no_launch_server:
        print(
            f"[{label}] not launching arm_server; connecting to {args.host}:{config.port}",
            flush=True,
        )
        return None

    if not args.force_launch_server and _port_accepts_connection(args.host, config.port):
        print(
            f"[{label}] portal port {args.host}:{config.port} is already open; "
            "reusing the existing server",
            flush=True,
        )
        return None

    cmd = [
        sys.executable,
        str(REPO_ROOT / "robot" / "yam" / "arm_server.py"),
        "--mode",
        "follower",
        "--side",
        side,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print(
        f"[{label}] launching {side} follower arm_server on "
        f"{config.can_interface} at localhost:{config.port}",
        flush=True,
    )
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        start_new_session=True,
    )


def _connect_follower(
    *,
    side: Side,
    host: str,
    port: int,
    timeout_s: float,
    server_proc: subprocess.Popen | None,
) -> FollowerRobotClient:
    deadline = time.monotonic() + timeout_s
    last_error: BaseException | None = None
    label = f"{side} follower arm"

    while time.monotonic() < deadline:
        if server_proc is not None and server_proc.poll() is not None:
            raise RuntimeError(
                f"{label} arm_server exited before connection; returncode={server_proc.returncode}"
            )
        try:
            client = FollowerRobotClient(host=host, port=port, label=label)
            client.get_joint_pos()
            return client
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)

    raise TimeoutError(
        f"timed out connecting to {label} arm_server at {host}:{port}"
    ) from last_error


def _enter_gravity_compensation(
    *,
    side: Side,
    client: FollowerRobotClient,
) -> np.ndarray:
    from enpire.env.forge.robot.constants import YAM_ARM_KD, YAM_GRIPPER_KD

    kd = np.asarray(YAM_ARM_KD + [YAM_GRIPPER_KD], dtype=np.float32)
    qpos = np.asarray(client.get_joint_pos(), dtype=np.float32).reshape(7)
    client.command_joint_state(
        {
            "pos": qpos.copy(),
            "vel": np.zeros(7, dtype=np.float32),
            "kp": np.zeros(7, dtype=np.float32),
            "kd": kd,
            "gripper_torque_limit_nm": 0.0,
        }
    )
    print(f"[{side}-yam] gravity compensation command sent (kp=0, kd={kd})", flush=True)
    return qpos


def _read_arm_observations(
    runtimes: dict[Side, ArmRuntime],
) -> tuple[dict[Side, np.ndarray], dict[Side, dict[str, np.ndarray]]]:
    qpos_by_side: dict[Side, np.ndarray] = {}
    obs_by_side: dict[Side, dict[str, np.ndarray]] = {}

    for side, runtime in runtimes.items():
        obs = runtime.client.get_observations()
        joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32).reshape(6)
        gripper_pos = np.asarray(obs["gripper_pos"], dtype=np.float32).reshape(1)
        qpos = np.concatenate([joint_pos, gripper_pos]).astype(np.float32)
        qpos_by_side[side] = qpos
        obs_by_side[side] = {
            key: np.asarray(value) for key, value in obs.items() if isinstance(value, np.ndarray)
        }
    return qpos_by_side, obs_by_side


def _eef_poses(
    *,
    kinematics: YamKinematics,
    qpos_by_side: dict[Side, np.ndarray],
) -> dict[Side, tuple[np.ndarray, np.ndarray]]:
    left_joint_pos = np.zeros(6, dtype=np.float32)
    right_joint_pos = np.zeros(6, dtype=np.float32)
    if "left" in qpos_by_side:
        left_joint_pos = np.asarray(qpos_by_side["left"][:6], dtype=np.float32)
    if "right" in qpos_by_side:
        right_joint_pos = np.asarray(qpos_by_side["right"][:6], dtype=np.float32)

    left_pos, left_quat_xyzw, right_pos, right_quat_xyzw = kinematics.forward_kinematics(
        left_joint_pos, right_joint_pos
    )
    poses: dict[Side, tuple[np.ndarray, np.ndarray]] = {}
    if "left" in qpos_by_side:
        poses["left"] = (np.asarray(left_pos), np.asarray(left_quat_xyzw))
    if "right" in qpos_by_side:
        poses["right"] = (np.asarray(right_pos), np.asarray(right_quat_xyzw))
    return poses


def _format_array(values: np.ndarray, precision: int = 4) -> str:
    return np.array2string(
        np.asarray(values),
        precision=precision,
        suppress_small=True,
        floatmode="fixed",
    )


def _print_live_state(
    *,
    qpos_by_side: dict[Side, np.ndarray],
    obs_by_side: dict[Side, dict[str, np.ndarray]],
    poses_by_side: dict[Side, tuple[np.ndarray, np.ndarray]],
) -> None:
    timestamp = time.strftime("%H:%M:%S")
    for side in ("left", "right"):
        if side not in qpos_by_side:
            continue
        qpos = qpos_by_side[side]
        obs = obs_by_side[side]
        eef_pos, eef_quat_xyzw = poses_by_side[side]
        eef_rpy_deg = Rotation.from_quat(eef_quat_xyzw).as_euler("xyz", degrees=True)
        pieces = [
            f"[{timestamp}] {side}_joint_state={_format_array(qpos)}",
            f"{side}_joint_pos={_format_array(qpos[:6])}",
            f"{side}_gripper={qpos[6]:.4f}",
        ]
        if "joint_vel" in obs:
            pieces.append(f"{side}_joint_vel={_format_array(obs['joint_vel'])}")
        if "joint_eff" in obs:
            pieces.append(f"{side}_joint_eff={_format_array(obs['joint_eff'])}")
        if "gravity_comp" in obs:
            pieces.append(f"{side}_gravity_comp={_format_array(obs['gravity_comp'])}")
        pieces.extend(
            [
                f"{side}_eef_xyz={_format_array(eef_pos)}",
                f"{side}_eef_quat_xyzw={_format_array(eef_quat_xyzw)}",
                f"{side}_eef_rpy_deg={_format_array(eef_rpy_deg, precision=2)}",
            ]
        )
        print(" ".join(pieces), flush=True)


def _draw_overlay(
    frame_rgb: np.ndarray,
    *,
    camera_name: str,
    qpos_by_side: dict[Side, np.ndarray],
    poses_by_side: dict[Side, tuple[np.ndarray, np.ndarray]],
    recorded_count: int = 0,
) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = frame_bgr.copy()

    side_label = "+".join(side for side in ("left", "right") if side in qpos_by_side)
    lines = [
        f"camera={camera_name}  {side_label} follower gravcomp  "
        f"r=record({recorded_count})  Enter=save YAML  q/ESC=quit",
    ]
    for side in ("left", "right"):
        if side not in qpos_by_side:
            continue
        qpos = qpos_by_side[side]
        eef_pos, eef_quat_xyzw = poses_by_side[side]
        lines.extend(
            [
                (f"{side} joint={_format_array(qpos[:6], precision=3)}  gripper={qpos[6]:.3f}"),
                f"{side} eef xyz={_format_array(eef_pos, precision=3)}",
                f"{side} eef quat xyzw={_format_array(eef_quat_xyzw, precision=3)}",
            ]
        )

    line_height = 25
    box_height = min(frame_bgr.shape[0], 18 + line_height * len(lines))
    cv2.rectangle(overlay, (0, 0), (frame_bgr.shape[1], box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

    y = 24
    for line in lines:
        cv2.putText(
            frame_bgr,
            line[:150],
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        y += line_height
    return frame_bgr


def _record_current_pose(
    *,
    qpos_by_side: dict[Side, np.ndarray],
    poses_by_side: dict[Side, tuple[np.ndarray, np.ndarray]],
    recorded: list[dict],
) -> None:
    entry: dict = {}
    for side in ("left", "right"):
        if side not in poses_by_side:
            continue
        qpos = qpos_by_side[side]
        eef_pos, eef_quat_xyzw = poses_by_side[side]
        rpy_deg = Rotation.from_quat(eef_quat_xyzw).as_euler("xyz", degrees=True)
        entry[side] = {
            "position": [round(float(v), 6) for v in eef_pos],
            "rpy_deg": [round(float(v), 6) for v in rpy_deg],
            "gripper_pos": [round(float(qpos[6]), 6)],
        }
    recorded.append(entry)
    print(f"[yam] recorded pose #{len(recorded)}: {entry}", flush=True)


def _save_recorded_poses(recorded: list[dict]) -> None:
    if not recorded:
        print("[yam] no poses recorded, nothing to save", flush=True)
        return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"recorded_poses_{timestamp}.yaml")
    data = {f"position_{i + 1}": entry for i, entry in enumerate(recorded)}
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False, allow_unicode=True)
    print(f"[yam] saved {len(recorded)} pose(s) to {out_path.resolve()}", flush=True)


def _start_stdin_key_thread(
    key_queue: "queue.Queue[bytes]",
    stop_event: threading.Event,
) -> threading.Thread:
    import termios
    import tty

    def _reader() -> None:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    ch = sys.stdin.buffer.read(1)
                    key_queue.put(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t


def _stop_server(server_proc: subprocess.Popen | None) -> None:
    if server_proc is None or server_proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        server_proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
        server_proc.wait(timeout=3.0)


def _prepare_runtimes(args: argparse.Namespace) -> dict[Side, ArmRuntime]:
    sides = _selected_sides(args)
    configs = _side_configs(args)
    if args.port is not None:
        if len(sides) != 1:
            raise ValueError("--port can only be used with --left or --right")
        side = sides[0]
        configs[side] = SideConfig(
            port=args.port,
            can_interface=configs[side].can_interface,
            expected_can_interface=configs[side].expected_can_interface,
        )

    runtimes: dict[Side, ArmRuntime] = {}
    started_procs: list[subprocess.Popen | None] = []
    try:
        for side in sides:
            config = configs[side]
            server_proc = _launch_arm_server(args=args, side=side, config=config)
            started_procs.append(server_proc)
            client = _connect_follower(
                side=side,
                host=args.host,
                port=config.port,
                timeout_s=args.connect_timeout,
                server_proc=server_proc,
            )
            _enter_gravity_compensation(side=side, client=client)
            runtimes[side] = ArmRuntime(
                side=side,
                port=config.port,
                client=client,
                server_proc=server_proc,
            )
    except Exception:
        runtime_procs = {id(runtime.server_proc) for runtime in runtimes.values()}
        for runtime in runtimes.values():
            _stop_server(runtime.server_proc)
        for server_proc in started_procs:
            if id(server_proc) not in runtime_procs:
                _stop_server(server_proc)
        raise
    return runtimes


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if not args.confirm_motion:
        raise RuntimeError(
            "Gravity compensation changes live actuator commands. Clear the workspace, "
            "keep an emergency stop reachable, then pass --confirm-motion."
        )
    if args.print_hz <= 0:
        raise ValueError("--print-hz must be positive")
    if args.camera == "none" and not args.no_gui:
        args.no_gui = True

    runtimes: dict[Side, ArmRuntime] = {}
    cameras: dict[str, NonBlockingCamera] = {}
    recorded_poses: list[dict] = []
    key_queue: queue.Queue[bytes] = queue.Queue()
    stop_event = threading.Event()

    try:
        runtimes = _prepare_runtimes(args)
        kinematics = YamKinematics()

        if args.camera != "none":
            cam_names = ["left", "right"] if args.camera == "both" else [args.camera]
            for cam_name in cam_names:
                print(f"[yam] opening {cam_name!r} camera feed", flush=True)
                cameras[cam_name] = NonBlockingCamera(cam_name)
            if not args.no_gui:
                print(
                    "[yam] press r=record EEF  Enter=save YAML  q/ESC=quit",
                    flush=True,
                )

        if args.no_gui:
            print(
                "[yam] press r=record EEF  Enter=save YAML  Ctrl-C=quit",
                flush=True,
            )
            _start_stdin_key_thread(key_queue, stop_event)

        print_period = 1.0 / args.print_hz
        next_print = 0.0
        deadline = time.monotonic() + args.duration if args.duration > 0 else None

        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            for runtime in runtimes.values():
                proc = runtime.server_proc
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(
                        f"{runtime.side} arm_server exited while running; "
                        f"returncode={proc.returncode}"
                    )

            qpos_by_side, obs_by_side = _read_arm_observations(runtimes)
            poses_by_side = _eef_poses(
                kinematics=kinematics,
                qpos_by_side=qpos_by_side,
            )

            now = time.monotonic()
            if now >= next_print:
                _print_live_state(
                    qpos_by_side=qpos_by_side,
                    obs_by_side=obs_by_side,
                    poses_by_side=poses_by_side,
                )
                next_print = now + print_period

            if cameras and not args.no_gui:
                frames = [
                    cv2.cvtColor(cam.get_image(), cv2.COLOR_RGB2BGR) for cam in cameras.values()
                ]
                if len(frames) > 1:
                    h = min(f.shape[0] for f in frames)
                    frames = [cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in frames]
                    display_frame = np.hstack(frames)
                else:
                    display_frame = frames[0]
                cv2.imshow("YAM Gravity Compensation", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == ord("r"):
                    _record_current_pose(
                        qpos_by_side=qpos_by_side,
                        poses_by_side=poses_by_side,
                        recorded=recorded_poses,
                    )
                elif key in (10, 13):  # Enter (LF or CR depending on platform)
                    _save_recorded_poses(recorded_poses)
            else:
                # Drain key queue from stdin thread
                try:
                    while True:
                        ch = key_queue.get_nowait()
                        if ch in (b"\x03", b"\x1b"):  # Ctrl-C or ESC
                            raise KeyboardInterrupt
                        elif ch == b"r":
                            _record_current_pose(
                                qpos_by_side=qpos_by_side,
                                poses_by_side=poses_by_side,
                                recorded=recorded_poses,
                            )
                        elif ch in (b"\r", b"\n"):  # Enter
                            _save_recorded_poses(recorded_poses)
                except queue.Empty:
                    pass
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[yam] interrupted", flush=True)
    finally:
        stop_event.set()
        for cam in cameras.values():
            cam.close()
        cv2.destroyAllWindows()
        for runtime in runtimes.values():
            _stop_server(runtime.server_proc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
