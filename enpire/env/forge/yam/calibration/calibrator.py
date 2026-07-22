# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-camera calibration pipeline.

Steps:
  1. Arm enters gravity mode → camera live-preview with detection → SPACE to confirm nominal
  2. FK model loads
  3. Arm visits calibration poses (offsets from nominal) and captures samples
  4. Hand-eye calibration → Fello station XML with D405 camera injected
  5. Optional MuJoCo viewer to inspect result

H key or Ctrl+C at any time: arm interpolates back to home (q=0) and exits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from . import config
from .arm_client import ArmClient
from .camera import RealSenseCamera, resolve_serial
from .charuco import CharucoDetector
from .hand_eye import Sample
from .hand_eye import run as run_hand_eye
from .xml_writer import update_body_extrinsic

FONT = cv2.FONT_HERSHEY_SIMPLEX
_DT = 1.0 / config.INTERP_HZ

# Per-joint convergence thresholds scaled by KP: looser for compliant joints.
_KP_ARR = np.array(config.ARM_KP, dtype=float)
_KP_NORM = _KP_ARR / _KP_ARR.max()  # [1, 1, 1, 0.5, 0.125, 0.125]
_CONV_TOL = 0.02  # effective tol for highest-KP joints


def _resolution(value: str) -> tuple[int, int]:
    """Parse a 'WxH' (or 'W,H') resolution string into an (int, int) tuple."""
    raw = value.strip().lower().replace(" ", "")
    for sep in ("x", ","):
        if sep in raw:
            w_str, h_str = raw.split(sep, 1)
            w, h = int(w_str), int(h_str)
            if w > 0 and h > 0:
                return (w, h)
            break
    raise argparse.ArgumentTypeError(f"Invalid resolution {value!r}; expected 'WxH' like '640x480'")


def _parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--camera",
        choices=["top", "left_wrist", "right_wrist"],
        default="top",
        help="Which camera to calibrate",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip env-var and viewer prompts (used by launch.py sequence)",
    )
    p.add_argument("--squares-x", type=int, default=config.SQUARES_X)
    p.add_argument("--squares-y", type=int, default=config.SQUARES_Y)
    p.add_argument("--square-length", type=float, default=config.SQUARE_LENGTH)
    p.add_argument("--marker-length", type=float, default=config.MARKER_LENGTH)
    p.add_argument(
        "--resolution",
        type=_resolution,
        default=config.CALIBRATION_RESOLUTION,
        metavar="WxH",
        help=f"Capture resolution (default {config.CALIBRATION_RESOLUTION[0]}x"
        f"{config.CALIBRATION_RESOLUTION[1]})",
    )
    p.add_argument(
        "--confirm-motion",
        action="store_true",
        help="confirm the board is mounted, workspace is clear, and arm motion is safe",
    )
    return p.parse_args(argv)


def _cam_config(camera: str):
    """Return a namespace with per-camera calibration parameters."""
    import types

    c = types.SimpleNamespace()
    if camera == "top":
        c.serial = config.CAMERA_SERIAL
        c.arm_port = config.ARM_SERVER_PORT
        c.fk_body = config.FK_BODY
        c.fk_joints = config.FK_JOINT_NAMES
        c.body_name = config.CALIB_BODY_NAME
        c.camera_name = "top_camera"
        c.eye_in_hand = False
    elif camera == "left_wrist":
        c.serial = config.WRIST_LEFT_CAMERA_SERIAL
        c.arm_port = config.ARM_SERVER_PORT
        c.fk_body = config.FK_BODY
        c.fk_joints = config.FK_JOINT_NAMES
        c.body_name = config.WRIST_LEFT_BODY_NAME
        c.camera_name = "left_wrist"
        c.eye_in_hand = True
    else:  # right_wrist
        c.serial = config.WRIST_RIGHT_CAMERA_SERIAL
        c.arm_port = config.ARM_SERVER_PORT_RIGHT
        c.fk_body = config.FK_BODY_RIGHT
        c.fk_joints = config.FK_JOINT_NAMES_RIGHT
        c.body_name = config.WRIST_RIGHT_BODY_NAME
        c.camera_name = "right_wrist"
        c.eye_in_hand = True
    return c


def _fk(model, data, body_id: int, joint_ids: list[int], q: np.ndarray) -> np.ndarray:
    for joint_id, value in zip(joint_ids, q, strict=True):
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T[:3, 3] = data.xpos[body_id]
    return T


def _gravity_loop(arm: ArmClient, stop: threading.Event) -> None:
    """Background thread: continuously update target = current pos with damping."""
    while not stop.is_set():
        try:
            arm.gravity_mode()
        except Exception:
            pass
        time.sleep(_DT)


def _do_home(arm: ArmClient, q_home: np.ndarray | None) -> None:
    if q_home is None:
        print("\n  [HOME] No start pose recorded — leaving arm in gravity mode.")
        return
    print(f"\n  [HOME] Returning to start pose: {np.round(q_home, 3)}")
    arm.home(q_home)
    print("  [HOME] Done.")


def _go_to_zero(arm: ArmClient, shutdown: bool = True) -> None:
    print("  Homing to joint zeros...")
    arm.home(np.zeros(6))
    print("  At zero.")
    if shutdown:
        arm.shutdown_server()
        print("  Arm server stopped.")


def _converged(diff: np.ndarray) -> bool:
    """True when weighted residual is within tolerance for all joints."""
    return bool(np.max(np.abs(diff) * _KP_NORM) < _CONV_TOL)


def _draw_nominal_overlay(ann, n_corners, board_ok, image_size):
    color = (0, 255, 0) if board_ok else (0, 0, 255)
    cv2.putText(ann, "NOMINAL SETUP", (10, 30), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ann, f"corners: {n_corners}", (10, 60), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        ann, "BOARD OK" if board_ok else "NO BOARD", (10, 90), FONT, 0.7, color, 2, cv2.LINE_AA
    )
    cv2.putText(
        ann,
        "SPACE: confirm pose  H: home",
        (10, ann.shape[0] - 15),
        FONT,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return ann


def _draw_ready_overlay(ann, n_poses):
    cv2.putText(ann, "READY", (10, 30), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        ann, f"{n_poses} poses queued", (10, 60), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        ann,
        "SPACE: start  H: home",
        (10, ann.shape[0] - 15),
        FONT,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return ann


def _draw_overlay(ann, pose_idx, total, n_corners, n_samples, board_ok, image_size):
    w, h = image_size
    color = (0, 255, 0) if board_ok else (0, 0, 255)
    cv2.putText(ann, f"Pose {pose_idx}/{total}", (10, 30), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ann, f"corners: {n_corners}", (10, 60), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(ann, f"samples: {n_samples}", (10, 85), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(ann, f"{w}x{h}", (10, 110), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        ann, "BOARD OK" if board_ok else "NO BOARD", (10, 140), FONT, 0.7, color, 2, cv2.LINE_AA
    )
    cv2.putText(
        ann,
        "H:home  Ctrl+C:abort",
        (10, ann.shape[0] - 15),
        FONT,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return ann


# The native env var plus the name expected by downstream repos that share
# this calibration result.
_ENV_VARS = ("YAM_STATION_CALIBRATED_XML", "YAM_STATION_CALIBRATED_XML_PATH")

# Per-camera calibration.json env vars, keyed by the calibrator camera_name
# (see _cam_config). Each points at the calibration.json holding that camera's
# intrinsics + distortion + extrinsics.
_CALIBRATION_JSON_ENV_VARS = {
    "top_camera": "YAM_TOP_CAMERA_CALIBRATION_JSON",
    "left_wrist": "YAM_LEFT_WRIST_CALIBRATION_JSON",
    "right_wrist": "YAM_RIGHT_WRIST_CALIBRATION_JSON",
}


def _shell_rc_path() -> Path:
    shell = os.environ.get("SHELL", "")
    return Path.home() / (".zshrc" if "zsh" in shell else ".bashrc")


def _write_env_exports(updates: dict[str, str]) -> None:
    """Add or override `export NAME="value"` lines in the user's shell rc file.

    Existing exports for the same NAME are replaced in place; new ones are
    appended. Also updates os.environ for the current process.
    """
    if not updates:
        return
    rc = _shell_rc_path()
    lines = rc.read_text().splitlines(keepends=True) if rc.exists() else []
    for name, value in updates.items():
        export_line = f'export {name}="{value}"\n'
        pattern = re.compile(rf"^\s*export\s+{re.escape(name)}\s*=")
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = export_line
                break
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(export_line)
        os.environ[name] = value

    rc.write_text("".join(lines))
    print(f"  Written to {rc}")
    print(f"  (Reload your shell or run: source {rc})")


def _offer_save_env_var(xml_path: str) -> None:
    for name in _ENV_VARS:
        current = os.environ.get(name)
        if current:
            print(f"\n  {name} is currently set to:")
            print(f"    {current}")
        else:
            print(f"\n  {name} is not set.")

    names = " and ".join(_ENV_VARS)
    choice = input(f"  ► Save/override {names}={xml_path} ? [y/N] ").strip().lower()
    if choice != "y":
        return

    print("  Add these lines to your station shell only if desired:")
    for name in _ENV_VARS:
        print(f'    export {name}="{xml_path}"')


def _offer_save_calibration_jsons(json_paths: dict[str, str]) -> None:
    """Offer to save per-camera calibration.json paths as env vars.

    ``json_paths`` maps the calibrator camera_name (e.g. "top_camera") to the
    absolute calibration.json path produced for that camera. Cameras missing
    from the mapping are skipped.
    """
    pending = {
        env_name: json_paths[camera]
        for camera, env_name in _CALIBRATION_JSON_ENV_VARS.items()
        if json_paths.get(camera)
    }
    if not pending:
        return

    print()
    for env_name, path in pending.items():
        current = os.environ.get(env_name)
        if current:
            print(f"  {env_name} is currently set to:")
            print(f"    {current}")
        else:
            print(f"  {env_name} is not set.")
        print(f"    → {path}")

    choice = (
        input(f"\n  ► Save/override the {len(pending)} calibration.json env var(s)? [y/N] ")
        .strip()
        .lower()
    )
    if choice != "y":
        return

    print("  Add these lines to your station shell only if desired:")
    for name, path in pending.items():
        print(f'    export {name}="{path}"')


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.confirm_motion:
        raise RuntimeError(
            "Calibration moves a live arm through multiple poses. Clear the workspace, "
            "keep an emergency stop reachable, then pass --confirm-motion."
        )
    missing_models = [str(path) for path in config.required_model_paths() if not path.is_file()]
    if missing_models:
        raise FileNotFoundError(
            "Missing station model assets. Set ENPIRE_YAM_MODEL_ROOT to the licensed "
            "YAM model bundle. Missing: " + ", ".join(missing_models)
        )
    cam = _cam_config(args.camera)
    interactive = not args.no_interactive
    squares_x = args.squares_x
    squares_y = args.squares_y
    square_length = args.square_length
    marker_length = args.marker_length

    print("\n" + "=" * 60)
    print(f"  {cam.camera_name} Calibration — Fello Gripper")
    print("=" * 60)
    print(f"  Camera  : {cam.serial}")
    print(f"  Arm port: {cam.arm_port}")
    print(
        f"  Board   : {squares_x}x{squares_y}  "
        f"sq={square_length * 1000:.0f}mm  mk={marker_length * 1000:.0f}mm"
    )
    print(f"  Resolution  : {args.resolution[0]}x{args.resolution[1]}")
    print(f"  Poses   : {len(config.CALIBRATION_POSE_OFFSETS)}  (offsets from nominal)")
    print(f"  Min samples : {config.MIN_SAMPLES}")
    print()

    home_event = threading.Event()

    def _on_sigint(sig, frame):
        home_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    # ── 1. Connect to arm server + open camera ────────────────────────────────
    print("[1/4] Connecting to arm server and camera...")
    arm = ArmClient(cam.arm_port)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            arm.get_joint_pos()
            break
        except Exception:
            time.sleep(0.2)
    else:
        raise ConnectionError("Could not reach the calibration arm server")
    print("  Connected.")

    serial = resolve_serial(cam.serial)
    camera = RealSenseCamera(serial=serial, fps=config.CAMERA_FPS, resolution=args.resolution)
    camera.open()
    for _ in range(10):
        camera.grab()
    K, dist, image_size = camera.get_intrinsics()
    w, h = image_size
    print(f"  RealSense {serial}: {w}x{h} @ {config.CAMERA_FPS}fps")
    print(f"  Intrinsics: fx={K[0, 0]:.1f}  fy={K[1, 1]:.1f}  {w}x{h}")
    print(f"  Dist:       {dist.round(5).tolist()}")

    charuco = CharucoDetector(squares_x, squares_y, square_length, marker_length, config.DICTIONARY)

    # ── Gravity mode: user drags arm while watching live detection ────────────
    grav_stop = threading.Event()
    grav_thread = threading.Thread(target=_gravity_loop, args=(arm, grav_stop), daemon=True)
    grav_thread.start()
    print(f"\n  Gravity mode active (kp=0, kd={config.GRAVITY_KD})")
    print("  Move arm to nominal calibration position.")
    print("  Check camera window — board must be visible.")
    print("  Press SPACE in the camera window to confirm, H to home.\n")

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

    while not home_event.is_set():
        if camera.grab():
            img = camera.get_image()
            corners, ids, mc, mi = charuco.detect(img)
            n = len(corners) if corners is not None else 0
            rvec, tvec = None, None
            if n >= 4:
                rvec, tvec = charuco.estimate_pose(corners, ids, K, dist)
            ann = img.copy()
            if mc is not None:
                cv2.aruco.drawDetectedMarkers(ann, mc, mi)
            if corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(ann, corners, ids)
            if rvec is not None:
                cv2.drawFrameAxes(ann, K, dist, rvec, tvec, square_length * 2, 2)
            ann = _draw_nominal_overlay(ann, n, rvec is not None, image_size)
            cv2.imshow("Calibration", ann)
        key = cv2.waitKey(int(_DT * 1000)) & 0xFF
        if key == ord(" "):
            break
        if key == ord("h"):
            home_event.set()

    if home_event.is_set():
        grav_stop.set()
        camera.close()
        cv2.destroyAllWindows()
        _go_to_zero(arm, shutdown=interactive)
        return 0

    q_start = arm.get_joint_pos()[:6]
    grav_stop.set()
    print(f"  Nominal pose locked: {np.round(q_start, 3)}")

    # ── 2. Load FK model ──────────────────────────────────────────────────────
    print("\n[2/4] Loading FK model...")
    mj_model = mujoco.MjModel.from_xml_path(config.FK_XML)
    mj_data = mujoco.MjData(mj_model)
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, cam.fk_body)
    assert body_id >= 0, f"Body '{cam.fk_body}' not found in {config.FK_XML}"
    joint_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in cam.fk_joints
    ]
    missing_joints = [
        name for name, joint_id in zip(cam.fk_joints, joint_ids, strict=True) if joint_id < 0
    ]
    assert not missing_joints, f"Joint(s) {missing_joints} not found in {config.FK_XML}"
    print("  Done.")

    # ── 3. Auto-capture ───────────────────────────────────────────────────────
    print(
        f"\n[3/4] Ready — {len(config.CALIBRATION_POSE_OFFSETS)} poses. Press SPACE in camera window to start."
    )

    while not home_event.is_set():
        if camera.grab():
            img = camera.get_image()
            ann = _draw_ready_overlay(img.copy(), len(config.CALIBRATION_POSE_OFFSETS))
            cv2.imshow("Calibration", ann)
        key = cv2.waitKey(int(_DT * 1000)) & 0xFF
        if key == ord(" "):
            break
        if key == ord("h"):
            home_event.set()

    if home_event.is_set():
        camera.close()
        cv2.destroyAllWindows()
        _do_home(arm, q_start)
        _go_to_zero(arm, shutdown=interactive)
        return 0

    samples: list[Sample] = []
    poses = [q_start + np.array(off) for off in config.CALIBRATION_POSE_OFFSETS]

    try:
        for i, q_target in enumerate(poses):
            if home_event.is_set():
                break

            pose_num = i + 1
            print(f"\n  [{pose_num:02d}/{len(poses)}] Target: {np.round(q_target, 3)}")

            max_step = config.MAX_VEL * _DT
            kp = np.array(config.ARM_KP)
            kd = np.array(config.ARM_KD)
            move_deadline = time.time() + 3.0
            q_setpoint = arm.get_joint_pos()[:6].copy()

            while not home_event.is_set():
                q = arm.get_joint_pos()[:6]
                diff = q_target - q
                if _converged(diff):
                    break
                if time.time() > move_deadline:
                    print(f"    [WARN] Move timeout — proceeding (residual={np.round(diff, 3)})")
                    break
                step = np.clip(q_target - q_setpoint, -max_step, max_step)
                q_setpoint = q_setpoint + step
                arm.command_joint_pos(q_setpoint, kp=kp, kd=kd)
                key = cv2.waitKey(int(_DT * 1000)) & 0xFF
                if key == ord("h"):
                    home_event.set()

            if home_event.is_set():
                break

            time.sleep(config.SETTLE_TIME)

            # Sample best detection over 1s
            best_rvec = best_tvec = None
            best_n = 0
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if not camera.grab():
                    continue
                img = camera.get_image()
                corners, ids, mc, mi = charuco.detect(img)
                n = len(corners) if corners is not None else 0
                rvec = tvec = None
                if n >= 4:
                    rvec, tvec = charuco.estimate_pose(corners, ids, K, dist)
                if n > best_n:
                    best_n = n
                    if rvec is not None:
                        best_rvec, best_tvec = rvec, tvec

                # Live overlay
                ann = img.copy()
                if mc is not None:
                    cv2.aruco.drawDetectedMarkers(ann, mc, mi)
                if corners is not None:
                    cv2.aruco.drawDetectedCornersCharuco(ann, corners, ids)
                if rvec is not None:
                    cv2.drawFrameAxes(ann, K, dist, rvec, tvec, square_length * 2, 2)
                ann = _draw_overlay(
                    ann, pose_num, len(poses), n, len(samples), rvec is not None, image_size
                )
                cv2.imshow("Calibration", ann)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("h"):
                    home_event.set()
                    break

            if home_event.is_set():
                break

            if best_rvec is None:
                print(f"    [SKIP] Board not detected (corners={best_n}).")
                continue

            q = arm.get_joint_pos()[:6]
            T_base_from_ee = _fk(mj_model, mj_data, body_id, joint_ids, q)
            T_cam_from_board = np.eye(4, dtype=np.float64)
            T_cam_from_board[:3, :3], _ = cv2.Rodrigues(best_rvec.ravel())
            T_cam_from_board[:3, 3] = best_tvec.ravel()

            samples.append(
                Sample(
                    q=q.copy(),
                    T_base_from_ee=T_base_from_ee,
                    T_ee_from_base=np.linalg.inv(T_base_from_ee),
                    T_cam_from_board=T_cam_from_board,
                )
            )
            print(f"    [OK] Sample {len(samples)} — corners={best_n}")

    except KeyboardInterrupt:
        home_event.set()

    cv2.destroyAllWindows()
    camera.close()

    # Home if requested
    if home_event.is_set():
        _do_home(arm, q_start)
        _go_to_zero(arm, shutdown=interactive)
        print(f"\n  Aborted after {len(samples)} samples.")
        return 0

    # Return to start then zero
    print("\n  Returning to nominal pose...")
    arm.smooth_move_to(q_start)

    print(f"\n  Captured {len(samples)}/{len(poses)} samples.")
    if len(samples) < config.MIN_SAMPLES:
        print(f"  [ABORT] Need {config.MIN_SAMPLES} samples, got {len(samples)}.")
        _go_to_zero(arm, shutdown=interactive)
        return 1

    # ── 4. Calibrate + generate XML ───────────────────────────────────────────
    print("\n[4/4] Running hand-eye calibration...")
    result = run_hand_eye(samples, eye_in_hand=cam.eye_in_hand)
    best_name = result["best"]
    best = result["all"][best_name]
    T = best["T"]

    print(f"\n  Best method : {best_name}")
    print(f"  Trans RMS   : {best['trans_rms_mm']:.2f} mm")
    print(f"  Rot RMS     : {best['rot_rms_deg']:.3f} deg")
    for n, v in result["all"].items():
        mark = " ◄" if n == best_name else ""
        print(f"    {n:<12} trans={v['trans_rms_mm']:.2f}mm  rot={v['rot_rms_deg']:.3f}deg{mark}")

    out_dir = config.OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    quat_wxyz = Rotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
    fovy = 2 * math.degrees(math.atan(h / (2 * K[1, 1])))

    cal = {
        "camera_name": cam.camera_name,
        "camera_serial": serial,
        "timestamp": datetime.now().isoformat(),
        "hand_eye": {
            "method": best_name,
            "T_base_from_camera": T.tolist(),
            "position_m": T[:3, 3].tolist(),
            "quat_wxyz": quat_wxyz.tolist(),
            "translation_rms_mm": best["trans_rms_mm"],
            "rotation_rms_deg": best["rot_rms_deg"],
        },
        "intrinsics": {
            "fx": K[0, 0],
            "fy": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2],
            "camera_matrix": K.tolist(),
            "dist_coeffs": dist.tolist(),
            "image_size": list(image_size),
            "fovy_deg": fovy,
        },
        "board_config": {
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_length_m": square_length,
            "marker_length_m": marker_length,
        },
        "all_methods": {
            n: {"translation_rms_mm": v["trans_rms_mm"], "rotation_rms_deg": v["rot_rms_deg"]}
            for n, v in result["all"].items()
        },
    }

    json_path = out_dir / "calibration.json"
    json_path.write_text(json.dumps(cal, indent=2))
    print(f"\n  Saved: {json_path}")

    # Chain: read from OUTPUT_XML if it already exists (preserves prior calibrations),
    # otherwise fall back to BASE_XML.
    base_xml = config.OUTPUT_XML if Path(config.OUTPUT_XML).exists() else config.BASE_XML
    update_body_extrinsic(
        base_xml=base_xml,
        body_name=cam.body_name,
        T=T,
        output_xml=config.OUTPUT_XML,
    )
    print(f"  Generated: {config.OUTPUT_XML}")
    print("\n" + "=" * 60)
    print("  Calibration complete.")
    print("=" * 60 + "\n")

    if interactive:
        _offer_save_env_var(config.OUTPUT_XML)

    _go_to_zero(arm, shutdown=interactive)

    # ── Optional MuJoCo viewer ────────────────────────────────────────────────
    if interactive:
        choice = input("  ► Visualize calibration in MuJoCo viewer? [y/N] ").strip().lower()
        if choice == "y":
            print(f"  Opening viewer: {config.OUTPUT_XML}")
            mujoco.viewer.launch(mujoco.MjModel.from_xml_path(config.OUTPUT_XML))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
