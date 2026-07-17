"""Test: verify CAP agent tools work through cap_server for the RoboCasa env.

Checks get_robot_state, get_camera_image/depth/intrinsics/extrinsics,
and the needs_optical_flip field that vision tools depend on.

Usage:
    uv run python -u cap/server/test_robocasa_tool_compat.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from enpire.env.forge.cap.server.cap_server import CapServer

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg, flush=True)
    if ok:
        PASS += 1
    else:
        FAIL += 1


# ---------------------------------------------------------------------------
# Start cap_server with RoboCasa
# ---------------------------------------------------------------------------

print("Starting CAP server with RoboCasa (no viewer)...", flush=True)
import os
os.environ.setdefault("ROBOCASA_LAYOUT_ID", "1")
os.environ.setdefault("ROBOCASA_STYLE_ID", "1")

server = CapServer(
    server_port=18501,
    enable_cameras=False,
    env_name="robocasa:PickPlaceCounterToCabinet",
    env_viewer=False,
)
server.start()
time.sleep(2.0)
print()

# ---------------------------------------------------------------------------
# 1. get_state — equivalent to what get_robot_state tool calls
# ---------------------------------------------------------------------------
print("=== get_state (get_robot_state tool) ===", flush=True)
state = server.get_state()

check("state is dict", isinstance(state, dict))
check("has right_ee_pos", "right_ee_pos" in state)
check("has right_ee_quat_xyzw", "right_ee_quat_xyzw" in state)
check("has right_joint_pos", "right_joint_pos" in state)
check("has right_gripper_pos", "right_gripper_pos" in state)

if "right_ee_pos" in state:
    ee = state["right_ee_pos"]
    check("ee_pos is 3D", len(ee) == 3)
    check("ee_pos not all zeros", not np.allclose(ee, 0.0),
          f"xyz={[round(float(x), 3) for x in ee]}")

if "right_ee_quat_xyzw" in state:
    eq = state["right_ee_quat_xyzw"]
    check("ee_quat is 4D", len(eq) == 4)
    qnorm = float(np.linalg.norm(eq))
    check("ee_quat is unit", abs(qnorm - 1.0) < 0.05, f"norm={qnorm:.4f}")

if "right_gripper_pos" in state:
    gp = float(state["right_gripper_pos"][0])
    check("gripper_pos in [0,1]", 0.0 <= gp <= 1.0, f"val={gp:.3f}")

# Mobile base fields
check("has base_pos (PandaOmron)", "base_pos" in state,
      f"keys={sorted(state.keys())}")

print()

# ---------------------------------------------------------------------------
# 2. Camera image — equivalent to what vlm_query / grasp tools call
# ---------------------------------------------------------------------------
print("=== get_camera_image (vlm_query tool) ===", flush=True)

for cam_name in ("top", "wrist"):
    rgb = server.get_camera_image(cam_name)
    check(f"{cam_name} rgb shape", rgb.ndim == 3 and rgb.shape[2] == 3,
          f"shape={rgb.shape}")
    check(f"{cam_name} rgb dtype", rgb.dtype == np.uint8)
    check(f"{cam_name} rgb not black", rgb.mean() > 1.0,
          f"mean={rgb.mean():.1f}")

print()

# ---------------------------------------------------------------------------
# 3. Camera depth — needed by AnyGrasp and BundleSDF
# ---------------------------------------------------------------------------
print("=== get_camera_depth ===", flush=True)

for cam_name in ("top", "wrist"):
    depth = server.get_camera_depth(cam_name)
    check(f"{cam_name} depth shape", depth.ndim == 2, f"shape={depth.shape}")
    check(f"{cam_name} depth dtype", depth.dtype == np.float32)
    check(f"{cam_name} depth has values", depth.max() > 0.0,
          f"min={depth.min():.3f} max={depth.max():.3f}")

print()

# ---------------------------------------------------------------------------
# 4. Camera intrinsics — needed by AnyGrasp and BundleSDF
# ---------------------------------------------------------------------------
print("=== get_camera_intrinsics ===", flush=True)

for cam_name in ("top", "wrist"):
    intr = server.get_camera_intrinsics(cam_name)
    check(f"{cam_name} intrinsics length", len(intr) == 4, f"val={intr}")
    fx, fy, cx, cy = intr
    check(f"{cam_name} fx > 0", fx > 0.0, f"fx={fx:.1f}")
    check(f"{cam_name} fy > 0", fy > 0.0, f"fy={fy:.1f}")
    check(f"{cam_name} cx > 0", cx > 0.0, f"cx={cx:.1f}")
    check(f"{cam_name} cy > 0", cy > 0.0, f"cy={cy:.1f}")

print()

# ---------------------------------------------------------------------------
# 5. Camera extrinsics + needs_optical_flip — the key fix we made
# ---------------------------------------------------------------------------
print("=== get_camera_extrinsics (+ needs_optical_flip) ===", flush=True)

for cam_name in ("top", "wrist"):
    extr = server.get_camera_extrinsics(cam_name)
    check(f"{cam_name} has 'position'", "position" in extr)
    check(f"{cam_name} has 'rotation'", "rotation" in extr)
    check(f"{cam_name} has 'needs_optical_flip'", "needs_optical_flip" in extr,
          "THIS IS THE KEY NEW FIELD")

    if "needs_optical_flip" in extr:
        check(f"{cam_name} needs_optical_flip is True (sim)",
              extr["needs_optical_flip"] is True)

    pos = extr.get("position", [])
    check(f"{cam_name} position is 3D", len(pos) == 3,
          f"pos={[round(x, 3) for x in pos]}")

    rot = np.asarray(extr.get("rotation", []), dtype=np.float64).reshape(3, 3)
    det = float(np.linalg.det(rot))
    check(f"{cam_name} rotation is valid SO(3)", abs(det - 1.0) < 0.01,
          f"det={det:.4f}")

    # Verify the full pipeline: Pinocchio → OpenCV flip
    R = rot
    F = np.diag([-1.0, -1.0, 1.0])
    R_opencv = R @ F
    det_cv = float(np.linalg.det(R_opencv))
    check(f"{cam_name} flipped rotation valid SO(3)", abs(det_cv - 1.0) < 0.01,
          f"det={det_cv:.4f}")

print()

# ---------------------------------------------------------------------------
# 6. Consistency: depth projection sanity check
# ---------------------------------------------------------------------------
print("=== Depth-to-3D projection sanity ===", flush=True)

cam = "top"
rgb = server.get_camera_image(cam)
depth = server.get_camera_depth(cam)
intr = server.get_camera_intrinsics(cam)
extr = server.get_camera_extrinsics(cam)

fx, fy, cx, cy = intr
H, W = depth.shape

# Pick the center pixel
u, v = int(W // 2), int(H // 2)
z = float(depth[v, u])
check("center pixel has depth", z > 0.0, f"z={z:.3f}m")

if z > 0.0:
    # Back-project to camera frame
    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    p_cam = np.array([x_cam, y_cam, z])

    # Camera → world (with optical flip as tools would do)
    R = np.asarray(extr["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(extr["position"], dtype=np.float64)
    if extr.get("needs_optical_flip", True):
        R_cv = R @ np.diag([-1.0, -1.0, 1.0])
    else:
        R_cv = R
    p_world = R_cv @ p_cam + t

    check("world point plausible z > 0", p_world[2] > 0.0,
          f"world_xyz={[round(float(x), 3) for x in p_world]}")
    check("world point plausible z < 3m", p_world[2] < 3.0,
          f"z={p_world[2]:.3f}")

print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 50)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL > 0:
    print("SOME TESTS FAILED")
    server.stop()
    sys.exit(1)
else:
    print("ALL TESTS PASSED")

server.stop()
print("Done!")
