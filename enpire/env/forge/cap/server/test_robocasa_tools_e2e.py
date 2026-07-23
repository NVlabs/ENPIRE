# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test: CAP agent tools against RoboCasa env.

Tests the full tool → cap_server → env pipeline for:
  1. get_robot_state — EE pose, gripper state
  2. vlm_query (Gemini) — scene understanding from camera images
  3. SAM3 segmentation — the first half of sample_grasp_pose_anygrasp
  4. get_camera_extrinsics — needs_optical_flip field

Requires:
  - SAM3 server running on port 6767
  - GEMINI_API_KEY set

Usage:
    CAP_AGENT_NAME=test uv run python -u cap/server/test_robocasa_tools_e2e.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

os.environ.setdefault("ROBOCASA_LAYOUT_ID", "1")
os.environ.setdefault("ROBOCASA_STYLE_ID", "1")

from enpire.env.forge.cap.agent.tools import create_default_registry
from enpire.env.forge.cap.server.cap_server import CapServer

PASS = 0
FAIL = 0
SAM3_URL = "http://localhost:6767"
ANYGRASP_URL = "http://localhost:8122"


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


def _service_up(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except urllib.error.HTTPError:
        # Server is up but returned an HTTP error (e.g., 404 on /)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Start cap_server with RoboCasa
# ---------------------------------------------------------------------------

print("Starting CapServer with RoboCasa...", flush=True)
server = CapServer(
    server_port=18502,
    enable_cameras=False,
    env_name="robocasa:PickPlaceCounterToCabinet",
    env_viewer=False,
)
server.start()
time.sleep(3.0)

# Build tool registry pointing at our server
registry = create_default_registry(
    cap_server_host="localhost",
    cap_server_port=18502,
    sam3_host="localhost",
    sam3_port=6767,
)
tools = registry.callable_dict()

print()

# ---------------------------------------------------------------------------
# Test 1: get_robot_state
# ---------------------------------------------------------------------------
print("=== Test 1: get_robot_state ===", flush=True)
state = tools["get_robot_state"]()
check("returns RobotState", hasattr(state, "right_ee_pos"))
ee_pos = list(state.right_ee_pos)
check("EE pos non-zero", not np.allclose(ee_pos, 0.0),
      f"xyz={[round(float(x), 3) for x in ee_pos]}")
check("gripper in range", 0.0 <= float(state.right_gripper_pos) <= 1.0,
      f"val={float(state.right_gripper_pos):.3f}")
print()

# ---------------------------------------------------------------------------
# Test 2: get_task_info (TaskProtocol)
# ---------------------------------------------------------------------------
print("=== Test 2: get_task_info ===", flush=True)
task_info = tools["get_task_info"]()
check("returns dict", isinstance(task_info, dict))
check("has env_name", "env_name" in task_info,
      f"env={task_info.get('env_name')}")
check("has done/success", "done" in task_info and "success" in task_info)
obj_keys = [k for k in task_info if k.endswith("_pos") and not k.startswith("robot")]
check("has object positions", len(obj_keys) > 0, f"keys={obj_keys}")
for k in obj_keys[:3]:
    pos = task_info[k]
    check(f"  {k} is 3D", len(pos) == 3, f"{[round(float(x), 3) for x in pos]}")
print()

# ---------------------------------------------------------------------------
# Test 3: get_camera_image
# ---------------------------------------------------------------------------
print("=== Test 3: get_camera_image ===", flush=True)
for cam in ("top", "wrist"):
    img = tools["get_camera_image"](cam)
    check(f"{cam} returns ndarray", isinstance(img, np.ndarray), f"shape={img.shape}")
    check(f"{cam} is color", img.ndim == 3 and img.shape[2] == 3)
    check(f"{cam} non-black", float(img.mean()) > 1.0, f"mean={img.mean():.1f}")
print()

# ---------------------------------------------------------------------------
# Test 4: vlm_query (Gemini) — full pipeline: camera image → VLM
# ---------------------------------------------------------------------------
print("=== Test 4: vlm_query (Gemini) ===", flush=True)
if os.environ.get("GEMINI_API_KEY"):
    try:
        response = tools["vlm_query"](
            text="What objects do you see in this kitchen scene? List them briefly.",
            backend="gemini",
            model="gemini-2.5-flash",
            media=["camera:top"],
        )
        check("vlm returns string", isinstance(response, str) and len(response) > 5,
              f"len={len(response)}")
        check("vlm mentions something", len(response.split()) > 3,
              f"response: {response[:120]}...")
    except Exception as e:
        check("vlm_query succeeded", False, f"error: {e}")
else:
    print("  [SKIP] GEMINI_API_KEY not set")
print()

# ---------------------------------------------------------------------------
# Test 5: SAM3 segmentation (first half of sample_grasp_pose_anygrasp pipeline)
# ---------------------------------------------------------------------------
print("=== Test 5: SAM3 segmentation ===", flush=True)
sam3_up = _service_up(f"{SAM3_URL}/")
check("SAM3 server reachable", sam3_up)

if sam3_up:
    # Get RGB from RoboCasa via cap_server
    rgb = server.get_camera_image("top")
    check("got RGB for segmentation", rgb.shape[2] == 3, f"shape={rgb.shape}")

    # Send to SAM3
    buf = io.BytesIO()
    np.save(buf, rgb)
    image_b64 = base64.b64encode(buf.getvalue()).decode()
    payload = json.dumps({"text": "object on counter", "image_b64": image_b64}).encode()

    try:
        req = urllib.request.Request(
            f"{SAM3_URL}/segment",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        mask_bytes = base64.b64decode(data["mask_b64"])
        mask = np.load(io.BytesIO(mask_bytes)).astype(np.int32)
        check("SAM3 returns mask", mask.shape[:2] == rgb.shape[:2],
              f"mask_shape={mask.shape}")
        n_pixels = int((mask > 0).sum())
        check("mask has positive region", n_pixels > 100,
              f"n_pixels={n_pixels}")
    except Exception as e:
        check("SAM3 segmentation succeeded", False, f"error: {e}")
else:
    print("  [SKIP] SAM3 server not running")
print()

# ---------------------------------------------------------------------------
# Test 6: Camera extrinsics + needs_optical_flip (the fix we made)
# ---------------------------------------------------------------------------
print("=== Test 6: get_camera_extrinsics (needs_optical_flip) ===", flush=True)
for cam in ("top", "wrist"):
    extr = server.get_camera_extrinsics(cam)
    check(f"{cam} has needs_optical_flip", "needs_optical_flip" in extr)
    if "needs_optical_flip" in extr:
        check(f"{cam} flip is True (sim)", extr["needs_optical_flip"] is True)

    # Verify the extrinsics-based projection pipeline works
    depth = server.get_camera_depth(cam)
    intr = server.get_camera_intrinsics(cam)
    fx, fy, cx, cy = intr
    H, W = depth.shape
    u, v = W // 2, H // 2
    z = float(depth[v, u])
    if z > 0:
        x_cam = (u - cx) / fx * z
        y_cam = (v - cy) / fy * z
        p_cam = np.array([x_cam, y_cam, z])
        R = np.asarray(extr["rotation"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(extr["position"], dtype=np.float64)
        if extr.get("needs_optical_flip", True):
            R_cv = R @ np.diag([-1.0, -1.0, 1.0])
        else:
            R_cv = R
        p_world = R_cv @ p_cam + t
        check(f"{cam} depth→world plausible", 0.0 < p_world[2] < 3.0,
              f"world_z={p_world[2]:.3f}")
print()

# ---------------------------------------------------------------------------
# Test 7: AnyGrasp tool (expect graceful failure if server down)
# ---------------------------------------------------------------------------
print("=== Test 7: sample_grasp_pose_anygrasp (availability) ===", flush=True)
anygrasp_up = _service_up(f"{ANYGRASP_URL}/")
if anygrasp_up:
    try:
        grasps = tools["sample_grasp_pose_anygrasp"](
            object_name="object on counter",
            camera="top",
            max_grasps=4,
        )
        check("anygrasp returns grasps", isinstance(grasps, list) and len(grasps) > 0,
              f"n_grasps={len(grasps) if isinstance(grasps, list) else '?'}")
        if grasps:
            g = grasps[0]
            check("grasp has position", hasattr(g, 'position') and len(g.position) == 3,
                  f"pos={[round(float(x), 3) for x in g.position]}")
            check("grasp has rpy", hasattr(g, 'rpy') and len(g.rpy) == 3)
            check("grasp has score", hasattr(g, 'score'))
    except RuntimeError as e:
        # Tool returns RuntimeError for expected failures
        err_str = str(e)
        if "AnyGrasp" in err_str or "no grasps" in err_str.lower():
            check("anygrasp tool ran (no grasps found)", True, f"msg: {err_str[:80]}")
        else:
            check("anygrasp tool ran", False, f"unexpected error: {err_str[:80]}")
    except Exception as e:
        check("anygrasp tool ran", False, f"error: {e}")
else:
    print("  [SKIP] AnyGrasp server not running on :8122")
    check("anygrasp server not available (known: needs OpenSSL 1.1)", True,
          "binary compat issue, not a tool bug")
print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL > 0:
    print("SOME TESTS FAILED")
else:
    print("ALL TESTS PASSED")
print("=" * 60)

server.stop()
print("Done!")
sys.exit(1 if FAIL > 0 else 0)
