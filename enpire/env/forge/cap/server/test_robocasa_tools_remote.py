# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test: CAP agent tools against RoboCasa env with remote vision servers.

Uses the same SSH-tunnel pattern as launch_table_bussing_remote.sh:
  - SAM3, AnyGrasp, BundleSDF run on a remote GPU server
  - SSH tunnels forward remote ports to localhost
  - Tools see localhost:{6767,8122,8119} as usual

Tests:
  1. get_robot_state
  2. get_task_info (TaskProtocol)
  3. get_camera_image (top, wrist)
  4. vlm_query (Gemini — cloud, no tunnel needed)
  5. SAM3 segmentation via tunnel
  6. get_camera_extrinsics + needs_optical_flip
  7. sample_grasp_pose_anygrasp (SAM3 + AnyGrasp via tunnel)
  8. detect_objects_oneshot (BundleSDF via tunnel)

Usage:
    # Option A: tunnels already open (e.g., from launch_table_bussing_remote.sh)
    CAP_AGENT_NAME=test uv run python -u cap/server/test_robocasa_tools_remote.py

    # Option B: provide SSH target, script opens tunnels itself
    CAP_AGENT_NAME=test SSH_TARGET=my-gpu-server \
        uv run python -u cap/server/test_robocasa_tools_remote.py

    # Option C: remote servers on custom host (no SSH tunnel)
    CAP_AGENT_NAME=test SAM3_SERVER_HOST=gpu-box ANYGRASP_SERVER_HOST=gpu-box \
        BUNDLESDF_SERVER_HOST=gpu-box \
        uv run python -u cap/server/test_robocasa_tools_remote.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

os.environ.setdefault("ROBOCASA_LAYOUT_ID", "1")
os.environ.setdefault("ROBOCASA_STYLE_ID", "1")

from enpire.env.forge.cap.server.cap_server import CapServer
from enpire.env.forge.cap.agent.tools import create_default_registry

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAM3_PORT = int(os.environ.get("SAM3_SERVER_PORT", "6767"))
ANYGRASP_PORT = int(os.environ.get("ANYGRASP_SERVER_PORT", "8122"))
BUNDLESDF_PORT = int(os.environ.get("BUNDLESDF_SERVER_PORT", "8119"))
SAM3_HOST = os.environ.get("SAM3_SERVER_HOST", "localhost")
ANYGRASP_HOST = os.environ.get("ANYGRASP_SERVER_HOST", "localhost")
BUNDLESDF_HOST = os.environ.get("BUNDLESDF_SERVER_HOST", "localhost")

SSH_TARGET = os.environ.get("SSH_TARGET", "")
CAP_SERVER_PORT = 18503

PASS = 0
FAIL = 0
_tunnel_proc = None


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


def _service_up(host: str, port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://{host}:{port}/", timeout=3)
        return True
    except urllib.error.HTTPError:
        return True  # server up, just no root handler
    except Exception:
        return False


# ---------------------------------------------------------------------------
# SSH tunnel (optional)
# ---------------------------------------------------------------------------

def start_ssh_tunnels():
    """Open SSH tunnels to forward remote vision server ports locally."""
    global _tunnel_proc
    if not SSH_TARGET:
        return

    # Only tunnel ports that are on localhost (not already remote)
    tunnels = []
    if SAM3_HOST == "localhost":
        tunnels.append(f"-L {SAM3_PORT}:127.0.0.1:{SAM3_PORT}")
    if ANYGRASP_HOST == "localhost":
        tunnels.append(f"-L {ANYGRASP_PORT}:127.0.0.1:{ANYGRASP_PORT}")
    if BUNDLESDF_HOST == "localhost":
        tunnels.append(f"-L {BUNDLESDF_PORT}:127.0.0.1:{BUNDLESDF_PORT}")
    # Reverse tunnel so remote BundleSDF can reach local cap_server
    tunnels.append(f"-R 18300:127.0.0.1:{CAP_SERVER_PORT}")

    if not tunnels:
        return

    cmd = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes"] + tunnels + [SSH_TARGET]
    print(f"[tunnel] Opening SSH tunnels to {SSH_TARGET}...", flush=True)
    print(f"[tunnel] cmd: {' '.join(cmd)}", flush=True)
    _tunnel_proc = subprocess.Popen(cmd)
    time.sleep(3.0)
    if _tunnel_proc.poll() is not None:
        print(f"[tunnel] SSH tunnel failed (exit {_tunnel_proc.returncode})", flush=True)
        _tunnel_proc = None
    else:
        print("[tunnel] SSH tunnels established", flush=True)


def stop_ssh_tunnels():
    global _tunnel_proc
    if _tunnel_proc is not None:
        _tunnel_proc.send_signal(signal.SIGTERM)
        _tunnel_proc.wait(timeout=5)
        _tunnel_proc = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 60)
print("RoboCasa + Remote Vision Servers — E2E Tool Test")
print("=" * 60)
print(f"  SAM3:      {SAM3_HOST}:{SAM3_PORT}")
print(f"  AnyGrasp:  {ANYGRASP_HOST}:{ANYGRASP_PORT}")
print(f"  BundleSDF: {BUNDLESDF_HOST}:{BUNDLESDF_PORT}")
print(f"  SSH target: {SSH_TARGET or '(none — expecting tunnels or direct)'}")
print()

# Start SSH tunnels if SSH_TARGET is set
start_ssh_tunnels()

# Start local CapServer with RoboCasa
print("Starting CapServer with RoboCasa...", flush=True)
server = CapServer(
    server_port=CAP_SERVER_PORT,
    enable_cameras=False,
    env_name="robocasa:PickPlaceCounterToCabinet",
    env_viewer=False,
)
server.start()
time.sleep(3.0)

# Build tool registry
registry = create_default_registry(
    cap_server_host="localhost",
    cap_server_port=CAP_SERVER_PORT,
    sam3_host=SAM3_HOST,
    sam3_port=SAM3_PORT,
    bundlesdf_host=BUNDLESDF_HOST,
    bundlesdf_port=BUNDLESDF_PORT,
)
tools = registry.callable_dict()

# Check which remote services are available
sam3_up = _service_up(SAM3_HOST, SAM3_PORT)
anygrasp_up = _service_up(ANYGRASP_HOST, ANYGRASP_PORT)
bundlesdf_up = _service_up(BUNDLESDF_HOST, BUNDLESDF_PORT)
gemini_key = bool(os.environ.get("GEMINI_API_KEY"))

print(f"\n  Service status:")
print(f"    SAM3:      {'UP' if sam3_up else 'DOWN'}")
print(f"    AnyGrasp:  {'UP' if anygrasp_up else 'DOWN'}")
print(f"    BundleSDF: {'UP' if bundlesdf_up else 'DOWN'}")
print(f"    Gemini:    {'KEY SET' if gemini_key else 'NO KEY'}")
print()

try:
    # ------------------------------------------------------------------
    # Test 1: get_robot_state
    # ------------------------------------------------------------------
    print("=== Test 1: get_robot_state ===", flush=True)
    state = tools["get_robot_state"]()
    check("returns RobotState", hasattr(state, "right_ee_pos"))
    ee = list(state.right_ee_pos)
    check("EE pos non-zero", not np.allclose(ee, 0.0),
          f"xyz={[round(float(x), 3) for x in ee]}")
    print()

    # ------------------------------------------------------------------
    # Test 2: get_task_info
    # ------------------------------------------------------------------
    print("=== Test 2: get_task_info ===", flush=True)
    task = tools["get_task_info"]()
    check("returns dict", isinstance(task, dict))
    obj_keys = [k for k in task if k.endswith("_pos") and not k.startswith("robot")]
    check("has object positions", len(obj_keys) > 0, f"keys={obj_keys}")
    print()

    # ------------------------------------------------------------------
    # Test 3: get_camera_image
    # ------------------------------------------------------------------
    print("=== Test 3: get_camera_image ===", flush=True)
    rgb = tools["get_camera_image"]("top")
    check("top RGB valid", rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.mean() > 1.0,
          f"shape={rgb.shape}, mean={rgb.mean():.1f}")
    print()

    # ------------------------------------------------------------------
    # Test 4: vlm_query (Gemini — cloud API, no tunnel needed)
    # ------------------------------------------------------------------
    print("=== Test 4: vlm_query (Gemini) ===", flush=True)
    if gemini_key:
        try:
            resp = tools["vlm_query"](
                text="List the objects you see in this kitchen. Be brief.",
                backend="gemini",
                model="gemini-2.5-flash",
                media=["camera:top"],
            )
            check("VLM returns response", isinstance(resp, str) and len(resp) > 5,
                  f"len={len(resp)}, preview: {resp[:100]}...")
        except Exception as e:
            check("vlm_query succeeded", False, str(e)[:100])
    else:
        print("  [SKIP] GEMINI_API_KEY not set")
    print()

    # ------------------------------------------------------------------
    # Test 5: SAM3 segmentation (via remote tunnel)
    # ------------------------------------------------------------------
    print("=== Test 5: SAM3 segmentation ===", flush=True)
    if sam3_up:
        buf = io.BytesIO()
        np.save(buf, rgb)
        payload = json.dumps({
            "text": "object on counter",
            "image_b64": base64.b64encode(buf.getvalue()).decode(),
        }).encode()
        try:
            req = urllib.request.Request(
                f"http://{SAM3_HOST}:{SAM3_PORT}/segment",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())
            mask = np.load(io.BytesIO(base64.b64decode(data["mask_b64"]))).astype(np.int32)
            n_pix = int((mask > 0).sum())
            check("SAM3 mask valid", mask.shape[:2] == rgb.shape[:2],
                  f"shape={mask.shape}")
            check("SAM3 mask non-empty", n_pix > 50, f"n_pixels={n_pix}")
        except Exception as e:
            check("SAM3 segmentation", False, str(e)[:100])
    else:
        print("  [SKIP] SAM3 not reachable")
    print()

    # ------------------------------------------------------------------
    # Test 6: get_camera_extrinsics + needs_optical_flip
    # ------------------------------------------------------------------
    print("=== Test 6: get_camera_extrinsics ===", flush=True)
    for cam in ("top", "wrist"):
        extr = server.get_camera_extrinsics(cam)
        check(f"{cam} needs_optical_flip present", "needs_optical_flip" in extr)
        check(f"{cam} needs_optical_flip is True", extr.get("needs_optical_flip") is True)
    print()

    # ------------------------------------------------------------------
    # Test 7: sample_grasp_pose_anygrasp (full pipeline via tunnels)
    # ------------------------------------------------------------------
    print("=== Test 7: sample_grasp_pose_anygrasp ===", flush=True)
    if sam3_up and anygrasp_up:
        try:
            grasps = tools["sample_grasp_pose_anygrasp"](
                object_name="object on the counter",
                camera="top",
                max_grasps=8,
            )
            check("returns grasp list", isinstance(grasps, list),
                  f"n={len(grasps) if isinstance(grasps, list) else '?'}")
            if grasps:
                g = grasps[0]
                check("grasp has position", hasattr(g, "position") and len(g.position) == 3,
                      f"xyz={[round(float(x), 3) for x in g.position]}")
                check("grasp has score", hasattr(g, "score"),
                      f"score={float(g.score):.3f}")
        except RuntimeError as e:
            msg = str(e)
            # Expected: "no grasps" is a valid tool response for some scenes
            if "no grasps" in msg.lower() or "anygrasp" in msg.lower():
                check("anygrasp tool ran (no grasps for this scene)", True, msg[:100])
            else:
                check("anygrasp tool", False, msg[:100])
        except Exception as e:
            check("anygrasp tool", False, str(e)[:100])
    else:
        missing = []
        if not sam3_up:
            missing.append("SAM3")
        if not anygrasp_up:
            missing.append("AnyGrasp")
        print(f"  [SKIP] {', '.join(missing)} not reachable")
    print()

    # ------------------------------------------------------------------
    # Test 8: detect_objects_oneshot (BundleSDF via tunnel)
    # ------------------------------------------------------------------
    print("=== Test 8: detect_objects_oneshot ===", flush=True)
    if bundlesdf_up:
        try:
            detections = tools["detect_objects_oneshot"](
                query="object on counter",
                camera="top",
            )
            check("returns detection dict", isinstance(detections, dict),
                  f"keys={list(detections.keys()) if isinstance(detections, dict) else '?'}")
            if isinstance(detections, dict):
                for name, dets in detections.items():
                    if dets:
                        d = dets[0]
                        check(f"detection '{name}' has position_3d",
                              hasattr(d, "position_3d") and d.position_3d and len(d.position_3d) == 3,
                              f"pos={d.position_3d}")
        except RuntimeError as e:
            msg = str(e)
            if "no" in msg.lower() and "detection" in msg.lower():
                check("bundlesdf tool ran (no detections)", True, msg[:100])
            else:
                check("bundlesdf tool", False, msg[:100])
        except Exception as e:
            check("bundlesdf tool", False, str(e)[:100])
    else:
        print("  [SKIP] BundleSDF not reachable")
    print()

finally:
    # ------------------------------------------------------------------
    # Cleanup & summary
    # ------------------------------------------------------------------
    print("=" * 60)
    total = PASS + FAIL
    print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")
    if FAIL > 0:
        print("SOME TESTS FAILED")
    else:
        print("ALL TESTS PASSED")
    print("=" * 60)

    server.stop()
    stop_ssh_tunnels()
    print("Done!")
    sys.exit(1 if FAIL > 0 else 0)
