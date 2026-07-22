# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo: pick up an object in RoboCasa using CAP tools.

Uses the same tool calls the LLM agent would generate:
get_robot_state, freespace_move, set_gripper — all routed through
cap_server to the RoboCasa env's native OSC_POSE controller.

Usage:
    PYTHONPATH=/path/to/robosuite:/path/to/robocasa:$PYTHONPATH \
        CAP_AGENT_NAME=demo uv run python -u cap/server/demo_robocasa_pick_tools.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

from enpire.env.forge.cap.server.cap_server import CapServer


# ---------------------------------------------------------------------------
# Start cap_server with RoboCasa
# ---------------------------------------------------------------------------

print("Starting CAP server with RoboCasa viewer...", flush=True)
server = CapServer(
    server_port=18500,
    enable_cameras=False,
    env_name="robocasa:PickPlaceCounterToCabinet",
    env_viewer=True,
)
server.start()
time.sleep(2.0)

# ---------------------------------------------------------------------------
# CAP tool calls — same as what the LLM agent would generate
# ---------------------------------------------------------------------------

# get_state
state = server.get_state()
print(f"\n--- get_state() ---")
print(f"  keys: {sorted(state.keys())}")
print(f"  EE pos:  {[round(float(x), 3) for x in state['right_ee_pos']]}")
print(f"  gripper: {state['right_gripper_pos'][0]:.2f}", flush=True)

# Object position (in real CAP, this comes from detect_object or VLM)
obj_pos = server._sim_backend._obs["obj_pos"].copy()
print(f"  object:  {[round(float(x), 3) for x in obj_pos]}", flush=True)

# set_gripper — open
print(f"\n--- set_gripper('right', 1.0) ---", flush=True)
server.set_gripper("right", 1.0)
print(f"  opened", flush=True)

# freespace_move — hover above object
print(f"\n--- freespace_move → hover ---", flush=True)
hover = obj_pos.copy()
hover[2] += 0.10
s = server.get_state()
hover_rpy = list(Rotation.from_quat(s["right_ee_quat_xyzw"]).as_euler('xyz', degrees=True))
result = server.freespace_move(right_target_pos=list(hover), right_target_rpy=hover_rpy, planning_speed=1.5)
print(f"  success: {result['success']}", flush=True)

# freespace_move — lower to object center
print(f"\n--- freespace_move → grasp ---", flush=True)
obj_now = server._sim_backend._obs["obj_pos"].copy()
s2 = server.get_state()
grasp_rpy = list(Rotation.from_quat(s2["right_ee_quat_xyzw"]).as_euler('xyz', degrees=True))
result = server.freespace_move(right_target_pos=list(obj_now), right_target_rpy=grasp_rpy, planning_speed=1.5)
print(f"  success: {result['success']}", flush=True)

# set_gripper — close
print(f"\n--- set_gripper('right', 0.0) ---", flush=True)
server.set_gripper("right", 0.0)
print(f"  closed", flush=True)

# freespace_move — lift
print(f"\n--- freespace_move → lift ---", flush=True)
s3 = server.get_state()
lift = np.array(s3["right_ee_pos"])
lift[2] += 0.20
lift_rpy = list(Rotation.from_quat(s3["right_ee_quat_xyzw"]).as_euler('xyz', degrees=True))
result = server.freespace_move(right_target_pos=list(lift), right_target_rpy=lift_rpy, planning_speed=1.5)
print(f"  success: {result['success']}", flush=True)

# Result
obj_final = server._sim_backend._obs["obj_pos"]
lifted = obj_final[2] - obj_pos[2]
print(f"\n--- Result ---")
print(f"  object lifted: {lifted:.3f}m")
print(f"  {'PICK SUCCESS' if lifted > 0.05 else 'PICK FAILED'}", flush=True)

# Hold for viewing
print(f"\n--- Holding. Close viewer or Ctrl+C to exit. ---", flush=True)
try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    pass

server.stop()
print("Done!")
