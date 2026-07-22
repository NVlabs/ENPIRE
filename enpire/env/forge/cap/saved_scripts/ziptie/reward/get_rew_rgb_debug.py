# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Giving ziptie reward (flat mode only!) in realtime.
This script first transports both arms to the moment we start plugging ziptie in,
waits for 5 seconds, and then starts computing reward at the given video fps.
Notice that this reward script only supports flat mode where in the beginning
the ziptie faces upward on the table, instead of lying on one side.

Set ZIPTIE_REWARD_NO_VIS=1 to suppress all vis/{top,right,merged} writes
and keep only the per-frame reward log line.
This script uses RGB-only inputs and no depth camera fetching
"""

import os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import enpire.env.forge.cap.agent.tools._artifact_log as _al
from enpire.env.forge.cap.agent.tools._artifact_log import render_pool_stats
from skill_library.namespace import get_camera_image

# Pull functions AND shared constants from the single source of truth _compute_rew.py
_compute_rew_func = load_module("ziptie/reward/_compute_rew_rgb.py")
get_reward_from_top_right_cam                    = _compute_rew_func["get_reward_from_top_right_cam"]
save_top_right_cam_tiled_async = _compute_rew_func["save_top_right_cam_tiled_async"]
save_top_right_cam_async       = _compute_rew_func["save_top_right_cam_async"]
TOP_STRAP_CROP                 = _compute_rew_func["TOP_STRAP_CROP"]

# Handover-pose move lives in _move_to_rew_pose.py (single source of truth, shared with TRT mirror).
move_to_rew_pose = load_module("ziptie/reward/_move_to_rew_pose.py")["move_to_rew_pose"]

# Disable all vis/* writes when ZIPTIE_REWARD_NO_VIS=1. The reward decision
# is still computed and printed each frame; only the disk artifacts go away.
NO_VIS = os.environ.get("ZIPTIE_REWARD_NO_VIS", "").lower() in ("1", "true", "yes")
if NO_VIS:
    _al.set_artifact_dir(None)
    print("[reward] ZIPTIE_REWARD_NO_VIS=1 — vis/* writes disabled (reward decision only)")

# ---- 1) move arms to the ziptie-handover pose (shared _move_to_rew_pose.py) ----
move_to_rew_pose()

# ---- 2) realtime loop ---------------------------------------------------
FPS = 15.0
DURATION_S = 600.0
print(f"[reward] running at {FPS} Hz for up to {DURATION_S:.0f}s → <log_dir>/vis/{{top,right,merged}}/...")
period  = 1.0 / FPS
t_start = time.time()
from enpire.env.forge.cap.agent.tools._artifact_log import _artifact_dir as _ARTIFACT_DIR  # noqa: E402
_prev_status = "FAIL"
_cap_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ziptie-cap")
while time.time() - t_start < DURATION_S:
    t0 = time.time()
    f_rgb_t = _cap_pool.submit(get_camera_image, "top")
    f_rgb_r = _cap_pool.submit(get_camera_image, "right")
    rgb_t, rgb_r = np.asarray(f_rgb_t.result()), np.asarray(f_rgb_r.result())
    t1 = time.time()
    (raw_t, rwd_t, fut_top, det_t), (raw_r, rwd_r, fut_right, det_r) = get_reward_from_top_right_cam(rgb_t, rgb_r)
    t2 = time.time()
    if _ARTIFACT_DIR is not None and not NO_VIS:
        save_top_right_cam_async(rgb_t, rgb_r, rwd_t, rwd_r, det_t, det_r, _ARTIFACT_DIR)
        save_top_right_cam_tiled_async(fut_top, fut_right, rwd_t, rwd_r, _ARTIFACT_DIR, TOP_STRAP_CROP, top_metrics=det_t)
    t3 = time.time()
    dt = time.time() - t0
    status = "SUCCESS" if (rwd_t == 1 and rwd_r == 1) else "FAIL"
    if status == "SUCCESS" and _prev_status != "SUCCESS":
        subprocess.Popen(["spd-say", "Yeah"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _prev_status = status
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {status}  top raw={raw_t:5.1f} rwd={rwd_t} | right raw={raw_r:5.1f} rwd={rwd_r}  (cam={(t1-t0)*1000:.0f}ms reward={(t2-t1)*1000:.0f}ms save={(t3-t2)*1000:.0f}ms total={dt*1000:.0f}ms)")
    if dt < period:
        time.sleep(period - dt)
_stats = render_pool_stats()
print(f"[reward] loop done after {time.time()-t_start:.0f}s; render pool "
      f"submitted={_stats['submitted']} pending={_stats['pending']} "
      f"dropped={_stats['dropped']} — atexit will drain remaining jobs.")

