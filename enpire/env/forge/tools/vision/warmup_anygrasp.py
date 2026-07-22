#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send one synthetic AnyGrasp request to warm the first real inference path."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request

import numpy as np


def _np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


def make_payload() -> dict[str, object]:
    h, w = 48, 64
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 1] = 96
    depth = np.full((h, w), 0.55, dtype=np.float32)
    cam_K = np.array(
        [[120.0, 0.0, w / 2.0], [0.0, 120.0, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    segmap = np.zeros((h, w), dtype=np.uint8)
    segmap[12:36, 20:44] = 1
    return {
        "rgb_base64": _np_to_b64(rgb),
        "depth_base64": _np_to_b64(depth),
        "cam_K_base64": _np_to_b64(cam_K),
        "segmap_base64": _np_to_b64(segmap),
        "segmap_id": 1,
        "z_range": [1e-6, 1.5],
        "max_grasps": 5,
        "workspace_margin": 0.02,
        "collision_detection": True,
        "object_input_mode": "segmented_object_cloud",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8122", help="AnyGrasp base URL")
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds for the warmup request",
    )
    args = parser.parse_args()

    url = args.url.rstrip("/") + "/plan_viz"
    req = urllib.request.Request(
        url,
        data=json.dumps(make_payload()).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    dt = time.time() - t0
    print(
        json.dumps(
            {
                "status": "ok",
                "elapsed_s": round(dt, 2),
                "n_grasps": int(body.get("n_grasps", 0) or 0),
                "best_score": body.get("best_score"),
            }
        )
    )


if __name__ == "__main__":
    main()
