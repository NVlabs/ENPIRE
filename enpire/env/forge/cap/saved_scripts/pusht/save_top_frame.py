# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save one top-camera RGB frame for PushT supervisor artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

out_path = os.environ.get("PUSHT_TOP_FRAME_PATH", "").strip()
if not out_path:
    raise RuntimeError("PUSHT_TOP_FRAME_PATH is required")

rgb = get_camera_image("top")
if rgb is None:
    raise RuntimeError("No top camera image returned")
arr = np.asarray(rgb)
if arr.ndim != 3 or arr.shape[2] < 3:
    raise RuntimeError(f"Unexpected top camera image shape: {arr.shape}")
if arr.dtype != np.uint8:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
arr = arr[..., :3]

path = Path(out_path).expanduser()
path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
print(f"[pusht_save_top_frame] saved {path}")

