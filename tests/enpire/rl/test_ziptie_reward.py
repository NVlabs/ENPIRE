# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_reward_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "enpire/env/forge/cap/saved_scripts/ziptie/reward/_compute_rew_rgb.py"
    )
    spec = importlib.util.spec_from_file_location("enpire_ziptie_reward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.background = lambda _function: None
    return module


def _det(mask: np.ndarray, score: float = 1.0) -> dict:
    ys, xs = np.nonzero(mask)
    return {
        "mask": mask,
        "score": score,
        "area": int(mask.sum()),
        "bbox_xywh": (
            int(xs.min()),
            int(ys.min()),
            int(np.ptp(xs) + 1),
            int(np.ptp(ys) + 1),
        ),
    }


def test_top_reward_requires_overlap_and_left_protrusion() -> None:
    module = _load_reward_module()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    head = np.zeros((64, 64), dtype=bool)
    head[24:34, 35:45] = True
    strap = np.zeros_like(head)
    strap[27:31, 5:45] = True
    _, reward, _, metrics = module._get_reward_from_top_cam(
        image, [[_det(strap)], [], [_det(head)]]
    )
    assert reward == 1
    assert metrics["inter_frac"] >= metrics["inter_thr"]
    assert metrics["protrude_mult"] >= metrics["protrude_thr"]


def test_right_reward_rejects_fragmented_strap() -> None:
    module = _load_reward_module()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    head = np.zeros((64, 64), dtype=bool)
    head[24:34, 24:34] = True
    strap = np.zeros_like(head)
    strap[10:12, 28:30] = True
    strap[27:29, 28:30] = True
    strap[44:46, 28:30] = True
    _, reward, _, _ = module._get_reward_from_right_cam(
        image, [[_det(head)], [_det(strap)]]
    )
    assert reward == 0
