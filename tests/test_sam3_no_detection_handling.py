# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


def _image_b64() -> str:
    buf = io.BytesIO()
    np.save(buf, np.zeros((4, 5, 3), dtype=np.uint8))
    return base64.b64encode(buf.getvalue()).decode()


def test_sam3_segment_returns_404_when_no_detection(monkeypatch) -> None:
    from enpire.env.forge.tools.vision import serve_sam3

    monkeypatch.setattr(serve_sam3, "text_to_masks", lambda *args, **kwargs: [])

    client = TestClient(serve_sam3.create_app())
    response = client.post(
        "/segment",
        json={"text": "white bot", "image_b64": _image_b64()},
    )

    assert response.status_code == 404
    assert "white bot" in response.json()["detail"]


def test_sample_anygrasp_returns_empty_list_on_tool_failure(monkeypatch) -> None:
    saved_scripts_dir = Path(__file__).resolve().parents[1] / "cap" / "saved_scripts"
    sys.path.insert(0, str(saved_scripts_dir))
    try:
        from skill_library import grasp_geometry
    finally:
        try:
            sys.path.remove(str(saved_scripts_dir))
        except ValueError:
            pass

    def _fail_anygrasp(**kwargs):
        raise RuntimeError("SAM3 could not find 'white bot'")

    monkeypatch.setattr(
        grasp_geometry,
        "sample_grasp_pose_anygrasp",
        _fail_anygrasp,
        raising=False,
    )

    assert grasp_geometry.sample_anygrasp("white bot") == []
