# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection tools for BundleSDF-based object localization and tracking.

This module also exports the SAM3 segment + filter + top-1 helpers that
serve_sam3_debug.py implements client-side in JS; the Python counterparts
here are the canonical reusable form for other scripts (e.g.
cap/saved_scripts/ziptie/reward_twocams.py) and for the LLM agent via the
:class:`Sam3SegmentFilterTool` wrapper below.

Public API for the SAM3 pipeline:
    sam3_segment(rgb, text, *, threshold, crop_xywh, url) -> list[dict]
    sam3_filter_dets(dets, metric, op, value, value_max=None) -> list[dict]
    sam3_select_top1(dets, metric, direction='max') -> dict | None
    Sam3SegmentFilterTool                              -- LLM-callable Tool
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import urllib.request
from typing import Any, Callable

import numpy as np
import requests
from PIL import Image

# orjson is ~3-5x faster than stdlib json at encoding/decoding the SAM3
# payloads (~100-300 KB of base64 image + mask data). Falls back to stdlib
# json if not installed — orjson is in [project.dependencies] but the
# fallback keeps cap/agent tests running outside the full uv env.
try:
    import orjson  # type: ignore
    def _dumps(obj) -> bytes: return orjson.dumps(obj)
    def _loads(buf): return orjson.loads(buf)
except ImportError:  # pragma: no cover
    def _dumps(obj) -> bytes: return json.dumps(obj).encode()
    def _loads(buf): return json.loads(buf)

from enpire.env.forge.cap.agent.tools._artifact_log import log_detection
from enpire.env.forge.cap.agent.tools.base import Detection3D, Tool, ToolParameter, ToolResult
from enpire.env.forge.cap.config import DETECTION_SERVER_PORT, make_bundlesdf_name

# Default SAM3 server endpoint. Resolved at call time inside the sam3_segment_*
# helpers (not captured into their `url=` defaults), so a downstream module can
# monkey-patch this attribute to redirect every SAM3 call to a different server
# (e.g. the TRT-backed variant on port 6868) without touching the callsite. See
# cap/saved_scripts/ziptie/reward/_compute_rew_rgb_trt.py for the live example.
SAM3_URL = "http://localhost:6767"
_SAM3_URL_DEFAULT = object()  # sentinel — tells the helpers to late-bind SAM3_URL


# ===========================================================================
# SAM3 segment + filter + top-1 — reusable across scripts and the LLM agent.
# Vocabulary matches serve_sam3_debug.py's JS: confidence | C | AR | area | P.
# ===========================================================================


def _sam3_post(rgb: np.ndarray, text: str, threshold: float, url: str) -> dict:
    buf = io.BytesIO()
    np.save(buf, rgb)
    payload = _dumps(
        {
            "text": text,
            "image_b64": base64.b64encode(buf.getvalue()).decode(),
            "score_threshold": float(threshold),
        }
    )
    req = urllib.request.Request(
        f"{url}/segment_all",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    return _loads(urllib.request.urlopen(req, timeout=30).read())


def _sam3_post_multi(
    rgb: np.ndarray, texts: list[str], threshold: float, url: str,
    max_per_prompt: int | None = None,
) -> dict:
    """Multi-prompt counterpart to :func:`_sam3_post`. POSTs ``/segment_multi``
    so the SAM3 server can amortize the image encoder cost across all
    ``texts``. Server returns ``{"results": [{"text", "detections", "count"}],
    "height", "width"}``. ``max_per_prompt`` (optional) caps each prompt's
    returned detections to top-K by score — saves the server from
    serializing masks the caller will throw away.
    """
    buf = io.BytesIO()
    np.save(buf, rgb)
    body = {
        "texts": list(texts),
        "image_b64": base64.b64encode(buf.getvalue()).decode(),
        "score_threshold": float(threshold),
    }
    if max_per_prompt is not None:
        body["max_per_prompt"] = int(max_per_prompt)
    req = urllib.request.Request(
        f"{url}/segment_multi",
        data=_dumps(body),
        headers={"Content-Type": "application/json"},
    )
    return _loads(urllib.request.urlopen(req, timeout=30).read())


def _sam3_post_multi_image(
    items: list[dict], threshold: float, url: str,
    max_per_prompt: int | None = None,
    image_size: int | None = None,
) -> dict:
    """Multi-image counterpart to :func:`_sam3_post_multi`. POSTs
    ``/segment_multi_image`` with N (image, prompts) pairs. Server batches
    the image encoder across every image — one forward pass instead of N —
    and replies with per-image detection lists for each prompt. Same
    ``max_per_prompt`` top-K cap as :func:`_sam3_post_multi``.

    ``image_size`` (optional, int) overrides the SAM3 processor's resize
    canvas for this request — passing 640 makes the encoder process
    ~2.5x fewer pixels than the default 1008. Per-request; doesn't affect
    other CAP consumers.

    ``items`` is a list of ``{"image_b64": str, "texts": list[str]}`` dicts.
    Response shape: ``{"items": [{"text_results": [{"text", "detections",
    "count"}], "height", "width"}]}``.
    """
    body = {"items": items, "score_threshold": float(threshold)}
    if max_per_prompt is not None:
        body["max_per_prompt"] = int(max_per_prompt)
    if image_size is not None:
        body["image_size"] = int(image_size)
    # Optional client-side breakdown: encode body → HTTP roundtrip → decode
    # response. Enable with SAM3_CLIENT_TRACE=1 when you need to know whether
    # a slow /segment_multi_image is server-side queueing (urlopen step) or
    # client-side GIL contention slowing JSON/base64 (dumps/loads steps).
    import os as _os
    import time as _time
    _trace = _os.environ.get("SAM3_CLIENT_TRACE", "").lower() in ("1", "true", "yes")
    _t0 = _time.perf_counter() if _trace else 0.0
    data = _dumps(body)
    _t1 = _time.perf_counter() if _trace else 0.0
    req = urllib.request.Request(
        f"{url}/segment_multi_image",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    raw = urllib.request.urlopen(req, timeout=30).read()
    _t2 = _time.perf_counter() if _trace else 0.0
    result = _loads(raw)
    _t3 = _time.perf_counter() if _trace else 0.0
    if _trace:
        print(
            f"[sam3-client] dumps={1000*(_t1-_t0):4.0f}ms  "
            f"http={1000*(_t2-_t1):4.0f}ms  "
            f"loads={1000*(_t3-_t2):4.0f}ms  "
            f"bytes_out={len(data):>7d}  bytes_in={len(raw):>7d}",
            flush=True,
        )
    return result


def _sam3_geom(mask: np.ndarray) -> dict:
    """Per-mask geometric metrics matching _compute_centroids in the debug UI:
    area, perimeter_px (zero-padded so masks touching the frame edge still
    count properly), compactness = 4πA/P², centroid uv, bbox_xywh."""
    m = mask.astype(np.int8)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return {
            "area": 0,
            "perimeter_px": 0,
            "compactness": None,
            "uv": None,
            "bbox_xywh": [0, 0, 0, 0],
        }
    mp = np.pad(m, 1, constant_values=0)
    P = int(
        np.abs(np.diff(mp[:, 1:-1], axis=0)).sum()
        + np.abs(np.diff(mp[1:-1, :], axis=1)).sum()
    )
    A = int(ys.size)
    return {
        "area": A,
        "perimeter_px": P,
        "compactness": float(4.0 * np.pi * A / (P * P)) if P > 0 else None,
        "uv": [float(xs.mean()), float(ys.mean())],
        "bbox_xywh": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ],
    }


def sam3_segment(
    rgb: np.ndarray,
    text: str,
    *,
    threshold: float = 0.1,
    crop_xywh: tuple[int, int, int, int] | None = None,
    depth: np.ndarray | None = None,
    intrinsics: dict | None = None,
    url: str | object = _SAM3_URL_DEFAULT,
) -> list[dict]:
    """Call SAM3 ``/segment_all``. If *crop_xywh* is given (full-frame px),
    slice the RGB before sending and lift every returned mask back into a
    full-HxW bool array so downstream filter/select operates in image coords.

    When *depth* is provided (and optionally *intrinsics*), each returned
    detection also gets ``depth_m`` (median depth under the mask, in metres)
    and ``xyz_cam`` (camera-frame xyz) populated, so callers can pass
    ``"depth"`` to :func:`sam3_filter_dets` or :func:`sam3_select_top1`.

    Returns: list of dicts ``{score, mask (bool HxW), area, perimeter_px,
    compactness, uv, bbox_xywh}`` plus ``depth_m`` / ``xyz_cam`` when depth
    is supplied."""
    if url is _SAM3_URL_DEFAULT: url = SAM3_URL  # late-bind so monkey-patches take effect
    H, W = rgb.shape[:2]
    if crop_xywh:
        cx, cy, cw, ch = (int(v) for v in crop_xywh)
        cx, cy = max(0, min(W - 1, cx)), max(0, min(H - 1, cy))
        cw, ch = max(1, min(W - cx, cw)), max(1, min(H - cy, ch))
        rgb_in = rgb[cy : cy + ch, cx : cx + cw]
    else:
        rgb_in = rgb
        cx = cy = ch = cw = 0
    data = _sam3_post(rgb_in, text, threshold, url)
    out: list[dict] = []
    # Pre-build the crop region as a HxW bool ROI mask once. Used at the
    # bottom of the loop to AND every returned mask with the crop ROI so
    # NO pixel can land outside the advertised crop region under any
    # circumstance — including buggy server responses, view/copy issues,
    # or any other bizarre path. Belt-and-suspenders for the operator's
    # report of "I see segmented pieces outside the crop".
    if crop_xywh:
        roi = np.zeros((H, W), dtype=bool)
        roi[cy : cy + ch, cx : cx + cw] = True
    else:
        roi = None
    for d in data.get("detections", []):
        out.append(_unpack_sam3_det(d, H, W, crop_xywh, cx, cy, cw, ch, roi, text))
    if depth is not None:
        _attach_depth_to_dets(out, depth, intrinsics)
    return out


def _unpack_sam3_det(
    d: dict,
    H: int, W: int,
    crop_xywh: tuple[int, int, int, int] | None,
    cx: int, cy: int, cw: int, ch: int,
    roi: np.ndarray | None,
    text: str,
) -> dict:
    """Decode one server-side detection dict into our canonical client dict.

    Lifts the server's cropped mask back into a full-HxW bool array, applies
    the hard ROI clip when ``crop_xywh`` is supplied, and adds geometric
    metrics. Factored out of :func:`sam3_segment` so the multi-prompt path
    (:func:`sam3_segment_multi`) can reuse the exact same lift-back contract.
    """
    m_small = np.load(io.BytesIO(base64.b64decode(d["mask_b64"]))) > 0
    if crop_xywh:
        mask = np.zeros((H, W), dtype=bool)
        # Guard: SAM3 should return a mask matching the input image's shape
        # (ch, cw). If the shapes disagree (e.g. server resized), clip / pad
        # to fit so mask pixels can never land outside the advertised crop
        # box. Mismatch is logged so the upstream issue is visible instead of
        # being silently corrected.
        if m_small.shape[:2] != (ch, cw):
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "sam3_segment: mask shape %s != crop (%d, %d) for text=%r — clipping",
                m_small.shape[:2], ch, cw, text,
            )
            mh = min(m_small.shape[0], ch)
            mw = min(m_small.shape[1], cw)
            mask[cy : cy + mh, cx : cx + mw] = m_small[:mh, :mw]
        else:
            mask[cy : cy + ch, cx : cx + cw] = m_small
        if roi is not None:
            mask &= roi  # hard ROI clip — nothing outside survives.
    else:
        mask = m_small.astype(bool)
    return {"score": float(d["score"]), "mask": mask, **_sam3_geom(mask)}


def sam3_segment_multi(
    rgb: np.ndarray,
    texts: list[str],
    *,
    threshold: float = 0.1,
    crop_xywh: tuple[int, int, int, int] | None = None,
    depth: np.ndarray | None = None,
    intrinsics: dict | None = None,
    max_per_prompt: int | None = None,
    url: str | object = _SAM3_URL_DEFAULT,
) -> list[list[dict]]:
    """Batched multi-prompt counterpart to :func:`sam3_segment`.

    Sends one HTTP request to the SAM3 server's ``/segment_multi`` endpoint
    so the image encoder runs ONCE for all ``texts``. The per-prompt result
    contract matches :func:`sam3_segment` exactly: each inner list contains
    detection dicts ``{score, mask (bool HxW), area, perimeter_px,
    compactness, uv, bbox_xywh}`` (plus ``depth_m`` / ``xyz_cam`` when
    *depth* is supplied), so callers can pipe each element through
    :func:`sam3_filter_dets` / :func:`sam3_select_top1` unchanged.

    Parameters
    ----------
    rgb : np.ndarray
        Full-frame HxWx3 RGB image. Cropped before sending if ``crop_xywh``
        is supplied.
    texts : list[str]
        N text prompts. Server returns one detection list per prompt, in
        the same order. An empty inner list means that prompt yielded no
        detections above ``threshold``.
    threshold : float, default 0.1
        Per-prompt score floor passed through to SAM3.
    crop_xywh : (x, y, w, h), optional
        If given, slice the RGB before sending and lift every returned mask
        back into the full HxW frame, with a hard ROI clip applied. The same
        crop is used for every prompt — pick the smallest bounding box that
        covers all your queries' regions of interest.
    depth, intrinsics : optional
        Same semantics as in :func:`sam3_segment`. Depth attach happens
        once per prompt-result-list after lift-back.
    url : str
        SAM3 server base URL (defaults to ``SAM3_URL``).

    Returns
    -------
    list[list[dict]]
        ``out[i]`` is the detection list for ``texts[i]``.

    Backwards compatibility
    -----------------------
    The single-prompt :func:`sam3_segment` is untouched and continues to hit
    ``/segment_all``. Callers can adopt this multi-prompt path incrementally
    where they currently fire several SAM3 queries against the same image.
    """
    if url is _SAM3_URL_DEFAULT: url = SAM3_URL  # late-bind so monkey-patches take effect
    H, W = rgb.shape[:2]
    if crop_xywh:
        cx, cy, cw, ch = (int(v) for v in crop_xywh)
        cx, cy = max(0, min(W - 1, cx)), max(0, min(H - 1, cy))
        cw, ch = max(1, min(W - cx, cw)), max(1, min(H - cy, ch))
        rgb_in = rgb[cy : cy + ch, cx : cx + cw]
        roi = np.zeros((H, W), dtype=bool)
        roi[cy : cy + ch, cx : cx + cw] = True
    else:
        rgb_in = rgb
        cx = cy = ch = cw = 0
        roi = None

    data = _sam3_post_multi(rgb_in, list(texts), threshold, url, max_per_prompt=max_per_prompt)
    server_results = data.get("results", [])
    # Server preserves prompt order, but we look up by text defensively in case
    # a future server revision returns them out of order.
    by_text: dict[str, list[dict]] = {r["text"]: r.get("detections", []) for r in server_results}

    out: list[list[dict]] = []
    for text in texts:
        dets_raw = by_text.get(text, [])
        dets = [
            _unpack_sam3_det(d, H, W, crop_xywh, cx, cy, cw, ch, roi, text)
            for d in dets_raw
        ]
        if depth is not None:
            _attach_depth_to_dets(dets, depth, intrinsics)
        out.append(dets)
    return out


def sam3_segment_multi_image(
    items: list[dict],
    *,
    threshold: float = 0.1,
    max_per_prompt: int | None = None,
    image_size: int | None = None,
    url: str | object = _SAM3_URL_DEFAULT,
) -> list[list[list[dict]]]:
    """Multi-image, multi-prompt SAM3 fused into ONE server call.

    Use this when you want to run several prompts against multiple distinct
    images in the same tick. The SAM3 server batches the image encoder across
    every image (one batched forward pass), then runs the detection head per
    (image, prompt) pair — wall time stays near a single batched encoder
    forward instead of growing linearly with the number of images.

    Parameters
    ----------
    items : list[dict]
        One entry per image. Each dict supports the same per-image options
        the single-image API would accept:

        - ``rgb`` (np.ndarray, required): full-frame RGB image.
        - ``texts`` (list[str], required): N prompts to run against this image.
        - ``crop_xywh`` (tuple, optional): same semantics as in
          :func:`sam3_segment` — the rgb is sliced to this xywh before
          sending and every returned mask is lifted back into the full
          ``rgb.shape[:2]`` frame with a hard ROI clip.
        - ``depth`` / ``intrinsics`` (optional): if supplied, each detection
          for this image gets ``depth_m`` / ``xyz_cam`` populated.
    threshold : float, default 0.1
        Score floor; shared across all prompts in the batch (per-image
        thresholds would require a server-protocol change for a marginal
        gain — easy to add later if needed).
    url : str
        SAM3 server base URL (defaults to :data:`SAM3_URL`).

    Returns
    -------
    list[list[list[dict]]]
        ``out[i][j]`` is the detection list for ``items[i]`` prompted with
        ``items[i]["texts"][j]``, in client-canonical form
        (``{score, mask, area, perimeter_px, compactness, uv, bbox_xywh}``,
        plus ``depth_m`` / ``xyz_cam`` when depth is provided). Empty inner
        lists mean that prompt yielded no detections above ``threshold``.

    Backwards compatibility
    -----------------------
    Neither :func:`sam3_segment` nor :func:`sam3_segment_multi` are touched;
    this is purely additive. Callers can adopt the multi-image path
    incrementally where they currently fire several multi-prompt calls
    against different images in the same tick (e.g. multi-camera reward
    loops).
    """
    if url is _SAM3_URL_DEFAULT: url = SAM3_URL  # late-bind so monkey-patches take effect
    if not items:
        return []

    # Crop each image up front and stash the lift-back state needed to map
    # masks back into full-frame coords. One dict per item — cleaner than
    # the half-dozen parallel arrays this otherwise wants to grow into.
    prepared: list[dict] = []
    payload_items: list[dict] = []
    for it in items:
        rgb = it["rgb"]
        texts = list(it["texts"])
        crop = it.get("crop_xywh")
        H, W = rgb.shape[:2]
        if crop:
            cx, cy, cw, ch = (int(v) for v in crop)
            cx, cy = max(0, min(W - 1, cx)), max(0, min(H - 1, cy))
            cw, ch = max(1, min(W - cx, cw)), max(1, min(H - cy, ch))
            rgb_in = rgb[cy : cy + ch, cx : cx + cw]
            roi = np.zeros((H, W), dtype=bool)
            roi[cy : cy + ch, cx : cx + cw] = True
        else:
            rgb_in = rgb
            cx = cy = ch = cw = 0
            roi = None
        prepared.append({
            "H": H, "W": W,
            "crop": crop, "cxcwch": (cx, cy, cw, ch), "roi": roi,
            "texts": texts,
            "depth": it.get("depth"),
            "intrinsics": it.get("intrinsics"),
        })
        buf = io.BytesIO()
        np.save(buf, rgb_in)
        payload_items.append({
            "image_b64": base64.b64encode(buf.getvalue()).decode(),
            "texts": texts,
        })

    data = _sam3_post_multi_image(
        payload_items, threshold, url,
        max_per_prompt=max_per_prompt, image_size=image_size,
    )
    server_items = data.get("items", [])

    out: list[list[list[dict]]] = []
    for prep, item_result in zip(prepared, server_items):
        H, W = prep["H"], prep["W"]
        crop = prep["crop"]
        cx, cy, cw, ch = prep["cxcwch"]
        roi = prep["roi"]
        depth = prep["depth"]
        intrinsics = prep["intrinsics"]
        # Defensive lookup by text in case the server returns prompts out of
        # order — current server preserves order, but the contract holds either way.
        by_text: dict[str, list[dict]] = {
            r["text"]: r.get("detections", [])
            for r in item_result.get("text_results", [])
        }
        per_prompt_out: list[list[dict]] = []
        for text in prep["texts"]:
            dets_raw = by_text.get(text, [])
            dets = [
                _unpack_sam3_det(d, H, W, crop, cx, cy, cw, ch, roi, text)
                for d in dets_raw
            ]
            if depth is not None:
                _attach_depth_to_dets(dets, depth, intrinsics)
            per_prompt_out.append(dets)
        out.append(per_prompt_out)
    return out


def _sam3_val(d: dict, m: str):
    """Same metric vocabulary as the debug UI's metricValue() function."""
    if m in ("score", "confidence"):
        return d.get("score")
    if m == "C":
        return d.get("compactness")
    if m == "area":
        return d.get("area")
    if m == "P":
        return d.get("perimeter_px")
    if m == "AR":
        _, _, w, h = d["bbox_xywh"]
        return w / h if h > 0 else None
    if m == "depth":
        return d.get("depth_m")
    return None


def sam3_mask_depth_m(mask: np.ndarray | None, depth: np.ndarray | None) -> float | None:
    """Median depth in metres under a bool mask.

    Returns None if depth is missing / not 2D / size <= 1 / no valid
    (finite, positive) depth pixel falls under the mask. Handles depth at
    a different resolution than the mask (common on RealSense — RGB at
    one resolution, depth at another) by nearest-neighbour scaling the
    mask coords into depth's coord system before sampling.

    Reusable from any saved script — replaces ad-hoc inline median-depth
    helpers that were duplicating this logic.
    """
    if mask is None or depth is None or depth.ndim != 2 or depth.size <= 1:
        return None
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape[:2]
    depth_h, depth_w = depth.shape[:2]
    sx = depth_w / float(w)
    sy = depth_h / float(h)
    dy = np.clip((ys * sy).astype(np.int64), 0, depth_h - 1)
    dx = np.clip((xs * sx).astype(np.int64), 0, depth_w - 1)
    md = depth[dy, dx]
    valid = np.isfinite(md) & (md > 0)
    return float(np.median(md[valid])) if valid.any() else None


def _intrinsics_to_fxfycxcy(intrinsics: Any) -> tuple[float, float, float, float] | None:
    """Coerce *intrinsics* into ``(fx, fy, cx, cy)`` regardless of whether the
    caller passed a {"fx","fy","cx","cy", ...} dict or a 4-element [fx, fy,
    cx, cy] list/tuple/ndarray. Returns None if it can't extract a usable
    set (also if any value is non-finite or fx/fy <= 0). Used by
    _attach_depth_to_dets — kept tolerant of the various shapes the camera
    helpers return through different transports."""
    if intrinsics is None:
        return None
    if isinstance(intrinsics, dict):
        try:
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            cx = float(intrinsics["cx"])
            cy = float(intrinsics["cy"])
        except (KeyError, TypeError, ValueError):
            return None
    else:
        try:
            arr = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if arr.size < 4:
            return None
        fx, fy, cx, cy = (float(arr[i]) for i in range(4))
    if not (np.isfinite([fx, fy, cx, cy]).all() and fx > 0 and fy > 0):
        return None
    return fx, fy, cx, cy


def _attach_depth_to_dets(
    dets: list[dict],
    depth: np.ndarray | None,
    intrinsics: Any,
) -> None:
    """Populate each det['depth_m'] (median depth under mask, metres) and
    det['xyz_cam'] (camera-frame xyz from intrinsics + depth) IN-PLACE.

    Handles the common RealSense shape mismatch (depth at a different
    resolution than rgb) by scaling the mask coords into depth's coord
    system at sample time (nearest-neighbour). Detections whose mask has
    no valid (finite, positive) depth pixel get depth_m / xyz_cam set to
    None.
    """
    if depth is None or depth.ndim != 2 or depth.size <= 1:
        for d in dets:
            d.setdefault("depth_m", None)
            d.setdefault("xyz_cam", None)
        return
    depth_h, depth_w = depth.shape[:2]
    intr = _intrinsics_to_fxfycxcy(intrinsics)
    fx = fy = cx = cy = None
    if intr is not None:
        fx, fy, cx, cy = intr
    for d in dets:
        d["depth_m"] = None
        d["xyz_cam"] = None
        m = d.get("mask")
        if m is None:
            continue
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        # Scale mask coords (rgb space) into depth's coord system.
        h, w = m.shape[:2]
        sx = depth_w / float(w)
        sy = depth_h / float(h)
        dy = np.clip((ys * sy).astype(np.int64), 0, depth_h - 1)
        dx = np.clip((xs * sx).astype(np.int64), 0, depth_w - 1)
        mask_depth = depth[dy, dx]
        valid = np.isfinite(mask_depth) & (mask_depth > 0)
        if not valid.any():
            continue
        z = float(np.median(mask_depth[valid]))
        d["depth_m"] = z
        if fx and fy and cx is not None and cy is not None and d.get("uv"):
            u, v = float(d["uv"][0]), float(d["uv"][1])
            d["xyz_cam"] = [(u - cx) * z / fx, (v - cy) * z / fy, z]


def sam3_filter_dets(
    dets: list[dict],
    metric: str,
    op: str,
    value: float,
    value_max: float | None = None,
) -> list[dict]:
    """Filter *dets* by *metric* using *op* ∈ {'>=', '<=', 'between'}. For
    'between' the bounds are inclusive (value ≤ v ≤ value_max)."""
    out = []
    for d in dets:
        v = _sam3_val(d, metric)
        if v is None or not np.isfinite(v):
            continue
        if op == ">=" and v >= value:
            out.append(d)
        elif op == "<=" and v <= value:
            out.append(d)
        elif op == "between" and value <= v <= (value_max if value_max is not None else value):
            out.append(d)
    return out


def sam3_select_top1(
    dets: list[dict], metric: str, direction: str = "max"
) -> dict | None:
    """Return the detection that maximises (or minimises) *metric*. Skips
    detections that don't define the metric. None if none qualify."""
    best, best_v = None, (float("-inf") if direction == "max" else float("inf"))
    for d in dets:
        v = _sam3_val(d, metric)
        if v is None or not np.isfinite(v):
            continue
        if (direction == "max" and v > best_v) or (direction == "min" and v < best_v):
            best, best_v = d, v
    return best


def _encode_rgb(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_depth(depth: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, depth.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _jsonify_extrinsics(extrinsics: dict[str, Any]) -> dict[str, Any]:
    R = np.asarray(extrinsics["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(extrinsics["position"], dtype=np.float64).reshape(-1)

    # Pre-apply optical flip so extrinsics are sent in OpenCV convention.
    # Remote BundleSDF servers may have stale needs_optical_flip() logic that
    # doesn't know about sim envs. By pre-flipping here, we ensure the
    # extrinsics arrive in OpenCV convention regardless.
    if extrinsics.get("needs_optical_flip", False):
        R = R @ np.diag([-1.0, -1.0, 1.0])

    return {
        "position": [float(x) for x in t.tolist()],
        "rotation": [float(x) for x in R.reshape(-1).tolist()],
        "needs_optical_flip": False,  # already flipped
    }


class DetectObjectTool(Tool):
    """Detect objects in a camera image via BundleSDF or simulation oracle.

    Supported backends:
      - ``bundlesdf``: 6-DOF pose tracking via BundleSDF. The first call
        starts a tracking session; subsequent calls poll the latest pose.
      - ``oracle``: ground-truth pose from simulation (sim-only).
    """

    name = "detect_object"
    description = (
        "Detect objects matching a text query. "
        "Returns Detection3D results with world-frame position and optional 6-DOF pose. "
        "Use backend='bundlesdf' for tracked 6-DOF pose, or backend='oracle' for "
        "ground-truth pose from simulation (sim-only)."
    )
    parameters = [
        ToolParameter("query", "str", "Text query describing the object to find."),
        ToolParameter(
            "camera", "str", "Camera to use: 'top', 'left', or 'right'.",
            required=False, default="top",
        ),
        ToolParameter(
            "backend", "str",
            "'bundlesdf' (default, 6-DOF pose tracking) or 'oracle' (ground-truth from sim, sim-only).",
            required=False, default="bundlesdf",
        ),
        ToolParameter(
            "max_retries", "int",
            "Number of retry attempts if detection fails (default 3).",
            required=False, default="3",
        ),
    ]

    def __init__(
        self,
        detection_host: str = "localhost",
        detection_port: int = DETECTION_SERVER_PORT,
        cap_server_host: str = "localhost",
        cap_server_port: int | None = None,
        bundlesdf_host: str = "localhost",
        bundlesdf_port: int | None = None,
        timeout: float = 30.0,
        env=None,
    ):
        self._env = env
        self._detection_host = detection_host
        self._detection_port = detection_port
        self._cap_server_host = cap_server_host
        self._cap_server_port = cap_server_port
        self._timeout = timeout
        self._portal_client = None

        # BundleSDF state
        from enpire.env.forge.cap.config import BUNDLESDF_SERVER_PORT
        bsdf_port = bundlesdf_port or BUNDLESDF_SERVER_PORT
        self._bundlesdf_url = f"http://{bundlesdf_host}:{bsdf_port}"
        self._bundlesdf_sessions: dict[str, str] = {}  # name_key → camera
        self._bundlesdf_active_query: str | None = None
        self._bundlesdf_active_camera: str | None = None

    _KNOWN_HALF_EXTENTS = {
        "red block": [0.025, 0.025, 0.06],
        "red_block": [0.025, 0.025, 0.06],
        "red stick": [0.025, 0.025, 0.06],
        "red_stick": [0.025, 0.025, 0.06],
        "green block": [0.05, 0.05, 0.005],
        "green_block": [0.05, 0.05, 0.005],
        "blue plate": [0.05, 0.05, 0.005],
        "blue_plate": [0.05, 0.05, 0.005],
    }

    def _capture_snapshot(self, camera: str) -> dict[str, Any]:
        if self._env is not None:
            rgb_raw = self._env.render_rgb(camera)
            depth_raw = self._env.render_depth(camera)
            intrinsics_raw = self._env.get_camera_intrinsics(camera)
            extrinsics = self._env.get_camera_extrinsics(camera)
        else:
            client = self._get_portal_client()
            rgb_raw = client.get_camera_image(camera).result()
            depth_raw = client.get_camera_depth(camera).result()
            intrinsics_raw = client.get_camera_intrinsics(camera).result()
            extrinsics = client.get_camera_extrinsics(camera).result()
        rgb = np.asarray(rgb_raw)
        depth = np.asarray(depth_raw)
        if rgb.size < 100:
            raise ValueError(f"No image returned for camera {camera!r}")
        if depth.size < 100:
            raise ValueError(f"No depth returned for camera {camera!r}")
        intrinsics = [float(x) for x in intrinsics_raw]
        return {
            "rgb": rgb,
            "depth": depth.astype(np.float32),
            "intrinsics": intrinsics,
            "extrinsics": _jsonify_extrinsics(extrinsics),
        }

    @staticmethod
    def _bbox_xywh_to_xyxy(bbox: list[float] | None) -> list[float]:
        if bbox and len(bbox) == 4:
            return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        return []

    def _bundlesdf_payload_to_detection(self, query: str, data: dict[str, Any]) -> Detection3D:
        he = data.get("half_extents") or self._KNOWN_HALF_EXTENTS.get(query, [])
        return Detection3D(
            label=query,
            score=data.get("score", 0.0),
            box_2d=self._bbox_xywh_to_xyxy(data.get("bbox")),
            position_3d=data["position_3d"],
            quaternion_xyzw=data.get("quaternion_xyzw", []),
            rpy=data.get("rpy", []),
            half_extents=he,
            vis_b64=data.get("vis_b64"),
        )

    def _execute_bundlesdf_single_frame(
        self,
        *,
        query: str,
        camera: str = "top",
        snapshot: dict[str, Any] | None = None,
    ) -> ToolResult:
        try:
            snap = snapshot or self._capture_snapshot(camera)
            payload = {
                "text": query,
                "camera": camera,
                "image_base64": _encode_rgb(snap["rgb"]),
                "depth_base64": _encode_depth(snap["depth"]),
                "intrinsics": snap["intrinsics"],
                "extrinsics": snap["extrinsics"],
            }
            resp = requests.post(
                f"{self._bundlesdf_url}/single_frame_pose",
                json=payload,
                timeout=max(self._timeout, 120.0),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("position_3d") is None:
                score = float(data.get("score", 0.0))
                bbox = data.get("bbox")
                return ToolResult(
                    success=False,
                    error=(
                        f"BundleSDF single-frame pose unavailable for {query!r}. "
                        f"score={score:.3f}, bbox={bbox}"
                    ),
                )
            det = self._bundlesdf_payload_to_detection(query, data)

            # Save annotated snapshot to log
            try:
                rgb = snap["rgb"]
                log_detection(rgb, [det], tag="detect_oneshot")
            except Exception:
                pass  # best-effort

            return ToolResult(success=True, data=[det])
        except requests.ConnectionError:
            return ToolResult(
                success=False,
                error=f"Cannot reach BundleSDF server at {self._bundlesdf_url}",
            )
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            return ToolResult(success=False, error=detail or str(e))
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    def _get_portal_client(self):
        if self._portal_client is None:
            import portal

            from enpire.env.forge.cap.config import CAP_SERVER_PORT

            port = self._cap_server_port or CAP_SERVER_PORT
            self._portal_client = portal.Client(f"{self._cap_server_host}:{port}")
        return self._portal_client

    @staticmethod
    def _cam_to_world(
        pos_cam: list[float],
        cam_pos: np.ndarray,
        cam_rot: np.ndarray,
    ) -> list[float]:
        """Transform a point from camera frame to world frame.

        RealSense camera convention: +x right, +y down, +z forward.
        The physical camera mount flips left/right relative to the world
        frame, so we negate x before applying the rotation.
        """
        p = np.asarray(pos_cam, dtype=np.float64)
        p[0] = -p[0]  # flip left/right to match world frame
        p[1] = -p[1]  # flip up/down to match world frame
        world_p = cam_pos + cam_rot @ p
        return [round(float(x), 4) for x in world_p]

    def execute(self, **kwargs: Any) -> ToolResult:
        import logging
        logger = logging.getLogger(__name__)

        backend: str = kwargs.get("backend", "bundlesdf")
        max_retries: int = int(kwargs.get("max_retries", 3))

        for attempt in range(1, max_retries + 1):
            if backend == "bundlesdf":
                result = self._execute_bundlesdf(**kwargs)
            elif backend == "oracle":
                result = self._execute_oracle(**kwargs)
            else:
                result = ToolResult(
                    success=False,
                    error=f"Unsupported detection backend: {backend}. Use 'bundlesdf' or 'oracle'.",
                )

            if result.success and result.data:
                return result

            if attempt < max_retries:
                logger.info(
                    f"[detect_object] Attempt {attempt}/{max_retries} failed: "
                    f"{result.error or 'no detections'}. Retrying in 1s..."
                )
                time.sleep(1.0)

        return result

    # -- Oracle backend (sim ground truth) -------------------------------------

    def _execute_oracle(self, **kwargs: Any) -> ToolResult:
        """Return the ground-truth pose of the queried object from simulation.

        Fuzzy-matches *query* against MuJoCo scene body names (case-insensitive
        substring match).  Only works when the cap_server is backed by a
        simulation (SimBackend).
        """
        query: str = kwargs["query"]
        try:
            client = self._get_portal_client()
            result = client.get_object_positions().result()
            if not result.get("ok"):
                return ToolResult(success=False, error="get_object_positions failed (is sim running?)")

            objects: dict[str, dict] = result["objects"]
            if not objects:
                return ToolResult(success=False, error="No scene objects found in simulation")

            # Fuzzy match: case-insensitive substring
            query_lower = query.lower()
            matches: list[tuple[str, dict]] = [
                (name, data)
                for name, data in objects.items()
                if query_lower in name.lower() or name.lower() in query_lower
            ]

            if not matches:
                available = ", ".join(objects.keys())
                return ToolResult(
                    success=False,
                    error=f"No object matching '{query}'. Available: {available}",
                )

            detections: list[Detection3D] = []
            for name, data in matches:
                pos = data["pos"]  # [x, y, z]
                quat_wxyz = data["quat"]  # MuJoCo convention: [w, x, y, z]
                quat_xyzw = quat_wxyz[1:] + quat_wxyz[:1]  # -> [x, y, z, w]
                size = data.get("size", [])  # geom half-extents from MuJoCo
                detections.append(
                    Detection3D(
                        label=name,
                        score=1.0,
                        box_2d=[],
                        position_3d=[round(float(x), 4) for x in pos],
                        quaternion_xyzw=[round(float(x), 4) for x in quat_xyzw],
                        half_extents=[round(float(x), 4) for x in size],
                    )
                )

            # Save camera snapshot with detections to log
            try:
                camera = kwargs.get("camera", "top")
                rgb = self._capture_snapshot(camera)["rgb"]
                log_detection(rgb, detections, tag="detect_oracle")
            except Exception:
                pass  # best-effort

            return ToolResult(success=True, data=detections)

        except Exception as e:
            return ToolResult(success=False, error=f"Oracle backend error: {e}")

    # -- BundleSDF backend -----------------------------------------------------

    # BundleSDF cold-start takes 30-60s (model loading + first SAM3 inference).
    # Subsequent calls with the same query are fast (~1s).
    _BUNDLESDF_POLL_INTERVAL = 2.0  # seconds between pose polls
    _BUNDLESDF_POLL_TIMEOUT = 60.0  # max wait for pose after start_tracking

    def _execute_bundlesdf(self, **kwargs: Any) -> ToolResult:
        import logging
        import urllib.parse
        logger = logging.getLogger(__name__)

        query: str = kwargs["query"]
        camera: str = kwargs.get("camera", "top")
        name = make_bundlesdf_name(query)
        encoded_name = urllib.parse.quote(name, safe="")

        try:
            # Auto-start if new query or camera changed
            cold_start = False
            if self._bundlesdf_active_query != query or self._bundlesdf_active_camera != camera:
                cold_start = True
                if self._bundlesdf_active_query is not None:
                    old_name = make_bundlesdf_name(self._bundlesdf_active_query)
                    logger.info("[bundlesdf] Stopping previous tracking session...")
                    requests.post(
                        f"{self._bundlesdf_url}/end_detection"
                        f"/{urllib.parse.quote(old_name, safe='')}",
                        timeout=30,
                    )

                logger.info(
                    f"[bundlesdf] Starting tracking for '{query}' on camera '{camera}' "
                    f"(cold start — model loading may take 30-60s)..."
                )
                resp = requests.post(
                    f"{self._bundlesdf_url}/add_detection",
                    json={"text": query, "camera": camera},
                    timeout=120,
                )
                if resp.status_code not in (200, 204, 409):
                    resp.raise_for_status()
                start_data = resp.json()
                logger.info(
                    f"[bundlesdf] Tracking started — "
                    f"bbox={start_data.get('bbox')}, "
                    f"first_score={start_data.get('first_score', '?')}"
                )
                self._bundlesdf_active_query = query
                self._bundlesdf_active_camera = camera

            # Poll pose with retries — BundleSDF needs time to converge
            timeout = self._BUNDLESDF_POLL_TIMEOUT if cold_start else 10.0
            interval = self._BUNDLESDF_POLL_INTERVAL
            elapsed = 0.0
            data = None

            while elapsed < timeout:
                resp = requests.get(
                    f"{self._bundlesdf_url}/get_detection/{encoded_name}",
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()

                tracking = data.get("tracking", False)
                has_pose = data.get("position_3d") is not None
                frame_idx = data.get("frame_idx", 0)
                score = data.get("score", 0.0)

                has_valid_depth = data.get("ob_in_cam") is not None

                if tracking and has_pose and has_valid_depth:
                    logger.info(
                        f"[bundlesdf] Pose ready — frame={frame_idx}, "
                        f"score={score:.3f}, elapsed={elapsed:.1f}s"
                    )
                    break

                if tracking and has_pose and not has_valid_depth:
                    logger.warning(
                        f"[bundlesdf] Depth invalid — SAM3 tracks the object "
                        f"(score={score:.3f}, frame={frame_idx}) but the depth "
                        f"sensor returned no valid data in the masked region. "
                        f"Position would be unreliable."
                    )

                logger.info(
                    f"[bundlesdf] Waiting for pose... "
                    f"tracking={tracking}, has_pose={has_pose}, "
                    f"frame={frame_idx}, score={score:.3f}, "
                    f"elapsed={elapsed:.1f}/{timeout:.0f}s"
                )
                time.sleep(interval)
                elapsed += interval
            else:
                self._bundlesdf_active_query = None  # reset so next call retries
                tracking = data.get("tracking", False) if data else False
                score = data.get("score", 0) if data else 0
                frame_idx = data.get("frame_idx", 0) if data else 0
                has_ob = (data.get("ob_in_cam") is not None) if data else False

                if tracking and score > 0.3 and not has_ob:
                    err = (
                        f"BundleSDF: object visually tracked (score={score:.3f}, "
                        f"frame={frame_idx}) but depth sensor returned no valid data "
                        f"in the masked region after {timeout:.0f}s. The object surface "
                        f"may be too reflective/shiny for the depth sensor. "
                        f"Try a different object or move it closer to the camera."
                    )
                else:
                    err = (
                        f"BundleSDF pose not available after {timeout:.0f}s. "
                        f"Last state: tracking={tracking}, "
                        f"frame={frame_idx}, score={score:.3f}"
                    )
                return ToolResult(success=False, error=err)

            # Convert bbox [x, y, w, h] to [x1, y1, x2, y2] for box_2d
            bbox = data.get("bbox")
            if bbox and len(bbox) == 4:
                box_2d = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            else:
                box_2d = []

            det = self._bundlesdf_payload_to_detection(query, data)
            det.box_2d = box_2d

            # Save annotated camera snapshot to log
            try:
                snap = self._capture_snapshot(camera)
                log_detection(snap["rgb"], [det], tag="detect_bundlesdf")
            except Exception:
                pass  # best-effort

            return ToolResult(success=True, data=[det])

        except requests.ConnectionError:
            return ToolResult(
                success=False,
                error=f"Cannot reach BundleSDF server at {self._bundlesdf_url}",
            )
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            return ToolResult(success=False, error=detail or str(e))
        except Exception as e:
            return ToolResult(success=False, error=str(e))

class DetectObjectsOneshotTool(Tool):
    """One-shot BundleSDF pose inference on a shared snapshot.

    This tool never starts a real-time tracking session. It captures one RGB-D
    snapshot, then runs BundleSDF single-frame pose estimation for one or more
    text queries against that same snapshot.
    """

    name = "detect_objects_oneshot"
    description = (
        "One-shot BundleSDF pose inference without real-time tracking. "
        "Accepts either a single query string or a list of query strings. "
        "Captures one shared RGB-D snapshot and returns a dict mapping each query "
        "to its Detection3D results. Best for quasistatic scenes and batch-style "
        "multi-object perception."
    )
    parameters = [
        ToolParameter(
            "query",
            "str | list[str]",
            "Single text query or list of text queries to localize in one shared snapshot.",
        ),
        ToolParameter(
            "camera", "str", "Camera to use: 'top', 'left', or 'right'.",
            required=False, default="top",
        ),
        ToolParameter(
            "max_retries", "int",
            "Number of retry attempts using one shared snapshot per attempt (default 3).",
            required=False, default="3",
        ),
    ]

    def __init__(
        self,
        detect_tool: DetectObjectTool | None = None,
        *,
        env=None,
        detection_host: str = "localhost",
        detection_port: int = DETECTION_SERVER_PORT,
        cap_server_host: str = "localhost",
        cap_server_port: int | None = None,
        bundlesdf_host: str = "localhost",
        bundlesdf_port: int | None = None,
        timeout: float = 30.0,
    ):
        self._detect_tool = detect_tool or DetectObjectTool(
            env=env,
            detection_host=detection_host,
            detection_port=detection_port,
            cap_server_host=cap_server_host,
            cap_server_port=cap_server_port,
            bundlesdf_host=bundlesdf_host,
            bundlesdf_port=bundlesdf_port,
            timeout=timeout,
        )

    @staticmethod
    def _normalize_queries(raw: Any) -> list[str]:
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, (list, tuple)) and all(isinstance(q, str) for q in raw):
            return [q for q in raw if q.strip()]
        raise ValueError("query must be a string or a list of strings")

    def execute(self, **kwargs: Any) -> ToolResult:
        camera = kwargs.get("camera", "top")
        max_retries = max(1, int(kwargs.get("max_retries", 3)))
        try:
            queries = self._normalize_queries(kwargs["query"])
        except Exception as e:
            return ToolResult(success=False, error=str(e))
        if not queries:
            return ToolResult(success=False, error="query list is empty")

        results: dict[str, list[Detection3D]] = {query: [] for query in queries}
        last_errors: dict[str, str] = {}
        pending = list(queries)

        for attempt in range(1, max_retries + 1):
            if not pending:
                break
            try:
                snapshot = self._detect_tool._capture_snapshot(camera)
            except Exception as e:
                return ToolResult(success=False, error=f"Failed to capture shared snapshot: {e}")

            next_pending: list[str] = []
            for query in pending:
                result = self._detect_tool._execute_bundlesdf_single_frame(
                    query=query, camera=camera, snapshot=snapshot
                )
                if result.success and result.data is not None:
                    results[query] = result.data
                else:
                    last_errors[query] = result.error or "unknown error"
                    next_pending.append(query)
            pending = next_pending
            if pending and attempt < max_retries:
                time.sleep(0.1)

        errors = [f"{query!r}: {last_errors[query]}" for query in pending if query in last_errors]
        if errors and not any(results.values()):
            return ToolResult(success=False, error="; ".join(errors))
        return ToolResult(success=True, data=results)


class DetectObjectRealtimeTool(Tool):
    """Continuously detect objects and update visualization until stopped.

    Blocks the executor thread, running detection in a loop. Stops when
    the stop_event is set (by Home/Stop/E-Stop buttons).
    """

    name = "detect_object_realtime"
    description = (
        "Continuously detect objects and update 3D visualization in real-time. "
        "Blocks until stopped by Home/Stop/E-Stop. "
        "Returns the last set of detections when stopped."
    )
    parameters = [
        ToolParameter("query", "str", "Text query describing the object to find."),
        ToolParameter(
            "camera", "str", "Camera to use: 'top', 'left', or 'right'.",
            required=False, default="top",
        ),
    ]

    def __init__(
        self,
        detect_tool: DetectObjectTool,
        stop_event: threading.Event,
        on_detections: Callable[[list[Detection3D], str], None] | None = None,
    ):
        self._detect_tool = detect_tool
        self._stop_event = stop_event
        self._on_detections = on_detections

    def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs["query"]
        camera: str = kwargs.get("camera", "top")

        self._stop_event.clear()
        last_detections: list[Detection3D] = []
        frame_count = 0

        print(f"[DetectRealtime] Starting continuous detection: query='{query}', camera={camera}")
        print("[DetectRealtime] Press Home/Stop/E-Stop to end")

        while not self._stop_event.is_set():
            result = self._detect_tool.execute(query=query, camera=camera)
            if result.success and result.data:
                last_detections = result.data
                frame_count += 1

                # Notify callback (updates Viser + UI bounding boxes)
                if self._on_detections is not None:
                    self._on_detections(last_detections, camera)

            # Small sleep to avoid hammering detection server
            # (detection inference takes ~100-500ms anyway)
            time.sleep(0.05)

        print(f"[DetectRealtime] Stopped after {frame_count} frames")
        return ToolResult(success=True, data=last_detections)


class Sam3SegmentFilterTool(Tool):
    """Single-shot SAM3 segment + crop + filter + top-1 — LLM-callable wrapper
    around :func:`sam3_segment` / :func:`sam3_filter_dets` / :func:`sam3_select_top1`.

    Captures one RGB snapshot from *camera* (via the wrapped
    :class:`DetectObjectTool` so env/portal selection follows the existing
    transport), runs SAM3 with optional crop, applies an optional filter, and
    optionally returns just the top-1 detection.

    Output: list of detection dicts ``{score, mask, area, perimeter_px,
    compactness, uv, bbox_xywh}``. When select_metric is set, the list has 0
    or 1 entries.
    """

    name = "sam3_segment_filter"
    description = (
        "Run SAM3 segment_all on a camera image with optional crop_xywh + filter "
        "(metric op value [value_max for 'between']) + top-1 select (metric, max/min). "
        "Metric vocabulary matches the debug UI: confidence | C | AR | area | P. "
        "Returns list of detection dicts including the full-frame mask."
    )
    parameters = [
        ToolParameter("text", "str", "SAM3 text prompt."),
        ToolParameter(
            "camera", "str", "Camera to use: 'top', 'left', or 'right'.",
            required=False, default="top",
        ),
        ToolParameter(
            "threshold", "float", "SAM3 score threshold.",
            required=False, default=0.1,
        ),
        ToolParameter(
            "crop_xywh", "list",
            "Optional [x, y, w, h] in full-frame pixel coords. RGB is sliced to this "
            "region before SAM3 sees it; returned masks are lifted back to full-frame.",
            required=False, default=None,
        ),
        ToolParameter(
            "filter_metric", "str",
            "Optional filter metric: confidence | C | AR | area | P.",
            required=False, default=None,
        ),
        ToolParameter(
            "filter_op", "str",
            "Optional filter op: '>=' | '<=' | 'between'.",
            required=False, default=None,
        ),
        ToolParameter(
            "filter_value", "float", "Filter threshold (or lower bound for 'between').",
            required=False, default=None,
        ),
        ToolParameter(
            "filter_value_max", "float", "Upper bound for 'between' op.",
            required=False, default=None,
        ),
        ToolParameter(
            "select_metric", "str",
            "Optional top-1 metric: confidence | C | AR | area | P. When set, "
            "result is the single best detection (or empty if none qualify).",
            required=False, default=None,
        ),
        ToolParameter(
            "select_direction", "str", "Top-1 direction: 'max' | 'min'.",
            required=False, default="max",
        ),
    ]

    def __init__(
        self,
        detect_tool: DetectObjectTool | None = None,
        *,
        env=None,
        cap_server_host: str = "localhost",
        cap_server_port: int | None = None,
        sam3_host: str = "localhost",
        sam3_port: int = 6767,
    ):
        self._detect_tool = detect_tool or DetectObjectTool(
            env=env,
            cap_server_host=cap_server_host,
            cap_server_port=cap_server_port,
        )
        self._sam3_url = f"http://{sam3_host}:{sam3_port}"

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            camera = kwargs.get("camera", "top")
            rgb = self._detect_tool._capture_snapshot(camera)["rgb"]
            crop = kwargs.get("crop_xywh")
            dets = sam3_segment(
                rgb,
                kwargs["text"],
                threshold=float(kwargs.get("threshold", 0.1)),
                crop_xywh=tuple(crop) if crop else None,
                url=self._sam3_url,
            )
            fm = kwargs.get("filter_metric")
            if fm:
                dets = sam3_filter_dets(
                    dets,
                    fm,
                    kwargs["filter_op"],
                    float(kwargs["filter_value"]),
                    (float(kwargs["filter_value_max"])
                     if kwargs.get("filter_value_max") is not None else None),
                )
            sm = kwargs.get("select_metric")
            if sm:
                best = sam3_select_top1(dets, sm, kwargs.get("select_direction", "max"))
                dets = [best] if best is not None else []
            return ToolResult(success=True, data=dets)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
