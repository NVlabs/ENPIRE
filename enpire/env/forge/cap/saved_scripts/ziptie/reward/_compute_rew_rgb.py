# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as _ndimage

from enpire.env.forge.cap.agent.tools._artifact_log import background
from enpire.env.forge.cap.agent.tools.detection import (
    sam3_segment_multi_image,
    sam3_select_top1,
)


def _load_viz_or_none():
    """Load the sibling viz helpers via CAP's ``load_module`` injection when
    available, else return ``None`` so plain-Python consumers (e.g. the
    RL runner's auto-reward worker, which only needs the reward decision
    and never spawns overlay renders) can still import this module."""
    try:
        return load_module("ziptie/reward/_visualize_rew.py")
    except NameError:
        return None


def _viz_noop(*_args, **_kwargs):
    """No-op stand-in for overlay-render helpers when running outside the
    CAP runtime. Reward computation is unaffected; only the background
    artifact-render lambdas resolve to this and return ``None``."""
    return None


_viz = _load_viz_or_none()
# Re-export the viz helpers so CAP-runtime consumer scripts can pick them up
# via a single ``load_module`` call to _compute_rew. When _viz is None we hand
# back the no-op stub for every name so attribute access never fails.
render_top                     = _viz["render_top"]                     if _viz else _viz_noop
render_right                   = _viz["render_right"]                   if _viz else _viz_noop
save_top_right_cam_tiled       = _viz["save_top_right_cam_tiled"]       if _viz else _viz_noop
save_top_right_cam_tiled_async = _viz["save_top_right_cam_tiled_async"] if _viz else _viz_noop
save_top_right_cam             = _viz["save_top_right_cam"]             if _viz else _viz_noop
save_top_right_cam_async       = _viz["save_top_right_cam_async"]       if _viz else _viz_noop

# SAM3 prompt strings live in the standalone _prompts.py (single source, also used by
# _visualize_rew.py) — loaded here in both CAP (load_module) and plain-Python (importlib) contexts.
def _load_prompts():
    try:
        _loader = load_module  # noqa: F821 — CAP-injected
    except Exception:
        _loader = None
    if _loader is not None:
        return _loader("ziptie/reward/_prompts.py")
    import importlib.util as _ilu
    from pathlib import Path as _P
    _spec = _ilu.spec_from_file_location("_prompts", _P(__file__).with_name("_prompts.py"))
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    return {k: v for k, v in _m.__dict__.items() if not k.startswith("__")}


_prompts = _load_prompts()
ZIPTIE_COLOR = _prompts["ZIPTIE_COLOR"]
ZIPTIE_STRAP_NAME_IN_TOP_CAM = _prompts["ZIPTIE_STRAP_NAME_IN_TOP_CAM"]
ZIPTIE_HEAD_NAME_IN_TOP_CAM = _prompts["ZIPTIE_HEAD_NAME_IN_TOP_CAM"]
TOP_STRAP_CROP = (203, 7, 192, 81) #(184, 44, 324, 189)   # xywh — wide region the ziptie strap is searched in
TOP_HEAD_CROP  = (255, 24, 35, 25) #(217, 80, 108, 72)
# Top-cam SUCCESS needs the strap∩head overlap to cover at least this fraction of
# the ziptie head's pixel area — scale-invariant, replaces the old fixed 10-px test.
TOP_CAM_INTERSECTION_FRACTION = 0.3
# ...AND the strap must extend left of the head by at least this multiple of the
# head's area (in strap pixels) — scale-invariant, replaces the old fixed 50-px test.
TOP_CAM_STRAP_PROTRUDING_AREA_OVER_HEAD_AREA = 1.0

RIGHT_HEAD_CROP  = (235, 49, 175, 238)
RIGHT_STRAP_CROP = (258, 0, 135, 291)
ZIPTIE_HEAD_NAME_IN_RIGHT_CAM = _prompts["ZIPTIE_HEAD_NAME_IN_RIGHT_CAM"]
ZIPTIE_STRAP_NAME_IN_RIGHT_CAM = _prompts["ZIPTIE_STRAP_NAME_IN_RIGHT_CAM"]


def _xywh_union(*crops):
    """Smallest xywh box covering every input xywh crop."""
    x  = min(c[0] for c in crops)
    y  = min(c[1] for c in crops)
    x2 = max(c[0] + c[2] for c in crops)
    y2 = max(c[1] + c[3] for c in crops)
    return (x, y, x2 - x, y2 - y)

# Single crop for the right-cam multi-prompt SAM3 call. Splitting head/strap
# into separate crops would force two encoder forwards — the union is tight
# enough that SAM3's text-conditioning still picks the right concept.
RIGHT_CAM_UNION_CROP = _xywh_union(RIGHT_HEAD_CROP, RIGHT_STRAP_CROP)

# Module pool reused across frames: 1 SAM3 call + 2 Portal RPCs + reductions.
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ziptie-rew")


def _get_reward_from_top_cam(rgb, top_dets):
    """Top-cam reward half.

    Args:
        rgb: top-cam RGB frame (only used for the background overlay render).
        top_dets: 3-element list ``[strap_dets, knob_dets, block_dets]`` from
            the multi-image SAM3 call. Strap → top-2 union (SAM3 often splits
            a thin strap into 2 fragments). Head → top-1 of block; fall back
            to knob if block returned nothing.

    Returns ``(raw, rwd, fut, det)``: ``raw`` is the strap∩head pixel count,
    ``rwd ∈ {0,1}``, ``fut`` is the background overlay-render future, ``det``
    is ``{"head": bool, "zipt": bool}`` for diagnostic logging.
    """
    strap_dets, knob_dets, block_dets = top_dets
    head = sam3_select_top1(block_dets, "score", "max")
    if head is None:
        head = sam3_select_top1(knob_dets, "score", "max")
    top2 = sorted(strap_dets, key=lambda d: -d.get("score", 0))[:2]
    zipt = {"mask": np.logical_or.reduce([d["mask"] for d in top2])} if top2 else None

    # reward=1 iff ziptie∩head covers ≥ TOP_INTER_HEAD_AREA_FRAC of the head's area
    # AND ziptie extends left of head by ≥ TOP_STRAP_LEFT_HEAD_AREA_MULT × head area.
    raw, rwd, inter = 0.0, 0, None
    inter_frac = protrude_mult = None  # actual ratios vs the thresholds (merged-plot tuning readout)
    if zipt is not None and head is not None:
        inter_mask = zipt["mask"] & head["mask"]
        head_area = int(head["mask"].sum())
        head_left_x = int(np.nonzero(head["mask"])[1].min())
        left_area = int(zipt["mask"][:, :head_left_x].sum())
        if head_area > 0:
            inter_frac = float(inter_mask.sum()) / head_area
            protrude_mult = left_area / head_area
        zipt_left_of_head = left_area > TOP_CAM_STRAP_PROTRUDING_AREA_OVER_HEAD_AREA * head_area
        if head_area > 0 and inter_mask.sum() >= TOP_CAM_INTERSECTION_FRACTION * head_area and zipt_left_of_head:
            inter, raw, rwd = inter_mask, float(inter_mask.sum()), 1
    head_det = "detected" if head is not None else "missing"
    title = f"top raw={raw:.0f}px reward={rwd}  (zipt∩head + zipt-left-of-head, head={head_det})"

    _zipt = zipt["mask"] if zipt else None
    _head = head["mask"] if head else None
    fut = background(lambda: render_top(rgb, _zipt, _head, inter, title, TOP_HEAD_CROP))
    det = {"head": head is not None, "zipt": zipt is not None,
           "inter_frac": inter_frac, "inter_thr": TOP_CAM_INTERSECTION_FRACTION,
           "protrude_mult": protrude_mult, "protrude_thr": TOP_CAM_STRAP_PROTRUDING_AREA_OVER_HEAD_AREA}
    return raw, rwd, fut, det


def _get_reward_from_right_cam(rgb, right_dets):
    """Right-cam reward half (RGB-only).

    Args:
        rgb: right-cam RGB frame (only used for the background overlay render).
        right_dets: 2-element list ``[head_dets, strap_dets]``. Head → top-1.

    Reward logic on the strap's connected components:
      0 dets / 0 components       → reward=0
      1 component (continuous)    → reward=1 iff mask ∩ 2/3-shrunk head bbox
      2 components (split by head)→ bridge the gap and reward=1 iff
                                    union ∩ shrunk bbox is non-empty
      3+ components               → misclassification → reward=0

    Returns ``(raw, rwd, fut, det)`` — same shape as the top-cam half.
    """
    head_dets, strap_dets = right_dets
    # Head filter before top-1: keep only square-ish (aspect ratio >= 0.8) and large-enough
    # (>= 50 px) candidates, so the right-cam head is a real block — not an elongated / tiny
    # sliver of the strap. AR = min(w,h)/max(w,h) of the bbox; P = mask pixel area.
    def _head_ok(d):
        _, _, w, h = d["bbox_xywh"]
        ar = min(w, h) / max(w, h) if max(w, h) else 0.0
        return d.get("area", 0) >= 50 and ar >= 0.8
    head_dets = [d for d in head_dets if _head_ok(d)]
    head  = sam3_select_top1(head_dets,  "score", "max")
    strap = sam3_select_top1(strap_dets, "score", "max")

    raw, rwd, shrunk_bbox, bridge_line, n_comp = 0.0, 0, None, None, 0
    if head is not None and strap is not None:
        dilated = _ndimage.binary_dilation(strap["mask"], iterations=2)
        labeled, n_comp = _ndimage.label(dilated)
        ys_h, xs_h = np.nonzero(head["mask"])
        cx_h, cy_h = float(xs_h.mean()), float(ys_h.mean())
        _, _, bw, bh = head["bbox_xywh"]
        sw, sh = int(bw * 2/3), int(bh * 2/3)
        sx, sy = int(cx_h - sw/2), int(cy_h - sh/2)
        shrunk_bbox = (sx, sy, sw, sh)
        H_img, W_img = rgb.shape[:2]
        shrunk_mask = np.zeros((H_img, W_img), bool)
        shrunk_mask[max(0,sy):min(H_img,sy+sh), max(0,sx):min(W_img,sx+sw)] = True
        if n_comp == 1:
            inter = strap["mask"] & shrunk_mask
            if inter.any():
                raw, rwd = float(inter.sum()), 1
        elif n_comp == 2:
            comp_masks = [labeled == i for i in (1, 2)]
            segs = sorted(comp_masks, key=lambda m: float(np.nonzero(m)[0].mean()))
            ys0, xs0 = np.nonzero(segs[0]); ys1, xs1 = np.nonzero(segs[1])
            top_y = int(ys0.max()); top_x = int(xs0[ys0 == top_y].mean())
            bot_y = int(ys1.min()); bot_x = int(xs1[ys1 == bot_y].mean())
            bridge_line = (top_x, top_y, bot_x, bot_y)
            n_pts = max(abs(bot_x - top_x), abs(bot_y - top_y), 1)
            bridge_mask = np.zeros((H_img, W_img), bool)
            for t in np.linspace(0, 1, n_pts * 2 + 1):
                lx, ly = int(round(top_x + t*(bot_x-top_x))), int(round(top_y + t*(bot_y-top_y)))
                if 0 <= ly < H_img and 0 <= lx < W_img:
                    bridge_mask[ly, lx] = True
            test_mask = strap["mask"] | bridge_mask
            inter = test_mask & shrunk_mask
            if inter.any():
                raw, rwd = float(inter.sum()), 1

    h_s   = "detected" if head  is not None else "missing"
    s_s   = f"{n_comp} part(s)" if strap is not None else "missing"
    title = f"right raw={raw:.0f}px reward={rwd}  (top1-strap {n_comp}-part | head={h_s} strap={s_s})"
    _hm = head["mask"] if head else None; _hb = head["bbox_xywh"] if head else None
    _sm = strap["mask"] if strap else None
    fut = background(lambda: render_right(
        rgb, _hm, _hb, _sm, shrunk_bbox, bridge_line, title, RIGHT_HEAD_CROP, RIGHT_STRAP_CROP))
    det = {"head": head is not None, "strap": strap is not None}
    return raw, rwd, fut, det


def get_reward_from_top_right_cam(rgb_t, rgb_r):
    """Compute top + right reward halves in one fused SAM3 call (RGB only).

    Issues a single ``/segment_multi_image`` request carrying both cropped
    images and all 5 prompts. The server runs ``Sam3Model.vision_encoder``
    ONCE on the batched pixel_values and loops the cheap detection head per
    (image, prompt) pair against the cached embeds.

    Parameters
    ----------
    rgb_t, rgb_r : np.ndarray
        Native-resolution RGB frames for the top and right cameras.

    Returns ``(top_reward, right_reward)`` where each is the 4-tuple from
    the corresponding ``_get_reward_from_*_cam`` helper. Episode reward is
    typically ``1`` iff ``rwd_t == 1 and rwd_r == 1``.
    """
    import time as _t
    p = _POOL

    # Top cam uses TWO crops: a wide one for the strap and a tight one for the
    # head. Each item carries one crop, so the strap and head go in separate
    # items (one extra encoder forward, but far fewer false head detections).
    items = [
        {
            "rgb":       rgb_t,
            "texts":     [ZIPTIE_STRAP_NAME_IN_TOP_CAM],
            "crop_xywh": TOP_STRAP_CROP,
        },
        {
            "rgb":       rgb_t,
            "texts":     [ZIPTIE_HEAD_NAME_IN_TOP_CAM[0],
                          ZIPTIE_HEAD_NAME_IN_TOP_CAM[1]],
            "crop_xywh": TOP_HEAD_CROP,
        },
        {
            "rgb":       rgb_r,
            "texts":     [ZIPTIE_HEAD_NAME_IN_RIGHT_CAM,
                          ZIPTIE_STRAP_NAME_IN_RIGHT_CAM],
            "crop_xywh": RIGHT_CAM_UNION_CROP,
        },
    ]
    # max_per_prompt=3 short-circuits per-mask serialization on the server:
    # the strap uses top-2 (union), every other prompt uses top-1, so K=3
    # leaves headroom without paying for SAM3's dozens of weak candidates.
    t1 = _t.time()
    top_strap_dets, top_head_dets, right_dets = sam3_segment_multi_image(items, max_per_prompt=3)
    t_sam3 = _t.time() - t1
    # Reassemble the [strap_dets, knob_dets, block_dets] shape that
    # _get_reward_from_top_cam expects: strap from the wide crop's lone prompt,
    # knob/block from the tight head crop's two prompts.
    top_dets = [top_strap_dets[0], top_head_dets[0], top_head_dets[1]]

    t2 = _t.time()
    f_top   = p.submit(_get_reward_from_top_cam,   rgb_t, top_dets)
    f_right = p.submit(_get_reward_from_right_cam, rgb_r, right_dets)
    top_r = f_top.result(); right_r = f_right.result()
    t_red = _t.time() - t2
    print(f"[reward-trace] sam3_multi_image={t_sam3*1000:.0f}ms  reductions={t_red*1000:.0f}ms",
          flush=True)
    return top_r, right_r

