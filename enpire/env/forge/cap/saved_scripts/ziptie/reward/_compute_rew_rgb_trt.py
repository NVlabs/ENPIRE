"""TRT-backed variant of _compute_rew_rgb.py with tunable post-SAM3 thresholds.

Two responsibilities:

  1) Route every SAM3 call from this module to the TRT-backed server on :6868
     (the PyTorch SAM3 server stays on :6767, unaffected).

  2) Expose the per-half reward thresholds as keyword arguments. The 640-
     canvas TRT engine produces coarser masks (22 src-px per patch vs 14 at
     1008), shifting strap-∩-head intersections toward 0 on borderline-success
     frames. Dilation + lower thresholds let us recover recall without
     touching the PyTorch reward path (_compute_rew_rgb.py).

Tweak knobs (defaults match base PyTorch reward):
  top_dilate (int=0): dilate strap+head masks before intersection (px)
  top_inter_frac (float=0.8): min strap∩head overlap as a fraction of head area
  top_left_mult (float=3.0): min strap pixels left of head, as a multiple of head area
  right_strap_dilate (int=2): connected-components dilation on right strap
  right_shrunk_frac  (float=2/3): fraction of head bbox used as test region

Call get_reward_from_top_right_cam(rgb_t, rgb_r, top_dilate=2, top_inter_frac=0.6)
etc. to override per-call. Defaults give identical behaviour to the PyTorch
reward (so test_sam3_tensorrt_5090.py with no kwargs keeps producing the
same numbers we baselined).
"""

import enpire.env.forge.cap.agent.tools.detection as _det
_det.SAM3_URL = "http://localhost:6868"

# Re-export the sibling PyTorch reward helpers explicitly. CAP's load_module()
# only executes imports, assignments, functions, and classes; wrapping these
# exports in top-level control flow leaves callers without helper keys.
def _load_base_reward_module():
    try:
        _loader = load_module  # noqa: F821 - CAP-injected
    except Exception:
        _loader = None

    if _loader is not None:
        return _loader("ziptie/reward/_compute_rew_rgb.py")

    import importlib.util as _ilu
    from pathlib import Path as _P

    _spec = _ilu.spec_from_file_location(
        "_compute_rew_rgb", _P(__file__).with_name("_compute_rew_rgb.py")
    )
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    return {k: v for k, v in _m.__dict__.items() if not k.startswith("__")}


_base_reward = _load_base_reward_module()

render_top = _base_reward["render_top"]
render_right = _base_reward["render_right"]
save_top_right_cam_tiled = _base_reward["save_top_right_cam_tiled"]
save_top_right_cam_tiled_async = _base_reward["save_top_right_cam_tiled_async"]
save_top_right_cam = _base_reward["save_top_right_cam"]
save_top_right_cam_async = _base_reward["save_top_right_cam_async"]

ZIPTIE_COLOR = _base_reward["ZIPTIE_COLOR"]
TOP_STRAP_CROP = _base_reward["TOP_STRAP_CROP"]
TOP_HEAD_CROP = _base_reward["TOP_HEAD_CROP"]
TOP_CAM_INTERSECTION_FRACTION = _base_reward["TOP_CAM_INTERSECTION_FRACTION"]
TOP_CAM_STRAP_PROTRUDING_AREA_OVER_HEAD_AREA = _base_reward["TOP_CAM_STRAP_PROTRUDING_AREA_OVER_HEAD_AREA"]
ZIPTIE_STRAP_NAME_IN_TOP_CAM = _base_reward["ZIPTIE_STRAP_NAME_IN_TOP_CAM"]
ZIPTIE_HEAD_NAME_IN_TOP_CAM = _base_reward["ZIPTIE_HEAD_NAME_IN_TOP_CAM"]
RIGHT_HEAD_CROP = _base_reward["RIGHT_HEAD_CROP"]
RIGHT_STRAP_CROP = _base_reward["RIGHT_STRAP_CROP"]
ZIPTIE_HEAD_NAME_IN_RIGHT_CAM = _base_reward["ZIPTIE_HEAD_NAME_IN_RIGHT_CAM"]
ZIPTIE_STRAP_NAME_IN_RIGHT_CAM = _base_reward["ZIPTIE_STRAP_NAME_IN_RIGHT_CAM"]
RIGHT_CAM_UNION_CROP = _base_reward["RIGHT_CAM_UNION_CROP"]
_POOL = _base_reward["_POOL"]

# ── Tunable per-half reward overrides ──────────────────────────────────────
import numpy as _np
from scipy import ndimage as _ndimage_local
from enpire.env.forge.cap.agent.tools.detection import (
    sam3_segment_multi_image as _sam3_segment_multi_image,
    sam3_select_top1 as _sel_top1,
)
from enpire.env.forge.cap.agent.tools._artifact_log import background as _bg
import time as _time


def _get_reward_from_top_cam(rgb, top_dets, *, top_dilate: int = 0,
                             top_inter_frac: float = TOP_CAM_INTERSECTION_FRACTION,
                             top_left_mult: float = TOP_CAM_STRAP_PROTRUDING_AREA_OVER_HEAD_AREA):
    """Top-cam reward with tunable thresholds.

    With default kwargs this is byte-identical to the base _compute_rew_rgb
    version. Setting top_dilate>0 widens both the strap and head masks via
    binary_dilation BEFORE the intersection test — recovers borderline cases
    where 640-canvas coarseness made the masks miss by ~1 patch (~22 px).
    Setting top_inter_frac or top_left_mult lower makes the test more permissive.
    """
    strap_dets, knob_dets, block_dets = top_dets
    head = _sel_top1(block_dets, "score", "max")
    if head is None:
        head = _sel_top1(knob_dets, "score", "max")
    top2 = sorted(strap_dets, key=lambda d: -d.get("score", 0))[:2]
    zipt = {"mask": _np.logical_or.reduce([d["mask"] for d in top2])} if top2 else None

    raw, rwd, inter = 0.0, 0, None
    inter_frac = protrude_mult = None  # actual ratios vs the thresholds (merged-plot tuning readout)
    if zipt is not None and head is not None:
        z = zipt["mask"]
        h = head["mask"]
        if top_dilate > 0:
            z = _ndimage_local.binary_dilation(z, iterations=top_dilate)
            h = _ndimage_local.binary_dilation(h, iterations=top_dilate)
        inter_mask = z & h
        head_area = int(head["mask"].sum())
        head_left_x = int(_np.nonzero(h)[1].min())
        left_area = int(z[:, :head_left_x].sum())
        if head_area > 0:
            inter_frac = float(inter_mask.sum()) / head_area
            protrude_mult = left_area / head_area
        zipt_left_of_head = left_area > top_left_mult * head_area
        if head_area > 0 and inter_mask.sum() >= top_inter_frac * head_area and zipt_left_of_head:
            inter, raw, rwd = inter_mask, float(inter_mask.sum()), 1

    head_det = "detected" if head is not None else "missing"
    title = (f"top raw={raw:.0f}px reward={rwd}  "
             f"(dilate={top_dilate} inter≥{top_inter_frac:.0%}·head left>{top_left_mult:g}·head, head={head_det})")
    _zipt = zipt["mask"] if zipt else None
    _head = head["mask"] if head else None
    # render_top / TOP_HEAD_CROP come from the re-exported base globals.
    fut = _bg(lambda: render_top(rgb, _zipt, _head, inter, title, TOP_HEAD_CROP))  # noqa: F821
    det = {"head": head is not None, "zipt": zipt is not None,
           "inter_frac": inter_frac, "inter_thr": top_inter_frac,
           "protrude_mult": protrude_mult, "protrude_thr": top_left_mult}
    return raw, rwd, fut, det


def _get_reward_from_right_cam(rgb, right_dets, *, right_strap_dilate: int = 2,
                                right_shrunk_frac: float = 2 / 3):
    """Right-cam reward with tunable strap-dilation and shrunk-bbox fraction.

    The connected-components dilation already exists in the base path
    (`iterations=2`); we expose it as a knob. The shrunk_bbox region is
    derived from the head's detected bbox — making it larger admits more
    borderline intersections; smaller is stricter.
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
    head = _sel_top1(head_dets, "score", "max")
    strap = _sel_top1(strap_dets, "score", "max")
    raw, rwd, shrunk_bbox, bridge_line, n_comp = 0.0, 0, None, None, 0
    if head is not None and strap is not None:
        dilated = _ndimage_local.binary_dilation(strap["mask"], iterations=right_strap_dilate)
        labeled, n_comp = _ndimage_local.label(dilated)
        ys_h, xs_h = _np.nonzero(head["mask"])
        cx_h, cy_h = float(xs_h.mean()), float(ys_h.mean())
        _, _, bw, bh = head["bbox_xywh"]
        sw, sh = int(bw * right_shrunk_frac), int(bh * right_shrunk_frac)
        sx, sy = int(cx_h - sw / 2), int(cy_h - sh / 2)
        shrunk_bbox = (sx, sy, sw, sh)
        H_img, W_img = rgb.shape[:2]
        shrunk_mask = _np.zeros((H_img, W_img), bool)
        shrunk_mask[max(0, sy):min(H_img, sy + sh), max(0, sx):min(W_img, sx + sw)] = True
        if n_comp == 1:
            inter = strap["mask"] & shrunk_mask
            if inter.any():
                raw, rwd = float(inter.sum()), 1
        elif n_comp == 2:
            comp_masks = [labeled == i for i in (1, 2)]
            segs = sorted(comp_masks, key=lambda m: float(_np.nonzero(m)[0].mean()))
            ys0, xs0 = _np.nonzero(segs[0])
            ys1, xs1 = _np.nonzero(segs[1])
            top_y = int(ys0.max()); top_x = int(xs0[ys0 == top_y].mean())
            bot_y = int(ys1.min()); bot_x = int(xs1[ys1 == bot_y].mean())
            bridge_line = (top_x, top_y, bot_x, bot_y)
            n_pts = max(abs(bot_x - top_x), abs(bot_y - top_y), 1)
            bridge_mask = _np.zeros((H_img, W_img), bool)
            for t in _np.linspace(0, 1, n_pts * 2 + 1):
                lx = int(round(top_x + t * (bot_x - top_x)))
                ly = int(round(top_y + t * (bot_y - top_y)))
                if 0 <= ly < H_img and 0 <= lx < W_img:
                    bridge_mask[ly, lx] = True
            test_mask = strap["mask"] | bridge_mask
            inter = test_mask & shrunk_mask
            if inter.any():
                raw, rwd = float(inter.sum()), 1

    h_s = "detected" if head is not None else "missing"
    s_s = f"{n_comp} part(s)" if strap is not None else "missing"
    title = (f"right raw={raw:.0f}px reward={rwd}  "
             f"(strap_dilate={right_strap_dilate} shrunk={right_shrunk_frac:.2f} "
             f"head={h_s} strap={s_s})")
    _hm = head["mask"] if head else None
    _hb = head["bbox_xywh"] if head else None
    _sm = strap["mask"] if strap else None
    fut = _bg(lambda: render_right(  # noqa: F821 — re-exported from base
        rgb, _hm, _hb, _sm, shrunk_bbox, bridge_line, title,
        RIGHT_HEAD_CROP, RIGHT_STRAP_CROP,  # noqa: F821
    ))
    det = {"head": head is not None, "strap": strap is not None}
    return raw, rwd, fut, det


def get_reward_from_top_right_cam(
    rgb_t, rgb_r,
    *,
    top_dilate: int = 0,
    top_inter_frac: float = TOP_CAM_INTERSECTION_FRACTION,
    top_left_mult: float = TOP_CAM_STRAP_PROTRUDING_AREA_OVER_HEAD_AREA,
    right_strap_dilate: int = 2,
    right_shrunk_frac: float = 2 / 3,
    sam3_score_threshold: float = 0.05,
    sam3_max_per_prompt: int = 3,
):
    """TRT-backed reward with tunable thresholds.

    Default sam3_score_threshold=0.05 (vs base path's 0.1) is the tuned
    sweet spot from tests/sweep_640_score_threshold.py: on the 640 canvas
    the strap often gets detected with confidence 0.05-0.09, and the
    default 0.1 filter silently drops it → ~14 pp recall loss vs 1008.
    Dropping the bar to 0.05 surfaces those detections (precision stays
    1.000 — no spurious FPs surface in that interval). 0.02 and 0.01 start
    flipping top-1 selection toward wrong dets → recall regresses.

    Threshold=0.05 also benefits 1008/encb2 (small additional recall gain
    on borderline frames where head-detection confidence is 0.06-0.09).
    Same structure as base get_reward_from_top_right_cam: one fused
    /segment_multi_image call, then per-half reductions on the shared pool.
    """
    # Top cam uses TWO crops: wide TOP_STRAP_CROP for the strap, tight
    # TOP_HEAD_CROP for the head — so each goes in its own item (one crop each).
    items = [
        {"rgb": rgb_t,
         "texts": [ZIPTIE_STRAP_NAME_IN_TOP_CAM],   # noqa: F821 — re-exported
         "crop_xywh": TOP_STRAP_CROP},               # noqa: F821
        {"rgb": rgb_t,
         "texts": [ZIPTIE_HEAD_NAME_IN_TOP_CAM[0],   # noqa: F821
                   ZIPTIE_HEAD_NAME_IN_TOP_CAM[1]],  # noqa: F821
         "crop_xywh": TOP_HEAD_CROP},                # noqa: F821
        {"rgb": rgb_r,
         "texts": [ZIPTIE_HEAD_NAME_IN_RIGHT_CAM,    # noqa: F821
                   ZIPTIE_STRAP_NAME_IN_RIGHT_CAM],  # noqa: F821
         "crop_xywh": RIGHT_CAM_UNION_CROP},         # noqa: F821
    ]
    t1 = _time.time()
    top_strap_dets, top_head_dets, right_dets = _sam3_segment_multi_image(
        items, max_per_prompt=sam3_max_per_prompt, threshold=sam3_score_threshold,
    )
    t_sam3 = _time.time() - t1
    # Reassemble [strap_dets, knob_dets, block_dets] for _get_reward_from_top_cam.
    top_dets = [top_strap_dets[0], top_head_dets[0], top_head_dets[1]]

    p = _POOL  # noqa: F821 — re-exported pool from base
    t2 = _time.time()
    f_top = p.submit(
        _get_reward_from_top_cam, rgb_t, top_dets,
        top_dilate=top_dilate, top_inter_frac=top_inter_frac, top_left_mult=top_left_mult,
    )
    f_right = p.submit(
        _get_reward_from_right_cam, rgb_r, right_dets,
        right_strap_dilate=right_strap_dilate, right_shrunk_frac=right_shrunk_frac,
    )
    top_r = f_top.result()
    right_r = f_right.result()
    t_red = _time.time() - t2
    print(
        f"[reward-trace] sam3_multi_image={t_sam3*1000:.0f}ms  reductions={t_red*1000:.0f}ms",
        flush=True,
    )
    return top_r, right_r

