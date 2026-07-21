"""Visualization helpers for ziptie reward: per-cam overlays and merged composition."""
import threading
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

from enpire.env.forge.cap.agent.tools._artifact_log import annotate_image, log_masks_multi

# ── Overlay colors — ADJUST THESE to recolor the reward panels ─────────────
# Each value is a color name resolved by cap.agent.tools._artifact_log.
# Available names: red, bright_green, blue, dark_blue, cyan, yellow, pink,
# magenta, orange. (To add a new name, also add it to _LEGEND_COLORS below so
# the merged-view legend swatch matches.)

# Top-camera overlays
TOP_HEAD_MASK_COLOR    = "blue"          # the ziptie head
TOP_ZIPTIE_MASK_COLOR  = "bright_green"  # the ziptie strap
TOP_OVERLAP_MASK_COLOR = "pink"          # ziptie ∩ head overlap

# Right-camera overlays
RIGHT_HEAD_MASK_COLOR   = "blue"   # the ziptie head (mask + its crop/bbox)
RIGHT_STRAP_MASK_COLOR  = "bright_green"  # the strap (mask + crop box + bridge line)
RIGHT_TARGET_BOX_COLOR  = "pink"   # shrunk head bbox the strap must touch (reward zone)

# Name → RGB lookup used to draw the merged-view legend swatches. The overlay
# color macros above must use names that appear here.
_LEGEND_COLORS = {
    "red": (255,60,60), "bright_green": (60,255,60), "cyan": (0,230,230),
    "yellow": (255,230,0), "dark_blue": (30,70,170), "pink": (255,80,180),
    "magenta": (230,0,230), "orange": (255,140,0), "blue": (60,180,255),
}
# Load prompts from the standalone _prompts.py, NOT _compute_rew_rgb.py — the latter loads THIS module
# (via _load_viz_or_none), so loading it back here would form a circular load (-> RecursionError).
_p = load_module("ziptie/reward/_prompts.py")
ZIPTIE_HEAD_NAME_IN_RIGHT_CAM = _p["ZIPTIE_HEAD_NAME_IN_RIGHT_CAM"]
ZIPTIE_STRAP_NAME_IN_RIGHT_CAM = _p["ZIPTIE_STRAP_NAME_IN_RIGHT_CAM"]

def _draw_legend(draw, entries, x_right, y_bottom, font):
    swatch, pad, row_h = 12, 4, 18
    try:
        tw = max(int(font.getlength(lbl)) for lbl, _ in entries)
    except Exception:
        tw = max(len(lbl) * 7 for lbl, _ in entries)
    w = swatch + pad + tw + 2 * pad
    h = row_h * len(entries) + 2 * pad
    x0, y0 = x_right - w - 4, y_bottom - h - 4
    draw.rectangle([(x0, y0), (x0+w, y0+h)], fill=(0,0,0,180) if hasattr(draw, "_image") else (20,20,20))
    for i, (lbl, col) in enumerate(entries):
        ry = y0 + pad + i * row_h
        color = _LEGEND_COLORS.get(col, (200,200,200))
        draw.rectangle([(x0+pad, ry+2), (x0+pad+swatch, ry+2+swatch)], fill=color)
        draw.text((x0+pad+swatch+pad, ry), lbl, fill=(255,255,255), font=font)

_TOP_LEGEND = [
    ("ziptie head", TOP_HEAD_MASK_COLOR), ("ziptie", TOP_ZIPTIE_MASK_COLOR),
    ("ziptie ∩ head", TOP_OVERLAP_MASK_COLOR),
]
_RIGHT_LEGEND = [("ziptie head", RIGHT_HEAD_MASK_COLOR), ("strap", RIGHT_STRAP_MASK_COLOR)]


def render_top(rgb, zipt_mask, head_mask, inter_mask, title, top_crop):
    """Render top-cam overlays; returns path to combined panel for merged/."""
    log_masks_multi(rgb, [(head_mask, TOP_HEAD_MASK_COLOR)],
                    query="head_only", tag="top", subdir="top",
                    extra_text=f"top head_only  {title}",
                    crop_box=top_crop, legend=[("ziptie head", TOP_HEAD_MASK_COLOR)])
    return log_masks_multi(rgb,
                           [(zipt_mask, TOP_ZIPTIE_MASK_COLOR), (head_mask, TOP_HEAD_MASK_COLOR),
                            (inter_mask, TOP_OVERLAP_MASK_COLOR)],
                           query="", tag="top", subdir="top",
                           marker_mask=inter_mask, marker_scale=0.33,
                           extra_text="", legend=_TOP_LEGEND)


def render_right(rgb, head_mask, head_bbox, strap_mask, shrunk_bbox,
                 bridge_line, title, head_crop, strap_crop):
    """Render right-cam overlays; returns path to combined panel for merged/."""
    p_head = log_masks_multi(rgb, [(head_mask, RIGHT_HEAD_MASK_COLOR)],
                             query="head", tag="right_head", subdir="right",
                             extra_text=f"head  {title}",
                             legend=[(f"head ({ZIPTIE_HEAD_NAME_IN_RIGHT_CAM})", RIGHT_HEAD_MASK_COLOR)])
    p_strap = log_masks_multi(rgb, [(strap_mask, RIGHT_STRAP_MASK_COLOR)],
                              query="strap", tag="right_strap", subdir="right",
                              extra_text=f"strap  {title}",
                              legend=[(f"strap ({ZIPTIE_STRAP_NAME_IN_RIGHT_CAM})", RIGHT_STRAP_MASK_COLOR)])
    head_bboxes = [(*head_crop, RIGHT_HEAD_MASK_COLOR, "dashed")]
    if head_bbox is not None:
        head_bboxes += [(head_bbox[0], head_bbox[1], head_bbox[2], head_bbox[3], RIGHT_HEAD_MASK_COLOR, "dashed"),
                        (*shrunk_bbox, RIGHT_TARGET_BOX_COLOR, "solid")]
    annotate_image(p_head, bboxes=head_bboxes)
    strap_bboxes = [(*strap_crop, RIGHT_STRAP_MASK_COLOR, "dashed")]
    if shrunk_bbox is not None:
        strap_bboxes.append((*shrunk_bbox, RIGHT_TARGET_BOX_COLOR, "solid"))
    strap_lines = [(*bridge_line, RIGHT_STRAP_MASK_COLOR, "solid", 4)] if bridge_line is not None else []
    annotate_image(p_strap, bboxes=strap_bboxes, lines=strap_lines)
    p_combined = log_masks_multi(rgb,
                                 [(head_mask, RIGHT_HEAD_MASK_COLOR), (strap_mask, RIGHT_STRAP_MASK_COLOR)],
                                 query="", tag="right_combined", subdir="right",
                                 extra_text="", legend=None)
    if strap_lines:
        annotate_image(p_combined, lines=strap_lines)
    return p_combined


def save_top_right_cam_tiled(top_path, right_path, rwd_t, rwd_r, out_dir,
                             top_crop_xywh=None, top_legend=None, right_legend=None,
                             top_metrics=None):
    """Stitch the top-cam and right-cam reward panels side-by-side into one PNG.

    Synchronous worker — opens both per-cam PNG paths, crops the top panel by
    `top_crop_xywh`, resizes it to match the right panel's height, composes a
    side-by-side canvas, draws optional per-side legends and the SUCCESS/FAIL
    labels + overall REWARD=0/1 banner, and writes a single timestamped PNG
    to `<out_dir>/<HHMMSS_mmm>_merged.png`. Returns the output path, or None
    if either input path is None.

    Takes ~50–100 ms per frame (PIL composition + disk I/O), so the reward
    loop calls `save_top_right_cam_tiled_async` instead — see that function
    for the threaded wrapper.
    """
    if top_path is None or right_path is None:
        return None
    top_img   = Image.open(top_path).convert("RGB")
    right_img = Image.open(right_path).convert("RGB")
    if top_crop_xywh is not None:
        cx, cy, cw, ch = top_crop_xywh
        top_img = top_img.crop((cx, cy, cx + cw, cy + ch))
    top_scaled = top_img.resize(right_img.size, Image.BILINEAR)

    label_h = 84
    H, Wt, Wr = right_img.height, top_scaled.width, right_img.width
    canvas = Image.new("RGB", (Wt + Wr, H + label_h), (0, 0, 0))
    canvas.paste(top_scaled, (0, 0))
    canvas.paste(right_img,  (Wt, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font_leg   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_big   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except Exception:
        font_leg = font_small = font_big = ImageFont.load_default()

    if top_legend:
        _draw_legend(draw, top_legend, Wt, H, font_leg)
    if right_legend:
        _draw_legend(draw, right_legend, Wt + Wr, H, font_leg)

    # Top-cam tuning readout: actual ratio vs threshold for the two SUCCESS gates,
    # drawn top-left of the top panel so you can watch the numbers cross while
    # tuning TOP_CAM_INTERSECTION_FRACTION / TOP_CAM_STRAP_PROTRUDING_AREA_OVER_HEAD_AREA.
    # Each row is green when its gate passes, red when it fails.
    if top_metrics:
        rows = []
        a, t = top_metrics.get("inter_frac"), top_metrics.get("inter_thr")
        if a is not None and t is not None:
            rows.append((f"overlap  {a:.2f} / {t:.2f}", a >= t))
        a, t = top_metrics.get("protrude_mult"), top_metrics.get("protrude_thr")
        if a is not None and t is not None:
            rows.append((f"protrude {a:.2f} / {t:.2f}", a > t))
        if rows:
            pad, lh = 5, 17
            bw = max(int(font_leg.getlength(s)) for s, _ in rows) + 2 * pad
            bh = lh * len(rows) + 2 * pad
            draw.rectangle([(4, 4), (4 + bw, 4 + bh)], fill=(0, 0, 0))
            for i, (s, ok) in enumerate(rows):
                draw.text((4 + pad, 4 + pad + i * lh), s,
                          fill=(60, 220, 60) if ok else (235, 80, 80), font=font_leg)

    for rwd, x_off, w_sub, cam in [(rwd_t, 0, Wt, "TOP CAM"), (rwd_r, Wt, Wr, "RIGHT CAM")]:
        txt   = f"{cam}: {'SUCCESS' if rwd == 1 else 'FAIL'}"
        color = (60, 220, 60) if rwd == 1 else (220, 50, 50)
        try:   tw = int(font_small.getlength(txt))
        except Exception: tw = len(txt) * 13
        draw.text((x_off + (w_sub - tw) // 2, H + 6), txt, fill=color, font=font_small)

    ok      = rwd_t == 1 and rwd_r == 1
    big_txt = "REWARD=1" if ok else "REWARD=0"
    big_col = (60, 220, 60) if ok else (220, 50, 50)
    try:   bw = int(font_big.getlength(big_txt))
    except Exception: bw = len(big_txt) * 20
    draw.text(((Wt + Wr - bw) // 2, H + 38), big_txt, fill=big_col, font=font_big)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now().strftime('%H%M%S_%f')[:-3]}_merged.png"
    canvas.save(out_path)
    return out_path


def save_top_right_cam_tiled_async(fut_top, fut_right, rwd_t, rwd_r,
                                   artifact_dir, top_crop_xywh, top_metrics=None):
    """Fire-and-forget wrapper around `save_top_right_cam_tiled`.

    Spawns a daemon thread that:
      1. awaits `fut_top` and `fut_right` (10 s timeout each — these are the
         futures returned by `reward_top`/`reward_right` for the per-cam
         render jobs);
      2. calls `save_top_right_cam_tiled` with the resolved paths, using the
         module-level `_TOP_LEGEND` / `_RIGHT_LEGEND` so callers don't need
         to repeat the legend literals;
      3. writes one merged PNG to `<artifact_dir>/merged/`.

    Returns immediately so the per-frame reward loop doesn't block on PIL.
    Exceptions in the worker (timeouts, missing paths) are silently dropped
    to keep the loop running — failures show up as missing files, not crashes.
    """
    out_dir = artifact_dir / "merged"
    def _do_merge(ft=fut_top, fr=fut_right, wt=rwd_t, wr=rwd_r):
        try:
            top_p = ft.result(timeout=10.0); right_p = fr.result(timeout=10.0)
        except Exception:
            return
        save_top_right_cam_tiled(top_p, right_p, wt, wr, out_dir,
                                 top_crop_xywh=top_crop_xywh,
                                 top_legend=_TOP_LEGEND, right_legend=_RIGHT_LEGEND,
                                 top_metrics=top_metrics)
    threading.Thread(target=_do_merge, daemon=True).start()


def save_top_right_cam(rgb_t, rgb_r, rwd_t, rwd_r, det_t, det_r, artifact_dir):
    """Save unprocessed top + right RGB frames plus per-frame reward / detection metadata.

    Synchronous worker. Writes one PNG per camera per frame:
      - <artifact_dir>/top/<HHMMSS_mmm>_top_raw_masks.png
      - <artifact_dir>/right/<HHMMSS_mmm>_right_raw_masks.png
    and appends one JSONL row per frame with the boolean reward + detection
    summary dicts to <artifact_dir>/rewards.jsonl.

    Intended for offline debugging — the file pile-up makes this unsuitable
    for long production runs. For the reward loop, prefer
    `save_top_right_cam_async` (same args, threaded wrapper).
    """
    import json as _json
    ts = datetime.now().strftime('%H%M%S_%f')[:-3]
    from pathlib import Path as _Path
    d = _Path(artifact_dir)
    for rgb, sub, tag in [(rgb_t, "top", "top"), (rgb_r, "right", "right")]:
        out = d / sub; out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(out / f"{ts}_{tag}_raw_masks.png")
    with (d / "rewards.jsonl").open("a") as f:
        f.write(_json.dumps({"rwd_t": rwd_t, "rwd_r": rwd_r, "det_t": det_t, "det_r": det_r}) + "\n")


def save_top_right_cam_async(rgb_t, rgb_r, rwd_t, rwd_r, det_t, det_r, artifact_dir):
    """Fire-and-forget wrapper around `save_top_right_cam`.

    Spawns a daemon thread that calls the sync worker, so the per-frame
    reward loop returns immediately. Mirrors the sync/async pairing of
    `save_top_right_cam_tiled` / `save_top_right_cam_tiled_async`.
    """
    threading.Thread(
        target=save_top_right_cam,
        args=(rgb_t, rgb_r, rwd_t, rwd_r, det_t, det_r, artifact_dir),
        daemon=True,
    ).start()

