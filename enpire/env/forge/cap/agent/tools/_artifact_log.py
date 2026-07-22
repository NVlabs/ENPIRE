# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official logging tool: save visual artifacts (annotated images, masks) from tool calls to the run log.

Usage from any tool module::

    from enpire.env.forge.cap.agent.tools._artifact_log import log_detection, log_mask, set_artifact_dir

    # Called once from run_agent.py
    set_artifact_dir(session.run_dir)

    # Called from detection tools after a successful result
    log_detection(rgb, detections, tag="detect_object")
    log_mask(rgb, mask, query="red cup", tag="segment")
"""

from __future__ import annotations

import atexit
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from enpire.env.forge.cap.agent.tools.base import Detection3D

logger = logging.getLogger(__name__)

_artifact_dir: Path | None = None


# ============================================================================
# Background render pool — shared across all artifact callers
# ============================================================================
# Every log_* / annotate_image call PIL-encodes a PNG and writes to disk
# (typically 20-40 ms each). For realtime callers (e.g. the 15 Hz ziptie
# reward loop) that's enough to starve the next frame's inference. Rather
# than have every script re-build its own ThreadPoolExecutor + back-pressure
# queue + atexit drain, expose a single shared pool here and let scripts
# submit their render closures with one call:
#
#     from cap.agent.tools._artifact_log import background, log_masks_multi
#
#     def _render():
#         log_masks_multi(rgb, layers, ...)        # sync inside the worker
#         path = log_masks_multi(rgb, ..., query="combined")
#         annotate_image(path, bboxes=[...])
#         return path
#
#     future = background(_render)   # returns immediately
#
# The pool is lazily created on first use, runs daemon-friendly worker
# threads, and drains automatically at interpreter exit via atexit so
# pending PNGs make it to disk. Back-pressure cancels the OLDEST UNSTARTED
# job when the queue exceeds RENDER_POOL_CAP — drop late renders rather
# than let memory grow without bound.

RENDER_POOL_WORKERS = 3   # parallel worker threads
RENDER_POOL_CAP = 16      # max pending jobs before back-pressure kicks in

_render_pool: dict[str, Any] = {
    "executor": None,   # ThreadPoolExecutor, lazily created
    "pending":  [],     # in-flight + queued futures (FIFO)
    "dropped":  0,      # back-pressure cancel count (since pool init)
    "submitted": 0,     # total jobs submitted (since pool init)
}


def _get_render_executor() -> ThreadPoolExecutor:
    """Lazily create the shared render pool and register the atexit drain."""
    ex = _render_pool["executor"]
    if ex is None:
        ex = ThreadPoolExecutor(
            max_workers=RENDER_POOL_WORKERS,
            thread_name_prefix="artifact-render",
        )
        _render_pool["executor"] = ex
        atexit.register(_atexit_drain)
        logger.debug("Artifact render pool started (%d workers)", RENDER_POOL_WORKERS)
    return ex


def background(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    """Run ``fn(*args, **kwargs)`` on the shared artifact render pool.

    Use to push PNG-encoding / file-writing off the caller's hot path.
    Returns the resulting ``Future`` for chaining; callers that just want
    fire-and-forget can discard it. Drain happens automatically at
    interpreter exit so no explicit shutdown is required.

    Back-pressure: when the queue is at ``RENDER_POOL_CAP``, the OLDEST
    unstarted future is cancelled before submitting the new one. Jobs
    already running cannot be cancelled by ThreadPoolExecutor — those
    drain naturally. The cancel count is surfaced by
    :func:`render_pool_stats` so callers can detect when disk is the
    bottleneck."""
    pend = _render_pool["pending"]
    pend[:] = [f for f in pend if not f.done()]
    while len(pend) >= RENDER_POOL_CAP:
        old = pend.pop(0)
        if old.cancel():
            _render_pool["dropped"] += 1
    fut = _get_render_executor().submit(fn, *args, **kwargs)
    pend.append(fut)
    _render_pool["submitted"] += 1
    return fut


def render_pool_stats() -> dict[str, int]:
    """Snapshot of the render pool: in-flight job count, total submitted,
    and total dropped via back-pressure. Useful for end-of-run logs."""
    pend = _render_pool["pending"]
    return {
        "pending":   sum(1 for f in pend if not f.done()),
        "submitted": _render_pool["submitted"],
        "dropped":   _render_pool["dropped"],
    }


def shutdown_render_pool(wait: bool = True) -> None:
    """Drain (and shut down) the render pool. Idempotent; the atexit hook
    calls this so most callers never need to."""
    ex = _render_pool["executor"]
    if ex is None:
        return
    ex.shutdown(wait=wait)
    _render_pool["executor"] = None


def _atexit_drain() -> None:
    """atexit hook: drain pending render jobs before the interpreter exits.
    Daemon-style behaviour (don't block exit forever) is implicit because
    ThreadPoolExecutor.shutdown(wait=True) only waits for jobs the queue
    already contained — no new submissions happen after main exits."""
    try:
        shutdown_render_pool(wait=True)
    except Exception:
        logger.warning("Render pool atexit drain failed", exc_info=True)


def set_artifact_dir(path: Path | None) -> None:
    """Set the directory for saving visual artifacts. Creates ``vis/`` subdir."""
    global _artifact_dir, _vlm_query_counter
    _vlm_query_counter = 0
    if path is not None:
        vis = Path(path) / "vis"
        vis.mkdir(parents=True, exist_ok=True)
        _artifact_dir = vis
        logger.info("Artifact log dir: %s", vis)
    else:
        _artifact_dir = None


def _stamp() -> str:
    # HHMMSS_FFF — millisecond resolution so realtime callers (e.g. the
    # ziptie reward loop at 15 Hz) don't overwrite same-second files.
    return datetime.now().strftime("%H%M%S_%f")[:-3]


def _vis_subdir(name: str) -> Path | None:
    if _artifact_dir is None:
        return None
    path = _artifact_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_detection(
    rgb: np.ndarray,
    detections: list[Detection3D],
    *,
    tag: str = "detect",
) -> Path | None:
    """Save camera image annotated with detection bboxes and 3D positions."""
    if _artifact_dir is None or rgb is None:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(rgb.copy())
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
            )
        except Exception:
            font = ImageFont.load_default()

        for det in detections:
            label = det.label
            pos = det.position_3d
            score = det.score

            # Draw bbox if available
            if det.box_2d and len(det.box_2d) == 4:
                x1, y1, x2, y2 = [int(v) for v in det.box_2d]
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                text_pos = (x1, max(0, y1 - 16))
            else:
                text_pos = (10, 10)

            text = f"{label} [{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] s={score:.2f}"
            draw.text(text_pos, text, fill="lime", font=font)

        safe_label = (
            detections[0].label.replace(" ", "_")[:20] if detections else "none"
        )
        name = f"{_stamp()}_{tag}_{safe_label}.png"
        out_dir = _vis_subdir("detect")
        if out_dir is None:
            return None
        path = out_dir / name
        img.save(path)
        logger.debug("Saved detection artifact: %s", path)
        return path
    except Exception:
        logger.warning("Failed to save detection artifact", exc_info=True)
        return None


_MASK_OVERLAY_CHANNELS = {"red": 0, "green": 1, "blue": 2}
_MASK_OVERLAY_LABEL_FILL = {"red": "#ff5050", "green": "lime", "blue": "#5599ff"}

# Saturated RGB triplets for log_masks_multi. Each picks a perceptually
# distinct hue so multiple overlays on the same panel stay readable. Used
# instead of the channel-boost scheme in log_mask which can't render
# secondary colors (yellow / cyan / magenta) without bleeding into other
# layers. Pixels under the mask are painted directly to the tuple, then
# alpha-blended with the original RGB.
_LAYER_RGB: dict[str, tuple[int, int, int]] = {
    "red":          (255,  60,  60),
    "dark_red":     (140,  20,  20),
    "green":        ( 60, 255,  60),
    "bright_green": ( 60, 255,  60),
    "blue":         ( 60, 180, 255),
    "dark_blue":    ( 30,  70, 170),
    "yellow":       (255, 230,   0),
    "orange":       (255, 140,   0),
    "cyan":         (  0, 230, 230),
    "magenta":      (230,   0, 230),
    "pink":         (255,  80, 180),
}


def _resolve_color(color):
    """Map a _LAYER_RGB palette name to its RGB triple; otherwise pass
    through (PIL accepts RGB tuples, "#rrggbb" hex, and CSS color names).
    Lets callers say ``color="dark_blue"`` once and have the drawn shape
    match the legend swatch pixel-for-pixel."""
    if isinstance(color, str) and color in _LAYER_RGB:
        return _LAYER_RGB[color]
    return color


def _draw_dashed_rect(draw, x0, y0, x1, y1, *, color, dash=10, gap=6, width=2):
    """Dashed rectangle with corners (x0,y0)-(x1,y1)."""
    fill = _resolve_color(color)
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    for x in range(x0, x1, dash + gap):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=fill, width=width)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=fill, width=width)
    for y in range(y0, y1, dash + gap):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=fill, width=width)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=fill, width=width)


def _draw_dashed_line(draw, x1, y1, x2, y2, *, color, dash=10, gap=6, width=2):
    """Dashed line from (x1,y1) to (x2,y2). Horizontals, verticals, and
    diagonals are all sampled at the same dash/gap rate so a panel of
    mixed guide-lines reads consistently."""
    fill = _resolve_color(color)
    dx = x2 - x1
    dy = y2 - y1
    length = max(abs(dx), abs(dy), 1)
    steps = int(length / (dash + gap)) + 1
    for s in range(steps):
        t0 = s * (dash + gap) / length
        t1 = min(1.0, (s * (dash + gap) + dash) / length)
        if t0 >= 1.0:
            break
        draw.line(
            [
                (int(x1 + dx * t0), int(y1 + dy * t0)),
                (int(x1 + dx * t1), int(y1 + dy * t1)),
            ],
            fill=fill, width=width,
        )


def _draw_cross(draw, cx, cy, *, color="#ff0000", arm=14, width=4):
    """Bold cross at (cx, cy). For marker dots that must read against
    any background colour."""
    fill = _resolve_color(color)
    cx, cy = int(cx), int(cy)
    arm = max(1, int(arm))
    width = max(1, int(width))
    draw.line([(cx - arm, cy), (cx + arm, cy)], fill=fill, width=width)
    draw.line([(cx, cy - arm), (cx, cy + arm)], fill=fill, width=width)


def _normalize_bbox(entry):
    """(x,y,w,h) | (...,color) | (...,color,style) → 6-tuple.

    ``style`` ∈ {"dashed", "solid"}; defaults: blue dashed."""
    color = "blue"
    style = "dashed"
    if len(entry) == 4:
        x, y, w, h = entry
    elif len(entry) == 5:
        x, y, w, h, color = entry
    elif len(entry) == 6:
        x, y, w, h, color, style = entry
    else:
        raise ValueError(f"bbox entry must have 4-6 elements, got {len(entry)}")
    return int(x), int(y), int(w), int(h), color, style


def log_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    query: str = "",
    tag: str = "segment",
    alpha: float = 0.45,
    color: str = "green",
    extra_text: str | None = None,
    camera: str | None = None,
) -> Path | None:
    """Save camera image with mask overlay.

    Args:
        color: ``"red"``, ``"green"`` (default), or ``"blue"`` — which RGB
            channel to boost under the mask. Anything else falls back to
            green so old callers keep their behaviour.
        extra_text: optional second line drawn below ``query`` (e.g. the
            world-xyz of the segmented centroid).
        camera: accepted for compatibility with the standard segmentation
            tool; artifact routing continues to use ``tag``.
    """
    if _artifact_dir is None or rgb is None or mask is None:
        return None
    try:
        overlay = rgb.copy()
        mask_bool = mask.astype(bool)
        ch = _MASK_OVERLAY_CHANNELS.get(color, 1)
        # Default green stays soft (+80) for backward compat with existing
        # callers. Explicit red / blue uses a saturated overlay: +200 on the
        # chosen channel, -80 on the other two, and α bumped to 0.6 so the
        # mask reads as actual red / blue instead of a faint tint.
        strong = color != "green"
        boost = 200 if strong else 80
        overlay[mask_bool, ch] = np.clip(
            overlay[mask_bool, ch].astype(np.float32) + boost, 0, 255
        ).astype(np.uint8)
        if strong:
            for other_ch in (i for i in range(3) if i != ch):
                overlay[mask_bool, other_ch] = np.clip(
                    overlay[mask_bool, other_ch].astype(np.float32) - 80, 0, 255
                ).astype(np.uint8)
            alpha = max(alpha, 0.6)
        blended = (
            rgb.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha
        ).astype(np.uint8)

        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(blended)
        if query or extra_text:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
                )
            except Exception:
                font = ImageFont.load_default()
            label_fill = _MASK_OVERLAY_LABEL_FILL.get(color, "lime")
            if query:
                draw.text((10, 10), query, fill=label_fill, font=font)
            if extra_text:
                # Support multi-line extra_text by drawing each line at an
                # increasing y offset. Tab off the right of the query (one
                # 18-px row below it) regardless of query length.
                for i, line in enumerate(str(extra_text).split("\n")):
                    draw.text((10, 28 + 18 * i), line, fill="white", font=font)

        safe_q = query.replace(" ", "_")[:20] if query else "mask"
        name = f"{_stamp()}_{tag}_{safe_q}.png"
        out_dir = _vis_subdir("segment")
        if out_dir is None:
            return None
        path = out_dir / name
        img.save(path)
        logger.debug("Saved mask artifact: %s", path)
        return path
    except Exception:
        logger.warning("Failed to save mask artifact", exc_info=True)
        return None


def log_masks_multi(
    rgb: np.ndarray,
    layers: list[tuple[np.ndarray | None, str]],
    *,
    query: str = "",
    tag: str = "masks",
    alpha: float = 0.55,
    extra_text: str | None = None,
    marker_mask: np.ndarray | None = None,
    marker_scale: float = 1.0,
    subdir: str = "segment",
    crop_box: tuple[int, int, int, int] | list[tuple[int, int, int, int]] | None = None,
    legend: list[tuple[str, str]] | None = None,
    legend_position: str = "top_right",
    text_anchors: list[tuple[str, tuple[int, int]]] | None = None,
    side_lines: list[str] | None = None,
    side_lines_position: str = "right_edge",
    guide_lines: list[tuple[int, int, int, int, str]] | None = None,
    bboxes: list[tuple] | None = None,
) -> Path | None:
    """Save camera image with MULTIPLE masks overlaid in different colors.

    Each entry in *layers* is ``(mask, color)`` where color ∈ {'red', 'green',
    'blue'} (anything else is treated as green). Each layer uses the saturated
    overlay scheme from :func:`log_mask` (+200 on its channel, -80 on the
    others) so red / green / blue read cleanly side-by-side on a single panel.

    If *marker_mask* is given (typically a small intersection mask), a black
    cross is drawn at its centroid — useful for marking the overlap between
    two masks shown in the same panel.

    *crop_box* may be either a single ``(x, y, w, h)`` tuple OR a list of
    such tuples. When set, the saved image blacks out every pixel OUTSIDE
    the union of all boxes (so the operator can visually confirm SAM3 only
    saw those regions) and draws a yellow dashed rectangle outlining each
    box. Use a list when two different SAM3 queries were issued against
    non-identical crops (e.g. right-cam head crop + right-cam tail crop)
    so the panel reflects the actual SAM3 input regions accurately.

    If *legend* is given as a list of ``(label, color_name)`` pairs (where
    color_name is one of the _LAYER_RGB keys), a small colour-swatch legend
    is drawn in the top-right corner so the operator can see which colour
    corresponds to which mask.

    File path layout mirrors :func:`log_mask` exactly:
    ``{run_dir}/vis/{subdir}/{HHMMSS_FFF}_{tag}_{query}.png``. Silently no-ops
    when :func:`set_artifact_dir` has not been called.
    """
    if _artifact_dir is None or rgb is None:
        return None
    try:
        overlay = rgb.copy().astype(np.float32)
        for mask, color in layers:
            if mask is None:
                continue
            mb = np.asarray(mask).astype(bool)
            if not mb.any():
                continue
            # Paint mask pixels to the layer's saturated RGB. Supports
            # dark_red / bright_green / yellow / cyan / magenta cleanly,
            # unlike the legacy channel-boost which could only render
            # primaries. Falls back to bright green if the color string
            # is unknown so old callers keep working.
            rgb_tuple = _LAYER_RGB.get(color, _LAYER_RGB["green"])
            overlay[mb] = np.array(rgb_tuple, dtype=np.float32)
        blended = (
            rgb.astype(np.float32) * (1 - alpha) + overlay * alpha
        ).astype(np.uint8)

        # Crop visualisation: blacks out everything outside the union of
        # the crop boxes (so the operator can verify SAM3 only saw those
        # regions) and draws a yellow dashed rectangle around each box.
        # Applied AFTER blending so any mask leakage outside the union is
        # also blacked out — if you still see color outside the dashed
        # rectangles in the saved PNG, the upstream cropping has a real
        # bug.
        H_img, W_img = blended.shape[:2]
        crops_clipped: list[tuple[int, int, int, int]] = []
        if crop_box is not None:
            # Normalise single-tuple to a 1-element list so the loop
            # below handles both call patterns uniformly.
            raw_boxes = (
                [crop_box]
                if (len(crop_box) == 4 and not isinstance(crop_box[0], (list, tuple)))
                else list(crop_box)
            )
            for box in raw_boxes:
                cbx, cby, cbw, cbh = (int(v) for v in box)
                cbx = max(0, min(W_img - 1, cbx))
                cby = max(0, min(H_img - 1, cby))
                cbw = max(1, min(W_img - cbx, cbw))
                cbh = max(1, min(H_img - cby, cbh))
                crops_clipped.append((cbx, cby, cbw, cbh))
        if crops_clipped:
            outside = np.ones((H_img, W_img), dtype=bool)
            for (cbx, cby, cbw, cbh) in crops_clipped:
                outside[cby : cby + cbh, cbx : cbx + cbw] = False
            blended[outside] = 0

        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(blended)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
            )
        except Exception:
            font = ImageFont.load_default()

        # One yellow dashed rectangle per crop box — same indicator the
        # serve_sam3_debug.py UI uses for the crop tool so the PNG output
        # is consistent with what you saw when you drew the crop. When
        # multiple crops are supplied (right-cam head + tail) every box
        # gets its own dashed outline.
        for (cbx, cby, cbw, cbh) in crops_clipped:
            _draw_dashed_rect(
                draw, cbx, cby, cbx + cbw - 1, cby + cbh - 1,
                color="#ffeb3b", dash=8, gap=4, width=2,
            )

        # Caller-supplied dashed line segments — used to mark constraint
        # boundaries inside the crop (e.g. "head must be left of x=318").
        # Each entry: (x1, y1, x2, y2, fill_color_string). Always rendered
        # as a dashed segment from (x1, y1) to (x2, y2) — supports both
        # horizontal and vertical lines; diagonals are sampled at the same
        # dash/gap as horizontals so they look consistent.
        if guide_lines:
            for x1g, y1g, x2g, y2g, fill_c in guide_lines:
                _draw_dashed_line(draw, x1g, y1g, x2g, y2g, color=fill_c)

        # Caller-supplied bounding boxes. Each entry: (x, y, w, h) with
        # optional ``color`` (palette name like "blue" / "dark_red", or
        # any PIL color spec) and ``style`` ∈ {"dashed", "solid"}.
        # Defaults: blue dashed. Use this for search ROIs, constraint
        # regions, etc. — anything you'd otherwise post-process with
        # PIL on the saved PNG.
        if bboxes:
            for entry in bboxes:
                bx, by, bw, bh, bcolor, bstyle = _normalize_bbox(entry)
                bx1, by1 = bx + bw - 1, by + bh - 1
                if bstyle == "dashed":
                    _draw_dashed_rect(draw, bx, by, bx1, by1, color=bcolor)
                else:
                    draw.rectangle(
                        [(bx, by), (bx1, by1)],
                        outline=_resolve_color(bcolor),
                        width=2,
                    )

        if marker_mask is not None and np.any(marker_mask):
            # Bold red cross at the centroid of marker_mask, sized by
            # *marker_scale* (default 1.0 → 28-px arm, 8-px stroke; pass
            # ≈0.33 for a 3× smaller cross on the top cam, etc.). Bounded
            # to a minimum 1-px stroke + 4-px arm so it stays visible even
            # at very small scales.
            ys, xs = np.nonzero(marker_mask)
            _draw_cross(
                draw, int(xs.mean()), int(ys.mean()),
                color="#ff0000",
                arm=max(4, int(round(28 * marker_scale))),
                width=max(1, int(round(8 * marker_scale))),
            )

        if query:
            draw.text((10, 10), query, fill="white", font=font)
        if extra_text:
            for i, line in enumerate(str(extra_text).split("\n")):
                draw.text((10, 28 + 18 * i), line, fill="white", font=font)

        # Short status lines like "1: 1234 px d=0.21m". Two layouts:
        #   side_lines_position == "right_edge" (default) — plate floats
        #       on the image's right edge, below the legend if any.
        #   side_lines_position == "below"               — canvas is
        #       extended downward; the strip is drawn BELOW the image,
        #       so the operator can see every per-component number
        #       without occluding the segmentation pixels.
        # Each entry is either a plain string (rendered white) or a
        # (text, color) tuple where color is an RGB triple or any PIL
        # colour string (used by reward_right to grey out dropped lines).
        if side_lines:
            def _split_line(entry):
                if isinstance(entry, tuple) and len(entry) == 2:
                    return entry[0], entry[1]
                return entry, (255, 255, 255)
            try:
                side_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16
                )
            except Exception:
                side_font = ImageFont.load_default()
            line_texts = [_split_line(ln)[0] for ln in side_lines]
            try:
                line_widths = [
                    side_font.getbbox(ln)[2] - side_font.getbbox(ln)[0] for ln in line_texts
                ]
                line_w = max(line_widths) if line_widths else 0
            except Exception:
                line_w = max((len(ln) for ln in line_texts), default=0) * 10
            line_h = 22
            pad = 6
            box_w = line_w + 2 * pad
            box_h = line_h * len(side_lines) + 2 * pad

            if side_lines_position == "below":
                # Extend canvas downward (further if the legend already
                # extended it). Strip sits immediately below the existing
                # bottom of the canvas at the time of writing.
                strip_pad = 12
                cur_w, cur_h = img.size
                new_h = cur_h + box_h + 2 * strip_pad
                new_img = Image.new("RGB", (cur_w, new_h), (0, 0, 0))
                new_img.paste(img, (0, 0))
                img = new_img
                draw = ImageDraw.Draw(img)
                sx0 = strip_pad
                sy0 = cur_h + strip_pad
            else:
                # Stack below the existing legend (top-right placement).
                legend_offset = 0
                if legend and legend_position != "below":
                    legend_offset = 20 + 22 * len(legend)
                sx0 = max(0, W_img - box_w - 8)
                sy0 = 8 + legend_offset

            draw.rectangle(
                [(sx0, sy0), (sx0 + box_w, sy0 + box_h)],
                fill=(0, 0, 0),
                outline=(255, 255, 255),
                width=1,
            )
            for i, ln in enumerate(side_lines):
                text, color = _split_line(ln)
                draw.text(
                    (sx0 + pad, sy0 + pad + i * line_h),
                    text, fill=color, font=side_font,
                )

        # Big bold numeric / short text labels anchored to image-pixel
        # coordinates — used by reward_right to stamp "1" / "2" / "3" on
        # each connected component of the tail mask. Each anchor is
        # either (text, (x, y)) for default white or
        # (text, (x, y), color) where color is RGB or any PIL colour
        # string (used to grey out filter-dropped components). Black
        # halo behind so the text reads against any underlying colour.
        if text_anchors:
            try:
                big_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36
                )
            except Exception:
                big_font = ImageFont.load_default()
            for anchor in text_anchors:
                if len(anchor) == 3:
                    txt, (tx, ty), fill = anchor
                else:
                    txt, (tx, ty) = anchor
                    fill = (255, 255, 255)
                for dx in (-2, -1, 0, 1, 2):
                    for dy in (-2, -1, 0, 1, 2):
                        draw.text((tx + dx, ty + dy), txt, fill=(0, 0, 0), font=big_font)
                draw.text((tx, ty), txt, fill=fill, font=big_font)

        # Legend: colour swatch + label per layer. Two layouts:
        #   legend_position == "top_right" (default) — plate floats in the
        #       image's top-right corner. Compact but can occlude masks.
        #   legend_position == "below"               — canvas is extended
        #       downward by the legend block's height; the legend is drawn
        #       in the new strip BELOW the image so nothing inside the
        #       image is occluded. Use when overlays crowd the corners
        #       (right-cam panel) or when the panel is composited into a
        #       multi-camera figure where top-right overlap would clash.
        if legend:
            swatch = 14
            pad = 6
            row_h = max(swatch, 16) + 4
            try:
                # Pillow ≥ 10 uses font.getbbox; older versions had getsize.
                bbox_w = max(font.getbbox(lbl)[2] - font.getbbox(lbl)[0] for lbl, _ in legend)
            except Exception:
                bbox_w = max(len(lbl) * 8 for lbl, _ in legend)
            legend_w = swatch + pad + bbox_w + 2 * pad
            legend_h = row_h * len(legend) + 2 * pad

            if legend_position == "below":
                # Extend canvas downward so the legend lives outside the
                # current image area. Use img.size (CURRENT height) rather
                # than H_img (original) so any earlier downward extension
                # — e.g. a side_lines strip below — survives instead of
                # getting cropped off when we paste into the new canvas.
                strip_pad = 12
                cur_w, cur_h = img.size
                new_h = cur_h + legend_h + 2 * strip_pad
                new_img = Image.new("RGB", (cur_w, new_h), (0, 0, 0))
                new_img.paste(img, (0, 0))
                img = new_img
                draw = ImageDraw.Draw(img)
                x0 = strip_pad
                y0 = cur_h + strip_pad
            else:
                x0 = max(0, W_img - legend_w - 8)
                y0 = 8

            draw.rectangle(
                [(x0, y0), (x0 + legend_w, y0 + legend_h)],
                fill=(0, 0, 0),
                outline=(255, 255, 255),
                width=1,
            )
            for i, (label, color) in enumerate(legend):
                ry = y0 + pad + i * row_h
                sw_rgb = _LAYER_RGB.get(color, _LAYER_RGB["green"])
                draw.rectangle(
                    [(x0 + pad, ry), (x0 + pad + swatch, ry + swatch)],
                    fill=sw_rgb,
                    outline=(255, 255, 255),
                    width=1,
                )
                draw.text(
                    (x0 + pad + swatch + pad, ry - 1),
                    str(label), fill=(255, 255, 255), font=font,
                )

        safe_q = query.replace(" ", "_")[:20] if query else "masks"
        name = f"{_stamp()}_{tag}_{safe_q}.png"
        out_dir = _vis_subdir(subdir)
        if out_dir is None:
            return None
        path = out_dir / name
        img.save(path)
        logger.debug("Saved multi-mask artifact: %s", path)
        return path
    except Exception:
        logger.warning("Failed to save multi-mask artifact", exc_info=True)
        return None


def log_grasp(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    grasp_candidates: list,
    *,
    query: str = "",
    tag: str = "grasp",
    alpha: float = 0.35,
) -> Path | None:
    """Save camera image with SAM3 mask overlay + grasp pose arrows.

    Each grasp candidate should have ``.position`` (world xyz),
    ``.rpy`` (display degrees), ``.score``, and ``.width``.
    The grasp positions are projected to 2D if ``_project_fn`` is set,
    otherwise drawn at fixed positions as a fallback legend.
    """
    if _artifact_dir is None or rgb is None:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont

        canvas = rgb.copy()
        if mask is not None:
            mask_bool = mask.astype(bool)
            canvas[mask_bool, 1] = np.clip(
                canvas[mask_bool, 1].astype(np.float32) + 80, 0, 255
            ).astype(np.uint8)
            canvas = (
                rgb.astype(np.float32) * (1 - alpha) + canvas.astype(np.float32) * alpha
            ).astype(np.uint8)

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12
            )
        except Exception:
            font = ImageFont.load_default()

        colors = ["#FF4444", "#FF8800", "#FFCC00", "#44FF44", "#4488FF"]
        h = rgb.shape[0]
        for i, g in enumerate(grasp_candidates[:5]):
            color = colors[i % len(colors)]
            pos = g.position
            rpy = g.rpy
            score = g.score
            width = getattr(g, "width", 0.08)
            y_text = h - 18 * (len(grasp_candidates[:5]) - i) - 4
            text = (
                f"G{i}: [{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] "
                f"rpy=[{rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f}] "
                f"s={score:.3f} w={width:.3f}"
            )
            draw.text((4, y_text), text, fill=color, font=font)

        if query:
            draw.text((4, 4), query, fill="lime", font=font)

        safe_q = query.replace(" ", "_")[:20] if query else "grasp"
        name = f"{_stamp()}_{tag}_{safe_q}.png"
        out_dir = _vis_subdir("grasp")
        if out_dir is None:
            return None
        path = out_dir / name
        img.save(path)
        logger.debug("Saved grasp artifact: %s", path)
        return path
    except Exception:
        logger.warning("Failed to save grasp artifact", exc_info=True)
        return None


_vlm_query_counter: int = 0


def log_vlm_query(
    prompt: str,
    response: str,
    *,
    images: list[np.ndarray] | None = None,
    media_labels: list[str] | None = None,
    backend: str = "",
    tag: str = "vlm_query",
) -> Path | None:
    """Save VLM query input and output as separate markdown files.

    Creates text transcripts under ``{run_dir}/vlm/{NNN}_{tag}/`` and copies
    browser-visible input images under ``{run_dir}/vis/vlm/{NNN}_{tag}/``.
    """
    if _artifact_dir is None:
        return None
    global _vlm_query_counter
    _vlm_query_counter += 1
    idx = _vlm_query_counter

    try:
        # vlm/ keeps text transcripts; vis/vlm/ keeps browser-visible images.
        vlm_dir = _artifact_dir.parent / "vlm" / f"{idx:03d}_{tag}"
        vlm_dir.mkdir(parents=True, exist_ok=True)
        vlm_vis_root = _vis_subdir("vlm")
        if vlm_vis_root is None:
            return None
        vlm_vis_dir = vlm_vis_root / f"{idx:03d}_{tag}"
        vlm_vis_dir.mkdir(parents=True, exist_ok=True)

        # Input markdown
        input_lines = [f"# VLM Query #{idx}\n"]
        if backend:
            input_lines.append(f"**Backend:** {backend}\n")
        if media_labels:
            input_lines.append("**Images:** " + ", ".join(media_labels) + "\n")
        input_lines.append(f"\n## Prompt\n\n{prompt}\n")
        (vlm_dir / "input.md").write_text("\n".join(input_lines), encoding="utf-8")

        # Save input images
        if images:
            from PIL import Image

            for i, img_arr in enumerate(images):
                lbl = (
                    media_labels[i]
                    if media_labels and i < len(media_labels)
                    else f"image_{i}"
                )
                safe_lbl = (
                    lbl.replace(":", "_").replace("/", "_").replace(" ", "_")[:30]
                )
                img = Image.fromarray(img_arr)
                img.save(vlm_vis_dir / f"{safe_lbl}.png")

        # Output markdown
        output_lines = [f"# VLM Response #{idx}\n"]
        output_lines.append(f"\n## Response\n\n{response}\n")
        (vlm_dir / "output.md").write_text("\n".join(output_lines), encoding="utf-8")

        logger.debug("Saved VLM query #%d to %s", idx, vlm_dir)
        return vlm_dir
    except Exception:
        logger.warning("Failed to save VLM query artifact", exc_info=True)
        return None


def annotate_image(
    path: Path | str | None,
    *,
    bboxes: list[tuple] | None = None,
    lines: list[tuple] | None = None,
    crosses: list[tuple] | None = None,
    texts: list[tuple] | None = None,
) -> Path | None:
    """Post-hoc annotate an EXISTING PNG and save back in place.

    Use this when ``log_masks_multi`` already saved the panel and you
    want to add markup on top — search ROIs, constraint guide-lines,
    extra crosses, ad-hoc labels. Works regardless of whether the
    CAP-executor module cache picked up new kwargs on log_masks_multi.

    Each entry is a short tuple with positional defaults:

      bboxes: ``(x, y, w, h)`` | ``(...,color)`` | ``(...,color,style)``
              style ∈ {"dashed","solid"}; default blue dashed.
      lines:  ``(x1, y1, x2, y2)`` | ``(...,color)`` | ``(...,color,style)``
              style ∈ {"dashed","solid"}; default white dashed.
      crosses: ``(cx, cy)`` | ``(...,color)`` | ``(...,color,arm)``
               | ``(...,color,arm,width)``; default red, arm=14, width=4.
      texts:  ``(text, x, y)`` | ``(...,color)`` | ``(...,color,size)``;
              default white, 14 px. Color may be a palette name.

    Returns the same path it was given (for chaining), or ``None`` if
    the input path was ``None`` or the open/save failed."""
    if path is None:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)

        if bboxes:
            for entry in bboxes:
                bx, by, bw, bh, bcolor, bstyle = _normalize_bbox(entry)
                bx1, by1 = bx + bw - 1, by + bh - 1
                if bstyle == "dashed":
                    _draw_dashed_rect(draw, bx, by, bx1, by1, color=bcolor)
                else:
                    draw.rectangle(
                        [(bx, by), (bx1, by1)],
                        outline=_resolve_color(bcolor),
                        width=2,
                    )

        if lines:
            for entry in lines:
                color = "white"
                style = "dashed"
                lw = 2
                if len(entry) == 4:
                    x1, y1, x2, y2 = entry
                elif len(entry) == 5:
                    x1, y1, x2, y2, color = entry
                elif len(entry) == 6:
                    x1, y1, x2, y2, color, style = entry
                elif len(entry) == 7:
                    x1, y1, x2, y2, color, style, lw = entry
                else:
                    continue
                if style == "dashed":
                    _draw_dashed_line(draw, x1, y1, x2, y2, color=color, width=lw)
                else:
                    draw.line(
                        [(int(x1), int(y1)), (int(x2), int(y2))],
                        fill=_resolve_color(color), width=int(lw),
                    )

        if crosses:
            for entry in crosses:
                color = "#ff0000"
                arm = 14
                width = 4
                if len(entry) == 2:
                    cx, cy = entry
                elif len(entry) == 3:
                    cx, cy, color = entry
                elif len(entry) == 4:
                    cx, cy, color, arm = entry
                elif len(entry) == 5:
                    cx, cy, color, arm, width = entry
                else:
                    continue
                _draw_cross(draw, cx, cy, color=color, arm=arm, width=width)

        if texts:
            for entry in texts:
                color = "white"
                size = 14
                if len(entry) == 3:
                    txt, tx, ty = entry
                elif len(entry) == 4:
                    txt, tx, ty, color = entry
                elif len(entry) == 5:
                    txt, tx, ty, color, size = entry
                else:
                    continue
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                        int(size),
                    )
                except Exception:
                    font = ImageFont.load_default()
                draw.text(
                    (int(tx), int(ty)), str(txt),
                    fill=_resolve_color(color), font=font,
                )

        img.save(path)
        return Path(path)
    except Exception:
        logger.warning("Failed to annotate image %s", path, exc_info=True)
        return Path(path) if path else None


def log_image(
    image: np.ndarray,
    *,
    tag: str = "image",
    label: str = "",
    subdir: str = "image",
) -> Path | None:
    """Save a raw image to the artifact dir."""
    if _artifact_dir is None or image is None:
        return None
    try:
        from PIL import Image

        img = Image.fromarray(image)
        safe_l = label.replace(" ", "_")[:20] if label else "raw"
        name = f"{_stamp()}_{tag}_{safe_l}.png"
        out_dir = _vis_subdir(subdir)
        if out_dir is None:
            return None
        path = out_dir / name
        img.save(path)
        return path
    except Exception:
        logger.warning("Failed to save image artifact", exc_info=True)
        return None
