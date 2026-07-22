# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
High-end figure aesthetics: colors, typography, and matplotlib style.

Design philosophy
-----------------
  - Publication-quality (Nature / ICML / NeurIPS register)
  - Minimal ink, maximum signal
  - Muted, refined palette — readable on white, print-safe, colorblind-safe
  - Helvetica-adjacent typography; graceful system-font fallback
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

PALETTE = {
    # Two primary run colors — warm vs cool, clearly distinct, not loud
    "crimson":  "#B03A2E",   # deep terra-cotta — urgency without harshness
    "navy":     "#1F618D",   # measured steel blue — precision, calm
    # Supporting cast
    "amber":    "#CA6F1E",   # warm amber accent
    "sage":     "#1E8449",   # muted sage green
    "slate":    "#5D6D7E",   # neutral mid-grey for annotations / secondary
    "ash":      "#AAB7B8",   # light grey for de-emphasis
    "fog":      "#F2F3F4",   # near-white panel tint
}

# Default color order for multi-run plots (extend as needed)
RUN_COLORS = [
    PALETTE["crimson"],
    PALETTE["navy"],
    PALETTE["amber"],
    PALETTE["sage"],
]

# Transparency levels
ALPHA_FRONTIER = 1.00   # frontier line + markers: fully opaque
ALPHA_SCATTER  = 0.38   # non-frontier scatter dots: recede into background

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

# Font options — swap FONT to change the whole figure's typeface
FONT = "comic_sans"   # options: "helvetica" | "comic_sans"

_SANS_HELVETICA = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
_SANS_COMIC     = ["Comic Sans MS", "Comic Sans", "DejaVu Sans"]  # fallback if not installed

_SANS = _SANS_COMIC if FONT == "comic_sans" else _SANS_HELVETICA

FONTSIZE = {
    "title":  13,
    "label":  11,
    "tick":    9,
    "legend":  9,
    "annot":   8,
}

# ---------------------------------------------------------------------------
# rcParams style dict
# ---------------------------------------------------------------------------

STYLE: dict = {
    # Figure
    "figure.facecolor":      "white",
    "figure.dpi":            150,

    # Axes
    "axes.facecolor":        "white",
    "axes.edgecolor":        "#2C3E50",
    "axes.linewidth":        0.75,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.labelcolor":       "#1A1A1A",
    "axes.labelsize":        FONTSIZE["label"],
    "axes.titlesize":        FONTSIZE["title"],
    "axes.titleweight":      "normal",
    "axes.titlepad":         9,
    "axes.grid":             True,
    "grid.color":            "#D5D8DC",
    "grid.linewidth":        0.45,
    "grid.linestyle":        "--",
    "grid.alpha":            0.8,

    # Ticks
    "xtick.labelsize":       FONTSIZE["tick"],
    "ytick.labelsize":       FONTSIZE["tick"],
    "xtick.color":           "#3D3D3D",
    "ytick.color":           "#3D3D3D",
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      3.5,
    "ytick.major.size":      3.5,
    "xtick.major.width":     0.65,
    "ytick.major.width":     0.65,
    "xtick.minor.visible":   False,
    "ytick.minor.visible":   False,

    # Lines & markers
    "lines.linewidth":       1.8,
    "lines.markersize":      5,
    "lines.solid_capstyle":  "round",
    "lines.solid_joinstyle": "round",

    # Legend
    "legend.frameon":        False,
    "legend.fontsize":       FONTSIZE["legend"],
    "legend.labelspacing":   0.4,
    "legend.handlelength":   1.6,
    "legend.handleheight":   0.8,
    "legend.borderpad":      0.0,
    "legend.columnspacing":  1.0,

    # Font
    "font.family":           "sans-serif",
    "font.sans-serif":       _SANS,
    "text.color":            "#1A1A1A",

    # Save
    "savefig.bbox":          "tight",
    "savefig.dpi":           300,
    "savefig.facecolor":     "white",
}


def apply_style() -> None:
    """Apply the house style to matplotlib globally."""
    mpl.rcParams.update(STYLE)


def figure(w: float = 6.5, h: float = 4.2, **kwargs):
    """Return a styled (fig, ax) pair."""
    apply_style()
    return plt.subplots(figsize=(w, h), **kwargs)


def figures(nrows: int, ncols: int, w: float = 6.5, h: float = 4.2, **kwargs):
    """Return a styled (fig, axes) grid."""
    apply_style()
    return plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
