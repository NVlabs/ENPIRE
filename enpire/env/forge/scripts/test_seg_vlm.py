#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test workflow: segment objects then pass segmented image to VLM.

1. Load top + left camera images (from local files or live cameras)
2. Segment "red block" and "blue plate" from each image via SAM3
3. Combine masks to create an image showing only the segmented objects
4. Pass the segmented-only images to VLM for insertion check

Usage:
    # From local image files (no cap_server needed, SAM3 server required):
    uv run scripts/test_seg_vlm.py --dir ~/Downloads/cam_20260313_154817

    # From live cameras (cap_server + SAM3 server required):
    uv run scripts/test_seg_vlm.py --live

    # Override VLM backend:
    uv run scripts/test_seg_vlm.py --dir ~/Downloads/cam_20260313_154817 --backend gemini_pro
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np
import requests


def load_image(path: str) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Failed to decode image: {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def segment_image(rgb: np.ndarray, query: str, sam3_url: str) -> np.ndarray | None:
    """Send image to SAM3 server and return binary mask, or None on failure."""
    buf = io.BytesIO()
    np.save(buf, rgb)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{sam3_url}/segment",
                json={"text": query, "image_b64": image_b64},
                timeout=30,
            )
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
            print(f"  SAM3 error for '{query}': {e}")
            return None

    data = resp.json()
    score = data["score"]
    mask_area = data["mask_area"]
    print(f"  '{query}': score={score:.3f}, area={mask_area}px")

    if score < 0.1:
        print(f"  '{query}': score too low, skipping")
        return None

    mask_bytes = base64.b64decode(data["mask_b64"])
    return np.load(io.BytesIO(mask_bytes))


def apply_mask(rgb: np.ndarray, mask: np.ndarray, bg_color=(200, 200, 200)) -> np.ndarray:
    """Keep only masked region, fill background with solid color."""
    out = np.full_like(rgb, bg_color, dtype=np.uint8)
    out[mask > 0] = rgb[mask > 0]
    return out


def apply_highlight(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Dim non-masked region while keeping masked region bright. Preserves full image context."""
    out = (rgb * alpha).astype(np.uint8)
    out[mask > 0] = rgb[mask > 0]
    return out


def combine_masks(*masks: np.ndarray | None) -> np.ndarray:
    """OR together multiple binary masks, skipping None."""
    valid = [m for m in masks if m is not None]
    if not valid:
        raise ValueError("No valid masks to combine")
    combined = valid[0].copy()
    for m in valid[1:]:
        combined = np.maximum(combined, m)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Test segmentation + VLM workflow")
    parser.add_argument("--dir", "-d", default=None,
                        help="Directory with top.png and left.png")
    parser.add_argument("--live", action="store_true",
                        help="Capture from live cameras via cap_server")
    parser.add_argument("--backend", "-b", default="gemini_pro",
                        help="VLM backend (default: gemini_pro)")
    parser.add_argument("--reasoning", "-r", default="high",
                        help="Reasoning effort (default: high)")
    parser.add_argument("--sam3-host", default="localhost",
                        help="SAM3 server host (default: localhost)")
    parser.add_argument("--sam3-port", type=int, default=6767,
                        help="SAM3 server port (default: 6767)")
    parser.add_argument("--save", "-s", default=None,
                        help="Directory to save intermediate images")
    args = parser.parse_args()

    sam3_url = f"http://{args.sam3_host}:{args.sam3_port}"

    # Load images
    cameras = {}
    if args.live:
        import portal
        from enpire.env.forge.cap.config import CAP_SERVER_PORT
        client = portal.Client(f"localhost:{CAP_SERVER_PORT}")
        for cam in ("top", "left"):
            img = np.asarray(client.get_camera_image(cam).result())
            cameras[cam] = img
            print(f"Captured camera:{cam} ({img.shape})")
    elif args.dir:
        d = Path(args.dir).expanduser().resolve()
        for cam in ("top", "left"):
            cameras[cam] = load_image(str(d / f"{cam}.png"))
            print(f"Loaded {cam}.png ({cameras[cam].shape})")
    else:
        print("ERROR: specify --dir or --live")
        sys.exit(1)

    # Segment objects from each camera view
    queries = ["red block", "blue object"]
    masked_images = {}
    highlighted_images = {}

    for cam, rgb in cameras.items():
        print(f"\nSegmenting {cam} camera:")
        masks = []
        for query in queries:
            mask = segment_image(rgb, query, sam3_url)
            masks.append(mask)

        combined = combine_masks(*masks)
        masked_images[cam] = apply_mask(rgb, combined)
        highlighted_images[cam] = apply_highlight(rgb, combined)
        print(f"  Combined mask: {int(combined.sum())} pixels")

        if args.save:
            save_dir = Path(args.save).expanduser().resolve()
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / f"{cam}_masked.png"),
                        cv2.cvtColor(masked_images[cam], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_dir / f"{cam}_highlighted.png"),
                        cv2.cvtColor(highlighted_images[cam], cv2.COLOR_RGB2BGR))
            print(f"  Saved {cam}_masked.png and {cam}_highlighted.png")

    # Prompts for each method
    direct_prompt = (
        "Look at both images (Image 1: top-down view, Image 2: wrist camera view).\n\n"
        "The blue object is a thin plate with a through-hole.\n"
        "The red stick is longer than the plate thickness.\n\n"
        "The stick is considered fully inserted if it passes through the plate,\n"
        "even if it protrudes above the plate.\n\n"
        "First determine:\n"
        "1. Is the stick inside the hole?\n"
        "2. Is it aligned with the hole opening?\n"
        "3. Does it appear to pass through the plate thickness?\n\n"
        "Then answer only:\n"
        "<YES> or <NO>."
    )

    seg_prompt = "Is the red inserted in the blue? <YES> or <NO>."

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    # Helpers
    def call_gemini_pro(prompt, images):
        from enpire.env.forge.cap.agent.tools.vlm_query import _query_gemini_pro
        budget_map = {"none": 0, "low": 2048, "medium": 8192, "high": 16384}
        thinking_budget = budget_map.get(args.reasoning, 8192)
        return _query_gemini_pro(prompt, images, "gemini-3.1-pro-preview",
                                  api_key, thinking_budget)

    def call_gemini_flash(prompt, images):
        from enpire.env.forge.cap.agent.tools.vlm_query import _query_gemini
        return _query_gemini(prompt, images, "gemini-3-flash-preview", api_key, 0.2)

    # Run benchmark
    methods = [
        ("A) Direct raw → gemini_pro", direct_prompt,
         [cameras["top"], cameras["left"]], call_gemini_pro),
        ("B) Seg masked → gemini_flash", seg_prompt,
         [masked_images["top"], masked_images["left"]], call_gemini_flash),
        ("C) Seg highlighted → gemini_flash", seg_prompt,
         [highlighted_images["top"], highlighted_images["left"]], call_gemini_flash),
    ]

    print(f"\n{'='*60}")
    print(f"BENCHMARK")
    print(f"{'='*60}")

    for label, prompt, images, vlm_fn in methods:
        print(f"\n--- {label} ---")
        print(f"  Prompt: {prompt[:80]}")
        t0 = time.time()
        try:
            response = vlm_fn(prompt, images)
            elapsed = time.time() - t0
            ans = "YES" if "YES" in response.upper() else "NO"
            print(f"  Answer: {ans} ({elapsed:.2f}s)")
            print(f"  Full: {response[:200]}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
