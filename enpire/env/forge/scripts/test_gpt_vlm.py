#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for GPT VLM backend (gpt-5.4 via OpenAI API).

Usage:
    # Default: text-only, reasoning=high
    uv run scripts/test_gpt_vlm.py

    # Test different reasoning efforts
    uv run scripts/test_gpt_vlm.py --reasoning none
    uv run scripts/test_gpt_vlm.py --reasoning low
    uv run scripts/test_gpt_vlm.py --reasoning high

    # With a local image file:
    uv run scripts/test_gpt_vlm.py --image path/to/image.png --prompt "What do you see?"

    # With a camera capture from cap_server:
    uv run scripts/test_gpt_vlm.py --camera top --prompt "Describe the scene"
"""

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


def load_image(path: str) -> np.ndarray:
    import cv2

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Failed to decode image: {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def capture_camera(camera: str) -> np.ndarray:
    import portal

    from enpire.env.forge.cap.config import CAP_SERVER_PORT

    client = portal.Client(f"localhost:{CAP_SERVER_PORT}")
    img = client.get_camera_image(camera).result()
    img = np.asarray(img)
    if img.size < 100:
        raise RuntimeError(f"Empty image from camera:{camera}")
    return img


def run_query(prompt, images, model, api_key, temperature, reasoning_effort):
    from enpire.env.forge.cap.agent.tools.vlm_query import _query_gpt

    print(f"Model:       {model}")
    print(f"Temperature: {temperature}")
    print(f"Reasoning:   {reasoning_effort}")
    print(f"Prompt:      {prompt}")
    print(f"Images:      {len(images)}")
    print("-" * 50)

    t0 = time.time()
    response = _query_gpt(
        text=prompt,
        images=images,
        model=model,
        api_key=api_key,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    elapsed = time.time() - t0
    print(f"Response ({elapsed:.2f}s):")
    print(response)
    return response


def main():
    parser = argparse.ArgumentParser(description="Test GPT VLM backend")
    parser.add_argument("--prompt", "-p", default=None,
                        help="Text prompt to send")
    parser.add_argument("--image", "-i", default=None,
                        help="Path to a local image file")
    parser.add_argument("--camera", "-c", default=None,
                        help="Camera name to capture from cap_server (top/left/right)")
    parser.add_argument("--model", "-m", default=None,
                        help="Override model name (default: from config)")
    parser.add_argument("--temperature", "-t", type=float, default=0.2,
                        help="Sampling temperature (default: 0.2)")
    parser.add_argument("--reasoning", "-r", default=None,
                        choices=["none", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort (default: test all levels)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    from enpire.env.forge.cap.config import GPT_VL_MODEL

    model = args.model or GPT_VL_MODEL
    images: list[np.ndarray] = []

    if args.image:
        print(f"Loading image: {args.image}")
        images.append(load_image(args.image))
    elif args.camera:
        print(f"Capturing from camera:{args.camera}")
        images.append(capture_camera(args.camera))
    else:
        print("No image provided — sending text-only query")

    # If a specific reasoning level or prompt is given, run once
    if args.reasoning is not None or args.prompt is not None:
        prompt = args.prompt or "What is 15 * 27? Show your work."
        reasoning = args.reasoning or "high"
        try:
            run_query(prompt, images, model, api_key, args.temperature, reasoning)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    # Default: test reasoning=none vs reasoning=high
    prompt = "What is 15 * 27? Show your work."
    print("=" * 60)
    print("TEST 1: reasoning=none (no thinking)")
    print("=" * 60)
    try:
        run_query(prompt, images, model, api_key, args.temperature, "none")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("TEST 2: reasoning=high (thinking enabled)")
    print("=" * 60)
    try:
        run_query(prompt, images, model, api_key, args.temperature, "high")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
