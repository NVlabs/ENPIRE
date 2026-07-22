#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for Gemini VLM backend with thinking support.

Usage:
    # Default: text-only, thinking=high
    uv run scripts/test_gemini_vlm.py

    # With images
    uv run scripts/test_gemini_vlm.py --image top.png left.png --prompt "Describe the scene"

    # Different thinking levels
    uv run scripts/test_gemini_vlm.py --thinking none
    uv run scripts/test_gemini_vlm.py --thinking low
    uv run scripts/test_gemini_vlm.py --thinking high
"""

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def load_image_bytes(path: str) -> bytes:
    import cv2
    import numpy as np

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Failed to decode image: {p}")
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def run_query(prompt, image_bytes_list, model_name, api_key, thinking_level):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    contents: list = []
    for i, jpeg_bytes in enumerate(image_bytes_list):
        contents.append(types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"))
    contents.append(prompt)

    config_kwargs = {
        "max_output_tokens": 16384,
    }
    if thinking_level != "none":
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=8192,
        )

    print(f"Model:    {model_name}")
    print(f"Thinking: {thinking_level}")
    print(f"Prompt:   {prompt}")
    print(f"Images:   {len(image_bytes_list)}")
    print("-" * 50)

    t0 = time.time()
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    elapsed = time.time() - t0

    # Print thoughts if available
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "thought") and part.thought:
                print(f"Thinking ({elapsed:.2f}s):")
                print(part.text)
                print("-" * 50)

    print(f"Response ({elapsed:.2f}s):")
    print(response.text)
    return response.text


def main():
    parser = argparse.ArgumentParser(description="Test Gemini VLM backend")
    parser.add_argument("--prompt", "-p", default=None,
                        help="Text prompt to send")
    parser.add_argument("--image", "-i", nargs="+", default=None,
                        help="Path(s) to local image file(s)")
    parser.add_argument("--model", "-m", default="gemini-2.5-flash",
                        help="Model name (default: gemini-2.5-flash)")
    parser.add_argument("--thinking", "-t", default=None,
                        choices=["none", "low", "medium", "high"],
                        help="Thinking level (default: test none vs high)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set")
        sys.exit(1)

    image_bytes_list: list[bytes] = []
    if args.image:
        for img_path in args.image:
            print(f"Loading image: {img_path}")
            image_bytes_list.append(load_image_bytes(img_path))
    else:
        print("No image provided — sending text-only query")

    # Single run if specific thinking level or prompt given
    if args.thinking is not None or args.prompt is not None:
        prompt = args.prompt or "What is 15 * 27? Show your work."
        thinking = args.thinking or "high"
        try:
            run_query(prompt, image_bytes_list, args.model, api_key, thinking)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    # Default: test thinking=none vs thinking=high
    prompt = "What is 15 * 27? Show your work."
    print("=" * 60)
    print("TEST 1: thinking=none")
    print("=" * 60)
    try:
        run_query(prompt, image_bytes_list, args.model, api_key, "none")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 60)
    print("TEST 2: thinking=high")
    print("=" * 60)
    try:
        run_query(prompt, image_bytes_list, args.model, api_key, "high")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
