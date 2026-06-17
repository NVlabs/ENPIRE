#!/usr/bin/env python3
"""Benchmark: seg+VLM (Gemini 3 Flash) vs direct VLM (Gemini 3 Pro with thinking).

Iterates over labeled samples in a benchmark directory structured as:
    <bench_dir>/Inserted/cam_*/top.png, left.png
    <bench_dir>/Not Inserted/cam_*/top.png, left.png

For each sample, runs two methods:
  A) seg_vlm  — segment via SAM3, then Gemini 3 Flash on masked images
  B) vlm_pro  — raw images to Gemini 3.1 Pro with thinking (high budget)

Prints per-sample results and a final accuracy summary.

Usage:
    uv run scripts/bench_seg_vlm.py --dir ~/Downloads/test_bench
    uv run scripts/bench_seg_vlm.py --dir ~/Downloads/test_bench --save /tmp/bench_vis
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


# ---------------------------------------------------------------------------
# Image helpers (from test_seg_vlm.py)
# ---------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Failed to decode image: {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def segment_image(rgb: np.ndarray, query: str, sam3_url: str) -> np.ndarray | None:
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
            print(f"    SAM3 error for '{query}': {e}")
            return None

    data = resp.json()
    score = data["score"]
    if score < 0.1:
        return None

    mask_bytes = base64.b64decode(data["mask_b64"])
    return np.load(io.BytesIO(mask_bytes))


def apply_mask(rgb: np.ndarray, mask: np.ndarray, bg_color=(200, 200, 200)) -> np.ndarray:
    out = np.full_like(rgb, bg_color, dtype=np.uint8)
    out[mask > 0] = rgb[mask > 0]
    return out


def combine_masks(*masks: np.ndarray | None) -> np.ndarray:
    valid = [m for m in masks if m is not None]
    if not valid:
        raise ValueError("No valid masks to combine")
    combined = valid[0].copy()
    for m in valid[1:]:
        combined = np.maximum(combined, m)
    return combined


# ---------------------------------------------------------------------------
# VLM callers
# ---------------------------------------------------------------------------
def call_gemini_3_flash(prompt: str, images: list[np.ndarray], api_key: str) -> str:
    from cap.agent.tools.vlm_query import _query_gemini
    return _query_gemini(prompt, images, "gemini-3-flash-preview", api_key, 0.2)


def call_gemini_3_flash_thinking(prompt: str, images: list[np.ndarray], api_key: str) -> str:
    from cap.agent.tools.vlm_query import _query_gemini_pro
    return _query_gemini_pro(prompt, images, "gemini-3-flash-preview", api_key, 8192)


def call_gemini_25_flash(prompt: str, images: list[np.ndarray], api_key: str) -> str:
    from cap.agent.tools.vlm_query import _query_gemini
    return _query_gemini(prompt, images, "gemini-2.5-flash", api_key, 0.2)


def call_gemini_pro(prompt: str, images: list[np.ndarray], api_key: str) -> str:
    from cap.agent.tools.vlm_query import _query_gemini_pro
    return _query_gemini_pro(prompt, images, "gemini-3.1-pro-preview", api_key, 16384)


def call_gpt_no_reasoning(prompt: str, images: list[np.ndarray], api_key: str) -> str:
    from cap.agent.tools.vlm_query import _query_gpt
    return _query_gpt(prompt, images, "gpt-5.4", api_key, 0.2, reasoning_effort="none")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SEG_PROMPT = "Is the red inserted in the blue? <NOT INSERTED> or <YES INSERTED>."

PRO_PROMPT = (
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


def parse_answer(response: str) -> str:
    return "YES" if "YES" in response.upper() else "NO"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark seg+VLM vs direct VLM")
    parser.add_argument("--dir", "-d", required=True,
                        help="Benchmark directory with Inserted/ and Not Inserted/ subdirs")
    parser.add_argument("--sam3-host", default="localhost")
    parser.add_argument("--sam3-port", type=int, default=6767)
    parser.add_argument("--save", "-s", default=None,
                        help="Directory to save intermediate segmented images")
    parser.add_argument("--methods", "-m", nargs="+",
                        default=["seg_flash", "flash_think", "gpt_no_reason", "gemini25_flash"],
                        help="Methods to benchmark (default: all new ones). "
                             "Choices: seg_flash, seg_flash_think, pro_think, "
                             "flash_think, gpt_no_reason, gemini25_flash")
    args = parser.parse_args()

    bench_dir = Path(args.dir).expanduser().resolve()
    sam3_url = f"http://{args.sam3_host}:{args.sam3_port}"

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    # Method registry: name -> (label, needs_seg, prompt, vlm_fn, requires_key)
    METHOD_REGISTRY = {
        "seg_flash": (
            "seg+gemini3-flash", True, SEG_PROMPT,
            lambda p, imgs: call_gemini_3_flash(p, imgs, gemini_key), "gemini",
        ),
        "seg_flash_think": (
            "seg+gemini3-flash-think", True, SEG_PROMPT,
            lambda p, imgs: call_gemini_3_flash_thinking(p, imgs, gemini_key), "gemini",
        ),
        "seg_pro_think": (
            "seg+gemini3.1-pro-think", True, SEG_PROMPT,
            lambda p, imgs: call_gemini_pro(p, imgs, gemini_key), "gemini",
        ),
        "seg_gpt_no_reason": (
            "seg+gpt5.4-no-reasoning", True, SEG_PROMPT,
            lambda p, imgs: call_gpt_no_reasoning(p, imgs, openai_key), "openai",
        ),
        "pro_think": (
            "gemini3.1-pro-think", False, PRO_PROMPT,
            lambda p, imgs: call_gemini_pro(p, imgs, gemini_key), "gemini",
        ),
        "flash_think": (
            "gemini3-flash-think", False, PRO_PROMPT,
            lambda p, imgs: call_gemini_3_flash_thinking(p, imgs, gemini_key), "gemini",
        ),
        "gpt_no_reason": (
            "gpt5.4-no-reasoning", False, PRO_PROMPT,
            lambda p, imgs: call_gpt_no_reasoning(p, imgs, openai_key), "openai",
        ),
        "gemini25_flash": (
            "gemini2.5-flash", False, PRO_PROMPT,
            lambda p, imgs: call_gemini_25_flash(p, imgs, gemini_key), "gemini",
        ),
    }

    methods = []
    for m in args.methods:
        if m not in METHOD_REGISTRY:
            print(f"ERROR: unknown method '{m}'. Choices: {list(METHOD_REGISTRY.keys())}")
            sys.exit(1)
        label, needs_seg, prompt, vlm_fn, key_type = METHOD_REGISTRY[m]
        if key_type == "gemini" and not gemini_key:
            print(f"ERROR: GEMINI_API_KEY not set (needed for {m})")
            sys.exit(1)
        if key_type == "openai" and not openai_key:
            print(f"ERROR: OPENAI_API_KEY not set (needed for {m})")
            sys.exit(1)
        methods.append((m, label, needs_seg, prompt, vlm_fn))

    # Collect samples
    samples = []
    for label, subdir in [("YES", "Inserted"), ("NO", "Not Inserted")]:
        d = bench_dir / subdir
        if not d.exists():
            print(f"WARNING: {d} does not exist, skipping")
            continue
        for cam_dir in sorted(d.iterdir()):
            if cam_dir.is_dir() and (cam_dir / "top.png").exists():
                samples.append((cam_dir, label))

    any_seg = any(ns for _, _, ns, _, _ in methods)
    method_names = [m_id for m_id, _, _, _, _ in methods]

    print(f"Found {len(samples)} samples "
          f"({sum(1 for _, l in samples if l == 'YES')} inserted, "
          f"{sum(1 for _, l in samples if l == 'NO')} not inserted)")
    print(f"Methods: {', '.join(lbl for _, lbl, _, _, _ in methods)}\n")

    seg_queries = ["red block", "blue object"]
    # results[i] = {method_id: (ans, time)}
    results = []  # list of (name, gt, {method_id: (ans, elapsed)})

    for i, (sample_dir, gt) in enumerate(samples):
        name = f"{sample_dir.parent.name}/{sample_dir.name}"
        print(f"[{i+1}/{len(samples)}] {name}  (GT: {gt})")

        top = load_image(str(sample_dir / "top.png"))
        left = load_image(str(sample_dir / "left.png"))

        # Segment once if any method needs it
        masked = {}
        if any_seg:
            try:
                for cam_name, rgb in [("top", top), ("left", left)]:
                    masks = [segment_image(rgb, q, sam3_url) for q in seg_queries]
                    combined = combine_masks(*masks)
                    masked[cam_name] = apply_mask(rgb, combined)

                if args.save:
                    save_dir = Path(args.save).expanduser().resolve() / name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    for cam_name in ("top", "left"):
                        cv2.imwrite(str(save_dir / f"{cam_name}_masked.png"),
                                    cv2.cvtColor(masked[cam_name], cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"  segmentation ERROR — {e}")
                masked = {}

        method_results = {}
        for m_id, m_label, needs_seg, prompt, vlm_fn in methods:
            if needs_seg and not masked:
                method_results[m_id] = ("ERR", 0.0)
                print(f"  {m_label:<28} ERR (no segmentation)")
                continue
            imgs = [masked["top"], masked["left"]] if needs_seg else [top, left]
            try:
                t0 = time.time()
                resp = vlm_fn(prompt, imgs)
                elapsed = time.time() - t0
                ans = parse_answer(resp)
                print(f"  {m_label:<28} {ans} ({elapsed:.1f}s)  | {resp[:80]}")
                method_results[m_id] = (ans, elapsed)
            except Exception as e:
                print(f"  {m_label:<28} ERROR — {e}")
                method_results[m_id] = ("ERR", 0.0)

        results.append((name, gt, method_results))
        print()

    # --- Summary table ---
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    # Header
    hdr = f"  {'Sample':<40} {'GT':>3}"
    for m_id, m_label, _, _, _ in methods:
        short = m_label[:12]
        hdr += f"  {short:>12}"
    print(hdr)
    print(f"  {'-'*40} {'---':>3}" + "  ------------" * len(methods))

    # Per-sample rows
    for name, gt, mr in results:
        row = f"  {name:<40} {gt:>3}"
        for m_id, _, _, _, _ in methods:
            ans, _ = mr.get(m_id, ("ERR", 0))
            mark = "ok" if ans == gt else "X"
            row += f"  {ans:>9} {mark:>2}"
        print(row)

    # Accuracy summary
    print(f"\n{'='*80}")
    for m_id, m_label, _, _, _ in methods:
        answers = [(mr.get(m_id, ("ERR", 0))) for _, _, mr in results]
        valid = [(a, t) for a, t in answers if a != "ERR"]
        correct = sum(1 for (a, _), (_, gt, _) in zip(answers, results) if a == gt and a != "ERR")
        errs = len(answers) - len(valid)
        avg_t = sum(t for _, t in valid) / max(len(valid), 1)
        if valid:
            print(f"  {m_label:<28} {correct}/{len(valid)} correct "
                  f"({correct/len(valid)*100:.0f}%)  avg {avg_t:.1f}s  errors={errs}")
        else:
            print(f"  {m_label:<28} all errors")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
