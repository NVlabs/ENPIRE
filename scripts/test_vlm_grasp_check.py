#!/usr/bin/env python3
"""Evaluate VLM grasp alignment check on labeled saved camera images.

Iterates over logs/saved_cams/true/ (aligned) and logs/saved_cams/false/ (misaligned),
runs the VLM grasp check on each, saves per-folder log.txt, and prints accuracy summary.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np

from cap.agent.tools.vlm_query import _query_qwen, _query_gemini

VLM_BACKEND = "gemini"
VLM_TEMPERATURE = 0.2
NUM_VOTES = 1

SAVED_CAMS_DIR = Path(_ROOT) / "logs/saved_cams"
REF_DIR = Path(_ROOT) / "cap/tasks/peg_insertion"

PROMPT = (
    "A robot gripper is holding a red stick for peg insertion into a blue plate.\n\n"
    "Image 1: top-down camera of the scene.\n"
    "Image 2: wrist camera from the gripper.\n"
    "Image 3: CORRECT wrist camera example (aligned grasp).\n"
    "Image 4: INCORRECT top camera example (misaligned grasp).\n\n"
    "CHECK 1 — WRIST CAMERA (Image 2 vs Image 3):\n"
    "In Image 3 (correct): The red stick is a large red trapezoid centered between the "
    "two black gripper fingers. A green sphere is visible on the stick. "
    "The blue plate (if visible) has horizontal edges.\n"
    "→ Does Image 2 look similar? Stick large and centered with green sphere? "
    "If stick is missing, tiny, or off to the side → NO.\n\n"
    "CHECK 2 — TOP CAMERA (Image 1 vs Image 4):\n"
    "In Image 4 (incorrect): From above, the red stick is ROTATED — it sits at a diagonal angle "
    "relative to the gripper jaws. The stick edges are NOT aligned with the gripper jaw line.\n"
    "In a correct grasp: The red stick is aligned parallel to the gripper jaws from above.\n"
    "→ Does Image 1 show the stick rotated like Image 4? If yes → NO.\n\n"
    "Answer YES only if BOTH checks pass. Answer NO if either fails.\n"
    "Answer your reason then YES or NO."
)


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def query_vlm(images: list[np.ndarray], text: str) -> str:
    if VLM_BACKEND == "qwen":
        from cap.config import QWEN_VL_URL, QWEN_VL_MODEL
        return _query_qwen(text, images, QWEN_VL_URL, QWEN_VL_MODEL, VLM_TEMPERATURE)
    elif VLM_BACKEND == "gemini":
        import os
        from cap.config import GEMINI_VL_MODEL
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        return _query_gemini(text, images, GEMINI_VL_MODEL, api_key, VLM_TEMPERATURE)
    else:
        raise ValueError(f"Unknown backend {VLM_BACKEND}")


def run_check(cam_dir: Path, ground_truth: bool) -> dict:
    """Run grasp check on one saved cam folder. Returns result dict."""
    top_img = load_rgb(cam_dir / "top.png")
    left_img = load_rgb(cam_dir / "left.png")
    ref_good = load_rgb(REF_DIR / "aligned_grasp.png")
    ref_bad = load_rgb(REF_DIR / "misaligned_top.png")
    images = [top_img, left_img, ref_good, ref_bad]

    labels = (
        "Image 1: top camera, "
        "Image 2: wrist camera, "
        "Image 3: CORRECT wrist camera example, "
        "Image 4: INCORRECT top camera example"
    )
    text = f"[Images: {labels}]\n{PROMPT}"

    votes = []
    responses = []
    for i in range(NUM_VOTES):
        resp = query_vlm(images, text)
        vote = "YES" in resp.upper()
        votes.append(vote)
        responses.append(resp)

    yes_count = sum(votes)
    predicted = yes_count > NUM_VOTES // 2  # majority vote
    correct = predicted == ground_truth

    # Save log
    log_file = cam_dir / "log.txt"
    with open(log_file, "w") as f:
        f.write(f"Backend: {VLM_BACKEND}\n")
        f.write(f"Temperature: {VLM_TEMPERATURE}\n")
        f.write(f"Votes: {NUM_VOTES}, YES={yes_count}, NO={NUM_VOTES - yes_count}\n")
        f.write(f"Ground truth: {'aligned' if ground_truth else 'misaligned'}\n")
        f.write(f"Predicted: {'YES' if predicted else 'NO'} (majority)\n")
        f.write(f"Correct: {correct}\n\n")
        f.write(f"Prompt: {PROMPT}\n\n")
        for i, resp in enumerate(responses):
            f.write(f"--- Vote {i+1}: {'YES' if votes[i] else 'NO'} ---\n{resp}\n\n")
    response = f"[{yes_count}/{NUM_VOTES} YES] {responses[-1]}"

    return {
        "folder": cam_dir.name,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": correct,
        "response": response,
    }


def main():
    print(f"Backend: {VLM_BACKEND}, Temperature: {VLM_TEMPERATURE}")
    print(f"Prompt: {PROMPT[:80]}...\n")

    results = []

    for label, gt in [("true", True), ("false", False)]:
        label_dir = SAVED_CAMS_DIR / label
        if not label_dir.exists():
            print(f"Skipping {label_dir} (not found)")
            continue
        folders = sorted(label_dir.iterdir())
        folders = [f for f in folders if f.is_dir() and (f / "left.png").exists()]
        print(f"--- {label}/ ({len(folders)} folders, ground_truth={'aligned' if gt else 'misaligned'}) ---")

        for cam_dir in folders:
            try:
                result = run_check(cam_dir, gt)
                mark = "OK" if result["correct"] else "WRONG"
                pred_str = "YES" if result["predicted"] else "NO"
                print(f"  {cam_dir.name}: predicted={pred_str} [{mark}]")
                results.append(result)
            except Exception as e:
                print(f"  {cam_dir.name}: ERROR {e}")
                results.append({
                    "folder": cam_dir.name,
                    "ground_truth": gt,
                    "predicted": None,
                    "correct": False,
                    "response": str(e),
                })

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    true_pos = sum(1 for r in results if r["ground_truth"] and r["predicted"])
    true_neg = sum(1 for r in results if not r["ground_truth"] and not r["predicted"])
    false_pos = sum(1 for r in results if not r["ground_truth"] and r["predicted"])
    false_neg = sum(1 for r in results if r["ground_truth"] and not r["predicted"])

    print(f"\n{'='*50}")
    print(f"SUMMARY ({VLM_BACKEND}, temp={VLM_TEMPERATURE}, votes={NUM_VOTES})")
    print(f"{'='*50}")
    print(f"  Total:       {total}")
    print(f"  Accuracy:    {correct}/{total} ({100*correct/max(total,1):.1f}%)")
    print(f"  True pos:    {true_pos} (aligned → YES)")
    print(f"  True neg:    {true_neg} (misaligned → NO)")
    print(f"  False pos:   {false_pos} (misaligned → YES)")
    print(f"  False neg:   {false_neg} (aligned → NO)")
    if false_pos + false_neg > 0:
        print(f"\n  Wrong predictions:")
        for r in results:
            if not r["correct"]:
                gt_str = "aligned" if r["ground_truth"] else "misaligned"
                pred_str = "YES" if r["predicted"] else "NO"
                err_type = "FP" if r["predicted"] and not r["ground_truth"] else "FN"
                print(f"    [{err_type}] {r['folder']} (gt={gt_str}, pred={pred_str})")


if __name__ == "__main__":
    main()
