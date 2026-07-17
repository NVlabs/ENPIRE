"""Smoke test for image understanding via the NVIDIA inference gateway.

Sends one image + one text question to an OpenAI-compatible vision model
on ``https://inference-api.nvidia.com/v1/`` and prints the reply.

Defaults target Gemini-3-flash-preview via NVIDIA's gateway. If your key
is scoped to other models, pass ``--model <id>`` — e.g.
``aws/anthropic/bedrock-claude-opus-4-6`` also accepts vision. No
max_tokens is sent; the gateway/model picks its own ceiling.

Usage::

    export NVIDIA_API_KEY=<your-key>
    uv run --no-sync python scripts/test_nvidia_vlm.py
    uv run --no-sync python scripts/test_nvidia_vlm.py --model gcp/google/gemini-3-pro
    uv run --no-sync python scripts/test_nvidia_vlm.py --list-models   # probe allowed models
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = str(
    _REPO_ROOT / "cap" / "tasks" / "peg_insertion" / "aligned_grasp.png"
)
DEFAULT_PROMPT = (
    "Describe this scene in detail. What objects are visible, what is their "
    "arrangement, and what task does the image appear to depict?"
)
BASE_URL = "https://inference-api.nvidia.com/v1/"


def _encode_image(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gcp/google/gemini-3-flash-preview")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Probe /v1/models and print what this key can access.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY env var is not set.", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    if args.list_models:
        for m in client.models.list().data:
            print(m.id)
        return

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Model:  {args.model}")
    print(f"Image:  {image_path}")
    print(f"Prompt: {args.prompt!r}")
    print("---")

    data_url = _encode_image(image_path)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )

    reply = (response.choices[0].message.content or "").strip()
    print(reply)
    print("---")
    print("OK" if reply else "FAIL: empty response")


if __name__ == "__main__":
    main()
