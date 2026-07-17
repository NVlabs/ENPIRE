"""Quick smoke test for NVIDIA's OpenAI-compatible inference gateway.

Defaults to the Claude Opus 4.7 model string used elsewhere in this repo and
prints a useful failure message instead of crashing on non-JSON error bodies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests  # pyright: ignore[reportMissingModuleSource]

BASE_URL = "https://inference-api.nvidia.com/v1/"
DEFAULT_MODEL = "azure/anthropic/claude-opus-4-7"
DEFAULT_PROMPT = "Reply with exactly the string: PONG"


def _print_model_hint(model: str) -> None:
    if model == "aws/anthropic/bedrock-claude-opus-4-7":
        print(
            "Hint: this repo uses azure/anthropic/claude-opus-4-7 for Opus 4.7 "
            "on NVIDIA's gateway.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models visible to the configured NVIDIA key.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY env var is not set.", file=sys.stderr)
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        if args.list_models:
            response = requests.get(f"{BASE_URL}models", headers=headers, timeout=60)
        else:
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": args.system},
                    {"role": "user", "content": args.prompt},
                ],
                "max_tokens": args.max_tokens,
            }
            _claude4 = ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4")
            if not any(s in args.model.lower() for s in _claude4):
                payload["temperature"] = 0.7

            response = requests.post(
                f"{BASE_URL}chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
    except requests.RequestException as exc:
        print(f"ERROR: request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not response.ok:
        print(f"ERROR: HTTP {response.status_code}", file=sys.stderr)
        body = response.text.strip()
        if body:
            print(body, file=sys.stderr)
        _print_model_hint(args.model)
        sys.exit(1)

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print(
            "ERROR: expected a JSON response but the gateway returned "
            f"{response.headers.get('content-type', 'unknown content type')}.",
            file=sys.stderr,
        )
        body = response.text.strip()
        if body:
            print(body, file=sys.stderr)
        _print_model_hint(args.model)
        sys.exit(1)

    if args.list_models:
        for model in data.get("data", []):
            model_id = model.get("id")
            if model_id:
                print(model_id)
        return

    reply = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    print(reply if reply else json.dumps(data, indent=2))


if __name__ == "__main__":
    main()