"""Smoke test for the NVIDIA inference-gateway LLM backend.

Verifies that ``cap.agent.llm.NvidiaLLM`` can talk to
``https://inference-api.nvidia.com/v1/`` and return text.

Supports single key (NVIDIA_API_KEY) or rotated keys (NVIDIA_API_KEY_1..N).

Usage::

    export NVIDIA_API_KEY=<your-key>
    uv run --no-sync python scripts/test_nvidia_llm.py
    uv run --no-sync python scripts/test_nvidia_llm.py --model azure/openai/gpt-5.1

    # test key rotation
    export NVIDIA_API_KEY_1=<key1> NVIDIA_API_KEY_2=<key2>
    uv run --no-sync python scripts/test_nvidia_llm.py --rounds 4
"""

from __future__ import annotations

import argparse
import sys

from enpire.env.forge.cap.agent.llm import NvidiaLLM
from enpire.env.forge.cap.agent.tools.vlm.backends.nvidia import list_nvidia_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="azure/anthropic/claude-opus-4-6")
    parser.add_argument("--prompt", default="Reply with exactly the string: PONG")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of requests to send (>1 exercises key rotation)")
    args = parser.parse_args()

    keys = list_nvidia_keys()
    if not keys:
        print("ERROR: no NVIDIA key found. Set NVIDIA_API_KEY or NVIDIA_API_KEY_1..N.", file=sys.stderr)
        sys.exit(1)

    print(f"Model:   {args.model}")
    print(f"Prompt:  {args.prompt!r}")
    print(f"Keys:    {len(keys)} configured")
    print(f"Rounds:  {args.rounds}")
    print("---")

    llm = NvidiaLLM(model=args.model)
    all_ok = True
    for i in range(args.rounds):
        reply = llm.generate_text(args.prompt)
        status = "OK" if reply.strip() else "FAIL: empty response"
        print(f"[{i + 1}/{args.rounds}] {status}: {reply.strip()[:80]!r}")
        if not reply.strip():
            all_ok = False

    print("---")
    print("ALL OK" if all_ok else "SOME ROUNDS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
