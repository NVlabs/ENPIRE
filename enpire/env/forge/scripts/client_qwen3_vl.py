#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple client for chatting with a running vLLM server."""

from __future__ import annotations
import argparse
import base64
from pathlib import Path
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a running vLLM server.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--image", type=Path, default=None, help="Optional local image to attach."
    )
    parser.add_argument(
        "--no-thinking", action="store_true", help="Suppress <think> blocks."
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=Path("./system_prompt/null.md"),
        help="Path to a .md or .txt system prompt file.",
    )
    return parser.parse_args()


def build_content(text: str, image_path: Path | None) -> str | list:
    if image_path is None or not image_path.exists():
        return text
    data = base64.b64encode(image_path.read_bytes()).decode()
    suffix = image_path.suffix.lstrip(".").replace("jpg", "jpeg")
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/{suffix};base64,{data}"},
        },
        {"type": "text", "text": text},
    ]


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    history = []

    extra = (
        {"chat_template_kwargs": {"enable_thinking": False}} if args.no_thinking else {}
    )
    if args.system_prompt and args.system_prompt.exists():
        system_text = args.system_prompt.read_text().strip()
        history.append({"role": "system", "content": system_text})
        print(f"System prompt loaded: {args.system_prompt.name}")
    else:
        print("No system prompt provided. ")
    print(f"Chatting with {args.model} at {args.base_url}")
    print("Type 'quit' or Ctrl-C to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        # Only attach image on first turn
        content = build_content(user_input, args.image if not history else None)
        history.append({"role": "user", "content": content})

        resp = client.chat.completions.create(
            model=args.model,
            messages=history,
            stream=True,
            **extra,
        )
        print("\nAssistant: ", end="", flush=True)
        reply = ""
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print("\n")
        history.append({"role": "assistant", "content": reply})  # only once, after loop


if __name__ == "__main__":
    main()
