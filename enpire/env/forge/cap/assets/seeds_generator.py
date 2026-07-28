#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_TIME_SEED = "42"
DEFAULT_COUNT = 40
DEFAULT_LOW = 0
DEFAULT_HIGH = 2_147_483_647


def _time_seed_to_int(time_seed: str) -> int:
    digits = "".join(ch for ch in time_seed if ch.isdigit())
    if not digits:
        raise ValueError(
            f"time seed {time_seed!r} does not contain any digits to seed from"
        )
    return int(digits)


def generate_seeds(
    *,
    count: int,
    time_seed: str,
    low: int,
    high: int,
) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if low > high:
        raise ValueError("low must be <= high")

    population_size = high - low + 1
    if count > population_size:
        raise ValueError(
            f"cannot draw {count} unique seeds from range [{low}, {high}]"
        )

    rng = random.Random(_time_seed_to_int(time_seed))
    return rng.sample(range(low, high + 1), count)


def format_output(
    seeds: list[int],
    output_format: str,
) -> str:
    if output_format == "json":
        return json.dumps(seeds, indent=2)
    if output_format == "bash":
        return "SEEDS=(" + " ".join(str(seed) for seed in seeds) + ")"
    if output_format == "csv":
        return ",".join(str(seed) for seed in seeds)
    return "\n".join(str(seed) for seed in seeds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible random evaluation seeds from a fixed "
            "timestamp-style seed."
        )
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--time-seed", default=DEFAULT_TIME_SEED)
    parser.add_argument("--low", type=int, default=DEFAULT_LOW)
    parser.add_argument("--high", type=int, default=DEFAULT_HIGH)
    parser.add_argument(
        "--format",
        choices=("text", "json", "bash", "csv"),
        default="text",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    seeds = generate_seeds(
        count=args.count,
        time_seed=args.time_seed,
        low=args.low,
        high=args.high,
    )
    rendered = format_output(seeds, args.format)

    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
