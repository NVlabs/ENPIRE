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
DEFAULT_LAYOUT_LOW = 1
DEFAULT_LAYOUT_HIGH = 60
DEFAULT_STYLE_LOW = 1
DEFAULT_STYLE_HIGH = 60


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


def _sample_axis(
    rng: random.Random,
    *,
    count: int,
    low: int,
    high: int,
) -> list[int]:
    if low > high:
        raise ValueError("low must be <= high")
    population_size = high - low + 1
    if count <= population_size:
        return rng.sample(range(low, high + 1), count)
    return [rng.randint(low, high) for _ in range(count)]


def generate_robocasa_triplets(
    *,
    count: int,
    time_seed: str,
    low: int,
    high: int,
    layout_low: int,
    layout_high: int,
    style_low: int,
    style_high: int,
) -> list[tuple[int, int, int]]:
    if count <= 0:
        raise ValueError("count must be positive")

    rng = random.Random(_time_seed_to_int(time_seed))
    env_seeds = rng.sample(range(low, high + 1), count)
    layout_ids = _sample_axis(
        rng,
        count=count,
        low=layout_low,
        high=layout_high,
    )
    style_ids = _sample_axis(
        rng,
        count=count,
        low=style_low,
        high=style_high,
    )
    return list(zip(env_seeds, layout_ids, style_ids))


def format_output(
    seeds: list[int] | list[tuple[int, int, int]],
    output_format: str,
) -> str:
    is_triplets = bool(seeds) and isinstance(seeds[0], tuple)
    if output_format == "json":
        if is_triplets:
            rows = [
                {
                    "env_seed": env_seed,
                    "layout_id": layout_id,
                    "style_id": style_id,
                }
                for env_seed, layout_id, style_id in seeds
            ]
            return json.dumps(rows, indent=2)
        return json.dumps(seeds, indent=2)
    if output_format == "bash":
        if is_triplets:
            triplets = " ".join(
                f"\"{env_seed} {layout_id} {style_id}\""
                for env_seed, layout_id, style_id in seeds
            )
            return f"ROBOCASA_TRIPLETS=({triplets})"
        return "SEEDS=(" + " ".join(str(seed) for seed in seeds) + ")"
    if output_format == "csv":
        if is_triplets:
            return "\n".join(
                f"{env_seed},{layout_id},{style_id}"
                for env_seed, layout_id, style_id in seeds
            )
        return ",".join(str(seed) for seed in seeds)
    if is_triplets:
        return "\n".join(
            f"{env_seed} {layout_id} {style_id}"
            for env_seed, layout_id, style_id in seeds
        )
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
        "--robocasa-triplets",
        action="store_true",
        help="Emit env_seed layout_id style_id per line for RoboCasa evals",
    )
    parser.add_argument("--layout-low", type=int, default=DEFAULT_LAYOUT_LOW)
    parser.add_argument("--layout-high", type=int, default=DEFAULT_LAYOUT_HIGH)
    parser.add_argument("--style-low", type=int, default=DEFAULT_STYLE_LOW)
    parser.add_argument("--style-high", type=int, default=DEFAULT_STYLE_HIGH)
    parser.add_argument(
        "--format",
        choices=("text", "json", "bash", "csv"),
        default="text",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.robocasa_triplets:
        seeds = generate_robocasa_triplets(
            count=args.count,
            time_seed=args.time_seed,
            low=args.low,
            high=args.high,
            layout_low=args.layout_low,
            layout_high=args.layout_high,
            style_low=args.style_low,
            style_high=args.style_high,
        )
    else:
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
