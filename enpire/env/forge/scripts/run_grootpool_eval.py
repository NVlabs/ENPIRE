"""Run a RoboCasa365 task with GR00T inference routed through grootpool.

Minimal entry point to prove the pipeline end-to-end:
    RoboCasa365 env ──► obs ──► grootpool ──► GR00T N1.5 worker
                      ◄── action_chunk ──

Assumes:
    1. grootpool middleware is running (see tmux/launch_grootpool.sh).
    2. GROOTPOOL_ENDPOINT points at it (default tcp://127.0.0.1:7070).
    3. robocasa365 env is installed and assets are downloaded.

Example:
    # on OSMO, with middleware up on port 7070:
    uv run python scripts/run_grootpool_eval.py \\
        --env-name robocasa/PickPlaceCounterToCabinet \\
        --task-description "pick the object from the counter and place it in the cabinet" \\
        --n-episodes 1 --max-steps 300 \\
        --log-dir logs/grootpool_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _env_factory(env_name: str, *, split: str, adapter_n16: bool):
    """Create a picklable env factory.  adapter_n16=True returns the N1.6 obs-key
    remapped env; False returns the raw robocasa365 env (matches N1.5 keys)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import gymnasium as gym
    import robocasa.wrappers.gym_wrapper  # noqa: F401  — registers robocasa/* envs

    if not adapter_n16:
        return lambda: gym.make(env_name, enable_render=True, split=split)

    # N1.6 adapter is an optional optimization — skip unless requested.
    from enpire.env.forge.cap.saved_scripts.robocasa.policy_eval.gr00t._workers.robocasa365 import (
        _make_env_n16,
    )

    return lambda: _make_env_n16(env_name, split)


def main() -> int:
    p = argparse.ArgumentParser(description="RoboCasa365 × GR00T eval via grootpool")
    p.add_argument(
        "--env-name",
        required=True,
        help='robocasa365 env id, e.g. "robocasa/PickPlaceCounterToCabinet"',
    )
    p.add_argument(
        "--task-description",
        required=True,
        help="Free-text task description passed to GR00T as language conditioning.",
    )
    p.add_argument(
        "--endpoint",
        default=os.environ.get("GROOTPOOL_ENDPOINT", "tcp://127.0.0.1:7070"),
    )
    p.add_argument("--model", default="n15")
    p.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=16,
        help="Actions to execute per policy query (16=official full chunk).",
    )
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--log-dir", default="logs/grootpool_eval")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    from enpire.env.forge.cap.policy.grootpool.backend import GrootpoolBackend
    from enpire.env.forge.cap.policy import InferencePolicyConfig, inference_policy

    env_fn = _env_factory(args.env_name, split=args.split, adapter_n16=False)

    cfg = InferencePolicyConfig(
        action_horizon=16,
        replan_horizon=args.n_action_steps,
        max_episode_steps=args.max_steps,
    )

    video_dir = str(log_dir / "videos") if args.record_video else None
    task_short = args.env_name.split("/")[-1]

    print(f"endpoint:       {args.endpoint}")
    print(f"env:            {args.env_name}  split={args.split}")
    print(f"task:           {args.task_description!r}")
    print(f"episodes:       {args.n_episodes}")
    print(f"max_steps:      {args.max_steps}")

    env = env_fn()
    try:
        with GrootpoolBackend(
            task_description=args.task_description,
            model=args.model,
            endpoint=args.endpoint,
        ) as backend:
            results = inference_policy(
                env,
                backend,
                cfg,
                n_episodes=args.n_episodes,
                video_dir=video_dir,
                task_name=task_short,
                model_name=f"grootpool/{args.model}",
            )
    finally:
        env.close()

    successes = [r.success for r in results]
    rate = float(sum(successes)) / max(1, len(successes)) * 100.0
    summary = {
        "env_name": args.env_name,
        "task_description": args.task_description,
        "model": args.model,
        "endpoint": args.endpoint,
        "n_episodes": len(successes),
        "success_rate_pct": rate,
        "successes": successes,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
