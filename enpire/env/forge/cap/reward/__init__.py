"""enpire.env.forge.cap.reward — two layers share this package.

- **RL reward servers** (``gemini_reward``, ``smolvlm_reward``,
  ``reward_server``, ``reward_client``, ``serve_reward``) — per-step scalar
  reward signals served over a socket, consumed by the RL training loop.
- **Task-success evaluators** (``evaluator``, ``oracle_reward``) — structured
  predicate reconstruction from ``result.json.details`` for the agent loop's
  reflection stage. Imported on demand via :func:`build_reward_evaluator`.

The two layers are deliberately kept in the same package but are otherwise
independent. This module exports only the task-success-evaluator API; RL
reward-server modules are imported by their full path.
"""

from enpire.env.forge.cap.reward.evaluator import (
    NoopRewardEvaluator,
    PredicateOutcome,
    PredicateStatus,
    RewardEvaluator,
    SeedReward,
    build_reward_evaluator,
)

__all__ = [
    "NoopRewardEvaluator",
    "PredicateOutcome",
    "PredicateStatus",
    "RewardEvaluator",
    "SeedReward",
    "build_reward_evaluator",
]
