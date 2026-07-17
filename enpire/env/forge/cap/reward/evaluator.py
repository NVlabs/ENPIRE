"""Task-success reward evaluators.

A ``RewardEvaluator`` takes a seed's ``result.json`` details and returns a
structured :class:`SeedReward` with per-predicate outcomes, metrics, and a
human-readable failure category. Runs *before* reflection so its diagnostics
can be injected into both the per-seed VLM prompt (Phase A) and the
cross-seed synthesis prompt (Phase B).

Not to be confused with ``cap.reward.gemini_reward`` et al., which are
**RL policy reward servers** (VLM-judged per-step scalar rewards for a
training loop). The module here is purely an analysis layer over the
agent loop's existing binary `success` flag — it reconstructs the
sub-predicates that the robocasa `_check_success` method would have
computed so that the reflection LLM sees ground-truth reasons instead of
guessing from pixels.

Public entry points::

    from enpire.env.forge.cap.reward import build_reward_evaluator
    evaluator = build_reward_evaluator(reward_cfg, env_name=env_cfg.name)
    seed_reward = evaluator.evaluate(details, seed=0, exec_id=0)
    md = evaluator.as_markdown(seed_reward)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class PredicateStatus(str, Enum):
    """Four-state classification of a sub-predicate's outcome.

    The oracle never contradicts the simulator. ``details["success"]`` is
    the only ground truth; per-predicate status reflects our **attribution**
    confidence, not a second source of truth.

    - ``CONFIRMED_PASS`` — simulator reports ``success=True``; by definition
      every sub-predicate in the conjunction passed, regardless of what our
      approximate threshold would say.
    - ``LIKELY_FAIL`` — simulator reports ``success=False`` AND the metric
      is clearly beyond the plausible threshold range. High confidence this
      predicate is the cause.
    - ``BOUNDARY`` — simulator reports ``success=False`` AND the metric is
      inside the plausible threshold range (could be this predicate or a
      different one). Low confidence.
    - ``UNKNOWN`` — simulator reports ``success=False`` but this predicate's
      metric looks fine (well past any plausible threshold on the pass side).
      The cause lies elsewhere — typically a contact check we can't recover
      from ``result.json``.
    """

    CONFIRMED_PASS = "confirmed_pass"
    LIKELY_FAIL = "likely_fail"
    BOUNDARY = "boundary"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PredicateOutcome:
    """One sub-predicate's classified outcome for a single seed.

    Thresholds are expressed as a **range** ``[low, high]`` instead of a
    single point, because the simulator's true threshold is often per-
    instance (e.g. ``container.horizontal_radius × 0.7``) and we only
    know its plausible spread. A metric inside the range is BOUNDARY.
    """

    name: str
    status: PredicateStatus
    metric: float | None = None
    threshold_low: float | None = None
    threshold_high: float | None = None
    direction: str = ""  # "<" → pass when metric < th; ">" → pass when metric > th
    threshold_rationale: str = ""
    description: str = ""

    @property
    def passed(self) -> bool:
        """Backward compat: truthy only for ``CONFIRMED_PASS``."""
        return self.status == PredicateStatus.CONFIRMED_PASS

    def _range_str(self) -> str:
        tl, th = self.threshold_low, self.threshold_high
        if tl is None and th is None:
            return ""
        if tl is not None and th is not None and tl == th:
            return f"{tl:.3f}"
        if tl is not None and th is not None:
            return f"{tl:.3f}–{th:.3f}"
        if tl is not None:
            return f">={tl:.3f}"
        return f"<={th:.3f}"

    def as_line(self) -> str:
        parts = [f"{self.name}: {self.status.value.upper()}"]
        if self.metric is not None:
            parts.append(f"(metric={self.metric:.3f}")
            rng = self._range_str()
            if rng:
                parts[-1] += f", {self.direction or '~'} {rng}"
            parts[-1] += ")"
        if self.description:
            parts.append(f"— {self.description}")
        return " ".join(parts)


@dataclass
class SeedReward:
    """Structured ground-truth diagnosis for one seed."""

    seed: int
    exec_id: int
    success: bool
    score: float
    obj_name: str = ""  # the specific instance sampled this seed (e.g. "potato")
    predicates: list[PredicateOutcome] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    failure_category: str = ""  # stable: "success" | "(a)" | "(b)" | "(c)" | "(d)"
    failure_cause: str = ""  # detailed text with metrics — for per-seed evidence
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "predicates": [asdict(p) for p in self.predicates],
        }

    def failed_predicates(self) -> list[PredicateOutcome]:
        return [p for p in self.predicates if not p.passed]


class RewardEvaluator(ABC):
    """Abstract interface for all reward evaluators."""

    #: Identifier used in prompts and logs.
    name: str = "reward"

    @abstractmethod
    def evaluate(
        self,
        details: dict[str, Any],
        seed: int,
        exec_id: int = 0,
    ) -> SeedReward:
        """Compute the structured diagnosis for one seed."""

    def as_markdown(self, reward: SeedReward) -> str:
        """Render a human-readable block for injection into reflection prompts."""
        header = (
            f"**Ground-truth predicates** (seed {reward.seed}, "
            f"success={reward.success}, score={reward.score:.3f}"
        )
        if reward.obj_name:
            header += f", obj_name={reward.obj_name!r}"
        header += "):"
        lines = [header]
        for p in reward.predicates:
            lines.append(f"  - {p.as_line()}")
        if reward.failure_cause:
            lines.append(f"**Classified failure:** {reward.failure_cause}")
        if reward.note:
            lines.append(f"_{reward.note}_")
        return "\n".join(lines)


class NoopRewardEvaluator(RewardEvaluator):
    """Pass-through evaluator that emits only the binary success flag."""

    name = "noop"

    def evaluate(
        self, details: dict[str, Any], seed: int, exec_id: int = 0
    ) -> SeedReward:
        return SeedReward(
            seed=seed,
            exec_id=exec_id,
            success=bool(details.get("success", False)),
            score=float(details.get("reward", 0.0)),
            obj_name=str(details.get("obj_name", "") or ""),
        )


def build_reward_evaluator(
    reward_cfg: Any,
    env_name: str | None = None,
) -> RewardEvaluator:
    """Construct an evaluator from a :class:`RewardConfig` dataclass.

    Supports::

        reward:
          evaluator: oracle   # or "noop" / "none"
          task: null          # defaults to env_name when None
    """
    kind = getattr(reward_cfg, "evaluator", "oracle") if reward_cfg else "oracle"
    if kind in ("none", "noop", "off", None):
        return NoopRewardEvaluator()
    if kind == "oracle":
        from enpire.env.forge.cap.reward.oracle_reward import OracleRewardEvaluator

        task = getattr(reward_cfg, "task", None) or env_name or ""
        return OracleRewardEvaluator(task=task)
    raise ValueError(f"Unknown reward evaluator: {kind!r}")
