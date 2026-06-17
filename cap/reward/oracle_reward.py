"""Oracle reward evaluator — attributes failure to specific sub-predicates.

Design principles (see docs/REWARD_MODULE.md):

1. **Ground truth is the anchor.** ``details["success"]`` is authoritative.
   The oracle never emits a verdict that contradicts the simulator; when
   ``success=True`` every sub-predicate is :class:`PredicateStatus.CONFIRMED_PASS`
   by definition, regardless of what our approximate thresholds would say.
2. **Thresholds are ranges.** Many sub-predicates in robocasa use
   per-instance thresholds (e.g. ``container.horizontal_radius × 0.7``) that
   vary ±2 cm across the object pool. We express each predicate's threshold
   as ``[low, high]`` and classify metrics inside that range as
   :class:`PredicateStatus.BOUNDARY` rather than forcing a PASS/FAIL verdict.
3. **Three phases, not one step.** Measurement → threshold classification →
   failure attribution. Adding a new task only requires a list of
   :class:`PredicateSpec` entries; the attribution logic is task-agnostic.

Fields consumed from ``result.json.details``:
    - ``success`` (bool), ``reward`` (float), ``obj_name`` (str)
    - ``obj_pos``                — world-frame object xyz
    - ``container_pos``          — world-frame receptacle xyz (when present)
    - ``obj_to_robot0_eef_pos``  — rigid-transform-preserving offset; its
      L2 norm equals the Euclidean eef↔obj distance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from cap.reward.evaluator import (
    PredicateOutcome,
    PredicateStatus,
    RewardEvaluator,
    SeedReward,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 — metric extractors (pure arithmetic over details)
# ---------------------------------------------------------------------------

MetricFn = Callable[[dict[str, Any]], "MetricResult"]


@dataclass(frozen=True)
class MetricResult:
    """Output of a metric extractor.

    ``primary`` is the scalar the predicate's threshold compares against;
    ``extra`` is a dict of auxiliary numbers surfaced to the reflection
    prompt for additional context (e.g. Δz between object and container).
    """

    primary: float | None
    extra: dict[str, float] = field(default_factory=dict)


def _l2(v: Any) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _l2_xy(a: Any, b: Any) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def _metric_obj_container_xy(details: dict[str, Any]) -> MetricResult:
    obj = details.get("obj_pos")
    rec = details.get("container_pos")
    if obj is None or rec is None:
        return MetricResult(None, {})
    d_xy = _l2_xy(obj, rec)
    d_z = float(obj[2]) - float(rec[2])
    return MetricResult(
        primary=d_xy,
        extra={
            "d_xy_obj_container": round(d_xy, 4),
            "d_z_obj_container": round(d_z, 4),
        },
    )


def _metric_container_z(details: dict[str, Any]) -> MetricResult:
    rec = details.get("container_pos")
    if rec is None:
        return MetricResult(None, {})
    z = float(rec[2])
    return MetricResult(primary=z, extra={"container_z": round(z, 4)})


def _metric_eef_obj_dist(details: dict[str, Any]) -> MetricResult:
    offset = details.get("obj_to_robot0_eef_pos")
    if offset is None:
        return MetricResult(None, {})
    d = _l2(offset)
    return MetricResult(primary=d, extra={"d_eef_obj": round(d, 4)})


# ---------------------------------------------------------------------------
# Phase 2 — predicate specs + threshold classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredicateSpec:
    """Declarative spec for one sub-predicate of a task's ``_check_success``.

    ``direction`` is ``'<'`` when the predicate passes with a **smaller**
    metric (``d_xy < th``) and ``'>'`` when it passes with a larger one
    (``d_eef_obj > th``). The range ``[th_low, th_high]`` is the plausible
    interval for the simulator's true threshold; a metric inside is
    :class:`PredicateStatus.BOUNDARY`.
    """

    name: str
    metric_fn: MetricFn
    th_low: float
    th_high: float
    direction: str  # "<" or ">"
    description: str
    threshold_rationale: str


def _classify(
    metric: float | None,
    spec: PredicateSpec,
    sim_success: bool,
) -> PredicateStatus:
    """Turn a raw metric + ground truth into a status.

    Anchors on ``sim_success`` first: if the simulator says success, every
    predicate is CONFIRMED_PASS — we never contradict ground truth.
    """
    if sim_success:
        return PredicateStatus.CONFIRMED_PASS
    if metric is None:
        return PredicateStatus.UNKNOWN

    if spec.direction == "<":
        # Pass when metric is small. th_low is the "definitely pass"
        # ceiling; th_high is the "definitely fail" floor.
        if metric > spec.th_high:
            return PredicateStatus.LIKELY_FAIL
        if metric < spec.th_low:
            return PredicateStatus.UNKNOWN  # looks fine, must be elsewhere
        return PredicateStatus.BOUNDARY
    if spec.direction == ">":
        # Pass when metric is large.
        if metric < spec.th_low:
            return PredicateStatus.LIKELY_FAIL
        if metric > spec.th_high:
            return PredicateStatus.UNKNOWN
        return PredicateStatus.BOUNDARY
    raise ValueError(f"bad direction {spec.direction!r} on predicate {spec.name!r}")


def _outcome_from_spec(
    spec: PredicateSpec,
    details: dict[str, Any],
    metrics: dict[str, float],
    sim_success: bool,
) -> PredicateOutcome:
    mr = spec.metric_fn(details)
    metrics.update(mr.extra)
    status = _classify(mr.primary, spec, sim_success)
    return PredicateOutcome(
        name=spec.name,
        status=status,
        metric=mr.primary,
        threshold_low=spec.th_low,
        threshold_high=spec.th_high,
        direction=spec.direction,
        threshold_rationale=spec.threshold_rationale,
        description=spec.description,
    )


# ---------------------------------------------------------------------------
# Phase 3 — failure attribution (task-agnostic over PredicateOutcomes)
# ---------------------------------------------------------------------------


#: Maps a predicate name → (a)/(b)/(c)/(d) label. Shared across pick-place
#: tasks; add entries here if new predicates are introduced.
_PREDICATE_CAUSE_LABELS: dict[str, str] = {
    "obj_in_recep": "object not in target receptacle",
    "recep_on_counter": "receptacle no longer on counter",
    "gripper_obj_far": "arm too close to object after placement",
}


def _attribute_failure(
    outcomes: list[PredicateOutcome],
    details: dict[str, Any],
    sim_success: bool,
) -> tuple[str, str]:
    """Map per-predicate statuses to a ``(category, detailed_cause)`` pair.

    ``category`` is a short stable string (``"success"``, ``"(a)"``,
    ``"(b)"``, ``"(c)"``, ``"(d)"``) suitable for cross-seed histograms.
    ``detailed_cause`` carries per-seed metrics and is used inside the
    reflection evidence block.

    Anchors on ``sim_success`` — if the simulator says success, returns
    ``("success", "success")`` regardless of what the approximate predicates
    show.
    """
    if sim_success:
        return "success", "success"

    likely = [o for o in outcomes if o.status == PredicateStatus.LIKELY_FAIL]
    boundary = [o for o in outcomes if o.status == PredicateStatus.BOUNDARY]

    def _label(o: PredicateOutcome, confidence: str) -> tuple[str, str]:
        if o.name == "gripper_obj_far":
            return "(c)", (
                f"(c) arm too close after placement "
                f"[confidence={confidence}]"
            )
        if o.name == "obj_in_recep":
            obj = details.get("obj_pos")
            rec = details.get("container_pos")
            if obj is not None and rec is not None:
                d_z = float(obj[2]) - float(rec[2])
                if d_z < -0.05:
                    return "(a)", (
                        f"(a) not picked up — object still below container "
                        f"level (Δz={d_z:+.3f} m) [confidence={confidence}]"
                    )
                return "(b)", (
                    f"(b) placed at wrong position — object moved but not "
                    f"into container [confidence={confidence}]"
                )
            return "(a)", f"(a) not picked up [confidence={confidence}]"
        if o.name == "recep_on_counter":
            return "(d)", (
                f"(d) other — container no longer on counter "
                f"[confidence={confidence}]"
            )
        return "(d)", (
            f"(d) other — {_PREDICATE_CAUSE_LABELS.get(o.name, o.name)} "
            f"[confidence={confidence}]"
        )

    if len(likely) == 1:
        return _label(likely[0], "high")
    if len(likely) > 1:
        cat, primary = _label(likely[0], "high")
        rest = ", ".join(o.name for o in likely[1:])
        return cat, f"{primary}  (also likely-failed: {rest})"
    if boundary:
        cat, primary = _label(boundary[0], "low")
        return cat, (
            f"{primary}  (in threshold range {boundary[0]._range_str()})"
        )

    return "(d)", (
        "(d) other — all tracked metrics are inside pass ranges; "
        "likely cause is a sub-predicate we cannot recover from result.json "
        "(e.g. contact check) or approximation drift at the threshold boundary"
    )


# ---------------------------------------------------------------------------
# Task specs — one list of PredicateSpec per task
# ---------------------------------------------------------------------------


#: Thresholds for ``check_obj_in_receptacle`` when the receptacle is a
#: container (plate, bowl, tray). Simulator uses
#: ``container.horizontal_radius × 0.7`` which varies with the sampled
#: instance — the range below covers the full ``obj_groups="container"``
#: pool for robocasa kitchen tasks.
_OBJ_IN_CONTAINER_TH_LOW = 0.07
_OBJ_IN_CONTAINER_TH_HIGH = 0.12
_OBJ_IN_CONTAINER_RATIONALE = (
    "robocasa uses container.horizontal_radius × 0.7 — per instance, this "
    "spans ~0.07 m (small bowl) to ~0.12 m (large plate/tray) across the "
    "obj_groups='container' pool."
)

#: Height threshold for ``recep_on_counter`` proxy (contact check is not
#: recoverable from result.json). A container that fell to the floor has
#: z ≈ 0.05; kitchen counters sit at z ≈ 0.90–0.95. Between 0.80 and 0.88
#: is in-between territory (e.g. cabinet shelf, island edge).
_RECEP_ON_COUNTER_TH_LOW = 0.80
_RECEP_ON_COUNTER_TH_HIGH = 0.88
_RECEP_ON_COUNTER_RATIONALE = (
    "no contact check in result.json — use container_pos[2] as proxy. Counter "
    "heights span ~0.90–0.95 m across robocasa layouts; below ~0.80 m almost "
    "certainly means the container dropped to the floor."
)

#: ``gripper_obj_far`` uses an exact threshold of 0.25 m — not an
#: approximation. The low/high collapse to a single point; BOUNDARY is a
#: zero-width zone here (effectively impossible).
_GRIPPER_FAR_TH = 0.25
_GRIPPER_FAR_RATIONALE = (
    "robocasa.utils.object_utils.gripper_obj_far uses th=0.25 m verbatim; "
    "no approximation needed."
)


#: Specs for ``PickPlaceSinkToCounter._check_success`` (line 463).
SINK_TO_COUNTER_SPECS: list[PredicateSpec] = [
    PredicateSpec(
        name="obj_in_recep",
        metric_fn=_metric_obj_container_xy,
        th_low=_OBJ_IN_CONTAINER_TH_LOW,
        th_high=_OBJ_IN_CONTAINER_TH_HIGH,
        direction="<",
        description="horizontal distance obj → container",
        threshold_rationale=_OBJ_IN_CONTAINER_RATIONALE,
    ),
    PredicateSpec(
        name="recep_on_counter",
        metric_fn=_metric_container_z,
        th_low=_RECEP_ON_COUNTER_TH_LOW,
        th_high=_RECEP_ON_COUNTER_TH_HIGH,
        direction=">",
        description="container z-height (proxy for counter-contact)",
        threshold_rationale=_RECEP_ON_COUNTER_RATIONALE,
    ),
    PredicateSpec(
        name="gripper_obj_far",
        metric_fn=_metric_eef_obj_dist,
        th_low=_GRIPPER_FAR_TH,
        th_high=_GRIPPER_FAR_TH,
        direction=">",
        description="end-effector distance from object",
        threshold_rationale=_GRIPPER_FAR_RATIONALE,
    ),
]

#: Specs for ``PickPlaceCounterToSink._check_success`` (line 352).
#: Simulator passes ``th=0.07`` explicitly — no range needed, but we still
#: express it as a near-zero range for consistency.
COUNTER_TO_SINK_SPECS: list[PredicateSpec] = [
    PredicateSpec(
        name="obj_in_recep",
        metric_fn=_metric_obj_container_xy,
        th_low=0.07,
        th_high=0.07,
        direction="<",
        description="horizontal distance obj → sink-basin receptacle",
        threshold_rationale="simulator uses th=0.07 verbatim (not instance-dependent)",
    ),
    PredicateSpec(
        name="gripper_obj_far",
        metric_fn=_metric_eef_obj_dist,
        th_low=_GRIPPER_FAR_TH,
        th_high=_GRIPPER_FAR_TH,
        direction=">",
        description="end-effector distance from object",
        threshold_rationale=_GRIPPER_FAR_RATIONALE,
    ),
]

#: Specs for ``PickPlaceCounterToMicrowave._check_success`` (line 586).
COUNTER_TO_MICROWAVE_SPECS: list[PredicateSpec] = [
    PredicateSpec(
        name="obj_in_recep",
        metric_fn=_metric_obj_container_xy,
        th_low=0.07,
        th_high=0.07,
        direction="<",
        description="horizontal distance obj → container inside microwave",
        threshold_rationale="simulator uses th=0.07 verbatim",
    ),
    PredicateSpec(
        name="gripper_obj_far",
        metric_fn=_metric_eef_obj_dist,
        th_low=_GRIPPER_FAR_TH,
        th_high=_GRIPPER_FAR_TH,
        direction=">",
        description="end-effector distance from object",
        threshold_rationale=_GRIPPER_FAR_RATIONALE,
    ),
]

#: Specs for ``PickPlaceMicrowaveToCounter._check_success`` (line 712).
MICROWAVE_TO_COUNTER_SPECS: list[PredicateSpec] = SINK_TO_COUNTER_SPECS  # identical shape


#: Task class name → spec list. Populate for each new task by reading the
#: matching ``_check_success`` method in robocasa's source.
TASK_SPECS: dict[str, list[PredicateSpec]] = {
    "PickPlaceSinkToCounter": SINK_TO_COUNTER_SPECS,
    "PickPlaceCounterToSink": COUNTER_TO_SINK_SPECS,
    "PickPlaceCounterToMicrowave": COUNTER_TO_MICROWAVE_SPECS,
    "PickPlaceMicrowaveToCounter": MICROWAVE_TO_COUNTER_SPECS,
}


def _normalize_task(env_name: str) -> str:
    """Extract the robocasa task class name from an env_name string.

    Examples::

        "robocasa:PickPlaceSinkToCounter"            -> "PickPlaceSinkToCounter"
        "robocasa:PickPlaceSinkToCounter:PandaOmron" -> "PickPlaceSinkToCounter"
        "PickPlaceSinkToCounter"                     -> "PickPlaceSinkToCounter"
    """
    if not env_name:
        return ""
    parts = env_name.split(":")
    if len(parts) == 1:
        return parts[0]
    if parts[0].lower() == "robocasa":
        return parts[1] if len(parts) >= 2 else ""
    return parts[-1]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class OracleRewardEvaluator(RewardEvaluator):
    """Three-phase oracle: measure → classify → attribute.

    Ground truth (``details["success"]``) is the anchor. Predicate status
    reflects attribution confidence, never a second opinion on success.
    """

    name = "oracle"

    def __init__(self, task: str) -> None:
        self._task = _normalize_task(task)
        self._specs: list[PredicateSpec] | None = TASK_SPECS.get(self._task)
        if self._specs is None and self._task:
            logger.warning(
                "OracleRewardEvaluator: no spec for task %r — falling back "
                "to binary success. Add an entry to TASK_SPECS in "
                "cap/reward/oracle_reward.py.",
                self._task,
            )

    @property
    def task(self) -> str:
        return self._task

    @property
    def has_recipe(self) -> bool:
        return self._specs is not None

    def evaluate(
        self,
        details: dict[str, Any],
        seed: int,
        exec_id: int = 0,
    ) -> SeedReward:
        obj_name = str(details.get("obj_name", "") or "")
        sim_success = bool(details.get("success", False))
        score = float(details.get("reward", 0.0))

        if self._specs is None:
            return SeedReward(
                seed=seed,
                exec_id=exec_id,
                success=sim_success,
                score=score,
                obj_name=obj_name,
                note=f"no oracle spec registered for task {self._task!r}",
            )

        metrics: dict[str, float] = {}
        outcomes: list[PredicateOutcome] = []
        try:
            for spec in self._specs:
                outcomes.append(
                    _outcome_from_spec(spec, details, metrics, sim_success)
                )
            category, cause = _attribute_failure(outcomes, details, sim_success)
        except Exception as e:
            logger.warning(
                "OracleRewardEvaluator.evaluate failed for seed=%d task=%r: %s",
                seed,
                self._task,
                e,
            )
            return SeedReward(
                seed=seed,
                exec_id=exec_id,
                success=sim_success,
                score=score,
                obj_name=obj_name,
                note=f"oracle spec raised: {e}",
            )

        return SeedReward(
            seed=seed,
            exec_id=exec_id,
            success=sim_success,
            score=score,
            obj_name=obj_name,
            predicates=outcomes,
            metrics=metrics,
            failure_category=category,
            failure_cause=cause,
        )

    def as_markdown(self, reward: SeedReward) -> str:
        """Override to expose threshold rationale when status is non-trivial."""
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
            # Only annotate with rationale when the status is non-trivial
            # (BOUNDARY or LIKELY_FAIL), to keep successful-seed blocks tight.
            if p.status in (PredicateStatus.BOUNDARY, PredicateStatus.LIKELY_FAIL):
                if p.threshold_rationale:
                    lines.append(f"      _{p.threshold_rationale}_")
        if reward.failure_cause:
            lines.append(f"**Classified failure:** {reward.failure_cause}")
        if reward.note:
            lines.append(f"_{reward.note}_")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Back-compat shim for the old recipe-function interface (TASK_RECIPES)
# ---------------------------------------------------------------------------

#: Legacy alias — callers that imported ``TASK_RECIPES`` still get a mapping,
#: but each value is now a lightweight closure that runs the spec list
#: through the evaluator. Prefer ``TASK_SPECS`` directly.
def _legacy_recipe_for(task: str) -> Callable[[dict, int, int], SeedReward]:
    def _run(details: dict[str, Any], seed: int, exec_id: int) -> SeedReward:
        return OracleRewardEvaluator(task=task).evaluate(details, seed, exec_id)

    return _run


TASK_RECIPES: dict[str, Callable[[dict, int, int], SeedReward]] = {
    task: _legacy_recipe_for(task) for task in TASK_SPECS
}
