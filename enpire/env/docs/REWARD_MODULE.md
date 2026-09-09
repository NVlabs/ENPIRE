# Reward Module — Task-success Evaluators

> Paths written as `cap/…`, `robot/…`, `experimental/…`, `tmux/…` or
> `experiments/…` are relative to `enpire/env/forge/`.
`cap/reward/` hosts **two independent layers** that share a package name:

1. **RL reward servers** (`gemini_reward.py`, `smolvlm_reward.py`, `reward_server.py`, `reward_client.py`, `serve_reward.py`) — per-step scalar-reward servers for the RL training loop. Not covered here.
2. **Task-success evaluators** (`evaluator.py`, `oracle_reward.py`) — structured predicate **attribution** from `result.json.details` for the agent loop's reflection stage. **This doc.**

## Design invariants

Three principles drive the design:

1. **Ground truth is sacred.** `details["success"]` is authoritative. The oracle never contradicts the simulator. When `success=True`, every sub-predicate is `CONFIRMED_PASS` regardless of what our approximation would say — because by definition the simulator's conjunction passed.
2. **Thresholds are ranges, not points.** Most task sub-predicates have per-instance thresholds that vary across the object pool (e.g. receptacle radius × 0.7 spans ±2 cm depending on container size). We carry `[th_low, th_high]` and classify metrics inside as `BOUNDARY`, not `FAIL`.
3. **Three phases, not one step.** Measurement → threshold classification → failure attribution. Adding a new task is a declarative spec list; the attribution logic is task-agnostic.

## What it does

Each task defines success as a conjunction of sub-predicates. Example — a YAM pick-and-place task:

```python
# success = object in receptacle AND gripper released AND arm clear
obj_in_recep   = check_obj_in_receptacle(obj_pos, container_pos, radius)
gripper_obj_far = eef_dist_to_obj > FAR_THRESHOLD
arm_safe        = eef_z > MIN_RETRACT_HEIGHT
return obj_in_recep and gripper_obj_far and arm_safe
```

When the agent observes `success=False`, *which* predicate failed is the crucial signal. The VLM can't always see it (e.g. the arm being 12 cm from the object vs 25 cm looks identical in the frame), but every needed field is already in `result.json.details`. The oracle **attributes** the failure to the most likely sub-predicate, with confidence annotation.

### PredicateStatus

Each `PredicateOutcome` carries one of four statuses:

| Status | Meaning |
|---|---|
| `CONFIRMED_PASS` | Simulator reports `success=True`; this predicate passed by definition. |
| `LIKELY_FAIL` | Simulator reports `success=False` **and** metric is clearly beyond the plausible threshold range. High-confidence cause. |
| `BOUNDARY` | Simulator reports `success=False` and metric is inside the plausible threshold range. Low-confidence cause (could be this predicate or another). |
| `UNKNOWN` | Simulator reports `success=False` but this predicate's metric looks fine. Cause is elsewhere (typically the contact check we can't recover). |

### Example output (failed seed, `(a) not picked up`)

```
**Ground-truth predicates** (seed 1, success=False, score=0.000, obj_name='cube'):
  - obj_in_recep: LIKELY_FAIL (metric=0.564, < 0.070–0.120) — horizontal distance obj → container
      _threshold spans ~0.07 m (small container) to ~0.12 m (large tray) across the object pool._
  - arm_safe:     UNKNOWN (metric=0.929, > 0.800–0.880) — end-effector z height above table
  - gripper_obj_far: UNKNOWN (metric=0.600, > 0.250) — end-effector distance from object
**Classified failure:** (a) not picked up — object still below container level (Δz=-0.170 m) [confidence=high]
```

### Example output (successful seed)

```
**Ground-truth predicates** (seed 0, success=True, score=1.000, obj_name='potato'):
  - obj_in_recep:     CONFIRMED_PASS (metric=0.008, < 0.070–0.120)
  - recep_on_counter: CONFIRMED_PASS (metric=0.927, > 0.800–0.880)
  - gripper_obj_far:  CONFIRMED_PASS (metric=0.724, > 0.250)
**Classified failure:** success
```

Note: threshold rationale is only surfaced on non-trivial statuses (`BOUNDARY`, `LIKELY_FAIL`) to keep successful-seed blocks compact.

## Pipeline integration

- **Where it runs:** `SubprocessExecutorStep` Phase 4.5 — *after* execution, *before* Phase A per-seed VLM reflection.
- **Persisted artifact:** `iter_NNN/reward_diagnostics.json`.
- **Injected into Phase A prompt:** the per-seed VLM call gets a "Ground-truth predicates" block and is told to treat it as authoritative.
- **Injected into Phase B prompt:** `_build_per_seed_evidence()` in `cap/agent/reflection.py` renders the predicate block into each seed's evidence chunk; `_cross_seed_stats()` adds a `failure_histogram` line to the prompt header (e.g. `(c) arm too close × 18; (b) wrong position × 4`).

## Configuration

Selected by Hydra group `reward=`:

```yaml
# experiments/reward/oracle.yaml  (default — wired into experiments/config.yaml)
reward:
  evaluator: "oracle"
  task: null                      # null → use cfg.env.name
  inject_into_per_seed_vlm: true
```

Disable with `reward=none`.

## Adding a new task spec

The redesigned module is **declarative**. A task is just a list of `PredicateSpec` — no recipe function needed.

1. Define the task's success conditions as a conjunction of measurable sub-predicates. Note each predicate's metric source (robot state, vision estimate, contact flag) and its threshold. Thresholds that vary across the object pool should be expressed as a range `[th_low, th_high]`.
2. Compose a `list[PredicateSpec]` with one entry per sub-predicate:
   ```python
   MY_TASK_SPECS = [
       PredicateSpec(
           name="obj_in_recep",
           metric_fn=_metric_obj_container_xy,
           th_low=0.07, th_high=0.12,   # range, not point
           direction="<",
           description="horizontal distance obj → container",
           threshold_rationale="why this range covers reality",
       ),
       # … more sub-predicates
   ]
   ```
3. Register in `TASK_SPECS` under a task name key (e.g. `"pusht"`, `"pick_place"`). No code changes beyond the spec list — classification and attribution are task-agnostic.

If a sub-predicate needs a metric we don't have (e.g. a contact check), either forward it from the env layer into `result.json.details`, or omit it from the spec and let attribution fall through to `UNKNOWN` / `(d) other`. The framework handles the resulting ambiguity honestly rather than guessing.

## Extending beyond oracle

Add another subclass of `RewardEvaluator` (e.g. `VlmRewardEvaluator`, `GeometricRewardEvaluator`) and select it in `experiments/reward/<name>.yaml`. The evaluator's `as_markdown()` output is all that the pipeline consumes; everything downstream treats it as an opaque string.

## Key files

| File | Purpose |
|---|---|
| `cap/reward/__init__.py` | Re-exports `build_reward_evaluator`, `RewardEvaluator`, `SeedReward`, `PredicateOutcome`, `PredicateStatus` |
| `cap/reward/evaluator.py` | Base types, `PredicateStatus` enum, `NoopRewardEvaluator`, factory |
| `cap/reward/oracle_reward.py` | Three-phase oracle (measure → classify → attribute); `PredicateSpec`, `TASK_SPECS` registry, per-task spec lists |
| `cap/agent/agent_config.py` | `RewardConfig` dataclass |
| `cap/agent/agent_step.py` | `SubprocessExecutorStep` Phase 4.5 wiring + Phase A VLM prompt injection |
| `cap/agent/reflection.py` | `_build_per_seed_evidence` / `_cross_seed_stats` surface diagnostics in Phase B (histogram keys on `failure_category`) |
| `cap/prompt/system/cross_seed_reflection.md` | Phase B prompt template — references predicates + failure_histogram + obj_breakdown |
| `experiments/reward/oracle.yaml` | Default Hydra config group |
