# Beam Search Agent Loop

> **Cross-references**: [AGENT_LOOP_REDESIGN](../AGENT_LOOP_REDESIGN.md) | [two_stage_code_generator](two_stage_code_generator.md) | [AGENT_PIPELINE_DESIGN](../AGENT_PIPELINE_DESIGN.md)

## 1. Motivation

The current loop is pure refinement: every iteration attempts to improve the
same tool strategy the LLM first picked. Three mechanisms keep it anchored
there — `ExecutionMemory` shows the LLM which tools it used before, the
champion surfaces a specific tool pattern as the template, and reflection
diagnoses failures *within* the chosen strategy rather than questioning it.

From `AGENT_LOOP_REDESIGN.md` (evidence runs T071845 / T092735):
> `vertical_grasp_v1` (sr=1.0, never the bottleneck) was refined 7 times.
> `lift_v1` (sr=0.31, the actual bottleneck) was never targeted.

The three fixes in `AGENT_LOOP_REDESIGN.md` improve refinement reliability.
They do not address the prior question: was the right tool strategy chosen?

Running K LLM samples with temperature alone does not help — they converge on
K variants of the same strategy. Genuine exploration requires **explicitly
conditioning each assembly on a structurally different tool plan** before code
generation begins.

---

## 2. Design

One new step (`StrategyPlannerStep`), one modified step (`AssemblyGeneratorStep`
called K times in parallel), and a small extension to `SubprocessExecutorStep`'s
job list. The dispatch machinery inside the executor — GPU semaphores,
`ThreadPoolExecutor`, process group killing, timeout — is **not touched**.

```
observer
    ↓
SkillAuthorStep          (once, exactly as today)
    ↓
StrategyPlannerStep      (NEW — generates K distinct tool strategies)
    ↓
AssemblyGeneratorStep×K  (parallel LLM calls, each conditioned on strategy_k)
    → code_0.py, code_1.py, ..., code_{K-1}.py
    ↓
SubprocessExecutorStep   (job queue extended to K×N pairs — see §4)
    → per_job results grouped by beam_id
    ↓
pick winner by score     → ctx.code = winner_code, ctx.evaluation = winner_result
    ↓
SelfReflectionStep       (unchanged, runs on winner)
```

---

## 3. StrategyPlannerStep

This is the only genuinely new component.

### 3.1 What it produces

Given the task, the current skill library index, and the prior iteration's
failure history, it generates K short natural language strategy prompts — one
per beam. No structured output, no tool catalog validation.

Example output for a pick-and-place task:

```
Strategy 0: "Use the high-level pick_object primitive directly. Trust it to
handle grasp internally rather than decomposing into detect + move + grasp."

Strategy 1: "Get fresh object coordinates from detect_object first, then use
freespace_move with explicit target positions for fine-grained control over
the grasp approach angle."

Strategy 2: "Assume the first grasp attempt may fail. Build in a nudge +
regrasp loop as the primary mechanism rather than as a fallback."
```

### 3.2 Diversity constraint

The planner prompt instructs the model to name one anchor tool per strategy
and to use a different anchor tool for each beam. A one-line string check
verifies the K anchor tool names are distinct before accepting the output.
On collision, re-prompt (≤2 retries). That is the entire validator.

### 3.3 Strategy conditioning in AssemblyGeneratorStep

Each parallel assembly call receives its strategy as a single prepended line
in the system prompt:

```
Strategy for this beam: Use the high-level pick_object primitive directly.
Trust it to handle grasp internally rather than decomposing into detect + move + grasp.
```

The existing assembly prompt and validation rules are otherwise unchanged.

---

## 4. Executor: job queue extension

The current executor builds a flat job list and dispatches it through a
GPU semaphore queue:

```python
# today
seeds = [(seed_start + i, i) for i in range(n_seeds)]
# → ThreadPoolExecutor submits N jobs, each running _run_seed(seed, i, gpu_slot)
# → gpu_slot = i % n_render  (round-robin across GPUs)
# → _gpu_sem[gpu_slot] limits concurrent jobs per GPU
```

The beam extension replaces `seeds` with a flat list of `(code_path, seed,
beam_id, exec_id)` tuples — K×N items instead of N:

```python
# beam extension
jobs = [
    (code_paths[k], seed_row, k, k * n_seeds + j)
    for k in range(n_codes)
    for j, seed_row in enumerate(selected_seeds)
]
# → same ThreadPoolExecutor, same _gpu_sem, same _run_with_gpu_queue
# → gpu_slot = job_index % n_render  (unchanged round-robin)
```

`_build_cmd` gains one parameter — `code_path: Path` — instead of closing
over the outer `code_path`. `exec_dir` is parameterised as
`iter_NNN/beam_k/exec_j` instead of `iter_NNN/exec_j`. Everything else in
`_run_seed`, `_kill_process_group`, the semaphore logic, timeout handling, and
result collection is unchanged.

After all K×N jobs finish, group results by `beam_id` and aggregate per beam:

```python
from itertools import groupby
by_beam = {k: [r for r in per_job if r["beam_id"] == k] for k in range(n_codes)}
beam_scores = {
    k: (
        sum(r["success"] for r in rs) / len(rs),   # success_rate
        sum(r["score"]   for r in rs) / len(rs),   # avg_score
    )
    for k, rs in by_beam.items()
}
winner_k = max(beam_scores, key=lambda k: beam_scores[k])
```

`ctx.code` and `ctx.evaluation` are set from the winner beam, exactly as
today. The downstream steps (reflection, snapshot, wandb logging) see no
difference.

---

## 5. Skill library stays simple

`SkillAuthorStep` runs once per iteration, as today. It writes to the shared
`skill_library/` before the parallel assembly phase begins. All K assemblies
read from the same library — no isolation, no forks, no merges needed.

The constraint: assembly-level diversity (how skills are combined and ordered)
rather than skill-level diversity (different skills per beam). This is
acceptable for a first version — most of the unexplored space lies in
assembly strategy, not in which skills exist.

If skill-level diversity becomes necessary later, `SkillAuthorStep` can be run
K times sequentially (each conditioned on its strategy, writing to a temp
directory), with only the winner's skills promoted. That is a follow-on
change and does not affect the rest of this design.

---

## 6. Directory layout

Minimal extension of the existing layout:

```
run_dir/
  iter_NNN/
    strategies.json           ← K strategies from StrategyPlannerStep (NEW)
    beam_scores.json          ← per-beam (score, success_rate, winner) (NEW)
    beam_000/
      code.py                 ← from AssemblyGeneratorStep beam 0
      exec_000/ … exec_NNN/   ← from SubprocessExecutorStep (unchanged paths)
    beam_001/
      code.py
      exec_000/ … exec_NNN/
    code.py                   ← symlink or copy of winner's code.py (as today)
    result.json               ← winner's aggregated result (as today)
```

Reflection and downstream logging consume `iter_NNN/code.py` and
`iter_NNN/result.json` exactly as today. The `beam_NNN/` subdirectories are
for debugging only.

---

## 7. Files and changes

### 7.1 New files

| Path | Purpose |
|------|---------|
| `cap/agent/strategy_planner_step.py` | `StrategyPlannerStep` class |
| `cap/prompt/system/strategy_planner.md` | Planner prompt |

### 7.2 Files that change

| Path | Change |
|------|--------|
| `cap/agent/agent_step.py` | (1) `AssemblyGeneratorStep.run()` accepts optional `strategy` arg + parallel-call helper; (2) `SubprocessExecutorStep`: job list extended to `(code_path, seed, beam_id, exec_id)` tuples, `_build_cmd` gains `code_path` param, `exec_dir` parameterised by beam, post-exec grouping by `beam_id` |
| `cap/agent/agent_pipeline.py` | `from_config` inserts `StrategyPlannerStep` before `AssemblyGeneratorStep` when `beam_search.enabled` |
| `cap/agent/agent_config.py` | Add `BeamSearchConfig(enabled, k_beams, planner_model, planner_max_retries)` |
| `cap/agent/agent_session.py` | `save_iteration()` writes `strategies.json` and `beam_scores.json`; `exec_dir()` accepts optional `beam_id` |
| `experiments/experiment/pick_place_sink_to_counter.yaml` | Add `beam_search:` block (see §8) |

### 7.3 Files not touched

`SkillAuthorStep`, `SelfReflectionStep`, `AgentContext`, `SkillLibrary`,
snapshot/promotion helpers, GPU semaphore dispatch logic — unchanged.

---

## 8. Config shape

```yaml
# experiments/experiment/pick_place_sink_to_counter.yaml

beam_search:
  enabled: true
  k_beams: 3
  planner_model: gemini-flash   # fast/cheap; planner call is small
  planner_max_retries: 2

# Everything below unchanged
steps:
  - observer
  - skill_author
  - assembly_generator   # pipeline runs this K times when beam_search.enabled
  - executor
  - self_reflection

skill_author:
  max_skills_per_iteration: 1
```

---

## 9. History integration

The existing `history.md` mechanism (`fix_history_and_escalation.md`) writes
one LLM paragraph per iteration summarising what was tried and what scored.
`SkillAuthorStep` reads it; stagnation triggers an escalation prompt.

Beam search changes the role of history from a **refinement log** (tracking
one trajectory) into an **exploration map** (recording what every tool strategy
learned).

### 9.1 All beams write history entries

Currently only the winner calls `_write_history_entry`. Under beam search,
every beam writes an entry — labelled `winner` or `explored` instead of
`keep/discard`:

```markdown
## iter_002 — beam_0 — pick_object — score=0.625 — winner
<history>Used pick_object directly. 5/8 seeds succeeded. Remaining 3 seeds
fail at approach angle for objects near the sink edge.</history>

## iter_002 — beam_1 — detect_object — score=0.375 — explored
<history>Used detect_object + freespace_move with explicit coordinates. 3/8
seeds. IK_Failed at grasp descent in 5 seeds — the explicit coordinate path
hits a feasibility boundary that pick_object avoids internally.</history>

## iter_002 — beam_2 — nudge — score=0.125 — explored
<history>Used nudge + regrasp as the primary mechanism. 1/8 seeds. Object
moved unpredictably during nudge in most seeds.</history>
```

`_write_history_entry` is called K times per iteration with beam-specific
context (`beam_id`, `strategy`, `score`, `code`). No other change to the
function.

### 9.2 StrategyPlannerStep reads history

Currently history is only injected into `SkillAuthorStep`. The planner is
added as a second reader. With the full exploration map visible, it can:

- Avoid re-assigning a strategy whose anchor tool already appears as `explored`
  in recent iterations
- Build on partial successes: "detect_object scored 0.375 but hit IK at
  descent — try a shallower approach angle this iteration"
- Deprioritise strategies that have failed across multiple iterations

The planner prompt receives the same `=== EXPERIMENT HISTORY ===` block that
`SkillAuthorStep` already receives.

### 9.3 `explored` skill status

History and the skill library tell the same story at different granularities.
Skills authored by non-winning beams that were actually executed get a new
`explored` status in `index.json`:

```json
{
  "name": "detect_and_grasp_v1",
  "status": "explored",
  "score": 0.375,
  "source_beam": 1,
  "iteration": 2,
  "failure_summary": "IK_Failed at grasp descent in 5/8 seeds"
}
```

`explored` skills are **not active** — the assembly generator will not import
them by default. But `SkillAuthorStep` can see them and explicitly `refine`
one into `pending` if the strategy planner assigns that tool region again.
This prevents the next beam that tries `detect_object` from starting from
scratch.

Lifecycle states with this addition:

| Status | Meaning |
|--------|---------|
| `pending` | Authored, not yet executed |
| `verified` | Executed, success rate above threshold |
| `deprecated` | Executed, consistently failed |
| `explored` | Executed by a non-winning beam; partial signal; available for refinement |

### 9.4 Escalation changes

The current escalation fires when N consecutive iterations all `discard` on
a single trajectory. The beam-search equivalent: **all K beams scored below
the current baseline for M consecutive iterations**.

```python
def _count_all_beams_stagnant(history_text: str, k_beams: int) -> int:
    """Count consecutive iterations where no beam beat the baseline."""
    count = 0
    for block in _iter_iterations(history_text):  # newest first
        if any("winner" in entry for entry in block):
            break  # a beam improved — reset
        if len([e for e in block if "explored" in e or "winner" in e]) == k_beams:
            count += 1  # full iteration, no winner beat baseline
    return count
```

When this fires, the escalation message targets the planner, not just
`SkillAuthorStep`:

```
⚠ ESCALATION: All K strategies tried in the last M iterations failed to
improve the baseline. Do NOT re-assign any anchor tool from those iterations.
Propose K strategies using tool families not yet explored, or combine
previously explored tools in a qualitatively different order.
```

---

## 10. Does this fundamentally solve exploitation-only?

Yes, provided `StrategyPlannerStep` produces genuinely diverse strategies.

| | Current loop | This design |
|--|-------------|-------------|
| Tool strategy per iter | One (LLM prior) | K (planner-forced) |
| Source of diversity | Temperature noise (weak) | Explicit tool plans (strong) |
| History role | Refinement log (one trajectory) | Exploration map (all K beams) |
| Explored tool knowledge | Discarded | Retained via `explored` status + history |
| Escalation trigger | N consecutive discards | All K beams stagnant for M iterations |
| Executor | Job queue extended to K×N | — |
| Reflection | Runs on winner | — |

---

## 11. What this does not address

- **Skill-level diversity**: all K assemblies draw from the same skill library.
  See §5 for the follow-on path if needed.
- **The three fixes from AGENT_LOOP_REDESIGN.md**: Fix 1 (first-failure
  attribution), Fix 2 (one-skill-per-iteration), Fix 3 (clean baseline) apply
  unchanged and are not superseded.
- **Stochastic seed variance**: K beams are scored on the same N seeds for fair
  comparison. The significance floor concern from `AGENT_LOOP_REDESIGN.md` is
  unchanged.

---

## 12. Files and changes

### 12.1 New files

| Path | Purpose |
|------|---------|
| `cap/agent/strategy_planner_step.py` | `StrategyPlannerStep` class |
| `cap/prompt/system/strategy_planner.md` | Planner prompt (reads history, outputs K strategy texts) |

### 12.2 Files that change

| Path | Change |
|------|--------|
| `cap/agent/agent_step.py` | (1) `AssemblyGeneratorStep.run()` accepts optional `strategy` arg + parallel-call helper; (2) `SubprocessExecutorStep`: job list → `(code_path, seed, beam_id, exec_id)`, `_build_cmd` gains `code_path`, `exec_dir` parameterised by beam, post-exec grouping by `beam_id` |
| `cap/agent/agent_pipeline.py` | Insert `StrategyPlannerStep` when `beam_search.enabled`; call `_write_history_entry` K times after executor; extend escalation to `_count_all_beams_stagnant`; inject history into planner prompt |
| `cap/agent/agent_config.py` | Add `BeamSearchConfig(enabled, k_beams, planner_model, planner_max_retries, stagnation_threshold)` |
| `cap/agent/agent_session.py` | `exec_dir()` accepts optional `beam_id`; `save_iteration()` writes `strategies.json` + `beam_scores.json` |
| `cap/agent/skill_library.py` | Add `explored` to valid lifecycle statuses; `index_for_prompt()` renders `explored` skills with score + failure_summary |
| `experiments/experiment/pick_place_sink_to_counter.yaml` | Add `beam_search:` block |

### 12.3 Files not touched

`SkillAuthorStep`, `SelfReflectionStep`, `AgentContext`, snapshot/promotion
helpers, GPU semaphore dispatch logic — unchanged.

---

## 13. Implementation order

1. **`explored` status in `SkillLibrary`** — add status, update
   `index_for_prompt` rendering. Zero risk, purely additive.
2. **`StrategyPlannerStep`** — write class and prompt; confirm diversity
   validator catches anchor-tool collisions.
3. **Strategy-conditioned `AssemblyGeneratorStep`** — optional `strategy` arg;
   parallel-call helper.
4. **Executor job list extension** — `(code_path, seed, beam_id, exec_id)`
   tuples; `_build_cmd` param; per-beam aggregation.
5. **Beam-aware history** — call `_write_history_entry` K times; add
   `beam_id` + `winner/explored` label; inject history into planner prompt;
   extend escalation to `_count_all_beams_stagnant`.
6. **Config and pipeline wiring** — `BeamSearchConfig`; `from_config` branch;
   session writes `strategies.json` / `beam_scores.json`.
7. **Sanity check** (K=2, 1 iteration) — confirm layout, history entries,
   winner selection, GPU semaphore respected.
8. **Full experiment** (K=3, 5 iterations) vs. baseline.

---

## 14. Success criteria

Run a 5-iteration, K=3 pick-place experiment and confirm:

- `history.md` contains K entries per iteration (K-1 `explored`, 1 `winner`).
- `index.json` shows `explored` entries for skills executed by non-winning
  beams, with `score` and `failure_summary` populated.
- `StrategyPlannerStep` at iter≥2 avoids re-assigning anchor tools that appear
  as `explored` in the two most recent iterations (verified by inspecting
  `strategies.json` across iterations).
- Escalation fires correctly when all K beams score below baseline for 3
  consecutive iterations.
- Final success rate meets or exceeds the 12-iteration single-beam baseline
  within the same total seed budget.
