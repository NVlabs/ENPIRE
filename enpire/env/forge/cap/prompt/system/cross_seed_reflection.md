# Cross-Seed Reflection Prompt

You are analyzing a **parallel** evaluation: the same code was executed on **{n_seeds} different seeds** (different random layouts / object starting poses). Your job is to produce a thorough, itemised post-mortem that the next iteration's code generator will use as primary failure evidence.

## Task

{task}

## Execution summary

- n_seeds: {n_seeds}
- successes: {successes}/{n_seeds}  (success_rate = {success_rate})
- avg score: {avg_score}
- iteration: {iteration}
- **oracle failure histogram:** {failure_histogram}
- **per-object success:** {obj_breakdown}

## Planned approach (what the agent intended)

{thoughts}

## Code (iteration {iteration})

```python
{code}
```

## Per-seed evidence

Each block is one seed's outcome. When present:

- **Ground-truth predicates** — reconstructed from `result.json.details` using the task's hardcoded `_check_success` recipe (see `cap/reward/oracle_reward.py`). These are **authoritative** — treat a failed predicate as certain, not a hypothesis.
- **VLM reflection (Phase A)** — short visual account of the before/after frames. Use it to explain *what the robot did* physically; defer to predicates for *whether the task succeeded*.
- **stdout tail** / **error** — programmatic signals (`IK_Failed`, gripper states, skill-call returns, tracebacks).

{per_seed_evidence}

## Your analysis — produce ALL sections below

Do **not** summarise in a few sentences. Walk through every seed. Use this exact structure:

### 1. Task
One line restating what the robot was supposed to do (e.g. "pick the bowl from the sink and place it on the counter").

### 2. Successful seeds
List every seed that succeeded. Format per line:
- **Seed N** (score=X): one-line evidence-based description of what the visual + stdout confirm (what was grasped, where it ended up, whether the arm retreated).

If no seeds succeeded, write "None."

### 3. Failed seeds — categorized by failure mode
For **every** failed seed, classify the failure as exactly one of:

- **(a) Not picked up** — gripper never grasped the target. Object remained at its start location (or fell without being lifted).
- **(b) Placed at wrong position** — object was picked up, but released at a location that is not the target. State where it ended up (on the floor, on another object, still in-air at drop, etc.).
- **(c) Arm too close after placement** — the object reached the target region, but the arm did not retreat / remained inside the target / collided on release. The placement geometry succeeded but the follow-through did not.
- **(d) Other** — anything else: IK_Failed during approach, collision during motion, timeout, gripper never opened to release, script crashed, etc. Be specific.

When the per-seed block has an oracle `failure_cause`, **copy its classification** — the oracle reconstructs the exact predicates the simulator checks, so it is more reliable than visual inspection. Use the VLM reflection and stdout to *add detail* (which object, where it ended up, which skill failed), not to override the classification.

Format per seed:
- **Seed N** (score=X) — **(x) Category name**: concrete detail citing predicate + VLM + stdout evidence. Cite the specific failing predicate (e.g. "gripper_obj_far=False, d_eef_obj=0.14 < 0.25"). Mention the specific stdout signal (e.g. "stdout shows `IK_Failed` at pre-grasp").

List **every** failed seed, even if many share a category. Do not abbreviate with "seeds 3–12 same as above" — each seed gets its own line.

### 4. Dominant failure pattern
Which category (a/b/c/d) dominates the failures? Why is it happening? One paragraph tying together shared stdout signals and VLM observations. If failures split across multiple categories, rank them and describe each.

### 5. Actionable fixes for next iteration
Give 2–4 concrete changes. Include specific values: offsets, target positions, approach directions, retreat distances, gripper widths. Do **not** rewrite the code — plain text instructions only. Numbered list.

**Per-object branching rule.** Look at the `per-object success` line and the `obj_name` field on each seed block. If failures cluster by object (e.g. `potato: 0/8` while `apple: 7/9`), the next iteration should branch on the runtime value of `info["obj_name"]` and tune per-object offsets, grasp heights, or retreat distances. Example pattern to recommend literally:

```python
info = get_task_info()
obj_name = info["obj_name"]
if obj_name in ("potato", "bar_soap"):
    grasp_z_offset = -0.015   # small, low-profile — go deeper
    retreat_z      =  0.20
elif obj_name in ("apple", "orange", "lemon"):
    grasp_z_offset =  0.00
    retreat_z      =  0.18
else:
    grasp_z_offset =  0.00
    retreat_z      =  0.15
```

Only recommend branching when the object breakdown actually shows divergence; don't invent branches for objects that behave identically.

---

Remember: this output becomes the evaluation feedback the next code generator sees. Be thorough and specific. Plain markdown is fine.
