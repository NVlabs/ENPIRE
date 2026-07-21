# Skill Library — Shared Reference

The skill library is a per-run, append-only store of reusable atomic helpers. It is rebuilt from scratch at the start of every `run_agent.py` invocation and persists across the iterations of that run, not across runs.

Two prompts do the real work: `skill_author` (stage 1) writes new / refined / replaced skills into `skill_library/<base>.py`; `assembly_generator` (stage 2) imports what it needs and writes the orchestration in `code.py`. This doc defines the vocabulary both stages share.

## Lifecycle of a saved skill

Every saved skill carries a `status`, displayed as a marker in the index:

- ⚪ **pending** — authored but has no successful call yet. Shown in the index. Safe to import; the first successful call transitions it to `verified`.
- 🟢 / 🟡 **verified** — at least one successful call on record. 🟢 ≥ 0.8 success rate, 🟡 ≥ 0.5. Live `calls` and `success_rate` accumulate across all prior iterations.
- **deprecated** (hidden) — called ≥ 10 times with ≤ 0.2 success rate. Kept on disk for provenance, hidden from the index so nobody reuses it.

**Every entry in the "Available skills" index is importable.** If a family is listed you must treat it as the source of truth — redefining it inline is forbidden in both stages.

## Family naming: reuse vs. refine vs. replace

A skill's identity is its `base_name` (the part before `_vN`). The `base_name` must describe the **mechanism**, never the **scene**:

| Base name | Good | Bad |
|---|---|---|
| Mechanism-oriented | `vertical_grasp`, `anygrasp_grasp`, `curobo_plan` | — |
| Scene-oriented | — | `sink_grasp`, `counter_place`, `cabinet_hover` |

Scene context (sink / counter / cabinet) is carried by the **position argument** you pass, not by the skill's name. Two skills with identical mechanisms but different scene names are parallel duplicates and will split the success-rate signal.

When the task calls for a change, classify it:

1. **Reuse** — call the existing skill with different arguments. Default choice when an existing skill's success rate is ≥ 0.8.
2. **Refine** — same mechanism, tuned params or minor logic. Bump the version within the family: `X_v1` → `X_v2`. `base_name` stays the same.
3. **Replace** — genuinely different mechanism (e.g. `vertical_grasp_v1` → `anygrasp_grasp_v1`). New `base_name`. Docstring's first line must name the different mechanism.

## Skill body contract (for stage 1 authors)

Every saved skill is:

1. A top-level `def` decorated with exactly one `@skill` (from `cap.agent.skill_registry`).
2. Returns `(val, {...})` — a 2-tuple whose second element is a dict literal or a local variable bound to a dict. `val` is typically a bool.
3. Calls only namespace tools (`freespace_move`, `set_gripper`, `get_task_info`, `np.*`, …). Never another skill, never anything imported from `skill_library.*`.
4. Imports its own module dependencies inside the function body: `import numpy as np`, `from scipy.spatial.transform import Rotation as R`, etc. Don't import `time` — `@skill` already supplies `time_s` in the returned log.

Stage 1's prompt (`system/skill_author`) has the full grammar and examples. Stage 2 never authors skills.

## Assembly contract (for stage 2 orchestration)

`code.py` is pure orchestration:

- Imports + top-level statements only. No `def`, no `class`, no `lambda`, no `async def`.
- No `try` / `except` anywhere. Check status-bearing return values instead (`r.status == "Success"`, `info["has_object"]`, `s, log = skill(...); if not s: ...`).
- Only imports skills that appear in the "Available skills" index. Missing ones stay missing; reflection can ask the author to add them on the next iter.

Stage 2's prompt (`system/assembly_generator`) has the full rules and an end-to-end example.

## Index format (what both stages see)

```
Available skills (import from skill_library.<base_name>):
  🟢 vertical_grasp_v1     [vertical_grasp.py:5-28]   success=0.97  calls=30
    "Descend onto obj_pos, compliant-close, verify grasp."
  ⚪ anygrasp_grasp_v1     [anygrasp_grasp.py:5-40]   untested (pending)
    "Sample grasp poses via AnyGrasp, top-score, compliant-close."
```

- Up to 15 families shown, ranked by verified-first then by `success_rate × log(calls+1)`.
- Within a family, only the top-scoring version is shown by default; the rest are noted as "N other versions in same family — see skill_library/<file>".
- Deprecated entries are hidden.
