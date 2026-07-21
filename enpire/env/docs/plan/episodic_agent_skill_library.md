# Episodic Agent Skill Library

## Overview

A skill library that grows monotonically within a single `run_agent.py` execution,
accumulating atomic reusable functions across iterations. It is **not** persistent
across runs — each run starts with an empty library and builds it up iteration by
iteration.

## Directory Layout

```
logs/<run_dir>/
  skill_library/
    namespace.py        # auto-generated at run start: re-exports all namespace tools
    pick_object.py      # all versions of pick_object (v1, v2, ...)
    open_drawer.py      # all versions of open_drawer
    ...                 # one file per base skill name
    index.json          # per-skill metadata: name, version, line range, success stats
```

Each `<skill_name>.py` file contains all versions of that skill (`_v1`, `_v2`, …),
appended over iterations. Files are never deleted or overwritten.

## `namespace.py`

Generated once at run start by inspecting the tool namespace dict (from the env,
not CapServer for RoboCasa). Contains re-exports of all callable tools so that
skill files can import them as a proper Python module.

```python
# namespace.py — auto-generated, do not edit
__all__ = [
    "get_robot_state",
    "move_to_pose",
    "set_gripper",
    "get_task_info",
    # ... all tools in the current namespace
]
# Actual implementations are injected via exec() namespace at runtime by
# run_script.py; this file exists so static importers see the symbol list.
```

The **parent** of `skill_library/` is prepended to `sys.path` in `run_script.py`
(commit `e15d5309`), so `from skill_library.<name> import ...` resolves at
subprocess execution time.

## `<skill_name>.py`

One file per base skill name (e.g. `pick_object.py`, `open_drawer.py`). Rules:

- **Atomic only**: each skill does exactly one conceptual action.
- **No skill-calls-skill**: skills may only call tools from `namespace.py`, never
other skills. This eliminates circular imports and keeps each skill independently
testable.
- **Never deleted**: old versions stay. New versions are appended to the same file
as `<name>_v2`, `<name>_v3`, etc.
- **Importable**: agent-generated code imports selectively, e.g.
`from skill_library.pick_object import pick_object_v1` or
`from skill_library.pick_object import `*.

### Skill format

Each skill is decorated with `@skill` from `cap.agent.skill_registry`, which
automatically registers it and wraps it to capture execution logs transparently.

```python
# pick_object.py
from skill_library.namespace import *
from cap.agent.skill_registry import skill

@skill
def pick_object_v1(arm: str, obj_pos: list, grasp_height_offset: float = 0.02):
    """Grasp object at obj_pos using arm."""
    import time as _time
    t0 = _time.time()
    move_to_pose(arm, [obj_pos[0], obj_pos[1], obj_pos[2] + grasp_height_offset])
    set_gripper(arm, 0.0)
    state = get_robot_state()
    success = state.arms[arm].gripper_pos < 0.05
    return success, {
        # --- shared fields (always present) ---
        "success": success,
        "time_s": round(_time.time() - t0, 3),
        # --- skill-specific fields (defined by agent) ---
        "object_pos": obj_pos,
        "final_gripper": state.arms[arm].gripper_pos,
    }

@skill
def pick_object_v2(arm: str, obj_pos: list, grasp_height_offset: float = 0.02):
    """Improved grasp with post-grasp force check."""
    ...
```

### `execution_log` schema


| Field             | Type  | Required | Description                                          |
| ----------------- | ----- | -------- | ---------------------------------------------------- |
| `success`         | bool  | ✅        | Whether the skill achieved its goal                  |
| `time_s`          | float | ✅        | Wall-clock duration of the skill call                |
| *(skill-defined)* | any   | ✗        | e.g. `object_pos`, `failure_reason`, `final_gripper` |


Skill-specific fields are defined by the agent when writing the skill. They are
passed raw to the reflection LLM, which reasons over them directly to identify
failure patterns and decide how to write improved versions.

## `SkillProfile` and `SkillRegistry`

Defined in `cap/agent/skill_registry.py` (static, part of codebase).

```python
@dataclass
class SkillProfile:
    name: str           # e.g. "pick_object_v1"
    base_name: str      # e.g. "pick_object"
    version: int        # e.g. 1
    docstring: str
    calls: list[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float: ...

    def record(self, log: dict) -> None:
        self.calls.append(log)

    def logs(self) -> list[dict]:
        """All execution logs for this skill. Fed raw to reflection LLM."""
        return self.calls


class SkillRegistry:
    """Singleton owning all SkillProfiles for the current process."""

    def register(self, fn: Callable) -> Callable:
        """@skill decorator: registers fn and wraps it to capture execution_log."""
        ...

    def profiles(self) -> dict[str, SkillProfile]: ...
    def to_json(self) -> list[dict]: ...


skill = SkillRegistry.get().register  # decorator alias used in skill files
```

The `@skill` decorator wraps each function so that `(val, execution_log)` is
intercepted and appended to the profile's `calls` list automatically — the agent
code and the skill implementation are both unchanged.

## Subprocess Integration

Each seed runs `run_script.py` as a subprocess. `SubprocessExecutorStep` passes
the skill library path as a CLI argument:

```python
# cap/agent/agent_step.py — SubprocessExecutorStep._build_cmd()
skill_library_dir = ctx.session.run_dir / "skill_library"
cmd.append(f"--skill_library_path={skill_library_dir}")
```

`run_script.py` receives it, prepends to `sys.path` before exec, and flushes
skill logs at exit:

```python
# run_script.py
parser.add_argument("--skill_library_path", default=None)
args, _ = parser.parse_known_args()

if args.skill_library_path:
    sys.path.insert(0, args.skill_library_path)  # enables skill imports

# ... exec agent code ...

# At exit: flush SkillRegistry to disk
from cap.agent.skill_registry import SkillRegistry
registry = SkillRegistry.get()
if registry.profiles():
    (Path(script_output_dir) / "skill_log.json").write_text(
        json.dumps(registry.to_json(), indent=2)
    )
```

If `--skill_library_path` is not passed (e.g. running `run_script.py` standalone),
the skill library feature is silently skipped.

The parent (`SubprocessExecutorStep`) reads all `exec_NNN/skill_log.json` files
after seeds complete and merges them into `iterations/iter_NNN/skill_log.json`.
At end of iteration, logs are appended to `skill_library/<skill_name>_log.json`
for the reflection step to consume.

## Agent Code Structure

The code generator produces three layers in one file per iteration:

```python
# Only import skills that already exist in the library
from skill_library.pick_object import pick_object_v1, pick_object_v2
# open_drawer doesn't exist yet — no import for it

# 1. NEW atomic skills defined inline (plain functions, no @skill)
#    Reflection will promote these into skill_library/open_drawer.py after this iteration
def open_drawer_v1(arm: str, handle_pos: list):
    """Open drawer by grasping handle and pulling back."""
    t0 = _time.time()
    move_to_pose(arm, handle_pos)
    set_gripper(arm, 0.0)
    success = True
    return success, {"success": success, "time_s": round(_time.time() - t0, 3),
                     "handle_pos": handle_pos}

# 2. Assembly: compose skills for this task
def run_pick_place(arm):
    s1, _ = pick_object_v1(arm, obj_pos=[0.3, 0.1, 0.85])
    s2, _ = open_drawer_v1(arm, handle_pos=[0.5, 0.0, 0.9])
    return s1 and s2

# 3. Main loop
for _ in range(3):
    if run_pick_place("right"):
        break
```

The `@skill` decorator and `cap.agent.skill_registry` import **never appear in
agent-generated code**. They only appear in `skill_library/<skill_name>.py` files
written by the reflection step.

## Code Generator Integration

At the start of each iteration, the skill library index is injected into the code
generator's system prompt:

```
Available skills:
  pick_object_v1   [pick_object.py:5-28]   success_rate=0.62   calls=18
    "Grasp object at obj_pos using arm."
  pick_object_v2   [pick_object.py:30-55]  success_rate=0.78   calls=9
    "Improved grasp with post-grasp force check."
  open_drawer_v1   [open_drawer.py:5-28]   success_rate=0.44   calls=7
    "Open drawer by grasping handle and pulling back."
```

Full skill source code is **not** injected — only name, file/line range, success
rate, call count, and docstring. Token cost stays bounded as the library grows.

The code generator reasons over this index to decide:

- Which existing skills to reuse
- Which atomic operations have no existing skill and need a new one written (with `@skill`)

## Skill Router

Skill selection from the index is handled by a pluggable `SkillRouter` protocol
so routing strategies can be swapped for research:

```python
class SkillRouter(Protocol):
    def select(self, base_name: str, versions: list[SkillProfile]) -> list[SkillProfile]:
        """Return versions to surface to the LLM for a given base skill name."""
        ...
```

Default implementation returns all versions. Future implementations may rank by
success rate, recency, or learned preference.

## Code Generator Skill Promotion

Promotion runs in **`CodeGeneratorStep`** (commit `9ffb2a09`), not in
`SelfReflectionStep` — skills must exist on disk before the subprocess runs so
`from skill_library.<name> import ...` resolves.

On each iteration, after code generation and before subprocess spawn,
`SkillPromoter(sl).promote(ctx, iteration_skill_logs)` in `cap/agent/reflection.py`:

1. Parses the generated code and identifies plain top-level functions (agent code
  does not include `@skill` — the decorator is prepended at promotion time).
2. For each candidate, skips if **any** of:
  - It calls another agent-defined function or already-imported skill (commit
    `d8c1ac79` — blocks promotion of functions that call imported skills).
  - It does not have a `(val, dict)`-style tuple return AND was not called at
    least once this iteration.
  - A skill with this base name already exists.
3. Otherwise promotes: prepends `@skill` and appends to `<skill_name>.py`,
  updates `index.json`.

**Current gap**: `CodeGeneratorStep` currently passes `iteration_skill_logs=[]`
to `SkillPromoter.promote(...)`, so `update_stats_from_logs` and
`append_skill_logs` are not yet wired in. Per-seed `exec_NNN/skill_log.json`
are still merged into `iterations/iter_NNN/skill_log.json` by
`_aggregate_skill_logs()` in `cap/agent/agent_step.py`, but downstream
propagation into `skill_library/<name>_log.json` / success-stats updates is
pending.

## `index.json` Schema

```json
{
  "skills": [
    {
      "name": "pick_object_v1",
      "base_name": "pick_object",
      "version": 1,
      "file": "pick_object.py",
      "start_line": 5,
      "end_line": 28,
      "line_count": 24,
      "docstring": "Grasp object at obj_pos using arm.",
      "introduced_iteration": 0,
      "calls": 18,
      "successes": 11,
      "success_rate": 0.611
    }
  ],
  "total_skills": 3
}
```

(Note: the current `_save_index()` writes only `skills` and `total_skills` — the
planned `last_updated_iteration` field is not yet wired in.)

`line_count` per skill allows tracking whether the agent tends to write longer or
shorter skills over iterations — observable trend without enforcing a constraint.

## Logging

- `index.json` is updated in-place after each iteration.
- Each `<skill_name>.py` is append-only; never overwritten.
- Raw execution logs are stored in `skill_library/<skill_name>_log.json`,
appended after each iteration, and fed directly to the reflection LLM.
- The skill library is **not** uploaded to W&B. It stays on disk under the run dir.

## Components to Implement


| Component                                          | Location                      | Description                                                                                                                         |
| -------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `SkillProfile`, `SkillRegistry`, `skill` decorator | `cap/agent/skill_registry.py` | Core registry; `@skill` wraps functions to capture logs transparently                                                               |
| `SkillLibrary`                                     | `cap/agent/skill_library.py`  | Owns per-skill `.py` files + `index.json`; generates `namespace.py`; provides index for prompt injection; handles versioned appends |
| `SkillRouter`                                      | `cap/agent/skill_library.py`  | Protocol + default all-versions implementation                                                                                      |
| Subprocess flush                                   | `run_script.py`               | Accept `--skill_library_path`; prepend to `sys.path`; serialize `SkillRegistry` to `exec_NNN/skill_log.json` at exit               |
| Log aggregation                                    | `cap/agent/agent_step.py`     | Pass `--skill_library_path` in subprocess cmd; merge per-seed `skill_log.json` → `iter_NNN/skill_log.json` → `skill_library/<name>_log.json` |
| Reflection extension                               | `cap/agent/reflection.py`     | Detect promotable `@skill` functions; append to library; feed raw logs to LLM                                                       |
| Code generator prompt                              | `cap/prompt/system/`          | Skill index injection; instruct agent to import existing skills and define new ones as plain inline functions                        |
| Executor `sys.path` injection                      | `cap/agent/agent_step.py`     | Add `skill_library/` to path before exec                                                                                            |


