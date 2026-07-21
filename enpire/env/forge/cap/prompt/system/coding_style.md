# Coding Style — Hard Rules

These rules apply to every Python program you generate. They are not suggestions. The rest of the pipeline (skill promotion, reflection, profiling) depends on them.

## No `try` / `except` — ever

**Do not wrap any statement in `try/except`.** No `try` around imports, no `try` around tool calls, no bare `except`.

Why this matters:

- Silent failures are the single largest source of wasted iterations in this system. If a tool crashes, an import fails, or an assumption is wrong, we **want** the traceback — the reflection LLM reads the stderr from each seed and can only fix what it can see.
- `try` around imports causes `def` statements to land inside the `except` clause. Tooling that operates on your generated code may not recognise those nested defs as module-scope bindings.
- `try` around tool calls throws away the structured status the tool returns (`FreespaceResult.status`, `MoveResult.reached`, `info["has_object"]`, …) — that status is exactly what you'd use to pick a recovery strategy.

### Wrong

```python
try:
    from skill_library.vertical_grasp import vertical_grasp_v1
except ImportError:
    def vertical_grasp_v1(...):
        ...
```

```python
try:
    r = freespace_move(right_target_pos=pos, side="right")
except Exception:
    r = None  # hides why we couldn't reach `pos`
```

### Right

Only import skills that appear in the "Available skills" index you were given. If a skill is not in the index, do not try to import it — define it fresh at module top level in Layer 2.

```python
from skill_library.vertical_grasp import vertical_grasp_v1   # listed in the index

r = freespace_move(right_target_pos=pos, side="right")
if r.status != "Success":
    print(f"freespace_move failed: status={r.status} reason={getattr(r, 'reason', '')}")
    # take a recovery path — go_home, re-plan with a higher hover, etc.
```

If a tool you expect to exist is actually missing from the namespace, the resulting `NameError` is the correct behaviour — it tells us the embodiment spec or the library save is broken, and we fix that, not the agent code.

## Check every status-bearing return

Every motion / grasp / planning tool returns a dataclass with a status field. Read it. Branch on it. Never call the tool and ignore the return. "Success" status does not prove the EE reached the target — use it as a necessary but not sufficient signal, and cross-check with `get_task_info()` or `get_gripper_info()` where relevant.

## Print diagnostics before every critical step

Reflection runs per-seed on your stdout. Print the target pose before motion, the gripper state after grasp, the task-info delta after place. Every `print` is a data point the next iteration can learn from — terse line format, include the identifiers that matter (side, object name, final position, status).

## No bare `assert` as error handling

`assert` is stripped under `python -O`. Use `if cond: raise ValueError(...)` when you genuinely need to fail fast, but prefer the pattern above (`if r.status != "Success": print(...); recovery`) for expected tool failures.

## No `sys.exit`, no `os._exit`, no `signal.*`

The executor runs your code against many seeds in threads / subprocesses. Exiting the process kills sibling seeds. If you reach a terminal failure, `print` the reason and `return` / let the script fall through to the end.
