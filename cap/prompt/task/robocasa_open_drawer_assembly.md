# RoboCasa OpenDrawer — Assembly Guide

Orchestration guide for `OpenDrawer` under the two-stage code generator.
The skill-author step has already written the needed skills; your job is to import and
sequence them correctly.

## Key facts

- Single drawer handle in most layouts; `controls` may have more than one — loop all.
- Open fraction target is **0.95** — `drawer_handle_finish_v1` stops automatically.
- Always re-fetch oracle inside the loop (`get_oracle_targets()`) so progress reflects
  the latest state.
- `get_task_info().get("success", False)` is the authoritative success signal — check
  it before each control to short-circuit if already done.

## Reference assembly (proven pattern from human oracle)

```python
from cap.saved_scripts.robocasa_skill_library.drawer_handle_targets import drawer_handle_targets_v1
from cap.saved_scripts.robocasa_skill_library.drawer_handle_grasp import drawer_handle_grasp_v1
from cap.saved_scripts.robocasa_skill_library.drawer_handle_finish import drawer_handle_finish_v1
from cap.saved_scripts.robocasa_skill_library.drawer_control_state import get_control_target

oracle = get_oracle_targets()
if oracle.get("task_name") != "OpenDrawer":
    raise RuntimeError(f"wrong task: expected OpenDrawer, got {oracle.get('task_name')}")
if not oracle.get("supported"):
    raise RuntimeError(f"oracle target unsupported: {oracle.get('reason')}")

_, target_log = drawer_handle_targets_v1(oracle, set_markers=False)
control_names = target_log["ordered_controls"]
last_error = ""

for control_name in control_names:
    if get_task_info().get("success", False):
        print(f"Task already successful before {control_name}; stopping")
        break

    latest_oracle = get_oracle_targets()
    target = get_control_target(latest_oracle, control_name)
    if target is None:
        print(f"Skipping {control_name}: target unavailable")
        continue

    s_grasp, grasp_log = drawer_handle_grasp_v1(control_name, target)
    print(f"{control_name} grasp: success={s_grasp}, log={grasp_log}")
    if not s_grasp:
        last_error = grasp_log.get("reason", "grasp failed")
        continue

    s_finish, finish_log = drawer_handle_finish_v1(control_name, grasp_log)
    print(f"{control_name} finish: success={s_finish}, log={finish_log}")
    if not s_finish:
        last_error = finish_log.get("reason", "drawer finish failed")
        continue

final = get_task_info()
print(f"Final state: {get_oracle_targets().get('fixture_state')}")
print(f"Success: {final.get('success', False)}")
print(f"Reward: {final.get('reward', 0.0)}")
if not final.get("success", False):
    raise RuntimeError(last_error or "drawer not opened successfully")
```

## Common failure patterns and fixes

- **All tilt candidates fail at pre-grasp**: arm configuration is poor. Add `go_home(arm)`
  before calling `drawer_handle_grasp_v1` (the skill does this internally, but if the
  home pose itself is infeasible, check the embodiment config).
- **Grasp succeeds but finish stalls at low progress**: pull direction is reversed. The
  skill probes both signs on step 1 and flips on low progress within the first 3 steps —
  if it still stalls, the drawer may require more travel distance than the oracle predicts.
- **`target is None` for every control**: `oracle["controls"]` is empty and
  `oracle["active_control"]` is not set. The oracle `supported=True` but produced an
  incomplete target — log `oracle` and raise.
- **RuntimeError from oracle validation**: `supported=False` means this fixture layout
  has no oracle target. Do not attempt drawer skills — the oracle cannot guide the arm.
