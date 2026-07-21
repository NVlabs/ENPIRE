# Adding a New Task to ENPIRE

This document is the single reference for setting up a new task and launching
auto-research on it.  It is machine-readable; an agent given only this file
can work end-to-end.

---

## Architecture overview

Every ENPIRE task is built from four primitives:

```
reset()  →  policy.act(obs)  →  step(action)  →  verify()
```

| Primitive | Contract | Where it lives |
|-----------|----------|----------------|
| `reset` | Bring world to a known initial state | `cap/saved_scripts/<task>/` or `tmux/realworld_rl/` |
| `policy` | Choose the next action | CaP script **or** PLD actor |
| `step` | Execute the action on the robot | Forge runtime (YAM arm servers, tool registry) |
| `verify` | Return a scalar success signal | Task-specific verifier (visual, contact, geometry) |

### Two research modes

| Mode | Policy | Edit surface | What improves |
|------|--------|-------------|---------------|
| **CaP (heuristic)** | Python script calling robot tools | `cap/saved_scripts/<task>/` and `skill_library/` | Code quality, skill sequencing, perception calls |
| **PLD (neural)** | JAX actor trained by SERL/HIL-SERL | `enpire/policy/pld/` and reward function | Neural network weights |

**Push-T is CaP/heuristic.**  The research agent edits the CaP scripts; the
supervisor measures success via a visual heuristic (red-T mask match).
No neural network is involved.

---

## Step 1 — define the task contract

Create `cap/saved_scripts/<task>/` and add:

```
cap/saved_scripts/<task>/
├── main.py           # primary CaP script (executed by `enpire cap run`)
├── reset.py          # reset the scene to a known initial state
└── verify.py         # compute and print a success score from camera + state
```

### `main.py` template

```python
# main.py — runs inside cap/agent with skill_library.namespace.*  in scope
from skill_library.namespace import *  # noqa: F401,F403

# 1. Read current state
state = get_robot_state()

# 2. Perceive
image = get_camera_image("top")

# 3. Plan and execute using skill library tools
result = freespace_move(side="left", left_target_pos=[0.5, 0.0, 0.9], ...)

# 4. Check
print(f"done status={result.status}")
```

Available tools (from `uv run enpire tools list`):

| Category | Tools |
|----------|-------|
| Arm control | `freespace_move`, `goto_bev`, `nudge`, `follow_traj` |
| Gripper | `grasp_slip` |
| Perception | `get_camera_image`, `get_camera_intrinsics`, `get_camera_extrinsics` |
| Grasping | `sample_grasp_pose_anygrasp`, `sample_grasp_pose_2d` |
| State | `get_robot_state`, `go_home` |
| Detection | `detect_objects` |

### `verify.py` template

```python
# verify.py — standalone script; print a JSON result at the end
import json, sys
from skill_library.namespace import *  # noqa

image = get_camera_image("top")
score = compute_my_success_metric(image)   # your vision / contact heuristic

result = {"success": score >= 0.5, "score": float(score)}
print(json.dumps(result))
sys.exit(0 if result["success"] else 1)
```

The verifier must be **independent of policy state**; it reads only cameras
and robot state, never internal policy variables.

---

## Step 2 — register the task

Add an entry to `enpire/env/forge/yam/commands.py` under the `cap run`
subparser, or create `enpire/env/examples/<NN>_<task>/example.yaml`:

```yaml
name: my-new-task
description: Brief human-readable description.
hardware: yam
entrypoint: cap/saved_scripts/my_task/main.py
motion: true
install_extras: [vision, planning, control-yam, camera-realsense]
```

Then register it in `enpire/env/examples/__init__.py` (if the example loader
does not auto-discover yaml files, add the path there).

---

## Step 3 — verify hardware-free first

ENPIRE tasks must run with `--dry-run` before any real motion:

```bash
uv run enpire cap run my-new-task --station my-yam --dry-run
```

With `--dry-run`, all motion primitives return immediately without sending
commands.  The full perception and planning stack still runs.

---

## Step 4 — run the task with motion

Prerequisites (see `enpire/env/docs/REAL_WORLD_WORKFLOWS.md`):

```bash
# Start AnyGrasp service (if needed)
export ANYGRASP_SDK_ROOT=... ANYGRASP_CHECKPOINT=... ANYGRASP_LICENSE_ZIP=...
uv run enpire services start --profile cap-real

# Start arm servers
uv run enpire services start --profile robot --station my-yam --confirm-motion

# Run the task
uv run enpire cap run my-new-task --station my-yam --confirm-motion
```

Logs and camera frames are written to:
```
enpire/env/forge/logs/<task>_<timestamp>/
├── events.jsonl     # timestamped tool calls and results
├── result.json      # final success/score from verify()
└── images/          # per-step camera snapshots
```

---

## Step 5 — CaP auto-research loop

This is the recommended mode for tasks like push-T where the policy is code.

```
hypothesis → edit CaP script → run task → read logs → measure success → iterate
```

### Required sequence for each iteration

```bash
# 1. Understand the current failure from logs
#    → read enpire/env/forge/logs/<task>_<timestamp>/events.jsonl
#    → look at images/, read result.json

# 2. Form a falsifiable hypothesis (one sentence, in a comment at the top of main.py)

# 3. Make the smallest change to CaP code that tests it
#    → edit cap/saved_scripts/<task>/main.py
#    → or edit cap/saved_scripts/skill_library/<skill>.py

# 4. Run the task — count the number of trials explicitly
uv run enpire cap run my-new-task --station my-yam --confirm-motion

# 5. Run the verifier independently to get a clean score
uv run python cap/saved_scripts/<task>/verify.py

# 6. Keep the change only if it passes success threshold
#    → revert with git checkout if it regresses
```

### What the agent may edit

| Allowed | Forbidden |
|---------|-----------|
| `cap/saved_scripts/<task>/main.py` | `reset.py` or `verify.py` logic |
| `cap/saved_scripts/skill_library/*.py` — tool parameters, heuristics | Verifier threshold |
| `enpire/env/examples/<task>/example.yaml` — Hydra overrides | Robot-side reward / event code |
| `enpire/policy/` — if using PLD alongside | `autoresearch_instruction.md` |

Safety invariants that must not change across iterations:
- The success definition in `verify.py`
- The reset procedure in `reset.py`
- Workspace limits and emergency-stop hooks

### Logging what you changed

Each iteration must record:
1. **Hypothesis** — one falsifiable sentence
2. **Diff** — `git diff cap/saved_scripts/<task>/`
3. **Result** — success boolean + score from `result.json`
4. **Decision** — keep / revert

---

## Push-T reference implementation

Push-T is an existing CaP/heuristic task. Study it as the canonical example.

| File | Purpose |
|------|---------|
| `cap/saved_scripts/pusht/reset_t_skill.py` | Red-T detection and pick-place primitives |
| `cap/saved_scripts/place_grasped_t_reset.py` | Full reset sequence (main policy script) |
| `cap/saved_scripts/pusht/go_home_fast.py` | Fast home skill for push-T |
| `cap/tasks/pusht/standard_initial_top.png` | Reference image for `reset_ok_v1` verifier |
| `cap/tasks/pusht/standard_initial_top_meta.json` | Crop metadata for the verifier |
| `enpire/env/forge/tmux/realworld_rl/rl_pusht.sh` | Robot-side supervisor (runs the CaP reset script in a loop) |
| `enpire/env/forge/tmux/realworld_rl/pusht_supervisor.py` | Supervisor that calls the reset script automatically on success |
| `enpire/env/forge/tmux/realworld_rl/tasks_config/pusht/pusht.yaml` | Task config (data path, camera names, auto-reset threshold) |

**How push-T auto-research works:**

```bash
export RL_DATA_PATH=/path/outside/repo/rl-data

# Start the supervisor — it runs place_grasped_t_reset.py in a loop,
# calls verify automatically after each trial, and restarts when the
# reset_ok score crosses the threshold.
bash tmux/realworld_rl/rl_pusht.sh --station my-yam --use-spacemouse

# The agent edits cap/saved_scripts/pusht/reset_t_skill.py to improve the
# reset success rate; the supervisor picks up changes on the next trial.
```

The supervisor writes per-trial records to `$RL_DATA_PATH`.  After N trials:

```bash
uv run enpire rl score \
  --data-dir /path/outside/repo/rl-data/<run-id> \
  --window 50 \
  --plot
```

---

## PLD (neural RL) tasks — pin insertion, GPU insertion, zip-tie

These tasks use a JAX neural actor trained online by SERL.  The agent edits
`enpire/policy/pld/` and the reward configuration.  See
`enpire/policy/autoresearch_instruction.md` for the required sequencing.

Quick reference:

```bash
export RL_DATA_PATH=/path/outside/repo/rl-data
export ENPIRE_YAM_STATION=my-yam
export ENPIRE_RL_INITIAL_POSITIONS=/path/outside/repo/init_positions.yaml
export ENPIRE_RL_REWARD_CONFIG=/path/outside/repo/reward.yaml

uv run enpire rl control health
uv run enpire rl control pause --confirm-control
uv run enpire rl control restart --confirm-control   # → prints run_dir=...
uv run enpire rl learner --task pin_insertion
uv run enpire rl actor  --task pin_insertion
bash tmux/realworld_rl/rl_gear.sh --task pin_insertion --station my-yam
uv run enpire rl control resume --confirm-control
# ... at budget boundary ...
uv run enpire rl control pause --confirm-control
uv run enpire rl score --data-dir /path/outside/repo/rl-data/<run-id> --window 50 --plot
```

---

## Checklist for a new task

- [ ] `cap/saved_scripts/<task>/main.py` written and `--dry-run` passes
- [ ] `cap/saved_scripts/<task>/reset.py` restores the initial state reliably
- [ ] `cap/saved_scripts/<task>/verify.py` returns a scalar score from observations only
- [ ] Task registered in example.yaml
- [ ] `uv run pytest -q tests/enpire` passes
- [ ] `uv run ruff check enpire tests/enpire` clean
- [ ] No workstation paths, credentials, or device serials in any file
- [ ] First live trial run with `--confirm-motion`; operator present
- [ ] Baseline success rate recorded before any auto-research starts
