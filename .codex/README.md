# ENPIRE — Agent Onboarding

This file is the single entry point for a coding or research agent working in
this repository.  Read it top to bottom before touching any file, then follow
the section that matches your assigned task.

Repository: `https://github.com/NVlabs/ENPIRE`  
Version: `0.1.0`  
Branch convention: feature work branches from `release`.

---

## What ENPIRE is

ENPIRE is a **reset → execute → verify → record → refine** harness for robot
policy improvement.  Two research modes are fully supported:

| Mode | Policy | Edit surface | Example tasks |
|------|--------|-------------|---------------|
| **CaP (heuristic)** | Python script calling robot tools | `cap/saved_scripts/<task>/` + `skill_library/` | cube-pick, push-T, GPU insertion, zip-tie |
| **PLD (neural)** | JAX actor trained by SERL/HIL-SERL | `enpire/policy/pld/` + reward config | pin-insertion |

> **New task setup and auto-research pipeline:** `enpire/env/docs/NEW_TASK.md`

Key capabilities:

| Capability | Entry point |
|-----------|-------------|
| Hardware-free simulation | `uv run enpire examples run 00_hello_environment` |
| CaP — cube pick, GPU insertion, zip-tie, push-T | `uv run enpire cap run <task> --station my-yam --confirm-motion` |
| YAM arm calibration (ChArUco, hand-eye) | `uv run enpire station calibrate-all --station my-yam --confirm-motion` |
| PLD actor/learner (pin insertion) | `uv run enpire rl learner/actor --task <task>` |
| RL control surface | `uv run enpire rl control <verb>` |
| Tool and service registry | `uv run enpire tools list`, `uv run enpire services start` |

---

## 1. Install

Requires: **Python 3.11**, **uv**, **Linux x86-64**, **tmux**, Git + Git LFS.

```bash
git clone https://github.com/NVlabs/ENPIRE.git
cd ENPIRE

# Minimal — hardware-free tests and simulation only
uv sync --extra dev

# Full real-YAM stack (robot, cameras, perception, planning, calibration, RL bridge)
uv sync \
  --extra dev \
  --extra cap \
  --extra vision \
  --extra vision-local \
  --extra grasping-local \
  --extra planning \
  --extra planning-local \
  --extra control-yam \
  --extra camera-realsense \
  --extra calibration \
  --extra real-rl

# PLD learner/actor — isolated JAX runtime, must be separate
uv sync --project enpire/policy/pld/runtime --extra dev
```

`planning-local` compiles the vendored cuRobo CUDA extensions; a CUDA toolkit
and NVIDIA driver must already be present.

Verify the install:

```bash
uv run enpire --version
uv run enpire doctor
uv run enpire tools list
uv run pytest -q tests/enpire
uv run ruff check enpire tests/enpire
```

---

## 2. Repository layout

```
ENPIRE/
├── enpire/
│   ├── env/
│   │   ├── forge/          runtime, tool registry, YAM station support, CaP runner
│   │   ├── examples/       ordered learning path (00_hello_environment, 10_real_cube_pick)
│   │   └── docs/           INSTALL.md, REAL_WORLD_WORKFLOWS.md, DEPENDENCIES.md,
│   │                       source_provenance.yaml
│   └── policy/
│       ├── interface.py    common code/learned policy contract
│       ├── autoresearch_instruction.md   auto-research safety contract (READ FIRST)
│       └── pld/            PLD actor/learner; isolated runtime in pld/runtime/
├── cap/                    source-faithful Forge CaP implementations
├── tmux/realworld_rl/      robot-side RL launchers and reset loops
├── third_party/            vendored: curobo, pyroki, i2rt
├── AGENTS.md               implementation rules for coding agents
├── THIRD_PARTY_NOTICES.md  third-party attributions
├── THIRD_PARTY_LICENSES.md full license texts for vendored components
└── VERSION                 0.1.0
```

---

## 3. Station setup (one-time, per physical station)

External files required (never committed to Git):
- Licensed YAM model assets — set `ENPIRE_YAM_MODEL_ROOT`
- Calibrated station XML — written by `calibrate-all` to the path you specify
- Device serials / CAN aliases — written by `station register` to ENPIRE data home

```bash
# Create and populate station profile
uv run enpire station init --station my-yam
uv run enpire station register --station my-yam   # unplugs/replugs, write-only
uv run enpire station show --station my-yam
uv run enpire station doctor --station my-yam

# Run all three camera calibrations in sequence (arm servers start automatically)
export ENPIRE_YAM_MODEL_ROOT=/path/to/yam-model-assets
uv run enpire station calibrate-all \
  --station my-yam \
  --output-xml /path/outside/repo/station_calibrated.xml \
  --confirm-motion

# Validate the emitted record without hardware
uv run enpire station validate-calibration /path/outside/repo/calibration.json
```

---

## 4. External service prerequisites

Before running CaP or RL tasks, start the perception and planning services.
These run in a managed tmux session named `enpire`.

```bash
# Preview commands without starting anything
uv run enpire services start --profile cap-real --dry-run

# AnyGrasp (local) — set these before starting
export ANYGRASP_SDK_ROOT=/path/to/anygrasp_sdk
export ANYGRASP_CHECKPOINT=/path/to/checkpoint_detection.tar
export ANYGRASP_LICENSE_ZIP=/path/to/license.zip

# Start perception + planning (no robot motion yet)
uv run enpire services start --profile cap-real

# Start YAM arm servers (motion-capable)
uv run enpire services start --profile robot --station my-yam --confirm-motion

# Or everything at once
uv run enpire services start --profile all --station my-yam --confirm-motion

# Inspect running services
uv run enpire services status
tmux attach -t enpire
```

---

## 5. Code-as-Policy (CaP) pipeline

CaP tasks are Python scripts that drive the robot through tools from
`skill_library/`.  The `enpire cap run` entry sets `FORGE_ROOT` and
`skill_library_path` correctly; do not invoke scripts directly.

```bash
uv run enpire cap run cube-pick          --station my-yam --confirm-motion
uv run enpire cap run gpu-handover       --station my-yam --confirm-motion
uv run enpire cap run gpu-reset          --station my-yam --confirm-motion
uv run enpire cap run ziptie-reset       --station my-yam --confirm-motion
```

Logs and camera frames → `enpire/env/forge/logs/<task>_<timestamp>/`.

---

## 6. PLD reinforcement learning pipeline (pin insertion)

The JAX learner requires the isolated runtime at `enpire/policy/pld/runtime/`.

```bash
export RL_DATA_PATH=/path/outside/repo/rl-data
export ENPIRE_YAM_STATION=my-yam

uv run enpire rl learner --task pin_insertion    # terminal 1
uv run enpire rl actor  --task pin_insertion     # terminal 2
bash tmux/realworld_rl/rl_gear.sh \
  --task pin_insertion --station my-yam --use-spacemouse   # terminal 3
```

---

## 7. Auto-research — two modes

**Read `enpire/env/docs/NEW_TASK.md` before starting.**  It covers both modes,
the allowed edit surface, and the per-iteration record format.

### Mode A — CaP / heuristic (push-T, GPU insertion, zip-tie, cube-pick)

The "policy" is code.  Auto-research = LLM edits scripts, re-runs, measures.

```bash
export RL_DATA_PATH=/path/outside/repo/rl-data

# Push-T: supervisor calls place_grasped_t_reset.py in a loop,
# measures success via red-T mask score, records per-trial results.
bash tmux/realworld_rl/rl_pusht.sh --station my-yam --use-spacemouse

# After N trials — score the run
uv run enpire rl score \
  --data-dir /path/outside/repo/rl-data/<run-id> --window 50 --plot
```

Agent edits: `cap/saved_scripts/<task>/main.py`, `skill_library/*.py`.
Agent must NOT change: `verify.py`, `reset.py`, verifier threshold.

### Mode B — PLD / neural (pin insertion)

```bash
uv run enpire rl control health
uv run enpire rl control pause --confirm-control
uv run enpire rl control restart --confirm-control   # → run_dir=...
uv run enpire rl learner --task pin_insertion
uv run enpire rl actor  --task pin_insertion
uv run enpire rl control resume --confirm-control
# at budget:
uv run enpire rl control pause --confirm-control
uv run enpire rl score \
  --data-dir /path/outside/repo/rl-data/<run-id> --window 50 --plot
```

Agent edits: `enpire/policy/pld/` (network, hyperparameters).
Agent must NOT change: reward function, reset, score implementation.

### Common rules (both modes)

An auto-research agent **must not** edit:
- Reset procedure, verifier, success gate, safety limits
- Evaluation seeds, artifact schema, score implementation
- Robot-side event/reward code in `tmux/realworld_rl/`

---

## 8. Development rules

1. Run `uv run pytest -q tests/enpire` and `uv run ruff check enpire tests/enpire` before every commit.
2. Consult `enpire/env/docs/source_provenance.yaml` before moving migrated Forge code.
3. Add a characterization test for existing behavior before refactoring it.
4. Never embed workstation paths, device serials, IP addresses, or credentials anywhere in the repo.
5. Optional dependencies must stay lazy — `import enpire` must not import hardware libraries.
6. Hardware, network, integration, and slow tests use explicit pytest markers and are not part of the default loop.
7. `enpire cap run` sets cwd to `FORGE_ROOT`; never call `run_script.py` directly.

---

## 9. Key environment variables

| Variable | Purpose |
|----------|---------|
| `ENPIRE_STATION` | Station identity (e.g. `my-yam`) |
| `ENPIRE_YAM_MODEL_ROOT` | Path to licensed YAM model XML assets |
| `ENPIRE_YAM_CALIBRATED_XML_OUTPUT` | Where `calibrate-all` writes the calibrated XML |
| `ANYGRASP_SDK_ROOT` | Local AnyGrasp SDK directory |
| `ANYGRASP_CHECKPOINT` | Path to `checkpoint_detection.tar` |
| `ANYGRASP_LICENSE_ZIP` | Path to AnyGrasp license archive |
| `RL_DATA_PATH` | Root for PLD run directories (external to repo) |
| `ENPIRE_RL_INITIAL_POSITIONS` | YAML with robot reset joint positions |
| `ENPIRE_RL_REWARD_CONFIG` | YAML reward configuration for the chosen task |

Credentials are supplied only through the process environment or an external
secret manager.  Never write them to files inside the repository.

---

## 10. Further reading

| Document | Purpose |
|----------|---------|
| `enpire/env/docs/NEW_TASK.md` | **How to add a new task and launch auto-research** |
| `AGENTS.md` | Implementation rules for coding agents |
| `enpire/policy/autoresearch_instruction.md` | PLD auto-research safety contract |
| `enpire/env/docs/INSTALL.md` | Full install reference |
| `enpire/env/docs/REAL_WORLD_WORKFLOWS.md` | All operational commands |
| `enpire/env/docs/DEPENDENCIES.md` | Complete dependency inventory |
| `enpire/env/docs/source_provenance.yaml` | Source branch / commit / license status |
| `THIRD_PARTY_NOTICES.md` | Third-party attributions |
| `THIRD_PARTY_LICENSES.md` | Full license texts for vendored components |
| `SECURITY.md` | Credential and hardware safety policy |
