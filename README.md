# ENPIRE: Agentic Robot Policy Self-Improvement in the Real World

<p align="center">
  <img src="assets/main_figure.png" alt="ENPIRE overview" width="100%">
</p>

ENPIRE is a research harness for autonomous robot policy improvement on real hardware.
An LLM agent proposes hypotheses, writes or edits policy code, runs trials on the
physical robot, reads the outcome, and iterates — all without human intervention
between trials.

The loop is: **reset → execute → verify → record → refine.**

---

## Highlights

- **Code-as-Policy (CaP)** — the policy is Python. The agent edits skill scripts and
  re-runs them; success is measured by a vision or contact heuristic the agent cannot modify.
- **Online RL (PLD)** — a JAX actor trained live by SERL/HIL-SERL on real robot data.
  The agent tunes hyperparameters and reward shaping between trial budgets.
- **Both modes on the same station** — CaP and PLD tasks share the YAM arm, cameras,
  and calibration infrastructure.
- **One-command calibration** — `enpire station calibrate-all` launches arm servers,
  runs all three ChArUco/hand-eye sequences in tmux, and writes the calibrated XML.
- **Agent-readable** — `.codex/README.md` is a self-contained onboarding file;
  an agent given only the repo URL can install, calibrate, and run auto-research
  end-to-end.

### Demonstrated tasks

| Task | Mode | Policy | Notes |
|------|------|--------|-------|
| Cube pick | CaP | `cap/saved_scripts/examples/pick_cube.py` | Hardware-free quickstart |
| **Push-T** | **CaP + PLD** | `cap/saved_scripts/pusht/` · `enpire/policy/rl/pusht/` | **Fully reproducible end-to-end autoresearch example** — includes CaP reset loop, vision reward, RL training, and 3D-printable T-block (`robot/models/objects/meshes/t_block.stl`) |
| GPU insertion | CaP | `cap/saved_scripts/skill_library/` | — |
| Pin insertion | PLD (online RL) | `enpire/policy/pld/` | — |

---

## Install

**Requirements:** Python 3.11, [uv](https://docs.astral.sh/uv/), Linux x86-64, tmux.

```bash
git clone https://github.com/NVlabs/ENPIRE.git
cd ENPIRE

# Hardware-free baseline (simulation + tests)
uv sync --extra dev

# Full real-robot stack
uv sync --extra dev --extra cap --extra vision --extra vision-local \
        --extra grasping-local --extra planning --extra planning-local \
        --extra control-yam --extra camera-realsense --extra calibration \
        --extra real-rl

# JAX PLD learner/actor (isolated environment)
uv sync --project enpire/policy/pld/runtime --extra dev
```

Run the hardware-free hello-world to verify the install:

```bash
uv run enpire examples run 00_hello_environment
```

Full installation notes: [`enpire/env/docs/INSTALL.md`](enpire/env/docs/INSTALL.md)

---

## Station setup

One-time setup per physical station (YAM arms + cameras):

```bash
export ENPIRE_YAM_MODEL_ROOT=/path/to/yam-model-assets

uv run enpire station init     --station my-yam
uv run enpire station register --station my-yam   # detects CAN/USB serials
uv run enpire station calibrate-all \
  --station my-yam \
  --output-xml /path/outside/repo/station_calibrated.xml \
  --confirm-motion
```

The `calibrate-all` command starts both arm servers automatically in a tmux
session, runs the intrinsic → extrinsic → hand-eye sequence, and writes the
calibrated MuJoCo XML to the path you specify.

---

## Running tasks

### Start services (perception + arm servers)

```bash
uv run enpire services start --profile cap-real          # AnyGrasp, cameras
uv run enpire services start --profile robot \
  --station my-yam --confirm-motion                      # YAM arm servers
```

### Code-as-Policy tasks

```bash
uv run enpire cap run cube-pick    --station my-yam --confirm-motion
uv run enpire cap run gpu-handover --station my-yam --confirm-motion
uv run enpire cap run gpu-reset    --station my-yam --confirm-motion
uv run enpire cap run ziptie-reset --station my-yam --confirm-motion
```

### Push-T (CaP auto-research)

```bash
export RL_DATA_PATH=/path/outside/repo/rl-data

# Supervisor runs the CaP reset script in a loop and records per-trial results
bash tmux/realworld_rl/rl_pusht.sh --station my-yam --use-spacemouse

# Score a completed run
uv run enpire rl score --data-dir "$RL_DATA_PATH/<run-id>" --window 50 --plot
```

### Pin insertion (PLD online RL)

```bash
export RL_DATA_PATH=/path/outside/repo/rl-data
export ENPIRE_YAM_STATION=my-yam

uv run enpire rl control health
uv run enpire rl control pause   --confirm-control
uv run enpire rl control restart --confirm-control        # → prints run_dir
uv run enpire rl learner --task pin_insertion             # terminal 1
uv run enpire rl actor   --task pin_insertion             # terminal 2
bash tmux/realworld_rl/rl_gear.sh \
  --task pin_insertion --station my-yam --use-spacemouse  # terminal 3
uv run enpire rl control resume  --confirm-control
```

---

## Repository layout

```
ENPIRE/
├── assets/                   figures for this README
├── cap/saved_scripts/
│   ├── skill_library/        shared robot tools (freespace_move, grasp, detect …)
│   ├── pusht/                push-T CaP skills
│   ├── gpu/                  GPU insertion scripts
│   └── ziptie/               zip-tie scripts
├── enpire/
│   ├── env/
│   │   ├── forge/            runtime, tool registry, YAM station, CaP runner
│   │   ├── examples/         learning path + task capsules
│   │   └── docs/             INSTALL.md, REAL_WORLD_WORKFLOWS.md, NEW_TASK.md
│   └── policy/
│       ├── pld/              JAX PLD actor/learner (isolated runtime)
│       └── autoresearch_instruction.md
├── tmux/realworld_rl/        supervisors and RL launchers
├── third_party/              vendored: cuRobo, PyRoki, i2rt, robocasa
├── .codex/README.md          agent onboarding (full setup + auto-research)
└── AGENTS.md                 coding-agent implementation rules
```

---

## Adding a new task / launching auto-research

See [`enpire/env/docs/NEW_TASK.md`](enpire/env/docs/NEW_TASK.md) for the
complete guide: environment contract, file templates, per-iteration loop,
allowed edit surface, and pre-live checklist.

---

## Development

```bash
uv run pytest -q tests/enpire
uv run ruff check enpire tests/enpire
```

Consult `enpire/env/docs/source_provenance.yaml` before moving code migrated
from upstream Forge branches.  Add a characterization test before refactoring.

---

## License

Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under the [Apache License 2.0](LICENSE).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for third-party attributions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions must be signed off
under the Developer Certificate of Origin and licensed under Apache-2.0.

## Security

To report a security vulnerability, visit
[https://www.nvidia.com/en-us/security/](https://www.nvidia.com/en-us/security/).
See [SECURITY.md](SECURITY.md) for credential and hardware safety rules.
