# ENPIRE

**Reset → execute → verify → record → refine.**

ENPIRE is a harness for repeatable robot policy improvement.  Two research
modes are supported and can coexist on the same station:

| Mode | Policy | What auto-research edits | Example tasks |
|------|--------|--------------------------|---------------|
| **CaP (heuristic)** | Python script calling robot tools | `cap/saved_scripts/<task>/` + `skill_library/` | cube-pick, push-T, GPU insertion, zip-tie |
| **PLD (neural)** | JAX actor trained by SERL/HIL-SERL | `enpire/policy/pld/` + reward config | pin-insertion |

> **Agent / LLM users:** read `.codex/README.md` for full setup and
> auto-research instructions.  `enpire/env/docs/NEW_TASK.md` explains how to
> add a new task and launch auto-research on it.  Read `AGENTS.md` for
> implementation rules.

---

## Install

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), Linux x86-64, tmux.

```bash
git clone https://github.com/DarthUtopian/gear-enpire.git
cd gear-enpire
uv sync --extra dev                  # hardware-free baseline
```

Full real-robot stack:

```bash
uv sync --extra dev --extra cap --extra vision --extra vision-local \
        --extra grasping-local --extra planning --extra planning-local \
        --extra control-yam --extra camera-realsense --extra calibration \
        --extra real-rl

uv sync --project enpire/policy/pld/runtime --extra dev   # JAX learner/actor
```

Verify:

```bash
uv run enpire --version && uv run enpire doctor && uv run enpire tools list
uv run pytest -q tests/enpire && uv run ruff check enpire tests/enpire
```

---

## Station setup (one-time)

```bash
export ENPIRE_YAM_MODEL_ROOT=/path/to/yam-model-assets
uv run enpire station init --station my-yam
uv run enpire station register --station my-yam
uv run enpire station calibrate-all \
  --station my-yam \
  --output-xml /path/outside/repo/station_calibrated.xml \
  --confirm-motion
```

---

## Repository layout

```
cap/saved_scripts/
├── skill_library/          shared robot tools (freespace_move, grasp, detect, …)
├── pusht/                  push-T CaP skills (reset_t_skill.py)
├── gpu/                    GPU insertion CaP scripts
├── ziptie/                 zip-tie CaP scripts
└── place_grasped_t_reset.py  push-T full reset policy (primary CaP script)

enpire/
├── env/
│   ├── forge/              runtime, tool registry, YAM station, CaP runner
│   ├── examples/           learning path + task capsules (example.yaml + main.py)
│   └── docs/
│       ├── NEW_TASK.md          ← how to add a task and launch auto-research
│       ├── REAL_WORLD_WORKFLOWS.md
│       ├── INSTALL.md
│       └── DEPENDENCIES.md
└── policy/
    ├── autoresearch_instruction.md   PLD auto-research safety contract
    ├── interface.py                  code/learned policy contract
    └── pld/                          JAX actor/learner (isolated runtime)

tmux/realworld_rl/          robot-side supervisors and RL launchers
third_party/                vendored: cuRobo, PyRoki, i2rt, robocasa
```

---

## CaP auto-research — push-T

Push-T is a heuristic (CaP) task.  The "policy" is `place_grasped_t_reset.py`
and the skills in `cap/saved_scripts/pusht/`.  Auto-research means an LLM
agent edits those scripts and re-runs them; the supervisor measures success
via a vision heuristic (red-T mask match score).

```bash
export RL_DATA_PATH=/path/outside/repo/rl-data

# Start AnyGrasp and arm servers first (see services setup in .codex/README.md)

# Run the push-T supervisor — it calls the reset script in a loop,
# measures success automatically, and records per-trial results.
bash tmux/realworld_rl/rl_pusht.sh --station my-yam --use-spacemouse

# Score a completed run
uv run enpire rl score \
  --data-dir /path/outside/repo/rl-data/<run-id> --window 50 --plot
```

The agent edits `cap/saved_scripts/pusht/reset_t_skill.py` (detection and
motion primitives) or `place_grasped_t_reset.py` (sequencing).  The verifier
(`reset_ok_v1`) must not be changed between trials.

For the complete CaP auto-research loop and new-task setup, see
`enpire/env/docs/NEW_TASK.md`.

---

## PLD auto-research — pin insertion

```bash
export RL_DATA_PATH=/path/outside/repo/rl-data
export ENPIRE_YAM_STATION=my-yam

uv run enpire rl control health
uv run enpire rl control pause --confirm-control
uv run enpire rl control restart --confirm-control   # prints run_dir=...
uv run enpire rl learner --task pin_insertion        # terminal 1
uv run enpire rl actor  --task pin_insertion         # terminal 2
bash tmux/realworld_rl/rl_gear.sh \
  --task pin_insertion --station my-yam --use-spacemouse   # terminal 3
uv run enpire rl control resume --confirm-control
# at trial budget:
uv run enpire rl control pause --confirm-control
uv run enpire rl score \
  --data-dir /path/outside/repo/rl-data/<run-id> --window 50 --plot
```

---

## Other CaP tasks

```bash
# Start services first
uv run enpire services start --profile cap-real
uv run enpire services start --profile robot --station my-yam --confirm-motion

uv run enpire cap run cube-pick      --station my-yam --confirm-motion
uv run enpire cap run gpu-handover   --station my-yam --confirm-motion
uv run enpire cap run gpu-reset      --station my-yam --confirm-motion
uv run enpire cap run ziptie-reset   --station my-yam --confirm-motion
```

---

## Development

```bash
uv run pytest -q tests/enpire
uv run ruff check enpire tests/enpire
```

Consult `enpire/env/docs/source_provenance.yaml` before moving migrated Forge
code.  Add a characterization test before refactoring existing behavior.

---

## Safety

- Never put API keys, device serials, calibration results, or checkpoints in Git.
- Never start robot motion without `--confirm-motion` / `--confirm-control`.
- The verifier, reset, and score implementations are read-only to a research agent.
- See `SECURITY.md` and `enpire/policy/autoresearch_instruction.md`.
