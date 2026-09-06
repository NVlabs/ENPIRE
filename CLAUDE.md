# ENPIRE Agent Guide

ENPIRE is a reset-execute-verify-refine harness for code-as-policy, robot tools,
real-world RL, and automated policy research. Read `AGENTS.md` first; it is the
canonical repository workflow for both human and coding-agent contributors.

## Start here

```bash
# Required first: cuRobo is a submodule that uv resolves as an editable path
# dependency, so an uninitialized third_party/curobo breaks EVERY uv command
# with "does not appear to be a Python project" — including uv sync --extra dev.
git submodule update --init --recursive

uv sync --extra dev
uv run enpire tools list
uv run enpire examples list
uv run pytest -q tests/enpire
```

For real workflows, read `enpire/env/docs/REAL_WORLD_WORKFLOWS.md` and
`enpire/env/docs/DEPENDENCIES.md`. External dependencies that cannot be
installed automatically have their own guides:
`enpire/env/docs/CUROBO_SETUP.md`, `enpire/env/docs/ANYGRASP_SETUP.md`
(per-machine licence), and `enpire/env/docs/CALIBRATION_BOARD.md` (needs a
printed board and a human). Inspect commands with `--dry-run` before enabling
motion. The PLD learner is installed from its isolated runtime project.

Do not start robot motion, install udev rules, connect to external services, or
run credentialed evaluations unless the user explicitly requests that action.
Never put credentials in source, commands, prompts, fixtures, or artifacts.

## Architecture

- `enpire/env/forge/` — public runtime, tool registry, artifacts, station support.
- `enpire/env/examples/` — numbered learning path and task capsules.
- `enpire/env/docs/` — practitioner and migration documentation.
- `enpire/policy/interface.py` — common code/learned policy contract.
- `enpire/policy/pld/` — optional real-world PLD actor and learner.
- `cap/` — original Forge implementations retained behind public adapters.
- `tmux/realworld_rl/` — source-faithful robot-side RL launchers and reset loops.

## Source fidelity

Before moving task code, consult
`enpire/env/docs/source_provenance.yaml`. Preserve the original algorithms and
write characterization tests before refactoring. GPU insertion and zip-tie
scripts originate from internal Forge feature branches; see
`enpire/env/docs/source_provenance.yaml` for commit provenance.
