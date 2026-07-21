# ENPIRE PLD runtime

This isolated uv project contains the real-world PLD actor/learner code migrated
from `minimal_policy@81988f0`. Isolation preserves its tested JAX 0.6.1,
TensorFlow 2.18, NumPy 1.x, Gymnasium 0.29, and SciPy 1.11 stack without
constraining ENPIRE's camera, calibration, simulation, or planning extras.

```bash
uv sync --project enpire/policy/pld/runtime
uv run enpire rl learner --task pin_insertion
uv run enpire rl actor --task pin_insertion
```

Set `RL_DATA_PATH` to an external data/checkpoint root. No datasets, model
weights, W&B credentials, or station addresses belong in this directory.
