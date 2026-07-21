# RoboCasa Randomness & Determinism

How to control every level of randomness in the RoboCasa benchmark.

**Cross-references**: [ROBOCASA_INTEGRATION](ROBOCASA_INTEGRATION.md) | [VISER_CUROBO_PLANNER](VISER_CUROBO_PLANNER.md)

---

## Randomness hierarchy

| # | Dimension | Options | Control | Sampling |
|---|-----------|---------|---------|----------|
| 1 | **Task** | 362+ tasks | `env_name` / `load_task()` | Deterministic |
| 2 | **Layout** | 60 (1-10 test, 11-60 train) | `layout_ids` / `ROBOCASA_LAYOUT_ID` | Uniform per reset |
| 3 | **Style** | 60 (1-10 test, 11-60 train) | `style_ids` / `ROBOCASA_STYLE_ID` | Uniform per reset |
| 4 | **Generative Textures** | ~103-104 per surface type (cabinet, counter, floor, wall) | `generative_textures="100p"` | Uniform per episode |
| 5 | **Object Category** | 198 categories | Task-specific `_get_obj_cfgs()` | Task-dependent |
| 6 | **Object Instance** | Multiple per category (Objaverse, Lightwheel, AI-gen) | `obj_instance_split`, `obj_registries` | Uniform per placement |
| 7 | **Object Placement** | Continuous within fixture reset regions | `PlacementInitializer` | Random per reset |
| 8 | **Robot Base Position** | Continuous (default: +/-15cm x, +/-5cm y) | `robot_spawn_deviation_pos_*` | Gaussian |
| 9 | **Robot Base Rotation** | Continuous (default: +/-0 rad) | `robot_spawn_deviation_rot` | Uniform |
| 10 | **Camera Poses** | Continuous noise on 4 cameras | `randomize_cameras=True` | Gaussian (0.1m pos, 6deg rot) |
| 11 | **Fixture Variants** | 6 cabinet panel types + 18 visual mesh panels + handle variants | Via layout+style YAML | Per-style |
| 12 | **Clutter** | On/off | `clutter_mode=0/1` | Binary |
| 13 | **Joint Init Noise** | Continuous | `initialization_noise` | Gaussian/Uniform |

---

## Layout/style group shortcuts

| ID | Meaning | Layouts |
|----|---------|---------|
| `-1` | Test set | 1-10 |
| `-2` | Train set | 11-60 |
| `-3` | All | 1-60 |
| `-4` | No-island | [1, 3, 5, 6, 8] |
| `-5` | Island | [2, 4, 7, 9, 10] |
| `-6` | Dining | [2, 4, 7, 8, 9, 10] |

Style groups: `-1` (test 1-10), `-2` (train 11-60), `-3` (all 1-60).

Layout+style YAML configs live in:
- `third_party/robocasa/robocasa/models/assets/scenes/kitchen_layouts/{test|train}/layoutXXX.yaml`
- `third_party/robocasa/robocasa/models/assets/scenes/kitchen_styles/{test|train}/styleXXX.yaml`

---

## How `seed` propagates

RoboCasa uses a single master RNG: `self.rng = np.random.default_rng(seed)` (in `robosuite/environments/base.py:145`).

This `env.rng` is passed to and controls:

| Subsystem | File | How |
|-----------|------|-----|
| Layout/style sampling | `kitchen.py:595` | `self.rng.choice(self.layout_and_style_ids)` |
| Object sampling | `kitchen_object_utils.py:357` | `sample_kitchen_object(..., rng=self.rng)` |
| Object placement | `placement_samplers.py:50` | `PlacementInitializer(rng=self.rng)` |
| Robot base position | `env_utils.py:1531` | `env.rng.uniform(...)` |
| Generative textures | `texture_swap.py:430` | `get_random_textures(self.rng)` |
| Agentview cameras | `camera_utils.py:221` | `env.rng.normal(...)` |

### Known bug: eye-in-hand camera

`third_party/robocasa/robocasa/utils/camera_utils.py:225-233` uses **unseeded `np.random.normal()`** instead of `env.rng.normal()` for the eye-in-hand camera. This means `randomize_cameras=True` is **not fully deterministic**.

**Workaround**: keep `randomize_cameras=False` (the default).

---

## Pinning all randomness (levels 1-12)

### Environment variables

| Env var | Purpose | Example |
|---------|---------|---------|
| `ROBOCASA_LAYOUT_ID` | Pin layout | `3` |
| `ROBOCASA_STYLE_ID` | Pin style | `5` |
| `ROBOCASA_SEED` | Master RNG seed (controls levels 4-10) | `42` |

These are read in `cap/env/__init__.py` and forwarded to `robosuite.make()`.

### Launch command (fully deterministic)

```bash
ROBOCASA_LAYOUT_ID=3 \
ROBOCASA_STYLE_ID=5 \
ROBOCASA_SEED=42 \
CAP_CUROBO_PORT=8611 CAP_ROBOT_TYPE=panda CAP_AGENT_NAME=demo \
  uv run python -u run_script.py \
  --file cap/saved_scripts/robocasa_test_planning.py \
  --env "robocasa:PickPlaceSinkToCounter" --cap-port 18600 --no-log --record
```

### What each control pins

| Control | Pins levels |
|---------|-------------|
| `env_name` (task in `--env`) | 1 (task) |
| `ROBOCASA_LAYOUT_ID` | 2 (layout) |
| `ROBOCASA_STYLE_ID` | 3 (style) |
| `ROBOCASA_SEED` | 4 (textures), 5-6 (object category/instance), 7 (object placement), 8-9 (robot base), 10 (agentview cameras) |
| Layout+style YAML | 11 (fixture variants) — deterministic for a given layout+style pair |
| `clutter_mode=0` (default) | 12 (clutter) |

Level 13 (joint init noise) uses robosuite's `initialization_noise` parameter.

### Python API (without env vars)

```python
from cap.env.robocasa import RoboCasaEnv

env = RoboCasaEnv(
    env_name="PickPlaceSinkToCounter",
    robot="PandaOmron",
    layout_ids=3,
    style_ids=5,
    seed=42,
    randomize_cameras=False,      # default, keep for determinism
    generative_textures=None,     # disable texture randomization
    # robot_spawn_deviation_pos_x=0.0,  # optionally zero out robot jitter
    # robot_spawn_deviation_pos_y=0.0,
)
```

---

## Object details

### Object registries

| Registry | Description |
|----------|-------------|
| `objaverse` | Photorealistic 3D scans |
| `lightwheel` | Simpler geometry |
| `aigen` | AI-generated models |

Default: `("objaverse", "lightwheel")`. Controlled via `obj_registries` parameter.

### Object instance splits

| Split | Meaning |
|-------|---------|
| `None` | All instances |
| `"pretrain"` | All but last 4 (or first half) |
| `"target"` | Last 4 (or second half) |

### Object groups (for task-specific sampling)

Pre-defined: `"food"`, `"in_container"`, `"container"`, `"cookware"`, `"pots_and_pans"`, `"oven_ready"`, `"freezer_items"`, etc.

198 object categories total (apple, banana, bowl, plate, can, ...).

---

## Reset Non-Determinism and `_post_reset_state`

### The problem: `env.reset()` is not deterministic across calls

Even with a fixed seed, calling `env.reset()` multiple times produces **different scenes** each time. This is because:

1. **RNG advances on each `reset()`** — The master RNG (`env.rng = np.random.default_rng(seed)`) advances through layout/style sampling, object sampling, placement sampling, etc. on every `reset()` call. The second reset consumes different random numbers than the first, yielding different objects and positions.

2. **`sim_state_initial` is captured too early** — robosuite captures `sim_state_initial` in `MujocoEnv.__init__()` before `_reset_internal()` runs. Restoring it gives an empty scene with no objects placed.

3. **Unseeded RNG in fixture constructors** — Some RoboCasa fixture constructors call `np.random.default_rng()` without a seed argument, which reads from `/dev/urandom`. This makes even full teardown+recreate (destroying and re-constructing the env) non-deterministic.

### The solution: post-reset sim state snapshot

`RoboCasaEnv` captures a MuJoCo sim state snapshot **after** the first full `env.reset()` completes (objects placed and settled):

```python
# In RoboCasaEnv.__init__() — after env.reset() finishes:
self._post_reset_state = self._env.sim.get_state()  # env.py:119
```

`reset_to_initial()` restores this snapshot instead of calling `env.reset()`:

```python
def reset_to_initial(self):
    sim.set_state(self._post_reset_state)
    sim.forward()
```

This gives **identical** object instances, positions, and robot state on every call — no RNG advancement, no re-randomization.

### Agent retry loop integration

The agent retry loop (`agent_step.py:ObserverStep`) resets the env between iterations so each attempt starts from the same state:

- **Prefers `reset_to_initial`** (deterministic sim state restore) in the skills namespace
- **Falls back to `reset_env`** if `reset_to_initial` is not available
- For RoboCasa direct mode, `reset_env` IS in the namespace and calls `reset_to_initial()` internally (`cap/env/robocasa/skills.py:392-400`)
- For CapServer mode, `reset_to_initial` is exposed as a separate RPC alongside `reset_env`

### What this means for experiments

| Scenario | Behavior |
|----------|----------|
| Agent retry (same run) | `reset_to_initial` restores exact same scene — deterministic |
| `load_task()` (switch task) | Full teardown+recreate — new seed, new scene |
| New `run_agent.py` invocation with same seed/layout/style | First reset is deterministic (same seed); subsequent retries use snapshot |
| `env.reset()` called directly (bypass wrapper) | Non-deterministic — RNG advances, different objects |

---

## Standard benchmark evaluation

The full RoboCasa benchmark evaluates over **task x layout x style** combinations. With `layout_ids=-3, style_ids=-3`, each `reset()` samples from all 60x60 = 3,600 scene combinations.

For controlled experiments, pin layout+style and vary only `seed` to get within-scene randomness (object instances, placements, robot base).

---

## Key source files

| File | Role |
|------|------|
| `cap/env/__init__.py` | Env var parsing (`ROBOCASA_LAYOUT_ID`, `ROBOCASA_STYLE_ID`, `ROBOCASA_SEED`) |
| `cap/env/robocasa.py` | Re-export stub (backward compat — imports from `robocasa/env.py`) |
| `cap/env/robocasa/env.py` | Refactored `RoboCasaEnv` with `_post_reset_state` snapshot and `reset_to_initial()` |
| `cap/env/robocasa/skills.py` | Direct-mode skills namespace (`reset_env` calls `reset_to_initial`) |
| `cap/agent/agent_step.py` | `ObserverStep` retry reset logic (prefers `reset_to_initial`, falls back to `reset_env`) |
| `third_party/robocasa/robocasa/environments/kitchen/kitchen.py` | Main Kitchen env, `_setup_model()`, `_reset_internal()` |
| `third_party/robocasa/robocasa/models/scenes/scene_registry.py` | Layout/style enums and group unpacking |
| `third_party/robocasa/robocasa/models/objects/kitchen_objects.py` | 198 object categories |
| `third_party/robocasa/robocasa/models/objects/kitchen_object_utils.py` | `sample_kitchen_object()` |
| `third_party/robocasa/robocasa/utils/camera_utils.py` | Camera configs and randomization |
| `third_party/robocasa/robocasa/utils/env_utils.py` | Robot base placement |
| `third_party/robocasa/robocasa/utils/texture_swap.py` | Generative texture sampling |
| `third_party/robocasa/robocasa/utils/placement_samplers.py` | Object placement initializer |
