# Real-world workflows

These commands preserve the working Forge/YAM/PLD control flow behind clearer
entry points. Commands that can move hardware require an explicit confirmation
or use a clearly motion-capable shell launcher. Keep an emergency stop within
reach and clear the workspace before every run.

## 1. Create and validate a station

Station identifiers, device serials, paths, poses, and calibration results live
outside the repository.

```bash
uv run enpire station init --station my-yam
uv run enpire station register --station my-yam
uv run enpire station show --station my-yam
uv run enpire station doctor --station my-yam
```

`station register` performs the original unplug-and-identify flow and writes
owner-only registration artifacts under the ENPIRE data home. It does not use
`sudo` or install system files. Review the generated udev rules and aliases
before an administrator installs them, then reload udev and reconnect devices.

## 2. Gravity compensation and calibration

Inspect the gravity-comp command without launching hardware by consulting its
help, then explicitly permit motion:

```bash
uv run enpire station gravcomp --station my-yam --both --camera both --confirm-motion
```

Calibration needs a printed ChArUco board and cannot be automated. See
[`CALIBRATION_BOARD.md`](CALIBRATION_BOARD.md) for how to generate, print at the
correct scale, and mount it — the board goes on the gripper for the top camera
but stays fixed in the world for the wrist cameras.

Mount the ChArUco board rigidly to the instructed gripper, set the external YAM
model root, and run the integrated calibration:

```bash
export ENPIRE_YAM_MODEL_ROOT=/path/to/yam-model-assets
uv run enpire station calibrate \
  --station my-yam \
  --camera top \
  --output-xml /path/outside/repo/station_calibrated.xml \
  --confirm-motion
```

Validate the emitted record without hardware:

```bash
uv run enpire station validate-calibration /path/outside/repo/calibration.json
```

Update the external station profile's `calibration_bundle` only after the
residual checks and a physical sanity check pass.

## 3. Start services in one command

Export required station-local paths in the launching process environment.
Service panes inherit that environment; ENPIRE does not implicitly source
checkout-local environment files. Credentials remain in a secret manager.

```bash
# SAM3 + licensed AnyGrasp
uv run enpire services start --profile perception

# SAM3 + AnyGrasp + cuRobo + optional local NVIDIA VLM adapter
uv run enpire services start --profile cap-real

# YAM servers (motion-capable)
uv run enpire services start --profile robot --confirm-motion

# Everything above (motion-capable)
uv run enpire services start --profile all --confirm-motion

uv run enpire services status
tmux attach -t enpire
```

For local AnyGrasp, configure externally obtained files first:

```bash
export ANYGRASP_SDK_ROOT=/path/to/anygrasp_sdk
export ANYGRASP_CHECKPOINT=/path/to/checkpoint_detection.tar
export ANYGRASP_LICENSE_ZIP=/path/to/license.zip
# Only when the SDK build tree does not contain MinkowskiEngineBackend/_C:
export ANYGRASP_MINKOWSKI_BACKEND=/path/to/MinkowskiEngineBackend/_C.so
```

Use `--dry-run` on `services start` to inspect every process command.

## 4. Run code-as-policy tasks

List or inspect a task without motion:

```bash
uv run enpire cap list
uv run enpire cap run pickup --prompt "blue cube" --station my-yam --dry-run
```

Run the generic prompted pickup using the standard segmentation, grasp planning,
and control tools:

```bash
uv run enpire cap run pickup --prompt "blue cube" \
  --station my-yam --confirm-motion
```

Only tasks whose complete, reviewed source is distributed are exposed by
`enpire cap list`. GPU and zip-tie CaP scripts from internal deployments are
not part of this release. The separately licensed PLD configurations for
`gpu_insertion` and `ziptie` remain available through the PLD entry points.

## 5. Pin-insertion PLD pipeline

Install the isolated runtime once, choose an external dataset root, and supply
station-local reset/reward files. Never use the zero-valued example pose files
on real hardware.

```bash
uv sync --project enpire/policy/pld/runtime --extra dev
export RL_DATA_PATH=/path/outside/repo/rl-data
export ENPIRE_YAM_STATION=my-yam
export ENPIRE_RL_INITIAL_POSITIONS=/path/outside/repo/pin_initial_positions.yaml
export ENPIRE_RL_REWARD_CONFIG=/path/outside/repo/pin_reward.yaml
```

Launch one process per terminal:

```bash
# terminal 1: learner
uv run enpire rl learner --task pin_insertion

# terminal 2: actor
uv run enpire rl actor --task pin_insertion

# terminal 3: robot-side bridge/data collection
bash tmux/realworld_rl/rl_gear.sh --task pin_insertion \
  --station my-yam --use-spacemouse
```

Hydra overrides pass through with repeated `--override`, for example:

```bash
uv run enpire rl actor --task pin_insertion \
  --override train.eval_mode=true \
  --override train.connect_to_learner=false \
  --override train.resume_checkpoint_path=/path/outside/repo/checkpoint
```

The same actor/learner entry supports `gpu_insertion` and `ziptie`.

## 6. PushT environment/reset loop

The source-faithful PushT robot environment and supervisor are included. Supply
a taught station pose and a top-camera goal image with a sibling
`goal_top_meta.json` success-region description:

```bash
export RL_DATA_PATH=/path/outside/repo/rl-data
export ENPIRE_YAM_STATION=my-yam
export ENPIRE_RL_INITIAL_POSITIONS=/path/outside/repo/pusht_initial_positions.yaml
export PUSHT_GOAL_IMAGE=/path/outside/repo/pusht_goal_top.png
bash tmux/realworld_rl/rl_pusht.sh
```

The PushT bridge speaks the same policy-server contract at `localhost:8965`.
Unlike pin/GPU/zip-tie, a PushT-specific PLD learner preset is not claimed by
this release; connect a compatible actor or add a characterized policy preset.

## 7. GPU insertion and full reset loop

Start the PLD learner and actor with `--task gpu_insertion`, then run the
robot-side bridge. Set socket selection and whether the CaP handover prepares
the GPU before learning:

```bash
export RL_DATA_PATH=/path/outside/repo/rl-data
export ENPIRE_YAM_STATION=my-yam
GPU_RL_PREPARE=1 GPU_TARGET_SOCKET_NUMBER=3 \
  bash tmux/realworld_rl/rl_gear.sh \
  --task gpu_insertion --station my-yam --use-spacemouse \
  --episode-timeout-s 12.0
```

The two-slot insertion/press/unplug/reset loop is:

```bash
GPU_RL_PREPARE=1 GPU_TARGET_SOCKET_NUMBER=1 \
  bash tmux/realworld_rl/gpu_insertion_dual_full_cycle.sh \
  --station my-yam --use-spacemouse
```

## 8. Auto-research reset/evaluate loop

The robot bridge exposes a small control surface and the source PLD metric is
packaged in the isolated runtime. All state-changing calls are explicit:

```bash
uv run enpire rl control health
uv run enpire rl control help
uv run enpire rl control pause --confirm-control
uv run enpire rl control restart --confirm-control
# Start learner, verify the returned run directory is being ingested, then actor:
uv run enpire rl learner --task pin_insertion
uv run enpire rl actor --task pin_insertion
uv run enpire rl control resume --confirm-control
```

At the experiment boundary, park the robot and score the exact run directory
returned by `restart`:

```bash
uv run enpire rl control pause --confirm-control
uv run enpire rl score --data-dir /path/outside/repo/run --window 50
```

The score reports all-episode and pure-RL cumulative/final/peak rolling success
rates and writes a CSV. Add `--plot` for a PNG. Read
`enpire/policy/autoresearch_instruction.md` before giving an agent write access;
reset, success, safety, evaluation, and metric code are outside policy-edit
scope.

All datasets, model checkpoints, calibration, learned goal metadata, and run
artifacts remain outside Git. See [SECURITY.md](../SECURITY.md).
