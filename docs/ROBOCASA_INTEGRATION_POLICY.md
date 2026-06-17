# RoboCasa Policy Evaluation

> **Status**: Active development on `tonghe/robocasa365-policy` branch
> **Last verified**: 2026-04-15

**Cross-references**: [ROBOCASA_INTEGRATION](ROBOCASA_INTEGRATION.md) | [ROBOCASA_RANDOMNESS](ROBOCASA_RANDOMNESS.md) | [remote_serving](remote_serving.md)

---

## 1. Overview

Two policy evaluation pipelines for RoboCasa environments:

| Pipeline | Tasks | Model | Env Wrapper | Scene Source |
|----------|-------|-------|-------------|--------------|
| **PandaOmron24** | 24 atomic tasks | GR00T N1.6 | `GrootRoboCasaEnv` (old robocasa) | Old robocasa scenes |
| **RoboCasa365 Benchmark** | 50 tasks (18 atomic_seen + 16 composite_seen + 16 composite_unseen) | GR00T N1.5 / N1.6 | `RoboCasaGymEnv` (robocasa365) + `N16RoboCasaEnv` | robocasa365 2500 pretrain scenes |

GR00T N1.6 was trained on old robocasa scenes (PandaOmron24). Running N1.6 on robocasa365 scenes is out-of-distribution and expected to underperform.

GR00T N1.5 is the official benchmark model, trained on robocasa365's 2500 pretrain scenes. It is the primary model for the RoboCasa365 benchmark.

## 2. Repository Dependencies

Four repos are needed. Only `lecar-tbd` is modified; the others are read-only dependencies (we create venvs inside them but never commit).

| Repo | What it provides | Needed for | Path on lecar-s1 |
|------|-----------------|------------|-------------------|
| **lecar-tbd** | Eval orchestrators, inference engine, docs | Both N1.5 and N1.6 | `/usr0/tonghez/lecar-tbd` |
| **Isaac-GR00T** (NVIDIA) | N1.6 model code, `GrootRoboCasaEnv` wrapper, old robocasa | **N1.6 PandaOmron24 only** | `/usr0/tonghez/Isaac-GR00T` |
| **Isaac-GR00T-benchmark** (robocasa-benchmark fork) | N1.5 model code, gymnasium-1.0-compatible `MultiStepWrapper` | **N1.5 RoboCasa365** (also used by N1.6 for MultiStepWrapper) | `/usr0/tonghez/Isaac-GR00T-benchmark` |
| **robocasa365** (robocasa365_release branch) | 365 task classes, `RoboCasaGymEnv`, `TASK_SET_REGISTRY`, kitchen assets | **RoboCasa365 benchmark** (both N1.5 and N1.6) | `/usr0/tonghez/robocasa365` |

### 2.1 Clone commands

```bash
# For N1.6 PandaOmron24 eval:
git clone https://github.com/Tonghe-Zhang/Isaac-GR00T.git

# For N1.5 RoboCasa365 benchmark:
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/robocasa-benchmark/Isaac-GR00T.git Isaac-GR00T-benchmark
cd Isaac-GR00T-benchmark && git checkout HEAD -- . && cd ..

# For robocasa365 environments (both N1.5 and N1.6 on 365 tasks):
git clone --branch robocasa365_release https://github.com/robocasa/robocasa.git robocasa365
```

### 2.2 Which repos are needed for which pipeline?

```
PandaOmron24 + N1.6:
  lecar-tbd ──► Isaac-GR00T (server venv + client venv)

RoboCasa365 + N1.5:
  lecar-tbd ──► Isaac-GR00T-benchmark (server venv + client venv)
              ──► robocasa365 (task classes + assets)

RoboCasa365 + N1.6:
  lecar-tbd ──► Isaac-GR00T (server venv only)
              ──► Isaac-GR00T-benchmark (client venv + MultiStepWrapper)
              ──► robocasa365 (task classes + assets)
```

## 3. Environment Setup

There are **4 separate venvs** — two for N1.6/PandaOmron24 and two for N1.5/RoboCasa365. They must stay isolated (different mujoco, numpy, robocasa versions).

### 3.0 OSMO one-command shortcut

On OSMO, the fastest path is now:

```bash
cd /mnt/amlfs-02/shared/<you>/forge
bash scripts/setup_robocasa365_eval.sh
```

That script is idempotent and does the full RoboCasa365 benchmark setup:

1. Runs `scripts/setup_env_osmo.sh` for the shared forge env and base assets.
2. Runs `scripts/setup_grootpool.sh` for the N1.5 server env and checkpoint.
3. Clones `robocasa365_release` into `/mnt/amlfs-02/shared/<you>/robocasa365` by default.
4. Builds an isolated `Isaac-GR00T-benchmark/client_venv/` with the benchmark client deps (`gymnasium==1.0.0`, `mujoco==3.3.1`, `pyzmq`, `msgpack`, `scipy`, `opencv-python-headless`, `tqdm`, `av`).
5. Reuses the `setup_env_osmo.sh` asset download by symlinking the six RoboCasa asset directories into the `robocasa365` clone when possible; otherwise downloads them into the clone.
6. Writes `.env.grootpool` with `GROOT_N15_MODEL_PATH` and `ROBOCASA365_CLIENT_PYTHON`, so `run_eval_365.py` works after a simple `source .env.grootpool`.

### 3.1 N1.6 Server Venv (for PandaOmron24)

Loads the GR00T N1.6 model and serves actions via ZMQ (msgpack serialization).

```bash
cd Isaac-GR00T/gr00t/eval/sim/robocasa
python3.12 -m venv model_server_venv
source model_server_venv/bin/activate

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Flash Attention (prebuilt for cu128 + Python 3.12)
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl

# GR00T N1.6 model code (editable from the repo)
pip install -e ../../..  # installs gr00t package from Isaac-GR00T root

# Server deps
pip install pyzmq msgpack tyro transformers accelerate diffusers einops timm albumentations

# Old robocasa (for GrootRoboCasaEnv obs key mapping)
pip install --no-deps -e ../../../external_dependencies/robocasa
pip install numpy==1.26.4 mujoco==3.2.6 numba==0.60.0 llvmlite==0.43.0
```

**Path**: `Isaac-GR00T/gr00t/eval/sim/robocasa/model_server_venv/`

### 3.2 N1.6 Client Venv (for PandaOmron24)

Runs robosuite environments, connects to the N1.6 server.

```bash
cd Isaac-GR00T/gr00t/eval/sim/robocasa
python3.12 -m venv robocasa_uv/.venv
source robocasa_uv/.venv/bin/activate

# Old robocasa + robosuite
pip install --no-deps -e ../../../external_dependencies/robocasa
pip install numpy==1.26.4 mujoco==3.2.6 numba==0.60.0 llvmlite==0.43.0
pip install gymnasium==0.29.1 pyzmq msgpack scipy opencv-python-headless tqdm
```

**Path**: `Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/`

> **WARNING**: Do NOT install robocasa365 into this venv. It replaces the robocasa package, upgrades numpy/mujoco, and breaks PandaOmron24.

### 3.3 N1.5 Server Venv (for RoboCasa365 benchmark)

Loads the GR00T N1.5 model and serves actions via ZMQ (torch serialization — different protocol from N1.6).

```bash
cd Isaac-GR00T-benchmark
python3.12 -m venv model_server_venv
source model_server_venv/bin/activate

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Flash Attention
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl

# GR00T N1.5 model code (editable from benchmark repo)
pip install -e .

# Server deps
pip install diffusers transformers accelerate pyzmq msgpack einops timm albumentations==1.4.18
```

**Path**: `Isaac-GR00T-benchmark/model_server_venv/`

### 3.4 N1.5/N1.6 Client Venv (shared for all RoboCasa365 eval)

Runs robocasa365 environments. Used by both N1.5 and N1.6 eval on RoboCasa365 tasks.

```bash
cd Isaac-GR00T-benchmark
python3.12 -m venv client_venv
source client_venv/bin/activate

# GR00T benchmark code (for MultiStepWrapper, SimulationInferenceClient)
pip install -e .

# robocasa365 (editable install from the cloned repo)
pip install -e /path/to/robocasa365

# Pin gymnasium to 1.0.0 (robocasa365 deps want <1.0, but we need 1.0 for AsyncVectorEnv API)
pip install gymnasium==1.0.0

# Ensure correct mujoco for robocasa365
pip install mujoco==3.3.1

# Client deps
pip install pyzmq msgpack scipy opencv-python-headless tqdm
```

**Path**: `Isaac-GR00T-benchmark/client_venv/`

> **Note**: `pip install -e robocasa365` may downgrade gymnasium. Always re-run `pip install gymnasium==1.0.0` after installing robocasa365.
>
> **Shortcut on OSMO**: if you already ran `scripts/setup_env_osmo.sh`, `run_eval_365.py` can reuse the shared forge env instead of this dedicated `client_venv`. Pass `--client-python /mnt/amlfs-02/shared/<you>/python_envs/forge/bin/python`, or set `ROBOCASA365_CLIENT_PYTHON`. The orchestrator resolves the client interpreter in this order: CLI flag, env var, `UV_PROJECT_ENVIRONMENT/bin/python`, repo `.venv/bin/python`, then the legacy `Isaac-GR00T-benchmark/client_venv/bin/python`.

### 3.5 robocasa365 Kitchen Assets (~10 GB)

robocasa365 needs kitchen fixtures, objects, and textures from HuggingFace. The NVIDIA assets are split into 86 individual zips (not one big file).

```bash
# Set HF_HOME to a large disk (root / may be small)
export HF_HOME=/usr0/tonghez/.cache/huggingface

# Use any Python with huggingface_hub installed
python3 -c "
from huggingface_hub import HfApi, hf_hub_download
from zipfile import ZipFile
import os

ROBOCASA = '/path/to/robocasa365/robocasa/models/assets'

# 1. robocasa-assets repo (textures, objaverse, aigen objects)
for fname in ['textures.zip', 'generative_textures.zip', 'objaverse.zip', 'aigen_objs.zip']:
    print(f'Downloading {fname}...')
    zpath = hf_hub_download('robocasa/robocasa-assets', fname, repo_type='dataset')
    ZipFile(zpath).extractall(ROBOCASA if 'texture' in fname else f'{ROBOCASA}/objects')

# 2. NVIDIA assets (86 per-category zips)
api = HfApi()
files = [f for f in api.list_repo_files('nvidia/PhysicalAI-Kitchen-Assets', repo_type='dataset') if f.endswith('.zip')]
for f in files:
    print(f'Downloading {f}...')
    zpath = hf_hub_download('nvidia/PhysicalAI-Kitchen-Assets', f, repo_type='dataset')
    dest = f'{ROBOCASA}/fixtures' if f.startswith('fixtures') else f'{ROBOCASA}/objects/lightwheel'
    os.makedirs(dest, exist_ok=True)
    ZipFile(zpath).extractall(dest)

print('Done')
"
```

### 3.6 Venv Summary

| Venv | Location | Python | Key packages | Pipeline |
|------|----------|--------|-------------|----------|
| N1.6 server | `Isaac-GR00T/.../model_server_venv/` | 3.12 | torch+cuda, flash-attn, gr00t (N1.6), mujoco 3.2.6 | PandaOmron24 |
| N1.6 client | `Isaac-GR00T/.../robocasa_uv/.venv/` | 3.12 | old robocasa, mujoco 3.2.6, numpy 1.26.4, gymnasium 0.29 | PandaOmron24 |
| N1.5 server | `Isaac-GR00T-benchmark/model_server_venv/` | 3.12 | torch+cuda, flash-attn, gr00t (N1.5), diffusers | RoboCasa365 |
| N1.5 client | `Isaac-GR00T-benchmark/client_venv/` | 3.12 | robocasa365, mujoco 3.3.1, gymnasium 1.0.0, numpy 2.x | RoboCasa365 |

## 4. Model Checkpoints

| Model | Source | Path on lecar-s1 |
|-------|--------|-------------------|
| GR00T N1.6-3B | https://huggingface.co/nvidia/GR00T-N1.6-3B | /usr0/tonghez/PretrainedModels/GR00T-N1.6-3B |
| GR00T N1.5 multitask | robocasa/robocasa365_checkpoints on HF | /usr0/tonghez/PretrainedModels/gr00t_n1-5/multitask_learning/checkpoint-120000 |

N1.5 download:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="robocasa/robocasa365_checkpoints",
    allow_patterns="gr00t_n1-5/multitask_learning/checkpoint-120000/*",
    local_dir="/usr0/PretrainedModels/gr00t_n1-5_multitask_checkpoint-120000",
    max_workers=4, resume_download=True,
)
```

## 5. Pipeline Architecture

### File Layout

```
cap/saved_scripts/robocasa/policy_eval/gr00t/
├── _common/
│   ├── __init__.py          # Shared imports
│   ├── pandaomron.py        # N1.6 paths, ModelServer (PandaOmron24)
│   └── robocasa365.py       # N1.5/N1.6 paths, ModelServer365 (RoboCasa365)
├── _servers/
│   └── n15.py               # Standalone N1.5 ZMQ server
├── _workers/
│   ├── pandaomron.py        # PandaOmron24 client (GrootRoboCasaEnv)
│   └── robocasa365.py       # RoboCasa365 client (RoboCasaGymEnv + N16RoboCasaEnv)
├── run_PandaOmron24.py      # PandaOmron24 orchestrator
├── run_eval_365.py          # RoboCasa365 orchestrator
└── task_registry_365.json   # Pre-extracted task lists + horizons
```

### System Diagram

```
run_eval_365.py (orchestrator)
  ├── ModelServer365 (1 per GPU, ZMQ server)
  │     N1.5: _servers/n15.py (torch serialization)
  │     N1.6: Isaac-GR00T/run_gr00t_server.py (msgpack)
  ├── _workers/robocasa365.py (1 per task, subprocess)
  │     N1.5: N15PolicyBackend (torch) + RoboCasaGymEnv
  │     N1.6: ZMQPolicyBackend (msgpack) + N16RoboCasaEnv
  └── inference_policy() (batched AsyncVectorEnv + MultiStepWrapper)
```

```
run_PandaOmron24.py (orchestrator)
  ├── ModelServer (1 per GPU, ZMQ server via Isaac-GR00T)
  ├── _worker.py (1 per task, subprocess, GrootRoboCasaEnv)
  └── inference_policy() (batched AsyncVectorEnv + MultiStepWrapper)
```

### GPU Affinity

Per worker subprocess:
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> MUJOCO_EGL_DEVICE_ID=<gpu_id>
```
This pins both PyTorch (model inference) and MuJoCo EGL rendering to the same physical GPU, avoiding cross-GPU memory traffic.

## 6. PandaOmron24 Evaluation (GR00T N1.6)

Batch offline evaluation of NVIDIA GR00T N1.6 (and compatible policies) against the full RoboCasa 24-task PandaOmron benchmark. The pipeline runs N parallel environments per GPU, distributes tasks across GPUs, and reports success rates with 95% confidence intervals.

### Pipeline Overview

```
run_PandaOmron24.py          # orchestrator -- argument parsing, GPU dispatch, reporting
  ├─ _common.py              # path constants, ModelServer CM, TeeStream
  ├─ _worker.py              # per-task eval loop (inference_policy -> EpisodeResult[])
  └─ cap/policy/inference.py # batched env stepping (AsyncVectorEnv + MultiStepWrapper)
```

| Component | File | Description |
|-----------|------|-------------|
| Orchestrator | `cap/saved_scripts/robocasa/policy_eval/gr00t/run_PandaOmron24.py` | CLI, ThreadPoolExecutor GPU dispatch, print_summary, save_results |
| Shared helpers | `cap/saved_scripts/robocasa/policy_eval/gr00t/_common.py` | `ModelServer` CM, `TeeStream`, machine-detect path constants |
| Per-task worker | `cap/saved_scripts/robocasa/policy_eval/gr00t/_worker.py` | Calls `inference_policy()`, writes `task_results.json` |
| Inference engine | `cap/policy/inference.py` | `inference_policy()` -- single/batched env stepping |
| Video recorder | `cap/policy/video.py` | `BatchedVideoTracker`, H.264 encoding, overlay text |

### ModelServer (GPU-local server lifecycle)

Each GPU runs one `ModelServer` context manager that:
1. Launches `gr00t_inference_server.py` as a subprocess with the GPU's `CUDA_VISIBLE_DEVICES`
2. Polls a ZMQ ping until the server is ready (timeout: 180 s), with fast-fail if the proc dies
3. On exit: SIGTERM -> wait(10 s) -> SIGKILL if stuck; closes log file

Workers connect to their GPU's ZMQ port (`5555 + gpu_idx` by default). In `--server-host` mode, no local server is started -- the worker connects to the remote host directly.

### Launch Commands

```bash
cd cap/saved_scripts/robocasa/policy_eval/gr00t

# All 24 PandaOmron tasks -- single GPU (default GPU 0):
python run_PandaOmron24.py \
    --n-eps-per-task 10 --n-parallel-envs 10 --record-video

# All 24 tasks -- 4 GPUs in parallel (tasks split evenly):
python run_PandaOmron24.py \
    --n-eps-per-task 10 --n-parallel-envs 10 --gpus 0 1 2 3 --record-video

# Subset of tasks:
python run_PandaOmron24.py \
    --tasks OpenDrawer CloseDrawer CoffeeSetupMug \
    --n-eps-per-task 10 --n-parallel-envs 5 --gpus 0 1

# Remote model server (no local server start):
python run_PandaOmron24.py \
    --server-host 10.0.0.5 --tasks OpenDrawer --n-eps-per-task 10

# With explicit model checkpoint path (via env var):
GROOT_MODEL_PATH=/path/to/gr00t_n16_checkpoint \
python run_PandaOmron24.py \
    --n-eps-per-task 10 --n-parallel-envs 10 --gpus 0 1 2 3
```

> **Note**: On machines where `python` is not in PATH, use the venv interpreter directly:
> `/usr0/tonghez/lecar-tbd/.venv/bin/python run_PandaOmron24.py ...`

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--n-eps-per-task` | 10 | Episodes per task |
| `--n-parallel-envs` | 2 | Parallel AsyncVectorEnv workers per GPU |
| `--gpus` | `[0]` | GPU IDs to use (one server per GPU) |
| `--tasks` | all 24 | Subset of task short names to evaluate |
| `--base-port` | 5555 | First server port (increments per GPU) |
| `--record-video` | off | Save per-episode per-camera MP4s |
| `--server-host` | local | Remote model server IP (no local server started) |

### Output Layout

```
logs/GR00TN1.6/eval_panda_ip_<timestamp>/
  benchmark_results.json       # per-task success rates + 95% CIs (Wilson + Clopper-Pearson)
  stdout.log / stderr.log
  <TaskShortName>/
    task_results.json           # per-episode success list
    client.log                  # worker stdout
    videos/ep_000_s1/           # per-episode videos (if --record-video)
      side_left.mp4
      wrist.mp4
  server_gpu<N>.log             # model server stdout
```

`benchmark_results.json` format:
```json
{
  "CloseDrawer": {
    "success_rate": 0.8,
    "n_episodes": 10,
    "ci_wilson_95": [0.49, 0.94],
    "ci_cp_95": [0.44, 0.97],
    "successes": [true, false, true, "..."]
  }
}
```

### Reproducing the Official GR00T N1.6 Numbers

The 24 PandaOmron tasks and official success rates are hardcoded in `run_PandaOmron24.py:TASKS`. To reproduce:

```bash
python run_PandaOmron24.py \
    --n-eps-per-task 10 --n-parallel-envs 10 --gpus 0 1 2 3 --record-video
```

Expected: results within Wilson 95% CI of official numbers for most tasks.

## 7. RoboCasa365 Benchmark Evaluation (GR00T N1.5 / N1.6)

### 7.1 Task Sets

Defined in `task_registry_365.json` (pre-extracted from robocasa365's dataset registry to avoid importing robocasa in the orchestrator):

| Task Set | Count | Description |
|----------|-------|-------------|
| `atomic_seen` | 18 | Single-step tasks seen during training |
| `composite_seen` | 16 | Multi-step tasks seen during training |
| `composite_unseen` | 16 | Multi-step tasks NOT seen during training |
| **Total** | **50** | Official benchmark suite |

### 7.2 Chunking: Full Chunk Execution (NOT Receding Horizon)

The official benchmark uses `n_action_steps=16`: the model predicts 16 actions and **all 16 are executed** before re-querying the model. This differs from the PandaOmron24 pipeline which uses receding horizon (predict 16, execute 8).

In code: `_workers/robocasa365.py` sets `InferencePolicyConfig(action_horizon=16, replan_horizon=args.n_action_steps)`. With the default `--n-action-steps 16`, all 16 predicted actions are executed per query.

Reference: `robocasa-benchmark/Isaac-GR00T/scripts/run_eval.py` line 156-161, `MultiStepWrapper.step()` loops `for step in range(self.n_action_steps)`.

### 7.3 N1.6 Adapter (N16RoboCasaEnv)

The robocasa365 `RoboCasaGymEnv` produces a subset of observation states. N1.6 expects additional joint/ee states not exposed by robocasa365. The `N16RoboCasaEnv` (defined inside `_make_env_n16()` in `_workers/robocasa365.py`) subclasses `RoboCasaGymEnv` and overrides `get_observation()` to extract extra states from the same raw robosuite obs dict -- zero overhead, no double `_get_observations()` call.

Extra states added:

| Raw robosuite key | Mapped state key | Shape |
|-------------------|------------------|-------|
| `robot0_joint_pos` | `state.joint_position` | (7,) |
| `robot0_joint_pos_cos` | `state.joint_position_cos` | (7,) |
| `robot0_joint_pos_sin` | `state.joint_position_sin` | (7,) |
| `robot0_joint_vel` | `state.joint_velocity` | (7,) |
| `robot0_eef_pos` | `state.end_effector_position_absolute` | (3,) |
| `robot0_eef_quat` | `state.end_effector_rotation_absolute` | (4,) |
| `robot0_gripper_qvel` | `state.gripper_qvel` | (2,) |

Observation key renames:

| robocasa365 key | N1.6 key |
|-----------------|----------|
| `video.robot0_agentview_left` | `video.res256_image_side_0` |
| `video.robot0_agentview_right` | `video.res256_image_side_1` |
| `video.robot0_eye_in_hand` | `video.res256_image_wrist_0` |
| `annotation.human.task_description` | `annotation.human.action.task_description` |

### 7.4 ZMQ Protocol Difference

| Property | N1.5 Server | N1.6 Server |
|----------|-------------|-------------|
| Serialization | `torch.save`/`torch.load` over ZMQ | `msgpack` over ZMQ |
| Server script | `_servers/n15.py` (standalone) | `Isaac-GR00T/gr00t/eval/run_gr00t_server.py` |
| Server venv | `Isaac-GR00T-benchmark/model_server_venv/` | `Isaac-GR00T/gr00t/eval/sim/robocasa/model_server_venv/` |
| Client backend | `N15PolicyBackend` (torch) | `ZMQPolicyBackend` (msgpack) |

The worker auto-selects the backend based on `--model-version`.

### 7.5 Launch Commands

```bash
cd cap/saved_scripts/robocasa/policy_eval/gr00t

# N1.5 on all 50 tasks, 4 GPUs:
python run_eval_365.py --model-version n15 \
  --task-set atomic_seen composite_seen composite_unseen \
  --gpus 0 1 2 3 --record-video

# N1.6 on atomic_seen only:
python run_eval_365.py --model-version n16 \
  --task-set atomic_seen --gpus 0 1

# Quick test:
python run_eval_365.py --model-version n15 \
  --tasks OpenDrawer --n-eps-per-task 2 --n-parallel-envs 1

# Quick test using the shared forge env from scripts/setup_env_osmo.sh:
python run_eval_365.py --model-version n15 \
  --client-python /mnt/amlfs-02/shared/<you>/python_envs/forge/bin/python \
  --tasks OpenDrawer --n-eps-per-task 2 --n-parallel-envs 1

# Quick test after scripts/setup_robocasa365_eval.sh:
source /mnt/amlfs-02/shared/<you>/forge/.env.grootpool
/mnt/amlfs-02/shared/<you>/Isaac-GR00T-benchmark/model_server_venv/bin/python run_eval_365.py \
  --model-version n15 \
  --tasks OpenDrawer --n-eps-per-task 2 --n-parallel-envs 1

# Stats only (reprint results from existing run):
python run_eval_365.py --stats-only --log-dir <dir>
```

### 7.6 CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model-version` | `n15` | Model version: `n15` or `n16` |
| `--task-set` | `atomic_seen composite_seen composite_unseen` | Task groups to evaluate (space-separated) |
| `--tasks` | (none) | Individual task names (overrides `--task-set`) |
| `--split` | `pretrain` | Data split: `pretrain` or `target` |
| `--n-eps-per-task` | 50 | Episodes per task |
| `--n-parallel-envs` | 5 | Parallel AsyncVectorEnv workers per GPU |
| `--n-action-steps` | 16 | Steps executed per policy query (16=official full chunk, 8=receding horizon) |
| `--gpus` | `[0]` | GPU IDs to use (one server per GPU) |
| `--base-port` | 5555 | First server port (increments per GPU) |
| `--server-host` | (none) | Remote server host (skip local server launch) |
| `--client-python` | auto-resolved | Worker interpreter override; defaults to `ROBOCASA365_CLIENT_PYTHON`, then `UV_PROJECT_ENVIRONMENT/bin/python`, then `.venv/bin/python`, then `Isaac-GR00T-benchmark/client_venv/bin/python` |
| `--record-video` | off | Save per-episode per-camera MP4s |
| `--stats-only` | off | Print stats from existing results (requires `--log-dir`) |
| `--log-dir` | auto-generated | Log directory (for `--stats-only` or override) |

### 7.7 Output Layout

```
logs/robocasa365/{model_version}_{split}_{timestamp}/
  benchmark_results.json        # per-task success rates + 95% CIs
  stdout.log / stderr.log
  server_gpu{N}.log             # model server stdout per GPU
  {TaskName}/
    task_results.json           # per-episode success list
    client.log                  # worker stdout
    videos/ep_000/
      side_left.mp4
      side_right.mp4
      wrist.mp4
```

`benchmark_results.json` format:
```json
{
  "timestamp": "20260415T123456",
  "model_version": "n15",
  "model_path": "/usr0/tonghez/PretrainedModels/gr00t_n1-5/...",
  "split": "pretrain",
  "n_eps_per_task": 50,
  "tasks_completed": 50,
  "tasks_total": 50,
  "aggregate": {
    "mean": 42.5,
    "std": 15.3,
    "n_tasks": 50
  },
  "per_task": {
    "OpenDrawer": {
      "success_rate": 70.0,
      "successes": 7,
      "total": 10,
      "horizon": 500,
      "wilson_ci_95": [39.68, 89.22],
      "clopper_pearson_ci_95": [34.75, 93.33],
      "per_episode": [true, false, true, "..."]
    }
  }
}
```

### 7.8 Resume Support

The orchestrator skips tasks with existing `task_results.json` in the log directory. Safe to Ctrl-C and re-run with the same `--log-dir` -- completed tasks are loaded from cache and only remaining tasks are evaluated.

## 8. Benchmark Results — GR00T N1.5 × RoboCasa365 (10 eps/task)

> Run date: 2026-04-15, lecar-s1 (4× RTX PRO 6000 Blackwell 98GB)

### Summary

| Task Set | Tasks | Avg Success | Std |
|:---|:---:|:---:|:---:|
| **atomic_seen** | 18 | **40.6%** | 28.2% |
| **composite_seen** | 16 | **11.9%** | 17.4% |
| **composite_unseen** | 16 | **2.5%** | 5.6% |

### atomic_seen (18 tasks)

| Task | Rate |
|:---|:---:|
| PickPlaceCounterToCabinet | 90% |
| PickPlaceSinkToCounter | 90% |
| OpenStandMixerHead | 70% |
| PickPlaceCounterToStove | 70% |
| PickPlaceToasterToCounter | 70% |
| OpenDrawer | 60% |
| CloseFridge | 40% |
| OpenCabinet | 40% |
| SlideDishwasherRack | 40% |
| TurnOnSinkFaucet | 40% |
| PickPlaceDrawerToCounter | 30% |
| TurnOnElectricKettle | 30% |
| CloseBlenderLid | 20% |
| CloseToasterOvenDoor | 20% |
| TurnOnMicrowave | 20% |
| CoffeeSetupMug | 0% |
| NavigateKitchen | 0% |
| TurnOffStove | 0% |

### composite_seen (16 tasks)

| Task | Rate |
|:---|:---:|
| LoadDishwasher | 70% |
| RinseSinkBasin | 30% |
| PreSoakPan | 20% |
| ScrubCuttingBoard | 20% |
| KettleBoiling | 10% |
| PackIdenticalLunches | 10% |
| SearingMeat | 10% |
| SetUpCuttingStation | 10% |
| StirVegetables | 10% |
| DeliverStraw | 0% |
| GetToastedBread | 0% |
| PrepareCoffee | 0% |
| StackBowlsCabinet | 0% |
| SteamInMicrowave | 0% |
| StoreLeftoversInBowl | 0% |
| WashLettuce | 0% |

### composite_unseen (16 tasks)

| Task | Rate |
|:---|:---:|
| WaffleReheat | 20% |
| GarnishPancake | 10% |
| WashFruitColander | 10% |
| ArrangeBreadBasket | 0% |
| ArrangeTea | 0% |
| BreadSelection | 0% |
| CategorizeCondiments | 0% |
| CuttingToolSelection | 0% |
| GatherTableware | 0% |
| HeatKebabSandwich | 0% |
| MakeIceLemonade | 0% |
| PanTransfer | 0% |
| PortionHotDogs | 0% |
| RecycleBottlesByType | 0% |
| SeparateFreezerRack | 0% |
| WeighIngredients | 0% |

### Key observations

- **Seen atomic (~41%) >> Seen composite (~12%) >> Unseen composite (~2.5%)** — clear performance gradient.
- Composite tasks require the policy to implicitly track multi-phase progress from pixels alone — see `docs/ROBOCASA_SYSTEM2_SURVEY.md` for analysis.
- GR00T N1.5 was finetuned on `pretrain300` (100 demos/task). N1.6 was NOT trained on RoboCasa (zero-shot only).

## 9. Known Issues

1. **N1.6 on robocasa365 is OOD**: N1.6 was trained on old robocasa scenes. robocasa365 uses different kitchen assets, layouts, and 2500 pretrain scenes. Expect near-zero success on most tasks.

2. **gymnasium 1.0.0 must be pinned** in the benchmark client venv. lerobot (a transitive dependency) wants `gymnasium<1.0`, but we need `gymnasium>=1.0` for the `AsyncVectorEnv` API used by `inference_policy()`.

3. **Root disk on lecar-s1 is small (49 GB)**: Always set `HF_HOME=/usr0/tonghez/.cache/huggingface` before downloading models or assets. The default `~/.cache/huggingface` lives on root and will fill it.

4. **robocasa365 asset download**: The `download_kitchen_assets.py` script expects a single zip per category, but the `nvidia/PhysicalAI-Kitchen-Assets` HF repo has 86 individual per-subcategory zips. Manual extraction may be required.

5. **AsyncVectorEnv EGL deadlock**: When other users have GPU processes with EGL contexts (e.g. MuJoCo, Isaac Sim), `AsyncVectorEnv(context='spawn')` with `n_envs>1` can deadlock during sub-process spawn. Workarounds: (a) kill competing GPU processes before launching, (b) use `--n-parallel-envs 1`, or (c) use `--log-dir` to resume after partial completion.

5. **N1.5 server startup is slow (~2-3 min)**: The `ModelServer365` has a 300 s startup timeout (vs 180 s for N1.6) because the N1.5 model loads slowly. If the server still times out, check GPU memory and ensure no other processes are occupying the GPU.

## 10. Composite Task Evaluation — Detailed Instructions

### 10.1 Running composite_seen (16 tasks, N1.5)

SSH to lecar-s1 and run inside tmux:

```bash
ssh LeCAR_4xRTX6000BlackWell_97GB
tmux new -s robocasa_eval
cd /usr0/tonghez/lecar-tbd/cap/saved_scripts/robocasa/policy_eval/gr00t

# Use the N1.5 server venv (has numpy, scipy, msgpack, zmq):
/usr0/tonghez/Isaac-GR00T-benchmark/model_server_venv/bin/python run_eval_365.py \
  --model-version n15 \
  --task-set composite_seen \
  --n-eps-per-task 10 \
  --gpus 0 1 2 3 \
  --record-video
```

This evaluates 16 composite_seen tasks (4 per GPU), 10 episodes each, with `n_envs=5` (default) and video recording.

**composite_seen tasks** (defined in `task_registry_365.json`):
DeliverStraw, GetToastedBread, KettleBoiling, LoadDishwasher, PackIdenticalLunches,
PreSoakPan, PrepareCoffee, RinseSinkBasin, ScrubCuttingBoard, SearingMeat,
SetUpCuttingStation, StackBowlsCabinet, SteamInMicrowave, StirVegetables,
StoreLeftoversInBowl, WashLettuce.

### 10.2 Monitoring

```bash
# From another terminal, watch stdout:
ssh LeCAR_4xRTX6000BlackWell_97GB
tail -f /usr0/tonghez/lecar-tbd/logs/robocasa365/n15_pretrain_<TIMESTAMP>/stdout.log

# Or check per-task results as they complete:
find /usr0/tonghez/lecar-tbd/logs/robocasa365/n15_pretrain_<TIMESTAMP>/ \
  -name task_results.json -exec sh -c 'echo "=== $(dirname {} | xargs basename) ==="; cat {}' \;
```

### 10.3 Resume after failure

Re-run the same command with `--log-dir <existing_dir>` to skip already-completed tasks:

```bash
/usr0/tonghez/Isaac-GR00T-benchmark/model_server_venv/bin/python run_eval_365.py \
  --model-version n15 \
  --task-set composite_seen \
  --n-eps-per-task 10 \
  --gpus 0 1 2 3 \
  --record-video \
  --log-dir /usr0/tonghez/lecar-tbd/logs/robocasa365/n15_pretrain_<TIMESTAMP>
```

### 10.4 View results

```bash
/usr0/tonghez/Isaac-GR00T-benchmark/model_server_venv/bin/python run_eval_365.py \
  --stats-only \
  --log-dir /usr0/tonghez/lecar-tbd/logs/robocasa365/n15_pretrain_<TIMESTAMP>
```

## File Reference

| File | Description |
|------|-------------|
| `cap/saved_scripts/robocasa/policy_eval/gr00t/_common/pandaomron.py` | N1.6 paths, `ModelServer` CM, `TeeStream` |
| `cap/saved_scripts/robocasa/policy_eval/gr00t/_common/robocasa365.py` | N1.5/N1.6 paths, `ModelServer365` CM, `get_model_config()` |
| `cap/saved_scripts/robocasa/policy_eval/gr00t/_servers/n15.py` | Standalone N1.5 ZMQ server (launches `Gr00tPolicy` + `RobotInferenceServer`) |
| `cap/saved_scripts/robocasa/policy_eval/gr00t/_workers/pandaomron.py` | PandaOmron24 client subprocess |
| `cap/saved_scripts/robocasa/policy_eval/gr00t/_workers/robocasa365.py` | RoboCasa365 client subprocess (N15PolicyBackend, N16RoboCasaEnv) |
| `cap/saved_scripts/robocasa/policy_eval/gr00t/run_PandaOmron24.py` | PandaOmron24 orchestrator (24 tasks, multi-GPU) |
| `cap/saved_scripts/robocasa/policy_eval/gr00t/run_eval_365.py` | RoboCasa365 orchestrator (50 tasks, multi-GPU, resume) |
| `cap/saved_scripts/robocasa/policy_eval/gr00t/task_registry_365.json` | Pre-extracted task lists + per-task horizons |
| `cap/policy/inference.py` | `inference_policy()` -- batched env stepping engine |
| `cap/policy/video.py` | `BatchedVideoTracker` -- H.264 video recording |
