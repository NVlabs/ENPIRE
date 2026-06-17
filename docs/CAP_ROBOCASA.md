# CAP — RoboCasa (Sim Benchmark, Eval & AutoResearch)

> Complement to [`CAP_DESIGN.md`](CAP_DESIGN.md). `CAP_DESIGN.md` is the overall
> CAP architecture and the **real-world YAM** setup; **this doc** covers the
> **RoboCasa** sim benchmark — how it plugs into the same CAP stack, the two ways
> it is used (code-as-policy agent vs GR00T policy eval), the runnable commands,
> and the RoboCasa **autoresearch** loop it feeds.

RoboCasa is the mobilie-manipulation benchmark (robosuite/MuJoCo + RoboCasa
fixtures and assets). In CAP it serves three purposes:

1. A **sim embodiment** for the CAP code-as-policy agent (same tools as real YAM).
2. A **policy-eval harness** for GR00T (N1.5 / N1.6) on the RoboCasa365 task suite.
3. The **scoring substrate** for RoboCasa autoresearch (autonomous skill discovery).

---

## 1. How RoboCasa plugs into the CAP stack

CAP's layering is **Agent → Tools → Server → Env** (see `CLAUDE.md > Env Layer`).
RoboCasa is one `Env` backend (`cap/env/robocasa/`), so the agent and tool layers
are identical to real YAM — only the bottom layer changes.

```
CAP Agent (LLM code-as-policy)        cap/agent/cap_agent.py · run_agent.py
   │  same tool namespace as real YAM
Tools  (freespace_move, detect_object, set_gripper, vlm_query, …)
   │
Env    RoboCasaEnv                     cap/env/robocasa/env.py
       ├─ EnvProtocol      (step / observation / render)
       ├─ EefControlProtocol (per-tick EE delta → OSC_POSE)
       └─ TaskProtocol     (reset / reward / success)
   │
robosuite (MuJoCo OSC controller) + robocasa (kitchen tasks & assets)
```

Key integration facts (verify in `cap/env/robocasa/env.py`, `cap/env/__init__.py`):

- **Controller is pinned to robosuite `OSC_POSE`.** The env consumes EE-delta
  actions (world-frame position delta + axis-angle orientation delta, clipped to
  ±1.0). There is **no external IK** in the env — OSC handles it. cuRobo joint
  trajectories (from `freespace_move`) are executed by per-waypoint FK + OSC
  tick-stepping inside `cap/env/robocasa/skills.py`.
- **Env naming:** `robocasa:<TaskName>:<RobotName>` (e.g.
  `robocasa:PickPlaceCounterToCabinet:PandaOmron`). Bare `robocasa` defaults to
  `PickPlaceCounterToCabinet` + `PandaOmron`. Parsed in `cap/env/__init__.py`.
- **Embodiments:** `PandaOmron` (single-arm Panda on Omron base) and
  `GR1ArmsOnly` (humanoid arms). Embodiment specs in `cap/prompt/embodiment/`,
  profiles in `cap/env/profile.py`.
- **Cameras:** CAP camera names map to RoboCasa obs keys via
  `profile.camera_obs_key_map` — e.g. `top → robot0_agentview_left_image`,
  `wrist → robot0_eye_in_hand_image`. Rendering runs on a dedicated thread
  (EGL context affinity); set `MUJOCO_GL=egl` for headless.
- **Determinism:** RoboCasa randomizes layout/style/seed. Pin them with
  `ROBOCASA_LAYOUT_ID`, `ROBOCASA_STYLE_ID`, `ROBOCASA_SEED` (see
  [`ROBOCASA_RANDOMNESS.md`](ROBOCASA_RANDOMNESS.md)). Agent-mode `reset_env()`
  restores a post-reset snapshot rather than re-randomizing.
- **Two entry styles:** direct in-process (`run_agent.py` / `run_script.py`
  build `RoboCasaEnv` directly) or a thin Portal-RPC `EnvServer`
  (`cap/env/robocasa/server.py`) used by the tmux launcher + Viser.

See [`ROBOCASA_INTEGRATION.md`](ROBOCASA_INTEGRATION.md),
[`ROBOCASA_INTEGRATION_POLICY.md`](ROBOCASA_INTEGRATION_POLICY.md),
[`ROBOCASA_ORACLE.md`](ROBOCASA_ORACLE.md).

---

## 2. Two usage modes

### Mode A — CAP agent (code-as-policy)
The LLM (or a human, oracle mode) writes Python using the CAP tool namespace to
solve a RoboCasa task. cuRobo plans collision-free motion (remote GPU via
`CAP_CUROBO_HOST`/`CAP_CUROBO_PORT`); perception is SAM3 + depth + VLM; success
is read from the task oracle for scoring. This is the same agent used on real
YAM — RoboCasa just supplies the env, cameras, and ground-truth success check.

### Mode B — GR00T policy evaluation (RoboCasa365)
A trained GR00T policy (N1.5 / N1.6) is rolled out batch-style over the
RoboCasa365 task suite with no agent in the loop. A per-GPU model server answers
action-chunk requests; parallel sim workers step the env and record per-episode
success. Used to benchmark checkpoints. Driver:
`cap/saved_scripts/archive/robocasa/policy_eval/gr00t/run_eval_365.py`
(+ `run_PandaOmron24.py` for the older 24-task PandaOmron suite). See
[`ROBOCASA_INTEGRATION_POLICY.md`](ROBOCASA_INTEGRATION_POLICY.md).

---

## 3. Skill library

`cap/saved_scripts/robocasa_skill_library/` (~58 modules) holds reusable,
versioned RoboCasa skills the agent composes or extends. Categories:

- **Grasp / motion:** `vertical_grasp`, `incremental_grasp`, `robust_grasp`,
  `descend_and_grasp`, `arm_motion`, `post_grasp_lift`, `lift`, `move_to_target_xyz`.
- **Fixtures:** `cabinet_handle_*`, `drawer_handle_*`, `fridge_handle_*`,
  `*_control_state`, `stove_knob_*`, `sink_faucet_motion`, `kettle_switch_motion`.
- **Pick-and-place:** `pnp_counter_to_cabinet_*`, `vertical_place`,
  `place_with_orientation`.
- **Specialized:** `pan_burner_alignment`, `stirring_motion`, `scrubbing_motion`,
  `under_water_holding`, `thin_object_insertion`, `bowl_stacking`,
  `sink_spout_rotation`, `toaster_extract`, `microwave_control`,
  `dishwasher_control`, `coffee_machine_control`.
- **Perception/utility:** `detection`, `planner_world`, `show_debug_balls`,
  `hover_orientation_search`, `nudge_down_and_regrasp`, `groot_only`.

Skills are plain Python exposing `<name>_v1(...)` functions over the shared tool
namespace; generated code imports them (`from skill_library.<mod> import <fn>`).
See [`SKILL_LIBRARY_ROBOCASA.md`](SKILL_LIBRARY_ROBOCASA.md).

---

## 4. Runnable commands

### 4.0 Install
```bash
# robosuite is a git submodule; robocasa is a vendored tree under third_party/.
git submodule update --init third_party/robosuite

# robosuite + robocasa (editable path sources) + numba/qpsolvers/pynput/hidapi.
# Add --extra cap_tools too — freespace_move needs nvidia-curobo for motion
# planning, which lives in the cap_tools extra:
uv sync --extra robocasa --extra cap_tools

# Kitchen assets. Full set is ~10 GB; you can fetch a subset by --type.
# A pick-place / open-drawer scene needs textures + lightwheel fixtures +
# objaverse/aigen/lightwheel objects (~2.6 GB total). 'all' gets everything.
echo y | uv run python third_party/robocasa/robocasa/scripts/download_kitchen_assets.py \
  --type tex fixtures_lw objs_objaverse objs_aigen objs_lw
# (the OSMO helper scripts/setup_robocasa365_eval.sh is Lustre-specific and is
#  for the GR00T365 client venv, not this code-as-policy path.)
```

### 4.1 Launch RoboCasa env server + Viser (interactive)
`scripts/launch_robocasa_tmux.sh` opens a split tmux: left = `EnvServer`, right =
Viser cuRobo planner UI.
```bash
bash scripts/launch_robocasa_tmux.sh --task robocasa:PrepareCoffee
bash scripts/launch_robocasa_tmux.sh \
  --task robocasa:PickPlaceCounterToCabinet --seed 7 \
  --env-port 18600 --viser-port 8890 --curobo-port 9611
```
Requires cuRobo already running on `--curobo-port`; SAM3/AnyGrasp hosts optional.
Open the UI at `http://127.0.0.1:8890`.

### 4.2 Run the CAP agent on a RoboCasa task (code-as-policy)
```bash
# pick a config from experiments/experiment/
uv run python run_agent.py experiment=pick_place_sink_to_counter

# point at a remote cuRobo GPU
CAP_CUROBO_HOST=<gpu-ip> CAP_CUROBO_PORT=8611 \
  uv run python run_agent.py experiment=open_drawer

# deterministic eval (pin layout/style/seed)
ROBOCASA_LAYOUT_ID=3 ROBOCASA_STYLE_ID=5 ROBOCASA_SEED=42 \
  uv run python run_agent.py experiment=pnp_counter_to_cabinet
```
Available RoboCasa experiment configs (`experiments/experiment/*.yaml`):
`pick_place_sink_to_counter`, `pick_place_v1`, `pnp_counter_to_cabinet`,
`pnp_counter_to_stove`, `pnp_counter_to_stove_single_stage`,
`pnp_drawer_to_counter`, `pnp_toaster_to_counter`, `open_drawer`,
`open_cabinet_single_stage`, `open_stand_mixer_head`, `close_fridge`,
`close_blender_lid`, `close_toaster_oven_door`, `coffee_setup_mug`,
`microwave_v1`, `turn_on_microwave`, `turn_on_electric_kettle`,
`turn_on_sink_faucet`, `turn_off_stove`, `slide_dishwasher_rack`,
`rc_open_drawer`, `rc_open_cabinet`, `rc_close_fridge`, `rc_navigate_kitchen`.

Run one saved/generated script directly (used by eval):
```bash
# run_script.py auto-starts a local cuRobo planner when runtime.curobo_port=0
# (the default agent_config value is 9400 → external server; override for local).
MUJOCO_GL=egl ROBOCASA_SEED=42 ROBOCASA_LAYOUT_ID=3 ROBOCASA_STYLE_ID=5 \
  uv run python run_script.py script_file=<code.py> \
  env.name=robocasa:OpenDrawer env.seed=42 runtime.curobo_port=0
# result.json (success + score) is written into the run's log dir; success/score
# are read from the RoboCasa task oracle via get_task_info().
```

**Oracle vs vision-only scripts.** `run_script.py` builds the tool namespace with
`runtime_role="script"`, which intentionally drops the oracle surfaces
(`get_task_info`, `reset_env`, per-fixture `_debug_info`) so eval `code.py`
cannot cheat (see `cap/env/robocasa/skills.py:make_namespace`). Oracle
code-as-policy scripts (e.g. `cap/saved_scripts/robocasa_pnp_test.py`, which
reads `get_task_info()` for object/container poses and final success) therefore
need a namespace built with `runtime_role="agent"`. This is the role the MCP
`env_worker` / agent loop use.

**cuRobo auto-start gotcha.** When `curobo_port=0`, `freespace_move` spawns the
cuRobo Portal planner via `portal.Process` (multiprocessing **spawn**). Spawn
re-imports the entry module, so any runner must keep all top-level work under
`if __name__ == "__main__":` — otherwise `_check_not_importing_main()` raises and
the parent deadlocks on the never-started child. `run_agent.py`/`run_script.py`
already satisfy this (Hydra `main()`); custom harnesses must too.

**Verified (2026-06):** an end-to-end oracle eval on `PickPlaceSinkToCounter`
(seed 42, layout 3, style 5) on a single local RTX 5090 — local cuRobo
auto-start, all 6 `freespace_move` waypoints planned, object grasped and placed —
returned `success=True, reward=1.0` from the task oracle.

### 4.3 GR00T RoboCasa365 policy eval
```bash
# one-time setup: clones benchmark + robocasa365, builds client venv, writes .env.grootpool
bash scripts/setup_robocasa365_eval.sh
source .env.grootpool

# quick smoke test (1 task, 2 eps, 1 GPU)
"$GROOTPOOL_SERVER_PYTHON" \
  cap/saved_scripts/archive/robocasa/policy_eval/gr00t/run_eval_365.py \
  --model-version n15 --tasks OpenDrawer --n-eps-per-task 2 --n-parallel-envs 1 --gpus 0

# full suite (N1.5, 4 GPUs)
"$GROOTPOOL_SERVER_PYTHON" \
  cap/saved_scripts/archive/robocasa/policy_eval/gr00t/run_eval_365.py \
  --model-version n15 \
  --task-set atomic_seen composite_seen composite_unseen \
  --n-eps-per-task 50 --n-parallel-envs 5 --gpus 0 1 2 3 --record-video
```
Key `run_eval_365.py` flags: `--model-version {n15,n16}`, `--task-set`,
`--tasks`, `--n-eps-per-task`, `--n-parallel-envs`, `--n-action-steps`
(16 = full chunk), `--gpus`, `--server-host` (use a remote server),
`--record-video`, `--stats-only --log-dir <dir>`. Results land in
`logs/robocasa365/<model>_<split>_<ts>/benchmark_results.json` (per-task success
+ 95% CIs).

### 4.4 GR00T via the grootpool middleware (sticky session routing)
```bash
bash scripts/launch_grootpool.sh 8                 # 8-GPU pool, endpoint tcp://127.0.0.1:7070
uv run python scripts/run_grootpool_eval.py \
  --env-name robocasa/PickPlaceCounterToCabinet \
  --task-description "pick the object from the counter and place it in the cabinet" \
  --n-episodes 1 --max-steps 300 --log-dir logs/grootpool_eval
```
See [`GROOTPOOL.md`](GROOTPOOL.md) (helpers: `scripts/setup_grootpool.sh`,
`wait_for_grootpool.sh`, `grootpool_dashboard.py`).

---

## 5. RoboCasa autoresearch pipeline

**Goal:** autonomously discover and codify robust RoboCasa skills — the policy is
*Python code*, not a neural net — and improve it until it passes a fixed seed
benchmark. Full design in
[`OPENCODE_ROBOCASA_AUTORL_HANDOFF.md`](OPENCODE_ROBOCASA_AUTORL_HANDOFF.md) and
[`ROBOCASA_SYSTEM2_SURVEY.md`](ROBOCASA_SYSTEM2_SURVEY.md).

### Loop
```
plan ─► explore (parallel sessions: perception / grasp / place)
     ─► codify  (write standalone code.py from exploration traces)
     ─► evaluate (tiered: 5 → 10 → 20 → 40 fixed seeds)
     ─► reflect  (VLM analyzes before/after frames, categorizes failures)
     ─► patch ─► (loop)        stop when 40-seed success ≥ 0.8
```
- **Tiered eval:** seed prefixes nest (tier-1 ⊂ tier-2 ⊂ … from
  `cap/assets/robocasa_40_seeds.txt`, each row `seed layout_id style_id`).
  `success_rate ≥ 0.8` escalates a tier; a miss patches the code and restarts at
  tier 1; tier-4 ≥ 0.8 stops. Each seed runs in its own subprocess
  (`run_script.py`) on a deterministic service slot + render GPU.
- **Vision-only constraint:** generated `code.py` may use the task *description*
  and cameras/SAM3/VLM, but **not** the oracle (`CAP_DISABLE_TASK_INFO_IN_SCRIPT=1`
  blocks `get_task_info`/oracle detection). Success is graded privately via the
  RoboCasa oracle; see [`REWARD_MODULE.md`](REWARD_MODULE.md).
- **Skill accumulation:** improved skills are written back into the
  `robocasa_skill_library` for reuse by later tasks.

### Harness & sandbox
- **Harness:** an LLM agent (Claude, driven by the OpenCode TUI) is the
  "researcher": it plans, spawns explorer subagents, codifies, and reflects. All
  robot actions go through an **MCP server** exposing `mcp__robot__*` tools
  (`env_create/reset/destroy`, `detect_object`, `freespace_move`, `gripper`,
  `vlm_query`, `evaluate_code`, `skill_list/read/write`).
- **Sandbox:** each agent/explorer gets an isolated **simulator session**
  (`session_id` + `service_slot`) so parallel agents never race shared MuJoCo
  state; generated code is graded in **per-seed subprocesses** with pinned seeds,
  slots, and GPUs.
- **In-tree vs external (this branch):** the *substrate* is present —
  `run_script.py` eval, the 40-seed list, the skill library, grootpool, and the
  GR00T eval drivers. The *driver* described in the handoff doc —
  `scripts/run_opencode_pnp_sink_to_counter.sh`, the `cap/mcp` server, the
  OpenCode TUI sidebar, and `tmux/remote_serving/launch_osmo.sh` slot pools — is
  the OpenCode harness and is **not all checked into this branch**. Treat §5 as
  the design + operational shape; wire the OpenCode/MCP front-end separately, or
  reuse the same loop logic over the in-tree pieces.

---

## 6. Task families
- **Pick & place:** counter↔cabinet, sink→counter, counter→stove,
  toaster→counter, drawer→counter.
- **Articulated:** open/close drawer, cabinet, fridge, toaster-oven door,
  dishwasher rack, stand-mixer head, blender lid.
- **Appliance control:** microwave, electric kettle, sink faucet, stove knobs,
  coffee machine.
- **Composite (RoboCasa365):** multi-step recipes — PrepareCoffee, KettleBoiling,
  WashLettuce, StackBowlsCabinet, LoadDishwasher, etc. (seen + unseen splits).

Task strategy guides live in `cap/prompt/task/robocasa_*.md`; the eval registry is
`cap/saved_scripts/archive/robocasa/policy_eval/gr00t/task_registry_365.json`.

---

## 7. Key files
| Concern | Path |
|---|---|
| Env backend | `cap/env/robocasa/{env,server,skills,mppi_client}.py`, `cap/env/__init__.py` |
| Embodiment profiles | `cap/env/profile.py`, `cap/prompt/embodiment/robocasa_*.md` |
| Agent adapter | `cap/agent/robot_adapters/robocasa.py` |
| GR00T policy runner | `cap/policy/robocasa_runner.py` |
| Agent entry | `run_agent.py`, `run_script.py`, `experiments/experiment/*.yaml` |
| Skill library | `cap/saved_scripts/robocasa_skill_library/` |
| GR00T eval | `cap/saved_scripts/archive/robocasa/policy_eval/gr00t/{run_eval_365,run_PandaOmron24}.py` |
| Setup / launch | `scripts/{setup_robocasa365_eval,launch_robocasa_tmux,setup_grootpool,launch_grootpool}.sh`, `scripts/run_grootpool_eval.py` |
| Eval seeds | `cap/assets/robocasa_40_seeds.txt` |
| Task guides | `cap/prompt/task/robocasa_*.md` |

---

## 8. Note on hardcoded hosts/paths (cleanup candidates)
A few imported files hardcode lab-specific hosts/paths (defaults only, overridable
by env var) — worth parameterizing if this leaves the lab:
- `scripts/launch_robocasa_tmux.sh` — SAM3/AnyGrasp default host `lecar-yam.wifi.local.cmu.edu`.
- `cap/saved_scripts/archive/robocasa/policy_eval/gr00t/_common/{robocasa365,pandaomron}.py`
  — `lecar-s1` fqdn check + `/usr0/<user>` and `/home/lecar/...` checkpoint paths.
- `docs/OPENCODE_ROBOCASA_AUTORL_HANDOFF.md` — `/mnt/amlfs-02/shared/...` Lustre paths.
None are credentials; they are infra/username disclosures.
