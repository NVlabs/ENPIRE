# Survey: System 2 Claude Agent for RoboCasa Composite Tasks

## 1. RoboCasa-365 Task Structure

**65 atomic + 300 composite = 365 total tasks.**

### Atomic tasks (65)
Single-step manipulation primitives. Examples:
- `OpenDrawer`, `CloseCabinet`, `CoffeeSetupMug`, `TurnOnMicrowave`
- `num_subtasks = 1` for all atomic tasks

### Composite tasks (300)
Multi-phase tasks grouped into ~65 activity categories:

| Subtask count | # of tasks |
|:---:|:---:|
| 1 | 7 |
| 2 | 89 |
| 3 | 43 |
| 4 | 56 |
| 5 | 24 |
| 6 | 9 |
| 7 | 34 |
| 8 | 17 |
| 9 | 5 |
| 11 | 11 |
| 12 | 2 |
| 15 | 2 |
| 16 | 1 |

Most complex: `DivideBuffetTrays` (16 subtasks).

### Class hierarchy — there is NO separate composite base class

Both atomic and composite tasks directly subclass `Kitchen(ManipulationEnv)`:

```
robosuite.environments.base.MujocoEnv
  └── robosuite.environments.robot_env.RobotEnv
        └── robosuite.environments.manipulation.ManipulationEnv
              └── robocasa.environments.kitchen.kitchen.Kitchen   ← ALL tasks
                    ├── atomic/   (e.g. OpenDrawer)
                    └── composite/ (e.g. MultistepSteaming)
```

Source: `third_party/robocasa/robocasa/environments/kitchen/kitchen.py:77`

---

## 2. The Core Question: Task Boundaries

### Q: Does the task description change between subtask boundaries?

**No.** The language instruction is set once at `reset()` and never changes during the episode.

Flow:
1. `env.reset()` → calls `get_ep_meta()` → sets `ep_meta["lang"]` describing ALL phases at once
2. `env.step()` → `GymWrapper` extracts `env.get_ep_meta().get("lang", "")` every step, but it's the same string
3. The obs key `"annotation.human.task_description"` is constant for the entire episode

Source: `third_party/robocasa/robocasa/wrappers/gym_wrapper.py:264,284`

### Q: Is the task description provided by the model or the env?

**By the env.** Each task class overrides `get_ep_meta()` to generate a language instruction that describes ALL phases. Example from `MultistepSteaming`:

```python
ep_meta["lang"] = (
    "Turn on the sink faucet. "
    f"Then move the {vegetable_name} from the counter to the sink. "
    "Turn off the sink. Move the vegetable from the sink to the pot next to the stove. "
    f"Finally move the pot to the {self.knob.replace('_', ' ')} burner."
)
```

Source: `third_party/robocasa/robocasa/environments/kitchen/composite/steaming_food/multistep_steaming.py:38-46`

### Q: How to know the task boundary?

**The env does NOT expose subtask boundaries to the policy.** Specifically:

1. **No subtask signal in observations.** The obs dict contains images + proprioception + one fixed language string. No phase index, no subtask progress indicator.

2. **No subtask signal in `info` dict.** `step()` returns `info = {"success": bool}` — only the final all-or-nothing success. Source: `gym_wrapper.py:341`

3. **Internal progress tracking is hidden.** Tasks use private flags like `self.water_was_turned_on`, `self.vegetable_was_in_sink` to track progress, but these are internal to `_check_success()` and never appear in the observation or info dict.

4. **`num_subtasks` in `task_attributes.json` is documentation metadata only.** It's not exposed as a runtime API — it exists in the docs folder for the website, not in the environment code.

**Bottom line:** A System 1 policy (end-to-end neural net) has no way to know when it crosses a subtask boundary. A System 2 agent must infer boundaries from visual observation or task description parsing.

---

## 3. Benchmarking Protocol

### Official eval: end-to-end, no subtask boundaries

Source: `third_party/robocasa/docs/benchmarking/multitask_learning.md`

- **50 episodes** randomly sampled per task
- **Binary success** — reward is `1.0` if `_check_success()` returns True, else `0.0`. No partial credit.
- **Single horizon per task** — no mid-episode reset or phase transition
- All composite tasks treated identically to atomic tasks: `reset()` → step loop → done

### Success is ALL-OR-NOTHING

For composite tasks, `_check_success()` requires ALL conditions to be met simultaneously:

```python
# MultistepSteaming._check_success()
return (
    self.water_was_turned_on        # phase 1 happened
    and self.vegetable_was_in_sink   # phase 2 happened
    and (not water_on)               # phase 3 happened (turned off sink)
    and pot_on_burner                # phase 4 happened
    and vegetable_in_pot             # phase 5 happened
)
```

Some conditions are stateful (use flags that latch True once satisfied), others are instantaneous (must be true at the moment of checking). This means some ordering is enforced (water must have been on before vegetable was in sink) but the final check requires everything done.

### Pretraining splits

| Split | Atomic | Composite | Total |
|:---:|:---:|:---:|:---:|
| pretrain50 | 18 | 32 | 50 |
| pretrain100 | 65 | 35 | 100 |
| pretrain200 | 65 | 135 | 200 |
| pretrain300 | 65 | 235 | 300 |

Standard benchmark uses pretrain300 (100 demos/task, ~482 hours of data).

Evaluation splits:
- `atomic_seen` — atomic tasks in pretraining
- `composite_seen` — composite tasks in pretraining
- `composite_unseen` — composite tasks NOT in pretraining (generalization test)

### Leaderboard baselines

**GR00T N1.6-3B was NOT trained on RoboCasa data.** The N1.6 pretraining mixture includes YAM bimanual, AGIBot Genie1, simulated Galaxea R1 Pro (BEHAVIOR), and Unitree G1 loco-manipulation — but no RoboCasa. NVIDIA's official N1.6 eval is a **zero-shot** benchmark on 24 atomic PandaOmron tasks only (avg 66.22%). No composite tasks were evaluated.

**GR00T N1.5 WAS finetuned on RoboCasa** — using the `pretrain300` split (100 demos/task, ~482 hours). The N1.5 checkpoint at `multitask_learning/checkpoint-120000` is the finetuned model.

| Model | Trained on RoboCasa? | Eval scope |
|:---|:---:|:---|
| GR00T N1.5 (finetuned) | **Yes** — finetuned on pretrain300 | 50 tasks (our task_registry_365.json) |
| GR00T N1.6-3B (base) | **No** — zero-shot only | 24 atomic PandaOmron tasks |

Our `task_registry_365.json` defines a broader 50-task eval (18 atomic_seen + 16 composite_seen + 16 composite_unseen). Composite task results from learned policies are expected to be significantly lower than atomic task results.

---

## 4. What a System 2 Agent Needs

### The problem with System 1

Current pipeline (`cap/policy/inference.py`, `cap/saved_scripts/robocasa/policy_eval/gr00t/_workers/robocasa365.py`) runs open-loop:
```
policy = load_model()
obs = env.reset()          # gets language instruction
while not done:
    action = policy(obs)   # System 1: obs → action, no reasoning
    obs, reward, done, info = env.step(action)
```

For atomic tasks (1 subtask), this works fine. For composite tasks with 4-16 subtasks and horizons up to 3500 steps, the policy must implicitly track which phase it's in, what to do next, and handle failures — all from pixels alone. This is where System 1 policies fail.

### System 2 architecture: Claude as task planner + monitor

```
┌─────────────────────────────────────────────────────┐
│  Claude Agent (System 2 — slow, deliberative)       │
│                                                     │
│  1. Parse language instruction → subtask sequence   │
│  2. For each subtask:                               │
│     a. Generate subtask-specific instruction         │
│     b. Dispatch to low-level policy (System 1)      │
│     c. Monitor visual progress (VLM)                │
│     d. Detect subtask completion or failure          │
│     e. If failed: replan / retry / skip             │
│  3. Aggregate result                                │
└──────────────┬──────────────────────────────────────┘
               │ subtask instruction + start/stop
               ▼
┌──────────────────────────────────────────────────────┐
│  Low-level Policy (System 1 — fast, reactive)        │
│  e.g. GR00T N1.6, Pi0.5, Diffusion Policy           │
│                                                      │
│  obs (images + proprio + language) → action           │
│  Runs at control_freq (20Hz)                          │
└──────────────┬───────────────────────────────────────┘
               │ action
               ▼
┌──────────────────────────────────────────────────────┐
│  RoboCasa Environment                                 │
│  obs, reward (sparse), done, info                     │
└──────────────────────────────────────────────────────┘
```

### Key design decisions

#### A. Task decomposition (before execution)

Claude parses the env-provided language instruction into an ordered subtask list:

```
Input:  "Turn on the sink faucet. Then move the broccoli from the counter
         to the sink. Turn off the sink. Move the vegetable from the sink
         to the pot next to the stove. Finally move the pot to the front
         left burner."

Output: [
  {"id": 1, "instruction": "Turn on the sink faucet", "type": "manipulate_fixture"},
  {"id": 2, "instruction": "Move the broccoli from counter to sink", "type": "pick_place"},
  {"id": 3, "instruction": "Turn off the sink faucet", "type": "manipulate_fixture"},
  {"id": 4, "instruction": "Move the broccoli from sink to the pot", "type": "pick_place"},
  {"id": 5, "instruction": "Move the pot to the front left burner", "type": "pick_place"},
]
```

This is a text-only LLM call — no vision needed. Can be done once at the start of each episode.

#### B. Subtask-specific language injection

Instead of feeding the low-level policy the full composite instruction, feed it the CURRENT subtask instruction. This narrows the policy's attention:

```python
# Instead of:
obs["annotation.human.task_description"] = full_composite_instruction  # 5 sentences

# Do:
obs["annotation.human.task_description"] = current_subtask_instruction  # 1 sentence
```

**Open question:** Does the pretrained policy (GR00T, Pi0.5) generalize to shorter/different instructions than what it was trained on? If trained on full composite descriptions, single-sentence instructions may be out-of-distribution. Needs empirical testing.

#### C. Subtask completion detection (during execution)

Since the env provides NO subtask boundary signals, the agent must detect them. Options:

| Method | Latency | Accuracy | Cost |
|:---|:---:|:---:|:---:|
| VLM vision query every N steps | ~1-2s | High | $$$ |
| Proprioception heuristics (gripper state, EE position) | ~0ms | Medium | Free |
| Reward signal (reward=1.0 means ALL done, not useful for subtasks) | 0ms | N/A | Free |
| Env internal state (hack: read `env.env._check_success()` internals) | 0ms | Perfect | Fragile |

**Recommended hybrid approach:**
1. Use proprioception heuristics for quick checks (did gripper open? did EE move away from object?)
2. Use VLM query every ~50-100 steps (2.5-5 sec of sim time at 20Hz) for authoritative phase detection
3. Use timeout-based fallback: if a subtask hasn't completed within its budget, move on or retry

#### D. Mid-episode sim state checkpointing

RoboCasa runs on MuJoCo. MuJoCo state can be saved/restored:

```python
# Save checkpoint at subtask boundary
state = env.sim.get_state()

# If subtask N+1 fails, restore and retry
env.sim.set_state(state)
env.sim.forward()
```

The existing `RoboCasaEnv.reset_to_initial()` already does this for the initial state. Extending it to subtask boundaries is straightforward.

**Caveat:** This only works in simulation. For real-robot System 2, checkpointing isn't possible.

#### E. Replanning on failure

If a subtask fails (detected via VLM or timeout):
1. **Retry** — restore checkpoint, re-attempt same subtask (max N retries)
2. **Adapt** — ask Claude to generate an alternative approach for the failed subtask
3. **Skip** — if subtask is non-blocking, move to next (risky — most subtasks are sequential)

---

## 5. Concrete Integration Points

### Existing code to extend

| Component | File | What to add |
|:---|:---|:---|
| Env wrapper | `cap/env/robocasa/env.py` | Expose sim state save/restore, subtask progress hooks |
| Policy inference | `cap/policy/inference.py` | System 2 loop: decompose → dispatch → monitor → replan |
| Eval pipeline | `cap/saved_scripts/.../run_eval_365.py` | System 2 eval mode, per-subtask metrics |
| Task registry | `cap/saved_scripts/.../task_registry_365.json` | Add `subtask_count`, `subtask_descriptions` metadata |
| VLM query | `cap/agent/tools/vlm_query.py` | Subtask completion queries |

### What the env already gives us (no changes needed)

- Full language instruction describing all phases (`obs["annotation.human.task_description"]`)
- Camera observations (3 views: agentview_left, agentview_right, eye_in_hand)
- Proprioception (gripper qpos, EE position/rotation relative to base)
- Final success signal (`info["success"]`)
- MuJoCo state access (`env.sim.get_state()` / `env.sim.set_state()`)

### What we need to build

1. **Task decomposition prompt** — Claude prompt that parses composite instructions into subtask lists
2. **Subtask dispatch** — Mechanism to override `obs["annotation.human.task_description"]` per-subtask
3. **Progress monitor** — VLM-based subtask completion detector (periodic visual queries)
4. **Checkpoint manager** — Save/restore sim state at subtask boundaries
5. **Orchestration loop** — The System 2 outer loop that ties it all together

---

## 6. Open Questions

1. **Does instruction rewriting help or hurt?** If GR00T was trained on full composite descriptions, will single-subtask instructions be OOD?
2. **How often to query VLM?** Every 50 steps (2.5s sim time) vs every 100 steps (5s). Tradeoff between latency and detection accuracy.
3. **Is sim state restore fair for benchmarking?** The official benchmark doesn't allow it. For a fair comparison, System 2 should be evaluated both with and without checkpointing.
4. **Can we extract subtask success from env internals?** Reading `self.water_was_turned_on` etc. is fragile and task-specific. But for a targeted benchmark on specific tasks, it gives perfect ground truth for subtask boundary detection evaluation.
5. **Horizon budget allocation:** How to divide the total horizon (e.g., 2000 steps) across subtasks? Even split? Proportional to complexity? Dynamic based on progress?

---

## 7. Key Source Files Reference

| File | Purpose |
|:---|:---|
| `third_party/robocasa/robocasa/environments/kitchen/kitchen.py` | Base Kitchen class, `get_ep_meta()` (L1167), `_check_success()` (L1542), `_post_action()` (L1476) |
| `third_party/robocasa/robocasa/wrappers/gym_wrapper.py` | `step()` (L313), `get_observation()` (L268), language injection (L264,284) |
| `third_party/robocasa/robocasa/utils/dataset_registry.py` | ATOMIC_TASK_DATASETS (L8), COMPOSITE_TASK_DATASETS (L558), pretraining splits (L2130) |
| `third_party/robocasa/docs/composite_tasks/task_attributes.json` | All 365 tasks with `num_subtasks` metadata |
| `third_party/robocasa/robocasa/environments/kitchen/composite/` | All 300 composite task implementations |
| `third_party/robosuite/robosuite/environments/base.py:532` | `_post_action()` — proves `info = {}` (empty) from base |
| `cap/env/robocasa/env.py` | Our RoboCasaEnv wrapper |
| `cap/policy/inference.py` | Current System 1 eval pipeline |
| `cap/saved_scripts/robocasa/policy_eval/gr00t/` | RoboCasa-365 GR00T eval scripts |
