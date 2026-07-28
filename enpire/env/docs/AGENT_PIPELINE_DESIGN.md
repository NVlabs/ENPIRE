# Agent Pipeline Design

Extensible LLM agent system for code-generate → execute → evaluate loops with unified logging.

## Entry Points

| Script | Purpose |
|--------|---------|
| `run_agent.py` | Agent loop: LLM generates code, executes it, observes, re-generates |
| `run_script.py` | Static script execution (unchanged, now imports from shared modules) |

### `run_agent.py` usage (Hydra)

Configuration uses Hydra structured configs. The dataclass hierarchy in
`cap/agent/agent_config.py` is the single schema — YAML configs are validated
against it, typos are caught at load time.

```bash
# Run an experiment (everything in one YAML)
PYTHONHASHSEED=42 uv run python run_agent.py experiment=pick_place_sink_to_counter

# Override fields from CLI (dot notation)
uv run python run_agent.py experiment=pick_place_sink_to_counter \
  env.seed=99 max_iterations=10

# Dry-run test
uv run python run_agent.py experiment=dry_run

# Show resolved config without running
uv run python run_agent.py experiment=pick_place_sink_to_counter --cfg job

# Show available experiments
uv run python run_agent.py --help
```

**Config structure:**
```
experiments/
  config.yaml                     # base defaults + env var fallbacks
  experiment/                     # experiment configs (select with experiment=NAME)
    pick_place_sink_to_counter.yaml
    pick_place_v1.yaml
    microwave_v1.yaml
    dry_run.yaml
```

**Schema** (`cap/agent/agent_config.py`):
- `AgentConfig` — top-level: name, task, max_iterations, steps
- `EnvConfig` — environment: name, viewer, layout_id, style_id, seed
- `LLMConfig` — LLM: backend, model. For `backend: nvidia`, `cap.agent.llm.NvidiaLLM`
  sends direct HTTPS `POST` requests to NVIDIA's `/v1/chat/completions`
  endpoint, rotates configured `NVIDIA_API_KEY_1..N` keys on HTTP 429, and
  automatically omits explicit `temperature` for Bedrock Claude models because
  the gateway rejects that field.
- `RuntimeConfig` — infrastructure: agent_name, dry_run, log_dir, no_cameras, quiet, saved_scripts_dir, host/port pairs (curobo, pyroki, mppi, anygrasp, sam3, bundlesdf, viser, bridge, voice)
- `PolicyConfig` — policy backend: backend, model, endpoint (for `use_policy_output`)
- `SkillLibraryConfig` — episodic skill library options
- `CodeGeneratorConfig` — prompts, context, retry behaviour
- `ReflectionConfig` — strategy, cameras, VLM backend
- `ExecutionConfig` — subprocess seeds, timeout
- `RecordingConfig`, `WandbConfig`, `PromptsConfig`

Env var fallbacks (in `experiments/config.yaml`) provide backward compatibility:
`CAP_AGENT_NAME`, `ROBOCASA_LAYOUT_ID`, `ROBOCASA_STYLE_ID` still work when
not set in the experiment YAML. Service ports live in `experiments/infra/ports.yaml`.
Experiment configs override both env vars and port defaults.

## Architecture

### Pipeline

```
AgentContext (shared state)
    │
    └── AgentPipeline
          ├── ObserverStep              reads get_robot_state(), get_task_info()
          ├── CodeGeneratorStep         LLM generates Python code
          ├── CodeReviewerStep          (optional) second LLM reviews code
          ├── SubprocessExecutorStep    spawn run_script.py N times (different seeds)
          └── SelfReflectionStep        (optional) LLM analyzes failure, gives feedback
```

**Execution:** `run_agent.py` uses `SubprocessExecutorStep` unconditionally — it spawns
`run_script.py` N times with different seeds, each subprocess creates its own env,
executes code, and saves `result.json`. Results are aggregated into `eval_summary.json`
(success rate, average score). The reward evaluation is folded into subprocess execution
(no separate `RewardEvaluatorStep` in the pipeline; the `_FOLDED` set in
`agent_pipeline.py` marks `reward_evaluator` as folded). Legacy `ExecutorStep` /
`RewardEvaluatorStep` classes still exist but are not wired by `AgentPipeline.from_config`.

Each step reads/writes `AgentContext`. `AgentPipeline` iterates until `ctx.should_stop` or `max_iterations`.

### Retry Reset Behavior

`ObserverStep` resets the environment between iterations (iteration > 0) so each attempt starts from the same initial state. The reset strategy uses a two-level fallback:

1. **`reset_to_initial`** (preferred) — Deterministic state restore. For sim, this restores a MuJoCo state snapshot captured after the first full reset. For real YAM, this returns the arm to its recorded home pose.
2. **`reset_env`** (fallback) — Used when `reset_to_initial` is not in the namespace.

### Reward evaluator (runs *before* reflection)

`cap/reward/` provides a pluggable task-success evaluator. For real-world tasks, the evaluator reads structured predicates from `result.json.details` (populated by the reward server) and reconstructs per-predicate pass/fail status for the reflection stage.

Example predicate output for a pick-and-place task:

```
**Ground-truth predicates** (seed 0, success=False):
  - obj_lifted:     LIKELY_FAIL (metric=0.012 m above table, threshold >0.05 m)
  - obj_in_target:  UNKNOWN (object not yet lifted)
**Classified failure:** (a) not picked up
```

**Configured via Hydra group `reward=`:**

```yaml
# experiments/reward/oracle.yaml  (default)
reward:
  evaluator: "oracle"        # or "noop" / "none"
  task: null                 # defaults to cfg.env.name
  inject_into_per_seed_vlm: true
```

**Runs in `SubprocessExecutorStep` Phase 4.5** — after execution, before Phase A per-seed VLM. Persists to `iter_NNN/reward_diagnostics.json`. Its `as_markdown()` output is injected into:

1. **Phase A per-seed VLM prompt** (one call per seed) — "Use the ground-truth predicates above as authoritative; the images should confirm, not contradict."
2. **Phase B cross-seed synthesis prompt** — rendered into the per-seed evidence block via `_build_per_seed_evidence()`, plus a `failure_histogram` line in the header (e.g. `(c) arm too close after placement × 18; (b) placed at wrong position × 4`).

**Adding a new task recipe:** add an entry to `TASK_SPECS` in `cap/reward/oracle_reward.py` with a list of `PredicateSpec` entries. See [`REWARD_MODULE.md`](REWARD_MODULE.md) for the full guide.

### Reflection — two-phase, cross-seed aware

With parallel execution (N seeds per iteration), reflection runs in **two phases**:

**Phase A — per-seed VLM calls (inside `SubprocessExecutorStep`).** After all seeds finish, one VLM call per seed analyzes that seed's `exec_NNN/vis/*_before_after.png`. Calls run in parallel via `ThreadPoolExecutor` (for NVIDIA, one worker per API key). The raw per-seed reflections are:

- appended to `ctx.evaluation.feedback` under `=== PER-SEED VISUAL ANALYSIS ===`
- stashed on `ctx.evaluation.details["per_seed_reflections_by_seed"]` (dict `{seed: text}`)
- **persisted to `iterations/iter_NNN/per_seed_reflections.md`** via `AgentRunSession.save_per_seed_reflections()`. This file survives the Phase B overwrite and is the canonical on-disk record of every seed's observation.

**Phase B — cross-seed synthesis (inside `SelfReflectionStep`).** One LLM call per iteration. When per-seed data is present on `ctx.evaluation.details["per_seed"]`, all three strategies (`Text`/`Vision`/`Composite`) route through `_reflect_cross_seed()` which:

1. Walks every `per_seed` entry, pulling `exec_NNN/exec.log` (stdout+error) via `session.load_run_artifacts(iteration, exec_id=N)` and the matching Phase A VLM reflection.
2. Assembles a per-seed evidence block (status, score, VLM reflection, stdout tail, error).
3. Renders `cap/prompt/system/cross_seed_reflection.md` and asks the LLM for a structured post-mortem: task, successful seeds, failed seeds **categorized** as (a) not picked up, (b) placed at wrong position, (c) arm too close after placement, (d) other, dominant pattern, actionable fixes.

When no per-seed data is present (inline executor, n_seeds=1 before per-seed gating), strategies fall back to their original single-seed behaviour (stdout/error from `exec_000` + optional fresh VLM scene query).

```
ObserverStep          → ctx.frames_before (captured once in parent env)
  ↓
CodeGeneratorStep     → generates code (failure_history = all prior iterations' feedback)
  ↓
SubprocessExecutorStep
   ├── dispatches N seeds → exec_NNN/{exec.log, result.json, vis/*_before_after.png}
   ├── Phase A: N parallel VLM calls → reflections_by_seed
   ├── writes iter_NNN/per_seed_reflections.md                          [new]
   └── ctx.evaluation.{feedback, details.per_seed, details.per_seed_reflections_by_seed}
  ↓
SelfReflectionStep    → _reflect_cross_seed() if details.per_seed else single-seed fallback
                          → reads exec_NNN/exec.log for every seed
                          → reads details.per_seed_reflections_by_seed
                          → renders cross_seed_reflection.md
                          → one LLM call → iter_NNN/reflection.md
                          → overwrites ctx.evaluation.feedback with the synthesis
                          → feedback flows to next iteration via failure_history
```

**Activation:** `cfg.reflection.strategy` chooses the strategy; cross-seed path engages automatically whenever `ctx.evaluation.details["per_seed"]` is populated (i.e. any `SubprocessExecutorStep` run).

**Artifacts:**
- `iterations/iter_NNN/per_seed_reflections.md` — raw Phase A (every seed, never overwritten)
- `iterations/iter_NNN/reflection.md` — Phase B synthesis (categorized list)
- `iterations/iter_NNN/exec_NNN/vis/*_before_after.png` — per-seed comparison images
- Legacy `vision_scene.md` / `visual_diff.md` are written only on the single-seed fallback path.

**Implementation:** `cap/agent/reflection.py` — `_build_per_seed_evidence()`, `_render_cross_seed_prompt()`, `_reflect_cross_seed()`. `cap/agent/agent_step.py` — `SubprocessExecutorStep` Phase 5/6 + `SelfReflectionStep`.

### Key files

| File | Purpose |
|------|---------|
| `cap/agent/agent_context.py` | `AgentContext`, `ExecutionMemory`, `ToolCallRecord`, `AgentEvaluation` |
| `cap/prompt/loader.py` | `PromptMemory` — index-based markdown prompt manager |
| `cap/agent/agent_step.py` | `AgentStep` ABC + all built-in steps (incl. `SubprocessExecutorStep`) |
| `cap/agent/agent_config.py` | Hydra structured configs: `AgentConfig`, `EnvConfig`, `LLMConfig`, `RuntimeConfig`, etc. |
| `cap/agent/agent_pipeline.py` | `AgentPipeline` iteration loop |
| `cap/agent/agent_session.py` | `AgentRunSession` — unified log folder + conversation logging |
| `cap/agent/wandb_logger.py` | `WandbLogger` — W&B metrics (success rate, step timing, per-iteration profiling) |
| `cap/agent/reflection.py` | Reflection strategies: Text, Vision, Composite + `_compute_visual_diff()` |
| `cap/agent/tools/_artifact_log.py` | `log_detection()`, `log_mask()`, `log_image()` — visual artifact saving |
| `cap/agent/tool_handle.py` | `ToolHandle` + `ToolRunner` — background execution + stop |
| `cap/agent/recorder.py` | `ScriptRecorder` — camera video capture |
| `cap/agent/tools/direct.py` | `make_direct_callables()`, `make_cancel_callables()` |
| `cap/agent/llm/bridge_llm.py` | `BridgeLLMBackend` — adapts bridge providers |
| `cap/env/yam_mujoco.py` | `YamMuJoCoEnv` (re-exports SimBackend) |
| `cap/env/yam_warp.py` | `YamWarpEnv` (re-exports WarpSimBackend, lazy) |
| `cap/env/adapters/sim.py` | `SimArmAdapter`, `SimCameraAdapter` |

## AgentContext

The central data structure flowing through all steps:

```python
@dataclass
class AgentContext:
    task: str
    max_iterations: int = 5
    iteration: int = 0

    # --- per-iteration pipeline state (reset each iteration) ---
    robot_state: Any = None            # RobotState from ObserverStep
    task_info: dict | None = None      # from get_task_info()
    thoughts: str | None = None        # LLM reasoning before code
    code: str | None = None            # generated by CodeGeneratorStep
    review_feedback: str | None = None # from CodeReviewerStep
    execution_result: Any = None       # ExecutionResult from ExecutorStep
    evaluation: AgentEvaluation | None = None
    frames_before: dict[str, Any] | None = None  # pre-execution camera frames
    frames_after: dict[str, Any] | None = None   # post-execution camera frames

    # --- accumulated across iterations ---
    history: list[IterationRecord]
    messages: list[dict]               # conversation log
    memory: ExecutionMemory            # all tool calls across iterations

    # --- control flow ---
    should_stop: bool = False
    stop_reason: str = ""

    # --- injected dependencies (set by run_agent.py before pipeline.run) ---
    namespace: dict                    # injected tool callables
    tool_schemas: list[dict]
    env_spec: dict | None = None       # from cap.prompt.env_spec.load_env_spec()
    config: Any = None                 # AgentConfig from YAML
    session: AgentRunSession
    prompt_memory: PromptMemory | None # index-based markdown prompt manager

    # --- paused execution support (wait_for_agent) ---
    paused_namespace: dict | None = None
```

## Prompt Memory

All prompts live as markdown files on disk in `cap/prompt/`, organized into semantic subfolders:

```
cap/prompt/
├── system/        # system prompts (identity, workflow, rules, reflection, review, retry, etc.)
├── tools/         # one markdown per tool (detect_object, bundlesdf_track, yam_coordinate_system)
├── embodiment/    # per-robot specs (yam.md, etc.)
├── task/          # task-specific strategies (pick_place.md, etc.)
├── heuristics/    # agent-learned heuristics (cross-run persistent, initially empty)
└── loader.py      # PromptMemory class
```

`PromptMemory` (`cap/prompt/loader.py`) provides index-based access:

```python
pm = PromptMemory()                    # defaults to cap/prompt/
pm.scan()                              # build keyword → file path index
pm.load("system", "code_review", task="...", code="...")  # load + substitute
pm.load_section("system", "vision_reflection", "Scene Query", task="...")
pm.resolve(["yam", "pick"])            # keyword search → list[Path]
pm.inject(paths, mode="full")          # render full content or index-only
pm.load_embodiment("yam")              # → {"tool_docs": ..., "env_notes": ...}
```

## Execution Memory

Every tool call from every thread is captured via `profiler.set_tool_event_hooks()`:

```python
memory.summarize_for_llm()  # compact summary fed to next code generation
memory.to_json()             # saved to tool_calls.json
```

## Concurrent Execution + Stop

Generated code can use background tasks with controlled stop:

```python
# Run any tool in background
handle = run_in_background(freespace_move, "right", pos, quat)

# Stop any running tool — injects ToolStopped exception via PyThreadState_SetAsyncExc
handle.stop()

# Stop everything
stop_all_tools()

# Per-arm motion cancel (cooperative, faster than stop())
cancel_motion("right")   # sets threading.Event checked in _ik_servo's inner loop
```

`CapServer.cancel_motion(side)` is also exposed as an RPC method.

## Log Folder Structure (Layout v2)

```
logs/
  {timestamp}_{agent_type}_{task}/      # per-run artifact directory
    config.yaml                          # experiment config (frozen at start)
    metadata.json                        # run summary (layout_version: 2)

    iterations/                          # per-iteration artifacts
      iter_000/
        code.py                          # generated code
        thoughts.md                      # LLM reasoning
        reflection.md                    # Phase B cross-seed synthesis (if --reflect)
        per_seed_reflections.md          # Phase A raw per-seed VLM output (N seeds)
        reward_diagnostics.json          # oracle predicate outcomes per seed
        review.md                        # code review (if --review)
        vision_scene.md                  # VLM scene description (single-seed fallback only)
        visual_diff.md                   # before/after VLM diff (single-seed fallback only)
        eval_summary.json                # aggregated results across N seeds
        eval.json                        # single eval (inline executor only)
        exec.log                         # exec log (inline executor only)
        exec_000/                        # seed=0 execution (subprocess mode)
          result.json                    # success, reward, feedback
          exec.log                       # stdout/stderr
          profiling.txt                  # tool call timing
          video/                         # camera recordings
          frames_after/                  # after-execution camera PNGs
        exec_001/                        # seed=1 execution
          ...
      iter_001/
        ...

    results/                             # experiment-level aggregation
      tool_calls.json                    # all tool invocations (all iterations)
      reward_trace.json                  # per-tool reward samples

    conversations/                       # LLM I/O (all iterations)
      iter_000_generator_HHMMSS.md
      iter_000_reflection_HHMMSS.md
      ...

    vis/                                 # detection/segmentation/visual diff images
    vlm/                                 # VLM query artifacts
```

**Subprocess execution** (direct mode, `SubprocessExecutorStep`): Each `exec_MMM/` directory contains the complete output of one `run_script.py` invocation with a specific seed. `eval_summary.json` aggregates results across all seeds.

**Inline execution** (CapServer mode, `ExecutorStep`): `exec.log` and `eval.json` are written directly into `iter_NNN/` without the `exec_MMM/` subdirectory.

## Live Dashboard (`run_script.py`)

When `run_script.py` runs on a TTY, it renders a `rich.live.Live` dashboard
(`cap/agent/live_dashboard.py`) that replaces the plain `[profile …]` lines:

- **Header** — script · env · seed · layout / style · running elapsed
- **In-flight** — `▶ #N tool_name(args) · live-ticking ms` (yellow when >1s)
- **Recent** — last 20 completed calls, color-coded (green ok, yellow slow, red error)
- **Footer** — call count · errors · total tool ms

The dashboard hooks the existing profiler via `set_tool_event_hooks()`; no
profiler changes are required.  Non-TTY runs (captured subprocess, CI logs)
fall back to plain prints automatically.

**Parallel mode with tmux:** when `execution.mode=parallel`, `n_seeds>1`,
and the parent agent is inside tmux (`$TMUX` set), `SubprocessExecutorStep`
launches each seed in its own tmux window (`exec_s{seed}_iter{NNN}`)
instead of capturing stdout, so all seeds render their dashboards live in
parallel.  The parent polls a `.tmux_done` sentinel per exec_dir to detect
completion.  Outside tmux, parallel mode keeps `capture_output=True` so
nothing is lost to interleaving.

## Weights & Biases Logging

Agent runs can log metrics to W&B for tracking success rates and profiling across experiments.

### Enabling

```bash
# Via CLI override
uv run python run_agent.py experiment=pick_place_sink_to_counter wandb.enabled=true

# Via experiment YAML (add to your experiment config)
# wandb:
#   enabled: true
#   project: cap-agent
#   entity: null
#   tags: [yam, pick_place]
```

### What gets logged

| Scope | Metrics |
|-------|---------|
| **Config** (wandb.config) | experiment name, task, env, LLM backend/model, pipeline steps, reflection strategy, station name, seed |
| **Per-iteration** (wandb.log) | `iter/total_ms`, `iter/score`, `iter/success`, `iter/code_length`, `iter/tool_calls`, `iter/has_error`, `step/<name>_ms` for each pipeline step |
| **Run summary** (wandb.summary) | `success`, `iterations`, `duration_s`, `final_score`, `stop_reason` |

### Key files

| File | Purpose |
|------|---------|
| `cap/agent/wandb_logger.py` | `WandbLogger` — thin wrapper (init/log/finish), graceful no-op when wandb not installed |
| `cap/agent/agent_config.py` | `WandbConfig` dataclass + YAML parsing |
| `cap/agent/agent_pipeline.py` | Per-iteration step timing collection and `wandb.log()` |
| `run_agent.py` | Init wandb before pipeline, log summary + finish in finally block |

### Dependencies

`wandb` is in the `cap` optional deps group. Install with `uv sync --extra cap` or `uv pip install wandb`.

## Custom Pipeline Example

```python
from cap.agent.agent_pipeline import AgentPipeline
from cap.agent.agent_step import AgentStep, ObserverStep, CodeGeneratorStep, ExecutorStep
from cap.agent.agent_context import AgentContext, AgentEvaluation
from cap.agent.llm.cloud import CloudLLM

class MyEvaluator(AgentStep):
    name = "custom_eval"
    def run(self, ctx):
        task_info = ctx.namespace["get_task_info"]()
        success = task_info.get("obj_pos", [0,0,0])[2] > 0.15
        ctx.evaluation = AgentEvaluation(success=success, score=float(success), feedback="", method="custom")
        if success: ctx.should_stop = True
        return ctx

llm = CloudLLM()
pipeline = AgentPipeline([ObserverStep(), CodeGeneratorStep(llm), ExecutorStep(), MyEvaluator()])
ctx = AgentContext(task="...", namespace=ns, tool_schemas=schemas, session=session)
ctx = pipeline.run(ctx)
```

## Env Layer

```
cap/env/
  base.py            — (compat shim, re-exports base/)
  base/
    protocols.py     — EnvProtocol, EefControlProtocol, SceneProtocol, TaskProtocol
    profile.py       — RobotProfile, ArmProfile
    skill_library.py — shared skill library (VLM backends, grasp helpers)
  profile.py         — (compat shim, re-exports base/profile)
  setup.py           — create_runtime() factory for direct mode
  yam.py             — YAM pinocchio IK
  yam_mujoco.py      — YAM MuJoCo (re-exports SimBackend)
  yam_warp.py        — YAM Warp (lazy re-export)
  adapters/
    sim.py           — SimArmAdapter, SimCameraAdapter
```

`create_env(env_name)` in `cap/env/__init__.py` is the single factory.
`create_runtime(env_name)` in `cap/env/setup.py` returns `(env, namespace)` for direct mode (used by `run_agent.py` and `run_script.py`).
Real hardware clients remain in `cap_server.py` (future: `YamHardwareEnv`).
