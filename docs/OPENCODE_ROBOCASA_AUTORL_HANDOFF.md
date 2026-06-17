# OpenCode RoboCasa Auto-Improve Handoff

This document records the work done during the conversation around running the OpenCode-based RoboCasa auto-improve loop in this repository. It is intended as a detailed handoff for future debugging, continuation, or cleanup.

The current working directory for this work was:

```bash
/mnt/amlfs-02/shared/wenli_vla_ft/forge-code
```

The original reference repository was:

```bash
/mnt/amlfs-02/shared/wenli_vla_ft/forge
```

The main user-facing goal is to replace or modernize the previous local-agent workflow with an OpenCode-driven workflow that can iteratively discover, codify, evaluate, and improve RoboCasa skills for `PickPlaceSinkToCounter`, while using the already-launched shared external services instead of trying to start services from the preinstalled `uv` environment.

## High-Level Goal

The desired workflow is:

1. Start the GR00T policy pool.
2. Start the remote OSMO service pool with SAM3, AnyGrasp, cuRobo, and pyroki services.
3. Run an OpenCode auto-improve loop for RoboCasa `PickPlaceSinkToCounter`.
4. Have OpenCode explore the task through MCP robot tools, use vision rather than oracle state, codify a `code.py`, and evaluate the code over a fixed tiered seed ladder.
5. Continue improving until the generated code reaches a target success rate of at least `0.8` on the canonical 40-seed benchmark.

The old launch sequence was:

```bash
cd /mnt/amlfs-02/shared/wenli_vla_ft/forge/

bash tmux/launch_grootpool.sh 8 0,1,2,3,4,5,6,7

bash tmux/remote_serving/launch_osmo.sh \
  --sam3-n 16 \
  --curobo-n 16 \
  --anygrasp-n 16 \
  --pyroki-n 16 \
  --num-gpus 8 || true
```

The desired command shape in this codebase is now cleaner for GR00T:

```bash
bash tmux/launch_grootpool.sh 8
```

The GPU id list should no longer need to be passed as `0,1,2,3,4,5,6,7`. The launcher should infer the standard contiguous GPU set from the GPU count.

The OSMO remote serving command remains conceptually the same:

```bash
bash tmux/remote_serving/launch_osmo.sh \
  --sam3-n 16 \
  --curobo-n 16 \
  --anygrasp-n 16 \
  --pyroki-n 16 \
  --num-gpus 8 || true
```

The OpenCode run command is:

```bash
bash scripts/run_opencode_pnp_sink_to_counter.sh
```

There is also a validation-only mode:

```bash
bash scripts/run_opencode_pnp_sink_to_counter.sh --check-only
```

That check has previously verified that SAM3, AnyGrasp, and cuRobo slots were reachable for slots `0..15`.

## Main Files Touched

The work touched several layers:

- `tmux/launch_grootpool.sh`
- `tmux/remote_serving/launch_osmo.sh`
- `scripts/run_opencode_pnp_sink_to_counter.sh`
- `.opencode/opencode.json`
- `.opencode/instructions.md`
- `.opencode/skills/auto-improve.md`
- `.opencode/skills/explore-task.md`
- `.opencode/skills/codify-trace.md`
- `cap/mcp/server.py`
- `cap/env/robocasa/skills.py`
- `run_script.py`
- `cap/saved_scripts/robocasa_skill_library/groot_only.py`
- `cap/saved_scripts/robocasa_skill_library/descend_and_grasp.py`
- `cap/saved_scripts/robocasa_skill_library/incremental_grasp.py`
- `third_party/opencode/packages/opencode/src/cli/cmd/tui/plugin/internal.ts`
- `third_party/opencode/packages/opencode/src/cli/cmd/tui/routes/session/index.tsx`
- `third_party/opencode/packages/opencode/src/cli/cmd/tui/feature-plugins/sidebar/eval-status.tsx`

There were also large binary or artifact files copied into this workspace earlier in the conversation, including AnyGrasp/MinkowskiEngine related `.so` artifacts and model/license archives. Those are not described as design changes here, but they are part of the dirty worktree and should not be accidentally reverted without checking why they were copied.

## Launch And Environment Work

The user wanted to know whether the old launch sequence from `/forge` could be run in this codebase, and then asked to copy the needed pieces. The important compatibility requirements were:

- Use the same external service pool layout as before.
- Do not rely on the current preinstalled `uv` environment for services that were known not to run there.
- Preserve the ability to use 16 service slots for SAM3, AnyGrasp, cuRobo, and pyroki.
- Preserve 8 render GPUs for parallel evaluation.
- Keep GR00T pool launch simple and avoid manually passing explicit GPU id lists.

The OpenCode run script now configures the service pool assumptions explicitly. The relevant values are:

```text
SERVICE_SLOT=0
SERVICE_SLOTS=16
RENDER_GPUS=8
EVAL_MAX_WORKERS=16
SEEDS_PER_GPU=2
CAP_PLANNER_BACKEND=curobo
SAM3=127.0.0.1:9500
AnyGrasp=127.0.0.1:9300
cuRobo=127.0.0.1:9400
pyroki=127.0.0.1:9600
CAP_MCP_MAX_SESSIONS=8
```

For parallel eval, service slot `i` maps ports as base plus slot:

```text
SAM3     9500 + i
AnyGrasp 9300 + i
cuRobo   9400 + i
pyroki   9600 + i
```

This matters because individual eval subprocesses use different service slots. Explorer sessions also use distinct service slots.

## OpenCode Auto-Improve Prompt Design

The prompt in `scripts/run_opencode_pnp_sink_to_counter.sh` was expanded from a simple "run this task" prompt into a structured auto-improve protocol.

The intended loop is:

1. Run `mcp__robot__health_check`.
2. Create the main environment:

   ```text
   mcp__robot__env_create(
     task="PickPlaceSinkToCounter",
     seed=0,
     session_id="main",
     service_slot=0
   )
   ```

3. Get the natural-language task description from the environment:

   ```text
   mcp__robot__env_get_task_description(session_id="main")
   ```

4. Inspect the pre-seeded RoboCasa skill library:

   ```text
   mcp__robot__skill_list(session_id="main")
   mcp__robot__skill_read(...)
   ```

5. Spawn an exploration team before codifying.
6. Integrate exploration results into one main-session trace.
7. Spawn a codifier subagent to write standalone `code.py`.
8. Evaluate through a tiered ladder:

   ```text
   tier 1: n_seeds=5
   tier 2: n_seeds=10
   tier 3: n_seeds=20
   tier 4: n_seeds=40
   ```

9. Escalate only if the current tier has `success_rate >= 0.8`.
10. If any tier fails, patch and restart from tier 1.
11. Stop only when tier 4 reaches at least `0.8`.

The prompt also reminds OpenCode to pass the required parallel eval arguments on every `evaluate_code` call:

```text
parallel=true
service_slots=16
render_gpus=8
max_workers=16
seeds_per_gpu=2
```

It also passes runtime overrides so child `run_script.py` processes use the same service pool:

```text
overrides=[
  "runtime.sam3_host=127.0.0.1",
  "runtime.sam3_port=9500",
  "runtime.anygrasp_host=127.0.0.1",
  "runtime.anygrasp_port=9300",
  "runtime.curobo_host=127.0.0.1",
  "runtime.curobo_port=9400"
]
```

The MCP server also auto-forwards active session service ports, including pyroki, when a live session is active.

## Agent Team And Multi-Session Design

The user wanted OpenCode to launch an agent team for exploration before codifying. The concern was whether the current infrastructure supported independent RoboCasa sessions for each agent.

The design implemented in `cap/mcp/server.py` added named sessions:

```python
@dataclasses.dataclass
class _RobotSession:
    session_id: str
    env: Any
    namespace: dict[str, Any]
    trace: list[dict[str, Any]]
    env_name: str
    seed: int
    service_slot: int
    service_ports: dict[str, Any]
```

The server now tracks:

```python
self.sessions: dict[str, _RobotSession] = {}
self.active_session_id = "main"
self.max_sessions = int(os.environ.get("CAP_MCP_MAX_SESSIONS", "8"))
```

Most robot tools accept an optional `session_id`. The server activates the requested session before running the tool. The main usage pattern is:

```text
main:
  session_id="main"
  service_slot=0

explore_perception:
  session_id="explore_perception"
  service_slot=1

explore_grasp:
  session_id="explore_grasp"
  service_slot=2

explore_place:
  session_id="explore_place"
  service_slot=3
```

The important constraint is: do not reuse a service slot across live sessions. Service ports are assigned as base plus slot, so two sessions sharing a service slot would contend for the same SAM3, AnyGrasp, cuRobo, and pyroki endpoints.

The prompt and skills now instruct explorer agents to destroy their environment when done:

```text
mcp__robot__env_destroy(session_id="<explorer_session_id>")
```

This was added after observing that idle explorer sessions can leave RoboCasa renderer/GPU resources alive inside the MCP server, causing high memory and thread counts.

### What "Serialize Individual Tool Calling" Means

There are two different kinds of parallelism in this setup.

The first is evaluation parallelism. `mcp__robot__evaluate_code` fans out over seeds by spawning `run_script.py` subprocesses through a `ThreadPoolExecutor`. That is the heavy parallel path. It uses service slots, render GPUs, and per-GPU concurrency limits.

The second is exploration parallelism. OpenCode subagents can be launched in parallel, and each explorer is instructed to create a separate RoboCasa session with a separate `session_id` and `service_slot`. This isolates simulator state and service ports.

However, individual MCP tool calls into one MCP server are still requests to one process, and tool calls against one robot session should be treated as effectively serialized at the behavioral level. The design does not depend on two agents simultaneously mutating the same RoboCasa environment. If parallel explorers are used, each must have its own named session. This avoids having multiple agents racing on the same simulator state.

## Oracle Information Removal

The user noticed OpenCode was getting oracle task information through `get_task_info`, including object positions. The requested behavior was to disable oracle task information and force the policy/code to infer object and target positions through vision tools.

The implementation direction was:

- Disable `get_task_info()` in evaluated scripts.
- Disable `get_oracle_targets()` in evaluated scripts.
- Disable script-level oracle `detect_object(...)` in evaluated scripts.
- Keep private oracle access only for scoring/evaluation internals.
- Use the language instruction the same way `groot_only` policy does.

The relevant changes are in:

- `run_script.py`
- `cap/mcp/server.py`
- `cap/saved_scripts/robocasa_skill_library/groot_only.py`
- `cap/saved_scripts/robocasa_skill_library/descend_and_grasp.py`
- `cap/saved_scripts/robocasa_skill_library/incremental_grasp.py`

The key environment variable is:

```text
CAP_DISABLE_TASK_INFO_IN_SCRIPT=1
```

`cap/mcp/server.py` sets this for eval subprocesses. That means generated `code.py` cannot use oracle helpers and must rely on:

- `get_task_description()`
- camera images
- `detect_objects_oneshot(...)`
- SAM3 plus depth
- VLM reasoning
- motion/planner feedback

The prompt explicitly says:

```text
Treat the natural-language instruction as the only task semantic source;
object and target XYZ must come from vision.
```

It also reminds OpenCode:

```text
mcp__robot__detect_object uses real vision (SAM3 + depth), not oracle ground truth.
Score < 1.0 is normal, occlusion is real, and camera switching may be needed.
```

## Skill Library And Placement Orientation Search

The user asked about a skill in `cap/saved_scripts/yam_autorl` that places an object based on XYZ by proposing several orientations and using cuRobo to find the most feasible one.

The inspected pattern was from:

```text
cap/saved_scripts/yam_autorl/pin_insert/move_to_target_xyz.py
```

The useful design idea is:

1. Start with a desired target XYZ.
2. Generate multiple RPY candidates for the same XYZ.
3. Convert candidate RPY values to grasp/motion candidates.
4. Batch-rank the candidates using cuRobo feasibility.
5. Execute the best feasible pose.

The conclusion was that RoboCasa skill library code can use the same idea, but should not directly import the nested YAM file because it has YAM-specific poses, fingertip offsets, and package-relative assumptions.

The prompt now tells OpenCode to adapt the idea if needed using RoboCasa-compatible primitives:

- `cap.env.base.skill_library.GraspCandidate`
- `select_best_grasp(..., batch_top_k=16)`
- `display_rpy_to_quat`
- `freespace_move`

Relevant pre-seeded RoboCasa skills include:

- `vertical_grasp`
- `descend_and_grasp`
- `hover_above`
- `post_grasp_lift`
- `robust_grasp`
- `lift`
- `vertical_place`
- `place_with_orientation`
- `hover_orientation_search`

The intended coding style is to reuse these skills first:

```python
from skill_library.<name> import <fn>_v1
```

New skills should be written only for genuinely new mechanisms.

## Parallel Evaluation Design

The old non-OpenCode agent, `run_agent.py`, had a parallel evaluation design. The OpenCode MCP flow was updated to preserve the same important behavior.

`mcp__robot__evaluate_code` now:

- Uses the canonical fixed 40-seed list from:

  ```text
  cap/assets/robocasa_40_seeds.txt
  ```

- Treats larger tiers as strict prefixes of the same fixed list. For example, `n_seeds=10` includes the first 10 seeds and is a strict superset of the first 5.
- Fails if `n_seeds` exceeds the preset pool size.
- Creates per-iteration directories under the run log directory:

  ```text
  logs/<run_id>_opencode_pnp_sink_to_counter/iterations/iter_NNN/
  ```

- Writes:

  ```text
  code.py
  eval_status.json
  eval_summary.json
  exec_000/
  exec_001/
  ...
  ```

- Runs child processes with:

  ```bash
  python -u run_script.py \
    script_file=<code.py> \
    env.name=<env_name> \
    env.seed=<seed> \
    env.layout_id=<layout_id> \
    env.style_id=<style_id> \
    script_output_dir=<exec_dir>
  ```

- Forwards service and GPU environment variables per seed.
- Applies process group cleanup on timeout.
- Writes real-time eval status as each seed moves through:

  ```text
  pending
  queued
  running
  success
  failed
  timeout
  error
  ```

The current eval status file is intended to be:

```text
logs/<run_id>_opencode_pnp_sink_to_counter/eval_status.json
```

This top-level status file is what the OpenCode TUI sidebar watches.

## Real-Time Eval Status In OpenCode TUI

The user wanted a real-time evaluation status panel in the OpenCode sidebar.

The implementation added a new internal OpenCode TUI plugin:

```text
third_party/opencode/packages/opencode/src/cli/cmd/tui/feature-plugins/sidebar/eval-status.tsx
```

It is registered from:

```text
third_party/opencode/packages/opencode/src/cli/cmd/tui/plugin/internal.ts
```

The plugin reads:

```text
${CAP_RUN_LOG_DIR}/eval_status.json
```

It polls once per second and renders:

- phase
- completed/total seeds
- success percentage
- active running/queued counts
- elapsed time
- successes
- failures
- iteration number
- per-seed status with service slot and render GPU

Before the first `evaluate_code` call, `eval_status.json` does not exist. The first version displayed raw `ENOENT`, which was ugly. It was then patched so future TUI starts show:

```text
waiting for first evaluation
```

During the live `oc-5:0.0` run, the sidebar was confirmed visible. It showed:

```text
MCP
LSP
Todo
Eval waiting
```

At that moment, this was correct because the run was still in the exploration stage and no `evaluate_code` call had happened. The current live run directory was:

```text
logs/20260503T043757_opencode_pnp_sink_to_counter
```

That directory had only the seeded `skill_library` and no `eval_status.json` yet, so `Eval waiting` was the expected state.

Important operational note: OpenCode TUI source changes do not hot-reload into an already-running TUI. If `eval-status.tsx` or `internal.ts` changes, the OpenCode TUI must be restarted to load the new plugin behavior.

## OpenCode TUI Freeze Investigation

The user noticed that the OpenCode TUI can get stuck after a few iterations.

A stale live pane was investigated in `tmux oxc-4:0.0`. The important facts were:

- The TUI still visually showed a running `mcp__robot__evaluate_code` call.
- The status file showed eval had completed.
- OpenCode logs showed the session had gone idle.
- The pane did not repaint after `Ctrl-L`.
- Bun was using roughly 100 percent CPU and around 25 GB RSS.
- The MCP Python process was using roughly 190 percent CPU and around 18 GB RSS with hundreds of threads.

The completed eval status for that stale run was:

```json
{
  "phase": "completed",
  "message": "0/5 seeds succeeded (0%)",
  "counts": {
    "pending": 0,
    "queued": 0,
    "running": 0,
    "completed": 5,
    "successes": 0,
    "failures": 5
  }
}
```

The run log path was:

```text
logs/20260503T025056_opencode_pnp_sink_to_counter
```

The conclusion was that the backend had completed but the TUI was stale or spinning on rendering. The concrete renderer issue found was in:

```text
third_party/opencode/packages/opencode/src/cli/cmd/tui/routes/session/index.tsx
```

The generic tool input formatter was rendering all primitive tool inputs inline:

```text
key=value
```

For `mcp__robot__evaluate_code`, this included the entire generated `code` string. After multiple iterations, the session view contained repeated 10-20 KB code blobs inside tool call titles, causing heavy rendering cost and likely contributing to the stuck TUI.

The formatter was patched to summarize large or multiline primitive values:

```text
code=<18000 chars>
```

instead of rendering the full source inline.

The relevant constants are:

```typescript
const MAX_INLINE_INPUT_VALUE = 120
const MAX_INLINE_INPUT_TOTAL = 600
```

This fix also requires restarting OpenCode because the old renderer stays loaded in existing TUI processes.

## Current Live Tmux State Observed

The current OpenCode run was in:

```text
tmux session: oc-5
pane: oc-5:0.0
```

The pane title was:

```text
OC | RoboCasa PickPlaceSinkToCounter auto-...
```

The live TUI was in exploration, with three explorer tasks:

- perception for lemon wedge and plate
- grasp strategy for lemon wedge
- placement onto plate

The sidebar was visible. It showed:

- `MCP`: robot connected
- `LSP`: waiting until files are read
- `Todo`: active checklist
- `Eval`: waiting

The process environment for the live OpenCode Bun process included:

```text
CAP_RUN_LOG_DIR=/mnt/amlfs-02/shared/wenli_vla_ft/forge-code/logs/20260503T043757_opencode_pnp_sink_to_counter
CAP_MCP_MAX_SESSIONS=8
PWD=/mnt/amlfs-02/shared/wenli_vla_ft/forge-code
```

At the time of inspection, the eval status file did not exist yet:

```text
logs/20260503T043757_opencode_pnp_sink_to_counter/eval_status.json
```

That was expected because no tiered eval had started.

## MCP Server Design Assessment

The user asked whether `cap/mcp/server.py` was an unstructured mess or a systematic design.

The practical assessment is:

- It is a stateful adapter between OpenCode/MCP and RoboCasa/CAP.
- It has clear major sections:
  - setup and tool schema registration
  - session helpers
  - dispatch
  - environment lifecycle
  - state/camera/perception tools
  - motion tools
  - health checks
  - evaluation
  - skill library
- It is large and has operational complexity, but it is not random. It is accumulating several responsibilities in one file.

The design pressure points are:

- It holds live RoboCasa envs in-process.
- It holds skill library state.
- It owns eval subprocess orchestration.
- It owns per-session service port routing.
- It writes status files.
- It has to protect against oracle leakage.

The better long-term design would split it into modules:

```text
cap/mcp/server.py              - MCP registration and dispatch only
cap/mcp/sessions.py            - RobotSession lifecycle and service slots
cap/mcp/evaluation.py          - evaluate_code and status writing
cap/mcp/tools/env.py           - env lifecycle/state/camera tools
cap/mcp/tools/perception.py    - detect_object and VLM tools
cap/mcp/tools/motion.py        - freespace_move, gripper, nudge
cap/mcp/tools/skills.py        - skill library tools
cap/mcp/health.py              - service checks
```

That refactor was discussed as a better design but not fully performed during this conversation. The immediate implementation focused on preserving behavior while adding multi-session support and eval status.

## Remote Serving And AnyGrasp

The user asked to read `tmux [osmo-remote-serving]` and determine why AnyGrasp was not up. The broader servicing design is that `launch_osmo.sh` starts multiple service types across ports.

The relevant services and bases are:

```text
SAM3     9500
AnyGrasp 9300
cuRobo   9400
pyroki   9600
```

The current check-only OpenCode launch previously confirmed multiple slots reachable. If AnyGrasp is missing for a slot, use:

```bash
tmux capture-pane -pt osmo-remote-serving:<window>.0 -S -200
```

and inspect the slot-specific window. The observed session has 64 windows, corresponding to the service pool.

## Permissions And Auto-Approval

There were several questions about why OpenCode asked for permission again, and why something was not auto-approved.

The important facts are:

- OpenCode permission rules are configured in `.opencode/opencode.json`.
- Some permissions are tool-name specific.
- MCP tools need explicit allow rules such as:

  ```text
  mcp__robot__env_create
  mcp__robot__env_reset
  mcp__robot__env_destroy
  mcp__robot__env_get_state
  mcp__robot__env_get_task_description
  mcp__robot__env_get_cameras
  mcp__robot__env_check_success
  mcp__robot__env_get_trace
  mcp__robot__health_check
  mcp__robot__detect_object
  mcp__robot__freespace_move
  mcp__robot__nudge
  mcp__robot__gripper
  mcp__robot__vlm_query
  mcp__robot__evaluate_code
  mcp__robot__skill_list
  mcp__robot__skill_read
  mcp__robot__skill_write
  ```

- `external_directory` reads/writes may still trigger prompts unless explicit allow patterns cover the path.
- `.opencode/skills/auto-improve.md` missing caused an earlier visible error. That file has since been created/updated.

The user also switched the model from Claude Opus 4.6 to Claude Opus 4.7 through the OpenCode config. The live TUI showed:

```text
Robot · Claude Opus 4.7 (Bedrock via NVIDIA)
```

## Important Validation Commands Already Used

These checks have passed at points during the conversation:

```bash
/mnt/amlfs-02/shared/wenli_vla_ft/python_envs/forge/bin/python \
  -m py_compile \
  cap/mcp/server.py \
  cap/env/robocasa/skills.py \
  run_script.py \
  cap/saved_scripts/robocasa_skill_library/groot_only.py
```

```bash
jq empty .opencode/opencode.json
```

```bash
bash -n scripts/run_opencode_pnp_sink_to_counter.sh
```

```bash
git diff --check
```

```bash
/root/.bun/bin/bun run --cwd third_party/opencode/packages/opencode typecheck
```

```bash
scripts/run_opencode_pnp_sink_to_counter.sh --check-only
```

The check-only mode confirmed service reachability for the configured service slots at the time it was run.

## Operational Runbook

### 1. Start service pool

From the repo root:

```bash
bash tmux/launch_grootpool.sh 8
```

Then start remote serving:

```bash
bash tmux/remote_serving/launch_osmo.sh \
  --sam3-n 16 \
  --curobo-n 16 \
  --anygrasp-n 16 \
  --pyroki-n 16 \
  --num-gpus 8 || true
```

### 2. Confirm services

```bash
bash scripts/run_opencode_pnp_sink_to_counter.sh --check-only
```

This should report SAM3, AnyGrasp, and cuRobo reachable across slots. If a service is missing, inspect the relevant tmux service window.

### 3. Run OpenCode auto-improve

```bash
bash scripts/run_opencode_pnp_sink_to_counter.sh
```

### 4. Watch tmux

List sessions:

```bash
tmux list-sessions
```

List panes:

```bash
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command} #{pane_pid} #{pane_title}'
```

Capture the OpenCode pane:

```bash
tmux capture-pane -pt oc-5:0.0 -S -120
```

The exact session number changes. Current pattern is usually `oc-*`.

### 5. Watch eval status

Find the current run directory from the OpenCode process environment:

```bash
tr '\0' '\n' < /proc/<opencode_pid>/environ | rg 'CAP_RUN_LOG_DIR'
```

Then inspect:

```bash
jq . "$CAP_RUN_LOG_DIR/eval_status.json"
```

During pre-eval exploration, this file may not exist yet.

### 6. If the TUI appears stuck

Check whether evaluation really is running:

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | rg 'opencode|bun|cap.mcp|run_script'
```

Check latest eval status:

```bash
jq '{phase,message,counts}' logs/<run_id>/eval_status.json
```

Check OpenCode logs:

```bash
tail -200 ~/.local/share/opencode/log/dev.log
```

If logs say the session is idle but the pane still shows a running tool, the stale TUI renderer issue may have recurred. Restarting the TUI is the practical recovery. The truncation patch should make this less likely in future runs.

## Current Known Issues

1. Existing running OpenCode processes do not hot-reload TUI changes.

   Restart OpenCode after changes to:

   ```text
   third_party/opencode/packages/opencode/src/cli/cmd/tui/*
   ```

2. Before first eval, `eval_status.json` does not exist.

   This is normal. The sidebar should show waiting. Older already-running TUI instances may still show raw `ENOENT`.

3. Explorer sessions can leak resources if not destroyed.

   The prompt now tells the primary agent to call `env_destroy` for explorer sessions. Future code could enforce this more strongly from the MCP server side.

4. `cap/mcp/server.py` is still too large.

   It has systematic sections, but the responsibilities should eventually be split.

5. OpenCode can consume a lot of memory on long sessions.

   The tool input truncation patch addresses one concrete cause, but long sessions with many subagents and tool results can still be heavy.

6. Generated policies must avoid oracle shortcuts.

   `get_task_info()`, `get_oracle_targets()`, and script-level oracle `detect_object(...)` are disabled in evaluated scripts. If a generated policy starts relying on hidden state again, treat that as a regression.

## Desired Future Improvements

### Enforce Session Cleanup

The prompt now asks for `env_destroy`, but the MCP server could enforce or assist cleanup by:

- adding session idle TTLs
- adding `env_list_sessions`
- adding `env_destroy_all_explorers`
- warning when live sessions exceed a small threshold before eval
- optionally destroying non-main sessions at the start of `evaluate_code`

### Make Eval Status More Complete

The sidebar currently reads `eval_status.json`. It could also show:

- current iteration number
- tier number
- last completed tier
- current code path
- last failure category summary
- time since last status update
- whether eval appears stale

### Split MCP Server Modules

The server should eventually be split into session management, tool handlers, evaluation orchestration, status writing, and health checks.

### Improve Prompt Enforcement

The prompt could more explicitly require:

- no tier escalation after failure
- destroy explorer sessions before eval
- report live session ids before eval
- do not spawn more than 3 explorers unless necessary
- use vision-detected XYZ only

### Add Regression Tests

Useful tests would include:

- `evaluate_code(n_seeds=0)` status smoke
- session create/destroy smoke
- duplicate service slot rejection
- TUI `input()` formatter truncates multiline code
- eval sidebar missing-file state displays as waiting
- oracle helper disabled under `CAP_DISABLE_TASK_INFO_IN_SCRIPT=1`

## Current Mental Model

The system now has three major loops:

1. OpenCode reasoning loop.

   This decides what to inspect, what subagents to launch, and when to codify/evaluate.

2. MCP interactive robot loop.

   This gives OpenCode access to RoboCasa state, camera images, VLM queries, SAM3 detections, motion primitives, and the skill library.

3. Evaluation loop.

   This runs generated `code.py` over fixed benchmark seeds using subprocesses and shared remote services.

The most important design rule is that exploration and evaluation should not share mutable simulator state accidentally:

- main session uses `session_id="main"`
- explorers use distinct `session_id` values
- each live session gets a distinct `service_slot`
- eval subprocesses get their own per-seed service slots
- non-main explorer sessions should be destroyed before heavy eval

The second most important rule is that the agent must learn task geometry from vision, not oracle state.

## Snapshot Of The Current User Intent

The user is trying to make OpenCode a serious replacement for the older local `run_agent.py` loop:

- It should run in this repo with the current service setup.
- It should use Claude Opus 4.7 through the configured OpenCode provider.
- It should use a team of subagents for exploration.
- It should use multiple isolated RoboCasa sessions without service port conflicts.
- It should evaluate in parallel over the canonical benchmark seeds.
- It should expose real-time evaluation status in the TUI sidebar.
- It should avoid oracle task information and rely on vision.
- It should reuse the curated RoboCasa skill library.
- It should remain debuggable through tmux, logs, and status files.

The current state is close to that architecture. The main remaining work is hardening:

- cleanup of explorer sessions
- stronger tests around MCP session lifecycle
- better TUI status polish
- modularizing `cap/mcp/server.py`
- watching memory/CPU over long OpenCode sessions
