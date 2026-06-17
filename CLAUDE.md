# lecar-tbd

YAM bimanual robot platform — Code as Policy (CAP) agent framework with RL training pipeline.

## Env Layer (cap/env/)

CAP has a 4-layer architecture: Agent → Tools → Server → Env.
The env layer (`cap/env/`) owns all robot/sim-specific logic. Adding a new robot = one new file.

- `cap/env/base.py` — Protocols: `EnvProtocol`, `MotionProtocol`, `SceneProtocol`, `TaskProtocol`
- `cap/env/profile.py` — `RobotProfile` / `ArmProfile` dataclasses + factory functions
- `cap/env/yam.py` — YAM MuJoCo sim (pinocchio IK)
- `cap/env/yam_mujoco.py` — re-exports `SimBackend` as `YamMuJoCoEnv`
- `cap/env/yam_warp.py` — re-exports `WarpSimBackend` as `YamWarpEnv` (lazy, GPU)
- `cap/env/robocasa.py` — RoboCasa365 (robosuite OSC_POSE controller)
- `cap/env/adapters/sim.py` — `SimArmAdapter`, `SimCameraAdapter` (CapServer glue)

### Remaining YAM-specific debt (to clean up when testing on physical YAM)

- `cap/server/cap_server.py` lines ~980-1030, ~1210-1230: Legacy pinocchio IK path for real hardware (runs only when no `MotionProtocol` env, i.e. physical YAM without an env wrapper)
- `cap/agent/visualizer.py` lines ~75-77, ~857: `YamKinematics` hardcoded for Viser 3D viz
- `cap/agent/tools/freespace_move.py` lines ~53, ~287, ~935: `YamKinematics` hardcoded for cuRobo motion planning
- Fix: Make real hardware YAM go through a `YamHardwareEnv` that implements `MotionProtocol`, then delete the legacy paths. Make visualizer and freespace_move read kinematics from the robot profile.

## Architecture

See `docs/` for design documentation:
- `docs/AGENT_PIPELINE_DESIGN.md` — Agent pipeline (run_agent.py): pipeline steps, AgentContext, PromptMemory, ExecutionMemory, ToolHandle, log folder
- `docs/BUNDLESDF_OBJECT_DETECTION.md` — BundleSDF multi-object 6-DOF pose tracking system
- `docs/CAP_DESIGN.md` — CAP system architecture and layer stack (overall design + real-world YAM setup)
- `docs/CAP_ROBOCASA.md` — RoboCasa sim benchmark: env integration, code-as-policy agent vs GR00T policy eval, runnable commands, autoresearch loop
- `docs/PHYSICAL_TOOLS_GUIDE.md` — Physical tools (perception/planning/contact): GT/oracle vs real-transferable tools, camera path + visual-input formats + BundleSDF I/O, compliant gripper close, end-to-end example
- `docs/CAP_SYSTEM_DASHBOARD.md` — Bringup system dashboard
- `docs/CAP_UI_DESIGN.md` — CAP web UI layout, components, data flow, and design decisions
- `docs/CLAUDE_SETTINGS.md` — Claude Code project settings: hooks (pre-commit docs check, ruff, session start), attribution, plugins
- `docs/CUROBO_ISAACSIM_SETUP.md` — YAM cuRobo robot configs, Isaac Sim environment setup, and Lula sphere fitting workflow
- `docs/CUROBO_UPDATE_SUMMARY.md` — cuRobo integration update: robot configs, world collision, depth scene, MuJoCo validation
- `docs/DATA_STUDIO.md` — TBD Data Studio: `tbd data` CLI (inspect/label/replay), overlay viz server, episode format, replay policy, recording wrapper
- `docs/MACMINI_SETUP.md` — Mac Mini setup for CAP sim mode (macOS deps, Darwin auto-detect, Fello leader arm)
- `docs/MULTI_CAMERA_CONFIG.md` — Multi-source camera configuration (per-station RealSense/ZED profiles)
- `docs/RL_PIPELINE_DESIGN.md` — RL training pipeline design (serve_rl_policy, learn_skill, diagnostics)
- `docs/plan/REAL_YAM_ENV.md` — Plan: Real YAM env — `RealYamEnv` implementing `EnvProtocol` for physical YAM bimanual hardware, camera depth/intrinsics/extrinsics, FK/IK via `YamKinematics`
- `docs/plan/episodic_agent_skill_library.md` — As-built notes: per-run episodic skill library (`skill_library/`, `@skill` decorator, `SkillProfile`/`SkillRegistry`, subprocess flush, log aggregation)
- `docs/plan/beam_search_agent_loop.md` — Plan (pending): K-beam strategy diversity — StrategyPlannerStep + parallel AssemblyGeneratorStep + beam-aware history/escalation
- `docs/ROBOCASA_INTEGRATION.md` — RoboCasa env integration: RoboCasaEnv, OSC controller, camera system, task management
- `docs/ROBOCASA_INTEGRATION_POLICY.md` — GR00T policy evaluation: PandaOmron24 (N1.6), RoboCasa365 benchmark (N1.5/N1.6), env setup, model checkpoints
- `docs/ROBOCASA_RANDOMNESS.md` — RoboCasa randomness hierarchy (13 levels), determinism controls, seed propagation, env vars for pinning layout/style/seed
- `docs/ROBOCASA_SYSTEM2_SURVEY.md` — Survey: System 2 Claude Agent for RoboCasa composite tasks (open-loop vs closed-loop policy invocation)
- `docs/REWARD_MODULE.md` — Pluggable task-success evaluators (`cap/reward/`): oracle reconstructs each `_check_success` sub-predicate from `result.json.details`, injected into Phase A VLM + Phase B reflection
- `docs/SAFETY_ZONE_DESIGN.md` — Task-aware EE safety zones for RL exploration
- `docs/SERIAL_FOOTSWITCH.md` — Serial 3-button footswitch (Waveshare RP2040-Zero) integration
- `docs/SKILL_LIBRARY.md` — Skill library overview: full tool catalog (~43 tools), data types, per-env comparison
- `docs/SKILL_LIBRARY_ROBOCASA.md` — Skill library for RoboCasa: OSC vs cuRobo modes, pick/place patterns, task strategies
- `docs/SKILL_LIBRARY_YAM.md` — Skill library for YAM: bimanual tools, table bussing, RL workflow, hardware notes
- `docs/TABLE_BUSSING_SKILLS.md` — Table bussing skill tools (tracking, freespace move, nudge, gripper)
- `docs/TODO_DAMIAO_FOR_MAC.md` — TODO: Damiao motor macOS (gs_usb) support
- `docs/TODO_DEBUG_OVERRUN.md` — Debugging control loop overruns (timing, profiling, thread analysis)
- `docs/VISER_CUROBO_PLANNER.md` — Interactive Viser 3D UI for cuRobo motion planning in RoboCasa (gizmo target, trajectory visualization, execution)
- `docs/VISER_IK_TELEOP_DESIGN.md` — Viser 6D gizmo IK teleoperation (ScriptedPolicy, gizmo live-teleop, Move-To panel, motion planner integration)
- `docs/VLM_QUERY.md` — VLM query tool: multi-backend vision-language queries (Gemini, Claude, OpenAI, Qwen3)
- `docs/VOICE_INPUT.md` — Voice input and output integration
- `docs/data_summary_new.md` — Data loader survey for online robot foundation model training
- `docs/debug_shit_data_collection_infra.md` — Debug plan and findings for data collection timing issues
- `docs/deploy_openpi.md` — Deploy OpenPI policy server (Pi0.5)
- `docs/frequency_relationship.md` — Frequency and timing relationships (control loop, policy, camera)
- `docs/grasp_orientation.md` — Grasp orientation conventions and grasping strategy
- `docs/INSTALL.md` — End-to-end environment install via `bash install/install_cap.sh`; covers `install/compile_bundlesdf.sh`, `.forge_env`, conda build envs, project-local `third_party/{bundlesdf_5090,anygrasp_libs,nodejs}/`, driver/CUDA safety, and launcher integration
- `docs/GROOTPOOL.md` — GR00T policy pool middleware: sticky-per-episode session routing, CAP integration via `use_policy_output` (auto-resolves backend / task prompt / endpoint; asserts OSC pinning), `RoboCasaEnv.as_gym_env()`, `GrootpoolN15Backend` / `GrootpoolN16Backend` with obs-spec metadata, OSMO launch via `tmux/launch_grootpool.sh`, `/status` admin
- `docs/lfs_setup.md` — Git LFS setup: why `fetchexclude = *` is the lightweight default, how launcher scripts materialize binaries, manual override steps, and what breaks if LFS objects are missing
- `docs/remote_serving.md` — Remote GPU serving setup (LeCAR-S1 deployment)

## Key Directories

- `cap/` — CAP agent framework (server, agent, UI, reward, diagnostics)
- `cap/server/cap_server.py` — `CONTROL_FREQ_HZ` control loop, Portal RPC server
- `cap/agent/cap_agent.py` — FastAPI orchestrator + WebSocket UI backend
- `cap/debug_ui/` — lightweight read-only run monitor (auto-launched by `run_script.py`)
- `robot/` — Hardware drivers (Fello arms, grippers, motors)
- `bringup/` — System launcher and dashboard
- `hardware/` — USB udev rules and CAN bringup (MKS `gs_usb` vs CANable2 slcan); see `hardware/README.md`
- `cap/prompt/` — Agent prompt & memory (markdown-on-disk, managed by `PromptMemory`)
  - `system/` — System prompts (identity, workflow, rules, reflection, review, retry)
  - `tools/` — Per-tool documentation (detect_object, bundlesdf, yam_coordinate_system)
  - `embodiment/` — Per-robot specs (robocasa_panda, robocasa_gr1)
  - `task/` — Task-specific strategy guides (pick_place, microwave, table_bussing)
  - `heuristics/` — Agent-learned heuristics (cross-run persistent)
  - `loader.py` — `PromptMemory` class: index-based prompt resolution & injection
- `experiments/` — Hydra experiment configs (validated against `cap/agent/agent_config.py` schemas)
  - `config.yaml` — Base defaults + env var fallbacks
  - `experiment/` — Per-experiment configs (select with `experiment=NAME`)
- `docs/` — Design docs and architecture diagrams
- `tmux/` — Launch scripts for services

## Conventions

- Python 3.11+, managed with `uv`
- Portal RPC for inter-process communication
- WebSocket for UI real-time updates
- `emit()` for UDP diagnostic events (consumed by `cap/diag/dashboard.py`)

## Running

```bash
# Launch core services
tmux/launch_basics.sh

# Launch sim mode
tmux/launch_sim.sh

# Run CAP agent (Hydra config — everything in one YAML)
uv run python run_agent.py experiment=pick_place_sink_to_counter

# Override fields from CLI
uv run python run_agent.py experiment=pick_place_sink_to_counter env.seed=99

# Show resolved config
uv run python run_agent.py experiment=pick_place_sink_to_counter --cfg job

# Run CAP agent (bridge mode, interactive)
uv run cap/agent/cap_agent.py
```

## Monitoring UI

`run_script.py` / `run_agent.py` auto-launch a lightweight read-only debug UI
(`cap/debug_ui`, `debug_ui.enabled=true`) that watches the run's log directory.
There is no separate frontend build step. (The legacy React `cap/ui` + `cap_agent`
bridge UI has been removed; see `docs/CAP_DESIGN.md`.)

## Git Workflow

- Run `git config core.hooksPath .githooks` after cloning to enable git hooks
- Branch naming: `<developer>/<feature>` (e.g. `wenlix/sim_insertion_task`) — enforced by pre-push hook
- Never push directly to `main` — always use PRs (enforced by pre-push hook)
- Branches are auto-deleted after PR merge (both on GitHub and locally via post-merge hook)

## Rules

1. Never push directly to `main` — always use PRs
2. Branch names must follow `<developer>/<feature>` convention (e.g. `wenlix/sim_insertion_task`)
3. Run `uv run ruff check --fix . && uv run ruff format .` before committing Python changes
4. Do not commit `.env` files, secrets, or large binary data
5. **Always update `docs/`** after any implementation or code design work — create a new design doc or update the relevant existing one. Keep `docs/` as the single source of truth for system architecture and design decisions. Link new docs from the Architecture section above.
6. **At the start of every session**, read all `docs/*.md` files before doing any work. After reading them, output a short confirmation message listing the docs you read (e.g. "Docs loaded: CAP_DESIGN.md, ROBOCASA_INTEGRATION.md, ...").
