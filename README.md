# ENPIRE: Agentic Robot Policy Self-Improvement in the Real World

> A harness for closing the **physical feedback loop** so robot policies can improve themselves — **EN**vironment construction · **P**olicy **I**mprovement · **R**ollout · **E**volution.

**🌐 Project website: [research.nvidia.com/labs/gear/enpire](https://research.nvidia.com/labs/gear/enpire/)**

![ENPIRE](asset/main_figure.png)

[main figure (PDF)](asset/main_figure.pdf)

*(This repo, internally "CAP" / forge, is the framework that implements ENPIRE on the YAM bimanual robot and the RoboCasa simulator.)*

## Introduction

2026's defining AI themes are recursive self-improvement, autoresearch, and agents that get stronger every iteration. Wherever a domain offers a *repeatable feedback loop* — propose, run, observe, revise, repeat — agents can iterate on their own. That is why so much of the digital world (games, ML experiments, codebase maintenance, GPU kernels) is becoming "agent-solvable": a "run" is just one command. Robotics is different. In the physical world a rollout is a real event: you must reset the scene, execute safely, judge success, and read the logs, video, and reward before you can improve. A coding agent can write training code and propose algorithms, but if every experiment still needs a human to reset, label, and tune, robot research is not actually automated.

ENPIRE's bet is that the missing piece is exactly that physical feedback loop — **reset → execute → verify → refine** — that makes the real world iterable. For tasks with a generation–verification gap this is tractable: producing a successful behavior is hard, but *resetting the scene* and *checking success* are often easy to assemble from motion planners, OpenCV, SAM, and similar tools. (Take GPU or pin insertion: learning the fine motion is hard; "put the part back" and "check whether it is inserted" are not.) Once the loop closes, agents do what they are best at — read failure logs, edit training code, mix BC / RL / code-as-policy, keep what works and discard what does not — except now the object of iteration is *robot behavior*, not just code. After pretraining, RLHF, and RLVR, the next recipe may simply be: **close the feedback loop wherever the world lets us.**

## The ENPIRE loop

| Module | What it does | Code | Docs |
|---|---|---|---|
| **EN**vironment construction | Auto-reset + auto-verification env interfaces and reward checkers | `cap/env/` (`real_bimanual_yam/`, `robocasa/`), `cap/reward/` | CAP_DESIGN, CAP_ROBOCASA, REWARD_MODULE |
| **P**olicy improvement | Improve training code / algorithms / policies (BC · RL · code-as-policy) | `run_agent.py`, `tmux/realworld_rl/` (learner/actor live in the separate `minimal_policy` repo) | RL_PIPELINE_DESIGN, AGENT_PIPELINE_DESIGN, PHYSICAL_TOOLS_GUIDE |
| **R**ollout | Execute and evaluate on hardware or sim | `run_script.py` (direct-mode), `tmux/realworld_rl/rl_gear.sh` | CAP_DESIGN (execution model), CAP_ROBOCASA (commands) |
| **E**volution | Select across hypotheses / agents / robots (autoresearch) | autoresearch loop | OPENCODE_ROBOCASA_AUTORL_HANDOFF, ROBOCASA_SYSTEM2_SURVEY |

The same **Agent → Tools → Env** stack runs unchanged across embodiments; only the bottom `cap/env/` layer swaps between real YAM hardware, MuJoCo/Warp sim, and RoboCasa.

## Quickstart

```bash
# Install — full environment setup (see docs/INSTALL.md)
bash install/install_cap.sh

# Run the code-as-policy agent in RoboCasa sim (see docs/CAP_ROBOCASA.md)
uv run python run_agent.py experiment=pick_place_sink_to_counter

# Real-world RL env loop (learner/actor run from the minimal_policy repo)
bash ./tmux/realworld_rl/rl_gear.sh --task gpu_insertion --station yam-auto-n --use-spacemouse
```

See **docs/CAP_DESIGN.md** ("Real-World RL, Inference & Data Collection") for the full real-world pipeline and **docs/INSTALL.md** for install details.

## Documentation index

Start with **CAP_DESIGN**, **CAP_ROBOCASA**, **PHYSICAL_TOOLS_GUIDE**, and **RL_PIPELINE_DESIGN**.

**Architecture**
- [CAP_DESIGN](docs/CAP_DESIGN.md) — overall architecture, direct-mode `run_script`/`run_agent`, real-world YAM setup.
- [AGENT_PIPELINE_DESIGN](docs/AGENT_PIPELINE_DESIGN.md) — agent pipeline steps, context, memory, logging.
- [CAP_SYSTEM_DASHBOARD](docs/CAP_SYSTEM_DASHBOARD.md) — service catalog, ports, launch profiles.

**Tools & Skills**
- [PHYSICAL_TOOLS_GUIDE](docs/PHYSICAL_TOOLS_GUIDE.md) — perception / planning / contact tools, GT-vs-real behavior.
- [SKILL_LIBRARY](docs/SKILL_LIBRARY.md) — full tool catalog, data types, per-env comparison.
- [SKILL_LIBRARY_YAM](docs/SKILL_LIBRARY_YAM.md) / [SKILL_LIBRARY_ROBOCASA](docs/SKILL_LIBRARY_ROBOCASA.md) — per-embodiment skill notes.
- [BUNDLESDF_OBJECT_DETECTION](docs/BUNDLESDF_OBJECT_DETECTION.md) — 6-DOF multi-object pose tracking.
- [VLM_QUERY](docs/VLM_QUERY.md) — multi-backend vision-language queries.
- [CUROBO_ISAACSIM_SETUP](docs/CUROBO_ISAACSIM_SETUP.md) / [VISER_CUROBO_PLANNER](docs/VISER_CUROBO_PLANNER.md) — cuRobo motion planning.

**Sim / RoboCasa**
- [CAP_ROBOCASA](docs/CAP_ROBOCASA.md) — sim benchmark, code-as-policy vs GR00T eval, autoresearch loop, commands.
- [ROBOCASA_INTEGRATION](docs/ROBOCASA_INTEGRATION.md) / [ROBOCASA_INTEGRATION_POLICY](docs/ROBOCASA_INTEGRATION_POLICY.md) — env integration and GR00T policy eval.
- [GROOTPOOL](docs/GROOTPOOL.md) — GR00T policy pool middleware.
- [ROBOCASA_RANDOMNESS](docs/ROBOCASA_RANDOMNESS.md) / [ROBOCASA_ORACLE](docs/ROBOCASA_ORACLE.md) — determinism controls and task oracle.

**Real-World RL & Self-Improvement**
- [RL_PIPELINE_DESIGN](docs/RL_PIPELINE_DESIGN.md) — RL training pipeline, serving, diagnostics.
- [REWARD_MODULE](docs/REWARD_MODULE.md) — pluggable task-success evaluators.
- [SAFETY_ZONE_DESIGN](docs/SAFETY_ZONE_DESIGN.md) — task-aware EE safety zones for exploration.
- [OPENCODE_ROBOCASA_AUTORL_HANDOFF](docs/OPENCODE_ROBOCASA_AUTORL_HANDOFF.md) — autoresearch / auto-improve loop.
- [ROBOCASA_SYSTEM2_SURVEY](docs/ROBOCASA_SYSTEM2_SURVEY.md) — System 2 agent survey for composite tasks.
- [DATA_STUDIO](docs/DATA_STUDIO.md) — episode browser, replay, labeling.

**Infra & Setup**
- [INSTALL](docs/INSTALL.md) — end-to-end environment install.
- [MULTI_CAMERA_CONFIG](docs/MULTI_CAMERA_CONFIG.md) — per-station RealSense/ZED camera profiles.
- [remote_serving](docs/remote_serving.md) — remote GPU serving.
- [MACMINI_SETUP](docs/MACMINI_SETUP.md) — Mac Mini sim dev environment.
- [SERIAL_FOOTSWITCH](docs/SERIAL_FOOTSWITCH.md) / [VOICE_INPUT](docs/VOICE_INPUT.md) — footswitch and voice I/O.
- [lfs_setup](docs/lfs_setup.md) — Git LFS configuration.

## Repo layout

```
cap/        CAP agent framework (env, agent, tools, server, reward, UI, diagnostics)
robot/      Hardware drivers (Fello arms, grippers, motors)
bringup/    System launcher and dashboard
tmux/       Launch scripts (sim, real-world RL, serving)
experiments/ Hydra experiment configs
docs/       Design docs (index above)
asset/      Figures and media
```

See **CLAUDE.md** for the detailed directory map, env-layer design, and contributor conventions.

## Git workflow

```bash
git config core.hooksPath .githooks
```

- Branch naming: `<developer>/<feature>` (e.g. `wenlix/sim_insertion_task`).
- Never push directly to `main` — use PRs.

## Citation

If you build on ENPIRE, please cite:

```bibtex
@misc{enpire2026,
  title  = {ENPIRE: Agentic Robot Policy Self-Improvement in the Real World},
  author = {Wenli Xiao and Jia Xie and Tonghe Zhang and Haotian Lin and Letian Fu and
            Haoru Xue and Jalen Lu and Yi Yang and Cunxi Dai and Zi Wang and Jimmy Wu and
            Guanzhi Wang and S. Shankar Sastry and Ken Goldberg and Linxi Fan and
            Yuke Zhu and Guanya Shi},
  year   = {2026},
  url    = {https://research.nvidia.com/labs/gear/enpire/},
  note   = {NVIDIA, CMU, UC Berkeley},
}
```
