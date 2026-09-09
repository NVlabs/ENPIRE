# ENPIRE: Agentic Robot Policy Self-Improvement in the Real World

[Project Page](https://research.nvidia.com/labs/gear/enpire/) ·
[Paper](https://arxiv.org/abs/2606.19980) ·
[Documentation](enpire/env/docs/index.html)

<p align="center">
  <img src="assets/main_figure.png" alt="ENPIRE overview" width="100%">
</p>

ENPIRE is a harness for coding agents to improve robot policies through
repeatable real-world experiments. Agents propose changes, evaluate them on a
physical station, inspect measurements and recordings, and use that evidence
to guide the next iteration.

The framework connects four modules:

- **Environment (EN):** reset the scene and verify task outcomes.
- **Policy Improvement (PI):** Iteratively improve performance via different learning paradigms (RL, BC, Heuristic Learning, etc.)
- **Rollout (R):** evaluate policies on one or more robots.
- **Evolution (E):** analyze failures, develop hypotheses, and improve the next experiment.

Together, they form the loop **reset → execute → verify → record → refine**.
Code-as-Policy (CaP) supports heuristic learning through Python skill scripts;
the real-world RL environment supports neural policy training with an
actor/learner pipeline. Both use the station's robot, perception, and calibration tools.

## Quick Start

### Run with a coding agent

Launch a coding agent and give it the following prompt:

```text
Clone https://github.com/NVlabs/ENPIRE.git with submodules and read AGENTS.md.
Follow enpire/env/docs/INSTALL.md and run 00_hello_environment.
Use the existing configured environment when one is provided.
Summarize the available tools and task entry points.
For real-robot work, follow enpire/env/docs/REAL_WORLD_WORKFLOWS.md:
complete station and calibration preflight, and obtain explicit motion authorization.
```

[AGENTS.md](AGENTS.md) describes the repository layout, implementation rules,
and environment ownership. The [auto-research instructions](enpire/policy/autoresearch_instruction.md)
define the experiment loop and the reset, verification, and safety boundaries
that a policy researcher must preserve.

### Manual setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Start with the
hardware-free example:

```bash
git clone --recurse-submodules https://github.com/NVlabs/ENPIRE.git
cd ENPIRE
uv sync --extra dev
uv run enpire examples run 00_hello_environment
```

For an existing checkout, run `git submodule update --init --recursive` before
using uv. The [installation guide](enpire/env/docs/INSTALL.md) covers hardware
extras; `uv sync` replaces the selected extras, so include all required extras
in one invocation.

Real-robot runs require a registered station, calibrated cameras and robot
transforms, and the task's perception and control services. Follow the
[station and task workflow](enpire/env/docs/REAL_WORLD_WORKFLOWS.md) for setup.
The [real-world RL environment](enpire/policy/pld/runtime/README.md) uses an
isolated dependency project for its actor and learner.

### Set up a custom real-world environment for auto-research

For each environment, the user must provide a **reset function** and a
**reward/verification function**, using CaP or other task-specific scripts.
A coding agent can help write and validate these during environment setup.
The [project website](https://research.nvidia.com/labs/gear/enpire/) illustrates
physical resets for Push-T, pin insertion, GPU insertion, and zip-tie tasks,
along with vision-based reward evaluation.

1. **Prepare the station.** Register the robot and cameras, calibrate the station,
   and start the required services using the
   [real-world setup guide](enpire/env/docs/REAL_WORLD_WORKFLOWS.md).
2. **Define and validate the environment.** Specify observations and actions.
   Write a reset that restores the scene and checks readiness, and a reward
   function that scores progress or success from camera images, robot state,
   or contact measurements. Follow the [task guide](enpire/env/docs/NEW_TASK.md)
   for script templates and task registration.
3. **Run auto-research.** Validate repeated reset and evaluation cycles on the
   real station, then keep reset, reward, verification, and safety rules fixed
   while the agent improves policy or training code. Set a trial budget and
   retain outcomes, logs, and videos as described in the
   [auto-research contract](enpire/policy/autoresearch_instruction.md).

Examples to build from:

- [Hello environment](enpire/env/examples/00_hello_environment/README.md): a minimal reset–execute–verify loop.
- [Real object pickup](enpire/env/examples/10_real_object_pick/README.md): CaP composition of perception, planning, and robot control.
- [Push-T](enpire/env/docs/NEW_TASK.md#push-t-reference-implementation): a physical reset workflow and heuristic policy improvement.

## Contribution Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions must be signed off
under the Developer Certificate of Origin and licensed under Apache-2.0.

### Security

To report a security vulnerability, visit
[https://www.nvidia.com/en-us/security/](https://www.nvidia.com/en-us/security/).
See [SECURITY.md](SECURITY.md) for credential and hardware safety rules.

## License

Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Licensed under the [Apache License 2.0](LICENSE).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for third-party attributions.

## Citation

If you use ENPIRE in your research, please cite the [paper](https://arxiv.org/abs/2606.19980):

```bibtex
@misc{xiao2026enpireagenticrobotpolicy,
      title={ENPIRE: Agentic Robot Policy Self-Improvement in the Real World},
      author={Wenli Xiao and Jia Xie and Tonghe Zhang and Haotian Lin and
              Letian "Max" Fu and Haoru Xue and Jalen Lu and Yi Yang and
              Cunxi Dai and Zi Wang and Jimmy Wu and Guanzhi Wang and
              S. Shankar Sastry and Ken Goldberg and Linxi "Jim" Fan and
              Yuke Zhu and Guanya Shi},
      year={2026},
      eprint={2606.19980},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.19980},
}
```
