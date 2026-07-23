# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# groot_only.py — skill library, append-only
from skill_library.namespace import *  # noqa: F401, F403

from enpire.env.forge.cap.agent.skill_registry import skill


@skill
def groot_only_v1(model="grootpool/n15", replan_horizon=16):
    """Run GR00T policy for the current task, selecting max_steps by task type.

    Tasks that require articulated manipulation (cabinet, drawer, dishwasher)
    get a larger step budget than simple pick-place tasks.
    """
    task_description = get_task_description()
    task_text = task_description.lower()

    if "cabinet" in task_text or "fridge" in task_text:
        max_steps = 1500
    elif "drawer" in task_text or "dishwasher" in task_text:
        max_steps = 1000
    elif "sink faucet" in task_text and "turn on" in task_text:
        # Lever rotation is consistently slower under n15 than knob-press
        # tasks; 800 steps timed out 4/5 seeds for TurnOnSinkFaucet at
        # layout_id=3 style_id=5 (logs/20260506T224451_agent_…).
        max_steps = 1200
    elif "turn off" in task_text:
        # Same shape as TurnOnSinkFaucet — TurnOff* (stove/microwave/sink-
        # faucet/simmered-sauce) is lever/knob rotation; 800 under-budgets.
        max_steps = 1200
    elif "turn on" in task_text or "lower heat" in task_text:
        max_steps = 800
    elif "navigate" in task_text:
        max_steps = 1500
    elif "set up the coffee mug" in task_text:
        # CoffeeSetupMug — placement under the coffee-machine spout is a
        # tight clearance; the default 500-step PickPlace bucket under-
        # budgets the rollout.
        max_steps = 1200
    elif "open the stand mixer head" in task_text:
        # OpenStandMixerHead — hinge-rotation articulation; the default
        # 500-step bucket under-budgets the rollout.
        max_steps = 1200
    else:
        # PickPlace* and everything else
        max_steps = 500

    print(
        f"[groot_only_v1] instruction={task_description!r}  "
        f"max_steps={max_steps}  model={model}"
    )
    result = use_policy_output(
        model=model,
        replan_horizon=replan_horizon,
        max_episode_steps=max_steps,
        task_description=task_description,
    )
    return result
