# Composite Reflection Prompt

## Scene Query

The robot was attempting this task: {task}

Describe the robot's current state at the END of this attempt. Focus on: where is the gripper/fingertip relative to the target, did the robot accomplish the task goal, and what visually went wrong.

## Analysis Prompt

Jointly consider the planned approach, the code, the visual evidence, and the execution results. In a single unified analysis:
1. Where did reality diverge from the plan? What does the visual evidence confirm?
2. What is the root cause?
3. Give 2-4 specific, actionable fixes for the NEXT attempt.
Be concrete. Do NOT rewrite the code. Under 250 words.

## Skill library discipline

If your recommended fixes touch a skill that is in the "Available skills" index, classify **every** change as one of:

- **reuse** — call the existing skill with different arguments (different target, different side, different clearance).
- **refine** — same mechanism, tuned parameters / minor logic. Bump the version: `X_v1` → `X_v2`. Base name stays.
- **replace** — genuinely different mechanism (different tool, different algorithm). New base name. Name the different mechanism in one sentence.

Default to **reuse**. Never recommend rewriting a skill whose `success_rate > 0.8` — if it works for most seeds, the failing seeds need different arguments, not different code.

Renaming the same code under a new base_name (e.g. `sink_hover_v1` when `hover_above_v1` already exists and covers it) is **forbidden**. The position argument already carries scene context.
