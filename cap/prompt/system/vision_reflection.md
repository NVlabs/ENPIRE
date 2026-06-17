# Vision Reflection Prompt

## Scene Query

The robot was attempting this task: {task}

Describe the robot's current state at the END of this attempt. Focus on: (1) where is the gripper/fingertip relative to the target, (2) did the robot accomplish the task goal (e.g. press a button, pick up an object, etc.), (3) what visually went wrong — be specific about positions and orientations.

## Analysis Prompt

You are analyzing a failed robot manipulation attempt to guide the next try.

Task: {task}

=== PLANNED APPROACH (what the agent expected to happen) ===
{thoughts}

=== VISUAL SCENE (what actually happened — cameras: {cameras}) ===
{scene_description}

=== VISUAL CHANGES (before -> after execution) ===
{visual_diff}

=== Task evaluation ===
{eval_info}

=== Execution error ===
{error}

=== Tool call history ===
{tool_history}

Compare the PLANNED APPROACH vs what the VISUAL SCENE and VISUAL CHANGES show:
1. Where did reality diverge from the plan?
2. What does the camera show that contradicts the expected behavior?
3. What specific changes occurred between the start and end of execution?
4. Give 2-4 specific, actionable fixes for the NEXT attempt.
Be concrete — include corrected positions or sequences. Plain text only, under 300 words.
