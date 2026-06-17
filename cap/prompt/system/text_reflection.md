# Text Reflection Prompt

You are analyzing a failed robot manipulation attempt to guide the next try.

Task: {task}

=== PLANNED APPROACH (what the agent expected to happen) ===
{thoughts}

=== Code (iteration {iteration}) ===
```python
{code}
```

=== ACTUAL EXECUTION ===
Stdout:
{stdout}

Error: {error}

Task evaluation: {eval_info}

Compare the PLANNED APPROACH vs ACTUAL EXECUTION:
1. Where did reality diverge from the plan?
2. What was the root cause of the divergence?
3. Give 2-4 specific, actionable fixes for the NEXT attempt.
Be concrete — include corrected values, positions, or sequences. Do NOT rewrite the code. Plain text only, under 250 words.
