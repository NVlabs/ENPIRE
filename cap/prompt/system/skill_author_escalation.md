# Escalation: Parameter Refinements Exhausted

⚠ **ESCALATION**: The experiment history shows the baseline score has been
unchanged for {stagnant} consecutive iterations. Parameter refinements
(`refine:`) have been exhausted — every variant tried has failed to improve
on the current baseline.

This iteration you **MUST** propose either:

- `new:` — a completely new skill family with a qualitatively different
  underlying mechanism (different motion primitive, different approach
  direction, different control strategy)
- `replace:` — replace an existing skill whose mechanism is fundamentally
  wrong for the observed failure mode, not just mis-tuned

**Do NOT** author another `refine:` of an existing skill. That direction has
already been tried repeatedly and failed.

To choose the right new mechanism, read the experiment history above and ask:
*what has NOT been tried yet?* If all attempts have used straight-down descent,
try a lateral or angled approach. If all attempts have used freespace_move for
the critical step, try incremental nudge-based motion. If the grasp mechanism
is unchanged across all iterations, replace it.
