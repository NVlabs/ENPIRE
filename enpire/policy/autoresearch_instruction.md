# ENPIRE Auto-Research Contract

An auto-research agent may improve policy and training code, but it must not
change the environment, reset procedure, verifier, safety limits, evaluation
seeds, or artifact schema.

## Required loop

1. Record one falsifiable hypothesis.
2. Make the smallest policy-side change needed to test it.
3. Run the configured unit tests and offline evaluation first.
4. Run only the approved number of real-world trials.
5. Inspect `result.json`, events, metrics, and video evidence.
6. Keep the change only when it passes the configured success and regression gates.

## Real-world sequencing

For online PLD experiments, preserve this order. The learner discovers the
current robot-side data directory only when it starts.

1. Start the robot-side bridge and confirm `rl control health` succeeds.
2. Pause any active rollout.
3. Restart the bridge output directory and record the returned external path.
4. Start the learner; verify its disk ingestor watches that exact path.
5. Start the actor and verify the learner/actor handshake.
6. Resume the robot loop only after the operator-approved trial budget begins.
7. Monitor learner ingestion and robot events; pause immediately on a stall,
   unsafe state, or invalid observation.
8. Pause at the budget boundary and compute the frozen rolling-success metric.

```bash
uv run enpire rl control health
uv run enpire rl control pause --confirm-control
uv run enpire rl control restart --confirm-control
uv run enpire rl control resume --confirm-control
uv run enpire rl score --data-dir /external/run/path --window 50
```

Do not replace signal-based monitoring with blind sleeps. A run is invalid if
the learner watches a stale directory, the policy sees a changed reset or
success contract, or human actions are counted as pure-RL actions.

## Safety and integrity

- Never place credentials in code, configuration, prompts, logs, or artifacts.
- Never weaken or bypass a hardware confirmation, workspace limit, or emergency stop.
- Never read privileged simulator state from policy code.
- Never redefine success inside the policy.
- Stop when device identity, calibration, or environment readiness checks fail.
- Preserve failed trials; they are part of the experiment record.
- The score implementation and robot-side event/reward code are read-only to a
  policy researcher; changing them invalidates comparison with prior runs.

## Experiment record

Each iteration must retain the hypothesis, diff, source commit, resolved public
configuration, data/checkpoint references, trial budget, results, and decision.
Dataset and checkpoint paths may be external, but credentials are always redacted.
