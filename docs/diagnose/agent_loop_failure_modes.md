# Agent Loop Failure Modes

This note explains why the current agent loop can drift into worse behavior even when it is technically “using feedback.” The issue is not a single broken module. It is the way feedback is allowed to mutate policy without a curation gate.

## 1. What the loop is doing now

The loop collects per-seed failures, per-iteration reflections, and skill-authoring prompts, then feeds that material directly into the next synthesis step. In practice, that means the next code generation round sees a large amount of recent failure context and is encouraged to “fix” it immediately.

This is effective at producing change, but it is not effective at preserving a good policy. A better iteration can be found, but the loop has no strong mechanism for protecting it from later over-specialized edits.

## 2. Why this causes regressions

The loop is too permissive about what can change at once.

The `skill_author` stage can invent or refine many skills, and the `assembly_generator` stage can rewrite orchestration around those skills. That gives the model a wide mutation surface. When the reflection signal says “small objects are failing” or “IK is failing in the sink,” the synthesis step tends to respond by making the grasp deeper, more aggressive, or more specialized. That may help the failing seeds, but it can break seeds that were already working.

The result is a classic local-search failure:

- a promising baseline appears,
- the next round overreacts to the most visible failures,
- the policy narrows,
- new brittle helpers are added,
- performance oscillates or degrades.

## 3. The systematic fix

The loop needs a curation layer between diagnosis and mutation. The goal is not another hotfix. The goal is to change the control flow so the agent cannot keep drifting away from the best known policy simply because a recent failure was loud.

The curation layer should:

- compress raw failures into one or two dominant hypotheses,
- preserve the last kept baseline as the only hard anchor,
- select a single mutation target per iteration,
- keep failure data as evidence, not as direct instructions,
- discard or archive bad trials without letting them become the live policy state,
- avoid feeding the full discarded history back into generation as if it were the current target.

In other words, the loop should not let reflection act like an optimizer directly. Reflection should produce evidence. A separate decision layer should decide what evidence is worth turning into a code change.

The concrete control-flow change should look like this:

1. Observe.
   - Collect logs, scores, per-seed reflections, and video or key frames.

2. Diagnose.
   - Reduce the raw evidence into a small number of failure hypotheses.
   - Example: “grasp misses small round objects,” not “add three more retries and two new skill variants.”

3. Choose one mutation.
   - One skill family or one assembly change per iteration.
   - Do not rewrite grasp, lift, place, and retract all at once.

4. Generate from the kept baseline.
   - Start from the last promoted snapshot only.
   - Do not use the latest discarded iteration as the live anchor.

5. Validate.
   - Compare against the kept baseline and a small holdout slice if available.
   - Reject trials that improve one failure mode by breaking the broader policy.

6. Promote or discard.
   - If the trial wins, promote it to baseline.
   - If it loses, restore the baseline and archive the trial.

This is the main difference from the current behavior visible in the generated code. The failing run did not stay centered on one stable champion; it repeatedly added object-specific heuristics, alternate skill versions, and fallback branches. That is the exact failure mode this control flow is meant to prevent.

There is also a lower-friction version of this fix: instead of adding a new curation module, tighten the existing reflection prompt so it performs the curation role itself. In that version, reflection should not just summarize failures. It should emit a structured decision memo that names the dominant failure, preserves the kept baseline, selects one allowed mutation, and lists forbidden changes. Skill authoring and assembly generation should then consume that memo as the primary input.

That approach is attractive because it keeps the architecture smaller. But it only works if downstream prompts stop treating the raw failure dump as co-equal with the curated decision. If reflection says “keep the champion and only change grasp depth,” but the generator still sees the full discarded history and multiple competing retries, the prompt will remain too soft.

## 4. Design invariants to protect

Any future redesign should preserve these invariants:

- A discarded trial must not become the new baseline.
- The next iteration should start from the last kept state, not from the latest failed state.
- One iteration should usually change one main idea, not multiple interacting layers.
- The loop should prefer preserving a workable policy over chasing every remaining failure mode.
- If reflection is the curation layer, downstream synthesis must treat its output as binding guidance rather than one more opinion in the prompt.

## 5. Implementation Plan

This should be implemented as a prompt and control-flow change first, not as a new orchestration module.

1. Tighten the reflection prompt.
   - Change reflection from open-ended failure summarization into a structured decision memo.
   - Require fields such as `dominant_failure`, `kept_baseline`, `allowed_mutation`, `forbidden_changes`, `confidence`, and `evidence`.
   - Make the reflection step choose one primary hypothesis, not a grab bag of fixes.

2. Make downstream generators consume the memo as the primary input.
   - Update `skill_author` and `assembly_generator` prompts so the reflection memo is the first-class guidance.
   - Keep raw failure artifacts available only as supporting evidence.
   - Reduce the amount of raw discarded history shown alongside the curated decision.

3. Preserve the kept baseline as the hard anchor.
   - Keep the snapshot rollback behavior for mutable skill-library state.
   - Ensure the next iteration starts from the last promoted baseline, not from the latest discarded attempt.
   - Do not let discarded trials become the live state that the next prompt silently inherits.

4. Constrain mutation scope per iteration.
   - Enforce that one round should usually modify one main idea.
   - Prefer one skill family or one assembly change, not multiple interacting rewrites.
   - Reject prompts that try to rewrite grasp, lift, place, and retract all at once unless the evidence clearly justifies it.

5. Add tests for prompt behavior and rollback semantics.
   - Verify that reflection emits a structured decision memo.
   - Verify that `skill_author` and `assembly_generator` prompts include the memo prominently.
   - Verify that discarded snapshot state does not become the next live baseline.

6. Validate on a real run.
   - Compare the next run against a known baseline run.
   - Check that the code no longer drifts into growing stacks of fallback variants.
   - Check that the best policy is preserved longer instead of being overwritten by the latest failure.

## 6. Review order

Please review this note in order:

1. Section 1: whether the current-loop summary matches your understanding.
2. Section 2: whether the regression mechanism is the right root cause.
3. Section 3: whether the curation-layer redesign is the right systematic fix.
4. Section 4: whether the invariants are the right ones to lock in before implementation.
5. Section 5: whether this implementation plan is the right sequence of changes.
