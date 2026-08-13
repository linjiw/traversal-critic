# Temporal-Matched v1 Evaluation and Control Protocol — 2026-08-11

**Status:** prospective and outcome-blind. This protocol is frozen before any
`traversal_critic_temporal_matched_v1` checkpoint, validation response,
selection result, control score, or OOD score exists. It cannot alter the
historical v5 checkpoints, iteration-100 PPO critic, E1 factorial, policy
matrix, or temporal-alignment amendment.

## Detailed goal statement

Complete the current submission-grade G1--G6 adjudication without touching its
running jobs or inspecting unreleased policy outcomes, while preparing the
smallest corrected experiment that resolves the historical E5 confound. Train
one separately named critic under the already sealed historical 800-iteration
budget and the shared `traversal-temporal-v1` materializer; evaluate all eight
exports on the unchanged 432-item, 108-scene validation split through the
exact free-generation interface used by the corrected policy scorer; select
once using a fixed zero-invalid rule; and challenge the selected model with a
constrained-digit sensitivity, duration-only and terminal-frame-only probes,
onset-aligned prefixes, endpoint masking, and the frozen 42-clip robot OOD
corpus. Determine whether a matched critic has held-out ordinal signal and
whether endpoint appearance remains a viable explanation. Report a negative,
ambiguous, or positive result with the same prominence. Never use the
corrected generation to relabel, repair, or replace the historical policy
experiment.

The current critical path remains the terminal historical matrix, held-out
policy evaluation, registered uncertainty analysis, and C1--C5 adjudication.
Corrected training is gated on that path and on completed E1. Until both gates
close, only outcome-blind planning and verification may proceed.

## Fixed validation interface and checkpoint selection

- Namespace: `traversal_critic_temporal_matched_v1`.
- Candidates: exactly iterations 100, 200, ..., 800 from the separately
  audited corrected export tree. Every candidate must score all 432 original
  validation items from all 108 held-out scenes; balanced-manifest duplicates
  are never evaluation units.
- Primary interface: shared temporal materialization, explicit original-frame
  metadata, processor sampling disabled, `enable_thinking=False`, greedy
  generation, batch size one, and `max_new_tokens=8`. Parse the first
  standalone digit in `[1,5]`, exactly as the corrected policy scorer does.
- Eligibility: an export is eligible only if its export audit passes, all 432
  expected sample IDs occur exactly once, all ground truths parse, every model
  response parses, and every metric is finite. No missing or invalid response
  may be imputed, clipped, recovered from logits, or accepted from another
  interface.
- Selection: among eligible exports, maximize Pearson correlation; values
  equal within `1e-12` are broken by lower macro-MAE, then the earlier
  iteration. If no export is eligible, selection terminates as
  `failed_no_policy_interface_eligible_checkpoint`. The constrained interface
  cannot rescue or select a checkpoint.
- Report for every candidate: parse failures, output histogram, exact accuracy,
  within-one accuracy, Pearson, Spearman, Kendall tau-b, macro-MAE over ground-
  truth classes, and per-class recall. Invalid predictions count as misses for
  exact and within-one metrics and are absent from parsed-pair statistics; this
  distinction is reported even though selection requires zero invalids.

The selected checkpoint is evaluated a second time using first-generation-step
argmax restricted to the exact single-token digit IDs 1--5 under the same
generation prefix. Token-prefix and single-token contracts must pass before
scoring. This is a decoding sensitivity only. Both interfaces use the same
materialized pixels, source indices, timestamps, prompt, and selected model.

All confidence intervals use the already materialized E1 bootstrap artifact:
10,000 resamples of the 108 scene clusters with replacement using
`random.Random(20260811)`. Items from a scene always move together. The exact
stored draw matrix is reused, not regenerated after scores exist. Selection is
made on full-sample point estimates, never bootstrap optimism.

## Prospectively fixed shortcut controls

These controls are secondary and cannot choose the checkpoint.

### Duration-only and terminal-frame-only

Reuse the existing leakage-controlled specification on unique original media:
five-fold scene-grouped train-only ridge selection, fold-local standardization,
and untouched 108-scene validation evaluation. Report duration-only and
terminal-frame-only separately; duration plus terminal frame and fixed-seed
permuted labels remain descriptive sentinels. The historical post-hoc values
are not treated as corrected-model outcomes. The corrected report independently
replays the feature/media fingerprints and compares the selected critic with
the fixed probes using paired scene-bootstrap differences in Pearson and
macro-MAE.

### Onset-aligned prefix

Privileged state defines onsets; model outputs never do. On each validation
episode:

1. `adverse_onset` is the first 50 Hz frame with `fall`,
   `person_collision`, or `collision` true.
2. `clean_success_onset` is the first frame with `goal_dist_m <= 0.4`, but is
   eligible only when the entire episode has no adverse onset.
3. Episodes containing both success and an adverse event belong only to the
   adverse cohort. Episodes with neither event remain in the full validation
   analysis but not the onset contrast.
4. The prefix endpoint is exactly 0.5 seconds before the applicable onset.
   It includes source time zero through that endpoint, with the endpoint
   converted by floor in privileged-frame coordinates. A prefix must contain
   at least two frames selected by the shared materializer or is reported
   ineligible; it is never padded with future pixels.
5. Apply the same shared approximately-2-Hz, maximum-32-frame sampling rule to
   the real prefix interval. Preserve original source FPS and source-index
   coordinates, disable downstream sampling, and record every selected source
   index.

Score the original and prefix once through the selected free interface and
once through the constrained sensitivity interface. For adverse episodes,
report `prefix_score - original_score`; for clean-success episodes, report
`original_score - prefix_score`. Report means, medians, positive-direction
fractions, parse failures, per-event strata, and scene-clustered 95% intervals.
Because prefixes change duration and do not have independently recomputed
overall labels, these are evidence-sensitivity estimands, not accuracy claims.

### Endpoint-masked

Mask only after the shared full-clip materializer has selected its frames. For
an item with `n` selected frames, set `k=max(1, ceil(n/4))`. In the endpoint
variant replace each of the last `k` RGB frames with a copy of frame
`n-k-1`; source indices, timestamps, frame count, prompt, and all non-pixel
metadata remain unchanged. Items with fewer than two selected frames are
ineligible and reported.

The mandatory operator-placebo uses the same `k` and same forward-hold
operation on a contiguous interior block centered as closely as possible on
half of the selected-frame sequence, never touching the first or last frame.
If such a block cannot be formed without touching an endpoint, the item is
reported placebo-ineligible rather than assigned a different operator.

Score original, endpoint-held, and interior-held variants through both
interfaces. Report paired score-change histograms, mean absolute changes,
signed changes by label and event cohort, Pearson and macro-MAE for each
variant, and the paired excess endpoint effect
`abs(endpoint-original) - abs(interior-original)` with scene-bootstrap 95%
intervals. A freeze artifact is therefore measured rather than silently
attributed to endpoint removal. This diagnostic can establish sensitivity to
endpoint pixels; it cannot by itself establish semantic temporal reasoning.

## Fixed OOD follow-up

After selection, score the exact audited 42-clip robot corpus through the
selected free interface and constrained sensitivity. Require the existing
manifest, corpus audit, and clip hashes to replay. Report zero/number of parse
failures, the complete 1--5 output distribution, each regime mean, and whether
the single impaired-gait clip scores strictly below the mean of the ten clean-
walk clips. The corpus contains no collision/fall negatives, so even a positive
ordering supports only this narrow positive-motion control. It does not
establish broad real-world reward validity.

## Decision and claim boundary

- A clean matched-interface in-domain result requires an eligible selected
  checkpoint, Pearson at least 0.50, and a scene-bootstrap 95% lower bound
  above zero. All other metrics and all eight candidates remain visible.
- Evidence beyond the two fitted shortcuts requires positive lower 95% bounds
  for the selected critic minus both duration-only and terminal-frame-only
  Pearson. Failure keeps endpoint/duration prediction as an unresolved
  alternative; it is not repaired by constrained decoding.
- A decoding-interface diagnosis is descriptive unless the constrained-minus-
  free paired interval excludes zero; it never changes the policy interface or
  selected checkpoint.
- Onset and masking results govern wording about evidence timing and endpoint
  dependence. They cannot establish pre-fall balance perception because this
  v5 corpus was not constructed as scene-matched stable/fall pairs.
- Corrected OOD ordering is reported separately from historical v4/v5. It may
  motivate the already planned challenge/readout study but cannot retroactively
  promote historical G3/G4.
- No corrected checkpoint becomes a deployment or policy-reward result without
  a new, separately preregistered policy experiment. Historical iteration 100
  remains the sole critic for the active 3-by-3 matrix.

The most promising direction after this goal is therefore conditional. If the
corrected model clears the matched and shortcut gates, test it on a new
scene-matched clean/collision/natural-fall challenge and only then run a new
policy replication. If it retains endpoint dependence or fails OOD ordering,
prioritize the frozen-representation ordinal/calibration readout and negative-
support study already specified in
`next_readout_challenge_goal_2026-08-09.md`; another unconstrained SFT run would
not answer the observed failure.
