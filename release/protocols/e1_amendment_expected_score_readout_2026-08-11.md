# E1 Amendment — Expected-Score Decoding Readout — 2026-08-11

**Status:** written 2026-08-11, under the frozen protocol's amendment clause
("any change to this protocol requires a dated amendment written before the
affected result is observed"), and before any comparator cell was generated.
At writing time the canonical E1 output root contains only the sealed
execution plan, its independent plan audit, and the bootstrap index artifact;
`e1_result.json` does not exist, no `cells/` item has been written, and the
latest supervised handoffs record `e1_outputs_observed=false` and
`e1_result_present=false`. This amendment is additive. It changes no existing
cell definition, no generation pass, no selection rule, and cannot affect the
frozen iteration-100 matrix critic.

## What this amendment adds

A third decoding readout, **expected score**, computed post-hoc from
artifacts the protocol already requires. The frozen release conditions
mandate recording the tokenizer IDs of the five candidate responses `1`–`5`,
failing if any candidate is multi-token after the exact generation prefix,
and saving per-item logits for those five candidates at the first score-token
position. The expected-score readout consumes exactly those saved per-item
logits:

- At the same first-score-token position used by the constrained argmax
  readout, take the softmax over exactly the five recorded candidate token
  IDs — no other vocabulary entries participate.
- Compute E[s] = Σ_{s=1..5} s · p(s), a continuous value in [1, 5].

Because it is derived from logits the protocol already saves for every newly
generated item, the readout requires **zero additional inference**, no change
to the sealed execution plan or supervisor, and applies to every cell of both
temporal routes (historical file-path and SFT-exact) and to the OOD
extension.

## Revised factorial shape

The comparator therefore becomes a 2 temporal-route × 3 decoding-readout
design: free generation, first-score-token argmax, and expected score.
All three readouts are reported for all eight fixed-budget v5 exports
(iterations 100–800), every cell, with no post-hoc route or readout
selection. The 32 sealed generation cells are unchanged; expected-score
cells are reporting cells derived from the per-item logits of the
corresponding constrained generation pass.

## Reporting for expected-score cells

- Primary: Pearson, Spearman, and Kendall tau-b against the validation
  labels, computed on the continuous E[s] values over all 432 items.
- Secondary: because the readout is continuous, exact accuracy, within-one
  accuracy, and per-class recall are computed on the rounded expected score
  (round-half-to-even to the nearest integer in 1–5) and reported as
  secondary only.
- Output distribution: reported as a histogram of the continuous values,
  not of the rounded scores.
- Scene-clustered bootstrap intervals reuse the frozen 108-scene RNG
  seed/index artifact already declared for the other cells and remain
  descriptive for the same reason.

## No selection power

The expected-score readout has **no selection power**. The protocol's
descriptive constrained-checkpoint selector (zero invalid outputs first,
then maximum Pearson, ties within `1e-12` on lower macro-MAE then earlier
iteration) is unchanged and continues to operate on the constrained argmax
cells only. No expected-score result can replace the frozen iteration-100
matrix critic, alter the free-decoding evidence of record, or feed back into
the active PPO matrix in any form.

## Motivation

This is the Q-Align-style readout (Wu et al., 2023, "Q-Align: Teaching LMMs
for visual scoring via discrete text-defined levels", arXiv:2312.17090),
the strongest known decoding interface for level-token scoring LMMs. Without
it, the comparator could conclude "constrained argmax ≈ free generation"
while the standard-practice readout for this model class was never measured
— the most predictable reviewer objection to the interface contrast. The
expected score also yields a continuous value whose ordering is invariant to
rounding ties, which sharpens the probe-vs-critic readout comparison: the
linear probe emits continuous scores, and until now every fine-tuned readout
was quantized to five levels, conflating readout quality with quantization
loss.

## OOD extension

The OOD extension inherits the third readout unchanged: the frozen 42-clip
OOD set is scored under all three readouts while holding its existing
predecoded temporal route fixed, and expected-score boundedness and regime
ordering are reported descriptively. Support-restricted softmax guarantees a
value in [1, 5]; it does not establish calibration, ordering, or policy
utility.
