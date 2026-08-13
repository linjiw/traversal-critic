# Constrained-Decoding Interface Comparator Protocol — 2026-08-11

**Status:** prospectively frozen before any constrained-decoding result is
generated. This comparator is additive. It cannot change the selected v5
checkpoint, the free-decoding evidence of record, the active PPO matrix, its
scorer, or any G1–G6/C1–C5 decision rule.

## Question

How much held-out ranking is lost to free-form digit generation, separately
from the temporal-preprocessing mismatch found by the release-blocking E5
audit on 2026-08-11?

## Fixed factorial

Evaluate all eight fixed-budget v5 exports (iterations 100–800) on the same
432-item, 108-scene validation manifest under:

1. the historical file-path temporal route used by the recorded sweep;
2. an SFT-exact temporal route that replays the captured launch dataflow,
   including its observed four-frame second sampling and timestamp metadata;
3. free generation with `enable_thinking=False`, unchanged parser and token
   budget; and
4. first-score-token argmax restricted to the tokenizer encodings of the five
   responses `1`, `2`, `3`, `4`, and `5`.

This is a 2 temporal-route × 2 decoding-interface comparison. The historical
free/file-path cell must replay the existing per-item outputs before any new
cell is interpreted. The policy scorer's predecoded 2-fps/32-frame route is
reported as the frozen policy interface but is not allowed to masquerade as
either the historical validation route or the SFT-exact route.

## Frozen reporting and selection

- Report every checkpoint and every cell; do not select a route after seeing
  results.
- Primary interface contrast: within each temporal route, constrained minus
  free Pearson and Spearman on all 432 items. Unparseable free outputs remain
  misses and are separately counted; the constrained cell must emit exactly
  one of five scores for every item.
- Secondary metrics: Kendall tau-b, macro-MAE, per-class recall, exact
  accuracy, within-one accuracy, and output distribution.
- Descriptive constrained-checkpoint selector: zero invalid outputs first,
  then maximum Pearson; ties within `1e-12` break on lower macro-MAE and then
  earlier iteration. This selector is for the comparator only.
- The frozen iteration-100 checkpoint remains the sole critic used by the
  active policy experiment. No comparator result can retroactively replace it.
- Scene-clustered bootstrap intervals use the 108 validation scenes and a
  declared RNG seed/index artifact. They are descriptive because the same
  validation set participated in the historical checkpoint sweep.

## OOD extension

After the validation factorial is complete, score the frozen 42-clip OOD set
under free and constrained decoding while holding its existing predecoded
temporal route fixed. Report boundedness and regime ordering separately.
Constrained support guarantees a valid digit; it does not establish
calibration, ordering, or policy utility.

## Release conditions

The implementation must record tokenizer IDs for all five candidate responses,
fail if any candidate is multi-token after the exact generation prefix, bind
checkpoint/processor/source hashes, save per-item logits and selected scores,
and pass a historical-cell replay. The run must be supervised and must not
start while its GPU memory or I/O demand could interfere with the frozen PPO
matrix. Any change to this protocol requires a dated amendment written before
the affected result is observed.
