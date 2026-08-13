# Shortcut-Battery Extension Protocol — 2026-08-11

**Status:** prospectively frozen on 2026-08-11 before any extended-battery row
is computed. This battery is additive and diagnostic. It cannot change the
selected v5 checkpoint, the active PPO matrix, its scorer, or any G1–G6/C1–C5
decision rule. The existing duration/terminal 8×8 ridge row of 2026-08-11
(`autoresearch/run-260811-1753/temporal_shortcut_results.json`, SHA-256
`02aa81fd26857744b47e49c769e256354909d27e8d380f1ab29b34073f893ca2`) is hereby
classified as a **post-hoc descriptive pilot**. It may not be upgraded to
preregistered status, cited as a confirmatory battery result, or partially
merged into battery tables; the frozen battery recomputes its rows under this
protocol and reports the recomputed values alongside the pilot for
transparency.

## Question

How much of the frozen labeler-v3 traversal score is recoverable, on the
unchanged scene-disjoint validation split, from trivially shallow features —
duration, single frames, temporally truncated or masked frame sets — and how
much predictability exists *before* the outcome unfolds versus *only at* the
endpoint? This bounds what any learned readout must beat to claim temporal
content, and quantifies the temporal increment of endpoint appearance. It
answers the readout challenge expansion required by the 2026-08-11 execution
report (item 6) and plan item P1c.

## Fixed data, labels, and split

- Training set: the same 1,568 unique original training clips used by the
  pilot. Balanced-manifest duplicates are excluded as independent samples,
  exactly as in the pilot; each unique original medium appears once.
- Validation set: the unchanged 432-clip, 108-scene held-out validation split.
  No item may be added, dropped, or re-split.
- Labels: the frozen labeler-v3 rubric scores (integers 1–5), identical to the
  scores used for v5 SFT and the pilot. No relabeling, recalibration, or label
  smoothing.
- Media identity is verified against the pilot's recorded fingerprints in the
  pilot `_meta` (`media_fingerprints`): train
  `77fa2c4a4eb9544159a41f457b8b07a8b18c92f1933bab5fd2a8ac49bfbbb698`, val
  `eaa47ef334a285bcc379dc396bdc1f8ace4c7c319337a7c3670f4686b102bba4`; a
  mismatch aborts the run.

## Fixed feature and estimator specification

- Frame basis: each clip's MP4 is fully decoded to RGB at native resolution
  and native fps; frames are indexed `0 … N-1` in source order, where `N` is
  the clip's total decoded frame count.
- Frame feature: each selected frame is resized to 8×8 pixels with **bilinear
  interpolation** (the pilot's resize), then flattened to 192 dimensions
  (8 × 8 × 3). No other pixel preprocessing.
- Duration features: exactly the pilot's four —
  `duration/24`, `(duration/24)^2`, `(duration/24)^3`, and the cap indicator
  `duration >= 23.99 s`.
- Estimator: ridge regression with fold-local feature standardization and an
  unpenalized intercept, exactly as in the pilot.
- Lambda grid (fixed): `{1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}`.
- Lambda selection: train-only, five-fold, scene-grouped cross-validation on
  the 1,568 training clips. Fold assignment: the 392 unique training scene IDs
  are sorted, shuffled once with `random.Random(20260811)`, and dealt
  round-robin into five folds. Lambda is chosen per row by out-of-fold Pearson
  on the pooled train out-of-fold predictions; the selected lambda is then
  refit on all 1,568 training clips and evaluated once on the 432 validation
  clips. Validation data never touches selection.
- Master RNG seed: `20260811`. It seeds fold assignment, the shuffled-label
  permutation, the random-single-frame draws, the frame-order permutations,
  and the bootstrap draw matrix. A seed/index artifact recording the fold
  assignment, every per-clip drawn or permuted index, the label permutation,
  and the full bootstrap draw matrix is saved next to the results and hashed.

## Preregistered rows

Every row below is computed and reported regardless of outcome. No row may be
added, dropped, or redefined after any extended-row result exists, except by
dated amendment written before the affected result is observed.

1. **duration-only** — the four duration features.
2. **first-frame-only** — frame `0` (192 dims).
3. **terminal-frame-only** — frame `N-1` (192 dims); recomputes the pilot row
   under this protocol.
4. **first+terminal** — frames `0` and `N-1` concatenated (384 dims).
5. **duration+terminal** — duration features plus frame `N-1`; recomputes the
   pilot row under this protocol.
6. **duration+first+terminal** — duration features plus rows 2 and 4's frames
   (388 dims).
7. **random-single-frame** — one frame per clip drawn uniformly from
   `0 … N-1` with the master seed (192 dims); required by plan item P1c as
   the position-agnostic single-frame baseline. Drawn indices are saved in the
   seed/index artifact.
8. **onset-aligned prefix, 25%** — the prefix is frames
   `0 … m-1` with `m = max(1, floor(0.25 · N))`: everything from clip start up
   to one quarter of the decoded frames, testing what is predictable before
   the outcome unfolds. Features: `K = 4` frames at the equally spaced indices
   `round(j · (m-1) / 3)` for `j = 0, 1, 2, 3` within the prefix (indices may
   repeat when `m < 4`), concatenated in temporal order (768 dims). No frame
   at or beyond index `m` may contribute.
9. **onset-aligned prefix, 50%** — identical rule with
   `m = max(1, floor(0.50 · N))`.
10. **endpoint-masked, last 10% removed** — drop the final
    `k = max(1, ceil(0.10 · N))` frames; the terminal feature is then taken
    from the last surviving frame, index `N-1-k` (192 dims). Tests how much
    of the terminal-frame signal dies with the endpoint.
11. **endpoint-masked, last 25% removed** — identical rule with
    `k = max(1, ceil(0.25 · N))`.
12. **temporally shuffled frame order** — for every multi-frame row (rows 4,
    6, 8, 9), a companion row in which the selected frames' temporal order is
    permuted per clip before concatenation, using a per-clip permutation drawn
    from the master seed and the clip ID (saved in the seed/index artifact).
    Feature dimensionality and pixel content are unchanged; only slot order
    moves. Tests whether any multi-frame gain depends on temporal order at
    all. Four companion rows.
13. **shuffled-label negative control** — training labels permuted once with
    the master seed, fit on the richest feature set (row 6), evaluated on the
    unpermuted validation labels. Expected near zero; a large value voids the
    run pending investigation.

Total: 17 rows (rows 1–11, four order-shuffled companions, one shuffled-label
control), plus the pilot values reprinted for reference and clearly labeled
`post-hoc pilot`.

## Frozen reporting rules

- Per row: validation Pearson, Spearman, and rounded Pearson (predictions
  rounded to the nearest integer and clipped to `[1, 5]` before correlation),
  plus the selected lambda and the out-of-fold selection Pearson.
- Intervals: scene-clustered bootstrap over the 108 validation scenes —
  10,000 resamples of scenes with replacement, items from a scene always
  moving together, drawn once with `random.Random(20260811)` and saved as an
  index artifact before any interval is read. All intervals are labeled
  **descriptive**, because this validation set participated in the historical
  checkpoint sweep.
- Every row is reported against **both** fixed reference points, as
  descriptive ratios of Pearson values and never as variance explained:
  1. the selected v5 critic's historical record, Pearson `0.5646`, explicitly
     labeled a *mismatched-interface historical record* per the E5 audit; and
  2. the frozen-tower probe result of record, Pearson `0.7053`.
- No row, ratio, or interval may be omitted, reordered by outcome, or moved to
  an appendix while others are promoted; the battery is published as one
  table.

## Frozen interpretation rules

- A shortcut row beating a learned readout shows that the label is recoverable
  from that feature set on scene-disjoint validation. It does **not** show
  that the model uses that shortcut; usage claims require the separately
  frozen masking/prefix interventions on the model itself.
- The temporal increment of any learned readout is defined as
  (full-input score − the corresponding endpoint-masked or terminal-only
  ceiling from this battery), reported descriptively with its scene-clustered
  interval; it is not a causal decomposition.
- Prefix rows bound *pre-outcome* predictability; endpoint-masked rows bound
  *endpoint-dependent* predictability. Neither licenses claims about balance
  perception or fall anticipation.
- No row in this battery licenses any claim about the separately named
  corrected-generation critic; corrected-generation comparisons occur only
  under the temporal-matched v1 protocol.

## Release conditions

- The battery is **CPU-only**. It must not allocate GPU memory, GPU I/O, or
  otherwise contend with the running frozen PPO matrix; it must not read any
  unreleased matrix outcome.
- Results are saved under `autoresearch/` in a dated run directory, with a
  `_meta` block recording labeler version, dynamics, scene range, data
  authorities, script SHA-256, feature-cache SHA-256, media fingerprints, and
  the seed/index artifact SHA-256. The accompanying run report lists SHA-256
  values for every saved artifact.
- The implementation replays the pilot's media fingerprints before fitting and
  aborts on mismatch.
- Any change to this protocol requires a dated amendment written before the
  affected result is observed; an amendment cannot retroactively reinterpret a
  row already computed.
