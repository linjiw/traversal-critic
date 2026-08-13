# E5 Temporal-Interface Amendment — 2026-08-11

**Status: canonical E5 amendment of record.** Written 2026-08-11, before any
critic-code change and before any E1 comparator cell was generated (only the
frozen E1 plan and preregistration artifacts existed at writing time). This
document records the E5 verdict and its claim consequences; it preserves the
historical numbers and interfaces rather than overwriting them. It is
additive: it cannot improve a metric, select a different checkpoint, modify
the frozen PPO matrix, or substitute any corrected model into a historical
table. It complements the conservative claim-restriction rules in
`docs/reviews/temporal_alignment_amendment_2026-08-11.md` and the frozen
comparator in
`docs/reviews/constrained_decoding_interface_comparator_protocol_2026-08-11.md`.

## 1. What E5 preregistered, and the verdict

E5 was the release-blocking preregistered audit requiring that the clean-v5
SFT training path, the historical validation path, and the policy-scorer path
present the **same temporal input interface** — identical frame indices,
timestamps, metadata, prompt tokens, and resulting pixel tensors — checked on
deterministic short, capped, successful, falling, and timeout strata.

**Verdict: FAIL.** The three paths are three different temporal interfaces:

1. **SFT (captured launch):** the training dataflow first pre-sampled each
   MP4 toward 2 fps and at most 32 frames, and the captured Edge processor
   then sampled that already sampled list a second time. Because no explicit
   video metadata reached the processor, it defaulted to 24 fps and reduced
   every audited clip to **four second-stage frames**.
2. **Historical validation:** the file-path route, passing MP4 pathnames and
   selecting **4–48 frames at 2 fps**.
3. **Policy scorer:** predecoded **5–32 frames** with explicit metadata.

All five audited strata (short/capped/success/fall/timeout) mismatched on
source indices, timestamps, prompt tokens, and pixel tensors: none of the
five examples had equal training-versus-validation source indices,
timestamps, prompt tokens, or pixel tensors, and training also did not equal
the policy scorer. This is not a source-reading inference. The audit executed
the real processor, tensors, and tokenizer; independently reproduced the
evaluator's actual source indices; and byte-verified the four relevant
preprocessing modules against the immutable SFT process capture
(`semantic_alignment_passed=false` in the saved audit record).

## 2. Companion shortcut diagnostic

Reproduced verbatim from the execution report
(`docs/reviews/claude_fable_plan_execution_report_2026-08-11.md`), which used
only unique original training clips, train-only five-fold scene-grouped
lambda selection, and the unchanged 108-scene validation split:

| Readout | Validation Pearson | Spearman | Rounded Pearson |
|---|---:|---:|---:|
| Duration only | 0.2681 | 0.4397 | 0.2758 |
| Final 8×8 RGB frame only | 0.6661 | 0.7003 | — |
| Duration + final 8×8 RGB frame | **0.6827** | **0.7820** | 0.6665 |
| Fixed-seed permuted train labels | 0.0402 | 0.0756 | — |
| Selected v5 critic reference | 0.5646 | not recorded | digit output |

Per the execution report's caveat, which this amendment adopts: duration
alone recovers about 47.5% of the selected critic's Pearson, and duration
plus terminal appearance reaches 120.9% of it, but **those ratios are
descriptive comparisons, not variance-explained estimates**. The
low-resolution endpoint probe beating the selected critic does **not** show
that the VLM uses this shortcut; it shows that the label is highly
predictable from terminal appearance even across scene-disjoint validation,
which makes endpoint shortcut controls a necessary causal diagnostic.

## 3. Consequences of record

Historical numbers are preserved, not overwritten. Henceforth:

1. **Pearson 0.5646 is historical mismatched-interface evidence.** It is the
   result of a file-path inference interface applied to a model trained
   through a different four-frame interface, and is no longer reported as a
   clean matched-interface train/validation estimate.
2. **The draft's prior claim that SFT consumed up to 32 frames is factually
   false for the captured launch.** The byte-bound launch path consumed four
   second-stage frames per clip in every audited stratum.
3. **C1 claim-slot wording is MARKED as requiring revision by the
   finalizer.** This amendment flags the C1 slot; it does not and must not
   edit the slot itself, which is materialized only by the finalizer after
   the required evidence gates close.
4. **G5/G6 remain valid for the exact frozen historical system** — the
   5–32-frame scoring interface over the four-frame-trained checkpoint —
   but, whatever their result, they cannot validate the originally described
   32-frame critic, because the policy scorer is a third temporal interface.
5. **E3-style replication of the unchanged recipe would replicate the bug.**
   Such replications may estimate historical-run variance but do not repair
   the intended critic. Any corrected generation must be separately named
   and never mixed into v5 tables.
6. **The frozen 3-seed × 3-arm PPO matrix continues untouched.** Its
   orchestration, scorer daemon, queue, logs, and checkpoints are not
   modified, stopped, or rescheduled by this amendment or by any work it
   triggers.

## 4. Saved evidence

- Temporal audit:
  `autoresearch/run-260811-1753/temporal_preprocessing_audit.json`
  SHA-256 `39f59319506dbc1b7cc3ce0a007efde32125c52104ced8dc4cf84dc796537847`
- Shortcut results:
  `autoresearch/run-260811-1753/temporal_shortcut_results.json`
  SHA-256 `02aa81fd26857744b47e49c769e256354909d27e8d380f1ab29b34073f893ca2`
- Shortcut feature cache:
  `autoresearch/run-260811-1753/temporal_shortcut_features.npz`
  SHA-256 `f8a9b5f81c94bca5bd240e0af8a31cf6692aab8808fcd36bb83f7636144fb852`
- E1 frozen protocol:
  `docs/reviews/constrained_decoding_interface_comparator_protocol_2026-08-11.md`
  SHA-256 `0d621c48984f5d28017a76cad640e6d99d305b4df296f9de25484b50a9fa49d6`
- Outcome-blind live checks:
  `autoresearch/run-260811-1753/live_orchestration_status.txt` and
  `live_queue_binding_status.txt`.
