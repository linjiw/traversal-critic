# Temporal-Matched v1 Prelaunch Seal — 2026-08-11

**Status:** code-complete and tensor-audited before any corrected training or
evaluation result exists. This is a new-generation interface seal. It does not
repair, relabel, rescore, or replace historical v5 evidence or the active PPO
critic.

## Fixed interface

The corrected namespace is `traversal_critic_temporal_matched_v1`. Its shared
materializer is `traversal-temporal-v1` and is the only component allowed to
sample video frames.

1. Read the decoder's positive original frame count and original average FPS;
   missing or nonpositive metadata fails closed.
2. Select original source indices at approximately 2 FPS using
   `stride=max(1, round(source_fps / 2))`.
3. If more than 32 candidates remain, deterministically select 32 positions
   across the entire candidate span, including its first and last entries.
4. Return the PIL frames, strictly increasing original source indices,
   original source FPS, original frame count, and explicit metadata.
5. SFT, validation, and policy scoring pass the preselected frames with
   `do_sample_frames=False`. Timestamp metadata uses the original indices and
   original FPS; sampled frames are never renumbered to `0..N-1`.

The historical v5 dataflow, evaluator, scorer, selection, and active matrix
were not edited. The corrected SFT recipe has a separate Hydra experiment
name, the corrected evaluator has a separate module, and the corrected policy
daemon has a separate executable. No corrected SFT or model-scoring job has
been launched.

## Executed equality audit

The replayable audit independently traversed the corrected SFT bytes route,
validation path route, and policy path route through the iteration-100 Edge
processor. It covered the same five preregistered E5 strata: short, capped,
successful, falling, and timeout. Every stratum matched exactly for:

- `input_ids`;
- `pixel_values_videos`;
- `video_grid_thw`;
- source indices, source FPS, frame count, and explicit metadata; and
- timestamp tokens.

The result is
`autoresearch/run-260811-1906/temporal_matched_route_audit.json`, SHA-256
`5e9d6fc9e99e708fc0064526fa628a3807c0eb866a52a784a7430f1cf931fe1e`.
It binds the processor files, historical E5 sample authority, audit source,
and the four executed route sources. Independent current-byte replay passes.

## Claim and launch boundary

This seal authorizes only a separately named corrected-generation prelaunch
after its source and evidence bytes pass canonical verification and the shared
GPU scheduler says it cannot interfere with the historical matrix. It is not a
positive historical E5 result: `historical_v5_e5_remains_negative=true`, the
canonical v5 amendment remains unchanged, and historical C1 remains blocked
as a clean matched-interface claim.

The canonical authority is the direct one-link record
`<DATA>/traversal-critic/data/critic_temporal_matched_v1_prelaunch_seal.json`.
It binds this protocol, the five-stratum route audit, iteration-100 processor
files and selection, repository and executed Cosmos sources, the frozen E1
protocol, and the independently replayed negative historical E5 amendment.
Capture fails if any of these separately named corrected roots already exists:

- `cosmos_outputs/train/cosmos3/vlm_traversal_critic_sft/traversal_critic_temporal_matched_v1`;
- `eval_temporal_matched_v1`; or
- `score_queue_temporal_matched_v1`.

From the repository root, capture once and verify before launch with:

```bash
$SONIC_ROOT/.venv_sim/bin/python scripts/record_temporal_matched_prelaunch_seal.py capture \
  --repo "$PWD" \
  --data-root <DATA>/traversal-critic/data \
  --cosmos-root <REPO>/cosmos-framework \
  --out <DATA>/traversal-critic/data/critic_temporal_matched_v1_prelaunch_seal.json

$SONIC_ROOT/.venv_sim/bin/python scripts/record_temporal_matched_prelaunch_seal.py verify \
  --record <DATA>/traversal-critic/data/critic_temporal_matched_v1_prelaunch_seal.json \
  --repo "$PWD" \
  --data-root <DATA>/traversal-critic/data \
  --cosmos-root <REPO>/cosmos-framework \
  --require-prelaunch-absence
```

After corrected outputs legitimately exist, omit
`--require-prelaunch-absence`; byte, ownership, source-copy, nested-evidence,
sampler, tensor, timestamp, and historical-boundary replay remain mandatory.

Before corrected training, preserve this sampler and budget. Any semantic
change requires a new materializer version, dated amendment, fresh five-stratum
audit, and new generation namespace. After matrix-safe capacity exists, the
already frozen E1 all-export factorial remains the next diagnostic on existing
v5 checkpoints; corrected SFT follows under its own table and artifact root.
