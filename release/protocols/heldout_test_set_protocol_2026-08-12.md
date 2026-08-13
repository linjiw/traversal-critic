# Held-Out Fresh-Scene Test-Set Protocol — 2026-08-12

**Status:** prospectively frozen on 2026-08-12, **before any test-scene
generation exists**. No clip, label, feature, seal, or score governed by this
protocol has been produced at writing time. This protocol is additive. It
cannot alter the running frozen 3-seed × 3-arm PPO matrix, its scorer, queue,
or schedule; any G1–G6 or C1–C5 decision rule; the selected iteration-100 v5
critic; or any existing frozen protocol (the E1 comparator and its
expected-score amendment, the shortcut-battery extension, the temporal-matched
v1 protocol and prelaunch seal, the matrix interpretation note). Execution is
**deferred**: generation and scoring are compute-heavy and may not begin while
they could contend with the frozen matrix (release conditions below). Writing
this document consumed no simulation, GPU, or generation compute.

## Question

Convert the project's key in-domain comparisons from descriptive to
confirmatory. Every in-domain statistic produced so far — and every
preregistered upcoming one (the E1 factorial, the shortcut-battery extension,
the temporal-matched v1 selection) — is evaluated on the same 432-clip,
108-scene validation split, which participated in the historical checkpoint
sweep and is therefore permanently descriptive under this project's own frozen
reporting rules (`shortcut_battery_extension_protocol_2026-08-11.md`,
`constrained_decoding_interface_comparator_protocol_2026-08-11.md`). This
protocol preregisters a scene-fresh test corpus that **no model training,
checkpoint selection, lambda selection, calibration, or protocol tuning has
ever touched**, and a frozen roster of readouts each evaluated on it **exactly
once**. The first and only use of this corpus is the sealed single-shot batch
defined below; that is what makes its intervals confirmatory.

## Consumed scene-seed ranges (exclusion inventory)

The new range must be provably disjoint from every scene index the project has
ever consumed or reserved. The complete inventory, each entry bound to its
primary authority:

| # | Range | Convention | Consumer | Primary authority |
|---|---|---|---|---|
| 1 | `[1000,1500)` | half-open at the CLI (`generate_physics_rollouts.py` line 209–211, `range(lo, hi)`); normalized to inclusive `[1000,1499]` in `_meta` (`audit_physics_rollouts.py` lines 502–503) | critic-v5 corpus: 2,000 rollouts, four per scene (`scene_id` `physhouse_1000`–`physhouse_1499`, `generate_physics_rollouts.py` line 274); its scene-hash subset is the 392-scene/1,568-clip train and 108-scene/432-clip validation split | launch configs `ops/gen_phys_v5_supervisor.sh` lines 38–39 and 89 (`--scenes 1000-1500 --episodes_per_scene 4 --schedule_scene_lo 1000`) and the five frozen tail shards in `ops/gen_phys_v5_tail_shard.sh` lines 16–22 (half-open sub-shards `1244-1279 … 1473-1500`, no new indices); pipeline expectation `ops/v5_pipeline.sh` line 28; launch-derivation contract `1000 <= lo < hi <= 1500` + `global_schedule_scene_lo=1000` enforced in `scripts/audit_physics_rollouts.py` lines 253–280; gate binding `scripts/audit_research_gates.py` lines 1363/1396 (`scene_range == [1000, 1499]`); `paper/draft.md` §3.2 |
| 2 | `[0,200]` | inclusive, 201 scenes (`sonic_physics_nav_env.py` line 211: `scene_rng.randint(*self.scene_range)`) | PPO training scenes for all nine matrix runs | launch `ops/ppo_supervisor.sh` lines 99–101 (`--scene_lo 0 --scene_hi 200`), entered by every arm/seed via `ops/critic_matrix_lane.sh`/`ops/control_matrix_lane.sh`; defaults `sim_rl/train_nav_policy_ppo.py` lines 278–279; frozen audit binding `scripts/audit_ppo_matrix.py` lines 291–292 and 3636–3637 ("inclusive endpoints (201 possible training scenes)"); goal-doc policy protocol §Training |
| 3 | `[400,440]` | inclusive, 41 scenes | frozen held-out policy evaluation (100 episodes per policy, evaluator seed 123, 18 registered endpoints) and the scripted-route baseline | launch `ops/policy_eval_lane.sh` line 283 (`--scene_lo 400 --scene_hi 440 --episodes 100`); `sim_rl/eval_nav_policy.py` lines 1088–1089 and 1275–1276 ("inclusive_endpoints (environment uses randint(lo, hi))"); `sim_rl/scripted_route_baseline.py` lines 100–101; auditors `scripts/audit_policy_eval.py`, `scripts/audit_research_gates.py` lines 2572–2576 (`"train": [0, 200], "evaluation": [400, 440], "semantics": "inclusive"`); goal-doc §Held-out evaluation |
| 4 | `[600,619]` | fixed inclusive list; **600–607 completed, 608 attempted and terminally burned, 609–619 reserved but never generated** | force-induced causal-balance mechanism corpus (terminated max-20 protocol, retained as a terminal negative; never training data) | `sim_rl/generate_balance_causal_set.py` line 37 (`SCENES = tuple(range(600, 620))`) and lines 554–557 (`_meta` purpose); `scripts/audit_causal_feasibility_failure.py` lines 21–27 (`COMPLETED_SCENES = tuple(range(600, 608))`, `FAILED_SCENE = 608`); amendment record `scripts/record_causal_feasibility_amendment.py` (retains `"scenes": [600, 619]`); goal-doc §Frozen causal balance follow-up. The full reservation `[600,619]` is excluded here regardless of partial materialization |
| 5 | `[10000,16600)` | half-open, 6,600 scenes; **plan-level reservation only** — a repository-wide search finds zero generation footprint | reserved exclusively for the future readout-challenge study (pilot/development/test blocks already assigned per seed and target) | `next_readout_challenge_goal_2026-08-09.md` §Data contract lines 311–325 ("Reserve the half-open scene-index block `[10000,16600)` exclusively for this study", with the full block-assignment table). Excluded here as reserved even though unconsumed |

No other pipeline consumes simulator scene indices: the temporal-matched v1
routes/controls, the shortcut pilot and battery, and the E1 factorial all
re-derive from the same `[1000,1499]` corpus (each records
`"scene_range": [1000, 1499]` in its `_meta`:
`scripts/audit_temporal_matched_routes.py` line 209,
`scripts/evaluate_temporal_shortcuts.py` line 322,
`scripts/run_shortcut_battery_extension.py` line 672,
`scripts/audit_e1_factorial.py`); the 42-clip OOD corpus and the probe control
record `"scene_range": None` (real Unitree G1 footage and a separate sim2sim
demonstration domain; `sim_rl/eval_critic_on_real_videos.py` line 193,
`sim_rl/compare_ood_scores.py` line 98, `scripts/audit_probe_control.py`
line 792). The generator's argparse default `--scenes 0-800`
(`generate_physics_rollouts.py` lines 25/185) is a never-launched default,
not a consumption, and is not treated as consumed.

Scene identity is constructed from the index by the environment rule
`SceneSpec(scene_index, random.Random(1000 + scene_index))`
(`sim_rl/traversal_nav_env.py`, line 198). Both the index blocks above **and**
their derived per-scene RNG seeds (2000–2499, 1000–1200, 1400–1440,
1600–1619, 11000–17599) are treated as consumed.

The validation membership of the `[1000,1499]` corpus is fixed by the
deterministic scene-hash rule of
`cosmos_overlay/cosmos_framework/scripts/reasoner/prepare_traversal_critic_dataset.py`
lines 361–365 (`bucket = int.from_bytes(sha256(scene_id)[:4], "big") / 2**32`;
`val` iff `bucket < val_scene_frac`, with `val_scene_frac = 0.2` pinned in
`scripts/materialize_critic_dataset.py` lines 95–96 and `ops/v5_pipeline.sh`
line 50; independently re-implemented in `scripts/audit_critic_dataset.py`
lines 23–27). Executing that rule over `physhouse_1000`–`physhouse_1499`
reproduces exactly the 108 validation / 392 training scenes of record. Every
scene of the new corpus is outside that universe entirely; no hash rule
partitions it.

## New scene range (fixed)

**Test scenes: the half-open block `[20000,20125)` — 125 scenes, four
rollouts per scene, exactly 500 clips.** Reserved from this document forward
exclusively for this protocol.

- Disjoint from every consumed or reserved index block above, with margin.
- Its derived `SceneSpec` RNG seeds, 21000–21124 (21000–21299 including the
  declared extension cap below), are disjoint from every derived seed of
  every consumed block, so no procedural scene can coincide even in the seed
  namespace.
- Per the challenge-study precedent, disjointness is verified from **recorded
  scene identities and scene hashes in the generated corpus**, not from
  numeric ranges alone: the corpus audit must list every `scene_id`
  (`physhouse_20000`–`physhouse_20124`) and confirm zero intersection with
  the scene identities recorded in the canonical critic-v5 rollout audit, the
  matrix training/evaluation configurations, and the causal-corpus manifest.

## Fixed generation specification

Same pipeline, same knobs, new scenes. Every parameter below is the audited
critic-v5 corpus value unless explicitly stated.

- **Generator:** `sim_rl/generate_physics_rollouts.py`, run from `sim_rl/`
  under `.venv_sim`, byte-identical to the sources bound by the canonical
  rollout audit of record
  (`4130c21ceee0dc84bba9534a18062a89fb2e9d4d4649deb50257cb55c38e36cc`); the
  run report records the executed source SHA-256s and any deviation fails
  closed.
- **Invocation contract:**
  `--scenes 20000-20125 --episodes_per_scene 4 --max_seconds 24.0
  --seed 20260812 --schedule_scene_lo 20000`, output root declared below.
  The v5 corpus used the generator's default master seed 0
  (`generate_physics_rollouts.py` line ~201, never overridden by
  `ops/gen_phys_v5_supervisor.sh` or the tail shards); the new master seed
  `20260812` is fresh by intention: the band and environment-seed schedule
  is drawn from the same frozen per-episode RNG rule (`episode_parameters`:
  `master.choice([1,2,3,4,5])`, `master.randint(0, 1 << 30)`) but shares no
  drawn value stream with the v5 corpus. Shard launches, if used, must
  declare half-open subsets of `[20000,20125)` and preserve
  `--schedule_scene_lo 20000`, mirroring the v5 shard contract
  (`ops/gen_phys_v5_tail_shard.sh`). Episode IDs
  (`phys_s20000_e0_b?` …) and scene IDs (`physhouse_20000` …) cannot collide
  with any existing ID: the `%04d` format pads without truncating, so
  five-digit indices remain distinct from every four-digit v5 identity.
- **Launch provenance:** every launch records a `source_launches.json` entry
  via `scripts/record_source_launch.py` and a source archive via
  `scripts/capture_source_provenance.py`, exactly as the v5 chain did
  (`generation_source_provenance` tree on the data root), so the corpus
  auditor can bind launch derivations.
- **Dynamics and rendering:** full SONIC contact physics
  (`dynamics:"physics"`), `PhysicsTraversalNavEnv`
  (`sim_rl/sonic_physics_nav_env.py`; MuJoCo scene
  `scene_29dof_mujoco.xml`; `CTRL_HZ = 50`, `SIM_DT = 0.005`,
  `DECIMATION = 4` per `sim_rl/sonic_physics_env.py` lines 44–46),
  third-person chase camera, 256×256 at 25 fps (`record_fps=25`), episodes
  capped at 24 s, nonreactive mocap people with collision geometry
  disabled, 50 Hz privileged sidecars — unchanged from the v5 corpus
  (`paper/draft.md` §3.2).
- **Rollout policy mix:** the same banded scripted drivers (`BandDriver`,
  `generate_physics_rollouts.py` lines 53–146; quality aims 1–5 drawn
  uniformly per episode from the master RNG schedule, expected ≈100 episodes
  per band over 500). **No learned policy, matrix checkpoint, or critic
  output participates in generation**, exactly as in the v5 corpus; the only
  frozen model weights in the loop are the SONIC whole-body controller ONNX
  triple (planner/encoder/decoder, `sim_rl/sonic_physics_env.py` lines
  39–42), whose hashes enter the rollout audit's source closure. The
  privileged labeler — not the driver band — assigns every label
  (`audit_physics_rollouts.py` treats `band_aim` as diagnostic only).
- **Labeler:** rubric version 3, recomputed by the same materializer,
  `cosmos_overlay/cosmos_framework/scripts/reasoner/prepare_traversal_critic_dataset.py`
  (`LABELER_VERSION = 3`, line 145, with the version ledger at lines
  137–144; the overlay copy is the repository authority and the installed
  cosmos-framework copy must byte-match it), byte-identical to the sources
  bound by the canonical dataset audit of record
  (`837cd7a408b90b0a72ecb29a672d9623f3d6d760edf8ad93d3b634acc8c1956d`). No
  threshold, axis, or rubric change of any kind; a `labeler_version != 3`
  anywhere fails closed.
- **Materialization:** one un-split, un-balanced test manifest, produced
  through the same staging/atomic-rename wrapper
  (`scripts/materialize_critic_dataset.py`) with `--val_scene_frac 1.0`,
  which deterministically routes every scene to the validation-format tree
  (`scene_split` assigns `val` for every hash bucket below 1.0; buckets lie
  in `[0,1)` by construction): no training split exists, no manifest
  balancing or media duplication occurs, and every evaluator consumes the
  same manifest format it already consumes for the 432-clip split. The
  materialized tree is the test corpus of record.
- **Target size and rationale (descriptive, not powered):** 500 clips over
  125 scenes at exactly four clips per scene structurally matches the
  validation split (432 = 108 × 4) with a modest scene-count margin. Because
  generator, driver-band mix, scene family, and labeler are unchanged, the
  natural label distribution is expected to resemble the validation split's
  197/140/32/38/25 (classes 1–5 of 432; `paper/draft.md` §4.1); scaled to
  500 clips the rarest class expects ≈29 members. The corpus is acceptable
  only if all five classes are represented with **at least 20 clips each**
  (the G1 corpus's minimum-per-score rule). If the natural draw of
  `[20000,20125)` fails that floor, the range is extended prospectively and
  deterministically in declared 25-scene blocks (`[20125,20150)`, then
  `[20150,20175)`, …, hard cap `[20000,20300)`), regenerating nothing and
  discarding nothing: every generated clip stays in the corpus, and the
  extension rule is label-blind at the scene level and fixed here before any
  clip exists. **No power claim is made or implied by this size.** Under the
  register's discipline, power-based sizing requires a separately
  materialized simulation amendment (cf. the challenge study's power
  artifacts); this corpus is sized by descriptive stratification-matching
  only, and its confirmatory character comes from untouched single-shot use,
  not from a sample-size argument.
- **Audit chain (rerun, not waived):** before any scoring, the new corpus
  must pass the same audit chain the 2,000-rollout corpus passed, in the
  v5 pipeline's order (`ops/v5_pipeline.sh`):
  1. `scripts/audit_physics_rollouts.py` — expected-episode/scene/schedule
     reconstruction by replaying the master RNG
     (`expected_episode_schedule(20000, 20125, 4, 20260812)`), rejection of
     any off-schedule episode ID, per-episode sidecar validation, full
     video probing via `--check-videos`, labeler-v3 recomputation,
     `--min-per-score 20`, launch-manifest validation, and the
     `_meta.episode_schedule_sha256` / `_meta.source_sha256` bindings
     (including the three SONIC ONNX hashes and the scene XML);
  2. `scripts/audit_critic_dataset.py` — independent split re-derivation
     over the materialized tree, copied media/conversation hash
     verification, five-class floor, labeler-version binding, and SHA-256
     binding to the rollout audit;
  3. an exact-equality replay of both audits in the style of
     `scripts/verify_g1_data_audits.py` before the seal is captured.
  Both auditors carry hard-coded v5 corpus constants — the launch-range
  contract `1000 <= lo < hi <= 1500` and `global_schedule_scene_lo=1000`
  (`audit_physics_rollouts.py` lines 253–280), and the dataset auditor's
  `scene_range == [1000, 1499]` binding (`audit_critic_dataset.py` lines
  61–65) — which must be extended to *additionally* accept this corpus's
  declared range, seed, and schedule anchor **by a tested code change made
  and hashed before generation begins**; the v5 contract itself, the
  canonical v5 audits (`4130c21c…`, `837cd7a4…`), and the G1 gate binding
  in `scripts/audit_research_gates.py` are not touched. Both new audit
  reports are saved with `_meta` and SHA-256s and are prerequisites for
  the seal.

## Fixed evaluation rule (core)

### Frozen readout roster

The following roster is frozen now. Each rostered readout is evaluated on the
test corpus **exactly once, ever**, in one declared batch. A readout enters
the batch only if (i) the corpus audits above have passed and (ii) the readout
itself is already frozen by its own protocol of record. Nothing outside this
roster — and nothing inside it whose artifact is not hash-pinned in the
pre-execution seal — may ever be evaluated on this corpus as part of the
confirmatory batch.

1. **R1 — frozen-tower ridge probe (coefficients of record; no refit).** The
   exact G4 probe of record: the fit/coefficient artifact and feature
   pipeline bound by `scripts/audit_probe_control.py` (generator
   `sim_rl/frozen_probe_baseline.py`; pinned base checkpoint revision
   `2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2`; selected regularization
   1000; historical validation record discrete r = 0.7053). Test-clip
   features are extracted with the probe's exact recorded frame-selection
   and feature procedure; the frozen coefficient vector is applied as-is.
   No refit, no lambda reselection, no recalibration. Continuous outputs
   are primary; rounded-and-clipped outputs are reported secondarily,
   matching the probe's record.
2. **R2 — selected v5 iteration-100 critic, historical interface (reported
   as historical).** The frozen iteration-100 checkpoint under its
   historical file-path temporal route and free-generation digit interface,
   unchanged parser and token budget. Per the canonical E5 amendment
   (`e5_temporal_interface_amendment_2026-08-11.md`), this number is
   labeled **historical mismatched-interface evidence** wherever it
   appears; its test-set value is confirmatory *for that historical
   system*, not a clean matched-interface estimate. Parse failures remain
   misses and are counted.
3. **R3 — the E1-selected constrained readout.** Eligible only after the E1
   factorial completes and its frozen descriptive selector (zero invalid
   outputs, then maximum Pearson, ties within `1e-12` on lower macro-MAE
   then earlier iteration) has fixed one checkpoint × temporal-route ×
   constrained-argmax cell, as recorded in the canonical `e1_result.json`.
   That exact cell definition — checkpoint, route, decoding contract — is
   sealed and run once on the test corpus. The E1 selection itself remains
   descriptive (it used the 432-clip split); the sealed cell's test-set
   number is confirmatory for the *selected* readout.
4. **R4 — temporal-matched v1 corrected critic.** Eligible only after the
   temporal-matched v1 protocol's own selection freezes
   (`temporal_matched_v1_evaluation_protocol_2026-08-11.md`: eligibility,
   zero-invalid rule, max-Pearson selection). The selected corrected
   checkpoint is run once through that protocol's primary interface (shared
   `traversal-temporal-v1` materializer, explicit metadata, processor
   sampling disabled, `enable_thinking=False`, greedy, batch size one,
   `max_new_tokens=8`, first-standalone-digit parser). If that selection
   terminates `failed_no_policy_interface_eligible_checkpoint`, R4 is
   reported as **not evaluable** — it is not replaced, substituted, or
   deferred into a later batch.
5. **R5 — shortcut duration+terminal row (frozen battery coefficients of
   record; refit: NO).** The duration + terminal 8×8-frame ridge readout as
   recomputed by the frozen shortcut-battery extension (its row 5), **not**
   the 2026-08-11 post-hoc pilot. Coefficients are the battery's
   coefficients of record: fit on the 1,568 training clips only, under the
   battery's frozen train-only five-fold scene-grouped lambda selection.
   Because the battery persists results, features, and its seed/index
   artifact but not the coefficient vector, the coefficient vector is
   materialized **before the seal** by a deterministic byte-replay of the
   battery's row-5 fit that must reproduce the battery's recorded row-5
   validation predictions exactly; the exported coefficient artifact is
   then hash-pinned. No coefficient is ever fit, tuned, or selected using
   any test clip. Test features use the battery's fixed featurization
   (bilinear 8×8 RGB terminal frame, the four frozen duration features).
6. **R6 (conditional) — external zero-shot baseline.** Included only if a
   separate external-baseline protocol (model list, exact prompt, parser,
   one pass) is frozen and its artifacts are pinnable before seal capture.
   If it is not frozen in time, R6 is recorded `absent_at_seal` and may
   never join this corpus's confirmatory batch.

**Companion reporting readouts (declared now, zero extra inference):** every
generative pass (R2, R3, R4, and R6 if present) must save per-item logits for
the five candidate score tokens at the first score-token position, following
the E1 release conditions. First-score-token argmax and Q-Align-style
expected score E[s] derived post-hoc from those saved logits are declared
companion readouts of the same sealed pass — they involve no additional
inference, are reported alongside the pass's primary parse, and give every
fine-tuned readout a continuous value so the probe-vs-critic comparison is
not confounded by quantization.

### Pre-execution seal

Before any test-corpus clip, label, or feature is read by any scoring or
feature-extraction process, a seal record is captured (a new
`scripts/record_heldout_test_seal.py`, modeled on
`scripts/record_temporal_matched_prelaunch_seal.py`, written and tested before
use). The seal binds, with path + SHA-256:

- this protocol file;
- the corpus manifest, labels, both new audit reports, and the generation
  launch/derivation records;
- for each rostered readout: every artifact it needs — checkpoint trees,
  coefficient/fit artifacts, processor and tokenizer files, evaluator and
  parser sources, the pinned base-checkpoint revision — plus, for R3 and R4,
  the upstream selection records (`e1_result.json`; the temporal-matched
  selection artifact) proving the readout was frozen before sealing;
- the declared bootstrap seed and the saved bootstrap draw-matrix artifact
  (drawn immediately after the corpus audit passes, before any score
  exists);
- the roster itself, including `absent_at_seal` entries.

**Anything not hash-pinned in the seal may never be evaluated on this corpus
as a confirmatory readout.** One evaluation pass per rostered readout, ever.
No second looks, no reruns with changed configuration, no interface swaps. A
pass that fails is reported as failed with its failure mode; the sole
exception is a pure infrastructure crash that produced **zero** per-item
outputs, which may be relaunched byte-identically once with the incident
logged — the moment any per-item output exists, the pass is consumed. All
sealed passes run in one declared batch, in the seal's roster order; no
correlation, metric, or interval is computed or read until every sealed pass
has completed or terminally failed (per-item outputs are write-only during
the batch).

## Frozen reporting

- **Per readout:** Pearson, Spearman, and Kendall tau-b against the frozen
  labeler-v3 test labels; per-class recall; macro-MAE over ground-truth
  classes; exact and within-one accuracy (secondary, given class imbalance);
  parse-failure counts (misses, never imputed); full output distribution.
  Continuous readouts (R1, expected-score companions) report continuous
  metrics primarily and rounded metrics secondarily.
- **Intervals:** scene-clustered bootstrap over the test scenes — 10,000
  resamples of scenes with replacement, items from a scene always moving
  together, drawn once with `random.Random(20260824)` (a fresh seed used
  nowhere else in this project) and saved as an index artifact before any
  score exists. **These intervals are confirmatory**: this is the first and
  only use of a corpus untouched by any training, selection, calibration, or
  protocol-tuning process.
- **Paired contrasts:** for every pair of rostered readouts that completed,
  the paired Pearson difference and paired macro-MAE difference with
  scene-clustered bootstrap 95% intervals from the same saved draw matrix.
  All pairs are reported; none is promoted or omitted by outcome. These are
  derived from the sealed per-item outputs and involve no new evaluation.
- **Completeness:** every rostered readout appears in the published table —
  evaluated, failed, not-evaluable, or `absent_at_seal` — none omitted,
  reordered by outcome, or split across venues while others are promoted.
- **Post-seal additions are descriptive, forever.** Any readout evaluated on
  this corpus after the seal (which requires a dated amendment written
  before that readout's result is observed, and may happen only after the
  confirmatory batch's results are published) is labeled **descriptive** and
  can never be upgraded, because the corpus's labels and score distributions
  are known by then.
- Historical framing rules are inherited: R2 is reported with the E5
  historical-mismatched-interface label; no ratio is reported as variance
  explained; no result here relabels, repairs, or replaces any historical
  table or the frozen matrix critic.

## Contamination rules

- **Storage:** rollouts at
  `<DATA>/traversal-critic/data/rollouts_heldout_test_v1/`,
  materialized corpus at
  `<DATA>/traversal-critic/data/critic_heldout_test_v1/`, seal at
  `<DATA>/traversal-critic/data/critic_heldout_test_v1_seal.json`,
  results and reports in a dated `autoresearch/` run directory with copies
  of all SHA-256s.
- **Access:** the corpus's clip files, labels, and features may be read only
  by (a) the generator writing them, (b) the labeler/materializer, (c) the
  two corpus auditors, (d) the seal capture/verify tool, and (e) the sealed
  batch's scoring processes. No training, checkpoint-selection, lambda- or
  hyperparameter-selection, calibration, prompt-design, or protocol-design
  process may ever read them. The run report logs every reader (tool,
  purpose, timestamp).
- **Violation consequence:** any computation on the corpus before the seal —
  including "just looking" at labels, score histograms, or features — voids
  confirmatory status for every artifact the computation touched; the
  affected readouts are demoted to descriptive on this corpus and the
  incident is reported, not repaired.
- The corpus never feeds any manifest, reward daemon, or PPO process, and no
  unreleased matrix outcome is read by any step of this protocol.

## Release conditions

- **Non-contention:** generation (CPU-heavy simulation) and scoring
  (GPU passes) must not contend with the frozen PPO matrix's compute, GPU
  memory, or I/O. Execution is deferred until matrix closure (expected
  ~2026-08-16) or until cleared hardware is available under the shared
  scheduler's non-interference verdict, whichever comes first. All runs are
  supervised launches.
- **Ordering:** (1) tested auditor-parameterization change; (2) generation;
  (3) materialization; (4) both corpus audits pass; (5) bootstrap draw
  matrix saved; (6) upstream protocols freeze their readouts (E1 result,
  temporal-matched selection, battery coefficients export); (7) seal
  capture and verification; (8) the single confirmatory batch; (9) run
  report and publication.
- **Provenance:** every saved artifact carries a `_meta` block (labeler
  version 3, dynamics `physics`, scene range `[20000,20125)` plus declared
  extensions, generation seed, schedule anchor, source SHA-256s, data
  authorities) and the run report lists SHA-256 values for the corpus
  manifest, labels, audit reports, seal, bootstrap index artifact,
  per-item outputs, and results file.
- **Amendment clause:** any change to this protocol requires a dated
  amendment written before the affected observation exists. An amendment
  cannot reinterpret a computed result, reopen a consumed evaluation pass,
  add a readout to the *confirmatory* batch after the seal, or upgrade a
  post-seal readout to confirmatory. If the corpus itself is ever touched
  contrary to the contamination rules, no amendment can restore its
  confirmatory status; the violation is reported and the corpus is retired
  to descriptive use.

## Why this matters

The 2026-08-12 red-team review ranks the absence of an untouched test set as
weakness 4 and gold-tier: "Every planned evaluation — E1, battery, even
temporal-matched v1 selection — reuses the same 432 clips that participated in
the historical sweep; all intervals stay descriptive forever under current
plans. … **Nothing currently planned does this.**" Reviewer 3's first
must-fix is exactly this instrument. Under the project's own honest register,
this protocol is the *only* path to a confirmatory in-domain number for any
readout — probe, historical critic, E1-selected interface, corrected critic,
or shortcut ceiling — and it serves the corrected-generation protocol too,
whose selection would otherwise remain descriptive on the reused split. The
ICLR 2027 abstract deadline (~Sep 24–26, 2026) requires these numbers to exist
before late September; generation of ~500 fresh-scene episodes is cheap
relative to the 2,000 already audited, and the marginal cost of doing it under
seal — rather than informally after the matrix closes — is one document and
one discipline: evaluate once, report everything.

---

*Frozen 2026-08-12, before any test-scene generation. Additive; defers
execution; changes nothing that is running.*
