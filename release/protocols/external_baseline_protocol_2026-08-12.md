# External-Baseline Protocol — Zero-Shot Judge and Similarity Reference — 2026-08-12

**Status:** prospectively frozen 2026-08-12, before any external-baseline
result exists. At writing time no zero-shot score, no similarity score, and no
B-cell artifact of any kind has been generated. This protocol is additive. It
cannot change the selected v5 iteration-100 checkpoint, the active PPO matrix,
its scorer, any G1–G6/C1–C5 decision rule, the frozen E1 factorial or its
expected-score amendment, the frozen shortcut battery, or the temporal-matched
v1 protocol. GPU execution is deferred until it cannot contend with the frozen
matrix (see Release conditions). It converts the objection-map promise
"planned zero-shot baseline closes the loop empirically"
(`paper/positioning_and_polish_2026-08-11.md` §8) — identified by the red-team
review (`docs/reviews/red_team_review_2026-08-12.md`, weakness 7) as a promise
with no registered protocol — into a preregistered instrument.

## Question

Does fine-tuning on privileged rubric labels add ranking value over (a) the
same backbone used zero-shot and (b) generic vision-language similarity — the
two external reference points the VLM-reward literature the paper cites
(VLM-RMs, RoboCLIP, RL-VLM-F, Guan et al.; `paper/draft.md` §2) would demand,
and which §4 currently does not run? The existing ridge probe
(`probe_baseline_v5.json`, discrete Pearson 0.7053) already occupies the
*trained* CLIP-feature-regressor cell; §4.2 should say so explicitly. The
missing cells are the zero-shot judge and the zero-training similarity score.
This protocol freezes both.

## Fixed evaluation substrate (shared by all cells)

- **Validation set:** the unchanged 432-item, 108-scene held-out split at
  `<DATA>/traversal-critic/data/critic_v5/traversal_critic_val`
  (`meta.json` SHA-256
  `88cdae865c36814d8ac6f23b0a03bb3468aaad4d972cfe28ec256a4853d59147`,
  `labels.csv` SHA-256
  `119d4025152f8ea6376f8aff65523a6acd1002c9f59d6457a6e7ebc29f2ae251`), under
  the authority of the dataset audit
  `<DATA>/traversal-critic/data/critic_v5_dataset_audit.json`
  (SHA-256
  `837cd7a408b90b0a72ecb29a672d9623f3d6d760edf8ad93d3b634acc8c1956d`, the same
  hash pinned by the sealed E1 plan). No item may be added, dropped, or
  re-split; labels are the frozen labeler-v3 scores.
- **Temporal route:** the corrected single-pass route `corrected_v6` of the
  shared materializer `scripts/temporal_route_materializer.py` (source SHA-256
  `7efd595b8ef82e15c9958eb94a8f7c9d313bde72c242898460a0d97a986229bf`) —
  2 fps, ≤32 frames, explicit source-coordinate metadata,
  `do_sample_frames=False`. The baselines run on the corrected route so they
  are not sabotaged by the temporal-interface defect the release-blocking E5
  audit found in the historical pipeline; the run binds the route's
  `config_hash` and records every selected source index.
- **Bootstrap:** scene-clustered intervals reuse the frozen 108-scene draw
  matrix
  `<DATA>/traversal-critic/data/e1_factorial_v5/e1_scene_bootstrap_indices.json`
  (SHA-256
  `32df654a4f240dfbea93a6877436b983d300203f5cec7a0e4f5612f4f36b188b`; 10,000
  resamples, `random.Random(20260811)`, items from a scene always moving
  together). The stored matrix is reused, never regenerated after scores
  exist.
- **Parser (free generation only):** the first standalone digit in `[1,5]`,
  regex `\b([1-5])\b`, exactly the `_SCORE_RE`/`SCORE_RE` shared by
  `eval_videophy2.py` and the E1 harness
  (`cosmos_overlay/cosmos_framework/scripts/reasoner/eval_traversal_critic_e1_factorial.py`,
  source SHA-256
  `391538369842a62d48a3561234731041c1400e9540c5dc848bc08d4ee2bb4a6b`).

## Fixed baseline roster

### B1 — zero-shot same-backbone judge (frozen)

- **Model:** the Cosmos3-Edge BASE (pre-SFT) checkpoint, hash-pinned:
  - Path:
    `<DATA>/traversal-critic/huggingface/hub/models--nvidia--Cosmos3-Edge/snapshots/2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2`
    (also reachable via the `~/.cache/huggingface/hub` symlink).
  - Revision: `2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2`.
  - Tree SHA-256 (config + indexed weight bytes, per
    `sim_rl/frozen_probe_baseline.py::checkpoint_tree_record`, as recorded in
    `probe_baseline_v5.json` `_meta.base_checkpoint`):
    `79514e9ab4336f2a8752354bf644608b220bd8362ba5e62816fa2e4818836aa5`.
  - The run must recompute the tree hash and abort on mismatch. The base
    snapshot's `chat_template.jinja` (SHA-256
    `7120ee6666468d4e9b2dc11e133ac5c2fa765fa5907706bf0f906270aa5510c8`) is
    byte-identical to the v5 exports' template, so the chat framing is
    matched to the fine-tuned critic by construction.
- **Prompt:** the frozen zero-shot text quoted verbatim in the next section.
  It is the exact SFT rubric prompt embedded in every materialized validation
  conversation (user-turn text SHA-256
  `d16f9fef4ba32598c7971a956ae765c0fa4b9b41c0f0caa37bbde7f25e4bb011`) with
  **one appended line and no other change**. Rationale: the fine-tuned critic
  acquires the single-digit response format from supervision; the zero-shot
  judge receives the same format contract as an explicit instruction instead.
- **Generation contract:** `enable_thinking=False`, greedy, batch size one —
  the E1 contract.
- **Readouts (both mandatory, from the E1 design plus its expected-score
  amendment; free generation as an optional third):**
  1. **Constrained first-score-token argmax:** first generated token
     restricted to the tokenizer encodings of the five responses `1`–`5`;
     tokenizer IDs recorded; the run fails if any candidate is multi-token
     after the exact generation prefix; per-item logits for the five
     candidates saved.
  2. **Expected score:** the E1-amendment readout reused verbatim
     (`docs/reviews/e1_amendment_expected_score_readout_2026-08-11.md`): at
     the same first-score-token position, softmax over exactly the five
     recorded candidate token IDs, E[s] = Σ_{s=1..5} s·p(s) ∈ [1, 5],
     derived post-hoc from the saved logits with zero additional inference.
  3. **Free generation (optional third):** `max_new_tokens=32`, the shared
     parser above; unparseable outputs remain misses, are separately counted,
     and are never imputed or recovered from logits.

### B2 — zero-training vision-language similarity score (frozen)

- **Cell intent:** the RoboCLIP/VLM-RMs-style text-video similarity number —
  the *zero-training* similarity cell. The existing ridge probe is the
  *trained* feature-regressor cell; together the pair brackets the similarity
  family, and neither may be presented as covering the other.
- **Tower fact, established 2026-08-12 by a safetensors-header audit of the
  pinned base snapshot:** the critic's vision tower ships as
  `vision_encoder/model.safetensors` (SHA-256
  `2180ad739ecc96b5c1e9386892d3c5c08bfa42b9cdab9aabc53b028671db89b3`),
  containing only the SigLIP2-class vision transformer (`model.visual.*`) and
  the LM projector (`model.projector.*`). The SigLIP2 contrastive
  attention-pooling head, text tower, and logit scale/bias are **absent**, so
  a contrastive text-video similarity cannot be computed from the Edge
  snapshot alone.
- **Frozen scoring model:** the unmodified public SigLIP2 pair of the same
  class, repo id `google/siglip2-so400m-patch16-256` (its vision config must
  match the Edge tower class: hidden 1152, 27 layers, intermediate 4304,
  patch 16, 256 patches; mismatch or unavailability requires a dated
  pre-execution amendment naming the substitute). The exact revision and
  per-file SHA-256s are recorded in the run `_meta` at fetch time, **before
  any B2 score exists**. Fetching this public checkpoint moves no private
  data off the machine.
- **Tower-drift audit (frozen, CPU-only, reported alongside B2):** map the
  Edge `model.visual.*` tensors onto the public vision tower's tensors and
  report per-tensor max-abs-diff and cosine, classifying the Edge tower as
  byte-identical to or drifted from the public SigLIP2 vision weights. This
  bounds how literally B2 speaks for "the critic's own tower"; it has no
  selection power.
- **Frame set:** exactly the `corrected_v6` frames the shared materializer
  selects for each clip — the same frames B1 consumes; indices recorded.
- **Scoring (frozen):** each selected frame is preprocessed by the pinned
  SigLIP2 processor and embedded; frame embeddings are L2-normalized; the
  clip embedding is the **mean over frames, re-L2-normalized** (mean-pool
  aggregation, fixed). The five anchor texts below are embedded once and
  L2-normalized. With cos(s) the cosine between the clip embedding and
  anchor s, p = softmax over the five values `logit_scale · cos(s)` using the
  pinned checkpoint's learned `logit_scale` (the additive `logit_bias`
  cancels in a softmax over a shared bias and is omitted); the B2 score is
  the expected level Σ_{s=1..5} s·p(s) ∈ [1, 5]. The argmax level and the
  full 432×5 cosine matrix are saved and reported as secondary.

### B3 — frontier API-based VLM judge (OPTION, not frozen, user decision required)

A frontier internet-API video-capable VLM judge (the strongest available
GPT/Gemini-class judge at execution time) prompted with the same frozen B1
text is listed as an **unfrozen option with the decision explicitly
deferred**: scoring it requires uploading the private validation clips to an
external provider, and private clips leaving this machine is a user-level
call this document cannot make. If the user approves, a dated pre-execution
amendment must freeze the model identifier, endpoint, prompt (byte-identical
to B1's), parser (the shared regex), temperature 0, and a single pass before
any API score is observed. Until then, B3 appears in every roster table as
`deferred_user_decision`.

## Frozen B1 zero-shot prompt (verbatim; UTF-8 SHA-256 `c5bded43c5581158b4657b94ae72ccea5d064ebb246519b81b5f87fb30bbed37`)

The text between the BEGIN and END markers is the frozen prompt. Everything
above the final line is byte-identical to the materialized SFT user turn; the
final line is the single zero-shot addition.

```
BEGIN FROZEN B1 PROMPT
You are evaluating a short egocentric or third-person video of a humanoid robot moving through a cluttered indoor space that may contain people. Score the traversal quality.

Evaluation Criteria:
1. **Collision & Contact Safety:** Does the robot avoid all contact with obstacles, furniture, and people, and stay upright (no falls or stumbles)?
2. **Clearance Management:** In narrow passages, does the robot keep sensible margins and adapt — slowing down, turning its shoulders, crouching — instead of skimming past obstacles at speed?
3. **Motion Quality:** Is the motion smooth, purposeful, and humanlike, without oscillation, freezing, or erratic corrections?
4. **Social Compliance & Progress:** Does the robot respect people's personal space and yield when appropriate, while still making efficient progress toward its goal?

Instructions for Scoring:
- **1:** Dangerous traversal. Collision with a person, a fall, or reckless motion through occupied space.
- **2:** Poor. Solid collision with obstacles, or barging through tight gaps with no adaptation.
- **3:** Moderate. No serious collision, but skimpy clearance margins, jerky or hesitant motion, or weak progress.
- **4:** Good. Safe and mostly smooth, with only minor issues such as one grazing contact or brief hesitation.
- **5:** Excellent. Safe, well-cleared, smooth, socially compliant, and efficient traversal.

Response Template:
Analyze the video carefully and answer according to the following template:

[Score between 1 and 5.]

Example Responses:
2

Respond with only a single digit from 1 to 5 and nothing else.
END FROZEN B1 PROMPT
```

The implementation must byte-verify its prompt string against the SHA-256
above before scoring and abort on mismatch.

## Frozen B2 anchor texts (verbatim)

One anchor per rubric level, each derived from the corresponding
"Instructions for Scoring" level description with a shared stem; UTF-8
SHA-256 beside each.

1. `A humanoid robot traversing a cluttered indoor space dangerously: it collides with a person, falls over, or moves recklessly through occupied space.`
   — `2ddfc1d2b318ae30740e67377aed01372d4cce9512c2d8ac2fb83b44da4e43da`
2. `A humanoid robot traversing a cluttered indoor space poorly: it collides solidly with obstacles or barges through tight gaps with no adaptation.`
   — `93f939e13026cc94fa14b3ad0d0d4cc92424fb0e8d2dee219e3b3ced0d0b9257`
3. `A humanoid robot traversing a cluttered indoor space moderately well: no serious collision, but skimpy clearance margins, jerky or hesitant motion, or weak progress.`
   — `c5434686cd682113af615ff1c58cab929479800f29b0bd0e5cdcc06591a8f9fb`
4. `A humanoid robot traversing a cluttered indoor space well: safe and mostly smooth, with only minor issues such as one grazing contact or brief hesitation.`
   — `1c3e836caee3cdc249ffce603c596499c85111c0160b6c9f6bbb7602eb729e99`
5. `A humanoid robot traversing a cluttered indoor space excellently: safe, well-cleared, smooth, socially compliant, and efficient.`
   — `2a0e61d529027817154857886494799565e06309e34337b91e0c600ab3b43600`

## Frozen reporting

- **Per cell** (B1-constrained, B1-expected, B1-free if run, B2, B3 if
  approved): Pearson, Spearman, and Kendall tau-b against the validation
  labels over all 432 items — on continuous values for the expected-score and
  B2 cells, on parsed pairs for discrete cells. Secondary, per the E1
  amendment's conventions: for continuous cells, exact accuracy, within-one
  accuracy, and per-class recall computed on the rounded value
  (round-half-to-even to the nearest integer in 1–5), and the output
  distribution reported as a histogram of the continuous values; for discrete
  cells, exact accuracy, within-one accuracy, macro-MAE, per-class recall,
  and the output distribution, with unparseable free outputs counted as
  misses and reported separately.
- **Intervals:** scene-clustered bootstrap reusing the frozen 108-scene
  index artifact above. Every interval is labeled **descriptive**, because
  this validation set participated in the historical checkpoint sweep.
- **Every cell is reported.** No cell, metric, ratio, or interval may be
  omitted, reordered by outcome, or moved aside while others are promoted;
  the roster is published as one table.
- **Fixed comparison rows,** printed with every publication of the table,
  as descriptive Pearson comparisons and never as variance explained:
  1. the selected v5 critic's historical record, Pearson `0.5646`,
     explicitly labeled a *mismatched-interface historical record* per the
     E5 audit;
  2. the frozen-tower probe result of record, Pearson `0.7053` (discrete;
     continuous `0.7159`), the trained feature-regressor cell;
  3. the duration + terminal 8×8 frame shortcut, Pearson `0.6827`,
     explicitly labeled a *post-hoc descriptive pilot* per the shortcut
     battery extension protocol; the battery's preregistered recomputation
     supersedes it in the printed row once it exists.
- **No selection of any kind flows from these results.** No B-cell number
  may select, replace, or re-rank any checkpoint, influence the PPO matrix
  or its interpretation branches, alter any G/C rule, modify the E1,
  shortcut-battery, or temporal-matched instruments, or by itself promote or
  demote any claim slot. The cells are external reference rows only.

## Interaction clauses

- **Untouched test set:** if the preregistered untouched-test-set protocol
  (dated 2026-08-12; it may be written concurrently with this document)
  freezes before any B1/B2 execution, B1 (both mandatory readouts) and B2
  join its hash-pinned roster for exactly **one** confirmatory pass under
  that protocol's rules, carrying this document's prompt and anchor texts
  over byte-identically. If B1/B2 have already been executed on the 432-clip
  validation set when that protocol freezes, their membership in its roster
  is decided by that protocol, not retroactively by this one.
- **OOD extension:** after the validation pass, score the frozen 42-clip
  robot OOD corpus
  (`<DATA>/traversal-critic/data/real_g1_eval_v5/manifest.json`,
  SHA-256 `9dbfb03df61b14acb4cad78407a3f075c5801d6714836081facdf2255f44c5e1`;
  `corpus_audit.json` SHA-256
  `8e6e81a4741c95e0b63451bc8a47ca31063038d50137ffd0070e44695b3bbc00`) with
  B1 (both readouts) and B2, holding the corpus's existing predecoded
  temporal route fixed, exactly mirroring E1's OOD extension. Boundedness
  and regime ordering are reported descriptively; the corpus contains no
  collision/fall negatives, so no ordering result there establishes broad
  real-world validity.

## Release conditions

- **Contention:** B1 GPU inference is deferred until after the frozen PPO
  matrix closes and is adjudicated, or until it can run on hardware
  demonstrably not shared with the matrix; it must not start while its GPU
  memory or I/O demand could interfere with the matrix, and it must not read
  any unreleased matrix outcome. B2 and the tower-drift audit may run
  earlier only in CPU-only mode under the shortcut battery's contention
  rules (no GPU allocation of any kind); otherwise they observe the same
  deferral. Every run is supervised.
- **Contracts:** the B1 implementation reuses the E1 harness contracts —
  record the tokenizer IDs of all five candidate responses, fail if any
  candidate is multi-token after the exact generation prefix, save per-item
  logits and selected scores — plus, for this protocol: byte-verify the
  prompt and anchor texts against the SHA-256s frozen above, recompute and
  verify the base-checkpoint tree hash, verify the validation `meta.json`
  and `labels.csv` hashes, and bind the `corrected_v6` route config hash,
  all before the first item is scored; any mismatch aborts the run.
- **Artifacts:** results are saved under `autoresearch/` in a dated run
  directory with a `_meta` block recording every authority named in this
  document (dataset hashes, base-checkpoint tree hash, SigLIP2 repo id +
  revision + file hashes, materializer source hash and route config hash,
  bootstrap artifact hash, implementation script SHA-256s, prompt and anchor
  SHA-256s). The accompanying run report lists SHA-256 values for every
  saved artifact.
- **Immutability:** the B1 prompt text and B2 anchor texts may never be
  edited after any baseline result exists, except by a dated amendment
  written before the affected result is observed. Any other change to this
  protocol requires the same: a dated amendment written before the affected
  result is observed; an amendment cannot retroactively reinterpret a cell
  already computed.
