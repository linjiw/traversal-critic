# External Review: Direction, Infrastructure Validity, and Open Questions

*2026-08-05. Independent verification pass over the paper draft, project page, the
`feat/traversal-critic` branch of cosmos-framework, and the experimental artifacts in
`/home/ubuntu/traversal_data`. Unlike the two prior reviews (review.md's literature pass
and docs/independent_review_2026-08-05.md's claims audit), this one traced every headline
number to its artifact on disk and read the reward/eval/data code paths line by line.*

---

## 1. Verdict

**The research direction is sound and the infrastructure is fundamentally valid.** The
numbers in the paper are real: every headline claim traces to a result file on disk, the
train/val scene split is verifiably leak-free, the critic-bonus credit assignment in PPO is
correct, and the three-arm comparison is paired properly (decoupled scene RNG streams,
identical shaping form for critic and oracle arms). The project's habit of adversarial
self-review has already caught and fixed the worst classes of bugs (dead critic bonus,
4-frame decode, camera confound, completion-blind demos).

That said, I found **one factual error in the paper, one experimental-design gap that a
reviewer will reject on, and several reproducibility hazards**. None of them invalidates the
direction; all of them are fixable before submission.

---

## 2. What I verified as correct (traced to artifacts)

| Claim | Artifact | Status |
|---|---|---|
| 5% / 39% / 53% three-arm success | `nav_eval_arms_peak.json` + `nav_eval_critic.json`, n=100 each | ✅ matches exactly |
| Single-yardstick curve 0.148 → 0.314 → 0.337 → 0.369 → 0.534 with CIs | `critic_ci_common.json` (n=589–590) | ✅ matches exactly |
| Peak-checkpoint selection symmetric across arms | recomputed window-10 smoothed training success from all three `train_log.jsonl`s → peaks at 246k/92k/114k | ✅ matches paper |
| v4 train/val scene split leak-free | `critic_v4` labels.csv: 241 train scenes ∩ 59 val scenes = **0 overlap**; 2,410/590 clips | ✅ |
| Scene-range hygiene across pipeline stages | PPO trains scenes 0–200; policy eval uses 400–440; critic v4 data from houses 1000–1299 — three disjoint ranges | ✅ |
| Class balancing is train-only | `prepare_traversal_critic_dataset.py` gates oversampling on `split == "train"`; duplicates are manifest-only | ✅ |
| Critic bonus credit assignment | `train_nav_policy_ppo.py:254-269` — bonus lands on the scored episode's own terminal step *before* GAE; truncation bootstraps V(s′) instead of zero | ✅ correct |
| Paired-arm scene streams | `traversal_nav_env.py` decouples scene-selection RNG from per-step planner RNG (episode-count-indexed), so arms see identical scene sequences despite divergent episode lengths | ✅ correct design |
| Critic actually shaped training | score_queue: 5,577 clips submitted, 5,577 scored, 0 unscored; score distribution during RL {1:163, 2:131, 3:192, 4:14 in first 500} — no critic-hacking-to-5 signature | ✅ |
| Daemon decode path matches training | `critic_scorer_daemon.py:_decode_frames` — uniform re-spread (not truncation), max_frames=32, `do_sample_frames=False` + explicit `video_metadata`, `enable_thinking=False` | ✅ post-fix |
| Baseline's 5% is not an eval artifact | 75/100 baseline episodes end in timeout (wandering on novel layouts), consistent with train-scene overfitting (0.51 train success), not a broken eval | ✅ plausible |
| Labeler axis logic | safety/clearance/social/progress scorers + `min(safety, half-down-round(mean))` match the documented contract; unit-tested | ✅ |

## 3. Errors found (fix before anything ships)

### 3.1 The "42 real G1 clips" claim is factually wrong — HIGH priority

The manifest (`real_g1_eval/manifest.json`) shows **15 real G1 clips + 27 clips from
`sim2sim.mp4` (regime "other-sim") = 42 total**. The draft §4.3 says "42 clips of real
Unitree G1 footage … plus 27 clips from a different simulator" (double-counts the 27), and
the project page says "42/42 real Unitree G1 clips." The already-thin real-transfer
evidence (all-positive clips, no negatives) drops from n=42 to **n=15 real clips**, of
which 10 are "clean walk." Per-regime means quoted in the paper are computed over as few
as 1–2 clips. Correct everywhere: paper §4.3, index.html stat tile, README table,
pitch.md. The honest framing: "15 real G1 clips + 27 clips from a second simulator."

### 3.2 Single seed per arm — the headline experiment is n=1 — HIGH priority

All three arm configs are `seed: 0`. The draft's "paired-seed PPO" phrasing implies a
multi-seed matrix that doesn't exist; your own review triage (review.md §8 item 1) demands
≥3 seeds and calls this "the hinge of the paper." With vanilla-PPO instability documented
in your own §4.2 (peaks then collapses; critic-arm final checkpoint = 0.00 success),
single-seed peak-vs-peak differences of this size *could* plausibly be seed variance.
39% vs 5% is a large gap and probably survives, but a reviewer only needs to ask once.
This is the cheapest high-value experiment left: the arms are CPU-trainable and the
supervisor scripts exist. Run seeds 1 and 2 for all three arms before writing "identical
seeds" anywhere else.

### 3.3 "Identical 300k steps" is not literally true — MEDIUM

The baseline run crash-resumed twice with the in-process step counter reset
(`train_log.jsonl` shows step resets at 143k and 169k) — it effectively experienced
~612k env steps of training against ~300k (critic) and ~320k (oracle). The resume code
itself warns optimizer/RNG state is not restored. This is *conservative in your favor*
(baseline trained longer and still lost), but the paper's "identical … steps" claim is
falsifiable from your own logs. Fix the wording ("at least 300k steps; the baseline
received ~2× due to crash-resume, favoring it") or re-run the baseline cleanly alongside
the multi-seed matrix (3.2).

### 3.4 Labeler version skew: published numbers used the *old* progress axis — MEDIUM

The completion-aware `score_progress` v2 (commit e9aafe3, Aug 5 21:56) landed **after**
every published eval ran (real-video 02:59, common-val 06:06, nav arms 18:20–18:38).
Consequences:
- The oracle arm's 53% was shaped by the old rate-based labeler; the critic was *trained*
  on old-labeler targets. That is internally consistent, but any rerun now silently uses a
  different reward function.
- v5 data, the oracle arm, and the eval labeler must move to the new axis **together**, and
  the paper must state which labeler version produced which table. Recommend a
  `LABELER_VERSION` constant written into every result JSON and labels.csv.

### 3.5 Stale public artifacts — LOW effort, reputational risk

- `paper/draft.pdf` (both repos) predates the Aug 5 honesty edits (kinematic-playback
  qualifiers, progress-axis caveat). The project page links to the stale PDF while the
  page itself carries the scope banner — inconsistent story to the same visitor.
- `traversal-critic-project/paper/draft.md` is an older snapshot of the cosmos-framework
  draft (missing the same honesty edits). Two divergent copies of the paper is how a wrong
  version gets submitted; make the project-page copy a build artifact of the canonical one.
- README.md and pitch.md still headline v2-era numbers (r=0.48 @ 1,530 clips; baseline
  success 0.15) and the 42-real-clips claim. The index.html abstract also still says
  r = 0.48 while the stat tile above it says 0.53.

### 3.6 v4 data predates the scene fixes — the "clean" dataset still has both scene bugs — HIGH

The scene-generator fixes in commit 0273635 (goal-adjacent gap wall producing spurious
negative clearance; "ghost walkers" phasing through gap walls in ~20% of peopled scenes)
were committed **2026-08-04**, but `rollouts_v4` was generated **2026-08-02** (file
timestamps; confirmed by RNG reconstruction — v4 scenes still use the old gap-position
population `[2,3,4,5]`, and ~25 of the 59 v4 val scenes contain the gap-at-5 geometry).
So v4 fixed **only the camera**; the paper's implication that v4 is the cleaned
regeneration is wrong on two of three known data bugs. Both the labels (clearance axis fed
by buggy geometry) and the pixels (walkers clipping through walls) are affected. A v5
regeneration with the fixed generator + fixed camera + completion-aware labeler is the
coherent move — it resolves this, 3.4, and the labeler-version pinning in one run.

### 3.7 "v4 isolates the camera confound" is not a clean isolation — MEDIUM

v4 trained with materially different budgets than v1–v3: `max_tokens=4500`,
`qwen_max_video_token_length=4096`, `grad_accum_iter=21` (v4 supervisor overrides,
confirmed in the run's config.yaml) vs 8000/8192/16 for earlier generations. The
0.369 → 0.534 jump attributed to the clean camera is confounded with a changed
video-token/resolution budget and effective batch size. Either rerun v3-data training
under the v4 budget (one training run) or soften the isolation claim to "camera +
training-budget change."

### 3.8 Smaller measurement issues — LOW

- **Select-on-test asymmetry on the headline curve**: v4_best (iter_700) was chosen by
  Pearson on the same 590-clip common yardstick it is reported on, while v1–v3 bests were
  chosen on their own val sets. Effect is small (iter_600 = 0.526, iter_800 = 0.533) but
  the protocol favors v4; disclose it or pick v4's checkpoint on its own in-domain val.
- **Parse-failure handling differs between tools**: `eval_videophy2` counts an unparseable
  reply as a miss (n stays 590); `critic_metrics_ci.py` and `analyze_critic_eval.py` drop
  it (n=589 for nearbase). ≤1 clip per run here, but the paper quotes acc from one tool
  and CIs from the other — unify.
- **Real-video clips play at ~0.8× speed**: `eval_critic_on_real_videos.py` strides with
  `int(fps // 12)` — for 29.97-fps sources that's stride 2 (≈15 fps) written with fps=12
  metadata, mild slow-motion vs training. A residual decode-domain mismatch after the
  4-frame fix; direction of bias unknown; fix the stride rounding before the negative-clips
  rerun.

### Additional checks that came back clean

- Scene-cluster bootstrap (10 clips/scene clustering) barely moves the v4 CI:
  [0.461, 0.602] vs clip-level [0.465, 0.595] — the clip-level CIs in the paper are
  defensible.
- Prompt byte-identical across all generations and matches `PROMPT_TEMPLATE`; common-val
  re-eval used identical script/args for all five points on the curve.
- Clip-duration shortcut ruled out: Pearson(frame count, label) = 0.059 on v4 val.
- Balancing duplicate-id scheme is collision-free and train-manifest-only.

## 4. Design-validity concerns (not bugs — decisions to defend or change)

1. **Critic checkpoint selected on the same val set it reports on.** iter_700 was chosen by
   val Pearson over 8 checkpoints, then r=0.534 is reported on that same 590-clip set. The
   optimism is mild (a selection over 8) but a strict reviewer will note it. Cheapest fix:
   hold out a small untouched test split from v5 data, or state the selection protocol
   explicitly and report the runner-up checkpoint's number too (0.526/0.533 nearby —
   the claim is robust; say so).
2. **The bonus-drop rate is invisible.** `n_bonus_applied`/`n_bonus_dropped` are counted
   but never logged (train_nav_policy_ppo.py) — you cannot currently report what fraction
   of episodes actually received their critic bonus, nor the score-arrival latency
   distribution. One line in the JSONL record fixes it; the paper's async-shaping story
   needs that number ("X% of episodes received their bonus within the same rollout").
3. **Stale-score semantics.** Scores arriving after `critic_wait_s` are dropped, so the
   *slowest-scored* episodes (often the longest/most eventful ones) systematically get no
   shaping. Probably minor at your queue throughput (all 5,577 clips were scored), but
   quantify it via (2) rather than asserting it.
4. **Auto-labeler axis quirks worth documenting**: social=5 when no people are in the scene
   (vacuous compliance inflates 4s/5s in people-free layouts, correlated with layout type);
   clearance=3 as "no channel" neutral; and person-collision terminates the episode at the
   *env* level (−2.0, terminate) but the labeler's safety axis reads it from frames — these
   are consistent today, but each is a threshold choice a reviewer can probe.
5. **Kinematic playback remains the biggest external-validity limit** — already honestly
   scoped by the Aug 5 independent review; keep that framing. The "collision-episode rate
   0.88–0.95 saturated" observation from that review stands: switch endpoint metrics to
   SPL / contact-seconds / collisions-per-meter for the next round.

## 5. Direction assessment — are you on the right track?

**Yes, with a re-scope that is already half-done.** The three prior reviews (literature,
triage, independent) converged on the same structural insight, and the artifacts now back
it: what you have is a **reward-model paper with an unusually rigorous evaluation
methodology**, not yet a robot-capability paper. The strongest, defensible claims today:

1. A 2B video-LM fine-tuned on ~2.4k auto-labeled clips reads traversal quality from
   pixels on held-out scenes (r=0.534, monotone single-yardstick progression, CIs,
   confound controlled by the shortcut probe). **Solid.**
2. Used as PPO shaping, the pixels-only critic recovers ~¾ of the privileged oracle's
   success gain over a hand-crafted baseline — the distillation-gap experiment nobody in
   your related-work table has run in this form. **Solid contingent on multi-seed (3.2).**
3. Transfer: **weakest leg** — 15 real clips, all positive, band-compressed. Say "transfer
   without collapse, n=15" or collect the ~20 deliberately-flawed real/cross-sim negatives
   the triage already lists.

The literature review's two "paper lives or dies" experiments have status: **arm (d)
oracle-direct — done** (that's your 53%); **VLM-vs-from-scratch on identical labels — still
missing** and is now the highest-value *new* experiment. A frozen-SigLIP+ordinal-probe
baseline on the exact v4 manifests would answer "why the 2B video prior?" — and either
result is publishable.

### Priority queue (my ordering, cost-weighted)

1. Fix the 42-clips error everywhere (hours).
2. **v5 dataset regeneration** with fixed scene generator + clean camera +
   completion-aware labeler + pinned `LABELER_VERSION` — resolves 3.4 and 3.6 in one run;
   retrain the critic on it under a documented budget (resolves 3.7's ambiguity going
   forward). This supersedes patching v4.
3. Multi-seed arms matrix, ≥3 seeds × 3 arms, clean no-resume runs, oracle on the v5
   labeler (days, CPU).
4. Log bonus-applied/dropped + regenerate PDF + sync the two paper copies + fix the
   real-video stride (hours).
5. Frozen-SigLIP / from-scratch probe baseline on the same manifests (days, the one
   missing scientific control).
6. Real/cross-sim negative clips for the transfer section (days).
7. Q-Align-style expectation head (cheap, addresses the measured central-tendency bias).
8. Isaac Lab physics-in-the-loop (the 6–12-month track — start, but don't gate the paper
   on it).

## 6. Questions for you (the author)

1. **Which paper are you writing for the March deadline?** The evidence supports
   "Traversal Critics: turning video foundation models into navigation rewards"
   (reward-model paper, ICLR-shaped) today. The humanoid-capability framing needs the
   Isaac Lab track. Pick one now; the intro, title, and figure set differ.
2. **Multi-seed budget**: the three-arm matrix at 3 seeds is ~9 CPU-days of wall clock on
   this box (arms run serially against one scorer daemon). Do you want the critic arms to
   share one daemon sequentially, or is it worth renting CPU to parallelize?
3. **Labeler v2 migration**: regenerate v5 data + retrain critic + rerun oracle arm under
   the completion-aware axis (one coherent story, more compute), or freeze everything at
   labeler-v1 for this paper and stage v2 for the physics phase? Mixing versions across
   tables is the one option that isn't defensible.
4. **The "critic right, labeler wrong" figure** (triage question 2) remains the single
   artifact that would turn "distillation" into "judgment" for reviewers — do the ~40
   near-threshold clips needed for it exist in v4's val pool, or does that need a
   targeted generation run?

---

*Verification methodology: all numeric claims cross-checked against
`/home/ubuntu/traversal_data/{nav_eval_arms_peak.json, nav_eval_critic.json,
critic_ci_common.json, real_g1_eval/manifest.json, critic_v4/*/labels.csv,
nav_ppo_*_r4/{config.json,train_log.jsonl}, eval_v4/iter_*/summary.json}`; code paths read
in full: `train_nav_policy_ppo.py`, `traversal_nav_env.py` (reward + CriticClient),
`critic_scorer_daemon.py`, `eval_nav_policy.py`, `prepare_traversal_critic_dataset.py`
(labeler + split + balancing), `generate_traversal_rollouts.py` (bands).*
