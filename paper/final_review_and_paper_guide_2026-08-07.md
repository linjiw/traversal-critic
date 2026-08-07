# Final Review & Complete Paper Guide

*2026-08-07. A final-round review of the paper draft and every experimental
result, followed by (A) the step-by-step path to the research goal and (B) the
full plan for arranging and writing the ICLR/ICRA submission. Companion to
`docs/advisor_response_2026-08-06.md` and the three prior reviews.*

---

# Part 1 — Final review of the paper and results

## 1.1 What is measured and solid (submission-grade evidence)

| # | Claim | Evidence | Artifact | Confidence |
|---|---|---|---|---|
| 1 | Fine-tuning extracts traversal judgment from pixels | r 0.148 → 0.534 [0.471, 0.595], strictly monotone across 4 interventions on ONE fixed 590-clip scene-held-out yardstick; clip- and scene-cluster bootstrap CIs agree | `critic_ci_common.json` | **High** |
| 2 | Critic shaping recovers most of privileged shaping (kinematic) | 5% / 39% / 53% success, paired seeds/scenes, n=100 | `nav_eval_arms_peak.json` | High (single seed disclosed) |
| 3 | **Critic shaping works under real contact physics** | baseline 0.20 / oracle 0.24 / **critic 0.36** success; SPL 0.19/0.20/**0.31**; falls 0.01/**0.11**/0.00 | `nav_eval_phys_peak.json` (labeler v3, `_meta`-stamped) | Medium-high (seed 0; matrix in flight) |
| 4 | The video prior buys OOD calibration, not in-domain accuracy | frozen-probe control: in-domain probe 0.545 ≈ critic 0.534; OOD probe emits 7.6–9.6 on a 1–5 scale with inverted ordering, critic stays in-band and correctly ordered | `probe_baseline_v4.json`, `probe_real_transfer.json` | **High** — and this is the paper's sharpest scientific point |
| 5 | Real-video transfer without collapse | 15 real G1 + 27 cross-sim clips, 0 parse fails, all in the correct band, regime ordering sensible | `real_g1_eval/scores.json` | Medium (n=15, no negatives — scoped honestly) |
| 6 | Hand-crafted rewards fail in measurable, mechanistic ways | (a) stand-still collapse (documented arithmetic); (b) baseline overfits: 0.51 train → 0.05 held-out; (c) **privileged oracle learns an 11%-fall gait under physics because its reward cannot see balance** | logs + phys eval | High |
| 7 | Learned pixels policy > privileged scripted expert | 0.36 (critic arm) vs 0.25 (waypoints + yield rule) on the same 100 episodes | `nav_eval_scripted_phys.json` | Medium-high |

Also strong and unusual for a paper at this stage: the **auditability layer**
— every result JSON carries `labeler_version` + dynamics + scene range; the
labeler has 23 unit tests; the dataset builder writes an audit CSV; all three
external reviews trace every number to an artifact on disk.

## 1.2 The single most important reframing (read this twice)

The physics result **flips the story from "distillation" to "the video prior
knows things the privileged state does not."** In the kinematic world the
critic was a lossy copy of the labeler (39% vs 53%). Under physics the critic
arm *beats* its own supervision source on every endpoint, and the mechanism is
identifiable: the labeler reads clearance/contact/progress scalars but has no
channel for balance or gait stability; video has both. Combined with the probe
control (finding 4), the paper now has a two-legged scientific thesis:

> **A video-LM reward is not a cheaper stand-in for privileged reward — it is
> a *differently-informed* reward.** (i) Under domain shift it stays bounded
> and calibrated where an appearance probe explodes; (ii) under real dynamics
> it credits stability information that privileged scalars omit.

Both legs are measured. That is the ICLR pitch. Do not bury it in §4 — it
must be in the abstract, the contributions, and the conclusion.

## 1.3 Honest weaknesses (rank-ordered by reviewer damage)

1. **Multi-seed matrix incomplete.** Done: baseline×3, oracle s0–s1, critic
   s0. Running: critic s1/s2, oracle s2 (invalidated critic runs from the
   daemon outage were caught and cleanly restarted — see §1.4). Until it
   lands, both three-arm tables are seed-0 with disclosure. **This is the
   last blocking experiment.**
2. **All shaped/critic gains are peak-checkpoint numbers under an unstable
   PPO.** All arms peak ~50k then decline. Peak-vs-peak is defensible and
   symmetric, but a reviewer will push. The `--target_kl`/`--anneal_lr` flags
   exist and are OFF for comparability; one stabilized critic-arm run (as an
   appendix ablation, not the headline) would defuse the question.
3. **Real transfer is n=15, positives only.** Scoped honestly, but
   contribution 3 stays "preliminary" until ~20 deliberately-flawed real or
   cross-sim negatives exist.
4. **Related work is still a bullet list.** The VIPER/VLM-RM/RL-VLM-F/Eureka
   engagement must be prose with explicit deltas (video-not-image; ordinal
   calibrated output; auto-labels not preferences; training-time-only; the
   OOD-calibration control that none of them run).
5. **v5 critic not yet evaluated** (sweep running). The paper currently rests
   on v4; v5 (physics data, labeler v3, fixed scene bugs, documented budget)
   should replace it everywhere if it clears v4's bar — or be reported as a
   negative alongside v4 if it doesn't.
6. **Smaller**: peak selection by smoothed train success can be fooled by
   early-window noise (baseline_s1 "peak" at step 512 from a 20-episode
   window) — the eval protocol needs a burn-in floor (e.g. ignore first 20k
   steps) written down *before* the matrix eval, not after looking at
   results. Person collisions are analytic (mocap capsules, no physical
   contact). Yield behavior in v5 data is scripted, not learned. Progress
   axis v2 is in v5 data only.

## 1.4 Infrastructure incident of record (for the repro appendix)

The scorer daemon restarted by the sequencer was missing `LD_LIBRARY_PATH`
for torchcodec's NPP libs → it returned `score=None` for ~3.7k clips, and the
critic matrix arms trained ~50k/190k steps with **zero bonuses applied** —
silently training the baseline reward. Caught via the new
`bonus_applied/dropped/pending` log fields (advisor fix §4.2 paid for itself
same-week); invalid runs deleted; daemon restarted with the lib path; null
scores requeued and now scoring (~0.7s/clip); critic s1/s2 relaunched clean.
**Lesson for the paper's async-shaping section: report the bonus-coverage
number per run; a critic arm without it is unfalsifiable.**

## 1.5 Verdict

The research direction survived four adversarial reviews and produced two
publishable findings beyond the original plan (probe OOD control; oracle
fall-rate mechanism). The kinematic→physics transition — the load-bearing gap
every review flagged — is closed at seed-0 and closing at n=3 seeds. What
remains is **completion and packaging**, not discovery: finish the matrix,
evaluate v5, write related work, freeze the claim set.

---

# Part 2 — Next-step guide to the research goal

Research goal (PI's words): *"robot could walk across cluttered and narrow
household"* — with our thesis that the missing piece is a learned, perceptual
reward. Two tracks: **Paper track** (freeze and submit) and **Capability
track** (the robot actually crossing rooms, continuing after submission).

## Phase A — Complete the evidence (now → ~5 days, all automated or cheap)

| Step | What | How | Gate |
|---|---|---|---|
| A1 | Finish multi-seed matrix | critic s1/s2 + oracle s2 running under supervisors; ~30h/run serial | all 9 runs `[ppo] done:` |
| A2 | **Pre-register the matrix eval protocol** (do this BEFORE A3) | peak = best 10-window smoothed train success with step ≥ 20k burn-in; 100 episodes, scenes 400–440, seed 123; metrics: success, SPL, contact-s, falls, labeler score; report mean ± min/max across seeds, per-seed table in appendix | protocol text committed before any s1/s2 eval runs |
| A3 | Run the matrix eval | `eval_nav_policy.py --physics` on 9 peak ckpts | `nav_eval_phys_matrix.json` |
| A4 | v5 critic sweep + common-yardstick re-eval | running (`eval_v5`); then score v5-best on the v4 590-clip yardstick AND v5's own val; run the probe on v5 manifests too (same control, cleaner data) | v5 ≥ v4 on its own val → v5 becomes "the critic"; else report both |
| A5 | Critic-vs-labeler disagreement figure (advisor Q4) | from v5 val: sample ~40 clips with \|critic − label\| ≥ 2, adjudicate manually against the rubric, report fraction where the critic is defensible | turns "distillation error" into "judgment" |
| A6 | Real/cross-sim negatives | generate ~20 deliberately-flawed rollouts in the SONIC MuJoCo sim2sim viewer (collisions, falls, wall-scraping) + any flawed real clips available; score with v5-best | closes the transfer caveat |
| A7 | λ-sensitivity mini-ablation | 3 short critic-arm runs (λ ∈ {0.25, 0.5, 1.0}, 100k steps, physics); appendix figure | cheap; answers a guaranteed question |

## Phase B — Freeze the claim set and write (days 5–12)

The five claims of record (everything else is supporting material):

1. **C1 (judge):** A 2B video-LM fine-tuned on ~2.4k auto-labeled sim clips
   predicts held-out-scene traversal quality (r=0.53; monotone scaling).
2. **C2 (control):** A frozen-encoder probe matches it in-domain but is
   uncalibrated OOD; the video prior's contribution is bounded judgment under
   shift. ← *the scientific heart*
3. **C3 (shaping):** Critic shaping beats hand-crafted reward under kinematic
   AND physics dynamics, n=3 seeds, and matches-or-beats privileged shaping
   under physics.
4. **C4 (mechanism):** Privileged shaping fails in an identifiable way under
   physics (11% falls — reward blind to balance); the video reward is not.
5. **C5 (transfer, scoped):** Zero-collapse transfer to real robot video at
   n=15+27 with correct banding; negatives from A6.

Anything not needed for C1–C5 goes to the appendix or gets cut.

## Phase C — Capability track (post-submission; the PI's end goal)

1. **Learned yielding**: recurrent policy (frame-stack or GRU) with `ego_obs`
   head camera — the scripted expert's 47% person-collision rate is the
   target to beat from pixels. This is the single biggest capability gap.
2. **Curriculum over scene difficulty** (gap width ↓, people ↑, table height
   ↓) using the two hooks already in the env; target ≥60% held-out success
   under physics.
3. **Critic v6 on on-policy data**: re-label rollouts from the *trained*
   policy (closes the distribution gap between scripted-band training data
   and policy behavior).
4. **Contactable people** (replace mocap capsules with a physically-coupled
   crowd model) — required before any social-navigation capability claim.
5. **Isaac Lab port** for GPU-parallel envs when scale demands it (the
   MuJoCo harness is CPU-bound at ~8 policy steps/s/process).
6. **Sim2real staging**: SONIC's ZMQ command path on a real G1; the deployed
   artifact is only the small policy + planner commands. Gate: sustained
   sim success ≥60% and a safety-cage protocol.

---

# Part 3 — Arranging and writing the ICLR/ICRA paper

## 3.1 Venue decision

**Primary: ICLR** (deadline typically late September — verify). The
contribution is a *learning* result (reward models, foundation-model priors,
an OOD-calibration control) evaluated in simulation; ICLR does not demand
hardware. **ICRA fallback** (September 15-ish) if the PI prefers the robotics
audience: same content, compressed to 6+n pages, C4 (falls mechanism)
promoted, C2 (probe control) compressed.

**Title options** (pick one that leads with the finding, not the system):
- *"The World Model as Judge: Video-Language Rewards that Stay Calibrated
  Where Privileged Rewards Cannot See"*
- *"Traversal Critics: Turning a Video Foundation Model into a Navigation
  Reward"* (current site framing; safer, flatter)
- *"Judge, Don't Simulate: Training-Time Video Rewards for Humanoid
  Navigation in Clutter"*

## 3.2 Paper skeleton (ICLR, 9 pages + appendix)

```
Abstract        150–200 words. One sentence per: problem (reward
                specification is perceptual) → method (critic from auto-
                labels) → C1 number → C3 physics numbers → C2 punchline
                (probe control) → C4 (oracle falls) → training-time-only.

1 Introduction  1.25 pp. Keep current §1.1–1.2 (they are good). Rebuild the
                contribution list as C1–C5 verbatim from Part 2B. Fig 1
                (system) on page 2. End with the two-legged thesis sentence
                from §1.2 of this review.

2 Related Work  0.75 pp, prose, four paragraphs:
                (a) VLM/foundation rewards: VLM-RMs, RL-VLM-F, RoboCLIP,
                    MineCLIP, Eureka — delta: video not image; ordinal
                    calibrated digits; auto-labels not preferences/LLM code.
                (b) Video models as rewards: VIPER (closest prior — engage
                    directly: VIPER uses likelihood under a generative
                    model; we use discriminative judgment + we run the
                    frozen-probe control they don't).
                (c) Learned rewards classic: Christiano, PEBBLE, VIP/LIV/R3M.
                (d) Humanoid WBC + social nav: SONIC/GR00T consumed frozen;
                    HumanoidPF etc.; social-nav rewards are hand-crafted —
                    ours is the learned-reward piece.

3 Method        1.5 pp. 3.1 problem + three-layer system (compress current
                §3.1+system fig caption). 3.2 auto-labeler (axes table +
                min/half-down rule + LABELER_VERSION discipline in one
                sentence). 3.3 critic SFT (one paragraph + the two
                reproduction landmines: thinking-mode off, val-loss is the
                wrong selector). 3.4 critic-shaped PPO (reward eq., async
                protocol, bonus-coverage number, credit assignment).

4 Experiments   3.5 pp — restructure around the claims:
                4.1 Judge quality (C1): monotone table + Fig 2; disclosures
                    inline (select-on-test bounded, budget confound).
                4.2 Probe control (C2): in-domain tie + OOD explosion.
                    Fig: scatter probe-vs-critic scores on real clips —
                    probe off-scale, critic in-band. THE figure of the paper.
                4.3 Shaping (C3): ONE table, two blocks — kinematic
                    (5/39/53) and physics matrix (mean ± range over 3
                    seeds); Fig 4 small multiples; scripted-expert row as
                    reference line; PPO instability + peak protocol +
                    burn-in floor stated.
                4.4 Mechanism (C4): oracle fall-rate story + one frame strip
                    of an oracle fall vs critic-arm crouch.
                4.5 Real transfer (C5): 15+27 with per-regime n, negatives
                    from A6, limitations verbatim from current draft.
                4.6 Hand-crafted failure vignette (current §4.5, one para).

5 Limitations   0.5 pp. Fold current §7: single robot; sim-only training;
                mocap people; auto-labeler blind spots (gait channel);
                completion-axis history; PPO instability; n=15 real.

6 Conclusion    0.25 pp. The two-legged thesis + capability roadmap sentence.

Appendix        A: env + scene generator details (+ the physics-vs-kinematic
                labeler recalibration measurements — reviewers love this).
                B: full per-seed matrix tables + training curves.
                C: labeler thresholds + version history + unit-test summary.
                D: async scorer protocol + bonus-coverage stats + the
                daemon-outage incident as a cautionary methods note.
                E: reproduction recipe (branch, budgets, seeds, one L4).
                F: λ sensitivity + stabilized-PPO ablation.
```

## 3.3 Figure plan (final set of 6)

| Fig | Content | Status |
|---|---|---|
| 1 | Three-layer system; train-time vs deploy-time split | done (light refresh: add physics env box) |
| 2 | Single-yardstick monotone progression w/ CIs | done |
| 3 | **Probe-vs-critic OOD scatter** (probe explodes off-scale, critic in-band) | **build after A4** — the paper's signature figure |
| 4 | Three-arm × {kinematic, physics} small multiples, mean ± seed range | rebuild after A3 |
| 5 | Confusion matrices (balancing ablation) | done |
| 6 | Real-transfer per-regime dots + probe overlay + negatives | extend after A6 |

Optional 7 (if space): oracle-fall vs critic-crouch frame strip (C4).

## 3.4 Writing order (fastest path to a full draft)

1. §4.3 physics matrix table the day A3 lands (numbers → prose).
2. §4.2 probe section — mostly written in the current draft; add Fig 3.
3. §2 related work prose (half a day; the deltas are already listed).
4. §1 contributions rewrite to C1–C5 + abstract rewrite (the current
   abstract still ends on the kinematic 39/53 — it must end on physics +
   probe).
5. §5 limitations fold-in; appendices from existing docs (mostly paste).
6. Two full read-throughs: one for claim-evidence pairing (every number in
   §1/abstract must appear in §4 with an artifact), one for tone (no
   "lovely", no hype; the honesty voice is this paper's brand).

## 3.5 Pre-submission checklist

- [ ] All 9 matrix runs done; eval protocol committed BEFORE matrix eval ran
- [ ] v5 sweep complete; v4-vs-v5 decision recorded; probe re-run on v5
- [ ] Every abstract/intro number traces to a `_meta`-stamped artifact
- [ ] Related work engages VIPER + VLM-RMs + RL-VLM-F + Eureka in prose
- [ ] Fig 3 (OOD scatter) built and validated (dataviz palette + validator)
- [ ] Negatives result in §4.5 (or the claim explicitly stays "no-collapse")
- [ ] PDF builds from the canonical draft; site + repo copies are the same file
- [ ] Anonymized repo for supplementary (strip usernames/paths)
- [ ] Advisor sign-off on the C1–C5 claim set specifically
```

The single sentence to keep on the wall while writing: **the paper's product
is trust** — every review cycle bought it; the writing must spend it
precisely, never past the evidence.
