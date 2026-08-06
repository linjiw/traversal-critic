# Response to the External Review of 2026-08-05

*2026-08-06. Status of every finding, and answers to the reviewer's four
questions. Companion to `~/traversal-critic-project/paper/advisor_review_2026-08-05.md`.*

## Findings — status

| Item | Status |
|---|---|
| 3.1 42-clips error | **FIXED** everywhere (paper §4.3 with per-regime n, pitch, README, site tile + abstract). Framing now "transfer without collapse at n=15". |
| 3.2 single-seed headline | **RUNNING** — seeds 1,2 × 3 arms under physics (two serial lanes, `ppo_matrix_lanes.sh`); disclosure added to §4.2 meanwhile. |
| 3.3 step-counter resets | **FIXED** (log-continuous counter on resume) + §4.2 discloses the baseline's ~2× budget (in its favor). |
| 3.4 labeler version skew | **FIXED** — `LABELER_VERSION` (now 3) written into every dataset (`dataset_info.json`) and eval JSON (`_meta`); §6 documents which version produced which table. |
| 3.5 stale public artifacts | **FIXED** — PDF rebuilt from the canonical draft with all six figures; project-page copies are now synced from cosmos-framework in the same script; README/site numbers updated. |
| 3.6 v4 scene bugs | **SUPERSEDED BY v5** — physics regeneration in flight (fixed generator + clean camera + labeler v3 + `dynamics` stamp), ~2,000 episodes. |
| 3.7 v4 budget confound | Disclosed in §4.1; v5 will train under one documented budget. |
| 3.8 stride/parse-handling | Stride **FIXED** (true-12fps resample); parse-handling unification pending the v5 eval round. |
| 4.2 bonus-drop invisibility | **FIXED** — `bonus_applied/dropped/pending` logged per rollout. |
| Probe control (priority 5) | **DONE — measured, both directions.** See below. |

## The probe result (new since the review)

Frozen SigLIP2 (the critic's own tower) + weighted ridge on the exact v4
manifests, CV-on-train λ, digit-for-digit Pearson:

- **In-domain: probe 0.545 ≈ critic 0.534.** The auto-labeler's signal is
  largely linearly recoverable from frozen appearance + temporal variance.
- **Off-distribution: probe collapses, critic doesn't.** On the real-G1 +
  cross-sim clips the probe emits 7.6–9.6 on a 1–5 scale with inverted
  regime ordering; the critic keeps all 42 clips in the valid 4–5 band,
  correctly ordered.

This reframes the contribution exactly as the review hoped: the video-LM
prior is not about in-domain accuracy — it buys **bounded, calibrated
judgment under domain shift**, which is the property a reward model must
keep when the policy distribution moves during RL. Written as §4.4.

## Answers to the four questions

**Q1 — which paper for March?** The reward-model paper:
*"Traversal Critics: turning video foundation models into navigation
rewards."* The evidence supports it today, and the probe result gives it a
crisp scientific spine: (i) auto-labeled supervision scales without humans;
(ii) shaping recovers ¾ of privileged-oracle gain from pixels; (iii) the
video-LM prior specifically buys OOD calibration that a frozen-encoder
probe demonstrably lacks. The physics track continues in parallel and
feeds the *next* paper (robot capability); §7 reports it as staging, not as
a claim. Intro/title/figures will be aligned to this framing in the next
paper pass.

**Q2 — multi-seed budget?** Running now at zero rental cost: two serial
lanes on this box (~2.7 steps/s per run alongside everything else; ≈30 h
per run, ≈4 days per lane). The critic arms share the existing daemon —
its queue never backed up at 2 arms and physics episodes are longer, so
contention is acceptable. If a deadline crunch appears, renting CPU for
lane 3 is the first thing to buy.

**Q3 — labeler migration?** Decision taken: **freeze published tables at
labeler v1 (disclosed), move everything new to v3 atomically.** v5 data,
the physics oracle arm (restarted on v3 the moment the calibration landed),
and all physics evals use v3; every artifact now carries its version. No
mixed-version table can occur by construction.

**Q4 — "critic right, labeler wrong" figure?** v4's val pool does contain
near-threshold clips (the p5-clearance distribution straddles the 0.15/0.30
cuts), but the cleaner source is the v5 physics pool where the labeler's
blind spots are sharper (head-clearance events, fall-adjacent recoveries).
Plan: after v5 SFT, sample ~40 val clips where |critic − labeler| ≥ 2,
manually adjudicate with the rubric, and report the fraction where the
critic's read is defensible. Scheduled after the v5 eval sweep.

## In flight right now

- 3 × seed-0 physics arms (~50k/300k), 2 × matrix lanes (seeds 1–2)
- v5 physics dataset ~700/2,000 episodes; pipeline armed (build → SFT)
- Next checkpoints: v5 SFT when generation completes; paired physics eval
  (with SPL/contact-seconds/fall-rate + `_meta` stamps) when arms peak.
