# Matrix Interpretation Note (Outcome-Blind) — 2026-08-11

**Status:** frozen 2026-08-11, before any held-out endpoint of the 3-seed ×
3-arm × 300,032-step PPO matrix existed or was observed by anyone on the
analysis side. Training lanes are still running; the 18 registered held-out
evaluations (selected + final checkpoints × 3 arms × 3 seeds; 100 episodes,
scenes 400–440, evaluator seed 123) have not started. This note pre-commits
the interpretation and the exact sentence families the paper may use for every
registered outcome branch. Any change to this note after the first held-out
endpoint exists requires a dated amendment that names the change, and that
amendment forfeits outcome-blindness for whatever it touches: an amended
branch text is post-hoc interpretation and must be labeled as such in the
paper.

**Blinding disclosure.** The author of this note has read the frozen goal
document `research_goal_and_execution_plan_2026-08-09.md`, which cites the
pre-registration seed-0 exploratory contact-physics evaluation among its
historical motivating artifacts. That evaluation predates the frozen matrix,
is named in the preregistration itself as "not a substitute for the frozen
multi-seed protocol," and is not an outcome of the matrix. No PPO training
log, return curve, reward trace, held-out endpoint, scorer output
distribution, or partial arm comparison of the frozen matrix was read. The
outcome-blind orchestration and queue checks saved in
`claude_fable_plan_execution_report_2026-08-11.md` were the only live-state
evidence consulted.

**Non-interference.** This note changes nothing. It does not modify, stop,
reschedule, or reprioritize any matrix lane, scorer, queue, evaluator,
analysis, gate rule, claim slot, or schedule. It adds interpretation
discipline only.

---

## 1. The object under test (central premise)

The critic arm of the matrix tests exactly one system: **the frozen
iteration-100 v5 checkpoint, scored through the frozen policy-scorer
interface that predecodes 5–32 frames with explicit metadata.** The E5 audit
(release-blocking, executed 2026-08-11, `semantic_alignment_passed=false`)
established that this checkpoint was trained through an accidental four-frame
second-stage interface, historically validated through a 4–48-frame file-path
interface, and is scored in the matrix through the third, predecoded
interface. Per the Temporal-Alignment Release Amendment (2026-08-11), G5 and
G6 keep their registered meanings and may adjudicate policy utility **for the
exact historical system only**.

Whatever G5/G6 conclude therefore attaches to that historical system — the
specific checkpoint bytes plus the specific scoring interface that actually
ran — and to nothing else. No branch below licenses any statement about the
*intended* matched-interface 32-frame critic, in either direction.

## 2. Registered decision rules (restated so the branches bind to them)

These restatements are for reference; the frozen prose in
`research_goal_and_execution_plan_2026-08-09.md` (H3/H4 and the 2026-08-10
04:02 protocol lock) is authoritative and governs on any discrepancy.

- **G5 (critic vs baseline, H3):** selected-checkpoint held-out success;
  promotion requires (i) critic success higher than baseline in at least two
  of three seeds, (ii) the registered nested paired 95% bootstrap interval
  for the success difference excludes zero, and (iii) the unique-scene-crossed
  clustered sensitivity interval also excludes zero (20,000 draws, seed
  20260809). Final checkpoints are mandatory sensitivity results and cannot
  rescue a failed selected-checkpoint gate. Otherwise: report point estimates
  and both intervals with no superiority claim.
- **G6-policy (critic vs oracle, H4):** the strong "critic is non-inferior to
  its supervision source" wording requires a non-worse critic-minus-oracle
  success point estimate with both nested and crossed lower bounds at or
  above the fixed −0.05 absolute margin, plus a lower fall-rate point
  estimate with both interval upper bounds strictly below zero. Mean ordering
  without interval support is descriptive only.
- **G6-causal** is terminally infeasible; C4 can only be `reframed` or
  `rejected` regardless of the policy result. No branch below revives it.
- **Oracle-arm condition (interpretive only, not a registered gate).** No
  registered gate adjudicates oracle-vs-baseline. For the branch structure
  below, "oracle arm succeeds" means: the registered analysis output shows
  the oracle arm's selected-checkpoint success exceeding baseline in at least
  two of three seeds with both G5-style intervals excluding zero — the G5
  computation applied, descriptively, to the oracle–baseline contrast the
  analyzer already publishes. "Oracle arm fails" is any other outcome. This
  conditioning variable shapes interpretation only; the paper may state the
  oracle–baseline contrast descriptively but may not present it as a
  registered gate decision.

Why the oracle condition matters: the oracle arm is shaped by the privileged
labeler-v3 score itself — the very labels the critic distills. If even the
privileged labels fail to improve the policy over the hand-crafted baseline,
the label→reward premise is undermined for **all** learned-reward arms, and
no critic result (positive or negative) can be read as evidence about the
critic's distillation quality. The oracle arm frames the ceiling; its failure
removes the ceiling from the argument.

## 3. The two asymmetries (bind before reading any endpoint)

**Asymmetry 1 — a positive critic result is a lower bound, not a
validation.** If G5 passes, the licensed reading is: *even an
interface-mismatched judge — trained through four frames, scored through
5–32, whose training label is largely recoverable from terminal appearance —
supplied a reward that improved a contact-physics policy.* That is a real,
reportable result about reward shaping. It is **not** evidence that the
intended 32-frame critic works, that temporal video understanding
contributed, or that the interface mismatch was harmless. The mismatch caps
the claim from above: the paper may say the historical system sufficed; it
may not say the approach was validated as designed.

**Asymmetry 2 — a negative critic result indicts only the historical
system.** If G5 fails or the critic arm is inferior, the interface mismatch
is a live confound: the deployed system is not the system the method
intended, so the failure cannot be attributed to the critic *concept*, to
VLM-derived rewards generally, or to the intended matched-interface critic.
The licensed claim is exactly: *this historical system, under this scoring
interface, did not improve (or degraded) the policy under the registered
rule.* The corrected, separately named generation — not any reanalysis of
these runs — is the only path to evidence about the intended critic.

These asymmetries are deliberately unequal. The positive branch yields a
bounded but usable claim; the negative branch yields almost no claim about
the method, only about the artifact. Writing that down now, before endpoints,
is what makes either sentence usable in the paper.

## 4. Outcome branches

Six branches: three registered critic outcomes (G5 superior /
indistinguishable / inferior) crossed with the two oracle conditions. Each
branch fixes one interpretation paragraph, one sentence family the paper may
instantiate (with numbers and intervals filled in from the adjudicated
analysis, wording otherwise fixed), and an explicit list of what may not be
claimed. G6-policy is adjudicated separately inside the relevant branches.
"Indistinguishable" below means the G5 conjunction fails without the critic
point estimate being consistently negative — including the partial case
(e.g., 2/3 seeds positive but an interval retaining zero), which is reported
under the same branch with no softening adjectives. "Inferior" means the
critic success point estimate is below baseline in at least two of three
seeds; any interval evidence for that reversal is reported descriptively
(no registered gate adjudicates critic inferiority, and the paper must not
imply one did).

### Branch A — G5 passes; oracle arm succeeds

**Interpretation.** The label→reward premise holds at both ends: privileged
labels shape the policy, and a pixels-only distillation of those labels —
even through a mismatched temporal interface — also shapes the policy. This
is the strongest available branch and it is still a lower-bound result
(Asymmetry 1). If G6-policy also passes, the non-inferiority wording is
additionally licensed; if G6-policy fails, the critic–oracle comparison is
reported descriptively and the baseline comparison carries the claim.

**Sentence family.** "Under the registered G5 rule, PPO shaping with the
frozen historical critic system (iteration-100 checkpoint, 5–32-frame
predecoded scoring interface) improved held-out success over the hand-crafted
baseline (point estimates and both registered intervals reported), and the
privileged-label oracle arm also exceeded the baseline descriptively. Because
the deployed checkpoint was trained through a four-frame interface and its
training label is substantially predictable from terminal appearance alone,
this establishes a lower bound: even an interface-mismatched judge can supply
a useful contact-physics shaping signal. It does not validate the intended
matched-interface critic, and it does not establish that the reward signal
was temporal." If G6-policy passes, append: "Under the registered G6
conjunction, the critic arm was non-inferior to its privileged supervision
source on success (margin −0.05) with a lower fall rate."

**May not be claimed:** validation of the intended 32-frame critic; temporal
or pre-fall judgment; that the interface mismatch was benign; mechanism
(robot-visual or otherwise; C4 stays reframed/rejected); real-robot or
deployment benefit; generality beyond the evaluated scene range and
sim2sim-free in-domain regime; "critic beats its supervision source" unless
the full G6 conjunction holds.

### Branch B — G5 passes; oracle arm fails

**Interpretation.** Anomalous and to be reported as such: the distilled
reward improved the policy while the privileged source reward did not. This
pattern weakens the label-fidelity reading of the critic's benefit — if the
labels themselves do not demonstrably help, the critic's effect cannot be
attributed to faithful distillation of those labels and may instead reflect
reward-shaping dynamics (smoothness, density, calibration of the mapped
bonus) or chance structure. The oracle failure caps the premise for all
arms. A G6-policy "pass" in this branch (non-inferiority to an oracle that
itself failed to beat baseline) must not be presented as "matches privileged
supervision" in any headline; it is a comparison to a non-working reference.

**Sentence family.** "The historical critic arm improved held-out success
over the baseline under the registered G5 rule, but the privileged-oracle arm
did not show a corresponding improvement. Because the oracle arm is shaped by
the very labels the critic distills, this dissociation means the critic's
benefit cannot be attributed to label fidelity; we report it as an unresolved
shaping-dynamics observation attached to the exact historical system."

**May not be claimed:** everything excluded in Branch A, plus: any
label-fidelity or distillation-quality attribution for the critic's benefit;
any G6-based "matches its supervision source" headline; any claim that the
labeling rubric itself is a validated reward.

### Branch C — G5 indistinguishable; oracle arm succeeds

**Interpretation.** The premise survives — privileged labels can shape the
policy — but the historical critic system did not demonstrably transfer that
signal under the registered rule. This is consistent with, though it does not
prove, degradation along the audited axes: the four-frame training
interface, the train/score interface mismatch, and shortcut-diluted signal.
Per Asymmetry 2, this outcome does not indict the intended critic and does
not support "VLM reward critics do not work." The corrected generation and
the probe-shaping secondary study become the decisive next experiments.

**Sentence family.** "Under the registered G5 rule, shaping with the frozen
historical critic system was not distinguishable from the hand-crafted
baseline (point estimates and both intervals reported; k of 3 seeds
favorable), while the privileged-label oracle arm exceeded the baseline
descriptively. The labels carry a usable shaping signal; whether a
matched-interface pixels-only critic can transfer it remains open and is the
registered question for the corrected generation."

**May not be claimed:** that the critic (historical or intended) cannot
shape a policy; that the null is explained by the interface mismatch (the
mismatch is a confound, not a demonstrated cause); any superiority or
"trend" language for partial seed patterns; equivalence (no equivalence test
was registered).

### Branch D — G5 indistinguishable; oracle arm fails

**Interpretation.** The study is uninformative about the critic. With the
premise arm flat, a flat critic arm cannot separate "the critic lost the
signal" from "the labels-as-terminal-bonus channel carries no usable signal
at this budget, mapping, and environment." No critic conclusion is licensed.
The reportable products are the matrix itself as a negative-controlled
artifact and the reward-channel question (bonus mapping λ, terminal-bonus
density, budget) as registered future work.

**Sentence family.** "Neither the privileged-oracle arm nor the historical
critic arm was distinguishable from the hand-crafted baseline under the
registered analyses. This outcome does not adjudicate critic quality: it
indicates that, at this budget and reward mapping, the label-derived terminal
bonus channel itself did not demonstrably improve the policy, which caps what
any judge distilling those labels could have shown."

**May not be claimed:** any statement about critic quality, positive or
negative; that the labels are uninformative in general (only that this
channel, budget, and mapping did not show benefit); equivalence of any arms.

### Branch E — critic arm inferior to baseline; oracle arm succeeds

**Interpretation.** The historical system's reward was actively
counterproductive relative to the hand-crafted baseline while the label
ceiling held. This is the sharpest licensed negative and it attaches
strictly to the historical system (Asymmetry 2): a judge trained through
four frames and scored through a third interface can inject a harmful
signal. That is itself a publishable caution for VLM-reward pipelines — it
is the failure mode the paper's audit chain exists to catch — but it says
nothing about the matched-interface critic, which was never run.

**Sentence family.** "The historical critic arm underperformed the
hand-crafted baseline descriptively (per-seed point estimates and both
intervals reported; no registered gate adjudicates inferiority), while the
privileged-oracle arm exceeded the baseline. We attribute this only to the
exact deployed system — a four-frame-trained checkpoint scored through a
mismatched 5–32-frame interface — and present it as direct evidence that
interface audits of the kind reported in Section [interface-audit] are
load-bearing for VLM reward deployment."

**May not be claimed:** that the intended critic is harmful or that VLM
rewards are harmful in general; a causal attribution of the harm to the
interface mismatch specifically (confounded with checkpoint quality and
shortcut structure); any registered-gate framing for the inferiority itself.

### Branch F — critic arm inferior to baseline; oracle arm fails

**Interpretation.** The shaping channel is suspect end to end. With the
oracle arm failing, even the harm observed in the critic arm cannot be
attributed to the critic's distillation rather than to the terminal-bonus
channel or its mapping. No label→reward conclusion of any polarity is
licensed. The paper reports the full matrix descriptively and the study's
contribution rests on the audited benchmark, the interface/shortcut audit,
and the diagnosis machinery.

**Sentence family.** "Neither learned-reward arm improved on the
hand-crafted baseline, and the historical critic arm was descriptively worse.
Because the privileged-label arm also failed the same comparison, these runs
do not adjudicate the critic: they indicate the registered terminal-bonus
channel did not help at this budget, and any critic-specific effect is
unidentifiable within this design."

**May not be claimed:** any critic-specific conclusion; any general claim
about learned or VLM rewards; that the hand-crafted baseline is superior to
learned rewards as a class.

## 5. Interaction with the E5 shortcut finding (applies to every branch)

The shortcut probe showed the training label is substantially recoverable
from duration plus a terminal 8×8 frame (r = 0.683 on scene-disjoint
validation) — more so than by the selected critic itself (r = 0.565). Two
consequences bind here:

1. **Policy benefit, in any branch, does not establish temporal judgment.**
   A reward correlated with terminal-appearance-recoverable structure could
   carry the entire shaping effect. Even a Branch-A pass is compatible with
   the critic functioning as an expensive endpoint detector. No branch text
   may use "temporal," "watches," "anticipates," or "pre-fall" language for
   the critic's contribution.
2. **The diagnostic is already registered, not improvised.** The
   preregistered shortcut-battery extension (scene-clustered intervals;
   first-frame-only, random-single-frame, and shuffled-frame-order probe
   rows; onset-aligned prefix and endpoint-masked controls) is the named
   instrument for separating temporal from terminal signal, and the optional
   final-frame-ridge-shaped reward arm is the corresponding policy-level
   control. Until those complete, temporal attribution is out of bounds in
   every branch, and the paper should say the question is open and
   registered.

## 6. Pre-declared descriptive language for 3-seed variance

- Per-seed results are reported as counts ("higher in k of 3 seeds"), never
  as proportions, percentages, or "most seeds."
- Aggregate contrasts are reported as the point estimate with **both**
  registered intervals (nested paired primary; unique-scene-crossed
  sensitivity), always together, never the narrower one alone.
- "Statistically significant," "significant," and asterisk notation are not
  used anywhere. The only promotable phrasings are the registered gate
  verdicts: "passed the registered G5 conjunction" / "did not pass."
- Partial patterns are reported neutrally: "the registered conjunction was
  not met" — without "trend," "marginal," "nearly," "approaching," or any
  synonym.
- No per-seed hypothesis tests, no post-hoc subgroup or per-scene tests, and
  no additional intervals beyond the registered analysis output. Three seeds
  support replication counting and the registered bootstraps; they support
  nothing else.
- Final-checkpoint results are labeled "sensitivity" in every table and
  sentence; they cannot upgrade or headline any claim.

## 7. Claims no outcome can license

Independent of branch, the following are out of bounds for this matrix:

1. **Real-robot transfer.** All evidence is MuJoCo contact physics with
   chase-camera observations; no real-G1 policy result exists.
2. **Matched-interface critic validity.** The intended 32-frame critic was
   never trained or deployed; E5 remains scientifically `false` until a
   separately named corrected generation passes the matched tensor audit.
3. **Temporal-judgment attribution.** Blocked by the shortcut finding until
   the registered prefix/masking battery completes (Section 5).
4. **Mechanism claims.** G6-causal is terminally infeasible; C4 cannot be
   promoted; no robot-visual or pre-fall wording attaches to any policy
   result.
5. **OOD or deployment-average generality.** G4 is terminal-negative; the
   matrix evaluates scenes 400–440 in-domain only.
6. **Class-level claims about VLM rewards** ("VLM reward judges work" /
   "do not work") in either direction, from any branch.

## 8. Amendment rule

This note is complete as of its date. If any wording here conflicts with the
frozen gate prose in `research_goal_and_execution_plan_2026-08-09.md` or the
Temporal-Alignment Release Amendment, the frozen documents govern and the
conflict is resolved by dated amendment to this note only. After the first
held-out endpoint exists, this note may be corrected only by a dated
amendment that (i) quotes the original text, (ii) states what was seen at the
time of the amendment, and (iii) marks every amended branch as
post-observation interpretation. The paper may quote sentence families only
from the version of this note that predates endpoint existence, except where
an amendment is disclosed inline.
