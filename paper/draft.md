# Do Video-Language Models Make Good Traversal Judges? Privileged-to-Visual Reward Distillation for Humanoid Navigation

*(Working manuscript for evidence-gated review; research planning and gate
history remain in `docs/reviews/`. Target venues: ICRA / ICLR. The checked-in
PDF and factual result figures predate the v5 evidence and are not submission
artifacts until the final evidence manifest validates their replacements.)*

---

## Abstract

Video-language models (VLMs) are increasingly used as robot reward judges, yet
they are typically validated by label correlation alone—without the privileged
signal that supplied their labels, without a simple readout of their own frozen
visual features, and without a policy-level test. We run all three controls in
humanoid traversal. A deterministic simulator rubric labels exactly 2,000
contact-physics rollouts, and a compact 2B-parameter VLM is fine-tuned to emit
one ordinal score from each video. The controls change the story a headline
correlation would tell. In-domain, the only checkpoint with valid outputs on
all 432 scene-disjoint validation clips reaches Pearson r = 0.565; later
checkpoints rank better (r = 0.705) but each emits one or two invalid
generations, exposing a tension between ranking quality and output reliability
under this model's autoregressive digit decoding. A release-blocking preprocessing audit
also finds that captured SFT, historical validation, and policy scoring used
different temporal interfaces, so r = 0.565 is not a clean matched-interface
estimate. A media-grouped ridge probe on the same
frozen vision tower reaches r = 0.705 with no autoregressive tuning at all,
while a preregistered 17-row shortcut battery recovers r = 0.683 from clip
duration plus one terminal 8×8 frame yet only r = 0.44 from the first half of
each clip: label-predictive signal survives the vision tower but concentrates
at the endpoint, and where the fine-tuned path loses quality remains an open,
preregistered question.
Under visual domain shift (42 real-robot and
separate-domain simulator clips), each readout fails a different criterion:
the VLM remains within [1,5] but ranks impaired gait above clean walking,
while the probe restores that ordering but places 8/42 unclipped predictions
outside the ordinal range. Because this OOD corpus contains no collision or
fall examples, neither readout establishes robust transfer or calibration. The
decisive test—a preregistered three-seed by three-arm PPO comparison of
hand-crafted, privileged, and VLM-derived shaping under a fixed budget—is
still in progress, so we make no policy-benefit claim. The current evidence,
all of it from a single 2B backbone at one scale, supports a controlled
privileged-to-visual diagnostic and a label-predictive frozen
representation—not an advantage for the fine-tuned autoregressive judge.

---

## 1. Introduction

### 1.1 When is a visual reward judge useful?

Modern whole-body controllers can walk, crawl, and recover from disturbances.
A different problem begins above that control layer: choosing how to move
through a narrow gap, beneath a low table, or around a person. The controller
may already contain the required motion; learning when and where to use it is a
decision and reward problem.

This setting makes VLM reward judging attractive but also unusually easy to
overclaim. A useful judge must pass three tests. First, its visual
representation must preserve the ordering encoded by the target rubric.
Second, its output interface must return a reliable scalar under scene and
visual-domain shift. Third, that scalar must improve policy learning relative
to the hand-crafted reward—and should be interpreted against direct access to
the privileged teacher. Validation correlation alone answers none of the last
two questions.

We conduct these tests for short humanoid-traversal videos, on one system: a
compact 2B VLM fine-tuned once (§3.2). For that system the answer is mixed.
The frozen visual representation contains substantial label-predictive signal,
but the selected autoregressive VLM readout does not outperform a weighted
linear probe. Later VLM checkpoints rank clips better yet occasionally fail to
emit a valid digit. Under visual domain shift, the VLM remains bounded but
misorders impaired gait, while the probe orders those clips correctly but is
unbounded. The final test—policy utility—is being measured in a preregistered
contact-physics PPO matrix and is not yet a result.

### 1.2 The privileged-to-visual test

The central objection is immediate: if privileged simulator state produces the
target, why not use that deterministic score directly? Inside the source
simulator, that is the cheaper and more accurate choice. A learned visual judge
has scientific or practical value only if the privileged rubric survives its
transfer to pixels and becomes usable where the underlying state is absent,
such as RGB-only logs, real-robot video, or another simulator without the same
instrumentation. We use simulation not because the VLM is needed there, but
because it supplies a controlled teacher, paired videos, and an oracle reference
with which to measure this abstraction gap. Our present OOD results do not yet
demonstrate that portability.

The VLM is used as a **judge**, not as a simulator or policy. It consumes a
completed clip and emits one ordinal token; it neither generates a future nor
controls the robot. The 2B model is used asynchronously during policy training
and discarded afterward. The evaluated learned component is a
0.43M-parameter navigation policy over a frozen whole-body controller. We
compare the VLM with both the privileged-labeler reference and a
frozen-vision-tower ridge probe. The former measures the cost of replacing the
teacher with pixels; the latter asks whether autoregressive VLM tuning adds
value beyond the base visual representation. Repository artifacts retain the
historical name `oracle` for the privileged-labeler arm.

### 1.3 Contributions

1. **A controlled benchmark for visual reward judging.** We pair exactly 2,000
   contact-physics traversal videos with a deterministic four-axis simulator
   rubric under a scene-hash 1,568/432 split, and evaluate a fine-tuned ordinal
   VLM against both its privileged teacher and a media-grouped frozen-feature
   readout—the controls that decide whether a visual judge, which matters only
   where privileged state is absent, deserves credit beyond its labels and its
   backbone.
2. **A representation–interface decomposition.** The only fully parseable VLM
   checkpoint reaches r = 0.565, whereas the frozen-tower ridge probe reaches
   r = 0.705. Later VLM checkpoints approach the probe's correlation but each
   produces one or two invalid generations. Captured SFT also reduced the five
   audited strata to four second-stage frames while validation and policy
   scoring used different frame counts. The evidence therefore diagnoses a
   combined temporal/readout interface problem; it does not localize a causal
   failure to instruction tuning.
3. **A two-sided domain-shift result.** Across 42 real-robot and
   separate-domain simulator clips, the VLM is bounded but fails
   impaired-versus-clean ordering; the probe restores the ordering but is
   unbounded. The corpus has no collision or fall examples, so we reject—not
   merely defer—claims of OOD calibration or adverse-event discrimination.
4. **An auditable, preregistered policy-utility test.** A checkpoint-bound
   asynchronous scorer attaches each timely VLM score to its originating
   episode. A preregistered three-seed by three-arm PPO matrix holds the
   controller, policy, scenes, and 300,032-transition budget fixed while
   comparing hand-crafted, privileged-labeler, and VLM shaping. Its outcome is
   in progress and is not counted as a positive empirical contribution.

### 1.4 Scope and assumptions

We do not claim that the critic replaces task reward: sparse goal reward and
privileged safety terms remain the backbone, and the critic only shapes
training. Nor do we test generative world dynamics. The task is local
goal-conditioned traversal, not room-scale exploration or general social
navigation. The clean reproduction uses a third-person chase camera and the
full SONIC stack under MuJoCo contact physics. The nine-run matrix is still
training, so historical single-seed and kinematic experiments support no policy
conclusion. Every critic-side finding likewise derives from one backbone at
one scale (a 2B Cosmos3-Edge) and a single SFT run; §6 states what that
scoping means for the mechanism claims. Real-robot dynamics, sensing, and
deployment remain future work (§6).

---

## 2. Related Work

**Learned rewards and visual representations.** Preference-based RL learns
objectives from comparisons between trajectory segments rather than manually
written rewards ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741));
[PEBBLE](https://arxiv.org/abs/2106.05091) improves feedback efficiency through
unsupervised pretraining, active queries, and replay relabeling. Video
pretraining can instead supply a reusable representation or distance-based
reward: [R3M](https://arxiv.org/abs/2203.12601) learns a frozen manipulation
representation, while [VIP](https://arxiv.org/abs/2210.00030) and
[LIV](https://arxiv.org/abs/2306.00958) learn value-implicit visual or
vision-language rewards from action-free human video. Closer to our recipe,
[Video-Language Critic](https://arxiv.org/abs/2405.19988) trains a contrastive
video-language reward on cross-embodiment data for manipulation RL, and
[VideoScore](https://arxiv.org/abs/2406.15252) fine-tunes a VLM into a numeric
video scorer from human ratings. Our labels require no online human
comparisons, but they inherit the simulator rubric and therefore need the
held-out, OOD, probe, and policy controls reported here.

**Foundation models as reward sources.** MineDojo uses MineCLIP, a pretrained
video-language model, as a learned reward for free-form Minecraft tasks
([Fan et al., 2022](https://arxiv.org/abs/2206.08853)).
[VLM-RMs](https://arxiv.org/abs/2310.12921) use pretrained image-language
similarity as a zero-shot reward — training a MuJoCo humanoid from a text
prompt — and document its Goodharting failure modes;
[RoboCLIP](https://arxiv.org/abs/2310.07899) derives manipulation rewards from
one video or text demonstration; and
[RL-VLM-F](https://arxiv.org/abs/2402.03681) queries a VLM for preferences over
image pairs before fitting a reward model.
[Language to Rewards](https://arxiv.org/abs/2306.08647) and
[Eureka](https://arxiv.org/abs/2310.12931) instead have an LLM author reward
code over privileged simulator state — privileged-rubric authoring rather than
visual judging. Recent systems bring VLM rewards toward our domains:
[Large Reward Models](https://arxiv.org/abs/2603.16065) generate online VLM
rewards — process, completion, and temporal-contrastive — for manipulation RL,
[MVR](https://arxiv.org/abs/2603.01694) shapes HumanoidBench locomotion with
multi-view VLM video rewards, and
[VLM-Social-Nav](https://arxiv.org/abs/2404.00210) scores candidate navigation
behaviors with a zero-shot VLM at planning time. In contrast, we fine-tune a
short-video ordinal critic on deterministic privileged-state labels, use it
asynchronously and only during policy training, and — unlike any of the above —
compare it against both the privileged signal that supplied its labels and a
simple readout of its own frozen visual backbone before asking whether it
moves a policy.

**Do VLMs judge robot behavior well?** [Guan et al.
(2024)](https://arxiv.org/abs/2402.04210) benchmark zero-shot VLM behavior
critics and document systematic failure modes.
[Generative Value Learning](https://arxiv.org/abs/2411.04549) poses visual
progress estimation as temporal ordering rather than direct scalar prediction,
and [OpenGVL](https://arxiv.org/abs/2509.17321) benchmarks that ability across
models — both evaluate by rank correlation alone.
[ProgressLM](https://arxiv.org/abs/2601.15224) finds that most VLMs fail
progress estimation and degrade under viewpoint shift, and [Kumar et al.
(2026)](https://arxiv.org/abs/2604.25235) report VLM judges that rank reliably
yet emit uninformative absolute scores — the mirror image of our
bounded-but-misordered domain-shift result in §4.3. On the data side,
[AHA](https://arxiv.org/abs/2410.00371)
trains explicitly on procedurally generated robotic failures,
[RoboReward](https://arxiv.org/abs/2601.00675) augments success-heavy robot
corpora with negative, near-miss, and partial-progress examples, and
[Tian et al. (2026)](https://arxiv.org/abs/2606.01036) argue that embodied
reward models systematically need bad-behavior data. These diagnostics
motivate our media-grouped readout and missing-negative-support controls, but
none tests a fine-tuned video judge against a privileged oracle and a
frozen-tower probe through policy training.

[ERL-VLM](https://proceedings.mlr.press/v267/luu25a.html) uses absolute VLM
trajectory ratings while explicitly addressing instability from imbalanced
data and noisy labels. [SOLE-R1](https://arxiv.org/abs/2603.28730) instead
trains temporally grounded video-language reasoning and reports that generic
VLM evaluators can be exploited under partial observability and distribution
shift. These findings reinforce our exact parser/range/accounting controls,
natural-adverse follow-up, and retention of sparse task plus privileged safety
reward; they do not establish that this ordinal critic is calibrated, robust
to reward hacking, or sufficient as a sole reward.

**Privileged information and teacher–student distillation.** Learning with
training-time-only privileged information is a classical paradigm
([Vapnik & Vashist, 2009](https://doi.org/10.1016/j.neunet.2009.06.042)).
Robotics uses it to distill privileged teachers into deployable students:
[Learning by Cheating](https://arxiv.org/abs/1912.12294) for vision-based
driving, [Lee et al. (2020)](https://doi.org/10.1126/scirobotics.abc5986) and
[RMA](https://arxiv.org/abs/2107.04034) for legged locomotion, and
[Humanoid Parkour Learning](https://arxiv.org/abs/2406.10759) for
height-scan-to-depth humanoid traversal. That recipe transfers *policies* out
of privileged state; we transpose it to the *reward path* — distilling a
deterministic privileged rubric into a pixels-only judge — and measure what
the transfer loses, with the privileged-labeler arm as the gold reference in
the sense of proxy-reward overoptimization
([Gao et al., 2023](https://arxiv.org/abs/2210.10760)).
[Driving Beyond Privilege](https://arxiv.org/abs/2512.04279) recently studied
privileged dense-reward knowledge distilled into sparse-reward driving
policies; our transfer target is a visual reward model rather than a world
model.

**Representation versus readout.** Fine-tuning can distort pretrained features
and underperform frozen-feature probes
([Kumar et al., 2022](https://arxiv.org/abs/2202.10054)), and linear probes
are the standard instrument for measuring what frozen representations encode
([Alain & Bengio, 2017](https://arxiv.org/abs/1610.01644)). Probing internal
states outperforms generated outputs for LLM truthfulness
([Orgad et al., 2025](https://arxiv.org/abs/2410.02707)) and VLM hallucination
detection ([HALP](https://arxiv.org/abs/2603.05465));
[TOPReward](https://arxiv.org/abs/2602.19313) extracts robot rewards from
token probabilities instead of generated progress values;
[Q-Align](https://arxiv.org/abs/2312.17090) reads ordinal visual scores from
level-token probabilities rather than free-form digits; and
[RALI](https://arxiv.org/abs/2510.11369) argues that representation alignment,
not generative reasoning, drives MLLM visual quality scoring. Our contribution
to this line is a controlled same-backbone comparison — a media-grouped ridge
probe on the exact frozen tower versus the same model fully fine-tuned as an
autoregressive digit judge — on scene-disjoint robot video, carried through to
an RL reward.

**Video prediction as reward.** [VIPER](https://arxiv.org/abs/2305.14343) is
the closest video-model precedent: it trains an autoregressive predictor on
expert video and rewards a policy using trajectory likelihood, including
temporal information without generating rollouts at reward time. Our critic is
discriminative rather than likelihood-based, and its supervision spans five
quality bands rather than only the expert distribution. That design makes the
1–5 interface explicit, but the failed OOD ordering test in §4.3 shows that
bounded decoding must not be conflated with calibrated judgment.

**Humanoid navigation over control substrates.**
[SONIC](https://arxiv.org/abs/2511.07820) scales motion tracking and exposes a
kinematic planner for downstream tasks, and
[HOVER](https://arxiv.org/abs/2410.21229) distills a multi-mode neural
whole-body controller behind a command interface; we consume the released
SONIC stack frozen rather than train a new locomotion substrate. High-level
policies over such controllers exist via imitation —
[NaVILA](https://arxiv.org/abs/2412.04453) commands a learned low-level policy
from a vision-language-action model on legged robots including the G1 — but
not, to our knowledge, via RL with a learned reward.
[Collision-Free Humanoid Traversal](https://arxiv.org/abs/2601.16035) learns
traversal of cluttered indoor scenes with a hand-designed humanoid–obstacle
potential field and demonstrates real transfer, and
[FocusNav](https://arxiv.org/abs/2601.12790) performs G1 local navigation via
spatial selective attention with waypoint guidance; neither includes moving
people or, to our knowledge, a reward learned from video.
[Habitat 3.0](https://arxiv.org/abs/2310.13724) provides humanoid simulation
and social-navigation/rearrangement tasks,
[SocNavGym](https://arxiv.org/abs/2304.14102) reports that data-driven social
rewards can outperform hand-crafted ones, and
[Francis et al. (2025)](https://arxiv.org/abs/2306.16740) argue that
evaluating social-navigation quality is itself unsolved — the gap a learned
traversal judge targets. Our narrower question is whether an ordinal video
judge can shape a humanoid navigation policy under contact physics beyond an
otherwise identical hand-crafted baseline and privileged-labeler reference.

---

## 3. Controlled Study Design

### 3.1 Problem setup

We study local goal-conditioned traversal in procedurally cluttered indoor
scenes with moving people. A frozen whole-body controller (C) combines SONIC
latent-token motion tracking at 50 Hz with a planner that exposes motion mode,
direction, speed, and height. A small visual policy π maps RGB, the relative
goal, and the previous planner command to a new planner command at approximately
3 Hz. We train π with PPO while holding the controller, policy architecture,
scene distribution, and training budget fixed across reward arms. The variable
under study is the shaping signal.

The registered reproduction uses a third-person chase camera. It is therefore
a controlled reward-learning experiment, not evidence that the same policy
works from an egocentric deployment view. Appendix D gives the complete policy
architecture, action mapping, and PPO hyperparameters.

→ `paper/figures/fig1_system.svg`

### 3.2 Privileged-to-visual reward distillation

**Data.** We generate four rollouts in each of 500 procedural scene seeds
[1000,1499], for exactly 2,000 episodes. Scenes contain doorway gaps of
0.6–1.2 m, furniture, low tables, prescribed-path people, and per-scene visual
randomization. The people are nonreactive mocap capsules: their physical
collision geometry is disabled to avoid infinite-mass contacts, while an
analytic distance threshold terminates person-collision episodes. Thus
"contact physics" refers to the robot, floor, and static clutter, not dynamic
human interaction. Banded scripted drivers generate a range of traversal
quality, but the privileged-state labeler—not the driver band—assigns every
target. Each chase-camera clip is rendered at 256 × 256 and 25 fps for at
most 24 s. Privileged channels recorded at 50 Hz include analytic point-to-box
clearance, minimum person distance, base speed, goal distance, obstacle and
person collision, and a reference-relative fall indicator.

**Auto-labeler.** Fixed thresholds produce four ordinal axes: safety from
contact duration, clearance from the fifth-percentile margin and
slow-when-tight behavior, a personal-space proxy from minimum person distance,
and progress from goal rate with a freezing penalty. Safety is 1 after any
fall or person collision, 2 after more than 0.4 s total obstacle contact, 4
after at most 0.4 s grazing contact, and 5 with no contact; it has no score-3
case. The overall label is the smaller of the safety score and the
half-down-rounded mean of the four axes, making safety a hard ceiling.
Labeler v3 makes progress completion-aware: non-arriving episodes are capped at
4, a score of 5 requires reaching within 0.4 m of the goal, and the axis loses
one point when more than 40% of frames are effectively frozen. Appendix B
gives every threshold. The sparse RL task reward separately retains terminal
completion credit.

A scene-hash split yields 392 training scenes (1,568 original clips) and 108
validation scenes (432 clips). Before balancing, training labels 1–5 occur
707/523/99/179/60 times. Manifest-level oversampling repeats training entries
to 707 per class (3,535 rows) without duplicating validation media. The
validation set is scene-disjoint, but it is used to select the checkpoint and
is not an untouched test set.

**Model.** We supervise Cosmos3-Edge (a Nemotron-2B language model with a
SigLIP2 vision tower) to map a clip and rubric prompt to one digit. Training
unfreezes the multimodal projector, final normalization, and last eight
language blocks while keeping the vision tower and language-model head fixed
(approximately 278M trainable parameters). The single clean SFT run uses AdamW
with learning rate 1 × 10⁻⁶, weight decay 0.05, a 50-iteration warm-up,
cosine decay through iteration 800, gradient accumulation 21, and a
4,500-token limit. It saves and validates every 100 iterations on one RTX 5090.
All clean-reproduction labels use rubric version 3. The prompt supplies the
rubric but no explicit goal location. Consequently, the progress target is only
partly observable from the clip, creating a possible incentive to use duration
or terminal-state correlates.

The intended training data path first strides 25-fps videos toward 2 fps and
caps that intermediate list at 32 frames. The preregistered temporal-interface
audit (2026-08-11) found, however, that the captured launch sampled that
already sampled list a second time under defaulted 24-fps metadata, reducing
every one of the five audited short/capped/success/fall/timeout clips to four
second-stage frames. Section 4.4 reports the full three-interface finding and
its consequences.

**Decoding and selection.** Two details materially affect the result. First,
generation-time reasoning is disabled because the critic is trained to emit a single digit. Second,
checkpoint selection uses task metrics but treats parseability as a hard
eligibility condition. Later v5 checkpoints rank clips better, yet each emits
one or two out-of-range generations; iteration 100 is the only eligible model.

### 3.3 Policy-utility test

All three arms share the same per-step baseline reward:

`r_t = 3 Δd_t I[no person collision] − 0.02 − 0.2 I[contact] − 2 I[person collision] − 5 I[fall] + 10 I[success]`,

where `Δd_t` is reduction in goal distance, `contact` means that any obstacle
contact occurred during the policy interval, and success requires goal
distance below 0.4 m with neither a fall nor person collision. The shaped arms
add one bonus to the terminal transition of episode `e`:

`r_t^shaped = r_t^base + I[t = T_e] λ (s(τ_e) − 3) / 2`,

with λ = 0.5 and completed clip `τ_e`. The critic arm uses the predicted score
`s(τ_e)`; the privileged-labeler reference uses the same rubric computed
directly from simulator state. This design holds
the baseline safety terms, policy, controller, and training budget fixed while
changing only the source of the episode-level score.

The critic path is asynchronous. Each environment writes completed episode
clips to a file queue, and a GPU daemon returns checkpoint-bound scores. After
each 512-step rollout, the trainer waits for at most 60 s so available scores
can be attached to their own terminal transitions before GAE. Late, failed, and
unmatched scores are counted rather than silently substituted. No policy step
blocks on a 2B forward pass, although a rollout boundary may wait. Section 5
describes the runtime, controller, queue, and checkpoint replay contract.

---

## 4. Experiments

### 4.1 What survives privileged-to-visual distillation?

We evaluate prediction on scene-disjoint validation clips using exact accuracy,
accuracy within one point, Pearson correlation, and parser failures. The clean
reproduction contains exactly 2,000 labeler-v3
contact-physics rollouts, split 1,568/432 by scene hash. Manifest-level
balancing expands the training partition to 3,535 rows without duplicating
validation media.

All eight fixed-budget checkpoints were evaluated on the same 432 validation
clips. Eligibility requires a finite Pearson correlation and a valid parsed
prediction for every item.

| iteration | Pearson | parse failures | eligible |
| --------: | ------: | -------------: | :------: |
| **100** | **0.564556** | **0** | **yes** |
| 200 | 0.630392 | 2 | no |
| 300 | 0.675511 | 2 | no |
| 400 | 0.696216 | 1 | no |
| 500 | 0.682023 | 1 | no |
| 600 | 0.704402 | 1 | no |
| 700 | 0.705093 | 1 | no |
| 800 | 0.686778 | 1 | no |

Iteration 100 is therefore selected. It reaches accuracy within one of
0.740741 and exact accuracy of 0.398148. Later checkpoints rank the validation
set better but each emits one or two parser-invalid outputs; we retain those
failures rather than repair generations or select a later checkpoint post hoc.

The validation labels are strongly imbalanced—scores 1–5 occur
197/140/32/38/25 times—which makes exact and within-one accuracy uninformative
on this set: constant predictors chosen from the training distribution alone,
without looking at validation outcomes, reach exact accuracy 0.456 (majority
label 1) and within-one accuracy 0.854 (median label 2), exceeding the
selected critic's 0.398 and 0.741. We therefore rely on correlation and
per-class diagnostics rather than accuracy. Pearson correlation is defined for
the nonconstant critic but undefined for either constant predictor; r = 0.565
therefore establishes linear association with the ordinal labels, not
calibrated five-class performance.

→ `paper/figures/fig2_scaling.svg`

The selected model's count and row-normalized confusion matrices are shown in
Fig. 5. For qualitative inspection, Fig. 3 shows one deterministically selected
validation clip per rubric label; neither figure is used for selection.

→ `paper/figures/fig5_confusion.svg`

→ `paper/figures/fig3_qualitative.png`

Earlier generations provide development context but are not
directly comparable because their data and cameras changed. The repository
does not contain the exact v4 weights or common-yardstick media required for
the registered paired v4/v5 non-inferiority test. The historical v4 aggregate
(r = 0.533692 on 590 clips) and the v5 result are therefore explicitly
unpaired. We do not infer monotone scaling, a clean-camera causal effect, or v5
superiority from those aggregates.
<!-- CLAIM_SLOT:C1:BEGIN -->
The fixed-budget v5 run yields an iteration-100 checkpoint with zero parser
failures and Pearson r = 0.565 on the scene-disjoint validation set used for
checkpoint selection, but E5 shows that this historical file-path evaluation
does not match the captured four-frame SFT or policy-scorer interface and is
therefore not a clean matched-interface estimate.
<!-- CLAIM_SLOT:C1:END -->

This result comes from one SFT run and has no independent critic-training
replication. More importantly, it separates two questions that a single
headline correlation would conflate: whether the model contains an ordinal
signal, and whether its generative interface returns that signal reliably.

### 4.2 Does VLM tuning outperform a frozen visual readout?

We fit a weighted ridge ordinal probe on the exact frozen
SigLIP2 tower used by v5 (8 frames, concatenated temporal mean and standard
deviation). The 1,568 unique training media expand to the same 3,535 balanced
rows used by the critic. Regularization is selected by five-fold train-only
cross-validation that groups all duplicates of a media path into one fold. The
432 scene-disjoint validation clips are not used to fit or tune the probe,
although they were already used for critic checkpoint selection. We compare
the critic with the probe's rounded digit predictions for in-domain Pearson and
retain continuous probe predictions for the OOD boundedness audit.

This is a representation control, not a parameter- or input-matched competing
model. The probe is order-invariant and uses 8 frames; the intended VLM route
can accept an ordered sequence of up to 32, but the captured SFT route consumed
four second-stage frames (§4.4). Its purpose is to test whether a
simple readout of the available frozen features already explains the claimed
traversal signal, not to provide a matched-input model comparison.

The selected ridge probe (regularization 1000) reaches discrete r = 0.7053,
exceeding the selected critic's r = 0.564556 on the same 432 validation clips.
It also matches the best raw correlation among the parser-ineligible VLM
checkpoints. VLM tuning under the registered selection and decoding
interface therefore does not improve on the simpler readout.

**What the probe does and does not show.** This result is more diagnostic than
a simple winner: the frozen base visual representation contains substantial
label-predictive signal under this preprocessing, so total absence of such signal in the tower is not a sufficient
explanation for the critic's OOD failure. However, temporal mean and standard
deviation are invariant to frame order. The probe therefore establishes neither
temporal reasoning nor attention to robot motion; static scene, posture,
duration, and terminal-state shortcuts remain viable explanations, and §4.4
quantifies how much of the label such shortcuts recover. The
remaining hypotheses concern the temporal input interface, instruction tuning,
the language/digit readout, calibration, and the adverse-event-deficient OOD
distribution. Distinguishing them requires the preregistered factorial readout
and shortcut-battery studies of §4.4, not a causal conclusion from this
control. Convergent evidence from other domains
makes the readout-bottleneck hypothesis the leading candidate — fine-tuning can
distort pretrained features ([Kumar et al., 2022](https://arxiv.org/abs/2202.10054)),
token-probability readouts outperform generated values for robot rewards
([TOPReward](https://arxiv.org/abs/2602.19313)), and representation rather than
generative reasoning drives MLLM visual scoring
([RALI](https://arxiv.org/abs/2510.11369)) — but it remains untested here.

**External similarity baseline and tower provenance (preregistered, executed
2026-08-12).** Two cells of the frozen external-baseline protocol
(`docs/reviews/external_baseline_protocol_2026-08-12.md`) completed before
matrix closure, CPU-only. First, the *zero-training* similarity cell — the
RoboCLIP/VLM-RMs-style readout, using the pinned public SigLIP2 checkpoint of
the Edge tower's class with five frozen rubric-anchor texts over the same
corrected-route frames — carries essentially no ranking signal: expected-level
Pearson r = −0.136 (scene-clustered 95% interval [−0.231, −0.033],
descriptive), with every clip's expected level falling in a 0.2-wide band
around level 2 and the argmax readout constant. Together with the probe, this
brackets the similarity family on this task: a *trained* linear readout of
SigLIP2-class features reaches r = 0.705 while *zero-training* text-anchor
similarity reaches approximately zero, so the tower's traversal signal is
linearly accessible but not aligned with rubric text in the public contrastive
embedding. Second, the tower-drift audit: the Edge vision tower is **not** the
public SigLIP2 release — all 437 mapped tensors differ (cosine mean 0.905,
minimum 0.483), i.e., a substantially retrained tower of the same
architecture. The probe result therefore characterizes a privately
continued-pretrained representation, and the zero-training cell speaks for the
tower's architecture class, not for the critic's own weights. The remaining
external cells are pending: the zero-shot same-backbone judge (B1) runs after
matrix closure under the frozen contention rules, and the frontier-API judge
(B3) awaits its finalized amendment.

### 4.3 Do either readout's criteria survive visual domain shift?

We score **15 clips of real Unitree G1 footage** from the GEAR-SONIC release
(in-the-wild navigation, impaired gait, crawling, and posture changes) and
**27 clips from a separate SONIC sim2sim demonstration domain**. Because these
clips have no privileged labels, this is an OOD stress test rather than an
accuracy benchmark.

The two readouts fail different criteria. All 42 VLM outputs parse and remain
in [1,5], but ten clean-walk clips average 4.3 while the single deliberately
impaired-gait clip (n = 1) scores 5.0. The probe gives the intended ordering—impaired
gait = 3.760853 versus a clean-walk mean of 4.833806—but 8/42 continuous
predictions lie outside [1,5], with an overall range of 2.346648–5.961985.
Because the VLM emits a digit while the ridge output is deliberately unclipped,
this comparison does not establish an intrinsic boundedness–ordering tradeoff.
It does show that neither model satisfies the registered conjunction under its
evaluated interface.
<!-- CLAIM_SLOT:C2:BEGIN -->
The media-grouped frozen-tower control does not support an OOD-calibration
advantage: the critic is bounded but misorders impaired gait, while the probe
satisfies the registered ordering but is unbounded on 8 of 42 clips.
<!-- CLAIM_SLOT:C2:END -->

Bounded output is evidence against numerical collapse, not evidence of
calibrated traversal judgment. The corpus is small and contains no collision or
fall examples—15 clips are real, and most non-clean regimes contain only one or
two clips—so it cannot measure adverse-event discrimination. We therefore
reject the OOD-calibration and positive-transfer claims.
<!-- CLAIM_SLOT:C5:BEGIN -->
The adverse-event-deficient OOD corpus shows bounded critic output but does not
support a positive-transfer claim because the impaired-gait ordering test fails
and no collision or fall discrimination is measured.
<!-- CLAIM_SLOT:C5:END -->

→ `paper/figures/fig6_real_transfer.svg`

A future study must collect scene-matched clean, collision, and naturally
falling clips with onset-aligned prefixes before making either claim.

### 4.4 Interface and shortcut audit

A reward model is not only weights; it is weights behind an input interface.
If training, validation, and deployment present different tensors for the same
clip, a validation correlation measures a system that never runs and a
deployed system that was never measured. Reward-judge validation in this
literature is typically correlation-only (§2) and rarely checks this
condition. We therefore preregistered a release-blocking interface audit (E5)
requiring that the SFT training path, the historical validation path, and the
policy-scorer path present the same temporal input interface—identical frame
indices, timestamps, metadata, prompt tokens, and resulting pixel
tensors—checked on deterministic short, capped, successful, falling, and
timeout strata.

The audit failed, and the mechanism is instructive. The captured SFT launch
double-sampled its input: the dataflow first pre-sampled each MP4 toward 2 fps
and at most 32 frames, and the captured processor then sampled that already
sampled list a second time; because no explicit video metadata reached the
processor, it silently defaulted to 24 fps and reduced every audited clip to
four second-stage frames. Historical validation instead passed MP4 paths and
selected 4–48 frames at 2 fps, and the policy scorer presents predecoded
5–32-frame tensors with explicit metadata—three different temporal interfaces,
with zero of the five audited strata matching on source indices, timestamps,
prompt tokens, or pixel tensors. This is an executed audit, not a
source-reading inference: it ran the real processor, tensors, and tokenizer,
independently reproduced the evaluator's actual source indices, and
byte-verified the four relevant preprocessing modules against the immutable
SFT process capture. The amendment of record is
`docs/reviews/e5_temporal_interface_amendment_2026-08-11.md`.

A companion shortcut diagnostic then asked how much of the label a trivially
shallow readout recovers on the unchanged scene-disjoint validation split.
A same-day pilot (duration, terminal 8×8 RGB frame, both; classified post-hoc
descriptive) motivated a 17-row battery that was prospectively frozen
(`docs/reviews/shortcut_battery_extension_protocol_2026-08-11.md`) and then
executed under that protocol (`autoresearch/run-260812-0034/`): ridge
regressions with train-only, scene-grouped regularization selection, a
media-fingerprint replay binding the corpus to the audited split (matched),
and a sanity gate requiring the recomputed pilot rows to reproduce — they did,
to full precision (duration+terminal r = 0.6827). All intervals are
scene-clustered bootstrap over the 108 validation scenes and remain
descriptive.

| battery row (preregistered, executed 2026-08-12) | Pearson | 95% CI | Spearman |
| --- | ---: | --- | ---: |
| duration only | 0.269 | [0.178, 0.356] | 0.429 |
| first frame only | 0.148 | [0.037, 0.257] | 0.134 |
| terminal frame only | 0.666 | [0.605, 0.722] | 0.700 |
| first + terminal | 0.650 | [0.579, 0.713] | 0.667 |
| duration + terminal | 0.683 | [0.635, 0.730] | 0.782 |
| duration + first + terminal | 0.674 | [0.616, 0.727] | 0.754 |
| random single frame | 0.372 | [0.272, 0.461] | 0.378 |
| onset prefix, first 25% of frames | 0.326 | [0.224, 0.424] | 0.325 |
| onset prefix, first 50% of frames | 0.440 | [0.335, 0.535] | 0.449 |
| endpoint-masked, last 10% removed | 0.574 | [0.495, 0.644] | 0.595 |
| endpoint-masked, last 25% removed | 0.492 | [0.421, 0.560] | 0.551 |
| shuffled frame order (first+terminal) | 0.642 | [0.583, 0.694] | 0.675 |
| shuffled frame order (dur+first+term) | 0.658 | [0.614, 0.700] | 0.751 |
| shuffled frame order (prefix 25%) | 0.237 | [0.142, 0.328] | 0.232 |
| shuffled frame order (prefix 50%) | 0.322 | [0.216, 0.421] | 0.344 |
| shuffled-label negative control | 0.010 | [−0.099, 0.121] | 0.010 |
| selected v5 critic (historical, mismatched interface) | 0.565 | — | — |
| frozen-tower probe (§4.2, reference) | 0.705 | — | — |

The battery bounds the temporal structure of the label itself. Pre-outcome
predictability is modest: the first quarter of a clip recovers r = 0.326 and
the first half r = 0.440. The endpoint dominates: one terminal frame recovers
r = 0.666, and removing just the final tenth of the frames drops the best
masked readout to 0.574 (final quarter: 0.492). The signal is not scene
identity — the first frame alone recovers only 0.148, and a random single
frame 0.372. Frame order carries little for endpoint-anchored rows (shuffling
costs ≤ 0.02) but measurably more for prefixes (0.326 → 0.237,
0.440 → 0.322), so what pre-outcome signal exists is partly ordered-motion
signal. Against these bounds, the frozen-tower probe's 0.705 exceeds the
terminal-frame row by ≈ 0.04 Pearson with overlapping intervals — a small
measured increment over low-resolution endpoint appearance — and the selected
critic's historical 0.565 sits below every endpoint-bearing row while
exceeding every endpoint-free row except the masked-10% row. These are
descriptive comparisons of label recoverability, not usage claims about any
model.

Two caveats are mandatory, and we adopt them as binding. First, comparisons
among these Pearson values are descriptive ratios, not variance-explained
estimates. Second, recoverability is not usage: the endpoint probe outscoring
the selected critic does not show that the VLM exploits terminal appearance;
it shows that the label is highly predictable from terminal appearance even
across scene-disjoint validation, which makes endpoint controls a necessary
causal diagnostic rather than an optional one. Part of this recoverability is
plausibly rubric-intrinsic—failed episodes end in fallen postures and
successful ones end at the goal—which is exactly why prefix and
endpoint-masked controls, not intuition, must quantify it. The near-zero
shuffled-label control indicates that the diagnostic pipeline itself does not
leak.

The audit reclassifies three of this paper's numbers. The selected critic's
r = 0.565 is historical mismatched-interface evidence—a file-path inference
interface applied to a model trained through a different four-frame
interface—and is no longer reported as a clean matched-interface estimate
(C1 in §4.1 is worded accordingly). The probe's r = 0.705 must now be read
against the 0.683 duration-plus-terminal reference: the temporal increment of
any readout over endpoint appearance is the decisive quantity, for the probe
as much as for the critic, and the battery above now bounds it for shallow
readouts — whether any *model* readout uses more than the endpoint remains a
usage question the factorial and corrected generation must answer. And the policy matrix (§4.5) retains its registered
meaning only for the exact frozen historical system—the four-frame-trained
iteration-100 checkpoint scored through the 5–32-frame predecoded interface;
whatever its outcome, it cannot validate the intended 32-frame critic, which
was never trained. An outcome-blind interpretation note frozen before any
held-out endpoint existed
(`docs/reviews/matrix_interpretation_note_outcome_blind_2026-08-11.md`)
pre-commits the licensed reading of every registered outcome branch.

The remaining preregistered instruments, all frozen before their results
exist, are assigned to the open questions. A frozen
2 temporal-route × 3 decoding-readout factorial
evaluates all eight fixed-budget checkpoints under the historical file-path
route and an SFT-exact route, each with free generation, first-score-token
constrained decoding, and a Q-Align-style expected score over the five
score-token probabilities
([Q-Align](https://arxiv.org/abs/2312.17090);
`docs/reviews/e1_amendment_expected_score_readout_2026-08-11.md`); no cell of
that factorial has selection power over the frozen matrix critic. Finally, a
corrected critic generation trained through a matched, tensor-audited
temporal route is required and separately named; it is never mixed into v5
tables, and replication of the unchanged recipe is explicitly disallowed as a
repair because it would replicate the bug.

We draw one methodological conclusion now, because it does not depend on any
pending result. Each control reported in this paper caught something a
headline correlation hides: parser eligibility caught invalid generations in
seven of eight checkpoints (§4.1); the same-backbone probe caught a
fine-tuned readout that, under the registered selection and decoding
interface, does not improve on frozen features in-domain (§4.2); the
domain-shift test caught a bounded but misordering judge (§4.3); the
interface audit caught three different temporal interfaces between training,
validation, and deployment; and the preregistered shortcut battery caught a
label largely recoverable from terminal appearance and only modestly
predictable before the outcome unfolds. The privileged oracle arm plays
the same role for the pending policy matrix: if privileged labels themselves
do not shape the policy, no judge distilling them can be credited or blamed
for its distillation. We take this chain—oracle reference,
frozen-representation probe, interface audit, shortcut battery—as the minimum
answer to the title question: this is what it takes to trust a VLM reward
judge, and none of it is visible in a validation correlation.

### 4.5 Does critic shaping improve policy learning? *(registered test in progress)*

This is the decisive experiment. Paired-seed PPO compares three reward arms
under identical scenes, policy architecture, controller, and step budget:
(a) the shared hand-crafted baseline, (b) privileged-labeler reference
shaping, and (c) selected-critic shaping. Held-out endpoints include success,
SPL, fall rate, contact seconds, person collision, minimum clearance, labeler
overall, and labeler safety. Success and SPL are the primary endpoints; the
remaining measures are registered safety or mechanism endpoints.

**Evidence-of-record protocol.** At each of three training seeds, all arms run
to the 300,000-step threshold—300,032 executed transitions at the next complete
512-step rollout boundary—on the 201-scene training support [0,200] under the
full SONIC MuJoCo contact-physics stack. The critic checkpoint is fixed before
policy training, and every critic run must retain at least 90% independently
verified applied-bonus coverage with correct checkpoint and queue ownership. A
1,024-step preflight applied all 13 requested bonuses with no unresolved or
wrong-model response.

For each arm and seed, we evaluate both the training-selected checkpoint and
the final checkpoint on identical held-out scene streams. Only the selected
checkpoint enters the primary G5/G6 decisions; the final checkpoint is required
sensitivity evidence and cannot rescue a failed primary. The registered stream
is 100 deterministic draws (evaluator seed 123) from the 41 possible scene IDs
[400,440], of which 40 are realized. The registered analysis reports both a
nested paired interval and a conservative interval that crosses training seeds
with unique held-out scenes. Pairing fixes scene identity and policy action
determinism, but not every low-level random draw: policies can consume the
persistent SONIC planner stream at different rates as their trajectories
diverge.

For G5, critic-over-baseline success requires a positive per-seed difference
in at least two of three seeds and positive 95% interval lower bounds under
both the nested paired bootstrap and the crossed seed-by-unique-scene
sensitivity. For G6's strong critic-versus-reference policy wording,
critic-minus-reference success must be nonnegative in point estimate, and both
interval lower bounds must be at least -0.05. Critic-minus-reference fall rate
must be negative in point estimate, and both interval upper bounds must be
below zero. Point ordering alone cannot pass either gate. Moreover, the
infeasible causal intervention in §6 blocks any robot-visual or pre-fall
mechanism claim even if the policy part of G6 passes.

The independent endpoint audit recomputes success from full-precision terminal
distance using the strict `<0.4 m` threshold and requires no fall or person
collision. It also reconstructs policy steps from the 50 Hz frame count and
15-frame action cadence. All 18 policy-evaluation artifacts (selected and final
checkpoints for three arms and three seeds) must share one pre-evaluation source
and runtime capture. The nine-run matrix is still running.
<!-- CLAIM_SLOT:C3:BEGIN -->
No policy-superiority result is available before the registered matrix and
held-out analysis complete.
<!-- CLAIM_SLOT:C3:END -->

Figure 4 is reserved for the terminal selected/final policy comparison. The
single-seed kinematic and physics pilots that motivated this protocol are
retained as historical artifacts, explicitly outside the evidence of record.

→ `paper/figures/fig4_arms.svg`

### 4.6 What a hand-crafted reward failure does—and does not—show

An early kinematic engineering run—not evidence of record—collapsed toward
idling because its per-frame contact penalties could outweigh the maximum
per-step progress term. The corrected baseline makes progress dominant, bounds
the contact cost per policy step, and adds time pressure. This failure
motivates testing a holistic episode-level critic, but does not show that a
learned reward is exploitation-proof; the registered held-out and coverage
analyses are required to determine whether it improves the policy.

### 4.7 Registered follow-up direction

After the current matrix is adjudicated, the registered follow-up studies
critic checkpoint quality versus policy gain, λ sensitivity, and scene-matched
negative OOD examples. The preregistered shortcut battery, decoding-readout
factorial, and separately named corrected generation of §4.4 do not read or
depend on the matrix outcome; the on-policy challenge-corpus design is
described in §6.

---

## 5. Reproducibility and auditability

### 5.1 Artifact and runtime provenance

The clean reproduction runs on one 32 GiB RTX 5090 workstation. Machine-readable
records bind the data, manifests, fixed-budget checkpoints, exports, selected
critic, OOD outputs, frozen-probe features and fit, policy sources, controller
assets, and scoring queues. `LABELER_VERSION` pins the rubric; both v5 data and
the privileged-labeler reference arm use version 3.

The runtime record inventories both Python environments, direct dependency
trees, installed versions, and native libraries mapped by representative policy
and scorer processes. Separate reports bind the SONIC planner, encoder, decoder,
scene XML, included robot XML, and 36 mesh assets. These reports were recovered
after policy training began, so we treat them as replay evidence rather than as
pre-launch archives. Their recorded file modification times precede the matrix,
and the held-out evaluator replays them before producing any endpoint.

### 5.2 Matrix and checkpoint integrity

Terminal matrix publication requires all nine trainers to stop and their writer
locks to quiesce. The auditor then checks the complete 300,032-step log sequence,
every checkpoint expected at the 10,240-step cadence, and one finite tensor
schema across all periodic and final policies. Raw policy state dictionaries do
not contain an internal step counter, so checkpoint attribution is limited to
the frozen save/filename contract and source provenance; we do not claim a
stronger internal attestation.

For critic arms, every request must bind one clip and one response from the
selected critic checkpoint. The terminal report retains dropped, failed,
null, late, and unmatched categories rather than normalizing them away, and it
requires the registered bonus-coverage threshold. Queue completeness is checked
again after the scorer stops. Any missing or changed queue member invalidates
the policy comparison.

Policy checkpoint selection is independently reimplemented from the bound
training logs and checkpoint bytes. The replay recomputes the 20k-step burn-in,
ten-record trailing-success mean, earlier-step tie-break, and checkpoint
mapping at full precision. It must pass before held-out evaluation begins.

### 5.3 Held-out evaluation, analysis, and release

Each seed-level held-out artifact contains six policy evaluations: selected and
final policies for the baseline, privileged-labeler reference, and critic arms.
Completed episodes are published atomically, and infrastructure recovery
preserves the committed prefix while restoring both Python and SONIC planner
RNG state. The evaluator
and independent auditor rederive episode metrics from raw records and require
one source, runtime, controller, asset, matrix, and selection generation across
all 18 policy evaluations.

The analysis auditor is source-separated from the primary analyzer. It
independently reconstructs Wilson summaries, the nested seed/episode bootstrap,
the crossed unique-scene bootstrap, and the exact G5/G6 rules, then requires
full scientific equality. Differential tests cover all four combinations of
G5/G6 pass and fail outcomes.

Factual figures are regenerated from current evidence rather than accepted by
image hash alone. Likewise, each C1–C5 statement occupies one designated
source slot and is materialized only after all required evidence gates close.
Final PDF verification reruns the gate, claim, figure, and provenance checks.
Historical v1–v4 measurements used different recipes and hardware and are not
presented as part of the clean reproduction.

## 6. Limitations and future work

The clean reproduction is entirely simulation-based. Contact physics makes
falls, wedging, and force-dependent clearance meaningful within MuJoCo, but it
does not establish transfer to real robot dynamics, sensors, people, or
deployment latency. The policy observes a third-person chase view rather than
an egocentric deployment camera. The moving people follow prescribed mocap
paths, do not react, and do not participate in contact dynamics; person
collision is an analytic terminal condition. The labeler's social axis is only
a minimum-distance proxy, not a measure of intent, yielding, comfort, or
long-horizon interaction. This is local traversal, not a general social-
navigation benchmark.

The 42-clip OOD corpus contains no collision or fall examples, is small, and is
imbalanced across regimes. Bounded predictions therefore do not demonstrate
collision or fall discrimination, and the selected critic fails even the
registered impaired-versus-clean ordering gate. We also have not completed a
behavioral reward-hacking analysis; queue integrity proves which reward was
applied, not that the policy could not exploit it.

The registered v4/v5 paired comparison is unresolved because the exact v4
weights and common-yardstick media are unavailable. The stronger raw Pearson
of parser-ineligible late-v5 checkpoints also exposes a tension between
ordinal ranking and strict digit generation that this experiment does not
resolve. Restricting decoding to valid score tokens — or reading an expected
score over level-token probabilities as in
[Q-Align](https://arxiv.org/abs/2312.17090) — is a straightforward and
important comparator: it may remove parse failures and change which checkpoint
is preferable. We do not apply it retroactively because the output interface
and checkpoint rule were frozen before this result. The preregistered
factorial of §4.4 evaluates the same checkpoints under free generation,
constrained score decoding, and the expected-score readout, with no selection
power over the frozen matrix critic. Even perfect parseability would not by
itself establish calibration, OOD ordering, or policy utility.

Auto-labels inherit threshold and rubric choices, including their
rate-based notion of progress. The critic receives no explicit goal location,
so some of that target is not identifiable from pixels alone. The frozen-tower
control also shows that a simple order-invariant readout can outperform the
selected SFT critic in-domain. Neither the single SFT training run nor its
checkpoint selection has been independently replicated. The current C1 result
is a post-selection point estimate; the final paper should add descriptive
scene-clustered uncertainty over the 108 validation scenes and per-class
diagnostics without treating them as confirmatory tests.

Every mechanism finding in this paper is likewise measured on one backbone at
one scale: Cosmos3-Edge, a Nemotron-2B language model over a SigLIP2 vision
tower, chosen because the full reproduction—SFT, the asynchronous scorer
daemon, and PPO training—had to share a single 32 GiB RTX 5090 workstation.
The parse-eligibility tension (§4.1), the probe-over-critic gap (§4.2), the
bounded-but-misordered domain-shift behavior (§4.3), and the consequences of
the three-interface mismatch (§4.4) are therefore claims about this system,
not about VLM judges in general; with one SFT run and no second backbone, we
cannot rule out that a larger model or a different vision–language pairing
would shrink, shift, or reverse any of them. What the study is designed to
export is the audit methodology rather than its outcomes: the interface
audit, the shortcut battery, the same-backbone probe-versus-critic
comparison, and the temporal-route × readout factorial are backbone-agnostic
instruments that any fine-tuned video judge can be run through, and whether
they catch the same failures on other backbones is exactly the untested
question. The one registered instrument with any reach beyond this backbone
is the tower-drift audit in the external-baseline protocol
(`docs/reviews/external_baseline_protocol_2026-08-12.md`, B2), which compares
the Edge vision tower tensor-by-tensor against public SigLIP2 weights of the
same class. Executed 2026-08-12, it found the tower *drifted* — all 437
mapped tensors differ from the public release (§4.2) — so the probe result of
§4.2 characterizes a privately continued-pretrained representation, and even
the frozen-representation finding does not transfer to publicly available
weights without retesting. A genuine generality test would require at least
two additional backbones, differing in vision tower and language-model family
or scale, carried through this same frozen protocol—data, rubric, split,
temporal-route materializer, decoding contracts, and audit gates unchanged.
We name that as future work and do not promise it here.

Video preprocessing is a measured confound, not merely a possibility. The
preregistered E5 interface audit found that the captured SFT, historical
validation, and policy-scoring paths presented three different temporal
interfaces, and its companion diagnostic showed the label largely recoverable
from duration and terminal appearance alone—without showing that the VLM uses
that shortcut (§4.4;
`docs/reviews/e5_temporal_interface_amendment_2026-08-11.md`). The historical
Pearson is therefore a mismatched-interface estimate, and repairing the
intended critic requires the separately named corrected generation with
matched temporal routing and endpoint/prefix controls, not a reanalysis or
replication of the captured recipe.

The force-induced causal-balance experiment is a terminal negative
feasibility result. After a prospectively frozen amendment increased the cap
from 5 to 20 deterministic initializations per scene, the required 20-pair
corpus still could not be completed under the unchanged force schedule. We do
not extend the cap or alter the intervention after observing that failure.
<!-- CLAIM_SLOT:C4:BEGIN -->
Consequently neither robot-visual dependence nor a pre-fall mechanism is
supported, and any policy fall-rate difference must not be interpreted as
mechanistic evidence.
<!-- CLAIM_SLOT:C4:END -->

The three-seed policy matrix and held-out analysis are incomplete.
Until all nine fixed-threshold runs (300,032 executed transitions each),
queue-coverage audits, checkpoint selection, identical held-out evaluations,
and registered uncertainty analyses are complete,
the historical single-seed point estimates support no policy-superiority or
parity claim. Even after completion, three training seeds and one deterministic
100-episode stream containing repeated draws from 40 realized unique held-out
scenes among 41 possible IDs
limit generalization across optimizer seeds and scene populations. The nested
and unique-scene-clustered intervals are paired checks under this fixed design,
not a substitute for a larger replication. Pairing fixes scene identity, but
not every low-level random draw: the physics wrapper retains one SONIC planner
random stream per policy evaluation, so policy-dependent episode lengths and
mode choices can shift later planner draws across arms. That reproducible
planner noise is part of this fixed evaluator and is not isolated as another
random factor.

The most promising next experiment—only after this matrix is adjudicated—is a
preregistered on-policy challenge corpus with scene-matched clean, collision,
and naturally falling trajectories and onset-aligned prefixes. A factorial
comparison of free and constrained digit decoding, frozen-tower
ordinal/regression, and calibration-only readouts under robot-hidden,
background-hidden, and placebo masks can test where signal is lost and whether
it exists before the realized fall becomes visible.

## 7. Conclusion

Do VLMs make good traversal judges? The present answer is conditional. The
frozen vision tower carries a label-predictive scene-disjoint signal—much of
it recoverable from terminal appearance alone (§4.4)—but the registered
autoregressive VLM interface does not improve its ranking and makes
checkpoint choice depend on output validity. Under visual domain shift, one
readout preserves the ordinal range and the other preserves the registered
ordering; neither satisfies both criteria, and the adverse-event-deficient
corpus cannot test collision or fall judgment. These are design findings, not
evidence that visual reward learning is intrinsically futile.

The remaining question is whether either statistical association matters for
control. The frozen three-seed PPO comparison against hand-crafted and
privileged-labeler shaping is designed to answer that question without treating
critic correlation as a proxy for reward utility. Until its held-out analysis
closes, the evidence supports a controlled privileged-to-visual diagnostic, a
label-predictive frozen representation, and an auditable asynchronous reward
path—but not a benefit from VLM shaping, parity with the privileged teacher,
real-robot transfer, or a mechanistic account of any fall-rate difference.

---

## Appendix A. Historical development evidence

The following studies motivated the clean v5 protocol but are not evidence of
record. They use different datasets, cameras, selection procedures, or training
histories and must not be combined into a scaling or causal claim.

### A.1 Earlier critic generations

| model | train clips | validation clips | Pearson | within one | exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| near-base (20 clips seen) | — | 140 | 0.024 | 0.557 | 0.314 |
| critic v1 | 460 | 140 | 0.377 | 0.679 | 0.457 |
| critic v2 | 1,530 | 470 | 0.482 | 0.755 | 0.517 |
| critic v3 (balanced) | 2,930 | 670 | 0.452 | 0.743 | 0.460 |
| critic v4 (clean camera) | 2,410 | 590 | 0.534 | 0.768 | 0.476 |

These rows were measured on changing validation corpora. The exact v4
checkpoint and its 590-clip corpus are unavailable, so the registered
common-yardstick comparison with v5 cannot be reconstructed.

### A.2 Kinematic policy pilot

One training seed per arm was evaluated on 100 episodes. Checkpoints were
selected by smoothed training success, but the baseline crash-resumed under a
faulty step counter and received roughly twice the shaped arms' training
budget.

| arm | success ↑ | collision episodes ↓ | labeler score ↑ | steps to end ↓ |
| --- | ---: | ---: | ---: | ---: |
| hand-crafted baseline (peak, 246k) | 0.05 | 0.95 | 1.80 | 56 |
| critic-shaped (peak, 92k) | 0.39 | 0.89 | 2.00 | 46 |
| privileged-labeler-shaped (peak, 114k) | 0.53 | 0.88 | 2.22 | 29 |

The pilot has one seed, unstable peak selection, unequal training budgets, and
a nearly saturated collision endpoint. It supports no causal reward-arm claim.

### A.3 Contact-physics policy pilot

The single-seed physics pilot used the full SONIC stack, labeler v3, peaks near
51k steps selected by the same training-only rule, and 100 evaluation episodes.

| arm | success ↑ | SPL ↑ | contact seconds ↓ | fall rate ↓ | steps to end ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| hand-crafted baseline | 0.20 | 0.19 | 14.2 | 0.01 | 75 |
| privileged-labeler-shaped | 0.24 | 0.20 | 8.8 | 0.11 | 61 |
| critic-shaped | 0.36 | 0.31 | 7.4 | 0.00 | 60 |

The critic has the most favorable point estimate, but one seed cannot separate
a reward effect from seed variance or historical checkpoint selection. The
reference arm's fall rate is not evidence that the labeler lacks fall awareness:
labeler v3 maps every realized fall to safety 1. Only the registered matrix in
§4.5 is eligible for the final policy claim.

## Appendix B. Labeler-v3 rubric

The labeler consumes the 50 Hz privileged trace. These thresholds are frozen
for all clean-v5 data and for the privileged-labeler policy arm.

| axis | fixed labeler-v3 rule |
| --- | --- |
| safety | 1 for any fall or person collision; otherwise 5 for no obstacle contact, 4 for at most 0.4 s total contact, and 2 for longer contact. Score 3 is not emitted. |
| clearance | Compute the nearest-rank fifth percentile `p5` of analytic robot-to-clutter clearance. Physics base scores are 5 if `p5 ≥ 0.15 m`, 4 if `p5 ≥ 0.08 m`, 2 if `p5 ≥ 0.02 m`, and 1 otherwise. For base 2 or 4, add one if median speed in the tightest clearance decile is at most 0.75 times overall median speed; otherwise reduce base 4 to 3. |
| social-distance proxy | 1 for person collision; otherwise 5 if no person appears or minimum distance is at least 1.2 m, 4 at 0.8–1.2 m, 3 at 0.45–0.8 m, and 2 below 0.45 m. |
| progress | From net goal-distance reduction per second: base 5 at `≥ 0.4 m/s`, 4 at `≥ 0.2`, 3 at `≥ 0.05`, 2 at `> -0.05`, and 1 otherwise. Reaching `≤ 0.4 m` raises the score to at least 4 and to 5 when rate is `≥ 0.2 m/s`; not reaching caps it at 4. Subtract one if speed is below 0.05 m/s for more than 40% of frames. |

The overall score is `min(safety, round_half_down(mean of four axes))`, then
clamped to the 1–5 ordinal range.

## Appendix C. OOD corpus composition

Seven GEAR-SONIC release videos are partitioned into non-overlapping four-second
clips, resized to 480 pixels wide, and encoded at 12 fps. Complete windows are
kept; for the sim2sim source only, a final tail of at least one second is kept
and padded by repeating its final frame. Regime names are fixed from the source
demonstration, not inferred from critic output.

| regime | n | release source(s) | interpretation |
| --- | ---: | --- | --- |
| clean walk | 10 | `Navigation.mp4`, `planner_stealth.mp4`, `planner_happy.mp4` | in-the-wild navigation and stylized walking |
| impaired gait | 1 | `planner_injured.mp4` | deliberately limping walk; registered ordering diagnostic |
| crawl | 2 | `hand_crawling.mp4` | hand-crawling demonstration |
| posture change | 2 | `planner_kneeling.mp4` | kneeling demonstration |
| separate sim2sim domain | 27 | `sim2sim.mp4` | official G1 sim2sim loop, outside the traversal-scene distribution |

The corpus contains no collision or fall examples and is not an accuracy test.

## Appendix D. Navigation-policy and PPO specification

**Observation and policy.** The environment renders 256 × 256 chase-camera
RGB; the policy resizes it to 84 × 84 by area interpolation and scales pixels
to [0,1]. A 10-dimensional vector contains clipped goal distance divided by
8 m, goal bearing divided by π, and the previous eight-dimensional command.
The recurrent-free actor–critic has three ReLU convolutional layers
(3→16, kernel 8, stride 4; 16→32, kernel 4, stride 2; 32→32, kernel 3,
stride 1), a shared 256-unit ReLU layer, and separate continuous-action,
mode-logit, and value heads. It has 427,099 trainable parameters.

The action comprises a Gaussian heading/speed pair before squashing and a
categorical mode. Tanh maps heading to [-1,1], which the environment scales to
±60°; sigmoid maps speed to [0,1], scaled to 0–1.2 m/s. The six categorical
SONIC modes are idle (0), slow walk (1), walk (2), run (3), crouch (22), and
hand crawl (8). One action is held for 15 controller ticks at 50 Hz, giving a
3.33 Hz policy rate. Held-out evaluation uses the continuous means and modal
argmax deterministically.

**Frozen PPO settings.** All arms use the same values.

| setting | value |
| --- | ---: |
| rollout length | 512 policy steps |
| minibatch / epochs | 64 / 4 |
| optimizer / learning rate | Adam / 3 × 10⁻⁴, constant |
| discount `γ` / GAE `λ` | 0.99 / 0.95 |
| policy clip | 0.2 |
| value-loss coefficient | 0.5 |
| entropy coefficient | 0.01 |
| gradient-norm cap | 0.5 |
| target KL / learning-rate annealing | disabled / disabled |
| episode limit | 30 s (100 policy decisions unless terminated early) |
| training budget | 300,032 executed steps at the complete-rollout boundary |

Advantages are normalized per rollout. Time-limit truncations bootstrap the
recorded next-state value while cutting GAE across the episode boundary.
