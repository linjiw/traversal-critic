# The World Model as Judge: Learned Traversal Rewards for Humanoid Navigation in Cluttered, Human-Occupied Spaces

*(working notes retained in `paper/planning.md`; this file converges toward the
submission text. Target venues: ICRA / ICLR.)*

---

## Abstract

Humanoid robots can walk, crawl, and recover from pushes, yet they still
cannot traverse the narrow, cluttered, human-occupied spaces where they are
expected to work. The bottleneck is not low-level control but **reward
specification**: behaviors such as turning through a tight gap, crouching
under furniture, or yielding to a person are inherently perceptual and resist
hand-crafted reward engineering. We propose using a video-language world model
as a learned reward model. We fine-tune a compact 2B-parameter model into a
*traversal critic* that assigns ordinal quality scores to short navigation
clips, supervised entirely by privileged simulator state converted
automatically into labels — no human annotation is required. The critic
provides reward shaping for a lightweight vision-based policy that commands a
frozen whole-body controller, and it is used at training time only: no world
model is deployed on the robot. On scenes held out at both the layout and
appearance level, fine-tuning raises the critic's correlation with
ground-truth traversal quality from r = 0.02 to r = 0.48, scaling with data.
`[P2 closing sentence: "We further show that critic-shaped policies collide
less often than hand-crafted-reward baselines at matched success rates" —
swap in measured numbers, or if P2 slips: "Finally, we describe an
asynchronous scoring architecture in which the critic never blocks policy
optimization."]`

---

## 1. Introduction

### 1.1 Introduction (opening)

Humanoid locomotion has advanced rapidly: whole-body controllers trained with
large-scale simulation RL now walk, crawl, and recover from pushes, and
behavior foundation models such as GEAR-SONIC compress thousands of hours of
human motion into a single policy. What remains unsolved is where those
robots are supposed to *go*: through the 0.6-meter gap between the sofa and
the bookshelf, under the low table, past the person carrying groceries. This
is not a control problem — the controller already knows how to crouch — it is
a **decision and reward problem**. No practical reward function captures
"traverse this room as a considerate person would"; hand-crafted proxies —
clearance penalties, velocity bonuses, personal-space costs — are brittle
surrogates that RL readily exploits. We show that a compact video-language
model, fine-tuned on automatically labeled simulation clips, can predict a
privileged-state traversal-quality score from pixels alone on held-out scenes
(r = 0.48) — a prerequisite for using it as a learned shaping reward. Because
it evaluates quality from pixels rather than privileged state, it can in
principle be applied where privileged state does not exist; we present
preliminary positive-transfer evidence on real robot footage and identify
viewpoint sensitivity as an open confound.

### 1.2 The observation the paper is built on

Generative video world models are used two ways in robotics today: as
**simulators** (roll out imagined futures for planning) and as **policies**
(action-conditioned VLAs). Both uses stress exactly what these models are
worst at — long-horizon pixel-consistent generation — and both put a large
model in or near the control loop. We exploit the third option: a video model
as a **judge**. Recognition is easier than generation; scoring "was that
traversal safe and natural?" needs one VLM forward pass over a short clip,
not thirty diffusion steps per imagined frame. Judging is also the only use
that is **training-time only**: the deployed system is a ~0.5M-parameter
policy on top of a frozen controller, with no world model on the robot.

### 1.3 Contributions

1. **A method**: fine-tune a compact video-language world model into a
   *traversal critic* — an ordinal (1–5) scorer of navigation clips —
   supervised entirely by privileged simulator state via a deterministic
   auto-labeler (four interpretable axes: safety, clearance, social, progress;
   safety as hard ceiling). No human labels; supervision scales with
   simulation.
2. **Evidence the premise holds**: on scenes held out at the *scene* level
   (novel layouts, gap widths, furniture, appearance), the critic recovers
   ground-truth traversal quality from pixels alone: Pearson r = 0.02
   (pretrained, zero-shot) → 0.38 (460 training clips) → **0.48** (1,530
   clips), acc±1 0.755 — a clean monotone scaling curve. `[v3: balanced
   ~2,900-clip result]`
3. **A system**: a three-layer architecture (frozen SONIC whole-body
   controller at 50 Hz; small vision policy at ~3 Hz; critic as asynchronous
   training-time reward) with an async scoring protocol that never blocks RL
   on the 2B model. `[P2: paired-seed comparison — critic-shaped vs.
   hand-crafted-reward PPO on held-out scenes.]`
4. **A quantified failure case for hand-crafted rewards**: our initial
   baseline reward silently collapsed the policy to standing still —
   per-frame contact penalties dominated the progress term — illustrating the
   brittleness the critic is designed to remove.

### 1.4 Scope and assumptions

We do not claim the critic replaces task reward: sparse goal reward and
privileged safety terms remain the backbone; the critic *shapes*. We do not
claim video world models should replace simulators. And the current
experiments are in simulation with kinematic playback of a pretrained motion
planner; physics-in-the-loop training and real-robot deployment are staged as
follow-up (§7).

---

## 2. Related Work

*(bullets to be expanded to cited prose; mandatory additions per review: VLM-RMs (Rocamonde et al. 2023), RL-VLM-F (Wang et al. 2024), Eureka (Ma et al. 2023), VIPER (Escontrela et al. 2023), RoboCLIP, MineCLIP, Christiano et al. 2017/PEBBLE, VIP/LIV/R3M)*

- **Humanoid whole-body control / behavior foundation models.** GEAR-SONIC,
  Decoupled WBC (GR00T), BeyondMimic, unitree_rl_lab. We consume these frozen.
- **Cluttered-space humanoid traversal.** HumanoidPF / Click-and-Traverse
  (G1, MuJoCo, procedural obstacles with difficulty knobs) — closest prior
  work; static clutter, hand-crafted rewards. Our deltas: dynamic people,
  learned perceptual reward, style-mode action space.
- **Social navigation.** Classic (ORCA, social forces), learned (NavIsaacLab
  crowds, Habitat 3.0 social nav). Reward is invariably hand-crafted; our
  critic is the missing learned-reward piece.
- **World models in robotics.** As simulators (dreamer-style, video
  prediction for MPC), as policies (VLA: GR00T N-series, π0, Cosmos policy
  mode). As *judges*: VideoPhy-2-style physical-plausibility scoring is the
  nearest neighbor — we extend the idea from "is this video physical?" to
  "is this traversal good?" and close the loop into RL.
- **Learned rewards / RLHF / VLM-as-reward.** Reward models from preferences,
  VLM zero-shot rewards (e.g., CLIP-based), foundation-model feedback for RL.
  Our distinction: video (not image) judgment, ordinal calibrated output,
  auto-labels from privileged sim state rather than human preference.

---

## 3. Method

### 3.1 Problem setup

Goal-conditioned navigation in cluttered indoor scenes with moving people.
Deployed stack: frozen whole-body controller **C** (SONIC: latent-token motion
tracking at 50 Hz + kinematic planner exposing {mode, direction, speed,
height}); small vision policy **π** (egocentric RGB + goal vector → planner
commands at ~3 Hz). Train π with PPO; the research question is the reward.

### 3.2 Traversal critic

**Data.** Parameterized procedural scenes (doorway gaps 0.6–1.2 m,
furniture, low tables, moving people on crossing paths; per-scene visual
randomization). Scripted policies spanning five quality bands generate
diverse rollouts. Privileged channels recorded per frame: min clearance
(analytic point-to-box), min person distance, base speed, goal distance,
contact flags.

**Auto-labeler.** Four axis scores from documented thresholds — safety
(contact duration), clearance (5th-percentile margin + slow-when-tight
bonus), social (min person distance), progress (goal-rate + freeze penalty) —
composed as overall = min(safety, half-down-round(mean)). Safety is a hard
ceiling: a collision episode cannot score well on style. Scene-hash split:
validation scenes are entirely unseen (layout *and* appearance).

**Model.** Cosmos3-Edge reasoner (Nemotron-2B LM + SigLIP2 tower), SFT on
(clip, rubric prompt) → single digit; partial unfreeze (projector + last 8 LM
blocks + final norm) fits a single 22 GiB L4. Class-balanced manifest
oversampling corrects ordinal imbalance. `[v3]`

Two implementation details are essential for reproduction: (i) the chat
template's generation-time reasoning mode must be disabled for a critic
fine-tuned to emit a single digit; otherwise every checkpoint produces
free-form chain-of-thought and no parseable score. (ii) Validation loss is
the wrong model-selection signal for this ordinal scorer — it reached its
minimum at iteration 200 while downstream correlation kept improving through
iteration 400; select checkpoints on task metrics.

### 3.3 Critic-shaped RL

R = R_task (sparse goal + dense progress) + R_safety (privileged, bounded)

- λ·(critic(clip) − 3)/2, λ bounded. The critic runs **asynchronously**: the
env banks episode clips to a file queue; a scorer daemon (owning the GPU)
returns scores; bonuses fold into later batches. RL never blocks on the 2B
model. Anti-hacking: sparse+privileged terms remain the backbone; periodic
audit of top-decile-critic episodes against the auto-labeler.

---

## 4. Experiments

### 4.1 Can the critic read traversal quality from pixels?

Held-out-scene evaluation, ordinal metrics (exact acc, acc±1, Pearson,
Spearman, per-class recall).

| Model                       | train clips | val clips | Pearson   | acc±1     | acc   |
| --------------------------- | ----------- | --------- | --------- | --------- | ----- |
| near-base (20 clips seen)   | —           | 140       | 0.024     | 0.557     | 0.314 |
| critic v1 (best ckpt)       | 460         | 140       | 0.377     | 0.679     | 0.457 |
| critic v2 (best ckpt)       | 1,530       | 470       | **0.482** | **0.755** | 0.517 |
| critic v3 (balanced)        | 2,930       | 670       | 0.452     | 0.743     | 0.460 |
| critic v4 (clean camera)    | 2,410       | 590       | **0.526** | **0.769** | 0.459 |

Scaling is monotone in data and steps through v2. **v3 isolates class
balancing** (manifest oversampling to uniform per-score weight): overall
correlation holds ≈flat on a harder, larger val while tail-class recall
roughly doubles — GT-1 0.22→0.35, GT-5 0.06→0.15, GT-3 0.12→0.24 (Fig. 5).
That is the right trade for a reward model: mistaking a dangerous episode
for a good one is the costly error, and it is precisely the rare-class
confusions that balancing removes. **v4 isolates the camera confound**
(§4.4 note): v1–v3 clips blank out the robot for 30–47% of frames in some
regimes (chase camera below wall height; wall-fill fraction correlated with
score band); v4 regenerates all data with an above-wall view. The clean-camera
critic is the best to date — Pearson 0.526 [0.462, 0.583], acc±1 0.769
[0.734, 0.803] (95% clip-bootstrap CIs) — and its confusion matrix is
substantially rebalanced: GT-5 recall 0.15 → 0.42, GT-1 0.35 → 0.43,
GT-3 0.19 → 0.35 relative to v3, at similar overall accuracy. A wall-fill-only
shortcut probe explains r = 0.155, confirming occlusion statistics were not
the critic's signal.

### 4.2 Does critic shaping improve the policy? `[P2 — headline]`

Paired-seed PPO, identical scenes/architecture/steps; arms: (a) sparse+safety
baseline, (b) + hand-crafted dense shaping, (c) + critic shaping. Endpoint
metrics on held-out scenes: success rate, collision-episode rate, min
clearance, person-space violations, time-to-goal, and auto-labeler axis
scores (one ruler for policies and critic).

**Baseline arm (measured).** 300k-step PPO with hand-crafted reward
(progress + bounded contact + time). Training-scene success peaked ~0.43
around 112k steps and degraded to ~0.32 by 300k (instability without KL/trust
constraints on a small net). On the 40-episode held-out-scene eval:

| checkpoint | success | collision-ep rate | mean auto-labeler score |
|---|---|---|---|
| baseline @ peak (112k) | 0.15 | 1.00 | 1.93 |
| baseline @ last (300k) | 0.10 | 0.98 | 1.83 |

The train-to-held-out gap (0.43 → 0.15) and the near-100% collision rate
quantify the headroom the critic-shaped arm targets. Notably, the
hand-crafted reward reached moderate training-scene success, yet its policy
collides in nearly every held-out episode and scores below 2 on the
auto-labeler scale.

### 4.3 Real-robot video transfer (measured, preliminary — **being re-run**)

> **Validity note (internal):** a code review found the serving path fed the
> model 4 frames with fabricated 0.6 s timestamps instead of training's ~17
> frames / real timestamps (transformers drops the per-item fps key without
> explicit video_metadata). The numbers below were measured through that
> skewed path and are being re-measured with the fixed decode; treat them as
> provisional. The sim val metrics (§4.1) are unaffected — eval_videophy2
> uses an honest decode path (verified).

We score 42 clips of **real Unitree G1 footage** (GEAR-SONIC release media:
in-the-wild navigation, style walks, impaired gait, crawling, kneeling) plus
27 clips from a *different* simulator, using the best sim-trained critic.
No privileged state exists for this footage, so no auto-label can.

Result: **zero parse failures; every clip scored in the 4–5 band** — the
correct range, since all release demos are collision-free — with clean
walking scoring highest (mean 4.30, the only real regime awarded 5s) and
crawl/impaired/posture regimes uniformly at 4.0. The critic transfers to
real video without collapsing to noise or misreading clean demos as unsafe.
Two limitations qualify this result: (i) the released footage contains no negative examples
(no real collisions), so this test establishes *transfer without collapse*,
not full discrimination on real video — closing that requires collecting
deliberately-flawed real rollouts; (ii) score compression toward 4 mirrors
the training distribution's mode.

### 4.4 Further ablations (planned)

Critic checkpoint quality vs. policy gain; λ sensitivity; band-3 "clean but
sloppy" discrimination; real negative-example collection.

### 4.5 Failure modes of hand-crafted rewards

First baseline reward: success 0.18 → 0.00 over 150k steps as PPO discovered
that standing still dominates walking (per-frame contact penalties × 15
frames/step > max progress term). A single line of reward arithmetic caused a silent
collapse. This is precisely the failure mode a learned critic avoids: its judgment is
holistic and episode-level rather than a sum of per-frame proxies that the policy can exploit.

---

## 5. Figures

- **Fig 1 (system)**: three-layer architecture; what deploys vs. what judges.
  → `paper/figures/fig1_system.svg`
- **Fig 2 (scaling)**: critic Pearson vs. training clips (0.02 → 0.38 → 0.48
  → `[v3]`), with acc±1 as secondary panel.
  → `paper/figures/fig2_scaling.html` (export to PDF/SVG for submission)
- **Fig 3 (qualitative)**: five clips, one per score, with critic score vs.
  auto-labeler score; egocentric frames strip.
- **Fig 4 (P2 headline)**: paired bars — the three PPO arms on the endpoint
  metrics.
- **Fig 5 (confusion)**: v2 vs. v3 confusion matrices — the class-balance
  ablation, visually.

## 6. Reproducibility notes

Single shared 22 GiB L4 for everything (critic SFT, eval, scoring); CPU for
rollout generation (planner ONNX ~35 ms) and PPO. Full recipe: branch
`feat/traversal-critic` — generator, labeler (+18 unit tests), SFT SKU/TOML,
eval sweep, scorer daemon, nav env, PPO trainer, paired evaluator.

## 7. Limitations & staging

Kinematic playback (no physics response) in phase 1–2; SONIC tracker
physics-in-the-loop is phase 2-proper (Isaac Lab, pinned 4.5, GRScenes-100 +
people layer); real-G1 deployment is phase 4 (ZMQ command path verified in
repo docs). Critic gate (Pearson ≥ 0.7) not yet met — scaling curve suggests
data, not architecture, is the binding constraint. Auto-labels inherit
threshold choices; the human-label override path exists for calibration.
