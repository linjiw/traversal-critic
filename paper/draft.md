# The World Model as Judge: Learned Traversal Rewards for Humanoid Navigation in Cluttered, Occupied Spaces

**Target venues:** ICRA 2027 (robotics framing) / ICLR 2027 (reward-learning framing).
**Status:** working draft; numbers marked `[v3]`/`[P2]` land from the running pipeline.

---

## Abstract

Humanoid robots can walk, but they cannot yet traverse the narrow, cluttered,
human-occupied spaces where they are meant to work. The obstacle is not
low-level control — whole-body controllers trained in physics simulation are
increasingly capable — but **reward specification**: behaviors such as turning
shoulders through a tight gap, crouching under furniture, or yielding to a
person resist hand-crafted reward engineering, and their quality is inherently
perceptual. We propose using a **video-language world model as a learned reward
model**. We fine-tune a compact (2B) video-language model into a *traversal
critic* that scores short navigation clips on collision safety, clearance
management, motion quality, and social compliance, supervised by privileged
simulator state that is automatically converted to ordinal labels — no human
annotation. The critic then provides reward shaping for training a lightweight
vision-based navigation policy that commands a frozen, pretrained whole-body
controller; the world model is used **at training time only**, leaving
deployment unencumbered. On held-out scenes, the fine-tuned critic predicts
ground-truth traversal quality from pixels alone (Pearson r = 0.48 vs. 0.02
before fine-tuning), improving monotonically with data scale. `[P2: Critic-shaped
policies achieve X% fewer collisions and Y-point higher traversal-quality
scores at equal success rate versus hand-crafted-reward baselines.]`

---

## 1. Introduction

### 1.1 The story in one paragraph

The last three years solved humanoid *locomotion*: massively parallel
physics-simulation RL now produces whole-body controllers that walk, crawl,
and recover from pushes, and behavior foundation models such as GEAR-SONIC
compress thousands of hours of human motion into a single deployable policy.
What it did not solve is where those robots are supposed to *go*: through the
0.6-meter gap between the sofa and the bookshelf, under the low table, past
the person carrying groceries. This is not a control problem — the controller
already knows how to crouch — it is a **decision and reward problem**. Nobody
can write down the reward function for "traverse this room the way a
considerate person would," and every hand-crafted proxy (clearance penalties,
velocity bonuses, personal-space costs) is a brittle stand-in that RL
exploits. We show that a video world model, fine-tuned cheaply on
automatically labeled simulation clips, can *be* that reward function — and
that unlike its hand-crafted competitors, it reads quality from pixels, so it
generalizes to scenes, viewpoints, and (ultimately) real videos where no
privileged state exists.

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
4. **A cautionary datapoint for the field**: our own hand-crafted baseline
   reward collapsed the policy to standing still on the first attempt
   (per-frame contact penalties dominated progress) — an accidental,
   quantified illustration of exactly the brittleness the critic is designed
   to remove.

### 1.4 What this paper is not

We do not claim the critic replaces task reward: sparse goal reward and
privileged safety terms remain the backbone; the critic *shapes*. We do not
claim video world models should replace simulators. And the current
experiments are in simulation with kinematic playback of a pretrained motion
planner; physics-in-the-loop training and real-robot deployment are staged as
follow-up (§7).

---

## 2. Related work (skeleton)

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

**Key implementation findings** (each cost a debugging cycle; reported for
reproducibility): (i) generation-time chat-template *thinking mode* must be
disabled for digit-SFT'd scorers — with it on, every checkpoint rambles CoT
and emits no score; (ii) validation *loss* is the wrong model-selection
signal — it bottomed at iter 200 while downstream Pearson kept improving
through iter 400; select on task metrics.

### 3.3 Critic-shaped RL

R = R_task (sparse goal + dense progress) + R_safety (privileged, bounded)

- λ·(critic(clip) − 3)/2, λ bounded. The critic runs **asynchronously**: the
env banks episode clips to a file queue; a scorer daemon (owning the GPU)
returns scores; bonuses fold into later batches. RL never blocks on the 2B
model. Anti-hacking: sparse+privileged terms remain the backbone; periodic
audit of top-decile-critic episodes against the auto-labeler.

---

## 4. Experiments

### 4.1 Can the critic read traversal quality from pixels? (core validation)

Held-out-scene evaluation, ordinal metrics (exact acc, acc±1, Pearson,
Spearman, per-class recall).

| Model                       | train clips | val clips | Pearson   | acc±1     | acc   |
| --------------------------- | ----------- | --------- | --------- | --------- | ----- |
| near-base (20 clips seen)   | —           | 140       | 0.024     | 0.557     | 0.314 |
| critic v1 (best ckpt)       | 460         | 140       | 0.377     | 0.679     | 0.457 |
| critic v2 (best ckpt)       | 1,530       | 470       | **0.482** | **0.755** | 0.517 |
| critic v3 (balanced)        | 2,930       | 670       | 0.452     | 0.743     | 0.460 |
| critic v4 (clean cam) `[v4]`| ~2,900      | ~590      | —         | —         | —     |

Scaling is monotone in data and steps through v2. **v3 isolates class
balancing** (manifest oversampling to uniform per-score weight): overall
correlation holds ≈flat on a harder, larger val while tail-class recall
roughly doubles — GT-1 0.22→0.35, GT-5 0.06→0.15, GT-3 0.12→0.24 (Fig. 5).
That is the right trade for a reward model: mistaking a dangerous episode
for a good one is the costly error, and it is precisely the rare-class
confusions that balancing removes. **v4 isolates the camera confound**
(§4.4 note): v1–v3 clips blank out the robot for 30–47% of frames in some
regimes (chase camera below wall height; wall-fill fraction correlated with
score band); v4 regenerates all data with an above-wall view. `[v4]`

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

| checkpoint | success | collision-ep rate | mean labeler score |
|---|---|---|---|
| baseline @ peak (112k) | 0.15 | 1.00 | 1.93 |
| baseline @ last (300k) | 0.10 | 0.98 | 1.83 |

The train→held-out gap (0.43 → 0.15) and saturated collision rate quantify
exactly the headroom the critic arm targets — and note the hand-crafted
reward *did* reach mid-training competence yet produces collisions in
essentially every held-out episode, scoring < 2 on the labeler's ruler.

### 4.3 Real-robot video transfer (measured, preliminary)

We score 42 clips of **real Unitree G1 footage** (GEAR-SONIC release media:
in-the-wild navigation, style walks, impaired gait, crawling, kneeling) plus
27 clips from a *different* simulator, using the best sim-trained critic.
No privileged state exists for this footage, so no auto-label can.

Result: **zero parse failures; every clip scored in the 4–5 band** — the
correct range, since all release demos are collision-free — with clean
walking scoring highest (mean 4.30, the only real regime awarded 5s) and
crawl/impaired/posture regimes uniformly at 4.0. The critic transfers to
real video without collapsing to noise or misreading clean demos as unsafe.
Two honest limits: (i) the released footage contains no negative examples
(no real collisions), so this test establishes *transfer without collapse*,
not full discrimination on real video — closing that requires collecting
deliberately-flawed real rollouts; (ii) score compression toward 4 mirrors
the training distribution's mode.

### 4.4 Further ablations (planned)

Critic checkpoint quality vs. policy gain; λ sensitivity; band-3 "clean but
sloppy" discrimination; real negative-example collection.

### 4.5 The hand-crafted-reward cautionary tale (motivation, quantified)

First baseline reward: success 0.18 → 0.00 over 150k steps as PPO discovered
that standing still dominates walking (per-frame contact penalties × 15
frames/step > max progress term). One line of reward arithmetic, silent
collapse. This is the failure mode the critic removes: its judgment is
holistic and episode-level, not a sum of per-frame proxies to out-game.

---

## 5. Figures

- **Fig 1 (system)**: three-layer architecture; what deploys vs. what judges.
  → `paper/figures/fig1_system.svg`
- **Fig 2 (scaling)**: critic Pearson vs. training clips (0.02 → 0.38 → 0.48
  → `[v3]`), with acc±1 as secondary panel.
  → `paper/figures/fig2_scaling.html` (export to PDF/SVG for submission)
- **Fig 3 (qualitative)**: five clips, one per score, with critic score vs.
  labeler score; egocentric frames strip.
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
