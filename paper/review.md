# Review & Literature Positioning — "The World Model as Judge" draft

*Prepared 2026-08-03. Covers: draft critique, three literature sweeps (VLM-as-reward;
humanoid/social navigation systems; reward-model evaluation & hacking methodology),
consolidated positioning, prioritized experiment list, and open questions for the author.*

*Caveat: 2026 arXiv IDs (26xx.xxxxx) were retrieved by live search; load-bearing ones were
verified at abstract level only — spot-check full PDFs before citing.*

---

## 1. Verdict in three sentences

The idea is sound and the lane is real, but narrower than the draft claims: what is yours is
the **conjunction** — fine-tuned video-LM + ordinal safety-capped rubric + privileged-sim
auto-labels + humanoid traversal among people + training-time-only PPO shaping — plus the
generalization argument (pixels port to where privileged state doesn't). No single prior
paper has that combination; every pillar in isolation is taken. The paper lives or dies on
two experiments it doesn't yet have: the four-arm P2 comparison including an
**auto-labeler-direct reward arm**, and evidence the 2B VLM beats a from-scratch classifier
on the same labels.

---

## 2. Strengths of the current draft

- Crisp, memorable framing: "simulator / policy / judge — we take the third option";
  "recognition is easier than generation."
- Training-time-only + async scorer daemon is a genuine, reusable systems contribution.
- Honest failure reporting (§4.5 standing-still collapse, camera-confound disclosure,
  "what this paper is not") reads as mature. Keep this voice.
- The real-video transfer test is the right instinct — it directly supports the
  "pixels port, privileged state doesn't" thesis.

## 3. Load-bearing weaknesses

1. **Headline claim unproven.** Everything hinges on §4.2 [P2]. A reward-model paper
   without the downstream RL result is a probe/dataset paper. Decide the fallback story now
   in case P2 is neutral.
2. **The distillation objection is the kill-shot.** Labels are a deterministic function of
   privileged state; privileged-reward + pixel-policy in sim is standard (asymmetric
   actor-critic, Pinto et al. 1710.06542; Eureka 2310.12931 exploits privileged state and
   beat human rewards on 83% of tasks). The critic's only justification is generalization
   beyond sim. **P2 needs arm (d): auto-labeler score used directly as reward.** If critic ≈
   labeler in-sim, the story becomes "matches the oracle and additionally transfers" — which
   currently rests on 42 all-positive real clips.
3. **r = 0.48 needs an anchor — and the ceiling argument cuts against you.** G-Eval GPT-4 hit
   Spearman 0.51 vs humans; GVL's useful datasets sit at 0.5–0.8 rank-corr — so 0.48 looks
   comparable to SOTA judge alignment. But those ceilings are human label noise; **your labels
   are deterministic, so your ceiling is 1.0**. GVL (2411.04549) found low-correlation value
   signals actively *hurt* policy learning, clear benefit only above ~0.6. Mitigations:
   ordinal-appropriate metrics (Spearman, quadratic-weighted kappa, per-class confusion,
   safety-axis false-negative rate) and downstream PPO as the real test (PPE 2410.14872:
   static RM metrics don't predict post-RL outcomes).
4. **Camera confound (v4) unresolved.** 30–47% blank frames correlated with score band means
   v1–v3 numbers are partially contaminated. Landing v4 is the top experimental priority
   after P2.
5. **Statistical exposures.** Val set changes across v1/v2/v3 (140→470→670) — the "monotone
   scaling curve" isn't apples-to-apples; evaluate all checkpoints on one fixed final val
   set with bootstrap CIs. v3 Pearson dropped (0.48→0.45); frame balancing purely as a
   tail-recall / cost-sensitivity trade, not part of a monotone curve.
6. **Ordinal head has a documented failure mode aimed at your safety mechanism.**
   Central-tendency bias in MLLM ordinal scoring (2605.16386) erodes 1s and 5s — exactly the
   safety ceiling. Your own Fig. 5 shows it (GT-1 recall 0.35, GT-5 0.15 post-balancing;
   real-video scores compress to 4). Adopt Q-Align (2312.17090): text-defined level words +
   **expectation over level-token probabilities** instead of argmax digit. Probably the
   cheapest technical upgrade available.

---

## 4. Literature sweep A — VLM / video models as reward

### Must cite and differentiate (reviewers will check)

| Work | Why it's dangerous | Your delta |
|---|---|---|
| **Video-Language Critic** (Alakuijala et al., 2405.19988, TMLR 2025) | Name + mechanism collision: video-language reward model scoring clips, in-loop RL shaping, no human labels (temporal ranking) | Navigation vs manipulation; ordinal vs contrastive; privileged-state labels vs self-supervision; generative VLM backbone. Differentiate in the intro, not just related work |
| **Rewarding DINO** (2603.16978, 2026) | **Your supervision recipe**: distills privileged analytical sim rewards into a visual reward model (rank loss, frozen DINO), manipulation | Video-language backbone, ordinal head, humanoid navigation. Kills "no human labels" as a novelty pillar |
| **MVR** (2603.01694, ICLR 2026) | Video-based reward shaping for **humanoids** (HumanoidBench), frozen VLM video-text similarity, policy-invariance-preserving | Fine-tuned ordinal critic vs frozen similarity; traversal vs locomotion skills. Likely a **required baseline** |
| **SuccessVQA** (Du et al., 2303.07280) | Canonical fine-tune-a-VLM-into-a-judge; showed pretrained VLM beats bespoke RMs OOD | Ordinal vs binary; sim auto-labels vs human labels; closes the RL loop |
| **SOLE-R1** (2603.28730, 2026) | Fine-tuned video-language reasoning model as sole dense reward for online robot RL, benchmarks hacking robustness | Navigation vs manipulation; rubric labels from privileged state vs synthesized reasoning traces |
| **Eureka** (2310.12931) / **Text2Reward** (2309.11489) | Forces the "why not privileged state directly?" answer | Critic ports judgment to pixels; must be *proven* (arm d + transfer) |

### Standard related work (cite, one-line differentiate)

VIPER 2305.14343 (generative likelihood as reward — ancestor of the judge framing);
RL-VLM-F 2402.03681 (GPT-4V preferences → RM, manipulation); VLM-RMs/Rocamonde 2310.12921
(zero-shot CLIP reward, MuJoCo humanoid — reviewers expect you to beat a frozen-CLIP
baseline); RoboCLIP 2310.07899; VIP 2210.00030 / R3M 2203.12601 / LIV 2306.00958
(representation-distance rewards); Diffusion Reward 2312.14134; GenRL 2406.18043;
GVL 2411.04549 (zero-shot frontier-VLM value — expected baseline); VideoPhy-2 2503.06800
(ordinal video judge precedent); Cosmos-Reason1 2503.15558; VLA-RL 2505.18719 (auto
pseudo-labeled VLM RM); Self-Improving EFM 2509.15155 (steps-to-go auto-labels);
ReWiND 2505.10911; VLLR 2604.00055 (VLM reward + PPO + navigation, zero-shot);
Beyond Binary Preferences 2603.02232 (principled ordinal-regression loss — consider
adopting); Q-Align 2312.17090; GT-SVJ 2602.05202; GE-Sim 2.0 2605.27491 ("world judge" —
occupies your phrase); RoboAlign-R1 2605.03821; CATNAV 2603.22800 / G2-Nav 2607.16956
(deployment-time VLM traversability — your training-time-only contrast).

### Narrative gift

**Wan-R1** (2603.27866, 2026): multimodal reward models "fail catastrophically" for
navigation-adjacent video RL; verifiable task-metric grounding essential. Your sim-verified
ordinal labels are exactly that fix — use as motivation anchor.

### Saturated claims — stop making them

"VLM as reward for RL" · "fine-tuned VLM judge" (SuccessVQA 2023) · "no human labels"
(VLC, VIPER, Rewarding DINO, Self-Improving EFM) · "ordinal video judging" (VideoPhy-2,
Beyond Binary Preferences) · **"world model as judge"** (GE-Sim 2.0, RoboAlign-R1, GT-SVJ;
and a 2B discriminative clip-scorer arguably isn't a world model — no dynamics prediction
anywhere in the pipeline).

---

## 5. Literature sweep B — humanoid navigation & social nav systems

### The scenario is crowded; the reward is the open lane

At least six 2025–26 works put a G1 in cluttered indoor scenes, all with hand-crafted
rewards, none with moving people or a learned perceptual reward:

- **Click-and-Traverse / HumanoidPF** (Xue et al., 2601.16035) — **one paper, not two**
  (fix the related-work skeleton). Humanoid Potential Field + PPO specialists → DAgger
  generalist; G1; real deployment; released code. **The mandatory experimental comparison**:
  same scenarios, same robot class. Reviewers will ask why a learned VLM reward beats its
  hand-crafted PF reward.
- Gallant 2511.14625 (voxel-LiDAR end-to-end, overhead clutter); LP-NavOA 2606.23249;
  Perceptive Humanoid Parkour 2602.15827; TTT-Parkour 2602.02331; VR clutter benchmark
  2603.05993 (348 trajectories, 145 scenes — useful dataset citation);
  Humanoid Parkour Learning (Zhuang) 2406.10759; Adaptive Safety Margins 2607.18200
  (learned geometric critic, inference-time — closest "learned critic" flavor).

### Hierarchy is an established pattern — claim it as engineering, not contribution

ANYmal Parkour (Hoeller, 2306.14874) is the canonical high-level-policy-over-locomotion-
skills hierarchy; expect *"this is ANYmal Parkour on a humanoid with a VLM reward"* and
*"ANYmal Parkour + RoboCLIP glued together."* The rebuttal must be behavioral: show the
critic materially changes what the policy does (yielding, slowing in tight gaps, smoother
traversal) under the identical hierarchy. Also cite: SONIC 2511.07820 (your load-bearing
dependency — characterize its command interface; check for SONIC-downstream nav papers
before camera-ready, that's the scoop risk), HOVER 2410.21229, LeVERB 2506.13751,
ExBody 2402.16796 / ExBody2 2412.13196, HumanPlus 2406.10454, BeyondMimic 2508.08241,
Belli et al. 2607.24083, NaVILA 2412.04453 (closest VLA-on-legged-controller — contrast
0.5M params + RL vs billions + imitation), NaVid 2402.15852, Uni-NaVid 2412.06224.

### Social navigation — real prior art the draft misses

SACSoN 2306.01874 (learned counterfactual social objective — strongest precedent);
SocialNav foundation model 2511.21135 (SAFE-GRPO, explicit social-compliance reward);
SocialNav-MoE 2512.14757; IRL line (S-MEDIRL 2501.06946, SoLo T-DIRL 2209.07996, SCAND
2203.15041); Habitat 3.0 2310.13724; benchmarks SEAN 2009.04300, SocNavBench 2103.00047
(reviewers will expect these metrics, not min-person-distance alone); ORCA / social forces
for crowd simulation. **Either build one social-compliance behavioral result or retreat
"occupied" to a secondary axis.**

Benchmarks/sims: GRUtopia/GRScenes 2407.10943; ProcTHOR 2206.06994; MetaUrban 2407.08725.

---

## 6. Literature sweep C — evaluation methodology & reward hacking

### Reward hacking (objection: "frozen RM + PPO = Goodhart; half your variance is noise")

- Gao et al. 2210.10760 — *the* overoptimization citation. **You have the gold reward in
  sim: plot gold rubric score vs PPO steps (and vs KL from init) during critic-shaped
  training. Nearly free; its absence would be conspicuous; a clean curve single-handedly
  answers the objection.**
- Skalse 2209.13085 (hackability theory); Ibarz 1811.06521 (earliest learned-RM
  exploitation; fix = on-policy relabel + retrain — your analogue: relabel PPO rollouts
  with the rubric, fine-tune the critic); VICE/RAQ 1904.07854 (RL finds adversarial states
  fooling an image classifier trained on positives only).
- Mitigations to cite/ablate: ensembles (Coste 2310.02743), WARM weight averaging
  2401.12187 (cheap for a 2B model), constrained multi-axis RL (Moskovitz 2310.04373 —
  directly relevant to your 4-axis rubric), reward capping (Singhal 2310.03716),
  discretization reduces hacking (2606.21795 — incidental support for your ordinal design).
- Multimodal reward hacking study 2607.09492 (up to 48% hacking rate with outcome-only
  rewards, 2B–32B).

### What "good enough" looks like in prior work

VIPER and Video-Language Critic report downstream RL only, no correlation tables; RL-VLM-F
works despite ~60–80% preference accuracy; GVL's Value-Order Correlation: helps at 0.6–0.75,
*hurts* at 0.1–0.2. EPIC (Gleave 2006.13900) is the principled alternative reviewers from
RL theory may demand instead of raw Pearson.

### Ordinal scoring pitfalls

G-Eval 2303.16634 (probability-weighted scores; integer clustering); Q-Align 2312.17090
(level words + expectation > numeric SFT — **adopt**); Decoding-based Regression 2501.19383;
Prometheus-Vision 2401.06591; central-tendency bias 2605.16386; Likert compression/tie
inflation 2509.24678; conformal intervals for judge scores 2509.18658.

### Distillation critique — both attack branches, precedented

(a) Privileged reward directly: asymmetric actor-critic standard (Pinto 1710.06542, theory
2412.00985, drone racing 2406.12505, Learning by Cheating 1912.12294). (b) From-scratch
classifier on same labels: SuccessVQA shows pretraining wins OOD but is not param-matched;
**nobody has run the exact 2B-VLM vs from-scratch-video-CNN-on-identical-auto-labels
comparison — running it is a publishable finding either way.**

---

## 7. Repositioned contribution statement (proposal)

> Hand-crafted rewards for humanoid traversal in occupied spaces fail measurably (our §4.5;
> Click-and-Traverse's reward is collision-geometric only); privileged-state rewards don't
> leave the simulator. We show a compact video-language critic, auto-supervised in
> simulation, ports traversal judgment to pixels — matching privileged reward
> in-distribution and remaining usable where no privileged state exists — at zero
> deployment cost.

Note: this commits to the labeler-direct comparison (arm d). That is now unavoidable.

**Title:** drop "world model" from the headline. Candidate #3 (*"Traversal Critics: Turning
Video Foundation Models into Navigation Rewards"*) is now the strongest; #1 the weakest.

---

## 8. Prioritized experiment list (by paper-survival value)

1. **P2 with four arms**: sparse+safety / hand-crafted dense / **auto-labeler-direct** /
   critic-shaped. The hinge of the paper.
2. **Gao-style overoptimization curve** during arm (d) — nearly free.
3. **v4 clean-camera numbers** — v1–v3 are contaminated until this lands.
4. **Param-matched from-scratch (or frozen-SigLIP+probe) reward baseline** on identical
   labels and splits; report generalization gap vs label count.
5. **Q-Align-style expectation head** + per-level reliability diagrams + safety-FN stress
   set (near-threshold clips, ROC of the safety-ceiling trigger).
6. **Real negative examples** (~20 deliberately flawed G1 clips) — upgrades transfer from
   "doesn't collapse" to "discriminates."
7. **Fixed final val set for all checkpoints + bootstrap CIs.**
8. (If time) binary-vs-ordinal head ablation; preference-head (Bradley-Terry) ablation;
   hard-ceiling vs additive aggregation ablation; MVR-style frozen-similarity baseline;
   Habitat-3.0/SocNavBench-style social metrics.

If compute-limited: cut the social axis and the second simulator before cutting arms (d)
or the from-scratch baseline — those two decide acceptance.

---

## 9. Open questions for the author

1. **Venue follows from arm (d)'s expected outcome.** Critic merely matches labeler-direct
   in-sim → ICLR-style reward-model study with transfer as the payoff. Critic beats
   hand-crafted in a regime where no labeler exists → ICRA systems paper. Which do you
   actually expect from what you've seen in training?
2. **Can you produce a "critic right, labeler wrong" figure?** Clips where thresholds
   misjudge (lurching-but-clear gait; forgiven near-miss) and the critic scores sensibly.
   The one artifact that turns "distillation" into "judgment" for a reviewer — and no
   currently planned experiment produces it.
3. **If P2 is neutral or negative, what is the fallback story?** Decide before investing.
4. **Why ordinal 1–5 rather than pairwise preferences?** Preferences are also free from the
   labeler, standard in reward modeling, and dodge calibration. If no principled reason, a
   preference-head ablation preempts the question.
5. **Social compliance: invest or retreat?** It's in the title but the evidence is one
   min-distance threshold.
