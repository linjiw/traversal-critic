# The World Model as Judge — pitch kit

*For advisor meetings, hallway explanations, and the paper's story spine.*
*Target venues: ICRA 2027 / ICLR 2027.*

## 30-second spoken pitch

> "Humanoids can walk now — but they still can't cross a cluttered living room
> with people in it. The blocker isn't balance, it's that nobody can *write
> down the reward* for 'squeeze through that gap politely, duck under the
> table, don't crowd the person.' Our idea: don't write it — **learn it from a
> video foundation model**. We fine-tune a video-language world model into a
> *traversal critic* that watches a rollout clip and scores it 1–5, then use
> that score as the reward signal to train a small navigation policy over a
> frozen whole-body controller. The critic is training-time only — nothing big
> deploys on the robot. And the key premise is validated: the critic scores
> unseen scenes from pixels alone, and its accuracy scales cleanly with data."

## One sentence

**We turn a video world model into a judge — not a simulator, not a policy —
and use it as the reward model for training humanoid navigation in cluttered,
occupied indoor spaces.**

## The three moves of the story

1. **The problem is the reward, not the robot.** Locomotion is solved-enough
   (behavior foundation models); traversal quality in occupied clutter is
   perceptual and unwritable. Our own quantified anecdote: our first
   hand-crafted reward silently collapsed PPO to standing still.
2. **Judging is the right job for a video model.** Simulator use fights
   generation consistency; policy use puts 2B+ in the control loop. Judging
   is one forward pass, async, training-time only — matched to what these
   models are reliable at (recognition > generation).
3. **The supervision is free.** Privileged sim state → deterministic ordinal
   labels (safety/clearance/social/progress, safety as hard ceiling). No
   humans. The critic then *generalizes the reward to pixels* — which is the
   answer to "why not use privileged state directly": privileged state
   doesn't exist outside sim; pixels do. (Killer follow-up experiment: score
   real-robot videos, where no labeler can exist.)

## Numbers that carry the pitch (current)

| claim                                   | number                                             |
| --------------------------------------- | -------------------------------------------------- |
| Base model has no signal                | Pearson **0.02** zero-shot                         |
| Fine-tuning extracts signal from pixels | **0.38** @ 460 clips                               |
| It scales                               | **0.48** @ 1,530 clips (acc±1 0.755), monotone     |
| Balancing fixes the tails               | GT-1/-5 recall roughly **doubles** at flat Pearson |
| **It transfers to real robot video**    | 42 real G1 clips: **0 parse fails, all in the correct 4–5 band**, clean walk ranked highest |
| Labels are free                         | ~6,000 clips auto-labeled, 0 human annotations     |
| Deployment is light                     | policy ~0.5M params; world model **never** deploys |
| Hand-crafted rewards fail measurably    | baseline PPO: held-out success 0.15, collisions in ~every episode |

## Anticipated objections, one-line answers

- *"Why not privileged state as the reward directly?"* It doesn't exist
  outside sim; the critic ports the judgment to pixels (→ real video, restyled
  domains) and inherits the video prior on qualities thresholds can't encode.
- *"Isn't the critic just distilling your labeler?"* On seen distributions,
  partially — the value is generalization to unseen scenes (measured) and
  unlabelable domains (planned), plus robustness to the labeler's blind spots.
- *"Reward hacking?"* Sparse task + privileged safety stay the backbone; λ is
  bounded; top-decile critic episodes are audited against the labeler.
- *"Kinematic sim only?"* Phase-staged: physics-in-the-loop (Isaac Lab +
  SONIC tracker) then real G1; the critic itself is sim-agnostic — it reads
  video.

## Figures

- `figures/fig1_system.svg/.png` — architecture: training loop vs. deployment;
  the critic never crosses the line.
- `figures/fig2_scaling.svg/.png` — the scaling curve, gates marked.
- Fig 3 (planned): 5 clips, one per score — critic vs. labeler.
- Fig 4 (headline, pending P2): three PPO arms on endpoint metrics.

## Title candidates

1. *The World Model as Judge: Learned Traversal Rewards for Humanoid
   Navigation in Cluttered, Occupied Spaces*
2. *Watch and Score: Video World Models as Reward Models for Humanoid
   Navigation*
3. *Traversal Critics: Turning Video Foundation Models into Navigation
   Rewards*
