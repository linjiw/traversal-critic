# The World Model as Judge

**Learned traversal rewards for humanoid navigation in cluttered, occupied indoor spaces.**

> We turn a video world model into a **judge** — not a simulator, not a policy — and use it
> as the reward model for training humanoid navigation in cluttered, occupied indoor spaces.

🌐 **[Project page](https://linjiw.github.io/traversal-critic/)** ·
📄 **[Draft paper (PDF)](paper/draft.pdf)** ·
🗒️ **[Pitch kit](paper/pitch.md)**

## The idea

Humanoids can walk — but they can't cross a cluttered living room with people in it,
because nobody can *write down the reward* for "squeeze through that gap politely, duck
under the table, don't crowd the person." We fine-tune a compact (2B) video-language world
model (Cosmos3-Edge) into a **traversal critic** that scores rollout clips 1–5 on collision
safety, clearance, motion quality, and social compliance — supervised entirely by privileged
simulator state (no human labels) — and use it as reward shaping for a small vision policy
over a frozen GEAR-SONIC whole-body controller. The world model **never deploys**.

![system](assets/fig1_system.png)

## Results so far

| claim | number |
|---|---|
| Fine-tuning extracts signal from pixels | Pearson **0.148** near-base → **0.534** best critic (one fixed 590-clip held-out-scene yardstick, strictly monotone across interventions) |
| Balancing fixes the rare scores | GT-1/GT-5 recall roughly **doubles** |
| **Critic shaping works** | held-out success: baseline **5%** → critic-shaped **39%** (oracle upper bound 53%) — kinematic env; physics rerun in flight |
| It transfers to real robot video | **15 real G1 clips** + 27 cross-sim: 0 parse failures, all in the correct band (transfer without collapse; no real negatives yet) |
| Hand-crafted rewards fail measurably | baseline PPO overfits: 0.51 train success → 0.05 held-out |

![scaling](assets/fig2_scaling.png)

## Repository layout

This repo hosts the project page and paper. The full implementation (rollout generator,
auto-labeler, critic SFT recipes, eval sweeps, PPO trainer, async scorer daemon) lives on
the `feat/traversal-critic` branch of our cosmos-framework fork — see the paper's
reproducibility section.

## Acknowledgments

Built on [Cosmos3](https://github.com/nvidia-cosmos/cosmos-framework) (world model) and
[GEAR-SONIC / GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
(whole-body control). Real-robot footage in Fig. 6 is from the GEAR-SONIC release media.
