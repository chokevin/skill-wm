# Research: Existing comparables for object-aware Skill-JEPA hazard reranking

**Date:** 2026-05-03
**Asker:** Kevin / Skill-JEPA MiniHack thesis context
**Decision:** ADAPT

## Question

What existing methods are the closest comparables to our object-aware Skill-JEPA decision-time hazard reranker, and what empirical baselines/results should we compare against for a credible MiniHack/NLE paper?

## TL;DR

We have external data on close comparables, but not a direct published equivalent to "object-head NLL from a JEPA-style predictor reranks held-out MiniHack hazard actions." The right move is **ADAPT**: treat Dreamer/THICK/Continual-Dreamer as world-model baselines/related work, C-SWM as object-centric representation precedent, and shielding/action-filtering as the closest conceptual competitor.

## What I read

| Source | Type | Date | What it said (1 line) |
|---|---:|---:|---|
| [MiniHack the Planet](https://arxiv.org/abs/2109.13202) | paper | 2021 | MiniHack is explicitly for controlled RL testbeds, from small rooms to procedural worlds, with NetHack entities/dynamics. |
| [MiniHack README/docs](https://github.com/facebookresearch/minihack) | official-docs/code-repo | 2026 | MiniHack provides TorchBeast and RLlib baseline integrations and lists world-model papers using MiniHack. |
| [The NetHack Learning Environment](https://arxiv.org/abs/2006.13760) | paper | 2020 | NLE targets exploration, planning, skill acquisition, and language-conditioned RL; baseline data used distributed deep RL and RND. |
| [NLE README](https://github.com/facebookresearch/nle) | official-docs/code-repo | 2026 | NLE ships a TorchBeast starter agent and records related NLE papers/datasets. |
| [NetHack Challenge report](https://nethackchallenge.com/report.html) | benchmark-report | 2021 | Symbolic bots beat neural agents by a wide margin; no agent ascended in over half a million games. |
| [DreamerV3](https://arxiv.org/abs/2301.04104) / [repo](https://github.com/danijar/dreamerv3) | paper/code-repo | 2023-2025 | Dreamer learns a latent world model and trains an actor-critic from imagined trajectories across many domains. |
| [Continual-Dreamer](https://arxiv.org/abs/2211.15944) / [repo](https://github.com/skezle/continual-dreamer) | paper/code-repo | 2022-2023 | DreamerV2 variants were evaluated on Minigrid and MiniHack continual-task suites, including Plan2Explore/replay variants. |
| [THICK](https://openreview.net/forum?id=TjCDNssXKU) / [repo](https://github.com/CognitiveModeling/THICK) | paper/code-repo | 2024 | Hierarchical world models improve MBRL/planning and include MiniHack commands with DreamerV2 baselines. |
| [C-SWM](https://arxiv.org/abs/1911.12247) / [repo](https://github.com/tkipf/c-swm) | paper/code-repo | 2019-2020 | Contrastive object-centric world models learn object representations and relational dynamics from pixels. |
| [Safe RL via shielding](https://arxiv.org/abs/1708.08611) | paper | 2017 | A shield can provide safe action sets or correct an unsafe learner action at decision time. |
| [ADVICE adaptive shielding](https://openreview.net/forum?id=82VzAtBZGk) | paper | 2024-2025 | A contrastive autoencoder distinguishes safe/unsafe state-action features and post-shields hazardous actions. |
| [BALROG](https://arxiv.org/abs/2411.13543) / [leaderboard](https://balrogai.com/) | paper/leaderboard | 2025-2026 | LLM/VLM agents still struggle on dynamic games; leaderboard NetHack scores remain very low relative to easier games. |

(Read budget: 12 sources across 4 source types. Stopped because the answer was stable: no direct MiniHack JEPA hazard-reranking equivalent surfaced, but several strong comparator families did.)

## Findings

1. **MiniHack/NLE give us accepted benchmark surfaces, not a direct method match.** MiniHack is designed for controlled environments and scaling task complexity, and its docs expose TorchBeast/RLlib baselines. NLE adds distributed deep RL and RND as early baseline evidence. Sources: MiniHack paper/docs; NLE paper/docs.

2. **World-model baselines exist for MiniHack, but they optimize policies through imagination rather than exposing an object-level hazard score.** Continual-Dreamer has MiniHack training commands for DreamerV2 variants, and THICK has MiniHack experiments plus a default DreamerV2 baseline. DreamerV3 is the broader world-model control baseline. Sources: Continual-Dreamer; THICK; DreamerV3.

3. **Object-centric world models support the representation claim, but not the MiniHack decision-time claim.** C-SWM shows contrastive object-structured dynamics can outperform pixel reconstruction models in structured environments and learn interpretable object representations, but its reported environments are compositional toy/Atari/physics tasks rather than MiniHack hazard reranking. Source: C-SWM.

4. **Safety shielding is the closest conceptual competitor.** Classical shields filter or correct unsafe actions; ADVICE is especially close because it learns state-action safe/unsafe features via contrastive representation and post-shields actions. Our differentiator cannot be "we filter unsafe actions"; it must be "a self-supervised/object-predictive JEPA head supplies the hazard/OOD score without hand-coded temporal logic or a supervised unsafe-action oracle." Sources: safe RL shielding; ADVICE.

5. **Symbolic/rule baselines are mandatory, not optional.** The NetHack Challenge found symbolic bots clearly ahead of neural agents, and BALROG shows modern LLM/VLM agents still struggle on MiniHack/NetHack. A reviewer will expect an oracle/rule shield or symbolic lava-avoidance baseline to bound how much of our result is just known gridworld structure. Sources: NetHack Challenge report; BALROG.

## Counter-evidence

The strongest case against novelty is that our current reranker resembles an action shield: it detects that `east` is unsafe and substitutes a safe action. Shielding literature already formalizes both pre-shields that restrict the action set and post-shields that correct the selected action. ADVICE goes further by learning state-action safe/unsafe features with a contrastive autoencoder, which is uncomfortably close to "learn a representation, then filter hazardous actions."

The strongest case against sufficiency is THICK/Continual-Dreamer: both are published world-model methods with MiniHack-facing code, while our current data is a controlled lava interaction rather than a trained policy suite. That means the paper should not claim broad world-model superiority. It should claim a narrower contribution: object-predictive JEPA signals can expose held-out hazardous object interactions and can be used as a lightweight learned safety reranker in a controlled MiniHack setting.

## Decision: ADAPT

Proceed, but adapt the plan so the live rerank eval includes baselines that answer the obvious comparator objections. The method still looks distinct enough to study, but only if we frame it against shields, object-centric world models, and MiniHack world-model baselines rather than pretending there are no equivalents.

## What this means in practice

- **First concrete move:** add live MiniHack comparison rows for `lava_probe`, `oracle_object_shield`, `latent_mse_rerank`, and `object_head_rerank`; keep `scripted_nav` as the expert/reference path.
- **Watch-fors:** if object-head reranking only matches a trivial oracle lava mask on one hazard, add a second held-out object/hazard before making a paper claim.
- **Out of scope for this research:** reproducing DreamerV3/THICK/Continual-Dreamer locally; evaluating full NLE ascension; integrating an LLM planner.

## Open questions / what I'd read next

1. Do THICK or Continual-Dreamer report per-task MiniHack tables that include lava/trap-like hazards, or only aggregate continual-task scores?
2. Can we create a non-oracle learned shield baseline using the same training rows but no JEPA/object head?
3. Does the object head generalize to a new hazard glyph/object, or is it just learning an explicit lava tile detector?
