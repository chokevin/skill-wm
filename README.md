# skill-wm

A testbed for the question: **can a small, separately trained world model improve LLM agents — or are LLMs already good enough as their own world model?**

This repo is intentionally small. It is the **prediction-only smell test** for the thesis direction "learned world models can make LLM agents more reliable on long-horizon tasks." It deliberately avoids the agent loop, deliberately avoids Minecraft, and deliberately avoids JEPA — those come only after the prediction premise is validated here.

## Thesis being tested

Recent work (Wang et al., TMLR 2025, [arXiv:2411.08794](https://arxiv.org/abs/2411.08794)) shows that **using LLMs as world models** (LLM-as-WM) degrades on long-horizon tasks and becomes unstable when combining functions. The natural counter-design is to pair the LLM with a **separately trained, parameterized** outcome model.

This testbed asks the cheapest version of that question first:

> Given a state and an action in Crafter, can a small trained model predict the outcome (success / inventory delta / vitals delta) better-calibrated than a frontier LLM prompted zero-shot?

If the answer is no, the broader thesis is in trouble at its premise.
If the answer is yes, we have a foundation to build on.

## Tiered plan

| Tier | Scope | Goal |
|---|---|---|
| **T0** | one skill, one model | does *anything* about this work? |
| **T1** | full Crafter skill set, all baselines, calibration metrics | go/no-go gate for the thesis |
| **T2** | LLM agent uses WM scores at decision time | does better prediction translate to better agents? |

We are currently building toward T1.

## What's in here

```
skill_wm/
  envs/crafter_env.py     # CrafterWrapper that emits clean Transition records
  data/schema.py          # Transition dataclass, action_success, deltas
  data/collect.py         # Random / biased-random rollout collection
  models/                 # (T1) trained WM, LLM-as-WM baseline
  eval/                   # (T1) Brier, ECE, per-skill accuracy
scripts/
  collect_rollouts.py     # entry point: dump npz transitions to disk
tests/
  test_smoke.py           # end-to-end smoke test
configs/
data/rollouts/            # gitignored: collected transitions
```

## Setup

```bash
make install              # uv sync
make check                # lint + tests
make smoke                # 5-episode rollout end-to-end
make all                  # install + check + smoke
```

Or call uv directly:

```bash
uv sync
uv run pytest -q                    # run smoke tests
uv run ruff check .                 # lint
uv run python -m scripts.collect_rollouts --episodes 50 --max-steps 200
```

Optional, for the LLM-as-WM baseline (later):
```bash
cp .env.example .env                # add OPENAI_API_KEY
```

## Baselines we will compare in T1

| Baseline | What it predicts from |
|---|---|
| Random | predicts most-common outcome with chance probability |
| Marginal | predicts marginal frequency of success per action |
| Hand-coded preconditions | rule-based per-action precondition check |
| **LLM-as-WM (zero-shot)** | GPT-4o-mini given textual state, asked for outcome |
| **LLM-as-WM (few-shot)** | same, with a few in-context examples |
| **Trained WM (this work)** | small transformer/MLP trained on logged transitions |

Primary metric: **calibrated success prediction on held-out world seeds** (Brier score and ECE), broken out per-skill and per-horizon.

## What this repo is NOT

- not an agent
- not a Minecraft project
- not a JEPA paper
- not a publishable result on its own

It is a falsifiability harness for one claim. Once T1 finishes, we either advance to T2 with a working WM, or we kill the thesis cheaply.
