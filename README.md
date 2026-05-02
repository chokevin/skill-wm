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
experiments/              # rune-py jobs for the voice-agent-flex AKS cluster
  collect_rollouts/config.py   # parallel rollout collection across N pods
  train_wm/config.py           # (stub) train the trained-WM checkpoint
  eval_baselines/config.py     # (stub) score trained-WM vs LLM-as-WM
bin/
  setup.sh                # one-shot: az login + kubeconfig + venv + rune CLI
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

## Running on Rune (voice-agent-flex)

This repo is small enough to run end-to-end on a laptop. Rune is used when we want **parallel rollout collection** across many pods, and (later) when the trained-WM model needs a real GPU for training.

One-time setup:

```bash
make rune-setup     # az login, voice-agent-flex kubeconfig, uv sync (incl. rune-py)
```

`bin/setup.sh` targets `voice-agent-flex` in `voice-agent-flex-rg`, namespace `ray`. Override via `SKILL_WM_CLUSTER_NAME` / `SKILL_WM_CLUSTER_RG` / `SKILL_WM_NS` if needed.

You also need the `rune` CLI on your `$PATH`. The script tells you how to install it from `azure-management-and-platforms/aks-ai-runtime` if missing.

### Sanity-check the rune wiring locally (no cluster)

The `@rune.train` / `@rune.eval` handles are callable directly — they synthesize a `Ctx` rooted at `cwd` and run the function in this process. No GPU scheduling, no Kueue admission, no manifest applied:

```bash
make rune-collect-local                            # runs collect_rollouts() in-process
SKILL_WM_TOTAL_EPISODES=20 make rune-collect-local
```

Output lands under `./skill-wm-collect-smoke/rank-000/ep_*.npz` (cwd-rooted because `is_remote=False`).

### Render the cluster manifest (no submit)

`--dry-run` shells to the rune Go CLI in `client` mode — validates the manifest and prints the rendered RayJob without applying anything:

```bash
make rune-collect-dry
make rune-train-dry
make rune-eval-dry
```

### Submit jobs to the cluster

You must set `RUNE_NAME` per submit so each run is uniquely named in Kueue:

```bash
# Collect: rune envelope is correct, but submit will fail until skill_wm is
# published — see TODO(skill-wm-rune-publish) in collect_rollouts/config.py.
RUNE_NAME=skill-wm-collect-001 SKILL_WM_TOTAL_EPISODES=500 SKILL_WM_WORKERS=10 \
    make rune-collect

# Train: stub — not runnable yet (skill-wm-trained-wm todo).
RUNE_NAME=skill-wm-train-001 SKILL_WM_DATA_RUN=skill-wm-collect-001 \
    uv run python experiments/train_wm/config.py

# Eval: stub — also requires --upstream-checkpoint at submit time
# (rune.eval has no default).
RUNE_NAME=skill-wm-eval-001 SKILL_WM_DATA_RUN=skill-wm-collect-001 \
                            SKILL_WM_TRAIN_RUN=skill-wm-train-001 \
    uv run python experiments/eval_baselines/config.py \
        --upstream-checkpoint /data/skill-wm/checkpoints/skill-wm-train-001/wm.pt
```

All experiments default to `team="experimental"` (the only safe Kueue queue for research on voice-agent-flex). Outputs land under `<ctx.data_dir>/skill-wm/{rollouts,checkpoints,eval}/<run-name>/` — locally that's cwd, on the cluster it's the PVC mount.

**Known gaps before this can actually run on the cluster:**
- `skill_wm` package needs to be `pip install`-able from the cluster (publish the repo and add `"skill-wm @ git+https://..."` to `RUNTIME_PIP`). Until then, `--local` and `--dry-run` work; cluster submit will fail with `ImportError` on the first pod. See `TODO(skill-wm-rune-publish)` in `experiments/collect_rollouts/config.py`.
- `train_wm` and `eval_baselines` are stubs (raise `NotImplementedError`). Real bodies depend on the `skill-wm-trained-wm` / `skill-wm-llm-baseline` / `skill-wm-eval-metrics` todos.

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
