# skill-wm

A testbed for the question: **can a small, separately trained world model improve LLM agents — or are LLMs already good enough as their own world model?**

This repo is intentionally small. It started as the **prediction-only smell test** for the thesis direction "learned world models can make LLM agents more reliable on long-horizon tasks." T1 validated the prediction premise, so the repo now also contains a minimal **T2 live-agent harness** for testing whether one-step WM scores actually improve Crafter behavior. It still deliberately avoids Minecraft and JEPA — those come only after the Crafter agent result is understood.

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

T1 is complete; the active work is T2.

## What's in here

```
skill_wm/
  envs/crafter_env.py     # CrafterWrapper that emits clean Transition records
  envs/minihack_env.py    # optional MiniHack/NLE adapter for second-env experiments
  data/schema.py          # Transition dataclass, action_success, deltas
  data/collect.py         # Random / biased-random rollout collection
  data/collect_minihack.py # MiniHack random rollout collector
  models/
    baselines.py          # Random / Marginal / Precondition predictors
    state_text.py         # Crafter state -> ASCII prompt for the LLM
    llm_wm.py             # OpenAI client + logprob-based p(success)
    trained_wm.py         # small CNN+MLP trained on (crop, inv, action) -> p(success)
    skill_jepa.py         # MiniHack Skill-JEPA prototype: state+skill -> latent outcome
  eval/
    dataset.py            # ScoringRow loader, seed-disjoint split, manifest
    metrics.py            # Brier (headline), ECE (adaptive bins, gated by support)
    run.py                # CLI: load -> split -> fit -> predict -> table
  agent/
    run.py                # T2 live-agent loop: baseline policies + WM reranking
tests/
  test_smoke.py           # env + schema + crop alignment
  test_eval_dataset_metrics.py
  test_models.py          # baselines + state_text + llm_wm (with FakeLLMClient)
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
make minihack-smoke       # optional MiniHack rollout (installs --extra minihack)
make skill-jepa-smoke     # collect MiniHack smoke + train tiny Skill-JEPA gate
make skill-jepa-eval      # held-out-seed MiniHack Skill-JEPA eval JSON
make skill-jepa-task-eval # held-out-task MiniHack Skill-JEPA eval JSONs
make skill-jepa-coverage-eval # noisy-room to lava coverage diagnostic
make skill-jepa-hazard-eval # noisy-room to deliberate lava-probe diagnostic
make skill-jepa-interaction-eval # safe-lava to lava-probe interaction diagnostic
make skill-jepa-cjepa-eval # baseline vs object-aux Skill-JEPA diagnostic
make eval-local           # run Random/Marginal/Precondition baselines on data/rollouts/smoke
make eval-llm             # add the LLM-as-WM baseline (needs OPENAI_API_KEY)
make all                  # install + check + smoke
```

Or call uv directly:

```bash
uv sync
uv run pytest -q                                               # all tests
uv run ruff check .                                            # lint
uv run python -m skill_wm.data.collect --episodes 50           # collect rollouts
uv run python -m skill_wm.eval.run --data data/rollouts/smoke \
    --baselines random marginal precondition                   # score
uv run python -m skill_wm.agent.run --train-data data/rollouts/collect-002-combined \
    --policies biased_random scripted_craft precondition-greedy trained-greedy
uv run python -m skill_wm.agent.run --train-data data/rollouts/collect-002-combined \
    --policies mixed precondition-rerank-mixed trained-rerank-mixed
uv run python -m skill_wm.agent.run --train-data data/rollouts/collect-002-combined \
    --policies mixed precondition-rerank-mixed trained-achievement-rerank-mixed
uv run --extra minihack python -m skill_wm.data.collect_minihack \
    --env-id skillwm-lava-detour --episodes 10 --max-steps 50 --policy scripted_nav
uv run --extra train python -m skill_wm.models.skill_jepa \
    --data data/rollouts/minihack-smoke --epochs 10 --batch-size 8 \
    --out data/eval/minihack-skill-jepa-smoke.json
```

## Second environment: MiniHack/NLE

Crafter now looks exhausted for the "learned WM beats rules" contribution:
rules + backoff dominate trained WMs as predictors and as live-agent filters.
The next environment is **MiniHack/NLE**, but only as an adapted custom-task
suite rather than full NetHackScore.

The MiniHack adapter is optional and logs the raw state primitives we need before
committing to a shared predictor schema: glyph crop, BLStats, message text,
inventory strings, action, reward, terminal, and a generic success label
(`success`/`task_success` from `info`, falling back to `reward > 0`).

The current controlled tasks are:

| Task | Purpose |
|---|---|
| `skillwm-room-goal` | Minimal coordinate/staircase success path. |
| `skillwm-lava-detour` | Same goal with a lava column requiring a route around local hazards. |

The first JEPA-shaped gate is intentionally tiny: `skill_wm.models.skill_jepa`
trains `state_before + action -> latent(state_after)` on MiniHack transition
NPZs, with glyph crops, BLStats, messages, and inventory strings in the state
encoder. It reports prediction error as a surprise score so we can test the
LeCun-inspired shape before committing to a larger Skill-JEPA benchmark. Use
`make skill-jepa-eval` for the repeatable controlled gate: it collects both
tasks with matched seeds, trains on a seed-disjoint split, and writes
`data/eval/minihack-skill-jepa-controlled.json`. Use
`make skill-jepa-task-eval` for the distribution-shift gate: it trains on one
controlled task and evaluates on the other, writing room-to-lava and
lava-to-room JSON summaries under `data/eval/`. The JSON summaries include
coverage diagnostics (`action_oov_rate`, glyph OOV rates, and OOV-conditioned
surprise) so the task-shift gap can be separated into unseen-action versus
unseen-object/hazard effects. Use `make skill-jepa-coverage-eval` to collect a
noisy room-goal training set with broader action coverage before evaluating on
lava-detour. Use `make skill-jepa-hazard-eval` for the sharper probe: the model
trains on noisy room-goal transitions, then evaluates on lava-detour episodes
that deliberately attempt one unsafe lava step before recovering. Use
`make skill-jepa-interaction-eval` for the cleanest object-interaction split:
train on safe lava-detour trajectories, then hold out the `lava_probe` policy in
the same task. That keeps actions and lava glyphs in-distribution while making
the unsafe `east|.->L->.` object signature out-of-distribution. Use
`make skill-jepa-cjepa-eval` to compare the baseline latent predictor against an
object-auxiliary variant (`--object-aux-weight`) that learns target/after tile
heads alongside the latent prediction objective.

MiniHack pulls in NLE. Prefer the maintained NLE line (`nle>=1.3`) and install
CMake first if your platform has to build NLE from source:

```bash
brew install cmake        # macOS, if no wheel is available
make minihack-smoke
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
# Collect: installs skill-wm from the public GitHub repo by default.
# Pin the exact commit for reproducible cluster runs:
RUNE_NAME=skill-wm-collect-001 SKILL_WM_TOTAL_EPISODES=500 SKILL_WM_WORKERS=10 \
    SKILL_WM_REPO_REF=$(git rev-parse HEAD) make rune-collect

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

**Cluster status (verified 2026-05-02)**: `collect_rollouts` runs end-to-end on voice-agent-flex. `skill-wm-collect-smoke-004` (5 episodes, biased random) wrote 5 npz to `/data/datasets/skill-wm/rollouts/skill-wm-collect-smoke-004/rank-000/` with the expected schema (192 transitions in ep 0, 36% per-action success rate, 3 achievements unlocked).

**T1 baseline harness status (verified locally on a 5-episode smoke set, ~500 transitions, 60/40 seed-disjoint split)**:

| baseline | overall Brier | overall ECE |
|---|---|---|
| random (always 0.5) | 0.250 | n/a |
| marginal (per-action prior) | ~0.10 | ~0.05 |
| precondition (hand-coded rules) | ~0.07 | ~0.04 |

These are the harness floor/ceiling references. The trained WM and LLM-as-WM baselines are next; see `skill-wm-trained-wm` and the LLM-as-WM toggle in `make eval-llm`. 5 episodes is far too sparse for honest per-action ECE — that requires the next big collect submit (~50+ episodes), but the headline Brier ordering is already meaningful.

**Known gaps before T1 can actually run on the cluster:**
- `train_wm` and `eval_baselines` are stubs (raise `NotImplementedError`). Real bodies depend on the `skill-wm-trained-wm` / `skill-wm-llm-baseline` / `skill-wm-eval-metrics` todos.

**Cluster-side friction (rune issues, with workarounds):**
- `skill_wm` is `pip install`-able from the cluster via `RUNTIME_PIP`'s `git+https://github.com/chokevin/skill-wm.git@<ref>`. Repo is public to avoid cluster-side auth provisioning. The underlying friction (rune-py shipping single files only) is tracked upstream as [aks-ai-runtime#289](https://github.com/azure-management-and-platforms/aks-ai-runtime/issues/289). When that lands, we can ship the source tree directly via `runtime.working_dir`.
- **Pip-spec shell-quoting** ([#295](https://github.com/azure-management-and-platforms/aks-ai-runtime/issues/295)): rune's entrypoint generator joins `runtime.pip` entries unquoted on a `/bin/sh` line. `<` / `>` in version constraints are parsed as redirects (`<3` → `cannot open 3`); `pkg @ url` PEP 508 specs are word-split. **Workaround:** use only `==` exact pins and bare `git+https://...@<ref>` form (no `pkg @`).
- **RayCluster leak on FAILED** ([#292](https://github.com/azure-management-and-platforms/aks-ai-runtime/issues/292)): `shutdownAfterJobFinishes: true` is honored but `ttlSecondsAfterFinished` defaults to 24h, so a failed job's RayCluster holds its DRA GPU claim for a day, blocking subsequent submits from the same team. **Workaround:** `kubectl delete rayjob <name> -n ray`.
- **`gpus=0` Python/Go contract drift** ([#296](https://github.com/azure-management-and-platforms/aks-ai-runtime/issues/296)): `@rune.train` Python decorator accepts `gpus=0`, rune Go CLI rejects `compute.gpus=0` with `want 1..8`. CPU-only Crafter rollouts therefore burn an H100. Default is `gpus=1`.
- **No `runtime.image` override** ([#235](https://github.com/azure-management-and-platforms/aks-ai-runtime/issues/235)): cluster image is hardcoded `rayproject/ray:2.39.0-py310-gpu` (Python 3.10). All cluster deps must be 3.10-compatible. `requires-python = ">=3.10"` and `numpy>=2.0` (not `>=2.4.4`) here.

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

- not a full LLM agent yet (the current T2 harness uses local proposal policies and WM reranking)
- not a Minecraft project
- not a JEPA paper
- not a publishable result on its own

It is a falsifiability harness for one claim at a time. T1 showed trained WMs beat LLM-as-WM prompting on calibration; T2 now asks whether that calibration translates into better live-agent behavior.
