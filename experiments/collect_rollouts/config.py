"""Parallel rollout collection on the voice-agent-flex Rune cluster.

The cluster function calls into ``skill_wm.data.collect.collect`` directly
so the cluster artifact is byte-identical to laptop output. This means
**submitting to cluster requires the skill_wm package to be importable on
the pod**, which today is NOT wired (see TODO below). Local invocation and
manifest dry-runs work without that step.

Each Ray worker pod takes a disjoint range of episode indices, runs Crafter
episodes, and writes one .npz per episode to the shared PVC mount under
``<ctx.data_dir>/skill-wm/rollouts/<run-name>/rank-<NNN>/``.

Local sanity (laptop, in-process, real schema):

    cd ~/nonwork/skill-wm
    source .venv/bin/activate
    SKILL_WM_TOTAL_EPISODES=5 python experiments/collect_rollouts/config.py --local

Cluster manifest dry-run (no submit):

    python experiments/collect_rollouts/config.py --dry-run

Cluster submit (TODO: requires shipping skill_wm — see runtime.pip below):

    RUNE_NAME=skill-wm-collect-001 \\
    SKILL_WM_TOTAL_EPISODES=500 \\
    SKILL_WM_WORKERS=10 \\
    python experiments/collect_rollouts/config.py

Override knobs (env vars):

    RUNE_NAME                unique run name              default: skill-wm-collect-smoke
    SKILL_WM_TOTAL_EPISODES  total episodes across pods   default: 50
    SKILL_WM_MAX_STEPS       max steps per episode        default: 200
    SKILL_WM_POLICY          random | biased_random       default: biased_random
    SKILL_WM_WORKERS         pod count (Ray Train workers) default: 1
    SKILL_WM_GPUS            per-pod GPUs                  default: 1 (wasted; see GPUS comment below)
    RUNE_TEAM                Kueue team routing           default: experimental
    RUNE_PRESET              optional Rune preset
"""

from __future__ import annotations

import argparse
import os

import rune

NAME = os.environ.get("RUNE_NAME", "skill-wm-collect-smoke")
TOTAL_EPISODES = int(os.environ.get("SKILL_WM_TOTAL_EPISODES", "50"))
MAX_STEPS = int(os.environ.get("SKILL_WM_MAX_STEPS", "200"))
POLICY = os.environ.get("SKILL_WM_POLICY", "biased_random")
WORKERS = int(os.environ.get("SKILL_WM_WORKERS", "1"))
# Crafter rollouts are CPU-only — gpus=0 is what we'd want. rune-py's
# @rune.train decorator accepts gpus=0 (no validator), but the rune Go CLI
# rejects compute.gpus=0 with `want 1..8 (per-worker)` at submit/dry-run
# time. Contract drift between Python decorator and Go validator. Default
# to 1 to unblock; this wastes 1 GPU per worker until the upstream contract
# is fixed (follow-up to aks-ai-runtime#289).
GPUS = int(os.environ.get("SKILL_WM_GPUS", "1"))
TEAM = os.environ.get("RUNE_TEAM", "experimental")
PRESET = os.environ.get("RUNE_PRESET") or None


SKILL_WM_REPO = os.environ.get(
    "SKILL_WM_REPO_URL",
    "git+https://github.com/chokevin/skill-wm.git",
)
SKILL_WM_REF = os.environ.get("SKILL_WM_REPO_REF", "main")


# Cluster-side pip list. Crafter rollouts only need crafter + numpy + tqdm,
# plus skill-wm itself (so the pod can `from skill_wm.data.collect import collect`).
#
# CRITICAL: rune's entrypoint generator does NOT shell-quote each pip spec.
# It joins them with spaces inside a /bin/sh `pip install ... ...` line.
# That means:
#   - any `<` or `>` in a version constraint is parsed as a shell redirect
#     (`numpy>=2.0,<3` → /bin/sh tries `<3` and fails)
#   - any `pkg @ url` PEP508 spec is word-split into 3 args, breaking pip
# Use ONLY exact `==` pins and bare `git+https://...@<ref>` (no `pkg @`).
# Tracked separately from #289 — different bug, same general theme.
#
# TODO(skill-wm-rune-publish): rune-py's --extra-script ships exactly one .py
# file (this config), so the skill_wm package is not on the pod by default.
# Tracked upstream:
#   https://github.com/azure-management-and-platforms/aks-ai-runtime/issues/289
#   ("rune-py: ship caller's local source tree to the cluster")
#
# Workaround: pin a published commit of chokevin/skill-wm into RUNTIME_PIP.
# Override SKILL_WM_REPO_URL / SKILL_WM_REPO_REF per submit:
#   SKILL_WM_REPO_REF=$(git rev-parse HEAD) make rune-collect
#
# The repo is currently public to avoid provisioning cluster-side git auth
# for chokevin/skill-wm. When aks-ai-runtime#289 lands and we can ship the
# source tree directly via runtime.working_dir, this can flip back to private.
RUNTIME_PIP = [
    "crafter==1.8.3",
    "numpy==2.1.3",
    "tqdm==4.67.1",
    "imageio==2.37.0",
    "pyyaml==6.0.2",
    f"{SKILL_WM_REPO}@{SKILL_WM_REF}",
]


@rune.train(
    name=NAME,
    gpus=GPUS,
    workers=WORKERS,
    team=TEAM,
    preset=PRESET,
    extra_manifest={
        "runtime": {"pip": RUNTIME_PIP},
        "skill_wm": {
            "total_episodes": TOTAL_EPISODES,
            "max_steps": MAX_STEPS,
            "policy": POLICY,
        },
    },
)
def collect_rollouts(ctx):
    """Cluster-side rollout collection sharded across Ray workers.

    Calls into ``skill_wm.data.collect.collect`` so the .npz files are
    byte-identical to laptop output. Sharding splits ``total_episodes``
    across ``workers`` pods using divmod (no over-allocation); each pod
    gets a contiguous, disjoint range of episode indices and writes to
    its own ``rank-NNN/`` subdir.
    """
    from pathlib import Path

    # Single source of truth — same code path as `make collect-small`.
    from skill_wm.data.collect import collect

    cfg = ctx.manifest["skill_wm"]
    total_episodes = int(cfg["total_episodes"])
    max_steps = int(cfg["max_steps"])
    policy_name = cfg["policy"]

    # Rank discovery. In local single-process mode (is_remote=False) and in
    # single-pod remote mode (workers==1), there is no Ray Train context and
    # rank=0/world_size=1 is correct. In multi-worker remote mode we MUST
    # get a real rank from Ray Train; silently falling back to rank 0 would
    # cause every pod to overwrite the same files. Hard-fail in that case.
    is_remote = bool(ctx.is_remote)
    declared_workers = int(ctx.workers)
    if is_remote and declared_workers > 1:
        from ray import train as ray_train

        world_size = ray_train.get_context().get_world_size()
        rank = ray_train.get_context().get_world_rank()
    else:
        world_size = 1
        rank = 0

    # Divmod sharding: rank r gets `base + (1 if r < extras else 0)` episodes.
    # Works correctly when world_size > total_episodes (extra ranks get 0).
    base, extras = divmod(total_episodes, world_size)
    counts = [base + (1 if r < extras else 0) for r in range(world_size)]
    starts = [sum(counts[:r]) for r in range(world_size)]
    seed_start = starts[rank]
    n_episodes = counts[rank]

    # Path under rune's idiomatic datasets dir. Locally this is `cwd/datasets/`;
    # on the cluster it's the PVC dataset subdir injected by the rune wrapper.
    # Path under rune's durable datasets dir (persistent across runs).
    # Locally: `cwd/datasets/...`. On the cluster: `/data/datasets/...` on
    # the PVC mount, readable by downstream training/eval jobs.
    out_dir = Path(ctx.durable_datasets_dir) / "skill-wm" / "rollouts" / ctx.name / f"rank-{rank:03d}"

    if n_episodes == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(
            f'{{"rank": {rank}, "world_size": {world_size}, "episodes": 0}}\n'
        )
        print(f"[rank {rank}] no episodes assigned (world_size={world_size} > "
              f"total_episodes={total_episodes})")
        return

    stats = collect(
        out_dir=out_dir,
        num_episodes=n_episodes,
        max_steps_per_episode=max_steps,
        policy_name=policy_name,
        seed_start=seed_start,
    )
    print(
        f"[rank {rank}/{world_size}] wrote {n_episodes} episodes "
        f"(seeds {seed_start}..{seed_start + n_episodes - 1}) -> {out_dir}: {stats}"
    )


def main():
    parser = argparse.ArgumentParser(description="Skill-WM rollout collection")
    parser.add_argument("--local", action="store_true",
                        help="run inside this process (no submit, writes to cwd/skill-wm/rollouts/<name>/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="render manifest via rune CLI client-side, do not apply")
    args = parser.parse_args()

    if args.local:
        collect_rollouts()
        return
    collect_rollouts.submit(dry_run="client" if args.dry_run else None)


if __name__ == "__main__":
    main()
