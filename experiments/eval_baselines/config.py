"""Evaluate trained-WM vs LLM-as-WM baselines on held-out rollouts (T1 model #3).

NOT YET IMPLEMENTED — stub showing the rune wiring shape. Uses ``@rune.eval``
because the workload is shaped like fanned-out scoring: 1 GPU head pod for
trained-WM batched inference + N CPU worker pods for LLM API fanout (the
LLM-as-WM baselines call OpenAI / Azure OpenAI from CPU workers).

Local sanity:

    OPENAI_API_KEY=... python experiments/eval_baselines/config.py --local

Cluster submit:

    RUNE_NAME=skill-wm-eval-001 \\
    SKILL_WM_DATA_RUN=skill-wm-collect-001 \\
    SKILL_WM_TRAIN_RUN=skill-wm-train-001 \\
    python experiments/eval_baselines/config.py
"""

from __future__ import annotations

import argparse
import os

import rune

NAME = os.environ.get("RUNE_NAME", "skill-wm-eval-smoke")
DATA_RUN = os.environ.get("SKILL_WM_DATA_RUN", "skill-wm-collect-smoke")
TRAIN_RUN = os.environ.get("SKILL_WM_TRAIN_RUN", "skill-wm-train-smoke")
WORKERS = int(os.environ.get("SKILL_WM_EVAL_WORKERS", "4"))
TEAM = os.environ.get("RUNE_TEAM", "experimental")
PRESET = os.environ.get("RUNE_PRESET") or None

RUNTIME_PIP = [
    "torch>=2.4,<3",
    "numpy>=2.0,<3",
    "openai>=2.0",
    "tqdm>=4.66",
    "pyyaml>=6",
]


@rune.eval(
    name=NAME,
    gpus=1,
    cpu_workers=WORKERS,
    team=TEAM,
    preset=PRESET,
    extra_manifest={
        "runtime": {"pip": RUNTIME_PIP},
        "skill_wm": {
            "data_run": DATA_RUN,
            "train_run": TRAIN_RUN,
        },
    },
)
def eval_baselines(ctx):
    """Score trained-WM and LLM-as-WM on held-out test seeds.

    TODO(skill-wm-eval-metrics + skill-wm-llm-baseline): implement.
    Expected shape:
      1. Load held-out (state, action, success) tuples from
         {durable_datasets_dir}/skill-wm/rollouts/{data_run}/.
      2. Load trained WM checkpoint from ctx.upstream_checkpoint (rune.eval
         injects this from the chained train run, or it's passed via
         --upstream-checkpoint at submit time).
      3. Score trained WM in batched inference on the GPU head pod.
      4. Use ray.remote on CPU workers to fan out OpenAI calls for LLM-as-WM
         (zero-shot + few-shot). Stash answers, then compute Brier/ECE per
         action and per horizon.
      5. Write metrics + calibration plots to
         {durable_checkpoints_dir}/skill-wm/{ctx.name}/.
    """
    from pathlib import Path

    cfg = ctx.manifest["skill_wm"]
    rollouts_dir = Path(ctx.durable_datasets_dir) / "skill-wm" / "rollouts" / cfg["data_run"]
    out_dir = Path(ctx.durable_checkpoints_dir) / "skill-wm" / ctx.name
    print(f"[eval_baselines STUB] would eval data={rollouts_dir}")
    print(
        f"[eval_baselines STUB] would load WM ckpt from ctx.upstream_checkpoint={ctx.upstream_checkpoint}"
    )
    print(f"[eval_baselines STUB] would write to {out_dir}/")
    raise NotImplementedError(
        "eval_baselines body not implemented yet — see skill-wm-eval-metrics + "
        "skill-wm-llm-baseline todos"
    )


def main():
    parser = argparse.ArgumentParser(description="Skill-WM baseline evaluation")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--upstream-checkpoint",
        help="absolute pod-side path to the trained-WM checkpoint "
        "(required for non-dry-run cluster submit; rune.eval has no default)",
    )
    args = parser.parse_args()
    if args.local:
        eval_baselines()
        return
    if args.dry_run:
        eval_baselines.submit(dry_run="client")
        return
    if not args.upstream_checkpoint:
        raise SystemExit(
            "eval_baselines: --upstream-checkpoint is required for cluster submit "
            "(rune.eval expects an absolute pod-side path to the trained-WM ckpt). "
            "Use --dry-run to render the manifest without submitting."
        )
    eval_baselines.submit(upstream_checkpoint=args.upstream_checkpoint)


if __name__ == "__main__":
    main()
