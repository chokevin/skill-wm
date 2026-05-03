"""End-to-end eval CLI: load rollout dir, run baselines, print metrics table.

Usage::

    python -m skill_wm.eval.run \
        --data data/rollouts/skill-wm-collect-smoke-005 \
        --baselines random marginal precondition \
        --train-frac 0.6 \
        --out eval_results.json

Add ``llm-zero`` to ``--baselines`` to also run the LLM-as-WM zero-shot
baseline (requires ``OPENAI_API_KEY``). LLM responses are cached on disk
under ``--llm-cache`` so reruns are free.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from skill_wm.eval.dataset import format_manifest, load_dir, manifest, split_by_seed
from skill_wm.eval.metrics import (
    aggregate_metrics,
    format_per_action_table,
    per_action_breakdown,
)
from skill_wm.models.baselines import (
    MarginalPredictor,
    PreconditionPredictor,
    Predictor,
    RandomPredictor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("skill_wm.eval.run")


def build_baseline(name: str, llm_cache: Path | None) -> Predictor:
    if name == "random":
        return RandomPredictor()
    if name == "marginal":
        return MarginalPredictor()
    if name == "precondition":
        return PreconditionPredictor()
    if name == "trained":
        # Lazy import: torch only required if the trained baseline is used.
        from skill_wm.models.trained_wm import TrainedWM

        return TrainedWM()
    if name == "llm-zero":
        # Lazy import so non-LLM runs don't need openai installed.
        from skill_wm.models.llm_wm import LLMWorldModel, make_default_client_or_skip

        client = make_default_client_or_skip()
        return LLMWorldModel(client=client, cache_path=llm_cache)
    raise ValueError(f"unknown baseline: {name}")


def run(args: argparse.Namespace) -> dict:
    rows = load_dir(args.data)
    log.info("loaded %d rows from %s", len(rows), args.data)
    train, evalu = split_by_seed(rows, train_frac=args.train_frac, rng_seed=args.split_seed)
    log.info("split: train=%d eval=%d", len(train), len(evalu))

    overall_manifest = manifest(rows)
    eval_manifest = manifest(evalu)
    print("\n=== full dataset manifest ===")
    print(format_manifest(overall_manifest))
    print("\n=== eval split manifest ===")
    print(format_manifest(eval_manifest))

    results: dict[str, dict] = {}
    for name in args.baselines:
        log.info("running baseline %s", name)
        predictor = build_baseline(name, args.llm_cache)
        predictor.fit(train)
        probs = predictor.predict(evalu)
        agg = aggregate_metrics(evalu, probs)
        per_action = per_action_breakdown(evalu, probs, min_support_for_ece=args.min_ece_support)
        results[name] = {"aggregate": agg, "per_action": per_action}
        print(f"\n=== {name} (aggregate) ===")
        print(json.dumps(agg, indent=2))
        print(f"\n=== {name} (per-action) ===")
        print(format_per_action_table(per_action))

    out = {
        "data_dir": str(args.data),
        "n_rows": len(rows),
        "n_train": len(train),
        "n_eval": len(evalu),
        "train_frac": args.train_frac,
        "split_seed": args.split_seed,
        "baselines": list(args.baselines),
        "results": results,
        "manifest": eval_manifest,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, default=_json_default))
        log.info("wrote %s", args.out)

    print("\n=== headline (overall Brier on eval split) ===")
    for name, r in results.items():
        agg = r["aggregate"]
        print(
            f"  {name:<14s}  brier={agg['brier']:.4f}  "
            f"ece={agg['ece']:.4f}  mean_pred={agg['mean_pred']:.3f}"
        )
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate skill-WM baselines on a rollout dir.")
    p.add_argument("--data", type=Path, required=True, help="dir of .npz rollouts (recursive)")
    p.add_argument(
        "--baselines",
        nargs="+",
        default=["random", "marginal", "precondition"],
        choices=["random", "marginal", "precondition", "trained", "llm-zero"],
        help="which baselines to score (llm-zero needs OPENAI_API_KEY)",
    )
    p.add_argument("--train-frac", type=float, default=0.6, help="seed-disjoint train fraction")
    p.add_argument("--split-seed", type=int, default=1337)
    p.add_argument(
        "--min-ece-support",
        type=int,
        default=30,
        help="suppress per-action ECE below this row count (per-action Brier still reported)",
    )
    p.add_argument("--out", type=Path, default=None, help="write JSON results here")
    p.add_argument(
        "--llm-cache",
        type=Path,
        default=Path(".skill_wm_llm_cache.json"),
        help="JSON file to cache LLM responses (per row_id + prompt hash)",
    )
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
