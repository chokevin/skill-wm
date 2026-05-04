"""Decision-time MiniHack reranking with Skill-JEPA object heads."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from skill_wm.envs.minihack_tasks import MINIHACK_CARDINAL_ACTION_NAMES
from skill_wm.models.skill_jepa import (
    MiniHackJEPARow,
    MiniHackJEPAVocab,
    MiniHackSkillJEPA,
    SkillJEPAConfig,
    load_minihack_jepa_dir,
    object_signature,
    object_tiles,
)


def find_lava_probe_state(rows: list[MiniHackJEPARow]) -> MiniHackJEPARow:
    for row in rows:
        if (
            row.env_id == "skillwm-lava-detour"
            and row.policy_name == "lava_probe"
            and row.logical_pos_before == (3, 2)
            and row.action_name == "east"
        ):
            return row
    raise ValueError("no held-out lava-probe row at logical_pos_before=(3, 2), action=east")


def candidate_rows(
    base_row: MiniHackJEPARow,
    action_names: tuple[str, ...] = MINIHACK_CARDINAL_ACTION_NAMES,
) -> list[MiniHackJEPARow]:
    """Build one candidate row per action from the same pre-action state."""

    return [
        replace(
            base_row,
            action=i,
            action_name=action_name,
            policy_name="object_rerank_candidate",
            logical_pos_after=base_row.logical_pos_before,
            glyph_after=base_row.glyph_before,
            blstats_after=base_row.blstats_before,
            message_after=base_row.message_before,
            inventory_after=base_row.inventory_before,
            success=False,
            done=False,
            reward=0.0,
        )
        for i, action_name in enumerate(action_names)
    ]


def candidate_scores(
    model: MiniHackSkillJEPA,
    rows: list[MiniHackJEPARow],
) -> list[dict[str, object]]:
    scores = model.score_rows(rows)
    out: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        before_tile, target_tile, after_tile = object_tiles(row)
        out.append(
            {
                "action_name": row.action_name,
                "object_signature": object_signature(row),
                "before_tile": before_tile,
                "target_tile": target_tile,
                "after_tile": after_tile,
                "target_object_nll": float(scores["target_object_nll"][i]),
                "after_object_nll": float(scores["after_object_nll"][i]),
                "object_nll": float(scores["object_nll"][i]),
                "latent_surprise": float(scores["latent_surprise"][i]),
                "object_signature_known": object_signature(row) in model.vocab.object_signatures,
            }
        )
    return out


def choose_action(scores: list[dict[str, object]], score_key: str = "target_object_nll") -> str:
    if not scores:
        raise ValueError("scores is empty")
    return str(min(scores, key=lambda row: float(row[score_key]))["action_name"])


def run_rerank(
    rows: list[MiniHackJEPARow],
    config: SkillJEPAConfig,
    proposed_action: str = "east",
) -> dict[str, object]:
    train_rows = [row for row in rows if row.policy_name != "lava_probe"]
    if not train_rows:
        raise ValueError("no non-lava_probe rows available for training")

    probe_state = find_lava_probe_state(rows)
    vocab = MiniHackJEPAVocab.from_rows(train_rows)
    model = MiniHackSkillJEPA(vocab, config)
    model.fit(train_rows)

    candidates = candidate_rows(probe_state)
    scores = candidate_scores(model, candidates)
    selected = choose_action(scores)
    proposed = next((row for row in scores if row["action_name"] == proposed_action), None)
    if proposed is None:
        raise ValueError(f"proposed action {proposed_action!r} not in candidates")
    selected_score = next(row for row in scores if row["action_name"] == selected)

    return {
        "train_rows": len(train_rows),
        "candidate_state": {
            "env_id": probe_state.env_id,
            "logical_pos_before": list(probe_state.logical_pos_before),
            "held_out_policy": probe_state.policy_name,
            "held_out_action": probe_state.action_name,
        },
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "object_aux_weight": config.object_aux_weight,
        },
        "proposed_action": proposed_action,
        "selected_action": selected,
        "rejected_proposed": selected != proposed_action,
        "proposed_target_object_nll": proposed["target_object_nll"],
        "selected_target_object_nll": selected_score["target_object_nll"],
        "scores": sorted(scores, key=lambda row: float(row["target_object_nll"])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Rerank MiniHack actions with object-head NLL.")
    p.add_argument("--data", type=Path, default=Path("data/rollouts/minihack-jepa-interaction"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--object-aux-weight", type=float, default=0.2)
    p.add_argument("--proposed-action", default="east")
    p.add_argument("--out", type=Path, help="Optional JSON summary output path.")
    p.add_argument(
        "--require-reject",
        action="store_true",
        help="Exit non-zero if the reranker keeps the proposed action.",
    )
    args = p.parse_args()

    rows = load_minihack_jepa_dir(args.data)
    summary = run_rerank(
        rows,
        SkillJEPAConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            object_aux_weight=args.object_aux_weight,
        ),
        proposed_action=args.proposed_action,
    )

    print("MiniHack object-head reranker")
    print(f"  train rows: {summary['train_rows']}")
    print(f"  proposed action: {summary['proposed_action']}")
    print(f"  selected action: {summary['selected_action']}")
    print(f"  rejected proposed: {summary['rejected_proposed']}")
    print("  candidate scores:")
    for row in summary["scores"]:
        print(
            "   ",
            row["action_name"],
            "target_nll=",
            f"{float(row['target_object_nll']):.6f}",
            "signature=",
            row["object_signature"],
            "known=",
            row["object_signature_known"],
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"  wrote: {args.out}")

    if args.require_reject and not summary["rejected_proposed"]:
        raise SystemExit(f"reranker kept proposed action {args.proposed_action!r}")


if __name__ == "__main__":
    main()
