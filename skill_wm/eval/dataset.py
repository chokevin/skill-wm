"""Load skill-WM rollout shards into a flat list of scoring rows.

A `ScoringRow` is one (state, action, ground-truth-success) record taken
verbatim from a collected `.npz` file. Predictors consume rows; metrics
consume the predictions plus the rows. Keeping the row a thin dataclass
(no env, no live Crafter dependency) means the eval harness can run on
any machine that has the npz files and numpy.

Two design choices worth flagging:

1. **Recursive glob.** Local rollouts live in a flat dir
   (`data/rollouts/<run>/*.npz`); cluster rollouts nest by Ray worker
   rank (`/data/datasets/skill-wm/rollouts/<run>/rank-NNN/*.npz`). We
   glob `**/*.npz` so the same loader works for both.

2. **Seed-disjoint train/eval split, not row-disjoint.** All transitions
   from the same world seed share environment dynamics; splitting rows
   at random would leak that into eval. We split *files* by seed and
   then concatenate rows. The split function is deterministic given a
   seed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skill_wm.data.schema import ACTION_NAMES, ITEM_KEYS, VITAL_KEYS

INVENTORY_KEYS = list(ITEM_KEYS) + list(VITAL_KEYS)


@dataclass(frozen=True)
class ScoringRow:
    """One transition viewed from the predictor's side.

    All fields are state-BEFORE-the-action plus the action and the
    ground-truth `success` label. State-after fields are deliberately
    NOT included: predictors must not see the future.
    """

    # Stable identity for caching async LLM calls. (seed, episode, step)
    # uniquely identifies a transition across a run; also computed as a
    # short hex digest for log readability.
    seed: int
    episode: int
    step: int

    # Action being predicted
    action: int
    action_name: str

    # State BEFORE the action
    inventory_before: np.ndarray  # (16,) int32, INVENTORY_KEYS order
    player_pos_before: tuple[int, int]
    facing_before: tuple[int, int]
    sleeping_before: bool
    semantic_crop_before: np.ndarray  # (15, 15) int

    # Ground truth label
    success: bool

    @property
    def row_id(self) -> str:
        """Stable, short ID for caching/log lines."""
        s = f"{self.seed}-{self.episode}-{self.step}"
        return f"{s}@{hashlib.md5(s.encode()).hexdigest()[:6]}"

    def inventory_dict(self) -> dict[str, int]:
        """Convenience: inventory as keyed dict, for textualization."""
        return {k: int(v) for k, v in zip(INVENTORY_KEYS, self.inventory_before, strict=True)}


def load_shard(npz_path: Path) -> list[ScoringRow]:
    """Load one .npz shard into a list of ScoringRow."""
    f = np.load(npz_path)
    n = int(f["action"].shape[0])
    rows: list[ScoringRow] = []
    for i in range(n):
        action = int(f["action"][i])
        rows.append(
            ScoringRow(
                seed=int(f["seed"][i]),
                episode=int(f["episode"][i]),
                step=int(f["step"][i]),
                action=action,
                action_name=ACTION_NAMES[action],
                inventory_before=np.asarray(f["inventory_before"][i], dtype=np.int32),
                player_pos_before=tuple(int(x) for x in f["player_pos_before"][i]),
                facing_before=tuple(int(x) for x in f["facing_before"][i]),
                sleeping_before=bool(f["sleeping_before"][i]),
                semantic_crop_before=np.asarray(f["semantic_crop_before"][i]),
                success=bool(f["success"][i]),
            )
        )
    return rows


def load_dir(root: Path) -> list[ScoringRow]:
    """Recursively load all .npz shards under `root`."""
    paths = sorted(Path(root).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz files under {root}")
    rows: list[ScoringRow] = []
    for p in paths:
        rows.extend(load_shard(p))
    return rows


def split_by_seed(
    rows: list[ScoringRow],
    train_frac: float = 0.6,
    rng_seed: int = 1337,
) -> tuple[list[ScoringRow], list[ScoringRow]]:
    """Split rows into (train, eval) by world seed.

    All rows from the same seed go to the same split; this prevents
    same-world rows from leaking dynamics across the split boundary.
    Deterministic in `rng_seed`.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    seeds = sorted({r.seed for r in rows})
    if len(seeds) < 2:
        raise ValueError(f"need at least 2 distinct seeds to split, got {len(seeds)}: {seeds}")
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(len(seeds))
    n_train = max(1, int(round(len(seeds) * train_frac)))
    n_train = min(n_train, len(seeds) - 1)  # leave at least one seed for eval
    train_seeds = {seeds[i] for i in perm[:n_train]}
    train = [r for r in rows if r.seed in train_seeds]
    evalu = [r for r in rows if r.seed not in train_seeds]
    return train, evalu


def manifest(rows: list[ScoringRow]) -> dict:
    """Summary stats for a row set; print this before scoring."""
    actions = np.array([r.action for r in rows])
    succ = np.array([r.success for r in rows], dtype=bool)
    seeds = sorted({r.seed for r in rows})
    per_action: dict[str, dict[str, int]] = {}
    for a, name in enumerate(ACTION_NAMES):
        mask = actions == a
        n = int(mask.sum())
        s = int(succ[mask].sum()) if n else 0
        per_action[name] = {"count": n, "successes": s}
    return {
        "rows": len(rows),
        "seeds": len(seeds),
        "seed_list": seeds,
        "per_action": per_action,
    }


def format_manifest(m: dict) -> str:
    """Pretty-print a manifest for stdout."""
    lines = [
        f"rows: {m['rows']}",
        f"seeds: {m['seeds']}",
        "per-action (count / successes):",
    ]
    for name, stats in m["per_action"].items():
        lines.append(f"  {name:<20s} {stats['count']:>5d} / {stats['successes']:>4d}")
    return "\n".join(lines)


def main() -> None:
    """CLI: print a manifest summary for one rollout dir."""
    import argparse

    p = argparse.ArgumentParser(description="Inspect a skill-WM rollout dir.")
    p.add_argument("data_dir", type=Path)
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = p.parse_args()
    rows = load_dir(args.data_dir)
    m = manifest(rows)
    if args.json:
        print(json.dumps(m, indent=2))
    else:
        print(format_manifest(m))


if __name__ == "__main__":
    main()
