"""Trivial baselines for the skill-WM calibration comparison.

These are floor and ceiling references for the LLM-as-WM and trained-WM:

- `RandomPredictor`: always 0.5. Lower bound on Brier for any reasonable
  predictor (Brier of 0.5-everywhere is exactly the marginal variance,
  ~0.25 for a balanced binary). If a "real" predictor underperforms
  this, it's actively miscalibrated.

- `MarginalPredictor`: per-action empirical P(success | action) fitted
  on train. This is the strongest non-context-aware predictor. Beating
  it requires using state. If LLM-as-WM and the trained WM can't beat
  it, neither is using context meaningfully.

- `PreconditionPredictor`: hand-coded rules per action — checks the
  semantic crop / inventory / facing for the obvious necessary
  conditions (e.g. `make_wood_pickaxe` needs wood ≥ 1 and an adjacent
  table). This is a *deterministic* {0, 1} predictor — it deliberately
  outputs 0 or 1, no in-between. Useful as a "domain-knowledge floor":
  if a learned model can't beat hand-coded rules, the rules are doing
  the work.

All three implement the same `Predictor` protocol so the eval driver
treats them uniformly.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from skill_wm.data.schema import ACTION_NAMES
from skill_wm.eval.dataset import INVENTORY_KEYS, ScoringRow


class Predictor(Protocol):
    """Common surface area for everything in the baseline horse race."""

    name: str

    def fit(self, train_rows: list[ScoringRow]) -> None:
        """Optional training step. Stateless predictors implement as no-op."""

    def predict(self, eval_rows: list[ScoringRow]) -> np.ndarray:
        """Return p(success) in [0, 1] for each row, in input order."""


class RandomPredictor:
    name = "random"

    def fit(self, train_rows: list[ScoringRow]) -> None:
        pass

    def predict(self, eval_rows: list[ScoringRow]) -> np.ndarray:
        return np.full(len(eval_rows), 0.5, dtype=np.float64)


class MarginalPredictor:
    """P(success | action) from training data.

    Falls back to the global base rate for actions with zero training
    examples (otherwise the estimate is undefined and we'd silently
    output 0). This is honest: with no information, predict the prior.
    """

    name = "marginal"

    def __init__(self) -> None:
        self.per_action_rate: np.ndarray = np.full(len(ACTION_NAMES), 0.5)
        self.global_rate: float = 0.5
        self.per_action_count: np.ndarray = np.zeros(len(ACTION_NAMES), dtype=np.int64)

    def fit(self, train_rows: list[ScoringRow]) -> None:
        actions = np.array([r.action for r in train_rows])
        labels = np.array([r.success for r in train_rows], dtype=np.float64)
        self.global_rate = float(labels.mean()) if labels.size else 0.5
        for a in range(len(ACTION_NAMES)):
            mask = actions == a
            n = int(mask.sum())
            self.per_action_count[a] = n
            self.per_action_rate[a] = float(labels[mask].mean()) if n > 0 else self.global_rate

    def predict(self, eval_rows: list[ScoringRow]) -> np.ndarray:
        actions = np.array([r.action for r in eval_rows])
        return self.per_action_rate[actions].astype(np.float64)


# Crafter tile-id legend. Order matches Crafter's SemanticView construction
# (engine.py: `[None] + materials` for terrain ids 0..12, then `obj_types`
# for entities 13..18). Verified against
# crafter/env.py:47 (SemanticView object list) and crafter/constants.py
# (`materials = ['water','grass','stone','path','sand','tree','lava',
#                'coal','iron','diamond','table','furnace']`).
TILE_LEGEND: dict[int, str] = {
    0: "void",
    1: "water",
    2: "grass",
    3: "stone",
    4: "path",
    5: "sand",
    6: "tree",
    7: "lava",
    8: "coal",
    9: "iron",
    10: "diamond",
    11: "table",
    12: "furnace",
    13: "player",
    14: "cow",
    15: "zombie",
    16: "skeleton",
    17: "arrow",
    18: "plant",
}

WALKABLE_TILES = {"grass", "path", "sand"}
ATTACK_TILES = {"zombie", "skeleton", "cow"}
DRINK_TILES = {"water"}
EAT_TILES = {"plant", "cow"}
COLLECTABLE_TILES = {"tree", "stone", "coal", "iron", "diamond"}


def _tile_at(crop: np.ndarray, dx: int, dy: int) -> str:
    """Read the tile at offset (dx, dy) FROM PLAYER on the world-aligned crop.

    Crafter convention: ``world_target = player_pos + (dx, dy)``;
    ``crop[half+dx, half+dy] == sem[player_pos[0]+dx, player_pos[1]+dy]``.
    """
    h, w = crop.shape
    cx, cy = h // 2, w // 2
    rr, cc = cx + dx, cy + dy
    if not (0 <= rr < h and 0 <= cc < w):
        return "void"
    return TILE_LEGEND.get(int(crop[rr, cc]), "unknown")


def _front_tile(row: ScoringRow) -> str:
    """Tile in the cell the player is facing (target of `do` and `place_*`)."""
    fx, fy = row.facing_before
    return _tile_at(row.semantic_crop_before, fx, fy)


def _has_adjacent(row: ScoringRow, target: str) -> bool:
    """True if any of the 4-neighbors is `target` (used for `is_at_table` etc.)."""
    for d in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if _tile_at(row.semantic_crop_before, *d) == target:
            return True
    return False


class PreconditionPredictor:
    """Hand-coded {0, 1} predictor based on Crafter's documented rules.

    Rules are the *necessary* conditions taken from Crafter source. They
    can over-predict success (e.g. predict make_wood_pickaxe=1 when
    you're at a table with wood, even though the trained policy may
    have moved away by the time the action resolves), but they should
    rarely under-predict.

    Implemented as a stateless rules engine; `fit` is a no-op.
    """

    name = "precondition"

    def fit(self, train_rows: list[ScoringRow]) -> None:
        pass

    def predict(self, eval_rows: list[ScoringRow]) -> np.ndarray:
        return np.array([float(self._rule(r)) for r in eval_rows], dtype=np.float64)

    @staticmethod
    def _inv(row: ScoringRow, key: str) -> int:
        idx = INVENTORY_KEYS.index(key)
        return int(row.inventory_before[idx])

    @classmethod
    def _rule(cls, row: ScoringRow) -> bool:
        name = row.action_name
        front = _front_tile(row)

        if name == "noop":
            return False  # noop is never success per schema

        if name.startswith("move_"):
            # Crafter `_move`: directions = dict(left=(-1,0), right=(+1,0),
            # up=(0,-1), down=(0,+1)). Move success iff target tile is walkable.
            offsets = {
                "move_left": (-1, 0),
                "move_right": (1, 0),
                "move_up": (0, -1),
                "move_down": (0, 1),
            }
            tile = _tile_at(row.semantic_crop_before, *offsets[name])
            return tile in WALKABLE_TILES

        if name == "do":
            # `do` interacts with the cell in front. Success modes:
            #   tree -> collect_wood, stone -> collect_stone (needs wood_pickaxe),
            #   coal -> needs wood_pickaxe, iron -> stone_pickaxe, diamond -> iron_pickaxe,
            #   water -> drink, plant/cow -> eat, zombie/skeleton -> defeat (needs sword)
            if front == "tree":
                return True
            if front == "stone":
                return cls._inv(row, "wood_pickaxe") > 0
            if front == "coal":
                return cls._inv(row, "wood_pickaxe") > 0
            if front == "iron":
                return cls._inv(row, "stone_pickaxe") > 0
            if front == "diamond":
                return cls._inv(row, "iron_pickaxe") > 0
            if front in DRINK_TILES:
                return True
            if front in EAT_TILES:
                return True
            if front in ATTACK_TILES:
                # Need any sword to reliably damage; bare hands work too but
                # often miss. Predict success only if armed.
                return any(
                    cls._inv(row, k) > 0 for k in ("wood_sword", "stone_sword", "iron_sword")
                )
            return False

        if name == "sleep":
            # Sleep enters successfully whenever the cell is walkable and
            # there's no adjacent enemy. We don't have full enemy adjacency
            # in the local crop check; approximate: no enemy in the 4-neighborhood.
            return not any(
                _tile_at(row.semantic_crop_before, *d) in ATTACK_TILES
                for d in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )

        if name == "place_stone":
            return cls._inv(row, "stone") > 0 and front in WALKABLE_TILES
        if name == "place_table":
            return cls._inv(row, "wood") > 0 and front in WALKABLE_TILES
        if name == "place_furnace":
            return cls._inv(row, "stone") > 0 and front in WALKABLE_TILES
        if name == "place_plant":
            # plant requires a sapling and grass in front
            return cls._inv(row, "sapling") > 0 and front == "grass"

        # Crafting actions: need ingredients AND adjacency to table (and
        # furnace for iron tools).
        at_table = _has_adjacent(row, "table")
        at_furnace = _has_adjacent(row, "furnace")
        if name == "make_wood_pickaxe":
            return at_table and cls._inv(row, "wood") >= 1
        if name == "make_wood_sword":
            return at_table and cls._inv(row, "wood") >= 1
        if name == "make_stone_pickaxe":
            return at_table and cls._inv(row, "wood") >= 1 and cls._inv(row, "stone") >= 1
        if name == "make_stone_sword":
            return at_table and cls._inv(row, "wood") >= 1 and cls._inv(row, "stone") >= 1
        if name == "make_iron_pickaxe":
            return (
                at_table
                and at_furnace
                and cls._inv(row, "wood") >= 1
                and cls._inv(row, "coal") >= 1
                and cls._inv(row, "iron") >= 1
            )
        if name == "make_iron_sword":
            return (
                at_table
                and at_furnace
                and cls._inv(row, "wood") >= 1
                and cls._inv(row, "coal") >= 1
                and cls._inv(row, "iron") >= 1
            )

        return False
