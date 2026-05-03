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
    19: "masked",  # sentinel for tiles outside the predictor's observation budget
}

WALKABLE_TILES = {"grass", "path", "sand"}
ATTACK_TILES = {"zombie", "skeleton", "cow"}
DRINK_TILES = {"water"}
EAT_TILES = {"plant", "cow"}
COLLECTABLE_TILES = {"tree", "stone", "coal", "iron", "diamond"}

# Distinguished tile id for "outside observation radius". Returned by
# `_tile_at` as the string "masked" (different from "void", which means
# off the world entirely, and "unknown", which means an id we don't
# have a name for). Predictors should treat a "masked" answer as "I
# can't see this cell" — same epistemic status as off-world for rules,
# but the trained WM can learn what is statistically likely there.
MASKED_TILE_ID: int = 19


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


class PreconditionWithBackoff:
    """Precondition rules + per-action marginal fallback when state is masked.

    The vanilla `PreconditionPredictor` is brittle under partial obs: any
    rule that queries a masked cell trivially fails (returns 0) because
    "masked" doesn't match any named tile set. That's a *deliberately
    cautious* answer ("I can't see, so I assume not"), but it's not the
    strongest hand-built competitor a reviewer would credit. A reasonable
    rule-based system would notice "I can't read the cell I need" and
    abstain, falling back to the empirical per-action prior.

    This baseline implements that. For each row we ask: "would the rule
    have inspected a masked cell to make its decision?" If yes, we abstain
    and emit the marginal P(success | action). If no, we emit the rule's
    {0, 1} answer.

    The "would inspect a masked cell" check is conservative: we mark the
    row as "depends on a masked cell" iff any of the cells in the player's
    3×3 neighborhood (the maximum span of the existing rules — both
    `_front_tile` and `_has_adjacent` query inside this window) is the
    MASKED sentinel.

    This is the strongest hand-built baseline: rules where they apply,
    learned per-action priors where they don't.
    """

    name = "precondition+backoff"

    def __init__(self) -> None:
        self._rules = PreconditionPredictor()
        self._marginal = MarginalPredictor()

    def fit(self, train_rows: list[ScoringRow]) -> None:
        # MarginalPredictor needs train data; rules are stateless.
        self._marginal.fit(train_rows)

    def predict(self, eval_rows: list[ScoringRow]) -> np.ndarray:
        out = np.empty(len(eval_rows), dtype=np.float64)
        marg = self._marginal.predict(eval_rows)
        for i, row in enumerate(eval_rows):
            if self._row_depends_on_masked(row):
                out[i] = marg[i]
            else:
                out[i] = float(self._rules._rule(row))
        return out

    @staticmethod
    def _row_depends_on_masked(row: ScoringRow) -> bool:
        crop = row.semantic_crop_before
        h, w = crop.shape
        cx, cy = h // 2, w // 2
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                rr, cc = cx + dx, cy + dy
                if 0 <= rr < h and 0 <= cc < w and int(crop[rr, cc]) == MASKED_TILE_ID:
                    return True
        return False


class InventoryOnlyMLP:
    """Per-action logistic regression on inventory features only.

    Critical control for the partial-observability story. At extreme
    masking the trained CNN's spatial channel collapses to a constant
    "masked" embedding, so anything it learns has to come from
    action+inventory features. If this *non-spatial* learned baseline
    matches the trained CNN's Brier under heavy masking, the CNN is not
    doing world modeling — it's just learning conditional priors that a
    plain logistic model would also learn.

    Implementation: independent logistic regression per action id, fit
    on that action's inventory rows only. We hand-roll batch gradient
    descent with L2 regularization to avoid taking on a sklearn
    dependency for one baseline. If an action has fewer than 5 examples
    or only one observed class, we fall back to its marginal rate
    (same fallback as `MarginalPredictor`).
    """

    name = "inv-mlp"

    # Hand-rolled logistic regression hyperparams. Inventory features are
    # unnormalized (some dims range 0-2 for tools, others 0-9 for stacks),
    # so we standardize per-feature and use small LR with many iters for
    # stable convergence on the small action subsets.
    _LR: float = 0.05
    _N_ITERS: int = 2000
    _L2: float = 1e-3  # NB: applied to weights only, NOT bias

    def __init__(self) -> None:
        # Per-action: (weights, bias, feature_mean, feature_std). None means use fallback.
        self._models: dict[int, tuple[np.ndarray, float, np.ndarray, np.ndarray]] = {}
        self._fallback_rate: np.ndarray = np.full(len(ACTION_NAMES), 0.5)
        self._global_rate: float = 0.5

    def fit(self, train_rows: list[ScoringRow]) -> None:
        actions = np.array([r.action for r in train_rows])
        labels = np.array([r.success for r in train_rows], dtype=np.float64)
        invs = np.array([r.inventory_before for r in train_rows], dtype=np.float64)
        self._global_rate = float(labels.mean()) if labels.size else 0.5
        for a in range(len(ACTION_NAMES)):
            mask = actions == a
            y = labels[mask]
            x = invs[mask]
            self._fallback_rate[a] = float(y.mean()) if y.size else self._global_rate
            if y.size < 5 or len(np.unique(y)) < 2:
                continue
            self._models[a] = self._fit_logreg(x, y)

    @classmethod
    def _fit_logreg(
        cls, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        # Standardize features (zero-mean, unit-variance) for stable GD;
        # store mean/std so predict can apply the same transform. Skip
        # constant features (std==0) by treating their std as 1 → centered
        # to 0 → contribute 0 to the linear function.
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        sd_safe = np.where(sd > 0, sd, 1.0)
        x_n = (x - mu) / sd_safe
        n, d = x_n.shape
        # Initialize bias to logit(mean(y)) so iter 0 is already calibrated.
        py = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        bias = float(np.log(py / (1 - py)))
        w = np.zeros(d, dtype=np.float64)
        for _ in range(cls._N_ITERS):
            z = x_n @ w + bias
            p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
            err = p - y
            grad_w = x_n.T @ err / n + cls._L2 * w  # regularize weights only
            grad_b = float(err.mean())  # bias gets pure log-likelihood gradient
            w -= cls._LR * grad_w
            bias -= cls._LR * grad_b
        return w, bias, mu, sd_safe

    def predict(self, eval_rows: list[ScoringRow]) -> np.ndarray:
        out = np.empty(len(eval_rows), dtype=np.float64)
        for i, r in enumerate(eval_rows):
            mdl = self._models.get(r.action)
            if mdl is None:
                out[i] = self._fallback_rate[r.action]
            else:
                w, bias, mu, sd = mdl
                xn = (r.inventory_before.astype(np.float64) - mu) / sd
                z = float(xn @ w + bias)
                if z >= 0:
                    out[i] = 1.0 / (1.0 + np.exp(-z))
                else:
                    ez = np.exp(z)
                    out[i] = ez / (1.0 + ez)
        return out
