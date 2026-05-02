"""Transition schema for skill-WM experiments.

A transition is a single (state, action, next_state) tuple emitted by the
environment, plus everything we may want to predict from (state, action):
inventory delta, vitals delta, achievement unlocks, success, terminal.

Designed so that the same record can be:
  - logged to disk (JSON Lines or numpy npz),
  - replayed for prediction-only training,
  - used to compute per-skill success/calibration metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# Crafter inventory keys we treat as "items" (countable resources/tools)
ITEM_KEYS: tuple[str, ...] = (
    "sapling",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
)

# Crafter inventory keys we treat as "vitals" (bounded scalars 0..9)
VITAL_KEYS: tuple[str, ...] = ("health", "food", "drink", "energy")

# Crafter's 17 actions (must match crafter.Env().action_names ordering).
ACTION_NAMES: tuple[str, ...] = (
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
)


def action_success(
    action_name: str,
    info_before: dict[str, Any],
    info_after: dict[str, Any],
) -> bool:
    """Per-action success definition.

    The point of this function is to give the predictor a clean binary target
    that captures "did the agent's intent succeed", not just "did anything
    change". Move counts as success only when position changed; do counts as
    success only when something tangible was gained.
    """
    inv_b = info_before["inventory"]
    inv_a = info_after["inventory"]

    if action_name.startswith("make_"):
        item = action_name[len("make_") :]
        return inv_a.get(item, 0) > inv_b.get(item, 0)

    if action_name.startswith("move_"):
        return not np.array_equal(info_before["player_pos"], info_after["player_pos"])

    if action_name.startswith("place_"):
        # place_table/place_furnace/place_plant unlock achievements; place_stone
        # consumes a stone from inventory.
        if action_name == "place_stone":
            return inv_a.get("stone", 0) < inv_b.get("stone", 0)
        ach_b = info_before["achievements"]
        ach_a = info_after["achievements"]
        return any(ach_a[k] > ach_b[k] for k in ach_a)

    if action_name == "do":
        # `do` is the universal interact: chop tree, mine stone, drink water,
        # eat plant, attack zombie. Success = any item delta or any new ach.
        for k in ITEM_KEYS:
            if inv_a.get(k, 0) != inv_b.get(k, 0):
                return True
        ach_b = info_before["achievements"]
        ach_a = info_after["achievements"]
        return any(ach_a[k] > ach_b[k] for k in ach_a)

    if action_name == "sleep":
        return inv_a.get("energy", 0) > inv_b.get("energy", 0)

    # noop: no semantic intent; we mark False so it doesn't pollute success rate.
    return False


def inventory_delta(inv_before: dict[str, int], inv_after: dict[str, int]) -> np.ndarray:
    """Return a length-len(ITEM_KEYS) vector of int deltas."""
    return np.array(
        [inv_after.get(k, 0) - inv_before.get(k, 0) for k in ITEM_KEYS],
        dtype=np.int8,
    )


def vitals_delta(inv_before: dict[str, int], inv_after: dict[str, int]) -> np.ndarray:
    """Return a length-len(VITAL_KEYS) vector of int deltas."""
    return np.array(
        [inv_after.get(k, 0) - inv_before.get(k, 0) for k in VITAL_KEYS],
        dtype=np.int8,
    )


def achievements_unlocked(ach_before: dict[str, int], ach_after: dict[str, int]) -> list[str]:
    """Names of achievements that went from 0 -> 1 across this transition."""
    return sorted([k for k in ach_after if ach_after[k] > 0 and ach_before.get(k, 0) == 0])


@dataclass
class Transition:
    """One (state, action, outcome) record.

    Fields are kept primitive so we can dump either as JSON Lines (for cheap
    inspection) or as numpy npz (for batch training).
    """

    # Identity
    seed: int
    episode: int
    step: int

    # Action taken
    action: int
    action_name: str

    # State BEFORE the action (compact features, no pixels here)
    inventory_before: dict[str, int]
    player_pos_before: tuple[int, int]
    semantic_crop_before: np.ndarray = field(repr=False)

    # State AFTER the action
    inventory_after: dict[str, int]
    player_pos_after: tuple[int, int]
    semantic_crop_after: np.ndarray = field(repr=False)

    # Predictor targets (computed once, stored for cheap eval)
    success: bool
    inventory_delta: np.ndarray = field(repr=False)
    vitals_delta: np.ndarray = field(repr=False)
    achievements_unlocked: list[str]
    reward: float
    done: bool

    def to_jsonable(self) -> dict[str, Any]:
        """JSON-serializable view; arrays converted to lists."""
        d = asdict(self)
        d["semantic_crop_before"] = self.semantic_crop_before.tolist()
        d["semantic_crop_after"] = self.semantic_crop_after.tolist()
        d["inventory_delta"] = self.inventory_delta.tolist()
        d["vitals_delta"] = self.vitals_delta.tolist()
        d["player_pos_before"] = list(self.player_pos_before)
        d["player_pos_after"] = list(self.player_pos_after)
        return d
