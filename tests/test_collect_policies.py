"""Tests for scripted_craft + mixed policies in collect.py."""
import numpy as np

from skill_wm.data.collect import (
    POLICIES,
    _move_toward_target,
    biased_random_policy,
    mixed_policy,
    random_policy,
    scripted_craft_policy,
)
from skill_wm.data.schema import ACTION_NAMES

NUM_ACTIONS = len(ACTION_NAMES)


def test_policies_registered():
    for name in ("random", "biased_random", "scripted_craft", "mixed"):
        assert name in POLICIES


def test_legacy_signature_still_callable():
    rng = np.random.default_rng(0)
    a = random_policy(rng, NUM_ACTIONS)
    assert 0 <= a < NUM_ACTIONS
    a2 = biased_random_policy(rng, NUM_ACTIONS)
    assert 0 <= a2 < NUM_ACTIONS


def test_scripted_falls_back_when_info_missing():
    rng = np.random.default_rng(0)
    a = scripted_craft_policy(rng, NUM_ACTIONS, None)
    assert 0 <= a < NUM_ACTIONS
    a2 = scripted_craft_policy(rng, NUM_ACTIONS, {})
    assert 0 <= a2 < NUM_ACTIONS


def _make_info(sem, pos=(7, 7), facing=(0, 1), inv=None):
    return {
        "semantic": sem,
        "player_pos": np.array(pos),
        "facing": facing,
        "inventory": inv or {},
    }


def test_scripted_makes_iron_when_at_table_furnace_with_ingredients():
    sem = np.full((15, 15), 2, dtype=np.uint8)  # all grass
    sem[7, 7] = 13  # player
    sem[6, 7] = 11  # table to the left
    sem[8, 7] = 12  # furnace to the right
    info = _make_info(sem, inv={"wood": 1, "coal": 1, "iron": 1})
    rng = np.random.default_rng(0)
    a = scripted_craft_policy(rng, NUM_ACTIONS, info)
    assert ACTION_NAMES[a] in ("make_iron_pickaxe", "make_iron_sword")


def test_scripted_does_tree_when_facing_one():
    sem = np.full((15, 15), 2, dtype=np.uint8)
    sem[7, 7] = 13  # player
    sem[7, 8] = 6  # tree below (facing=(0,1) is +y/down)
    info = _make_info(sem, facing=(0, 1))
    rng = np.random.default_rng(0)
    a = scripted_craft_policy(rng, NUM_ACTIONS, info)
    assert ACTION_NAMES[a] == "do"


def test_scripted_skips_stone_without_pickaxe():
    sem = np.full((15, 15), 2, dtype=np.uint8)
    sem[7, 7] = 13
    sem[7, 8] = 3  # stone in front
    # No pickaxe in inventory -- should NOT do; should navigate elsewhere or fall back
    info = _make_info(sem, facing=(0, 1), inv={})
    rng = np.random.default_rng(0)
    a = scripted_craft_policy(rng, NUM_ACTIONS, info)
    assert ACTION_NAMES[a] != "do"


def test_scripted_places_table_only_with_two_wood():
    sem = np.full((15, 15), 2, dtype=np.uint8)  # all grass; no nearby table
    sem[7, 7] = 13
    # Only 1 wood -- should NOT place_table (Crafter requires 2)
    info = _make_info(sem, facing=(0, 1), inv={"wood": 1})
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = scripted_craft_policy(rng, NUM_ACTIONS, info)
        assert ACTION_NAMES[a] != "place_table"
    # With 2 wood -- should place_table
    info["inventory"] = {"wood": 2}
    a = scripted_craft_policy(rng, NUM_ACTIONS, info)
    assert ACTION_NAMES[a] == "place_table"


def test_move_toward_target_paths_around_blocked_direct_route():
    sem = np.full((15, 15), 2, dtype=np.uint8)  # grass
    pos = np.array([7, 7])
    sem[7, 7] = 13  # player
    sem[8, 7] = 3  # stone blocks the direct route to coal
    sem[9, 7] = 8  # coal target
    rng = np.random.default_rng(0)
    action = _move_toward_target(rng, sem, pos, {8})
    assert action is not None
    assert ACTION_NAMES[action] != "move_right"


def test_mixed_calls_both_branches():
    # With many calls, mixed should produce a mix of scripted (intelligent)
    # and biased_random (uniform-ish) action choices.
    sem = np.full((15, 15), 2, dtype=np.uint8)
    sem[7, 7] = 13
    info = _make_info(sem)
    rng = np.random.default_rng(42)
    actions = [mixed_policy(rng, NUM_ACTIONS, info) for _ in range(200)]
    assert all(0 <= a < NUM_ACTIONS for a in actions)
    # Should produce diverse actions (not collapse to one)
    assert len(set(actions)) > 3
