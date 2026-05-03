"""End-to-end smoke test for the testbed.

Verifies:
  - Crafter wrapper imports and steps,
  - schema computes deltas/success without crashing,
  - rollouts can be written to disk and reloaded,
  - inventory_after = inventory_before + inventory_delta on a sample of records.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skill_wm.data.collect import collect
from skill_wm.data.schema import (
    ACTION_NAMES,
    ITEM_KEYS,
    VITAL_KEYS,
    action_success,
)
from skill_wm.envs.crafter_env import CrafterWrapper


def test_action_names_are_complete():
    assert "noop" in ACTION_NAMES
    assert "do" in ACTION_NAMES
    assert sum(1 for a in ACTION_NAMES if a.startswith("make_")) == 6
    assert sum(1 for a in ACTION_NAMES if a.startswith("place_")) == 4
    assert sum(1 for a in ACTION_NAMES if a.startswith("move_")) == 4


def test_wrapper_steps_and_emits_transition():
    env = CrafterWrapper(seed=0)
    obs, info = env.reset(episode=0)
    assert obs.shape == (64, 64, 3)
    assert "inventory" in info
    transition, _, _, _ = env.step(ACTION_NAMES.index("noop"))
    assert transition.action_name == "noop"
    # noop never counts as success in our schema.
    assert transition.success is False


def test_inventory_delta_consistency():
    env = CrafterWrapper(seed=1)
    env.reset(episode=0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = int(rng.integers(0, env.num_actions))
        transition, _, _, done = env.step(a)
        # inventory_after - inventory_before, restricted to ITEM_KEYS, must
        # equal the recorded inventory_delta exactly.
        delta = np.array(
            [
                transition.inventory_after.get(k, 0) - transition.inventory_before.get(k, 0)
                for k in ITEM_KEYS
            ],
            dtype=np.int8,
        )
        assert np.array_equal(delta, transition.inventory_delta)
        if done:
            env.reset(episode=1)


def test_collect_writes_loadable_npz(tmp_path: Path):
    stats = collect(
        out_dir=tmp_path,
        num_episodes=2,
        max_steps_per_episode=20,
        policy_name="biased_random",
        seed_start=42,
    )
    assert stats["episodes"] == 2
    assert stats["transitions"] > 0
    files = sorted(tmp_path.glob("ep_*.npz"))
    assert len(files) == 2
    loaded = np.load(files[0])
    assert "action" in loaded
    assert "success" in loaded
    assert "inventory_delta" in loaded
    assert loaded["action"].shape[0] == loaded["success"].shape[0]
    assert loaded["semantic_crop_before"].shape[1] == loaded["semantic_crop_before"].shape[2]


def test_action_success_definition_basics():
    """Synthetic before/after states confirm action_success semantics."""
    inv_before = {k: 0 for k in ITEM_KEYS} | {k: 9 for k in VITAL_KEYS}
    inv_after = dict(inv_before)
    inv_after["wood"] = 1
    info_before = {
        "inventory": inv_before,
        "player_pos": np.array([5, 5]),
        "achievements": {"collect_wood": 0},
    }
    info_after = {
        "inventory": inv_after,
        "player_pos": np.array([5, 5]),
        "achievements": {"collect_wood": 1},
    }
    assert action_success("do", info_before, info_after) is True
    # No-op style action with no change: not a success.
    assert (
        action_success(
            "make_wood_pickaxe",
            {**info_before, "inventory": dict(inv_before)},
            {**info_before, "inventory": dict(inv_before)},
        )
        is False
    )


def test_sleep_success_uses_sleeping_flag_not_energy_delta():
    """Crafter sleep takes ticks; energy rarely changes in one step. The
    intent-took-effect signal is `info_after['sleeping']`, not `energy_after >
    energy_before` (which biases the label toward almost-always-False)."""
    inv_full = {k: 0 for k in ITEM_KEYS} | {k: 9 for k in VITAL_KEYS}
    info_before = {
        "inventory": inv_full,
        "player_pos": np.array([5, 5]),
        "achievements": {},
        "sleeping": False,
    }
    # sleep entered, energy unchanged: this is success.
    info_after_entered = {**info_before, "sleeping": True}
    assert action_success("sleep", info_before, info_after_entered) is True
    # sleep tried, never entered (e.g. enemy nearby): failure.
    info_after_failed = {**info_before, "sleeping": False}
    assert action_success("sleep", info_before, info_after_failed) is False


def test_transition_carries_facing_and_sleeping(tmp_path: Path):
    """Wrapper must populate facing/sleeping into both Transition and the npz."""
    stats = collect(
        out_dir=tmp_path,
        num_episodes=1,
        max_steps_per_episode=30,
        policy_name="biased_random",
        seed_start=0,
    )
    assert stats["transitions"] > 0
    files = sorted(tmp_path.glob("ep_*.npz"))
    loaded = np.load(files[0])
    for k in ("facing_before", "facing_after", "sleeping_before", "sleeping_after"):
        assert k in loaded, f"missing {k}"
    n = loaded["action"].shape[0]
    assert loaded["facing_before"].shape == (n, 2)
    assert loaded["facing_after"].shape == (n, 2)
    assert loaded["sleeping_before"].shape == (n,)
    assert loaded["sleeping_after"].shape == (n,)
    # Crafter facing is one of the four cardinal unit vectors.
    valid = {(-1, 0), (1, 0), (0, -1), (0, 1)}
    for f in loaded["facing_before"]:
        assert tuple(int(x) for x in f) in valid
