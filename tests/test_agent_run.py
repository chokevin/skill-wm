"""Tests for the T2 agent-loop harness."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from skill_wm.agent.run import PredictorPolicy, _state_action_rows
from skill_wm.data.schema import ACTION_NAMES


class _DummyPredictor:
    name = "dummy"

    def fit(self, train_rows):
        pass

    def predict(self, eval_rows):
        # Prefer place_table if it is present; otherwise prefer the largest id.
        target = ACTION_NAMES.index("place_table")
        return np.array([1.0 if r.action == target else r.action / 100.0 for r in eval_rows])


def _info() -> dict:
    return {
        "semantic": np.zeros((64, 64), dtype=np.uint8),
        "player_pos": np.array([32, 32], dtype=np.int32),
        "facing": (1, 0),
        "sleeping": False,
        "inventory": {"wood": 2, "health": 9, "food": 9, "drink": 9, "energy": 9},
    }


def test_state_action_rows_builds_one_row_per_candidate() -> None:
    env = SimpleNamespace(_seed=123)
    rows = _state_action_rows(env, _info(), episode=7, step=11, actions=[1, 8])

    assert [r.action_name for r in rows] == ["move_left", "place_table"]
    assert rows[0].seed == 123
    assert rows[0].episode == 7
    assert rows[0].step == 11
    assert rows[0].semantic_crop_before.shape == (15, 15)
    assert rows[0].inventory_dict()["wood"] == 2


def test_predictor_policy_greedy_chooses_highest_scored_action() -> None:
    env = SimpleNamespace(_seed=123, num_actions=len(ACTION_NAMES))
    policy = PredictorPolicy(
        name="dummy-greedy",
        predictor=_DummyPredictor(),
        mode="greedy",
    )
    action = policy.act(np.random.default_rng(0), env, _info(), episode=0, step=0)
    assert ACTION_NAMES[action] == "place_table"


def test_predictor_policy_rerank_uses_candidate_subset() -> None:
    env = SimpleNamespace(_seed=123, num_actions=len(ACTION_NAMES))
    policy = PredictorPolicy(
        name="dummy-rerank-biased",
        predictor=_DummyPredictor(),
        mode="rerank",
        proposal="biased",
        num_candidates=1,
    )
    action = policy.act(np.random.default_rng(0), env, _info(), episode=0, step=0)
    assert 0 <= action < len(ACTION_NAMES)
