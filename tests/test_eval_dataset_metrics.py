"""Tests for skill_wm.eval.{dataset, metrics} on a tiny fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skill_wm.eval.dataset import load_dir, manifest, split_by_seed
from skill_wm.eval.metrics import (
    aggregate_metrics,
    brier_score,
    expected_calibration_error,
    per_action_breakdown,
)


def _make_npz(path: Path, n_rows: int, seed: int, episode: int = 0) -> None:
    """Write a minimal valid npz shard. All fields must match the schema
    produced by skill_wm.data.collect.transitions_to_npz."""
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 17, size=n_rows).astype(np.int32)
    success = rng.integers(0, 2, size=n_rows).astype(np.bool_)
    data = {
        "episode": np.full(n_rows, episode, dtype=np.int32),
        "step": np.arange(n_rows, dtype=np.int32),
        "seed": np.full(n_rows, seed, dtype=np.int32),
        "action": actions,
        "success": success,
        "reward": np.zeros(n_rows, dtype=np.float32),
        "done": np.zeros(n_rows, dtype=np.bool_),
        "player_pos_before": np.zeros((n_rows, 2), dtype=np.int32),
        "player_pos_after": np.zeros((n_rows, 2), dtype=np.int32),
        "facing_before": np.zeros((n_rows, 2), dtype=np.int8) + np.array([0, 1], dtype=np.int8),
        "facing_after": np.zeros((n_rows, 2), dtype=np.int8) + np.array([0, 1], dtype=np.int8),
        "sleeping_before": np.zeros(n_rows, dtype=np.bool_),
        "sleeping_after": np.zeros(n_rows, dtype=np.bool_),
        "semantic_crop_before": np.zeros((n_rows, 15, 15), dtype=np.uint8),
        "semantic_crop_after": np.zeros((n_rows, 15, 15), dtype=np.uint8),
        "inventory_delta": np.zeros((n_rows, 12), dtype=np.int8),
        "vitals_delta": np.zeros((n_rows, 4), dtype=np.int8),
        "inventory_before": np.zeros((n_rows, 16), dtype=np.int32),
        "inventory_after": np.zeros((n_rows, 16), dtype=np.int32),
        "achievements_unlocked": np.array([""] * n_rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def test_load_dir_recursive(tmp_path: Path) -> None:
    """load_dir must find shards in nested rank-NNN/ subdirs (cluster layout)."""
    _make_npz(tmp_path / "rank-000" / "ep0.npz", n_rows=4, seed=0)
    _make_npz(tmp_path / "rank-001" / "ep0.npz", n_rows=3, seed=1)
    rows = load_dir(tmp_path)
    assert len(rows) == 7
    seeds = {r.seed for r in rows}
    assert seeds == {0, 1}


def test_load_dir_flat(tmp_path: Path) -> None:
    """Flat dir layout (local) also works."""
    _make_npz(tmp_path / "ep0.npz", n_rows=5, seed=0)
    _make_npz(tmp_path / "ep1.npz", n_rows=5, seed=1)
    rows = load_dir(tmp_path)
    assert len(rows) == 10


def test_load_dir_preserves_reward_and_achievements(tmp_path: Path) -> None:
    path = tmp_path / "ep.npz"
    _make_npz(path, n_rows=2, seed=0)
    data = dict(np.load(path))
    data["reward"] = np.array([1.0, 0.0], dtype=np.float32)
    data["achievements_unlocked"] = np.array(["collect_wood|place_table", ""])
    np.savez_compressed(path, **data)

    rows = load_dir(tmp_path)
    assert rows[0].reward == pytest.approx(1.0)
    assert rows[0].achievements_unlocked == ("collect_wood", "place_table")
    assert rows[1].reward == pytest.approx(0.0)
    assert rows[1].achievements_unlocked == ()


def test_load_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dir(tmp_path / "nope")


def test_split_by_seed_is_seed_disjoint(tmp_path: Path) -> None:
    """Same seed must not appear in both train and eval."""
    for s in range(5):
        _make_npz(tmp_path / f"s{s}.npz", n_rows=3, seed=s)
    rows = load_dir(tmp_path)
    train, evalu = split_by_seed(rows, train_frac=0.6, rng_seed=42)
    train_seeds = {r.seed for r in train}
    eval_seeds = {r.seed for r in evalu}
    assert train_seeds.isdisjoint(eval_seeds)
    assert train_seeds | eval_seeds == set(range(5))


def test_split_requires_two_seeds(tmp_path: Path) -> None:
    _make_npz(tmp_path / "single.npz", n_rows=10, seed=0)
    rows = load_dir(tmp_path)
    with pytest.raises(ValueError, match="at least 2 distinct seeds"):
        split_by_seed(rows)


def test_split_keeps_at_least_one_eval_seed(tmp_path: Path) -> None:
    """train_frac=0.99 with 2 seeds must still leave 1 in eval."""
    _make_npz(tmp_path / "a.npz", n_rows=2, seed=0)
    _make_npz(tmp_path / "b.npz", n_rows=2, seed=1)
    rows = load_dir(tmp_path)
    train, evalu = split_by_seed(rows, train_frac=0.99, rng_seed=42)
    assert len(train) > 0
    assert len(evalu) > 0


def test_brier_score_perfect_and_chance() -> None:
    """Perfect predictor scores 0, max-entropy predictor scores variance(label)."""
    labels = np.array([0, 1, 0, 1], dtype=np.float64)
    perfect = labels.copy()
    chance = np.full_like(labels, 0.5)
    assert brier_score(perfect, labels) == pytest.approx(0.0)
    # variance of [0,1,0,1] around 0.5 = mean(0.25) = 0.25
    assert brier_score(chance, labels) == pytest.approx(0.25)


def test_brier_score_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        brier_score(np.array([0.5, 0.5]), np.array([1.0]))


def test_ece_perfect_predictor() -> None:
    """A predictor whose confidence == accuracy in every bin should have ECE ≈ 0."""
    rng = np.random.default_rng(0)
    n = 1000
    probs = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < probs).astype(np.float64)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    # In expectation 0; with finite samples small but bounded.
    assert ece < 0.05, f"expected near-perfect calibration, got {ece:.4f}"


def test_ece_returns_nan_when_too_few_samples() -> None:
    probs = np.array([0.1, 0.5, 0.9])
    labels = np.array([0.0, 1.0, 1.0])
    assert np.isnan(expected_calibration_error(probs, labels, n_bins=10))


def test_per_action_breakdown_suppresses_ece_below_support(tmp_path: Path) -> None:
    """ECE column must be NaN for actions below min_support_for_ece."""
    _make_npz(tmp_path / "a.npz", n_rows=20, seed=0)  # too few for any action
    _make_npz(tmp_path / "b.npz", n_rows=20, seed=1)
    rows = load_dir(tmp_path)
    probs = np.full(len(rows), 0.5)
    breakdown = per_action_breakdown(rows, probs, min_support_for_ece=30)
    # No action should have >= 30 rows in 40 random transitions across 17 actions.
    for name, m in breakdown.items():
        if m["count"] > 0 and m["count"] < 30:
            assert np.isnan(m["ece"]), f"{name} count={m['count']} should suppress ECE"


def test_aggregate_metrics_includes_base_rate(tmp_path: Path) -> None:
    _make_npz(tmp_path / "a.npz", n_rows=50, seed=0)
    _make_npz(tmp_path / "b.npz", n_rows=50, seed=1)
    rows = load_dir(tmp_path)
    probs = np.full(len(rows), 0.5)
    agg = aggregate_metrics(rows, probs)
    assert agg["n"] == 100
    assert 0 <= agg["base_rate"] <= 1
    assert agg["mean_pred"] == pytest.approx(0.5)


def test_manifest_counts_per_action(tmp_path: Path) -> None:
    _make_npz(tmp_path / "a.npz", n_rows=20, seed=0)
    _make_npz(tmp_path / "b.npz", n_rows=20, seed=1)
    rows = load_dir(tmp_path)
    m = manifest(rows)
    assert m["rows"] == 40
    assert m["seeds"] == 2
    total = sum(s["count"] for s in m["per_action"].values())
    assert total == 40
