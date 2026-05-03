"""Tests for the end-to-end eval runner CLI helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from skill_wm.eval.run import run


def _make_npz(path: Path, n_rows: int, seed: int, episode: int = 0) -> None:
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 17, size=n_rows).astype(np.int32)
    data = {
        "episode": np.full(n_rows, episode, dtype=np.int32),
        "step": np.arange(n_rows, dtype=np.int32),
        "seed": np.full(n_rows, seed, dtype=np.int32),
        "action": actions,
        "success": rng.integers(0, 2, size=n_rows).astype(np.bool_),
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


def test_run_eval_data_uses_separate_train_and_eval_dirs(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    _make_npz(train_dir / "seed0.npz", n_rows=3, seed=0)
    _make_npz(train_dir / "seed1.npz", n_rows=4, seed=1)
    _make_npz(eval_dir / "seed10.npz", n_rows=5, seed=10)

    out = run(
        argparse.Namespace(
            data=train_dir,
            eval_data=eval_dir,
            baselines=["random"],
            train_frac=0.6,
            split_seed=1337,
            min_ece_support=30,
            out=None,
            eval_limit=None,
            obs_radius=None,
            llm_cache=tmp_path / "llm-cache.json",
        )
    )

    assert out["data_dir"] == str(train_dir)
    assert out["eval_data_dir"] == str(eval_dir)
    assert out["n_rows"] == 7
    assert out["n_eval_source_rows"] == 5
    assert out["n_train"] == 7
    assert out["n_eval"] == 5
