"""Tests for skill_wm.models.trained_wm.

The model is small enough to train fully in unit tests on synthetic data.
We verify:
  - the network forward pass produces the right shape
  - fit + predict implements the Predictor protocol shape
  - on a learnable synthetic problem (label = 1 iff action == 5), training
    drives val Brier well below the chance baseline
  - save/load round-trips
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from skill_wm.eval.dataset import INVENTORY_KEYS, ScoringRow
from skill_wm.models.trained_wm import (
    NUM_ACTIONS,
    NUM_TILES,
    TrainConfig,
    TrainedWM,
    TrainedWMNet,
)


def _row(action: int, success: bool, *, seed: int = 0, step: int = 0) -> ScoringRow:
    inv = np.zeros(len(INVENTORY_KEYS), dtype=np.int32)
    crop = np.zeros((15, 15), dtype=np.uint8)
    crop[7, 7] = 13
    return ScoringRow(
        seed=seed,
        episode=0,
        step=step,
        action=action,
        action_name=f"action_{action}",
        inventory_before=inv,
        player_pos_before=(32, 32),
        facing_before=(1, 0),
        sleeping_before=False,
        semantic_crop_before=crop,
        success=success,
    )


def test_net_forward_shape() -> None:
    net = TrainedWMNet()
    crop = torch.zeros((4, 15, 15), dtype=torch.long)
    inv = torch.zeros((4, len(INVENTORY_KEYS)), dtype=torch.float32)
    action = torch.zeros((4,), dtype=torch.long)
    out = net(crop, inv, action)
    assert out.shape == (4,)


def test_net_handles_full_tile_id_range() -> None:
    """Embedding bounds: tile ids 0..NUM_TILES-1 must not error."""
    net = TrainedWMNet()
    crop = torch.full((1, 15, 15), NUM_TILES - 1, dtype=torch.long)
    inv = torch.zeros((1, len(INVENTORY_KEYS)), dtype=torch.float32)
    action = torch.tensor([NUM_ACTIONS - 1], dtype=torch.long)
    net(crop, inv, action)


def test_predict_before_fit_raises() -> None:
    wm = TrainedWM()
    with pytest.raises(RuntimeError, match="fit"):
        wm.predict([_row(0, False)])


def test_fit_predict_returns_probs_in_unit_interval() -> None:
    rows = [_row(a % NUM_ACTIONS, bool(a % 2)) for a in range(80)]
    wm = TrainedWM(TrainConfig(epochs=2, batch_size=16))
    wm.fit(rows)
    out = wm.predict(rows)
    assert out.shape == (len(rows),)
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_fit_learns_synthetic_action_label_mapping() -> None:
    """Synthetic task: action == 5 -> success, else not. With 200 rows
    and 30 epochs, the model should drive Brier well below 0.25 (chance)."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(200):
        a = int(rng.integers(0, NUM_ACTIONS))
        rows.append(_row(a, success=(a == 5), step=i))
    wm = TrainedWM(TrainConfig(epochs=30, batch_size=32, lr=3e-3, val_frac=0.2))
    wm.fit(rows)
    probs = wm.predict(rows)
    labels = np.array([r.success for r in rows], dtype=np.float64)
    brier = float(np.mean((probs - labels) ** 2))
    # Action 5 is selected ~6% of the time (1/17). Constant 0.06 predictor
    # would Brier ~0.055. We want to learn structure beyond that.
    assert brier < 0.05, f"trained WM should learn this synthetic task, got brier={brier:.4f}"


def test_save_load_round_trip(tmp_path: Path) -> None:
    rows = [_row(a % NUM_ACTIONS, bool(a % 3)) for a in range(40)]
    wm = TrainedWM(TrainConfig(epochs=2, batch_size=16))
    wm.fit(rows)
    out_before = wm.predict(rows)
    wm.save(tmp_path / "wm.pt")

    wm2 = TrainedWM(TrainConfig(epochs=1))
    wm2.load(tmp_path / "wm.pt")
    out_after = wm2.predict(rows)
    np.testing.assert_allclose(out_after, out_before, rtol=1e-5, atol=1e-6)


def test_fit_empty_rows_raises() -> None:
    wm = TrainedWM()
    with pytest.raises(ValueError, match="empty"):
        wm.fit([])


def test_history_is_populated() -> None:
    rows = [_row(a % NUM_ACTIONS, bool(a % 2)) for a in range(40)]
    wm = TrainedWM(TrainConfig(epochs=3, batch_size=16))
    wm.fit(rows)
    assert len(wm.history) > 0
    assert "train_loss" in wm.history[0]
    assert "val_brier" in wm.history[0]
