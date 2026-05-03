"""Tests for partial-observability masking via apply_obs_mask."""

from __future__ import annotations

import numpy as np

from skill_wm.eval.dataset import INVENTORY_KEYS, ScoringRow, apply_obs_mask
from skill_wm.models.baselines import (
    MASKED_TILE_ID,
    PreconditionPredictor,
    _tile_at,
)
from skill_wm.models.state_text import render_crop


def _row(crop: np.ndarray, action: int = 5, action_name: str = "do") -> ScoringRow:
    inv = np.zeros(len(INVENTORY_KEYS), dtype=np.int32)
    return ScoringRow(
        seed=0,
        episode=0,
        step=0,
        action=action,
        action_name=action_name,
        inventory_before=inv,
        player_pos_before=(32, 32),
        facing_before=(1, 0),
        sleeping_before=False,
        semantic_crop_before=crop,
        success=False,
    )


def test_apply_obs_mask_radius_none_is_noop() -> None:
    crop = np.full((15, 15), 6, dtype=np.uint8)  # all trees
    crop[7, 7] = 13  # player
    row = _row(crop)
    out = apply_obs_mask([row], None)
    assert out[0] is row, "radius=None must be a literal no-op (no copy)"


def test_apply_obs_mask_radius_2_keeps_5x5() -> None:
    crop = np.full((15, 15), 6, dtype=np.uint8)  # trees everywhere
    crop[7, 7] = 13  # player
    row = _row(crop)
    out = apply_obs_mask([row], 2)
    masked = out[0].semantic_crop_before
    # 5×5 center (rows 5..9, cols 5..9) preserved
    assert (masked[5:10, 5:10] == crop[5:10, 5:10]).all()
    # Outside that window: all MASKED_TILE_ID
    assert masked[0, 0] == MASKED_TILE_ID
    assert masked[14, 14] == MASKED_TILE_ID
    # Player marker at center retained (it's inside the window)
    assert masked[7, 7] == 13


def test_apply_obs_mask_radius_0_keeps_only_center() -> None:
    crop = np.full((15, 15), 6, dtype=np.uint8)
    crop[7, 7] = 13  # player
    row = _row(crop)
    out = apply_obs_mask([row], 0)
    masked = out[0].semantic_crop_before
    # Only the player tile is unmasked
    assert masked[7, 7] == 13
    # Cell next to player is masked
    assert masked[7, 8] == MASKED_TILE_ID
    assert masked[8, 7] == MASKED_TILE_ID


def test_apply_obs_mask_does_not_mutate_input() -> None:
    crop = np.full((15, 15), 6, dtype=np.uint8)
    crop[7, 7] = 13
    row = _row(crop)
    original_crop = row.semantic_crop_before.copy()
    _ = apply_obs_mask([row], 2)
    # The original row's crop must be unchanged.
    assert (row.semantic_crop_before == original_crop).all()


def test_tile_at_returns_masked_string() -> None:
    crop = np.full((15, 15), MASKED_TILE_ID, dtype=np.uint8)
    crop[7, 7] = 13
    assert _tile_at(crop, 1, 0) == "masked"
    assert _tile_at(crop, 0, 0) == "player"


def test_precondition_degrades_under_masking() -> None:
    """Rules MUST predict False when the cell they need to read is masked.

    Setup: player faces a tree (offset +1, 0). With full obs, the rule
    predicts True (do→tree always succeeds). With radius=0 mask, the
    front cell becomes "masked" → rule falls through to False.
    """
    crop = np.full((15, 15), 2, dtype=np.uint8)  # grass
    crop[7, 7] = 13  # player
    crop[8, 7] = 6  # tree directly to the right
    row = _row(crop, action=5, action_name="do")

    # Full obs: rule predicts True
    p_full = PreconditionPredictor().predict([row])[0]
    assert p_full == 1.0

    # Radius=0: front cell masked → rule predicts False
    masked_row = apply_obs_mask([row], 0)[0]
    p_masked = PreconditionPredictor().predict([masked_row])[0]
    assert p_masked == 0.0


def test_render_crop_uses_question_mark_for_masked() -> None:
    crop = np.full((15, 15), MASKED_TILE_ID, dtype=np.uint8)
    crop[7, 7] = 13  # player
    text = render_crop(crop)
    assert "?" in text
    assert "@" in text  # player still visible
