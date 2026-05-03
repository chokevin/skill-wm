"""Tests for skill_wm.models.{baselines, state_text, llm_wm}."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from skill_wm.eval.dataset import INVENTORY_KEYS, ScoringRow
from skill_wm.models.baselines import (
    MarginalPredictor,
    PreconditionPredictor,
    RandomPredictor,
    _front_tile,
    _has_adjacent,
    _tile_at,
)
from skill_wm.models.llm_wm import (
    FakeLLMClient,
    LLMResponse,
    LLMWorldModel,
    _renormalize,
)
from skill_wm.models.state_text import (
    SYSTEM_PROMPT,
    render_crop,
    render_user_prompt,
)


def _row(
    action: int,
    action_name: str,
    *,
    seed: int = 0,
    episode: int = 0,
    step: int = 0,
    inventory: dict[str, int] | None = None,
    facing: tuple[int, int] = (1, 0),
    crop: np.ndarray | None = None,
    success: bool = False,
    sleeping: bool = False,
) -> ScoringRow:
    """Build a ScoringRow for a unit test scenario."""
    inv = inventory or {}
    inv_arr = np.array([inv.get(k, 0) for k in INVENTORY_KEYS], dtype=np.int32)
    if crop is None:
        crop = np.zeros((15, 15), dtype=np.uint8)
        crop[7, 7] = 13  # player marker at center
    return ScoringRow(
        seed=seed,
        episode=episode,
        step=step,
        action=action,
        action_name=action_name,
        inventory_before=inv_arr,
        player_pos_before=(32, 32),
        facing_before=facing,
        sleeping_before=sleeping,
        semantic_crop_before=crop,
        success=success,
    )


# ===== RandomPredictor =====


def test_random_predictor_always_half() -> None:
    rows = [_row(0, "noop") for _ in range(5)]
    out = RandomPredictor().predict(rows)
    assert np.allclose(out, 0.5)


# ===== MarginalPredictor =====


def test_marginal_predictor_per_action_rate_from_train() -> None:
    """If train shows make_wood_pickaxe succeeds 3/4 times, eval prediction
    for that action should be 0.75."""
    train = [_row(11, "make_wood_pickaxe", success=True) for _ in range(3)] + [
        _row(11, "make_wood_pickaxe", success=False)
    ]
    p = MarginalPredictor()
    p.fit(train)
    eval_rows = [_row(11, "make_wood_pickaxe")]
    assert p.predict(eval_rows)[0] == pytest.approx(0.75)


def test_marginal_predictor_falls_back_to_global_rate_for_unseen_action() -> None:
    """Unseen action gets the global base rate, not 0."""
    train = [_row(5, "do", success=True), _row(5, "do", success=False)]  # base rate 0.5
    p = MarginalPredictor()
    p.fit(train)
    # Action 11 (make_wood_pickaxe) was never in train.
    out = p.predict([_row(11, "make_wood_pickaxe")])
    assert out[0] == pytest.approx(0.5)


# ===== PreconditionPredictor =====


def test_tile_at_world_aligned() -> None:
    """`_tile_at(crop, dx, dy)` reads the world-aligned offset, not transposed."""
    crop = np.zeros((15, 15), dtype=np.uint8)
    crop[7 + 1, 7 + 0] = 6  # tree at (dx=+1, dy=0) — to the player's right
    crop[7 + 0, 7 + 1] = 1  # water at (dx=0, dy=+1) — below the player
    assert _tile_at(crop, 1, 0) == "tree"
    assert _tile_at(crop, 0, 1) == "water"


def test_precondition_move_blocked_by_stone() -> None:
    """move_right onto stone is unwalkable -> predict 0."""
    crop = np.zeros((15, 15), dtype=np.uint8)
    crop[7, 7] = 13  # player
    crop[7 + 1, 7 + 0] = 3  # stone to the right (dx=+1)
    row = _row(2, "move_right", crop=crop)
    out = PreconditionPredictor().predict([row])
    assert out[0] == 0.0


def test_precondition_move_onto_grass_succeeds() -> None:
    crop = np.zeros((15, 15), dtype=np.uint8)
    crop[7, 7] = 13
    crop[7 + 1, 7 + 0] = 2  # grass to the right
    row = _row(2, "move_right", crop=crop)
    assert PreconditionPredictor().predict([row])[0] == 1.0


def test_precondition_do_tree_always_succeeds_no_pickaxe_needed() -> None:
    crop = np.zeros((15, 15), dtype=np.uint8)
    crop[7, 7] = 13
    crop[7 + 1, 7 + 0] = 6  # tree in front (player faces right by default in fixture)
    row = _row(5, "do", facing=(1, 0), crop=crop)
    assert PreconditionPredictor().predict([row])[0] == 1.0


def test_precondition_do_stone_needs_wood_pickaxe() -> None:
    crop = np.zeros((15, 15), dtype=np.uint8)
    crop[7, 7] = 13
    crop[7 + 1, 7 + 0] = 3  # stone in front
    no_pick = _row(5, "do", facing=(1, 0), crop=crop)
    with_pick = _row(5, "do", facing=(1, 0), crop=crop, inventory={"wood_pickaxe": 1})
    out = PreconditionPredictor().predict([no_pick, with_pick])
    assert out[0] == 0.0
    assert out[1] == 1.0


def test_precondition_make_wood_pickaxe_needs_table_and_wood() -> None:
    crop = np.zeros((15, 15), dtype=np.uint8)
    crop[7, 7] = 13
    crop[7 + 1, 7 + 0] = 11  # table to the right
    no_wood = _row(11, "make_wood_pickaxe", crop=crop)
    has_wood = _row(11, "make_wood_pickaxe", crop=crop, inventory={"wood": 1})
    no_table_no_wood = _row(11, "make_wood_pickaxe")  # blank crop, blank inv
    out = PreconditionPredictor().predict([no_wood, has_wood, no_table_no_wood])
    assert out[0] == 0.0  # adjacent to table but no wood
    assert out[1] == 1.0  # adjacent to table AND wood
    assert out[2] == 0.0


def test_precondition_place_plant_needs_sapling_and_grass_in_front() -> None:
    crop = np.zeros((15, 15), dtype=np.uint8)
    crop[7, 7] = 13
    crop[7 + 1, 7 + 0] = 2  # grass in front
    sapling = _row(10, "place_plant", crop=crop, inventory={"sapling": 1}, facing=(1, 0))
    no_sapling = _row(10, "place_plant", crop=crop, facing=(1, 0))
    no_grass_crop = np.zeros((15, 15), dtype=np.uint8)
    no_grass_crop[7, 7] = 13
    no_grass_crop[7 + 1, 7 + 0] = 4  # path in front (not grass)
    no_grass = _row(10, "place_plant", crop=no_grass_crop, inventory={"sapling": 1}, facing=(1, 0))
    out = PreconditionPredictor().predict([sapling, no_sapling, no_grass])
    assert out[0] == 1.0
    assert out[1] == 0.0
    assert out[2] == 0.0


def test_has_adjacent_finds_table_in_any_direction() -> None:
    for d in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        crop = np.zeros((15, 15), dtype=np.uint8)
        crop[7 + d[0], 7 + d[1]] = 11  # table at offset d
        row = _row(11, "make_wood_pickaxe", crop=crop)
        assert _has_adjacent(row, "table"), f"failed for offset {d}"


def test_front_tile_uses_facing_not_default_direction() -> None:
    crop = np.zeros((15, 15), dtype=np.uint8)
    crop[7, 7] = 13
    crop[7 + 0, 7 + (-1)] = 6  # tree above (dy=-1)
    crop[7 + 1, 7 + 0] = 7  # lava to the right (dx=+1)
    row_facing_up = _row(5, "do", facing=(0, -1), crop=crop)
    row_facing_right = _row(5, "do", facing=(1, 0), crop=crop)
    assert _front_tile(row_facing_up) == "tree"
    assert _front_tile(row_facing_right) == "lava"


# ===== state_text =====


def test_render_crop_includes_player_glyph() -> None:
    crop = np.zeros((15, 15), dtype=np.uint8)
    crop[7, 7] = 13  # player
    out = render_crop(crop)
    assert "@" in out
    assert out.count("\n") == 14  # 15 rows


def test_render_user_prompt_contains_action_and_facing() -> None:
    row = _row(2, "move_right", facing=(1, 0))
    prompt = render_user_prompt(row)
    assert "move_right" in prompt
    assert "facing right" in prompt
    assert "Yes or No" in prompt
    # SYSTEM_PROMPT mentions Crafter rules; user prompt mentions the legend.
    assert "grass" in SYSTEM_PROMPT or "grass" in prompt


# ===== llm_wm =====


def test_renormalize_yes_dominant() -> None:
    """If Yes has logprob -0.1 and No has -2.0, p_yes ~ 0.87."""
    top = [("Yes", -0.1), ("No", -2.0)]
    p = _renormalize(top)
    expected = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.0))
    assert p == pytest.approx(expected)


def test_renormalize_no_yes_no_in_top_returns_half() -> None:
    """Defensive: if neither token is in top, default to 0.5."""
    top = [("maybe", -0.5), ("perhaps", -0.7)]
    assert _renormalize(top) == 0.5


def test_renormalize_handles_only_yes() -> None:
    """If only Yes appears (No was outside top-K), p_yes -> 1.0."""
    top = [("Yes", -0.1)]
    assert _renormalize(top) == 1.0


def test_fake_llm_client_returns_requested_probability() -> None:
    """FakeLLMClient must return whatever answer_fn says."""
    client = FakeLLMClient(answer_fn=lambda _u: 0.8)
    resp = client.complete("sys", "user")
    assert resp.p_yes == pytest.approx(0.8)
    assert client.calls == 1


def test_llm_wm_caches_predictions(tmp_path: Path) -> None:
    """Second call on the same row must hit cache (no extra LLM call)."""
    cache = tmp_path / "cache.json"
    client = FakeLLMClient(answer_fn=lambda _u: 0.7)
    wm = LLMWorldModel(client=client, cache_path=cache)
    rows = [_row(2, "move_right", seed=1, episode=0, step=0)]
    out1 = wm.predict(rows)
    out2 = wm.predict(rows)
    assert client.calls == 1, "second predict should hit cache, not call LLM again"
    assert np.allclose(out1, out2)
    assert np.allclose(out1, 0.7)


def test_llm_wm_cache_invalidates_on_prompt_change(tmp_path: Path) -> None:
    """If we rebuild LLMWorldModel after editing the prompt, the cached key
    won't match (because key includes prompt md5), so the LLM is called again."""
    cache = tmp_path / "cache.json"
    client1 = FakeLLMClient(answer_fn=lambda _u: 0.3)
    wm1 = LLMWorldModel(client=client1, cache_path=cache)
    row = _row(2, "move_right")
    wm1.predict([row])

    # New WM with same cache; the row hashes the same, so cached value reused.
    client2 = FakeLLMClient(answer_fn=lambda _u: 0.99)
    wm2 = LLMWorldModel(client=client2, cache_path=cache)
    out = wm2.predict([row])
    assert client2.calls == 0, "same prompt should hit cache"
    assert np.allclose(out, 0.3)


def test_llm_wm_predict_preserves_row_order(tmp_path: Path) -> None:
    """Order of returned probs must match order of input rows."""
    cache = tmp_path / "cache.json"
    actions = [(0, "noop"), (2, "move_right"), (5, "do")]

    def per_action_p(user: str) -> float:
        if "noop" in user:
            return 0.1
        if "move_right" in user:
            return 0.5
        return 0.9

    wm = LLMWorldModel(client=FakeLLMClient(answer_fn=per_action_p), cache_path=cache)
    rows = [_row(a, n, seed=1, step=i) for i, (a, n) in enumerate(actions)]
    out = wm.predict(rows)
    assert out[0] == pytest.approx(0.1)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(0.9)


def test_llm_response_dataclass() -> None:
    r = LLMResponse(p_yes=0.42, raw_top=[("Yes", -1.0)])
    assert r.p_yes == 0.42
