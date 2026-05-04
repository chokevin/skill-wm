from __future__ import annotations

from pathlib import Path

import numpy as np

from skill_wm.models.skill_jepa import (
    MiniHackJEPARow,
    MiniHackJEPAVocab,
    MiniHackSkillJEPA,
    SkillJEPAConfig,
    load_minihack_jepa_shard,
    split_rows_by_seed,
    train_eval_summary,
)


def _row(i: int, action_name: str = "east") -> MiniHackJEPARow:
    before = np.full((15, 15), 100 + i % 3, dtype=np.int16)
    after = before.copy()
    after[7, 7] = 200 + (i % 5)
    return MiniHackJEPARow(
        seed=i // 4,
        episode=0,
        step=i,
        action=0,
        action_name=action_name,
        glyph_before=before,
        glyph_after=after,
        blstats_before=np.arange(27, dtype=np.int32),
        blstats_after=np.arange(27, dtype=np.int32) + 1,
        message_before="start",
        message_after="done" if i % 5 == 0 else "",
        inventory_before="a key|an apple",
        inventory_after="a key|an apple",
        success=i % 5 == 0,
        done=i % 5 == 0,
        reward=0.0,
    )


def test_minihack_jepa_vocab_encodes_glyphs_and_actions() -> None:
    rows = [_row(0, "east"), _row(1, "north")]
    vocab = MiniHackJEPAVocab.from_rows(rows)
    encoded = vocab.encode_glyphs(rows[0].glyph_before)
    assert encoded.shape == (15, 15)
    assert encoded.max() > 0
    assert vocab.encode_action("east") != vocab.encode_action("north")


def test_load_minihack_jepa_shard(tmp_path: Path) -> None:
    row = _row(0)
    path = tmp_path / "one.npz"
    np.savez_compressed(
        path,
        seed=np.array([row.seed], dtype=np.int32),
        episode=np.array([row.episode], dtype=np.int32),
        step=np.array([row.step], dtype=np.int32),
        action=np.array([row.action], dtype=np.int32),
        action_name=np.array([row.action_name]),
        glyph_crop_before=np.stack([row.glyph_before]),
        glyph_crop_after=np.stack([row.glyph_after]),
        blstats_before=np.stack([row.blstats_before]),
        blstats_after=np.stack([row.blstats_after]),
        message_before=np.array([row.message_before]),
        message_after=np.array([row.message_after]),
        inventory_before=np.array([row.inventory_before]),
        inventory_after=np.array([row.inventory_after]),
        success=np.array([row.success], dtype=np.bool_),
        done=np.array([row.done], dtype=np.bool_),
        reward=np.array([row.reward], dtype=np.float32),
    )

    rows = load_minihack_jepa_shard(path)
    assert len(rows) == 1
    assert rows[0].action_name == "east"
    assert rows[0].glyph_before.shape == (15, 15)
    assert rows[0].inventory_before == "a key|an apple"


def test_skill_jepa_fit_and_surprise_shape() -> None:
    rows = [_row(i, "east" if i % 2 else "north") for i in range(24)]
    vocab = MiniHackJEPAVocab.from_rows(rows)
    model = MiniHackSkillJEPA(vocab, SkillJEPAConfig(epochs=2, batch_size=8, seed=3))
    model.fit(rows)
    scores = model.surprise(rows)
    assert len(model.history) == 2
    assert scores.shape == (len(rows),)
    assert np.isfinite(scores).all()


def test_split_rows_by_seed_is_disjoint() -> None:
    rows = [_row(i) for i in range(24)]
    train, evalu = split_rows_by_seed(rows, train_frac=0.5, seed=4)
    train_seeds = {r.seed for r in train}
    eval_seeds = {r.seed for r in evalu}
    assert train_seeds
    assert eval_seeds
    assert train_seeds.isdisjoint(eval_seeds)
    assert len(train) + len(evalu) == len(rows)


def test_train_eval_summary_is_json_ready() -> None:
    rows = [_row(i, "east" if i % 2 else "north") for i in range(24)]
    summary = train_eval_summary(
        rows,
        SkillJEPAConfig(epochs=2, batch_size=8, seed=5),
        train_frac=0.5,
    )
    assert summary["rows"] == len(rows)
    assert summary["train"]["rows"] > 0
    assert summary["eval"]["rows"] > 0
    assert np.isfinite(summary["final_loss"])
