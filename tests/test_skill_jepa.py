from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from skill_wm.eval.minihack_object_rerank import (
    candidate_rows,
    choose_action,
    find_lava_probe_state,
    run_rerank,
)
from skill_wm.models.skill_jepa import (
    MiniHackJEPARow,
    MiniHackJEPAVocab,
    MiniHackSkillJEPA,
    SkillJEPAConfig,
    coverage_summary,
    load_minihack_jepa_shard,
    object_signature,
    split_rows_by_env,
    split_rows_by_policy,
    split_rows_by_seed,
    train_eval_summary,
)


def _row(
    i: int,
    action_name: str = "east",
    env_id: str = "skillwm-room-goal",
    policy_name: str = "scripted_nav",
) -> MiniHackJEPARow:
    before = np.full((15, 15), 100 + i % 3, dtype=np.int16)
    after = before.copy()
    after[7, 7] = 200 + (i % 5)
    return MiniHackJEPARow(
        seed=i // 4,
        env_id=env_id,
        policy_name=policy_name,
        episode=0,
        step=i,
        action=0,
        action_name=action_name,
        logical_pos_before=(1, 2),
        logical_pos_after=(2, 2),
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
    assert vocab.encode_action("unseen") == 0


def test_load_minihack_jepa_shard(tmp_path: Path) -> None:
    row = _row(0)
    path = tmp_path / "one.npz"
    np.savez_compressed(
        path,
        seed=np.array([row.seed], dtype=np.int32),
        env_id=np.array([row.env_id]),
        policy_name=np.array([row.policy_name]),
        episode=np.array([row.episode], dtype=np.int32),
        step=np.array([row.step], dtype=np.int32),
        action=np.array([row.action], dtype=np.int32),
        action_name=np.array([row.action_name]),
        logical_pos_before=np.array([row.logical_pos_before], dtype=np.int32),
        logical_pos_after=np.array([row.logical_pos_after], dtype=np.int32),
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
    assert rows[0].env_id == "skillwm-room-goal"
    assert rows[0].policy_name == "scripted_nav"
    assert rows[0].action_name == "east"
    assert rows[0].logical_pos_before == (1, 2)
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


def test_split_rows_by_env_holds_out_task() -> None:
    rows = [
        _row(i, env_id="skillwm-room-goal" if i < 12 else "skillwm-lava-detour") for i in range(24)
    ]
    train, evalu = split_rows_by_env(rows, eval_env_id="skillwm-lava-detour")
    assert {r.env_id for r in train} == {"skillwm-room-goal"}
    assert {r.env_id for r in evalu} == {"skillwm-lava-detour"}


def test_split_rows_by_policy_holds_out_interaction_policy() -> None:
    rows = [_row(i, policy_name="scripted_nav" if i < 12 else "lava_probe") for i in range(24)]
    train, evalu = split_rows_by_policy(rows, eval_policy_name="lava_probe")
    assert {r.policy_name for r in train} == {"scripted_nav"}
    assert {r.policy_name for r in evalu} == {"lava_probe"}


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
    assert summary["coverage"]["train"]["action_oov_rate"] == 0.0
    assert np.isfinite(summary["final_loss"])


def test_train_eval_summary_supports_task_split() -> None:
    rows = [
        _row(
            i,
            action_name="east" if i % 2 else "north",
            env_id="skillwm-room-goal" if i < 12 else "skillwm-lava-detour",
        )
        for i in range(24)
    ]
    summary = train_eval_summary(
        rows,
        SkillJEPAConfig(epochs=2, batch_size=8, seed=6),
        split="task",
        eval_env_id="skillwm-lava-detour",
    )
    assert summary["split"]["mode"] == "task"
    assert summary["split"]["train_env_ids"] == ["skillwm-room-goal"]
    assert summary["split"]["eval_env_ids"] == ["skillwm-lava-detour"]


def test_coverage_summary_explains_action_and_glyph_shift() -> None:
    train = [_row(i, action_name="east", env_id="skillwm-room-goal") for i in range(4)]
    evalu = [_row(20, action_name="north", env_id="skillwm-lava-detour")]
    evalu[0].glyph_after[:, :] = 9999
    vocab = MiniHackJEPAVocab.from_rows(train)
    coverage = coverage_summary(evalu, vocab, np.array([3.0]))
    assert coverage["action_oov_names"] == ["north"]
    assert coverage["action_oov_rate"] == 1.0
    assert coverage["glyph_after_oov_rate"] == 1.0
    assert coverage["action_oov_surprise"] == 3.0


def test_coverage_summary_identifies_lava_probe_transition() -> None:
    train = [_row(i, action_name="east", env_id="skillwm-room-goal") for i in range(4)]
    evalu = [
        _row(20, action_name="east", env_id="skillwm-lava-detour"),
        _row(21, action_name="north", env_id="skillwm-lava-detour"),
    ]
    evalu[0] = replace(evalu[0], logical_pos_before=(3, 2), logical_pos_after=(3, 2))
    vocab = MiniHackJEPAVocab.from_rows(train)
    coverage = coverage_summary(evalu, vocab, np.array([5.0, 1.0]))
    assert coverage["lava_probe_rate"] == 0.5
    assert coverage["lava_probe_surprise"] == 5.0


def test_object_signature_marks_lava_target() -> None:
    row = replace(
        _row(0, action_name="east", env_id="skillwm-lava-detour"),
        logical_pos_before=(3, 2),
        logical_pos_after=(3, 2),
    )
    assert object_signature(row) == "east|.->L->."


def test_object_signature_marks_water_target() -> None:
    row = replace(
        _row(0, action_name="east", env_id="skillwm-water-detour"),
        logical_pos_before=(3, 2),
        logical_pos_after=(3, 2),
    )
    vocab = MiniHackJEPAVocab.from_rows([row])

    assert object_signature(row) == "east|.->}->."
    assert vocab.object_vocab_size >= 8


def test_train_eval_summary_supports_policy_split() -> None:
    rows = [
        _row(
            i,
            action_name="east",
            env_id="skillwm-lava-detour",
            policy_name="scripted_nav" if i < 16 else "lava_probe",
        )
        for i in range(32)
    ]
    rows[-1] = replace(rows[-1], logical_pos_before=(3, 2), logical_pos_after=(3, 2))
    summary = train_eval_summary(
        rows,
        SkillJEPAConfig(epochs=2, batch_size=8, seed=8),
        split="policy",
        eval_policy_name="lava_probe",
    )
    assert summary["split"]["mode"] == "policy"
    assert summary["split"]["train_policy_names"] == ["scripted_nav"]
    assert summary["split"]["eval_policy_names"] == ["lava_probe"]
    assert summary["coverage"]["eval"]["object_signature_oov_rate"] > 0.0


def test_train_eval_summary_reports_object_aux_when_enabled() -> None:
    rows = [
        _row(
            i,
            action_name="east",
            env_id="skillwm-lava-detour",
            policy_name="scripted_nav" if i < 16 else "lava_probe",
        )
        for i in range(32)
    ]
    rows[-1] = replace(rows[-1], logical_pos_before=(3, 2), logical_pos_after=(3, 2))
    summary = train_eval_summary(
        rows,
        SkillJEPAConfig(epochs=2, batch_size=8, seed=9, object_aux_weight=0.2),
        split="policy",
        eval_policy_name="lava_probe",
    )
    assert "object_aux" in summary
    assert summary["object_aux"]["eval"]["mean_object_nll"] > 0.0
    assert summary["object_aux"]["eval"]["object_signature_oov_rate"] > 0.0


def test_object_rerank_candidate_rows_use_same_state() -> None:
    base = replace(
        _row(0, action_name="east", env_id="skillwm-lava-detour", policy_name="lava_probe"),
        logical_pos_before=(3, 2),
        logical_pos_after=(3, 2),
    )
    candidates = candidate_rows(base, ("north", "east", "south", "west"))
    assert [row.action_name for row in candidates] == ["north", "east", "south", "west"]
    assert {row.logical_pos_before for row in candidates} == {(3, 2)}
    assert object_signature(candidates[1]) == "east|.->L->."


def test_choose_action_takes_lowest_target_object_nll() -> None:
    scores = [
        {"action_name": "east", "target_object_nll": 10.0},
        {"action_name": "north", "target_object_nll": 1.0},
    ]
    assert choose_action(scores) == "north"


def test_find_lava_probe_state() -> None:
    rows = [
        _row(0, action_name="east", env_id="skillwm-lava-detour", policy_name="scripted_nav"),
        replace(
            _row(1, action_name="east", env_id="skillwm-lava-detour", policy_name="lava_probe"),
            logical_pos_before=(3, 2),
        ),
    ]
    assert find_lava_probe_state(rows).policy_name == "lava_probe"


def test_run_rerank_reports_candidate_scores() -> None:
    safe_rows = [
        _row(
            i,
            action_name="east" if i % 2 else "north",
            env_id="skillwm-lava-detour",
            policy_name="scripted_nav",
        )
        for i in range(16)
    ]
    probe = replace(
        _row(30, action_name="east", env_id="skillwm-lava-detour", policy_name="lava_probe"),
        logical_pos_before=(3, 2),
        logical_pos_after=(3, 2),
    )
    summary = run_rerank(
        [*safe_rows, probe],
        SkillJEPAConfig(epochs=2, batch_size=8, seed=10, object_aux_weight=0.2),
    )
    assert summary["proposed_action"] == "east"
    assert len(summary["scores"]) == 4
    assert any(row["action_name"] == "east" for row in summary["scores"])
