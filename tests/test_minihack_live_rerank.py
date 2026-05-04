from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from skill_wm.envs.minihack_tasks import MINIHACK_CARDINAL_ACTION_NAMES
from skill_wm.eval.minihack_live_rerank import (
    is_unsafe_lava_action,
    live_candidate_rows,
    oracle_object_shield_action,
    proposed_object_signature_known,
    require_object_improves,
)
from skill_wm.models.skill_jepa import object_signature


def _obs_for_logical_pos(
    logical_pos: tuple[int, int],
    coord_offset: tuple[int, int] = (34, 9),
) -> dict:
    chars = np.full((5, 7), ord("."), dtype=np.int16)
    glyphs = np.arange(35, dtype=np.int16).reshape(5, 7)
    chars[2, 3] = ord("@")
    glyphs[2, 3] = 999
    return {
        "chars": chars,
        "glyphs": glyphs,
        "message": np.array([0], dtype=np.uint8),
        "blstats": np.array(
            [logical_pos[0] + coord_offset[0], logical_pos[1] + coord_offset[1], 0],
            dtype=np.int32,
        ),
        "inv_strs": np.zeros((1, 4), dtype=np.uint8),
    }


def test_live_candidate_rows_do_not_need_future_state() -> None:
    candidates = live_candidate_rows(
        _obs_for_logical_pos((3, 2)),
        env_id="skillwm-lava-detour",
        action_names=MINIHACK_CARDINAL_ACTION_NAMES,
        coord_offset=(34, 9),
        seed=1,
        step=2,
    )

    assert [row.action_name for row in candidates] == ["north", "east", "south", "west"]
    assert {row.logical_pos_before for row in candidates} == {(3, 2)}
    assert {row.logical_pos_after for row in candidates} == {(3, 2)}
    assert object_signature(candidates[1]) == "east|.->L->."


def test_oracle_object_shield_blocks_lava_target() -> None:
    selected = oracle_object_shield_action(
        "skillwm-lava-detour",
        (3, 2),
        "east",
        MINIHACK_CARDINAL_ACTION_NAMES,
    )

    assert selected != "east"
    assert not is_unsafe_lava_action("skillwm-lava-detour", (3, 2), selected)
    assert (
        oracle_object_shield_action(
            "skillwm-lava-detour",
            (3, 2),
            "north",
            MINIHACK_CARDINAL_ACTION_NAMES,
        )
        == "north"
    )


def test_proposed_object_signature_known_uses_training_signatures() -> None:
    model = SimpleNamespace(
        vocab=SimpleNamespace(object_signatures=frozenset({"east|.->.->."}))
    )

    assert proposed_object_signature_known(
        model,
        _obs_for_logical_pos((1, 2)),
        env_id="skillwm-lava-detour",
        action_names=MINIHACK_CARDINAL_ACTION_NAMES,
        coord_offset=(34, 9),
        proposed_action="east",
    )
    assert not proposed_object_signature_known(
        model,
        _obs_for_logical_pos((3, 2)),
        env_id="skillwm-lava-detour",
        action_names=MINIHACK_CARDINAL_ACTION_NAMES,
        coord_offset=(34, 9),
        proposed_action="east",
    )


def test_require_object_improves_checks_success_and_unsafe_moves() -> None:
    summary = {
        "policies": {
            "lava_probe": {
                "success_rate": 0.0,
                "unsafe_lava_executed": 4,
                "overrides": 0,
            },
            "object_rerank_lava_probe": {
                "success_rate": 1.0,
                "unsafe_lava_executed": 0,
                "overrides": 4,
            },
        }
    }

    require_object_improves(summary)
