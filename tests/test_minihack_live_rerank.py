from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from skill_wm.envs.minihack_tasks import MINIHACK_CARDINAL_ACTION_NAMES
from skill_wm.eval.minihack_live_rerank import (
    LIVE_PROBES,
    aggregate_live_summaries,
    is_unsafe_lava_action,
    live_candidate_rows,
    live_probe_policy,
    next_action_toward_pos,
    oracle_object_shield_action,
    proposed_object_signature_known,
    require_object_improves,
    unsafe_target_tiles,
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


def test_oracle_object_shield_blocks_water_target() -> None:
    assert unsafe_target_tiles("skillwm-water-detour") == frozenset({"}"})
    selected = oracle_object_shield_action(
        "skillwm-water-detour",
        (3, 2),
        "east",
        MINIHACK_CARDINAL_ACTION_NAMES,
    )

    assert selected != "east"
    assert not is_unsafe_lava_action("skillwm-water-detour", (3, 2), selected)


def test_next_action_toward_west_probe_avoids_lava_column() -> None:
    assert next_action_toward_pos("skillwm-lava-detour", (3, 2), (5, 2)) != "east"


def test_live_probe_policy_supports_top_east_probe() -> None:
    memory: dict[str, bool] = {}
    info = {
        "env_id": "skillwm-lava-detour",
        "action_names": MINIHACK_CARDINAL_ACTION_NAMES,
        "coord_offset": (34, 9),
        "policy_memory": memory,
        "obs": {"blstats": np.array([37, 10, 0], dtype=np.int32)},
    }

    first = live_probe_policy(
        np.random.default_rng(0),
        len(MINIHACK_CARDINAL_ACTION_NAMES),
        info,
        LIVE_PROBES["east_top"],
    )
    second = live_probe_policy(
        np.random.default_rng(0),
        len(MINIHACK_CARDINAL_ACTION_NAMES),
        info,
        LIVE_PROBES["east_top"],
    )

    assert MINIHACK_CARDINAL_ACTION_NAMES[first] == "east"
    assert memory["east_top_lava_probe_done"] is True
    assert MINIHACK_CARDINAL_ACTION_NAMES[second] != "east"


def test_live_probe_policy_supports_west_side_probe() -> None:
    memory: dict[str, bool] = {}
    info = {
        "env_id": "skillwm-lava-detour",
        "action_names": MINIHACK_CARDINAL_ACTION_NAMES,
        "coord_offset": (34, 9),
        "policy_memory": memory,
        "obs": {"blstats": np.array([39, 11, 0], dtype=np.int32)},
    }

    first = live_probe_policy(
        np.random.default_rng(0),
        len(MINIHACK_CARDINAL_ACTION_NAMES),
        info,
        LIVE_PROBES["west"],
    )
    second = live_probe_policy(
        np.random.default_rng(0),
        len(MINIHACK_CARDINAL_ACTION_NAMES),
        info,
        LIVE_PROBES["west"],
    )

    assert MINIHACK_CARDINAL_ACTION_NAMES[first] == "west"
    assert memory["west_lava_probe_done"] is True
    assert MINIHACK_CARDINAL_ACTION_NAMES[second] != "west"


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
            "latent_mse_rerank": {
                "success_rate": 0.5,
                "unsafe_lava_executed": 1,
                "overrides": 3,
            },
        }
    }

    require_object_improves(summary)


def test_require_object_improves_can_allow_latent_match() -> None:
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
            "latent_mse_rerank": {
                "success_rate": 1.0,
                "unsafe_lava_executed": 0,
                "overrides": 4,
            },
        }
    }

    require_object_improves(summary, require_beat_latent=False)


def test_require_object_improves_can_allow_success_match() -> None:
    summary = {
        "policies": {
            "lava_probe": {
                "success_rate": 1.0,
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

    require_object_improves(summary, require_success_improvement=False)


def test_aggregate_live_summaries_combines_model_seed_runs() -> None:
    def summary(model_seed: int, latent_unsafe: int) -> dict[str, object]:
        return {
            "env_id": "skillwm-lava-detour",
            "probe": {"name": "east", "target_pos": (3, 2), "unsafe_action": "east"},
            "train_rows": 10,
            "config": {"seed": model_seed},
            "episodes": 1,
            "seed_start": 5000,
            "max_steps": 50,
            "object_threshold": 1.0,
            "latent_threshold": 0.0,
            "policies": {
                "lava_probe": {
                    "policy_name": "lava_probe",
                    "episodes": 1,
                    "successes": 0,
                    "success_rate": 0.0,
                    "transitions": 1,
                    "reward_total": 0.0,
                    "unsafe_lava_proposals": 1,
                    "unsafe_lava_executed": 1,
                    "overrides": 0,
                    "probe_selected_actions": ["east"],
                    "episodes_detail": [
                        {
                            "success": False,
                            "transitions": 1,
                            "reward_total": 0.0,
                            "unsafe_lava_proposals": 1,
                            "unsafe_lava_executed": 1,
                            "overrides": 0,
                            "probe_selected_actions": ["east"],
                        }
                    ],
                },
                "latent_mse_rerank": {
                    "policy_name": "latent_mse_rerank",
                    "episodes": 1,
                    "successes": int(latent_unsafe == 0),
                    "success_rate": float(latent_unsafe == 0),
                    "transitions": 1,
                    "reward_total": 0.0,
                    "unsafe_lava_proposals": 1,
                    "unsafe_lava_executed": latent_unsafe,
                    "overrides": int(latent_unsafe == 0),
                    "probe_selected_actions": ["east"],
                    "episodes_detail": [
                        {
                            "success": latent_unsafe == 0,
                            "transitions": 1,
                            "reward_total": 0.0,
                            "unsafe_lava_proposals": 1,
                            "unsafe_lava_executed": latent_unsafe,
                            "overrides": int(latent_unsafe == 0),
                            "probe_selected_actions": ["east"],
                        }
                    ],
                },
                "object_rerank_lava_probe": {
                    "policy_name": "object_rerank_lava_probe",
                    "episodes": 1,
                    "successes": 1,
                    "success_rate": 1.0,
                    "transitions": 1,
                    "reward_total": 0.0,
                    "unsafe_lava_proposals": 1,
                    "unsafe_lava_executed": 0,
                    "overrides": 1,
                    "probe_selected_actions": ["north"],
                    "episodes_detail": [
                        {
                            "success": True,
                            "transitions": 1,
                            "reward_total": 0.0,
                            "unsafe_lava_proposals": 1,
                            "unsafe_lava_executed": 0,
                            "overrides": 1,
                            "probe_selected_actions": ["north"],
                        }
                    ],
                },
            },
        }

    aggregate = aggregate_live_summaries([summary(1, 1), summary(7, 0)])

    assert aggregate["model_seeds"] == [1, 7]
    policies = aggregate["policies"]
    assert isinstance(policies, dict)
    assert policies["latent_mse_rerank"]["unsafe_lava_executed"] == 1
    assert policies["object_rerank_lava_probe"]["unsafe_lava_executed"] == 0
    assert policies["object_rerank_lava_probe"]["episodes_detail"][0]["model_seed"] == 1
    require_object_improves(aggregate)
