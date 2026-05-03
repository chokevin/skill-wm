from __future__ import annotations

import numpy as np

from skill_wm.data.collect_minihack import minihack_transitions_to_npz
from skill_wm.envs.minihack_env import (
    MiniHackWrapper,
    crop_grid,
    decode_inventory,
    decode_nle_text,
    find_player_pos,
)


def _obs(row: int, col: int, message: str = "") -> dict:
    chars = np.full((5, 7), ord("."), dtype=np.int16)
    glyphs = np.arange(35, dtype=np.int16).reshape(5, 7)
    chars[row, col] = ord("@")
    glyphs[row, col] = 999
    msg = np.array([ord(c) for c in message] + [0, 0], dtype=np.uint8)
    inv = np.zeros((2, 12), dtype=np.uint8)
    inv[0, :5] = np.frombuffer(b"a key", dtype=np.uint8)
    return {
        "chars": chars,
        "glyphs": glyphs,
        "message": msg,
        "blstats": np.array([col, row, 1, 2, 3], dtype=np.int32),
        "inv_strs": inv,
    }


def test_crop_grid_is_centered_and_padded():
    grid = np.arange(9, dtype=np.int16).reshape(3, 3)
    crop = crop_grid(grid, (0, 0), half=1, pad_value=-1)
    assert crop.tolist() == [[-1, -1, -1], [-1, 0, 1], [-1, 3, 4]]


def test_decode_nle_text_and_inventory():
    raw = np.array([ord("h"), ord("i"), 0, ord("!")], dtype=np.uint8)
    assert decode_nle_text(raw) == "hi!"
    assert decode_inventory(_obs(1, 2)) == ("a key",)


def test_find_player_pos_uses_chars_plane():
    assert find_player_pos(_obs(3, 4)) == (3, 4)


class _ActionSpace:
    n = 2


class _FakeMiniHackEnv:
    action_space = _ActionSpace()

    def __init__(self) -> None:
        self.reset_calls: list[int] = []

    def reset(self, seed=None):
        self.reset_calls.append(seed)
        return _obs(2, 3, "start"), {"reset_seed": seed}

    def step(self, action: int):
        assert action == 1
        return _obs(2, 4, "opened"), 1.0, True, False, {"success": True}


def test_wrapper_emits_minihack_transition_from_gymnasium_api():
    env = _FakeMiniHackEnv()
    wrapper = MiniHackWrapper(
        seed=10,
        env=env,
        action_names=("noop", "open"),
        crop_half=1,
    )

    obs, info = wrapper.reset(episode=2)
    assert obs["glyphs"].shape == (5, 7)
    assert info == {"reset_seed": 12}

    transition, _, _, done = wrapper.step(1)
    assert done is True
    assert transition.seed == 10
    assert transition.episode == 2
    assert transition.action_name == "open"
    assert transition.success is True
    assert transition.state_before.player_pos == (2, 3)
    assert transition.state_after.player_pos == (2, 4)
    assert transition.state_before.message == "start"
    assert transition.state_after.message == "opened"
    assert transition.state_before.glyph_crop.shape == (3, 3)
    assert transition.state_before.inventory == ("a key",)


def test_minihack_npz_writer_preserves_core_fields():
    wrapper = MiniHackWrapper(
        seed=0,
        env=_FakeMiniHackEnv(),
        action_names=("noop", "open"),
        crop_half=1,
    )
    wrapper.reset(episode=0)
    transition, *_ = wrapper.step(1)
    data = minihack_transitions_to_npz([transition])

    assert data["action_name"].tolist() == ["open"]
    assert data["success"].tolist() == [True]
    assert data["glyph_crop_before"].shape == (1, 3, 3)
    assert data["message_after"].tolist() == ["opened"]
    assert data["inventory_before"].tolist() == ["a key"]
