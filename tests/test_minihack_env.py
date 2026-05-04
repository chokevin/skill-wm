from __future__ import annotations

import numpy as np

from skill_wm.data.collect_minihack import (
    lava_probe_policy,
    minihack_transitions_to_npz,
    scripted_nav_noisy_policy,
    scripted_nav_policy,
    scripted_nav_safe_west_policy,
)
from skill_wm.envs.minihack_env import (
    MiniHackWrapper,
    crop_grid,
    decode_inventory,
    decode_nle_text,
    find_player_pos,
    minihack_success,
)
from skill_wm.envs.minihack_tasks import get_minihack_task_spec


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
    data = minihack_transitions_to_npz(
        [transition],
        env_id="skillwm-room-goal",
        policy_name="scripted_nav",
    )

    assert data["action_name"].tolist() == ["open"]
    assert data["env_id"].tolist() == ["skillwm-room-goal"]
    assert data["policy_name"].tolist() == ["scripted_nav"]
    assert data["logical_pos_before"].tolist() == [[1, 2]]
    assert data["logical_pos_after"].tolist() == [[2, 2]]
    assert data["success"].tolist() == [True]
    assert data["glyph_crop_before"].shape == (1, 3, 3)
    assert data["message_after"].tolist() == ["opened"]
    assert data["inventory_before"].tolist() == ["a key"]


def test_minihack_success_reads_end_status():
    class _TaskStatus:
        name = "TASK_SUCCESSFUL"

    assert minihack_success(0.0, {"end_status": _TaskStatus()}) is True
    assert minihack_success(0.0, {"end_status": "StepStatus.TASK_SUCCESSFUL"}) is True
    assert minihack_success(0.0, {"end_status": "RUNNING"}) is False


def test_registered_tasks_define_des_and_paths():
    room = get_minihack_task_spec("skillwm-room-goal")
    assert room is not None
    assert "STAIR:(7,2),down" in room.des_file
    assert room.next_action_toward_goal((1, 2)) == "east"
    assert room.next_action_toward_goal((35, 11), coord_offset=(34, 9)) == "east"

    lava = get_minihack_task_spec("skillwm-lava-detour")
    assert lava is not None
    assert lava.next_action_toward_goal((3, 2)) != "east"


def test_scripted_nav_policy_uses_task_shortest_path():
    spec = get_minihack_task_spec("skillwm-room-goal")
    assert spec is not None
    info = {
        "env_id": spec.env_id,
        "action_names": spec.action_names,
        "coord_offset": (34, 9),
        "obs": {"blstats": np.array([35, 11, 0], dtype=np.int32)},
    }
    action = scripted_nav_policy(np.random.default_rng(0), len(spec.action_names), info)
    assert spec.action_names[action] == "east"


def test_noisy_scripted_nav_still_returns_valid_action():
    spec = get_minihack_task_spec("skillwm-room-goal")
    assert spec is not None
    info = {
        "env_id": spec.env_id,
        "action_names": spec.action_names,
        "coord_offset": (34, 9),
        "obs": {"blstats": np.array([35, 11, 0], dtype=np.int32)},
    }
    action = scripted_nav_noisy_policy(np.random.default_rng(0), len(spec.action_names), info)
    assert 0 <= action < len(spec.action_names)


def test_lava_probe_policy_takes_one_unsafe_probe_then_recovers():
    spec = get_minihack_task_spec("skillwm-lava-detour")
    assert spec is not None
    memory: dict[str, bool] = {}
    info = {
        "env_id": spec.env_id,
        "action_names": spec.action_names,
        "coord_offset": (34, 9),
        "policy_memory": memory,
        "obs": {"blstats": np.array([37, 11, 0], dtype=np.int32)},
    }
    first = lava_probe_policy(np.random.default_rng(0), len(spec.action_names), info)
    second = lava_probe_policy(np.random.default_rng(0), len(spec.action_names), info)
    assert spec.action_names[first] == "east"
    assert memory["lava_probe_done"] is True
    assert spec.action_names[second] != "east"


def test_safe_west_policy_adds_one_safe_west_action_then_recovers():
    spec = get_minihack_task_spec("skillwm-lava-detour")
    assert spec is not None
    memory: dict[str, bool] = {}
    info = {
        "env_id": spec.env_id,
        "action_names": spec.action_names,
        "coord_offset": (34, 9),
        "policy_memory": memory,
        "obs": {"blstats": np.array([36, 9, 0], dtype=np.int32)},
    }
    first = scripted_nav_safe_west_policy(np.random.default_rng(0), len(spec.action_names), info)
    second = scripted_nav_safe_west_policy(np.random.default_rng(0), len(spec.action_names), info)

    assert spec.action_names[first] == "west"
    assert memory["safe_west_done"] is True
    assert spec.action_names[second] != "west"


def test_safe_west_trigger_is_on_lava_detour_scripted_route():
    spec = get_minihack_task_spec("skillwm-lava-detour")
    assert spec is not None
    pos = spec.start_pos
    route = [pos]
    deltas = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
    for _ in range(12):
        action = spec.next_action_toward_goal(pos)
        if action is None:
            break
        dx, dy = deltas[action]
        pos = (pos[0] + dx, pos[1] + dy)
        route.append(pos)

    assert (2, 0) in route
    assert spec.map_lines[0][1] == "."
    assert spec.map_lines[0][2] == "."
