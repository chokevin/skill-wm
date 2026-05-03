"""MiniHack/NLE adapter for the second-environment skill-WM track.

This intentionally sits next to the Crafter wrapper instead of replacing the
Crafter `Transition` schema. MiniHack exposes different state primitives
(glyphs, BLStats, messages, inventory strings), so the first port should log
those faithfully before deciding which generic predictor schema to share.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

MINIHACK_CROP_HALF = 7
PAD_GLYPH = 0


@dataclass(frozen=True)
class MiniHackState:
    """Predictor-visible MiniHack/NLE state snapshot."""

    player_pos: tuple[int, int]
    glyph_crop: np.ndarray = field(repr=False)
    blstats: np.ndarray = field(repr=False)
    message: str
    inventory: tuple[str, ...]

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["glyph_crop"] = self.glyph_crop.tolist()
        d["blstats"] = self.blstats.tolist()
        d["player_pos"] = list(self.player_pos)
        return d


@dataclass(frozen=True)
class MiniHackTransition:
    """One MiniHack state-action-outcome record."""

    seed: int
    episode: int
    step: int
    action: int
    action_name: str
    state_before: MiniHackState
    state_after: MiniHackState
    reward: float
    done: bool
    success: bool

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["state_before"] = self.state_before.to_jsonable()
        d["state_after"] = self.state_after.to_jsonable()
        return d


def crop_grid(
    grid: np.ndarray,
    center: tuple[int, int],
    half: int = MINIHACK_CROP_HALF,
    pad_value: int = PAD_GLYPH,
) -> np.ndarray:
    """Return a world-aligned square crop around ``center`` with padding."""

    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D grid, got shape {arr.shape}")
    row, col = int(center[0]), int(center[1])
    out = np.full((2 * half + 1, 2 * half + 1), pad_value, dtype=arr.dtype)
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            sr, sc = row + dr, col + dc
            if 0 <= sr < arr.shape[0] and 0 <= sc < arr.shape[1]:
                out[dr + half, dc + half] = arr[sr, sc]
    return out


def decode_nle_text(value: Any) -> str:
    """Decode NLE byte/int character arrays into a stripped Python string."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").replace("\x00", "").strip()

    arr = np.asarray(value)
    if arr.ndim == 0:
        return decode_nle_text(arr.item())
    if arr.dtype.kind in {"i", "u"}:
        chars = [chr(int(x)) for x in arr.ravel() if int(x) != 0]
        return "".join(chars).strip()
    if arr.dtype.kind == "S":
        return (
            b"".join(bytes(x) for x in arr.ravel())
            .decode("utf-8", errors="ignore")
            .replace("\x00", "")
            .strip()
        )
    if arr.dtype.kind == "U":
        return "".join(str(x) for x in arr.ravel()).replace("\x00", "").strip()
    return str(value).strip()


def decode_inventory(obs: dict[str, Any]) -> tuple[str, ...]:
    """Decode NLE ``inv_strs`` into non-empty inventory item strings."""

    raw = obs.get("inv_strs")
    if raw is None:
        return ()
    arr = np.asarray(raw)
    if arr.ndim <= 1:
        item = decode_nle_text(arr)
        return (item,) if item else ()
    items: list[str] = []
    for row in arr:
        item = decode_nle_text(row)
        if item:
            items.append(item)
    return tuple(items)


def find_player_pos(obs: dict[str, Any]) -> tuple[int, int]:
    """Find the player on the visible terminal grid.

    NLE exposes both ``chars`` and ``glyphs``. The character plane is the most
    stable way to find the visible player marker ("@"). If it is absent, fall
    back to the center of the glyph plane; callers still get a valid crop.
    """

    chars = obs.get("chars")
    if chars is not None:
        char_arr = np.asarray(chars)
        matches = np.argwhere(char_arr == ord("@"))
        if matches.size:
            row, col = matches[0]
            return int(row), int(col)

    glyphs = np.asarray(obs.get("glyphs", obs.get("chars")))
    if glyphs.ndim != 2:
        raise ValueError("MiniHack observation must contain a 2D 'glyphs' or 'chars' array")
    return int(glyphs.shape[0] // 2), int(glyphs.shape[1] // 2)


def state_from_obs(obs: dict[str, Any], half: int = MINIHACK_CROP_HALF) -> MiniHackState:
    """Build the predictor-visible MiniHack state snapshot from one observation."""

    glyphs = np.asarray(obs.get("glyphs", obs.get("chars")))
    if glyphs.ndim != 2:
        raise ValueError("MiniHack observation must contain a 2D 'glyphs' or 'chars' array")
    pos = find_player_pos(obs)
    return MiniHackState(
        player_pos=pos,
        glyph_crop=crop_grid(glyphs, pos, half=half),
        blstats=np.asarray(obs.get("blstats", []), dtype=np.int32),
        message=decode_nle_text(obs.get("message")),
        inventory=decode_inventory(obs),
    )


def minihack_success(reward: float, info_after: dict[str, Any]) -> bool:
    """Generic one-step success label for MiniHack custom tasks.

    Positive reward is the stable cross-task signal. Some wrappers also expose
    explicit booleans such as ``success`` or ``task_success``; honor those when
    present so custom tasks can provide sharper labels later.
    """

    for key in ("success", "task_success", "goal_reached"):
        value = info_after.get(key)
        if isinstance(value, bool):
            return value
    return float(reward) > 0.0


class MiniHackWrapper:
    """Episode-managing wrapper that emits ``MiniHackTransition`` records."""

    def __init__(
        self,
        env_id: str = "MiniHack-Room-5x5-v0",
        seed: int = 0,
        *,
        env: Any | None = None,
        action_names: tuple[str, ...] | None = None,
        crop_half: int = MINIHACK_CROP_HALF,
    ) -> None:
        self._seed = seed
        self._episode = 0
        self._step = 0
        self._last_obs: dict[str, Any] | None = None
        self._last_info: dict[str, Any] | None = None
        self._crop_half = crop_half

        self.env = env if env is not None else self._make_env(env_id)
        self._action_names = action_names or self._infer_action_names()

    @staticmethod
    def _make_env(env_id: str) -> Any:
        try:
            import gymnasium as gym
            import minihack  # noqa: F401  # registers environments
        except ImportError as e:
            raise RuntimeError(
                "MiniHack support is optional. Install it with "
                "`uv sync --extra minihack` or `pip install minihack`."
            ) from e
        return gym.make(env_id)

    @property
    def num_actions(self) -> int:
        return int(self.env.action_space.n)

    @property
    def action_names(self) -> tuple[str, ...]:
        return self._action_names

    def _infer_action_names(self) -> tuple[str, ...]:
        unwrapped = getattr(self.env, "unwrapped", self.env)
        actions = getattr(unwrapped, "actions", None)
        if actions is not None and len(actions) == self.num_actions:
            return tuple(str(a).split(".")[-1].lower() for a in actions)
        return tuple(f"action_{i}" for i in range(self.num_actions))

    def reset(self, episode: int) -> tuple[dict[str, Any], dict[str, Any]]:
        self._episode = episode
        self._step = 0

        reset_sig = inspect.signature(self.env.reset)
        if "seed" in reset_sig.parameters:
            result = self.env.reset(seed=self._seed + episode)
        else:
            result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        self._last_obs = dict(obs)
        self._last_info = dict(info)
        return self._last_obs, self._last_info

    def step(self, action: int) -> tuple[MiniHackTransition, dict[str, Any], dict[str, Any], bool]:
        if self._last_obs is None or self._last_info is None:
            raise RuntimeError("Call reset() before step().")
        if not 0 <= int(action) < self.num_actions:
            raise ValueError(f"action {action} out of range for {self.num_actions} actions")

        obs_before = self._last_obs
        result = self.env.step(int(action))
        if not isinstance(result, tuple):
            raise TypeError(f"env.step() must return a tuple, got {type(result).__name__}")
        if len(result) == 5:
            obs_after, reward, terminated, truncated, info_after = result
            done = bool(terminated or truncated)
        elif len(result) == 4:
            obs_after, reward, done, info_after = result
            done = bool(done)
        else:
            raise ValueError(f"env.step() returned {len(result)} values; expected 4 or 5")

        info_after = dict(info_after)
        transition = MiniHackTransition(
            seed=self._seed,
            episode=self._episode,
            step=self._step,
            action=int(action),
            action_name=self._action_names[int(action)],
            state_before=state_from_obs(obs_before, half=self._crop_half),
            state_after=state_from_obs(dict(obs_after), half=self._crop_half),
            reward=float(reward),
            done=done,
            success=minihack_success(float(reward), info_after),
        )
        self._step += 1
        self._last_obs = dict(obs_after)
        self._last_info = info_after
        return transition, self._last_obs, info_after, done
