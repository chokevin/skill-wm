"""Crafter env wrapper.

Goal: a thin layer that gives us a stable observation tuple for both the
trained world model and the LLM-as-WM baseline, plus a per-step `info_before`
snapshot so we can compute action_success cleanly without re-querying the env.
"""

from __future__ import annotations

import copy
from typing import Any

import crafter
import numpy as np

from skill_wm.data.schema import (
    ACTION_NAMES,
    Transition,
    achievements_unlocked,
    action_success,
    inventory_delta,
    vitals_delta,
)

# Half-width of the local semantic crop centered on the player.
# 7 -> 15x15 crop, plenty for predicting outcome of one action.
SEMANTIC_CROP_HALF = 7


def crop_semantic(
    semantic: np.ndarray, pos: np.ndarray, half: int = SEMANTIC_CROP_HALF
) -> np.ndarray:
    """Return a (2*half+1, 2*half+1) crop around player_pos, padded with 0 at edges."""
    h, w = semantic.shape
    px, py = int(pos[0]), int(pos[1])
    out = np.zeros((2 * half + 1, 2 * half + 1), dtype=semantic.dtype)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            sx = px + dx
            sy = py + dy
            if 0 <= sx < h and 0 <= sy < w:
                out[dy + half, dx + half] = semantic[sx, sy]
    return out


class CrafterWrapper:
    """Episode-managing wrapper that yields well-formed Transition records.

    Usage:
        env = CrafterWrapper(seed=0)
        obs, info = env.reset(episode=0)
        for step in range(N):
            action = pick_action(obs, info)
            transition, obs, info, done = env.step(action)
            ...
            if done:
                obs, info = env.reset(episode=episode + 1)
    """

    def __init__(self, seed: int = 0, area: tuple[int, int] = (64, 64)):
        self._seed = seed
        # crafter.Env constructor takes (area, view, size, length, seed)
        self.env = crafter.Env(seed=seed, area=area)
        # Sanity: action ordering matches our schema.
        env_names = list(self.env.action_names)
        if env_names != list(ACTION_NAMES):
            raise RuntimeError(
                f"Crafter action names mismatch: env={env_names} vs schema={ACTION_NAMES}"
            )
        self._episode: int = 0
        self._step: int = 0
        self._last_info: dict[str, Any] | None = None

    @property
    def num_actions(self) -> int:
        return self.env.action_space.n

    @property
    def action_names(self) -> tuple[str, ...]:
        return ACTION_NAMES

    def reset(self, episode: int) -> tuple[np.ndarray, dict[str, Any]]:
        obs = self.env.reset()
        # crafter.reset doesn't return info; do a noop step is wrong, instead
        # synthesize a starting info dict from a single noop... but that
        # mutates state. Better: peek into env._world / env._player.
        # Cheapest correct path: do a noop step that we *don't* count.
        # Since reset state has full vitals, no items, and no achievements,
        # we can synthesize the info ourselves.
        info = self._synth_initial_info()
        self._episode = episode
        self._step = 0
        self._last_info = info
        return obs, info

    def _synth_initial_info(self) -> dict[str, Any]:
        """Build the info dict that should match Crafter's post-step info shape."""
        # Crafter exposes its world/player via private attrs; use a cheap noop
        # to harvest the ground-truth starting info instead.
        # We reset again first to ensure determinism.
        # NOTE: crafter.Env.reset returns obs only, so we step noop once and
        # treat its info as the *initial* info for episode bookkeeping. The
        # collector below ignores this synthesized step.
        action_noop = ACTION_NAMES.index("noop")
        _, _, _, info = self.env.step(action_noop)
        return info

    def step(self, action: int) -> tuple[Transition, np.ndarray, dict[str, Any], bool]:
        """Take one action and return a fully-populated Transition."""
        if self._last_info is None:
            raise RuntimeError("Call reset() before step().")
        info_before = copy.deepcopy(self._last_info)
        obs_after, reward, done, info_after = self.env.step(action)
        action_name = ACTION_NAMES[action]

        crop_before = crop_semantic(info_before["semantic"], info_before["player_pos"])
        crop_after = crop_semantic(info_after["semantic"], info_after["player_pos"])

        transition = Transition(
            seed=self._seed,
            episode=self._episode,
            step=self._step,
            action=action,
            action_name=action_name,
            inventory_before=dict(info_before["inventory"]),
            player_pos_before=tuple(int(x) for x in info_before["player_pos"]),
            semantic_crop_before=crop_before,
            inventory_after=dict(info_after["inventory"]),
            player_pos_after=tuple(int(x) for x in info_after["player_pos"]),
            semantic_crop_after=crop_after,
            success=action_success(action_name, info_before, info_after),
            inventory_delta=inventory_delta(info_before["inventory"], info_after["inventory"]),
            vitals_delta=vitals_delta(info_before["inventory"], info_after["inventory"]),
            achievements_unlocked=achievements_unlocked(
                info_before["achievements"], info_after["achievements"]
            ),
            reward=float(reward),
            done=bool(done),
        )
        self._step += 1
        self._last_info = info_after
        return transition, obs_after, info_after, done
