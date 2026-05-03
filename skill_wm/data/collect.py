"""Roll out policies in Crafter and dump transitions to disk.

For the smell test, we use a simple random policy and a "biased random"
policy (which weights `do` and movement higher than crafting), since pure
random rarely produces success on craft actions.

For meaningful per-action calibration on crafting actions, we add a
`scripted_craft` policy that drives the agent toward crafting unlocks
(tree -> wood -> table -> wood_pickaxe -> stone -> furnace -> stone tools
-> iron). Without it, all `make_*` actions stay at 0 positives forever
and per-action ECE on crafting is undefined.

Transitions are written one episode per .npz file under data/rollouts/.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from skill_wm.data.schema import ACTION_NAMES, Transition
from skill_wm.envs.crafter_env import CrafterWrapper

# Action policies #################################################


def random_policy(
    rng: np.random.Generator, num_actions: int, info: dict[str, Any] | None = None
) -> int:
    return int(rng.integers(0, num_actions))


def biased_random_policy(
    rng: np.random.Generator, num_actions: int, info: dict[str, Any] | None = None
) -> int:
    """Bias toward exploration actions; deweight rare crafting actions.

    Random policy almost never satisfies make_*_pickaxe preconditions, so the
    raw success rate for crafting will be ~0%. We add a small bias toward
    move/do/sleep to get richer transition diversity.
    """
    weights = np.ones(num_actions, dtype=np.float64)
    for i, name in enumerate(ACTION_NAMES):
        if name == "noop":
            weights[i] = 0.5
        elif name == "do":
            weights[i] = 4.0
        elif name.startswith("move_"):
            weights[i] = 3.0
        elif name == "sleep":
            weights[i] = 0.5
        elif name.startswith("place_"):
            weights[i] = 1.0
        elif name.startswith("make_"):
            weights[i] = 1.0
    weights /= weights.sum()
    return int(rng.choice(num_actions, p=weights))


# Tile-id constants. Must match the TILE_LEGEND in skill_wm.models.baselines.
_TILE_WATER, _TILE_GRASS, _TILE_STONE, _TILE_PATH, _TILE_SAND = 1, 2, 3, 4, 5
_TILE_TREE, _TILE_LAVA, _TILE_COAL, _TILE_IRON, _TILE_DIAMOND = 6, 7, 8, 9, 10
_TILE_TABLE, _TILE_FURNACE = 11, 12
_WALKABLE_IDS = {_TILE_GRASS, _TILE_PATH, _TILE_SAND}

# Crafter direction convention: dict(left=(-1,0), right=(1,0), up=(0,-1), down=(0,1)).
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR_TO_MOVE = {
    (-1, 0): "move_left",
    (1, 0): "move_right",
    (0, -1): "move_up",
    (0, 1): "move_down",
}


def _aidx(name: str) -> int:
    return ACTION_NAMES.index(name)


def _front_tile_id(
    sem: np.ndarray, pos: np.ndarray, facing: tuple[int, int]
) -> int:
    fx, fy = facing
    px, py = int(pos[0]), int(pos[1])
    sx, sy = px + fx, py + fy
    if 0 <= sx < sem.shape[0] and 0 <= sy < sem.shape[1]:
        return int(sem[sx, sy])
    return 0


def _adjacent_tile_ids(sem: np.ndarray, pos: np.ndarray) -> set[int]:
    ids: set[int] = set()
    px, py = int(pos[0]), int(pos[1])
    for dx, dy in _DIRS:
        sx, sy = px + dx, py + dy
        if 0 <= sx < sem.shape[0] and 0 <= sy < sem.shape[1]:
            ids.add(int(sem[sx, sy]))
    return ids


def _nearest_target(
    sem: np.ndarray, pos: np.ndarray, target_ids: set[int], max_radius: int = 20
) -> tuple[int, int] | None:
    """Return (dx, dy) to the L1-nearest cell in `target_ids` within `max_radius`,
    or None if not found. (dx, dy) is in Crafter convention."""
    px, py = int(pos[0]), int(pos[1])
    h, w = sem.shape
    best: tuple[int, int] | None = None
    best_dist = max_radius + 1
    for dx in range(-max_radius, max_radius + 1):
        for dy in range(-max_radius, max_radius + 1):
            sx, sy = px + dx, py + dy
            if not (0 <= sx < h and 0 <= sy < w):
                continue
            if int(sem[sx, sy]) in target_ids:
                d = abs(dx) + abs(dy)
                if d < best_dist:
                    best_dist = d
                    best = (dx, dy)
    return best


def _move_toward(rng: np.random.Generator, dx: int, dy: int) -> int:
    """Pick a move action that reduces |dx|+|dy|. Tie-broken randomly."""
    if abs(dx) > abs(dy):
        return _aidx("move_right" if dx > 0 else "move_left")
    if abs(dy) > abs(dx):
        return _aidx("move_down" if dy > 0 else "move_up")
    if rng.random() < 0.5:
        return _aidx("move_right" if dx > 0 else "move_left")
    return _aidx("move_down" if dy > 0 else "move_up")


def scripted_craft_policy(
    rng: np.random.Generator, num_actions: int, info: dict[str, Any] | None = None
) -> int:
    """Goal-directed policy that drives toward crafting unlocks.

    The point of this policy is data: biased_random produces zero positive
    examples for ``make_*`` because it never assembles the prerequisites
    (wood -> table -> wood_pickaxe -> ...). This policy will. It is NOT
    trying to be optimal; it is trying to **execute crafting actions when
    their preconditions are met**, so the eval harness has positive
    examples to score against.

    Priority (highest first):
      1. ``make_iron_*`` if at table+furnace with wood+coal+iron in inventory.
      2. ``make_stone_*`` if at table with wood+stone.
      3. ``make_wood_*`` if at table with wood.
      4. ``place_table`` if we have wood and no table within crop reach.
      5. ``place_furnace`` if at table with stone and no furnace within reach.
      6. ``do`` on the facing tile if it's collectible and we have the tool.
      7. Navigate one step toward the nearest interesting tile in priority
         tree > stone > coal > iron (gated by tool requirements).
      8. Fallback: biased random.
    """
    if info is None or "semantic" not in info or "player_pos" not in info:
        # Reset-time call before the first env.step() — fall back.
        return biased_random_policy(rng, num_actions, info)

    sem = info["semantic"]
    pos = np.asarray(info["player_pos"])
    facing = tuple(int(x) for x in info.get("facing", (0, 1)))
    inv = info.get("inventory", {})

    front = _front_tile_id(sem, pos, facing)
    adj = _adjacent_tile_ids(sem, pos)
    at_table = _TILE_TABLE in adj
    at_furnace = _TILE_FURNACE in adj

    if (
        at_table
        and at_furnace
        and inv.get("wood", 0) >= 1
        and inv.get("coal", 0) >= 1
        and inv.get("iron", 0) >= 1
    ):
        return _aidx(rng.choice(["make_iron_pickaxe", "make_iron_sword"]))
    if at_table and inv.get("wood", 0) >= 1 and inv.get("stone", 0) >= 1:
        return _aidx(rng.choice(["make_stone_pickaxe", "make_stone_sword"]))
    if at_table and inv.get("wood", 0) >= 1:
        return _aidx(rng.choice(["make_wood_pickaxe", "make_wood_sword"]))

    if inv.get("wood", 0) >= 2 and front in _WALKABLE_IDS:
        if _nearest_target(sem, pos, {_TILE_TABLE}, max_radius=8) is None:
            return _aidx("place_table")

    if at_table and inv.get("stone", 0) >= 4 and front in _WALKABLE_IDS:
        if _nearest_target(sem, pos, {_TILE_FURNACE}, max_radius=8) is None:
            return _aidx("place_furnace")

    # Do on facing tile (collectibles, gated by tool)
    if front == _TILE_TREE:
        return _aidx("do")
    if front == _TILE_STONE and inv.get("wood_pickaxe", 0) >= 1:
        return _aidx("do")
    if front == _TILE_COAL and inv.get("wood_pickaxe", 0) >= 1:
        return _aidx("do")
    if front == _TILE_IRON and inv.get("stone_pickaxe", 0) >= 1:
        return _aidx("do")
    if front == _TILE_DIAMOND and inv.get("iron_pickaxe", 0) >= 1:
        return _aidx("do")
    if front == _TILE_WATER and rng.random() < 0.3:
        return _aidx("do")

    # Navigate to nearest interesting tile
    nav_targets: list[tuple[set[int], str | None]] = [
        ({_TILE_TREE}, None),
        ({_TILE_STONE}, "wood_pickaxe"),
        ({_TILE_COAL}, "wood_pickaxe"),
        ({_TILE_IRON}, "stone_pickaxe"),
    ]
    for target_ids, prereq in nav_targets:
        if prereq is not None and inv.get(prereq, 0) == 0:
            continue
        offset = _nearest_target(sem, pos, target_ids)
        if offset is not None:
            return _move_toward(rng, *offset)

    return biased_random_policy(rng, num_actions, info)


def mixed_policy(
    rng: np.random.Generator, num_actions: int, info: dict[str, Any] | None = None
) -> int:
    """70% scripted, 30% biased random.

    The scripted side gives us crafting positives; the biased-random side
    gives us natural negatives across all action types (e.g. trying make_*
    without ingredients, trying move_* into stone). Both signals are needed
    for honest per-action calibration on the eval harness.
    """
    if rng.random() < 0.7:
        return scripted_craft_policy(rng, num_actions, info)
    return biased_random_policy(rng, num_actions, info)


POLICIES: dict[str, Callable] = {
    "random": random_policy,
    "biased_random": biased_random_policy,
    "scripted_craft": scripted_craft_policy,
    "mixed": mixed_policy,
}


# Storage #########################################################


def transitions_to_npz(transitions: list[Transition]) -> dict[str, np.ndarray]:
    """Pack a list of transitions into numpy arrays for one .npz file."""
    n = len(transitions)
    if n == 0:
        return {}

    data = {
        "episode": np.array([t.episode for t in transitions], dtype=np.int32),
        "step": np.array([t.step for t in transitions], dtype=np.int32),
        "seed": np.array([t.seed for t in transitions], dtype=np.int32),
        "action": np.array([t.action for t in transitions], dtype=np.int32),
        "success": np.array([t.success for t in transitions], dtype=np.bool_),
        "reward": np.array([t.reward for t in transitions], dtype=np.float32),
        "done": np.array([t.done for t in transitions], dtype=np.bool_),
        "player_pos_before": np.array([t.player_pos_before for t in transitions], dtype=np.int32),
        "player_pos_after": np.array([t.player_pos_after for t in transitions], dtype=np.int32),
        "facing_before": np.array([t.facing_before for t in transitions], dtype=np.int8),
        "facing_after": np.array([t.facing_after for t in transitions], dtype=np.int8),
        "sleeping_before": np.array([t.sleeping_before for t in transitions], dtype=np.bool_),
        "sleeping_after": np.array([t.sleeping_after for t in transitions], dtype=np.bool_),
        "semantic_crop_before": np.stack([t.semantic_crop_before for t in transitions]),
        "semantic_crop_after": np.stack([t.semantic_crop_after for t in transitions]),
        "inventory_delta": np.stack([t.inventory_delta for t in transitions]),
        "vitals_delta": np.stack([t.vitals_delta for t in transitions]),
    }

    # Inventory before/after: flatten to (n, n_items+n_vitals) keyed in schema order.
    from skill_wm.data.schema import ITEM_KEYS, VITAL_KEYS

    keys = list(ITEM_KEYS) + list(VITAL_KEYS)
    inv_before = np.zeros((n, len(keys)), dtype=np.int32)
    inv_after = np.zeros((n, len(keys)), dtype=np.int32)
    for i, t in enumerate(transitions):
        for j, k in enumerate(keys):
            inv_before[i, j] = t.inventory_before.get(k, 0)
            inv_after[i, j] = t.inventory_after.get(k, 0)
    data["inventory_before"] = inv_before
    data["inventory_after"] = inv_after

    # Achievements unlocked: pack as variable-length lists via object array of strings.
    data["achievements_unlocked"] = np.array(
        ["|".join(t.achievements_unlocked) for t in transitions]
    )
    return data


def collect(
    out_dir: Path,
    num_episodes: int,
    max_steps_per_episode: int,
    policy_name: str,
    seed_start: int,
) -> dict[str, int]:
    """Collect rollouts and write one .npz per episode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = POLICIES[policy_name]
    rng = np.random.default_rng(seed_start)

    total_transitions = 0
    total_successes = 0
    achievements_per_kind: dict[str, int] = {}

    for ep in tqdm(range(num_episodes), desc=f"rollouts({policy_name})"):
        ep_seed = seed_start + ep
        env = CrafterWrapper(seed=ep_seed)
        _, info = env.reset(episode=ep)
        transitions: list[Transition] = []
        done = False
        for _ in range(max_steps_per_episode):
            action = policy(rng, env.num_actions, info)
            transition, _, info, done = env.step(action)
            transitions.append(transition)
            total_transitions += 1
            if transition.success:
                total_successes += 1
            for ach in transition.achievements_unlocked:
                achievements_per_kind[ach] = achievements_per_kind.get(ach, 0) + 1
            if done:
                break
        data = transitions_to_npz(transitions)
        out_path = out_dir / f"ep_{ep:06d}_seed_{ep_seed:06d}_{policy_name}.npz"
        np.savez_compressed(out_path, **data)

    return {
        "episodes": num_episodes,
        "transitions": total_transitions,
        "successes": total_successes,
        "achievements_total": sum(achievements_per_kind.values()),
        **{f"ach_{k}": v for k, v in achievements_per_kind.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/rollouts"))
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--policy", choices=list(POLICIES.keys()), default="biased_random")
    p.add_argument("--seed-start", type=int, default=0)
    args = p.parse_args()

    stats = collect(
        out_dir=args.out,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        policy_name=args.policy,
        seed_start=args.seed_start,
    )
    print("Rollout stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
