"""Roll out policies in Crafter and dump transitions to disk.

For the smell test, we use a simple random policy and a "biased random"
policy (which weights `do` and movement higher than crafting), since pure
random rarely produces success on craft actions.

Transitions are written one episode per .npz file under data/rollouts/.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from skill_wm.data.schema import ACTION_NAMES, Transition
from skill_wm.envs.crafter_env import CrafterWrapper

# Action policies #################################################


def random_policy(rng: np.random.Generator, num_actions: int):
    return int(rng.integers(0, num_actions))


def biased_random_policy(rng: np.random.Generator, num_actions: int):
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


POLICIES: dict[str, Callable] = {
    "random": random_policy,
    "biased_random": biased_random_policy,
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
        _, _ = env.reset(episode=ep)
        transitions: list[Transition] = []
        done = False
        for _ in range(max_steps_per_episode):
            action = policy(rng, env.num_actions)
            transition, _, _, done = env.step(action)
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
