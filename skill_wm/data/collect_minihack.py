"""Collect MiniHack/NLE transitions for the second-environment track."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from skill_wm.envs.minihack_env import MiniHackTransition, MiniHackWrapper


def random_policy(
    rng: np.random.Generator, num_actions: int, info: dict[str, Any] | None = None
) -> int:
    return int(rng.integers(0, num_actions))


POLICIES: dict[str, Callable[[np.random.Generator, int, dict[str, Any] | None], int]] = {
    "random": random_policy,
}


def minihack_transitions_to_npz(transitions: list[MiniHackTransition]) -> dict[str, np.ndarray]:
    """Pack MiniHack transitions into arrays for one shard."""

    if not transitions:
        return {}

    return {
        "episode": np.array([t.episode for t in transitions], dtype=np.int32),
        "step": np.array([t.step for t in transitions], dtype=np.int32),
        "seed": np.array([t.seed for t in transitions], dtype=np.int32),
        "action": np.array([t.action for t in transitions], dtype=np.int32),
        "action_name": np.array([t.action_name for t in transitions]),
        "success": np.array([t.success for t in transitions], dtype=np.bool_),
        "reward": np.array([t.reward for t in transitions], dtype=np.float32),
        "done": np.array([t.done for t in transitions], dtype=np.bool_),
        "player_pos_before": np.array(
            [t.state_before.player_pos for t in transitions], dtype=np.int32
        ),
        "player_pos_after": np.array(
            [t.state_after.player_pos for t in transitions], dtype=np.int32
        ),
        "glyph_crop_before": np.stack([t.state_before.glyph_crop for t in transitions]),
        "glyph_crop_after": np.stack([t.state_after.glyph_crop for t in transitions]),
        "blstats_before": np.stack([t.state_before.blstats for t in transitions]),
        "blstats_after": np.stack([t.state_after.blstats for t in transitions]),
        "message_before": np.array([t.state_before.message for t in transitions]),
        "message_after": np.array([t.state_after.message for t in transitions]),
        "inventory_before": np.array(["|".join(t.state_before.inventory) for t in transitions]),
        "inventory_after": np.array(["|".join(t.state_after.inventory) for t in transitions]),
    }


def collect(
    out_dir: Path,
    env_id: str,
    num_episodes: int,
    max_steps_per_episode: int,
    policy_name: str,
    seed_start: int,
) -> dict[str, int | str]:
    """Collect MiniHack rollouts and write one compressed npz per episode."""

    out_dir.mkdir(parents=True, exist_ok=True)
    policy = POLICIES[policy_name]
    rng = np.random.default_rng(seed_start)

    total_transitions = 0
    total_successes = 0
    total_reward = 0.0

    for ep in tqdm(range(num_episodes), desc=f"minihack({env_id},{policy_name})"):
        ep_seed = seed_start + ep
        env = MiniHackWrapper(env_id=env_id, seed=ep_seed)
        _, info = env.reset(episode=ep)
        transitions: list[MiniHackTransition] = []

        for _ in range(max_steps_per_episode):
            action = policy(rng, env.num_actions, info)
            transition, _, info, done = env.step(action)
            transitions.append(transition)
            total_transitions += 1
            total_successes += int(transition.success)
            total_reward += float(transition.reward)
            if done:
                break

        data = minihack_transitions_to_npz(transitions)
        safe_env = env_id.replace("/", "_")
        out_path = out_dir / f"ep_{ep:06d}_seed_{ep_seed:06d}_{safe_env}_{policy_name}.npz"
        np.savez_compressed(out_path, **data)

    return {
        "env_id": env_id,
        "episodes": num_episodes,
        "transitions": total_transitions,
        "successes": total_successes,
        "reward_total": f"{total_reward:.3f}",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/rollouts/minihack-smoke"))
    p.add_argument("--env-id", default="MiniHack-Room-5x5-v0")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--policy", choices=list(POLICIES.keys()), default="random")
    p.add_argument("--seed-start", type=int, default=0)
    args = p.parse_args()

    stats = collect(
        out_dir=args.out,
        env_id=args.env_id,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        policy_name=args.policy,
        seed_start=args.seed_start,
    )
    print("MiniHack rollout stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
