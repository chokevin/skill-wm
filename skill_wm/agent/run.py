"""T2 agent-loop harness: use outcome predictors at decision time.

This is deliberately small and local-first. It answers a sharper question than
the T1 prediction harness:

    Does a calibrated one-step success predictor improve a live Crafter agent?

The harness supports two decision modes:

1. Plain rollout policies from `skill_wm.data.collect` (`biased_random`,
   `scripted_craft`, `mixed`, ...).
2. Predictor-scored policies:
   - `*-greedy`: score all 17 actions and take argmax P(success).
   - `*-rerank-<proposal>`: sample K candidates from a proposal policy, score
     them, and take the best. This mimics "LLM/planner proposes a few plausible
     actions; WM filters/reranks" without spending LLM calls.

The key metric is not just per-step success. A one-step success model can choose
safe, locally-successful actions forever while making no long-horizon progress.
So we report success rate, reward, action mix, and achievements.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from skill_wm.data.collect import (
    POLICIES,
    biased_random_policy,
)
from skill_wm.data.schema import ACTION_NAMES
from skill_wm.envs.crafter_env import CrafterWrapper, crop_semantic
from skill_wm.eval.dataset import INVENTORY_KEYS, ScoringRow, load_dir
from skill_wm.models.baselines import (
    InventoryOnlyMLP,
    MarginalPredictor,
    PreconditionPredictor,
    PreconditionWithBackoff,
    Predictor,
)

log = logging.getLogger("skill_wm.agent.run")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class AgentPolicy(Protocol):
    name: str

    def act(
        self,
        rng: np.random.Generator,
        env: CrafterWrapper,
        info: dict,
        episode: int,
        step: int,
    ) -> int:
        """Return an action id for the current live env state."""


class RolloutPolicy:
    """Adapter for existing collection policies."""

    def __init__(self, name: str):
        if name not in POLICIES:
            raise ValueError(f"unknown rollout policy {name!r}; choices={sorted(POLICIES)}")
        self.name = name
        self._fn = POLICIES[name]

    def act(
        self,
        rng: np.random.Generator,
        env: CrafterWrapper,
        info: dict,
        episode: int,
        step: int,
    ) -> int:
        return int(self._fn(rng, env.num_actions, info))


class PredictorPolicy:
    """Score candidate actions with a T1 predictor, then choose the best."""

    def __init__(
        self,
        name: str,
        predictor: Predictor,
        mode: str,
        *,
        proposal: str = "biased_random",
        num_candidates: int = 8,
        epsilon: float = 0.0,
    ):
        if mode not in {"greedy", "rerank"}:
            raise ValueError(f"unknown predictor policy mode {mode!r}")
        self.name = name
        self.predictor = predictor
        self.mode = mode
        self.proposal = _canonical_proposal_name(proposal)
        self.num_candidates = num_candidates
        self.epsilon = epsilon

    def act(
        self,
        rng: np.random.Generator,
        env: CrafterWrapper,
        info: dict,
        episode: int,
        step: int,
    ) -> int:
        if self.epsilon > 0 and rng.random() < self.epsilon:
            return int(biased_random_policy(rng, env.num_actions, info))

        candidates = self._candidates(rng, env, info)
        rows = _state_action_rows(env, info, episode, step, candidates)
        probs = self.predictor.predict(rows)
        best = np.flatnonzero(probs == probs.max())
        return int(candidates[int(rng.choice(best))])

    def _candidates(
        self, rng: np.random.Generator, env: CrafterWrapper, info: dict
    ) -> list[int]:
        if self.mode == "greedy":
            return list(range(env.num_actions))

        # Sample K proposal actions from the cheap "LLM/planner stand-in" policy,
        # preserving order and deduplicating. Always keep at least one action.
        proposer = POLICIES[self.proposal]
        seen: set[int] = set()
        out: list[int] = []
        for _ in range(max(1, self.num_candidates)):
            a = int(proposer(rng, env.num_actions, info))
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out or [0]


def _canonical_proposal_name(name: str) -> str:
    aliases = {
        "biased": "biased_random",
        "scripted": "scripted_craft",
    }
    out = aliases.get(name, name)
    if out not in POLICIES:
        raise ValueError(f"unknown proposal policy {name!r}; choices={sorted(POLICIES)}")
    return out


def _inventory_array(info: dict) -> np.ndarray:
    inv = info["inventory"]
    return np.array([inv.get(k, 0) for k in INVENTORY_KEYS], dtype=np.int32)


def _state_action_rows(
    env: CrafterWrapper,
    info: dict,
    episode: int,
    step: int,
    actions: list[int],
) -> list[ScoringRow]:
    crop = crop_semantic(info["semantic"], info["player_pos"])
    inv = _inventory_array(info)
    pos = tuple(int(x) for x in info["player_pos"])
    facing = tuple(int(x) for x in info.get("facing", (0, 1)))
    sleeping = bool(info.get("sleeping", False))
    rows: list[ScoringRow] = []
    for action in actions:
        rows.append(
            ScoringRow(
                seed=env._seed,
                episode=episode,
                step=step,
                action=int(action),
                action_name=ACTION_NAMES[int(action)],
                inventory_before=inv,
                player_pos_before=pos,
                facing_before=facing,
                sleeping_before=sleeping,
                semantic_crop_before=crop,
                success=False,
            )
        )
    return rows


@dataclass
class RolloutStats:
    episodes: int = 0
    transitions: int = 0
    successes: int = 0
    total_reward: float = 0.0
    action_counts: Counter[str] | None = None
    action_successes: Counter[str] | None = None
    achievement_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        self.action_counts = Counter()
        self.action_successes = Counter()
        self.achievement_counts = Counter()

    def to_dict(self) -> dict:
        assert self.action_counts is not None
        assert self.action_successes is not None
        assert self.achievement_counts is not None
        action_success_rate = {
            a: self.action_successes[a] / self.action_counts[a]
            for a in sorted(self.action_counts)
            if self.action_counts[a] > 0
        }
        return {
            "episodes": self.episodes,
            "transitions": self.transitions,
            "successes": self.successes,
            "success_rate": self.successes / max(self.transitions, 1),
            "total_reward": self.total_reward,
            "avg_reward_per_episode": self.total_reward / max(self.episodes, 1),
            "total_achievements": sum(self.achievement_counts.values()),
            "avg_achievements_per_episode": sum(self.achievement_counts.values())
            / max(self.episodes, 1),
            "unique_achievements": len(self.achievement_counts),
            "achievement_counts": dict(sorted(self.achievement_counts.items())),
            "action_counts": dict(sorted(self.action_counts.items())),
            "action_success_rate": action_success_rate,
        }


def rollout_policy(
    policy: AgentPolicy,
    *,
    episodes: int,
    max_steps: int,
    seed_start: int,
) -> dict:
    stats = RolloutStats(episodes=episodes)
    assert stats.action_counts is not None
    assert stats.action_successes is not None
    assert stats.achievement_counts is not None

    for ep in range(episodes):
        env = CrafterWrapper(seed=seed_start + ep)
        _, info = env.reset(episode=ep)
        rng = np.random.default_rng(seed_start + ep)
        done = False
        step = 0
        while step < max_steps and not done:
            action = int(policy.act(rng, env, info, ep, step))
            transition, _, info, done = env.step(action)
            stats.transitions += 1
            stats.total_reward += transition.reward
            stats.action_counts[transition.action_name] += 1
            if transition.success:
                stats.successes += 1
                stats.action_successes[transition.action_name] += 1
            for ach in transition.achievements_unlocked:
                stats.achievement_counts[ach] += 1
            step += 1

    return stats.to_dict()


def _build_predictor(kind: str) -> Predictor:
    if kind == "marginal":
        return MarginalPredictor()
    if kind == "precondition":
        return PreconditionPredictor()
    if kind == "precondition-backoff":
        return PreconditionWithBackoff()
    if kind == "inv-mlp":
        return InventoryOnlyMLP()
    if kind == "trained":
        from skill_wm.models.trained_wm import TrainedWM

        return TrainedWM()
    raise ValueError(f"unknown predictor kind {kind!r}")


def build_agent_policy(
    name: str,
    train_rows: list[ScoringRow] | None,
    args,
    predictor_cache: dict[str, Predictor],
) -> AgentPolicy:
    if name in POLICIES:
        return RolloutPolicy(name)

    if name.endswith("-greedy"):
        kind = name[: -len("-greedy")]
        if train_rows is None:
            raise ValueError(f"{name} requires --train-data")
        if kind not in predictor_cache:
            predictor = _build_predictor(kind)
            log.info("fitting predictor kind %s on %d rows", kind, len(train_rows))
            predictor.fit(train_rows)
            predictor_cache[kind] = predictor
        return PredictorPolicy(
            name=name,
            predictor=predictor_cache[kind],
            mode="greedy",
            num_candidates=args.num_candidates,
            epsilon=args.epsilon,
        )

    if "-rerank-" in name:
        kind, proposal = name.split("-rerank-", 1)
        if train_rows is None:
            raise ValueError(f"{name} requires --train-data")
        if kind not in predictor_cache:
            predictor = _build_predictor(kind)
            log.info("fitting predictor kind %s on %d rows", kind, len(train_rows))
            predictor.fit(train_rows)
            predictor_cache[kind] = predictor
        return PredictorPolicy(
            name=name,
            predictor=predictor_cache[kind],
            mode="rerank",
            proposal=proposal,
            num_candidates=args.num_candidates,
            epsilon=args.epsilon,
        )

    raise ValueError(f"unknown agent policy {name!r}")


def run(args: argparse.Namespace) -> dict:
    train_rows = load_dir(args.train_data) if args.train_data else None
    if train_rows is not None:
        log.info("loaded %d training rows from %s", len(train_rows), args.train_data)

    results: dict[str, dict] = {}
    predictor_cache: dict[str, Predictor] = {}
    for name in args.policies:
        log.info("rolling policy %s", name)
        policy = build_agent_policy(name, train_rows, args, predictor_cache)
        results[name] = rollout_policy(
            policy,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed_start=args.seed_start,
        )
        r = results[name]
        print(
            f"{name:<28s} success={r['success_rate']:.3f} "
            f"avg_reward={r['avg_reward_per_episode']:.3f} "
            f"avg_ach={r['avg_achievements_per_episode']:.3f} "
            f"unique_ach={r['unique_achievements']}"
        )

    out = {
        "train_data": str(args.train_data) if args.train_data else None,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed_start": args.seed_start,
        "policies": list(args.policies),
        "num_candidates": args.num_candidates,
        "epsilon": args.epsilon,
        "results": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, default=_json_default))
        log.info("wrote %s", args.out)
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run live Crafter agent-loop experiments.")
    p.add_argument("--train-data", type=Path, default=None, help="rollout dir for learned/rule fit")
    p.add_argument(
        "--policies",
        nargs="+",
        default=["biased_random", "scripted_craft", "precondition-greedy", "trained-greedy"],
        help=(
            "agent policies: rollout policies {random,biased_random,scripted_craft,mixed}; "
            "or predictor policies like trained-greedy, precondition-greedy, "
            "trained-rerank-biased, trained-rerank-mixed, precondition-rerank-scripted"
        ),
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed-start", type=int, default=10_000)
    p.add_argument("--num-candidates", type=int, default=8, help="K for *-rerank-biased")
    p.add_argument("--epsilon", type=float, default=0.0, help="epsilon random actions for predictor")
    p.add_argument("--out", type=Path, default=None, help="write JSON results here")
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
