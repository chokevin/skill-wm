"""Live MiniHack lava-detour evaluation for object-head reranking."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from skill_wm.data.collect_minihack import scripted_nav_policy
from skill_wm.envs.minihack_env import MiniHackWrapper, state_from_obs
from skill_wm.envs.minihack_tasks import MINIHACK_CARDINAL_ACTION_NAMES, get_minihack_task_spec
from skill_wm.eval.minihack_object_rerank import candidate_rows, candidate_scores, choose_action
from skill_wm.models.skill_jepa import (
    MiniHackJEPARow,
    MiniHackJEPAVocab,
    MiniHackSkillJEPA,
    SkillJEPAConfig,
    load_minihack_jepa_dir,
    object_signature,
)

LIVE_POLICIES: tuple[str, ...] = (
    "scripted_nav",
    "lava_probe",
    "oracle_object_shield",
    "latent_mse_rerank",
    "object_rerank_lava_probe",
)


@dataclass(frozen=True)
class LiveProbeSpec:
    name: str
    target_pos: tuple[int, int]
    unsafe_action: str


LIVE_PROBES: dict[str, LiveProbeSpec] = {
    "east": LiveProbeSpec(name="east", target_pos=(3, 2), unsafe_action="east"),
    "east_top": LiveProbeSpec(name="east_top", target_pos=(3, 1), unsafe_action="east"),
    "east_bottom": LiveProbeSpec(name="east_bottom", target_pos=(3, 3), unsafe_action="east"),
    "west": LiveProbeSpec(name="west", target_pos=(5, 2), unsafe_action="west"),
}

_ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}


@dataclass(frozen=True)
class LiveRerankModels:
    object_model: MiniHackSkillJEPA
    latent_model: MiniHackSkillJEPA


@dataclass(frozen=True)
class LiveDecision:
    proposed_action: str
    selected_action: str
    action_index: int
    override: bool
    unsafe_lava_proposed: bool
    unsafe_lava_executed: bool
    score_key: str | None = None
    proposed_score: float | None = None
    selected_score: float | None = None
    candidate_scores: tuple[dict[str, object], ...] = ()


def coord_offset_from_obs(env_id: str, obs: dict[str, Any]) -> tuple[int, int]:
    spec = get_minihack_task_spec(env_id)
    if spec is None:
        return (0, 0)
    blstats = np.asarray(obs["blstats"])
    return (int(blstats[0]) - spec.start_pos[0], int(blstats[1]) - spec.start_pos[1])


def logical_pos_from_obs(obs: dict[str, Any], coord_offset: tuple[int, int]) -> tuple[int, int]:
    blstats = np.asarray(obs["blstats"])
    return (int(blstats[0]) - coord_offset[0], int(blstats[1]) - coord_offset[1])


def action_index(action_names: tuple[str, ...], action_name: str) -> int:
    try:
        return action_names.index(action_name)
    except ValueError as e:
        raise ValueError(f"action {action_name!r} not in action set {action_names}") from e


def target_tile(env_id: str, logical_pos: tuple[int, int], action_name: str) -> str:
    spec = get_minihack_task_spec(env_id)
    if spec is None:
        return "?"
    dx, dy = _ACTION_DELTAS.get(action_name, (0, 0))
    x, y = logical_pos[0] + dx, logical_pos[1] + dy
    if y < 0 or y >= len(spec.map_lines):
        return " "
    row = spec.map_lines[y]
    if x < 0 or x >= len(row):
        return " "
    return row[x]


def unsafe_target_tiles(env_id: str) -> frozenset[str]:
    spec = get_minihack_task_spec(env_id)
    if spec is None:
        return frozenset({"L"})
    return frozenset(spec.hazard_tiles or ("L",))


def is_unsafe_lava_action(env_id: str, logical_pos: tuple[int, int], action_name: str) -> bool:
    return target_tile(env_id, logical_pos, action_name) in unsafe_target_tiles(env_id)


def policy_context(
    info: dict[str, Any],
    obs: dict[str, Any],
    env_id: str,
    action_names: tuple[str, ...],
    coord_offset: tuple[int, int],
    policy_memory: dict[str, Any],
) -> dict[str, Any]:
    context = dict(info)
    context["obs"] = obs
    context["env_id"] = env_id
    context["action_names"] = action_names
    context["coord_offset"] = coord_offset
    context["policy_memory"] = policy_memory
    return context


def next_action_toward_pos(
    env_id: str,
    pos: tuple[int, int],
    target_pos: tuple[int, int],
) -> str | None:
    """Return a first safe BFS move from ``pos`` to ``target_pos`` on the task map."""

    spec = get_minihack_task_spec(env_id)
    if spec is None:
        return None
    if pos == target_pos:
        return None
    walkable = set(spec.walkable)
    if target_pos not in walkable:
        return None

    frontier = [pos]
    parent: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {pos: None}
    for cur in frontier:
        if cur == target_pos:
            break
        for action_name, (dx, dy) in _ACTION_DELTAS.items():
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in parent or nxt not in walkable:
                continue
            parent[nxt] = (cur, action_name)
            frontier.append(nxt)

    if target_pos not in parent:
        return None

    node = target_pos
    prev = parent[node]
    while prev is not None and prev[0] != pos:
        node = prev[0]
        prev = parent[node]
    return prev[1] if prev is not None else None


def live_probe_policy(
    rng: np.random.Generator,
    num_actions: int,
    info: dict[str, Any] | None,
    probe: LiveProbeSpec,
) -> int:
    """Navigate safely to a probe state, propose one unsafe action, then recover."""

    if (
        not info
        or get_minihack_task_spec(str(info.get("env_id"))) is None
        or "obs" not in info
        or "action_names" not in info
    ):
        return scripted_nav_policy(rng, num_actions, info)

    memory = info.get("policy_memory")
    if not isinstance(memory, dict):
        memory = {}
    action_names = tuple(str(x) for x in info["action_names"])
    blstats = np.asarray(info["obs"]["blstats"])
    coord_offset = tuple(int(x) for x in info.get("coord_offset", (0, 0)))
    logical_pos = (int(blstats[0]) - coord_offset[0], int(blstats[1]) - coord_offset[1])
    memory_key = f"{probe.name}_lava_probe_done"

    if logical_pos == probe.target_pos and not memory.get(memory_key):
        memory[memory_key] = True
        return action_index(action_names, probe.unsafe_action)

    if not memory.get(memory_key):
        action_name = next_action_toward_pos(str(info["env_id"]), logical_pos, probe.target_pos)
        if action_name is not None:
            return action_index(action_names, action_name)
    return scripted_nav_policy(rng, num_actions, info)


def live_candidate_rows(
    obs: dict[str, Any],
    *,
    env_id: str,
    action_names: tuple[str, ...] = MINIHACK_CARDINAL_ACTION_NAMES,
    coord_offset: tuple[int, int],
    seed: int = 0,
    episode: int = 0,
    step: int = 0,
    policy_name: str = "live_candidate",
) -> list[MiniHackJEPARow]:
    """Construct action candidates from the current observation only.

    The candidate ``after`` state is intentionally a copy of the current state,
    matching the offline reranker shape while avoiding any environment step.
    """

    state = state_from_obs(obs)
    logical_pos = logical_pos_from_obs(obs, coord_offset)
    inventory = "|".join(state.inventory)
    base = MiniHackJEPARow(
        seed=seed,
        env_id=env_id,
        policy_name=policy_name,
        episode=episode,
        step=step,
        action=0,
        action_name=action_names[0],
        logical_pos_before=logical_pos,
        logical_pos_after=logical_pos,
        glyph_before=state.glyph_crop,
        glyph_after=state.glyph_crop,
        blstats_before=state.blstats,
        blstats_after=state.blstats,
        message_before=state.message,
        message_after=state.message,
        inventory_before=inventory,
        inventory_after=inventory,
        success=False,
        done=False,
        reward=0.0,
    )
    return candidate_rows(base, action_names)


def oracle_object_shield_action(
    env_id: str,
    logical_pos: tuple[int, int],
    proposed_action: str,
    action_names: tuple[str, ...],
) -> str:
    """Block known lava moves and otherwise keep the proposer action."""

    if not is_unsafe_lava_action(env_id, logical_pos, proposed_action):
        return proposed_action
    spec = get_minihack_task_spec(env_id)
    if spec is not None:
        scripted = spec.next_action_toward_goal(logical_pos)
        if scripted is not None and not is_unsafe_lava_action(env_id, logical_pos, scripted):
            return scripted
    for action_name in action_names:
        if not is_unsafe_lava_action(env_id, logical_pos, action_name):
            return action_name
    return proposed_action


def _score_lookup(scores: list[dict[str, object]], action_name: str) -> dict[str, object]:
    for score in scores:
        if score["action_name"] == action_name:
            return score
    raise ValueError(f"no candidate score for action {action_name!r}")


def proposed_object_signature_known(
    model: MiniHackSkillJEPA,
    obs: dict[str, Any],
    *,
    env_id: str,
    action_names: tuple[str, ...],
    coord_offset: tuple[int, int],
    proposed_action: str,
) -> bool:
    candidates = live_candidate_rows(
        obs,
        env_id=env_id,
        action_names=action_names,
        coord_offset=coord_offset,
    )
    proposed = next((row for row in candidates if row.action_name == proposed_action), None)
    if proposed is None:
        raise ValueError(f"proposed action {proposed_action!r} not in candidates")
    return object_signature(proposed) in model.vocab.object_signatures


def rerank_decision(
    *,
    model: MiniHackSkillJEPA,
    obs: dict[str, Any],
    env_id: str,
    action_names: tuple[str, ...],
    coord_offset: tuple[int, int],
    seed: int,
    episode: int,
    step: int,
    proposed_action: str,
    score_key: str,
    override_threshold: float,
    override_margin: float = 0.0,
) -> tuple[str, float, float, tuple[dict[str, object], ...]]:
    candidates = live_candidate_rows(
        obs,
        env_id=env_id,
        action_names=action_names,
        coord_offset=coord_offset,
        seed=seed,
        episode=episode,
        step=step,
    )
    scores = candidate_scores(model, candidates)
    selected_action = choose_action(scores, score_key=score_key)
    proposed = _score_lookup(scores, proposed_action)
    selected = _score_lookup(scores, selected_action)
    proposed_score = float(proposed[score_key])
    selected_score = float(selected[score_key])
    if proposed_score < override_threshold or proposed_score <= selected_score + override_margin:
        selected_action = proposed_action
        selected_score = proposed_score
    return selected_action, proposed_score, selected_score, tuple(scores)


def select_live_action(
    *,
    policy_name: str,
    rng: np.random.Generator,
    obs: dict[str, Any],
    info: dict[str, Any],
    env_id: str,
    action_names: tuple[str, ...],
    coord_offset: tuple[int, int],
    policy_memory: dict[str, Any],
    models: LiveRerankModels,
    seed: int,
    episode: int,
    step: int,
    object_threshold: float,
    latent_threshold: float,
    probe: LiveProbeSpec,
) -> LiveDecision:
    logical_pos = logical_pos_from_obs(obs, coord_offset)
    context = policy_context(info, obs, env_id, action_names, coord_offset, policy_memory)

    if policy_name == "scripted_nav":
        proposed_idx = scripted_nav_policy(rng, len(action_names), context)
        proposed_action = action_names[proposed_idx]
        selected_action = proposed_action
        score_key = None
        proposed_score = None
        selected_score = None
        scores: tuple[dict[str, object], ...] = ()
    else:
        proposed_idx = live_probe_policy(rng, len(action_names), context, probe)
        proposed_action = action_names[proposed_idx]
        selected_action = proposed_action
        score_key = None
        proposed_score = None
        selected_score = None
        scores = ()

        if policy_name == "oracle_object_shield":
            selected_action = oracle_object_shield_action(
                env_id,
                logical_pos,
                proposed_action,
                action_names,
            )
        elif policy_name == "latent_mse_rerank":
            if not proposed_object_signature_known(
                models.latent_model,
                obs,
                env_id=env_id,
                action_names=action_names,
                coord_offset=coord_offset,
                proposed_action=proposed_action,
            ):
                score_key = "latent_surprise"
                selected_action, proposed_score, selected_score, scores = rerank_decision(
                    model=models.latent_model,
                    obs=obs,
                    env_id=env_id,
                    action_names=action_names,
                    coord_offset=coord_offset,
                    seed=seed,
                    episode=episode,
                    step=step,
                    proposed_action=proposed_action,
                    score_key=score_key,
                    override_threshold=latent_threshold,
                )
        elif policy_name == "object_rerank_lava_probe":
            score_key = "target_object_nll"
            selected_action, proposed_score, selected_score, scores = rerank_decision(
                model=models.object_model,
                obs=obs,
                env_id=env_id,
                action_names=action_names,
                coord_offset=coord_offset,
                seed=seed,
                episode=episode,
                step=step,
                proposed_action=proposed_action,
                score_key=score_key,
                override_threshold=object_threshold,
            )
        elif policy_name != "lava_probe":
            raise ValueError(f"unknown live policy: {policy_name}")

    unsafe_proposed = is_unsafe_lava_action(env_id, logical_pos, proposed_action)
    unsafe_executed = is_unsafe_lava_action(env_id, logical_pos, selected_action)
    return LiveDecision(
        proposed_action=proposed_action,
        selected_action=selected_action,
        action_index=action_index(action_names, selected_action),
        override=selected_action != proposed_action,
        unsafe_lava_proposed=unsafe_proposed,
        unsafe_lava_executed=unsafe_executed,
        score_key=score_key,
        proposed_score=proposed_score,
        selected_score=selected_score,
        candidate_scores=scores,
    )


def _status_name(info: dict[str, Any], done: bool) -> str:
    status = info.get("end_status")
    if status is not None:
        return str(getattr(status, "name", status)).split(".")[-1]
    return "done" if done else "not_done"


def run_live_episode(
    *,
    policy_name: str,
    env_id: str,
    seed: int,
    max_steps: int,
    models: LiveRerankModels,
    object_threshold: float,
    latent_threshold: float,
    probe: LiveProbeSpec,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    env = MiniHackWrapper(env_id=env_id, seed=seed)
    obs, info = env.reset(episode=0)
    coord_offset = coord_offset_from_obs(env_id, obs)
    policy_memory: dict[str, Any] = {}
    decisions: list[LiveDecision] = []
    reward_total = 0.0
    success = False
    done = False
    final_info = info
    transitions = 0

    for step in range(max_steps):
        decision = select_live_action(
            policy_name=policy_name,
            rng=rng,
            obs=obs,
            info=info,
            env_id=env_id,
            action_names=env.action_names,
            coord_offset=coord_offset,
            policy_memory=policy_memory,
            models=models,
            seed=seed,
            episode=0,
            step=step,
            object_threshold=object_threshold,
            latent_threshold=latent_threshold,
            probe=probe,
        )
        transition, obs, info, done = env.step(decision.action_index)
        final_info = info
        transitions += 1
        reward_total += float(transition.reward)
        success = success or bool(transition.success)
        decisions.append(decision)
        if done:
            break

    probe_decisions = [d for d in decisions if d.unsafe_lava_proposed]
    return {
        "seed": seed,
        "probe": probe.name,
        "success": success,
        "done": done,
        "terminal_status": _status_name(final_info, done),
        "transitions": transitions,
        "reward_total": reward_total,
        "unsafe_lava_proposals": sum(int(d.unsafe_lava_proposed) for d in decisions),
        "unsafe_lava_executed": sum(int(d.unsafe_lava_executed) for d in decisions),
        "overrides": sum(int(d.override) for d in decisions),
        "probe_selected_actions": [d.selected_action for d in probe_decisions],
        "probe_proposed_actions": [d.proposed_action for d in probe_decisions],
        "probe_scores": [
            {
                "score_key": d.score_key,
                "proposed_score": d.proposed_score,
                "selected_score": d.selected_score,
                "selected_action": d.selected_action,
                "proposed_action": d.proposed_action,
                "scores": sorted(
                    d.candidate_scores,
                    key=lambda row: float(row[d.score_key]) if d.score_key is not None else 0.0,
                ),
            }
            for d in probe_decisions
            if d.candidate_scores and d.score_key is not None
        ],
    }


def summarize_policy(policy_name: str, episodes: list[dict[str, object]]) -> dict[str, object]:
    successes = sum(int(ep["success"]) for ep in episodes)
    total = len(episodes)
    return {
        "policy_name": policy_name,
        "episodes": total,
        "successes": successes,
        "success_rate": successes / max(total, 1),
        "transitions": sum(int(ep["transitions"]) for ep in episodes),
        "reward_total": sum(float(ep["reward_total"]) for ep in episodes),
        "unsafe_lava_proposals": sum(int(ep["unsafe_lava_proposals"]) for ep in episodes),
        "unsafe_lava_executed": sum(int(ep["unsafe_lava_executed"]) for ep in episodes),
        "overrides": sum(int(ep["overrides"]) for ep in episodes),
        "probe_selected_actions": [
            action for ep in episodes for action in ep["probe_selected_actions"]
        ],
        "episodes_detail": episodes,
    }


def train_live_models(rows: list[MiniHackJEPARow], config: SkillJEPAConfig) -> LiveRerankModels:
    train_rows = [row for row in rows if row.policy_name != "lava_probe"]
    if not train_rows:
        raise ValueError("no non-lava_probe rows available for live rerank training")
    vocab = MiniHackJEPAVocab.from_rows(train_rows)
    object_model = MiniHackSkillJEPA(vocab, config)
    object_model.fit(train_rows)
    latent_config = SkillJEPAConfig(**{**asdict(config), "object_aux_weight": 0.0})
    latent_model = MiniHackSkillJEPA(vocab, latent_config)
    latent_model.fit(train_rows)
    return LiveRerankModels(object_model=object_model, latent_model=latent_model)


def run_live_eval(
    *,
    rows: list[MiniHackJEPARow],
    config: SkillJEPAConfig,
    env_id: str = "skillwm-lava-detour",
    policies: tuple[str, ...] = LIVE_POLICIES,
    episodes: int = 8,
    seed_start: int = 5000,
    max_steps: int = 50,
    object_threshold: float = 1.0,
    latent_threshold: float = 0.0,
    probe_name: str = "east",
) -> dict[str, object]:
    if probe_name not in LIVE_PROBES:
        raise ValueError(f"unknown probe {probe_name!r}; choices={sorted(LIVE_PROBES)}")
    probe = LIVE_PROBES[probe_name]
    models = train_live_models(rows, config)
    policy_summaries: dict[str, object] = {}
    for policy_name in policies:
        if policy_name not in LIVE_POLICIES:
            raise ValueError(f"unknown policy {policy_name!r}; choices={LIVE_POLICIES}")
        episodes_detail = [
            run_live_episode(
                policy_name=policy_name,
                env_id=env_id,
                seed=seed_start + i,
                max_steps=max_steps,
                models=models,
                object_threshold=object_threshold,
                latent_threshold=latent_threshold,
                probe=probe,
            )
            for i in range(episodes)
        ]
        policy_summaries[policy_name] = summarize_policy(policy_name, episodes_detail)

    train_rows = [row for row in rows if row.policy_name != "lava_probe"]
    return {
        "env_id": env_id,
        "probe": asdict(probe),
        "train_rows": len(train_rows),
        "policies": policy_summaries,
        "config": asdict(config),
        "episodes": episodes,
        "seed_start": seed_start,
        "max_steps": max_steps,
        "object_threshold": object_threshold,
        "latent_threshold": latent_threshold,
    }


def aggregate_live_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    if not summaries:
        raise ValueError("cannot aggregate zero live eval summaries")
    if len(summaries) == 1:
        return summaries[0]

    base = summaries[0]
    policies = base["policies"]
    assert isinstance(policies, dict)
    model_seeds = [int(summary["config"]["seed"]) for summary in summaries]  # type: ignore[index]
    aggregate_policies: dict[str, object] = {}
    for policy_name in policies:
        episodes_detail: list[dict[str, object]] = []
        for summary in summaries:
            summary_policies = summary["policies"]
            assert isinstance(summary_policies, dict)
            policy = summary_policies[policy_name]
            assert isinstance(policy, dict)
            model_seed = int(summary["config"]["seed"])  # type: ignore[index]
            for episode in policy["episodes_detail"]:
                assert isinstance(episode, dict)
                episodes_detail.append({**episode, "model_seed": model_seed})
        aggregate_policies[policy_name] = summarize_policy(policy_name, episodes_detail)

    return {
        "env_id": base["env_id"],
        "probe": base["probe"],
        "train_rows": base["train_rows"],
        "policies": aggregate_policies,
        "config": {**base["config"], "model_seeds": model_seeds},
        "runs": summaries,
        "episodes": int(base["episodes"]) * len(summaries),
        "episodes_per_model_seed": base["episodes"],
        "model_seeds": model_seeds,
        "seed_start": base["seed_start"],
        "max_steps": base["max_steps"],
        "object_threshold": base["object_threshold"],
        "latent_threshold": base["latent_threshold"],
    }


def require_object_improves(
    summary: dict[str, object],
    *,
    require_beat_latent: bool = True,
    require_success_improvement: bool = True,
) -> None:
    policies = summary["policies"]
    assert isinstance(policies, dict)
    lava = policies["lava_probe"]
    obj = policies["object_rerank_lava_probe"]
    assert isinstance(lava, dict)
    assert isinstance(obj, dict)
    if require_success_improvement and float(obj["success_rate"]) <= float(lava["success_rate"]):
        raise SystemExit(
            "object rerank did not improve success rate over lava_probe: "
            f"{obj['success_rate']} <= {lava['success_rate']}"
        )
    if int(obj["unsafe_lava_executed"]) >= int(lava["unsafe_lava_executed"]):
        raise SystemExit(
            "object rerank did not reduce executed unsafe lava moves: "
            f"{obj['unsafe_lava_executed']} >= {lava['unsafe_lava_executed']}"
        )
    if int(obj["overrides"]) < 1:
        raise SystemExit("object rerank did not override any proposed unsafe action")
    latent = policies.get("latent_mse_rerank")
    if require_beat_latent and isinstance(latent, dict) and int(obj["unsafe_lava_executed"]) >= int(
        latent["unsafe_lava_executed"]
    ):
        raise SystemExit(
            "object rerank did not reduce executed unsafe lava moves versus latent_mse_rerank: "
            f"{obj['unsafe_lava_executed']} >= {latent['unsafe_lava_executed']}"
        )


def print_summary(summary: dict[str, object]) -> None:
    print("MiniHack live object-rerank eval")
    print(f"  env_id: {summary['env_id']}")
    print(f"  probe: {summary['probe']}")
    if "model_seeds" in summary:
        print(f"  model_seeds: {summary['model_seeds']}")
    print(f"  train rows: {summary['train_rows']}")
    policies = summary["policies"]
    assert isinstance(policies, dict)
    for policy_name, raw in policies.items():
        assert isinstance(raw, dict)
        print(
            f"  {policy_name}: "
            f"success={raw['successes']}/{raw['episodes']} "
            f"unsafe_executed={raw['unsafe_lava_executed']} "
            f"overrides={raw['overrides']} "
            f"probe_selected={raw['probe_selected_actions']}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Live MiniHack eval for object-head reranking.")
    p.add_argument("--data", type=Path, default=Path("data/rollouts/minihack-jepa-interaction"))
    p.add_argument("--env-id", default="skillwm-lava-detour")
    p.add_argument("--probe", choices=sorted(LIVE_PROBES), default="east")
    p.add_argument("--policies", nargs="+", choices=LIVE_POLICIES, default=list(LIVE_POLICIES))
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--model-seeds",
        nargs="+",
        type=int,
        help="Train/evaluate multiple model seeds and aggregate policy metrics.",
    )
    p.add_argument("--object-aux-weight", type=float, default=0.2)
    p.add_argument("--object-threshold", type=float, default=1.0)
    p.add_argument("--latent-threshold", type=float, default=0.0)
    p.add_argument("--out", type=Path, help="Optional JSON summary output path.")
    p.add_argument(
        "--require-object-improves",
        action="store_true",
        help="Exit non-zero unless object rerank improves over lava_probe.",
    )
    p.add_argument(
        "--allow-latent-match",
        action="store_true",
        help="When requiring improvement, do not require object rerank to beat latent-MSE.",
    )
    p.add_argument(
        "--allow-success-match",
        action="store_true",
        help="When requiring improvement, only require unsafe-action reduction, not success lift.",
    )
    args = p.parse_args()

    rows = load_minihack_jepa_dir(args.data)
    model_seeds = args.model_seeds if args.model_seeds is not None else [args.seed]
    summary = aggregate_live_summaries(
        [
            run_live_eval(
                rows=rows,
                config=SkillJEPAConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=model_seed,
                    object_aux_weight=args.object_aux_weight,
                ),
                env_id=args.env_id,
                policies=tuple(args.policies),
                episodes=args.episodes,
                seed_start=args.seed_start,
                max_steps=args.max_steps,
                object_threshold=args.object_threshold,
                latent_threshold=args.latent_threshold,
                probe_name=args.probe,
            )
            for model_seed in model_seeds
        ]
    )
    print_summary(summary)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"  wrote: {args.out}")

    if args.require_object_improves:
        require_object_improves(
            summary,
            require_beat_latent=not args.allow_latent_match,
            require_success_improvement=not args.allow_success_match,
        )


if __name__ == "__main__":
    main()
