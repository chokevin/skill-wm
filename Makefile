.PHONY: help install sync test lint fix format check smoke minihack-smoke skill-jepa-smoke minihack-jepa-data skill-jepa-eval skill-jepa-task-eval minihack-jepa-coverage-data skill-jepa-coverage-eval minihack-jepa-hazard-data skill-jepa-hazard-eval collect-small collect-full eval-local \
        agent-pilot agent-goal-pilot agent-achievement-pilot clean all \
        rune-setup rune-collect-local rune-collect rune-collect-dry rune-train-dry rune-eval-dry

help:
	@echo "Local targets:"
	@echo "  install       sync deps via uv (creates .venv if missing)"
	@echo "  test          run pytest"
	@echo "  lint          ruff check (no fixes)"
	@echo "  fix           ruff check --fix + ruff format"
	@echo "  format        ruff format"
	@echo "  check         lint + test (CI-equivalent)"
	@echo "  smoke         5-episode rollout to verify end-to-end"
	@echo "  minihack-smoke  optional MiniHack rollout smoke test"
	@echo "  skill-jepa-smoke  train tiny MiniHack Skill-JEPA on smoke rollouts"
	@echo "  skill-jepa-eval  held-out-seed MiniHack Skill-JEPA eval"
	@echo "  skill-jepa-task-eval  held-out-task MiniHack Skill-JEPA eval"
	@echo "  skill-jepa-coverage-eval  noisy-room to lava coverage diagnostic"
	@echo "  skill-jepa-hazard-eval  noisy-room to deliberate lava-probe diagnostic"
	@echo "  collect-small 50-episode rollout (~30s)"
	@echo "  collect-full  500-episode rollout (~5min)"
	@echo "  eval-local    run baselines on data/rollouts/smoke (no LLM)"
	@echo "  eval-llm      run baselines + llm-zero (needs OPENAI_API_KEY)"
	@echo "  agent-pilot   run a small T2 live-agent pilot (trains local WM)"
	@echo "  agent-goal-pilot  run T2 mixed-proposer rerank pilot"
	@echo "  agent-achievement-pilot  run T2 achievement-target rerank pilot"
	@echo "  clean         remove caches and rollouts"
	@echo "  all           install + check + smoke"
	@echo ""
	@echo "Rune (voice-agent-flex AKS cluster) targets:"
	@echo "  rune-setup           az login + kubeconfig + venv (run once)"
	@echo "  rune-collect-local   run collect_rollouts function in this process (no submit)"
	@echo "  rune-collect-dry     render rollout-collection manifest, do not apply"
	@echo "  rune-collect         submit rollout-collection job to cluster"
	@echo "  rune-train-dry       render trained-WM training manifest"
	@echo "  rune-eval-dry        render baseline eval manifest"

install sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

fix:
	uv run ruff check --fix .
	uv run ruff format .

format:
	uv run ruff format .

check: lint test

smoke:
	uv run python -m skill_wm.data.collect --episodes 5 --max-steps 100 --policy biased_random --out data/rollouts/smoke

minihack-smoke:
	uv run --extra minihack python -m skill_wm.data.collect_minihack --env-id skillwm-room-goal --episodes 2 --max-steps 30 --policy scripted_nav --out data/rollouts/minihack-smoke/room-goal
	uv run --extra minihack python -m skill_wm.data.collect_minihack --env-id skillwm-lava-detour --episodes 2 --max-steps 50 --policy scripted_nav --out data/rollouts/minihack-smoke/lava-detour

skill-jepa-smoke: minihack-smoke
	uv run --extra train python -m skill_wm.models.skill_jepa --data data/rollouts/minihack-smoke --epochs 10 --batch-size 8

minihack-jepa-data:
	rm -rf data/rollouts/minihack-jepa-controlled
	uv run --extra minihack python -m skill_wm.data.collect_minihack --env-id skillwm-room-goal --episodes 8 --max-steps 30 --policy scripted_nav --seed-start 1000 --out data/rollouts/minihack-jepa-controlled/room-goal
	uv run --extra minihack python -m skill_wm.data.collect_minihack --env-id skillwm-lava-detour --episodes 8 --max-steps 50 --policy scripted_nav --seed-start 1000 --out data/rollouts/minihack-jepa-controlled/lava-detour

skill-jepa-eval: minihack-jepa-data
	uv run --extra train python -m skill_wm.models.skill_jepa --data data/rollouts/minihack-jepa-controlled --epochs 20 --batch-size 16 --train-frac 0.5 --seed 7 --out data/eval/minihack-skill-jepa-controlled.json

skill-jepa-task-eval: minihack-jepa-data
	uv run --extra train python -m skill_wm.models.skill_jepa --data data/rollouts/minihack-jepa-controlled --epochs 20 --batch-size 16 --split task --eval-env-id skillwm-lava-detour --seed 7 --out data/eval/minihack-skill-jepa-room-to-lava.json
	uv run --extra train python -m skill_wm.models.skill_jepa --data data/rollouts/minihack-jepa-controlled --epochs 20 --batch-size 16 --split task --eval-env-id skillwm-room-goal --seed 7 --out data/eval/minihack-skill-jepa-lava-to-room.json

minihack-jepa-coverage-data:
	rm -rf data/rollouts/minihack-jepa-coverage
	uv run --extra minihack python -m skill_wm.data.collect_minihack --env-id skillwm-room-goal --episodes 32 --max-steps 40 --policy scripted_nav_noisy --seed-start 3000 --out data/rollouts/minihack-jepa-coverage/room-goal-noisy
	uv run --extra minihack python -m skill_wm.data.collect_minihack --env-id skillwm-lava-detour --episodes 8 --max-steps 50 --policy scripted_nav --seed-start 1000 --out data/rollouts/minihack-jepa-coverage/lava-detour

skill-jepa-coverage-eval: minihack-jepa-coverage-data
	uv run --extra train python -m skill_wm.models.skill_jepa --data data/rollouts/minihack-jepa-coverage --epochs 20 --batch-size 16 --split task --eval-env-id skillwm-lava-detour --seed 7 --out data/eval/minihack-skill-jepa-room-noisy-to-lava.json

minihack-jepa-hazard-data:
	rm -rf data/rollouts/minihack-jepa-hazard
	uv run --extra minihack python -m skill_wm.data.collect_minihack --env-id skillwm-room-goal --episodes 32 --max-steps 40 --policy scripted_nav_noisy --seed-start 3000 --out data/rollouts/minihack-jepa-hazard/room-goal-noisy
	uv run --extra minihack python -m skill_wm.data.collect_minihack --env-id skillwm-lava-detour --episodes 8 --max-steps 50 --policy lava_probe --seed-start 1000 --out data/rollouts/minihack-jepa-hazard/lava-probe

skill-jepa-hazard-eval: minihack-jepa-hazard-data
	uv run --extra train python -m skill_wm.models.skill_jepa --data data/rollouts/minihack-jepa-hazard --epochs 20 --batch-size 16 --split task --eval-env-id skillwm-lava-detour --seed 7 --out data/eval/minihack-skill-jepa-room-noisy-to-lava-probe.json

collect-small:
	uv run python -m skill_wm.data.collect --episodes 50 --max-steps 200 --policy biased_random --out data/rollouts/small

collect-full:
	uv run python -m skill_wm.data.collect --episodes 500 --max-steps 300 --policy biased_random --out data/rollouts/full

eval-local:
	uv run python -m skill_wm.eval.run --data data/rollouts/smoke --baselines random marginal precondition --train-frac 0.6 --out eval-local.json

eval-llm:
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "set OPENAI_API_KEY before eval-llm"; exit 1; fi
	uv run python -m skill_wm.eval.run --data data/rollouts/smoke --baselines random marginal precondition llm-zero --train-frac 0.6 --out eval-llm.json

agent-pilot:
	uv run python -m skill_wm.agent.run \
		--train-data data/rollouts/collect-002-combined \
		--policies biased_random scripted_craft precondition-greedy trained-greedy trained-rerank-biased \
		--episodes 10 --max-steps 200 --seed-start 20000 \
		--out eval-results-agent/t2-pilot-10ep.json

agent-goal-pilot:
	uv run python -m skill_wm.agent.run \
		--train-data data/rollouts/collect-002-combined \
		--policies mixed scripted_craft precondition-rerank-mixed trained-rerank-mixed \
		--episodes 10 --max-steps 200 --seed-start 21000 \
		--out eval-results-agent/t2-goal-rerank-10ep.json

agent-achievement-pilot:
	uv run python -m skill_wm.agent.run \
		--train-data data/rollouts/collect-002-combined \
		--policies mixed precondition-rerank-mixed trained-achievement-rerank-mixed \
		--episodes 30 --max-steps 200 --seed-start 23000 \
		--out eval-results-agent/t2-achievement-rerank-30ep.json

clean:
	rm -rf .pytest_cache .ruff_cache data/rollouts
	find . -type d -name __pycache__ -exec rm -rf {} +

all: install check smoke

# ---- Rune (voice-agent-flex) -----------------------------------------------
# These targets only render or submit. They never auto-submit a job; a
# top-level `RUNE_NAME=...` is required for cluster submits.

rune-setup:
	bash bin/setup.sh

rune-collect-local:
	uv run python experiments/collect_rollouts/config.py --local

rune-collect-dry:
	uv run python experiments/collect_rollouts/config.py --dry-run

rune-collect:
	@if [ -z "$$RUNE_NAME" ]; then echo "set RUNE_NAME=skill-wm-collect-NNN before submitting"; exit 1; fi
	uv run python experiments/collect_rollouts/config.py

rune-train-dry:
	uv run python experiments/train_wm/config.py --dry-run

rune-eval-dry:
	uv run python experiments/eval_baselines/config.py --dry-run
