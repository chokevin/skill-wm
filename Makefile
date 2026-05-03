.PHONY: help install sync test lint fix format check smoke collect-small collect-full eval-local clean all \
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
	@echo "  collect-small 50-episode rollout (~30s)"
	@echo "  collect-full  500-episode rollout (~5min)"
	@echo "  eval-local    run baselines on data/rollouts/smoke (no LLM)"
	@echo "  eval-llm      run baselines + llm-zero (needs OPENAI_API_KEY)"
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

collect-small:
	uv run python -m skill_wm.data.collect --episodes 50 --max-steps 200 --policy biased_random --out data/rollouts/small

collect-full:
	uv run python -m skill_wm.data.collect --episodes 500 --max-steps 300 --policy biased_random --out data/rollouts/full

eval-local:
	uv run python -m skill_wm.eval.run --data data/rollouts/smoke --baselines random marginal precondition --train-frac 0.6 --out eval-local.json

eval-llm:
	@if [ -z "$$OPENAI_API_KEY" ]; then echo "set OPENAI_API_KEY before eval-llm"; exit 1; fi
	uv run python -m skill_wm.eval.run --data data/rollouts/smoke --baselines random marginal precondition llm-zero --train-frac 0.6 --out eval-llm.json

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
