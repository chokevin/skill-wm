.PHONY: help install sync test lint fix format check smoke collect-small collect-full clean all

help:
	@echo "Targets:"
	@echo "  install       sync deps via uv (creates .venv if missing)"
	@echo "  test          run pytest"
	@echo "  lint          ruff check (no fixes)"
	@echo "  fix           ruff check --fix + ruff format"
	@echo "  format        ruff format"
	@echo "  check         lint + test (CI-equivalent)"
	@echo "  smoke         5-episode rollout to verify end-to-end"
	@echo "  collect-small 50-episode rollout (~30s)"
	@echo "  collect-full  500-episode rollout (~5min)"
	@echo "  clean         remove caches and rollouts"
	@echo "  all           install + check + smoke"

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
	uv run python -m scripts.collect_rollouts --episodes 5 --max-steps 100 --policy biased_random --out data/rollouts/smoke

collect-small:
	uv run python -m scripts.collect_rollouts --episodes 50 --max-steps 200 --policy biased_random --out data/rollouts/small

collect-full:
	uv run python -m scripts.collect_rollouts --episodes 500 --max-steps 300 --policy biased_random --out data/rollouts/full

clean:
	rm -rf .pytest_cache .ruff_cache data/rollouts
	find . -type d -name __pycache__ -exec rm -rf {} +

all: install check smoke
