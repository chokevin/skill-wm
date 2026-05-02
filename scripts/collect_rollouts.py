"""Top-level script: run a small rollout and print summary stats.

Usage: uv run python -m scripts.collect_rollouts --episodes 50 --max-steps 200
"""

from skill_wm.data.collect import main

if __name__ == "__main__":
    main()
