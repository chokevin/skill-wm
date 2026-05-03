"""LLM-as-WM zero-shot baseline.

We ask a frontier LLM (default: gpt-4o-mini) to predict per-action success
in Crafter, and read a graded probability from the token logprobs. The
prompt constrains the output to a single token "Yes" or "No"; we then
compute::

    p(success) = exp(lp_yes) / (exp(lp_yes) + exp(lp_no))

This matters because T1's headline metric is **calibration** (Brier /
ECE). Mapping the LLM's reply to a hard {0, 1} would bias the
comparison against the LLM by construction — it would never be allowed
to express uncertainty.

The OpenAI Chat Completions API returns top-K alternative tokens with
logprobs at each generated position when called with
``logprobs=True, top_logprobs=K``; we look at the first generated token
and search its top-K alternatives for "Yes"/"No" (case-insensitive).

Caching is keyed by ``ScoringRow.row_id`` (a stable hash of seed-episode-
step) so we don't double-pay for the same row across multiple eval runs.

Tests use ``client=FakeLLMClient(...)`` to inject deterministic
responses without hitting the real API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from skill_wm.eval.dataset import ScoringRow
from skill_wm.models.state_text import SYSTEM_PROMPT, render_user_prompt

log = logging.getLogger(__name__)

# Tokens we accept as the answer position. Mostly "Yes"/"No"; the LLM
# may also produce " Yes" with a leading space depending on tokenizer.
YES_TOKENS = {"Yes", "yes", " Yes", " yes", "YES"}
NO_TOKENS = {"No", "no", " No", " no", "NO"}


@dataclass
class LLMResponse:
    """One row's LLM verdict, including raw logprob debug info."""

    p_yes: float  # in [0, 1], renormalized over yes/no
    raw_top: list[tuple[str, float]]  # top tokens at the answer position


class LLMClient(Protocol):
    """Minimal surface area we depend on. Real impl wraps OpenAI."""

    def complete(self, system: str, user: str) -> LLMResponse: ...


class FakeLLMClient:
    """Deterministic, callable LLM for tests.

    `answer_fn(row_user_prompt) -> p_yes` lets the test author decide
    what probability to return per row. Default returns 0.5 (max
    entropy), which is the "I don't know" answer.
    """

    def __init__(self, answer_fn=None):
        self._answer_fn = answer_fn or (lambda _user: 0.5)
        self.calls = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls += 1
        p = float(self._answer_fn(user))
        p = min(max(p, 1e-6), 1 - 1e-6)
        # Build a plausible top_logprobs layout for whichever side wins.
        if p >= 0.5:
            top = [("Yes", math.log(p)), ("No", math.log(1 - p))]
        else:
            top = [("No", math.log(1 - p)), ("Yes", math.log(p))]
        return LLMResponse(p_yes=p, raw_top=top)


class OpenAIClient:
    """Real OpenAI Chat Completions client.

    Lazily imports the SDK so the rest of the codebase doesn't pull
    `openai` into import paths. Reads the API key from `OPENAI_API_KEY`.
    Retries on transient errors with exponential backoff.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 4,
        backoff_base: float = 2.0,
    ):
        from openai import OpenAI  # type: ignore[import-not-found]

        self._OpenAI = OpenAI
        self._client = OpenAI()
        self.model = model
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def complete(self, system: str, user: str) -> LLMResponse:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=1,
                    temperature=0,
                    logprobs=True,
                    top_logprobs=20,
                )
                choice = resp.choices[0]
                lp_data = choice.logprobs
                if lp_data is None or not lp_data.content:
                    return LLMResponse(p_yes=0.5, raw_top=[])
                token_lp = lp_data.content[0]
                tops = [(t.token, float(t.logprob)) for t in (token_lp.top_logprobs or [])]
                # Include the chosen token itself in case the top_logprobs
                # list omitted it (unusual but possible).
                tops.append((token_lp.token, float(token_lp.logprob)))
                return LLMResponse(p_yes=_renormalize(tops), raw_top=tops)
            except Exception as e:  # broad: includes RateLimit, APIConnection, etc.
                last_exc = e
                wait = self.backoff_base**attempt
                log.warning("openai retry %d after %.1fs: %s", attempt + 1, wait, e)
                time.sleep(wait)
        raise RuntimeError(f"openai client failed after {self.max_retries} retries: {last_exc}")


def _renormalize(top: list[tuple[str, float]]) -> float:
    """Compute p(Yes) by renormalizing yes/no token logprobs.

    If neither Yes nor No appears in the top tokens, return 0.5 (we
    have no evidence either way). If only one appears, treat the other
    as having ~zero probability.
    """
    yes_lps = [lp for tok, lp in top if tok in YES_TOKENS]
    no_lps = [lp for tok, lp in top if tok in NO_TOKENS]
    yes_p = sum(math.exp(lp) for lp in yes_lps)
    no_p = sum(math.exp(lp) for lp in no_lps)
    total = yes_p + no_p
    if total <= 0:
        return 0.5
    return yes_p / total


class LLMWorldModel:
    """Predictor wrapper around an LLMClient + on-disk cache."""

    name = "llm_zero_shot"

    def __init__(
        self,
        client: LLMClient | None = None,
        cache_path: Path | None = None,
    ):
        self.client = client if client is not None else OpenAIClient()
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, float] = self._load_cache()

    def _load_cache(self) -> dict[str, float]:
        if self.cache_path and self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                log.warning("could not read cache %s: %s", self.cache_path, e)
        return {}

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))

    @staticmethod
    def _prompt_key(user: str) -> str:
        """Cache key includes the prompt text so prompt edits invalidate."""
        return hashlib.md5(user.encode()).hexdigest()[:12]

    def fit(self, train_rows: list[ScoringRow]) -> None:
        """Zero-shot: no training. (Few-shot variant will populate examples here.)"""
        return None

    def predict(self, eval_rows: list[ScoringRow]) -> np.ndarray:
        out = np.empty(len(eval_rows), dtype=np.float64)
        for i, row in enumerate(eval_rows):
            user = render_user_prompt(row)
            key = f"{row.row_id}:{self._prompt_key(user)}"
            if key in self._cache:
                out[i] = self._cache[key]
                continue
            resp = self.client.complete(SYSTEM_PROMPT, user)
            self._cache[key] = resp.p_yes
            out[i] = resp.p_yes
        self._save_cache()
        return out


def make_default_client_or_skip() -> Any:
    """Helper for the CLI: build a real OpenAIClient if env is set, else error.

    Importing `openai` is deferred until this is called so dry-runs and
    pure-baseline runs don't require the package.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Either export it, or run the eval "
            "without `llm-zero` in --baselines to skip the LLM call."
        )
    return OpenAIClient()
