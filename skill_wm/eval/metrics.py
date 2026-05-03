"""Calibration metrics for skill-WM baseline comparison.

Headline metric: **Brier score** (lower is better; 0 is perfect, 0.25 is
chance for a balanced binary). It's well-defined regardless of how
peaked the predictor's distribution is, and degrades gracefully on
small N — unlike ECE, which is essentially noise below ~hundreds of
samples per action.

ECE is reported only at the aggregate level by default. Per-action ECE
is suppressed below `min_support` because 10-bin ECE on 50 rows (or 5
positives) is theater. Per-action Brier is fine.

References:
- Brier (1950) "Verification of forecasts expressed in terms of probability"
- Naeini et al. (2015) "Obtaining well calibrated probabilities using Bayesian binning"
"""

from __future__ import annotations

import numpy as np

from skill_wm.data.schema import ACTION_NAMES
from skill_wm.eval.dataset import ScoringRow


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and 0/1 labels.

    Range [0, 1]; 0 is perfect; 0.25 is the chance score for a balanced
    binary task with predictor-output 0.5 everywhere.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.shape != labels.shape:
        raise ValueError(f"shape mismatch: probs {probs.shape} vs labels {labels.shape}")
    if probs.size == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    adaptive: bool = True,
) -> float:
    """Weighted average gap between bin confidence and bin accuracy.

    Two binning strategies:
      - `adaptive=True`: equal-mass bins (each bin holds ~N/n_bins rows).
        More robust on tiny / skewed predictor distributions.
      - `adaptive=False`: equal-width bins on [0, 1].

    Returns NaN if `len(probs) < n_bins` (not enough samples to fill bins
    meaningfully).
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = probs.size
    if n < n_bins:
        return float("nan")

    if adaptive:
        # Equal-mass bins by quantile. Drop duplicate edges so bins with
        # ties still get a single entry instead of crashing.
        edges = np.unique(np.quantile(probs, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            return float("nan")
        bin_idx = np.clip(np.searchsorted(edges[1:-1], probs, side="right"), 0, len(edges) - 2)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.clip(np.searchsorted(edges[1:-1], probs, side="right"), 0, n_bins - 1)

    ece = 0.0
    for b in range(int(bin_idx.max()) + 1):
        mask = bin_idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += (nb / n) * abs(conf - acc)
    return float(ece)


def per_action_breakdown(
    rows: list[ScoringRow],
    probs: np.ndarray,
    min_support_for_ece: int = 30,
) -> dict[str, dict[str, float]]:
    """Per-action Brier (always) and ECE (when n >= min_support_for_ece).

    Also reports `count`, `positives`, `mean_pred`, `mean_label` so dataset
    sparsity is visible alongside the metric. ECE is `None` when support
    is insufficient — flagged in the table rather than silently reported.
    """
    actions = np.array([r.action for r in rows])
    labels = np.array([r.success for r in rows], dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)

    out: dict[str, dict[str, float]] = {}
    for a, name in enumerate(ACTION_NAMES):
        mask = actions == a
        n = int(mask.sum())
        if n == 0:
            out[name] = {
                "count": 0,
                "positives": 0,
                "mean_pred": float("nan"),
                "mean_label": float("nan"),
                "brier": float("nan"),
                "ece": float("nan"),
            }
            continue
        p = probs[mask]
        y = labels[mask]
        out[name] = {
            "count": n,
            "positives": int(y.sum()),
            "mean_pred": float(p.mean()),
            "mean_label": float(y.mean()),
            "brier": brier_score(p, y),
            "ece": (
                expected_calibration_error(p, y, n_bins=min(10, max(2, n // 10)))
                if n >= min_support_for_ece
                else float("nan")
            ),
        }
    return out


def aggregate_metrics(
    rows: list[ScoringRow],
    probs: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Headline: overall Brier + overall ECE + base-rate."""
    labels = np.array([r.success for r in rows], dtype=np.float64)
    return {
        "n": int(labels.size),
        "base_rate": float(labels.mean()) if labels.size else float("nan"),
        "mean_pred": float(np.asarray(probs).mean()) if probs.size else float("nan"),
        "brier": brier_score(probs, labels),
        "ece": expected_calibration_error(probs, labels, n_bins=n_bins, adaptive=True),
    }


def format_per_action_table(
    breakdown: dict[str, dict[str, float]],
    headline_metric: str = "brier",
) -> str:
    """Markdown-style table for stdout."""
    lines = [
        "| action               | count |  pos | mean_pred | mean_lbl |  brier |    ece |",
        "|----------------------|------:|-----:|----------:|---------:|-------:|-------:|",
    ]
    for name, m in breakdown.items():
        if m["count"] == 0:
            continue

        def _fmt(x: float) -> str:
            return "      —" if np.isnan(x) else f"{x:6.3f}"

        lines.append(
            f"| {name:<20s} | {int(m['count']):>5d} | {int(m['positives']):>4d} "
            f"| {_fmt(m['mean_pred'])} | {_fmt(m['mean_label'])} "
            f"| {_fmt(m['brier'])} | {_fmt(m['ece'])} |"
        )
    return "\n".join(lines)
