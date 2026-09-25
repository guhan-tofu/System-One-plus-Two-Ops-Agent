"""Jev's `confidence` as a function of an answer's probabilities.

Reverse-engineered from live Jev responses (2026-09-25; 15 score and 9 choice
samples, all matching to Jev's 2-decimal rounding) and confirmed against the
official Jev via Vercel AI Gateway, whose reported confidence matches exactly.
Used when a provider omits confidence (and by `llm_fallback`) so that thresholds
mean the same thing whichever provider answered.
"""

from __future__ import annotations

from collections.abc import Sequence


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def choice_confidence(probabilities: Sequence[float]) -> float:
    """Top probability rescaled so uniform -> 0 and certain -> 1: (n*p_max - 1) / (n - 1)."""
    n = len(probabilities)
    if n < 2:
        raise ValueError("need at least 2 options")
    return _clamp((n * max(probabilities) - 1) / (n - 1))


def score_confidence(probabilities: Sequence[float]) -> float:
    """1 - E|level - modal level| / D_n.

    Ordinal: mass on neighbouring levels costs less than mass far away. D_n is the
    mean distance from the middle level under a uniform distribution, so the scale
    is comparable across 2..10 levels.
    """
    n = len(probabilities)
    if n < 2:
        raise ValueError("need at least 2 levels")
    mode = max(range(n), key=lambda i: probabilities[i])
    spread = sum(p * abs(i - mode) for i, p in enumerate(probabilities))
    centre = (n - 1) / 2
    d_n = sum(abs(i - centre) for i in range(n)) / n
    return _clamp(1 - spread / d_n)
