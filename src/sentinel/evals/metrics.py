"""Eval metrics: accuracy, Brier score, calibration (10 bins), percentiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def brier_multiclass(probs: Mapping[str, float], label: str, labels: Sequence[str]) -> float:
    """Sum over classes of (p - y)^2. 0 is perfect, 2 is worst."""
    return sum((probs.get(k, 0.0) - (1.0 if k == label else 0.0)) ** 2 for k in labels)


def brier_binary(p: float, outcome: bool) -> float:
    """(p - y)^2. 0 is perfect, 1 is worst."""
    return (p - (1.0 if outcome else 0.0)) ** 2


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_predicted: float | None
    observed: float | None
    """Fraction of items in the bin where the predicted event happened."""


def calibration_bins(pairs: Sequence[tuple[float, bool]], n_bins: int = 10) -> list[CalibrationBin]:
    """Bin (predicted probability, event happened) pairs into equal-width bins on [0, 1]."""
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, happened in pairs:
        idx = min(int(p * n_bins), n_bins - 1) if p >= 0 else 0
        grouped[idx].append((p, happened))
    bins = []
    for i, group in enumerate(grouped):
        n = len(group)
        bins.append(
            CalibrationBin(
                lo=i / n_bins,
                hi=(i + 1) / n_bins,
                n=n,
                mean_predicted=sum(p for p, _ in group) / n if n else None,
                observed=sum(1 for _, h in group if h) / n if n else None,
            )
        )
    return bins


def expected_calibration_error(bins: Sequence[CalibrationBin]) -> float | None:
    """Sample-weighted mean |predicted - observed| over non-empty bins."""
    total = sum(b.n for b in bins)
    if total == 0:
        return None
    return (
        sum(
            b.n * abs(b.mean_predicted - b.observed)
            for b in bins
            if b.n and b.mean_predicted is not None and b.observed is not None
        )
        / total
    )


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None
