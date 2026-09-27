"""Answer sanity checks shared by the HTTP Jev parsers."""

from __future__ import annotations

import math
from collections.abc import Mapping

from sentinel.jev.errors import JevResponseError


def probability(qid: str, name: str, value: float) -> float:
    """`value` if it is a finite probability, else JevResponseError."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise JevResponseError(f"answer {qid!r}: {name} {value} is outside [0, 1]")
    return value


def check_distribution(qid: str, probs: Mapping[str, float], allowed: list[str]) -> None:
    """Every label is one we offered and every probability is in [0, 1]."""
    unknown = sorted(set(probs) - set(allowed))
    if unknown:
        raise JevResponseError(f"answer {qid!r}: probabilities for unknown labels {unknown}")
    for label, p in probs.items():
        probability(qid, f"probability for {label!r}", p)
