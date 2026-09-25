from __future__ import annotations

import pytest

from sentinel.jev.confidence import choice_confidence, score_confidence

# (probabilities, confidence) pairs observed from live Jev responses
# (2026-09-25). The vendor computes confidence from unrounded probabilities but
# shows both rounded to 2 decimals; on longer scales that input rounding moves the
# expected distance by a few hundredths, hence the tolerance.
TOLERANCE = 0.02
OBSERVED_CHOICE = [
    ([0.19, 0.81], 0.62),
    ([0.14, 0.0, 0.4, 0.46], 0.27),
    ([0.11, 0.26, 0.34, 0.28], 0.12),
    ([0.92, 0.0, 0.08, 0.0], 0.89),
    ([0.0, 1.0], 1.0),
]
OBSERVED_SCORE = [
    ([0.23, 0.77], 0.54),
    ([0.97, 0.03], 0.94),
    ([0.03, 0.46, 0.51], 0.23),
    ([0.82, 0.18, 0.0], 0.73),
    ([0.88, 0.12, 0.0], 0.81),
    ([0.01, 0.21, 0.7, 0.08], 0.69),
    ([0.31, 0.66, 0.03, 0.0], 0.66),
    ([0.09, 0.66, 0.23, 0.02, 0.0], 0.70),
    ([0.0, 0.3, 0.37, 0.3, 0.03], 0.45),
    ([0.58, 0.19, 0.1, 0.08, 0.02, 0.03], 0.44),
    ([0.38, 0.28, 0.14, 0.13, 0.04, 0.03], 0.17),
]


@pytest.mark.parametrize(("probs", "expected"), OBSERVED_CHOICE)
def test_choice_confidence_matches_jev(probs: list[float], expected: float) -> None:
    assert choice_confidence(probs) == pytest.approx(expected, abs=TOLERANCE)


@pytest.mark.parametrize(("probs", "expected"), OBSERVED_SCORE)
def test_score_confidence_matches_jev(probs: list[float], expected: float) -> None:
    assert score_confidence(probs) == pytest.approx(expected, abs=TOLERANCE)


def test_bounds() -> None:
    assert choice_confidence([0.25] * 4) == 0.0
    assert choice_confidence([1.0, 0.0, 0.0]) == 1.0
    assert score_confidence([0.0, 0.0, 1.0]) == 1.0
    # Mass at both extremes of a long scale would go negative; clamp to 0.
    assert score_confidence([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]) == 0.0


def test_score_penalises_distant_mass_more() -> None:
    near = score_confidence([0.0, 0.3, 0.7, 0.0])
    far = score_confidence([0.3, 0.0, 0.7, 0.0])
    assert near > far


@pytest.mark.parametrize("fn", [choice_confidence, score_confidence])
def test_needs_two_labels(fn: object) -> None:
    with pytest.raises(ValueError):
        fn([1.0])  # type: ignore[operator]
