from __future__ import annotations

import pytest

from sentinel.evals.metrics import (
    brier_binary,
    brier_multiclass,
    calibration_bins,
    expected_calibration_error,
    percentile,
)


def test_brier_multiclass() -> None:
    labels = ["a", "b", "c"]
    assert brier_multiclass({"a": 1.0}, "a", labels) == 0.0
    assert brier_multiclass({"b": 1.0}, "a", labels) == 2.0
    assert brier_multiclass({"a": 0.5, "b": 0.5}, "a", labels) == pytest.approx(0.5)


def test_brier_binary() -> None:
    assert brier_binary(1.0, True) == 0.0
    assert brier_binary(0.2, False) == pytest.approx(0.04)
    assert brier_binary(0.2, True) == pytest.approx(0.64)


def test_calibration_bins_edges() -> None:
    bins = calibration_bins([(0.0, False), (0.05, True), (0.95, True), (1.0, True), (1.0, False)])
    assert len(bins) == 10
    assert bins[0].n == 2 and bins[0].observed == 0.5
    assert bins[9].n == 3  # p == 1.0 lands in the last bin
    assert bins[9].mean_predicted == pytest.approx((0.95 + 1 + 1) / 3)
    assert all(b.n == 0 and b.observed is None for b in bins[1:9])


def test_ece() -> None:
    perfect = calibration_bins([(0.95, True)] * 19 + [(0.95, False)])
    assert expected_calibration_error(perfect) == pytest.approx(0.0)
    overconfident = calibration_bins([(0.95, False)] * 10)
    assert expected_calibration_error(overconfident) == pytest.approx(0.95)
    assert expected_calibration_error(calibration_bins([])) is None


def test_percentile() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile(list(map(float, range(1, 101))), 95) == pytest.approx(95.05)
