from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel.llm.pricing import ModelPrice, PriceTable, TokenUsage

REPO_POLICY = Path(__file__).parents[2] / "policies" / "model_tiers.yaml"


def test_cost_with_cached_input() -> None:
    table = PriceTable(prices={"m": ModelPrice(input=2.0, cached_input=0.5, output=8.0)})
    usage = TokenUsage(input_tokens=1_000_000, cached_input_tokens=400_000, output_tokens=500_000)
    # 600k * 2 + 400k * 0.5 + 500k * 8, per 1M
    assert table.cost(usage, "m") == pytest.approx(1.2 + 0.2 + 4.0)


def test_cached_rate_defaults_to_input() -> None:
    table = PriceTable(prices={"m": ModelPrice(input=1.0, output=1.0)})
    usage = TokenUsage(input_tokens=1_000_000, cached_input_tokens=1_000_000)
    assert table.cost(usage, "m") == pytest.approx(1.0)


def test_first_matching_name_wins() -> None:
    table = PriceTable(
        prices={"alias": ModelPrice(input=1.0, output=0), "m-2026": ModelPrice(input=2.0, output=0)}
    )
    usage = TokenUsage(input_tokens=1_000_000)
    assert table.cost(usage, "m-2026", "alias") == pytest.approx(2.0)
    assert table.cost(usage, "unknown", "alias") == pytest.approx(1.0)


def test_unknown_model_cost_is_none() -> None:
    assert PriceTable().cost(TokenUsage(input_tokens=5), "m") is None


def test_usage_adds() -> None:
    a = TokenUsage(input_tokens=1, cached_input_tokens=1, output_tokens=2, reasoning_tokens=1)
    assert a + a == TokenUsage(
        input_tokens=2, cached_input_tokens=2, output_tokens=4, reasoning_tokens=2
    )


def test_load_yaml(tmp_path: Path) -> None:
    path = tmp_path / "tiers.yaml"
    path.write_text("prices:\n  m:\n    input: 1.5\n    output: 6\n")
    table = PriceTable.load(path)
    assert table.prices["m"] == ModelPrice(input=1.5, output=6.0)


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert PriceTable.load(tmp_path / "nope.yaml").prices == {}


def test_load_rejects_bad_price(tmp_path: Path) -> None:
    path = tmp_path / "tiers.yaml"
    path.write_text("prices:\n  m:\n    input: -1\n    output: 1\n")
    with pytest.raises(ValidationError):
        PriceTable.load(path)


def test_repo_policy_file_loads() -> None:
    PriceTable.load(REPO_POLICY)
