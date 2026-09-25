from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from sentinel.evals.dataset import load_dataset

DATASET = Path(__file__).parents[2] / "evals" / "datasets" / "triage.jsonl"


def test_repo_dataset_is_valid_and_balanced() -> None:
    items = load_dataset(DATASET)
    assert 50 <= len(items) <= 200
    assert set(Counter(i.labels.category for i in items)) == {
        "billing", "technical", "account", "sales",
    }  # fmt: skip
    assert set(Counter(i.labels.path for i in items)) == {"lookup", "generate", "human"}
    assert set(Counter(i.labels.urgency for i in items)) == {0, 1, 2, 3}
    assert sum(i.labels.is_abusive for i in items) >= 5


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    line = DATASET.read_text().splitlines()[0]
    path = tmp_path / "dup.jsonl"
    path.write_text(f"{line}\n{line}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)
