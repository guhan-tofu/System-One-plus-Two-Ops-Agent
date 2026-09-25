from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from sentinel.cli import app
from sentinel.evals.dataset import EvalItem, load_dataset
from sentinel.evals.report import render
from sentinel.evals.runner import run_provider, score_run
from sentinel.jev.errors import JevOverloadedError
from sentinel.jev.models import (
    Answer,
    ChoiceAnswer,
    JevResult,
    NoulAnswer,
    Question,
    ScoreAnswer,
    State,
)
from sentinel.policy.engine import Thresholds
from tests.fixtures.vercel import EVAL_URL, GATEWAY_URL

ROOT = Path(__file__).parents[2]
DATASET = ROOT / "evals" / "datasets" / "triage.jsonl"
POLICY = Thresholds.load(ROOT / "policies" / "thresholds.yaml").triage


def _choice(value: str, options: list[str], p: float) -> ChoiceAnswer:
    rest = (1 - p) / (len(options) - 1)
    probs = {o: p if o == value else rest for o in options}
    return ChoiceAnswer(type="choice", choice=value, probabilities=probs, confidence=p)


class OracleJev:
    """Answers from the labels (optionally wrong on purpose) to test the scoring."""

    name = "oracle"

    def __init__(self, items: list[EvalItem], *, wrong_category: set[str] = frozenset()) -> None:  # type: ignore[assignment]
        self.by_message = {i.body: i for i in items}
        self.wrong_category = wrong_category

    async def evaluate(
        self, state: State, questions: Mapping[str, Question], model: str | None = None
    ) -> JevResult:
        assert isinstance(state, dict)
        item = self.by_message[state["message"]]
        labels = item.labels
        category = "sales" if item.id in self.wrong_category else labels.category
        answers: dict[str, Answer] = {
            "category": _choice(category, ["billing", "technical", "account", "sales"], 0.9),
            "path": _choice(labels.path, ["lookup", "generate", "human"], 0.9),
            "urgency": ScoreAnswer(
                type="score",
                score=float(labels.urgency),
                legend={str(i): str(i) for i in range(4)},
                probabilities={str(i): 1.0 if i == labels.urgency else 0.0 for i in range(4)},
                confidence=1.0,
            ),
            "is_abusive": NoulAnswer(type="noul", noul=0.9 if labels.is_abusive else 0.05),
        }
        return JevResult(
            provider=self.name,
            model="oracle-1",
            model_verified=True,
            answers=answers,
            usage={"cost_usd": 0.001},
            latency_ms=10.0,
        )

    async def aclose(self) -> None:
        return None


class BrokenJev(OracleJev):
    name = "broken"

    async def evaluate(self, *args: Any, **kwargs: Any) -> JevResult:
        raise JevOverloadedError("HTTP 529: overloaded", status_code=529)


async def test_perfect_provider_scores_perfectly() -> None:
    items = load_dataset(DATASET)
    score = score_run(await run_provider(OracleJev(items), items), POLICY)

    assert score.errors == []
    for q in ("category", "path", "urgency", "is_abusive"):
        assert score.questions[q].accuracy == 1.0, q
    assert score.questions["urgency"].mean_brier == 0.0
    assert score.cost_per_item == pytest.approx(0.001)
    assert score.p50_ms == 10.0
    # Every item needing a human (path human or abusive) is escalated by the policy.
    assert score.policy.needs_human_escalated == score.policy.needs_human > 0
    assert score.policy.proceeded_category_correct == score.policy.proceeded


async def test_mistakes_show_up() -> None:
    items = load_dataset(DATASET)[:10]
    wrong = {items[0].id, items[1].id}
    score = score_run(await run_provider(OracleJev(items, wrong_category=wrong), items), POLICY)
    assert score.questions["category"].accuracy == pytest.approx(0.8)
    assert {m[0] for m in score.questions["category"].misses} == wrong


async def test_rate_limit_spaces_request_starts() -> None:
    import time

    items = load_dataset(DATASET)[:4]
    started = time.perf_counter()
    run = await run_provider(OracleJev(items), items, concurrency=4, max_rate=20)
    assert len(run.runs) == 4
    assert time.perf_counter() - started >= 3 / 20 * 0.9  # 4 starts, 3 gaps of 50 ms


async def test_errors_are_counted_not_scored() -> None:
    items = load_dataset(DATASET)[:5]
    score = score_run(await run_provider(BrokenJev(items), items), POLICY)
    assert len(score.errors) == 5 and "529" in score.errors[0][1]
    assert score.questions["category"].accuracy is None
    assert score.cost_per_item is None


async def test_report_renders_side_by_side() -> None:
    items = load_dataset(DATASET)[:8]
    scores = [
        score_run(await run_provider(OracleJev(items), items), POLICY),
        score_run(await run_provider(BrokenJev(items), items), POLICY),
    ]
    md = render(
        dataset=DATASET,
        items=items,
        scores=scores,
        generated_at=datetime(2026, 9, 25, tzinfo=UTC),
        thresholds_path=Path("policies/thresholds.yaml"),
    )
    assert "| | oracle | broken |" in md
    assert "| Errors | 0/8 | 8/8 |" in md
    assert "| **category** accuracy | 100.0% | - |" in md
    assert "### category: calibration" in md and "### path: confusion" in md
    assert "## broken" in md and "**Errors** (first 5):" in md


def gateway_answer(request: httpx.Request) -> httpx.Response:
    """Always the first offered option, 0.9 confident (enough to exercise the report)."""
    questions = json.loads(request.content)["questions"]
    answers: dict[str, Any] = {}
    for qid, q in questions.items():
        if q["type"] == "choice":
            options = list(q["criteria"])
            answers[qid] = {
                "type": "choice",
                "choice": options[0],
                "probabilities": {
                    o: 0.9 if i == 0 else 0.1 / (len(options) - 1) for i, o in enumerate(options)
                },
            }
        elif q["type"] == "score":
            answers[qid] = {
                "type": "score",
                "score": 1.0,
                "probabilities": {
                    str(i): 1.0 if i == 1 else 0.0 for i in range(len(q["criteria"]))
                },
            }
        else:
            answers[qid] = {"type": "boolean", "probability": 0.1}
    return httpx.Response(200, json={"answers": answers, "warnings": []})


@respx.mock
def test_cli_eval_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "fake-gateway-key")  # pragma: allowlist secret
    monkeypatch.setenv("AI_GATEWAY_BASE_URL", GATEWAY_URL)
    respx.post(EVAL_URL).mock(side_effect=gateway_answer)

    result = CliRunner().invoke(
        app,
        ["eval", "--provider", "vercel", "--provider", "thejevai", "--limit", "6",
         "--out", str(tmp_path)],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    [report] = list(tmp_path.glob("*-triage.md"))
    md = report.read_text()
    assert "| | vercel | thejevai |" in md
    assert "| Errors | 0/6 | 6/6 |" in md  # thejevai has no key here
    assert "provider not available" in md
    assert "fake-gateway-key" not in md
    assert "vercel: errors 0/6" in result.stdout
