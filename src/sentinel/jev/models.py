"""Request/response models for Jev (System One).

Wire format (confirmed by `sentinel probe`, see `sentinel.jev.parsing`):
    request:  {"model": str, "state": str | object | [str], "questions": {id: Question}}
    response: {"code": 0, "message": "ok",
               "data": {"creditsUsed", "result": {"answers": {id: Answer}, "usage", "elapsedMs"}}}
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
State = str | dict[str, Any] | list[str]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- questions ---------------------------------------------------------------


class ChoiceQuestion(_Frozen):
    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    criteria: dict[str, str]

    @field_validator("criteria")
    @classmethod
    def _check_options(cls, v: dict[str, str]) -> dict[str, str]:
        if not 2 <= len(v) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice needs 2..{MAX_CHOICE_OPTIONS} options, got {len(v)}")
        if any(not k.strip() for k in v):
            raise ValueError("choice option names must be non-empty")
        return v


class ScoreQuestion(_Frozen):
    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    criteria: list[str]
    """Ordered low -> high."""

    @field_validator("criteria")
    @classmethod
    def _check_levels(cls, v: list[str]) -> list[str]:
        if not MIN_SCORE_LEVELS <= len(v) <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels, got {len(v)}"
            )
        if len(set(v)) != len(v):
            raise ValueError("score levels must be unique")
        return v


class NoulCriteria(_Frozen):
    true: str = Field(min_length=1)
    false: str = Field(min_length=1)


class NoulQuestion(_Frozen):
    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1)
    criteria: NoulCriteria | None = None


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class JevRequest(_Frozen):
    model: str = Field(min_length=1)
    state: State
    questions: dict[str, Question] = Field(min_length=1)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


# --- answers -----------------------------------------------------------------


class _Answer(BaseModel):
    # Vendors may add fields; ignore rather than fail, but keep what we need strict.
    model_config = ConfigDict(frozen=True, extra="ignore")


class ChoiceAnswer(_Answer):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class ScoreAnswer(_Answer):
    type: Literal["score"]
    score: float
    """Expected level index, 0 (lowest) .. len(levels) - 1."""
    legend: dict[str, str]
    """Level index (as a string) -> level text."""
    probabilities: dict[str, Probability]
    """Level index (as a string) -> probability."""
    confidence: Probability


class NoulAnswer(_Answer):
    type: Literal["noul"]
    noul: Probability
    """Probability of yes."""


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


class JevResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    """Which JevProvider answered, e.g. "thejevai" or "llm_fallback"."""
    model: str
    """Model ID to audit: the vendor's if it returned one, else the one we requested."""
    model_verified: bool
    """True only if the provider reported the model ID. thejevai.com currently does not."""
    answers: dict[str, Answer]
    usage: dict[str, Any] | None = None
    credits_used: float | None = None
    vendor_elapsed_ms: float | None = None
    """Timing as reported by the vendor (informational; untrusted)."""
    latency_ms: float
    """Client-side wall-clock latency (authoritative for evals)."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Provider trace IDs worth auditing, e.g. the gateway's generation ID and the
    upstream provider that actually answered. No state, no answers."""
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    def choice(self, qid: str) -> ChoiceAnswer:
        return _typed(self.answers, qid, ChoiceAnswer)

    def score(self, qid: str) -> ScoreAnswer:
        return _typed(self.answers, qid, ScoreAnswer)

    def noul(self, qid: str) -> NoulAnswer:
        return _typed(self.answers, qid, NoulAnswer)


def _typed[T: _Answer](answers: dict[str, Answer], qid: str, kind: type[T]) -> T:
    answer = answers[qid]
    if not isinstance(answer, kind):
        raise TypeError(f"answer {qid!r} is {type(answer).__name__}, not {kind.__name__}")
    return answer
