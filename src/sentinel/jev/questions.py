"""Question builders and the initial question catalog (PLAN.md section 7).

One judgment per question: never pack several decisions into one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sentinel.jev.models import (
    ChoiceQuestion,
    NoulCriteria,
    NoulQuestion,
    Question,
    ScoreQuestion,
)


def choice(instructions: str, options: Mapping[str, str]) -> ChoiceQuestion:
    return ChoiceQuestion(instructions=instructions, criteria=dict(options))


def score(instructions: str, levels: Sequence[str]) -> ScoreQuestion:
    """`levels` are ordered from low to high."""
    return ScoreQuestion(instructions=instructions, criteria=list(levels))


def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> NoulQuestion:
    """Yes/no. Optional `criteria` must have exactly the keys "true" and "false"."""
    if criteria is None:
        return NoulQuestion(instructions=instructions)
    if set(criteria) != {"true", "false"}:
        raise ValueError('noul criteria must have exactly the keys "true" and "false"')
    return NoulQuestion(
        instructions=instructions,
        criteria=NoulCriteria(true=criteria["true"], false=criteria["false"]),
    )


TRIAGE: dict[str, Question] = {
    "category": choice(
        "Which team should handle this?",
        {
            "billing": "Payments, invoices, refunds, charges",
            "technical": "Bugs, outages, errors, integrations",
            "account": "Login, access, profile, security",
            "sales": "Pricing, upgrades, new accounts",
        },
    ),
    "urgency": score(
        "How urgent is this for the customer's business?",
        ["Not urgent", "Soon", "Today", "Immediately / outage"],
    ),
    "path": choice(
        "What does resolving this need next?",
        {
            "lookup": "Needs account/order data before answering",
            "generate": "Can be answered from the message and policies",
            "human": "Legal, threats, sensitive, or ambiguous",
        },
    ),
    "is_abusive": noul("Is the message abusive or threatening?"),
}

ROUTE_MODEL: dict[str, Question] = {
    "tier": choice(
        "Which approved model tier fits this task?",
        {
            "fast": "Short, routine, low-risk replies",
            "strong": "Multi-step reasoning, long context, disputes, money",
        },
    ),
}

GUARD: dict[str, Question] = {
    "safe_to_run": noul(
        "Is it safe to run this tool call without human approval?",
        criteria={
            "true": "Read-only or reversible, in scope, within policy",
            "false": "Destructive, irreversible, out of scope, or unclear",
        },
    ),
}

VERIFY: dict[str, Question] = {
    "claim_supported": noul("Is every factual claim in `draft` supported by `tool_results`?"),
    "task_status": choice(
        "Based on tool results, what is the task status?",
        {
            "complete": "All required actions succeeded",
            "verify_more": "Partial evidence, needs another check",
            "failed": "A required action failed or was not run",
        },
    ),
}
