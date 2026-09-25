"""Fake OpenAI Responses API bodies. Values are invented; no real data."""

from __future__ import annotations

import json
from typing import Any

import httpx

RESPONSES_URL = "https://api.openai.com/v1/responses"
FAKE_OPENAI_MODEL = "fake-model-2026-01-01"
# Tiny Retry-After so SDK retries don't slow the tests down.
FAST_RETRY = {"retry-after-ms": "1"}


def response_body(
    text: str,
    *,
    model: str = FAKE_OPENAI_MODEL,
    input_tokens: int = 100,
    cached_tokens: int = 0,
    output_tokens: int = 20,
    status: str = "completed",
    refusal: bool = False,
) -> dict[str, Any]:
    content = (
        {"type": "refusal", "refusal": "I can't help with that."}
        if refusal
        else {"type": "output_text", "text": text, "annotations": []}
    )
    body: dict[str, Any] = {
        "id": "resp_fake",
        "object": "response",
        "created_at": 1,
        "model": model,
        "status": status,
        "output": [
            {
                "type": "message",
                "id": "msg_fake",
                "role": "assistant",
                "status": "completed",
                "content": [content],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    if status == "incomplete":
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    return body


def error_body(
    message: str, *, param: str | None = None, code: str | None = None
) -> dict[str, Any]:
    return {
        "error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}
    }


def _resolve(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if isinstance(ref, str):
        name = ref.rsplit("/", 1)[-1]
        return dict(schema["$defs"][name])
    return node


def labels_from_request(request: httpx.Request) -> list[str] | None:
    """Allowed answers from a structured-output request, or None for a yes/no schema."""
    schema = json.loads(request.content)["text"]["format"]["schema"]
    if "distribution" not in schema["properties"]:
        return None
    item = _resolve(schema, schema["properties"]["distribution"]["items"])
    return list(item["properties"]["answer"]["enum"])


def judge(request: httpx.Request) -> httpx.Response:
    """Answer any fallback question: 0.8 on the first allowed answer, rest spread evenly."""
    labels = labels_from_request(request)
    if labels is None:
        payload: dict[str, Any] = {"probability_yes": 0.2}
    else:
        rest = 0.2 / (len(labels) - 1)
        payload = {
            "distribution": [
                {"answer": label, "probability": 0.8 if i == 0 else rest}
                for i, label in enumerate(labels)
            ]
        }
    return httpx.Response(200, json=response_body(json.dumps(payload)))
