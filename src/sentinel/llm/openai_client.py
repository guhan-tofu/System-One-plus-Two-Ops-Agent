"""OpenAI (System Two) client: tiered generation with token and cost accounting.

- Tiers ("fast", "strong") map to model names from settings; nothing else is callable.
- Uses the Responses API with `store=False` so OpenAI does not retain our state.
- The SDK retries 408/409/429/5xx, timeouts and connection errors (exponential
  backoff with jitter, honours Retry-After), `MAX_RETRIES` times.
- SDK errors are mapped to `LLMError` subclasses whose messages never contain
  the API key, the prompt, or the model's output.
- Prompts and outputs are never logged; token counts, cost and model IDs are.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, cast

import httpx
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from sentinel.config import LLMTier, Settings
from sentinel.llm.pricing import PriceTable, TokenUsage
from sentinel.log import get_logger

log = get_logger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
MAX_RETRIES = 3
_DETAIL_LIMIT = 200


# --- errors ------------------------------------------------------------------


class LLMError(Exception):
    """Base class for all OpenAI client failures."""


class LLMConfigError(LLMError):
    """Missing key or tier model name."""


class LLMAuthError(LLMError):
    """401/403."""


class LLMRateLimitError(LLMError):
    """429 after all retries."""


class LLMServerError(LLMError):
    """5xx after all retries."""

    def __init__(self, message: str, *, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(message)


class LLMRequestError(LLMError):
    """400/404/422: OpenAI rejected the request. Not retried."""

    def __init__(self, message: str, *, param: str | None = None) -> None:
        self.detail = message
        self.param = param
        super().__init__(f"{message} (param: {param})" if param else message)


class LLMHTTPError(LLMError):
    """Any other non-success status."""

    def __init__(self, message: str, *, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(message)


class LLMTimeoutError(LLMError):
    """Timed out after all retries."""


class LLMTransportError(LLMError):
    """Network failure after all retries."""


class LLMResponseError(LLMError):
    """Refusal, truncated output, or output that does not fit the schema."""


# --- results -----------------------------------------------------------------


class Generation(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    tier: LLMTier
    requested_model: str
    model: str
    """Model ID returned by OpenAI (usually a dated version). Always audit this."""
    usage: TokenUsage
    cost_usd: float | None
    """None when the model has no entry in the price table."""
    latency_ms: float


class OpenAIClient:
    def __init__(
        self,
        *,
        api_key: SecretStr,
        models: Mapping[LLMTier, str],
        prices: PriceTable | None = None,
        client: httpx.AsyncClient | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        if not api_key.get_secret_value():
            raise LLMConfigError("OPENAI_API_KEY is not set")
        self._models = dict(models)
        self._prices = prices or PriceTable()
        self._owns_client = client is None
        self._http = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        # The SDK is typed for httpx2 but supports legacy httpx clients and timeouts
        # at runtime. We pass httpx so the whole app shares one HTTP stack (and
        # respx can mock it in tests).
        self._sdk = AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            http_client=cast(Any, self._http),
            max_retries=max_retries,
            timeout=cast(Any, DEFAULT_TIMEOUT),
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, client: httpx.AsyncClient | None = None
    ) -> OpenAIClient:
        return cls(
            api_key=settings.openai_api_key,
            models={"fast": settings.openai_model_fast, "strong": settings.openai_model_strong},
            prices=PriceTable.load(settings.model_tiers_path),
            client=client,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    def model_for(self, tier: LLMTier) -> str:
        model = self._models.get(tier)
        if not model:
            raise LLMConfigError(
                f"no model configured for tier {tier!r} (OPENAI_MODEL_{tier.upper()})"
            )
        return model

    async def generate(
        self,
        tier: LLMTier,
        *,
        instructions: str,
        input: str,
        max_output_tokens: int | None = None,
    ) -> Generation:
        """Free-text generation on the given tier."""
        model = self.model_for(tier)
        started = time.perf_counter()
        response = await self._call(
            self._sdk.responses.create,
            model=model,
            instructions=instructions,
            input=input,
            **_optional(max_output_tokens=max_output_tokens),
        )
        _check_complete(response)
        text = response.output_text
        if not text:
            raise LLMResponseError(_refusal_or("model returned no text", response))
        return self._record(tier, model, response, text, started)

    async def structured[T: BaseModel](
        self,
        tier: LLMTier,
        *,
        instructions: str,
        input: str,
        schema: type[T],
        max_output_tokens: int | None = None,
    ) -> tuple[T, Generation]:
        """Generation constrained to `schema` via structured outputs (strict JSON schema)."""
        model = self.model_for(tier)
        started = time.perf_counter()
        try:
            response = await self._call(
                self._sdk.responses.parse,
                model=model,
                instructions=instructions,
                input=input,
                text_format=schema,
                **_optional(max_output_tokens=max_output_tokens),
            )
        except ValidationError as exc:
            fields = ", ".join(".".join(map(str, e["loc"])) or "<root>" for e in exc.errors())
            raise LLMResponseError(f"output does not match schema (fields: {fields})") from None
        _check_complete(response)
        parsed = response.output_parsed
        if parsed is None:
            raise LLMResponseError(_refusal_or("model returned no parsable output", response))
        return parsed, self._record(tier, model, response, response.output_text, started)

    async def _call(self, method: Any, **kwargs: Any) -> Any:
        try:
            return await method(store=False, **kwargs)
        except openai.APITimeoutError:
            raise LLMTimeoutError("request timed out") from None
        except openai.APIConnectionError:
            raise LLMTransportError("connection error") from None
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise LLMAuthError(f"HTTP {exc.status_code}: authentication failed") from None
        except openai.RateLimitError:
            raise LLMRateLimitError("HTTP 429: rate limited") from None
        except (
            openai.BadRequestError,
            openai.NotFoundError,
            openai.UnprocessableEntityError,
        ) as exc:
            raise LLMRequestError(
                f"HTTP {exc.status_code}: {_detail(exc)}", param=_param(exc)
            ) from None
        except openai.InternalServerError as exc:
            raise LLMServerError(f"HTTP {exc.status_code}", status_code=exc.status_code) from None
        except openai.APIStatusError as exc:
            raise LLMHTTPError(f"HTTP {exc.status_code}", status_code=exc.status_code) from None
        except (openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError) as exc:
            raise LLMResponseError(type(exc).__name__) from None

    def _record(
        self, tier: LLMTier, requested: str, response: Any, text: str, started: float
    ) -> Generation:
        usage = _usage(response)
        returned = str(response.model or requested)
        cost = self._prices.cost(usage, returned, requested)
        generation = Generation(
            text=text,
            tier=tier,
            requested_model=requested,
            model=returned,
            usage=usage,
            cost_usd=cost,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        log.info(
            "llm.generate",
            tier=tier,
            requested_model=requested,
            model=returned,
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cost_usd=cost,
            latency_ms=round(generation.latency_ms, 1),
        )
        if cost is None:
            log.warning("llm.price_unknown", model=returned)
        return generation


def _optional(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return TokenUsage(
        input_tokens=usage.input_tokens or 0,
        cached_input_tokens=getattr(input_details, "cached_tokens", 0) or 0,
        output_tokens=usage.output_tokens or 0,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) or 0,
    )


def _check_complete(response: Any) -> None:
    status = getattr(response, "status", None)
    if status not in (None, "completed"):
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        raise LLMResponseError(f"response {status}" + (f" ({reason})" if reason else ""))


def _refusal_or(default: str, response: Any) -> str:
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) == "refusal":
                return "model refused"
    return default


def _param(exc: openai.APIStatusError) -> str | None:
    param = getattr(exc, "param", None)
    return str(param) if param else None


def _detail(exc: openai.APIStatusError) -> str:
    body = exc.body
    if isinstance(body, Mapping):
        message = body.get("message")
        if isinstance(message, str) and message:
            return message[:_DETAIL_LIMIT]
    return "request rejected"
