"""Shared HTTP machinery for Jev providers (thejevai.com, Vercel AI Gateway).

- Retries 429/529 (plus any provider-specific `retry_statuses`), exponential
  backoff with jitter, max 4 attempts.
- Auth failures, validation errors, other statuses, timeouts and transport
  errors fail immediately.
- The API key is sent per request and never logged or put in exception text.
- Request state is never logged (it may contain customer data even after redaction).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import httpx
from pydantic import SecretStr
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from sentinel.jev.errors import (
    JevAuthError,
    JevConfigError,
    JevHTTPError,
    JevOverloadedError,
    JevRateLimitError,
    JevResponseError,
    JevRetryableError,
    JevTimeoutError,
    JevTransportError,
    JevValidationError,
)
from sentinel.jev.models import JevRequest, JevResult, Question, State
from sentinel.log import get_logger

log = get_logger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
MAX_ATTEMPTS = 4
_DETAIL_LIMIT = 300


def default_retry_wait() -> wait_base:
    return wait_exponential_jitter(initial=0.5, max=8.0, jitter=0.5)


class HTTPJevProvider(ABC):
    name: str
    key_env: ClassVar[str]
    """Env var holding the key, named in config errors."""
    validation_statuses: ClassVar[tuple[int, ...]] = (422,)
    """Statuses meaning "the provider rejected our request" (surfaced with the field)."""
    retry_statuses: ClassVar[tuple[int, ...]] = (529,)
    """Transient "overloaded" statuses retried like 529 (429 is always retried)."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        url: str,
        default_model: str,
        client: httpx.AsyncClient | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        retry_wait: wait_base | None = None,
    ) -> None:
        if not api_key.get_secret_value():
            raise JevConfigError(f"{self.key_env} is not set")
        self._api_key = api_key
        self._url = url
        self._default_model = default_model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        self._max_attempts = max_attempts
        self._retry_wait = retry_wait or default_retry_wait()

    # --- provider-specific hooks -------------------------------------------

    @abstractmethod
    def payload(self, request: JevRequest) -> dict[str, Any]:
        """JSON body for the request."""

    def headers(self, request: JevRequest) -> dict[str, str]:
        """Extra headers (the Authorization header is added separately)."""
        return {}

    @abstractmethod
    def parse(self, raw: Any, request: JevRequest, *, latency_ms: float) -> JevResult:
        """Strictly parse a 2xx body. Raise JevResponseError on anything unexpected."""

    @abstractmethod
    def describe_shape(self, raw: Any) -> dict[str, Any]:
        """Structural summary (keys and types, no values) for `sentinel probe`."""

    # --- shared behaviour ---------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def build_request(
        self, state: State, questions: Mapping[str, Question], model: str | None = None
    ) -> JevRequest:
        return JevRequest(model=model or self._default_model, state=state, questions=questions)

    async def evaluate(
        self,
        state: State,
        questions: Mapping[str, Question],
        model: str | None = None,
    ) -> JevResult:
        request = self.build_request(state, questions, model)
        raw, latency_ms = await self.post_raw(request)
        result = self.parse(raw, request, latency_ms=latency_ms)
        log.info(
            "jev.evaluate",
            provider=self.name,
            requested_model=request.model,
            model=result.model,
            model_verified=result.model_verified,
            questions=sorted(request.questions),
            latency_ms=round(latency_ms, 1),
            vendor_elapsed_ms=result.vendor_elapsed_ms,
            credits_used=result.credits_used,
            metadata=result.metadata,
        )
        return result

    async def post_raw(self, request: JevRequest) -> tuple[Any, float]:
        """POST with retries; return the decoded JSON body and total latency in ms."""
        payload = self.payload(request)
        headers = {
            **self.headers(request),
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
        }
        started = time.perf_counter()
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(JevRetryableError),
            stop=stop_after_attempt(self._max_attempts),
            wait=self._retry_wait,
            before_sleep=self._log_retry,
            reraise=True,
        )
        body: Any = None
        async for attempt in retrying:
            with attempt:
                body = await self._post_once(payload, headers)
        return body, (time.perf_counter() - started) * 1000

    async def _post_once(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        try:
            response = await self._client.post(self._url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise JevTimeoutError(f"request timed out ({type(exc).__name__})") from None
        except httpx.TransportError as exc:
            raise JevTransportError(f"transport error ({type(exc).__name__})") from None

        status = response.status_code
        if 200 <= status < 300:
            try:
                return response.json()
            except ValueError:
                raise JevResponseError(f"HTTP {status} with a non-JSON body") from None
        if status in (401, 403):
            raise JevAuthError(_status_detail(status, response, "authentication failed"))
        if status in self.validation_statuses:
            field, detail = _validation_detail(response)
            raise JevValidationError(f"HTTP {status}: {detail}", field=field)
        if status == 429:
            raise JevRateLimitError("HTTP 429: rate limited", status_code=status)
        if status in self.retry_statuses:
            raise JevOverloadedError(
                _status_detail(status, response, "overloaded"), status_code=status
            )
        raise JevHTTPError(_status_detail(status, response), status_code=status)

    def _log_retry(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        log.warning(
            "jev.retry",
            provider=self.name,
            attempt=state.attempt_number,
            status_code=getattr(exc, "status_code", None),
            sleep_s=round(state.upcoming_sleep, 2),
        )


def error_message(body: Any) -> str | None:
    """Vendor error text from `{"message"}` or `{"error": {"message"} | str}`, truncated."""
    if not isinstance(body, Mapping):
        return None
    message = body.get("message")
    error = body.get("error")
    if isinstance(error, Mapping) and isinstance(error.get("message"), str):
        message = error["message"]
    elif isinstance(error, str):
        message = error
    return str(message)[:_DETAIL_LIMIT] if isinstance(message, str) and message else None


def _status_detail(status: int, response: httpx.Response, default: str | None = None) -> str:
    """`HTTP <status>: <vendor message>` when the body has one."""
    try:
        message = error_message(response.json())
    except ValueError:
        message = None
    message = message or default
    return f"HTTP {status}: {message}" if message else f"HTTP {status}"


def _validation_detail(response: httpx.Response) -> tuple[str | None, str]:
    """Pull the offending field out of a validation error body, whatever shape it takes."""
    try:
        body = response.json()
    except ValueError:
        return None, response.text[:_DETAIL_LIMIT] or "validation error"

    def from_obj(obj: Any) -> tuple[str | None, str | None]:
        if not isinstance(obj, Mapping):
            return None, str(obj) if obj else None
        field: str | None = None
        for key in ("field", "param", "path", "loc"):
            value = obj.get(key)
            if isinstance(value, list):
                field = ".".join(str(p) for p in value)
            elif isinstance(value, str | int):
                field = str(value)
            if field:
                break
        message = obj.get("message") or obj.get("msg")
        return field, str(message) if message else None

    candidates: list[Any] = []
    if isinstance(body, Mapping):
        for key in ("error", "detail", "errors"):
            value = body.get(key)
            if isinstance(value, list):
                candidates.extend(value)
            elif value is not None:
                candidates.append(value)
        candidates.append(body)
    for candidate in candidates:
        field, message = from_obj(candidate)
        if field or message:
            return field, (message or "validation error")[:_DETAIL_LIMIT]
    return None, "validation error"
