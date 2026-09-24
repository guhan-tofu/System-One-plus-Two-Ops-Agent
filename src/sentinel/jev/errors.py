"""Jev provider errors. Messages never include credentials or request state."""

from __future__ import annotations


class JevError(Exception):
    """Base class for all Jev provider failures."""


class JevConfigError(JevError):
    """Provider is misconfigured (e.g. missing API key)."""


class JevAuthError(JevError):
    """401/403: bad or missing credentials. Not retried."""


class JevValidationError(JevError):
    """422: the provider rejected our request. Not retried."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        self.field = field
        super().__init__(f"{message} (field: {field})" if field else message)


class JevRetryableError(JevError):
    """A transient failure we retry with backoff (429, 529)."""

    def __init__(self, message: str, *, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(message)


class JevRateLimitError(JevRetryableError):
    """429 Too Many Requests."""


class JevOverloadedError(JevRetryableError):
    """529 Overloaded."""


class JevHTTPError(JevError):
    """Any other non-success status. Not retried."""

    def __init__(self, message: str, *, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(message)


class JevTimeoutError(JevError):
    """Connect or read timeout."""


class JevTransportError(JevError):
    """Network-level failure other than a timeout."""


class JevResponseError(JevError):
    """The response was not in a shape we can trust. Never guessed around."""
