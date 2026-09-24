"""structlog JSON logging with a last-line-of-defence secret scrubber."""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

REDACTED = "[REDACTED]"
_SENSITIVE_MARKERS = ("key", "secret", "token", "password", "authorization", "cookie")


def _is_sensitive(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _SENSITIVE_MARKERS)


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: REDACTED if _is_sensitive(str(k)) else _scrub(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_scrub(v) for v in value)
    return value


def redact_secrets(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Replace values of sensitive-looking keys (recursively) with a placeholder."""
    for k in list(event_dict):
        event_dict[k] = REDACTED if _is_sensitive(k) else _scrub(event_dict[k])
    return event_dict


def _stderr_logger(*_args: Any) -> structlog.PrintLogger:
    # Resolve sys.stderr per logger so redirected/replaced streams are honoured.
    return structlog.PrintLogger(file=sys.stderr)


def configure_logging(level: str = "INFO") -> None:
    log_level = logging.getLevelNamesMapping()[level.upper()]
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=log_level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_secrets,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=_stderr_logger,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
