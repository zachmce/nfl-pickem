"""structlog JSON logging to stdout, with redaction of sensitive keys.

Call ``configure_logging()`` once at process startup — the bot does this in
``app.bot.client.main`` and the API can do it in its lifespan.
"""

import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "token",
        "secret",
        "authorization",
        "cookie",
        "set-cookie",
        "database_url",
        "secret_key",
    }
)

_SENSITIVE_PATTERN = re.compile(
    r"(" + "|".join(re.escape(k) for k in _SENSITIVE_KEYS) + r")",
    re.IGNORECASE,
)


def _redact_processor(
    logger: Any,  # noqa: ARG001 — structlog processor signature
    method: str,  # noqa: ARG001 — structlog processor signature
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Replace values of sensitive keys with '[REDACTED]' before rendering."""
    for key in list(event_dict.keys()):
        if _SENSITIVE_PATTERN.search(key):
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON output to stdout. Call once at startup."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            _redact_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
