"""Keep credentials out of logs and user-visible error texts.

The Telegram Bot API puts the bot token in request and file-download URLs.
``httpx`` logs request URLs at INFO, and aiohttp errors (e.g. a failed voice
download) embed the full URL in the exception text and traceback. Entry
points that configure logging (bot, watchdog, doctor, pipeline) call
:func:`install` right after ``logging.basicConfig``; error texts sent to a
chat go through :func:`redact`.
"""

from __future__ import annotations

import logging
import re

_PATTERNS = (
    re.compile(r"\d{6,12}:[A-Za-z0-9_-]{30,}"),  # also inside ".../bot<token>/..."
    re.compile(r"\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[opsur]_[A-Za-z0-9]{30,})"),
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"),
)
REDACTED = "<redacted>"


def redact(text: str) -> str:
    for pattern in _PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


class RedactingFilter(logging.Filter):
    """Redacts the rendered message (covers handlers that bypass formatters)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - malformed args: let logging report it
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True


class RedactingFormatter(logging.Formatter):
    """Wraps a handler's formatter; redacts the final text incl. tracebacks."""

    def __init__(self, inner: logging.Formatter | None) -> None:
        super().__init__()
        self.inner = inner or logging.Formatter()

    def format(self, record: logging.LogRecord) -> str:
        return redact(self.inner.format(record))


def _protect(handler: logging.Handler) -> None:
    if not any(isinstance(f, RedactingFilter) for f in handler.filters):
        handler.addFilter(RedactingFilter())
    if not isinstance(handler.formatter, RedactingFormatter):
        handler.setFormatter(RedactingFormatter(handler.formatter))


def install() -> None:
    """Silence per-request HTTP logs; redact secrets on every existing handler."""
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    loggers = [logging.getLogger()] + [
        logger for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]
    for logger in loggers:
        for handler in logger.handlers:
            _protect(handler)
