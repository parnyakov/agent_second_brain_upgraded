"""Keep the Telegram bot token out of log output.

A Bot API URL carries the token (``https://api.telegram.org/bot<id>:<secret>/...``).
httpx logs every request URL at INFO, and an httpx error can put the same URL
into a WARNING or a traceback, so raising the level alone is not enough.
"""

import logging
import re

_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]+")


def redact(text: str) -> str:
    return _TOKEN_RE.sub("bot***", text)


class TokenRedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        clean = redact(message)
        if clean != message:
            record.msg, record.args = clean, ()
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def quiet_http_logging() -> None:
    """Call right after ``logging.basicConfig``."""
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for handler in logging.getLogger().handlers:
        handler.addFilter(TokenRedactFilter())
