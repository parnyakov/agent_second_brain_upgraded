import logging

from d_brain.log_redact import TokenRedactFilter, redact

URL = "https://api.telegram.org/bot123456:AA-bc_DEF/sendMessage"


def test_redact_replaces_token():
    assert redact(URL) == "https://api.telegram.org/bot***/sendMessage"


def test_filter_redacts_args_and_traceback():
    try:
        raise RuntimeError(URL)
    except RuntimeError:
        import sys

        record = logging.LogRecord(
            "httpx", logging.WARNING, "", 0, "HTTP Request: %s", (URL,), sys.exc_info()
        )
    TokenRedactFilter().filter(record)
    text = logging.Formatter().format(record)
    assert "123456:" not in text
    assert "bot***" in text
