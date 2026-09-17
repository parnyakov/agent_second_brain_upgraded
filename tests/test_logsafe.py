# ruff: noqa: E501
"""Bot tokens never reach logs (the Bot API puts them in request URLs)."""

import logging

from d_brain import logsafe


def test_httpx_request_url_with_token_is_not_logged(caplog):
    token = "1234567" + ":" + "Ab" * 20
    root = logging.getLogger()
    handler = logging.StreamHandler()
    root.addHandler(handler)
    try:
        logsafe.install()
        assert logging.getLogger("httpx").level == logging.WARNING
        assert any(isinstance(f, logsafe.RedactingFilter) for f in handler.filters)
        record = logging.LogRecord(
            "d_brain.watchdog", logging.INFO, __file__, 1,
            "HTTP Request: POST https://api.telegram.org/bot%s/sendMessage", (token,), None,
        )
        for f in handler.filters:
            f.filter(record)
        assert token not in record.getMessage()
        assert logsafe.REDACTED in record.getMessage()
    finally:
        root.removeHandler(handler)


def test_redact_keeps_ordinary_text():
    assert logsafe.redact("ask took 12.5s, 3 retries") == "ask took 12.5s, 3 retries"


def test_entry_points_install_the_filter():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "d_brain"
    for rel in ("__main__.py", "pipeline.py", "services/doctor.py", "services/watchdog.py"):
        assert "logsafe.install()" in (src / rel).read_text(), rel


def test_real_httpx_request_log_has_no_token():
    """Reproduces the install smoke finding: doctor/watchdog configure INFO
    logging and post to the Bot API through httpx."""
    import subprocess
    import sys

    token = "7654321" + ":" + "Zz" * 20
    code = f"""
import logging, httpx
logging.basicConfig(level=logging.INFO)
from d_brain import logsafe
logsafe.install()
transport = httpx.MockTransport(lambda request: httpx.Response(200, json={{"ok": True}}))
with httpx.Client(transport=transport) as client:
    client.post("https://api.telegram.org/bot{token}/sendMessage", data={{"text": "x"}})
logging.getLogger("d_brain.watchdog").info("alert url https://api.telegram.org/bot{token}/sendMessage")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert token not in out.stdout + out.stderr
    assert "alert url" in out.stderr and logsafe.REDACTED in out.stderr


def test_without_install_the_leak_is_real():
    """Control: the same request logs the token when the filter is absent."""
    import subprocess
    import sys

    token = "7654321" + ":" + "Yy" * 20
    code = f"""
import logging, httpx
logging.basicConfig(level=logging.INFO)
transport = httpx.MockTransport(lambda request: httpx.Response(200, json={{"ok": True}}))
with httpx.Client(transport=transport) as client:
    client.post("https://api.telegram.org/bot{token}/sendMessage")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert token in out.stderr


def test_traceback_with_token_url_is_redacted():
    """A failed file download (aiohttp ClientResponseError) carries the full
    file URL with the token in the exception text and traceback."""
    import subprocess
    import sys

    token = "7000001" + ":" + "Qq" * 20
    code = f"""
import logging
logging.basicConfig(level=logging.INFO)
from d_brain import logsafe
logsafe.install()
log = logging.getLogger("d_brain.bot.handlers.chat")
try:
    raise RuntimeError("500, url='https://api.telegram.org/file/bot{token}/voice/file_1.oga'")
except RuntimeError:
    log.exception("Error processing voice message")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert token not in out.stdout + out.stderr
    assert "Error processing voice message" in out.stderr and logsafe.REDACTED in out.stderr


def test_malformed_log_call_does_not_raise_from_the_filter():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "%s %s", ("only-one",), None)
    assert logsafe.RedactingFilter().filter(record) is True


def test_chat_error_replies_are_redacted():
    from pathlib import Path

    chat = (Path(__file__).resolve().parents[1] / "src/d_brain/bot/handlers/chat.py").read_text()
    assert "html.escape(str(e)" not in chat
    assert chat.count("logsafe.redact(str(e))") == 3
