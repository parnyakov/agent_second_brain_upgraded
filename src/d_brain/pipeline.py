"""Headless pipeline entrypoint for the daily cron timer.

Replaces the old `claude --print -p` phases in process.sh: the command runs
through the shared persistent interactive session (subscription billing). The
surrounding shell shim still handles graph rebuild, memory decay, git and
Telegram delivery.

    uv run python -m d_brain.pipeline daily   # prints the HTML report
    uv run python -m d_brain.pipeline daily 2026-10-25   # re-run a missed day

Exit code 0 on success (report), 1 otherwise.
"""

import logging
import sys
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


def run(cmd: str, proc: Any, day: date | None = None) -> tuple[str, bool]:
    """Run a pipeline command against a processor. Returns (text, ok)."""
    if cmd == "daily":
        res = proc.process_daily() if day is None else proc.process_daily(day)
    else:
        return f"unknown command: {cmd}", False
    if res.get("report"):
        return res["report"], True
    if "report" in res and "error" not in res:
        # The turn finished with nothing to say ([SILENT] or no text): a
        # success with no output, so process.sh sends nothing.
        return "", True
    return res.get("error", "pipeline error"), False


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    from d_brain import logsafe
    from d_brain.log_redact import quiet_http_logging

    logsafe.install()
    # A Telegram Bot API URL carries the bot token — keep it out of logs.
    quiet_http_logging()
    from d_brain.config import get_settings
    from d_brain.services.runtime import get_processor

    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    day = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None
    proc = get_processor(get_settings())
    text, ok = run(cmd, proc, day)
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
