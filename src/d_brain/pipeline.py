"""Headless pipeline entrypoint for the daily cron timer.

Replaces the old `claude --print -p` phases in process.sh: the command runs
through the shared persistent interactive session (subscription billing). The
surrounding shell shim still handles graph rebuild, memory decay, git and
Telegram delivery.

    uv run python -m d_brain.pipeline daily   # prints the HTML report

Exit code 0 on success (report), 1 otherwise.
"""

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def run(cmd: str, proc: Any) -> tuple[str, bool]:
    """Run a pipeline command against a processor. Returns (text, ok)."""
    if cmd == "daily":
        res = proc.process_daily()
    else:
        return f"unknown command: {cmd}", False
    if res.get("report"):
        return res["report"], True
    return res.get("error", "pipeline error"), False


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    from d_brain import logsafe

    logsafe.install()
    # httpx logs every request URL at INFO, and a Telegram Bot API URL
    # carries the bot token — keep it out of the journal / log files.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    from d_brain.config import get_settings
    from d_brain.services.runtime import get_processor

    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    proc = get_processor(get_settings())
    text, ok = run(cmd, proc)
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
