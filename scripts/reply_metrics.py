#!/usr/bin/env python3
"""Share of NOT delivered replies over a time window (reply ledger).

    uv run python scripts/reply_metrics.py                  # ~/.dbrain, last 24h
    uv run python scripts/reply_metrics.py --since 7d
    uv run python scripts/reply_metrics.py --runtime-dir ~/.dbrain --json

Reads <runtime_dir>/turn-ledger.jsonl and turn-ledger.claims.jsonl (written by
the CLI hooks and by the bot). Read-only. See d_brain.services.reply_metrics
for the definitions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from d_brain.services import reply_metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path(os.environ.get("RUNTIME_DIR") or Path.home() / ".dbrain"),
    )
    ap.add_argument("--since", default="24h", help="window length: 90m, 24h, 7d")
    ap.add_argument("--until", default="0", help="window end, as an age (0 = now)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    now = time.time()
    until = now - reply_metrics.parse_window(args.until)
    since = until - reply_metrics.parse_window(args.since)
    if not (args.runtime_dir / "turn-ledger.jsonl").exists():
        print(f"нет журнала ходов в {args.runtime_dir}", file=sys.stderr)
        return 1
    report = reply_metrics.compute(args.runtime_dir, since=since, until=until, now=now)
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(reply_metrics.render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
