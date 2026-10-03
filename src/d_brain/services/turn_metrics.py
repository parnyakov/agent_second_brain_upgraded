"""Stuck-turn metric (design 2026-10-03, section 4 step 5).

An append-only journal ``<runtime_dir>/turn-events.jsonl`` — one JSON object
per line, ``{"ts": float, "kind": str, ...}`` — and a one-day summary of it
for ``/work`` and ``doctor``. Kinds written here: ``queued`` and
``dispatched`` (the chat queue). Others (``stale_override``, ``decision``)
are written by the turn-state code; this module only counts them.

Everything is best effort: a metric must never break delivery.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVENTS_FILE = "turn-events.jsonl"
STATE_FILE = "turn-state.json"
MAX_BYTES = 1_000_000
DAY = 86400.0
DELAY_TARGET_S = 5.0


def _events_path(runtime_dir: Path | str) -> Path:
    return Path(runtime_dir) / EVENTS_FILE


def _trim(path: Path) -> None:
    """Keep the newer half of an oversized journal, on a line boundary."""
    data = path.read_bytes()
    tail = data[len(data) // 2 :]
    nl = tail.find(b"\n")
    tail = tail[nl + 1 :] if nl >= 0 else b""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(tail)
    tmp.replace(path)


def append_event(runtime_dir: Path | str | None, kind: str, **fields: Any) -> None:
    """Append one event. Never raises."""
    if runtime_dir is None:
        return
    try:
        path = _events_path(runtime_dir)
        record = {"ts": float(fields.pop("ts", time.time())), "kind": str(kind)}
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        if path.stat().st_size > MAX_BYTES:
            _trim(path)
    except Exception:  # noqa: BLE001 — metrics must never break the caller
        logger.debug("turn-metrics: could not append %s", kind, exc_info=True)


def read_turn_state_raw(runtime_dir: Path | str | None) -> str:
    """``open`` / ``closed`` from the hook file, else ``unknown``."""
    if runtime_dir is None:
        return "unknown"
    try:
        raw = json.loads((Path(runtime_dir) / STATE_FILE).read_text(encoding="utf-8"))
        state = raw.get("state") if isinstance(raw, dict) else None
        return state if state in ("open", "closed") else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _read_events(runtime_dir: Path | str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        text = _events_path(runtime_dir).read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — no file ⇒ no events
        return out
    for line in text.splitlines():
        try:
            rec = json.loads(line)
            if isinstance(rec, dict) and isinstance(rec.get("kind"), str):
                rec["ts"] = float(rec["ts"])
                out.append(rec)
        except Exception:  # noqa: BLE001 — skip a broken line
            continue
    return out


def _delay(rec: dict[str, Any]) -> float | None:
    try:
        return float(rec.get("delay_s"))
    except (TypeError, ValueError):
        return None


def summary(
    runtime_dir: Path | str, *, now: float | None = None, window_s: float = DAY
) -> dict[str, Any]:
    moment = time.time() if now is None else float(now)
    since = moment - window_s
    events = [e for e in _read_events(runtime_dir) if since <= e["ts"] <= moment]
    queued = [e for e in events if e["kind"] == "queued"]
    dispatched = [e for e in events if e["kind"] == "dispatched"]
    decisions = [e for e in events if e["kind"] == "decision"]
    delays = [d for d in (_delay(e) for e in dispatched) if d is not None]
    closed_delays = [
        d
        for d in (
            _delay(e) for e in dispatched if e.get("turn_state_at_queue") == "closed"
        )
        if d is not None
    ]
    screen = sum(1 for e in decisions if e.get("source") == "screen")
    return {
        "window_s": window_s,
        "queued_total": len(queued),
        "parked_while_closed": sum(
            1 for e in queued if e.get("turn_state") == "closed"
        ),
        "max_queue_delay_s": max(delays) if delays else None,
        "max_queue_delay_closed_s": max(closed_delays) if closed_delays else None,
        "stale_overrides": sum(1 for e in events if e["kind"] == "stale_override"),
        "screen_decisions_share": (screen / len(decisions)) if decisions else None,
    }


def needs_attention(d: dict[str, Any]) -> bool:
    closed = d.get("max_queue_delay_closed_s")
    return bool(d.get("parked_while_closed")) or (
        closed is not None and closed > DELAY_TARGET_S
    )


def _sec(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} с"


def format_summary(d: dict[str, Any]) -> str:
    """Short Russian summary, Telegram HTML (b/i/code only)."""
    share = d.get("screen_decisions_share")
    share_txt = "—" if share is None else f"{share * 100:.0f}%"
    lines = [
        "<b>🧭 Застревания за сутки</b>",
        f"• в очередь: {d.get('queued_total', 0)}, "
        f"из них при закрытом ходе: <b>{d.get('parked_while_closed', 0)}</b>",
        f"• макс. ожидание в очереди: {_sec(d.get('max_queue_delay_s'))}, "
        f"при закрытом ходе: {_sec(d.get('max_queue_delay_closed_s'))}",
        f"• сброшено зависших «идёт»: {d.get('stale_overrides', 0)}",
        f"• решений по экрану: {share_txt}",
    ]
    return "\n".join(lines)
