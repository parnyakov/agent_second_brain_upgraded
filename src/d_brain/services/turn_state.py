"""Turn state written by the Claude CLI hooks (primary busy/idle signal).

The hook script writes ``<runtime_dir>/turn-state.json`` atomically; this
module only reads/validates it, lets the bot force it ``closed`` (interrupt,
restart, stale override) and keeps a small append-only event journal
(``turn-events.jsonl``) for the turn-source metric. The pane screen stays the
fallback whenever the file is missing or not trustworthy.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from d_brain.services.tmux_parse import is_agents_wait_only, is_main_turn_active

logger = logging.getLogger(__name__)

STATE_FILE = "turn-state.json"
EVENTS_FILE = "turn-events.jsonl"
FUTURE_TOLERANCE = 5.0
SUSPICIOUS_OPEN_SECONDS = 600.0
SUSPICIOUS_QUIET_SECONDS = 300.0
_STATES = ("open", "closed")


@dataclass(frozen=True)
class TurnState:
    state: str
    event: str
    ts: float
    last_event_ts: float
    session_id: str
    cli_pid: int
    background_tasks: int = 0
    origin: str | None = None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "event": self.event,
            "ts": self.ts,
            "last_event_ts": self.last_event_ts,
            "session_id": self.session_id,
            "cli_pid": self.cli_pid,
            "background_tasks": self.background_tasks,
            "origin": self.origin,
        }


def read(runtime_dir: Path) -> TurnState | None:
    """Parse the state file; None if missing or malformed."""
    try:
        raw = json.loads((Path(runtime_dir) / STATE_FILE).read_text())
        if not isinstance(raw, dict):
            return None
        ts = float(raw["ts"])
        return TurnState(
            state=str(raw["state"]),
            event=str(raw.get("event") or ""),
            ts=ts,
            last_event_ts=float(raw.get("last_event_ts") or ts),
            session_id=str(raw.get("session_id") or ""),
            cli_pid=int(raw.get("cli_pid") or 0),
            background_tasks=int(raw.get("background_tasks") or 0),
            origin=raw.get("origin"),
        )
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _is_claude_proc(pid: int) -> bool:
    """True if pid is alive and looks like the claude CLI.

    Without /proc, a successful kill(pid, 0) is taken as alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except OSError:
        return False
    proc = Path("/proc") / str(pid)
    if not Path("/proc/self").exists():
        return True
    try:
        comm = (proc / "comm").read_text(errors="replace")
    except OSError:
        comm = ""
    try:
        cmdline = (proc / "cmdline").read_bytes().decode(errors="replace")
    except OSError:
        cmdline = ""
    if not comm and not cmdline:
        return False
    return "claude" in comm.lower() or "claude" in cmdline.lower()


def valid(
    st: TurnState | None,
    *,
    session_id: str | None,
    now: float,
    pid_alive: Callable[[int], bool] | None = None,
) -> bool:
    if st is None or st.state not in _STATES:
        return False
    if not session_id or st.session_id != session_id:
        return False
    if st.ts > now + FUTURE_TOLERANCE:
        return False
    if st.cli_pid <= 0:
        return False
    check = pid_alive if pid_alive is not None else _is_claude_proc
    try:
        return bool(check(st.cli_pid))
    except Exception:  # noqa: BLE001 — unknown liveness → not trusted
        return False


def suspicious_open(st: TurnState, now: float) -> bool:
    return (
        st.state == "open"
        and now - st.ts > SUSPICIOUS_OPEN_SECONDS
        and now - st.last_event_ts > SUSPICIOUS_QUIET_SECONDS
    )


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def close(runtime_dir: Path, *, event: str, reason: str = "") -> None:
    """Force the state to closed (best-effort), keeping session_id/cli_pid."""
    try:
        runtime_dir = Path(runtime_dir)
        now = time.time()
        prev = read(runtime_dir)
        data = (
            prev.to_dict()
            if prev is not None
            else {
                "session_id": "",
                "cli_pid": 0,
                "background_tasks": 0,
                "origin": None,
            }
        )
        data.update(state="closed", event=event, ts=now, last_event_ts=now)
        if reason:
            data["reason"] = reason
        _atomic_write(runtime_dir / STATE_FILE, json.dumps(data))
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.debug("turn-state: close(%s) failed: %s", event, exc)


def append_event(runtime_dir: Path, kind: str, **fields) -> None:
    """Append one line to the shared event journal (see turn_metrics)."""
    from d_brain.services import turn_metrics

    turn_metrics.append_event(runtime_dir, kind, **fields)


def screen_busy(cap: str) -> bool:
    """The legacy screen predicate (fallback path)."""
    return is_main_turn_active(cap) and not is_agents_wait_only(cap)


# Per-process decision tracking (log/journal only on change).
_last_decision: dict[str, tuple[str, bool]] = {}
_last_disagree: dict[str, float] = {}
DISAGREE_LOG_INTERVAL = 60.0


def _record_decision(runtime_dir: Path, source: str, is_open: bool) -> None:
    key = str(runtime_dir)
    if _last_decision.get(key) == (source, is_open):
        return
    prev_source = _last_decision.get(key, (None, None))[0]
    _last_decision[key] = (source, is_open)
    if prev_source != source:
        logger.info("turn-source=%s", source)
    append_event(runtime_dir, "decision", source=source, open=is_open)


def turn_open(
    runtime_dir: Path,
    cap: str,
    session_id: str | None,
    capture_again: Callable[[], str] | None = None,
    *,
    now: float | None = None,
    pid_alive: Callable[[int], bool] | None = None,
) -> bool:
    """Is the main turn open? Hooks file first, screen as fallback."""
    runtime_dir = Path(runtime_dir)
    now = time.time() if now is None else now
    st = read(runtime_dir)
    if not valid(st, session_id=session_id, now=now, pid_alive=pid_alive):
        result = screen_busy(cap)
        _record_decision(runtime_dir, "screen", result)
        return result
    assert st is not None
    if st.state == "open":
        if suspicious_open(st, now) and not screen_busy(cap) and capture_again:
            try:
                cap2 = capture_again()
            except Exception:  # noqa: BLE001 — cannot confirm → keep open
                cap2 = None
            if cap2 is not None and not screen_busy(cap2):
                close(runtime_dir, event="stale_override", reason="screen idle twice")
                logger.warning("turn-state: stale open overridden")
                append_event(
                    runtime_dir,
                    "stale_override",
                    session_id=st.session_id,
                    open_since=st.ts,
                    last_event_ts=st.last_event_ts,
                )
                _record_decision(runtime_dir, "hooks", False)
                return False
        _record_decision(runtime_dir, "hooks", True)
        return True
    if screen_busy(cap):
        key = str(runtime_dir)
        if now - _last_disagree.get(key, float("-inf")) >= DISAGREE_LOG_INTERVAL:
            _last_disagree[key] = now
            logger.info("turn-state: screen disagrees (hooks=closed, screen=busy)")
    _record_decision(runtime_dir, "hooks", False)
    return False
