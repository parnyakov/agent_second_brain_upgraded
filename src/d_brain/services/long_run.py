"""Tracks unattended long-running turns for USER VISIBILITY only.

Background: during a long autonomous agent
cascade, the pane's main turn can stay visibly active (progressing) for far
longer than a normal chat turn, while nobody's `ask()` holds the process-wide
ask-lock — the interactive session is being driven by a background
task-notification, not by something `ask()` itself sent. From `ask()`'s point
of view that is indistinguishable from a leftover turn it must wait out
before typing, which is exactly right; what's missing is a way for the
WATCHDOG (which polls the pane every tick regardless) to notice "this has
been going for a while" and give the user an honest heads-up, and for `ask()`
itself to short-circuit its own busy-wait once that fact is already known
instead of re-discovering it the slow way every single time a message comes
in during the cascade.

Modeled structurally on `ask_health.py`: a pure transition function
(`next_state`) plus a small atomic JSON file (`long-run.json`) as the ONLY
cross-process channel between the watchdog (writer) and the bot (reader, via
`is_active`). A missing or unreadable file reads as "nothing happening" —
absence of evidence must never be reported as a fault, same contract as
`ask_health.read`.

Clock: ALL timestamps here are wall-clock (`time.time`), never
`ClaudeSession`'s internal `time.monotonic` clock — this file crosses a
process boundary (watchdog writes, bot reads), and monotonic clocks are not
comparable across processes. Every function that needs "now" takes it as an
explicit parameter so callers control the clock and tests stay deterministic
without patching global state.

Staleness is mandatory, not an optimization: if the process that writes this
marker dies (or simply drifts to a state where it stops ticking), the file
must degrade to "no long run in progress" on its own — never latch a fast
path or a scary notice on forever. See `is_active`'s `stale_after` guard.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

FILENAME = "long-run.json"


def path_for(runtime_dir: Path | str) -> Path:
    return Path(runtime_dir) / FILENAME


@dataclass(frozen=True)
class LongRun:
    """Current tracked state of an unattended long-running turn.

    ``since == 0.0`` is the sentinel for "no long run in progress" — the
    empty/default value, same convention as ``ask_health.Health``'s zeroed
    fields.
    """

    since: float = 0.0
    updated_ts: float = 0.0
    alerted: bool = False


def read(runtime_dir: Path | str) -> LongRun:
    """Current state. A missing/unreadable/malformed file reads as "nothing
    happening" — absence of evidence must never be reported as a fault."""
    try:
        raw = json.loads(path_for(runtime_dir).read_text())
    except (OSError, ValueError):
        return LongRun()
    if not isinstance(raw, dict):
        return LongRun()
    try:
        return LongRun(
            since=float(raw.get("since", 0.0)),
            updated_ts=float(raw.get("updated_ts", 0.0)),
            alerted=bool(raw.get("alerted", False)),
        )
    except (TypeError, ValueError):
        return LongRun()


def write(runtime_dir: Path | str, state: LongRun) -> bool:
    """Persist ``state`` atomically (tmp-then-rename, pid-suffixed temp name
    — same pattern as ``ask_health.record`` / ``ClaudeSession._atomic_write``).
    Best effort: a write failure is logged, never raised — this ledger must
    never be the reason a watchdog tick or a bot reply fails."""
    target = path_for(runtime_dir)
    payload = json.dumps(
        {
            "since": state.since,
            "updated_ts": state.updated_ts,
            "alerted": state.alerted,
        }
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(f".json.{os.getpid()}.tmp")
        tmp.write_text(payload)
        os.replace(tmp, target)
        return True
    except OSError as exc:
        logger.warning("could not write %s: %s", target, exc)
        return False


def clear(runtime_dir: Path | str) -> None:
    """Remove the marker entirely (e.g. on a clean shutdown)."""
    try:
        path_for(runtime_dir).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not clear %s: %s", path_for(runtime_dir), exc)


def next_state(
    previous: LongRun,
    *,
    main_turn_active: bool,
    attended: bool,
    now: float,
    alert_after: float,
) -> tuple[LongRun, str]:
    """Pure state transition — the whole policy, testable without a disk.

    Returns ``(new_state, event)`` where ``event`` is one of:

      "none"    — nothing worth telling the caller about this tick.
      "started" — an unattended long turn just began.
      "alert"   — the run crossed ``alert_after`` seconds and has not been
                  alerted about yet this run (latched via ``alerted``, so
                  this fires at most once per run).
      "paused"  — the turn is STILL running, but a human's ``ask()`` took
                  the lock, so it is no longer unattended.
      "ended"   — the turn itself stopped: the pane is no longer busy.

    An ATTENDED active turn (the caller's own ``ask()`` holds the lock) never
    STARTS a tracked run — that is an ordinary chat turn, not the
    unattended-cascade shape this module exists to catch.

    The ``paused`` / ``ended`` split exists because of a real false alarm:
    both used to fall into one "nothing unattended remains" branch and both
    produced ``ended``, on which the watchdog sends «✅ сессия снова
    свободна» — so writing to the bot during a long run made the pane's own
    busy-ness announce itself as freedom. The message was wrong about the
    world at the instant it was sent; renaming it would not have helped.

    Pausing deliberately KEEPS ``since`` and ``alerted``:

    * ``since``, because zeroing it would restart the clock when the human's
      turn ends and send a second «🛠 идёт длинная задача ~1 мин» about the
      very same task — moving a false message one step later instead of
      removing it;
    * ``alerted``, because it is the latch that guarantees ✅ only ever
      follows a 🛠 (blind-review fix F2, 2026-09).

    The marker therefore stays ACTIVE across a pause, which is correct and
    load-bearing: the pane really is busy, and ``claude_session.ask()``'s
    busy-active fast path reads exactly that — and re-confirms it against
    the live pane before trusting it, so nothing here can make it call a
    free session busy.
    """
    if main_turn_active and not attended:
        if previous.since == 0.0:
            return LongRun(since=now, updated_ts=now, alerted=False), "started"
        updated = LongRun(
            since=previous.since, updated_ts=now, alerted=previous.alerted
        )
        if (
            alert_after > 0
            and now - previous.since >= alert_after
            and not previous.alerted
        ):
            return (
                LongRun(since=previous.since, updated_ts=now, alerted=True),
                "alert",
            )
        return updated, "none"
    if main_turn_active:
        # Still running, just no longer unattended — a human took the lock.
        if previous.since != 0.0:
            return (
                LongRun(
                    since=previous.since, updated_ts=now, alerted=previous.alerted
                ),
                "paused",
            )
        return LongRun(), "none"
    # The turn itself is over: the pane is not busy. THIS is the only
    # transition that may claim the session is free again.
    if previous.since != 0.0:
        return LongRun(), "ended"
    return LongRun(), "none"


def is_active(
    runtime_dir: Path | str, *, now: float, stale_after: float
) -> tuple[bool, float]:
    """``(True, elapsed_seconds)`` iff a long run is tracked AND its marker
    was updated recently enough to trust (``stale_after`` guard) — otherwise
    ``(False, 0.0)``.

    The stale guard is mandatory: if the watchdog process that maintains
    this marker dies, the file must degrade to "no long run in progress" on
    its own rather than latch a stale "active" reading forever.
    """
    state = read(runtime_dir)
    if state.since <= 0.0:
        return False, 0.0
    if now - state.updated_ts > stale_after:
        return False, 0.0
    return True, max(0.0, now - state.since)
