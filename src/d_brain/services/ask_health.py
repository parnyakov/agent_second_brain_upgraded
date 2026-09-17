"""Delivery-health ledger: how the last few `session ask()` calls ended.

Written by the BOT process on every chat turn, read by the WATCHDOG — a
different process, in a different systemd unit and slice. A tiny JSON file
under runtime_dir is the entire channel between them: no imports, no socket,
no shared object. That is deliberate. On 2026-08-20 the bot stayed
`active (running)` while every reply it produced was thrown away, and nothing
outside the bot could tell. Anything the reader has to import from the bot is
something a wedged bot can take down with it.

What counts as a failure is narrow on purpose:

  ok             → the turn was delivered. Resets the streak.
  timeout/error/
  busy           → the user got a canned apology instead of an answer. Counts.
  anything else  → neutral (leaves the streak as it was).

`rate_limited` and `logged_out` are neutral rather than failures: the user
does get a truthful message, the pipe itself is intact, and the subscription
limit fires often enough (dozens of times on 2026-08-20) that counting it
would drown the signal we actually want.

B3 fix (Fable audit fix round, 2026-08-22): `busy` (the "pane still busy
with a previous turn after waiting" outcome — see claude_session.ask()'s
busy-wait branch) used to be excluded here on the theory that a busy pane
isn't evidence the delivery PATH is broken. That reasoning missed that
`is_working()`/`is_main_turn_active()` can key on a persistent footer/
spinner signature that never changes for a GENUINELY wedged pane, which
means the "busy" branch is exactly the shape a stuck session takes from
ask()'s point of view too — excluding it from FAILURE_STATUSES silently
disabled delivery_guard's restart backstop for that case. `busy` counts
now; only the user-facing WORDING stays distinct (friendly "still working,
try again shortly" instead of "❌ Ошибка сессии") — see
chat_session._STATUS_MESSAGES.

`busy_active` → neutral (agent-infra-backlog item 22, 2026-09): added
alongside plain `busy` when claude_session.ask()'s busy-wait branch can show
the pane made DEMONSTRABLE PROGRESS across the entire wait (the shared
"is_working_progressing() OR pane.log growth" signal, sampled on every poll
of the wait — see AskResult.busy_seconds / claude_session.py's `saw_progress`
tracking). The B3 reasoning above still holds for a pane that shows NO
progress the whole time — that is still exactly the shape a wedged session
takes, so plain `busy` keeps counting as a failure unchanged. `busy_active`
is deliberately NOT added to FAILURE_STATUSES: a pane that visibly kept
working for the whole busy-wait is, by construction, not the "delivery path
is broken" case this ledger exists to catch — it's a live turn taking a
while, most commonly an unattended multi-level agent cascade (see
long_run.py). It falls through to the same neutral handling as
`rate_limited`/`logged_out` below with no code change needed beyond that
fall-through already working.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

FILENAME = "ask-health.json"

# A turn that ends in any of these produced no answer for the user. `busy`
# added 2026-08-22 (B3 fix) — see the module docstring for why a "friendly"
# status still needs to count toward the delivery_guard restart backstop.
# `busy_active` is DELIBERATELY absent — see the module docstring's
# "busy_active → neutral" note (2026-09, agent-infra-backlog item 22).
FAILURE_STATUSES = frozenset({"timeout", "error", "busy"})
# Only this proves the whole path worked end to end.
SUCCESS_STATUS = "ok"


@dataclass(frozen=True)
class Health:
    """Last known state of the ask path."""

    fail_streak: int = 0
    last_status: str = ""
    last_ts: float = 0.0
    # When the CURRENT streak of failures started. Equal to last_ts on the
    # first failure; lets a reader ask "how fast are these piling up".
    streak_started_ts: float = 0.0

    @property
    def streak_span(self) -> float:
        """Seconds between the first and the last failure of this streak."""
        if self.fail_streak < 2:
            return 0.0
        return max(0.0, self.last_ts - self.streak_started_ts)


def path_for(runtime_dir: Path | str) -> Path:
    return Path(runtime_dir) / FILENAME


def read(runtime_dir: Path | str) -> Health:
    """Current health. A missing or unreadable file reads as healthy —
    absence of evidence must never be reported as a fault."""
    try:
        raw = json.loads(path_for(runtime_dir).read_text())
    except (OSError, ValueError):
        return Health()
    if not isinstance(raw, dict):
        return Health()
    try:
        return Health(
            fail_streak=int(raw.get("fail_streak", 0)),
            last_status=str(raw.get("last_status", "")),
            last_ts=float(raw.get("last_ts", 0.0)),
            streak_started_ts=float(raw.get("streak_started_ts", 0.0)),
        )
    except (TypeError, ValueError):
        return Health()


def next_health(previous: Health, status: str, now: float) -> Health:
    """Pure state transition — the whole policy, testable without a disk."""
    if status == SUCCESS_STATUS:
        return Health(0, status, now, 0.0)
    if status not in FAILURE_STATUSES:
        # Neutral: remember that we saw it, keep the streak untouched.
        return Health(previous.fail_streak, status, now, previous.streak_started_ts)
    streak = previous.fail_streak + 1
    started = previous.streak_started_ts if previous.fail_streak else now
    return Health(streak, status, now, started)


def record(
    runtime_dir: Path | str,
    status: str,
    *,
    clock_fn: Callable[[], float] = time.time,
) -> Health:
    """Fold one ask() outcome into the ledger. Best effort by contract: this
    runs on the reply path of every user message and must never be the reason
    a reply fails to go out."""
    now = clock_fn()
    updated = next_health(read(runtime_dir), status, now)
    target = path_for(runtime_dir)
    payload = json.dumps(
        {
            "fail_streak": updated.fail_streak,
            "last_status": updated.last_status,
            "last_ts": updated.last_ts,
            "streak_started_ts": updated.streak_started_ts,
        }
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: the watchdog polls this file every 15s and must never read
        # a half-written one. The pid in the temp name matters because the
        # failure this ledger exists to detect is TWO bot processes alive at
        # once (the 2026-08-20 orphan) — sharing one temp path would let one
        # writer publish the other's half-written bytes.
        tmp = target.with_suffix(f".json.{os.getpid()}.tmp")
        tmp.write_text(payload)
        os.replace(tmp, target)
    except OSError as exc:
        logger.warning("could not record ask health: %s", exc)
    return updated
