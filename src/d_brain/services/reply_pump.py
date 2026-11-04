"""Unsolicited-reply pump: the ONE delivery path for turns no ask() took.

With the reply ledger on (Settings.reply_ledger), every CLI turn ends with a
Stop hook carrying its final message (services/turn_ledger.py). An ``ask()``
claims the turn it is waiting for. Everything else lands here:

* a message steered into a turn that had just closed (it opened its own);
* a turn the CLI started by itself (``<task-notification>``);
* a bound turn whose ``ask()`` is gone — timed out on the chat ceiling, or
  the bot restarted mid-turn — once ``bound_grace`` has passed with no claim.

The pump runs in the bot process and sends through ``send_response`` (the
same Telegram-HTML sanitizing, 4096-char chunking and durable outbox every
normal reply uses). It never looks at the pane and never takes the ask-lock.
Dedup is the claims file, keyed by the CLI ``prompt_id``.

One pump per session runtime dir. The bot runs one for the main session
and, with ``Settings.cron_reply_pump`` on, one for the cron session
(2026-11-02: a cron job that dispatched a background agent closed its turn;
the agent's task-notification turn then answered in the cron session and
nobody took that reply). Each pump only reads and claims inside its own
dir, so the cron runner's ask() and the cron pump split a turn through the
same claims file exactly as the chat ask() and the main pump do.

A final message starting with ``[SILENT]``, or no final text at all, is
"nothing to report": recorded, not sent. Exception: with
``Settings.empty_reply_notice`` on, a turn carrying a steered message that
ended without text gets ``turn_ledger.EMPTY_REPLY``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from d_brain.services import turn_ledger

logger = logging.getLogger(__name__)

POLL_SECONDS = 2.0
# How long a BOUND turn's Stop waits for its own ask() (which polls every
# second) before the pump treats that ask() as gone.
BOUND_GRACE = 20.0
# Unbound turns: a breath, so a steer that raced a closing turn is folded
# into the ledger before we look.
UNBOUND_GRACE = 1.0
# Never deliver anything older than this (a ledger left behind by a long
# outage is history, not a reply).
MAX_AGE = 6 * 3600.0
SINCE_FILE = "turn-ledger.pump-since"

LATE_PREFIX = "↩️ <i>Ответ на предыдущее сообщение:</i>\n\n"
FAILED_NOTICE = (
    "⚠️ Ход прервался ошибкой на стороне Claude — ответа на последнее "
    "сообщение не будет. Повтори его, пожалуйста."
)

Deliver = Callable[[str], Awaitable[None]]


# The marker era's queue of unextracted replies (claude_session). Nothing
# reads it under the ledger; a pump that starts in a dir that still has one
# only reports it (see ReplyPump._report_history).
LEGACY_ORPHANS_FILE = "pending_orphans"


def _read_since(runtime_dir: Path, now: float) -> tuple[float, bool]:
    """When the pump first ran in this dir, and whether that is now. Turns
    that ended before it are history and are never delivered. Persisted, so
    a restart does not move it."""
    path = runtime_dir / SINCE_FILE
    try:
        return float(path.read_text().strip()), False
    except (OSError, ValueError):
        pass
    try:
        path.write_text(f"{now!r}\n")  # repr: reads back bit-exact
    except OSError as exc:
        logger.warning("reply pump: could not persist %s: %s", path, exc)
    return now, True


class ReplyPump:
    def __init__(
        self,
        runtime_dir: Path | str,
        deliver: Deliver,
        *,
        bound_grace: float = BOUND_GRACE,
        unbound_grace: float = UNBOUND_GRACE,
        max_age: float = MAX_AGE,
        clock: Callable[[], float] = time.time,
        empty_notice: bool = False,
        name: str = "main",
        late_prefix: str = LATE_PREFIX,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.name = name
        self._late_prefix = late_prefix
        self._deliver = deliver
        # Settings.empty_reply_notice: a steered turn with no text gets
        # turn_ledger.EMPTY_REPLY (on) or nothing (off, like [SILENT]).
        self._empty_notice = empty_notice
        self._bound_grace = bound_grace
        self._unbound_grace = unbound_grace
        self._max_age = max_age
        self._clock = clock
        self._reader = turn_ledger.LedgerReader(self.runtime_dir)
        self._book = turn_ledger.TurnBook()
        self._claims = turn_ledger.Claims(self.runtime_dir)
        self._since, self._fresh = _read_since(self.runtime_dir, clock())
        self._first_pass = True

    def _due(self, turn: turn_ledger.Turn, now: float) -> bool:
        end = turn.end_ts
        if end is None or end < self._since or now - end > self._max_age:
            return False
        wait = self._bound_grace if turn.bound_rid else self._unbound_grace
        return now - end >= wait

    async def step(self) -> int:
        """One pass. Returns how many messages were handed to Telegram."""
        if self._first_pass:
            # A turn whose submit sits in the rotated generation and whose
            # Stop is in the current file must still be one turn.
            self._first_pass = False
            self._book.extend(turn_ledger.read_rotated(self.runtime_dir))
            self._book.extend(self._reader.read_new())
            if self._fresh:
                self._report_history(self._claims.states())
        else:
            self._book.extend(self._reader.read_new())
        now = self._clock()
        self._book.prune(now - self._max_age)
        states = self._claims.states()
        sent = 0
        for turn in self._book.ordered():
            prev = states.get(turn.prompt_id)
            resume = (
                prev is not None
                and prev.get("state") == "claimed"
                and prev.get("by") == "pump"
            )
            if (
                prev is not None
                and prev.get("state") in turn_ledger.TERMINAL
                and not resume
            ):
                continue
            if not self._due(turn, now):
                continue
            sent += await self._handle(turn, now, resume=resume)
        return sent

    def _report_history(self, states: dict[str, dict]) -> None:
        """First run in this dir: say what stays undelivered on purpose.
        Turns that ended before the pump existed (nobody took them) and the
        marker era's ``pending_orphans`` are history: sending them now, hours
        or days late and out of order, would be spam, not a fix. Logged once,
        never sent; the files are left as they are."""
        unclaimed = [
            t.prompt_id
            for t in self._book.ordered()
            if t.stop is not None
            and (t.end_ts or 0.0) < self._since
            and t.prompt_id not in states
        ]
        try:
            orphans = (self.runtime_dir / LEGACY_ORPHANS_FILE).read_text().split()
        except OSError:
            orphans = []
        if unclaimed or orphans:
            logger.warning(
                "reply pump [%s]: first run in %s — %d earlier unclaimed turn(s) "
                "and %d legacy pending_orphans rid(s) are history, not sent "
                "(turns: %s; orphans: %s)",
                self.name,
                self.runtime_dir,
                len(unclaimed),
                len(orphans),
                ", ".join(pid[:8] for pid in unclaimed) or "-",
                ", ".join(orphans) or "-",
            )

    def _base(self, turn: turn_ledger.Turn, now: float) -> dict:
        return {
            "rid": turn.bound_rid,
            "bound": turn.bound_rid is not None,
            "kinds": turn.kinds,
            "origin": turn.origin,
            "latency": round(now - turn.first_ts, 3),
        }

    async def _handle(self, turn: turn_ledger.Turn, now: float, *, resume: bool) -> int:
        pid = turn.prompt_id
        base = self._base(turn, now)
        if turn.stop is None:
            # Failed or interrupted: nothing to relay. A bound ask() already
            # told the person; an unbound turn that carried a steered message
            # gets an honest notice instead of silence.
            if turn.failure is not None:
                if not self._claims.claim(pid, by="pump", state="failed", **base):
                    return 0
                if turn.bound_rid is None and "steer" in turn.kinds:
                    await self._deliver(FAILED_NOTICE)
                    return 1
                return 0
            self._claims.claim(pid, by="pump", state="interrupted", **base)
            return 0

        if turn.is_maint:
            # A maintenance turn (pipeline, doctor/deploy canary) whose ask()
            # gave up: its output is machine-facing, not a chat reply.
            self._claims.claim(pid, by="pump", state="skipped", **base)
            return 0
        text = turn_ledger.clean_reply(turn_ledger.stop_text(turn))
        if "steer" in turn.kinds:
            # A person's message is in this turn: any text after [SILENT] is
            # their answer. No text at all is an empty turn: the instance's
            # notice if it wants one, else logged below and not sent.
            text = turn_ledger.strip_sentinel(text)
            if not text and self._empty_notice:
                text = turn_ledger.EMPTY_REPLY
        if turn_ledger.is_silent(text):
            self._claims.claim(pid, by="pump", state="sentinel", **base)
            logger.info(
                "reply pump [%s]: turn %s ended silent (%s)",
                self.name,
                pid,
                turn_ledger.SENTINEL,
            )
            return 0
        if not text:
            self._claims.claim(pid, by="pump", state="empty", **base)
            logger.info(
                "reply pump [%s]: turn %s ended without text — not sent",
                self.name,
                pid,
            )
            return 0
        if not resume and not self._claims.claim(
            pid, by="pump", state="claimed", **base
        ):
            return 0
        message = self._late_prefix + text if turn.bound_rid else text
        try:
            await self._deliver(message)
        except Exception:  # noqa: BLE001 — stays "claimed": retried next pass
            logger.exception(
                "reply pump [%s]: delivery of turn %s failed", self.name, pid
            )
            return 0
        self._claims.record(
            prompt_id=pid, by="pump", state="delivered", chars=len(text), **base
        )
        logger.info(
            "reply pump [%s] delivered turn %s (%s, %d chars, %.1fs after it began)",
            self.name,
            pid,
            "late bound" if turn.bound_rid else "unbound",
            len(text),
            base["latency"],
        )
        return 1


async def run(*pumps: ReplyPump, poll_seconds: float = POLL_SECONDS) -> None:
    """Background loop for the bot process: every session's pump, in turn.
    Never raises out; one pump's bad pass does not hold up the others."""
    while True:
        for pump in pumps:
            try:
                await pump.step()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — one bad pass must not stop the pump
                logger.exception("reply pump [%s] pass failed", pump.name)
        await asyncio.sleep(poll_seconds)
