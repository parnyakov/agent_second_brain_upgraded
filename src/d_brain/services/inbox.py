"""Durable inbox: an incoming message is on disk BEFORE any work starts.

Step 2 of the reliability plan (``thoughts/projects/agent-infra-backlog.md``
item 33). Step 1 made sure a reply that was BORN could not be lost on the
way out. This one makes sure a message that ARRIVED cannot be lost on the
way in.

The pain, in the owner's words: he restarts the bot several times a day. Until
now an incoming update lived only in RAM — ``handle_chat_text`` &co read it
straight off the wire. A restart, a crash or a deploy landing between
"message received" and "reply sent" left nothing at all: the person wrote,
no answer came, and only he could notice and repeat himself.

Two holes, one fix
------------------

**The obvious one** — nothing on disk. Fixed by inverting the order: the
update is written to ``runtime_dir/inbox/`` before the dispatcher sees it,
and removed only once a handler has run to the end.

**The quiet one — the Telegram offset.** ``aiogram``'s
``Dispatcher._listen_updates`` yields an update, the polling loop spawns
``asyncio.create_task(...)`` for it and comes straight back to the
generator, which then does ``get_updates.offset = update.update_id + 1``.
The NEXT ``getUpdates`` call confirms that whole batch at Telegram — while
the handler tasks may not have run a single line yet. Die in that window
and Telegram will never resend: the update is gone from both sides. This is
the same hole the reference agent closes by persisting the last processed
update id; we close it one step earlier, by making the batch durable before
``_listen_updates`` is allowed to continue.

Where the single acceptance point is
------------------------------------

``accept_middleware`` is registered on ``bot.session.middleware`` — the
client-session request middleware, i.e. around the ``getUpdates`` call
itself. It is the ONE place every incoming update passes through, before
routers, before filters, before the auth middleware: text, voice, photos,
documents, albums, commands and callbacks all land there identically, and
no handler has to remember to enrol. That is what "одна точка приёма, а не
две" means here.

It is deliberately NOT a ``dp.update`` middleware. That one runs inside the
spawned task — after the offset has already moved on.

The one thing it does not see is an update that never came through
``getUpdates``: a webhook deployment would quietly reduce this module to a
duplicate gate over an empty queue. This install polls, and switching to a
webhook would need a second acceptance point here.

Why acceptance does not also drop duplicates
--------------------------------------------

Tempting, and wrong. ``_listen_updates`` advances the offset FROM THE
UPDATES IT YIELDS; hand it a filtered list and the dropped tail is never
confirmed, so Telegram redelivers it, so it is dropped again — forever, at
polling speed. So acceptance only RECORDS. The duplicate gate is
``handled_middleware`` on ``dp.update``, which runs after aiogram has
already counted the update and can safely refuse to process it.

Storage — the outbox's shape on purpose
---------------------------------------

Plain JSON files, atomic ``os.replace``, owner-only, no database::

    runtime_dir/inbox/
        <update_id>.json    one accepted update, zero-padded id, so plain
                            filename order IS arrival order
        receipts.json       capped ring of ids already handled
        stale/<id>.json     accepted, never answered, too old to replay

The id is the Telegram ``update_id``: monotonic per bot, so it orders
correctly against entries left by a PREVIOUS process, and it doubles as the
dedup key — the inbox's answer to the outbox's receipts.

Guarantees, and what they cost
------------------------------

* **Survives a restart.** ``replay()`` runs at boot, before polling, and
  feeds every accepted-but-unanswered update back through the dispatcher in
  id order, one at a time. Order inside a chat is preserved because
  ``update_id`` order is arrival order and nothing runs concurrently with
  the backlog.
* **No double answer.** A receipt is written BEFORE the entry is removed,
  exactly as in the outbox, so a crash in that window leaves the entry
  behind and the receipt makes the next boot drop it instead of answering
  twice. Within a process, ``claim()`` also refuses an update that is
  already in flight, which covers Telegram redelivering a batch we have not
  finished yet.
* **Nothing quietly piles up, and no boot inherits unbounded work.** An
  entry past the replay threshold, or past one failed replay attempt, is
  retired to ``stale/`` rather than deleted, and the whole boot-time replay
  is bounded in wall time. ``/work`` reports what is still waiting.
* **Degrades to today.** A queue that cannot be WRITTEN is logged and the
  update is processed anyway: durability is an upgrade over answering,
  never a precondition for it (the same call the outbox's blind review
  made, B2).

What it costs, stated plainly: the chat handlers are not idempotent. The
librarian safety net writes the daily entry (and saves the attachment)
BEFORE the turn starts, so a message replayed after a crash that happened
later in the turn leaves a duplicate daily line, and for media a second
copy of the file. A duplicated line in a daily note is a much smaller loss
than a message that never got an answer, so this step takes that trade
knowingly; the proper fix is a ``msg_id`` key inside ``VaultStorage``,
which is its own change.

Not here on purpose: cross-process locking (only the bot writes), a
per-chat worker, retries, priorities. One module, two middlewares, one
boot-time function.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiogram.methods import GetUpdates

from d_brain.services.outbox import write_json_atomic

logger = logging.getLogger(__name__)

DIRNAME = "inbox"
STALE_DIRNAME = "stale"
RECEIPTS_FILENAME = "receipts.json"

# How many handled ids the receipt ring keeps. Bigger than the outbox's 200
# on purpose (blind review 9): acceptance runs BEFORE the auth middleware —
# it has to, it is the transport — so any Telegram user can push ids through
# this ring, and flushing it is what would let a real duplicate through.
# A thousand 19-digit ids is a ~20 KB file and ten full getUpdates batches.
RECEIPTS_KEPT = 1000

# Default ceiling on how old an accepted-but-unanswered message may be and
# still be replayed at boot. Mirrors the reference agent's hour. Overridden
# by ``Settings.inbox_replay_max_age``.
DEFAULT_MAX_AGE = 3600.0

# How many times an entry may be fed to the dispatcher at boot before it is
# retired unanswered. One (blind review 3): the message that wedged the
# engine is exactly the message the next boot would feed to the same wedged
# engine, and while that turn stalls, polling has not started — no /work, no
# /new, no /stop. A second shot at an answer is not worth a deaf bot, and
# the attempt is recorded BEFORE the work so a hard kill still counts.
REPLAY_MAX_ATTEMPTS = 1

# Wall-clock ceiling on the whole boot-time replay. ``max_age`` bounds how
# OLD the backlog may be, never how long answering it takes: one replayed
# turn can legitimately ride ``chat_turn_timeout`` (1500s by default), and
# every second of it is a second the bot is not polling. Past this, the
# backlog is abandoned to its attempt counter and the channel comes back.
REPLAY_BUDGET = 300.0

# Update keys we know how to describe in a log line. Order is arbitrary —
# an update carries exactly one of them.
_EVENT_KEYS = (
    "message",
    "edited_message",
    "channel_post",
    "edited_channel_post",
    "business_message",
    "callback_query",
    "inline_query",
    "my_chat_member",
    "chat_member",
)


@dataclass(frozen=True)
class Entry:
    """One accepted update, exactly as Telegram sent it.

    ``kind`` and ``chat_id`` are denormalized copies pulled out of the raw
    payload for log lines only — nothing reads them back to make a
    decision, so a Telegram schema change can never break replay.
    """

    id: str
    update_id: int
    kind: str
    chat_id: int | None
    accepted_at: float
    update: dict[str, Any]
    attempts: int = 0


@dataclass(frozen=True)
class Stats:
    """What ``/work`` prints: how much arrived and never got an answer.

    ``pending`` deliberately EXCLUDES what this process is handling right
    now. ``/work`` is itself an update, and it reads these numbers from
    inside its own claim window, so counting in-flight entries made every
    single ``/work`` report "принято, без ответа: 1" on a completely idle
    bot — a permanent false alarm in the one block whose job is to raise a
    real one (blind review 2). What is left is what nobody is working on:
    a backlog, i.e. something actually stuck.
    """

    pending: int = 0
    in_flight: int = 0
    stale: int = 0
    oldest_age: float = 0.0


def entry_id(update_id: int) -> str:
    """Filename-sortable id for an update. 19 digits is ``time_ns`` width —
    the same padding the outbox uses, so both queues read alike."""
    return f"{int(update_id):019d}"


def describe(update: dict[str, Any]) -> tuple[str, int | None]:
    """``(kind, chat_id)`` for a raw update — best effort, never raises."""
    for key in _EVENT_KEYS:
        event = update.get(key)
        if not isinstance(event, dict):
            continue
        chat = event.get("chat")
        if not isinstance(chat, dict):
            # callback_query carries its chat one level down
            message = event.get("message")
            chat = message.get("chat") if isinstance(message, dict) else None
        chat_id = chat.get("id") if isinstance(chat, dict) else None
        return key, int(chat_id) if isinstance(chat_id, int) else None
    return "unknown", None


def _entry_from_dict(raw: dict[str, Any]) -> Entry:
    update = raw["update"]
    if not isinstance(update, dict):
        raise TypeError("entry carries no update payload")
    return Entry(
        id=str(raw["id"]),
        update_id=int(raw["update_id"]),
        kind=str(raw.get("kind", "unknown")),
        chat_id=raw.get("chat_id") if isinstance(raw.get("chat_id"), int) else None,
        accepted_at=float(raw.get("accepted_at", 0.0)),
        update=update,
        attempts=int(raw.get("attempts", 0)),
    )


def _entry_payload(entry: Entry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "update_id": entry.update_id,
        "kind": entry.kind,
        "chat_id": entry.chat_id,
        "accepted_at": entry.accepted_at,
        "attempts": entry.attempts,
        "update": entry.update,
    }


class Inbox:
    """The queue itself: files in, files out.

    Knows nothing about aiogram beyond the shape of a raw update dict, so
    the storage is testable without a bot and the wiring below is testable
    without a disk.
    """

    def __init__(
        self,
        runtime_dir: Path | str,
        *,
        clock_fn: Callable[[], float] = time.time,
    ) -> None:
        self.dir = Path(runtime_dir) / DIRNAME
        self.stale_dir = self.dir / STALE_DIRNAME
        self.receipts_file = self.dir / RECEIPTS_FILENAME
        self._clock = clock_fn
        self._receipts: list[str] | None = None
        # Ids being handled RIGHT NOW by this process. aiogram runs every
        # update as its own task, and Telegram happily redelivers a batch it
        # never got a confirmed offset for, so "already on disk" is not the
        # same question as "already being answered".
        self._running: set[str] = set()

    # ── clock and files ──────────────────────────────────────────────

    def now(self) -> float:
        return self._clock()

    def _path(self, item_id: str) -> Path:
        return self.dir / f"{item_id}.json"

    # ── acceptance ───────────────────────────────────────────────────

    def accept(self, update: dict[str, Any]) -> Entry:
        """Persist one incoming update. Raises only on a write failure.

        Idempotent in BOTH directions, and both of them matter:

        * An update already on disk keeps its ORIGINAL ``accepted_at`` and
          its attempt count. A redelivered batch must not reset the age that
          decides whether the message is still worth replaying, or a bot
          flapping in a restart loop would keep a stale message eligible
          forever.
        * An update that already has a RECEIPT is not written back at all
          (blind review 1). This is not an edge case, it is the feature's
          own headline path: batch written → crash before the next
          ``getUpdates`` confirms it → boot → replay answers it, receipt
          written, entry dropped → polling starts → Telegram redelivers the
          still-unconfirmed batch. Without this guard the entry would be
          re-created, ``handled_middleware`` would refuse to claim it, and
          nothing would ever remove it again: a permanent phantom in
          ``/work``, and a real second answer once the ring aged the receipt
          out.
        """
        kind, chat_id = describe(update)
        item_id = entry_id(update["update_id"])
        entry = Entry(
            id=item_id,
            update_id=int(update["update_id"]),
            kind=kind,
            chat_id=chat_id,
            accepted_at=self.now(),
            update=update,
        )
        if self.is_handled(item_id):
            # Answered already. Returned unwritten so the caller still has
            # something to log; the gate below will refuse it anyway.
            return entry
        existing = self.read(item_id)
        if existing is not None:
            return existing
        write_json_atomic(self._path(item_id), _entry_payload(entry))
        return entry

    def record_attempt(self, entry: Entry) -> Entry:
        """Persist one more replay attempt for ``entry``, BEFORE the work.

        Before, not after: a turn that wedges the engine hard enough to need
        a kill would never reach an after-the-fact counter, which is the one
        case the counter exists for.
        """
        bumped = Entry(**{**_entry_payload(entry), "attempts": entry.attempts + 1})
        write_json_atomic(self._path(entry.id), _entry_payload(bumped))
        return bumped

    def claim(self, item_id: str) -> bool:
        """May this process start handling ``item_id`` now?

        ``False`` for an update that already has a receipt (answered before
        a crash, and Telegram resent the unconfirmed batch) or that another
        task is handling this instant. Deliberately synchronous and
        await-free between the check and the set: that is what makes it
        atomic against concurrently scheduled handler tasks, exactly like
        ``chat.py``'s media-dispatch claim.
        """
        if item_id in self._running:
            return False
        if self.is_handled(item_id):
            return False
        self._running.add(item_id)
        return True

    def release(self, item_id: str) -> None:
        """Give the claim back WITHOUT signing the entry off — the turn did
        not finish and the message is still owed an answer."""
        self._running.discard(item_id)

    def mark_handled(self, item_id: str) -> None:
        """Receipt first, entry second — that order is the duplicate guard.

        The receipt is written even when there is no entry on disk (blind
        review 6). An earlier version skipped it to keep the ring clean, but
        the id with no entry is precisely the id whose ``accept`` FAILED —
        a full or unwritable disk — and that is exactly the state a bot
        restarts a lot in. No receipt there means a redelivered batch gets
        answered a second time, i.e. the one guarantee this module makes
        would evaporate in the one condition it is supposed to hold in.
        A thousand-id ring can afford the honesty.
        """
        self._running.discard(item_id)
        self._add_receipt(item_id)
        self.drop(item_id)

    def drop(self, item_id: str) -> None:
        """Remove an entry. Idempotent.

        A failure here is logged and swallowed: the entry survives, and the
        receipt written just before it is what stops the next boot from
        replaying it — the same trade the outbox makes.
        """
        try:
            self._path(item_id).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("inbox: could not drop %s: %s", item_id, exc)

    # ── reading ──────────────────────────────────────────────────────

    def read(self, item_id: str) -> Entry | None:
        try:
            return _entry_from_dict(json.loads(self._path(item_id).read_text()))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def pending(self) -> list[Entry]:
        """Everything accepted and not yet handled, in arrival order.

        A file that will not parse is moved to ``stale/`` rather than
        retried forever or deleted: the point of this queue is that nothing
        vanishes without a trace.
        """
        try:
            names = sorted(p.name for p in self.dir.glob("*.json"))
        except OSError:
            return []
        entries: list[Entry] = []
        for name in names:
            if name == RECEIPTS_FILENAME:
                continue
            path = self.dir / name
            try:
                entries.append(_entry_from_dict(json.loads(path.read_text())))
            except FileNotFoundError:
                continue
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.error("inbox: unreadable entry %s (%s)", name, exc)
                self._quarantine(name, str(exc))
        return entries

    def replayable(
        self, *, max_age: float, now: float | None = None
    ) -> tuple[list[Entry], list[Entry]]:
        """``(fresh, too_old)`` — what to replay and what to retire.

        The threshold is a BLAST-RADIUS bound, not a noise bound (blind
        review 8). It is tempting to justify it as "answering an hour-old
        «как дела» on boot is noise", but that reasoning does not survive
        contact with the rest of the system: an update Telegram never got to
        hand us has no age limit at all — it waits up to 24h on their side
        and is answered the moment polling starts. What the threshold really
        buys is a ceiling on how much work one boot inherits, and a
        guarantee that a queue nobody noticed cannot surprise a chat days
        later.

        ``max_age <= 0`` disables replay entirely: everything pending is
        retired. An operator who turns this off wants silence at boot, not a
        counter that grows forever.
        """
        moment = self.now() if now is None else now
        fresh: list[Entry] = []
        old: list[Entry] = []
        for entry in self.pending():
            if max_age > 0 and moment - entry.accepted_at <= max_age:
                fresh.append(entry)
            else:
                old.append(entry)
        return fresh, old

    # ── retiring ─────────────────────────────────────────────────────

    def retire(self, entry: Entry, *, reason: str) -> None:
        """Move an entry to ``stale/`` with the reason attached."""
        payload = _entry_payload(entry)
        payload["reason"] = reason
        payload["retired_at"] = self.now()
        try:
            write_json_atomic(self.stale_dir / f"{entry.id}.json", payload)
        except OSError as exc:
            logger.warning("inbox: could not retire %s: %s", entry.id, exc)
            return
        self.drop(entry.id)
        logger.warning(
            "inbox: not replaying %s (%s, chat %s) — %s",
            entry.id,
            entry.kind,
            entry.chat_id,
            reason,
        )

    def _quarantine(self, name: str, reason: str) -> None:
        """Retire a file we could not even parse: keep the bytes."""
        source = self.dir / name
        try:
            raw = source.read_text()
        except OSError:
            raw = ""
        try:
            write_json_atomic(
                self.stale_dir / name,
                {
                    "id": name,
                    "raw": raw,
                    "reason": f"unreadable: {reason}",
                    "retired_at": self.now(),
                },
            )
            source.unlink()
        except OSError as exc:
            logger.warning("inbox: could not quarantine %s: %s", name, exc)

    # ── receipts ─────────────────────────────────────────────────────

    def _load_receipts(self) -> list[str]:
        if self._receipts is None:
            try:
                raw = json.loads(self.receipts_file.read_text())
                ids = raw.get("ids", []) if isinstance(raw, dict) else []
                self._receipts = [str(i) for i in ids]
            except (OSError, ValueError, AttributeError):
                self._receipts = []
        return self._receipts

    def is_handled(self, item_id: str) -> bool:
        return item_id in self._load_receipts()

    def _add_receipt(self, item_id: str) -> None:
        ids = self._load_receipts()
        if item_id in ids:
            return
        ids.append(item_id)
        del ids[:-RECEIPTS_KEPT]
        try:
            write_json_atomic(self.receipts_file, {"ids": ids})
        except OSError as exc:
            # A lost receipt costs at most one duplicate answer after a
            # crash; it must never cost the answer itself.
            logger.warning("inbox: could not write receipt %s: %s", item_id, exc)

    # ── reporting ────────────────────────────────────────────────────

    def stats(self, *, now: float | None = None) -> Stats:
        """Best effort by contract — this feeds a status report.

        Deliberately does NOT reuse ``pending()``: that one quarantines a
        file it cannot parse, and a read-only status command must not mutate
        the queue it is describing (the outbox's blind review, S5).

        What is in flight right now is counted separately and left out of
        ``pending`` — see ``Stats``.
        """
        moment = self.now() if now is None else now
        try:
            paths = [p for p in self.dir.glob("*.json") if p.name != RECEIPTS_FILENAME]
        except OSError:
            paths = []
        oldest = 0.0
        waiting = 0
        in_flight = 0
        for path in paths:
            if path.stem in self._running:
                in_flight += 1
                continue
            waiting += 1
            try:
                accepted = float(json.loads(path.read_text())["accepted_at"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            oldest = max(oldest, moment - accepted)
        try:
            stale = len(list(self.stale_dir.glob("*.json")))
        except OSError:
            stale = 0
        return Stats(
            pending=waiting,
            in_flight=in_flight,
            stale=stale,
            oldest_age=max(0.0, oldest),
        )


# ── wiring: the acceptance point and the duplicate gate ──────────────


def accept_all(box: Inbox, updates: list[Any]) -> None:
    """Write a fetched batch to disk. Returns nothing on purpose.

    The caller MUST hand ``_listen_updates`` back the batch it was given,
    unfiltered — see the module docstring: a filtered list stalls the
    Telegram offset and the dropped updates are redelivered forever.

    A write that fails is logged and the update goes through anyway.
    """
    for update in updates:
        try:
            raw = update.model_dump(mode="json", exclude_none=True)
            box.accept(raw)
        except Exception:  # noqa: BLE001 — never let durability cost delivery
            logger.exception(
                "inbox: could not accept update %s — handling it anyway",
                getattr(update, "update_id", "?"),
            )


def accept_middleware(box: Inbox) -> Callable[..., Awaitable[Any]]:
    """Session middleware for ``bot.session.middleware(...)``.

    Fires around every Bot API call; only ``getUpdates`` is interesting.
    Because ``Session.__call__`` awaits this, ``_listen_updates`` cannot
    resume — and therefore cannot move the offset Telegram confirms
    against — until the whole batch is on disk.
    """

    async def middleware(make_request: Any, bot: Any, method: Any) -> Any:
        result = await make_request(bot, method)
        if isinstance(method, GetUpdates) and isinstance(result, list):
            accept_all(box, result)
        return result

    return middleware


def handled_middleware(box: Inbox) -> Callable[..., Awaitable[Any]]:
    """Outer middleware for ``dp.update.outer_middleware(...)``.

    Two jobs, both of which have to be OUTSIDE everything else — the auth
    middleware, the routers, the filters:

    * refuse an update that was already answered (receipt) or is already in
      flight — the "no double answer" half of the contract;
    * mark it handled once the chain is done.

    The mark happens in a ``finally``, i.e. even for a handler that raised.
    "Handled" here means "this process ran it to the end", not "it went
    well": a message that blows a handler up would otherwise be replayed on
    every single boot for the next hour, and an error reply has already been
    sent by the handlers themselves.

    CANCELLATION is the one exception (blind review 10). A cancelled turn
    was not run to the end and nobody answered it, so signing it off would
    delete the entry for a message that is still owed a reply — the exact
    loss this module exists to prevent. Today aiogram's shutdown never runs
    this ``finally`` (it cancels only the polling task and lets the handler
    tasks die with the loop), so this is a guard against a future aiogram
    that awaits them, and against ``replay``'s own budget cancelling a turn
    mid-flight.
    """

    async def middleware(handler: Any, event: Any, data: dict[str, Any]) -> Any:
        item_id = entry_id(event.update_id)
        if not box.claim(item_id):
            logger.info("inbox: %s already handled, not answering twice", item_id)
            return None
        ran_to_the_end = True
        try:
            return await handler(event, data)
        except asyncio.CancelledError:
            ran_to_the_end = False
            raise
        finally:
            if ran_to_the_end:
                box.mark_handled(item_id)
            else:
                logger.warning(
                    "inbox: %s was cancelled mid-turn — leaving it unanswered "
                    "in the queue",
                    item_id,
                )
                box.release(item_id)

    return middleware


async def replay(
    bot: Any,
    dp: Any,
    box: Inbox,
    *,
    max_age: float = DEFAULT_MAX_AGE,
) -> int:
    """Answer what the previous process accepted and never answered.

    Called at boot, BEFORE polling starts, and awaited. That ordering is
    what keeps a chat in order: the backlog goes through one update at a
    time, oldest first, with no new traffic interleaving.

    Replayed updates go through ``dp.feed_raw_update``, i.e. the same
    dispatcher, middlewares and handlers a live update takes — no second
    code path to drift. The handlers are not, however, IDEMPOTENT: the
    librarian safety net in ``bot/handlers/chat.py`` appends to the daily
    file and saves attachments before the turn begins, so a message that
    crashed AFTER that point gets a second daily line (and, for media, a
    second copy of the file) when it is replayed. That is a known, bounded
    cost of this step — a duplicate line in a daily note against a lost
    message — and the place to fix it is a ``msg_id`` key in
    ``VaultStorage``, not here.

    Two bounds, both learned from the blind review:

    * ``entry.attempts``, persisted BEFORE the work: an update that already
      cost one boot its polling is retired instead of being fed to the same
      engine it wedged.
    * the caller's ``REPLAY_BUDGET`` — see ``bot/main.py``.
    """
    try:
        fresh, old = box.replayable(max_age=max_age)
    except Exception:  # noqa: BLE001 — a bad replay must never block the bot
        logger.exception("inbox: could not read the queue at boot")
        return 0

    for entry in old:
        box.retire(entry, reason=f"older than {max_age:.0f}s at startup")

    if not fresh:
        logger.info("inbox: nothing to replay")
        return 0

    logger.info("inbox: replaying %d accepted-but-unanswered update(s)", len(fresh))
    done = 0
    for entry in fresh:
        if box.is_handled(entry.id):
            # Receipt written, entry not yet removed: we died between the
            # two. Do NOT answer it again — just finish the removal.
            logger.info("inbox: %s already answered, dropping the leftover", entry.id)
            box.mark_handled(entry.id)
            continue
        if entry.attempts >= REPLAY_MAX_ATTEMPTS:
            box.retire(
                entry,
                reason=f"{entry.attempts} replay attempt(s) did not finish",
            )
            continue
        try:
            box.record_attempt(entry)
        except OSError:
            # Cannot count the attempt → cannot promise this stops. Retiring
            # is the safe direction: a lost answer beats an unbootable bot.
            logger.exception(
                "inbox: could not record a replay attempt for %s", entry.id
            )
            box.retire(entry, reason="attempt counter unwritable")
            continue
        try:
            await dp.feed_raw_update(bot, entry.update)
            done += 1
        except asyncio.CancelledError:
            # The caller's budget ran out mid-turn. The message was not
            # answered, so it keeps its place in the queue; the attempt is
            # already counted, so the next boot retires it rather than
            # spending another budget on it.
            box.release(entry.id)
            raise
        except Exception:  # noqa: BLE001 — one bad entry must not strand the rest
            logger.exception("inbox: replay of %s failed", entry.id)
        # Belt and braces: handled_middleware already did this, unless the
        # update never reached it (a payload Telegram's schema no longer
        # validates). Idempotent either way.
        box.mark_handled(entry.id)
    return done


# ── process-wide handle ──────────────────────────────────────────────
#
# ``/work`` needs the LIVE queue, not a fresh reader of the same directory:
# only the live one knows which entries this process is handling right now,
# and without that every ``/work`` counts its own update as a stuck message
# (blind review 2). Same pattern, same three functions as ``outbox``.

_current: Inbox | None = None


def configure(runtime_dir: Path | str) -> Inbox:
    """Install the process-wide queue. Called once, from ``run_bot``."""
    global _current  # noqa: PLW0603
    _current = Inbox(runtime_dir)
    return _current


def current() -> Inbox | None:
    """The configured queue, or None outside the bot process."""
    return _current


def reset() -> None:
    """Drop the process-wide queue. For tests — the bot never unconfigures
    itself."""
    global _current  # noqa: PLW0603
    _current = None
