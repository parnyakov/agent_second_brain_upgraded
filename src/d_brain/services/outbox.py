"""Durable outbox: a reply is on disk BEFORE anything is sent to Telegram.

The pain this exists for (owner report, 2026-09-22): "ответ родился, но не дошёл".
Until now every reply went straight out of the handler with
``bot.send_message``. A network blip, a 429, a restart landing in that exact
second, or the process dying mid-send left an exception in the log and
nothing at all for the user — the answer existed for a moment and then did
not.

So the order is inverted. ``send_response`` writes the (already sanitized and
chunked) reply into this queue first, and only then asks for it to be
delivered. Nothing is ever *only* in memory.

Shape — deliberately the smallest thing that covers the requirement, and the
same shape the rest of this runtime dir already uses (``ask_health``,
``cron_store``): plain JSON files, atomic ``os.replace`` writes, owner-only
permissions. No database, no broker, no new layer::

    runtime_dir/outbox/
        <id>.json          one queued message, id = a zero-padded ns stamp,
                           so plain filename order IS arrival order
        receipts.json      capped ring of ids already delivered
        dead/<id>.json     gave up on this one, with the reason and when

Guarantees, and what they cost:

* **Survives a restart.** The queue is files; a new process picks up whatever
  the old one left. ``run()`` starts before polling does, so leftovers go out
  at boot rather than waiting for the next message.
* **At least once, with a receipt against the obvious duplicate.** A receipt
  is written *before* the queue entry is removed, so a crash in that window
  leaves the entry behind but the receipt makes the next drain drop it
  instead of sending it twice. The one window that cannot be closed from
  here is a crash between Telegram accepting the message and the receipt
  being written — that is inherent to at-least-once and is why this is not
  advertised as exactly-once.
* **Order inside a chat.** One drain at a time (an asyncio lock — the bot is
  a single process, single loop), entries in filename order, and a chat whose
  head entry is backing off or failing is skipped WHOLE for that pass. A
  stuck chat never lets its own later messages overtake it; other chats are
  unaffected.
* **Nothing disappears.** Attempts are capped; past the cap the entry moves
  to ``dead/`` with the reason and ``/work`` reports the count. A file that
  will not even parse is moved there too rather than deleted.

Not here on purpose: cross-process locking (only the bot writes), priorities,
per-chat worker tasks, a retry policy object. One module, one background
task, one function that talks to Telegram.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNotFound,
)

logger = logging.getLogger(__name__)

DIRNAME = "outbox"
DEAD_DIRNAME = "dead"
RECEIPTS_FILENAME = "receipts.json"

# Ten tries, doubling from 2s and capped at 5 min, is ~15 minutes of trying
# before an entry is declared dead. Long enough to ride out a Telegram blip
# or a rate-limit window; short enough that a genuinely broken channel shows
# up in /work while the owner still remembers asking the question.
MAX_ATTEMPTS = 10
BASE_BACKOFF = 2.0
MAX_BACKOFF = 300.0
# How many delivered ids the receipt ring keeps. Only the crash window above
# needs them, which is measured in the last message or two — 200 is already
# absurdly generous and keeps the file a few KB.
RECEIPTS_KEPT = 200
# Background sweep. Only ever picks up what the inline drain could not send
# (backoff, restart leftovers), so it can be lazy.
POLL_SECONDS = 2.0
# Pacing between two consecutive sends to the SAME chat — the 0.3s
# ``send_response`` has always slept between chunks, kept here now that the
# sending itself moved.
SEND_SPACING = 0.3
# Sends attempted in one pass. The pass holds the lock, and the pacing above
# is paid inside it, so an unbounded backlog would make a fresh reply wait
# out the whole queue before its own inline drain even starts. Bounded, the
# worst a new reply pays is one batch; the rest of the backlog goes out on
# the worker's next tick (blind review S3).
MAX_PER_PASS = 20


def write_json_atomic(target: Path, payload: dict[str, Any]) -> None:
    """Atomic, owner-only JSON write — the one used by BOTH durable queues.

    Module-level rather than a method because ``inbox.py`` needs exactly
    this and nothing else from here: chat ids, reply texts and incoming
    message bodies are private, so the mode is set on the temp file BEFORE
    the bytes land in it — the same precaution ``cron_store.save`` takes for
    prompts.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.touch()
    os.chmod(tmp, 0o600)
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    os.replace(tmp, target)


def backoff_seconds(attempts: int) -> float:
    """Delay before attempt number ``attempts + 1``. Pure, so the ladder is
    testable without a clock."""
    if attempts <= 0:
        return 0.0
    return min(BASE_BACKOFF * (2 ** (attempts - 1)), MAX_BACKOFF)


# Telegram saying "this will never work" — a bad chat id, a blocked bot, a
# message the API refuses outright. Retrying these ten times buys nothing and
# costs everything: the chat is held in order behind the doomed entry, so on
# a single-user bot one poison message is a bot-wide silence (blind review
# S2). Bury it at once instead and let the chat move on.
#
# Everything else — 429s, network errors, 5xx, anything unrecognized — is
# assumed transient and retried. Erring that way keeps a reply alive.
PERMANENT_ERRORS = (TelegramBadRequest, TelegramForbiddenError, TelegramNotFound)


def is_permanent(error: object) -> bool:
    return isinstance(error, PERMANENT_ERRORS)


def retry_delay(error: object, attempts: int) -> float:
    """How long to wait before the next attempt.

    Telegram's own ``retry_after`` wins when it is longer than our ladder:
    hammering inside an active flood-wait is what extends it.
    """
    ladder = backoff_seconds(attempts)
    after = getattr(error, "retry_after", None)
    if isinstance(after, int | float) and not isinstance(after, bool):
        return max(ladder, float(after) + 1.0)
    return ladder


@dataclass
class Item:
    """One queued Telegram message — already sanitized and already split to
    fit, because that work happens once, at enqueue time, not on every
    retry."""

    id: str
    chat_id: int
    text: str
    created_at: float
    attempts: int = 0
    next_attempt: float = 0.0
    last_error: str = ""
    # The message this chunk answers, when saying so is worth it — today
    # only a reply that comes out of the per-chat queue (item 33 step 5),
    # where minutes may have passed since the question. 0 = a plain send,
    # which is what every entry written before this field existed reads as.
    reply_to: int = 0


@dataclass(frozen=True)
class Stats:
    """What ``/work`` prints: how much is waiting, how much gave up."""

    pending: int = 0
    dead: int = 0
    oldest_age: float = 0.0


def _item_from_dict(raw: dict[str, Any]) -> Item:
    return Item(
        id=str(raw["id"]),
        chat_id=int(raw["chat_id"]),
        text=str(raw["text"]),
        created_at=float(raw.get("created_at", 0.0)),
        attempts=int(raw.get("attempts", 0)),
        next_attempt=float(raw.get("next_attempt", 0.0)),
        last_error=str(raw.get("last_error", "")),
        reply_to=int(raw.get("reply_to", 0) or 0),
    )


class Outbox:
    """The queue itself: files in, files out. No Telegram knowledge at all —
    that lives in ``send_one``/``drain`` below, so the storage is testable
    without a bot and the sending is testable without a disk."""

    def __init__(
        self,
        runtime_dir: Path | str,
        *,
        max_attempts: int = MAX_ATTEMPTS,
        clock_fn: Callable[[], float] = time.time,
    ) -> None:
        self.dir = Path(runtime_dir) / DIRNAME
        self.dead_dir = self.dir / DEAD_DIRNAME
        self.receipts_file = self.dir / RECEIPTS_FILENAME
        self._max_attempts = max_attempts
        self._clock = clock_fn
        self._last_ns = 0
        self._receipts: list[str] | None = None
        # Serializes drains. aiogram runs every update as its own task, so
        # two replies can reach drain() at the same instant; without this
        # they would both read the same pending list and send it twice.
        self.lock = asyncio.Lock()

    # ── ids and files ────────────────────────────────────────────────

    def now(self) -> float:
        return self._clock()

    def _new_id(self) -> str:
        """Monotonic within the process, sortable as a plain string.

        Wall-clock nanoseconds: they order correctly against entries left by
        a PREVIOUS process, which an in-memory counter would not. ``max``
        against the last id handed out covers two enqueues inside the same
        nanosecond tick.
        """
        ns = max(time.time_ns(), self._last_ns + 1)
        self._last_ns = ns
        return f"{ns:019d}"

    def _path(self, item_id: str) -> Path:
        return self.dir / f"{item_id}.json"

    def _write_json(self, target: Path, payload: dict[str, Any]) -> None:
        """See ``write_json_atomic`` — kept as a method so the call sites
        below read the same as they always did."""
        write_json_atomic(target, payload)

    # ── queue operations ─────────────────────────────────────────────

    def enqueue(self, chat_id: int, text: str, *, reply_to: int = 0) -> Item:
        """Persist one message. Returns the stored item (id included)."""
        item = Item(
            id=self._new_id(),
            chat_id=int(chat_id),
            text=text,
            created_at=self.now(),
            reply_to=int(reply_to or 0),
        )
        self._write_json(self._path(item.id), asdict(item))
        return item

    def pending(self) -> list[Item]:
        """Everything still waiting, in arrival order.

        A file that will not parse is moved to ``dead/`` rather than skipped
        forever or deleted: the whole point of this queue is that nothing
        vanishes without a trace.
        """
        try:
            names = sorted(p.name for p in self.dir.glob("*.json"))
        except OSError:
            return []
        items: list[Item] = []
        for name in names:
            if name == RECEIPTS_FILENAME:
                continue
            path = self.dir / name
            try:
                items.append(_item_from_dict(json.loads(path.read_text())))
            except FileNotFoundError:
                continue
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.error("outbox: unreadable entry %s (%s)", name, exc)
                self._bury_raw(name, {"reason": f"unreadable: {exc}"})
        return items

    def drop(self, item_id: str) -> None:
        """Remove a queue entry. Idempotent.

        A failure here is logged and swallowed: the entry survives, and the
        receipt written just before it is what stops the next drain from
        sending it again — until the receipt ring ages that id out. So a
        permanently unlinkable file is a (very slow) duplicate, not a loop.
        """
        try:
            self._path(item_id).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("outbox: could not drop %s: %s", item_id, exc)

    def mark_delivered(self, item: Item) -> None:
        """Receipt first, entry second — that order is the duplicate guard."""
        self._add_receipt(item.id)
        self.drop(item.id)

    def record_failure(self, item: Item, error: object, *, now: float) -> str:
        """Fold one failed send into the entry. Returns ``retry`` or ``dead``."""
        item.attempts += 1
        item.last_error = f"{type(error).__name__}: {error}"[:300]
        if is_permanent(error):
            self.bury(item, reason=f"refused: {item.last_error}", now=now)
            return "dead"
        if item.attempts >= self._max_attempts:
            self.bury(
                item,
                reason=f"{item.attempts} attempts failed: {item.last_error}",
                now=now,
            )
            return "dead"
        item.next_attempt = now + retry_delay(error, item.attempts)
        self._write_json(self._path(item.id), asdict(item))
        return "retry"

    def bury(self, item: Item, *, reason: str, now: float | None = None) -> None:
        """Move an entry to the dead queue, keeping the reason with it."""
        payload = asdict(item)
        payload["reason"] = reason
        payload["failed_at"] = self.now() if now is None else now
        self._write_json(self.dead_dir / f"{item.id}.json", payload)
        self.drop(item.id)
        logger.error(
            "outbox: giving up on %s for chat %s (%s)", item.id, item.chat_id, reason
        )

    def _bury_raw(self, name: str, extra: dict[str, Any]) -> None:
        """Dead-letter a file we could not even parse: keep the bytes."""
        source = self.dir / name
        try:
            raw = source.read_text()
        except OSError:
            raw = ""
        payload: dict[str, Any] = {"id": name, "raw": raw, "failed_at": self.now()}
        payload.update(extra)
        try:
            self._write_json(self.dead_dir / name, payload)
            source.unlink()
        except OSError as exc:
            logger.warning("outbox: could not quarantine %s: %s", name, exc)

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

    def is_delivered(self, item_id: str) -> bool:
        return item_id in self._load_receipts()

    def _add_receipt(self, item_id: str) -> None:
        ids = self._load_receipts()
        if item_id in ids:
            return
        ids.append(item_id)
        del ids[:-RECEIPTS_KEPT]
        try:
            self._write_json(self.receipts_file, {"ids": ids})
        except OSError as exc:
            # A lost receipt costs at most one duplicate after a crash; it
            # must never cost the delivery itself.
            logger.warning("outbox: could not write receipt %s: %s", item_id, exc)

    # ── reporting ────────────────────────────────────────────────────

    def stats(self, *, now: float | None = None) -> Stats:
        """Best effort by contract — this feeds a status report.

        The age is measured from ``created_at``, i.e. from when the reply was
        PRODUCED. An earlier version used the file's mtime, which
        ``record_failure`` rewrites on every retry: a reply stuck for an hour
        reported itself as five minutes old, which is the opposite of what a
        stuck-queue report is for (blind review S5).

        Deliberately does NOT reuse ``pending()``: that one quarantines a
        file it cannot parse, and a read-only status command must not mutate
        the queue it is describing.
        """
        moment = self.now() if now is None else now
        try:
            paths = [p for p in self.dir.glob("*.json") if p.name != RECEIPTS_FILENAME]
        except OSError:
            paths = []
        oldest = 0.0
        for path in paths:
            try:
                created = float(json.loads(path.read_text())["created_at"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            oldest = max(oldest, moment - created)
        try:
            dead = len(list(self.dead_dir.glob("*.json")))
        except OSError:
            dead = 0
        return Stats(pending=len(paths), dead=dead, oldest_age=max(0.0, oldest))


# ── the single point where a message reaches Telegram ────────────────


async def send_one(bot: Any, chat_id: int, text: str, *, reply_to: int = 0) -> None:
    """Hand one chunk to Telegram. THE one place a reply is sent.

    ``reply_to`` threads the message under the one it answers. It always
    travels with ``allow_sending_without_reply=True``: the answer matters,
    the threading is decoration, and a deleted original must never turn a
    real reply into a permanent ``TelegramBadRequest`` that the queue would
    then bury in ``dead/``.

    The HTML→plain retry is what ``send_response`` used to do inline: a 400
    "can't parse entities" means the markup is bad, and a readable plain-text
    reply beats no reply.

    It is narrowed to ``TelegramBadRequest`` on purpose (blind review B1).
    It used to catch everything, which made the classic Telegram failure —
    the server accepted the message and the client saw a timeout or a 429 —
    send a SECOND copy immediately. Harmless-ish when that cost one
    duplicate; with a retry ladder behind it, the same reply could be
    delivered twenty times and still be filed as lost. Only a 400 gets the
    second attempt now; everything else goes straight back to the queue.
    """
    extra: dict[str, Any] = {}
    if reply_to:
        extra = {
            "reply_to_message_id": int(reply_to),
            "allow_sending_without_reply": True,
        }
    try:
        await bot.send_message(chat_id, text, **extra)
    except TelegramBadRequest:
        await bot.send_message(chat_id, text, parse_mode=None, **extra)


async def drain(bot: Any, box: Outbox) -> None:
    """Try to deliver everything that is due. Never raises.

    Called inline by ``send_response`` (so a healthy reply still arrives in
    the same instant it always did) and on a timer by ``run`` (so a reply
    that could not go out then still goes out later).
    """
    async with box.lock:
        try:
            await _drain_locked(bot, box)
        except Exception:  # noqa: BLE001 — a failed drain must never fail a turn
            logger.exception("outbox: drain failed")


async def _drain_locked(bot: Any, box: Outbox) -> None:
    now = box.now()
    # Chats whose head entry could not be sent this pass. Skipping the rest
    # of the chat is what keeps replies in order — the alternative is a
    # retried message arriving after the answer that came later.
    blocked: set[int] = set()
    last_chat: int | None = None
    sent = 0
    for item in box.pending():
        if sent >= MAX_PER_PASS:
            return
        if item.chat_id in blocked:
            continue
        if item.next_attempt > now:
            blocked.add(item.chat_id)
            continue
        if box.is_delivered(item.id):
            # Receipt but no removal: we died between the two. Do NOT send.
            logger.info("outbox: %s already delivered, dropping duplicate", item.id)
            box.drop(item.id)
            continue
        if last_chat == item.chat_id:
            await asyncio.sleep(SEND_SPACING)
        sent += 1
        try:
            await send_one(bot, item.chat_id, item.text, reply_to=item.reply_to)
        except Exception as exc:  # noqa: BLE001 — that is what the queue is for
            blocked.add(item.chat_id)
            # The bookkeeping write can fail for the same reason the send
            # did (a full or unwritable disk). Letting that escape would end
            # the whole pass and strand every OTHER chat's healthy replies,
            # pass after pass, exactly when the queue matters most (blind
            # review S1). The chat is already blocked; that is enough.
            try:
                action = box.record_failure(item, exc, now=box.now())
            except OSError:
                logger.exception("outbox: could not record the failure of %s", item.id)
                continue
            logger.warning(
                "outbox: send of %s failed (attempt %d, %s): %s",
                item.id,
                item.attempts,
                action,
                exc,
            )
            continue
        box.mark_delivered(item)
        last_chat = item.chat_id


async def run(bot: Any, box: Outbox, *, poll_seconds: float = POLL_SECONDS) -> None:
    """The one background task: sweep the queue forever.

    Runs its first pass immediately, which is what makes a restart pick up
    the previous process's undelivered replies instead of stranding them.
    """
    logger.info("outbox worker started (%s, poll %.0fs)", box.dir, poll_seconds)
    while True:
        await drain(bot, box)
        await asyncio.sleep(poll_seconds)


# ── process-wide handle ──────────────────────────────────────────────
#
# ``send_response`` is called from four modules and takes no settings; the
# queue is one piece of bot-process runtime state, so it is configured once
# at startup rather than threaded through every call site. Same pattern as
# chat.py's session manager singleton.

_current: Outbox | None = None


def configure(runtime_dir: Path | str) -> Outbox:
    """Install the process-wide queue. Called once, from ``run_bot``."""
    global _current  # noqa: PLW0603
    _current = Outbox(runtime_dir)
    return _current


def current() -> Outbox | None:
    """The configured queue, or None outside the bot process."""
    return _current


def reset() -> None:
    """Drop the process-wide queue. For tests — the bot never unconfigures
    itself, and anything still queued at shutdown is picked up at boot."""
    global _current  # noqa: PLW0603
    _current = None
