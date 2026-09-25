"""What the durable queues PROVE about a message: that it was lost.

A false alarm once fired twice in a row ("🔴 канал доставки под вопросом, N
ответов подряд не доставлено" and "✅ длинная задача завершилась") — both
true statements about an internal flag and false statements about the
world, since the answers had in fact been delivered. The governing rule:
if a reply was actually delivered, no alert claiming it wasn't should ever
fire.

Every OTHER alerting signal in this system is an inference from something
adjacent to delivery — a unit's ActiveState, a file's age, a streak of turn
outcomes, a process count. Each of them can be true while the person is
reading their answers, which is how this module came to exist. The three
durable queues it builds on are different: they leave a FILE behind for
exactly the states that mean a message was lost, and this module counts
those files.

  ``outbox/dead/``       a reply that was produced and could not be delivered
  ``inbox/stale/``       a message accepted and retired without an answer
  ``chat-queue/stale/``  a message prepared and retired without an answer
  ``inbox/<id>.json``    accepted long ago, and nothing has touched it since

Why there is no "proof of DELIVERY" here
----------------------------------------

The obvious companion — "``outbox/receipts.json`` says something reached
Telegram after the last failed turn, so do not alert" — was written, and
removed in blind review. It is satisfied by every streak it was meant to
judge: the apology the user gets for a failed turn goes out through the
outbox like any other message, and its receipt is minted AFTER the ledger
row for that same turn. A timestamp veto built on it silently disarms the
whole delivery backstop while looking like a careful check.

The receipt still does its job, just not as a timestamp: it is what REMOVES
an entry from the outbox. So a reply that is still in the queue, or in
``dead/``, is by construction a reply Telegram never accepted — which is
what the counts below rest on. And the question the health ledger asks
("did the person get an ANSWER") is answered where the answer is produced,
by scoring the turn `ok` on every delivery route. See
``delivery_guard.decide``'s docstring.

Deliberately read-only and deliberately stdlib-only. Read-only because the
callers are alerting paths: ``Outbox.pending()`` and ``Inbox.pending()``
quarantine a file they cannot parse, and a check that describes a queue must
not mutate it (the same call the outbox's own blind review made for
``stats``). Stdlib-only because one of the callers is the watchdog process,
which must be able to answer "is anything actually lost" without importing
the bot's transport stack.

The directory names are duplicated here rather than imported for that same
reason — importing ``outbox`` pulls in aiogram. ``test_delivery_proof.py``
asserts every one of them still equals its source module's constant, so the
duplication cannot drift silently.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Mirrors of outbox.DIRNAME / outbox.DEAD_DIRNAME / outbox.RECEIPTS_FILENAME /
# inbox.DIRNAME / inbox.STALE_DIRNAME / chat_queue.DIRNAME /
# chat_queue.STALE_DIRNAME. Pinned by a test — see the module docstring.
OUTBOX_DIRNAME = "outbox"
OUTBOX_DEAD_DIRNAME = "dead"
RECEIPTS_FILENAME = "receipts.json"
INBOX_DIRNAME = "inbox"
INBOX_STALE_DIRNAME = "stale"
CHAT_QUEUE_DIRNAME = "chat-queue"
CHAT_QUEUE_STALE_DIRNAME = "stale"

# How long an accepted message may sit untouched in the inbox before its
# presence is evidence of a loss rather than evidence of work in progress.
#
# It has to clear the longest LEGITIMATE time an entry can stay on disk, and
# that is much longer than one turn: the entry is removed only when the whole
# `dp.update` handler chain returns, and `bot/handlers/chat.py` runs a SECOND
# full `send_message` when the first comes back empty. Two turns at
# `chat_turn_timeout` (1500s) plus a busy-wait is already ~50 minutes of
# entirely normal behaviour (blind review 5).
#
# Two hours is the number `chat_queue_max_age` already uses for the same
# judgement, with the same justification ("far beyond any legitimate turn"),
# so the two agree rather than each inventing a threshold.
#
# An external reader cannot see ``Inbox._running`` — that set is per-process
# memory — so age is the ONLY thing that separates "being answered right now"
# from "stranded". Erring long is the right error: a late alert about a real
# loss costs minutes, a premature one is the cry-wolf this item removes.
DEFAULT_STUCK_AFTER = 7200.0


@dataclass(frozen=True)
class Loss:
    """Messages this install can PROVE it failed to deliver.

    Every field is derived from files that exist on disk right now. Nothing
    here is inferred from a status, a streak or a flag — that is the whole
    point of the type.

    ``ids`` carries WHICH losses these are, not just how many. A caller that
    only remembered counts would miss the case where one loss is cleared and
    a different one appears in the same interval — the count is unchanged
    and the new message is never reported (blind review 4).
    """

    dead_replies: int = 0
    """``outbox/dead/`` — produced, then permanently undeliverable."""

    retired_messages: int = 0
    """``inbox/stale/`` — accepted, retired without an answer."""

    retired_jobs: int = 0
    """``chat-queue/stale/`` — prepared, retired without an answer."""

    stranded_messages: int = 0
    """``inbox/`` — accepted, untouched for longer than ``stuck_after``."""

    ids: tuple[str, ...] = field(default_factory=tuple)
    """``<category>/<filename>`` for each loss above, sorted."""

    @property
    def any(self) -> bool:
        return bool(self.ids)

    @property
    def total(self) -> int:
        return len(self.ids)

    def new_since(self, reported: object) -> tuple[str, ...]:
        """The losses in here that ``reported`` has not seen.

        ``reported`` is any iterable of ids — a previous ``Loss.ids`` or a
        list read back from a latch file.
        """
        try:
            seen = set(reported or ())  # type: ignore[arg-type]
        except TypeError:
            seen = set()
        return tuple(i for i in self.ids if i not in seen)

    def reasons(self) -> list[str]:
        """Human-readable lines, WITHOUT the counts.

        The numbers are deliberately left out:
        ``notify.sh``/``backup-notify.sh`` debounce on a checksum of the
        message, so a count inside the text mints a brand-new "message"
        every time it moves and alerts forever — which is exactly how a
        false alarm once arrived every five minutes. The exact numbers
        belong in the log.
        """
        out: list[str] = []
        if self.dead_replies:
            out.append("ответ не удалось доставить (лежит в мёртвой очереди)")
        if self.retired_messages or self.retired_jobs:
            out.append("сообщение принято, но ответ так и не ушёл")
        if self.stranded_messages:
            out.append("принятое сообщение зависло без ответа")
        return out


def _entries(directory: Path) -> list[Path]:
    try:
        return sorted(
            p for p in directory.glob("*.json") if p.name != RECEIPTS_FILENAME
        )
    except OSError:
        return []


def losses(
    runtime_dir: Path | str,
    *,
    now: float,
    stuck_after: float = DEFAULT_STUCK_AFTER,
) -> Loss:
    """Everything this install can prove it failed to deliver, right now.

    Never raises and never writes: a directory that does not exist yet (a
    fresh install, a queue that has never been used) simply contributes
    zero.
    """
    root = Path(runtime_dir)
    outbox_dir = root / OUTBOX_DIRNAME
    inbox_dir = root / INBOX_DIRNAME

    ids: list[str] = []

    def collect(label: str, paths: list[Path]) -> int:
        ids.extend(f"{label}/{p.name}" for p in paths)
        return len(paths)

    dead = collect("outbox-dead", _entries(outbox_dir / OUTBOX_DEAD_DIRNAME))
    retired_messages = collect(
        "inbox-stale", _entries(inbox_dir / INBOX_STALE_DIRNAME)
    )
    retired_jobs = collect(
        "queue-stale",
        _entries(root / CHAT_QUEUE_DIRNAME / CHAT_QUEUE_STALE_DIRNAME),
    )

    # Age from the file's MTIME, not from the `accepted_at` inside it.
    #
    # The question here is "is anyone working on this", and mtime is the one
    # that answers it: `Inbox.record_attempt` rewrites the entry when a boot
    # replay picks it up, which is precisely the moment it stops being
    # abandoned. Reading `accepted_at` instead would call a backlog that is
    # actively being replayed a proven loss (blind review 8), and would
    # disagree with `delivery-watch.sh`, which can only see mtime.
    stranded: list[Path] = []
    for path in _entries(inbox_dir):
        try:
            touched = path.stat().st_mtime
        except FileNotFoundError:
            # A turn finished between the glob and the stat — an ordinary
            # race on a busy install, and the opposite of a loss (blind
            # review 7). `Inbox.pending()` skips it for the same reason.
            continue
        except OSError:
            continue
        if now - touched > stuck_after:
            stranded.append(path)

    return Loss(
        dead_replies=dead,
        retired_messages=retired_messages,
        retired_jobs=retired_jobs,
        stranded_messages=collect("inbox-stranded", stranded),
        ids=tuple(sorted(ids)),
    )


# ── the "already told him" latch ─────────────────────────────────────


LATCH_FILENAME = "delivery-loss.seen"


def read_reported(runtime_dir: Path | str) -> tuple[str, ...]:
    """Loss ids a report has already gone out for. Missing reads as none."""
    try:
        raw = json.loads((Path(runtime_dir) / LATCH_FILENAME).read_text())
    except (OSError, ValueError):
        return ()
    if not isinstance(raw, dict):
        return ()
    return tuple(str(i) for i in raw.get("ids", []) or [])


def write_reported(runtime_dir: Path | str, ids: tuple[str, ...]) -> None:
    """Remember what has been reported. Best effort — a latch that cannot be
    written costs a repeated alert, never a missing one, and that is the
    right way round."""
    target = Path(runtime_dir) / LATCH_FILENAME
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"ids": list(ids)}))
    except OSError as exc:
        logger.warning("could not write %s: %s", target, exc)
