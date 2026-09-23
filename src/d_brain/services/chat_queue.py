"""Per-chat work queue: one turn at a time, everything else waits on disk.

Step 5 — the last — of the reliability plan
(``thoughts/projects/agent-infra-backlog.md`` item 33). The four before it
each closed one hole: a reply that was born could not be lost on the way out
(outbox), a message that arrived could not be lost on the way in (inbox), a
restart no longer cuts a live turn off (shutdown), and a long turn now shows
what it is doing (progress card). What was still missing is the case the
owner hits every single day: he writes a SECOND message while the first one
is still being worked on.

Until now that message had three possible fates, and all three were bad:

* "🔧 Предыдущая задача ещё выполняется… напиши ещё раз через минуту" — a
  brush-off that asks the human to poll the bot;
* an answer from the DUTY session, which has the same vault but no idea what
  the conversation is about;
* or, on the voice and media paths, nothing at all for minutes while
  ``ask()`` sat on the process-wide ask-lock waiting its turn in silence.

The owner's rule, verbatim: «отвечать мне должна основная сессия в 99%
случаев, у неё есть контекст». So busyness must lead to a QUEUE, and the
duty session goes back to being what it was meant to be — the emergency
path, used only when the main brain shows signs of being WEDGED.

The shape
---------

One lane per chat, one live runner in it, and a durable FIFO behind it::

    runtime_dir/chat-queue/
        <id>.json          one waiting message, id = a zero-padded ns stamp,
                           so plain filename order IS arrival order
        stale/<id>.json    gave up on this one, with the reason and when

The lane itself is deliberately IN MEMORY (``_running``): a live runner can
only exist inside a live process, so a lane that survived a restart would be
a lie — it would wedge the chat until a TTL expired. What survives a restart
is the part that must: the waiting messages themselves.

What a job carries, and why
---------------------------

``chat_id``, ``user_id``, ``message_id`` and the fully prepared prompt. Not
the raw Telegram update — the inbox already stores that, and re-feeding it
through the dispatcher would re-run the librarian safety net in
``bot/handlers/chat.py`` (the daily line, the saved attachment) a second
time for every queued message. That is a known, accepted cost of a CRASH
(see ``inbox.replay``); paying it on the ordinary path, for every second
message, would not be. So the handoff is: the inbox owns "arrived, not yet
prepared", this queue owns "prepared, not yet answered", and the outbox owns
"answered, not yet delivered". Three files, one direction, no overlap.

``message_id`` is kept so the late answer can be a Telegram REPLY to the
message it answers — ten minutes after the fact, "принял, отвечу следом" is
only useful if the eventual answer says which message it belongs to. It is
also what a reaction would need, should one ever be added.

Guarantees, and what they cost
------------------------------

* **Order.** Filename order is arrival order, ``ready()`` hands out the
  oldest job per chat, and a new message never overtakes a waiting one — the
  gate in ``chat.py`` parks a message whenever the lane is taken OR the chat
  already has something waiting.
* **Survives a restart.** The jobs are files. The worker's first pass runs at
  boot, so whatever the previous process had parked is answered by the new
  one.
* **At least once.** A job is removed only AFTER its turn finished, so a
  crash mid-turn re-runs it. The reverse (remove first) would lose exactly
  the message this module exists to keep. The duplicate window is the same
  one the inbox already documents.
* **Bounded.** ``max_waiting`` per chat, and the human is told plainly when
  it is reached instead of being quietly dropped. A job that keeps blowing
  its turn up, or that has been waiting longer than ``max_age``, is retired
  to ``stale/`` with an honest line to the owner — nothing piles up silently
  and nothing vanishes.
* **Degrades to today.** No queue configured (unit tests, one-off tools) ⇒
  ``current()`` is None and ``chat.py`` takes exactly the pre-queue path,
  duty session included.

Not here on purpose: cross-process locking (only the bot writes), priorities,
per-chat worker tasks, a fairness policy. Draining is SEQUENTIAL — one queued
turn at a time for the whole machine — which is also the "общий ограничитель
на число одновременных агентов" the reference agent keeps on top: the engine
is a single shared session behind a single ask-lock, so anything else would
be a lie about parallelism we do not have.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from d_brain.services.outbox import write_json_atomic

logger = logging.getLogger(__name__)

DIRNAME = "chat-queue"
STALE_DIRNAME = "stale"

# How often the worker looks for work it can start. The lane is also released
# synchronously by the inline turn that held it, so in the common case ("the
# answer to the first message just went out") the wait is at most one tick.
POLL_SECONDS = 2.0

# Per-chat ceiling on WAITING messages. Ten is well past anything a human
# types while waiting for one answer; past it the honest thing is to say so,
# not to keep accepting work nobody will read the answer to in time.
DEFAULT_MAX_WAITING = 10

# How long a message may wait before the queue gives up on it. Two hours is
# far beyond any legitimate turn (``chat_turn_timeout`` is 1500s and the
# watchdog closes an unattended run at 1800s), so reaching it means something
# is wrong rather than slow — and an answer to a two-hour-old question is
# worth less than knowing it never came.
DEFAULT_MAX_AGE = 7200.0

# How many times a job's turn may blow up before it is retired. Only real
# EXCEPTIONS count: a turn that simply could not start yet (the session is
# busy with live work) is not an attempt, it is the normal waiting state.
MAX_ATTEMPTS = 3

# Backstop for a lane that was somehow never released. The lane is given back
# in a ``finally`` on every path, so this should never fire; if it ever does,
# a chat recovers by itself instead of going permanently deaf — the same call
# ``chat.py``'s media claim makes, for the same reason.
LANE_TTL = 7200.0


class QueueFull(RuntimeError):
    """This chat already has ``max_waiting`` messages waiting."""


@dataclass(frozen=True)
class Job:
    """One prepared message waiting for the main session."""

    id: str
    chat_id: int
    user_id: int
    prompt: str
    queued_at: float
    message_id: int | None = None
    attempts: int = 0


@dataclass(frozen=True)
class Stats:
    """What ``/work`` prints."""

    waiting: int = 0
    running: int = 0
    stale: int = 0
    oldest_age: float = 0.0
    per_chat: dict[int, int] | None = None


def _job_from_dict(raw: dict[str, Any]) -> Job:
    message_id = raw.get("message_id")
    return Job(
        id=str(raw["id"]),
        chat_id=int(raw["chat_id"]),
        user_id=int(raw["user_id"]),
        prompt=str(raw["prompt"]),
        queued_at=float(raw.get("queued_at", 0.0)),
        message_id=int(message_id) if isinstance(message_id, int) else None,
        attempts=int(raw.get("attempts", 0)),
    )


def _job_payload(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "chat_id": job.chat_id,
        "user_id": job.user_id,
        "prompt": job.prompt,
        "queued_at": job.queued_at,
        "message_id": job.message_id,
        "attempts": job.attempts,
    }


class ChatQueue:
    """The queue itself: files in, files out, plus the in-memory lane.

    Knows nothing about aiogram or about how a turn is run — that lives in
    ``drain`` below and in ``chat.py`` — so the storage is testable without a
    bot and the wiring is testable without a disk.
    """

    def __init__(
        self,
        runtime_dir: Path | str,
        *,
        max_waiting: int = DEFAULT_MAX_WAITING,
        max_age: float = DEFAULT_MAX_AGE,
        clock_fn: Callable[[], float] = time.time,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dir = Path(runtime_dir) / DIRNAME
        self.stale_dir = self.dir / STALE_DIRNAME
        self.max_waiting = int(max_waiting)
        self.max_age = float(max_age)
        self._clock = clock_fn
        self._monotonic = monotonic_fn
        # chat_id → (when its live runner took the lane, the queued job it is
        # running or None for an inline turn). One entry per chat.
        self._running: dict[int, tuple[float, str | None]] = {}

    # ── clock and files ──────────────────────────────────────────────

    def now(self) -> float:
        return self._clock()

    def _new_id(self, arrived_ns: int | None = None) -> str:
        """The id IS the arrival stamp, zero-padded so filename order is
        arrival order — the outbox's id, for the same reason: it orders
        correctly against files left by a PREVIOUS process, which an
        in-memory counter would not.

        ``arrived_ns`` matters more than it looks. The moment a message is
        PARKED is not the moment it ARRIVED, and the gap is unbounded: the
        message that took the lane only discovers the session is busy after
        ``ask()``'s busy-wait, which can be five minutes, so it is parked
        LAST even though it came FIRST. Stamping at parking time therefore
        inverted the queue for exactly the case this module exists for (blind
        review 1). The caller takes the stamp before it touches the lane.

        Collisions step forward one nanosecond at a time rather than being
        bumped past the last id handed out: "never collide" must not be
        bought by "and never mind the order", which is precisely what a
        ``max(..., _last_ns + 1)`` guard would do to a late-parked arrival.
        """
        ns = int(arrived_ns) if arrived_ns is not None else time.time_ns()
        while True:
            candidate = f"{ns:019d}"
            if not self._path(candidate).exists():
                return candidate
            ns += 1

    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    # ── the lane (one live runner per chat) ──────────────────────────

    def try_acquire(self, chat_id: int, *, job_id: str | None = None) -> bool:
        """Take this chat's lane, or report it taken.

        ``job_id`` names the queued job the lane is being taken FOR; an
        inline turn (a message that never had to wait) passes nothing. It is
        what lets ``pending_count`` tell "waiting" from "being answered right
        now" — a queued job stays on disk for the whole of its turn, since
        removing it earlier is what would lose it.

        Deliberately synchronous and await-free between the check and the
        set: that is exactly what makes it atomic against concurrently
        scheduled handler tasks, the same property ``chat.py``'s media claim
        and ``inbox.claim`` rely on. aiogram runs every update as its own
        task, so two messages arriving in the same second BOTH reach this
        function before either reaches ``ask()``.
        """
        held = self._running.get(int(chat_id))
        if held is not None:
            if self._monotonic() - held[0] < LANE_TTL:
                return False
            logger.warning(
                "chat-queue: stale lane for chat %s (%.0fs old) — reclaiming",
                chat_id,
                self._monotonic() - held[0],
            )
        self._running[int(chat_id)] = (self._monotonic(), job_id)
        return True

    def release(self, chat_id: int) -> None:
        """Give the lane back. Idempotent — safe in a ``finally`` that may
        run twice."""
        self._running.pop(int(chat_id), None)

    def is_running(self, chat_id: int) -> bool:
        return int(chat_id) in self._running

    def running_job(self, chat_id: int) -> str | None:
        """The queued job this chat's lane is running, if it is running one."""
        held = self._running.get(int(chat_id))
        return held[1] if held is not None else None

    def _running_jobs(self) -> set[str]:
        return {job_id for _since, job_id in self._running.values() if job_id}

    # ── queue operations ─────────────────────────────────────────────

    def enqueue(
        self,
        chat_id: int,
        user_id: int,
        prompt: str,
        *,
        message_id: int | None = None,
        arrived_ns: int | None = None,
    ) -> tuple[Job, int]:
        """Park one prepared message. Returns ``(job, position)``.

        ``position`` is this job's actual PLACE in the line — how many of
        the chat's waiting messages will be answered before it, plus one —
        not simply how many are waiting. The two differ exactly when a
        message is parked out of arrival order (the lane-holder, parked last
        but stamped first), and telling that message "в очереди: 2" when it
        is about to be answered first would be a small lie in the one line
        this feature exists to make trustworthy.

        Raises ``QueueFull`` when the chat is already at ``max_waiting`` —
        the caller says so out loud; nothing is ever dropped quietly.

        ``arrived_ns`` is when the message reached the bot, not when it got
        here — see ``_new_id``.

        Fully synchronous on purpose: with no ``await`` between counting and
        writing, the limit cannot be raced past on a single event loop.
        """
        running = self.running_job(chat_id)
        ahead = [job for job in self.waiting(chat_id) if job.id != running]
        if self.max_waiting > 0 and len(ahead) >= self.max_waiting:
            raise QueueFull(f"chat {chat_id} already has {len(ahead)} waiting")
        job = Job(
            id=self._new_id(arrived_ns),
            chat_id=int(chat_id),
            user_id=int(user_id),
            prompt=prompt,
            queued_at=self.now(),
            message_id=int(message_id) if isinstance(message_id, int) else None,
        )
        write_json_atomic(self._path(job.id), _job_payload(job))
        return job, 1 + sum(1 for other in ahead if other.id < job.id)

    def waiting(self, chat_id: int | None = None) -> list[Job]:
        """Everything still waiting, oldest first; one chat's if asked.

        A file that will not parse is moved to ``stale/`` rather than retried
        forever or deleted — same contract as both sibling queues.
        """
        try:
            names = sorted(p.name for p in self.dir.glob("*.json"))
        except OSError:
            return []
        jobs: list[Job] = []
        for name in names:
            path = self.dir / name
            try:
                job = _job_from_dict(json.loads(path.read_text()))
            except FileNotFoundError:
                continue
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.error("chat-queue: unreadable job %s (%s)", name, exc)
                self._quarantine(name, str(exc))
                continue
            if chat_id is None or job.chat_id == int(chat_id):
                jobs.append(job)
        return jobs

    def pending_count(self, chat_id: int) -> int:
        """How many messages this chat still has WAITING — the number the
        human is told, and the one the ceiling is measured against.

        The job a runner is currently answering is deliberately left out
        although it is still on disk: it is removed only once its turn
        finished (that is what makes this at-least-once), so counting it
        would tell the next sender "в очереди: 2" when exactly one message
        is ahead of him and it is already being answered (blind review 3).
        """
        running = self.running_job(chat_id)
        jobs = self.waiting(chat_id)
        return len([job for job in jobs if job.id != running])

    def ready(self) -> list[Job]:
        """The head job of every chat whose lane is free, oldest first.

        One job per chat, because the whole point is one turn at a time; the
        rest of that chat's queue waits for the next pass.
        """
        heads: dict[int, Job] = {}
        for job in self.waiting():
            if job.chat_id in self._running or job.chat_id in heads:
                continue
            heads[job.chat_id] = job
        return sorted(heads.values(), key=lambda j: j.id)

    def expired(self, job: Job, *, now: float | None = None) -> bool:
        if self.max_age <= 0:
            return False
        moment = self.now() if now is None else now
        return (moment - job.queued_at) > self.max_age

    def record_attempt(self, job: Job) -> Job:
        """Persist one more failed attempt, so a job that kills its turn
        cannot be retried forever across restarts."""
        bumped = Job(**{**_job_payload(job), "attempts": job.attempts + 1})
        write_json_atomic(self._path(job.id), _job_payload(bumped))
        return bumped

    def done(self, job: Job) -> None:
        """Remove a finished job. Idempotent; a failure is logged, never
        raised — the worst case is one duplicate answer after a restart, and
        that is strictly better than an answer that never comes."""
        try:
            self._path(job.id).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("chat-queue: could not drop %s: %s", job.id, exc)

    def retire(self, job: Job, *, reason: str) -> None:
        """Move a job to ``stale/`` with the reason attached."""
        payload = _job_payload(job)
        payload["reason"] = reason
        payload["retired_at"] = self.now()
        try:
            write_json_atomic(self.stale_dir / f"{job.id}.json", payload)
        except OSError as exc:
            logger.warning("chat-queue: could not retire %s: %s", job.id, exc)
            return
        self.done(job)
        logger.warning(
            "chat-queue: giving up on %s (chat %s) — %s", job.id, job.chat_id, reason
        )

    def _quarantine(self, name: str, reason: str) -> None:
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
            logger.warning("chat-queue: could not quarantine %s: %s", name, exc)

    # ── reporting ────────────────────────────────────────────────────

    def stats(self, *, now: float | None = None) -> Stats:
        """Best effort by contract — this feeds a status report.

        Deliberately does NOT reuse ``waiting()``: that one quarantines a
        file it cannot parse, and a read-only status command must not mutate
        the queue it is describing (the outbox's blind review, S5).

        A job being answered right now is counted under ``running``, never
        under ``waiting`` — the same one-message-one-place rule
        ``pending_count`` keeps, so ``/work`` cannot show a single message
        twice.
        """
        moment = self.now() if now is None else now
        try:
            paths = list(self.dir.glob("*.json"))
        except OSError:
            paths = []
        in_flight = self._running_jobs()
        per_chat: dict[int, int] = {}
        oldest = 0.0
        for path in paths:
            if path.stem in in_flight:
                continue
            try:
                raw = json.loads(path.read_text())
                chat_id = int(raw["chat_id"])
                queued_at = float(raw.get("queued_at", 0.0))
            except (OSError, ValueError, KeyError, TypeError):
                continue
            per_chat[chat_id] = per_chat.get(chat_id, 0) + 1
            if queued_at:
                oldest = max(oldest, moment - queued_at)
        try:
            stale = len(list(self.stale_dir.glob("*.json")))
        except OSError:
            stale = 0
        return Stats(
            waiting=sum(per_chat.values()),
            running=len(self._running),
            stale=stale,
            oldest_age=max(0.0, oldest),
            per_chat=per_chat,
        )


# ── the worker ───────────────────────────────────────────────────────
#
# ``runner`` is ``chat.run_queued_job``, passed in rather than imported, so
# this module stays free of aiogram handlers and the handlers stay free of a
# background task. It returns True when the job is FINISHED (answered, or
# answered with an error the user can see) and False when the turn could not
# start yet — the session is busy with live work and the job keeps its place.

RETIRED_NOTICE = (
    "⚠️ Сообщение из очереди так и не дошло до основной сессии ({reason}). "
    "Оно сохранено, но ответа по нему не будет — повтори его, пожалуйста."
)


async def drain(
    bot: Any,
    queue: ChatQueue,
    runner: Any,
    *,
    should_start: Any = None,
    turns: Any = None,
) -> None:
    """One pass: start the oldest job of every chat whose lane is free.

    Never raises. Sequential by construction — the jobs it starts all end up
    behind the same process-wide ask-lock anyway, so running them one after
    another is honest about the parallelism that exists (none) instead of
    piling turns up inside the engine.

    ``should_start`` is the graceful stop's veto: once a stop is requested,
    starting a brand-new twenty-minute turn that the deadline will cut off in
    a few seconds helps nobody. Same reasoning ``run_bot`` gives for taking
    cron down first.

    ``turns`` is the other half of that, and it is why each job runs as its
    own task rather than being awaited inline. It is the graceful stop's
    registry (``shutdown.Shutdown``), which otherwise only ever hears about
    aiogram's update tasks — so a queued turn, the one kind of turn nobody
    is holding an update open for, was the only turn in the process that got
    NO grace at all and was killed where it stood (blind review 2). Tracked,
    it gets the same deadline every other in-flight turn gets, and
    ``wait_for_turns`` re-reads the set every pass so registering one mid-stop
    is safe.
    """
    try:
        jobs = queue.ready()
    except Exception:  # noqa: BLE001 — a bad read must never kill the worker
        logger.exception("chat-queue: could not read the queue")
        return
    for job in jobs:
        if should_start is not None and not should_start():
            return
        if not queue.try_acquire(job.chat_id, job_id=job.id):
            continue
        task = asyncio.ensure_future(_run_one(bot, queue, runner, job))
        if turns is not None:
            turns.track(task)
        try:
            await task
        except asyncio.CancelledError:
            # Shutdown, or the worker being torn down. The job was NOT
            # finished, so it keeps its place on disk and the next boot
            # answers it.
            raise
        except Exception:  # noqa: BLE001 — one bad job must not strand the rest
            logger.exception("chat-queue: job %s failed", job.id)
            await _count_failure(bot, queue, job)
        finally:
            if turns is not None:
                turns.untrack(task)
            queue.release(job.chat_id)


async def _run_one(bot: Any, queue: ChatQueue, runner: Any, job: Job) -> None:
    if queue.expired(job):
        queue.retire(job, reason=f"waited longer than {queue.max_age:.0f}s")
        await _tell(bot, job, RETIRED_NOTICE.format(reason="слишком долго ждало"))
        return
    if await runner(bot, job):
        queue.done(job)


async def _count_failure(bot: Any, queue: ChatQueue, job: Job) -> None:
    """Fold one blown-up turn into the job, and retire it past the cap."""
    try:
        bumped = queue.record_attempt(job)
    except OSError:
        # Cannot count the attempt ⇒ cannot promise this stops. Retiring is
        # the safe direction, exactly as ``inbox.replay`` decides it.
        logger.exception("chat-queue: could not record an attempt for %s", job.id)
        queue.retire(job, reason="attempt counter unwritable")
        await _tell(bot, job, RETIRED_NOTICE.format(reason="очередь не пишется"))
        return
    if bumped.attempts >= MAX_ATTEMPTS:
        queue.retire(bumped, reason=f"{bumped.attempts} failed attempt(s)")
        await _tell(
            bot, job, RETIRED_NOTICE.format(reason=f"{bumped.attempts} сбоя подряд")
        )


async def retire_leftovers(bot: Any, runtime_dir: Path | str, *, reason: str) -> int:
    """Empty a queue nobody is going to drain, telling every affected chat.

    Called at boot when the queue is turned OFF. Without it the rollback
    switch would be a silent broken promise: the messages a previous process
    parked were already answered with "отвечу следом", nothing would ever
    pick them up again, and even the age cap could not save them — retiring
    lives inside ``drain``, which is exactly what is not running. Rule B3
    cuts both ways; a promise that can no longer be kept has to be withdrawn
    out loud.

    Never raises: a bot that refuses to boot over this would be worse than
    the thing it is cleaning up after.
    """
    try:
        queue = ChatQueue(runtime_dir)
        jobs = queue.waiting()
    except Exception:  # noqa: BLE001
        logger.exception("chat-queue: could not read a disabled queue")
        return 0
    for job in jobs:
        try:
            queue.retire(job, reason=reason)
        except Exception:  # noqa: BLE001 — one bad job must not strand the rest
            logger.exception("chat-queue: could not retire %s", job.id)
            continue
        await _tell(bot, job, RETIRED_NOTICE.format(reason=reason))
    if jobs:
        logger.warning(
            "chat-queue: withdrew %d parked message(s) — %s", len(jobs), reason
        )
    return len(jobs)


async def _tell(bot: Any, job: Job, text: str) -> None:
    """One honest line to the chat. Never raises — a courtesy notice must not
    take the worker down with it."""
    from d_brain.bot.formatters import send_response

    try:
        await send_response(bot, job.chat_id, text, reply_to=job.message_id)
    except Exception:  # noqa: BLE001
        logger.warning("chat-queue: could not deliver a notice", exc_info=True)


async def run(
    bot: Any,
    queue: ChatQueue,
    runner: Any,
    *,
    poll_seconds: float = POLL_SECONDS,
    should_start: Any = None,
    turns: Any = None,
) -> None:
    """The one background task: drain the queue forever.

    Its first pass runs immediately, which is what makes a restart pick up
    whatever the previous process had parked instead of stranding it.

    The belt over ``drain``'s braces: ``drain`` is written not to raise, but
    the lane calls at its edges are outside its own ``try``. If one ever did
    escape, this loop would end, its task would die with an unread exception
    and EVERY parked message would hang forever — after the owner was told
    "отвечу следом". A worker that logs and keeps sweeping is the only shape
    that keeps that promise (blind review 8).
    """
    logger.info("chat-queue worker started (%s, poll %.0fs)", queue.dir, poll_seconds)
    while True:
        try:
            await drain(bot, queue, runner, should_start=should_start, turns=turns)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — see the docstring
            logger.exception("chat-queue: drain pass failed; sweeping again")
        await asyncio.sleep(poll_seconds)


# ── process-wide handle ──────────────────────────────────────────────
#
# Same pattern, same three functions as ``inbox`` and ``outbox``: the chat
# handlers take no settings, and the queue is one piece of bot-process
# runtime state. ``current()`` returning None is the "no queue configured"
# path — unit tests and one-off tools — where ``chat.py`` behaves exactly as
# it did before this step, duty session included.

_current: ChatQueue | None = None


def configure(
    runtime_dir: Path | str,
    *,
    max_waiting: int = DEFAULT_MAX_WAITING,
    max_age: float = DEFAULT_MAX_AGE,
) -> ChatQueue:
    """Install the process-wide queue. Called once, from ``run_bot``."""
    global _current  # noqa: PLW0603
    _current = ChatQueue(runtime_dir, max_waiting=max_waiting, max_age=max_age)
    return _current


def current() -> ChatQueue | None:
    """The configured queue, or None outside the bot process."""
    return _current


def reset() -> None:
    """Drop the process-wide queue. For tests — the bot never unconfigures
    itself."""
    global _current  # noqa: PLW0603
    _current = None
