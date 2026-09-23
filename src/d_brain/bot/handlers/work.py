"""``/work`` command handler — "what is running right now" (item 29).

A pure READ of state that other components already maintain on disk or in
the engine: the session's own busy flags, the watchdog's ``long-run.json``
marker, the cron store's ``jobs.json``, and the ask-path health ledger. It
answers the owner's three questions — what is in flight, how long it has
been going, and where something is stuck.

Two hard constraints shape the whole module:

* **It must work precisely when the session is busy.** So, like
  ``/resend``, it never calls the model and never takes the ask-lock: the
  moment this command needs the lock it stops being usable in the one
  situation it was built for. Everything it reads is either a file or a
  non-blocking engine probe.
* **It must survive both engines.** It speaks only the ``EngineDriver``
  Protocol (``is_turn_active`` / ``is_pane_turn_active``) plus the
  cross-process ledgers, and imports nothing tmux-shaped — under the Codex
  engine the same calls answer about a live exec process instead of a pane,
  and the report reads the same.

Every single reading is best-effort: a missing file, an unreadable marker
or an engine that refuses to answer produces an honest line ("состояние
недоступно", "неизвестно"), never an exception and never a confident lie.
Absence of evidence is reported as absence of evidence — the same contract
``ask_health.read`` and ``long_run.read`` keep.
"""

import asyncio
import html
import logging
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from d_brain.bot.formatters import send_response
from d_brain.config import Settings, get_settings
from d_brain.services import (
    ask_health,
    chat_queue,
    inbox,
    long_run,
    outbox,
    runtime,
)
from d_brain.services.claude_session import MAINT_PREFIX
from d_brain.services.cron_store import CronJob, CronStore

# MAINT_PREFIX above is one CONSTANT, not behavior: `maint-` is the
# request-id prefix BOTH engine drivers write into `inflight` for a
# maintenance turn, so it is part of the cross-process file format this
# module already reads — not engine-specific machinery. Copying the literal
# here instead would be a second source of truth for a format whose whole
# point is that two processes agree on it.

router = Router(name="work")
logger = logging.getLogger(__name__)

# How stale the watchdog's long-run.json may be before this report stops
# trusting it. Deliberately a local copy of claude_session's
# DEFAULT_LONG_RUN_STALE_AFTER (120s) rather than an import: this handler
# must not depend on the Claude-only session module, and the only cost of
# the duplication is that a marker is called "unknown" a little sooner or
# later than ask() would — a wording difference in a status report.
_MARKER_STALE_AFTER = 120.0
# Past this distance a "через N" / "N назад" clause stops being information
# (see _fmt_when).
_RELATIVE_HORIZON = 30 * 24 * 3600.0
# A cron job parked on the subscription limit is NOT a stuck job — the run
# did not fail, it never started. cron_runner records it with count=False
# for exactly that reason, and ask_health treats the same status as neutral.
# It gets its own line instead of being counted as a failure (F9).
_RATE_LIMITED = "rate_limited"


def _fmt_duration(seconds: float) -> str:
    """Human, coarse, and never more precise than the source deserves."""
    total = int(max(0.0, seconds))
    if total < 60:
        return f"{total} сек"
    minutes, _ = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} мин"
    hours, rest = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} ч {rest:02d} мин"
    # Days, because "497204 ч" is not a number anyone reads — and a
    # corrupt or absurd next_run in jobs.json is exactly how that value
    # gets here.
    days, hours = divmod(hours, 24)
    return f"{days} дн {hours} ч"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


def _fmt_when(moment: datetime, *, tz: str, now: float) -> str:
    """``HH:MM DD.MM (через N)`` / ``(N назад)`` in the install's timezone."""
    try:
        local = moment.astimezone(ZoneInfo(tz))
    except Exception:  # noqa: BLE001 — a bad tz must not break a report
        local = moment
    stamp = local.strftime("%H:%M %d.%m")
    delta = moment.timestamp() - now
    if abs(delta) > _RELATIVE_HORIZON:
        # Beyond a month either way the relative phrasing adds nothing and
        # usually signals bad data — show the bare timestamp and let the
        # reader judge it.
        return stamp
    if delta >= 0:
        return f"{stamp} (через {_fmt_duration(delta)})"
    return f"{stamp} ({_fmt_duration(-delta)} назад)"


def _inflight_age(runtime_dir: Path, *, now: float) -> float | None:
    """Seconds since the ``inflight`` marker was last written, or None.

    Honest-source note: ``inflight``'s SECOND line is a timestamp from
    ``ClaudeSession``'s internal ``time.monotonic`` clock — that of the
    process that wrote it. Monotonic clocks are not comparable across
    processes (nor across reboots), so reading that number and subtracting
    it from our own clock would produce confident nonsense. The file's
    mtime is wall-clock, written by the same act, and is the only
    cross-process-honest answer to "since when".
    """
    try:
        return max(0.0, now - (runtime_dir / "inflight").stat().st_mtime)
    except OSError:
        return None


def _attended_turn_kind(runtime_dir: Path) -> str:
    """What the held turn actually IS, from ``inflight``'s FIRST line.

    Blind review F8: an attended turn was reported as "кто-то ждёт ответ"
    unconditionally, but the ask-lock is just as often held by the nightly
    ``/process`` pipeline, the doctor canary or a startup/recovery turn —
    nobody is waiting on any of those, and telling the owner otherwise is
    the wrong prompt to reach for ``/stop``.

    The first line is the turn's request id, written by BOTH engine drivers
    in the same format (``{log_id}\\n{clock}\\n``); a maintenance turn tags
    it with ``MAINT_PREFIX``. A held lock with NO inflight record is
    startup/recovery/control — ``ClaudeSession.is_steerable_turn`` reads the
    same file the same way and draws the same three distinctions.
    """
    try:
        first = (runtime_dir / "inflight").read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        return "чей — неизвестно (нет записи inflight)"
    if not first:
        return "чей — неизвестно (пустая запись inflight)"
    if first.startswith(MAINT_PREFIX):
        return "фоновое обслуживание (пайплайн/доктор/старт)"
    return "ход пользователя, кто-то ждёт ответ"


def _main_session_lines(settings: Settings, *, now: float) -> list[str]:
    lines = ["<b>🧠 Основная сессия</b>"]
    try:
        session = runtime.get_session(settings)
    except Exception:  # noqa: BLE001 — report, never raise
        logger.warning("/work: could not obtain the main session", exc_info=True)
        lines.append("• состояние недоступно — не смог получить сессию")
        return lines

    try:
        attended: bool | None = bool(session.is_turn_active())
    except Exception:  # noqa: BLE001
        logger.warning("/work: is_turn_active failed", exc_info=True)
        attended = None
    try:
        pane_active: bool | None = bool(session.is_pane_turn_active())
    except Exception:  # noqa: BLE001
        logger.warning("/work: is_pane_turn_active failed", exc_info=True)
        pane_active = None

    if attended is None and pane_active is None:
        lines.append("• состояние недоступно — сессия не отвечает на опрос")
        return lines

    if attended:
        age = _inflight_age(settings.runtime_dir, now=now)
        spent = (
            f"идёт {_fmt_duration(age)}"
            if age is not None
            else "сколько идёт — неизвестно"
        )
        kind = _attended_turn_kind(settings.runtime_dir)
        lines.append(f"• присмотренный ход — {kind}, {spent}")
        return lines

    if pane_active:
        active, elapsed = long_run.is_active(
            settings.runtime_dir, now=now, stale_after=_MARKER_STALE_AFTER
        )
        if active:
            lines.append(
                "• неприсмотренный ход (фоновая работа), "
                f"идёт {_fmt_duration(elapsed)}"
            )
            cap = settings.long_run_max_seconds
            if cap > 0:
                left = cap - elapsed
                if left > 0:
                    lines.append(f"• автозакрытие через {_fmt_duration(left)}")
                else:
                    lines.append("• лимит превышен — вотчдог уже закрывает этот ход")
            else:
                lines.append("• автозакрытие выключено")
        else:
            # The pane says busy but the watchdog marker is missing or
            # stale: either the run is younger than one watchdog tick, or
            # the watchdog is not running. Say so instead of inventing a
            # duration.
            lines.append(
                "• неприсмотренный ход (фоновая работа), "
                "сколько идёт — неизвестно (нет свежей отметки вотчдога)"
            )
        return lines

    if attended is None or pane_active is None:
        lines.append("• похоже, свободна (часть проверок не ответила)")
    else:
        lines.append("• свободна")
    return lines


def _duty_session_lines(settings: Settings) -> list[str]:
    """The duty session's line — printed only if this build HAS one.

    The duty session arrives on a separate branch (item 29's point 2). The
    lookup is deliberately a ``getattr`` rather than an import so this file
    works identically before and after that branch lands: no line at all in
    a build without it, no import error, no merge conflict between the two.
    """
    factory = getattr(runtime, "get_duty_session", None)
    if factory is None:
        return []
    try:
        duty = factory(settings)
        busy = bool(duty.is_turn_active()) or bool(duty.is_pane_turn_active())
    except Exception:  # noqa: BLE001
        logger.warning("/work: duty session probe failed", exc_info=True)
        return ["<b>🧰 Дежурная сессия</b>", "• состояние недоступно"]
    return ["<b>🧰 Дежурная сессия</b>", "• занята" if busy else "• свободна"]


def _job_is_failing(job: CronJob) -> bool:
    """A job whose LAST outcome was bad. ``ok-silent`` is a success (the
    run happened, it just had nothing to say), hence the prefix test.

    ``rate_limited`` is NOT a failure (blind review F9): the run never
    happened, nothing about the job is broken, and ``cron_runner`` itself
    records it with ``count=False``. It used to land under "где застряло",
    which cried wolf on the single most common non-``ok`` status there is.
    It gets its own honest line instead — see ``_rate_limited_line``.

    The ``consecutive_errors`` clause is guarded by the same early return:
    a job that errored earlier and is NOW parked on the limit would
    otherwise still be listed as stuck on the strength of that stale count.
    """
    status = job.state.last_status or ""
    if status == _RATE_LIMITED:
        return False
    if status and not status.startswith("ok"):
        return True
    return job.state.consecutive_errors > 0


def _cron_session_line(settings: Settings) -> str:
    """Busy/free for the isolated cron brain — the same two Protocol probes
    ``_main_session_lines`` and ``_duty_session_lines`` use, so it reads
    identically under either engine. A cron session that is busy explains a
    job that looks overdue, which is exactly the question this report is
    asked during one."""
    try:
        cron = runtime.get_cron_session(settings)
        busy = bool(cron.is_turn_active()) or bool(cron.is_pane_turn_active())
    except Exception:  # noqa: BLE001 — report, never raise
        logger.warning("/work: cron session probe failed", exc_info=True)
        return "• сессия расписания: состояние недоступно"
    return "• сессия расписания: " + ("занята" if busy else "свободна")


def _cron_lines(settings: Settings, *, now: float) -> list[str]:
    lines = ["<b>⏰ Расписание</b>", _cron_session_line(settings)]
    try:
        jobs = CronStore(settings.cron_dir).load()
    except Exception:  # noqa: BLE001
        logger.warning("/work: could not read the cron store", exc_info=True)
        lines.append("• состояние недоступно — не смог прочитать задания")
        return lines

    if not jobs:
        lines.append("• заданий нет")
        return lines

    enabled = [j for j in jobs if j.enabled]
    lines.append(f"• включено: {len(enabled)} из {len(jobs)}")

    upcoming = [
        (moment, job)
        for job in enabled
        if (moment := _parse_iso(job.state.next_run)) is not None
    ]
    if upcoming:
        moment, job = min(upcoming, key=lambda pair: pair[0])
        lines.append(
            f"• ближайшее: <code>{html.escape(job.id)}</code> — "
            f"{_fmt_when(moment, tz=settings.tz, now=now)}"
        )
    elif enabled:
        lines.append("• ближайшее: время следующего запуска неизвестно")

    limited = [j for j in enabled if (j.state.last_status or "") == _RATE_LIMITED]
    if limited:
        ids = ", ".join(f"<code>{html.escape(j.id)}</code>" for j in limited[:5])
        more = f" …и ещё {len(limited) - 5}" if len(limited) > 5 else ""
        lines.append(f"• крон стоит на лимите ({len(limited)}): {ids}{more}")

    # Only ENABLED jobs (F9). A disabled job keeps the last_status that got
    # it disabled forever, so counting those meant an old, already-handled
    # error hung in "где застряло" for the rest of the install's life.
    failing = [j for j in enabled if _job_is_failing(j)]
    if not failing:
        lines.append("• сбоев нет")
        return lines
    lines.append(f"• где застряло ({len(failing)}):")
    for job in failing[:5]:
        status = html.escape(job.state.last_status or "?")
        detail = f"<code>{html.escape(job.id)}</code> — {status}"
        if job.state.consecutive_errors:
            detail += f", подряд ошибок: {job.state.consecutive_errors}"
        if job.state.last_error:
            detail += f" ({html.escape(job.state.last_error[:120])})"
        lines.append(f"  · {detail}")
    if len(failing) > 5:
        lines.append(f"  · …и ещё {len(failing) - 5}")
    return lines


def _delivery_lines(settings: Settings, *, now: float) -> list[str]:
    """Two independent readings under one heading: how the last turns ENDED
    (the ask-health ledger) and what is still waiting to GO OUT (the durable
    outbox). The queue block is appended on every path — an unreadable
    ledger is no reason to stop reporting a backlog of undelivered replies,
    which is precisely the situation where both go wrong together."""
    return (
        ["<b>📨 Доставка ответов</b>"]
        + _ask_health_lines(settings, now=now)
        + _outbox_lines(settings, now=now)
    )


def _inbox_lines(settings: Settings, *, now: float) -> list[str]:
    """The durable inbox, read straight off disk — the answer to "написал,
    а ответа нет": how many messages were ACCEPTED and nobody is working on,
    and how old the oldest of them is.

    "Nobody is working on" is the load-bearing part. Messages this process
    has in flight are left out, because this very report runs inside its own
    claim window: counting them made every ``/work`` on a completely idle
    bot announce one stuck message, which is worse than printing nothing at
    all. Nothing is hidden by that — a turn actually running is already the
    subject of the "Основная сессия" block above. What is left here is a
    backlog nobody is touching, which is the only reading that should worry
    anyone. The retired pile is mentioned only when it is not empty; it is
    forensics, not a daily reading.
    """
    try:
        # The LIVE queue when there is one: only it knows what this process
        # is handling right now, and /work is itself one of those — a fresh
        # reader would count the /work update as a stuck message on every
        # single report (blind review 2).
        box = inbox.current() or inbox.Inbox(settings.runtime_dir)
        stats = box.stats(now=now)
    except Exception:  # noqa: BLE001 — stats() swallows its own, belt+braces
        logger.warning("/work: could not read the inbox", exc_info=True)
        return ["<b>📥 Приём сообщений</b>", "• состояние недоступно"]

    lines = ["<b>📥 Приём сообщений</b>"]
    if stats.pending:
        age = (
            f", самому старому {_fmt_duration(stats.oldest_age)}"
            if stats.oldest_age
            else ""
        )
        lines.append(f"• принято, без ответа: {stats.pending}{age}")
    else:
        lines.append("• принято, без ответа: нет")
    if stats.stale:
        lines.append(
            f"• не доигрывалось (слишком старое): {stats.stale} — "
            f"<code>{html.escape(str(box.stale_dir))}</code>"
        )
    return lines


def _chat_queue_lines(settings: Settings, *, now: float) -> list[str]:
    """The per-chat queue (item 33, step 5): what has been acknowledged with
    "отвечу следом" and is still waiting for its turn.

    Reads the LIVE queue when there is one — only it knows which chats have a
    runner right now; a fresh reader off the same directory would report
    "ничего не идёт" while a turn is plainly running. Best effort like every
    other reading here: an unreadable queue is reported as unreadable, never
    as empty.
    """
    try:
        lane = chat_queue.current()
        if lane is None:
            lane = chat_queue.ChatQueue(settings.runtime_dir)
        stats = lane.stats(now=now)
    except Exception:  # noqa: BLE001 — stats() swallows its own, belt+braces
        logger.warning("/work: could not read the chat queue", exc_info=True)
        return ["<b>⏳ Очередь по чату</b>", "• состояние недоступно"]

    lines = ["<b>⏳ Очередь по чату</b>"]
    if stats.waiting:
        age = (
            f", самое старое ждёт {_fmt_duration(stats.oldest_age)}"
            if stats.oldest_age
            else ""
        )
        lines.append(f"• ждут в очереди: {stats.waiting}{age}")
        for chat_id, count in sorted((stats.per_chat or {}).items()):
            lines.append(f"  · чат <code>{html.escape(str(chat_id))}</code>: {count}")
    else:
        lines.append("• ждут в очереди: нет")
    lines.append(f"• разбирается прямо сейчас: {stats.running}")
    if stats.stale:
        lines.append(
            f"• брошено без ответа: {stats.stale} — "
            f"<code>{html.escape(str(lane.stale_dir))}</code>"
        )
    return lines


def _ask_health_lines(settings: Settings, *, now: float) -> list[str]:
    try:
        health = ask_health.read(settings.runtime_dir)
    except Exception:  # noqa: BLE001 — read() already swallows its own, belt+braces
        logger.warning("/work: could not read ask health", exc_info=True)
        return ["• состояние недоступно"]

    if not health.last_status:
        return ["• записей пока нет"]
    stamp = ""
    if health.last_ts:
        stamp = f", {_fmt_duration(max(0.0, now - health.last_ts))} назад"
    lines = [
        f"• последний статус: <code>{html.escape(health.last_status)}</code>{stamp}"
    ]
    if health.fail_streak:
        span = health.streak_span
        tail = f" за {_fmt_duration(span)}" if span else ""
        lines.append(f"• подряд неудач: {health.fail_streak}{tail}")
    else:
        lines.append("• серия неудач: нет")
    return lines


def _outbox_lines(settings: Settings, *, now: float) -> list[str]:
    """The durable queue, read straight off disk — the answer to "ответ
    родился, но не дошёл": how many replies are still waiting, and how many
    were given up on entirely.

    Same best-effort contract as every other reading here: a queue we cannot
    read is reported as unreadable, never as empty.
    """
    try:
        box = outbox.Outbox(settings.runtime_dir)
        stats = box.stats(now=now)
    except Exception:  # noqa: BLE001 — stats() swallows its own, belt+braces
        logger.warning("/work: could not read the outbox", exc_info=True)
        return ["• очередь отправки: состояние недоступно"]

    lines = []
    if stats.pending:
        age = (
            f", самой старой {_fmt_duration(stats.oldest_age)}"
            if stats.oldest_age
            else ""
        )
        lines.append(f"• ждут отправки: {stats.pending}{age}")
    else:
        lines.append("• ждут отправки: нет")
    if stats.dead:
        lines.append(
            f"• не доставлено совсем: {stats.dead} — "
            f"<code>{html.escape(str(box.dead_dir))}</code>"
        )
    else:
        lines.append("• потерянных ответов нет")
    return lines


def build_work_report(settings: Settings, *, now: float | None = None) -> str:
    """The whole report as Telegram HTML. Synchronous and blocking on
    purpose — the caller runs it in a thread; see ``cmd_work``."""
    moment = time.time() if now is None else now
    blocks: list[list[str]] = [
        _main_session_lines(settings, now=moment),
        _duty_session_lines(settings),
        _cron_lines(settings, now=moment),
        _inbox_lines(settings, now=moment),
        _chat_queue_lines(settings, now=moment),
        _delivery_lines(settings, now=moment),
    ]
    body = "\n\n".join("\n".join(block) for block in blocks if block)
    return f"{body}\n\n<i>Прервать текущий ход — /stop</i>"


@router.message(Command("work"))
async def cmd_work(message: Message) -> None:
    """Report what is running right now. No model call, no ask-lock.

    Delivery goes through ``send_response`` (blind review F10), the same
    path every chat reply uses: it sanitizes the HTML, falls back to plain
    text if Telegram rejects the markup, and splits at 4096 characters. This
    report is assembled from unbounded external data — cron job ids and
    ``last_error`` strings — so "it is always short and always valid" was an
    assumption, not a fact. The send sits INSIDE the same ``try``: a status
    command that raises out of its handler because the report was one
    character too long is the failure mode it exists to prevent.

    Since the outbox landed, ``send_response`` no longer raises when
    TELEGRAM refuses the message — that is now a queued retry, not this
    handler's problem. The ``try`` still covers the build, and it still
    covers a send that fails before anything can be queued, so the failure
    notice below is a narrower backstop than it was, not a dead branch.
    """
    settings = get_settings()
    try:
        # Off the event loop wholesale: is_pane_turn_active() does a tmux
        # capture under the Claude engine, and the ledger reads hit the
        # disk. None of it may stall the bot's polling loop.
        text = await asyncio.to_thread(build_work_report, settings)
        await send_response(message.bot, message.chat.id, text)
    except Exception:  # noqa: BLE001 — a status command must not 500 silently
        logger.exception("/work: report build or delivery failed")
        try:
            await message.answer("❌ Не смог собрать статус работ прямо сейчас.")
        except Exception:  # noqa: BLE001 — nothing left to try
            logger.exception("/work: could not deliver the failure notice")
