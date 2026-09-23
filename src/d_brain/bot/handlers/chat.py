"""Unified private chat handler with persistent Claude sessions.

Voice + text only (v3.0): replaces the legacy split handlers for private chats.
Every message is saved to daily (safety net) and routed IMMEDIATELY through
ChatSessionManager for Claude to process and respond — no debounce buffer.
"""

import asyncio
import contextlib
import html
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.types import Message

from d_brain import logsafe
from d_brain.bot.formatters import send_response
from d_brain.config import get_settings
from d_brain.services import chat_queue, progress
from d_brain.services.chat_session import (
    Busy,
    ChatSessionManager,
    busy_fallback_message,
)
from d_brain.services.claude_session import DEFAULT_TIMEOUT
from d_brain.services.media_prep import MediaPrep, prepare_for_model
from d_brain.services.session import SessionStore
from d_brain.services.storage import VaultStorage
from d_brain.services.transcription import DeepgramTranscriber

router = Router(name="chat")
logger = logging.getLogger(__name__)

# Only handle private chats
router.message.filter(F.chat.type == ChatType.PRIVATE)

MAX_RESPONSE_LENGTH = 4096

# Slash commands split by BEHAVIOR, not by the leading "/":
# - control: client-side Claude Code commands — no model turn, fire-and-forget
# - tui: interactive full-screen UIs — undrivable through a typed pane
# - everything else (incl. /skill-name) is a normal model turn → marker path
# /compact is NOT here: commands.router (registered earlier) intercepts it.
_CONTROL = {"/clear", "/model"}
_TUI_ONLY = {"/agents", "/config", "/login"}

_manager: ChatSessionManager | None = None


def classify_command(text: str) -> str:
    """'control' | 'tui' | 'turn' for an incoming chat text."""
    if not text.startswith("/"):
        return "turn"
    head = text.split(maxsplit=1)[0]
    if head in _CONTROL:
        return "control"
    if head in _TUI_ONLY:
        return "tui"
    return "turn"


_STOP_WORDS = {"/stop", "stop", "стоп"}

# The pre-duty-session wordings, kept verbatim: they are what the user sees
# whenever the duty path is disabled, unavailable or fails.
_MAINTENANCE_MESSAGE = (
    "🔧 Идёт фоновое обслуживание — повтори сообщение через "
    "несколько минут, отвечу как освобожусь."
)
_BUSY_FALLBACK_MESSAGE = busy_fallback_message()


def classify_concurrent_input(text: str, turn_active: bool) -> str:
    """'ask' | 'steer' | 'interrupt' — what to do with input that arrives
    while the agent may be busy. Plain text during an active turn STEERS it
    (injected mid-turn); a stop word interrupts; otherwise a normal turn."""
    if not turn_active:
        return "ask"
    if text.strip().lower() in _STOP_WORDS:
        return "interrupt"
    return "steer"


def _get_manager() -> ChatSessionManager:
    """Lazy-init ChatSessionManager singleton."""
    global _manager  # noqa: PLW0603
    if _manager is None:
        settings = get_settings()
        _manager = ChatSessionManager(settings.vault_path)
    return _manager


async def _dispatch_text(
    bot: Bot,
    chat_id: int,
    user_id: int,
    text: str,
    *,
    message_id: int | None = None,
) -> None:
    """Route a text by behavior: control → fire-and-forget; tui → hint;
    normal turn (incl. /skill-name) → session via the marker path."""
    kind = classify_command(text)
    if kind == "control":
        await _get_manager().send_control(text)
        await bot.send_message(
            chat_id, f"⌨️ <code>{html.escape(text)}</code> отправлена в сессию."
        )
        return
    if kind == "tui":
        await bot.send_message(
            chat_id,
            "Эта команда открывает интерактивный интерфейс — доступно только "
            "через <code>dbrain attach</code> на сервере.",
        )
        return

    manager = _get_manager()
    turn_active = manager.is_turn_active()
    if (
        not turn_active
        and text.strip().lower() in _STOP_WORDS
        and manager.is_pane_turn_active()
    ):
        # Step D: during an unattended long
        # cascade the ask-lock is free (nothing called ask() for this turn)
        # while the PANE itself is genuinely busy — classify_concurrent_input
        # only ever sees the lock-based is_turn_active(), so a stop-word here
        # used to fall through to the normal ask() busy path instead of
        # actually interrupting anything, leaving the user with no way to
        # reclaim the channel. classify_concurrent_input itself stays a pure
        # function — the pane check lives here, at the call site.
        await manager.interrupt()
        await bot.send_message(chat_id, "⏹ Останавливаю текущий ответ.")
        return
    mode = classify_concurrent_input(text, turn_active)
    if mode == "interrupt":
        await manager.interrupt()
        await bot.send_message(chat_id, "⏹ Останавливаю текущий ответ.")
        return
    if mode == "steer":
        if not manager.is_steerable_turn():
            # Maintenance turn (nightly pipeline / doctor / startup) holds
            # the session — injecting user text would contaminate it. Falls
            # through to the ordinary path, which parks it in the per-chat
            # queue; the "held by maintenance" check lives
            # at that single choke point so voice and media get it too (the
            # F6 lesson), not here.
            await _process_and_reply(bot, chat_id, user_id, text, message_id=message_id)
            return
        # A STEERABLE chat turn is the one busy case the queue deliberately
        # does NOT take: steering hands the text to the turn that is running
        # right now, which is strictly better than making it wait for a turn
        # of its own. "One work at a time per chat" still holds — steering
        # adds input to the live work, it does not start a second one.
        await manager.steer(text)
        await bot.send_message(chat_id, "↪️ Передал в текущую задачу.")
        return
    # mode == "ask". The busy gate used to live right here — it now lives
    # inside _run_turn, behind the single _process_and_reply choke point
    # (blind review F6), so that voice, media and album messages pass through
    # it too.
    await _process_and_reply(bot, chat_id, user_id, text, message_id=message_id)


async def _main_is_busy(manager: Any) -> bool:
    """Pre-ask busy gate — never fatal: a manager without the probe (or one
    that throws) means "not busy", i.e. today's path into ask()."""
    probe = getattr(manager, "is_main_busy", None)
    if probe is None:
        return False
    try:
        return bool(await probe())
    except Exception:  # noqa: BLE001
        logger.warning("main-busy probe failed", exc_info=True)
        return False


async def _reply_from_duty(
    bot: Bot,
    chat_id: int,
    user_id: int,
    text: str,
    *,
    fallback: str | None = None,
    busy_seconds: float | None = None,
) -> None:
    """Deliver an answer from the duty session (typing indicator included —
    the duty turn takes real time, just less of it)."""
    # One synchronous ping before the loop task: on this path the user has
    # already been waiting on a busy main session, so the indicator must
    # appear immediately rather than whenever the loop first gets scheduled.
    try:
        await bot.send_chat_action(chat_id, "typing")
    except Exception:  # noqa: BLE001 — an indicator is never worth a reply
        logger.debug("typing indicator failed", exc_info=True)
    typing_task = asyncio.create_task(_typing_loop(bot, chat_id))
    try:
        response = await _get_manager().answer_from_duty(
            user_id, text, busy_seconds=busy_seconds, fallback=fallback
        )
    except Exception:  # noqa: BLE001 — the fallback text is always deliverable
        logger.exception("duty reply failed for user %d", user_id)
        response = fallback or _BUSY_FALLBACK_MESSAGE
    finally:
        typing_task.cancel()
    await send_response(bot, chat_id, response or _BUSY_FALLBACK_MESSAGE)


@dataclass(frozen=True)
class _Later:
    """ "Not now" — the main session is working on something else.

    Carries the exact wording the user would have received before the
    per-chat queue existed, so an install WITHOUT a configured queue (unit
    tests, one-off tools) lands byte-for-byte on the old duty-session path.
    """

    fallback: str | None = None
    busy_seconds: float | None = None


def queued_ack(position: int) -> str:
    """The line the human gets the instant his message is parked.

    Deliberately makes a PROMISE ("отвечу следом") — the first wording in
    this file that is allowed to, and only because there is finally
    something on disk that keeps it. Every earlier busy message in this
    codebase was reworded away from exactly this phrasing (rule B3,
    2026-08-22) precisely because nothing re-sent the answer later.
    """
    return f"✅ Принял — отвечу следом (в очереди: {position})."


def queue_full_message(limit: int) -> str:
    """Honest refusal at the ceiling. Says what happened, what was kept and
    what to do — never silence, never a lie that it will be answered."""
    return (
        f"🧺 В очереди уже {limit} сообщени(й) — больше не беру, иначе отвечу "
        "на них тогда, когда они уже никому не нужны. Это сообщение "
        "сохранено в дневнике, но ответа по нему не будет. Дождись ответов "
        "или прерви текущую работу — /stop."
    )


def _queue() -> Any | None:
    """The process-wide chat queue, or None outside the bot process."""
    try:
        return chat_queue.current()
    except Exception:  # noqa: BLE001 — never let the gate cost a reply
        logger.warning("chat-queue: could not read the process queue", exc_info=True)
        return None


def _held_by_maintenance(manager: Any) -> bool:
    """True iff the session is held by a turn that is NOT this chat's and
    cannot take input: the nightly pipeline, the doctor canary, a startup or
    recovery turn. Entering ``ask()`` there means blocking on the
    process-wide ask-lock for however long that turn runs, with a typing
    indicator and nothing else to show for it — on the voice and media paths
    that was minutes of total silence, because only the text path ever
    looked at ``is_steerable_turn``.

    Never fatal: a manager without either probe (or one that throws) reads
    as free, i.e. today's path into ``ask()``.
    """
    try:
        if not bool(manager.is_turn_active()):
            return False
        return not bool(manager.is_steerable_turn())
    except Exception:  # noqa: BLE001
        logger.warning("steerability probe failed", exc_info=True)
        return False


async def _park(
    bot: Bot,
    queue: Any,
    chat_id: int,
    user_id: int,
    prompt: str,
    message_id: int | None,
    *,
    arrived_ns: int,
    later: _Later | None = None,
) -> None:
    """Put a prepared message on disk and acknowledge it at once.

    ``arrived_ns`` is stamped by the caller BEFORE it touches the lane, and
    it is what orders the queue. Stamping here instead would invert it: the
    message that took the lane learns the session is busy only after
    ``ask()``'s busy-wait (up to five minutes), so it is parked last although
    it came first — see ``chat_queue.ChatQueue._new_id``.

    A queue that cannot be WRITTEN falls back to the pre-queue behavior (the
    duty session) rather than losing the message: durability is an upgrade
    over answering, never a precondition for it — the same call the outbox's
    blind review made, B2.
    """
    try:
        _job, position = queue.enqueue(
            chat_id,
            user_id,
            prompt,
            message_id=message_id,
            arrived_ns=arrived_ns,
        )
    except chat_queue.QueueFull:
        logger.warning("chat-queue: chat %s is at its waiting limit", chat_id)
        await send_response(bot, chat_id, queue_full_message(queue.max_waiting))
        return
    except Exception:  # noqa: BLE001 — see the docstring
        logger.exception("chat-queue: could not park a message for chat %s", chat_id)
        fallback = later.fallback if later is not None else None
        seconds = later.busy_seconds if later is not None else None
        await _reply_from_duty(
            bot, chat_id, user_id, prompt, fallback=fallback, busy_seconds=seconds
        )
        return
    await send_response(bot, chat_id, queued_ack(position))


async def run_queued_job(bot: Bot, job: Any) -> bool:
    """Run one job off the queue. ``True`` ⇒ it is finished, drop it.

    ``False`` means the turn could not START — the session is busy with live
    work — so the job keeps its place and the worker tries again next pass.
    It deliberately runs the SAME ``_run_turn`` a fresh message does: same
    typing indicator, same progress card, same retry-on-empty, same error
    handling. A second code path here would drift.
    """
    later = await _run_turn(
        bot,
        job.chat_id,
        job.user_id,
        job.prompt,
        job.message_id,
        from_queue=True,
    )
    return later is None


async def _process_and_reply(
    bot: Bot,
    chat_id: int,
    user_id: int,
    prompt: str,
    *,
    message_id: int | None = None,
) -> None:
    """THE choke point every input path funnels through — and therefore the
    place the per-chat lane is taken.

    One live runner per chat. A message that arrives while that lane is
    taken, or while the chat already has something waiting, is parked on disk
    and acknowledged immediately instead of racing the first one into
    ``ask()``'s lock. Checking "already waiting" and not just "lane taken" is
    what keeps the queue FIFO: without it, a message arriving in the gap
    between two drains would overtake everything ahead of it.

    The lane is in memory, so it cannot survive a restart — which is correct,
    a live runner cannot either. What survives is the waiting messages.

    With NO queue configured (unit tests, one-off tools) this is exactly the
    pre-queue function: straight into ``_run_turn``, and a "not now" outcome
    goes to the duty session with its old wording.
    """
    # WHEN THIS MESSAGE ARRIVED — taken once, here, before anything can
    # await. It is the queue's sort key, and the two ways it can be parked
    # are minutes apart: straight away (the lane is taken) or only after
    # ``_run_turn`` has sat through ``ask()``'s busy-wait. Stamping at
    # parking time would therefore put the FIRST message behind every
    # message that arrived while it was waiting.
    arrived_ns = time.time_ns()
    queue = _queue()
    if queue is not None:
        try:
            crowded = bool(queue.pending_count(chat_id))
        except Exception:  # noqa: BLE001 — a bad read must not cost the reply
            logger.warning("chat-queue: could not count waiting", exc_info=True)
            crowded = False
        # No ``await`` between the two checks and the acquire, so this is
        # atomic against concurrently scheduled handler tasks — the property
        # the media claim and ``inbox.claim`` rely on for the same reason.
        if crowded or not queue.try_acquire(chat_id):
            await _park(
                bot,
                queue,
                chat_id,
                user_id,
                prompt,
                message_id,
                arrived_ns=arrived_ns,
            )
            return
    try:
        later = await _run_turn(
            bot, chat_id, user_id, prompt, message_id, from_queue=False
        )
    finally:
        if queue is not None:
            queue.release(chat_id)
    if later is None:
        return
    if queue is not None:
        await _park(
            bot,
            queue,
            chat_id,
            user_id,
            prompt,
            message_id,
            arrived_ns=arrived_ns,
            later=later,
        )
        return
    await _reply_from_duty(
        bot,
        chat_id,
        user_id,
        prompt,
        fallback=later.fallback,
        busy_seconds=later.busy_seconds,
    )


async def _run_turn(
    bot: Bot,
    chat_id: int,
    user_id: int,
    prompt: str,
    message_id: int | None = None,
    *,
    from_queue: bool = False,
) -> _Later | None:
    """Send the prompt to the shared session and deliver the reply.

    Returns ``None`` when the message is finished with — answered, or
    answered with an error the user can read — and a ``_Later`` when the main
    session is busy with live work and this message has to wait.

    ``from_queue`` marks the retry of an already-parked message and turns the
    two cheap pre-``ask()`` probes OFF. That is deliberate and load-bearing:
    the parked message has already been acknowledged, nobody is staring at a
    typing indicator, and going into ``ask()`` is what produces the honest
    ``busy`` / ``busy_active`` split AND writes the ``ask_health`` row that
    keeps ``delivery_guard``'s restart backstop armed. A drain that took the
    cheap "still busy" shortcut instead would keep the ledger blind for as
    long as the wedge lasted — the exact hole blind-review F1 closed.

    THE BUSY GATE LIVES HERE, not in ``_dispatch_text`` (blind review F6).
    It was originally placed on the text path only, which left the system's
    PRIMARY input channel outside it: ``handle_chat_voice`` and the
    media/album handlers all call this function directly, so a voice message
    arriving during an unattended cascade paid the full
    ``DEFAULT_BUSY_WAIT_BUDGET`` (up to 300s of silence) before the duty
    session answered it — on a voice-first assistant, the exact case this
    whole feature was built for. One gate at the single choke point every
    input path already funnels through fixes all four at once and cannot
    drift out of sync the way four copies would.

    It runs ONCE, before the first attempt: the retry below is for an empty
    reply from an already-idle session, and re-probing there would add a
    second ~3s pane check to a user who is already waiting on a second turn.
    The media path's own claim mechanism (``reject_media_if_busy``) is
    untouched and still runs earlier — it guards a different race (two
    attachments landing in the same second), not a long unattended turn.
    """
    typing_task: asyncio.Task | None = None
    card: progress.ProgressCard | None = None
    card_task: asyncio.Task | None = None
    # Thread the answer under the message it answers — but only for a message
    # that WAITED. A fresh reply already follows its question directly, and
    # quoting it there is noise; an answer that arrives ten minutes later
    # needs to say which of several questions it belongs to.
    reply_to = message_id if from_queue else None
    try:
        # One synchronous ping BEFORE the gate (review round 3, R3). When the
        # marker is fresh but the second pane probe finds the turn finished,
        # the gate spends ~_MAIN_BUSY_CONFIRM_SECONDS and returns False — the
        # owner would otherwise stare at nothing for those 3 seconds before
        # the loop below starts. Same shape (and same never-worth-a-reply
        # guard) as _reply_from_duty's opening ping.
        try:
            await bot.send_chat_action(chat_id, "typing")
        except Exception:  # noqa: BLE001 — an indicator is never worth a reply
            logger.debug("typing indicator failed", exc_info=True)
        manager = _get_manager()
        if not from_queue:
            # The session is held by work that takes no input — nightly
            # pipeline, doctor canary, startup/recovery. ask() would sit on
            # the process-wide lock for as long as that runs.
            if _held_by_maintenance(manager):
                return _Later(fallback=_MAINTENANCE_MESSAGE)
            # An UNATTENDED long turn holds no ask-lock (nothing called ask()
            # for it), so the lock-based check above reads the session as
            # idle while the engine is genuinely working. Entering ask() here
            # would burn the whole busy-wait budget before we could say
            # anything.
            if await _main_is_busy(manager):
                return _Later()
        typing_task = asyncio.create_task(_typing_loop(bot, chat_id))
        # Живая карточка прогресса. Открывается ДО
        # отправки промпта, потому что её курсор по стенограмме встаёт на
        # конец файла в момент создания — иначе первые события хода
        # окажутся позади курсора и человек их не увидит.
        card, card_task = _start_progress_card(bot, chat_id, manager)
        response = await manager.send_message(user_id, prompt)
        # Карточка снимается ПЕРЕД ответом, а не после: ответ должен быть
        # последним сообщением в чате, а не встать над ещё живой карточкой.
        await _stop_progress_card(card, card_task)
        card, card_task = None, None

        # ``Busy`` is not a reply: the engine refused to type over a turn
        # that is demonstrably alive. Nothing is delivered and nothing is
        # said here — the caller parks the message and acknowledges it once.
        if isinstance(response, Busy):
            return _Later(
                fallback=response.fallback, busy_seconds=response.busy_seconds
            )

        if response:
            await send_response(bot, chat_id, response, reply_to=reply_to)
        else:
            logger.warning(
                "Empty response from Claude for user %d, retrying...", user_id
            )
            # Retry once before giving up — don't reset session on first empty
            # Повторный ход — такой же полноценный ход, который может идти
            # минутами; без своей карточки он проходил бы в полной тишине
            # (слепое ревью, находка 5).
            card, card_task = _start_progress_card(bot, chat_id, manager)
            response = await manager.send_message(user_id, prompt)
            await _stop_progress_card(card, card_task)
            card, card_task = None, None
            if isinstance(response, Busy):
                return _Later(
                    fallback=response.fallback, busy_seconds=response.busy_seconds
                )
            if response:
                await send_response(bot, chat_id, response, reply_to=reply_to)
            else:
                logger.warning("Empty response after retry for user %d", user_id)
                await bot.send_message(
                    chat_id,
                    "Claude не ответил дважды. Повтори сообщение.",
                )

    except Exception as e:
        logger.exception("Chat session error for user %d", user_id)
        # Карточку снимаем ДО сообщения об ошибке: инвариант «последнее
        # сообщение в чате — это ответ, а не карточка» должен держаться и на
        # аварийном пути тоже (слепое ревью, находка 9).
        await _stop_progress_card(card, card_task)
        card, card_task = None, None
        error_text = f"Error: {html.escape(logsafe.redact(str(e))[:200])}"
        try:
            await bot.send_message(chat_id, error_text)
        except Exception:
            logger.exception("Failed to send error message")
    finally:
        if typing_task is not None:
            typing_task.cancel()
        # Страховка для аварийного выхода: на нормальном пути карточка уже
        # снята выше. Повторный вызов безопасен и ничего не делает.
        await _stop_progress_card(card, card_task)
    # Answered, or answered with an error the user can read: either way this
    # message is finished with, and a queued job holding it can be dropped.
    return None


def _start_progress_card(
    bot: Bot, chat_id: int, manager: Any
) -> tuple[Any | None, asyncio.Task | None]:
    """Открыть карточку прогресса и фоновую задачу, которая её ведёт.

    ``(None, None)``, если карточки не будет — под Codex, без закреплённой
    сессии, или если что угодно пошло не так. Не бросает: карточка
    вспомогательная, и её отсутствие не должно стоить ответа."""
    try:
        path = manager.progress_transcript()
        if path is None:
            return None, None
        card = progress.ProgressCard(
            bot,
            chat_id,
            progress.ProgressFeed(path),
            clock=time.monotonic,
        )
    except Exception:  # noqa: BLE001 — см. докстринг
        logger.debug("progress: could not start the card", exc_info=True)
        return None, None
    return card, asyncio.create_task(_progress_loop(card))


async def _progress_loop(card: Any) -> None:
    """Тикать карточку, пока задачу не отменят. ``tick`` не бросает сам."""
    try:
        while True:
            await card.tick()
            await asyncio.sleep(progress.POLL_SECONDS)
    except asyncio.CancelledError:
        pass


#: Потолок ожидания на снятие карточки — см. _stop_progress_card.
_CARD_CLOSE_TIMEOUT = 3.0


async def _stop_progress_card(card: Any | None, card_task: asyncio.Task | None) -> None:
    """Погасить задачу и удалить карточку. Не бросает НИЧЕГО: вызывается
    вплотную к доставке ответа, и её сбой не имеет права её сорвать."""
    # CancelledError наследует BaseException, а не Exception — suppress(Exception)
    # его НЕ ловит. Два реальных случая, когда он тут возникает: задачу отменили
    # до её первого запуска (тогда `await` бросает его сразу), и жёсткая отмена
    # самого хода на выключении бота (shutdown.cancel_turns). Утечка отсюда
    # срывала бы ровно то, ради чего написан этот finally, — и, на нормальном
    # пути, уносила бы ответ, потому что снятие карточки идёт ПЕРЕД его
    # отправкой.
    #
    # Честная оговорка (второе слепое ревью): если отмена ВСЕГО хода придёт
    # ровно в тот момент, когда мы висим на `await card_task`, asyncio
    # доставит её отменой именно этой задачи, `_progress_loop` её проглотит
    # и вернётся штатно — то есть отмена хода будет поглощена здесь. Это
    # осознанно оставлено: цена ограничена одним тиком (POLL_SECONDS), а
    # исход — ход доигрывает и ответ доходит, ровно то, ради чего сделана
    # мягкая остановка; `cancel_turns` всё равно ждёт
    # стрегглеров под своим `wait_for`.
    if card_task is not None:
        card_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await card_task
    if card is not None:
        # С ПОТОЛКОМ: удаление карточки — полный round-trip к Telegram, и он
        # стоит ПЕРЕД отправкой ответа. У aiogram дефолтный таймаут запроса
        # около минуты, так что подвисший или поймавший flood-wait вызов
        # задержал бы ответ на всё это время. Ответ прикрыт долговечной
        # очередью, а этот вызов — нет; ждать его дольше пары секунд не
        # стоит ничего хорошего (слепое ревью, находка 6).
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(card.close(), timeout=_CARD_CLOSE_TIMEOUT)


async def _typing_loop(bot: Bot, chat_id: int) -> None:
    """Send typing action every 4 seconds while processing."""
    try:
        while True:
            await bot.send_chat_action(chat_id, "typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


# --- Media input (photo / document / video / audio / animation / video_note) ---

UNSUPPORTED_REPLY = (
    "Я принимаю голос, текст, фото и файлы. Этот тип сообщения обработать не могу."
)

_MEDIA_EXTRACTORS = (
    # (kind, attr, default extension)
    ("document", "document", None),
    ("video", "video", "mp4"),
    ("audio", "audio", "mp3"),
    ("animation", "animation", "mp4"),
    ("video_note", "video_note", "mp4"),
)


def extract_media(message: Any) -> tuple[str, str, str, str | None]:
    """(kind, file_id, extension, original_name) for a media message.

    Photos are a size ladder — take the largest. Documents/audio keep the
    original file name (its extension wins over the default).
    """
    if getattr(message, "photo", None):
        return ("photo", message.photo[-1].file_id, "jpg", None)
    for kind, attr, default_ext in _MEDIA_EXTRACTORS:
        obj = getattr(message, attr, None)
        if obj is None:
            continue
        name = getattr(obj, "file_name", None)
        ext = default_ext or "bin"
        if name and "." in name:
            candidate = name.rsplit(".", 1)[-1].lower()
            # file_name is sender-controlled — never let it shape the path
            if re.fullmatch(r"[a-z0-9]{1,10}", candidate):
                ext = candidate
        return (kind, obj.file_id, ext, name)
    raise ValueError("message carries no known media")


def forward_note(origin: Any) -> str:
    """Human-readable forward attribution, or '' for a non-forward."""
    if origin is None:
        return ""
    user = getattr(origin, "sender_user", None)
    if user is not None:
        return f"[переслано от: {user.full_name}]\n"
    # MessageOriginChannel carries .chat, MessageOriginChat carries .sender_chat
    chat = getattr(origin, "chat", None) or getattr(origin, "sender_chat", None)
    if chat is not None:
        return f"[переслано из: {chat.title}]\n"
    name = getattr(origin, "sender_user_name", None)
    if name:
        return f"[переслано от: {name}]\n"
    return "[переслано]\n"


def build_media_prompt(
    *,
    kind: str,
    rel_path: str,
    original_name: str | None,
    caption: str,
    fwd: str,
    prep: MediaPrep | None = None,
) -> str:
    """Prompt for the brain: it lives in the vault and can Read the file
    itself (images, PDFs, text) — we hand it the path, the context and the
    instruction that ``media_prep`` decided is safe for this file.

    Stays a pure function: ``prep`` is a plain value object, all the I/O
    happened before the call.
    """
    prep = prep or MediaPrep()
    name_part = f" (имя файла: {original_name})" if original_name else ""
    meta_part = f" [{prep.meta}]" if prep.meta else ""
    caption_part = f"\nПодпись: {caption}" if caption else ""
    return (
        f"{fwd}Пользователь прислал {kind}: {rel_path}{name_part}{meta_part}"
        f"{caption_part}\n{prep.instruction}"
    )


# Telegram albums arrive as N separate messages sharing media_group_id.
# Buffer them briefly and hand the brain ONE prompt with all paths —
# otherwise an album means N long brain turns over the same context.
ALBUM_SETTLE = 1.5
_album_buf: dict[str, list[dict[str, Any]]] = {}
_album_tasks: dict[str, asyncio.Task] = {}


def build_album_prompt(items: list[dict[str, Any]]) -> str:
    fwd = next((i["fwd"] for i in items if i["fwd"]), "")
    captions = [i["caption"] for i in items if i["caption"]]
    lines = []
    for item in items:
        line = f"- {item['rel_path']} ({item['kind']})"
        model_rel = item.get("model_rel_path")
        if model_rel:
            line += f"\n  → читай вместо него: {model_rel}"
        lines.append(line)
    files = "\n".join(lines)
    caption_part = f"\nПодпись: {' / '.join(captions)}" if captions else ""
    # Per-file instructions from media_prep, deduplicated and in order — an
    # album can legitimately mix a downscaled photo with a video that must
    # not be read at all.
    instructions: list[str] = []
    for item in items:
        text = item.get("instruction")
        if text and text not in instructions:
            instructions.append(text)
    guidance = "\n".join(instructions) or (
        "Прочитай файлы (Read поддерживает изображения и PDF), сохрани суть "
        "в память по правилам vault одной записью и кратко ответь."
    )
    return (
        f"{fwd}Пользователь прислал альбом из {len(items)} файлов:\n"
        f"{files}{caption_part}\n{guidance}\n"
        "Ответь ОДНОЙ записью в память и одним кратким ответом на весь альбом."
    )


# --- Busy guard for the media path ---


def build_busy_media_reply(rel_paths: list[str]) -> str:
    """Honest reply for a file that arrived while a turn is already running.

    Deliberately different from the text path's busy wording: the file is
    ALREADY saved by the time we get here (the librarian safety net is
    untouched), and — per rule B3 — it must NOT promise a callback, because
    nothing re-sends an answer later.
    """
    if len(rel_paths) == 1:
        head = f"📎 Файл сохранил (<code>{html.escape(rel_paths[0])}</code>)."
        tail = "напомни про файл или пришли подпись к нему, обработаю"
    else:
        listed = "\n".join(f"• <code>{html.escape(p)}</code>" for p in rel_paths)
        head = f"📎 Файлы сохранил ({len(rel_paths)}):\n{listed}"
        tail = "напомни про них или пришли подпись, обработаю"
    return (
        f"{head}\nСессия сейчас занята предыдущей задачей — когда "
        f"освободится, {tail}. Ничего не потерялось."
    )


# ``is_turn_active()`` reflects the pane FILE LOCK, which is only acquired
# deep inside ``ask()``, in a worker thread — far downstream of this guard.
# main.py calls start_polling without overriding handle_as_tasks, which
# aiogram defaults to True, so every Telegram update is its own concurrent
# asyncio task: two heavy documents arriving in the same second
# (the literal 31.08 incident) BOTH run the guard before EITHER reaches
# ask(), both see an idle pane, and the second one silently queues on the
# lock instead of getting an honest busy reply. The in-process claim below
# closes that window: it is taken SYNCHRONOUSLY, with no await between the
# check and the set, so on a single event loop exactly one task can win it.
#
# TTL is a self-heal backstop only — the claim is released in a `finally` on
# every dispatch path. It sits just above the hardest turn budget
# (DEFAULT_TIMEOUT) so a legitimate hour-long turn never has its claim
# stolen, while a claim leaked by some path we did not anticipate expires
# instead of wedging the media path permanently (/23: never let one
# bug become a standing outage).
MEDIA_CLAIM_TTL = DEFAULT_TIMEOUT + 300

_media_claim_at: float | None = None


def _try_claim_media_dispatch() -> bool:
    """Take the single in-process media-dispatch slot, or report it taken.

    Deliberately synchronous and await-free: that is exactly what makes it
    atomic against concurrently scheduled handler tasks.
    """
    global _media_claim_at
    now = time.monotonic()
    held = _media_claim_at
    if held is not None:
        if now - held < MEDIA_CLAIM_TTL:
            return False
        logger.warning(
            "Stale media dispatch claim (%.0fs old) — reclaiming", now - held
        )
    _media_claim_at = now
    return True


def _release_media_dispatch() -> None:
    """Idempotent — safe to call from a `finally` that may run twice."""
    global _media_claim_at
    _media_claim_at = None


async def reject_media_if_busy(bot: Bot, chat_id: int, rel_paths: list[str]) -> bool:
    """True (and an honest reply sent) if this file must not be dispatched.

    This check lives in the HANDLER, before ``ask()`` is ever reached. That
    placement is the whole point: ``ask()`` returning "busy" is counted as a
    failure by ``ask_health`` (FAILURE_STATUSES, fix B3 of 2026-08-22), so a
    heavy attachment landing on a busy pane used to grow the fail-streak and,
    three in a row, fire a ``delivery_guard`` "channel broken" alert. Bailing
    out here never touches the ledger.

    Contract, in both of its halves:

    * WITHOUT a queue — returning ``False`` means the caller now HOLDS the
      dispatch claim and MUST release it with ``_release_media_dispatch()``
      in a ``finally``; returning ``True`` means no claim is held.
    * WITH a queue — this returns ``False`` having taken NOTHING, and the
      caller's ``finally`` is a harmless no-op (``_release_media_dispatch``
      is idempotent, and nothing in this process ever takes the claim while
      a queue is configured). The lane is the claim now; see below.

    WITH THE PER-CHAT QUEUE THERE IS NOTHING LEFT TO
    GUARD. This whole function is an earlier, weaker version of the lane in
    ``_process_and_reply``: the same synchronous, await-free claim, taken at
    the same point, against the same race (two heavy documents landing in
    the same second, the literal 31.08 incident). The difference is what
    happens to the loser — brushed off here, PARKED with a receipt there —
    and a file the owner sent is worth keeping, not apologizing for. So when
    a queue is configured this steps aside entirely and lets the lane do the
    work; without one (rollback, unit tests, tools) it is byte-for-byte the
    old guard.
    """
    if _queue() is not None:
        return False
    if not _try_claim_media_dispatch():
        # A sibling update is already on its way to ask() — the pane looks
        # idle only because that task has not reached the lock yet.
        await bot.send_message(chat_id, build_busy_media_reply(rel_paths))
        return True
    if _get_manager().is_turn_active():
        _release_media_dispatch()
        await bot.send_message(chat_id, build_busy_media_reply(rel_paths))
        return True
    return False


async def queue_album_item(
    bot: Bot, *, chat_id: int, user_id: int, group_id: str, item: dict[str, Any]
) -> asyncio.Task:
    """Buffer one album member and return the task that will dispatch the
    whole group.

    The RETURN is what makes albums durable (blind review 5). Every other
    input path awaits its own work inside the handler, so the inbox signs
    the entry off only once the turn is done. The album path used to return
    the moment the item was buffered, which signed all N entries off 1.5
    seconds before the album was even dispatched — a crash in that window,
    or during the album's turn, lost the whole group with no trace in the
    queue. The caller awaits this task, so the sign-off waits too.
    """
    _album_buf.setdefault(group_id, []).append(item)
    if group_id not in _album_tasks:
        _album_tasks[group_id] = asyncio.create_task(
            _flush_album(bot, chat_id, user_id, group_id)
        )
    return _album_tasks[group_id]


async def _flush_album(bot: Bot, chat_id: int, user_id: int, group_id: str) -> None:
    await asyncio.sleep(ALBUM_SETTLE)
    items = _album_buf.pop(group_id, [])
    _album_tasks.pop(group_id, None)
    if not items:
        return
    # A turn can have STARTED during the 1.5s settle window — re-check here,
    # still before ask(). One reply for the whole album, not one per file.
    if await reject_media_if_busy(bot, chat_id, [i["rel_path"] for i in items]):
        return
    try:
        await _process_and_reply(
            bot,
            chat_id,
            user_id,
            build_album_prompt(items),
            # The album is answered ONCE, so it is threaded under the first
            # file of the group — the message the owner would point at if
            # asked which album this answer belongs to.
            message_id=items[0].get("message_id"),
        )
    finally:
        _release_media_dispatch()


# --- Handlers ---


@router.message(F.voice)
async def handle_chat_voice(message: Message, bot: Bot) -> None:
    """Handle voice messages in private chat."""
    if not message.voice or not message.from_user:
        return

    settings = get_settings()
    storage = VaultStorage(settings.vault_path)
    transcriber = DeepgramTranscriber(settings.deepgram_api_key)

    try:
        file = await bot.get_file(message.voice.file_id)
        if not file.file_path:
            await message.answer("Failed to download voice")
            return

        file_bytes = await bot.download_file(file.file_path)
        if not file_bytes:
            await message.answer("Failed to download voice")
            return

        transcript = await transcriber.transcribe(file_bytes.read())
        if not transcript:
            await message.answer("Could not transcribe audio")
            return

        # Safety net: save to daily
        timestamp = datetime.fromtimestamp(message.date.timestamp())
        storage.append_to_daily(transcript, timestamp, "[voice]")

        # Log to session
        session = SessionStore(settings.vault_path)
        session.append(
            message.from_user.id,
            "voice",
            text=transcript,
            duration=message.voice.duration,
            msg_id=message.message_id,
        )

        await _process_and_reply(
            bot,
            message.chat.id,
            message.from_user.id,
            f"[voice] {transcript}",
            message_id=message.message_id,
        )

    except Exception as e:
        logger.exception("Error processing voice in chat")
        try:
            await message.answer(f"Error: {html.escape(logsafe.redact(str(e))[:200])}")
        except Exception:
            logger.exception("Failed to send voice error message")


@router.message(F.text)
async def handle_chat_text(message: Message, bot: Bot) -> None:
    """Handle text messages in private chat.

    Bot-level commands (/start, /help, …) are intercepted by routers
    registered earlier; anything that reaches here — including Claude Code
    slash commands and /skill-name invocations — is dispatched by behavior.
    """
    if not message.text or not message.from_user:
        return

    settings = get_settings()
    storage = VaultStorage(settings.vault_path)

    fwd = forward_note(getattr(message, "forward_origin", None))
    text = f"{fwd}{message.text}" if fwd else message.text

    # Safety net: save to daily
    timestamp = datetime.fromtimestamp(message.date.timestamp())
    storage.append_to_daily(text, timestamp, "[forward]" if fwd else "[text]")

    # Log to session
    session = SessionStore(settings.vault_path)
    session.append(
        message.from_user.id,
        "text",
        text=text,
        msg_id=message.message_id,
    )

    await _dispatch_text(
        bot,
        message.chat.id,
        message.from_user.id,
        text,
        message_id=message.message_id,
    )


@router.message(F.photo | F.document | F.video | F.audio | F.animation | F.video_note)
async def handle_chat_media(message: Message, bot: Bot) -> None:
    """Handle any file-bearing message: download into the vault's
    attachments and hand the PATH to the brain — it reads the file itself."""
    if not message.from_user:
        return

    settings = get_settings()
    storage = VaultStorage(settings.vault_path)
    timestamp = datetime.fromtimestamp(message.date.timestamp())
    caption = message.caption or ""
    fwd = forward_note(getattr(message, "forward_origin", None))

    try:
        kind, file_id, ext, original_name = extract_media(message)
    except ValueError:
        await message.answer(UNSUPPORTED_REPLY)
        return

    try:
        file = await bot.get_file(file_id)
        if not file.file_path:
            await message.answer("Не удалось скачать файл.")
            return
        file_bytes = await bot.download_file(file.file_path)
        if not file_bytes:
            await message.answer("Не удалось скачать файл.")
            return

        rel_path = storage.save_attachment(
            file_bytes.read(), timestamp.date(), timestamp, ext
        )

        # Safety net: save to daily with an Obsidian embed
        daily_entry = f"{fwd}![[{rel_path}]]"
        if caption:
            daily_entry += f"\n\n{caption}"
        storage.append_to_daily(daily_entry, timestamp, f"[{kind}]")

        session = SessionStore(settings.vault_path)
        session.append(
            message.from_user.id,
            kind,
            text=caption or rel_path,
            msg_id=message.message_id,
        )

        group_id = getattr(message, "media_group_id", None)

        # Step 1: the universal busy guard,
        # BEFORE anything can reach ask(). The file is already saved above,
        # so bailing out here loses nothing — and, unlike a "busy" result
        # from ask(), it never increments the ask_health fail-streak.
        # Album items are the one exception: they still get buffered so the
        # whole album produces ONE busy reply from _flush_album instead of
        # one per file (the plan asks for a single combined reply there).
        claimed = False
        if not group_id:
            if await reject_media_if_busy(bot, message.chat.id, [rel_path]):
                return
            claimed = True  # released in the finally below, on every path

        try:
            # Step 2-5: decide what the brain should actually open. Runs in a
            # worker thread — downscaling a 4284x4284 JPEG is real CPU work
            # and must not block the event loop.
            prep = await asyncio.to_thread(
                prepare_for_model, settings.vault_path, rel_path, kind, ext
            )

            if group_id:
                flush = await queue_album_item(
                    bot,
                    chat_id=message.chat.id,
                    user_id=message.from_user.id,
                    group_id=str(group_id),
                    item={
                        "kind": kind,
                        "rel_path": rel_path,
                        "caption": caption,
                        "fwd": fwd,
                        "message_id": message.message_id,
                        "model_rel_path": prep.model_rel_path,
                        "instruction": prep.instruction,
                    },
                )
                # Wait for the group's dispatch so this update is not signed
                # off in the inbox before its work has happened. Every member
                # of the album awaits the SAME task, so all N entries stay in
                # the queue until the one combined turn is over. Failures are
                # swallowed because _flush_album already replies for the whole
                # group — letting one bubble up here would produce N copies of
                # the handler's error message below.
                with contextlib.suppress(Exception):
                    await flush
                return

            prompt = build_media_prompt(
                kind=kind,
                rel_path=rel_path,
                original_name=original_name,
                caption=caption,
                fwd=fwd,
                prep=prep,
            )
            await _process_and_reply(
                bot,
                message.chat.id,
                message.from_user.id,
                prompt,
                message_id=message.message_id,
            )
        finally:
            # Runs while the exception is still propagating, i.e. BEFORE the
            # handler's own `except Exception` below sees it — so no failure
            # path out of the dispatch can leak the claim.
            if claimed:
                _release_media_dispatch()

    except Exception as e:
        logger.exception("Error processing media in chat")
        # Bot API can't hand us files >20MB — keep the librarian promise:
        # the fact and the caption still land in daily.
        if "too big" in str(e).lower():
            note = f"{fwd}(файл >20MB — Telegram не отдаёт его ботам)"
            if caption:
                note += f"\n\n{caption}"
            storage.append_to_daily(note, timestamp, f"[{kind}]")
            await message.answer(
                "Файл больше 20 МБ — Telegram не отдаёт такие ботам. "
                "Подпись сохранил; перешли файл иначе (ссылкой/частями)."
            )
            return
        try:
            await message.answer(f"Error: {html.escape(logsafe.redact(str(e))[:200])}")
        except Exception:
            logger.exception("Failed to send media error message")


@router.message()
async def handle_chat_other(message: Message) -> None:
    """Catch-all: never go silent — tell the user what the bot accepts."""
    await message.answer(UNSUPPORTED_REPLY)
