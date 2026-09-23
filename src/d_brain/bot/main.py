"""Telegram bot initialization and polling."""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, ReplyKeyboardRemove, Update

from d_brain.config import Settings
from d_brain.services import chat_queue, inbox, outbox, shutdown
from d_brain.services.cron_runner import run_cron
from d_brain.services.runtime import get_session
from d_brain.services.systemd_notify import notify, watchdog_interval

logger = logging.getLogger(__name__)


def create_bot(settings: Settings) -> Bot:
    """Create and configure the Telegram bot."""
    return Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


def create_dispatcher() -> Dispatcher:
    """Create and configure the dispatcher with routers."""
    from d_brain.bot.handlers import (
        # buttons,
        chat,
        commands,
        process,
        resend,
        work,
    )

    dp = Dispatcher(storage=MemoryStorage())

    # Register routers - ORDER MATTERS
    dp.include_router(commands.router)
    dp.include_router(process.router)
    dp.include_router(resend.router)
    # Read-only status of what's running — must be reachable while the
    # session is busy, so it goes BEFORE chat.router's catch-all.
    dp.include_router(work.router)
    # Reply-keyboard buttons DISABLED 2026-08-22: the owner kept
    # hitting "⚙️ Обработать" by accident. To restore: uncomment the import
    # above and this include_router call, and re-add
    # reply_markup=get_main_keyboard() to cmd_start() in handlers/commands.py.
    # keyboards.py / handlers/buttons.py themselves were left untouched.
    # dp.include_router(buttons.router)  # Reply keyboard buttons
    dp.include_router(chat.router)  # Catch-all for private chat (LAST)
    return dp


MiddlewareHandler = Callable[[Update, dict[str, Any]], Awaitable[Any]]
MiddlewareType = Callable[[MiddlewareHandler, Update, dict[str, Any]], Awaitable[Any]]


def event_user(event: Update) -> Any:
    """The human behind an update, if it carries one."""
    if event.message:
        return event.message.from_user
    if event.callback_query:
        return event.callback_query.from_user
    return None


def is_authorized(settings: Settings, event: Update) -> bool:
    """May this update be served? The decision only — no logging, no reply.

    Split out of ``create_auth_middleware`` (2026-09-22) because the graceful
    stop needs the same answer OUTSIDE the auth middleware: its refusal has to
    sit outside the inbox's duplicate gate, which is itself outside auth, and a
    stranger must not get the "перезапускаюсь" notice.

    An update with no ``from_user`` (service updates, channel posts) passes,
    exactly as it did before the split — this is a faithful extraction, not a
    tightening.
    """
    if settings.allow_all_users:
        return True
    if not settings.allowed_user_ids:
        return False
    user = event_user(event)
    return not (user and user.id not in settings.allowed_user_ids)


def create_auth_middleware(settings: Settings) -> MiddlewareType:
    """Create middleware to check user authorization."""

    async def auth_middleware(
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        if is_authorized(settings, event):
            return await handler(event, data)

        # If no users allowed and not allow_all_users -> deny everyone
        if not settings.allowed_user_ids:
            logger.warning(
                "Access denied: no allowed_user_ids configured and "
                "allow_all_users is False"
            )
            return None

        user = event_user(event)
        logger.warning(
            "Unauthorized access attempt from user %s", getattr(user, "id", None)
        )
        return None

    return auth_middleware


def bot_commands() -> list[BotCommand]:
    """Commands shown in Telegram's native "/" command menu."""
    return [
        BotCommand(command="status", description="Статус дня"),
        BotCommand(command="process", description="Обработать записи дня"),
        BotCommand(command="onboarding", description="Знакомство и настройка"),
        BotCommand(command="help", description="Справка"),
        BotCommand(command="new", description="Новый чат (сброс сессии)"),
        BotCommand(command="compact", description="Сжать контекст сессии"),
        BotCommand(command="resend", description="Повторить последний ответ"),
        BotCommand(command="work", description="Что сейчас в работе"),
        BotCommand(
            command="relogin",
            description="Пересоздать сессию (если после dbrain login просит вход)",
        ),
    ]


async def _send_keyboard_removal_once(bot: Bot, settings: Settings) -> None:
    """One-time proactive ping so already-subscribed users lose the old
    reply keyboard (new users never see it since cmd_start no longer sends
    reply_markup). Gated by a marker file under runtime_dir."""
    marker = settings.runtime_dir / "keyboard_removed"
    if marker.exists() or settings.admin_chat_id is None:
        return
    try:
        await bot.send_message(
            settings.admin_chat_id,
            "Кнопки под полем ввода убраны — команды теперь в "
            'нативном "/"-меню Telegram (слева от поля ввода).',
            reply_markup=ReplyKeyboardRemove(),
        )
    except Exception:
        logger.warning("keyboard removal ping failed", exc_info=True)
        return
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("sent\n")
    except Exception:
        logger.warning("keyboard removal marker write failed", exc_info=True)


async def _watchdog_pinger() -> None:
    """Ping systemd's watchdog while the event loop is healthy."""
    interval = watchdog_interval()
    while True:
        await asyncio.sleep(interval)
        notify("WATCHDOG=1")


async def run_bot(settings: Settings) -> None:
    """Run the bot with polling."""
    bot = create_bot(settings)
    # Before anything can produce a reply: every send_response from here on
    # is persisted first. Configuring it this early also means the worker
    # below inherits whatever the PREVIOUS process failed to deliver.
    box = outbox.configure(settings.runtime_dir)
    # The mirror image, on the way IN: every fetched update is written to
    # disk before aiogram is allowed to continue — and therefore before the
    # Telegram offset that confirms it can move. Registered here, on the
    # session, because that is the single point every incoming update
    # crosses (see services/inbox.py).
    mailbox = inbox.configure(settings.runtime_dir)
    bot.session.middleware(inbox.accept_middleware(mailbox))
    # The middle of the pipeline: inbox owns "arrived, not yet prepared",
    # this owns "prepared, not yet answered", outbox owns "answered, not yet
    # delivered". Configured BEFORE the replay below, so a message the
    # previous process parked is visible to the handlers the replay feeds.
    lane = None
    if settings.chat_queue_enabled:
        lane = chat_queue.configure(
            settings.runtime_dir,
            max_waiting=settings.chat_queue_max_waiting,
            max_age=settings.chat_queue_max_age,
        )
    else:
        # Turned OFF with messages still parked from a previous run. Nothing
        # will ever drain them, so the "отвечу следом" they were promised has
        # to be withdrawn out loud rather than left to rot on disk.
        await chat_queue.retire_leftovers(
            bot, settings.runtime_dir, reason="очередь по чату выключена"
        )
    try:
        await bot.set_my_commands(bot_commands())
    except Exception:
        logger.warning("set_my_commands failed", exc_info=True)
    await _send_keyboard_removal_once(bot, settings)

    dp = create_dispatcher()

    # The stop itself. Registered as the OUTERMOST middleware of all — before
    # the inbox's, which is before auth — because once a stop is requested a
    # new message must be refused WITHOUT the inbox claiming and signing it
    # off. Refused there, it stays on disk and the next boot answers it; a
    # refusal one layer deeper would delete it.
    stopper = shutdown.Shutdown(grace=settings.shutdown_grace_seconds)
    dp.update.outer_middleware(
        shutdown.stop_middleware(
            stopper, bot, is_allowed=lambda event: is_authorized(settings, event)
        )
    )
    # OUTER, so it wraps the auth middleware, the routers and every filter:
    # it refuses an update that already has a receipt (no double answer) and
    # signs the entry off once the chain is done.
    dp.update.outer_middleware(inbox.handled_middleware(mailbox))
    # Always add auth middleware for security (it handles allow_all_users internally)
    dp.update.middleware(create_auth_middleware(settings))

    # Bring the persistent Claude session up before serving requests; failure
    # here is non-fatal (ask() will retry ensure on demand).
    try:
        await asyncio.to_thread(get_session(settings).ensure_session)
    except Exception:
        logger.exception("Claude session failed to start at boot; retrying on demand")

    notify("READY=1")
    pinger = asyncio.create_task(_watchdog_pinger())
    # Starts BEFORE polling: its first pass flushes replies the previous
    # process had queued but not delivered (restart mid-send, crash, 429
    # backoff that outlived the process).
    outbox_task = asyncio.create_task(outbox.run(bot, box))
    cron_task = (
        asyncio.create_task(run_cron(settings, bot)) if settings.cron_enabled else None
    )

    # Whatever the previous process accepted and never answered goes out
    # BEFORE polling resumes — awaited, one update at a time, oldest first.
    # That ordering is what keeps a chat in order: the backlog cannot
    # interleave with messages arriving right now. Telegram holds updates
    # for 24h, so nothing is lost while this runs.
    #
    # Hard-bounded in wall time. ``inbox_replay_max_age`` limits how OLD the
    # backlog may be, never how long answering it takes — one replayed turn
    # can legitimately ride chat_turn_timeout (25 minutes by default), and
    # every second of that is a second with no /work, no /stop and no way
    # back in. Past the budget the backlog is abandoned to its attempt
    # counter and the channel comes back.
    try:
        await asyncio.wait_for(
            inbox.replay(bot, dp, mailbox, max_age=settings.inbox_replay_max_age),
            timeout=inbox.REPLAY_BUDGET,
        )
    except TimeoutError:
        logger.warning(
            "inbox replay hit its %.0fs budget; starting polling and leaving "
            "the rest to the next boot",
            inbox.REPLAY_BUDGET,
        )
    except Exception:  # noqa: BLE001 — a stuck backlog must never cost polling
        logger.exception("inbox replay failed; starting polling anyway")

    # AFTER the replay, so a message the previous process parked keeps its
    # place ahead of whatever the replay feeds in: the gate in chat.py parks
    # a new message whenever the chat already has one waiting, and with the
    # worker still idle the whole backlog lands in the queue in arrival order
    # before a single job starts.
    queue_task = None
    if lane is not None:
        from d_brain.bot.handlers import chat as chat_handlers

        queue_task = asyncio.create_task(
            chat_queue.run(
                bot,
                lane,
                chat_handlers.run_queued_job,
                should_start=lambda: not stopper.stopping,
                # A queued turn is the one turn nobody holds an aiogram
                # update open for, so `stop_middleware` never hears about it.
                # Registering it here is what gives it the same grace window
                # every other in-flight turn gets.
                turns=stopper,
            )
        )

    logger.info("Starting bot polling...")
    # Ours, not aiogram's: aiogram's handler stops polling the instant the
    # signal lands, and polling has to survive the grace window so a message
    # arriving mid-restart is still accepted to disk and still gets a line
    # back. Hence handle_signals=False, and close_bot_session=False so the
    # final outbox drain below still has a session to send through.
    #
    # HERE and not earlier, on purpose. Everything above — ensure_session, the
    # replay and its 300s budget — still dies outright on a signal, exactly as
    # it did before this existed. Arming the stop before the replay would make
    # stop_middleware refuse the updates the replay is feeding, and replay
    # signs every entry off afterwards regardless (its belt-and-braces
    # mark_handled), so a refusal there DELETES the message instead of keeping
    # it. Closing the boot window means teaching replay that difference; see
    # services/shutdown.py, "Three things this does NOT cover".
    shutdown.install(stopper)
    poller = asyncio.create_task(
        dp.start_polling(
            bot,
            allowed_updates=dp.resolve_used_update_types(),
            handle_signals=False,
            close_bot_session=False,
        )
    )
    try:
        asked_to_stop = asyncio.create_task(stopper.requested.wait())
        try:
            await asyncio.wait(
                {poller, asked_to_stop}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            asked_to_stop.cancel()
        if stopper.stopping:
            # Cron goes down FIRST, before the grace is waited out. A cron job
            # is new work, and claim_due persists the job's next run before it
            # runs it (at-most-once, by design) — so a recurring job that
            # starts inside the grace and is killed at the deadline is silently
            # skipped for that occurrence. Cancelling here keeps that window
            # exactly as short as it was before this feature existed, instead
            # of stretching it to the whole grace (blind review 3).
            if cron_task is not None:
                cron_task.cancel()
            await shutdown.stop(bot, dp, box, stopper, poller)
        else:
            # Polling ended on its own — surface whatever ended it.
            await poller
    finally:
        pinger.cancel()
        outbox_task.cancel()
        # Not cancelled before the grace, unlike cron: `should_start` above
        # already stopped it taking NEW jobs, and the one it may still be
        # running is tracked in `stopper`, so `wait_for_turns` gives it the
        # same deadline every other in-flight turn gets. Whatever it does not
        # finish stays on disk and is answered after the restart.
        if queue_task is not None:
            queue_task.cancel()
        if cron_task is not None:
            cron_task.cancel()
        # A clean graceful stop must not exit non-zero: that status is not in
        # SuccessExitStatus and would fire OnFailure=dbrain-notify (blind
        # review 8). Closing a session that is already going away is not worth
        # an alert.
        with contextlib.suppress(Exception):
            await bot.session.close()
        if stopper.stopping:
            # Deliberately a hard exit, not a return: a wedged engine thread
            # would otherwise hold asyncio.run's executor join for minutes,
            # well past TimeoutStopSec, and systemd would SIGKILL us anyway.
            # Both queues are on disk by now — see services/shutdown.py.
            stopper.finish()
