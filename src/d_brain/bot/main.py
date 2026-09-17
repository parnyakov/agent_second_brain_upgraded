"""Telegram bot initialization and polling."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, ReplyKeyboardRemove, Update

from d_brain.config import Settings
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
    )

    dp = Dispatcher(storage=MemoryStorage())

    # Register routers - ORDER MATTERS
    dp.include_router(commands.router)
    dp.include_router(process.router)
    dp.include_router(resend.router)
    # Reply-keyboard buttons DISABLED 2026-08-22 (backlog item 4): the owner kept
    # hitting "⚙️ Обработать" by accident. To restore: uncomment the import
    # above and this include_router call, and re-add
    # reply_markup=get_main_keyboard() to cmd_start() in handlers/commands.py.
    # keyboards.py / handlers/buttons.py themselves were left untouched.
    # dp.include_router(buttons.router)  # Reply keyboard buttons
    dp.include_router(chat.router)  # Catch-all for private chat (LAST)
    return dp


MiddlewareHandler = Callable[[Update, dict[str, Any]], Awaitable[Any]]
MiddlewareType = Callable[[MiddlewareHandler, Update, dict[str, Any]], Awaitable[Any]]


def create_auth_middleware(settings: Settings) -> MiddlewareType:
    """Create middleware to check user authorization."""

    async def auth_middleware(
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        # If explicitly allowed all users, just bypass check
        if settings.allow_all_users:
            return await handler(event, data)

        user = None
        if event.message:
            user = event.message.from_user
        elif event.callback_query:
            user = event.callback_query.from_user

        # If no users allowed and not allow_all_users -> deny everyone
        if not settings.allowed_user_ids:
            logger.warning(
                "Access denied: no allowed_user_ids configured and "
                "allow_all_users is False"
            )
            return None

        # Check if user is in allowed list
        if user and user.id not in settings.allowed_user_ids:
            logger.warning("Unauthorized access attempt from user %s", user.id)
            return None

        return await handler(event, data)

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
            "нативном \"/\"-меню Telegram (слева от поля ввода).",
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
    try:
        await bot.set_my_commands(bot_commands())
    except Exception:
        logger.warning("set_my_commands failed", exc_info=True)
    await _send_keyboard_removal_once(bot, settings)

    dp = create_dispatcher()

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
    cron_task = (
        asyncio.create_task(run_cron(settings, bot)) if settings.cron_enabled else None
    )

    logger.info("Starting bot polling...")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        pinger.cancel()
        if cron_task is not None:
            cron_task.cancel()
        await bot.session.close()
