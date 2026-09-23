"""``/resend`` command handler.

A manual, READ-ONLY escape hatch: re-sends the last assistant reply by
reading it straight from the session's JSONL transcript, bypassing the
flaky tmux-pane-scrape delivery path entirely. Exists because ~3.3% of
turns were measured producing no delivery path at all — no closing marker
ever appears in the pane for them. Because this is user-triggered and
read-only (worst case: it finds nothing, or shows something unexpected,
and the user just asks again), it does not need the same validation rigor
as the automatic delivery path — but every one of its five possible
outcomes must be reported honestly, never as a generic error.
"""

import logging
import time

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.types import Message

from d_brain.bot.formatters import send_response
from d_brain.config import get_settings
from d_brain.services.chat_session import ChatSessionManager

router = Router(name="resend")
logger = logging.getLogger(__name__)

_RESEND_PREFIX = "🔁 Повтор последнего ответа:\n\n"

_MESSAGES = {
    "empty": "🤷 Пока нечего повторять — истории ответов не нашёл.",
    "in_progress": "⏳ Предыдущий ответ ещё не готов — сессия ещё работает над ним.",
    "unavailable": "❌ Не смог прочитать состояние сессии/транскрипт прямо сейчас.",
    "no_markers": (
        "🤷 Последний ответ завершился, но я не нашёл в нём маркер для "
        "восстановления — повторить нечем. Просто задай вопрос ещё раз."
    ),
}

# Simple in-memory per-user anti-spam cooldown — no persistence needed, this
# is a low-stakes manual escape hatch, not a delivery guarantee. Keyed by
# Telegram user id, value is the monotonic time of the last accepted press.
_COOLDOWN_SECONDS = 5.0
_last_press: dict[int, float] = {}


@router.message(Command("resend"))
async def cmd_resend(message: Message, bot: Bot) -> None:
    """Re-send the last assistant reply, read directly from the transcript."""
    if not message.from_user:
        return
    user_id = message.from_user.id

    now = time.monotonic()
    last = _last_press.get(user_id)
    if last is not None and now - last < _COOLDOWN_SECONDS:
        await message.answer("⌛ Подожди немного перед повтором.")
        return
    _last_press[user_id] = now

    settings = get_settings()
    manager = ChatSessionManager(settings.vault_path)
    status, body = await manager.resend_last_reply(user_id)

    if status == "ready":
        logger.info(
            "resend: delivered for user %d, body len=%d", user_id, len(body or "")
        )
        await send_response(bot, message.chat.id, f"{_RESEND_PREFIX}{body or ''}")
        return

    logger.info("resend: status=%s for user %d", status, user_id)
    text = _MESSAGES.get(status, _MESSAGES["unavailable"])
    if status not in _MESSAGES:
        logger.warning("resend: unexpected status %r for user %d", status, user_id)
    await message.answer(text)
