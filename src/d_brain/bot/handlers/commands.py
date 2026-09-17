"""Bot commands: /start, /help, /status, /onboarding, /new, /compact, /relogin."""

import asyncio
from datetime import date

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.types import Message

from d_brain.bot.handlers import chat
from d_brain.config import get_settings
from d_brain.services.chat_session import ChatSessionManager
from d_brain.services.session import SessionStore
from d_brain.services.storage import VaultStorage

router = Router(name="commands")


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """Handle /start command."""
    await message.answer(
        "<b>d-brain</b> — персональный ассистент\n\n"
        "Просто пиши мне — я отвечу.\n"
        "Голосовые, текст, фото, пересланные — всё принимаю.\n\n"
        "<b>Команды:</b>\n"
        "/new — новый чат\n"
        "/compact — сжать контекст\n"
        "/relogin — пересоздать сессию, если после dbrain login просит вход\n"
        "/status — статус дня\n"
        "/process — обработать записи\n"
        "/onboarding — знакомство и настройка (можно продолжить в любой момент)\n"
        "/help — справка"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command."""
    await message.answer(
        "<b>d-brain — персональный ассистент</b>\n\n"
        "Просто отправляй что угодно — Claude обработает и ответит.\n\n"
        "🎤 Голосовое — транскрибирую и обработаю\n"
        "💬 Текст — обработаю как есть\n\n"
        "<b>Команды:</b>\n"
        "/new — новый чат (сброс сессии)\n"
        "/compact — сжать контекст сессии\n"
        "/relogin — пересоздать сессию, если после dbrain login просит вход\n"
        "/status — статус сегодняшнего дня\n"
        "/process — обработать записи дня\n"
        "/onboarding — продолжить знакомство: профиль, цели, заметки"
    )


ONBOARDING_PROMPT = (
    "Запусти skill onboarding (.claude/skills/onboarding/SKILL.md): продолжи "
    "онбординг с текущего шага. Сначала узнай прогресс через CLI онбординга "
    "(status --json), покажи короткую карту шагов и продолжай по инструкции."
)


def build_onboarding_prompt(text: str | None) -> str:
    """Agent prompt for /onboarding; text after the command is passed on."""
    parts = (text or "").split(maxsplit=1)
    extra = parts[1].strip() if len(parts) > 1 else ""
    if not extra:
        return ONBOARDING_PROMPT
    return f"{ONBOARDING_PROMPT}\nСообщение пользователя: {extra}"


@router.message(Command("onboarding"))
async def cmd_onboarding(message: Message, bot: Bot) -> None:
    """Hand onboarding to the agent through the normal chat-text path
    (same busy/steer/stop handling, lock, delivery and reply)."""
    if not message.from_user:
        return
    await chat._dispatch_text(
        bot,
        message.chat.id,
        message.from_user.id,
        build_onboarding_prompt(message.text),
    )


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    """Handle /status command."""
    user_id = message.from_user.id if message.from_user else 0
    settings = get_settings()
    storage = VaultStorage(settings.vault_path)

    # Log command
    session = SessionStore(settings.vault_path)
    session.append(user_id, "command", cmd="/status")

    today = date.today()
    content = storage.read_daily(today)

    if not content:
        await message.answer(f"📅 <b>{today}</b>\n\nЗаписей пока нет.")
        return

    lines = content.strip().split("\n")
    entries = [line for line in lines if line.startswith("## ")]

    voice_count = sum(1 for e in entries if "[voice]" in e)
    text_count = sum(1 for e in entries if "[text]" in e)
    photo_count = sum(1 for e in entries if "[photo]" in e)
    forward_count = sum(1 for e in entries if "[forward]" in e)

    total = len(entries)

    # Get weekly stats from session
    week_stats = ""
    stats = session.get_stats(user_id, days=7)
    if stats:
        week_stats = "\n\n<b>За 7 дней:</b>"
        for entry_type, count in sorted(stats.items()):
            week_stats += f"\n• {entry_type}: {count}"

    await message.answer(
        f"📅 <b>{today}</b>\n\n"
        f"Всего записей: <b>{total}</b>\n"
        f"- 🎤 Голосовых: {voice_count}\n"
        f"- 💬 Текстовых: {text_count}\n"
        f"- 📷 Фото: {photo_count}\n"
        f"- ↩️ Пересланных: {forward_count}"
        f"{week_stats}"
    )


@router.message(Command("new"))
async def cmd_new(message: Message) -> None:
    """Start fresh Claude session."""
    if not message.from_user:
        return

    settings = get_settings()
    manager = ChatSessionManager(settings.vault_path)
    # M1 fix (2026-08-22): reset() calls session.clear(), which now sends a
    # blocking /clear + up to a real ~10s resync poll (see B1) — offload it
    # so this doesn't block the bot's event loop for the whole call.
    await asyncio.to_thread(manager.reset, message.from_user.id)

    await message.answer("Новая сессия. Контекст очищен.")


@router.message(Command("relogin"))
async def cmd_relogin(message: Message) -> None:
    """Self-service recovery for a logged-out Claude session.

    Deliberately does NOT go through the Claude brain at all — the whole
    point is that the brain may be exactly what's broken (2026-08-24
    incident: expired OAuth token cached in a long-lived tmux session,
    `/new`'s `/clear` keystroke does nothing for that case since auth is
    only re-read at process start). This is plain aiogram + a tmux kill,
    same primitive the watchdog already uses for other dead-session
    recovery, just newly exposed for the one case it deliberately leaves to
    a human today (`logged_out` alerts but never auto-kills).
    """
    if not message.from_user:
        return

    settings = get_settings()
    manager = ChatSessionManager(settings.vault_path)
    recovered = await asyncio.to_thread(
        manager.force_recover, message.from_user.id
    )

    if recovered:
        await message.answer(
            "🔄 Сессия пересоздана. Напиши что-нибудь ещё раз через "
            "несколько секунд — должно ответить нормально."
        )
    else:
        await message.answer(
            "⏳ Сейчас идёт другой ответ — подожди, пока он закончится, и "
            "попробуй /relogin ещё раз."
        )


@router.message(Command("compact"))
async def cmd_compact(message: Message) -> None:
    """Compact current session context."""
    if not message.from_user:
        return

    settings = get_settings()
    manager = ChatSessionManager(settings.vault_path)

    await message.chat.do(action="typing")
    summary = await manager.compact(message.from_user.id)

    if summary and len(summary) > 500:
        summary_text = summary[:500] + "..."
    else:
        summary_text = summary or "No summary."

    await message.answer(f"Контекст сжат.\n\n<i>{summary_text}</i>")
