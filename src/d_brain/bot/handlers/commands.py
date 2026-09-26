"""Bot commands: /start, /help, /status, /onboarding, /new, /compact,
/relogin, /reset."""

import asyncio
import logging
from datetime import date

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.types import Message

from d_brain.bot.handlers import chat
from d_brain.config import get_settings
from d_brain.services import chat_queue
from d_brain.services.chat_session import ChatSessionManager, ResetOutcome
from d_brain.services.session import SessionStore
from d_brain.services.storage import VaultStorage

router = Router(name="commands")
logger = logging.getLogger(__name__)


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
        "/relogin — починить сессию, если разлогинилось\n"
        "/reset — полный перезапуск, если бот завис или всё время «занят»\n"
        "/status — статус дня\n"
        "/work — что сейчас в работе\n"
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
        "/relogin — починить сессию, если бот пишет «нужен повторный вход»\n"
        "/reset — полный перезапуск: прервать ход, пересоздать основную и "
        "дежурную сессии (контекст разговора начнётся заново, вольт на месте)\n"
        "/status — статус сегодняшнего дня\n"
        "/work — что сейчас в работе: идёт ли ход, сколько уже, где застряло\n"
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


_RESET_NAMES = {"main": "основная", "duty": "дежурная"}


def reset_report(
    outcomes: list[ResetOutcome], waiting: int, handoff: int = 0
) -> str:
    """The /reset reply. "всё чисто" only when EVERY session read back as
    up and idle — anything else names what did not come back."""
    failed = [o for o in outcomes if not o.ok]
    if failed:
        lines = ["⚠️ Перезапуск прошёл не до конца:"]
        for o in outcomes:
            name = _RESET_NAMES.get(o.name, o.name)
            state = "поднята и свободна" if o.ok else f"не поднялась ({o.detail})"
            lines.append(f"• {name} сессия — {state}")
        lines.append("Попробуй /reset ещё раз через минуту.")
        text = "\n".join(lines)
    else:
        text = (
            "🔌 Перезапущено, всё чисто: текущий ход прерван, сессии "
            "пересозданы и проверены — свободны. Контекст разговора начнётся "
            "заново, всё записанное в вольт на месте."
        )
    if waiting:
        text += (
            f"\n\n📨 В очереди ждут сообщений: {waiting}. Они не потеряны — "
            "отвечу на них из новой сессии по порядку."
        )
    if handoff:
        text += (
            f"\n\n🔁 Дежурная сессия отвечала без контекста на {handoff} "
            "сообщ. — передам их основной вместе с твоим следующим сообщением."
        )
    return text


@router.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    """Circuit breaker for a session that is stuck or falsely "busy".

    Like /relogin it never goes through the brain — the brain may be what
    is broken. Unlike /relogin it does not give up when a turn holds the
    session: it stops that turn first (ChatSessionManager.circuit_reset).
    Uses the chat handler's own manager, so the duty lock it holds during
    the duty restart is the same one the duty path takes.
    """
    if not message.from_user:
        return
    from d_brain.bot.handlers import chat

    manager = chat._get_manager()
    if manager.reset_in_progress():
        await message.answer("🔌 Перезапуск уже идёт — дождись его отчёта.")
        return
    await message.answer("🔌 Перезапускаю: прерываю текущий ход и пересоздаю сессии…")
    try:
        outcomes = await manager.circuit_reset(message.from_user.id)
    except Exception as exc:  # noqa: BLE001 — the owner must get an answer
        logger.exception("/reset failed")
        await message.answer(f"❌ Перезапуск не удался: {exc}")
        return
    queue = chat_queue.current()
    waiting = 0
    if queue is not None:
        try:
            waiting = queue.pending_count(message.chat.id)
        except Exception:  # noqa: BLE001 — a count must not cost the report
            logger.warning("/reset: could not read the chat queue", exc_info=True)
    try:
        handoff = manager.pending_handoff_count()
    except Exception:  # noqa: BLE001
        handoff = 0
    await message.answer(reset_report(outcomes, waiting, handoff))


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
