"""Unified private chat handler with persistent Claude sessions.

Voice + text only (v3.0): replaces the legacy split handlers for private chats.
Every message is saved to daily (safety net) and routed IMMEDIATELY through
ChatSessionManager for Claude to process and respond — no debounce buffer.
"""

import asyncio
import html
import logging
import re
import time
from datetime import datetime
from typing import Any

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.types import Message

from d_brain import logsafe
from d_brain.bot.formatters import send_response
from d_brain.config import get_settings
from d_brain.services.chat_session import ChatSessionManager
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


async def _dispatch_text(bot: Bot, chat_id: int, user_id: int, text: str) -> None:
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
        # Step D (agent-infra-backlog item 22): during an unattended long
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
            # the session — injecting user text would contaminate it.
            await bot.send_message(
                chat_id,
                "🔧 Идёт фоновое обслуживание — повтори сообщение через "
                "несколько минут, отвечу как освобожусь.",
            )
            return
        await manager.steer(text)
        await bot.send_message(chat_id, "↪️ Передал в текущую задачу.")
        return
    await _process_and_reply(bot, chat_id, user_id, text)


async def _process_and_reply(bot: Bot, chat_id: int, user_id: int, prompt: str) -> None:
    """Send the prompt to the shared session and deliver the reply."""
    typing_task = asyncio.create_task(_typing_loop(bot, chat_id))
    try:
        manager = _get_manager()
        response = await manager.send_message(user_id, prompt)

        if response:
            await send_response(bot, chat_id, response)
        else:
            logger.warning(
                "Empty response from Claude for user %d, retrying...", user_id
            )
            # Retry once before giving up — don't reset session on first empty
            response = await manager.send_message(user_id, prompt)
            if response:
                await send_response(bot, chat_id, response)
            else:
                logger.warning("Empty response after retry for user %d", user_id)
                await bot.send_message(
                    chat_id,
                    "Claude не ответил дважды. Повтори сообщение.",
                )

    except Exception as e:
        logger.exception("Chat session error for user %d", user_id)
        error_text = f"Error: {html.escape(logsafe.redact(str(e))[:200])}"
        try:
            await bot.send_message(chat_id, error_text)
        except Exception:
            logger.exception("Failed to send error message")
    finally:
        typing_task.cancel()


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
    "Я принимаю голос, текст, фото и файлы. "
    "Этот тип сообщения обработать не могу."
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
    happened before the call (agent-infra-backlog item 21, step 5).
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


# --- Busy guard for the media path (agent-infra-backlog item 21, step 1) ---


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
# instead of wedging the media path permanently (items 22/23: never let one
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

    Contract: returning ``False`` means the caller now HOLDS the dispatch
    claim and MUST release it with ``_release_media_dispatch()`` in a
    ``finally``. Returning ``True`` means no claim is held.
    """
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
) -> None:
    _album_buf.setdefault(group_id, []).append(item)
    if group_id not in _album_tasks:
        _album_tasks[group_id] = asyncio.create_task(
            _flush_album(bot, chat_id, user_id, group_id)
        )


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
        await _process_and_reply(bot, chat_id, user_id, build_album_prompt(items))
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
            bot, message.chat.id, message.from_user.id, f"[voice] {transcript}"
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

    await _dispatch_text(bot, message.chat.id, message.from_user.id, text)


@router.message(
    F.photo | F.document | F.video | F.audio | F.animation | F.video_note
)
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

        # Step 1 (agent-infra-backlog item 21): the universal busy guard,
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
                await queue_album_item(
                    bot,
                    chat_id=message.chat.id,
                    user_id=message.from_user.id,
                    group_id=str(group_id),
                    item={
                        "kind": kind,
                        "rel_path": rel_path,
                        "caption": caption,
                        "fwd": fwd,
                        "model_rel_path": prep.model_rel_path,
                        "instruction": prep.instruction,
                    },
                )
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
                bot, message.chat.id, message.from_user.id, prompt
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
