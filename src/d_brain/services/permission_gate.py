"""Permission dialogs: the CLI asks, the person answers from Telegram.

The pain (2026-11-05 incident): even under bypass
permissions Claude Code asks a person to approve some tool calls of
background agents — a dangerous ``rm`` on a glob, "This command requires
approval" — and draws the question over the input box of the bot's own
session::

    Do you want to proceed?
    ❯ 1. Yes
      2. No
    Esc to cancel · Tab to amend · …

The bot read that as an idle prompt and pasted the next message into it. The
paste is dropped and the Enter after it APPROVES the highlighted "Yes"
(verified live, Claude Code 2.1.318); on the instances the ask then hung for
15-25 minutes until the stall detector's Escape refused the agent's command
and the person got "❌ Ошибка сессии".

The owner's decision (2026-11-05): do not forbid such commands, and never answer
for the person. Ask instead: one Telegram message with «Да» / «Нет», and the
button pressed there is the only thing that ever answers the dialog.

How it fits together
--------------------

* ``tmux_parse.read_permission_dialog`` recognises the dialog
  (``PaneState.PERMISSION_DIALOG``). ``ClaudeSession`` never types into it:
  ask() hands its prompt back with status ``permission`` (the chat path parks
  it in the per-chat queue), and Enter is refused while the dialog is up.
* :class:`PermissionGate` (this module, one task in the bot process) looks at
  every Claude session's pane every few seconds. A new dialog becomes a
  record on disk (``<runtime_dir>/permission.json``) and a message with
  buttons. While that record exists the chat path does not even try the
  main session: messages wait in the queue.
* A button press carries the record's nonce. It acts only if it is the
  current record AND the very same dialog (same fingerprint) is still on the
  screen; then the dialog's own digit for "Yes" or "No" is pressed — never
  Enter, never the "don't ask again" option. A stale button says so and does
  nothing.
* A dialog that closes without us (answered in the terminal, agent stopped)
  drops its record; the message is edited to say so.

No answer is ever given by timeout. After ``remind_after`` seconds without an
answer the person gets ONE reminder; the dialog and its buttons stay valid
for as long as it is on screen. Nothing else in the bot is blocked: commands,
/work and /stop work, and the queue resumes the moment the dialog is gone.
Waiting messages are still bounded by the queue's own age cap, which retires
them out loud. The watchdog treats the dialog as serviceable (never "hung")
and its long-run cap does not interrupt over it — an Escape there would
refuse the command on the person's behalf.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from d_brain.services.outbox import write_json_atomic

logger = logging.getLogger(__name__)

STATE_NAME = "permission.json"
CALLBACK_PREFIX = "perm:"
# How often the panes are looked at. A capture-pane per Claude session.
POLL_SECONDS = 3.0
# One reminder after this long without an answer (Settings.permission_remind_after).
DEFAULT_REMIND_AFTER = 1800.0
# The command shown in the message is cut to this many characters.
COMMAND_LIMIT = 600

_ALLOWED = "✅ <b>Разрешил.</b>"
_DENIED = "🚫 <b>Отклонил.</b> Команда не выполнена."
_STALE = "⌛ <i>Запрос устарел: окно уже закрыто. Ничего не нажимал.</i>"
_CLOSED = "⌛ <i>Окно закрылось без ответа отсюда. Ничего не нажимал.</i>"
_QUESTION = "Разрешить?"
REMINDER = (
    "⏳ Всё ещё жду твоего решения по запросу выше. Сам ничего не разрешаю, "
    "а новые сообщения пока ждут в очереди."
)

_RM_RE = re.compile(
    r"Dangerous (rm|rmdir) operation on ([^:]+):\s*(.+?)"
    r"(?:\s+in\s+`.*`)?(?:\s+\(rewrite .*\))?\s*$"
)


# ── the record ──────────────────────────────────────────────────────────


@dataclass
class Record:
    """One dialog the person has been (or is about to be) asked about."""

    nonce: str
    slot: str
    fingerprint: str
    seen_at: float
    body: str  # the message without its last line ("Разрешить?" / outcome)
    message_id: int | None = None
    reminded_at: float | None = None


def _path(runtime_dir: Path | str) -> Path:
    return Path(runtime_dir) / STATE_NAME


def load(runtime_dir: Path | str) -> Record | None:
    try:
        raw = json.loads(_path(runtime_dir).read_text())
        return Record(**raw)
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — a broken record is no record
        logger.warning("permission: unreadable %s", _path(runtime_dir), exc_info=True)
        return None


def save(runtime_dir: Path | str, rec: Record) -> None:
    write_json_atomic(_path(runtime_dir), asdict(rec))


def clear(runtime_dir: Path | str) -> None:
    _path(runtime_dir).unlink(missing_ok=True)


def pending(runtime_dir: Path | str | None) -> bool:
    """True while a dialog in this session waits for the person. The chat
    path parks messages instead of entering ask() while this holds."""
    if not isinstance(runtime_dir, (str, Path)):
        return False
    return _path(runtime_dir).exists()


# ── what the person reads ──────────────────────────────────────────────


def _cut(text: str, limit: int = COMMAND_LIMIT) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _what(dialog: Any) -> str:
    reason = dialog.reason or ""
    m = _RM_RE.search(reason)
    if m:
        kind, why, target = m.group(1), m.group(2), m.group(3).strip()
        noun = "папки" if kind == "rmdir" else "файлы"
        folder, _, mask = target.rpartition("/")
        if folder and mask:
            line = (
                f"Удалить {noun} <code>{html.escape(mask)}</code> в папке "
                f"<code>{html.escape(folder)}</code>."
            )
        else:
            line = f"Удалить {noun} <code>{html.escape(target)}</code>."
        if "variable" in why:
            line += (
                " Путь задан переменной: если она окажется пустой, удаление "
                "заденет не ту папку."
            )
        return line
    if "could not be checked" in reason:
        return (
            "Выполнить скрипт, который удаляет файлы. Проверить его заранее не удалось."
        )
    if "requires approval" in reason or "requires confirmation" in reason:
        return "Выполнить команду, для которой нужно твоё подтверждение."
    if dialog.title.lower().startswith("bash"):
        return "Выполнить команду."
    return f"Действие: {html.escape(dialog.title or 'без названия')}."


def describe(dialog: Any, slot: str) -> str:
    """The message body (HTML, without the closing question)."""
    if dialog.agent:
        who = "Фоновая задача"
    elif slot == "cron":
        who = "Задача по расписанию"
    else:
        who = "Сессия"
    parts = [f"🔐 <b>{who} просит подтвердить действие</b>", "", _what(dialog)]
    if dialog.description:
        parts.append(f"<i>{html.escape(dialog.description)}</i>")
    if dialog.command:
        parts += ["", f"<code>{html.escape(_cut(dialog.command))}</code>"]
    if dialog.agent:
        parts += ["", f"<i>Спрашивает фоновый агент «{html.escape(dialog.agent)}».</i>"]
    parts += [
        "",
        "Пока ты не ответишь, я ничего не нажимаю, а новые сообщения ждут в очереди.",
    ]
    return "\n".join(parts)


def callback_data(slot: str, nonce: str, allow: bool) -> str:
    return f"{CALLBACK_PREFIX}{slot}:{nonce}:{'y' if allow else 'n'}"


def parse_callback(data: str) -> tuple[str, str, bool] | None:
    """``(slot, nonce, allow)`` of a button, or None for anything else."""
    if not data.startswith(CALLBACK_PREFIX):
        return None
    parts = data[len(CALLBACK_PREFIX) :].split(":")
    if len(parts) != 3 or parts[2] not in ("y", "n") or not parts[1]:
        return None
    return parts[0], parts[1], parts[2] == "y"


def keyboard(slot: str, nonce: str) -> Any:
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Да", callback_data=callback_data(slot, nonce, True)
                ),
                InlineKeyboardButton(
                    text="Нет", callback_data=callback_data(slot, nonce, False)
                ),
            ]
        ]
    )


def _without_question(current_html: str) -> str:
    text = current_html.rstrip()
    if text.endswith(_QUESTION):
        text = text[: -len(_QUESTION)].rstrip()
    return text


# ── the gate ─────────────────────────────────────────────────────────────


def _present(session: Any) -> bool:
    """Whether the session's pane exists. A session without a quiet
    ``exists()`` probe is assumed present (its ``permission_dialog`` copes)."""
    probe = getattr(session, "exists", None)
    if not callable(probe):
        return True
    return bool(probe())


class PermissionGate:
    """Watches the Claude sessions' panes and turns each permission dialog
    into one Telegram question; applies the person's answer.

    ``sessions`` maps a slot name ("main", "cron", "duty") to a session.
    Only sessions with ``permission_dialog()`` / ``answer_permission()``
    (``ClaudeSession``) take part; Codex has no such dialog."""

    def __init__(
        self,
        sessions: dict[str, Any],
        *,
        chat_id: int | None,
        allowed_ids: list[int] | None = None,
        remind_after: float = DEFAULT_REMIND_AFTER,
        clock: Any = time.time,
    ) -> None:
        self._sessions = {
            slot: s
            for slot, s in sessions.items()
            if s is not None and callable(getattr(s, "permission_dialog", None))
        }
        self._chat_id = chat_id
        self._allowed = set(allowed_ids or [])
        self._remind_after = remind_after
        self._clock = clock
        self._lock = asyncio.Lock()

    @staticmethod
    def _runtime_dir(session: Any) -> Path:
        return Path(session.runtime_dir)

    async def step(self, bot: Any) -> None:
        """One look at every pane. Never raises."""
        async with self._lock:
            for slot, session in self._sessions.items():
                try:
                    await self._step_one(bot, slot, session)
                except Exception:  # noqa: BLE001 — one session must not stop the rest
                    logger.exception("permission: step failed for %s", slot)

    async def _step_one(self, bot: Any, slot: str, session: Any) -> None:
        rt = self._runtime_dir(session)
        rec = load(rt)
        # Only a session that is actually there is probed. The gate watches
        # every slot the configuration allows, but some exist only on demand
        # (the duty session is built on the first busy-main fallback, and on
        # most instances never is): probing an absent one logged a tmux
        # "can't find session" every pass. Absent reads as "no dialog", so a
        # recorded question still closes below.
        if await asyncio.to_thread(_present, session):
            dialog = await asyncio.to_thread(session.permission_dialog)
        else:
            dialog = None
        if rec is not None and (
            dialog is None or dialog.fingerprint != rec.fingerprint
        ):
            logger.info(
                "permission: dialog %s in %s closed without us", rec.nonce, slot
            )
            clear(rt)
            await self._edit(bot, rec, _CLOSED)
            rec = None
        if dialog is None:
            return
        now = self._clock()
        if rec is None:
            rec = Record(
                nonce=secrets.token_hex(4),
                slot=slot,
                fingerprint=dialog.fingerprint,
                seen_at=now,
                body=describe(dialog, slot),
            )
            # On disk BEFORE the message: the chat path stops typing into
            # this session from now on, even if Telegram is unreachable.
            save(rt, rec)
            logger.warning(
                "permission: %s session asks to approve (%s): %s",
                slot,
                dialog.agent or "main",
                (dialog.reason or dialog.title)[:200],
            )
        if rec.message_id is None:
            rec.message_id = await self._ask(bot, rec)
            if rec.message_id is not None:
                save(rt, rec)
            return
        if (
            self._remind_after > 0
            and rec.reminded_at is None
            and now - rec.seen_at >= self._remind_after
        ):
            if await self._send(bot, REMINDER, reply_to=rec.message_id) is not None:
                rec.reminded_at = now
                save(rt, rec)

    async def _send(self, bot: Any, text: str, **kw: Any) -> int | None:
        if self._chat_id is None:
            return None
        extra: dict[str, Any] = {}
        if kw.get("reply_to"):
            extra = {
                "reply_to_message_id": int(kw["reply_to"]),
                "allow_sending_without_reply": True,
            }
        if kw.get("reply_markup") is not None:
            extra["reply_markup"] = kw["reply_markup"]
        try:
            msg = await bot.send_message(self._chat_id, text, **extra)
        except Exception:  # noqa: BLE001 — retried on the next step
            logger.warning("permission: could not send to Telegram", exc_info=True)
            return None
        return getattr(msg, "message_id", None)

    async def _ask(self, bot: Any, rec: Record) -> int | None:
        if self._chat_id is None:
            logger.warning("permission: no owner chat id — nobody to ask")
            return None
        return await self._send(
            bot,
            f"{rec.body}\n\n{_QUESTION}",
            reply_markup=keyboard(rec.slot, rec.nonce),
        )

    async def _edit(self, bot: Any, rec: Record, outcome: str) -> None:
        if rec.message_id is None or self._chat_id is None:
            return
        try:
            await bot.edit_message_text(
                f"{rec.body}\n\n{outcome}",
                chat_id=self._chat_id,
                message_id=rec.message_id,
                reply_markup=None,
            )
        except Exception:  # noqa: BLE001 — a cosmetic edit
            logger.warning("permission: could not edit the question", exc_info=True)

    async def answer(
        self,
        slot: str,
        nonce: str,
        allow: bool,
        *,
        user_id: int | None,
        current_html: str = "",
    ) -> tuple[str, str | None]:
        """Apply one button press. ``(toast, new message text or None)``."""
        if self._allowed and user_id not in self._allowed:
            logger.warning("permission: button pressed by a stranger (%s)", user_id)
            return "Эта кнопка не для тебя.", None
        stale = f"{_without_question(current_html)}\n\n{_STALE}"
        async with self._lock:
            session = self._sessions.get(slot)
            rec = load(self._runtime_dir(session)) if session is not None else None
            if rec is None or rec.nonce != nonce:
                return "Запрос устарел.", stale
            result = await asyncio.to_thread(
                session.answer_permission, rec.fingerprint, allow
            )
            if result == "stuck":
                return "Нажал, но окно ещё на экране. Попробуй ещё раз.", None
            clear(self._runtime_dir(session))
            if result != "done":
                return "Запрос устарел.", f"{rec.body}\n\n{_STALE}"
            logger.warning(
                "permission: %s dialog %s answered %s by the person",
                slot,
                rec.nonce,
                "YES" if allow else "NO",
            )
            outcome = _ALLOWED if allow else _DENIED
            return ("Разрешил." if allow else "Отклонил."), f"{rec.body}\n\n{outcome}"


async def run(
    bot: Any, gate: PermissionGate, *, poll_seconds: float = POLL_SECONDS
) -> None:
    """The background task: look at the panes forever."""
    logger.info("permission gate started (poll %.0fs)", poll_seconds)
    while True:
        try:
            await gate.step(bot)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the watcher must outlive a bad pass
            logger.exception("permission: pass failed")
        await asyncio.sleep(poll_seconds)


# ── process-wide handle (same pattern as chat_queue / outbox) ─────────────

_current: PermissionGate | None = None


def configure(gate: PermissionGate) -> PermissionGate:
    global _current  # noqa: PLW0603
    _current = gate
    return gate


def current() -> PermissionGate | None:
    return _current


def reset() -> None:
    global _current  # noqa: PLW0603
    _current = None
