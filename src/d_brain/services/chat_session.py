"""Persistent chat session manager — backed by ONE interactive Claude session.

Migrated from per-user `claude -p --resume` (which moves to the paid Agent SDK
credit on 2026-06-15) to a single long-lived interactive tmux session shared
across the bot. Conversational continuity lives in the live session itself
plus the vault (durable-state-first), so per-user --resume bookkeeping is
gone. The public interface (send_message / reset / compact) is unchanged so
the chat handlers don't need to change.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from d_brain.config import get_settings
from d_brain.services import ask_health
from d_brain.services.runtime import get_ask_lock, get_session

logger = logging.getLogger(__name__)

_UNRESOLVED = object()

_STATUS_MESSAGES = {
    "rate_limited": "⏳ Лимит подписки исчерпан. Вернусь, когда он обновится.",
    "logged_out": (
        "🔑 Нужен вход в подписку ИИ. На сервере выполните dbrain login: "
        "откройте ссылку на компьютере и введите код. Потом напишите сюда снова."
    ),
    "timeout": "⌛ Превышено время ожидания ответа. Попробуй ещё раз.",
    "error": "❌ Ошибка сессии. Попробуй позже.",
    # Busy-panel UX finding (2026-08-22): a legitimately busy panel (a
    # previous turn still running after the full busy-wait budget) reads as
    # honest, not scary — showing "❌ Ошибка сессии" here misled the owner into
    # thinking something had crashed at 04:27 UTC when nothing was actually
    # wrong. B3 fix (2026-08-22): the ORIGINAL wording here promised "отвечу,
    # как только освобожусь" (I'll answer once I'm free) — but no queue
    # mechanism exists to make that true; this message is a one-shot reply
    # to the message that triggered it, nothing re-sends the answer later.
    # Reworded to state the fact (still busy) without promising a callback.
    "busy": "🔧 Предыдущая задача ещё выполняется, ничего не сломалось. "
    "Напиши ещё раз через минуту-другую.",
    # agent-infra-backlog item 22 (2026-09): the pane is busy with a leftover
    # turn that demonstrably kept making progress across the whole wait — a
    # live, working turn (typically an unattended agent cascade), not a
    # wedged one. Deliberately does NOT promise a callback/auto-reply (same
    # constraint the "busy" wording above was reworded for): nothing
    # re-sends the answer later, this is a one-shot reply to the message
    # that triggered it.
    # Step D (agent-infra-backlog item 22): /stop now actually reaches the
    # interrupt path for a pane-active-but-lock-free turn too (see
    # ClaudeSession.is_pane_turn_active / chat.py's stop-word handling), so
    # this message can honestly point the user at it.
    "busy_active": "🛠 Идёт длинная фоновая задача — сессия занята, но канал "
    "доставки в порядке. Напиши ещё раз позже. Прервать её — /stop.",
}
# R2c (Fable audit, F2 class): the ceiling's honest message for "no reply
# markers ever appeared at all" — distinguished from the generic "timeout"
# message by a substring of AskResult.detail set in claude_session.ask().
_NO_MARKERS_DETAIL_MARKER = "no reply markers ever appeared"
_NO_MARKERS_MESSAGE = (
    "⌛ Ответ, похоже, готов в терминале, но его маркеры доставки потерялись "
    "— автоматически переслать не смог. Проверь <code>dbrain attach</code> "
    "или напиши ещё раз."
)


def _busy_active_message(busy_seconds: float | None) -> str:
    """The "busy_active" message, special-cased with elapsed time when
    available — mirrors how ``_NO_MARKERS_MESSAGE`` above is picked by a
    substring of ``AskResult.detail`` rather than the plain status→string
    map. Never promises a callback (see the map entry's comment)."""
    base = _STATUS_MESSAGES["busy_active"]
    if not busy_seconds or busy_seconds < 60:
        return base
    minutes = max(1, round(busy_seconds / 60))
    return (
        f"🛠 Идёт длинная фоновая задача (~{minutes} мин) — сессия занята, "
        "но канал доставки в порядке. Напиши ещё раз позже. Прервать её — "
        "/stop."
    )


class ChatSessionManager:
    """Routes chat messages to the shared interactive session."""

    def __init__(
        self,
        vault_path: Path | str,
        session: Any | None = None,
        health_dir: Path | str | None = None,
    ) -> None:
        self.vault_path = Path(vault_path)
        self._session = session if session is not None else get_session(get_settings())
        # Where the delivery-health ledger lives. Resolved lazily and never
        # fatally: a caller that hands us a session (tests, tools) may have no
        # Settings at all, and health bookkeeping must not gate replies.
        self._health_dir: Any = (
            Path(health_dir) if health_dir is not None else _UNRESOLVED
        )

    def _health_target(self) -> Path | None:
        if self._health_dir is _UNRESOLVED:
            try:
                self._health_dir = get_settings().runtime_dir
            except Exception:  # noqa: BLE001 — no settings ⇒ no ledger, no noise
                self._health_dir = None
        return self._health_dir

    def _record_health(self, status: str) -> None:
        target = self._health_target()
        if target is None:
            return
        try:
            ask_health.record(target, status)
        except Exception:  # noqa: BLE001 — never break a reply over telemetry
            logger.warning("ask-health record failed", exc_info=True)

    async def send_message(self, user_id: int, prompt: str) -> str:
        """Send a message to the session and return the reply text.

        Serialized via the process-wide ask-lock; runs the blocking ask() in a
        worker thread so the event loop stays responsive.
        """
        async with get_ask_lock():
            res = await asyncio.to_thread(self._session.ask, prompt)
        # Score what the USER got, not what ask() returned. An `ok` with an
        # empty body is a delivered-nothing turn: chat.py retries it once and,
        # if the retry is empty too, answers "Claude не ответил дважды".
        # Recording that as a success would clear the fail streak on exactly
        # the shape of failure this ledger exists to catch — a health signal
        # reading green while the user receives silence (2026-08-20).
        if res.ok and not (res.reply or "").strip():
            self._record_health("error")
        else:
            self._record_health(res.status)
        if res.ok:
            reply = res.reply or ""
            if res.salvaged:
                # backlog item 10 (2026-08-21): delivered from an
                # unterminated <<<R:id>>> span because the closing marker
                # never appeared. Health/streak accounting is unaffected —
                # from ask_health's perspective this IS "ok": the user
                # genuinely received a real answer.
                notice = "⚠️ <i>ответ восстановлен без закрывающего маркера</i>"
                reply = f"{notice}\n\n{reply}"
            return reply
        logger.warning("session ask for user %d returned %s", user_id, res.status)
        if (
            res.status == "timeout"
            and res.detail
            and _NO_MARKERS_DETAIL_MARKER in res.detail
        ):
            return _NO_MARKERS_MESSAGE
        if res.status == "busy_active":
            return _busy_active_message(res.busy_seconds)
        return _STATUS_MESSAGES.get(res.status, _STATUS_MESSAGES["error"])

    async def send_control(self, text: str) -> None:
        """Fire-and-forget a client-side Claude Code command into the session."""
        async with get_ask_lock():
            await asyncio.to_thread(self._session.send_control, text)

    # ── steering: deliberately NOT under the ask-lock — the lock is held by
    # the in-flight turn these calls are aimed at.

    def is_turn_active(self) -> bool:
        return self._session.is_turn_active()

    def is_pane_turn_active(self) -> bool:
        """Step D (agent-infra-backlog item 22): true iff the PANE shows the
        main turn running, regardless of whether the ask-lock is held — see
        ClaudeSession.is_pane_turn_active."""
        return self._session.is_pane_turn_active()

    def is_steerable_turn(self) -> bool:
        return self._session.is_steerable_turn()

    async def steer(self, text: str) -> None:
        await asyncio.to_thread(self._session.steer, text)

    async def interrupt(self) -> None:
        await asyncio.to_thread(self._session.interrupt)

    async def resend_last_reply(self, user_id: int) -> tuple[str, str | None]:
        """Status + optional body for ``/resend`` (backlog item 14).

        Deliberately NOT under ``get_ask_lock()`` — same reasoning as
        ``steer``/``interrupt`` above: the whole point of ``/resend`` is
        that it must work even while the main session is mid-turn and
        holding that lock.
        """
        logger.info("resend_last_reply requested by user %d", user_id)
        return await asyncio.to_thread(self._session.last_reply_for_resend)

    def reset(self, user_id: int) -> None:
        """Clear the live session context (durable data in files is kept)."""
        self._session.clear()
        logger.info("session cleared (reset) requested by user %d", user_id)

    def force_recover(self, user_id: int) -> bool:
        """User-facing `/relogin`: kill + recreate the tmux session so a
        fresh `claude` process re-reads the credential file at its own
        start. Unlike `reset()`, this survives a session that isn't just
        stale but actually logged out (2026-08-24 incident) — `/new`'s
        `clear()` types `/clear` into the live pane, which does nothing for
        an expired-token session since auth is only read at process start,
        never mid-session. Same primitive the watchdog already uses for
        dead/hung recovery (`force_recover` on the session object), just
        exposed here for the one case the watchdog deliberately leaves to a
        human: `logged_out` alerts but never auto-kills. Returns False (no
        action taken) if a real turn is currently in flight — never yanks
        the pane out from under a live reply.
        """
        recovered = self._session.force_recover()
        logger.info(
            "force_recover requested by user %d -> %s", user_id, recovered
        )
        return recovered

    async def compact(self, user_id: int) -> str:
        """Durable-state-first: clearing is the compaction; memory lives in
        files, so there is nothing to summarize into the session."""
        # M1 fix (2026-08-22): clear() now sends a blocking /clear + up to a
        # real ~10s resync poll (see B1) instead of being near-instant —
        # offload it so this coroutine doesn't block the event loop.
        await asyncio.to_thread(self._session.clear)
        return "🧹 Сессия очищена (важные данные сохранены в файлах)."
