"""Persistent chat session manager — backed by ONE interactive Claude session.

Migrated from per-user `claude -p --resume` (which moves to the paid Agent SDK
credit on 2026-06-15) to a single long-lived interactive tmux session shared
across the bot. Conversational continuity lives in the live session itself
plus the vault (durable-state-first), so per-user --resume bookkeeping is
gone. The public interface (send_message / reset / compact) is unchanged so
the chat handlers don't need to change.
"""

import asyncio
import functools
import html
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from d_brain.config import Settings, get_settings
from d_brain.services import ask_health, long_run
from d_brain.services.claude_session import (
    DEFAULT_LONG_RUN_STALE_AFTER,
    DEFAULT_TIMEOUT,
    AskResult,
)
from d_brain.services.runtime import get_ask_lock, get_duty_session, get_session

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
    # honest, not scary — showing "❌ Ошибка сессии" here misled the owner
    # into thinking something had crashed when nothing was actually
    # wrong. B3 fix (2026-08-22): the ORIGINAL wording here promised "отвечу,
    # как только освобожусь" (I'll answer once I'm free) — but no queue
    # mechanism exists to make that true; this message is a one-shot reply
    # to the message that triggered it, nothing re-sends the answer later.
    # Reworded to state the fact (still busy) without promising a callback.
    "busy": "🔧 Предыдущая задача ещё выполняется, ничего не сломалось. "
    "Напиши ещё раз через минуту-другую.",
    # (2026-09): the pane is busy with a leftover
    # turn that demonstrably kept making progress across the whole wait — a
    # live, working turn (typically an unattended agent cascade), not a
    # wedged one. Deliberately does NOT promise a callback/auto-reply (same
    # constraint the "busy" wording above was reworded for): nothing
    # re-sends the answer later, this is a one-shot reply to the message
    # that triggered it.
    # Step D: /stop now actually reaches the
    # interrupt path for a pane-active-but-lock-free turn too (see
    # ClaudeSession.is_pane_turn_active / chat.py's stop-word handling), so
    # this message can honestly point the user at it.
    "busy_active": "🛠 Идёт длинная фоновая задача — сессия занята, но канал "
    "доставки в порядке. Напиши ещё раз позже. Прервать её — /stop.",
    # The owner's /reset cut this very turn short (ClaudeSession
    # .request_abort). Neutral in ask_health: not a delivery failure.
    "reset": "⏹ Этот ход прерван командой /reset — если ответ ещё нужен, "
    "повтори сообщение.",
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


# agent-infra: the chat path now caps a main-session turn at
# settings.chat_turn_timeout instead of inheriting DEFAULT_TIMEOUT. On that
# ceiling ask() returns status "timeout" with detail "no reply in Ns" and —
# unlike every other exit — does NOT mark the rid handled and leaves the
# inflight marker in place (claude_session.ask's tail comment: "prompt is
# still physically in the pane → keep inflight"). So the reply, if it ever
# lands, is delivered later by the watchdog's orphan path
# (pop_orphan_replies). The generic "попробуй ещё раз" wording would be a
# lie here — a retry would start a SECOND turn behind the first.
#
# All of that is TRUE OF THE CLAUDE ENGINE ONLY. Codex emits the identical
# status/detail pair from a deadline that killed the turn's process, so the
# detail marker below identifies the SHAPE of the outcome and never its
# consequences — those are decided by _turn_limit_keeps_inflight().
_TURN_LIMIT_DETAIL_MARKER = "no reply in "
# F5 (blind review, 2026-09-20): the ORIGINAL wording promised only the good
# outcome ("если ответ появится, пришлю"), and ~30 minutes later the
# watchdog's hard unattended cap (settings.long_run_max_seconds) would close
# that very turn and send "закрыл автоматически" — two messages about the
# same turn that contradict each other, with two different numbers in them.
# Name BOTH branches up front instead, without quoting either limit: the two
# are configured independently and any number printed here would be the
# chat-side one, not the one the watchdog will actually enforce.
_TURN_LIMIT_MESSAGE = (
    "⏳ Ход идёт дольше лимита ожидания — я перестал его ждать, но он "
    "продолжается. Если он завершится сам, пришлю ответ отдельным "
    "сообщением; если затянется сверх лимита фоновой работы — вотчдог "
    "закроет его, и об этом я тоже сообщу."
)
# F4 (blind review, 2026-09-20): under the Codex engine the same ceiling is
# NOT a "keep waiting" outcome. CodexExecDriver's hard deadline terminates
# the turn's process (_terminate → SIGINT → kill) before returning the very
# same `timeout` / "no reply in Ns" pair, so there is no in-flight request
# left, no unmarked rid, and nothing for the watchdog's orphan path to
# deliver later. Promising a late reply there would be a plain lie.
_TURN_LIMIT_MESSAGE_KILLED = (
    "⌛ Ход прервался по лимиту ожидания — этот запрос уже не завершится и "
    "ответа по нему не будет. Повтори его, пожалуйста."
)

# ── neutral ledger row (blind review F2, 2026-09-20) ─────────────────────
# Written to ask_health for an outcome that is NEITHER a delivered answer
# nor a delivery failure. `next_health` treats any status outside
# FAILURE_STATUSES and different from SUCCESS_STATUS as neutral: it records
# `last_status` (so /work and a human reading the ledger see reality) while
# leaving `fail_streak` exactly as it was — it neither grows the streak nor
# clears a genuine one. `test_chat_session.py` pins that neutrality so a
# future edit to FAILURE_STATUSES cannot silently turn it into a restart
# trigger.
#
# A chat turn outlived `chat_turn_timeout` on an engine that keeps the
# request in flight. The user's answer still arrives, via the watchdog's
# orphan path; counting it as `timeout` (a FAILURE_STATUS) meant three
# legitimately long turns inside delivery_guard's window restarted
# dbrain-bot.service for no reason — a false failure by construction.
#
# NOTE — this row is written by a turn that ENDED. There is deliberately no
# sibling constant for the duty detour: see is_main_busy()'s docstring for
# why that path must not touch the ledger at all (review round 3, R2).
TURN_LIMIT_STATUS = "turn_limit"

# How long is_main_busy() waits between its two pane probes. A single probe
# would also fire on a turn that is a second away from finishing (the user
# would then be answered from the duty session for no reason), so the pane
# has to still look busy on a second look before we route around the main
# brain. Module-level so tests can shrink it.
_MAIN_BUSY_CONFIRM_SECONDS = 3.0

# /reset (circuit_reset): how long to wait for a turn to let go of a session
# after it was told to stop, and the poll step for that wait and for the
# read-back. Module-level so tests can shrink them.
_RESET_RELEASE_WAIT = 60.0
_RESET_READBACK_WAIT = 15.0
_RESET_POLL_SECONDS = 1.0

# Duty → main handoff. The duty session answers without the conversation's
# context and is told to "accept, record, and hand long work to the main
# session" — but that handoff used to be words only: nothing ever told the
# main session (2026-09-25, 22:31/22:34 — two handoffs nobody picked up).
# Now every message the duty session actually answered is noted here, and
# the NEXT main turn carries the list, so the main session picks it up the
# moment it is free. Cleared only after a main turn delivered a real reply.
_HANDOFF_FILE = "duty-handoff.json"
# Wall-clock time of the last /reset in this process. A main turn that was
# already running then and came back without an answer was cut short by it —
# reported as "reset" on BOTH engines (Claude's ask() says so itself via its
# abort stamp; Codex's SIGINT'ed turn comes back as a plain "error").
_last_reset_ts = 0.0
# One /reset at a time: a second one would kill the sessions the first is
# still reading back.
_reset_running = False
_HANDOFF_MAX = 10
_HANDOFF_TEXT_CAP = 1500


@dataclass(frozen=True)
class ResetOutcome:
    """What /reset did to ONE session, verified by reading it back."""

    name: str  # "main" | "duty"
    ok: bool  # recreated AND read back as up and idle
    detail: str = ""

# Stamp file (inside settings.duty_dir) holding the unix time of the last
# duty turn — the input to the idle-reset decision.
_DUTY_STAMP_NAME = "last_used"


@dataclass(frozen=True)
class Busy:
    """"The main session is working — this message has to wait its turn."

    Returned by ``send_message`` INSTEAD of a reply string when the engine
    reported ``busy_active``: the pane is busy with a turn that demonstrably
    kept making progress, i.e. healthy work, not a wedge. Until 's
    step 5 that outcome went straight to the duty session; now the caller
    parks the message in the per-chat queue and the MAIN session — the one
    with the conversation's context — answers it when it is free. The duty
    session stays for the ``busy`` case, where no progress was observed
    across the whole busy-wait budget and the pane is, by that evidence,
    wedged.

    Being a separate type rather than a magic string is what keeps every
    caller honest: a reply is a ``str``, and this is not one, so a path that
    forgets to handle it fails loudly instead of sending "\\x00busy" to the
    owner.

    ``fallback`` is the exact wording the user would have received before the
    queue existed. A caller with no queue configured (unit tests, one-off
    tools) hands it back to ``answer_from_duty`` and lands byte-for-byte on
    the pre-queue behavior.
    """

    busy_seconds: float | None = None
    fallback: str = ""


def wrap_duty_prompt(text: str) -> str:
    """Per-turn envelope for the duty session — the chat counterpart of
    ``cron_runner.wrap_job_prompt``.

    Pure function on purpose (no Settings, no session): the envelope is the
    contract the duty turn is judged by, so it is testable on its own. The
    marker instruction is NOT duplicated here — ``ask(wrap=True)`` appends
    it, exactly as for cron jobs.

    ACCEPTED RISK (blind review F12, 2026-09-20): the rule below tells the
    duty turn to write a thought into ``daily/`` — which the nightly
    pipeline may be processing at that very moment, since the whole point of
    this session is that the main one is busy. ``VaultStorage.append_to_daily``
    appends rather than rewrites, so a concurrent duty write cannot clobber
    the file wholesale; the general question of parallel vault writes is
    and is deliberately NOT solved here. The exclusion list
    below (``projects/*/status.md``, ``MEMORY.md``, ``.session/handoff.md``)
    covers the files that ARE rewritten whole.
    """
    return (
        "[ДЕЖУРНАЯ СЕССИЯ] Основная сессия сейчас занята длинной работой, "
        "поэтому это\nсообщение обрабатываешь ты — отдельная короткая сессия "
        "с тем же вольтом, но БЕЗ\nконтекста текущего разговора основной "
        "сессии.\n\n"
        "Правила этого хода:\n"
        "- Ответ короткий, в Telegram-HTML (<b> <i> <code> <a>), без "
        "Markdown.\n"
        "- Мысль, идею или задачу зафиксируй в вольте по обычным правилам "
        "(daily).\n"
        "- НЕ начинай длинную работу: никаких субагентов, веб-ресёрча, "
        "многошаговых\n  правок, рендера. Если запрос требует длинной "
        "работы — прими его, запиши и\n  скажи, что передашь в основную "
        "сессию, когда она освободится.\n"
        "- Не трогай файлы, с которыми прямо сейчас может работать основная "
        "сессия:\n  projects/*/status.md, MEMORY.md, .session/handoff.md.\n\n"
        f"Сообщение пользователя:\n{text}"
    )


def with_duty_handoff(prompt: str, items: list[dict]) -> str:
    """The main-session prompt with the duty session's pending handoff in
    front of it (see ``_HANDOFF_FILE``)."""
    lines = []
    for item in items:
        stamp = time.strftime("%d.%m %H:%M", time.localtime(float(item["ts"])))
        text = " ".join(str(item["text"]).split())
        lines.append(f"- [{stamp}] {text}")
    return (
        "[ПЕРЕДАЧА ОТ ДЕЖУРНОЙ СЕССИИ]\n"
        "Пока ты был занят, на сообщения ниже отвечала дежурная сессия — без "
        "контекста разговора. Всё, что она приняла «в очередь основной "
        "сессии» или обещала передать, теперь твоё: прочитай её записи "
        "«[дежурная сессия]» в сегодняшнем и вчерашнем daily, возьми "
        "обещанное в работу (длинное — агентам) и в начале ответа одной "
        "строкой скажи, что подхватил.\n\n"
        "Сообщения, на которые отвечала дежурная:\n"
        + "\n".join(lines)
        + "\n\nНовое сообщение пользователя:\n"
        + prompt
    )


def duty_header(busy_seconds: float | None) -> str:
    """The honest banner above a duty reply: which session is answering and
    what it does not know. Never silently impersonates the main brain."""
    if busy_seconds and busy_seconds >= 60:
        minutes = max(1, round(busy_seconds / 60))
        busy = f" (~{minutes} мин)"
    else:
        busy = ""
    return (
        f"🔁 <i>Основная сессия занята{busy} — отвечаю из дежурной, "
        "контекста текущего разговора у неё нет.</i>"
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


def busy_fallback_message(busy_seconds: float | None = None) -> str:
    """The pre-duty-session "main brain is busy" wording, for callers that
    need the fallback text itself (the chat handler's duty path). One
    definition, so the two paths can never drift apart."""
    return _busy_active_message(busy_seconds)


class ChatSessionManager:
    """Routes chat messages to the shared interactive session."""

    def __init__(
        self,
        vault_path: Path | str,
        session: Any | None = None,
        health_dir: Path | str | None = None,
        duty_session: Any | None = None,
    ) -> None:
        self.vault_path = Path(vault_path)
        self._session = session if session is not None else get_session(get_settings())
        # Where the delivery-health ledger lives. Resolved lazily and never
        # fatally: a caller that hands us a session (tests, tools) may have no
        # Settings at all, and health bookkeeping must not gate replies.
        self._health_dir: Any = (
            Path(health_dir) if health_dir is not None else _UNRESOLVED
        )
        # The duty session is auto-resolved from runtime ONLY when this
        # manager also built its main session from runtime. A caller that
        # injected a main session (tests, tools) would otherwise silently get
        # a REAL second engine session — started against the live runtime dir
        # — as the fallback for its fake one. Injected main ⇒ injected duty
        # or no duty at all.
        self._duty: Any = duty_session if duty_session is not None else _UNRESOLVED
        self._auto_duty = session is None
        # Serializes duty turns among themselves. Deliberately NOT the
        # process-wide ask-lock: the whole point of the duty session is to
        # run while the main one is busy (and, in the unattended-cascade
        # case, while nothing holds that lock at all).
        self._duty_lock = asyncio.Lock()
        self._settings: Any = _UNRESOLVED

    def _config(self) -> Settings | None:
        """Settings, resolved lazily and never fatally — same contract as
        ``_health_target``: no Settings ⇒ no duty session, not a broken
        reply."""
        if self._settings is _UNRESOLVED:
            try:
                self._settings = get_settings()
            except Exception:  # noqa: BLE001 — no settings ⇒ legacy behavior
                self._settings = None
        return self._settings

    def _health_target(self) -> Path | None:
        if self._health_dir is _UNRESOLVED:
            try:
                self._health_dir = get_settings().runtime_dir
            except Exception:  # noqa: BLE001 — no settings ⇒ no ledger, no noise
                self._health_dir = None
        return self._health_dir

    def _runtime_dir(self) -> Path | None:
        """Where the CROSS-PROCESS markers live: the ask-health ledger this
        process writes AND the watchdog's ``long-run.json`` it reads. One
        resolver for both, because in production they are literally the same
        directory (``settings.runtime_dir``) — and because a test that pins
        it must move both together or the two would describe different
        installs."""
        return self._health_target()

    def _engine(self) -> str:
        """Which engine backs the CHAT brain. Never fatal: no Settings ⇒
        "claude", the only engine this code shipped against."""
        settings = self._config()
        return str(getattr(settings, "chat_engine", "claude") or "claude")

    def _record_health(self, status: str) -> None:
        target = self._health_target()
        if target is None:
            return
        try:
            ask_health.record(target, status)
        except Exception:  # noqa: BLE001 — never break a reply over telemetry
            logger.warning("ask-health record failed", exc_info=True)

    async def send_message(self, user_id: int, prompt: str) -> str | Busy:
        """Send a message to the session and return the reply text — or
        ``Busy``, meaning "the main brain is working, park this".

        Serialized via the process-wide ask-lock; runs the blocking ask() in a
        worker thread so the event loop stays responsive.

        THE BUSY SPLIT is the heart of this method, and it
        is not a new judgement — it reads one ``ask()`` already makes:

        * ``busy_active`` — the pane is busy and made a REAL, RECENT change
          (chrome or pane.log growth) while we waited. That is live work by
          the main session, so the message waits for it: ``Busy`` goes back to
          the caller, which parks it in the per-chat queue. The duty session
          is not involved; the owner gets his answer from the session that
          has the conversation.
        * ``busy`` — no real progress at all across the whole busy-wait
          budget (up to 300s). That is failure class B3, the wedge signature,
          and it is exactly the emergency the duty session exists for. It
          keeps answering those, as it has since.

        The main outcome is scored in the health ledger on BOTH branches
        before anything else happens — that ledger tracks the MAIN delivery
        channel, and hiding a busy turn behind a queue ack or a duty reply
        would blind the DeliveryGuard. ``busy_active`` is neutral there and
        ``busy`` counts as a failure, which is what arms the restart backstop
        for a genuinely wedged pane.
        """
        handoff = self._pending_handoff()
        main_prompt = with_duty_handoff(prompt, handoff) if handoff else prompt
        started = time.time()
        async with get_ask_lock():
            res = await asyncio.to_thread(
                self._session.ask, main_prompt, timeout=self._main_turn_timeout()
            )
        if (
            not res.ok
            and res.status not in ("busy", "busy_active", "reset")
            and _last_reset_ts >= started
        ):
            res = AskResult("reset", detail=f"cut short by /reset ({res.status})")
        if handoff and res.ok and (res.reply or "").strip():
            self._clear_handoff(handoff[-1]["ts"])
        if res.status == "busy_active":
            self._record_health(res.status)
            logger.info(
                "main session busy with live work for user %d — parking the "
                "message instead of answering from the duty session",
                user_id,
            )
            # Notices are deliberately NOT popped here: nothing is being
            # delivered, so a parking notice would be dropped on the floor.
            # It rides out with whatever reply this message eventually gets.
            return Busy(
                busy_seconds=res.busy_seconds,
                fallback=_busy_active_message(res.busy_seconds),
            )
        if res.status == "busy":
            logger.warning(
                "main session showed no progress across the whole busy wait "
                "for user %d — falling back to the duty session",
                user_id,
            )
            text, answered = await self.answer_from_duty_detailed(
                user_id,
                prompt,
                busy_seconds=res.busy_seconds,
                fallback=_STATUS_MESSAGES["busy"],
            )
            # Scored AFTER the duty turn, on its outcome. The ledger's
            # question is "did the person get an answer", not "which
            # session produced it", and recording `busy` here before the
            # duty session had even been asked is what made a false-alarm
            # morning report «N ответов подряд не доставлено» about a
            # morning whose answers the owner had already read in Telegram
            # as they arrived.
            #
            # A duty session that ALSO came back empty leaves it at `busy`,
            # unchanged: then the person really did get nothing but a
            # brush-off, and the restart backstop the B3 fix armed for a
            # wedged pane (see ask_health's module docstring) must still see
            # the streak it was built for.
            #
            # THE TRADE, stated plainly (blind review 9): while the duty
            # session keeps covering, a permanently wedged main pane grows
            # no streak, so delivery_guard never restarts the bot over it.
            # That is the intended reading of the rule («→ ok либо
            # нейтральный статус, никогда busy») and it is the right call
            # here: restarting the BOT does not unwedge a tmux PANE, the
            # person is being answered, and the pane itself is covered by
            # two paths that do not depend on this ledger at all —
            # watchdog._is_hung → recovered_hung, and the long-run cap.
            # What this must never do is stay quiet while the person gets
            # nothing, and that case still scores `busy`.
            if answered:
                status = ask_health.SUCCESS_STATUS
            elif _last_reset_ts >= started:
                status = "reset"  # the duty turn was cut short by /reset
            else:
                status = res.status
            self._record_health(status)
        else:
            text = self._reply_text(user_id, res)
        # agent-infra: if this very turn had to park a pane
        # stuck on a background task, the owner must learn that the
        # conversation context is gone — right on the reply it affected.
        # Only on a non-empty reply: an empty one is chat.py's retry signal,
        # and the watchdog tick delivers the notice anyway. A duty reply
        # carries them too — the MAIN session's notices are about the
        # conversation this user is having, whoever ends up answering.
        if text.strip():
            notices = self._pop_notices()
            if notices:
                text = "\n\n".join(html.escape(n) for n in notices) + "\n\n" + text
        return text

    def _pop_notices(self, session: Any | None = None) -> list[str]:
        target = self._session if session is None else session
        pop = getattr(target, "pop_notices", None)
        if pop is None:
            return []
        try:
            return list(pop())
        except Exception:  # noqa: BLE001 — a notice must never cost the reply
            logger.warning("could not read session notices", exc_info=True)
            return []

    def _main_turn_timeout(self) -> float:
        """Ceiling for a chat-initiated main turn. 0 or no
        Settings ⇒ DEFAULT_TIMEOUT — exactly what this call passed before
        the setting existed."""
        settings = self._config()
        limit = getattr(settings, "chat_turn_timeout", 0.0) or 0.0
        return float(limit) if limit > 0 else float(DEFAULT_TIMEOUT)

    # ── duty session ───────────────────────────

    def _pane_active(self) -> bool:
        """Blocking pane probe, exception-proof. A driver without the probe
        (or one that throws) reads as idle: the duty detour is an
        optimization, never a reason to lose a message."""
        probe = getattr(self._session, "is_pane_turn_active", None)
        if probe is None:
            return False
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001 — never gate a reply on a probe
            logger.warning("pane-activity probe failed", exc_info=True)
            return False

    def _activity_fingerprint(self) -> Any | None:
        """The engine's "did anything move" snapshot, or None when the
        driver has none (Codex, test fakes) or it throws — None disables the
        no-change check rather than guessing."""
        probe = getattr(self._session, "activity_fingerprint", None)
        if probe is None:
            return None
        try:
            return probe()
        except Exception:  # noqa: BLE001 — never gate a reply on a probe
            logger.warning("pane activity fingerprint failed", exc_info=True)
            return None

    async def is_main_busy(self) -> bool:
        """True iff the main brain is demonstrably mid UNATTENDED long turn —
        the pre-ask gate the chat handler uses so a user does not pay ask()'s
        busy-wait just to be told the session is busy.

        WHAT IT MEANS CHANGED WITH 's STEP 5, the evidence did not.
        This used to be the "answer from the duty session" gate; now it is
        the "park it in the per-chat queue" gate. Both readings rest on the
        same two pieces of evidence below, but the new one is the weaker
        claim — "the main session is working" rather than "the main session
        cannot answer this" — and the weaker claim is the one the evidence
        actually supports. The blind-review worry that made requirement 1
        non-negotiable (a wedged pane looks busy to every probe we have, and
        routing around ``ask()`` leaves ``ask_health`` unfed, so
        ``delivery_guard`` can never restart anything) is narrower now than
        it was: a parked message is retried by ``chat_queue``'s worker, and
        that retry deliberately does NOT consult this gate — it goes into
        ``ask()``, pays the busy-wait and writes the ledger row. So the
        ledger is at most one drain pass behind, never blind.

        TWO INDEPENDENT PIECES OF EVIDENCE are required, and the first of
        them is a hard correctness constraint, not an optimization:

        1. A FRESH watchdog long-run marker (``long_run.is_active`` with
           ``DEFAULT_LONG_RUN_STALE_AFTER``). Blind-review finding F1: the
           pane probe ALONE is not evidence of a healthy busy session. A
           genuinely WEDGED pane holds a static footer signature forever
           (failure class B3 — see ask_health.py's module docstring), which
           reads as "busy" to every probe we have. Routing around ``ask()``
           on that reading meant ``ask_health.record()`` was never called,
           so ``fail_streak`` could not grow, so ``delivery_guard`` could
           never restart the unit or escalate — the owner would sit chatting
           with the stand-in indefinitely while "delivery health" stayed
           green. The marker is written by a DIFFERENT process that is
           independently alive, and it goes stale on its own; requiring it
           means the detour can only fire when something outside this
           process agrees a real run is in progress. When it is absent we
           return False and the message takes the ordinary path into
           ``ask()``, which busy-waits, records ``busy``/``busy_active`` and
           falls back to the duty session anyway. Slower in that rare case;
           the backstop stays armed.
        2. Two pane probes, ~3s apart: a turn that is about to finish must
           not cost the user a context-less duty reply. Both run in worker
           threads (the engine probe shells out), so the event loop stays
           free. Between them the pane must also visibly CHANGE (chrome or
           pane.log growth, when the driver can say so): a frozen frame
           with a busy signature on it is not a live turn.

        CODEX ENGINE: the watchdog's ``_track_long_run`` classifies a turn as
        active by running ``tmux_parse.is_main_turn_active`` over
        ``session.capture_text()``, and ``CodexExecDriver.capture_text``
        deliberately returns only this driver's own journal vocabulary —
        which cannot match those pane signatures by construction. So under
        ``chat_engine=codex`` no long-run marker is ever written, this gate
        is permanently False, and every message goes through ``ask()`` with
        its busy-wait and its duty fallback, exactly as before this gate
        existed. That is the correct degradation (no fast path, no blinded
        ledger) and is pinned by a test.

        THIS PATH NEVER TOUCHES ``ask_health`` (review round 3, R2). An
        earlier version wrote a neutral ``long_run`` row here so that
        ``last_status`` would not keep describing a turn from another hour.
        That was wrong. A neutral status leaves ``fail_streak`` alone but
        still moves ``Health.last_ts``, and ``delivery_guard.decide()``
        reads ``last_ts`` in two branches that rest on the invariant "the
        ledger only moves when a turn ends": the stale-evidence check, and —
        worse — ``health.last_ts <= self._last_restart_ts`` ("no failed turn
        since the last restart"). With a genuine ``fail_streak`` of 3 and
        one restart already spent, a single message sent during a long run
        would have read as fresh evidence and bought a second restart plus
        an escalation without the channel having been tried even once. The
        live state of a long run is already on disk in ``long-run.json``,
        which is exactly what ``/work`` reads; a log line is all this path
        owes anyone.
        """
        runtime_dir = self._runtime_dir()
        if runtime_dir is None:
            # No Settings and no injected dir ⇒ no marker to trust ⇒ take
            # the ordinary path. Never guess "busy" from nothing.
            return False
        try:
            active, _elapsed = long_run.is_active(
                runtime_dir,
                now=time.time(),
                stale_after=DEFAULT_LONG_RUN_STALE_AFTER,
            )
        except Exception:  # noqa: BLE001 — a marker read must never gate a reply
            logger.warning("long-run marker read failed", exc_info=True)
            return False
        if not active:
            return False
        before = await asyncio.to_thread(self._activity_fingerprint)
        if not await asyncio.to_thread(self._pane_active):
            return False
        await asyncio.sleep(_MAIN_BUSY_CONFIRM_SECONDS)
        if not await asyncio.to_thread(self._pane_active):
            return False
        after = await asyncio.to_thread(self._activity_fingerprint)
        if before is not None and before == after:
            # A live turn moves SOMETHING in a few seconds (spinner counter,
            # agent rows, the pane.log stream). A frozen frame that merely
            # carries a busy signature is not grounds for promising
            # "отвечу следом" (2026-09-25: a stale wait line parked every
            # message for ~9 hours). Take the ordinary path into ask(),
            # whose own busy-wait judges the pane and feeds the ledger.
            logger.info(
                "main session looks busy but nothing on the pane changed in "
                "%.0fs — not parking, taking the ordinary path",
                _MAIN_BUSY_CONFIRM_SECONDS,
            )
            return False
        logger.info(
            "main session is mid unattended long run (~%.0fs) — parking this "
            "message in the chat queue",
            _elapsed,
        )
        return True

    def _resolve_duty(self) -> Any | None:
        if self._duty is _UNRESOLVED:
            self._duty = None
            if self._auto_duty:
                try:
                    settings = self._config()
                    if settings is not None:
                        self._duty = get_duty_session(settings)
                except Exception:  # noqa: BLE001 — no duty ⇒ legacy message
                    logger.warning("duty session unavailable", exc_info=True)
                    self._duty = None
        return self._duty

    def _duty_stamp(self, settings: Settings) -> Path:
        return Path(settings.duty_dir) / _DUTY_STAMP_NAME

    def _duty_is_stale(self, settings: Settings, now: float) -> bool:
        """True ⇒ send /clear before this turn. An unreadable or missing
        stamp counts as stale: the safe direction is a fresh context, not a
        month-old one."""
        try:
            last = float(self._duty_stamp(settings).read_text().strip())
        except Exception:  # noqa: BLE001 — missing/corrupt ⇒ "long ago"
            return True
        return (now - last) > float(settings.duty_idle_reset_seconds)

    def _touch_duty(self, settings: Settings, now: float) -> None:
        stamp = self._duty_stamp(settings)
        tmp = stamp.with_name(stamp.name + ".tmp")
        try:
            stamp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(f"{now:.3f}\n")
            os.replace(tmp, stamp)  # atomic within the same dir
        except Exception:  # noqa: BLE001 — a lost stamp only costs a /clear
            logger.warning("could not write the duty stamp", exc_info=True)

    async def answer_from_duty(
        self,
        user_id: int,
        prompt: str,
        *,
        busy_seconds: float | None = None,
        fallback: str | None = None,
    ) -> str:
        """Answer a message from the duty session while the main one is busy.

        The text only — the shape every caller outside this module wants.
        ``send_message`` uses :meth:`answer_from_duty_detailed` instead,
        because it also has to score the turn in the health ledger and the
        text alone cannot say whether a real answer came back.
        """
        text, _answered = await self.answer_from_duty_detailed(
            user_id, prompt, busy_seconds=busy_seconds, fallback=fallback
        )
        return text

    async def answer_from_duty_detailed(
        self,
        user_id: int,
        prompt: str,
        *,
        busy_seconds: float | None = None,
        fallback: str | None = None,
    ) -> tuple[str, bool]:
        """``(text, answered)`` — the reply, and whether the DUTY SESSION is
        what produced it.

        ``answered`` is the fact this whole distinction turns on: a
        ``busy`` turn that the duty session covered is a turn the person got
        a real answer to, and must not count as a delivery failure. Anything
        else — the feature off, no session, a crash, an empty or failed duty
        turn — returns ``False``, i.e. the user got only the brush-off, and
        the ledger keeps scoring it as the failure it is.

        ``fallback`` is the message the user would have received before this
        feature existed; it is returned verbatim whenever the duty path is
        off, unavailable or broken — the rollback and the crash path land on
        exactly today's behavior. Defaults to the busy brush-off.
        """
        if fallback is None:
            fallback = _busy_active_message(busy_seconds)
        legacy = fallback
        try:
            settings = self._config()
            if settings is None or not settings.duty_session_enabled:
                return legacy, False
            duty = self._resolve_duty()
            if duty is None:
                return legacy, False
            async with self._duty_lock:
                now = time.time()
                if self._duty_is_stale(settings, now):
                    logger.info("duty session idle — clearing before the turn")
                    await asyncio.to_thread(duty.send_control, "/clear")
                self._touch_duty(settings, now)
                res = await asyncio.to_thread(
                    duty.ask,
                    wrap_duty_prompt(prompt),
                    timeout=settings.duty_turn_timeout,
                    request_id=f"duty-{user_id}-{int(now)}",
                )
                self._touch_duty(settings, time.time())
            text, answered = self._duty_reply_text(duty, res, busy_seconds, legacy)
            if answered:
                self._note_handoff(prompt)
            return text, answered
        except Exception:  # noqa: BLE001 — the duty path must never cost a reply
            logger.exception("duty session failed for user %d", user_id)
            return legacy, False

    def _duty_reply_text(
        self, duty: Any, res: Any, busy_seconds: float | None, legacy: str
    ) -> tuple[str, bool]:
        reply = (res.reply or "").strip() if res.ok else ""
        if not reply:
            # Never hand the user silence: say what the main session is doing
            # AND why the stand-in could not cover for it. An `ok` with an
            # empty body counts as a failure here — unlike the main path,
            # nothing retries a duty turn.
            detail = (
                _STATUS_MESSAGES["error"]
                if res.ok
                else _STATUS_MESSAGES.get(res.status, _STATUS_MESSAGES["error"])
            )
            logger.warning(
                "duty session returned %s", "empty" if res.ok else res.status
            )
            return (
                f"{legacy}\n\n🔁 <i>Дежурная сессия тоже не ответила:</i> {detail}",
                False,
            )
        body = f"{duty_header(busy_seconds)}\n\n{reply}"
        notices = self._pop_notices(duty)
        if notices:
            body = "\n\n".join(html.escape(n) for n in notices) + "\n\n" + body
        return body, True

    def _is_turn_limit(self, res: Any) -> bool:
        """True iff this result is the chat-turn CEILING
        rather than any other timeout — see ``_TURN_LIMIT_DETAIL_MARKER``.
        Both engines emit the same ``timeout`` / ``"no reply in Ns"`` pair
        for their hard deadline, which is why the engine question below is
        answered separately."""
        return (
            res.status == "timeout"
            and bool(res.detail)
            and _TURN_LIMIT_DETAIL_MARKER in res.detail
        )

    def _turn_limit_keeps_inflight(self) -> bool:
        """True iff, on THIS engine, hitting the chat-turn ceiling leaves the
        request running with a late reply still possible.

        Claude (tmux): yes — ``ask()``'s tail return deliberately does not
        mark the rid handled and leaves the ``inflight`` marker in place
        ("prompt is still physically in the pane → keep inflight"), so the
        watchdog's orphan path delivers a late reply.

        Codex: no — ``CodexExecDriver`` terminates the turn's process on its
        hard deadline. Decided here rather than in ``codex_driver.py`` on
        purpose: the driver's own contract is unchanged, it is the CHAT
        path's interpretation of that contract that differs.
        """
        return self._engine() != "codex"

    def _health_status_for(self, res: Any) -> str:
        """The ledger row for this outcome — deliberately NOT always
        ``res.status``. Split out of ``_reply_text`` (blind review F2) so
        scoring and wording are two decisions instead of one tangled if.
        """
        # Score what the USER got, not what ask() returned. An `ok` with an
        # empty body is a delivered-nothing turn: chat.py retries it once and,
        # if the retry is empty too, answers "Claude не ответил дважды".
        # Recording that as a success would clear the fail streak on exactly
        # the shape of failure this ledger exists to catch — a health signal
        # reading green while the user receives silence (2026-08-20).
        if res.ok and not (res.reply or "").strip():
            return "error"
        # F2: a legitimately long turn whose answer still reaches the user
        # through the orphan path is not a delivery failure — but ONLY on an
        # engine that keeps it in flight. Under Codex the same result really
        # is a lost reply, so it stays `timeout` and keeps counting.
        if self._is_turn_limit(res) and self._turn_limit_keeps_inflight():
            return TURN_LIMIT_STATUS
        return res.status

    def _reply_text(self, user_id: int, res: Any) -> str:
        self._record_health(self._health_status_for(res))
        if res.ok:
            reply = res.reply or ""
            if res.salvaged:
                # (2026-08-21): delivered from an
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
        if self._is_turn_limit(res):
            # Engine-dependent truth (F4). Claude: the turn is still running
            # and still in flight — "попробуй ещё раз" would queue a second
            # turn behind it. Codex: the turn's process is already dead, so
            # a retry is the ONLY thing that can produce an answer.
            if self._turn_limit_keeps_inflight():
                return _TURN_LIMIT_MESSAGE
            return _TURN_LIMIT_MESSAGE_KILLED
        if res.status == "busy_active":
            return _busy_active_message(res.busy_seconds)
        return _STATUS_MESSAGES.get(res.status, _STATUS_MESSAGES["error"])

    def progress_transcript(self) -> Path | None:
        """Файл стенограммы, по которому карточка прогресса (
        шаг 4) читает события хода, или ``None``, если карточку показывать не
        по чему.

        ``None`` возвращается в трёх случаях, и каждый — штатный:

        * **движок Codex.** Его ``current_transcript_path()`` отдаёт
          rollout-JSONL с совершенно другой схемой; парсить его парсером
          Claude Code нельзя, а заводить второй разбор ради вспомогательной
          карточки — не та цена. Под Codex карточки просто нет.
        * **сессия ещё ни разу не стартовала** — закреплённого id нет,
          читать нечего.
        * **файла ещё не существует.** Важно не проглядеть: ``TranscriptTail``
          при недоступном файле встаёт на смещение 0, то есть карточка на
          первом же опросе проиграла бы ВСЮ историю файла как «события этого
          хода». Проверка существования здесь — единственное место, где это
          ловится.

        Никогда не бросает: карточка не имеет права стоить ответа.
        """
        if self._engine() == "codex":
            return None
        try:
            path = self._session.current_transcript_path()
        except Exception:  # noqa: BLE001 — нет карточки лучше, чем нет ответа
            logger.debug("progress: could not resolve the transcript", exc_info=True)
            return None
        if path is None:
            return None
        try:
            return path if Path(path).exists() else None
        except OSError:
            return None

    async def send_control(self, text: str) -> None:
        """Fire-and-forget a client-side Claude Code command into the session."""
        async with get_ask_lock():
            await asyncio.to_thread(self._session.send_control, text)

    # ── steering: deliberately NOT under the ask-lock — the lock is held by
    # the in-flight turn these calls are aimed at.

    def is_turn_active(self) -> bool:
        return self._session.is_turn_active()

    def is_pane_turn_active(self) -> bool:
        """Step D: true iff the PANE shows the
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
        """Status + optional body for ``/resend``.

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

    # ── duty → main handoff ──────────────────────────────────────────

    def _handoff_path(self) -> Path | None:
        runtime_dir = self._runtime_dir()
        return None if runtime_dir is None else Path(runtime_dir) / _HANDOFF_FILE

    def _pending_handoff(self) -> list[dict]:
        """Messages the duty session answered that the main session has not
        seen yet, oldest first. Never raises; unreadable ⇒ none."""
        path = self._handoff_path()
        if path is None:
            return []
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        return [
            item
            for item in raw
            if isinstance(item, dict)
            and isinstance(item.get("ts"), (int, float))
            and isinstance(item.get("text"), str)
        ]

    def _write_handoff(self, items: list[dict]) -> None:
        path = self._handoff_path()
        if path is None:
            return
        try:
            if not items:
                path.unlink(missing_ok=True)
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(items, ensure_ascii=False))
            os.replace(tmp, path)
        except OSError:
            logger.warning("could not write the duty handoff", exc_info=True)

    def _note_handoff(self, prompt: str) -> None:
        items = self._pending_handoff()
        items.append({"ts": time.time(), "text": prompt[:_HANDOFF_TEXT_CAP]})
        self._write_handoff(items[-_HANDOFF_MAX:])

    def _clear_handoff(self, up_to_ts: float) -> None:
        """Drop what the main session was shown; keep anything the duty
        session answered while that main turn was running."""
        self._write_handoff(
            [item for item in self._pending_handoff() if item["ts"] > up_to_ts]
        )

    def pending_handoff_count(self) -> int:
        return len(self._pending_handoff())

    # ── /reset: the circuit breaker ──────────────────────────────────

    @staticmethod
    def reset_in_progress() -> bool:
        return _reset_running

    async def circuit_reset(self, user_id: int) -> list[ResetOutcome]:
        """Owner's /reset: stop the current turn and recreate the main and
        duty sessions from scratch, then prove they are up and idle.

        Built from the pieces that already exist, in this order:

        1. every session is told to stop — ``request_abort`` (the Claude
           driver: an ask() already running gives up at its next poll and
           releases the pane lock) plus ``interrupt`` (Escape / SIGINT);
        2. the process-wide ask-lock is taken (bounded wait), so the chat
           queue cannot start its next turn between the restart and the
           read-back and make a clean session look busy;
        3. ``force_recover`` — the /relogin and watchdog primitive: kill and
           recreate under the pane lock — retried until the stopped turn has
           let go (``_RESET_RELEASE_WAIT``);
        4. the watchdog's long-run marker is removed, so nothing keeps
           reporting the old run as live;
        5. read-back: healthy, no turn holding the lock, nothing active on
           the pane. Only that is reported as ``ok``.

        Engine-neutral: under Codex ``interrupt`` SIGINTs the exec process
        and ``force_recover`` kills a stray one; the thread is then dropped
        too, so both engines come back with a clean context (new process,
        new conversation — the vault is untouched). ``request_abort`` does
        not exist there and is skipped.
        """
        global _last_reset_ts, _reset_running  # noqa: PLW0603
        if _reset_running:
            raise RuntimeError("перезапуск уже идёт")
        _reset_running = True
        try:
            return await self._circuit_reset(user_id)
        finally:
            _reset_running = False

    async def _circuit_reset(self, user_id: int) -> list[ResetOutcome]:
        global _last_reset_ts  # noqa: PLW0603
        _last_reset_ts = time.time()
        logger.warning("/reset requested by user %d", user_id)
        targets: list[tuple[str, Any]] = [("main", self._session)]
        settings = self._config()
        if settings is not None and getattr(settings, "duty_session_enabled", False):
            duty = self._resolve_duty()
            if duty is not None:
                targets.append(("duty", duty))

        for _name, session in targets:
            await asyncio.to_thread(self._stop_turn, session)

        lock = get_ask_lock()
        try:
            await asyncio.wait_for(lock.acquire(), timeout=_RESET_RELEASE_WAIT)
            held = True
        except TimeoutError:
            logger.warning("/reset: ask-lock still held — restarting without it")
            held = False
        try:
            outcomes = []
            for name, session in targets:
                restart = functools.partial(
                    self._restart_and_verify,
                    name,
                    session,
                    drop_thread=self._engine() == "codex",
                )
                if name != "duty":
                    outcomes.append(await asyncio.to_thread(restart))
                    continue
                # The duty path serializes on this lock, not on the ask-lock:
                # hold it too, so a duty turn cannot start between the
                # restart and the read-back.
                try:
                    await asyncio.wait_for(
                        self._duty_lock.acquire(), timeout=_RESET_RELEASE_WAIT
                    )
                    duty_held = True
                except TimeoutError:
                    duty_held = False
                try:
                    outcomes.append(await asyncio.to_thread(restart))
                finally:
                    if duty_held:
                        self._duty_lock.release()
            runtime_dir = self._runtime_dir()
            if runtime_dir is not None:
                long_run.clear(runtime_dir)
        finally:
            if held:
                lock.release()
        logger.warning(
            "/reset done: %s",
            ", ".join(f"{o.name}={'ok' if o.ok else o.detail}" for o in outcomes),
        )
        return outcomes

    @staticmethod
    def _stop_turn(session: Any) -> None:
        """Best effort: signal a running turn to stop. Never raises."""
        for method in ("request_abort", "interrupt"):
            fn = getattr(session, method, None)
            if fn is None:
                continue
            try:
                fn()
            except Exception:  # noqa: BLE001 — a failed stop must not stop /reset
                logger.warning("/reset: %s failed", method, exc_info=True)

    @staticmethod
    def _restart_and_verify(
        name: str, session: Any, *, drop_thread: bool = False
    ) -> ResetOutcome:
        """Blocking: recreate one session, then read it back.

        Claude: ``force_recover`` kills the tmux session and starts a NEW
        ``claude`` process with a new session id — a clean context by
        construction. Codex: ``force_recover`` kills the exec process but
        deliberately keeps the thread, so ``drop_thread`` also sends the
        engine's ``/clear`` (forget ``thread_id``) — the next turn starts a
        fresh thread in a fresh process."""
        deadline = time.monotonic() + _RESET_RELEASE_WAIT
        recovered = False
        error = ""
        while True:
            try:
                recovered = bool(session.force_recover())
            except Exception as exc:  # noqa: BLE001 — reported, not raised
                logger.warning("/reset: %s force_recover failed", name, exc_info=True)
                error = str(exc) or exc.__class__.__name__
            if recovered or time.monotonic() >= deadline:
                break
            # Whoever holds the session now may be a turn that was only
            # WAITING when /reset began (the pipeline, a queued ask) and so
            # was not covered by the first stop — stop it too (review).
            ChatSessionManager._stop_turn(session)
            time.sleep(_RESET_POLL_SECONDS)
        if not recovered:
            return ResetOutcome(
                name, False, error or "сессию так и не отпустил предыдущий ход"
            )
        if drop_thread:
            try:
                session.send_control("/clear")
            except Exception as exc:  # noqa: BLE001 — reported, not raised
                logger.warning("/reset: %s thread drop failed", name, exc_info=True)
                return ResetOutcome(name, False, f"контекст не сброшен: {exc}")
        deadline = time.monotonic() + _RESET_READBACK_WAIT
        while True:
            try:
                if (
                    session.is_healthy()
                    and not session.is_turn_active()
                    and not session.is_pane_turn_active()
                ):
                    return ResetOutcome(name, True)
                detail = "после перезапуска сессия не выглядит свободной"
            except Exception as exc:  # noqa: BLE001
                detail = f"не удалось проверить сессию: {exc}"
            if time.monotonic() >= deadline:
                return ResetOutcome(name, False, detail)
            time.sleep(_RESET_POLL_SECONDS)

    async def compact(self, user_id: int) -> str:
        """Durable-state-first: clearing is the compaction; memory lives in
        files, so there is nothing to summarize into the session."""
        # M1 fix (2026-08-22): clear() now sends a blocking /clear + up to a
        # real ~10s resync poll (see B1) instead of being near-instant —
        # offload it so this coroutine doesn't block the event loop.
        await asyncio.to_thread(self._session.clear)
        return "🧹 Сессия очищена (важные данные сохранены в файлах)."
