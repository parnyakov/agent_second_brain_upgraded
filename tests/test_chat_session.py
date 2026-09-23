"""Tests for ChatSessionManager — chat messages routed to the shared session."""

import asyncio

from d_brain.services.claude_session import AskResult


class FakeSession:
    def __init__(
        self,
        result: AskResult,
        resend_result: tuple[str, str | None] = ("empty", None),
        force_recover_result: bool = True,
    ) -> None:
        self.result = result
        self.prompts: list[str] = []
        self.cleared = 0
        self.resend_result = resend_result
        self.resend_calls = 0
        self.force_recover_result = force_recover_result
        self.force_recover_calls = 0

        self.ask_kwargs: list[dict] = []
        self.controls: list[str] = []
        self.pane_active_answers: list[bool] = []

    def ask(self, prompt: str, **kwargs) -> AskResult:
        self.prompts.append(prompt)
        self.ask_kwargs.append(kwargs)
        return self.result

    def send_control(self, text: str) -> None:
        self.controls.append(text)

    def is_pane_turn_active(self) -> bool:
        # Pops the scripted answers in order; empty ⇒ idle (today's default).
        return self.pane_active_answers.pop(0) if self.pane_active_answers else False

    def clear(self) -> None:
        self.cleared += 1

    def last_reply_for_resend(self) -> tuple[str, str | None]:
        self.resend_calls += 1
        return self.resend_result

    def force_recover(self) -> bool:
        self.force_recover_calls += 1
        return self.force_recover_result


def _settings(tmp_path, **over):
    """An isolated Settings for a manager under test.

    ``_env_file=None`` and an EXPLICIT ``chat_engine`` are both load-bearing.
    ``tests/conftest.py`` already unbinds the repo-root ``.env`` suite-wide,
    but a contract that differs per engine (the turn-limit wording and its
    health row — blind review F4) must not ride on the *default* engine
    either: whichever branch a test means, it says so.
    """
    from d_brain.config import Settings

    base = dict(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path / "vault",
        runtime_dir=tmp_path / "rt",
        chat_engine="claude",
        _env_file=None,
    )
    base.update(over)
    return Settings(**base)


def _manager(tmp_path, result: AskResult, **over):
    from d_brain.services.chat_session import ChatSessionManager

    # health_dir is pinned to tmp_path on purpose. Left unset it resolves
    # lazily from Settings, which SUCCEEDS on the production checkout (there is
    # a real .env there) — and the suite would then write to the live
    # ~/.dbrain/ask-health.json that the running watchdog reads, quietly
    # clearing a genuine fail streak.
    m = ChatSessionManager(
        tmp_path, session=FakeSession(result), health_dir=tmp_path
    )
    # Settings are injected rather than resolved: an injected main session
    # means _auto_duty is False, so no real duty session is ever built, and
    # every engine-dependent branch is decided by this test's own value.
    m._settings = _settings(tmp_path, **over)
    return m


def test_send_message_returns_reply_on_ok(tmp_path):
    m = _manager(tmp_path, AskResult("ok", reply="привет"))
    reply = asyncio.run(m.send_message(1, "здравствуй"))
    assert reply == "привет"
    assert m._session.prompts == ["здравствуй"]


def test_send_message_prepends_parking_notice_once(tmp_path):
    """agent-infra: a turn that had to park the pane tells the
    owner, on that very reply, that the conversation context is gone."""
    m = _manager(tmp_path, AskResult("ok", reply="привет"))
    pending = ["⚠️ Контекст разговора сброшен: «<night>» отложена"]
    m._session.pop_notices = lambda: [pending.pop()] if pending else []
    first = asyncio.run(m.send_message(1, "x"))
    assert first.startswith("⚠️ Контекст разговора сброшен: «&lt;night&gt;»")
    assert first.endswith("\n\nпривет")
    assert asyncio.run(m.send_message(1, "x")) == "привет"


def test_send_message_leaves_notice_queued_on_empty_reply(tmp_path):
    """An empty reply is chat.py's retry signal — never make it non-empty;
    the watchdog delivers the notice instead."""
    m = _manager(tmp_path, AskResult("ok", reply=""))
    popped = []
    m._session.pop_notices = lambda: popped.append(1) or ["n"]
    assert asyncio.run(m.send_message(1, "x")) == ""
    assert popped == []


def test_send_message_maps_rate_limited(tmp_path):
    m = _manager(tmp_path, AskResult("rate_limited"))
    reply = asyncio.run(m.send_message(1, "x"))
    assert "Лимит" in reply


def test_send_message_maps_busy_to_an_honest_non_error_message(tmp_path):
    """Busy-panel UX finding: a legitimately busy panel must NOT read as
    '❌ Ошибка сессии' — that misled an operator into thinking something
    had crashed when nothing was actually wrong."""
    m = _manager(
        tmp_path, AskResult("busy", detail="pane still busy with a previous turn")
    )
    reply = asyncio.run(m.send_message(1, "x"))
    assert "Ошибка" not in reply
    assert "ещё выполняется" in reply or "busy" in reply.lower()


def test_send_message_returns_busy_for_a_live_progressing_turn(tmp_path):
    """``busy_active`` means the pane is busy with work that
    demonstrably kept progressing — healthy, not wedged. It is no longer an
    outcome the user hears about at all: the manager hands back ``Busy`` and
    the caller parks the message in the per-chat queue, so the MAIN session
    (the one with the conversation) answers it when it is free."""
    from d_brain.services.chat_session import Busy

    m = _manager(
        tmp_path,
        AskResult(
            "busy_active",
            detail="pane busy with a live, progressing turn",
            busy_seconds=30.0,
        ),
    )
    res = asyncio.run(m.send_message(1, "x"))
    assert isinstance(res, Busy)
    assert res.busy_seconds == 30.0


def test_busy_carries_the_pre_queue_wording_as_its_fallback(tmp_path):
    """The rollback path: a caller with no queue configured hands
    ``fallback`` straight to the duty session and lands byte-for-byte on the
    pre-queue behavior, elapsed-minutes parenthetical included."""
    m = _manager(
        tmp_path,
        AskResult(
            "busy_active",
            detail="pane busy with a live, progressing turn",
            busy_seconds=185.0,  # ~3 min
        ),
    )
    res = asyncio.run(m.send_message(1, "x"))
    assert "3 мин" in res.fallback
    assert "Ошибка" not in res.fallback

    bare = _manager(
        tmp_path,
        AskResult("busy_active", detail="pane busy with a live, progressing turn"),
    )
    assert "мин)" not in asyncio.run(bare.send_message(1, "x")).fallback


def test_busy_active_is_still_scored_in_the_health_ledger(tmp_path):
    """Parking a message must not hide the main channel's busy turn from
    ``ask_health`` — that ledger is what arms ``delivery_guard``."""
    from d_brain.services import ask_health

    m = _manager(tmp_path, AskResult("busy_active", busy_seconds=5.0))
    asyncio.run(m.send_message(1, "x"))
    assert ask_health.read(tmp_path).last_status == "busy_active"


def test_send_message_maps_no_markers_ceiling_to_an_honest_message(tmp_path):
    """R2c (Fable audit, F2 class): the ceiling's distinct detail for 'no
    reply markers ever appeared at all' must map to an honest message, not
    the generic '⌛ Превышено время ожидания ответа'."""
    m = _manager(
        tmp_path,
        AskResult(
            "timeout",
            detail="no reply markers ever appeared — the reply may be ready "
            "in the terminal but its delivery markers were lost",
        ),
    )
    reply = asyncio.run(m.send_message(1, "x"))
    assert "Превышено время ожидания" not in reply
    assert "маркер" in reply.lower()


def test_send_message_keeps_generic_timeout_message_for_ordinary_timeouts(tmp_path):
    """The honest F2 message must be scoped to its specific detail string —
    a timeout with any OTHER detail keeps the existing generic wording.

    The detail is the REAL generic-timeout string Claude's ask() produces
    ("no closing marker and no active main turn"); the previous
    "session stalled" was not a timeout detail at all in production — that
    is the detail of the `error` status — so the test was passing on a
    result shape that cannot occur (blind review F11).
    """
    m = _manager(
        tmp_path,
        AskResult("timeout", detail="no closing marker and no active main turn"),
    )
    reply = asyncio.run(m.send_message(1, "x"))
    assert "Превышено время ожидания" in reply


def test_send_message_turn_limit_timeout_says_the_turn_is_still_running(tmp_path):
    """the chat path now caps a main turn at
    chat_turn_timeout, and ask()'s tail return ("no reply in Ns") leaves the
    request IN FLIGHT (rid not marked handled) — the watchdog's orphan path
    still delivers a late reply. "Попробуй ещё раз" would be a lie that
    queues a second turn behind the first.

    The engine is named explicitly: this contract is the CLAUDE half of the
    split (see the codex test below), and it must not depend on which engine
    happens to be the default — or on an ambient .env, which is how this
    assertion once passed in a worktree and failed in the live checkout."""
    m = _manager(
        tmp_path,
        AskResult("timeout", detail="no reply in 1500s"),
        chat_engine="claude",
    )
    reply = asyncio.run(m.send_message(1, "x"))
    assert "Превышено время ожидания" not in reply
    assert "маркер" not in reply.lower()  # not the F2 message either
    assert "отдельным сообщением" in reply


def test_turn_limit_message_names_both_outcomes(tmp_path):
    """Blind review F5: the user was promised a late reply and then, ~30
    minutes later, got the watchdog's "закрыл автоматически" about the same
    turn — two messages contradicting each other. The wording must name both
    branches up front."""
    m = _manager(
        tmp_path,
        AskResult("timeout", detail="no reply in 1500s"),
        chat_engine="claude",
    )
    reply = asyncio.run(m.send_message(1, "x"))
    assert "отдельным сообщением" in reply  # it may still finish
    assert "вотчдог" in reply.lower()  # ...or it may be closed


def test_turn_limit_does_not_grow_the_fail_streak(tmp_path):
    """Blind review F2: `timeout` is in ask_health.FAILURE_STATUSES, so
    three legitimately long chat turns inside delivery_guard's window used
    to restart dbrain-bot.service — while the user's answer was on its way
    through the orphan path. A neutral row records reality without arming a
    restart."""
    from d_brain.services import ask_health
    from d_brain.services.chat_session import TURN_LIMIT_STATUS

    m = _manager(
        tmp_path,
        AskResult("timeout", detail="no reply in 1500s"),
        chat_engine="claude",
    )
    for _ in range(3):
        asyncio.run(m.send_message(1, "x"))
    health = ask_health.read(tmp_path)
    assert health.fail_streak == 0
    assert health.last_status == TURN_LIMIT_STATUS


def test_turn_limit_neither_clears_an_existing_streak(tmp_path):
    """Neutral means neutral in BOTH directions: a long turn must not mask a
    genuine failure streak the DeliveryGuard is already counting."""
    from d_brain.services import ask_health

    ask_health.record(tmp_path, "error", clock_fn=lambda: 1.0)
    ask_health.record(tmp_path, "error", clock_fn=lambda: 2.0)
    m = _manager(
        tmp_path,
        AskResult("timeout", detail="no reply in 1500s"),
        chat_engine="claude",
    )
    asyncio.run(m.send_message(1, "x"))
    assert ask_health.read(tmp_path).fail_streak == 2


def test_the_turn_limit_split_is_decided_by_the_engine_alone(tmp_path):
    """One and the same AskResult, two engines, two contracts.

    This is the test the 2026-09-20 green/red split should have had: the
    four turn-limit assertions above were silently reading the engine from
    the repo-root .env (DBRAIN_CHAT_ENGINE=codex, left over from the
    September experiment), so they passed in a clean worktree and failed in
    the live checkout. Both halves are now pinned side by side, each naming
    its own engine.
    """
    from d_brain.services import ask_health

    ceiling = AskResult("timeout", detail="no reply in 1500s")

    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    m = _manager(claude_dir, ceiling, chat_engine="claude")
    claude_reply = asyncio.run(m.send_message(1, "x"))

    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    m = _manager(codex_dir, ceiling, chat_engine="codex")
    codex_reply = asyncio.run(m.send_message(1, "x"))

    # Claude keeps the request in flight ⇒ a late reply is still promised,
    # and the row stays neutral. Codex kills the turn ⇒ no promise, and a
    # genuine delivery failure the DeliveryGuard must count.
    assert "отдельным сообщением" in claude_reply
    assert "отдельным сообщением" not in codex_reply
    assert ask_health.read(claude_dir).fail_streak == 0
    assert ask_health.read(codex_dir).fail_streak == 1


def test_the_turn_limit_status_is_really_neutral():
    """Guard against a future edit to FAILURE_STATUSES silently turning the
    neutral row into a restart trigger."""
    from d_brain.services import ask_health
    from d_brain.services.chat_session import TURN_LIMIT_STATUS

    assert TURN_LIMIT_STATUS not in ask_health.FAILURE_STATUSES
    assert TURN_LIMIT_STATUS != ask_health.SUCCESS_STATUS


def test_turn_limit_under_codex_is_a_real_failure_and_says_so(tmp_path):
    """Blind review F4: CodexExecDriver TERMINATES the turn's process on its
    hard deadline, returning the very same timeout/"no reply in Ns" pair.
    Nothing is left in flight, so the Claude wording would be a lie and the
    outcome is a genuine lost reply — it must keep counting as `timeout`."""
    from d_brain.services import ask_health

    m, _duty = _duty_manager(
        tmp_path,
        AskResult("timeout", detail="no reply in 1500s"),
        AskResult("ok", reply="-"),
        chat_engine="codex",
    )
    reply = asyncio.run(m.send_message(1, "x"))
    assert "отдельным сообщением" not in reply
    assert "Повтори" in reply
    assert ask_health.read(tmp_path).fail_streak == 1
    assert ask_health.read(tmp_path).last_status == "timeout"


def test_an_ok_with_no_reply_body_is_recorded_as_a_failure(tmp_path):
    """The 2026-08-20 shape: the health signal must not read green while the
    user receives silence. chat.py retries an empty reply once and then tells
    the user "Claude не ответил дважды" — that is a delivered-nothing turn."""
    from d_brain.services import ask_health

    m = _manager(tmp_path, AskResult("ok", reply="   "))
    assert asyncio.run(m.send_message(1, "x")) == "   "
    assert ask_health.read(tmp_path).fail_streak == 1


def test_a_real_reply_clears_the_streak(tmp_path):
    from d_brain.services import ask_health

    ask_health.record(tmp_path, "timeout", clock_fn=lambda: 1.0)
    m = _manager(tmp_path, AskResult("ok", reply="привет"))
    asyncio.run(m.send_message(1, "x"))
    assert ask_health.read(tmp_path).fail_streak == 0


def test_reset_clears_live_session(tmp_path):
    m = _manager(tmp_path, AskResult("ok", reply=""))
    m.reset(1)
    assert m._session.cleared == 1


# ── force_recover (/relogin — 2026-08-24 incident: logged-out session) ────


def test_force_recover_delegates_to_session_and_returns_true_on_success(tmp_path):
    m = _manager(tmp_path, AskResult("ok", reply=""))
    assert m.force_recover(1) is True
    assert m._session.force_recover_calls == 1


def test_force_recover_returns_false_when_session_reports_busy(tmp_path):
    from d_brain.services.chat_session import ChatSessionManager

    session = FakeSession(AskResult("ok", reply=""), force_recover_result=False)
    m = ChatSessionManager(tmp_path, session=session, health_dir=tmp_path)
    assert m.force_recover(1) is False
    assert session.force_recover_calls == 1


# ── resend_last_reply (/resend) ──────────────────────────


def _resend_manager(tmp_path, resend_result: tuple[str, str | None]):
    from d_brain.services.chat_session import ChatSessionManager

    session = FakeSession(AskResult("ok", reply=""), resend_result=resend_result)
    return ChatSessionManager(tmp_path, session=session, health_dir=tmp_path)


def test_resend_last_reply_delegates_to_session_and_returns_status(tmp_path):
    m = _resend_manager(tmp_path, ("ready", "the last answer"))
    status, body = asyncio.run(m.resend_last_reply(1))
    assert status == "ready"
    assert body == "the last answer"
    assert m._session.resend_calls == 1


def test_resend_last_reply_passes_through_each_status(tmp_path):
    for status in ("empty", "in_progress", "unavailable"):
        m = _resend_manager(tmp_path, (status, None))
        got_status, got_body = asyncio.run(m.resend_last_reply(1))
        assert got_status == status
        assert got_body is None


# ── duty session: answering while the main brain is busy
# ─────────────────────────────────────


def _duty_settings(tmp_path, **over):
    # One definition of "isolated Settings" for this module — see _settings.
    return _settings(tmp_path, **over)


def _duty_manager(tmp_path, main: AskResult, duty: AskResult, **over):
    """A manager whose MAIN session returns `main` and whose duty session is
    a fake returning `duty`.

    The duty session is INJECTED, never auto-resolved: a manager built with
    an injected main session must never reach runtime.get_duty_session() and
    start a real engine session against the live runtime dir.
    """
    from d_brain.services.chat_session import ChatSessionManager

    duty_fake = FakeSession(duty)
    m = ChatSessionManager(
        tmp_path,
        session=FakeSession(main),
        health_dir=tmp_path,
        duty_session=duty_fake,
    )
    m._settings = _duty_settings(tmp_path, **over)
    return m, duty_fake


def test_wrap_duty_prompt_states_the_contract_and_carries_the_message():
    from d_brain.services.chat_session import wrap_duty_prompt

    env = wrap_duty_prompt("запиши мысль про склад")
    assert env.startswith("[ДЕЖУРНАЯ СЕССИЯ]")
    # The rules the duty turn is judged by.
    assert "Telegram-HTML" in env and "Markdown" in env
    assert "daily" in env
    assert "субагентов" in env and "НЕ начинай длинную работу" in env
    assert "projects/*/status.md" in env and "MEMORY.md" in env
    # The user's text survives verbatim, last.
    assert env.rstrip().endswith("запиши мысль про склад")


def test_duty_header_mentions_minutes_only_when_known():
    from d_brain.services.chat_session import duty_header

    assert "мин" not in duty_header(None)
    assert "мин" not in duty_header(12.0)  # under a minute: no parenthetical
    assert "~3 мин" in duty_header(185.0)
    assert "дежурной" in duty_header(None)


def test_a_wedged_main_session_is_answered_by_the_duty_session(tmp_path):
    """The duty session's remaining job after and its only
    one: plain ``busy`` means the pane showed NO real progress across the
    whole busy-wait budget — failure class B3, the wedge signature. Queueing
    behind a wedge would be a promise nobody can keep, so the stand-in
    answers, labelled as the context-less stand-in it is."""
    m, duty = _duty_manager(
        tmp_path,
        AskResult("busy", busy_seconds=600.0),
        AskResult("ok", reply="<b>Записал</b> в daily."),
    )
    reply = asyncio.run(m.send_message(7, "запиши мысль"))
    assert "Записал" in reply
    assert reply.startswith("🔁")  # the honest stand-in banner
    assert "~10 мин" in reply
    # The duty turn got the envelope, its own timeout and a request id.
    assert duty.prompts[0].startswith("[ДЕЖУРНАЯ СЕССИЯ]")
    assert "запиши мысль" in duty.prompts[0]
    assert duty.ask_kwargs[0]["timeout"] == 600.0
    assert duty.ask_kwargs[0]["request_id"].startswith("duty-7-")


def test_a_busy_but_progressing_session_never_reaches_the_duty_session(tmp_path):
    """The product rule: the main session should answer almost every time,
    since it holds the context. Simple busyness must not spend a
    context-less stand-in turn any more."""
    from d_brain.services.chat_session import Busy

    m, duty = _duty_manager(
        tmp_path,
        AskResult("busy_active", busy_seconds=600.0),
        AskResult("ok", reply="не должно уехать"),
    )
    assert isinstance(asyncio.run(m.send_message(1, "x")), Busy)
    assert duty.prompts == []


def test_duty_disabled_restores_todays_exact_messages(tmp_path):
    """The rollback switch: DBRAIN_DUTY_SESSION=false ⇒ the two busy statuses
    map to their own pre-duty wordings and nothing reaches the duty
    session."""
    from d_brain.services.chat_session import Busy

    m, duty = _duty_manager(
        tmp_path,
        AskResult("busy_active", busy_seconds=185.0),
        AskResult("ok", reply="не должно уехать"),
        duty_session_enabled=False,
    )
    # busy_active never reaches the duty session at all now — it is parked.
    # Its fallback is still the exact pre-duty wording, which is what a
    # caller with no queue configured delivers.
    res = asyncio.run(m.send_message(1, "x"))
    assert isinstance(res, Busy)
    assert "3 мин" in res.fallback and "/stop" in res.fallback
    assert duty.prompts == []

    m2, duty2 = _duty_manager(
        tmp_path,
        AskResult("busy"),
        AskResult("ok", reply="не должно уехать"),
        duty_session_enabled=False,
    )
    assert "ещё выполняется" in asyncio.run(m2.send_message(1, "x"))
    assert duty2.prompts == []


def test_duty_session_is_cleared_when_it_has_been_idle(tmp_path):
    m, duty = _duty_manager(tmp_path, AskResult("busy"), AskResult("ok", reply="ок"))
    asyncio.run(m.send_message(1, "первое"))
    assert duty.controls == ["/clear"]  # no stamp yet ⇒ "long ago"
    # That turn wrote a fresh stamp, so the next message inside the same busy
    # window keeps the duty session's own short thread.
    asyncio.run(m.send_message(1, "второе"))
    assert duty.controls == ["/clear"]
    assert len(duty.prompts) == 2


def test_duty_session_is_cleared_again_after_the_idle_window(tmp_path):
    m, duty = _duty_manager(
        tmp_path,
        AskResult("busy"),
        AskResult("ok", reply="ок"),
        duty_idle_reset_seconds=0.0,
    )
    asyncio.run(m.send_message(1, "первое"))
    asyncio.run(m.send_message(1, "второе"))
    assert duty.controls == ["/clear", "/clear"]


def test_duty_stamp_is_written_atomically_into_duty_dir(tmp_path):
    m, _duty = _duty_manager(tmp_path, AskResult("busy"), AskResult("ok", reply="ок"))
    asyncio.run(m.send_message(1, "x"))
    stamp = m._settings.duty_dir / "last_used"
    assert stamp.exists() and float(stamp.read_text()) > 0
    assert not list(m._settings.duty_dir.glob("*.tmp"))  # no tmp left behind


def test_a_broken_duty_session_never_costs_the_reply(tmp_path):
    """Any unexpected failure in the duty path lands on exactly the message
    the user would have received before the feature existed."""
    m, duty = _duty_manager(
        tmp_path, AskResult("busy", busy_seconds=600.0), AskResult("ok")
    )

    def boom(*a, **kw):
        raise RuntimeError("tmux exploded")

    duty.ask = boom
    reply = asyncio.run(m.send_message(1, "x"))
    assert "ещё выполняется" in reply


def test_duty_failure_status_is_reported_honestly_not_as_silence(tmp_path):
    m, _duty = _duty_manager(tmp_path, AskResult("busy"), AskResult("rate_limited"))
    reply = asyncio.run(m.send_message(1, "x"))
    assert reply.strip()  # never empty: nothing retries a duty turn
    assert "ещё выполняется" in reply  # the main-session fact
    assert "Лимит" in reply  # and why the stand-in could not cover


def test_duty_empty_ok_reply_is_also_reported(tmp_path):
    m, _duty = _duty_manager(tmp_path, AskResult("busy"), AskResult("ok", reply="  "))
    reply = asyncio.run(m.send_message(1, "x"))
    assert reply.strip()
    assert "Дежурная сессия" in reply


def test_duty_notices_are_delivered_with_the_reply(tmp_path):
    """The duty session has its own owner notices and no watchdog
    watching it — same treatment send_message gives the main session's."""
    m, duty = _duty_manager(tmp_path, AskResult("busy"), AskResult("ok", reply="ответ"))
    pending = ["⚠️ <контекст> сброшен"]
    duty.pop_notices = lambda: [pending.pop()] if pending else []
    reply = asyncio.run(m.send_message(1, "x"))
    assert reply.startswith("⚠️ &lt;контекст&gt; сброшен")
    assert "ответ" in reply


def test_main_session_notices_still_ride_on_a_duty_reply(tmp_path):
    """'s notices are about the user's own conversation — they must
    not be swallowed just because the stand-in answered this round."""
    m, duty = _duty_manager(tmp_path, AskResult("busy"), AskResult("ok", reply="ответ"))
    pending = ["⚠️ контекст основной сессии сброшен"]
    m._session.pop_notices = lambda: [pending.pop()] if pending else []
    reply = asyncio.run(m.send_message(1, "x"))
    assert reply.startswith("⚠️ контекст основной сессии сброшен")
    assert "ответ" in reply and duty.prompts


def test_main_busy_status_is_still_scored_in_the_health_ledger(tmp_path):
    """A duty reply must not hide a busy main channel from the DeliveryGuard."""
    from d_brain.services import ask_health

    m, _duty = _duty_manager(tmp_path, AskResult("busy"), AskResult("ok", reply="ок"))
    asyncio.run(m.send_message(1, "x"))
    assert ask_health.read(tmp_path).last_status == "busy"


def test_main_turn_uses_the_chat_turn_timeout(tmp_path):
    m, _duty = _duty_manager(
        tmp_path, AskResult("ok", reply="ок"), AskResult("ok", reply="-")
    )
    asyncio.run(m.send_message(1, "x"))
    assert m._session.ask_kwargs[0]["timeout"] == 1500.0


def test_chat_turn_timeout_zero_restores_the_default_timeout(tmp_path):
    from d_brain.services.claude_session import DEFAULT_TIMEOUT

    m, _duty = _duty_manager(
        tmp_path,
        AskResult("ok", reply="ок"),
        AskResult("ok", reply="-"),
        chat_turn_timeout=0.0,
    )
    asyncio.run(m.send_message(1, "x"))
    assert m._session.ask_kwargs[0]["timeout"] == DEFAULT_TIMEOUT


def _fresh_long_run(runtime_dir, *, age: float = 60.0) -> None:
    """Write the watchdog marker the duty detour now requires as its first
    piece of evidence (blind review F1)."""
    import time

    from d_brain.services import long_run

    now = time.time()
    long_run.write(
        runtime_dir, long_run.LongRun(since=now - age, updated_ts=now, alerted=True)
    )


def test_is_main_busy_needs_two_confirming_probes(tmp_path, monkeypatch):
    """A turn a second away from finishing must not cost the user a
    context-less duty reply."""
    from d_brain.services import chat_session

    monkeypatch.setattr(chat_session, "_MAIN_BUSY_CONFIRM_SECONDS", 0.0)
    m = _manager(tmp_path, AskResult("ok", reply="x"))
    _fresh_long_run(tmp_path)

    m._session.pane_active_answers = [False]
    assert asyncio.run(m.is_main_busy()) is False

    m._session.pane_active_answers = [True, False]  # the turn finished
    assert asyncio.run(m.is_main_busy()) is False

    m._session.pane_active_answers = [True, True]
    assert asyncio.run(m.is_main_busy()) is True


def test_is_main_busy_needs_a_fresh_watchdog_marker(tmp_path, monkeypatch):
    """Blind review F1 (blocker): a busy-LOOKING pane is not enough.

    A wedged session holds a static footer signature forever (class B3), and
    every probe we own reads that as "busy". Diverting on the pane alone
    meant ask() was never entered, ask_health never recorded, fail_streak
    never grew, and delivery_guard's restart backstop never fired — the
    owner chats with the stand-in indefinitely while delivery health reads
    green. The marker is written by a different, independently-alive
    process and goes stale on its own: requiring it keeps the backstop
    armed.
    """
    from d_brain.services import chat_session, long_run

    monkeypatch.setattr(chat_session, "_MAIN_BUSY_CONFIRM_SECONDS", 0.0)
    m = _manager(tmp_path, AskResult("ok", reply="x"))

    # No marker at all: the pane can say "busy" as loudly as it likes.
    m._session.pane_active_answers = [True, True]
    assert asyncio.run(m.is_main_busy()) is False
    assert m._session.pane_active_answers == [True, True]  # not even probed

    # A STALE marker (the watchdog died mid-run) counts as no marker.
    import time

    now = time.time()
    long_run.write(
        tmp_path,
        long_run.LongRun(since=now - 3600.0, updated_ts=now - 3600.0, alerted=True),
    )
    assert asyncio.run(m.is_main_busy()) is False

    _fresh_long_run(tmp_path)
    assert asyncio.run(m.is_main_busy()) is True


def test_the_duty_detour_does_not_touch_the_health_ledger_at_all(
    tmp_path, monkeypatch
):
    """The other half of F1, corrected in review round 3 (R2).

    The detour skips ask(), so no turn ended — and the ledger must not move
    AT ALL, not even by a neutral row. A neutral status leaves fail_streak
    alone but still advances Health.last_ts, and delivery_guard.decide()
    reads last_ts in two branches that rest on "the ledger only moves when a
    turn ends": the stale-evidence check and `last_ts <= _last_restart_ts`
    ("no failed turn since the last restart"). With a real streak of 3 and
    one restart already spent, one message sent during a long run would have
    looked like fresh evidence and bought a second restart plus an
    escalation with the channel never having been tried.
    """
    from d_brain.services import ask_health, chat_session

    monkeypatch.setattr(chat_session, "_MAIN_BUSY_CONFIRM_SECONDS", 0.0)
    ask_health.record(tmp_path, "error", clock_fn=lambda: 1.0)
    before = ask_health.path_for(tmp_path).read_bytes()

    m = _manager(tmp_path, AskResult("ok", reply="x"))
    _fresh_long_run(tmp_path)
    m._session.pane_active_answers = [True, True]

    assert asyncio.run(m.is_main_busy()) is True
    # Byte-for-byte untouched: neither the streak, nor last_status, nor the
    # timestamp delivery_guard keys on.
    assert ask_health.path_for(tmp_path).read_bytes() == before


def test_without_a_marker_the_ordinary_path_still_scores_the_ledger(
    tmp_path, monkeypatch
):
    """The point of F1: when the detour does NOT fire, the message goes
    through ask() and the ledger keeps moving — so delivery_guard's backstop
    stays armed on a wedged pane instead of being blinded by the detour."""
    from d_brain.services import ask_health, chat_session

    monkeypatch.setattr(chat_session, "_MAIN_BUSY_CONFIRM_SECONDS", 0.0)
    m = _manager(tmp_path, AskResult("busy", detail="pane still busy"))
    m._session.pane_active_answers = [True, True]  # a wedged pane says "busy"

    assert asyncio.run(m.is_main_busy()) is False  # no marker ⇒ no detour
    asyncio.run(m.send_message(1, "x"))
    health = ask_health.read(tmp_path)
    assert health.last_status == "busy"
    assert health.fail_streak == 1  # the backstop is armed


def test_is_main_busy_is_false_when_the_probe_is_missing_or_throws(tmp_path):
    """Without a usable probe the message takes today's path into ask(),
    rather than being diverted to a context-less session."""
    class NoProbe(FakeSession):
        is_pane_turn_active = None  # a driver without the probe at all

    from d_brain.services.chat_session import ChatSessionManager

    _fresh_long_run(tmp_path)
    m = ChatSessionManager(
        tmp_path,
        session=NoProbe(AskResult("ok", reply="x")),
        health_dir=tmp_path,
    )
    assert asyncio.run(m.is_main_busy()) is False

    m2 = _manager(tmp_path, AskResult("ok", reply="x"))

    def boom():
        raise RuntimeError("no tmux")

    m2._session.is_pane_turn_active = boom
    assert asyncio.run(m2.is_main_busy()) is False


def test_under_codex_the_detour_is_off_and_everything_goes_through_ask(tmp_path):
    """F1's documented degradation: the watchdog classifies a turn as active
    by matching pane signatures over ``capture_text()``, and
    ``CodexExecDriver.capture_text`` returns only its own journal
    vocabulary — which cannot match them. So no long-run marker is ever
    written under that engine, this gate is permanently False, and every
    message takes the ask() path with its busy-wait and its duty fallback,
    exactly as before the gate existed. No fast path, but no blinded ledger
    either."""
    from d_brain.services import ask_health

    m, duty = _duty_manager(
        tmp_path,
        AskResult("busy", busy_seconds=600.0),
        AskResult("ok", reply="из дежурной"),
        chat_engine="codex",
    )
    # No marker on disk — which is the whole point; the pane may still be busy.
    m._session.pane_active_answers = [True, True]
    assert asyncio.run(m.is_main_busy()) is False

    # ...and the ordinary path still ends in a duty reply, with the main
    # session's busy outcome honestly scored.
    reply = asyncio.run(m.send_message(1, "x"))
    assert "из дежурной" in reply and duty.prompts
    assert ask_health.read(tmp_path).last_status == "busy"


def test_injected_main_session_never_auto_resolves_a_real_duty_session(tmp_path):
    """Safety rail for tools and tests: a manager handed a session must not
    quietly start a REAL second engine session as its fallback."""
    from d_brain.services.chat_session import ChatSessionManager

    m = ChatSessionManager(
        tmp_path, session=FakeSession(AskResult("busy")), health_dir=tmp_path
    )
    assert m._auto_duty is False
    assert m._resolve_duty() is None
