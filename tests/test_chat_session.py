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

    def ask(self, prompt: str, **kwargs) -> AskResult:
        self.prompts.append(prompt)
        return self.result

    def clear(self) -> None:
        self.cleared += 1

    def last_reply_for_resend(self) -> tuple[str, str | None]:
        self.resend_calls += 1
        return self.resend_result

    def force_recover(self) -> bool:
        self.force_recover_calls += 1
        return self.force_recover_result


def _manager(tmp_path, result: AskResult):
    from d_brain.services.chat_session import ChatSessionManager

    # health_dir is pinned to tmp_path on purpose. Left unset it resolves
    # lazily from Settings, which SUCCEEDS on the production checkout (there is
    # a real .env there) — and the suite would then write to the live
    # ~/.dbrain/ask-health.json that the running watchdog reads, quietly
    # clearing a genuine fail streak.
    return ChatSessionManager(
        tmp_path, session=FakeSession(result), health_dir=tmp_path
    )


def test_send_message_returns_reply_on_ok(tmp_path):
    m = _manager(tmp_path, AskResult("ok", reply="привет"))
    reply = asyncio.run(m.send_message(1, "здравствуй"))
    assert reply == "привет"
    assert m._session.prompts == ["здравствуй"]


def test_send_message_maps_rate_limited(tmp_path):
    m = _manager(tmp_path, AskResult("rate_limited"))
    reply = asyncio.run(m.send_message(1, "x"))
    assert "Лимит" in reply


def test_send_message_maps_busy_to_an_honest_non_error_message(tmp_path):
    """Busy-panel UX finding (2026-08-22): a legitimately busy panel must
    NOT read as '❌ Ошибка сессии' — that misled the owner on 2026-08-22 04:27
    UTC into thinking something had crashed when nothing was actually
    wrong."""
    m = _manager(
        tmp_path, AskResult("busy", detail="pane still busy with a previous turn")
    )
    reply = asyncio.run(m.send_message(1, "x"))
    assert "Ошибка" not in reply
    assert "ещё выполняется" in reply or "busy" in reply.lower()


def test_send_message_maps_busy_active_to_an_honest_non_error_message(tmp_path):
    """agent-infra-backlog item 22: a leftover turn that demonstrably kept
    progressing must map to a distinct, non-scary message that does NOT
    promise a callback (B3's wording constraint — nothing re-sends the
    answer later)."""
    m = _manager(
        tmp_path,
        AskResult(
            "busy_active",
            detail="pane busy with a live, progressing turn",
            busy_seconds=30.0,
        ),
    )
    reply = asyncio.run(m.send_message(1, "x"))
    assert "Ошибка" not in reply
    assert "отвечу, как только освобожусь" not in reply
    assert "занята" in reply.lower() or "busy" in reply.lower()


def test_send_message_busy_active_mentions_elapsed_minutes_when_long(tmp_path):
    m = _manager(
        tmp_path,
        AskResult(
            "busy_active",
            detail="pane busy with a live, progressing turn",
            busy_seconds=185.0,  # ~3 min
        ),
    )
    reply = asyncio.run(m.send_message(1, "x"))
    assert "3 мин" in reply


def test_send_message_busy_active_without_elapsed_uses_base_message(tmp_path):
    m = _manager(
        tmp_path,
        AskResult("busy_active", detail="pane busy with a live, progressing turn"),
    )
    reply = asyncio.run(m.send_message(1, "x"))
    assert "Ошибка" not in reply
    assert "мин)" not in reply  # no elapsed-time parenthetical


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
    an ordinary timeout (leftover marker, still generic) keeps the existing
    wording."""
    m = _manager(tmp_path, AskResult("timeout", detail="no reply in 3600s"))
    reply = asyncio.run(m.send_message(1, "x"))
    assert "Превышено время ожидания" in reply


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


# ── resend_last_reply (backlog item 14: /resend) ──────────────────────────


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
