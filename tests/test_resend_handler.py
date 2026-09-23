"""Tests for the /resend command handler.

Covers the five honest, distinct statuses last_reply_for_resend() can
return, and the 5s in-memory anti-spam cooldown. The ChatSessionManager
itself is faked out here (already covered by test_chat_session.py) so
these tests focus purely on the handler's status -> message mapping and
the cooldown gate.
"""

import asyncio

from d_brain.config import Settings


class FakeUser:
    def __init__(self, user_id: int):
        self.id = user_id


class FakeChat:
    def __init__(self, chat_id: int):
        self.id = chat_id


class FakeMessage:
    def __init__(self, user_id: int = 1, chat_id: int = 10):
        self.from_user = FakeUser(user_id)
        self.chat = FakeChat(chat_id)
        self.answers: list[str] = []

    async def answer(self, text, **kwargs):
        self.answers.append(text)


class FakeBot:
    def __init__(self):
        self.messages: list[tuple[int, str]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text))


class FakeManager:
    def __init__(self, result: tuple[str, str | None]):
        self.result = result
        self.calls = 0

    async def resend_last_reply(self, user_id: int) -> tuple[str, str | None]:
        self.calls += 1
        return self.result


def _settings(tmp_path):
    return Settings(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path,
        _env_file=None,
    )


def _setup(monkeypatch, tmp_path, result: tuple[str, str | None]):
    from d_brain.bot.handlers import resend

    resend._last_press.clear()  # tests must not leak cooldown state
    manager = FakeManager(result)
    monkeypatch.setattr(resend, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(resend, "ChatSessionManager", lambda vault_path: manager)
    return resend, manager


def test_resend_ready_sends_prefixed_reply_via_send_response(monkeypatch, tmp_path):
    resend, manager = _setup(monkeypatch, tmp_path, ("ready", "the last answer"))
    message = FakeMessage()
    bot = FakeBot()

    asyncio.run(resend.cmd_resend(message, bot))

    assert manager.calls == 1
    assert not message.answers  # went through send_response -> bot, not message.answer
    assert bot.messages
    assert "the last answer" in bot.messages[0][1]
    assert "Повтор" in bot.messages[0][1]


def test_resend_empty_status_is_honest_not_an_error(monkeypatch, tmp_path):
    resend, manager = _setup(monkeypatch, tmp_path, ("empty", None))
    message = FakeMessage()
    bot = FakeBot()

    asyncio.run(resend.cmd_resend(message, bot))

    assert manager.calls == 1
    assert not bot.messages
    assert message.answers
    text = message.answers[0].lower()
    assert "нечего" in text or "истори" in text


def test_resend_in_progress_status_says_still_working_not_lost(monkeypatch, tmp_path):
    resend, manager = _setup(monkeypatch, tmp_path, ("in_progress", None))
    message = FakeMessage()
    bot = FakeBot()

    asyncio.run(resend.cmd_resend(message, bot))

    assert manager.calls == 1
    assert message.answers
    text = message.answers[0].lower()
    assert "потерян" not in text
    assert "lost" not in text
    assert "работает" in text or "готов" in text


def test_resend_no_markers_status_is_honest_not_still_working(monkeypatch, tmp_path):
    """F-1 fix (round 2 review): 'no_markers' must read as 'nothing to
    recover, ask again' — never claim the session is still working (that
    would repeat the original bug this status was created to fix)."""
    resend, manager = _setup(monkeypatch, tmp_path, ("no_markers", None))
    message = FakeMessage()
    bot = FakeBot()

    asyncio.run(resend.cmd_resend(message, bot))

    assert manager.calls == 1
    assert not bot.messages
    assert message.answers
    text = message.answers[0].lower()
    assert "работает" not in text
    assert "потерян" not in text
    assert "lost" not in text
    assert "ещё раз" in text or "маркер" in text


def test_resend_unavailable_status_is_honest(monkeypatch, tmp_path):
    resend, manager = _setup(monkeypatch, tmp_path, ("unavailable", None))
    message = FakeMessage()
    bot = FakeBot()

    asyncio.run(resend.cmd_resend(message, bot))

    assert manager.calls == 1
    assert message.answers
    text = message.answers[0].lower()
    assert "не смог" in text or "❌" in message.answers[0]


def test_resend_cooldown_blocks_rapid_double_press(monkeypatch, tmp_path):
    resend, manager = _setup(monkeypatch, tmp_path, ("ready", "x"))
    bot = FakeBot()

    asyncio.run(resend.cmd_resend(FakeMessage(), bot))
    assert manager.calls == 1

    second = FakeMessage()
    asyncio.run(resend.cmd_resend(second, bot))
    assert manager.calls == 1  # not re-scanned / re-sent
    assert second.answers
    assert "подожди" in second.answers[0].lower()


def test_resend_cooldown_is_per_user(monkeypatch, tmp_path):
    resend, manager = _setup(monkeypatch, tmp_path, ("ready", "x"))
    bot = FakeBot()

    asyncio.run(resend.cmd_resend(FakeMessage(user_id=1), bot))
    assert manager.calls == 1

    other_user = FakeMessage(user_id=2)
    asyncio.run(resend.cmd_resend(other_user, bot))
    assert manager.calls == 2  # a different user is not gated by user 1's cooldown
    assert not other_user.answers


def test_resend_cooldown_expires_after_window(monkeypatch, tmp_path):
    resend, manager = _setup(monkeypatch, tmp_path, ("ready", "x"))
    bot = FakeBot()

    asyncio.run(resend.cmd_resend(FakeMessage(), bot))
    assert manager.calls == 1

    # Simulate the cooldown window having elapsed.
    resend._last_press[1] -= resend._COOLDOWN_SECONDS + 1

    second = FakeMessage()
    asyncio.run(resend.cmd_resend(second, bot))
    assert manager.calls == 2
    assert not second.answers
