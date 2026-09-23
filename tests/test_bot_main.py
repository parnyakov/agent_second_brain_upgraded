"""Tests for (Part A): reply-keyboard removal.

Covers:
- cmd_start no longer sends reply_markup
- buttons.router is disconnected from create_dispatcher(), other routers stay
- bot_commands() feeds Telegram's native "/" menu
- keyboards.py / handlers/buttons.py are left importable (disabled, not deleted)
- _send_keyboard_removal_once() one-shot marker-file gating
"""

import asyncio

from d_brain.config import Settings


def _settings(**over):
    base = dict(telegram_bot_token="t", deepgram_api_key="d", _env_file=None)
    base.update(over)
    return Settings(**base)


class FakeMessage:
    def __init__(self):
        self.calls: list[tuple[tuple, dict]] = []

    async def answer(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeBot:
    def __init__(self):
        self.messages: list[tuple[int, str, dict]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text, kwargs))


# ── cmd_start no longer ships the reply keyboard ───────────────────────────


def test_cmd_start_sends_no_reply_markup():
    from d_brain.bot.handlers.commands import cmd_start

    message = FakeMessage()
    asyncio.run(cmd_start(message))

    assert len(message.calls) == 1
    _, kwargs = message.calls[0]
    assert kwargs.get("reply_markup") is None


# ── router wiring ───────────────────────────────────────────────────────────


def test_buttons_router_not_included_others_are():
    from d_brain.bot.main import create_dispatcher

    dp = create_dispatcher()
    included_names = {router.name for router in dp.sub_routers}

    assert "buttons" not in included_names
    assert {"commands", "process", "chat", "resend", "work"} <= included_names
    # /work must be reachable while the session is busy — registered after
    # chat.router's catch-all it would never fire at all.
    ordered = [router.name for router in dp.sub_routers]
    assert ordered.index("work") < ordered.index("chat")


# ── native "/" command menu ─────────────────────────────────────────────────


def test_bot_commands_has_expected_entries():
    from d_brain.bot.main import bot_commands

    commands = bot_commands()
    assert len(commands) == 9

    by_command = {c.command: c.description for c in commands}
    assert set(by_command) == {
        "status",
        "process",
        "help",
        "onboarding",
        "new",
        "compact",
        "resend",
        "work",
        "relogin",
    }
    for description in by_command.values():
        assert description


# ── disabled, not deleted ───────────────────────────────────────────────────


def test_disabled_keyboard_code_still_importable():
    from d_brain.bot.handlers.buttons import btn_process, btn_status
    from d_brain.bot.keyboards import get_main_keyboard

    assert callable(get_main_keyboard)
    assert callable(btn_status)
    assert callable(btn_process)


# ── one-time keyboard-removal ping ──────────────────────────────────────────


def test_keyboard_removal_sent_once_then_gated_by_marker(tmp_path):
    from d_brain.bot.main import _send_keyboard_removal_once

    settings = _settings(runtime_dir=tmp_path / "rt", allowed_user_ids=[111])
    marker = settings.runtime_dir / "keyboard_removed"
    assert not marker.exists()

    bot = FakeBot()
    asyncio.run(_send_keyboard_removal_once(bot, settings))

    assert len(bot.messages) == 1
    chat_id, _text, kwargs = bot.messages[0]
    assert chat_id == 111
    assert "reply_markup" in kwargs
    assert marker.exists()

    # Second call: marker already present -> no further send.
    asyncio.run(_send_keyboard_removal_once(bot, settings))
    assert len(bot.messages) == 1


def test_keyboard_removal_marker_not_written_on_send_failure(tmp_path):
    """Regression: a transient send failure must not permanently suppress
    the notice. Marker must only be written after a successful send."""
    from d_brain.bot.main import _send_keyboard_removal_once

    class FailingBot:
        async def send_message(self, *args, **kwargs):
            raise RuntimeError("network blip")

    settings = _settings(runtime_dir=tmp_path / "rt", allowed_user_ids=[111])
    marker = settings.runtime_dir / "keyboard_removed"

    asyncio.run(_send_keyboard_removal_once(FailingBot(), settings))

    assert not marker.exists()


def test_keyboard_removal_skipped_without_admin_chat_id(tmp_path):
    from d_brain.bot.main import _send_keyboard_removal_once

    settings = _settings(runtime_dir=tmp_path / "rt", allowed_user_ids=[])
    bot = FakeBot()

    asyncio.run(_send_keyboard_removal_once(bot, settings))

    assert bot.messages == []
    assert not (settings.runtime_dir / "keyboard_removed").exists()


# ── /onboarding hands the conversation to the agent ─────────────────────────


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_onboarding_command_reuses_chat_dispatch(monkeypatch):
    from d_brain.bot.handlers import chat, commands

    calls = []

    async def fake_dispatch(bot, chat_id, user_id, text):
        calls.append((bot, chat_id, user_id, text))

    monkeypatch.setattr(chat, "_dispatch_text", fake_dispatch)
    bot = FakeBot()
    message = _Obj(
        text="/onboarding давай начнём с целей",
        from_user=_Obj(id=7),
        chat=_Obj(id=42),
    )

    asyncio.run(commands.cmd_onboarding(message, bot))

    assert len(calls) == 1
    got_bot, chat_id, user_id, prompt = calls[0]
    assert (got_bot, chat_id, user_id) == (bot, 42, 7)
    assert prompt.startswith("Запусти skill onboarding")
    assert prompt.endswith("Сообщение пользователя: давай начнём с целей")
    # A plain model turn, not a control/TUI slash command.
    assert chat.classify_command(prompt) == "turn"


def test_onboarding_prompt_without_text():
    from d_brain.bot.handlers.commands import ONBOARDING_PROMPT, build_onboarding_prompt

    assert build_onboarding_prompt("/onboarding") == ONBOARDING_PROMPT
    assert build_onboarding_prompt("/onboarding@my_bot   ") == ONBOARDING_PROMPT


def test_start_and_help_mention_onboarding():
    from d_brain.bot.handlers.commands import cmd_help, cmd_start

    for handler in (cmd_start, cmd_help):
        message = FakeMessage()
        asyncio.run(handler(message))
        assert "/onboarding" in message.calls[0][0][0]
