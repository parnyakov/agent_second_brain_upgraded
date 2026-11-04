"""A Telegram "Reply" carries its referent into the prompt."""

import asyncio
import io
from datetime import datetime

from aiogram.types import (
    Animation,
    Chat,
    Document,
    ExternalReplyInfo,
    Message,
    MessageOriginUser,
    PhotoSize,
    TextQuote,
    User,
    Voice,
)

from d_brain.bot.handlers import chat
from d_brain.bot.handlers.chat import REPLY_CONTEXT_LIMIT, reply_context

CHAT = Chat(id=10, type="private")
BOT = User(id=900, is_bot=True, first_name="Bot")
HUMAN = User(id=7, is_bot=False, first_name="User")
NOW = datetime(2026, 9, 4, 12, 30, 0)


def _msg(**kw) -> Message:
    base = {"message_id": 1, "date": NOW, "chat": CHAT, "from_user": HUMAN}
    base.update(kw)
    return Message(**base)


def _bot_msg(**kw) -> Message:
    return _msg(message_id=50, date=datetime(2026, 9, 4, 9, 5), from_user=BOT, **kw)


def test_plain_message_has_no_context():
    assert reply_context(_msg(text="Да")) == ""


def test_reply_to_bot_text_names_sender_time_and_text():
    msg = _msg(
        text="Да", reply_to_message=_bot_msg(text="Перенести встречу на пятницу?")
    )
    assert reply_context(msg) == (
        "[в ответ на сообщение от бота 09:05: «Перенести встречу на пятницу?»]\n"
    )


def test_reply_to_own_message_is_attributed_to_the_user():
    replied = _msg(message_id=40, date=datetime(2026, 9, 4, 8, 0), text="заметка")
    line = reply_context(_msg(text="и ещё", reply_to_message=replied))
    assert line.startswith("[в ответ на сообщение от пользователя 08:00:")


def test_quote_wins_over_full_text():
    msg = _msg(
        text="Да",
        reply_to_message=_bot_msg(text="Вариант A. Вариант B. Вариант C."),
        quote=TextQuote(text="Вариант B.", position=11),
    )
    line = reply_context(msg)
    assert ", цитата: «Вариант B.»" in line
    assert "Вариант A" not in line


def test_long_text_is_cut_keeping_the_start():
    long_text = "начало " + "x" * 2000 + " конец"
    line = reply_context(_msg(text="Да", reply_to_message=_bot_msg(text=long_text)))
    assert "«начало " in line and "конец" not in line
    assert line.endswith("…»]\n")
    assert len(line) < REPLY_CONTEXT_LIMIT + 80


def test_media_without_text_is_named_by_type():
    photo = _bot_msg(
        photo=[PhotoSize(file_id="p", file_unique_id="u", width=1, height=1)]
    )
    assert reply_context(_msg(text="ок", reply_to_message=photo)) == (
        "[в ответ на сообщение от бота 09:05 (фото)]\n"
    )
    voice = _msg(
        message_id=41,
        date=datetime(2026, 9, 4, 11, 0),
        voice=Voice(file_id="v", file_unique_id="u", duration=3),
    )
    assert "(голосовое)" in reply_context(_msg(text="ок", reply_to_message=voice))


def test_document_with_caption_keeps_both():
    doc = _bot_msg(
        document=Document(file_id="d", file_unique_id="u", file_name="report.pdf"),
        caption="Отчёт за неделю",
    )
    line = reply_context(_msg(text="спасибо", reply_to_message=doc))
    assert (
        line == "[в ответ на сообщение от бота 09:05 (документ): «Отчёт за неделю»]\n"
    )


def test_reply_to_an_older_day_shows_the_date():
    old = _msg(
        message_id=3, date=datetime(2026, 9, 1, 18, 45), from_user=BOT, text="вопрос"
    )
    assert "от бота 01.09 18:45:" in reply_context(
        _msg(text="Да", reply_to_message=old)
    )


def test_external_reply_uses_origin_and_quote():
    ext = ExternalReplyInfo(
        origin=MessageOriginUser(date=datetime(2026, 9, 4, 10, 0), sender_user=HUMAN)
    )
    msg = _msg(
        text="смотри", external_reply=ext, quote=TextQuote(text="фрагмент", position=0)
    )
    assert reply_context(msg) == (
        "[в ответ на сообщение из другого чата 10:00, цитата: «фрагмент»]\n"
    )


# ── the context actually reaches the session ───────────────────────────────


class _Settings:
    def __init__(self, vault_path):
        self.vault_path = vault_path
        self.deepgram_api_key = "k"


class _Mgr:
    def __init__(self):
        self.sent: list[str] = []

    async def send_message(self, user_id, prompt):
        self.sent.append(prompt)
        return "ок"

    def is_turn_active(self):
        return False


class _Bot:
    def __init__(self, payload=b"bytes"):
        self.payload = payload
        self.messages: list[str] = []

    async def send_message(self, chat_id, text, **kw):
        self.messages.append(text)

    async def send_chat_action(self, chat_id, action):
        pass

    async def get_file(self, file_id):
        class F:
            file_path = "x/file"

        return F()

    async def download_file(self, path):
        return io.BytesIO(self.payload)


def _wire(monkeypatch, tmp_path):
    mgr = _Mgr()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    monkeypatch.setattr(chat, "get_settings", lambda: _Settings(tmp_path))
    return mgr


QUESTION = _bot_msg(text="Отправить черновик сейчас?")
EXPECTED = "[в ответ на сообщение от бота 09:05: «Отправить черновик сейчас?»]\n"


def test_text_reply_reaches_the_session_with_context(monkeypatch, tmp_path):
    mgr = _wire(monkeypatch, tmp_path)
    asyncio.run(
        chat.handle_chat_text(_msg(text="Да", reply_to_message=QUESTION), _Bot())
    )
    assert mgr.sent == [EXPECTED + "Да"]


def test_context_does_not_change_command_classification(monkeypatch):
    calls: list[str] = []

    class Mgr(_Mgr):
        async def send_control(self, text):
            calls.append(text)

    mgr = Mgr()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    asyncio.run(chat._dispatch_text(_Bot(), 10, 1, "/clear", context=EXPECTED))
    assert calls == ["/clear"] and mgr.sent == []


def test_steer_carries_the_context(monkeypatch):
    steered: list[str] = []

    class Mgr(_Mgr):
        def is_turn_active(self):
            return True

        def is_steerable_turn(self):
            return True

        async def steer(self, text):
            steered.append(text)

    monkeypatch.setattr(chat, "_get_manager", lambda: Mgr())
    asyncio.run(chat._dispatch_text(_Bot(), 10, 1, "Да", context=EXPECTED))
    assert steered == [EXPECTED + "Да"]


def test_voice_reply_reaches_the_session_with_context(monkeypatch, tmp_path):
    mgr = _wire(monkeypatch, tmp_path)

    class Transcriber:
        def __init__(self, _key):
            pass

        async def transcribe(self, _payload):
            return "да, отправляй"

    monkeypatch.setattr(chat, "DeepgramTranscriber", Transcriber)
    msg = _msg(
        voice=Voice(file_id="v", file_unique_id="u", duration=2),
        reply_to_message=QUESTION,
    )
    asyncio.run(chat.handle_chat_voice(msg, _Bot()))
    assert mgr.sent == [EXPECTED + "[voice] да, отправляй"]


def test_document_reply_reaches_the_session_with_context(monkeypatch, tmp_path):
    mgr = _wire(monkeypatch, tmp_path)
    msg = _msg(
        document=Document(file_id="d", file_unique_id="u", file_name="notes.txt"),
        reply_to_message=QUESTION,
    )
    asyncio.run(chat.handle_chat_media(msg, _Bot()))
    assert len(mgr.sent) == 1 and mgr.sent[0].startswith(EXPECTED)
    assert "notes.txt" in mgr.sent[0]


def test_album_prompt_carries_the_first_items_context():
    items = [
        {
            "kind": "photo",
            "rel_path": "a.jpg",
            "caption": "",
            "fwd": "",
            "reply": EXPECTED,
        },
        {"kind": "photo", "rel_path": "b.jpg", "caption": "", "fwd": "", "reply": ""},
    ]
    prompt = chat.build_album_prompt(items)
    assert prompt.startswith(EXPECTED) and prompt.count("в ответ на") == 1


def test_slash_command_reply_keeps_the_command_first(monkeypatch):
    mgr = _Mgr()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    asyncio.run(chat._dispatch_text(_Bot(), 10, 1, "/process", context=EXPECTED))
    assert mgr.sent == ["/process\n" + EXPECTED.rstrip()]


def test_gif_is_named_as_gif_not_document():
    gif = _bot_msg(
        animation=Animation(
            file_id="a", file_unique_id="u", width=1, height=1, duration=1
        ),
        document=Document(file_id="a", file_unique_id="u"),
    )
    assert "(GIF)" in reply_context(_msg(text="ок", reply_to_message=gif))
