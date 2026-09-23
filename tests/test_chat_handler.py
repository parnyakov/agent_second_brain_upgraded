"""Tests for the unified chat handler (v3.0: immediate routing, no debounce)."""

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _reset_media_claim():
    """The media-dispatch claim is module-global process state. A test that
    left it held would silently wedge every test after it — exactly the
    "stale state trusted as fresh" trap this codebase has a history of."""
    from d_brain.bot.handlers import chat

    chat._release_media_dispatch()
    yield
    chat._release_media_dispatch()


class FakeManager:
    def __init__(self, reply="ответ"):
        self.reply = reply
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, user_id: int, prompt: str) -> str:
        self.sent.append((user_id, prompt))
        return self.reply

    def is_turn_active(self) -> bool:
        # Idle by default — the media busy-guard
        # consults this before anything can reach ask().
        return False


class FakeBot:
    def __init__(self):
        self.messages: list[tuple[int, str]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text))

    async def send_chat_action(self, chat_id, action):
        pass


def test_text_message_routed_immediately(monkeypatch):
    """v3.0: an incoming message reaches the session manager immediately —
    no debounce buffer, no delayed flush."""
    from d_brain.bot.handlers import chat

    mgr = FakeManager(reply="<b>готово</b>")
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    asyncio.run(chat._process_and_reply(bot, chat_id=10, user_id=1, prompt="привет"))

    assert mgr.sent == [(1, "привет")]
    assert bot.messages and "готово" in bot.messages[0][1]


def test_no_debounce_infrastructure_left():
    """The debounce buffer is fully removed."""
    from d_brain.bot.handlers import chat

    zombies = ("DEBOUNCE_SECONDS", "DebounceBuffer", "_add_to_buffer",
               "_debounce_flush", "_buffers")
    for name in zombies:
        assert not hasattr(chat, name), f"zombie debounce symbol: {name}"


# ── slash commands split by BEHAVIOR, not by leading "/" ───────────────────


def test_classify_command_skill_is_normal_turn():
    from d_brain.bot.handlers.chat import classify_command

    assert classify_command("/vault-note сохрани мысль") == "turn"
    assert classify_command("привет, как дела?") == "turn"


def test_classify_command_control_is_fire_and_forget():
    from d_brain.bot.handlers.chat import classify_command

    assert classify_command("/clear") == "control"
    assert classify_command("/model sonnet") == "control"


def test_classify_command_tui_is_unsupported():
    from d_brain.bot.handlers.chat import classify_command

    assert classify_command("/agents") == "tui"
    assert classify_command("/config") == "tui"
    assert classify_command("/login") == "tui"


def test_control_command_dispatches_fire_and_forget(monkeypatch):
    from d_brain.bot.handlers import chat

    class Mgr(FakeManager):
        def __init__(self):
            super().__init__()
            self.controls: list[str] = []

        async def send_control(self, text: str) -> None:
            self.controls.append(text)

    mgr = Mgr()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="/clear"))

    assert mgr.controls == ["/clear"]
    assert mgr.sent == []  # no marker turn started
    assert bot.messages  # got an acknowledgement


def test_tui_command_rejected_with_hint(monkeypatch):
    from d_brain.bot.handlers import chat

    mgr = FakeManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="/agents"))

    assert mgr.sent == []
    assert bot.messages and "attach" in bot.messages[0][1]


# ── concurrent input: steer / interrupt / queue-as-ask ─────────────────────


def test_classify_concurrent_input_modes():
    from d_brain.bot.handlers.chat import classify_concurrent_input

    assert classify_concurrent_input("привет", turn_active=False) == "ask"
    assert classify_concurrent_input("пиши короче", turn_active=True) == "steer"
    assert classify_concurrent_input("стоп", turn_active=True) == "interrupt"
    assert classify_concurrent_input("/stop", turn_active=True) == "interrupt"
    assert classify_concurrent_input("стоп", turn_active=False) == "ask"


class SteerableManager(FakeManager):
    def __init__(self, *, active=False):
        super().__init__()
        self.active = active
        self.steered: list[str] = []
        self.interrupts = 0

    def is_turn_active(self) -> bool:
        return self.active

    def is_steerable_turn(self) -> bool:
        return self.active  # a plain chat turn — steerable while active

    async def steer(self, text: str) -> None:
        self.steered.append(text)

    async def interrupt(self) -> None:
        self.interrupts += 1


def test_plain_text_during_active_turn_steers(monkeypatch):
    from d_brain.bot.handlers import chat

    mgr = SteerableManager(active=True)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="пиши короче"))
    assert mgr.steered == ["пиши короче"]
    assert mgr.sent == []  # no new turn started


def test_stop_word_interrupts_active_turn(monkeypatch):
    from d_brain.bot.handlers import chat

    mgr = SteerableManager(active=True)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="стоп"))
    assert mgr.interrupts == 1
    assert mgr.sent == [] and mgr.steered == []


def test_text_when_idle_goes_to_normal_turn(monkeypatch):
    from d_brain.bot.handlers import chat

    mgr = SteerableManager(active=False)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="привет"))
    assert mgr.sent == [(1, "привет")]
    assert mgr.steered == []


# ── Step D: /stop reaching a pane-active-but-lock-free turn (F3, blind-review
# round) ─────────────────────────────────────────────────────────────────


class PaneActiveManager(FakeManager):
    """Lock free (is_turn_active == False, so classify_concurrent_input
    alone always reads this as idle) but the PANE itself may still be
    genuinely busy — the exact shape an unattended long cascade produces:
    nothing called ask() for this turn, so no lock is held, while the pane
    keeps working. is_pane_turn_active() is the only thing that can see
    this; SteerableManager above deliberately does NOT implement it, so any
    test that never needs this branch (the vast majority) is unaffected."""

    def __init__(self, *, pane_active: bool) -> None:
        super().__init__()
        self.pane_active = pane_active
        self.interrupts = 0

    def is_turn_active(self) -> bool:
        return False

    def is_pane_turn_active(self) -> bool:
        return self.pane_active

    async def interrupt(self) -> None:
        self.interrupts += 1


def test_stop_word_reaches_pane_active_unattended_turn(monkeypatch):
    """The core Step D fix: with the ask-lock free but the pane genuinely
    busy, a stop-word must reach interrupt() directly — not fall through to
    classify_concurrent_input (which would read turn_active=False and route
    it as an ordinary 'ask', exactly the bug this step closes: the user had
    no way to reclaim the channel during an unattended cascade)."""
    from d_brain.bot.handlers import chat

    mgr = PaneActiveManager(pane_active=True)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="/stop"))
    assert mgr.interrupts == 1
    assert mgr.sent == []  # never fell through to a normal ask() turn
    assert bot.messages and "Останавливаю" in bot.messages[0][1]


def test_stop_word_with_lock_free_and_pane_idle_goes_to_normal_turn(monkeypatch):
    """The pane-active check must not fire when the pane really IS idle —
    a stop-word with nothing to interrupt is just ordinary text, handled by
    the normal classify_concurrent_input path unchanged."""
    from d_brain.bot.handlers import chat

    mgr = PaneActiveManager(pane_active=False)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="/stop"))
    assert mgr.interrupts == 0
    assert mgr.sent == [(1, "/stop")]


# ── media input: photo / document / video / audio (v3.0.x) ────────────────


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_extract_media_photo_takes_largest():
    from d_brain.bot.handlers.chat import extract_media

    msg = _Stub(
        photo=[_Stub(file_id="small"), _Stub(file_id="big")],
        document=None, video=None, audio=None, animation=None, video_note=None,
    )
    kind, file_id, ext, name = extract_media(msg)
    assert (kind, file_id, ext) == ("photo", "big", "jpg")


def test_extract_media_document_keeps_name_and_ext():
    from d_brain.bot.handlers.chat import extract_media

    doc = _Stub(file_id="d1", file_name="report Q2.pdf")
    msg = _Stub(photo=None, document=doc, video=None, audio=None,
                animation=None, video_note=None)
    kind, file_id, ext, name = extract_media(msg)
    assert (kind, file_id, ext, name) == ("document", "d1", "pdf", "report Q2.pdf")


def test_extract_media_video_note_is_mp4():
    from d_brain.bot.handlers.chat import extract_media

    msg = _Stub(photo=None, document=None, video=None, audio=None,
                animation=None, video_note=_Stub(file_id="v1"))
    kind, file_id, ext, name = extract_media(msg)
    assert (kind, ext) == ("video_note", "mp4")


def test_forward_note_variants():
    from d_brain.bot.handlers.chat import forward_note

    user = _Stub(sender_user=_Stub(full_name="Test User"))
    assert "Test User" in forward_note(user)
    channel = _Stub(sender_user=None, chat=_Stub(title="AI News"))
    assert "AI News" in forward_note(channel)
    hidden = _Stub(sender_user=None, chat=None, sender_user_name="Hidden Guy")
    assert "Hidden Guy" in forward_note(hidden)
    assert forward_note(None) == ""


def test_build_media_prompt_contract():
    from d_brain.bot.handlers.chat import build_media_prompt

    p = build_media_prompt(
        kind="document",
        rel_path="attachments/2026-06-10/img-120000.pdf",
        original_name="report.pdf",
        caption="квартальный отчёт",
        fwd="[переслано от: Тест]\n",
    )
    assert "attachments/2026-06-10/img-120000.pdf" in p
    assert "report.pdf" in p
    assert "квартальный отчёт" in p
    assert "Тест" in p
    # the brain must be told to actually open the file
    assert "Read" in p or "прочитай" in p.lower() or "посмотри" in p.lower()


def test_unsupported_content_reply_exists():
    from d_brain.bot.handlers.chat import UNSUPPORTED_REPLY

    assert "голос" in UNSUPPORTED_REPLY or "voice" in UNSUPPORTED_REPLY.lower()


# ── blind-review fixes: collisions, albums, attribution, escaping ──────────


def test_save_attachment_never_overwrites_same_second(tmp_path):
    from datetime import date, datetime

    from d_brain.services.storage import VaultStorage

    s = VaultStorage(tmp_path)
    ts = datetime(2026, 6, 10, 12, 0, 0)
    p1 = s.save_attachment(b"one", date(2026, 6, 10), ts, "jpg")
    p2 = s.save_attachment(b"two", date(2026, 6, 10), ts, "jpg")
    assert p1 != p2
    assert (tmp_path / "attachments/2026-06-10").glob("*")
    assert (tmp_path / p1).read_bytes() == b"one"
    assert (tmp_path / p2).read_bytes() == b"two"


def test_forward_note_anonymous_group_admin():
    from d_brain.bot.handlers.chat import forward_note

    origin = _Stub(sender_user=None, chat=None, sender_chat=_Stub(title="Work Chat"),
                   sender_user_name=None)
    assert "Work Chat" in forward_note(origin)


def test_extract_media_sanitizes_hostile_extension():
    from d_brain.bot.handlers.chat import extract_media

    doc = _Stub(file_id="d1", file_name="x.a/b")
    msg = _Stub(photo=None, document=doc, video=None, audio=None,
                animation=None, video_note=None)
    _, _, ext, _ = extract_media(msg)
    assert "/" not in ext and "\\" not in ext


def test_compact_not_a_control_command():
    """commands.router intercepts /compact earlier — keeping it in _CONTROL
    is dead code that lies about behavior."""
    from d_brain.bot.handlers.chat import classify_command

    assert classify_command("/compact") == "turn"


def test_album_items_flush_as_single_prompt(monkeypatch):
    from d_brain.bot.handlers import chat

    mgr = FakeManager(reply="ok")
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    monkeypatch.setattr(chat, "ALBUM_SETTLE", 0.01)
    bot = FakeBot()

    async def run():
        await chat.queue_album_item(
            bot, chat_id=10, user_id=1, group_id="g1",
            item={"kind": "photo", "rel_path": "attachments/a.jpg",
                  "caption": "подпись", "fwd": ""},
        )
        await chat.queue_album_item(
            bot, chat_id=10, user_id=1, group_id="g1",
            item={"kind": "photo", "rel_path": "attachments/b.jpg",
                  "caption": "", "fwd": ""},
        )
        await asyncio.sleep(0.1)

    asyncio.run(run())
    assert len(mgr.sent) == 1
    prompt = mgr.sent[0][1]
    assert "attachments/a.jpg" in prompt and "attachments/b.jpg" in prompt
    assert "подпись" in prompt


def test_text_during_maintenance_turn_gets_busy_reply(monkeypatch):
    """A message while the nightly pipeline / doctor holds the session must
    NOT be steered into the maintenance turn — the user gets a busy reply."""
    from d_brain.bot.handlers import chat

    class MaintenanceManager(SteerableManager):
        def is_steerable_turn(self) -> bool:
            return False

    mgr = MaintenanceManager(active=True)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="пиши короче"))
    assert mgr.steered == []  # nothing injected into the pipeline turn
    assert mgr.sent == []  # and no new turn forced past the lock
    assert bot.messages and "обслуживание" in bot.messages[0][1].lower()


# ── heavy attachments: pre-ask() busy guard + prompt contracts
# ───────────────────────────────────────────


class BusyManager(FakeManager):
    """A turn is already running. Nothing here may reach ask(): a 'busy'
    result from ask() is counted as a FAILURE by ask_health, and three of
    those in a row fire the delivery_guard 'channel broken' alert — the exact
    escalation that documents for 2026-08-31."""

    def is_turn_active(self) -> bool:
        return True


def test_busy_media_reply_names_the_saved_file_and_promises_no_callback():
    from d_brain.bot.handlers.chat import build_busy_media_reply

    text = build_busy_media_reply(["attachments/2026-09-04/img-022703.jpg"])
    assert "attachments/2026-09-04/img-022703.jpg" in text
    assert "занята" in text
    assert "Ничего не потерялось" in text
    # rule B3: never promise a callback — nothing re-sends an answer later
    assert "как освобожусь" not in text
    assert "отвечу" not in text.lower()


def test_busy_media_reply_lists_a_whole_album_in_one_message():
    from d_brain.bot.handlers.chat import build_busy_media_reply

    text = build_busy_media_reply(["attachments/d/a.jpg", "attachments/d/b.jpg"])
    assert "attachments/d/a.jpg" in text and "attachments/d/b.jpg" in text
    assert "(2)" in text


def test_media_guard_blocks_ask_and_replies_honestly(monkeypatch):
    from d_brain.bot.handlers import chat

    mgr = BusyManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    blocked = asyncio.run(
        chat.reject_media_if_busy(bot, 10, ["attachments/2026-09-04/img-1.jpg"])
    )

    assert blocked is True
    assert mgr.sent == []  # ask() never reached ⇒ ask_health untouched
    assert bot.messages and "img-1.jpg" in bot.messages[0][1]


def test_media_guard_passes_through_when_idle(monkeypatch):
    from d_brain.bot.handlers import chat

    mgr = FakeManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    blocked = asyncio.run(chat.reject_media_if_busy(bot, 10, ["attachments/a.jpg"]))

    assert blocked is False
    assert bot.messages == []
    # Contract: a False return means the caller now HOLDS the dispatch claim
    # and owes a _release_media_dispatch() in a finally.
    assert chat._media_claim_at is not None
    chat._release_media_dispatch()
    assert chat._media_claim_at is None


def test_stale_media_claim_self_heals_instead_of_wedging_the_path(monkeypatch):
    """A claim leaked by some path we did not anticipate must expire, not turn
    one bug into a standing outage on the whole media path."""
    from d_brain.bot.handlers import chat

    mgr = FakeManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    assert chat._try_claim_media_dispatch() is True
    assert chat._try_claim_media_dispatch() is False  # held, correctly refused

    # Age it past the TTL without waiting an hour.
    import time

    monkeypatch.setattr(
        chat, "_media_claim_at", time.monotonic() - chat.MEDIA_CLAIM_TTL - 1
    )
    blocked = asyncio.run(chat.reject_media_if_busy(bot, 10, ["attachments/a.jpg"]))

    assert blocked is False  # reclaimed, media path alive again
    assert bot.messages == []
    chat._release_media_dispatch()


def test_media_claim_ttl_outlives_the_longest_legitimate_turn():
    """The TTL is a leak backstop, not a turn budget — if it fired during a
    real hour-long turn it would re-open the very race it closes."""
    from d_brain.bot.handlers import chat
    from d_brain.services.claude_session import DEFAULT_TIMEOUT

    assert chat.MEDIA_CLAIM_TTL > DEFAULT_TIMEOUT


def test_album_flush_during_active_turn_sends_one_busy_reply(monkeypatch):
    """The 1.5s settle window is long enough for a turn to start — the guard
    is re-checked at flush, and the whole album gets ONE reply, not N."""
    from d_brain.bot.handlers import chat

    mgr = BusyManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    monkeypatch.setattr(chat, "ALBUM_SETTLE", 0.01)
    bot = FakeBot()

    async def run():
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            await chat.queue_album_item(
                bot, chat_id=10, user_id=1, group_id="g9",
                item={"kind": "photo", "rel_path": f"attachments/d/{name}",
                      "caption": "", "fwd": ""},
            )
        await asyncio.sleep(0.1)

    asyncio.run(run())

    assert mgr.sent == []  # no turn started
    assert len(bot.messages) == 1  # one reply for the whole album
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        assert name in bot.messages[0][1]


def test_media_prompt_with_downscaled_image_names_both_paths():
    from d_brain.bot.handlers.chat import build_media_prompt
    from d_brain.services.media_prep import MediaPrep

    prep = MediaPrep(
        model_rel_path="attachments/2026-09-04/derived/img-1-model.jpg",
        instruction=(
            "Читай уменьшенную копию: attachments/2026-09-04/derived/"
            "img-1-model.jpg. Оригинал attachments/2026-09-04/img-1.jpg НЕ "
            "открывай. В память ссылайся ТОЛЬКО на оригинал."
        ),
        meta="4284×4284 3.8 МБ → 1568×1568 400 КБ",
    )
    p = build_media_prompt(
        kind="document", rel_path="attachments/2026-09-04/img-1.jpg",
        original_name="IMG_0042.jpg", caption="", fwd="", prep=prep,
    )
    assert "attachments/2026-09-04/img-1.jpg" in p
    assert "attachments/2026-09-04/derived/img-1-model.jpg" in p
    assert "ТОЛЬКО на оригинал" in p
    assert "4284×4284" in p


def test_media_prompt_for_video_forbids_reading_the_file():
    from d_brain.bot.handlers.chat import build_media_prompt
    from d_brain.services.media_prep import MediaPrep, video_instruction

    p = build_media_prompt(
        kind="video", rel_path="attachments/2026-09-04/clip.mp4",
        original_name="clip.mp4", caption="смотри", fwd="",
        prep=MediaPrep(instruction=video_instruction("attachments/2026-09-04/clip.mp4")),
    )
    assert "НЕ читай" in p
    assert "обработать не могу" in p
    assert "смотри" in p


def test_media_prompt_for_big_pdf_points_at_the_sidecar(tmp_path):
    from d_brain.bot.handlers.chat import build_media_prompt
    from d_brain.services.media_prep import MediaPrep

    sidecar = "attachments/2026-09-04/derived/report-text.md"
    prep = MediaPrep(
        model_rel_path=sidecar,
        instruction=(
            f"Читай извлечённый текст: {sidecar}. Оригинал целиком не открывай."
        ),
        meta="36 стр., 4.1 МБ",
    )
    p = build_media_prompt(
        kind="document", rel_path="attachments/2026-09-04/report.pdf",
        original_name="report.pdf", caption="", fwd="", prep=prep,
    )
    assert sidecar in p
    assert "36 стр." in p


def test_album_prompt_uses_derivatives_and_per_file_instructions():
    from d_brain.bot.handlers.chat import build_album_prompt

    p = build_album_prompt([
        {"kind": "photo", "rel_path": "attachments/d/a.jpg", "caption": "",
         "fwd": "", "model_rel_path": "attachments/d/derived/a-model.jpg",
         "instruction": "Читай уменьшенную копию."},
        {"kind": "video", "rel_path": "attachments/d/c.mp4", "caption": "",
         "fwd": "", "model_rel_path": None,
         "instruction": "Видео обработать не могу. НЕ читай этот файл."},
    ])
    assert "attachments/d/derived/a-model.jpg" in p
    assert "НЕ читай этот файл" in p
    assert "ОДНОЙ записью" in p


def test_media_prompt_without_prep_keeps_todays_wording():
    """The fallback path (media_prep failed) must be byte-identical to the
    behavior that shipped before."""
    from d_brain.bot.handlers.chat import build_media_prompt
    from d_brain.services.media_prep import DEFAULT_INSTRUCTION

    p = build_media_prompt(
        kind="document", rel_path="attachments/d/x.bin", original_name=None,
        caption="", fwd="",
    )
    assert p.endswith(DEFAULT_INSTRUCTION)


# ── end-to-end media handler: file is saved BEFORE the guard bails out ─────


class _FakeFile:
    file_path = "photos/file_1.jpg"


class MediaBot(FakeBot):
    def __init__(self, payload=b"binary-bytes"):
        super().__init__()
        self.payload = payload

    async def get_file(self, file_id):
        return _FakeFile()

    async def download_file(self, path):
        import io

        return io.BytesIO(self.payload)


def _media_message(tmp_path, *, doc_name="IMG_0042.jpg", group_id=None):
    from datetime import datetime

    answers: list[str] = []

    class Msg(_Stub):
        async def answer(self, text, **kw):
            answers.append(text)

    msg = Msg(
        from_user=_Stub(id=7),
        chat=_Stub(id=10),
        message_id=555,
        date=datetime(2026, 9, 4, 12, 30, 0),
        caption="",
        forward_origin=None,
        photo=None,
        document=_Stub(file_id="d1", file_name=doc_name),
        video=None, audio=None, animation=None, video_note=None,
        media_group_id=group_id,
    )
    return msg, answers


def test_media_handler_saves_file_then_bails_out_when_busy(monkeypatch, tmp_path):
    """The librarian promise stays intact: the attachment and the daily entry
    are written BEFORE the guard returns, so nothing is lost — but ask() is
    never called, so the ask_health fail-streak never grows."""
    from d_brain.bot.handlers import chat

    mgr = BusyManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    monkeypatch.setattr(chat, "get_settings", lambda: _Stub(vault_path=tmp_path))
    bot = MediaBot()
    msg, _answers = _media_message(tmp_path)

    asyncio.run(chat.handle_chat_media(msg, bot))

    saved = list((tmp_path / "attachments/2026-09-04").glob("img-*.jpg"))
    assert saved and saved[0].read_bytes() == b"binary-bytes"
    daily = (tmp_path / "daily/2026-09-04.md").read_text(encoding="utf-8")
    assert "![[attachments/2026-09-04/" in daily
    assert mgr.sent == []  # ask() never reached
    assert bot.messages and "Ничего не потерялось" in bot.messages[0][1]
    # a busy turn must not have paid for a downscale either
    assert not (tmp_path / "attachments/2026-09-04/derived").exists()


def test_media_handler_downscales_and_dispatches_when_idle(monkeypatch, tmp_path):
    from PIL import Image

    from d_brain.bot.handlers import chat

    buf = __import__("io").BytesIO()
    Image.new("RGB", (4284, 4284), (10, 20, 30)).save(buf, format="JPEG")

    mgr = FakeManager(reply="ок")
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    monkeypatch.setattr(chat, "get_settings", lambda: _Stub(vault_path=tmp_path))
    bot = MediaBot(payload=buf.getvalue())
    msg, _answers = _media_message(tmp_path)

    asyncio.run(chat.handle_chat_media(msg, bot))

    assert len(mgr.sent) == 1
    prompt = mgr.sent[0][1]
    assert "/derived/" in prompt and "-model.jpg" in prompt
    derived = list((tmp_path / "attachments/2026-09-04/derived").glob("*-model.jpg"))
    assert derived
    with Image.open(derived[0]) as img:
        assert max(img.size) <= 1568
    # the dispatch claim is handed back, so the next file is not wedged out
    assert chat._media_claim_at is None


class LockingManager(FakeManager):
    """Mirrors the real timing: the pane lock lives deep inside ask(), so it is
    only taken once send_message is actually running. Everything upstream of
    that — i.e. the handler guard — still sees an IDLE pane."""

    def __init__(self):
        super().__init__(reply="ок")
        self.active = False
        self.in_flight = 0
        self.max_concurrent = 0

    def is_turn_active(self) -> bool:
        return self.active

    async def send_message(self, user_id: int, prompt: str) -> str:
        self.in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self.in_flight)
        self.active = True
        try:
            await asyncio.sleep(0.05)  # a turn takes time; that is the race
            self.sent.append((user_id, prompt))
            return self.reply
        finally:
            self.in_flight -= 1
            self.active = False


def test_two_documents_in_the_same_second_only_dispatch_one(monkeypatch, tmp_path):
    """A real incident's exact shape: two heavy documents arrive in the same
    second. main.py runs aiogram with handle_as_tasks=True, so each update is
    its own concurrent asyncio task — and is_turn_active() only reflects the
    pane lock taken far downstream, inside ask(). Without an in-process claim
    BOTH tasks see an idle pane, both pass the guard, and the second silently
    queues on the lock instead of getting an honest busy reply — which is what
    grew the ask_health fail-streak into a delivery_guard alert."""
    from d_brain.bot.handlers import chat

    mgr = LockingManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    monkeypatch.setattr(chat, "get_settings", lambda: _Stub(vault_path=tmp_path))
    bot = MediaBot()
    first, _a1 = _media_message(tmp_path, doc_name="IMG_022703.jpg")
    second, _a2 = _media_message(tmp_path, doc_name="IMG_022703_1.jpg")

    async def run():
        await asyncio.gather(
            chat.handle_chat_media(first, bot),
            chat.handle_chat_media(second, bot),
        )

    asyncio.run(run())

    # exactly one turn reached ask(), never two racing for the pane lock
    assert len(mgr.sent) == 1
    assert mgr.max_concurrent == 1
    # ...and the loser got an honest busy reply, not silence
    busy = [t for _cid, t in bot.messages if "Ничего не потерялось" in t]
    assert len(busy) == 1
    # librarian promise holds for BOTH files regardless
    assert len(list((tmp_path / "attachments/2026-09-04").glob("img-*.jpg"))) == 2
    assert chat._media_claim_at is None


# ── duty session routing ────────────────


class DutyManager(FakeManager):
    """Lock-free but pane-busy — the unattended-cascade shape — with the duty
    hooks the real ChatSessionManager exposes."""

    def __init__(self, *, main_busy: bool, duty_reply: str = "<b>из дежурной</b>"):
        super().__init__()
        self.main_busy = main_busy
        self.duty_reply = duty_reply
        self.duty_calls: list[tuple[int, str, str | None]] = []
        self.steerable = True
        self.steered: list[str] = []

    def is_turn_active(self) -> bool:
        return not self.steerable  # a maintenance turn holds the session

    def is_steerable_turn(self) -> bool:
        return self.steerable

    async def steer(self, text: str) -> None:
        self.steered.append(text)

    async def is_main_busy(self) -> bool:
        return self.main_busy

    async def answer_from_duty(self, user_id, prompt, *, busy_seconds=None,
                               fallback=None) -> str:
        self.duty_calls.append((user_id, prompt, fallback))
        return self.duty_reply


def test_busy_main_session_answers_from_duty_without_entering_ask(monkeypatch):
    """The owner's complaint: during an unattended cascade the
    ask-lock is free, so the message used to go into ask() and burn the
    whole busy-wait budget only to return a brush-off. The pane check now
    routes it to the duty session instead — ask() is never entered."""
    from d_brain.bot.handlers import chat

    mgr = DutyManager(main_busy=True)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="запиши мысль"))
    assert mgr.sent == []  # never reached the main session's ask()
    assert mgr.duty_calls == [(1, "запиши мысль", None)]
    assert bot.messages and "из дежурной" in bot.messages[0][1]


def test_idle_main_session_still_takes_the_normal_path(monkeypatch):
    """The gate must not divert anything when the pane is genuinely idle."""
    from d_brain.bot.handlers import chat

    mgr = DutyManager(main_busy=False)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="привет"))
    assert mgr.sent == [(1, "привет")]
    assert mgr.duty_calls == []


def test_maintenance_turn_is_answered_from_duty(monkeypatch):
    """A message during a non-steerable (maintenance) turn must still not be
    injected into that turn — but it now gets a real answer instead of the
    'повтори через несколько минут' brush-off."""
    from d_brain.bot.handlers import chat

    mgr = DutyManager(main_busy=False)
    mgr.steerable = False
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="пиши короче"))
    assert mgr.steered == []  # nothing contaminated the maintenance turn
    assert mgr.sent == []
    assert len(mgr.duty_calls) == 1
    # the pre-duty wording is handed down as the fallback, verbatim
    assert "обслуживание" in mgr.duty_calls[0][2]
    assert "из дежурной" in bot.messages[0][1]


def test_duty_path_shows_typing(monkeypatch):
    from d_brain.bot.handlers import chat

    class TypingBot(FakeBot):
        def __init__(self):
            super().__init__()
            self.typing = 0

        async def send_chat_action(self, chat_id, action):
            self.typing += 1

    mgr = DutyManager(main_busy=True)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = TypingBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="привет"))
    assert bot.typing >= 1


def test_duty_path_crash_still_delivers_the_old_message(monkeypatch):
    """A failure inside the duty path must leave the user with the pre-duty
    message, never with silence."""
    from d_brain.bot.handlers import chat

    class Broken(DutyManager):
        async def answer_from_duty(self, *a, **kw):
            raise RuntimeError("duty exploded")

    mgr = Broken(main_busy=True)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="привет"))
    assert bot.messages and "/stop" in bot.messages[0][1]


def test_a_manager_without_the_duty_probe_keeps_todays_path(monkeypatch):
    """Forward/backward compatibility: no is_main_busy ⇒ straight into the
    main session, exactly as before this feature."""
    from d_brain.bot.handlers import chat

    mgr = FakeManager()
    assert not hasattr(mgr, "is_main_busy")
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="привет"))
    assert mgr.sent == [(1, "привет")]


# ── the gate covers EVERY input path, not just text (blind review F6) ─────


def test_the_busy_gates_live_in_run_turn_not_in_dispatch_text():
    """Structural guard for F6. The gate was originally in _dispatch_text,
    which left voice — the system's PRIMARY channel — and media outside it:
    they call _process_and_reply directly. It now lives in _run_turn, behind
    the single _process_and_reply choke point every path funnels through; a
    future edit that moves either probe back to the text path would silently
    re-open the hole for three of the four paths, and every behavioral test
    below would still pass because they all enter through text.

    ``_held_by_maintenance`` is in the same list for the same reason: it
    replaced the ``is_steerable_turn`` branch that used to sit in
    _dispatch_text and therefore only ever protected typed text."""
    import inspect

    from d_brain.bot.handlers import chat

    turn = inspect.getsource(chat._run_turn)
    dispatch = inspect.getsource(chat._dispatch_text)
    for probe in ("_main_is_busy", "_held_by_maintenance"):
        assert probe in turn
        assert probe not in dispatch


def test_voice_path_gets_the_duty_answer_without_entering_ask(monkeypatch):
    """The voice-first case the whole feature exists for: before F6 a voice
    message arriving during an unattended cascade paid the full
    DEFAULT_BUSY_WAIT_BUDGET (up to 300s of silence) before the duty session
    answered."""
    from d_brain.bot.handlers import chat

    mgr = DutyManager(main_busy=True)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(
        chat._process_and_reply(bot, chat_id=10, user_id=1, prompt="[voice] мысль")
    )
    assert mgr.sent == []  # never reached the main session's ask()
    assert mgr.duty_calls == [(1, "[voice] мысль", None)]
    assert bot.messages and "из дежурной" in bot.messages[0][1]


class VoiceBot(MediaBot):
    """A bot that can hand back a voice file, like MediaBot does for docs."""


def _voice_message():
    from datetime import datetime

    return _Stub(
        from_user=_Stub(id=7),
        chat=_Stub(id=10),
        message_id=555,
        date=datetime(2026, 9, 4, 12, 30, 0),
        voice=_Stub(file_id="v1", duration=3),
        answer=None,
    )


def test_voice_handler_reaches_the_gate_end_to_end(monkeypatch, tmp_path):
    """Not just _process_and_reply in isolation: the REAL voice handler must
    funnel into it, so the gate it now carries actually applies to the
    voice-first channel."""
    from d_brain.bot.handlers import chat

    mgr = DutyManager(main_busy=True)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    monkeypatch.setattr(
        chat, "get_settings", lambda: _Stub(vault_path=tmp_path, deepgram_api_key="k")
    )

    class FakeTranscriber:
        def __init__(self, _key):
            pass

        async def transcribe(self, _payload):
            return "запиши мысль"

    monkeypatch.setattr(chat, "DeepgramTranscriber", FakeTranscriber)
    bot = VoiceBot()

    asyncio.run(chat.handle_chat_voice(_voice_message(), bot))

    assert mgr.sent == []  # ask() never entered
    assert mgr.duty_calls == [(7, "[voice] запиши мысль", None)]
    assert bot.messages and "из дежурной" in bot.messages[0][1]
    # the librarian safety net still ran first
    daily = (tmp_path / "daily/2026-09-04.md").read_text(encoding="utf-8")
    assert "запиши мысль" in daily


def test_media_prompt_path_gets_the_duty_answer_too(monkeypatch):
    """Single-file media reaches _process_and_reply after its own claim
    guard (which answers a different race), so it inherits the gate."""
    from d_brain.bot.handlers import chat

    mgr = DutyManager(main_busy=True)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(
        chat._process_and_reply(
            bot, chat_id=10, user_id=1, prompt="Пользователь прислал photo: a.jpg"
        )
    )
    assert mgr.sent == []
    assert mgr.duty_calls and "a.jpg" in mgr.duty_calls[0][1]
    assert chat._media_claim_at is None  # the claim path is untouched


def test_album_flush_honours_the_gate_and_releases_the_claim(monkeypatch):
    """An album must produce ONE duty reply for the whole group, and the
    dispatch claim must come back — F6 must not wedge the media path."""
    from d_brain.bot.handlers import chat

    mgr = DutyManager(main_busy=True)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    monkeypatch.setattr(chat, "ALBUM_SETTLE", 0.0)
    bot = FakeBot()
    chat._album_buf["g9"] = [
        {"kind": "photo", "rel_path": "a.jpg", "caption": "", "fwd": ""},
        {"kind": "photo", "rel_path": "b.jpg", "caption": "", "fwd": ""},
    ]
    asyncio.run(chat._flush_album(bot, 10, 1, "g9"))

    assert mgr.sent == []
    assert len(mgr.duty_calls) == 1  # ONE duty reply for the whole album
    assert chat._media_claim_at is None


def test_the_empty_reply_retry_does_not_re_probe_the_gate(monkeypatch):
    """The gate runs once, before the first attempt. The retry exists for an
    empty reply from an already-idle session; re-probing there would add a
    second ~3s pane check for a user already waiting on a second turn."""
    from d_brain.bot.handlers import chat

    class CountingGate(DutyManager):
        def __init__(self):
            super().__init__(main_busy=False)
            self.probes = 0
            self.reply = ""  # always empty ⇒ always retries

        async def is_main_busy(self) -> bool:
            self.probes += 1
            return False

    mgr = CountingGate()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._process_and_reply(bot, chat_id=10, user_id=1, prompt="x"))
    assert len(mgr.sent) == 2  # the retry happened
    assert mgr.probes == 1  # but the gate was consulted once


def test_typing_starts_before_the_busy_gate(monkeypatch):
    """Review round 3, R3: when the marker is fresh but the second pane
    probe finds the turn finished, the gate spends ~3s and returns False.
    The indicator must already be up by then, not start afterwards."""
    from d_brain.bot.handlers import chat

    order: list[str] = []

    class OrderBot(FakeBot):
        async def send_chat_action(self, chat_id, action):
            order.append("typing")

    class SlowGate(DutyManager):
        async def is_main_busy(self) -> bool:
            order.append("gate")
            return False

    mgr = SlowGate(main_busy=False)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    asyncio.run(chat._process_and_reply(OrderBot(), 10, 1, "привет"))
    assert order[0] == "typing"
    assert "gate" in order


def test_a_failing_typing_ping_never_costs_the_reply(monkeypatch):
    from d_brain.bot.handlers import chat

    class BrokenTyping(FakeBot):
        async def send_chat_action(self, chat_id, action):
            raise RuntimeError("telegram hiccup")

    mgr = DutyManager(main_busy=False)
    mgr.reply = "ответ"
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = BrokenTyping()
    asyncio.run(chat._process_and_reply(bot, 10, 1, "привет"))
    assert bot.messages and "ответ" in bot.messages[0][1]


def test_an_idle_session_pays_no_extra_delay_with_the_real_is_main_busy(
    monkeypatch, tmp_path
):
    """Integration (blind review F11): the handler against the REAL
    ChatSessionManager.is_main_busy, not a stub. With no watchdog marker on
    disk the gate must return immediately — no marker, no pane probes, no
    _MAIN_BUSY_CONFIRM_SECONDS sleep — and the message must take the
    ordinary path."""
    import time

    from d_brain.bot.handlers import chat
    from d_brain.services.chat_session import ChatSessionManager
    from d_brain.services.claude_session import AskResult

    class Probed:
        """A session whose pane probe would say "busy" if it were asked."""

        def __init__(self):
            self.probes = 0
            self.prompts: list[str] = []

        def ask(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return AskResult("ok", reply="ответ")

        def is_pane_turn_active(self) -> bool:
            self.probes += 1
            return True

        def is_turn_active(self) -> bool:
            return False

    session = Probed()
    mgr = ChatSessionManager(tmp_path, session=session, health_dir=tmp_path)
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    started = time.monotonic()
    asyncio.run(chat._dispatch_text(bot, chat_id=10, user_id=1, text="привет"))
    elapsed = time.monotonic() - started

    assert session.prompts == ["привет"]  # the normal path
    assert session.probes == 0  # not even probed: no marker to confirm
    assert elapsed < 1.0  # and certainly no 3s confirmation window
    assert bot.messages and "ответ" in bot.messages[0][1]
