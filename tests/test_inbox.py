"""Tests for the durable inbox (2026-09-22, step 2 of the reliability plan).

What is pinned down here is exactly what the feature promises, and nothing
about how the files happen to be named:

* a message is on disk BEFORE any handler runs;
* a restart with an accepted-but-unanswered message answers it;
* the same message is never answered twice — not on a re-feed, not after a
  crash in the receipt window, not when Telegram redelivers the batch;
* a message older than the threshold is NOT replayed (and is not deleted
  either);
* order inside a chat survives the replay;
* voice and files survive the disk round-trip, not just text;
* the Telegram offset hole is closed — and closing it did not break the
  offset by filtering the batch aiogram counts.
"""

import asyncio
import json
import os

import pytest
from aiogram import Dispatcher, Router
from aiogram.methods import GetUpdates, SendMessage
from aiogram.types import Update

from d_brain.config import Settings
from d_brain.services import inbox
from d_brain.services.inbox import Inbox, entry_id


class FakeBot:
    """Enough of a Bot for ``Update.model_validate(context={"bot": ...})``
    and for the dispatcher's identity check — aiogram never type-checks it."""

    id = 1

    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text))


class Clock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _box(tmp_path, clock=None) -> Inbox:
    return Inbox(tmp_path, clock_fn=clock or (lambda: 1000.0))


def _message(update_id: int, chat_id: int = 42, **payload) -> dict:
    body = {
        "message_id": update_id,
        "date": 1700000000,
        "chat": {"id": chat_id, "type": "private"},
        "from": {"id": 7, "is_bot": False, "first_name": "Owner"},
    }
    body.update(payload)
    return {"update_id": update_id, "message": body}


def _text(update_id: int, text: str = "привет", chat_id: int = 42) -> dict:
    return _message(update_id, chat_id=chat_id, text=text)


def _voice(update_id: int, chat_id: int = 42) -> dict:
    return _message(
        update_id,
        chat_id=chat_id,
        voice={"file_id": "voice-file", "file_unique_id": "vu", "duration": 7},
    )


def _document(update_id: int, chat_id: int = 42) -> dict:
    return _message(
        update_id,
        chat_id=chat_id,
        document={
            "file_id": "doc-file",
            "file_unique_id": "du",
            "file_name": "смета.pdf",
        },
        caption="посмотри смету",
    )


def _photo(update_id: int, chat_id: int = 42) -> dict:
    return _message(
        update_id,
        chat_id=chat_id,
        photo=[
            {"file_id": "small", "file_unique_id": "s", "width": 90, "height": 90},
            {"file_id": "big", "file_unique_id": "b", "width": 1280, "height": 1280},
        ],
    )


def _dispatcher(box: Inbox, seen: list) -> Dispatcher:
    """A dispatcher wired exactly the way ``run_bot`` wires the real one:
    the inbox gate OUTSIDE everything else."""
    dp = Dispatcher()
    dp.update.outer_middleware(inbox.handled_middleware(box))
    router = Router(name="probe")

    @router.message()
    async def handler(message):  # pragma: no cover - trivial
        seen.append(message)

    dp.include_router(router)
    return dp


# ── storage ──────────────────────────────────────────────────────────


def test_accept_persists_before_anything_is_handled(tmp_path):
    box = _box(tmp_path)
    entry = box.accept(_text(5, "первое"))

    on_disk = json.loads((tmp_path / "inbox" / f"{entry.id}.json").read_text())
    assert on_disk["update_id"] == 5
    assert on_disk["update"]["message"]["text"] == "первое"
    assert on_disk["accepted_at"] == 1000.0
    assert on_disk["chat_id"] == 42
    assert on_disk["kind"] == "message"


def test_entries_are_owner_only(tmp_path):
    box = _box(tmp_path)
    entry = box.accept(_text(1))
    mode = os.stat(tmp_path / "inbox" / f"{entry.id}.json").st_mode & 0o777
    assert mode == 0o600


def test_accept_is_idempotent_and_keeps_the_original_age(tmp_path):
    """A redelivered batch must not reset the clock that decides whether a
    message is still worth replaying — otherwise a bot in a restart loop
    keeps an ancient message eligible forever."""
    clock = Clock()
    box = _box(tmp_path, clock)
    box.accept(_text(1))
    clock.advance(500)
    again = box.accept(_text(1))

    assert again.accepted_at == 1000.0
    assert len(box.pending()) == 1


def test_pending_is_in_arrival_order_regardless_of_write_order(tmp_path):
    box = _box(tmp_path)
    for uid in (3, 1, 2):
        box.accept(_text(uid))
    assert [e.update_id for e in box.pending()] == [1, 2, 3]


def test_unreadable_entry_is_retired_not_skipped_forever(tmp_path):
    box = _box(tmp_path)
    box.accept(_text(1))
    (tmp_path / "inbox" / f"{entry_id(1)}.json").write_text("{not json")

    assert box.pending() == []
    assert (tmp_path / "inbox" / "stale" / f"{entry_id(1)}.json").exists()


def test_mark_handled_writes_the_receipt_before_dropping(tmp_path):
    box = _box(tmp_path)
    box.accept(_text(1))
    box.claim(entry_id(1))
    box.mark_handled(entry_id(1))

    assert box.pending() == []
    assert box.is_handled(entry_id(1))
    assert json.loads((tmp_path / "inbox" / "receipts.json").read_text())["ids"] == [
        entry_id(1)
    ]


def test_mark_handled_writes_a_receipt_even_without_an_entry(tmp_path):
    """The id with no entry on disk is the id whose accept() FAILED — a full
    or unwritable disk, i.e. exactly the state a bot restarts a lot in. No
    receipt there would mean a redelivered batch is answered twice, which is
    the one guarantee this module makes (blind review 6)."""
    box = _box(tmp_path)
    box.mark_handled(entry_id(99))
    assert box.is_handled(entry_id(99))


def test_receipts_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(inbox, "RECEIPTS_KEPT", 3)
    box = _box(tmp_path)
    for uid in range(1, 6):
        box.accept(_text(uid))
        box.mark_handled(entry_id(uid))
    ids = json.loads((tmp_path / "inbox" / "receipts.json").read_text())["ids"]
    assert ids == [entry_id(3), entry_id(4), entry_id(5)]


def test_claim_refuses_an_id_already_in_flight(tmp_path):
    box = _box(tmp_path)
    box.accept(_text(1))
    assert box.claim(entry_id(1)) is True
    assert box.claim(entry_id(1)) is False


def test_replayable_splits_by_age(tmp_path):
    clock = Clock()
    box = _box(tmp_path, clock)
    box.accept(_text(1, "старое"))
    clock.advance(7200)
    box.accept(_text(2, "свежее"))

    fresh, old = box.replayable(max_age=3600)
    assert [e.update_id for e in fresh] == [2]
    assert [e.update_id for e in old] == [1]


def test_a_zero_threshold_retires_everything(tmp_path):
    box = _box(tmp_path)
    box.accept(_text(1))
    fresh, old = box.replayable(max_age=0)
    assert fresh == []
    assert [e.update_id for e in old] == [1]


def test_stats_report_what_is_waiting(tmp_path):
    box = _box(tmp_path)
    box.accept(_text(1))
    box.accept(_text(2))
    stats = box.stats(now=1090.0)
    assert stats.pending == 2
    assert stats.oldest_age == 90.0
    assert stats.stale == 0


def test_stats_do_not_retire_what_they_cannot_read(tmp_path):
    box = _box(tmp_path)
    box.accept(_text(1))
    (tmp_path / "inbox" / f"{entry_id(1)}.json").write_text("{broken")
    stats = box.stats(now=1000.0)
    assert stats.pending == 1
    assert (tmp_path / "inbox" / f"{entry_id(1)}.json").exists()


def test_retired_entries_are_counted_and_kept(tmp_path):
    box = _box(tmp_path)
    entry = box.accept(_text(1))
    box.retire(entry, reason="too old")
    saved = json.loads((tmp_path / "inbox" / "stale" / f"{entry.id}.json").read_text())
    assert saved["reason"] == "too old"
    assert saved["update"]["message"]["text"] == "привет"
    assert box.stats().pending == 0
    assert box.stats().stale == 1


def test_describe_reads_a_callback_query_chat():
    kind, chat_id = inbox.describe(
        {
            "update_id": 1,
            "callback_query": {
                "id": "c",
                "from": {"id": 7, "is_bot": False, "first_name": "M"},
                "chat_instance": "x",
                "message": {
                    "message_id": 1,
                    "date": 1700000000,
                    "chat": {"id": 42, "type": "private"},
                },
            },
        }
    )
    assert (kind, chat_id) == ("callback_query", 42)


# ── acceptance: the single point, and the offset it must not break ───


def test_the_whole_batch_is_on_disk_before_aiogram_continues(tmp_path):
    box = _box(tmp_path)
    updates = [Update.model_validate(_text(uid)) for uid in (1, 2, 3)]
    seen_on_disk: list[int] = []

    async def make_request(bot, method):
        return updates

    async def scenario():
        middleware = inbox.accept_middleware(box)
        out = await middleware(make_request, FakeBot(), GetUpdates())
        seen_on_disk.extend(e.update_id for e in box.pending())
        return out

    out = asyncio.run(scenario())
    assert seen_on_disk == [1, 2, 3]
    # ...and the batch aiogram counts the offset from is handed back whole.
    assert out == updates


def test_duplicates_are_not_filtered_out_of_the_batch(tmp_path):
    """The offset trap: ``_listen_updates`` advances ``get_updates.offset``
    from the updates it is HANDED. Filter a fully-duplicate batch down to
    [] and the offset never moves, so Telegram redelivers the same batch
    forever, at polling speed. Acceptance records; it never filters."""
    box = _box(tmp_path)
    updates = [Update.model_validate(_text(uid)) for uid in (1, 2)]

    async def make_request(bot, method):
        return updates

    async def scenario():
        middleware = inbox.accept_middleware(box)
        bot = FakeBot()
        first = await middleware(make_request, bot, GetUpdates())
        second = await middleware(make_request, bot, GetUpdates())
        return first, second

    first, second = asyncio.run(scenario())
    assert len(first) == 2
    assert len(second) == 2
    assert len(box.pending()) == 2  # recorded once, not twice


def test_other_api_calls_are_left_alone(tmp_path):
    box = _box(tmp_path)

    async def make_request(bot, method):
        return "ok"

    async def scenario():
        middleware = inbox.accept_middleware(box)
        return await middleware(
            make_request, FakeBot(), SendMessage(chat_id=1, text="hi")
        )

    assert asyncio.run(scenario()) == "ok"
    assert box.pending() == []


def test_a_queue_that_cannot_be_written_still_lets_the_message_through(tmp_path):
    """Durability is an upgrade over answering, never a precondition for it
    (the outbox's blind review, B2)."""
    (tmp_path / "inbox").write_text("i am a file, not a directory")
    box = _box(tmp_path)
    updates = [Update.model_validate(_text(1))]

    async def make_request(bot, method):
        return updates

    async def scenario():
        middleware = inbox.accept_middleware(box)
        return await middleware(make_request, FakeBot(), GetUpdates())

    assert asyncio.run(scenario()) == updates


# ── the duplicate gate ───────────────────────────────────────────────


def test_a_fresh_update_reaches_the_handler_and_is_signed_off(tmp_path):
    box = _box(tmp_path)
    seen: list = []
    dp = _dispatcher(box, seen)
    raw = _text(1, "работает")
    box.accept(raw)

    asyncio.run(dp.feed_raw_update(FakeBot(), raw))

    assert [m.text for m in seen] == ["работает"]
    assert box.pending() == []
    assert box.is_handled(entry_id(1))


def test_the_same_message_is_never_answered_twice(tmp_path):
    box = _box(tmp_path)
    seen: list = []
    dp = _dispatcher(box, seen)
    raw = _text(1, "только один раз")
    box.accept(raw)

    async def scenario():
        bot = FakeBot()
        await dp.feed_raw_update(bot, raw)
        # Telegram redelivering the batch it never got a confirmed offset
        # for — the classic duplicate.
        await dp.feed_raw_update(bot, raw)

    asyncio.run(scenario())
    assert len(seen) == 1


def test_a_redelivered_answered_update_leaves_no_orphan(tmp_path):
    """The feature's own headline path, and it used to leave a permanent
    phantom (blind review 1): batch on disk → crash before the offset is
    confirmed → boot → replay answers it (receipt written, entry dropped) →
    polling starts → Telegram redelivers the still-unconfirmed batch.

    accept() must recognise the receipt and write nothing back. Otherwise
    the entry is re-created, the gate refuses to claim it, nothing ever
    removes it — /work reports a message that was answered, and once the
    receipt ring ages that id out the entry IS replayed: a second answer.
    """
    box = _box(tmp_path)
    seen: list = []
    dp = _dispatcher(box, seen)
    raw = _text(1, "отвечено один раз")

    async def scenario():
        bot = FakeBot()
        # Live path: accepted, handled, signed off.
        box.accept(raw)
        await dp.feed_raw_update(bot, raw)
        # Telegram redelivers the batch the dead process never confirmed.
        box.accept(raw)
        await dp.feed_raw_update(bot, raw)

    asyncio.run(scenario())

    assert len(seen) == 1
    assert box.pending() == []  # no orphan
    assert box.stats().pending == 0  # ...and /work says so

    # ...and it stays true after the receipt ring has rotated that id out.
    restarted = _box(tmp_path)
    again: list = []
    dp2 = _dispatcher(restarted, again)
    asyncio.run(inbox.replay(FakeBot(), dp2, restarted, max_age=3600))
    assert again == []


def test_a_handler_that_raised_still_counts_as_handled(tmp_path):
    """Otherwise a message that blows a handler up is replayed on every
    single boot for the next hour."""
    box = _box(tmp_path)
    dp = Dispatcher()
    dp.update.outer_middleware(inbox.handled_middleware(box))
    router = Router(name="angry")
    calls: list[int] = []

    @router.message()
    async def handler(message):
        calls.append(message.message_id)
        raise RuntimeError("boom")

    dp.include_router(router)
    raw = _text(1)
    box.accept(raw)

    # feed_raw_update re-raises after logging (aiogram 3.24); the live
    # polling path swallows it in _process_update and replay() catches it —
    # either way the sign-off below has already happened in the `finally`.
    with pytest.raises(RuntimeError):
        asyncio.run(dp.feed_raw_update(FakeBot(), raw))
    assert calls == [1]
    assert box.is_handled(entry_id(1))
    assert box.pending() == []


def test_a_crash_in_the_receipt_window_does_not_answer_twice(tmp_path):
    """Receipt written, entry not yet removed, process dies. The next boot
    must DROP the leftover, not replay it."""
    box = _box(tmp_path)
    raw = _text(1)
    box.accept(raw)
    box._add_receipt(entry_id(1))  # noqa: SLF001 — simulating the crash window

    restarted = _box(tmp_path)
    seen: list = []
    dp = _dispatcher(restarted, seen)
    asyncio.run(inbox.replay(FakeBot(), dp, restarted, max_age=3600))

    assert seen == []
    assert restarted.pending() == []


# ── replay ───────────────────────────────────────────────────────────


def test_a_restart_answers_what_the_dead_process_accepted(tmp_path):
    clock = Clock()
    box = _box(tmp_path, clock)
    box.accept(_text(1, "не потеряй меня"))
    # ...process dies here: accepted, never handled.

    restarted = _box(tmp_path, clock)
    seen: list = []
    dp = _dispatcher(restarted, seen)
    replayed = asyncio.run(inbox.replay(FakeBot(), dp, restarted, max_age=3600))

    assert replayed == 1
    assert [m.text for m in seen] == ["не потеряй меня"]
    assert restarted.pending() == []


def test_replay_keeps_the_order_inside_a_chat(tmp_path):
    box = _box(tmp_path)
    for uid, text in ((1, "раз"), (2, "два"), (3, "три")):
        box.accept(_text(uid, text))

    restarted = _box(tmp_path)
    seen: list = []
    dp = _dispatcher(restarted, seen)
    asyncio.run(inbox.replay(FakeBot(), dp, restarted, max_age=3600))

    assert [m.text for m in seen] == ["раз", "два", "три"]


def test_a_message_older_than_the_threshold_is_not_replayed(tmp_path):
    clock = Clock()
    box = _box(tmp_path, clock)
    box.accept(_text(1, "вчерашнее"))
    clock.advance(3601)
    box.accept(_text(2, "сегодняшнее"))

    restarted = Inbox(tmp_path, clock_fn=clock)
    seen: list = []
    dp = _dispatcher(restarted, seen)
    asyncio.run(inbox.replay(FakeBot(), dp, restarted, max_age=3600))

    assert [m.text for m in seen] == ["сегодняшнее"]
    # Not replayed, but not lost either.
    stale = json.loads(
        (tmp_path / "inbox" / "stale" / f"{entry_id(1)}.json").read_text()
    )
    assert stale["update"]["message"]["text"] == "вчерашнее"
    assert "older than" in stale["reason"]


def test_the_threshold_comes_from_settings(tmp_path):
    s = Settings(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path / "v",
        runtime_dir=tmp_path / "rt",
        _env_file=None,
    )
    assert s.inbox_replay_max_age == 3600.0
    assert (
        Settings(
            telegram_bot_token="t",
            deepgram_api_key="d",
            vault_path=tmp_path / "v",
            runtime_dir=tmp_path / "rt",
            inbox_replay_max_age=120.0,
            _env_file=None,
        ).inbox_replay_max_age
        == 120.0
    )


def test_the_threshold_can_be_set_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DBRAIN_INBOX_REPLAY_MAX_AGE", "90")
    s = Settings(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path / "v",
        runtime_dir=tmp_path / "rt",
        _env_file=None,
    )
    assert s.inbox_replay_max_age == 90.0


@pytest.mark.parametrize(
    "build, check",
    [
        (_voice, lambda m: m.voice.file_id == "voice-file" and m.voice.duration == 7),
        (
            _document,
            lambda m: m.document.file_name == "смета.pdf"
            and m.caption == "посмотри смету",
        ),
        (_photo, lambda m: m.photo[-1].file_id == "big"),
    ],
    ids=["voice", "document", "photo"],
)
def test_voice_and_files_survive_the_round_trip(tmp_path, build, check):
    """The primary input channel is voice, and files are the second. A
    replay that only rehydrated text would be worse than no replay."""
    box = _box(tmp_path)
    box.accept(build(1))

    restarted = _box(tmp_path)
    seen: list = []
    dp = _dispatcher(restarted, seen)
    asyncio.run(inbox.replay(FakeBot(), dp, restarted, max_age=3600))

    assert len(seen) == 1
    assert check(seen[0])


def test_replay_of_a_mixed_backlog_keeps_every_kind(tmp_path):
    box = _box(tmp_path)
    box.accept(_text(1, "сначала текст"))
    box.accept(_voice(2))
    box.accept(_document(3))

    restarted = _box(tmp_path)
    seen: list = []
    dp = _dispatcher(restarted, seen)
    assert asyncio.run(inbox.replay(FakeBot(), dp, restarted, max_age=3600)) == 3
    assert seen[0].text == "сначала текст"
    assert seen[1].voice.file_id == "voice-file"
    assert seen[2].document.file_id == "doc-file"


def test_one_broken_entry_does_not_strand_the_rest(tmp_path):
    box = _box(tmp_path)
    box.accept({"update_id": 1, "message": {"totally": "wrong"}})
    box.accept(_text(2, "всё равно отвечу"))

    restarted = _box(tmp_path)
    seen: list = []
    dp = _dispatcher(restarted, seen)
    asyncio.run(inbox.replay(FakeBot(), dp, restarted, max_age=3600))

    assert [m.text for m in seen] == ["всё равно отвечу"]
    assert restarted.pending() == []


def test_a_handler_blowing_up_mid_replay_does_not_strand_the_backlog(tmp_path):
    box = _box(tmp_path)
    box.accept(_text(1, "ядовитое"))
    box.accept(_text(2, "нормальное"))

    restarted = _box(tmp_path)
    seen: list = []
    dp = Dispatcher()
    dp.update.outer_middleware(inbox.handled_middleware(restarted))
    router = Router(name="flaky")

    @router.message()
    async def handler(message):
        if message.text == "ядовитое":
            raise RuntimeError("boom")
        seen.append(message.text)

    dp.include_router(router)
    asyncio.run(inbox.replay(FakeBot(), dp, restarted, max_age=3600))

    assert seen == ["нормальное"]
    # ...and the poison message is signed off, not queued up for every
    # future boot.
    assert restarted.pending() == []


def test_a_message_that_cost_one_boot_is_not_fed_to_the_next(tmp_path):
    """Blind review 3. The message that wedged the engine is exactly the
    message the next boot would hand to the same engine — and while that
    turn stalls, polling has not started, so /work, /new and /stop are all
    unreachable. One attempt, recorded BEFORE the work, then retire."""
    box = _box(tmp_path)
    box.accept(_text(1, "ядовитое"))

    # First boot: fed once, and the attempt is on disk before the turn.
    first = _box(tmp_path)
    attempts_seen: list[int] = []
    dp = Dispatcher()
    dp.update.outer_middleware(inbox.handled_middleware(first))
    router = Router(name="wedge")

    @router.message()
    async def handler(message):
        attempts_seen.append(first.read(entry_id(1)).attempts)
        raise asyncio.CancelledError  # the budget pulls the plug mid-turn

    dp.include_router(router)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(inbox.replay(FakeBot(), dp, first, max_age=3600))

    assert attempts_seen == [1]
    # Cancelled ≠ handled: the message is still owed an answer.
    assert [e.update_id for e in first.pending()] == [1]

    # Second boot: retired instead of wedging the bot again.
    second = _box(tmp_path)
    seen: list = []
    dp2 = _dispatcher(second, seen)
    asyncio.run(inbox.replay(FakeBot(), dp2, second, max_age=3600))

    assert seen == []
    assert second.pending() == []
    stale = json.loads(
        (tmp_path / "inbox" / "stale" / f"{entry_id(1)}.json").read_text()
    )
    assert "replay attempt" in stale["reason"]


def test_a_cancelled_turn_is_not_signed_off(tmp_path):
    """aiogram's shutdown does not run this `finally` today, but a future
    one that awaits its handler tasks would — and signing off a turn nobody
    finished deletes the entry for a message still owed a reply (blind
    review 10)."""
    box = _box(tmp_path)
    dp = Dispatcher()
    dp.update.outer_middleware(inbox.handled_middleware(box))
    router = Router(name="cancelled")

    @router.message()
    async def handler(message):
        raise asyncio.CancelledError

    dp.include_router(router)
    raw = _text(1)
    box.accept(raw)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(dp.feed_raw_update(FakeBot(), raw))

    assert [e.update_id for e in box.pending()] == [1]
    assert not box.is_handled(entry_id(1))
    # ...and the claim is back, so the next process can pick it up.
    assert box.claim(entry_id(1)) is True


def test_replay_does_not_count_a_leftover_it_only_dropped(tmp_path):
    box = _box(tmp_path)
    box.accept(_text(1))
    box._add_receipt(entry_id(1))  # noqa: SLF001 — the crash window again

    restarted = _box(tmp_path)
    seen: list = []
    dp = _dispatcher(restarted, seen)
    assert asyncio.run(inbox.replay(FakeBot(), dp, restarted, max_age=3600)) == 0
    assert seen == []


def test_replay_with_an_empty_queue_is_a_no_op(tmp_path):
    box = _box(tmp_path)
    seen: list = []
    dp = _dispatcher(box, seen)
    assert asyncio.run(inbox.replay(FakeBot(), dp, box, max_age=3600)) == 0
    assert seen == []


# ── visibility ───────────────────────────────────────────────────────


def _settings(tmp_path, **over):
    kwargs = dict(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path / "vault",
        runtime_dir=tmp_path / "rt",
        # The user every helper update above comes from: run_bot installs the
        # real auth middleware, which drops an update from anyone else.
        allowed_user_ids=[7],
        cron_enabled=False,
        _env_file=None,
    )
    kwargs.update(over)
    s = Settings(**kwargs)
    s.runtime_dir.mkdir(parents=True, exist_ok=True)
    return s


def test_work_does_not_count_its_own_message(tmp_path):
    """/work is itself an update, and it builds this report from INSIDE its
    own claim window. Counting in-flight entries made every single report on
    a completely idle bot announce one stuck message (blind review 2), which
    is a permanent false alarm in the one block meant to raise a real one.

    So drive it the way production does: through the gate, with the /work
    update accepted and claimed."""
    from d_brain.bot.handlers import work

    s = _settings(tmp_path)
    box = inbox.configure(s.runtime_dir)
    reports: list[str] = []

    dp = Dispatcher()
    dp.update.outer_middleware(inbox.handled_middleware(box))
    router = Router(name="work-probe")

    @router.message()
    async def handler(message):
        reports.append(work.build_work_report(s, now=1000.0))

    dp.include_router(router)
    raw = _text(1, "/work")
    box.accept(raw)
    try:
        asyncio.run(dp.feed_raw_update(FakeBot(), raw))
    finally:
        inbox.reset()

    assert len(reports) == 1
    assert "Приём сообщений" in reports[0]
    assert "принято, без ответа: нет" in reports[0]


def test_work_reports_an_empty_inbox(tmp_path):
    from d_brain.bot.handlers import work

    s = _settings(tmp_path)
    text = work.build_work_report(s, now=1000.0)
    assert "Приём сообщений" in text
    assert "принято, без ответа: нет" in text


def test_work_reports_a_waiting_message(tmp_path):
    from d_brain.bot.handlers import work

    s = _settings(tmp_path)
    box = Inbox(s.runtime_dir, clock_fn=lambda: 1000.0)
    box.accept(_text(1))
    box.accept(_voice(2))

    text = work.build_work_report(s, now=1120.0)
    assert "принято, без ответа: 2" in text
    assert "самому старому 2 мин" in text


def test_work_mentions_retired_messages_only_when_there_are_any(tmp_path):
    from d_brain.bot.handlers import work

    s = _settings(tmp_path)
    box = Inbox(s.runtime_dir, clock_fn=lambda: 1000.0)
    assert "не доигрывалось" not in work.build_work_report(s, now=1000.0)

    box.retire(box.accept(_text(1)), reason="older than 3600s at startup")
    assert "не доигрывалось (слишком старое): 1" in work.build_work_report(
        s, now=1000.0
    )


# ── wiring ───────────────────────────────────────────────────────────


class _FakeSession:
    def __init__(self) -> None:
        self.middlewares: list = []

    def middleware(self, handler):
        self.middlewares.append(handler)
        return handler

    async def close(self):
        pass


class _IdleSession:
    @staticmethod
    def ensure_session():
        pass


def _idle_session(settings):
    return _IdleSession()


class _BootBot(FakeBot):
    def __init__(self) -> None:
        super().__init__()
        self.session = _FakeSession()

    async def set_my_commands(self, commands):
        pass


def test_run_bot_accepts_before_it_fetches_and_replays_before_it_polls(
    tmp_path, monkeypatch
):
    """The three lines that make the whole thing real live in ``run_bot``,
    and their ORDER is the contract: acceptance installed before anything
    can fetch, the gate outside everything, and the backlog answered before
    polling lets new traffic in.

    Driven through the real ``run_bot`` rather than by reading its source —
    a source check would pass with the replay sitting inside ``if False``.
    """
    from d_brain.bot import main as bot_main

    s = _settings(tmp_path)
    # A message the previous process accepted and never answered.
    Inbox(s.runtime_dir).accept(_text(1, "остался с прошлого раза"))

    order: list[str] = []
    seen: list = []
    bot = _BootBot()

    dp = Dispatcher()
    router = Router(name="boot-probe")

    @router.message()
    async def handler(message):
        order.append("handled")
        seen.append(message.text)

    dp.include_router(router)

    async def fake_start_polling(*args, **kwargs):
        order.append("polling")

    monkeypatch.setattr(dp, "start_polling", fake_start_polling)
    monkeypatch.setattr(bot_main, "create_bot", lambda settings: bot)
    monkeypatch.setattr(bot_main, "create_dispatcher", lambda: dp)
    monkeypatch.setattr(bot_main, "get_session", _idle_session)

    try:
        asyncio.run(bot_main.run_bot(s))
    finally:
        inbox.reset()

    # Acceptance is installed on the SESSION — i.e. around getUpdates, before
    # aiogram can confirm an offset — not as a dispatcher middleware.
    assert len(bot.session.middlewares) == 1
    # ...and the backlog was answered before the first poll.
    assert order == ["handled", "polling"]
    assert seen == ["остался с прошлого раза"]
    assert Inbox(s.runtime_dir).pending() == []


def test_run_bot_survives_a_backlog_that_never_finishes(tmp_path, monkeypatch):
    """A replayed turn can ride chat_turn_timeout (25 minutes by default),
    and every second of it is a second with no /work and no /stop. The
    budget must hand the channel back (blind review 3)."""
    from d_brain.bot import main as bot_main

    s = _settings(tmp_path)
    Inbox(s.runtime_dir).accept(_text(1, "зависнет"))

    order: list[str] = []
    bot = _BootBot()
    dp = Dispatcher()
    router = Router(name="hangs")

    @router.message()
    async def handler(message):
        order.append("started")
        await asyncio.sleep(3600)

    dp.include_router(router)

    async def fake_start_polling(*args, **kwargs):
        order.append("polling")

    monkeypatch.setattr(dp, "start_polling", fake_start_polling)
    monkeypatch.setattr(bot_main, "create_bot", lambda settings: bot)
    monkeypatch.setattr(bot_main, "create_dispatcher", lambda: dp)
    monkeypatch.setattr(bot_main, "get_session", _idle_session)
    monkeypatch.setattr(inbox, "REPLAY_BUDGET", 0.05)

    try:
        asyncio.run(bot_main.run_bot(s))
    finally:
        inbox.reset()

    assert order == ["started", "polling"]
