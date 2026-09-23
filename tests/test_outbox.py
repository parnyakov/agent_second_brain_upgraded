"""Tests for the durable reply outbox.

What is pinned down here is exactly what the feature promises, and nothing
about how the files happen to be named:

* a failed send does not lose the reply, and the retry delivers it;
* a restart with a non-empty queue delivers what the dead process left;
* a receipt stops a duplicate send after a crash in the removal window;
* attempts are capped, and past the cap the reply lands in the dead queue
  WITH a reason rather than disappearing;
* messages inside one chat keep their order even when the first one fails,
  and a stuck chat does not hold up a different one.
"""

import asyncio
import json

import pytest
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)

from d_brain.services import outbox
from d_brain.services.outbox import Outbox


def _bad_request(message: str = "can't parse entities") -> TelegramBadRequest:
    return TelegramBadRequest(method=None, message=message)


def _forbidden(message: str = "bot was blocked by the user") -> TelegramForbiddenError:
    return TelegramForbiddenError(method=None, message=message)


class FakeBot:
    """Records sends; ``fail_texts`` makes chosen texts blow up with
    ``error`` (a transient network failure by default)."""

    def __init__(
        self,
        fail_texts: set[str] | None = None,
        error: Exception | None = None,
    ):
        self.sent: list[tuple[int, str]] = []
        self.fail_texts = fail_texts or set()
        self.error = error or RuntimeError("telegram is down")
        self.attempted: list[tuple[int, str]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.attempted.append((chat_id, text))
        if text in self.fail_texts:
            raise self.error
        self.sent.append((chat_id, text))


class Clock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _box(tmp_path, clock=None, **kw) -> Outbox:
    return Outbox(tmp_path, clock_fn=clock or (lambda: 1000.0), **kw)


# ── storage ──────────────────────────────────────────────────────────


def test_enqueue_persists_before_anything_is_sent(tmp_path):
    box = _box(tmp_path)
    item = box.enqueue(42, "the answer")

    on_disk = json.loads((box.dir / f"{item.id}.json").read_text())
    assert on_disk["chat_id"] == 42
    assert on_disk["text"] == "the answer"
    assert [i.text for i in box.pending()] == ["the answer"]


def test_pending_is_in_arrival_order(tmp_path):
    box = _box(tmp_path)
    for n in range(5):
        box.enqueue(1, f"m{n}")
    assert [i.text for i in box.pending()] == ["m0", "m1", "m2", "m3", "m4"]


def test_queue_files_are_owner_only(tmp_path):
    box = _box(tmp_path)
    item = box.enqueue(1, "private")
    mode = (box.dir / f"{item.id}.json").stat().st_mode & 0o777
    assert mode == 0o600


def test_unreadable_entry_is_quarantined_not_skipped_forever(tmp_path):
    box = _box(tmp_path)
    (box.dir).mkdir(parents=True, exist_ok=True)
    (box.dir / "0000000000000000001.json").write_text("{not json")

    assert box.pending() == []
    dead = list(box.dead_dir.glob("*.json"))
    assert len(dead) == 1
    assert "raw" in json.loads(dead[0].read_text())


# ── failure, retry, backoff ──────────────────────────────────────────


def test_backoff_grows_and_is_capped():
    ladder = [outbox.backoff_seconds(n) for n in range(1, 12)]
    assert ladder[0] == outbox.BASE_BACKOFF
    assert ladder == sorted(ladder)
    assert max(ladder) == outbox.MAX_BACKOFF


def test_failed_send_keeps_the_reply_and_the_retry_delivers_it(tmp_path):
    clock = Clock()
    box = _box(tmp_path, clock)
    bot = FakeBot(fail_texts={"hello"})
    box.enqueue(7, "hello")

    asyncio.run(outbox.drain(bot, box))

    assert bot.sent == []  # nothing got through
    still = box.pending()
    assert len(still) == 1  # and nothing was lost
    assert still[0].attempts == 1
    assert still[0].next_attempt > clock.now
    assert "telegram is down" in still[0].last_error

    # Telegram comes back; the backoff expires.
    bot.fail_texts.clear()
    clock.advance(outbox.MAX_BACKOFF + 1)
    asyncio.run(outbox.drain(bot, box))

    assert bot.sent == [(7, "hello")]
    assert box.pending() == []


def test_backoff_is_respected_before_it_expires(tmp_path):
    clock = Clock()
    box = _box(tmp_path, clock)
    bot = FakeBot(fail_texts={"hello"})
    box.enqueue(7, "hello")
    asyncio.run(outbox.drain(bot, box))

    bot.fail_texts.clear()
    asyncio.run(outbox.drain(bot, box))  # same instant — still backing off
    assert bot.sent == []
    assert box.pending()[0].attempts == 1


def test_a_non_markup_failure_is_attempted_exactly_once(tmp_path):
    """Blind review B1. The classic Telegram failure is "the server took the
    message, the client saw a timeout". The HTML→plain fallback used to fire
    on ANY exception, so every retry in the ladder delivered a second copy —
    up to twenty real messages for one reply. Only a 400 gets the fallback."""
    box = _box(tmp_path)
    bot = FakeBot(fail_texts={"hi"}, error=TimeoutError("read timeout"))
    box.enqueue(1, "hi")

    asyncio.run(outbox.drain(bot, box))

    assert bot.attempted == [(1, "hi")]  # ONE request, not two
    assert box.pending()[0].attempts == 1


def test_a_markup_failure_still_gets_the_plain_text_retry(tmp_path):
    box = _box(tmp_path)
    bot = FakeBot(fail_texts={"<b>broken"}, error=_bad_request())
    box.enqueue(1, "<b>broken")

    asyncio.run(outbox.drain(bot, box))

    assert bot.attempted == [(1, "<b>broken"), (1, "<b>broken")]


def test_telegram_retry_after_beats_our_own_ladder(tmp_path):
    """A 429 says how long to wait; hammering inside a flood-wait extends
    it. The ladder is a floor, never a ceiling."""
    clock = Clock()
    box = _box(tmp_path, clock)
    bot = FakeBot(
        fail_texts={"hi"},
        error=TelegramRetryAfter(method=None, message="flood", retry_after=120),
    )
    box.enqueue(1, "hi")

    asyncio.run(outbox.drain(bot, box))

    assert box.pending()[0].next_attempt >= clock.now + 120
    assert outbox.retry_delay(None, 1) == outbox.BASE_BACKOFF  # no hint → ladder


def test_a_refused_message_is_buried_at_once_and_frees_the_chat(tmp_path):
    """Blind review S2: a 403 (bot blocked, dead chat id) will never
    succeed. Retrying it ten times held every later reply to that chat
    behind it — on a one-user bot, ~18 minutes of total silence."""
    clock = Clock()
    box = _box(tmp_path, clock)
    bot = FakeBot(fail_texts={"doomed"}, error=_forbidden())
    box.enqueue(1, "doomed")
    box.enqueue(1, "the next answer")

    asyncio.run(outbox.drain(bot, box))  # buries the head, blocks the chat
    asyncio.run(outbox.drain(bot, box))  # …and the very next pass flows

    assert bot.sent == [(1, "the next answer")]
    assert box.pending() == []
    dead = json.loads(next(box.dead_dir.glob("*.json")).read_text())
    assert dead["text"] == "doomed"
    assert "blocked" in dead["reason"]
    assert dead["attempts"] == 1  # not ten


def test_a_bookkeeping_write_failure_does_not_strand_other_chats(tmp_path):
    """Blind review S1: record_failure writes to the same disk the send may
    have failed over. Letting that escape ended the whole pass — every other
    chat's healthy reply went undelivered, pass after pass."""
    clock = Clock()
    box = _box(tmp_path, clock)
    bot = FakeBot(fail_texts={"stuck"})
    box.enqueue(1, "stuck")
    box.enqueue(2, "fine")

    def no_disk(*_a, **_k):
        raise OSError("no space left on device")

    box.record_failure = no_disk  # type: ignore[method-assign]
    asyncio.run(outbox.drain(bot, box))

    assert bot.sent == [(2, "fine")]


def test_a_pass_is_bounded(tmp_path, monkeypatch):
    """Blind review S3: the pass holds the lock, so an unbounded backlog
    made a fresh reply wait out the whole queue."""
    monkeypatch.setattr(outbox, "SEND_SPACING", 0)
    monkeypatch.setattr(outbox, "MAX_PER_PASS", 3)
    box = _box(tmp_path)
    bot = FakeBot()
    for n in range(10):
        box.enqueue(1, f"m{n}")

    asyncio.run(outbox.drain(bot, box))
    assert [t for _c, t in bot.sent] == ["m0", "m1", "m2"]

    asyncio.run(outbox.drain(bot, box))
    assert [t for _c, t in bot.sent] == ["m0", "m1", "m2", "m3", "m4", "m5"]


# ── restart ──────────────────────────────────────────────────────────


def test_a_new_process_delivers_what_the_old_one_left(tmp_path):
    clock = Clock()
    dying = _box(tmp_path, clock)
    dying.enqueue(3, "queued just before the crash")
    del dying  # the process goes away; only files remain

    reborn = _box(tmp_path, clock)
    bot = FakeBot()
    asyncio.run(outbox.drain(bot, reborn))

    assert bot.sent == [(3, "queued just before the crash")]
    assert reborn.pending() == []


# ── duplicate protection ─────────────────────────────────────────────


def test_a_receipt_stops_the_duplicate_after_a_crash(tmp_path):
    """Crash between "Telegram accepted it" and "remove the entry": the
    receipt is written first, so the next drain drops the entry instead of
    sending the same reply twice."""
    clock = Clock()
    box = _box(tmp_path, clock)
    item = box.enqueue(5, "already gone out")

    box._add_receipt(item.id)  # receipt written... and then we died
    assert box.pending()  # the entry is still there

    reborn = _box(tmp_path, clock)
    bot = FakeBot()
    asyncio.run(outbox.drain(bot, reborn))

    assert bot.attempted == []  # not sent again
    assert reborn.pending() == []  # and cleaned up


def test_delivery_writes_a_receipt(tmp_path):
    box = _box(tmp_path)
    bot = FakeBot()
    item = box.enqueue(5, "x")
    asyncio.run(outbox.drain(bot, box))
    assert box.is_delivered(item.id)


def test_receipts_are_capped(tmp_path, monkeypatch):
    # The 0.3s inter-chunk pacing is real behavior, but 220 of them is 66s
    # of test runtime and proves nothing about receipts.
    monkeypatch.setattr(outbox, "SEND_SPACING", 0)
    box = _box(tmp_path)
    bot = FakeBot()
    total = outbox.RECEIPTS_KEPT + 20
    for n in range(total):
        box.enqueue(1, f"m{n}")
    while box.pending():  # a pass is bounded by MAX_PER_PASS
        asyncio.run(outbox.drain(bot, box))
    assert len(bot.sent) == total
    ids = json.loads(box.receipts_file.read_text())["ids"]
    assert len(ids) == outbox.RECEIPTS_KEPT


# ── dead queue ───────────────────────────────────────────────────────


def test_exhausted_attempts_land_in_the_dead_queue_with_a_reason(tmp_path):
    clock = Clock()
    box = _box(tmp_path, clock, max_attempts=3)
    bot = FakeBot(fail_texts={"doomed"})
    box.enqueue(9, "doomed")

    for _ in range(3):
        asyncio.run(outbox.drain(bot, box))
        clock.advance(outbox.MAX_BACKOFF + 1)

    assert box.pending() == []
    dead = list(box.dead_dir.glob("*.json"))
    assert len(dead) == 1
    payload = json.loads(dead[0].read_text())
    assert payload["text"] == "doomed"
    assert payload["chat_id"] == 9
    assert "telegram is down" in payload["reason"]
    assert payload["attempts"] == 3
    assert payload["failed_at"] >= 1000.0


def test_dead_letters_are_counted_in_stats(tmp_path):
    clock = Clock()
    box = _box(tmp_path, clock, max_attempts=1)
    bot = FakeBot(fail_texts={"doomed"})
    box.enqueue(9, "doomed")
    box.enqueue(9, "waiting")

    asyncio.run(outbox.drain(bot, box))

    stats = box.stats()
    assert stats.dead == 1
    assert stats.pending == 1


def test_stats_report_the_waiting_queue(tmp_path):
    box = _box(tmp_path)
    assert box.stats() == outbox.Stats(0, 0, 0.0)
    box.enqueue(1, "a")
    box.enqueue(1, "b")
    stats = box.stats()
    assert stats.pending == 2
    assert stats.dead == 0


def test_age_is_measured_from_when_the_reply_was_produced(tmp_path):
    """Blind review S5: the age used to come from the file's mtime, which a
    retry rewrites — a reply stuck for an hour reported itself as fresh."""
    clock = Clock()
    box = _box(tmp_path, clock)
    bot = FakeBot(fail_texts={"stuck"})
    box.enqueue(1, "stuck")

    for _ in range(3):
        clock.advance(600.0)
        asyncio.run(outbox.drain(bot, box))  # rewrites the file each time

    assert box.stats(now=clock.now).oldest_age >= 1800.0


def test_stats_do_not_quarantine_what_they_cannot_read(tmp_path):
    """A status command must not mutate the queue it describes."""
    box = _box(tmp_path)
    box.dir.mkdir(parents=True, exist_ok=True)
    (box.dir / "0000000000000000001.json").write_text("{not json")

    assert box.stats().pending == 1
    assert list(box.dead_dir.glob("*.json")) == []


# ── ordering ─────────────────────────────────────────────────────────


def test_a_failing_message_does_not_let_its_chat_overtake_it(tmp_path):
    """Order inside a chat is the promise: while "first" is stuck, "second"
    and "third" must wait, and when it finally goes out all three arrive in
    the order they were produced."""
    clock = Clock()
    box = _box(tmp_path, clock)
    bot = FakeBot(fail_texts={"first"})
    for text in ("first", "second", "third"):
        box.enqueue(1, text)

    asyncio.run(outbox.drain(bot, box))
    assert bot.sent == []  # nothing overtook the stuck head

    bot.fail_texts.clear()
    clock.advance(outbox.MAX_BACKOFF + 1)
    asyncio.run(outbox.drain(bot, box))

    assert [t for _c, t in bot.sent] == ["first", "second", "third"]


def test_a_backing_off_message_does_not_let_its_chat_overtake_it(tmp_path):
    clock = Clock()
    box = _box(tmp_path, clock)
    bot = FakeBot(fail_texts={"first"})
    box.enqueue(1, "first")
    asyncio.run(outbox.drain(bot, box))

    # A new reply for the same chat arrives while the first is backing off.
    box.enqueue(1, "second")
    bot.fail_texts.clear()
    asyncio.run(outbox.drain(bot, box))

    assert bot.sent == []  # "second" waited for "first"


def test_one_stuck_chat_does_not_block_another(tmp_path):
    clock = Clock()
    box = _box(tmp_path, clock)
    bot = FakeBot(fail_texts={"stuck"})
    box.enqueue(1, "stuck")
    box.enqueue(2, "fine")

    asyncio.run(outbox.drain(bot, box))

    assert bot.sent == [(2, "fine")]
    assert [i.text for i in box.pending()] == ["stuck"]


def test_concurrent_drains_do_not_double_send(tmp_path):
    """aiogram runs every update as its own task, so two replies can reach
    drain() at the same instant. The lock is what stops both passes from
    reading the same pending entry and sending it twice."""
    box = _box(tmp_path)
    bot = FakeBot()
    box.enqueue(1, "once")

    async def both():
        await asyncio.gather(outbox.drain(bot, box), outbox.drain(bot, box))

    asyncio.run(both())
    assert bot.sent == [(1, "once")]


# ── the sending function itself ──────────────────────────────────────


def test_send_one_falls_back_to_plain_text():
    class PickyBot:
        def __init__(self):
            self.calls: list[dict] = []

        async def send_message(self, chat_id, text, **kwargs):
            self.calls.append(kwargs)
            if "parse_mode" not in kwargs:
                raise _bad_request()

    bot = PickyBot()
    asyncio.run(outbox.send_one(bot, 1, "<b>broken"))
    assert len(bot.calls) == 2
    assert bot.calls[1]["parse_mode"] is None


def test_drain_never_raises(tmp_path):
    class ExplodingBox(Outbox):
        def pending(self):
            raise RuntimeError("disk is gone")

    box = ExplodingBox(tmp_path)
    asyncio.run(outbox.drain(FakeBot(), box))  # must not raise


# ── the process-wide handle ──────────────────────────────────────────


@pytest.fixture
def clean_outbox():
    outbox.reset()
    yield
    outbox.reset()


def test_configure_installs_and_reset_clears(tmp_path, clean_outbox):
    assert outbox.current() is None
    box = outbox.configure(tmp_path)
    assert outbox.current() is box
    outbox.reset()
    assert outbox.current() is None


def test_send_response_delivers_inline_and_leaves_nothing_behind(
    tmp_path, clean_outbox
):
    """The user-visible half of the promise: a healthy reply still goes out
    in the same await it always did, and the queue is empty afterwards."""
    from d_brain.bot.formatters import send_response

    box = outbox.configure(tmp_path)
    bot = FakeBot()
    asyncio.run(send_response(bot, 42, "<b>hi</b>"))

    assert bot.sent == [(42, "<b>hi</b>")]
    assert box.pending() == []


def test_send_response_keeps_a_reply_telegram_refused(tmp_path, clean_outbox):
    """The other half: the send failed, the caller was told nothing, and the
    answer is still on disk waiting for the worker."""
    from d_brain.bot.formatters import send_response

    box = outbox.configure(tmp_path)
    bot = FakeBot(fail_texts={"answer nobody received"})
    asyncio.run(send_response(bot, 42, "answer nobody received"))

    assert bot.sent == []
    waiting = box.pending()
    assert [i.text for i in waiting] == ["answer nobody received"]
    assert waiting[0].attempts == 1


def test_send_response_queues_a_long_reply_chunk_by_chunk(tmp_path, clean_outbox):
    """Chunking happens once, at enqueue: a retry then re-sends only the
    chunk that failed, and the queue order IS the chunk order."""
    from d_brain.bot.formatters import MAX_RESPONSE_LENGTH, send_response

    box = outbox.configure(tmp_path)
    bot = FakeBot()
    asyncio.run(send_response(bot, 42, "я" * 9000))

    assert len(bot.sent) > 1
    assert all(len(t) <= MAX_RESPONSE_LENGTH for _c, t in bot.sent)
    assert box.pending() == []
    assert "".join(t for _c, t in bot.sent).count("я") == 9000


def test_send_response_still_sends_when_the_queue_cannot_be_written(
    tmp_path, clean_outbox
):
    """Blind review B2: durability is an upgrade over sending, never a
    precondition. A full or unwritable disk used to mean the reply never
    reached Telegram at all — the exact failure this feature exists to
    prevent."""
    from d_brain.bot.formatters import send_response

    box = outbox.configure(tmp_path)

    def no_disk(*_a, **_k):
        raise OSError("no space left on device")

    box.enqueue = no_disk  # type: ignore[method-assign]
    bot = FakeBot()
    asyncio.run(send_response(bot, 42, "the answer"))

    assert bot.sent == [(42, "the answer")]


def test_a_partly_queued_reply_keeps_its_chunk_order(tmp_path, clean_outbox):
    """If the disk gives out halfway through a chunked reply, the chunks
    already queued must still go out FIRST — the fallback sends only the
    remainder, and only after the drain."""
    from d_brain.bot.formatters import send_response

    box = outbox.configure(tmp_path)
    real_enqueue = box.enqueue
    calls = {"n": 0}

    def fail_after_two(chat_id, text, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            raise OSError("no space left on device")
        return real_enqueue(chat_id, text, **kwargs)

    box.enqueue = fail_after_two  # type: ignore[method-assign]
    bot = FakeBot()
    asyncio.run(send_response(bot, 42, "я" * 9000))

    assert "".join(t for _c, t in bot.sent).count("я") == 9000
    assert box.pending() == []


def test_send_response_without_a_configured_outbox_still_sends(clean_outbox):
    """Unit tests and one-off scripts have no runtime dir — the reply goes
    out through the same send_one, just with nothing on disk behind it."""
    from d_brain.bot.formatters import send_response

    bot = FakeBot()
    asyncio.run(send_response(bot, 42, "hi"))
    assert bot.sent == [(42, "hi")]


def test_run_delivers_the_queue_and_keeps_going(tmp_path, clean_outbox):
    box = _box(tmp_path)
    bot = FakeBot()
    box.enqueue(1, "left over from the last process")

    async def scenario():
        task = asyncio.create_task(outbox.run(bot, box, poll_seconds=0.01))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if bot.sent:
                break
        task.cancel()

    asyncio.run(scenario())
    assert bot.sent == [(1, "left over from the last process")]
