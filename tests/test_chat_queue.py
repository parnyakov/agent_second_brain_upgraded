"""Per-chat queue — item 33, step 5.

The five things the owner asked for, one test each and then some: a second
message during work is parked and acknowledged at once, the queue drains
itself in order, it survives a restart, the ceiling is honest, and the duty
session stops being the answer to "busy" while staying the answer to
"wedged".
"""

import asyncio

import pytest

from d_brain.services import chat_queue
from d_brain.services.chat_session import Busy


@pytest.fixture
def queue(tmp_path):
    """A process-wide queue on a temp dir, torn down after every test."""
    box = chat_queue.configure(tmp_path, max_waiting=10, max_age=7200.0)
    yield box
    chat_queue.reset()


@pytest.fixture(autouse=True)
def _no_leftover_queue():
    chat_queue.reset()
    yield
    chat_queue.reset()


class FakeBot:
    def __init__(self):
        self.messages: list[tuple[int, str]] = []
        self.kwargs: list[dict] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text))
        self.kwargs.append(kwargs)

    async def send_chat_action(self, chat_id, action):
        pass

    def texts(self) -> str:
        return "\n".join(t for _c, t in self.messages)


class FakeManager:
    """Idle, answers instantly."""

    def __init__(self, reply="ответ"):
        self.reply = reply
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, user_id: int, prompt: str):
        self.sent.append((user_id, prompt))
        return self.reply

    def is_turn_active(self) -> bool:
        return False

    def is_steerable_turn(self) -> bool:
        return True


class SlowManager(FakeManager):
    """Holds its turn open until the test lets go of it."""

    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def send_message(self, user_id: int, prompt: str):
        self.sent.append((user_id, prompt))
        self.started.set()
        await self.release.wait()
        return self.reply


class BusyManager(FakeManager):
    """The engine refuses to type over a live, progressing turn."""

    def __init__(self):
        super().__init__()
        self.duty_calls: list[tuple] = []

    async def send_message(self, user_id: int, prompt: str):
        self.sent.append((user_id, prompt))
        return Busy(busy_seconds=600.0, fallback="🛠 старая отбивка /stop")

    async def answer_from_duty(
        self, user_id, prompt, *, busy_seconds=None, fallback=None
    ):
        self.duty_calls.append((user_id, prompt, fallback))
        return "из дежурной"


# ── 1. a second message during work ──────────────────────────────────


def test_a_second_message_during_work_is_parked_and_acknowledged(monkeypatch, queue):
    """The headline: while the first message is being worked on, the second
    one gets an immediate receipt with its place in line and waits ON DISK —
    not a brush-off, not a context-less stand-in."""
    from d_brain.bot.handlers import chat

    mgr = SlowManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    async def scenario():
        first = asyncio.create_task(
            chat._process_and_reply(bot, 10, 7, "первое", message_id=100)
        )
        await mgr.started.wait()
        await chat._process_and_reply(bot, 10, 7, "второе", message_id=101)
        mgr.release.set()
        await first

    asyncio.run(scenario())

    assert "в очереди: 1" in bot.texts()
    assert "Принял" in bot.texts()
    # Only the FIRST message reached the session; the second waited.
    assert mgr.sent == [(7, "первое")]
    waiting = queue.waiting(10)
    assert len(waiting) == 1
    job = waiting[0]
    # Everything an answer needs is on disk with it: who asked, and which
    # message the answer belongs to.
    assert (job.prompt, job.user_id, job.chat_id, job.message_id) == (
        "второе",
        7,
        10,
        101,
    )


def test_a_third_message_is_told_its_real_place_in_line(monkeypatch, queue):
    from d_brain.bot.handlers import chat

    mgr = SlowManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    async def scenario():
        first = asyncio.create_task(chat._process_and_reply(bot, 10, 7, "первое"))
        await mgr.started.wait()
        await chat._process_and_reply(bot, 10, 7, "второе")
        await chat._process_and_reply(bot, 10, 7, "третье")
        mgr.release.set()
        await first

    asyncio.run(scenario())
    assert "в очереди: 1" in bot.texts()
    assert "в очереди: 2" in bot.texts()
    assert len(queue.waiting(10)) == 2


def test_a_message_never_overtakes_one_that_is_already_waiting(monkeypatch, queue):
    """FIFO is not only about the lane. A message arriving in the gap
    between two drains — lane free, queue not empty — must go to the BACK of
    the queue, not straight into the session."""
    from d_brain.bot.handlers import chat

    queue.enqueue(10, 7, "жду с прошлого раза")
    mgr = FakeManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    asyncio.run(chat._process_and_reply(bot, 10, 7, "только что пришло"))

    assert mgr.sent == []  # nothing jumped the line
    assert [j.prompt for j in queue.waiting(10)] == [
        "жду с прошлого раза",
        "только что пришло",
    ]


# ── 2. the queue drains itself, in order ─────────────────────────────


def test_the_queue_drains_itself_one_at_a_time_in_order(monkeypatch, queue):
    from d_brain.bot.handlers import chat

    queue.enqueue(10, 7, "первое", message_id=1)
    queue.enqueue(10, 7, "второе", message_id=2)
    mgr = FakeManager(reply="готово")
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    asyncio.run(chat_queue.drain(bot, queue, chat.run_queued_job))
    assert mgr.sent == [(7, "первое")]  # one per pass, oldest first
    assert [j.prompt for j in queue.waiting(10)] == ["второе"]

    asyncio.run(chat_queue.drain(bot, queue, chat.run_queued_job))
    assert mgr.sent == [(7, "первое"), (7, "второе")]
    assert queue.waiting() == []


def test_a_queued_answer_is_threaded_under_the_message_it_answers(monkeypatch, queue):
    """The stored ``message_id`` is not decoration: ten minutes later the
    answer has to say which question it belongs to."""
    from d_brain.bot.handlers import chat

    queue.enqueue(10, 7, "вопрос", message_id=4242)
    monkeypatch.setattr(chat, "_get_manager", lambda: FakeManager(reply="ответ"))
    bot = FakeBot()

    asyncio.run(chat_queue.drain(bot, queue, chat.run_queued_job))

    assert bot.messages == [(10, "ответ")]
    assert bot.kwargs[0]["reply_to_message_id"] == 4242
    # Never load-bearing: a deleted original must not bury a real reply.
    assert bot.kwargs[0]["allow_sending_without_reply"] is True


def test_a_fresh_reply_is_not_threaded(monkeypatch, queue):
    """Only a message that WAITED gets the quote; a reply that follows its
    question directly would just be noise."""
    from d_brain.bot.handlers import chat

    monkeypatch.setattr(chat, "_get_manager", lambda: FakeManager(reply="ответ"))
    bot = FakeBot()
    asyncio.run(chat._process_and_reply(bot, 10, 7, "вопрос", message_id=4242))
    assert "reply_to_message_id" not in bot.kwargs[0]


def test_a_job_whose_turn_cannot_start_keeps_its_place(monkeypatch, queue):
    """The drain's other outcome: the session is still busy with live work,
    so the job is NOT dropped and no second receipt is sent."""
    from d_brain.bot.handlers import chat

    queue.enqueue(10, 7, "жду")
    mgr = BusyManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    asyncio.run(chat_queue.drain(bot, queue, chat.run_queued_job))

    assert [j.prompt for j in queue.waiting(10)] == ["жду"]
    assert bot.messages == []  # acknowledged once, at parking time, not again
    assert mgr.duty_calls == []


def test_the_drain_skips_a_chat_whose_lane_is_taken(queue):
    """One live runner per chat: a job must not start behind the back of an
    inline turn that is already running for the same chat."""
    queue.enqueue(10, 7, "жду")
    queue.try_acquire(10)
    started: list = []

    async def runner(_bot, job):
        started.append(job)
        return True

    asyncio.run(chat_queue.drain(FakeBot(), queue, runner))
    assert started == []
    queue.release(10)
    asyncio.run(chat_queue.drain(FakeBot(), queue, runner))
    assert len(started) == 1


# ── 3. it survives a restart ─────────────────────────────────────────


def test_the_queue_survives_a_restart(tmp_path, monkeypatch):
    """A parked message is a FILE. A brand-new process — new ChatQueue over
    the same runtime dir, empty in-memory lane — finds it and answers it."""
    from d_brain.bot.handlers import chat

    before = chat_queue.configure(tmp_path)
    before.enqueue(10, 7, "переживу перезапуск", message_id=99)
    before.try_acquire(10)  # a lane held by the process that is going away
    chat_queue.reset()

    after = chat_queue.configure(tmp_path)
    assert not after.is_running(10)  # the lane did NOT survive, and must not
    jobs = after.waiting(10)
    assert [j.prompt for j in jobs] == ["переживу перезапуск"]
    assert jobs[0].message_id == 99

    mgr = FakeManager(reply="поздний ответ")
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat_queue.drain(bot, after, chat.run_queued_job))
    assert mgr.sent == [(7, "переживу перезапуск")]
    assert after.waiting() == []
    chat_queue.reset()


def test_a_job_is_removed_only_after_its_turn_finished(queue):
    """At-least-once, deliberately: a crash mid-turn must leave the job on
    disk. Removing it first would lose exactly the message this queue
    exists to keep."""
    job, _pos = queue.enqueue(10, 7, "x")

    async def crashing(_bot, _job):
        assert queue.waiting(10)  # still on disk while the turn runs
        raise RuntimeError("engine exploded")

    asyncio.run(chat_queue.drain(FakeBot(), queue, crashing))
    assert [j.id for j in queue.waiting(10)] == [job.id]
    assert queue.waiting(10)[0].attempts == 1


def test_a_job_that_keeps_blowing_up_is_retired_with_an_honest_line(queue):
    """Nothing piles up silently and nothing vanishes: past the attempt cap
    the job moves to stale/ and the owner is told."""
    queue.enqueue(10, 7, "ядовитое", message_id=5)

    async def crashing(_bot, _job):
        raise RuntimeError("boom")

    bot = FakeBot()
    for _ in range(chat_queue.MAX_ATTEMPTS):
        asyncio.run(chat_queue.drain(bot, queue, crashing))

    assert queue.waiting(10) == []
    assert len(list(queue.stale_dir.glob("*.json"))) == 1
    assert "не дошло до основной сессии" in bot.texts()


def test_a_job_that_waited_too_long_is_retired_not_answered(tmp_path):
    box = chat_queue.ChatQueue(tmp_path, max_age=60.0, clock_fn=lambda: 1000.0)
    box.enqueue(10, 7, "старое")
    box._clock = lambda: 2000.0  # an hour later
    ran: list = []

    async def runner(_bot, job):
        ran.append(job)
        return True

    bot = FakeBot()
    asyncio.run(chat_queue.drain(bot, box, runner))
    assert ran == []
    assert box.waiting() == []
    assert "слишком долго ждало" in bot.texts()


# ── 4. the ceiling ───────────────────────────────────────────────────


def test_past_the_limit_the_sender_is_told_plainly(tmp_path, monkeypatch):
    """Not silence and not a lie: the message is refused out loud, with what
    was kept and what to do next."""
    from d_brain.bot.handlers import chat

    box = chat_queue.configure(tmp_path, max_waiting=2)
    mgr = SlowManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    async def scenario():
        first = asyncio.create_task(chat._process_and_reply(bot, 10, 7, "первое"))
        await mgr.started.wait()
        for text in ("второе", "третье", "четвёртое"):
            await chat._process_and_reply(bot, 10, 7, text)
        mgr.release.set()
        await first

    asyncio.run(scenario())

    assert len(box.waiting(10)) == 2  # the ceiling held
    assert "больше не беру" in bot.texts()
    assert "/stop" in bot.texts()
    chat_queue.reset()


def test_the_limit_is_per_chat_not_global(tmp_path):
    box = chat_queue.ChatQueue(tmp_path, max_waiting=1)
    box.enqueue(10, 7, "a")
    box.enqueue(11, 8, "b")  # another chat is unaffected
    with pytest.raises(chat_queue.QueueFull):
        box.enqueue(10, 7, "c")
    chat_queue.reset()


# ── 5. what became of the duty session ───────────────────────────────


def test_plain_busyness_parks_instead_of_spending_a_duty_turn(monkeypatch, queue):
    """The owner's rule: «отвечать мне должна основная сессия в 99% случаев».
    A live, progressing turn is busyness, not a failure — the message waits
    for the session that has the context."""
    from d_brain.bot.handlers import chat

    mgr = BusyManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    asyncio.run(chat._process_and_reply(bot, 10, 7, "запиши мысль", message_id=3))

    assert mgr.duty_calls == []
    assert "в очереди: 1" in bot.texts()
    assert [j.prompt for j in queue.waiting(10)] == ["запиши мысль"]


def test_an_unattended_long_run_parks_instead_of_spending_a_duty_turn(
    monkeypatch, queue
):
    """The pre-ask gate (a fresh watchdog marker + two pane probes) used to
    route straight to the duty session. It parks now."""
    from d_brain.bot.handlers import chat

    class Unattended(FakeManager):
        def __init__(self):
            super().__init__()
            self.duty_calls: list = []

        async def is_main_busy(self) -> bool:
            return True

        async def answer_from_duty(self, user_id, prompt, **kw):
            self.duty_calls.append(prompt)
            return "из дежурной"

    mgr = Unattended()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._process_and_reply(bot, 10, 7, "[voice] мысль"))

    assert mgr.sent == []  # ask() never entered
    assert mgr.duty_calls == []  # and no stand-in turn was spent
    assert [j.prompt for j in queue.waiting(10)] == ["[voice] мысль"]


def test_a_maintenance_turn_parks_the_message_on_every_input_path(monkeypatch, queue):
    """The nightly pipeline holds the session and takes no input. Before,
    only TYPED TEXT noticed (the check lived in _dispatch_text) — voice and
    media blocked on the ask-lock in silence. Now every path parks."""
    from d_brain.bot.handlers import chat

    class Maintenance(FakeManager):
        def __init__(self):
            super().__init__()
            self.duty_calls: list = []

        def is_turn_active(self) -> bool:
            return True

        def is_steerable_turn(self) -> bool:
            return False

        async def answer_from_duty(self, user_id, prompt, **kw):
            self.duty_calls.append(prompt)
            return "из дежурной"

    mgr = Maintenance()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    # the voice/media choke point, not the text path
    asyncio.run(chat._process_and_reply(bot, 10, 7, "[voice] мысль"))
    # and the text path, which routes into the same place
    asyncio.run(chat._dispatch_text(bot, 10, 7, "пиши короче", message_id=8))

    assert mgr.sent == []
    assert mgr.duty_calls == []
    assert [j.prompt for j in queue.waiting(10)] == ["[voice] мысль", "пиши короче"]


def test_a_wedged_session_still_reaches_the_duty_session(monkeypatch, queue):
    """The emergency path survives. ``ChatSessionManager`` answers a plain
    ``busy`` — no real progress across the whole busy-wait — from the duty
    session itself, so the handler just delivers what it returns and the
    message is finished with, never parked behind a wedge."""
    from d_brain.bot.handlers import chat

    class Wedged(FakeManager):
        async def send_message(self, user_id, prompt):
            self.sent.append((user_id, prompt))
            return "🔁 <i>Основная сессия занята</i>\n\nиз дежурной"

    monkeypatch.setattr(chat, "_get_manager", lambda: Wedged())
    bot = FakeBot()
    asyncio.run(chat._process_and_reply(bot, 10, 7, "привет"))

    assert "из дежурной" in bot.texts()
    assert queue.waiting() == []


def test_a_steerable_turn_still_steers_rather_than_queues(monkeypatch, queue):
    """The one busy case the queue deliberately does NOT take: steering
    hands the text to the turn running right now, which beats waiting for a
    turn of its own."""
    from d_brain.bot.handlers import chat

    class Steerable(FakeManager):
        def __init__(self):
            super().__init__()
            self.steered: list[str] = []

        def is_turn_active(self) -> bool:
            return True

        def is_pane_turn_active(self) -> bool:
            return True

        def is_steerable_turn(self) -> bool:
            return True

        async def steer(self, text: str) -> None:
            self.steered.append(text)

    mgr = Steerable()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._dispatch_text(bot, 10, 7, "добавь ещё пункт", message_id=9))

    assert mgr.steered == ["добавь ещё пункт"]
    assert queue.waiting() == []


# ── degradation, visibility and the stop ─────────────────────────────


def test_without_a_configured_queue_nothing_changes(monkeypatch):
    """The rollback (DBRAIN_CHAT_QUEUE=false) and every unit test that never
    configures a queue: busyness lands on exactly the pre-queue duty path,
    with its pre-queue wording."""
    from d_brain.bot.handlers import chat

    assert chat_queue.current() is None
    mgr = BusyManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._process_and_reply(bot, 10, 7, "привет"))

    assert mgr.duty_calls == [(7, "привет", "🛠 старая отбивка /stop")]
    assert "из дежурной" in bot.texts()


def test_an_unwritable_queue_falls_back_to_answering(monkeypatch, queue):
    """Durability is an upgrade over answering, never a precondition for it
    (the outbox's blind review, B2)."""
    from d_brain.bot.handlers import chat

    def no_disk(*_a, **_k):
        raise OSError("no space left on device")

    monkeypatch.setattr(queue, "enqueue", no_disk)
    mgr = BusyManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()
    asyncio.run(chat._process_and_reply(bot, 10, 7, "привет"))

    assert mgr.duty_calls  # the message was answered, not dropped
    assert "из дежурной" in bot.texts()


def test_work_report_shows_what_is_waiting_per_chat(tmp_path, queue):
    from d_brain.bot.handlers.work import _chat_queue_lines
    from d_brain.config import Settings

    queue.enqueue(10, 7, "a")
    queue.enqueue(10, 7, "b")
    queue.try_acquire(10)
    settings = Settings(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path,
        runtime_dir=tmp_path,
        _env_file=None,
    )
    text = "\n".join(_chat_queue_lines(settings, now=queue.now()))
    assert "ждут в очереди: 2" in text
    assert "чат <code>10</code>: 2" in text
    assert "разбирается прямо сейчас: 1" in text


def test_work_report_survives_an_unreadable_queue(tmp_path, monkeypatch, queue):
    from d_brain.bot.handlers.work import _chat_queue_lines
    from d_brain.config import Settings

    def boom(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(queue, "stats", boom)
    settings = Settings(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path,
        runtime_dir=tmp_path,
        _env_file=None,
    )
    assert "недоступно" in "\n".join(_chat_queue_lines(settings, now=0.0))


def test_a_requested_stop_keeps_the_worker_from_starting_new_jobs(queue):
    """Graceful stop: the job already running rides out the grace, but
    starting a brand-new turn seconds before the deadline helps nobody."""
    queue.enqueue(10, 7, "не начинай")
    started: list = []

    async def runner(_bot, job):
        started.append(job)
        return True

    asyncio.run(chat_queue.drain(FakeBot(), queue, runner, should_start=lambda: False))
    assert started == []
    assert len(queue.waiting(10)) == 1


def test_a_leaked_lane_expires_instead_of_deafening_the_chat(tmp_path):
    """Never let one bug become a standing outage (items 22/23). The lane is
    released in a finally on every path; if one ever leaks, the chat heals
    by itself."""
    clock = {"t": 0.0}
    box = chat_queue.ChatQueue(tmp_path, monotonic_fn=lambda: clock["t"])
    assert box.try_acquire(10) is True
    assert box.try_acquire(10) is False
    clock["t"] = chat_queue.LANE_TTL + 1
    assert box.try_acquire(10) is True
    chat_queue.reset()


def test_an_unparsable_job_is_quarantined_not_retried_forever(tmp_path):
    box = chat_queue.ChatQueue(tmp_path)
    box.dir.mkdir(parents=True, exist_ok=True)
    (box.dir / "0000000000000000001.json").write_text("{not json")
    assert box.waiting() == []
    assert (box.stale_dir / "0000000000000000001.json").exists()
    chat_queue.reset()


def test_a_file_arriving_during_a_turn_is_queued_not_brushed_off(monkeypatch, queue):
    """The media path had its own busy guard (item 21) that answered "файл
    сохранил, сессия занята" and stopped there. With the queue it steps
    aside: the file is prepared, parked and acknowledged like any other
    message, and the MAIN session answers it."""
    from d_brain.bot.handlers import chat

    mgr = SlowManager()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    async def scenario():
        first = asyncio.create_task(chat._process_and_reply(bot, 10, 7, "первое"))
        await mgr.started.wait()
        assert await chat.reject_media_if_busy(bot, 10, ["a.jpg"]) is False
        await chat._process_and_reply(
            bot, 10, 7, "Пользователь прислал photo: a.jpg", message_id=55
        )
        mgr.release.set()
        await first

    asyncio.run(scenario())

    assert "Файл сохранил" not in bot.texts()  # no brush-off
    assert "в очереди: 1" in bot.texts()
    assert [j.prompt for j in queue.waiting(10)] == [
        "Пользователь прислал photo: a.jpg"
    ]


def test_without_a_queue_the_media_busy_guard_is_untouched(monkeypatch):
    """Rollback: no queue ⇒ byte-for-byte item 21's guard."""
    from d_brain.bot.handlers import chat

    chat._release_media_dispatch()

    class Busy1(FakeManager):
        def is_turn_active(self) -> bool:
            return True

    monkeypatch.setattr(chat, "_get_manager", lambda: Busy1())
    bot = FakeBot()
    try:
        assert asyncio.run(chat.reject_media_if_busy(bot, 10, ["a.jpg"])) is True
        assert "Файл сохранил" in bot.texts()
    finally:
        chat._release_media_dispatch()


# ── regressions the blind review found ───────────────────────────────


class StuckProbe(FakeManager):
    """Busy, and slow to admit it — the gate itself takes real time
    (~3s of pane probes in production, ask()'s busy-wait up to 300s)."""

    def __init__(self):
        super().__init__()
        self.probing = asyncio.Event()
        self.release = asyncio.Event()

    async def is_main_busy(self) -> bool:
        self.probing.set()
        await self.release.wait()
        return True


def test_the_message_that_held_the_lane_keeps_its_place_in_line(monkeypatch, queue):
    """Blind review 1. The lane-holder is parked LAST — it only learns the
    session is busy after the probes and ask()'s busy-wait — so a queue
    ordered by PARKING time answered the second message first and told the
    first one it was second. The id is the ARRIVAL stamp for exactly this."""
    from d_brain.bot.handlers import chat

    mgr = StuckProbe()
    monkeypatch.setattr(chat, "_get_manager", lambda: mgr)
    bot = FakeBot()

    async def scenario():
        first = asyncio.create_task(
            chat._process_and_reply(bot, 10, 7, "ПЕРВОЕ", message_id=1)
        )
        await mgr.probing.wait()
        await chat._process_and_reply(bot, 10, 7, "ВТОРОЕ", message_id=2)
        mgr.release.set()
        await first

    asyncio.run(scenario())

    assert [j.prompt for j in queue.waiting(10)] == ["ПЕРВОЕ", "ВТОРОЕ"]
    assert [j.prompt for j in queue.ready()] == ["ПЕРВОЕ"]
    # ...and it was told its real place, not "you are second".
    assert bot.texts().count("в очереди: 1") == 2
    assert "в очереди: 2" not in bot.texts()


def test_a_queued_turn_is_tracked_by_the_graceful_stop(queue):
    """Blind review 2. A queued turn is the one turn no aiogram update task
    is held open for, so ``stop_middleware`` never registers it — and it was
    the only turn in the process killed with no grace at all."""
    from d_brain.services import shutdown

    queue.enqueue(10, 7, "долгий ход")
    seen: list[int] = []

    async def scenario():
        stopper = shutdown.Shutdown(grace=5.0)

        async def runner(_bot, _job):
            seen.append(stopper.in_flight)
            return True

        await chat_queue.drain(FakeBot(), queue, runner, turns=stopper)
        return stopper

    stopper = asyncio.run(scenario())
    assert seen == [1]  # the stop could see it while it ran
    assert stopper.in_flight == 0  # and it was given back afterwards


def test_the_job_being_answered_is_not_counted_as_waiting(queue):
    """Blind review 3. A job stays on disk for the whole of its turn (that
    is what makes this at-least-once), so the naive count told the next
    sender "в очереди: 2" when one message was ahead of him and it was
    already being answered."""
    from d_brain.bot.handlers import chat

    queue.enqueue(10, 7, "уже отвечается")
    seen: list[int] = []

    async def runner(_bot, _job):
        seen.append(queue.pending_count(10))
        stats = queue.stats()
        assert stats.waiting == 0 and stats.running == 1
        return True

    asyncio.run(chat_queue.drain(FakeBot(), queue, runner))
    assert seen == [0]
    assert chat.queued_ack(1).endswith("(в очереди: 1).")


def test_turning_the_queue_off_withdraws_the_promise(tmp_path):
    """Blind review 4. The rollback switch must not strand messages the
    previous process already answered with "отвечу следом": nothing would
    ever drain them, and even the age cap lives inside the drain."""
    box = chat_queue.ChatQueue(tmp_path)
    box.enqueue(10, 7, "обещали ответить", message_id=3)
    bot = FakeBot()

    withdrawn = asyncio.run(
        chat_queue.retire_leftovers(bot, tmp_path, reason="очередь выключена")
    )

    assert withdrawn == 1
    assert box.waiting() == []
    assert len(list(box.stale_dir.glob("*.json"))) == 1  # nothing deleted
    assert "ответа по нему не будет" in bot.texts()
    chat_queue.reset()


def test_an_album_is_parked_under_the_message_id_of_its_first_file(monkeypatch, queue):
    """Blind review 5. An album is answered once, so it has to carry ONE
    message id — without it a late album answer pointed at nothing."""
    from d_brain.bot.handlers import chat

    monkeypatch.setattr(chat, "_get_manager", lambda: BusyManager())
    monkeypatch.setattr(chat, "ALBUM_SETTLE", 0.0)
    chat._album_buf["g1"] = [
        {
            "kind": "photo",
            "rel_path": "a.jpg",
            "caption": "",
            "fwd": "",
            "message_id": 77,
        },
        {
            "kind": "photo",
            "rel_path": "b.jpg",
            "caption": "",
            "fwd": "",
            "message_id": 78,
        },
    ]
    asyncio.run(chat._flush_album(FakeBot(), 10, 7, "g1"))

    waiting = queue.waiting(10)
    assert len(waiting) == 1  # ONE job for the whole album
    assert waiting[0].message_id == 77


def test_the_settings_defaults_are_the_module_constants(tmp_path):
    """Blind review 9. Two copies of a default drift in silence; this is the
    seam that notices."""
    from d_brain.config import Settings

    s = Settings(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path,
        runtime_dir=tmp_path,
        _env_file=None,
    )
    assert s.chat_queue_max_waiting == chat_queue.DEFAULT_MAX_WAITING
    assert s.chat_queue_max_age == chat_queue.DEFAULT_MAX_AGE


def test_the_worker_keeps_sweeping_after_a_failed_pass(queue, monkeypatch):
    """Blind review 8. A drain that somehow raised would end the loop, kill
    the task with an unread exception and hang EVERY parked message — after
    the owner was told "отвечу следом"."""
    passes = {"n": 0}

    async def exploding(*_a, **_k):
        passes["n"] += 1
        if passes["n"] == 1:
            raise RuntimeError("lane on fire")
        raise asyncio.CancelledError

    monkeypatch.setattr(chat_queue, "drain", exploding)

    async def scenario():
        with pytest.raises(asyncio.CancelledError):
            await chat_queue.run(FakeBot(), queue, lambda *_a: None, poll_seconds=0.0)

    asyncio.run(scenario())
    assert passes["n"] == 2  # it came back for a second pass
