"""Tests for the graceful stop.

What is pinned down here is what a restart PROMISES, not how the module is
built:

* a turn already running when SIGTERM lands gets to finish, and its reply
  actually leaves;
* a turn that outlives the deadline does not cost the message — the incoming
  entry is still in the inbox afterwards, and the next boot answers it;
* a message arriving DURING the stop is never started, is kept on disk, and
  gets one short line back instead of silence — and a stranger gets nothing;
* a second signal leaves at once instead of waiting out the deadline;
* the systemd units wait longer than the application deadline, or the whole
  thing buys nothing.
"""

import asyncio
import re
from pathlib import Path

import pytest
from aiogram import Dispatcher, Router
from aiogram.types import Update

from d_brain.config import Settings
from d_brain.services import inbox, outbox, shutdown
from d_brain.services.inbox import Inbox, entry_id

REPO = Path(__file__).resolve().parents[1]


# ── scaffolding ──────────────────────────────────────────────────────


class FakeSession:
    def __init__(self) -> None:
        self.middlewares: list = []
        self.closed = False
        self.timeout = 5

    def middleware(self, handler):
        self.middlewares.append(handler)
        return handler

    async def close(self):
        self.closed = True


class FakeBot:
    """Enough of a Bot for ``Update.model_validate`` and the outbox."""

    id = 1

    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.session = FakeSession()

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text))

    async def set_my_commands(self, commands):
        pass


class IdleSession:
    @staticmethod
    def ensure_session():
        pass


class Clock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _settings(tmp_path, **over):
    kwargs = dict(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path / "vault",
        runtime_dir=tmp_path / "rt",
        # Every helper update below comes from user 7; run_bot installs the
        # real auth middleware, which drops anyone else.
        allowed_user_ids=[7],
        cron_enabled=False,
        _env_file=None,
    )
    kwargs.update(over)
    s = Settings(**kwargs)
    s.runtime_dir.mkdir(parents=True, exist_ok=True)
    # One-shot keyboard-removal ping: already done, so it does not turn up in
    # every assertion about what the bot sent.
    (s.runtime_dir / "keyboard_removed").write_text("sent\n")
    return s


def _text(update_id: int, text: str = "привет", chat_id: int = 42, user: int = 7):
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "date": 1700000000,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user, "is_bot": False, "first_name": "Owner"},
            "text": text,
        },
    }


class Harness:
    """One ``run_bot`` under test, with polling faked and the stop drivable.

    The real ``run_bot`` is what runs — middleware order, the inbox, the
    outbox and the stop sequence included. Only two things are replaced:
    ``start_polling`` (which would talk to Telegram) and the process exit.
    """

    def __init__(self, settings, monkeypatch, *, grace: float):
        from d_brain.bot import main as bot_main

        self.settings = settings
        self.bot = FakeBot()
        self.dp = Dispatcher()
        self.exits: list[int] = []
        self.polling_kwargs: dict = {}
        self.polling_started = asyncio.Event()
        self.shutdown: shutdown.Shutdown | None = None
        self.handled: list[str] = []
        self._grace = grace

        real_shutdown = shutdown.Shutdown

        def factory(**kwargs):
            kwargs["grace"] = self._grace
            # Recorded instead of taken: ``finish``/``force`` would otherwise
            # os._exit the test runner. Execution continues past the call,
            # which is harmless — both are the last thing that happens.
            sd = real_shutdown(exit_fn=self.exits.append, **kwargs)
            self.shutdown = sd
            return sd

        async def fake_start_polling(*bots, **kwargs):
            self.polling_kwargs = kwargs
            self.polling_started.set()
            await asyncio.Event().wait()  # cancelled by the stop sequence

        monkeypatch.setattr(bot_main.shutdown, "Shutdown", factory)
        monkeypatch.setattr(self.dp, "start_polling", fake_start_polling)
        monkeypatch.setattr(bot_main, "create_bot", lambda settings: self.bot)
        monkeypatch.setattr(bot_main, "create_dispatcher", lambda: self.dp)
        monkeypatch.setattr(bot_main, "get_session", lambda settings: IdleSession())

    def on_message(self, handler):
        router = Router(name="probe")
        router.message()(handler)
        self.dp.include_router(router)

    async def start(self):
        from d_brain.bot import main as bot_main

        self.task = asyncio.create_task(bot_main.run_bot(self.settings))
        await asyncio.wait_for(self.polling_started.wait(), 5)
        return self

    def feed(self, update: dict) -> asyncio.Task:
        """Hand one update in the way polling does: on disk first.

        The real acceptance point is a session middleware around
        ``getUpdates`` (services/inbox.py), which faked polling never calls —
        so the write it would have done happens here, in the same order.
        """
        mailbox = inbox.current()
        assert mailbox is not None
        mailbox.accept(update)
        return asyncio.create_task(self.dp.feed_raw_update(self.bot, update))

    async def stop_and_wait(self, *, reason="SIGTERM"):
        assert self.shutdown is not None
        self.shutdown.request(reason)
        await asyncio.wait_for(self.task, 30)


def _run(coro):
    try:
        return asyncio.run(coro)
    finally:
        inbox.reset()
        outbox.reset()


# ── the deadline itself ──────────────────────────────────────────────


def test_first_signal_stops_taking_work_and_arms_the_deadline():
    clock = Clock()
    sd = shutdown.Shutdown(grace=300.0, clock_fn=clock, exit_fn=lambda code: None)

    assert not sd.stopping
    sd.request("SIGTERM")

    assert sd.stopping
    assert sd.remaining() == pytest.approx(300.0)
    clock.advance(120.0)
    assert sd.remaining() == pytest.approx(180.0)
    clock.advance(500.0)
    assert sd.remaining() == 0.0


def test_second_signal_exits_immediately():
    """The owner is in a hurry. Nothing is lost by leaving — both queues are
    files — so the second signal must not inherit the first one's wait."""
    exits: list[int] = []
    sd = shutdown.Shutdown(grace=9999.0, exit_fn=exits.append)

    sd.request("SIGTERM")
    assert exits == []

    sd.request("SIGTERM")
    assert exits == [shutdown.EXIT_CODE]


def test_wait_for_turns_returns_when_the_turns_finish():
    async def scenario():
        sd = shutdown.Shutdown(grace=30.0, exit_fn=lambda code: None)
        done = []

        async def turn():
            await asyncio.sleep(0.05)
            done.append("finished")

        task = asyncio.create_task(turn())
        sd.track(task)
        sd.request("SIGTERM")
        left = await sd.wait_for_turns()
        return left, done

    left, done = asyncio.run(scenario())
    assert left == []
    assert done == ["finished"]


def test_wait_for_turns_gives_up_at_the_deadline():
    async def scenario():
        sd = shutdown.Shutdown(grace=0.05, exit_fn=lambda code: None)
        task = asyncio.create_task(asyncio.sleep(30))
        sd.track(task)
        sd.request("SIGTERM")
        left = await sd.wait_for_turns()
        await shutdown.cancel_turns(left, timeout=1.0)
        return left, task

    left, task = asyncio.run(scenario())
    assert len(left) == 1
    assert task.cancelled()


# ── a restart with a turn in flight ──────────────────────────────────


def test_a_turn_in_flight_finishes_and_its_reply_goes_out(tmp_path, monkeypatch):
    """The headline: SIGTERM no longer cuts an almost-finished answer off."""
    from d_brain.bot.formatters import send_response

    s = _settings(tmp_path)
    h = Harness(s, monkeypatch, grace=10.0)
    started = asyncio.Event()

    async def handler(message):
        started.set()
        await asyncio.sleep(0.2)
        await send_response(h.bot, message.chat.id, "дописал ответ")

    h.on_message(handler)

    async def scenario():
        await h.start()
        turn = h.feed(_text(1, "посчитай"))
        await asyncio.wait_for(started.wait(), 5)
        await h.stop_and_wait()
        await turn

    _run(scenario())

    # The reply left, the message is signed off, and the process exited
    # through the graceful path.
    assert [text for _chat, text in h.bot.sent] == ["дописал ответ"]
    assert Inbox(s.runtime_dir).pending() == []
    assert h.exits == [shutdown.EXIT_CODE]


def test_a_turn_longer_than_the_deadline_keeps_its_message(tmp_path, monkeypatch):
    """Past the deadline the turn is dropped — but the MESSAGE is not: it is
    still in the inbox, unanswered, and the next boot replays it."""
    from d_brain.bot import main as bot_main
    from d_brain.bot.formatters import send_response

    s = _settings(tmp_path)
    h = Harness(s, monkeypatch, grace=0.1)
    started = asyncio.Event()

    async def handler(message):
        started.set()
        await asyncio.sleep(60)  # never finishes inside the deadline

    h.on_message(handler)

    async def first_boot():
        await h.start()
        turn = h.feed(_text(7, "долгая задача"))
        await asyncio.wait_for(started.wait(), 5)
        await h.stop_and_wait()
        with pytest.raises(asyncio.CancelledError):
            await turn

    _run(first_boot())

    # Nothing was answered, nothing was deleted, nothing was retired.
    box = Inbox(s.runtime_dir)
    assert [e.id for e in box.pending()] == [entry_id(7)]
    assert not box.is_handled(entry_id(7))
    assert list(box.stale_dir.glob("*.json")) == []

    # ── the restart answers it ───────────────────────────────────────
    second = FakeBot()
    dp = Dispatcher()
    answered: list[str] = []
    router = Router(name="after-restart")

    @router.message()
    async def handler_again(message):
        answered.append(message.text)
        await send_response(second, message.chat.id, "доиграл после старта")

    dp.include_router(router)

    async def fake_start_polling(*bots, **kwargs):
        return  # polling "ends" right after the replay, so run_bot returns

    monkeypatch.setattr(dp, "start_polling", fake_start_polling)
    monkeypatch.setattr(bot_main, "create_bot", lambda settings: second)
    monkeypatch.setattr(bot_main, "create_dispatcher", lambda: dp)

    _run(bot_main.run_bot(s))

    assert answered == ["долгая задача"]
    assert [text for _chat, text in second.sent] == ["доиграл после старта"]
    assert Inbox(s.runtime_dir).pending() == []


# ── messages arriving DURING the stop ────────────────────────────────


def test_a_message_during_the_stop_is_told_so_and_replayed(tmp_path, monkeypatch):
    """Not started, not lost, not silent — the three things at once."""
    from d_brain.bot import main as bot_main
    from d_brain.bot.formatters import send_response

    s = _settings(tmp_path)
    h = Harness(s, monkeypatch, grace=10.0)
    handled: list[str] = []

    async def handler(message):
        handled.append(message.text)

    h.on_message(handler)

    async def scenario():
        await h.start()
        assert h.shutdown is not None
        h.shutdown.request("SIGTERM")
        # Polling is deliberately still alive here, so this is exactly what a
        # message arriving mid-restart does: accepted to disk, then refused a
        # turn.
        await h.feed(_text(11, "а это уже во время рестарта"))
        await h.feed(_text(12, "и ещё одно"))
        await asyncio.wait_for(h.task, 30)

    _run(scenario())

    # No turn ran for them.
    assert handled == []
    # One short line, once per chat — not once per message.
    assert [text for _chat, text in h.bot.sent] == [shutdown.NOTICE]
    # Both are still owed an answer: on disk, in order, no receipts.
    box = Inbox(s.runtime_dir)
    assert [e.id for e in box.pending()] == [entry_id(11), entry_id(12)]
    assert not box.is_handled(entry_id(11))
    assert not box.is_handled(entry_id(12))

    # The restart answers it.
    second = FakeBot()
    dp = Dispatcher()
    answered: list[str] = []
    router = Router(name="after-restart")

    @router.message()
    async def handler_again(message):
        answered.append(message.text)
        await send_response(second, message.chat.id, "ответ на отложенное")

    dp.include_router(router)

    async def fake_start_polling(*bots, **kwargs):
        return  # polling "ends" right after the replay, so run_bot returns

    monkeypatch.setattr(dp, "start_polling", fake_start_polling)
    monkeypatch.setattr(bot_main, "create_bot", lambda settings: second)
    monkeypatch.setattr(bot_main, "create_dispatcher", lambda: dp)

    _run(bot_main.run_bot(s))

    # Both replayed, oldest first — the order they arrived in.
    assert answered == ["а это уже во время рестарта", "и ещё одно"]
    assert Inbox(s.runtime_dir).pending() == []


def test_a_stranger_gets_no_restart_notice(tmp_path, monkeypatch):
    """The refusal sits OUTSIDE the auth middleware — it has to, or it would
    delete the message it is protecting — so it does its own auth check."""
    s = _settings(tmp_path)
    h = Harness(s, monkeypatch, grace=10.0)
    h.on_message(lambda message: asyncio.sleep(0))

    async def scenario():
        await h.start()
        assert h.shutdown is not None
        h.shutdown.request("SIGTERM")
        await h.feed(_text(21, "кто ты", user=999))
        await asyncio.wait_for(h.task, 30)

    _run(scenario())

    assert h.bot.sent == []


def test_polling_runs_without_aiograms_own_signal_handling(tmp_path, monkeypatch):
    """aiogram's handler stops polling the instant the signal lands, which is
    the behavior being replaced; and the bot session must outlive polling so
    the final outbox drain still has something to send through."""
    s = _settings(tmp_path)
    h = Harness(s, monkeypatch, grace=1.0)

    async def scenario():
        await h.start()
        await h.stop_and_wait()

    _run(scenario())

    assert h.polling_kwargs.get("handle_signals") is False
    assert h.polling_kwargs.get("close_bot_session") is False


def test_an_idle_stop_does_not_wait(tmp_path, monkeypatch):
    """The grace is only ever PAID when a turn is running. A restart of an
    idle bot — which is most of the six a day — must be instant."""
    s = _settings(tmp_path)
    h = Harness(s, monkeypatch, grace=600.0)

    async def scenario():
        await h.start()
        loop = asyncio.get_running_loop()
        began = loop.time()
        await h.stop_and_wait()
        return loop.time() - began

    took = _run(scenario())
    assert took < 5.0
    assert h.exits == [shutdown.EXIT_CODE]


# ── against aiogram's REAL polling, not a stand-in ───────────────────


class PollingBot(FakeBot):
    """Enough of a Bot for aiogram's real ``start_polling`` to run.

    ``_polling`` calls ``bot.me()`` once and then ``bot(GetUpdates(...))`` in
    a loop; ``_listen_updates`` reads ``bot.session.timeout`` to size its
    request timeout. Nothing else of the Bot API is touched.
    """

    def __init__(self, updates: list[dict] | None = None) -> None:
        super().__init__()
        self.session.timeout = 5
        self._queue = list(updates or [])
        self.get_updates_calls = 0

    async def me(self):
        from aiogram.types import User

        return User(id=self.id, is_bot=True, first_name="probe", username="probe")

    async def __call__(self, method, **kwargs):
        from aiogram.methods import GetUpdates

        if isinstance(method, GetUpdates):
            self.get_updates_calls += 1
            if self._queue:
                raw = self._queue.pop(0)
                return [Update.model_validate(raw, context={"bot": self})]
            await asyncio.sleep(0.02)
            return []
        raise AssertionError(f"unexpected API call: {type(method).__name__}")


def test_the_real_polling_loop_stops_on_the_real_stop_sequence(tmp_path, monkeypatch):
    """Everything else here fakes ``start_polling``, which means aiogram's own
    stop handshake — ``_running_lock``, ``stop_polling``'s ``_stopped_signal``,
    ``emit_shutdown``, ``close_bot_session`` — is never exercised, and a
    rename of ``handle_signals`` would be silently absorbed as workflow data
    (blind review 5). This one runs the real thing.
    """
    from d_brain.bot import main as bot_main
    from d_brain.bot.formatters import send_response

    s = _settings(tmp_path)
    bot = PollingBot([_text(31, "живой опрос")])
    dp = Dispatcher()
    exits: list[int] = []
    answered: list[str] = []
    started = asyncio.Event()
    router = Router(name="real-polling")

    @router.message()
    async def handler(message):
        started.set()
        await asyncio.sleep(0.2)
        answered.append(message.text)
        await send_response(bot, message.chat.id, "ответ из реального polling")

    dp.include_router(router)

    real_shutdown = shutdown.Shutdown
    holder: dict = {}

    def factory(**kwargs):
        kwargs["grace"] = 10.0
        sd = real_shutdown(exit_fn=exits.append, **kwargs)
        holder["sd"] = sd
        return sd

    monkeypatch.setattr(bot_main.shutdown, "Shutdown", factory)
    monkeypatch.setattr(bot_main, "create_bot", lambda settings: bot)
    monkeypatch.setattr(bot_main, "create_dispatcher", lambda: dp)
    monkeypatch.setattr(bot_main, "get_session", lambda settings: IdleSession())

    async def scenario():
        task = asyncio.create_task(bot_main.run_bot(s))
        await asyncio.wait_for(started.wait(), 10)
        holder["sd"].request("SIGTERM")
        await asyncio.wait_for(task, 30)

    _run(scenario())

    # aiogram really polled, really dispatched, and really stopped.
    assert bot.get_updates_calls >= 1
    assert answered == ["живой опрос"]
    assert [text for _chat, text in bot.sent] == ["ответ из реального polling"]
    # The turn in flight finished before the stop completed, and the process
    # left through the graceful path.
    assert exits == [shutdown.EXIT_CODE]
    # close_bot_session=False means run_bot owns the close — and it did it.
    assert bot.session.closed is True
    assert Inbox(s.runtime_dir).pending() == []


# ── the deadline and the unit must agree ─────────────────────────────


def _unit_or_skip(unit: str):
    """The multi-instance system unit is maintainer-only and is not part of
    the shipped distribution (manifest ``exclude``), so in a clean clone the
    file is simply absent. Skipping keeps the check honest where the file
    exists instead of failing where it was never meant to be."""
    path = REPO / unit
    if not path.exists():
        pytest.skip(f"{unit} is not part of this checkout")
    return path


def _timeout_stop_sec(path: Path) -> float:
    text = path.read_text()
    found = re.findall(r"^TimeoutStopSec=(\d+)", text, flags=re.MULTILINE)
    assert len(found) == 1, f"{path}: expected exactly one TimeoutStopSec"
    return float(found[0])


@pytest.mark.parametrize(
    "unit",
    ["deploy/dbrain-bot.service", "deploy/systemd/dbrain-bot@.service"],
)
def test_unit_waits_longer_than_the_application_deadline(unit):
    """If systemd's bound is the shorter one it SIGKILLs the process
    mid-grace and the whole feature buys nothing. Both the live system unit
    and the legacy user unit have to hold.

    The margin counts EVERY bounded step after the grace, not just the final
    drain: stopping polling, cancelling the stragglers and then the drain. An
    earlier version compared against grace + drain only, so a unit set to 315
    would have passed and still been SIGKILLed mid-drain (blind review 6).
    """
    settings = Settings(telegram_bot_token="t", deepgram_api_key="d", _env_file=None)
    needed = settings.shutdown_grace_seconds + shutdown.STOP_OVERHEAD
    assert _timeout_stop_sec(_unit_or_skip(unit)) >= needed


@pytest.mark.parametrize(
    "unit",
    ["deploy/dbrain-bot.service", "deploy/systemd/dbrain-bot@.service"],
)
def test_the_units_say_what_the_code_thinks_they_say(unit):
    """``UNIT_TIMEOUT_STOP_SEC`` is what the startup warning compares the
    configured grace against. If it drifts from the shipped units, the warning
    is about a number nobody uses."""
    assert _timeout_stop_sec(_unit_or_skip(unit)) == shutdown.UNIT_TIMEOUT_STOP_SEC


def test_a_grace_the_unit_cannot_honour_is_warned_about(caplog):
    """An operator raises DBRAIN_SHUTDOWN_GRACE in the same .env systemd reads
    as EnvironmentFile= — and silently gets the old hard kill back. Warn, never
    raise: a stop that is less graceful than intended must not be a bot that
    will not boot."""
    too_big = shutdown.UNIT_TIMEOUT_STOP_SEC - shutdown.STOP_OVERHEAD + 1
    with caplog.at_level("WARNING"):
        settings = Settings(
            telegram_bot_token="t",
            deepgram_api_key="d",
            shutdown_grace_seconds=too_big,
            _env_file=None,
        )
    assert settings.shutdown_grace_seconds == too_big  # accepted, not rejected
    assert "TimeoutStopSec" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        Settings(telegram_bot_token="t", deepgram_api_key="d", _env_file=None)
    assert "TimeoutStopSec" not in caplog.text


def test_the_default_grace_is_minutes_not_hours():
    settings = Settings(telegram_bot_token="t", deepgram_api_key="d", _env_file=None)
    assert 60.0 <= settings.shutdown_grace_seconds <= 600.0


def test_grace_is_configurable(monkeypatch):
    monkeypatch.setenv("DBRAIN_SHUTDOWN_GRACE", "42")
    settings = Settings(telegram_bot_token="t", deepgram_api_key="d", _env_file=None)
    assert settings.shutdown_grace_seconds == 42.0


# ── the auth split is faithful ───────────────────────────────────────


def test_is_authorized_matches_what_the_auth_middleware_used_to_do(tmp_path):
    from d_brain.bot.main import is_authorized

    bot = FakeBot()
    allowed = Update.model_validate(_text(1, user=7), context={"bot": bot})
    stranger = Update.model_validate(_text(2, user=999), context={"bot": bot})
    # A service update carries no from_user; the old middleware let it through
    # (its check was `if user and user.id not in ...`) and so must this one.
    faceless = Update.model_validate(
        {
            "update_id": 3,
            "poll": {
                "id": "p",
                "question": "?",
                "options": [],
                "total_voter_count": 0,
                "is_closed": False,
                "is_anonymous": True,
                "type": "regular",
                "allows_multiple_answers": False,
            },
        },
        context={"bot": bot},
    )

    s = _settings(tmp_path)
    assert is_authorized(s, allowed)
    assert not is_authorized(s, stranger)
    assert is_authorized(s, faceless)

    open_to_all = _settings(tmp_path, allow_all_users=True, allowed_user_ids=[])
    assert is_authorized(open_to_all, stranger)

    nobody = _settings(tmp_path, allow_all_users=False, allowed_user_ids=[])
    assert not is_authorized(nobody, allowed)
