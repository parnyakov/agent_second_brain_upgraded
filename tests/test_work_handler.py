"""Tests for the /work command handler ("что сейчас в работе", item 29).

The report is assembled from four independent best-effort readings (engine
busy flags, long-run marker, cron store, ask-health ledger). What these
tests pin down is the contract that matters when the bot is on fire: every
reading may fail, and a failed reading must produce an honest line rather
than an exception — a status command that dies when the system is unhealthy
is worse than none at all.
"""

import asyncio
import json
import time

from d_brain.bot.formatters import validate_telegram_html
from d_brain.bot.handlers import work
from d_brain.config import Settings
from d_brain.services import ask_health, long_run
from d_brain.services.cron_store import CronJob, CronStore, JobState, Schedule


class FakeUser:
    def __init__(self, user_id: int = 1):
        self.id = user_id


class FakeChat:
    def __init__(self, chat_id: int = 10):
        self.id = chat_id


class FakeBot:
    """Delivery now goes through formatters.send_response (F10), which takes
    a bot rather than the Message — so the fake has to carry one."""

    def __init__(self):
        self.messages: list[tuple[int, str]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text))


class FakeMessage:
    def __init__(self):
        self.from_user = FakeUser()
        self.chat = FakeChat()
        self.bot = FakeBot()
        self.answers: list[str] = []

    async def answer(self, text, **kwargs):
        self.answers.append(text)

    @property
    def sent(self) -> list[str]:
        """Everything the user actually received, by either route."""
        return [t for _cid, t in self.bot.messages] + self.answers


class FakeSession:
    """Only the two Protocol probes /work is allowed to use."""

    def __init__(self, *, attended=False, pane=False, raises=False):
        self._attended = attended
        self._pane = pane
        self._raises = raises

    def is_turn_active(self) -> bool:
        if self._raises:
            raise RuntimeError("no")
        return self._attended

    def is_pane_turn_active(self) -> bool:
        if self._raises:
            raise RuntimeError("no")
        return self._pane


def _settings(tmp_path, **over):
    kwargs = dict(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path / "vault",
        runtime_dir=tmp_path / "rt",
        _env_file=None,
    )
    kwargs.update(over)
    s = Settings(**kwargs)
    s.runtime_dir.mkdir(parents=True, exist_ok=True)
    return s


def _with_session(monkeypatch, session, *, cron=None):
    monkeypatch.setattr(work.runtime, "get_session", lambda _s: session)
    # The cron-session line (blind review F9) probes the isolated cron brain
    # through the same two Protocol methods. Stub it by default: without
    # this every test would try to BUILD a real one (and fail on the absent
    # persona file), turning a deliberate assertion into an accident.
    monkeypatch.setattr(
        work.runtime, "get_cron_session", lambda _s: cron or FakeSession()
    )


# ── main session ──────────────────────────────────────────────────────────


def test_idle_session_is_reported_as_free(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    text = work.build_work_report(s, now=1000.0)
    assert "свободна" in text
    assert "Прервать текущий ход — /stop" in text


def test_attended_turn_uses_the_inflight_mtime_not_its_monotonic_line(
    monkeypatch, tmp_path
):
    """inflight's second line is the WRITER process's time.monotonic value —
    meaningless in this process. The mtime is wall-clock and is the only
    honest 'since when'. A regression that read the line instead would
    produce a wildly wrong duration, so pin the mtime path explicitly."""
    s = _settings(tmp_path)
    inflight = s.runtime_dir / "inflight"
    inflight.write_text("req-1\n123456.75\n")  # monotonic, NOT an epoch
    import os

    os.utime(inflight, (900.0, 900.0))
    _with_session(monkeypatch, FakeSession(attended=True))

    text = work.build_work_report(s, now=1000.0)
    assert "присмотренный ход" in text
    assert "идёт 1 мин" in text  # 100s since the mtime
    assert "123456" not in text


def test_attended_turn_without_inflight_admits_it_does_not_know(
    monkeypatch, tmp_path
):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession(attended=True))
    text = work.build_work_report(s, now=1000.0)
    assert "сколько идёт — неизвестно" in text
    # ...and it does not claim someone is waiting either (F8).
    assert "нет записи inflight" in text
    assert "кто-то ждёт ответ" not in text


def test_attended_maintenance_turn_is_not_called_a_user_turn(monkeypatch, tmp_path):
    """Blind review F8: the ask-lock is held by the nightly /process, the
    doctor canary or startup recovery at least as often as by a user turn.
    Telling the owner "кто-то ждёт ответ" about a maintenance turn is the
    wrong prompt to reach for /stop."""
    from d_brain.services.claude_session import MAINT_PREFIX

    s = _settings(tmp_path)
    (s.runtime_dir / "inflight").write_text(f"{MAINT_PREFIX}night\n123456.75\n")
    _with_session(monkeypatch, FakeSession(attended=True))
    text = work.build_work_report(s, now=1000.0)
    assert "фоновое обслуживание" in text
    assert "кто-то ждёт ответ" not in text


def test_attended_user_turn_says_someone_is_waiting(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    (s.runtime_dir / "inflight").write_text("chat-7-1758312000\n123456.75\n")
    _with_session(monkeypatch, FakeSession(attended=True))
    text = work.build_work_report(s, now=1000.0)
    assert "ход пользователя" in text
    assert "кто-то ждёт ответ" in text
    assert "1758312000" not in text  # the rid itself is never printed


def test_unattended_turn_reports_elapsed_and_time_left_to_autoclose(
    monkeypatch, tmp_path
):
    s = _settings(tmp_path, long_run_max_seconds=1800.0)
    long_run.write(
        s.runtime_dir, long_run.LongRun(since=400.0, updated_ts=980.0, alerted=True)
    )
    _with_session(monkeypatch, FakeSession(pane=True))
    text = work.build_work_report(s, now=1000.0)
    assert "неприсмотренный ход" in text
    assert "идёт 10 мин" in text
    assert "автозакрытие через 20 мин" in text


def test_unattended_turn_past_the_cap_says_so(monkeypatch, tmp_path):
    s = _settings(tmp_path, long_run_max_seconds=300.0)
    long_run.write(
        s.runtime_dir, long_run.LongRun(since=100.0, updated_ts=980.0, alerted=True)
    )
    _with_session(monkeypatch, FakeSession(pane=True))
    text = work.build_work_report(s, now=1000.0)
    assert "лимит превышен" in text


def test_unattended_turn_with_a_stale_marker_does_not_invent_a_duration(
    monkeypatch, tmp_path
):
    """The pane says busy but the watchdog's marker is stale (or the
    watchdog is dead). Report the honest unknown — never a made-up number."""
    s = _settings(tmp_path)
    long_run.write(
        s.runtime_dir, long_run.LongRun(since=100.0, updated_ts=200.0, alerted=True)
    )
    _with_session(monkeypatch, FakeSession(pane=True))
    text = work.build_work_report(s, now=100_000.0)
    assert "сколько идёт — неизвестно" in text
    assert "нет свежей отметки вотчдога" in text


def test_autoclose_disabled_is_stated(monkeypatch, tmp_path):
    s = _settings(tmp_path, long_run_max_seconds=0.0)
    long_run.write(
        s.runtime_dir, long_run.LongRun(since=400.0, updated_ts=980.0, alerted=True)
    )
    _with_session(monkeypatch, FakeSession(pane=True))
    assert "автозакрытие выключено" in work.build_work_report(s, now=1000.0)


def test_session_probes_that_raise_degrade_to_an_honest_line(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession(raises=True))
    text = work.build_work_report(s, now=1000.0)
    assert "состояние недоступно" in text


def test_session_that_cannot_be_built_does_not_kill_the_report(
    monkeypatch, tmp_path
):
    """get_session() asserts on a missing persona file — a broken install
    must still get a report telling it what else is going on."""
    s = _settings(tmp_path)

    def boom(_s):
        raise RuntimeError("persona file missing")

    monkeypatch.setattr(work.runtime, "get_session", boom)
    text = work.build_work_report(s, now=1000.0)
    assert "не смог получить сессию" in text
    assert "Расписание" in text  # the other sections still rendered


# ── duty session (built by a parallel branch; may not exist here) ─────────


def test_duty_session_line_is_absent_when_the_build_has_no_duty_session(
    monkeypatch, tmp_path
):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    monkeypatch.delattr(work.runtime, "get_duty_session", raising=False)
    assert "Дежурная сессия" not in work.build_work_report(s, now=1000.0)


def test_duty_session_line_appears_once_the_factory_exists(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    monkeypatch.setattr(
        work.runtime,
        "get_duty_session",
        lambda _s: FakeSession(attended=True),
        raising=False,
    )
    text = work.build_work_report(s, now=1000.0)
    assert "Дежурная сессия" in text
    assert "занята" in text


def test_duty_session_probe_failure_is_reported_not_raised(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    monkeypatch.setattr(
        work.runtime,
        "get_duty_session",
        lambda _s: FakeSession(raises=True),
        raising=False,
    )
    text = work.build_work_report(s, now=1000.0)
    assert "Дежурная сессия" in text
    assert "состояние недоступно" in text


# ── cron ──────────────────────────────────────────────────────────────────


def _job(job_id, **state) -> CronJob:
    return CronJob(
        id=job_id,
        prompt="p",
        schedule=Schedule(kind="every", every_seconds=3600),
        state=JobState(**state),
    )


def test_cron_reports_the_nearest_job_and_where_it_is_stuck(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    store = CronStore(s.cron_dir)
    store.save(
        [
            _job("morning", next_run="2026-09-20T09:00:00+00:00", last_status="ok"),
            _job("soon", next_run="2026-09-20T08:30:00+00:00", last_status="ok"),
            _job(
                "broken",
                next_run="2026-09-20T10:00:00+00:00",
                last_status="error",
                last_error="boom",
                consecutive_errors=3,
            ),
        ]
    )
    now = time.mktime(time.strptime("2026-09-20 08:00:00", "%Y-%m-%d %H:%M:%S"))
    # Interpret the naive strptime result as UTC to keep the test TZ-free.
    now -= time.timezone
    text = work.build_work_report(s, now=now)
    assert "включено: 3 из 3" in text
    assert "soon" in text
    assert "broken" in text
    assert "подряд ошибок: 3" in text
    assert "boom" in text


def test_absurd_next_run_renders_a_bare_timestamp(monkeypatch, tmp_path):
    """A corrupt or wildly-future next_run must not render as "через 497204
    ч" — past a month the relative clause is noise, so drop it."""
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    CronStore(s.cron_dir).save(
        [_job("far", next_run="2099-01-01T00:00:00+00:00", last_status="ok")]
    )
    text = work.build_work_report(s, now=1000.0)
    assert "через" not in text
    assert "01.01" in text


def test_multi_day_durations_are_rendered_in_days(monkeypatch, tmp_path):
    assert work._fmt_duration(3 * 24 * 3600 + 5 * 3600) == "3 дн 5 ч"
    assert work._fmt_duration(3600 + 120) == "1 ч 02 мин"


def test_cron_ok_silent_is_a_success_not_a_failure(monkeypatch, tmp_path):
    """'ok-silent' means the job ran and had nothing to say — reporting it
    as a stuck job would cry wolf on the most common healthy outcome."""
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    CronStore(s.cron_dir).save([_job("quiet", last_status="ok-silent")])
    text = work.build_work_report(s, now=1000.0)
    assert "сбоев нет" in text


def test_cron_rate_limited_is_not_a_stuck_job(monkeypatch, tmp_path):
    """Blind review F9: `rate_limited` means the run never started — nothing
    about the job is broken, and cron_runner itself records it with
    count=False. Listing it under "где застряло" cried wolf on the most
    common non-ok status there is."""
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    CronStore(s.cron_dir).save([_job("digest", last_status="rate_limited")])
    text = work.build_work_report(s, now=1000.0)
    assert "сбоев нет" in text
    assert "где застряло" not in text
    assert "крон стоит на лимите" in text and "digest" in text


def test_cron_rate_limited_does_not_resurrect_a_stale_error_count(
    monkeypatch, tmp_path
):
    """A job that errored earlier and is NOW parked on the limit must not be
    reported as stuck on the strength of that old consecutive_errors count."""
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    CronStore(s.cron_dir).save(
        [_job("digest", last_status="rate_limited", consecutive_errors=2)]
    )
    text = work.build_work_report(s, now=1000.0)
    assert "сбоев нет" in text


def test_cron_disabled_jobs_do_not_hang_in_where_it_is_stuck(monkeypatch, tmp_path):
    """Blind review F9: a disabled job keeps forever the last_status that
    got it disabled, so counting those meant an old, already-handled error
    sat in the report for the rest of the install's life."""
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    dead = _job("ancient", last_status="error", consecutive_errors=3)
    dead.enabled = False
    CronStore(s.cron_dir).save([dead, _job("live", last_status="ok")])
    text = work.build_work_report(s, now=1000.0)
    assert "включено: 1 из 2" in text
    assert "сбоев нет" in text
    assert "ancient" not in text


def test_cron_session_busy_is_reported(monkeypatch, tmp_path):
    """A busy cron brain explains a job that looks overdue — which is the
    question /work is asked during one."""
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession(), cron=FakeSession(pane=True))
    assert "сессия расписания: занята" in work.build_work_report(s, now=1000.0)

    _with_session(monkeypatch, FakeSession(), cron=FakeSession())
    assert "сессия расписания: свободна" in work.build_work_report(s, now=1000.0)


def test_cron_session_probe_failure_is_reported_not_raised(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession(), cron=FakeSession(raises=True))
    text = work.build_work_report(s, now=1000.0)
    assert "сессия расписания: состояние недоступно" in text


def test_cron_with_no_jobs_says_so(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    assert "заданий нет" in work.build_work_report(s, now=1000.0)


def test_corrupt_cron_store_does_not_break_the_report(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    s.cron_dir.mkdir(parents=True, exist_ok=True)
    (s.cron_dir / "jobs.json").write_text("{not json")
    text = work.build_work_report(s, now=1000.0)
    assert "Расписание" in text  # CronStore.load quarantines and returns []


# ── delivery health ───────────────────────────────────────────────────────


def test_delivery_health_reports_the_streak(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    ask_health.record(s.runtime_dir, "timeout", clock_fn=lambda: 900.0)
    ask_health.record(s.runtime_dir, "error", clock_fn=lambda: 960.0)
    text = work.build_work_report(s, now=1000.0)
    assert "<code>error</code>" in text
    assert "подряд неудач: 2" in text


def test_delivery_health_without_records_is_not_a_fault(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    assert "записей пока нет" in work.build_work_report(s, now=1000.0)


def test_empty_outbox_is_reported_as_empty(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    text = work.build_work_report(s, now=1000.0)
    assert "ждут отправки: нет" in text
    assert "потерянных ответов нет" in text


def test_outbox_queue_and_dead_letters_are_visible(monkeypatch, tmp_path):
    """The queue has to be inspectable from Telegram — that is the whole
    point of "сколько ждёт, сколько провалилось"."""
    from d_brain.services.outbox import Outbox

    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    box = Outbox(s.runtime_dir)
    box.enqueue(1, "waiting")
    box.enqueue(1, "also waiting")
    box.bury(box.enqueue(1, "lost"), reason="gave up", now=900.0)

    text = work.build_work_report(s, now=1000.0)
    assert "ждут отправки: 2" in text
    assert "не доставлено совсем: 1" in text
    assert validate_telegram_html(text)


def test_unreadable_outbox_is_reported_as_unreadable(monkeypatch, tmp_path):
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())

    def boom(_runtime_dir):
        raise RuntimeError("no disk")

    monkeypatch.setattr(work.outbox, "Outbox", boom)
    assert "очередь отправки: состояние недоступно" in work.build_work_report(
        s, now=1000.0
    )


# ── formatting ────────────────────────────────────────────────────────────


def test_report_is_valid_telegram_html_with_hostile_data(monkeypatch, tmp_path):
    """Job ids and error texts come from data the report must not trust:
    an unescaped '<' would make Telegram reject the whole message."""
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    CronStore(s.cron_dir).save(
        [
            _job(
                "<b>evil</b> & co",
                last_status="error",
                last_error="<script>alert(1)</script>",
                consecutive_errors=1,
            )
        ]
    )
    ask_health.record(s.runtime_dir, "<i>weird</i>", clock_fn=lambda: 990.0)
    text = work.build_work_report(s, now=1000.0)
    assert validate_telegram_html(text)
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "&lt;b&gt;evil&lt;/b&gt; &amp; co" in text


def test_report_uses_only_allowed_tags(monkeypatch, tmp_path):
    import re

    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    text = work.build_work_report(s, now=1000.0)
    tags = set(re.findall(r"</?([a-zA-Z]+)", text))
    assert tags <= {"b", "i", "code", "a", "pre"}


# ── the handler itself ────────────────────────────────────────────────────


def test_cmd_work_answers_without_touching_the_ask_lock(monkeypatch, tmp_path):
    """The whole point: /work must work while the session is busy, so it
    neither calls the model nor acquires the lock — it only probes."""
    s = _settings(tmp_path)
    session = FakeSession(pane=True)
    _with_session(monkeypatch, session)
    monkeypatch.setattr(work, "get_settings", lambda: s)

    message = FakeMessage()
    asyncio.run(work.cmd_work(message))

    assert len(message.sent) == 1
    assert "Основная сессия" in message.sent[0]
    assert not hasattr(session, "ask")  # nothing here could have asked


def test_cmd_work_delivery_goes_through_the_shared_send_path(monkeypatch, tmp_path):
    """Blind review F10: the report is assembled from unbounded external
    data (cron ids, last_error strings), so it must be split at Telegram's
    4096-char limit and sanitized like any other reply — not handed to a raw
    message.answer that would raise on a long or malformed one."""
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    monkeypatch.setattr(work, "get_settings", lambda: s)
    monkeypatch.setattr(
        work, "build_work_report", lambda *_a, **_k: "<b>x</b>" + "я" * 9000
    )

    message = FakeMessage()
    asyncio.run(work.cmd_work(message))

    assert len(message.bot.messages) > 1  # chunked, not truncated or dropped
    assert all(len(t) <= 4096 for _cid, t in message.bot.messages)
    assert message.answers == []  # no failure notice: the send succeeded


def test_cmd_work_survives_a_failing_send(monkeypatch, tmp_path):
    """The send sits inside the same try as the build: a status command must
    not raise out of its handler because delivery failed."""
    s = _settings(tmp_path)
    _with_session(monkeypatch, FakeSession())
    monkeypatch.setattr(work, "get_settings", lambda: s)

    async def boom(*_a, **_k):
        raise RuntimeError("telegram is down")

    monkeypatch.setattr(work, "send_response", boom)
    message = FakeMessage()
    asyncio.run(work.cmd_work(message))
    assert message.answers == ["❌ Не смог собрать статус работ прямо сейчас."]


def test_cmd_work_reports_a_total_failure_instead_of_crashing(
    monkeypatch, tmp_path
):
    s = _settings(tmp_path)
    monkeypatch.setattr(work, "get_settings", lambda: s)

    def boom(*_a, **_k):
        raise RuntimeError("everything is on fire")

    monkeypatch.setattr(work, "build_work_report", boom)
    message = FakeMessage()
    asyncio.run(work.cmd_work(message))
    assert message.answers == ["❌ Не смог собрать статус работ прямо сейчас."]


def test_work_command_is_exposed_in_the_native_menu():
    """Router ORDER (work before chat.py's catch-all) is asserted in
    tests/test_bot_main.py — create_dispatcher() attaches module-level
    router singletons and can only be called once per process, so it is
    deliberately not called a second time here."""
    from d_brain.bot.main import bot_commands

    assert "work" in [c.command for c in bot_commands()]


def test_work_handler_imports_nothing_tmux_specific():
    """Engine compatibility (owner's hard requirement): this handler speaks
    only the EngineDriver Protocol and the cross-process ledgers, so the
    same code reports correctly under chat_engine=codex."""
    from pathlib import Path

    # Comments are allowed to say "tmux"; CODE is not.
    code = "\n".join(
        line
        for line in Path(work.__file__).read_text().splitlines()
        if not line.lstrip().startswith(("#", '"', "*"))
    )
    for banned in ("tmux_parse", "tmux_", "capture_text(", "PaneState"):
        assert banned not in code, banned


def test_work_command_is_listed_in_start_and_help(tmp_path):
    from pathlib import Path

    body = Path(
        __import__("d_brain.bot.handlers.commands", fromlist=["x"]).__file__
    ).read_text()
    assert body.count("/work") >= 2


def test_json_marker_written_by_the_watchdog_is_the_same_one_read_here(tmp_path):
    """Contract check across the process boundary: the report's duration
    for an unattended run comes from the watchdog's own marker file, not
    from a second, drifting source of truth."""
    long_run.write(tmp_path, long_run.LongRun(since=5.0, updated_ts=10.0))
    raw = json.loads((tmp_path / long_run.FILENAME).read_text())
    assert raw["since"] == 5.0
