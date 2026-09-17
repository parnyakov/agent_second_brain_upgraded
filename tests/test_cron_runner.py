"""Tests for the cron runner: ticker semantics, delivery, self-healing.

FakeSession pattern as in test_claude_session.py — no tmux, scripted
AskResults, recorder callables for deliver/alert, a manual clock.
"""

from datetime import UTC, datetime, timedelta

from d_brain.services.claude_session import AskResult
from d_brain.services.cron_runner import SILENT_MARKER, CronRunner, wrap_job_prompt
from d_brain.services.cron_store import CronJob, CronStore, Schedule

NOW = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)


class FakeSession:
    def __init__(
        self,
        results=None,
        on_ask=None,
        *,
        pane_text="",
        turn_active=False,
    ):
        self.results = list(results or [])
        self.on_ask = on_ask
        self.asked: list[str] = []
        self.controls: list[str] = []
        self.recovered = 0
        self.request_ids: list[str | None] = []
        self.pane_text = pane_text
        self.turn_active = turn_active

    def ask(self, prompt, *, timeout=0.0, wrap=True, request_id=None):
        self.asked.append(prompt)
        self.request_ids.append(request_id)
        if self.on_ask:
            self.on_ask()
        return self.results.pop(0) if self.results else AskResult("ok", reply="done")

    def send_control(self, text):
        self.controls.append(text)

    def force_recover(self):
        self.recovered += 1
        return True

    def capture_text(self):
        return self.pane_text

    def is_turn_active(self):
        return self.turn_active


class Recorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, *args):
        self.calls.append(args)


def _store(tmp_path):
    return CronStore(tmp_path / "cron")


def _add_job(store, job_id="j1", *, kind="every", next_run=NOW, **over):
    if kind == "every":
        schedule = Schedule(kind="every", every_seconds=3600)
    elif kind == "at":
        schedule = Schedule(kind="at", at=next_run.isoformat())
    else:
        schedule = Schedule(kind="cron", expr="0 9 * * *", tz="UTC")
    base = dict(id=job_id, prompt="do it", schedule=schedule)
    base.update(over)
    job = CronJob(**base)
    job.state.next_run = next_run.isoformat()
    store.mutate(lambda jobs: jobs.append(job))
    return job


def _runner(store, session, *, deliver=None, alert=None, clock=None, **over):
    base = dict(
        deliver=deliver or Recorder(),
        alert=alert or Recorder(),
        default_chat_id=111,
        job_timeout=10.0,
        max_consecutive_errors=3,
        retry_seconds=300.0,
        clock=clock or (lambda: NOW),
    )
    base.update(over)
    return CronRunner(store, session, **base)


# ── ticker semantics ─────────────────────────────────────────────────


async def test_due_job_runs_and_delivers(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "j1")
    deliver = Recorder()
    session = FakeSession([AskResult("ok", reply="<b>done</b>")])
    await _runner(store, session, deliver=deliver).tick()
    assert len(session.asked) == 1
    assert "do it" in session.asked[0]
    assert deliver.calls == [(111, "<b>done</b>")]


async def test_future_and_disabled_jobs_skipped(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "future", next_run=NOW + timedelta(hours=1))
    _add_job(store, "off", enabled=False)
    session = FakeSession()
    await _runner(store, session).tick()
    assert session.asked == []


async def test_next_run_persisted_before_ask(tmp_path):
    """At-most-once: a crash mid-ask must not refire the same slot."""
    store = _store(tmp_path)
    _add_job(store, "j1")
    seen = {}

    def snapshot():
        seen["next_run"] = store.load()[0].state.next_run

    session = FakeSession([AskResult("ok", reply="x")], on_ask=snapshot)
    await _runner(store, session).tick()
    assert seen["next_run"] == (NOW + timedelta(hours=1)).isoformat()


async def test_every_job_reschedules_after_run(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "j1")
    await _runner(store, FakeSession([AskResult("ok", reply="x")])).tick()
    job = store.load()[0]
    assert job.state.next_run == (NOW + timedelta(hours=1)).isoformat()
    assert job.state.last_status == "ok"
    assert job.state.last_run == NOW.isoformat()


async def test_hot_reload_picks_up_jobs_added_between_ticks(tmp_path):
    store = _store(tmp_path)
    session = FakeSession([AskResult("ok", reply="x")])
    runner = _runner(store, session)
    await runner.tick()
    assert session.asked == []
    _add_job(store, "late")  # e.g. brain CLI while the bot is running
    await runner.tick()
    assert len(session.asked) == 1


# ── delivery ─────────────────────────────────────────────────────────


async def test_job_chat_id_overrides_default(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "j1", chat_id=999)
    deliver = Recorder()
    await _runner(
        store, FakeSession([AskResult("ok", reply="x")]), deliver=deliver
    ).tick()
    assert deliver.calls == [(999, "x")]


async def test_silent_reply_suppresses_delivery(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "j1")
    deliver = Recorder()
    reply = f"{SILENT_MARKER} nothing new"
    await _runner(
        store, FakeSession([AskResult("ok", reply=reply)]), deliver=deliver
    ).tick()
    assert deliver.calls == []
    assert store.load()[0].state.last_status == "ok-silent"


async def test_clear_sent_after_successful_run(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "j1")
    session = FakeSession([AskResult("ok", reply="x")])
    await _runner(store, session).tick()
    assert "/clear" in session.controls


# ── one-shot lifecycle ───────────────────────────────────────────────


async def test_oneshot_deleted_after_success(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "once", kind="at", delete_after_run=True)
    await _runner(store, FakeSession([AskResult("ok", reply="x")])).tick()
    assert store.load() == []


async def test_oneshot_error_kept_with_retry(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "once", kind="at", delete_after_run=True)
    await _runner(store, FakeSession([AskResult("timeout")])).tick()
    jobs = store.load()
    assert len(jobs) == 1
    assert jobs[0].state.next_run == (NOW + timedelta(seconds=300)).isoformat()
    assert jobs[0].state.last_status == "timeout"


# ── self-healing ─────────────────────────────────────────────────────


async def test_error_recovers_session_and_counts(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "j1")
    session = FakeSession([AskResult("error", detail="boom")])
    await _runner(store, session).tick()
    assert session.recovered == 1
    job = store.load()[0]
    assert job.state.consecutive_errors == 1
    assert job.state.last_error == "boom"
    assert job.enabled is True


async def test_third_consecutive_error_disables_and_alerts(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "j1")

    def set_errors(jobs):
        jobs[0].state.consecutive_errors = 2

    store.mutate(set_errors)
    alert = Recorder()
    await _runner(store, FakeSession([AskResult("error")]), alert=alert).tick()
    job = store.load()[0]
    assert job.enabled is False
    assert job.state.consecutive_errors == 3
    assert len(alert.calls) == 1
    assert "j1" in alert.calls[0][0]


async def test_success_resets_error_counter(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "j1")

    def set_errors(jobs):
        jobs[0].state.consecutive_errors = 2

    store.mutate(set_errors)
    await _runner(store, FakeSession([AskResult("ok", reply="x")])).tick()
    assert store.load()[0].state.consecutive_errors == 0


async def test_rate_limited_skips_without_recover_or_count(tmp_path):
    """Subscription limit is not the job's fault: wait for the next slot."""
    store = _store(tmp_path)
    _add_job(store, "j1")
    session = FakeSession([AskResult("rate_limited")])
    await _runner(store, session).tick()
    job = store.load()[0]
    assert session.recovered == 0
    assert job.state.consecutive_errors == 0
    assert job.state.last_status == "rate_limited"
    assert "/clear" not in session.controls


# ── rate-limit self-recovery (Fix A/B, agent-infra-backlog item 23) ───
#
# 2026-08-28 incident: the cron pane got permanently parked at a stale
# rate-limit banner for 6 days — nothing ever sent it input to re-check
# the limit, because the cron session has no watchdog of its own (see
# runtime.py) and ask() refuses to type over a confirmed RATE_LIMITED
# pane. Fix A gives CronRunner its own recovery: track when the limit was
# first seen, and once past a deadline (the banner's own reset time, or a
# LIMIT_MAX_WAIT fallback when unreadable), send /clear to re-check and
# unstick it — cooling down between attempts. Fix B adds one escalating
# admin alert if it's been stuck a long time.

_LIMIT_BANNER = "  You've hit your session limit · resets 12am (UTC)\n❯\n"


def _write_state_file(tmp_path, name, value):
    d = tmp_path / "cron"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(f"{value}\n")


def _read_state_file(tmp_path, name):
    return (tmp_path / "cron" / name).read_text()


def _limit_runner(
    tmp_path, *, clock, pane_text=_LIMIT_BANNER, turn_active=False, alert=None
):
    store = _store(tmp_path)
    session = FakeSession(pane_text=pane_text, turn_active=turn_active)
    runner = _runner(store, session, clock=lambda: clock["now"], alert=alert)
    return store, session, runner


async def test_rate_limited_run_creates_limited_since(tmp_path):
    store = _store(tmp_path)
    job = _add_job(store, "j1")
    clock = {"now": NOW}
    runner = _runner(
        store, FakeSession([AskResult("rate_limited")]), clock=lambda: clock["now"]
    )
    await runner.run_job(job)
    assert _read_state_file(tmp_path, "limited_since") == f"{NOW.timestamp()}\n"


async def test_second_rate_limited_run_does_not_overwrite_limited_since(tmp_path):
    """The anchor is set on the FIRST hit only — a repeat hit must not push
    the recovery deadline (and Fix B's alert clock) further out."""
    store = _store(tmp_path)
    job = _add_job(store, "j1")
    clock = {"now": NOW}
    session = FakeSession([AskResult("rate_limited"), AskResult("rate_limited")])
    runner = _runner(store, session, clock=lambda: clock["now"])
    await runner.run_job(job)
    first = _read_state_file(tmp_path, "limited_since")
    clock["now"] = NOW + timedelta(hours=1)
    await runner.run_job(job)
    second = _read_state_file(tmp_path, "limited_since")
    assert first == second == f"{NOW.timestamp()}\n"


async def test_ok_run_clears_limit_state_and_resets_clear_attempts(tmp_path):
    store = _store(tmp_path)
    job = _add_job(store, "j1")
    _write_state_file(tmp_path, "limited_since", 123.0)
    _write_state_file(tmp_path, "limit_alert_ts", 456.0)
    runner = _runner(store, FakeSession([AskResult("ok", reply="x")]))
    runner._clear_attempts = 3
    await runner.run_job(job)
    assert not (tmp_path / "cron" / "limited_since").exists()
    assert not (tmp_path / "cron" / "limit_alert_ts").exists()
    assert runner._clear_attempts == 0


async def test_limit_recovery_before_due_sends_no_clear(tmp_path):
    since = NOW
    clock = {"now": since}
    tmp_store, session, runner = _limit_runner(tmp_path, clock=clock)
    _write_state_file(tmp_path, "limited_since", since.timestamp())
    await runner.tick()
    assert session.controls == []


async def test_limit_recovery_sends_exactly_one_clear_once_due(tmp_path):
    """_LIMIT_BANNER's own reset (12am UTC, 12h after `since`=noon) is
    farther out than the 6h LIMIT_MAX_WAIT fallback, so the fallback is
    what actually governs `due` here — matching the real incident, where
    the banner's reset time was unreadable at all."""
    since = NOW
    clock = {"now": since}
    tmp_store, session, runner = _limit_runner(tmp_path, clock=clock)
    _write_state_file(tmp_path, "limited_since", since.timestamp())
    clock["now"] = since + timedelta(hours=6)  # exactly at LIMIT_MAX_WAIT
    await runner.tick()
    assert session.controls == ["/clear"]
    assert runner._clear_attempts == 1


async def test_limit_recovery_noop_when_banner_gone(tmp_path):
    """If the pane no longer classifies as RATE_LIMITED, _limit_recovery
    must not touch anything — the next successful ask() (not this method)
    is what clears the limited_since anchor."""
    since = NOW
    clock = {"now": since + timedelta(hours=7)}  # well past due
    tmp_store, session, runner = _limit_runner(tmp_path, clock=clock, pane_text="❯\n")
    _write_state_file(tmp_path, "limited_since", since.timestamp())
    await runner.tick()
    assert session.controls == []
    assert (tmp_path / "cron" / "limited_since").exists()  # untouched


async def test_limit_recovery_defers_while_a_live_turn_holds_the_pane(tmp_path):
    since = NOW
    clock = {"now": since + timedelta(hours=6)}
    tmp_store, session, runner = _limit_runner(tmp_path, clock=clock, turn_active=True)
    _write_state_file(tmp_path, "limited_since", since.timestamp())
    await runner.tick()
    assert session.controls == []
    assert runner._clear_attempts == 0


async def test_limit_recovery_cooldown_blocks_immediate_second_clear(tmp_path):
    since = NOW
    clock = {"now": since}
    tmp_store, session, runner = _limit_runner(tmp_path, clock=clock)
    _write_state_file(tmp_path, "limited_since", since.timestamp())
    clock["now"] = since + timedelta(hours=6)
    await runner.tick()
    assert session.controls == ["/clear"]
    clock["now"] += timedelta(seconds=1)
    await runner.tick()
    assert session.controls == ["/clear"]  # cooldown blocked the second


async def test_limit_recovery_cooldown_doubles_up_to_the_cap(tmp_path):
    """Cooldown between recovery /clear attempts doubles each time:
    600s -> 1200s -> 2400s -> capped at 3600s (CLEAR_COOLDOWN_MAX). Checks
    the actual boundary math, not just "a second attempt is blocked"."""
    since = NOW
    clock = {"now": since}
    tmp_store, session, runner = _limit_runner(tmp_path, clock=clock)
    _write_state_file(tmp_path, "limited_since", since.timestamp())

    clock["now"] = since + timedelta(hours=6)  # attempt 1: due
    await runner.tick()
    assert session.controls == ["/clear"]
    assert runner._clear_attempts == 1
    t1 = clock["now"]

    clock["now"] = t1 + timedelta(seconds=1199)  # cooldown for #2 is 1200s
    await runner.tick()
    assert session.controls == ["/clear"]  # still blocked

    clock["now"] = t1 + timedelta(seconds=1200)
    await runner.tick()
    assert session.controls == ["/clear", "/clear"]
    assert runner._clear_attempts == 2
    t2 = clock["now"]

    clock["now"] = t2 + timedelta(seconds=2399)  # cooldown for #3 is 2400s
    await runner.tick()
    assert session.controls == ["/clear", "/clear"]  # still blocked

    clock["now"] = t2 + timedelta(seconds=2400)
    await runner.tick()
    assert session.controls == ["/clear", "/clear", "/clear"]
    assert runner._clear_attempts == 3
    t3 = clock["now"]

    clock["now"] = t3 + timedelta(seconds=3599)  # #4 cooldown 4800s capped at 3600s
    await runner.tick()
    assert session.controls == ["/clear", "/clear", "/clear"]  # still blocked

    clock["now"] = t3 + timedelta(seconds=3600)
    await runner.tick()
    assert session.controls == ["/clear", "/clear", "/clear", "/clear"]


async def test_limit_alert_fires_once_after_6h_not_before(tmp_path):
    since = NOW
    clock = {"now": since + timedelta(hours=5, minutes=59)}
    alerts = Recorder()
    # turn_active=True isolates this test to the alert path — no /clear
    # sent, so nothing else touches `alerts`.
    tmp_store, session, runner = _limit_runner(
        tmp_path, clock=clock, turn_active=True, alert=alerts
    )
    _write_state_file(tmp_path, "limited_since", since.timestamp())
    await runner.tick()
    assert alerts.calls == []  # not yet 6h

    clock["now"] = since + timedelta(hours=6)
    await runner.tick()
    assert len(alerts.calls) == 1
    assert "6" in alerts.calls[0][0]


async def test_limit_alert_does_not_repeat_inside_24h(tmp_path):
    since = NOW
    clock = {"now": since + timedelta(hours=6)}
    alerts = Recorder()
    tmp_store, session, runner = _limit_runner(
        tmp_path, clock=clock, turn_active=True, alert=alerts
    )
    _write_state_file(tmp_path, "limited_since", since.timestamp())
    await runner.tick()
    assert len(alerts.calls) == 1

    clock["now"] = since + timedelta(hours=20)  # 14h after the first alert
    await runner.tick()
    assert len(alerts.calls) == 1  # unchanged — inside the 24h repeat window


async def test_limit_alert_fires_again_after_24h(tmp_path):
    since = NOW
    clock = {"now": since + timedelta(hours=6)}
    alerts = Recorder()
    tmp_store, session, runner = _limit_runner(
        tmp_path, clock=clock, turn_active=True, alert=alerts
    )
    _write_state_file(tmp_path, "limited_since", since.timestamp())
    await runner.tick()
    assert len(alerts.calls) == 1

    clock["now"] = since + timedelta(hours=30)  # 24h after the first alert
    await runner.tick()
    assert len(alerts.calls) == 2


async def test_limit_alert_already_alerted_state_survives_a_fresh_runner(tmp_path):
    """The "already alerted" flag lives in a file (limit_alert_ts), not in
    the CronRunner instance — a fresh instance pointed at the same
    cron_dir must still respect the repeat window."""
    since = NOW
    store = _store(tmp_path)
    _write_state_file(tmp_path, "limited_since", since.timestamp())
    clock = {"now": since + timedelta(hours=6)}

    alerts1 = Recorder()
    session1 = FakeSession(pane_text=_LIMIT_BANNER, turn_active=True)
    runner1 = _runner(store, session1, clock=lambda: clock["now"], alert=alerts1)
    await runner1.tick()
    assert len(alerts1.calls) == 1

    alerts2 = Recorder()
    session2 = FakeSession(pane_text=_LIMIT_BANNER, turn_active=True)
    runner2 = _runner(store, session2, clock=lambda: clock["now"], alert=alerts2)
    clock["now"] = since + timedelta(hours=10)  # still within 24h of the first alert
    await runner2.tick()
    assert alerts2.calls == []


# ── crash-safety (blind-review findings) ─────────────────────────────


async def test_at_claim_rearms_retry_instead_of_none(tmp_path):
    """A bot restart between claim and record must not brick a one-shot:
    claim leaves a retry next_run, success/record clears or deletes it."""
    store = _store(tmp_path)
    _add_job(store, "once", kind="at", delete_after_run=True)
    runner = _runner(store, FakeSession())
    claimed = runner.claim_due(NOW)
    assert len(claimed) == 1
    persisted = store.load()[0]
    assert persisted.state.next_run == (NOW + timedelta(seconds=300)).isoformat()


async def test_oneshot_success_clears_next_run_when_kept(tmp_path):
    """kind=at without delete_after_run must not refire after success."""
    store = _store(tmp_path)
    _add_job(store, "once", kind="at", delete_after_run=False)
    await _runner(store, FakeSession([AskResult("ok", reply="x")])).tick()
    job = store.load()[0]
    assert job.state.next_run is None
    assert job.state.last_status == "ok"


async def test_deliver_crash_counts_as_failure_and_clears(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "once", kind="at", delete_after_run=True)

    async def bad_deliver(*args):
        raise RuntimeError("telegram down")

    session = FakeSession([AskResult("ok", reply="x")])
    await _runner(store, session, deliver=bad_deliver).tick()
    jobs = store.load()
    assert len(jobs) == 1  # NOT deleted — will retry
    assert jobs[0].state.last_status == "deliver_error"
    assert jobs[0].state.consecutive_errors == 1
    assert "/clear" in session.controls  # context still dropped


async def test_tick_survives_job_crash_and_runs_rest_of_batch(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "boom")
    _add_job(store, "fine")

    class ExplodingRecover(FakeSession):
        def force_recover(self):
            raise RuntimeError("session not ready")

    session = ExplodingRecover(
        [AskResult("error", detail="x"), AskResult("ok", reply="y")]
    )
    deliver = Recorder()
    await _runner(store, session, deliver=deliver).tick()
    assert len(session.asked) == 2  # second job still ran
    boom = next(j for j in store.load() if j.id == "boom")
    assert boom.state.last_status == "crash"
    assert boom.state.consecutive_errors == 1


async def test_claim_skips_malformed_next_run_without_killing_tick(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "bad")
    _add_job(store, "good")

    def corrupt(jobs):
        jobs[0].state.next_run = "definitely not a date"

    store.mutate(corrupt)
    session = FakeSession([AskResult("ok", reply="x")])
    await _runner(store, session).tick()
    assert len(session.asked) == 1  # good ran, bad skipped, no exception


async def test_claim_treats_naive_next_run_as_utc(tmp_path):
    store = _store(tmp_path)
    _add_job(store, "naive")

    def make_naive(jobs):
        jobs[0].state.next_run = NOW.replace(tzinfo=None).isoformat()

    store.mutate(make_naive)
    session = FakeSession([AskResult("ok", reply="x")])
    await _runner(store, session).tick()
    assert len(session.asked) == 1


def test_load_tolerates_concurrent_rename(tmp_path):
    """FileNotFoundError between exists() and read must not escape load()."""
    from d_brain.services.cron_store import CronStore as CS

    store = CS(tmp_path / "cron")
    store.save([])
    # Simulate the race directly: file vanishes after exists() check
    store.jobs_file.unlink()
    real_exists = type(store.jobs_file).exists
    try:
        type(store.jobs_file).exists = lambda self: True  # type: ignore[method-assign]
        assert store.load() == []
    finally:
        type(store.jobs_file).exists = real_exists  # type: ignore[method-assign]


# ── prompt wrapper ───────────────────────────────────────────────────


def test_wrap_job_prompt_contract():
    wrapped = wrap_job_prompt("j1", "check the inbox", scheduled_for="2026-06-10")
    assert "[CRON JOB j1]" in wrapped
    assert "check the inbox" in wrapped
    assert "2026-06-10" in wrapped
    assert SILENT_MARKER in wrapped
    # recursion guard: a scheduled run must not breed more jobs
    assert "d_brain.cron" in wrapped
    # marker instruction belongs to ask(wrap=True), never duplicated here
    assert "<<<R:" not in wrapped


def test_dormant_recurring_job_warns_once(tmp_path, caplog):
    # A recurring job with next_run=None can never fire again — that is
    # a broken state (hand-edited jobs.json), not a completed one-shot.
    # Surface it once, do not spam every tick.
    store = _store(tmp_path)
    _add_job(store, "stuck")
    store.mutate(lambda jobs: setattr(jobs[0].state, "next_run", None))
    runner = _runner(store, FakeSession([]))

    with caplog.at_level("WARNING"):
        assert runner.claim_due(NOW) == []
        assert runner.claim_due(NOW) == []

    warnings = [r for r in caplog.records if "stuck" in r.getMessage()]
    assert len(warnings) == 1


def test_completed_one_shot_does_not_warn(tmp_path, caplog):
    # at-jobs legitimately end with next_run=None after success.
    store = _store(tmp_path)
    _add_job(store, "done", kind="at")
    store.mutate(lambda jobs: setattr(jobs[0].state, "next_run", None))
    runner = _runner(store, FakeSession([]))

    with caplog.at_level("WARNING"):
        assert runner.claim_due(NOW) == []

    assert [r for r in caplog.records if "done" in r.getMessage()] == []
