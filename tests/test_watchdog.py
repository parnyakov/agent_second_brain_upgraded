"""Tests for the liveness watchdog.

Hang model (v3.0): a session is hung when its pane state is NOT serviceable
(not READY/RATE_LIMITED/LOGGED_OUT) AND the pane shows no active work (no
'esc to interrupt' spinner) PERSISTENTLY for stall_threshold. Silence is not
a signal: a long task that prints nothing still shows the working spinner and
must never be killed.
"""

from datetime import datetime

from d_brain.services.tmux_parse import PaneState
from d_brain.services.watchdog import Watchdog


class FakeSession:
    def __init__(
        self,
        *,
        healthy=True,
        state=PaneState.READY,
        recover_ok=True,
        working=False,
        orphan_reply=None,
        pane_text="",
        nudge_ok=True,
    ):
        self._healthy = healthy
        self.state = state
        self._recover_ok = recover_ok
        self.working = working
        self.recovered = 0
        self._orphan_reply = orphan_reply
        self.orphan_pops = 0
        self.pane_text = pane_text
        self._nudge_ok = nudge_ok
        self.nudges: list[str] = []

    def capture_text(self) -> str:
        return self.pane_text

    def nudge(self, text: str = "Continue") -> bool:
        if not self._nudge_ok:
            return False
        self.nudges.append(text)
        return True

    def is_healthy(self) -> bool:
        return self._healthy

    def current_state(self) -> PaneState:
        return self.state

    def is_working(self) -> bool:
        return self.working

    def force_recover(self) -> bool:
        self.recovered += 1
        if self._recover_ok:
            self._healthy = True
        return self._recover_ok

    def pop_orphan_replies(self) -> list[str]:
        self.orphan_pops += 1
        reply, self._orphan_reply = self._orphan_reply, None  # deliver once
        return [reply] if reply else []


def make_wd(tmp_path, session, *, disk_free=10_000_000_000, clock=None, alerts=None):
    clock = clock if clock is not None else {"now": 1000.0}
    return Watchdog(
        session,
        runtime_dir=tmp_path,
        disk_free_fn=lambda: disk_free,
        clock_fn=lambda: clock["now"],
        alert_fn=(alerts.append if alerts is not None else (lambda m: None)),
        min_disk_bytes=500_000_000,
        stall_threshold=300.0,
        alert_cooldown=3600.0,
    )


# ── basic states ─────────────────────────────────────────────────────────


def test_healthy_does_nothing(tmp_path):
    sess = FakeSession()
    wd = make_wd(tmp_path, sess)
    assert wd.check_once() == "healthy"
    assert sess.recovered == 0


# ── orphan reply delivery ────────────────────────────────────────────────


def test_tick_delivers_session_notices_once(tmp_path):
    """agent-infra-backlog item 28: parking at bot start-up or in
    force_recover() has no chat reply to ride on — the tick delivers it."""
    sess = FakeSession()
    pending = ["⚠️ Контекст разговора сброшен: … отложена как parked_x"]
    sess.pop_notices = lambda: [pending.pop()] if pending else []
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    wd.check_once()
    wd.check_once()
    assert alerts == ["⚠️ Контекст разговора сброшен: … отложена как parked_x"]


class _NoticeQueue:
    """pop_notices()/requeue_notices() the way ClaudeSession does them."""

    def __init__(self, *texts: str) -> None:
        self.items = list(texts)

    def pop(self) -> list[str]:
        out, self.items = self.items, []
        return out

    def requeue(self, texts: list[str]) -> None:
        self.items = list(texts) + self.items


def test_undelivered_notice_is_kept_for_the_next_tick(tmp_path):
    """The alerter reports failure (returns False: no admin_chat_id, API
    error) — the notice and every one after it go back to the queue, in
    order, and the next tick that can send delivers them."""
    sess = FakeSession()
    q = _NoticeQueue("first", "second")
    sess.pop_notices, sess.requeue_notices = q.pop, q.requeue
    sent: list[str] = []
    up = {"ok": False}

    def alert(msg: str) -> bool:
        if up["ok"]:
            sent.append(msg)
        return up["ok"]

    wd = make_wd(tmp_path, sess)
    wd._alert_fn = alert
    wd.check_once()
    assert sent == [] and q.items == ["first", "second"]
    up["ok"] = True
    wd.check_once()
    assert sent == ["first", "second"] and q.items == []


def test_notice_send_that_raises_is_requeued_from_that_point(tmp_path):
    sess = FakeSession()
    q = _NoticeQueue("a", "b", "c")
    sess.pop_notices, sess.requeue_notices = q.pop, q.requeue
    sent: list[str] = []

    def alert(msg: str) -> None:
        if msg == "b":
            raise RuntimeError("network")
        sent.append(msg)

    wd = make_wd(tmp_path, sess)
    wd._alert_fn = alert
    assert wd.check_once() == "healthy"
    assert sent == ["a"] and q.items == ["b", "c"]


def test_tick_survives_a_failing_notice_source(tmp_path):
    sess = FakeSession()

    def boom():
        raise OSError("disk")

    sess.pop_notices = boom
    assert make_wd(tmp_path, sess).check_once() == "healthy"


def test_ready_tick_delivers_pending_orphan_reply(tmp_path):
    sess = FakeSession(orphan_reply="Landing page shipped to staging.")
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    assert wd.check_once() == "healthy"
    assert alerts == ["Landing page shipped to staging."]
    assert sess.orphan_pops == 1


def test_ready_tick_strips_html_tags_from_salvaged_orphan_reply(tmp_path):
    """M3 fix (2026-08-22): the watchdog's own alert path
    (`_telegram_alerter`) sets no `parse_mode`, unlike chat_session.py's
    normal send — so a salvaged-orphan reply's "⚠️ <i>ответ восстановлен без
    закрывающего маркера</i>" prefix (or any b/i/code/pre/a markup in the
    reply itself) must be stripped before it goes out here, or the user
    sees the literal tag characters."""
    sess = FakeSession(
        orphan_reply="⚠️ <i>ответ восстановлен без закрывающего маркера</i>\n\n"
        "<b>Done.</b> See <code>foo.py</code>."
    )
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    assert wd.check_once() == "healthy"
    assert len(alerts) == 1
    assert "<i>" not in alerts[0] and "</i>" not in alerts[0]
    assert "<b>" not in alerts[0] and "<code>" not in alerts[0]
    assert "ответ восстановлен без закрывающего маркера" in alerts[0]
    assert "Done." in alerts[0]
    assert "foo.py" in alerts[0]


def test_ready_tick_with_no_orphan_reply_sends_no_alert(tmp_path):
    sess = FakeSession(orphan_reply=None)
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    assert wd.check_once() == "healthy"
    assert alerts == []
    assert sess.orphan_pops == 1


def test_non_ready_tick_never_polls_for_orphan_reply(tmp_path):
    """Only a healthy READY tick is a safe place to read the pane for an
    orphan reply — a rate-limited/logged-out/hung session must not."""
    sess = FakeSession(state=PaneState.RATE_LIMITED, orphan_reply="should not fire")
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    wd.check_once()
    assert sess.orphan_pops == 0
    assert alerts == []


def test_orphan_alert_failure_does_not_break_the_tick(tmp_path):
    """A broken Telegram send must not surface as an exception from
    check_once — best-effort delivery, never worth crash-looping over."""

    def boom(_msg):
        raise RuntimeError("network down")

    sess = FakeSession(orphan_reply="hello")
    wd = Watchdog(
        sess,
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10_000_000_000,
        clock_fn=lambda: 1000.0,
        alert_fn=boom,
        min_disk_bytes=500_000_000,
    )
    assert wd.check_once() == "healthy"


def test_orphan_alert_failure_logs_at_error_not_warning(tmp_path, caplog):
    """The rid was already marked handled BEFORE this send by
    pop_orphan_replies() — a failed send here loses the reply permanently.
    That must be loud (ERROR, with the exception attached), not a bare
    WARNING with no detail (review 2026-08-20)."""
    import logging

    def boom(_msg):
        raise RuntimeError("network down")

    sess = FakeSession(orphan_reply="hello there")
    wd = Watchdog(
        sess,
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10_000_000_000,
        clock_fn=lambda: 1000.0,
        alert_fn=boom,
        min_disk_bytes=500_000_000,
    )
    with caplog.at_level(logging.ERROR, logger="d_brain.services.watchdog"):
        assert wd.check_once() == "healthy"
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "expected an ERROR-level log for the lost reply"
    assert "lost" in error_records[0].message.lower()


def test_dead_session_is_recovered(tmp_path):
    sess = FakeSession(healthy=False)
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    assert wd.check_once() == "recovered_dead"
    assert sess.recovered == 1
    assert alerts


def test_rate_limited_is_not_killed(tmp_path):
    sess = FakeSession(state=PaneState.RATE_LIMITED)
    wd = make_wd(tmp_path, sess)
    assert wd.check_once() == "rate_limited"
    assert sess.recovered == 0


def test_logged_out_alerts_without_restart(tmp_path):
    sess = FakeSession(state=PaneState.LOGGED_OUT)
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    assert wd.check_once() == "logged_out"
    assert sess.recovered == 0
    assert alerts


# ── hang detection (stuck-without-work, persistence-based) ────────────────


def test_hung_when_stuck_without_work(tmp_path):
    """Non-serviceable + no working spinner, persisting past the threshold →
    recover. A single observation never kills."""
    clock = {"now": 1000.0}
    sess = FakeSession(state=PaneState.UNKNOWN, working=False)
    wd = make_wd(tmp_path, sess, clock=clock)
    assert wd.check_once() == "healthy"  # arms the stuck timer only
    assert sess.recovered == 0
    clock["now"] += 301.0
    assert wd.check_once() == "recovered_hung"
    assert sess.recovered == 1


def test_stuck_startup_is_recovered(tmp_path):
    clock = {"now": 1000.0}
    sess = FakeSession(state=PaneState.STARTING, working=False)
    wd = make_wd(tmp_path, sess, clock=clock)
    wd.check_once()
    clock["now"] += 301.0
    assert wd.check_once() == "recovered_hung"


def test_long_silent_task_not_killed(tmp_path):
    """The working spinner is visible → alive, however long and however quiet
    pane.log is. Silence is NOT a hang signal."""
    clock = {"now": 1000.0}
    sess = FakeSession(state=PaneState.UNKNOWN, working=True)
    wd = make_wd(tmp_path, sess, clock=clock)
    wd.check_once()
    clock["now"] += 10_000.0
    assert wd.check_once() == "healthy"
    assert sess.recovered == 0


def test_work_resumes_resets_stuck_timer(tmp_path):
    clock = {"now": 1000.0}
    sess = FakeSession(state=PaneState.UNKNOWN, working=False)
    wd = make_wd(tmp_path, sess, clock=clock)
    wd.check_once()  # arms timer
    clock["now"] += 200.0
    sess.working = True
    wd.check_once()  # work visible → timer reset
    sess.working = False
    clock["now"] += 200.0
    assert wd.check_once() == "healthy"  # only 200s into a NEW stuck window
    assert sess.recovered == 0


def test_idle_ready_with_orphan_inflight_not_killed(tmp_path):
    """CRITICAL regression: a healthy idle READY session with a leftover
    orphan inflight marker must NOT be recovered."""
    sess = FakeSession(state=PaneState.READY)
    (tmp_path / "inflight").write_text("orphan\n")
    wd = make_wd(tmp_path, sess)
    assert wd.check_once() == "healthy"
    assert sess.recovered == 0
    assert not (tmp_path / "inflight").exists()  # orphan cleared


# ── Fix C (agent-infra-backlog item 23): READY must not erase a LIVE
# inflight marker ──────────────────────────────────────────────────────
#
# 03.09 incident: a READY tick that lands in a "pane looks briefly idle"
# window mid-turn (the pane can render this way even while a real ask() is
# in progress — see the item's diagnosis) used to unconditionally delete
# inflight ("clear any orphan marker"). That erased a LIVE hold's marker,
# which made is_steerable_turn() see "lock held, no inflight" and bounce
# the user's next message with the "🔧 background maintenance" message
# instead of steering it into the running turn — the user's text never
# reached the pane at all (confirmed by grep'ing pane.log, see the
# backlog item). The fix: only clear inflight when is_turn_active() says
# no ask() actually holds the lock.


def test_ready_tick_with_live_turn_leaves_inflight_in_place(tmp_path):
    """READY + a live ask() holding the lock (is_turn_active() == True) +
    an inflight marker present → check_once() must NOT delete it. This is
    the Fix C regression test — it must FAIL against the unfixed
    unconditional `self._inflight.unlink(missing_ok=True)`."""
    sess = LongRunFakeSession(state=PaneState.READY, turn_active=True)
    (tmp_path / "inflight").write_text("9c62f041\n")
    wd = make_wd(tmp_path, sess)
    assert wd.check_once() == "healthy"
    assert (tmp_path / "inflight").exists()  # live hold's marker survives
    assert (tmp_path / "inflight").read_text() == "9c62f041\n"


def test_ready_tick_with_no_live_turn_still_clears_inflight(tmp_path):
    """READY + is_turn_active() == False (explicitly implemented, not just
    absent) → inflight is still cleared — unchanged prior behavior for the
    genuine orphan-marker case."""
    sess = LongRunFakeSession(state=PaneState.READY, turn_active=False)
    (tmp_path / "inflight").write_text("orphan\n")
    wd = make_wd(tmp_path, sess)
    assert wd.check_once() == "healthy"
    assert not (tmp_path / "inflight").exists()


def test_ready_tick_session_without_is_turn_active_does_not_raise(tmp_path):
    """A duck-typed session that implements no is_turn_active() at all (the
    plain FakeSession) must not crash check_once() — same defensive
    try/except contract as _track_long_run's sibling check — and keeps
    clearing inflight, matching prior behavior for such sessions."""
    sess = FakeSession(state=PaneState.READY)
    assert not hasattr(sess, "is_turn_active")
    (tmp_path / "inflight").write_text("orphan\n")
    wd = make_wd(tmp_path, sess)
    assert wd.check_once() == "healthy"
    assert not (tmp_path / "inflight").exists()


def test_recover_deferred_when_lock_busy(tmp_path):
    """force_recover returns False (a live request holds the lock) → report
    deferred, do not claim a restart happened."""
    clock = {"now": 1000.0}
    sess = FakeSession(state=PaneState.UNKNOWN, working=False, recover_ok=False)
    wd = make_wd(tmp_path, sess, clock=clock)
    wd.check_once()
    clock["now"] += 301.0
    assert wd.check_once() == "recover_deferred"


# ── alert debounce ────────────────────────────────────────────────────────


def test_logged_out_alert_debounced_across_ticks(tmp_path):
    sess = FakeSession(state=PaneState.LOGGED_OUT)
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    wd.check_once()
    wd.check_once()
    wd.check_once()
    assert len(alerts) == 1  # not 3


def test_alert_refires_after_recovery(tmp_path):
    sess = FakeSession(state=PaneState.LOGGED_OUT)
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    wd.check_once()  # alert #1
    sess.state = PaneState.READY
    wd.check_once()  # healthy → resets debounce
    sess.state = PaneState.LOGGED_OUT
    wd.check_once()  # alert #2
    assert len(alerts) == 2


def test_persistent_fault_alert_backs_off(tmp_path):
    """A persistent fault must not spam at a fixed hourly rate — the re-alert
    interval doubles (1h, 2h, 4h…), so an 8-hour overnight outage sends a few
    escalating reminders, not one every hour."""
    sess = FakeSession(state=PaneState.LOGGED_OUT)
    alerts = []
    clock = {"now": 0.0}
    wd = make_wd(tmp_path, sess, alerts=alerts, clock=clock)
    # Tick every 30 min for 8 hours (16 ticks).
    for _ in range(16):
        wd.check_once()
        clock["now"] += 1800.0
    # Fixed-hourly would fire ~8 times; back-off fires at 0h,1h,3h,7h = 4.
    assert len(alerts) == 4


# ── disk + status ─────────────────────────────────────────────────────────


def test_disk_full_alerts_and_does_not_restart(tmp_path):
    sess = FakeSession()
    alerts = []
    wd = make_wd(tmp_path, sess, disk_free=1_000_000, alerts=alerts)
    assert wd.check_once() == "disk_full"
    assert sess.recovered == 0
    assert alerts


def test_status_file_written(tmp_path):
    sess = FakeSession()
    wd = make_wd(tmp_path, sess)
    wd.check_once()
    assert (tmp_path / "STATUS.md").exists()


# ── R3 (Fable audit): one-time context-size notification ─────────────────


def _with_transcript(sess, tmp_path, monkeypatch, tokens: int):
    fake_transcript = tmp_path / "fake.jsonl"
    fake_transcript.touch()
    sess.current_transcript_path = lambda: fake_transcript
    monkeypatch.setattr(
        "d_brain.services.watchdog.latest_context_tokens",
        lambda path, **kw: tokens,
    )
    return sess


def test_context_alert_fires_once_when_crossing_threshold(tmp_path, monkeypatch):
    sess = _with_transcript(FakeSession(), tmp_path, monkeypatch, 450_000)
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    wd.check_once()
    assert any("Контекст сессии" in a for a in alerts)
    alerts.clear()
    wd.check_once()  # still above threshold, second tick
    assert alerts == []  # one-time, not repeated


def test_context_alert_does_not_fire_below_threshold(tmp_path, monkeypatch):
    sess = _with_transcript(FakeSession(), tmp_path, monkeypatch, 10_000)
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    wd.check_once()
    assert alerts == []


def test_context_alert_rearms_after_dropping_back_below_threshold(
    tmp_path, monkeypatch
):
    """E.g. after a `/clear` — a LATER crossing must alert again, not stay
    silent forever after the first one-time alert. B2 fix (2026-08-22): the
    rearm now requires DEFAULT_CONTEXT_REARM_TICKS CONSECUTIVE low ticks, so
    this drives it below threshold for that many ticks before checking the
    later crossing fires again."""
    from d_brain.services.watchdog import DEFAULT_CONTEXT_REARM_TICKS

    sess = FakeSession()
    fake_transcript = tmp_path / "fake.jsonl"
    fake_transcript.touch()
    sess.current_transcript_path = lambda: fake_transcript
    tokens = {"v": 450_000}
    monkeypatch.setattr(
        "d_brain.services.watchdog.latest_context_tokens",
        lambda path, **kw: tokens["v"],
    )
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    wd.check_once()
    assert len(alerts) == 1
    tokens["v"] = 10_000
    for _ in range(DEFAULT_CONTEXT_REARM_TICKS):
        wd.check_once()
    assert len(alerts) == 1  # no new alert while below
    tokens["v"] = 450_000
    wd.check_once()
    assert len(alerts) == 2  # crossed again -> alerts again


def test_context_alert_does_not_rearm_on_a_single_low_tick(tmp_path, monkeypatch):
    """B2 fix regression test: the actual bug this round fixes — one noisy
    low reading (e.g. an unrecognized transcript shape) sandwiched between
    two real high readings must NOT rearm the latch, or the alert would
    re-fire on the very next high tick instead of staying one-time."""
    sess = FakeSession()
    fake_transcript = tmp_path / "fake.jsonl"
    fake_transcript.touch()
    sess.current_transcript_path = lambda: fake_transcript
    tokens = {"v": 450_000}
    monkeypatch.setattr(
        "d_brain.services.watchdog.latest_context_tokens",
        lambda path, **kw: tokens["v"],
    )
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    wd.check_once()
    assert len(alerts) == 1
    tokens["v"] = 10_000
    wd.check_once()  # ONE low tick only — not enough to rearm
    tokens["v"] = 450_000
    wd.check_once()
    assert len(alerts) == 1  # still latched — must NOT have fired again


def test_context_alert_skips_silently_when_no_transcript(tmp_path):
    sess = FakeSession()
    sess.current_transcript_path = lambda: None
    alerts = []
    wd = make_wd(tmp_path, sess, alerts=alerts)
    assert wd.check_once() == "healthy"
    assert alerts == []


def test_context_alert_never_breaks_the_tick_if_session_lacks_the_method(tmp_path):
    """Older/duck-typed sessions (or existing tests) may not implement
    current_transcript_path() at all — must degrade to 'nothing to check',
    never crash the tick."""
    sess = FakeSession()  # no current_transcript_path attribute set
    wd = make_wd(tmp_path, sess)
    assert wd.check_once() == "healthy"


def test_context_alert_does_not_automatically_clear(tmp_path, monkeypatch):
    """R3 is explicit: notification only, no automatic /clear."""
    sess = _with_transcript(FakeSession(), tmp_path, monkeypatch, 450_000)
    wd = make_wd(tmp_path, sess)
    wd.check_once()
    assert sess.recovered == 0  # never force_recover()-ed / cleared over this


# ── subscription limit: wait for the reset, then wake the session ────────
#
# 2026-08-20 incident: a background workflow hit "You've hit your session
# limit · resets 12am (UTC)". Claude Code parks at that banner and only ever
# acts on the next input, so the brain stayed silent long after the reset —
# a manual `tmux send-keys` revived it instantly. The watchdog now does that
# send-keys itself once the banner's reset time has passed.

BANNER = "  You've hit your session limit · resets 12am (UTC)\n❯\n"


def _epoch(day: str, hhmm: str) -> float:
    return datetime.fromisoformat(f"{day}T{hhmm}:00+00:00").timestamp()


def _limited_wd(tmp_path, clock, alerts, *, pane=BANNER, nudge_ok=True):
    sess = FakeSession(state=PaneState.RATE_LIMITED, pane_text=pane, nudge_ok=nudge_ok)
    return sess, make_wd(tmp_path, sess, clock=clock, alerts=alerts)


def test_limit_before_reset_waits_without_nudging(tmp_path):
    """22:00 UTC, banner resets at 12am (midnight) — two hours to go."""
    clock = {"now": _epoch("2026-08-20", "22:00")}
    alerts = []
    sess, wd = _limited_wd(tmp_path, clock, alerts)
    assert wd.check_once() == "rate_limited"
    assert sess.nudges == []
    assert alerts == []


def test_limit_nudges_once_the_reset_time_has_passed(tmp_path):
    """Seen at 22:00, reset at midnight; by 00:30 the session must be woken."""
    clock = {"now": _epoch("2026-08-20", "22:00")}
    alerts = []
    sess, wd = _limited_wd(tmp_path, clock, alerts)
    assert wd.check_once() == "rate_limited"
    clock["now"] = _epoch("2026-08-21", "00:30")
    assert wd.check_once() == "limit_resumed"
    assert sess.nudges == ["Continue"]
    assert len(alerts) == 1
    assert "Лимит" in alerts[0]


def test_limit_does_not_nudge_every_tick(tmp_path):
    """A nudge that didn't take must not turn into a send-keys storm."""
    clock = {"now": _epoch("2026-08-20", "22:00")}
    alerts = []
    sess, wd = _limited_wd(tmp_path, clock, alerts)
    wd.check_once()
    clock["now"] = _epoch("2026-08-21", "00:30")
    assert wd.check_once() == "limit_resumed"
    clock["now"] += 15.0  # next tick, still parked
    assert wd.check_once() == "rate_limited"
    assert sess.nudges == ["Continue"]


def test_limit_nudge_deferred_while_a_live_turn_holds_the_pane(tmp_path):
    """nudge() returns False when a real ask() owns the lock — the session is
    already working, so there is nothing to wake."""
    clock = {"now": _epoch("2026-08-20", "22:00")}
    alerts = []
    sess, wd = _limited_wd(tmp_path, clock, alerts, nudge_ok=False)
    wd.check_once()
    clock["now"] = _epoch("2026-08-21", "00:30")
    assert wd.check_once() == "rate_limited"
    assert alerts == []


def test_unreadable_reset_time_still_wakes_up_eventually(tmp_path):
    """No parseable time ⇒ don't guess a schedule, but never park forever."""
    clock = {"now": _epoch("2026-08-20", "10:00")}
    alerts = []
    sess, wd = _limited_wd(tmp_path, clock, alerts, pane="  usage limit reached\n❯\n")
    assert wd.check_once() == "rate_limited"
    clock["now"] += 5 * 3600
    assert wd.check_once() == "rate_limited"  # inside the 6h fallback
    clock["now"] += 2 * 3600
    assert wd.check_once() == "limit_resumed"
    assert sess.nudges == ["Continue"]


def test_unmarked_reset_time_logs_a_warning(tmp_path, caplog):
    """A reset time IS on screen but without the UTC marker parse_reset_time
    requires must not fail silently — it is the one case, among several that
    all make parse_reset_time() return None, actually worth a loud log line
    (review 2026-08-20: the caller otherwise falls back to limit_max_wait
    with zero explanation)."""
    import logging

    clock = {"now": _epoch("2026-08-20", "10:00")}
    alerts = []
    sess, wd = _limited_wd(
        tmp_path, clock, alerts, pane="  weekly limit reached · resets 9am\n❯\n"
    )
    with caplog.at_level(logging.WARNING, logger="d_brain.services.watchdog"):
        assert wd.check_once() == "rate_limited"
    assert any("UTC marker" in r.message for r in caplog.records)


def test_recovering_from_a_limit_rearms_the_next_one(tmp_path):
    """State from one limit must not leak into the next: after the session
    goes healthy, a fresh banner starts its own wait."""
    clock = {"now": _epoch("2026-08-20", "22:00")}
    alerts = []
    sess, wd = _limited_wd(tmp_path, clock, alerts)
    wd.check_once()
    sess.state = PaneState.READY
    clock["now"] = _epoch("2026-08-21", "00:30")
    assert wd.check_once() == "healthy"
    sess.state = PaneState.RATE_LIMITED  # new limit, resets midnight again
    clock["now"] = _epoch("2026-08-21", "01:00")
    assert wd.check_once() == "rate_limited"
    assert sess.nudges == []


# ── unattended long-run tracking (agent-infra-backlog item 22) ────────────

_ACTIVE_PANE = "✻ Working…  (esc to interrupt)\n"
_IDLE_PANE = "❯\n"


class LongRunFakeSession(FakeSession):
    """Extends FakeSession with the two extra hooks _track_long_run needs:
    ``is_turn_active`` (lock-based attendance) and ``steer`` (Step E's
    one-shot nudge). Kept as its own subclass rather than adding these to
    the shared FakeSession above, so the OTHER watchdog tests keep exercising
    the "session lacks these methods" duck-typing path for free."""

    def __init__(self, *, turn_active: bool = False, **kw) -> None:
        super().__init__(**kw)
        self._turn_active = turn_active
        self.steered: list[str] = []

    def is_turn_active(self) -> bool:
        return self._turn_active

    def steer(self, text: str) -> None:
        self.steered.append(text)


def test_track_long_run_survives_a_session_without_is_turn_active(tmp_path):
    """A minimal/fake session that doesn't implement is_turn_active (the
    plain FakeSession above) must never crash a tick — same defensive
    duck-typing contract as _check_context_size's sibling checks."""
    sess = FakeSession(pane_text=_ACTIVE_PANE)
    wd = make_wd(tmp_path, sess)  # default long_run_alert_seconds (900, on)
    assert wd.check_once() == "healthy"
    assert not (tmp_path / "long-run.json").exists()


def test_track_long_run_disabled_writes_nothing(tmp_path):
    clock = {"now": 1000.0}
    sess = LongRunFakeSession(pane_text=_ACTIVE_PANE, turn_active=False)
    alerts = []
    wd = Watchdog(
        sess,
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10_000_000_000,
        clock_fn=lambda: clock["now"],
        alert_fn=alerts.append,
        min_disk_bytes=500_000_000,
        long_run_alert_seconds=0.0,
    )
    wd.check_once()
    assert not (tmp_path / "long-run.json").exists()
    assert alerts == []


def test_track_long_run_fires_exactly_one_alert_and_one_ended(tmp_path):
    # Baseline deliberately NONZERO: since == 0.0 is long_run.LongRun's
    # sentinel for "no run in progress" (see its docstring) — starting the
    # fake clock exactly at 0.0 would make the freshly-started run
    # indistinguishable from "not started" on the very first tick, a false
    # collision that never happens with a real time.time() clock.
    clock = {"now": 1000.0}
    sess = LongRunFakeSession(pane_text=_ACTIVE_PANE, turn_active=False)
    alerts = []
    wd = Watchdog(
        sess,
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10_000_000_000,
        clock_fn=lambda: clock["now"],
        alert_fn=alerts.append,
        min_disk_bytes=500_000_000,
        long_run_alert_seconds=60.0,
    )
    wd.check_once()  # started — no alert yet
    clock["now"] = 1030.0
    wd.check_once()  # still below threshold
    assert alerts == []
    clock["now"] = 1061.0
    wd.check_once()  # crosses alert_after
    alert_hits = [a for a in alerts if "занята уже" in a]
    assert len(alert_hits) == 1
    clock["now"] = 1200.0
    wd.check_once()  # still active — must not alert again
    assert len([a for a in alerts if "занята уже" in a]) == 1
    sess.pane_text = _IDLE_PANE
    clock["now"] = 1260.0
    wd.check_once()  # main turn no longer active — ends the run
    ended_hits = [a for a in alerts if "завершилась" in a]
    assert len(ended_hits) == 1


def test_track_long_run_attended_turn_never_alerts(tmp_path):
    """An ordinary chat turn (ask() holds the lock) must never be mistaken
    for an unattended cascade, however long the pane shows work."""
    clock = {"now": 1000.0}
    sess = LongRunFakeSession(pane_text=_ACTIVE_PANE, turn_active=True)
    alerts = []
    wd = Watchdog(
        sess,
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10_000_000_000,
        clock_fn=lambda: clock["now"],
        alert_fn=alerts.append,
        min_disk_bytes=500_000_000,
        long_run_alert_seconds=60.0,
    )
    for t in (1000.0, 1061.0, 1200.0):
        clock["now"] = t
        wd.check_once()
    assert alerts == []


def test_track_long_run_short_blip_never_alerts_on_either_end(tmp_path):
    """F2 fix (blind-review round, 2026-09): a run that starts and ends
    WITHOUT ever crossing alert_after must produce zero Telegram messages —
    including on the 'ended' side. Before the fix, 'ended' fired
    unconditionally on any run that had ever been tracked at all, so a
    15-second unattended blip (a proactive self-initiated turn, a leftover
    turn clearing right after a stall) sent an unsolicited "✅ Длинная
    автономная задача завершилась" message for something the user was
    never told had started."""
    clock = {"now": 1000.0}
    sess = LongRunFakeSession(pane_text=_ACTIVE_PANE, turn_active=False)
    alerts = []
    wd = Watchdog(
        sess,
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10_000_000_000,
        clock_fn=lambda: clock["now"],
        alert_fn=alerts.append,
        min_disk_bytes=500_000_000,
        long_run_alert_seconds=900.0,  # production default
    )
    wd.check_once()  # started — no alert (log-only)
    clock["now"] = 1015.0  # 15s blip — nowhere near alert_after (900s)
    sess.pane_text = _IDLE_PANE
    wd.check_once()  # ended — must NOT alert either
    assert alerts == []


def test_track_long_run_writes_a_marker_claude_session_can_read(tmp_path):
    """The whole point of this ledger: claude_session.ask()'s fast path
    reads long-run.json via long_run.is_active — verify the watchdog's write
    actually produces a marker that reads back as active."""
    from d_brain.services import long_run

    clock = {"now": 1000.0}
    sess = LongRunFakeSession(pane_text=_ACTIVE_PANE, turn_active=False)
    wd = Watchdog(
        sess,
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10_000_000_000,
        clock_fn=lambda: clock["now"],
        alert_fn=lambda _m: None,
        min_disk_bytes=500_000_000,
        long_run_alert_seconds=60.0,
    )
    wd.check_once()
    active, elapsed = long_run.is_active(tmp_path, now=1010.0, stale_after=120.0)
    assert active is True
    assert elapsed == 10.0


# ── Step E: default-off in-pane nudge ──────────────────────────────────────


def test_long_run_nudge_default_off_never_steers(tmp_path):
    """long_run_nudge_seconds defaults to 0.0 — zero live behavior change:
    steer() must never be called regardless of how long the run drags on."""
    clock = {"now": 1000.0}
    sess = LongRunFakeSession(pane_text=_ACTIVE_PANE, turn_active=False)
    wd = Watchdog(
        sess,
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10_000_000_000,
        clock_fn=lambda: clock["now"],
        alert_fn=lambda _m: None,
        min_disk_bytes=500_000_000,
        long_run_alert_seconds=60.0,
        # long_run_nudge_seconds left at its default (0.0)
    )
    for t in range(1000, 6000, 200):
        clock["now"] = float(t)
        wd.check_once()
    assert sess.steered == []


def test_long_run_nudge_enabled_fires_exactly_once_past_threshold(tmp_path):
    clock = {"now": 1000.0}
    sess = LongRunFakeSession(pane_text=_ACTIVE_PANE, turn_active=False)
    wd = Watchdog(
        sess,
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10_000_000_000,
        clock_fn=lambda: clock["now"],
        alert_fn=lambda _m: None,
        min_disk_bytes=500_000_000,
        long_run_alert_seconds=60.0,
        long_run_nudge_seconds=120.0,
    )
    for t in (1000.0, 1060.0, 1121.0, 1200.0, 1300.0):
        clock["now"] = t
        wd.check_once()
    assert len(sess.steered) == 1


# ── item 29: hard cap on an unattended turn ────────────────────────────────


class CapFakeSession(LongRunFakeSession):
    """LongRunFakeSession + a recording ``interrupt()``.

    The parent deliberately LACKS ``interrupt`` — that omission is itself a
    test case below (a duck-typed session must not crash a tick), so the
    method lives here instead of being pushed up.
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.interrupts = 0

    def interrupt(self) -> None:
        self.interrupts += 1


def _cap_wd(tmp_path, sess, clock, alerts, *, cap=300.0, alert_after=60.0):
    return Watchdog(
        sess,
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10_000_000_000,
        clock_fn=lambda: clock["now"],
        alert_fn=alerts.append,
        min_disk_bytes=500_000_000,
        long_run_alert_seconds=alert_after,
        long_run_max_seconds=cap,
    )


def test_long_run_cap_closes_the_turn_exactly_once(tmp_path):
    """The owner's rule, enforced: an unattended turn that outlives the cap
    is closed once and explained once — not re-interrupted every tick."""
    clock = {"now": 1000.0}
    sess = CapFakeSession(pane_text=_ACTIVE_PANE, turn_active=False)
    alerts: list[str] = []
    wd = _cap_wd(tmp_path, sess, clock, alerts, cap=300.0)

    wd.check_once()  # started
    clock["now"] = 1299.0
    wd.check_once()  # one second short of the cap
    assert sess.interrupts == 0

    clock["now"] = 1301.0
    wd.check_once()  # crosses the cap
    assert sess.interrupts == 1
    closed = [a for a in alerts if "закрыл его автоматически" in a]
    assert len(closed) == 1
    assert "~5 мин" in closed[0]

    # Later ticks still see the same run: no second interrupt, and no
    # second message until a whole cap has passed (escalation window).
    for t in (1310.0, 1400.0, 1500.0):
        clock["now"] = t
        wd.check_once()
    assert sess.interrupts == 1
    assert len([a for a in alerts if "закрыл его автоматически" in a]) == 1


def test_long_run_cap_escalates_at_most_once_per_cap_and_never_recovers(tmp_path):
    """A turn that ignores the interrupt gets a rare reminder — never a
    second interrupt and never force_recover (that would destroy the very
    context holding the partial work)."""
    clock = {"now": 1000.0}
    sess = CapFakeSession(pane_text=_ACTIVE_PANE, turn_active=False)
    alerts: list[str] = []
    wd = _cap_wd(tmp_path, sess, clock, alerts, cap=300.0)

    wd.check_once()
    clock["now"] = 1301.0
    wd.check_once()  # closed at 1301
    for t in (1400.0, 1500.0, 1600.0):
        clock["now"] = t
        wd.check_once()
    assert [a for a in alerts if "не закрылся" in a] == []

    clock["now"] = 1601.0  # a full cap after the close
    wd.check_once()
    assert len([a for a in alerts if "не закрылся" in a]) == 1
    for t in (1700.0, 1800.0, 1900.0):
        clock["now"] = t
        wd.check_once()
    assert len([a for a in alerts if "не закрылся" in a]) == 1

    clock["now"] = 1902.0  # another full cap later
    wd.check_once()
    assert len([a for a in alerts if "не закрылся" in a]) == 2
    assert sess.interrupts == 1
    assert sess.recovered == 0


def test_long_run_cap_disabled_never_interrupts(tmp_path):
    """0 is the rollback switch: the heads-up alert still works, the
    auto-close never fires however long the run drags on."""
    clock = {"now": 1000.0}
    sess = CapFakeSession(pane_text=_ACTIVE_PANE, turn_active=False)
    alerts: list[str] = []
    wd = _cap_wd(tmp_path, sess, clock, alerts, cap=0.0)
    for t in range(1000, 20000, 500):
        clock["now"] = float(t)
        wd.check_once()
    assert sess.interrupts == 0
    assert [a for a in alerts if "закрыл его автоматически" in a] == []
    assert len([a for a in alerts if "занята уже" in a]) == 1  # alert unaffected


def test_long_run_cap_never_closes_an_attended_turn(tmp_path):
    """An ordinary chat turn (ask() holds the lock) is the user waiting on
    their own answer — it is never tracked, so it is never closed, however
    far past the cap it runs."""
    clock = {"now": 1000.0}
    sess = CapFakeSession(pane_text=_ACTIVE_PANE, turn_active=True)
    alerts: list[str] = []
    wd = _cap_wd(tmp_path, sess, clock, alerts, cap=300.0)
    for t in range(1000, 8000, 250):
        clock["now"] = float(t)
        wd.check_once()
    assert sess.interrupts == 0
    assert alerts == []
    # The marker file is written every tick, but always as the empty
    # "nothing tracked" state — an attended turn is never a tracked run.
    from d_brain.services import long_run as _long_run

    assert _long_run.read(tmp_path).since == 0.0


def test_long_run_cap_survives_a_session_without_interrupt(tmp_path):
    """Duck-typed sessions (fakes, a driver missing the method) must not
    crash a tick — and the owner is told honestly that the close FAILED
    rather than being told the turn was closed when it was not."""
    clock = {"now": 1000.0}
    sess = LongRunFakeSession(pane_text=_ACTIVE_PANE, turn_active=False)
    assert not hasattr(sess, "interrupt")
    alerts: list[str] = []
    wd = _cap_wd(tmp_path, sess, clock, alerts, cap=300.0)
    wd.check_once()
    clock["now"] = 1301.0
    assert wd.check_once() == "healthy"  # tick survives
    assert [a for a in alerts if "закрыл его автоматически" in a] == []
    failed = [a for a in alerts if "закрыть его автоматически не получилось" in a]
    assert len(failed) == 1


def test_long_run_cap_relatches_for_the_next_run(tmp_path):
    """The latch is per-RUN, not per-process: a NEW unattended run after
    this one ends gets its own single interrupt."""
    clock = {"now": 1000.0}
    sess = CapFakeSession(pane_text=_ACTIVE_PANE, turn_active=False)
    alerts: list[str] = []
    wd = _cap_wd(tmp_path, sess, clock, alerts, cap=300.0)

    wd.check_once()
    clock["now"] = 1301.0
    wd.check_once()
    assert sess.interrupts == 1

    sess.pane_text = _IDLE_PANE  # run ends
    clock["now"] = 1400.0
    wd.check_once()

    sess.pane_text = _ACTIVE_PANE  # a brand-new run starts
    clock["now"] = 1500.0
    wd.check_once()
    clock["now"] = 1700.0
    wd.check_once()  # below the cap for THIS run
    assert sess.interrupts == 1
    clock["now"] = 1801.0
    wd.check_once()
    assert sess.interrupts == 2
