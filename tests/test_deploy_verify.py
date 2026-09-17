"""Tests for the post-deploy canary."""

from d_brain.services import deploy_verify as dv
from d_brain.services.claude_session import AskResult

# ── which changes need a canary at all ───────────────────────────────


def test_delivery_critical_files_are_recognized():
    changed = [
        "vault/daily/2026-08-20.md",
        "src/d_brain/services/claude_session.py",
        "deploy/dbrain-bot.service",
    ]
    assert dv.touches_delivery_path(changed) == [
        "deploy/dbrain-bot.service",
        "src/d_brain/services/claude_session.py",
    ]


def test_a_vault_only_change_needs_no_canary():
    assert dv.touches_delivery_path(["vault/MEMORY.md", "README.md"]) == []


# ── the runner ───────────────────────────────────────────────────────


def _ok(detail="fine"):
    return lambda: (True, detail)


def _bad(detail="broken"):
    return lambda: (False, detail)


def test_all_green_is_healthy():
    report = dv.verify([("a", _ok()), ("b", _ok())])
    assert report.ok
    assert "HEALTHY" in report.summary()


def test_first_failure_stops_the_rest():
    report = dv.verify([("a", _bad("unit dead")), ("b", _ok())])
    assert not report.ok
    assert report.failure.name == "a"
    assert report.checks[1].ok is None
    assert "not run" in report.checks[1].detail


def test_a_crashing_probe_counts_as_a_failure():
    def boom():
        raise RuntimeError("no systemd here")

    report = dv.verify([("a", boom)])
    assert not report.ok
    assert "probe raised" in report.checks[0].detail


def test_an_empty_probe_list_is_not_healthy():
    """A verification that checked nothing must never read as a pass."""
    assert not dv.verify([]).ok


# ── individual probes ────────────────────────────────────────────────


def test_unit_probe_rejects_an_inactive_unit():
    show = lambda _u: {"ActiveState": "failed", "SubState": "failed"}  # noqa: E731
    ok, detail = dv.unit_active_probe(show, "dbrain-bot.service")()
    assert not ok
    assert "failed" in detail


def test_unit_probe_rejects_a_unit_that_never_restarted():
    """Writing new bytes to disk is not a deploy — the process has to be
    replaced, or the canary would bless the old code."""
    show = lambda _u: {  # noqa: E731
        "ActiveState": "active",
        "SubState": "running",
        "ActiveEnterTimestampMonotonic": "5000000",  # 5.0s since boot
    }
    ok, detail = dv.unit_active_probe(
        show, "dbrain-bot.service", restarted_after_monotonic=60.0
    )()
    assert not ok
    assert "NOT restarted" in detail


def test_unit_probe_accepts_a_restart_after_the_reference():
    show = lambda _u: {  # noqa: E731
        "ActiveState": "active",
        "SubState": "running",
        "ActiveEnterTimestampMonotonic": "90000000",  # 90s since boot
    }
    ok, _ = dv.unit_active_probe(
        show, "dbrain-bot.service", restarted_after_monotonic=60.0
    )()
    assert ok


def test_single_instance_probe_catches_the_orphan():
    ok, detail = dv.single_instance_probe(lambda: [255428, 260271])()
    assert not ok
    assert "orphan" in detail


def test_single_instance_probe_catches_a_dead_bot():
    ok, detail = dv.single_instance_probe(lambda: [])()
    assert not ok
    assert "no `python -m d_brain`" in detail


def test_single_instance_probe_passes_on_exactly_one():
    ok, _ = dv.single_instance_probe(lambda: [265661])()
    assert ok


# ── the live round trip ──────────────────────────────────────────────


class FakeSession:
    def __init__(self, result, *, busy_ticks: int = 0) -> None:
        self.result = result
        self.busy_ticks = busy_ticks
        self.asked: list[tuple[str, dict]] = []

    def is_turn_active(self) -> bool:
        if self.busy_ticks:
            self.busy_ticks -= 1
            return True
        return False

    def ask(self, prompt, **kwargs):
        self.asked.append((prompt, kwargs))
        return self.result


def _canary(session, **kw):
    ticks = iter(range(0, 100_000))
    return dv.brain_canary_probe(
        session,
        clock_fn=lambda: float(next(ticks)),
        sleep_fn=lambda _s: None,
        rid_fn=lambda: "cafe",
        **kw,
    )


def test_canary_passes_on_pong():
    session = FakeSession(AskResult("ok", reply="PONG"))
    ok, detail = _canary(session)()
    assert ok
    assert "round trip" in detail


def test_canary_uses_a_short_timeout_not_the_runaway_ceiling():
    session = FakeSession(AskResult("ok", reply="PONG"))
    _canary(session)()
    _, kwargs = session.asked[0]
    assert kwargs["timeout"] == dv.CANARY_TIMEOUT
    assert dv.CANARY_TIMEOUT < 300  # a no-op turn; nowhere near DEFAULT_TIMEOUT


def test_canary_marks_itself_as_maintenance():
    """So a user message arriving mid-canary is not injected into it."""
    session = FakeSession(AskResult("ok", reply="PONG"))
    _canary(session)()
    _, kwargs = session.asked[0]
    assert kwargs["request_id"].startswith("maint-")


def test_canary_fails_on_a_timeout():
    session = FakeSession(AskResult("timeout"))
    ok, detail = _canary(session)()
    assert not ok
    assert "timeout" in detail


def test_canary_fails_when_the_reply_is_not_the_expected_word():
    session = FakeSession(AskResult("ok", reply="Привет! Чем помочь?"))
    ok, detail = _canary(session)()
    assert not ok
    assert "expected PONG" in detail


def test_canary_waits_for_a_live_turn_then_runs():
    session = FakeSession(AskResult("ok", reply="pong"), busy_ticks=3)
    ok, _ = _canary(session, idle_wait=1000.0)()
    assert ok
    assert len(session.asked) == 1


def test_canary_gives_up_rather_than_stealing_the_pane():
    session = FakeSession(AskResult("ok", reply="PONG"), busy_ticks=10**6)
    ok, detail = _canary(session, idle_wait=5.0)()
    assert not ok
    assert "cannot attest" in detail
    assert session.asked == []
