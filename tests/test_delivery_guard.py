"""Tests for the consecutive-delivery-failure guard.

The behaviour that matters is as much about what it REFUSES to do (roll code
back, restart forever) as about what it does.
"""

from d_brain.services import ask_health
from d_brain.services.ask_health import Health
from d_brain.services.claude_session import DEFAULT_STALL_TIMEOUT, DEFAULT_TIMEOUT
from d_brain.services.delivery_guard import DEFAULT_WINDOW, DeliveryGuard


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _guard(tmp_path, clock, **kw):
    restarts: list[str] = []
    alerts: list[str] = []
    guard = DeliveryGuard(
        tmp_path,
        restart_fn=lambda unit: (restarts.append(unit), True)[1],
        alert_fn=alerts.append,
        clock_fn=clock,
        **kw,
    )
    return guard, restarts, alerts


def _streak(n: int, *, start: float = 1000.0, span: float = 60.0) -> Health:
    return Health(n, "timeout", start + span, start)


def test_below_threshold_does_nothing(tmp_path):
    guard, restarts, _ = _guard(tmp_path, Clock())
    assert guard.decide(_streak(2)).action == "none"
    assert restarts == []


def test_default_window_tracks_the_session_timeouts(tmp_path):
    """Review 2026-08-20: a bare 1800s DEFAULT_WINDOW silently stopped
    matching claude_session's stall/hard timeouts once THOSE were raised
    (180s→900s, 1200s→3600s) — three genuine failures could then span longer
    than the window and the auto-restart guard would never fire. The window
    must scale with whichever ceiling is larger."""
    assert DEFAULT_WINDOW == 3 * max(DEFAULT_STALL_TIMEOUT, DEFAULT_TIMEOUT)
    # And concretely wide enough for three real failures at today's ceiling
    # to still register as "close together".
    assert DEFAULT_WINDOW >= 3 * DEFAULT_TIMEOUT


def test_threshold_triggers_one_restart(tmp_path):
    guard, restarts, alerts = _guard(tmp_path, Clock())
    ask_health.record(tmp_path, "timeout", clock_fn=lambda: 1000.0)
    ask_health.record(tmp_path, "timeout", clock_fn=lambda: 1030.0)
    ask_health.record(tmp_path, "timeout", clock_fn=lambda: 1060.0)
    assert guard.tick().action == "restart"
    assert restarts == ["dbrain-bot.service"]
    assert alerts and "перезапустил" in alerts[0]


def test_busy_active_run_never_triggers_a_restart(tmp_path):
    """agent-infra: a health ledger built entirely from
    'busy_active' outcomes (a live, progressing pane during an unattended
    agent cascade) must never look like a delivery outage — decide() stays
    'none' regardless of how many such turns are recorded."""
    guard, restarts, _ = _guard(tmp_path, Clock())
    ask_health.record(tmp_path, "busy_active", clock_fn=lambda: 1000.0)
    ask_health.record(tmp_path, "busy_active", clock_fn=lambda: 1030.0)
    ask_health.record(tmp_path, "busy_active", clock_fn=lambda: 1060.0)
    ask_health.record(tmp_path, "busy_active", clock_fn=lambda: 1090.0)
    health = ask_health.read(tmp_path)
    assert health.fail_streak == 0  # busy_active never accumulates a streak
    assert guard.decide(health).action == "none"
    assert restarts == []


def test_failures_spread_too_thin_are_not_an_outage(tmp_path):
    """Three bad turns across a working day are three bad turns."""
    guard, restarts, _ = _guard(tmp_path, Clock())
    slow = Health(3, "timeout", 100_000.0, 1000.0)
    assert guard.decide(slow).action == "none"
    assert restarts == []


def test_a_stale_streak_is_not_acted_on(tmp_path):
    """The ledger is only written when a turn ENDS, so a streak sits there
    unchanged until the next message. Re-reading it must not look like fresh
    breakage."""
    clock = Clock(10_000.0)
    guard, restarts, _ = _guard(tmp_path, clock, window=1800.0)
    assert guard.decide(_streak(3, start=1000.0)).action == "none"
    assert restarts == []


def test_no_second_restart_without_a_turn_in_between(tmp_path):
    """The false-escalation path: restart, wait out the cooldown with nobody
    sending anything, and the guard must NOT restart again and then announce
    that two restarts failed — it never saw a turn attempted."""
    clock = Clock(1060.0)
    guard, _, alerts = _guard(tmp_path, clock, restart_cooldown=900.0, window=1800.0)
    assert guard.decide(_streak(3)).action == "restart"
    clock.t += 1000  # cooldown is over, but the ledger has not moved
    assert guard.decide(_streak(3)).action == "none"
    clock.t += 1000
    assert guard.decide(_streak(3)).action == "none"
    assert not any("не восстановилась" in a for a in alerts)


def test_a_fresh_failure_after_a_restart_still_escalates(tmp_path):
    """...but a real turn that fails again after the restart must get through
    to escalation. Freshness gates noise, not the signal."""
    clock = Clock(1060.0)
    guard, _, _ = _guard(tmp_path, clock, restart_cooldown=10.0, max_restarts=1)
    assert guard.decide(_streak(3)).action == "restart"
    clock.t += 100
    # A new turn ran and failed: last_ts moves with it.
    assert guard.decide(Health(4, "timeout", clock.t, 1000.0)).action == "escalate"


def test_restart_is_not_repeated_inside_the_cooldown(tmp_path):
    clock = Clock()
    guard, restarts, _ = _guard(tmp_path, clock, restart_cooldown=900.0)
    assert guard.decide(_streak(3)).action == "restart"
    clock.t += 60
    assert guard.decide(_streak(4)).action == "none"
    clock.t += 900
    assert guard.decide(_streak(5)).action == "restart"
    assert len(restarts) == 0  # decide() alone never restarts; tick() does


def test_budget_exhausted_escalates_instead_of_looping(tmp_path):
    clock = Clock()
    guard, _, _ = _guard(tmp_path, clock, restart_cooldown=10.0, max_restarts=2)
    assert guard.decide(_streak(3)).action == "restart"
    clock.t += 20
    assert guard.decide(_streak(4)).action == "restart"
    clock.t += 20
    assert guard.decide(_streak(5)).action == "escalate"
    clock.t += 20
    # ...and then goes quiet rather than re-alerting every 15s tick.
    assert guard.decide(_streak(6)).action == "none"


def test_escalation_hands_over_the_rollback_it_will_not_run(tmp_path):
    clock = Clock()
    guard, _, alerts = _guard(
        tmp_path, clock, restart_cooldown=0.0, max_restarts=1, repo_dir="/opt/dbrain"
    )
    ask_health.record(tmp_path, "timeout", clock_fn=lambda: 1000.0)
    ask_health.record(tmp_path, "timeout", clock_fn=lambda: 1010.0)
    ask_health.record(tmp_path, "error", clock_fn=lambda: 1020.0)
    guard.tick()  # restart
    guard.tick()  # escalate
    assert any("git -C /opt/dbrain revert" in a for a in alerts)
    # The guard suggests the rollback; it never performs one.
    assert not any("revert" in a and "выполнил" in a for a in alerts)


def test_recovery_rearms_the_budget(tmp_path):
    clock = Clock()
    guard, _, _ = _guard(tmp_path, clock, restart_cooldown=10.0, max_restarts=1)
    assert guard.decide(_streak(3, start=1000.0)).action == "restart"
    clock.t += 20
    assert guard.decide(_streak(4, start=1000.0)).action == "escalate"
    assert guard.decide(Health()).action == "none"  # delivered again
    clock.t += 20
    assert guard.decide(_streak(3, start=5000.0)).action == "restart"


def test_a_failing_restart_does_not_claim_success(tmp_path):
    alerts: list[str] = []
    guard = DeliveryGuard(
        tmp_path,
        restart_fn=lambda _u: False,
        alert_fn=alerts.append,
        clock_fn=Clock(),
    )
    for t in (1000.0, 1010.0, 1020.0):
        ask_health.record(tmp_path, "timeout", clock_fn=lambda t=t: t)
    assert guard.tick().action == "restart"
    assert alerts == []


def test_tick_survives_a_broken_restarter(tmp_path):
    def boom(_unit):
        raise RuntimeError("systemctl exploded")

    guard = DeliveryGuard(tmp_path, restart_fn=boom, clock_fn=Clock())
    for t in (1000.0, 1010.0, 1020.0):
        ask_health.record(tmp_path, "timeout", clock_fn=lambda t=t: t)
    assert guard.tick().action == "none"  # swallowed, loop lives on


def test_watchdog_runs_the_guard_every_tick(tmp_path):
    """Wiring check: a READY brain must not stop the guard from looking."""
    from d_brain.services.tmux_parse import PaneState
    from d_brain.services.watchdog import Watchdog

    class FakeSession:
        def is_healthy(self):
            return True

        def is_working(self):
            return False

        def current_state(self):
            return PaneState.READY

        def pop_orphan_replies(self):
            return []

    class SpyGuard:
        def __init__(self):
            self.ticks = 0

        def tick(self):
            self.ticks += 1

    spy = SpyGuard()
    wd = Watchdog(
        FakeSession(),
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10**12,
        delivery_guard=spy,
    )
    assert wd.check_once() == "healthy"
    assert spy.ticks == 1


def test_watchdog_runs_the_guard_every_tick_even_while_rate_limited(tmp_path):
    """The guard sits above the RATE_LIMITED early-return in check_once() —
    a stuck subscription limit must not blind delivery_guard to failures on
    whatever the bot is doing meanwhile.
    """
    from d_brain.services.tmux_parse import PaneState
    from d_brain.services.watchdog import Watchdog

    class FakeSession:
        def is_healthy(self):
            return True

        def is_working(self):
            return False

        def current_state(self):
            return PaneState.RATE_LIMITED

        def nudge(self, text):
            return True

        def pop_orphan_replies(self):
            return []

    class SpyGuard:
        def __init__(self):
            self.ticks = 0

        def tick(self):
            self.ticks += 1

    spy = SpyGuard()
    wd = Watchdog(
        FakeSession(),
        runtime_dir=tmp_path,
        disk_free_fn=lambda: 10**12,
        delivery_guard=spy,
    )
    result = wd.check_once()
    assert result in ("rate_limited", "limit_resumed")
    assert spy.ticks == 1


# ── C4: systemd scope / unit parameterization ────────────────────────────


def test_restart_argv_user_scope_is_todays_command_exactly():
    """The "user" branch must be byte-identical to the pre-C4 hardcoded argv;
    anything else is a silent change to the live restart path."""
    from d_brain.services.delivery_guard import restart_argv

    assert restart_argv("dbrain-bot.service") == [
        "systemctl",
        "--user",
        "restart",
        "--no-block",
        "dbrain-bot.service",
    ]
    assert restart_argv("dbrain-bot.service", "user") == restart_argv(
        "dbrain-bot.service"
    )


def test_restart_argv_system_scope_uses_sudo_n():
    """System units need the sudoers whitelist; -n must never prompt, so a
    missing rule fails fast instead of hanging the watchdog tick."""
    from d_brain.services.delivery_guard import restart_argv

    argv = restart_argv("dbrain-bot@second.service", "system")
    assert argv == [
        "sudo",
        "-n",
        "systemctl",
        "restart",
        "--no-block",
        "dbrain-bot@second.service",
    ]
    assert "--user" not in argv


def test_guard_restarts_the_configured_unit(tmp_path):
    """The guard must restart ITS OWN instance's unit, not the default."""
    clock = Clock()
    guard, restarts, alerts = _guard(
        tmp_path, clock, unit="dbrain-bot@second.service", fail_threshold=3
    )
    for t in (clock.t - 60, clock.t - 30, clock.t):
        ask_health.record(tmp_path, "timeout", clock_fn=lambda t=t: t)
    guard.tick()
    assert restarts == ["dbrain-bot@second.service"]
    assert "dbrain-bot@second.service" in alerts[0]
    assert "dbrain-bot.service." not in alerts[0]


def test_escalate_text_user_scope_unchanged(tmp_path):
    """Default scope keeps the exact operator instructions the runbook
    documents."""
    from d_brain.services.delivery_guard import (
        ESCALATE_MSG,
        _journal_flag,
        _restart_cmd,
    )

    msg = ESCALATE_MSG.format(
        n=2,
        streak=5,
        repo="/repo",
        unit="dbrain-bot.service",
        journal_flag=_journal_flag("user"),
        restart_cmd=_restart_cmd("dbrain-bot.service", "user"),
    )
    assert "journalctl --user -u dbrain-bot.service -n 100" in msg
    assert "systemctl --user restart dbrain-bot.service" in msg
    assert "sudo" not in msg


def test_escalate_text_system_scope_targets_the_instance():
    from d_brain.services.delivery_guard import (
        ESCALATE_MSG,
        _journal_flag,
        _restart_cmd,
    )

    unit = "dbrain-bot@second.service"
    msg = ESCALATE_MSG.format(
        n=2,
        streak=5,
        repo="/repo",
        unit=unit,
        journal_flag=_journal_flag("system"),
        restart_cmd=_restart_cmd(unit, "system"),
    )
    # No --user anywhere, and no double space where the flag used to be.
    assert f"journalctl -u {unit} -n 100" in msg
    # No `-n` in the human-facing copy-paste command (see _restart_cmd).
    assert f"sudo systemctl restart {unit}" in msg
    assert "sudo -n" not in msg
    assert "--user" not in msg
    assert "journalctl  -u" not in msg


# The sudoers file lists one pair of units per instance and is maintainer-only
# (it is not part of the distribution), so both tests below read the instance
# names out of the file itself instead of carrying them as fixtures: a checkout
# without the file has nothing to pin, and no instance name lives in the test.
def _sudoers_text():
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent / "deploy" / "systemd" / "sudoers-dbrain"
    )
    if not path.exists():
        import pytest

        pytest.skip("sudoers-dbrain is not part of this checkout")
    return path.read_text()


def _sudoers_instances(text):
    import re

    return sorted(set(re.findall(r"dbrain-bot@([A-Za-z0-9_.-]+)\.service", text)))






# ── what the streak means now ─────────────────────────────────────────


def _apology_receipt(tmp_path, at: float) -> None:
    """The receipt the bot mints for its OWN brush-off.

    Every failed turn produces one: `_record_health` writes the ledger row,
    then the apology text goes out through `send_response` → the outbox →
    a receipt stamped after that row. This is the shape that killed the
    "a receipt newer than the last failure means the channel works" veto —
    it is present in every streak the guard exists for.
    """
    import json

    d = tmp_path / "outbox"
    d.mkdir(parents=True, exist_ok=True)
    (d / "receipts.json").write_text(json.dumps({"ids": [f"{int(at * 1e9):019d}"]}))


def test_a_streak_of_brush_offs_still_restarts_although_they_were_delivered(tmp_path):
    """THE REAL CASE, and the one a timestamp veto would have swallowed.

    Three turns in a row where the person got nothing but an apology. The
    apologies themselves were delivered perfectly — receipts and all — which
    is exactly why "something reached Telegram since the last failure" is
    not evidence that anything is working. What the ledger now says is
    narrow and true: three turns, no answer by any route."""
    clock = Clock(1100.0)
    guard, restarts, alerts = _guard(tmp_path, clock)
    ask_health.record(tmp_path, "busy", clock_fn=lambda: 1000.0)
    ask_health.record(tmp_path, "busy", clock_fn=lambda: 1030.0)
    ask_health.record(tmp_path, "busy", clock_fn=lambda: 1060.0)
    _apology_receipt(tmp_path, 1060.5)  # the brush-off for the third turn

    assert guard.tick().action == "restart"
    assert restarts == ["dbrain-bot.service"]
    assert len(alerts) == 1


def test_a_turn_answered_by_any_route_clears_the_streak_before_the_guard(tmp_path):
    """NO FALSE ALARM. The false-alarm morning, in ledger terms: the duty
    session covered those turns and the watchdog delivered the late ones, so
    each of them now scores `ok` where it is written — and the guard never
    sees a streak at all. The fix lives in the ledger, not in a veto here;
    this pins that the guard honours it."""
    guard, restarts, alerts = _guard(tmp_path, Clock(1100.0))
    ask_health.record(tmp_path, "busy", clock_fn=lambda: 1000.0)
    ask_health.record(tmp_path, "busy", clock_fn=lambda: 1030.0)
    ask_health.record(tmp_path, "ok", clock_fn=lambda: 1060.0)
    ask_health.record(tmp_path, "busy", clock_fn=lambda: 1070.0)

    assert guard.tick().action == "none"
    assert restarts == []
    assert alerts == []


def test_decide_takes_no_delivery_evidence(tmp_path):
    """A regression guard on the design decision itself: `decide` is a pure
    function of the ledger. Re-introducing a receipt/loss parameter here is
    the change that disarmed the backstop — read its docstring first."""
    import inspect

    guard, _restarts, _alerts = _guard(tmp_path, Clock())
    assert list(inspect.signature(guard.decide).parameters) == ["health"]
