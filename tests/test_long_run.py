"""Tests for long_run.py — unattended long-turn tracking. Table-driven over
the pure `next_state` transition, plus the read/write/clear file plumbing
and the `is_active` staleness guard."""

from d_brain.services import long_run
from d_brain.services.long_run import LongRun

# ── next_state: table-driven ──────────────────────────────────────────────


def test_idle_to_active_starts_a_run():
    prev = LongRun()
    new, event = long_run.next_state(
        prev, main_turn_active=True, attended=False, now=100.0, alert_after=900.0
    )
    assert event == "started"
    assert new.since == 100.0
    assert new.updated_ts == 100.0
    assert new.alerted is False


def test_repeated_active_below_threshold_is_none():
    prev = LongRun(since=100.0, updated_ts=100.0, alerted=False)
    new, event = long_run.next_state(
        prev, main_turn_active=True, attended=False, now=200.0, alert_after=900.0
    )
    assert event == "none"
    assert new.since == 100.0  # unchanged
    assert new.updated_ts == 200.0  # refreshed
    assert new.alerted is False


def test_crossing_alert_after_fires_alert_exactly_once():
    # Starts at a NONZERO "now" deliberately: since == 0.0 is the sentinel
    # for "no run in progress" (see LongRun's docstring) — starting exactly
    # at wall-clock 0.0 would make the freshly-started run indistinguishable
    # from "not started", which never happens with a real time.time() but
    # would be a false collision in this test.
    prev = LongRun()
    new, event = long_run.next_state(
        prev, main_turn_active=True, attended=False, now=1000.0, alert_after=900.0
    )
    assert event == "started"
    assert new.since == 1000.0
    # Advance to just past the threshold.
    new, event = long_run.next_state(
        new, main_turn_active=True, attended=False, now=1900.0, alert_after=900.0
    )
    assert event == "alert"
    assert new.alerted is True
    # A later tick, still past threshold, must NOT alert again (latched).
    new, event = long_run.next_state(
        new, main_turn_active=True, attended=False, now=2000.0, alert_after=900.0
    )
    assert event == "none"
    assert new.alerted is True


def test_active_to_idle_ends_the_run():
    prev = LongRun(since=100.0, updated_ts=500.0, alerted=True)
    new, event = long_run.next_state(
        prev, main_turn_active=False, attended=False, now=600.0, alert_after=900.0
    )
    assert event == "ended"
    assert new == LongRun()


def test_idle_to_idle_is_none():
    prev = LongRun()
    new, event = long_run.next_state(
        prev, main_turn_active=False, attended=False, now=100.0, alert_after=900.0
    )
    assert event == "none"
    assert new == LongRun()


def test_attended_active_turn_never_starts_a_run():
    """The caller's own ask() holding the lock is an ordinary chat turn, not
    the unattended-cascade shape this module exists to catch."""
    prev = LongRun()
    new, event = long_run.next_state(
        prev, main_turn_active=True, attended=True, now=100.0, alert_after=900.0
    )
    assert event == "none"
    assert new == LongRun()


def test_attended_turn_ends_a_previously_tracked_run():
    """A run that WAS unattended can become attended mid-flight (e.g. the
    user starts typing into it); that must end the tracked run too."""
    prev = LongRun(since=100.0, updated_ts=200.0, alerted=False)
    new, event = long_run.next_state(
        prev, main_turn_active=True, attended=True, now=300.0, alert_after=900.0
    )
    assert event == "ended"
    assert new == LongRun()


def test_alert_after_zero_never_alerts():
    prev = LongRun(since=0.0, updated_ts=0.0, alerted=False)
    new, _ = long_run.next_state(
        prev, main_turn_active=True, attended=False, now=0.0, alert_after=0.0
    )
    for now in (100.0, 10_000.0, 1_000_000.0):
        new, event = long_run.next_state(
            new, main_turn_active=True, attended=False, now=now, alert_after=0.0
        )
        assert event != "alert"
        assert new.alerted is False


# ── read / write / clear ───────────────────────────────────────────────────


def test_missing_file_reads_as_empty(tmp_path):
    assert long_run.read(tmp_path) == LongRun()


def test_write_then_read_roundtrips(tmp_path):
    state = LongRun(since=10.0, updated_ts=20.0, alerted=True)
    assert long_run.write(tmp_path, state) is True
    assert long_run.read(tmp_path) == state


def test_corrupt_file_reads_as_empty(tmp_path):
    long_run.path_for(tmp_path).write_text("{not json")
    assert long_run.read(tmp_path) == LongRun()


def test_clear_removes_the_marker(tmp_path):
    long_run.write(tmp_path, LongRun(since=1.0, updated_ts=1.0))
    long_run.clear(tmp_path)
    assert long_run.read(tmp_path) == LongRun()


def test_clear_on_missing_file_does_not_raise(tmp_path):
    long_run.clear(tmp_path)  # no marker was ever written


def test_write_leaves_no_temp_file_behind(tmp_path):
    long_run.write(tmp_path, LongRun(since=1.0, updated_ts=1.0))
    assert [p.name for p in tmp_path.iterdir()] == [long_run.FILENAME]


# ── is_active: staleness guard ─────────────────────────────────────────────


def test_is_active_true_for_a_fresh_marker(tmp_path):
    long_run.write(tmp_path, LongRun(since=100.0, updated_ts=190.0))
    active, elapsed = long_run.is_active(tmp_path, now=200.0, stale_after=120.0)
    assert active is True
    assert elapsed == 100.0


def test_is_active_false_for_a_stale_marker(tmp_path):
    """Mandatory degrade-to-inactive: if whatever writes this marker dies,
    it must never latch 'active' forever."""
    long_run.write(tmp_path, LongRun(since=100.0, updated_ts=100.0))
    active, elapsed = long_run.is_active(tmp_path, now=500.0, stale_after=120.0)
    assert active is False
    assert elapsed == 0.0


def test_is_active_false_when_no_run_is_tracked(tmp_path):
    active, elapsed = long_run.is_active(tmp_path, now=100.0, stale_after=120.0)
    assert active is False
    assert elapsed == 0.0


def test_is_active_false_for_missing_file(tmp_path):
    active, elapsed = long_run.is_active(tmp_path, now=100.0, stale_after=120.0)
    assert active is False
    assert elapsed == 0.0
