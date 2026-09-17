"""Tests for the delivery-health ledger the bot writes and the watchdog reads."""

from d_brain.services import ask_health
from d_brain.services.ask_health import Health


def test_missing_file_reads_as_healthy(tmp_path):
    assert ask_health.read(tmp_path) == Health()


def test_failures_accumulate_into_a_streak(tmp_path):
    ask_health.record(tmp_path, "timeout", clock_fn=lambda: 100.0)
    ask_health.record(tmp_path, "error", clock_fn=lambda: 160.0)
    h = ask_health.read(tmp_path)
    assert h.fail_streak == 2
    assert h.last_status == "error"
    assert h.streak_started_ts == 100.0
    assert h.streak_span == 60.0


def test_a_delivered_reply_clears_the_streak(tmp_path):
    ask_health.record(tmp_path, "timeout", clock_fn=lambda: 1.0)
    ask_health.record(tmp_path, "timeout", clock_fn=lambda: 2.0)
    ask_health.record(tmp_path, "ok", clock_fn=lambda: 3.0)
    h = ask_health.read(tmp_path)
    assert h.fail_streak == 0
    assert h.streak_started_ts == 0.0


def test_rate_limited_is_neutral_not_a_failure(tmp_path):
    """The user gets a truthful message and the pipe is intact — counting it
    would bury the signal (2026-08-20 fired dozens of these in one day)."""
    ask_health.record(tmp_path, "timeout", clock_fn=lambda: 10.0)
    ask_health.record(tmp_path, "rate_limited", clock_fn=lambda: 20.0)
    h = ask_health.read(tmp_path)
    assert h.fail_streak == 1
    assert h.last_status == "rate_limited"
    assert h.streak_started_ts == 10.0


def test_busy_counts_as_a_failure(tmp_path):
    """B3 fix (2026-08-22): a "busy" outcome (pane still busy with a
    previous turn after the full busy-wait budget) must still feed
    delivery_guard's restart backstop — only the user-facing wording is
    friendly, the health accounting is not."""
    ask_health.record(tmp_path, "busy", clock_fn=lambda: 10.0)
    ask_health.record(tmp_path, "busy", clock_fn=lambda: 20.0)
    h = ask_health.read(tmp_path)
    assert h.fail_streak == 2
    assert h.last_status == "busy"


def test_busy_active_is_neutral(tmp_path):
    """agent-infra-backlog item 22: 'busy_active' (a leftover turn that
    demonstrably kept progressing across the whole busy-wait) must NOT feed
    the restart backstop the way plain 'busy' does — it is a live turn, not
    evidence of a wedged pane."""
    ask_health.record(tmp_path, "busy_active", clock_fn=lambda: 10.0)
    ask_health.record(tmp_path, "busy_active", clock_fn=lambda: 20.0)
    ask_health.record(tmp_path, "busy_active", clock_fn=lambda: 30.0)
    h = ask_health.read(tmp_path)
    assert h.fail_streak == 0
    assert h.last_status == "busy_active"


def test_busy_active_does_not_reset_an_existing_fail_streak(tmp_path):
    """Only 'ok' resets the streak — 'busy_active' is neutral, same as
    rate_limited/logged_out, not a second kind of success."""
    ask_health.record(tmp_path, "timeout", clock_fn=lambda: 1.0)
    ask_health.record(tmp_path, "busy_active", clock_fn=lambda: 2.0)
    h = ask_health.read(tmp_path)
    assert h.fail_streak == 1
    assert h.last_status == "busy_active"


def test_streak_span_needs_two_failures():
    assert Health(1, "timeout", 500.0, 500.0).streak_span == 0.0


def test_corrupt_file_reads_as_healthy(tmp_path):
    ask_health.path_for(tmp_path).write_text("{not json")
    assert ask_health.read(tmp_path) == Health()


def test_unwritable_dir_does_not_raise(tmp_path):
    """Recording runs on the reply path of every message; it must never be
    the reason a reply fails to go out."""
    blocked = tmp_path / "file-not-a-dir"
    blocked.write_text("x")
    result = ask_health.record(blocked / "sub", "timeout", clock_fn=lambda: 1.0)
    assert result.fail_streak == 1  # in-memory answer still correct


def test_record_leaves_no_temp_file_behind(tmp_path):
    ask_health.record(tmp_path, "ok", clock_fn=lambda: 1.0)
    assert [p.name for p in tmp_path.iterdir()] == [ask_health.FILENAME]
