"""Tests for the daily pipeline entrypoint (no claude -p)."""

from d_brain.pipeline import run


class FakeProc:
    def __init__(self, daily=None):
        self._daily = daily or {}

    def process_daily(self):
        return self._daily


def test_run_daily_returns_report_and_ok():
    proc = FakeProc(daily={"report": "<b>done</b>", "processed_entries": 1})
    text, ok = run("daily", proc)
    assert ok
    assert text == "<b>done</b>"


def test_run_daily_error_is_not_ok():
    proc = FakeProc(daily={"error": "no daily file", "processed_entries": 0})
    text, ok = run("daily", proc)
    assert not ok
    assert "no daily file" in text


def test_run_weekly_is_unknown_command():
    """v3.0: the weekly digest is removed; 'weekly' is no longer a command."""
    text, ok = run("weekly", FakeProc())
    assert not ok
    assert "unknown" in text.lower()


def test_run_unknown_command():
    text, ok = run("frobnicate", FakeProc())
    assert not ok
    assert "unknown" in text.lower()
