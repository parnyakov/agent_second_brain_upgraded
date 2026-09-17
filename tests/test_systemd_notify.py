"""Tests for the minimal sd_notify client."""

from d_brain.services.systemd_notify import notify, watchdog_interval


def test_notify_no_socket_returns_false(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert notify("READY=1") is False


def test_watchdog_interval_default_when_unset(monkeypatch):
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)
    assert watchdog_interval(60) == 60


def test_watchdog_interval_is_half_of_usec(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "20000000")  # 20s deadline
    assert watchdog_interval() == 10.0


def test_watchdog_interval_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "not-a-number")
    assert watchdog_interval(30) == 30
