"""Tests for the hook-written turn state (services/turn_state.py) and the
ClaudeSession._turn_open predicate built on it."""

import json
import logging
import os
import time
from pathlib import Path

import pytest

from d_brain.services import turn_state
from d_brain.services.tmux_parse import is_agents_wait_only, is_main_turn_active

FIXTURES = Path(__file__).parent / "fixtures"
SID = "test-sid"


@pytest.fixture(autouse=True)
def _reset_process_state(monkeypatch):
    monkeypatch.setattr(turn_state, "_last_decision", {})
    monkeypatch.setattr(turn_state, "_last_disagree", {})


@pytest.fixture
def claude_pid(monkeypatch):
    """os.getpid() stands in for the CLI; 'is it claude' is patched."""
    monkeypatch.setattr(turn_state, "_is_claude_proc", lambda pid: pid == os.getpid())
    return os.getpid()


def write_state(rd: Path, **over) -> dict:
    now = time.time()
    data = {
        "state": "open",
        "event": "UserPromptSubmit",
        "ts": now,
        "last_event_ts": now,
        "session_id": SID,
        "cli_pid": os.getpid(),
        "background_tasks": 0,
        "origin": "user",
    }
    data.update(over)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / turn_state.STATE_FILE).write_text(json.dumps(data))
    return data


def events(rd: Path) -> list[dict]:
    p = rd / turn_state.EVENTS_FILE
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line]


THINKING = (
    "✢ Pondering… (12s · ↓ 300 tokens · esc to interrupt)\n"
    "──────────\n❯\n──────────\n  ? for shortcuts\n"
)
EMPTY = "──────────\n❯\n──────────\n  ? for shortcuts\n"


def test_screen_samples_mean_what_the_tests_assume():
    assert is_main_turn_active(THINKING) and not is_agents_wait_only(THINKING)
    assert not turn_state.screen_busy(EMPTY)


# ── read / valid ────────────────────────────────────────────────────────


def test_read_missing_file(tmp_path):
    assert turn_state.read(tmp_path) is None


def test_read_broken_json(tmp_path):
    (tmp_path / turn_state.STATE_FILE).write_text("{not json")
    assert turn_state.read(tmp_path) is None


def test_read_roundtrip(tmp_path):
    write_state(tmp_path, background_tasks=2, origin=None)
    st = turn_state.read(tmp_path)
    assert st.state == "open" and st.session_id == SID
    assert st.background_tasks == 2 and st.origin is None


def test_valid_ok(tmp_path, claude_pid):
    write_state(tmp_path)
    assert turn_state.valid(turn_state.read(tmp_path), session_id=SID, now=time.time())


def test_valid_foreign_session_id(tmp_path, claude_pid):
    write_state(tmp_path, session_id="other")
    st = turn_state.read(tmp_path)
    assert not turn_state.valid(st, session_id=SID, now=time.time())
    assert not turn_state.valid(st, session_id=None, now=time.time())


def test_valid_dead_pid(tmp_path):
    write_state(tmp_path, cli_pid=2**22 + 12345)
    st = turn_state.read(tmp_path)
    assert not turn_state.valid(st, session_id=SID, now=time.time())


def test_valid_pid_not_claude(tmp_path):
    import subprocess

    proc = subprocess.Popen(["sleep", "30"])
    try:
        write_state(tmp_path, cli_pid=proc.pid)
        st = turn_state.read(tmp_path)
        assert not turn_state.valid(st, session_id=SID, now=time.time())
    finally:
        proc.kill()
        proc.wait()


def test_valid_ts_from_future(tmp_path, claude_pid):
    now = time.time()
    write_state(tmp_path, ts=now + 60)
    st = turn_state.read(tmp_path)
    assert not turn_state.valid(st, session_id=SID, now=now)
    write_state(tmp_path, ts=now + 3)
    assert turn_state.valid(turn_state.read(tmp_path), session_id=SID, now=now)


def test_valid_bad_state(tmp_path, claude_pid):
    write_state(tmp_path, state="weird")
    assert not turn_state.valid(
        turn_state.read(tmp_path), session_id=SID, now=time.time()
    )


def test_valid_none():
    assert not turn_state.valid(None, session_id=SID, now=time.time())


# ── suspicious_open / close / append_event ─────────────────────────────


def test_suspicious_open():
    now = 10_000.0
    mk = lambda **kw: turn_state.TurnState(  # noqa: E731
        **{
            "state": "open",
            "event": "x",
            "ts": now - 700,
            "last_event_ts": now - 400,
            "session_id": SID,
            "cli_pid": 1,
            **kw,
        }
    )
    assert turn_state.suspicious_open(mk(), now)
    assert not turn_state.suspicious_open(mk(ts=now - 500), now)
    assert not turn_state.suspicious_open(mk(last_event_ts=now - 100), now)
    assert not turn_state.suspicious_open(mk(state="closed"), now)


def test_close_keeps_identity(tmp_path):
    write_state(tmp_path, cli_pid=4242)
    turn_state.close(tmp_path, event="bot_interrupt", reason="test")
    raw = json.loads((tmp_path / turn_state.STATE_FILE).read_text())
    assert raw["state"] == "closed" and raw["event"] == "bot_interrupt"
    assert raw["session_id"] == SID and raw["cli_pid"] == 4242


def test_close_without_file_and_on_missing_dir(tmp_path):
    turn_state.close(tmp_path, event="restart")
    assert turn_state.read(tmp_path).state == "closed"
    turn_state.close(tmp_path / "nope", event="restart")  # must not raise


def test_append_event_and_cap(tmp_path, monkeypatch):
    turn_state.append_event(tmp_path, "x", a=1)
    assert events(tmp_path)[0]["kind"] == "x"
    assert events(tmp_path)[0]["a"] == 1
    from d_brain.services import turn_metrics

    monkeypatch.setattr(turn_metrics, "MAX_BYTES", 2000)
    for i in range(200):
        turn_state.append_event(tmp_path, "y", i=i)
    p = tmp_path / turn_state.EVENTS_FILE
    assert p.stat().st_size < 4000
    evs = events(tmp_path)  # every line still valid JSON
    assert evs[-1]["i"] == 199
    turn_state.append_event(tmp_path / "missing", "z")  # must not raise


# ── turn_open ───────────────────────────────────────────────────────────


def test_turn_open_without_file_uses_screen(tmp_path):
    assert turn_state.turn_open(tmp_path, THINKING, SID) is True
    assert turn_state.turn_open(tmp_path, EMPTY, SID) is False
    decisions = [e for e in events(tmp_path) if e["kind"] == "decision"]
    assert [(d["source"], d["open"]) for d in decisions] == [
        ("screen", True),
        ("screen", False),
    ]


def test_decision_not_spammed(tmp_path):
    for _ in range(5):
        turn_state.turn_open(tmp_path, THINKING, SID)
    assert len([e for e in events(tmp_path) if e["kind"] == "decision"]) == 1


def test_turn_open_hooks_closed_beats_busy_screen(tmp_path, claude_pid, caplog):
    write_state(tmp_path, state="closed")
    with caplog.at_level(logging.INFO, logger="d_brain.services.turn_state"):
        assert turn_state.turn_open(tmp_path, THINKING, SID) is False
        assert turn_state.turn_open(tmp_path, THINKING, SID) is False
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs.count("turn-state: screen disagrees (hooks=closed, screen=busy)") == 1
    assert "turn-source=hooks" in msgs


def test_turn_open_hooks_open_beats_empty_screen(tmp_path, claude_pid):
    write_state(tmp_path, state="open")
    assert turn_state.turn_open(tmp_path, EMPTY, SID, capture_again=lambda: EMPTY)


def test_quiet_turn_with_spinner_is_not_closed(tmp_path, claude_pid):
    now = time.time()
    write_state(tmp_path, ts=now - 1000, last_event_ts=now - 900)
    assert turn_state.turn_open(tmp_path, THINKING, SID, capture_again=lambda: THINKING)
    assert turn_state.read(tmp_path).state == "open"


def test_stale_open_needs_two_empty_screens(tmp_path, claude_pid):
    now = time.time()
    write_state(tmp_path, ts=now - 1000, last_event_ts=now - 900)
    assert turn_state.turn_open(tmp_path, EMPTY, SID, capture_again=lambda: THINKING)
    assert turn_state.read(tmp_path).state == "open"


def test_stale_open_overridden(tmp_path, claude_pid, caplog):
    now = time.time()
    write_state(tmp_path, ts=now - 1000, last_event_ts=now - 900)
    with caplog.at_level(logging.WARNING, logger="d_brain.services.turn_state"):
        assert not turn_state.turn_open(
            tmp_path, EMPTY, SID, capture_again=lambda: EMPTY
        )
    st = turn_state.read(tmp_path)
    assert st.state == "closed" and st.event == "stale_override"
    assert st.session_id == SID
    assert "turn-state: stale open overridden" in [
        r.getMessage() for r in caplog.records
    ]
    assert any(e["kind"] == "stale_override" for e in events(tmp_path))


def test_invalid_file_falls_back_to_screen(tmp_path, claude_pid):
    write_state(tmp_path, state="closed", session_id="old-session")
    assert turn_state.turn_open(tmp_path, THINKING, SID) is True


# ── ClaudeSession integration ───────────────────────────────────────────


def _session(tmp_path, captures):
    from test_claude_session import FakeTmux, make_session

    clock = {"now": 0.0}
    fake = FakeTmux(captures, exists=True)
    s = make_session(tmp_path, fake, clock)
    return s, fake


def test_session_turn_open_hooks_closed_with_trap_screen(tmp_path, claude_pid):
    trap = (FIXTURES / "pane_agents_wait_live_turn.txt").read_text(encoding="utf-8")
    s, _ = _session(tmp_path, [trap])
    (tmp_path / ".dbrain" / "session_id").write_text(SID + "\n")
    assert s._turn_open(trap) == turn_state.screen_busy(trap)  # no file → screen
    write_state(tmp_path / ".dbrain", state="closed")
    assert s._turn_open(trap) is False
    assert s.is_pane_turn_active() is False


def test_session_turn_open_hooks_open_with_empty_screen(tmp_path, claude_pid):
    s, _ = _session(tmp_path, [EMPTY])
    (tmp_path / ".dbrain" / "session_id").write_text(SID + "\n")
    write_state(tmp_path / ".dbrain", state="open")
    assert s._turn_open(EMPTY) is True


def test_session_interrupt_writes_closed(tmp_path, claude_pid):
    s, _ = _session(tmp_path, [EMPTY])
    write_state(tmp_path / ".dbrain", state="open")
    s.interrupt()
    st = turn_state.read(tmp_path / ".dbrain")
    assert st.state == "closed" and st.event == "bot_interrupt"
    assert st.session_id == SID


def test_session_kill_writes_closed(tmp_path, claude_pid):
    s, _ = _session(tmp_path, [EMPTY])
    write_state(tmp_path / ".dbrain", state="open")
    s.kill()
    assert turn_state.read(tmp_path / ".dbrain").event == "restart"


# ── pre-send through ask() ──────────────────────────────────────────────


def test_ask_sends_at_once_when_hooks_closed_despite_busy_screen(tmp_path, claude_pid):
    from test_claude_session import _complete

    rid = "rid00001"
    s, fake = _session(tmp_path, [THINKING] * 8 + [_complete(rid)])
    (tmp_path / ".dbrain" / "session_id").write_text(SID + "\n")
    write_state(tmp_path / ".dbrain", state="closed")
    s.ask("ping", timeout=600)
    subs = fake.sent_subcommands()
    assert "paste-buffer" in subs
    assert subs[: subs.index("paste-buffer")].count("capture-pane") <= 3


def test_ask_waits_when_hooks_open_despite_empty_screen(tmp_path, claude_pid):
    s, fake = _session(tmp_path, [EMPTY])
    (tmp_path / ".dbrain" / "session_id").write_text(SID + "\n")
    write_state(tmp_path / ".dbrain", state="open")
    res = s.ask("ping", timeout=30)
    assert res.status in ("busy", "busy_active")
    assert "paste-buffer" not in fake.sent_subcommands()
