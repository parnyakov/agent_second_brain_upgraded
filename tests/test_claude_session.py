"""Tests for ClaudeSession orchestration.

The real tmux/subprocess layer is replaced by FakeTmux, which models the
pane as a small state machine and returns scripted capture-pane output.
Clock, sleep and rid generation are injected so polling, timeout
and stall detection are deterministic and fast.
"""

import json
import logging
import subprocess
import time
from pathlib import Path

import pytest

from d_brain.services.claude_session import (
    DEFAULT_BUSY_WAIT_BUDGET,
    DEFAULT_NO_MAIN_TURN_CEILING,
    DEFAULT_SALVAGE_STABLE,
    DEFAULT_STALL_TIMEOUT,
    DEFAULT_TIMEOUT,
    MAINT_PREFIX,
    STATIC_TRUST_WINDOW,
    AskResult,
    ClaudeSession,
)
from d_brain.services.tmux_parse import is_main_turn_active

READY = (
    "────────────────────\n❯\n────────────────────\n"
    "  hello | Opus 4.8 (1M context) | ~/p\n"
    "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
)
TRUST = (
    " Quick safety check: Is this a project you created or one you trust?\n"
    " ❯ 1. Yes, I trust this folder\n   2. No, exit\n"
)
BYPASS = (
    " WARNING: Claude Code running in Bypass Permissions mode\n\n"
    " In Bypass Permissions mode, Claude Code will not ask for your approval.\n"
    " ❯ 1. No, exit\n   2. Yes, I accept\n"
)
THINKING = "  ✻ Working…  (esc to interrupt)\n"
RATE = "  You've reached your usage limit. Your limit resets at 3:00 PM.\n❯\n"
LOGGED_OUT = "  Invalid API key · Please run /login to authenticate.\n❯\n"


def _complete(rid: str, reply: str = "PONG") -> str:
    return f"some echo <<<R:{rid}>>> inline\n<<<R:{rid}>>>\n{reply}\n<<<E:{rid}>>>\n❯\n"


def _inline_echo(rid: str) -> str:
    """Echo of the typed prompt: markers appear INLINE (mid-sentence)."""
    return f"> reply, wrap between <<<R:{rid}>>> and <<<E:{rid}>>> markers\n{THINKING}"


class FakeTmux:
    """Callable stand-in for subprocess.run over `tmux ...`."""

    def __init__(
        self,
        capture_script: list[str],
        exists: bool = False,
        window_size: str = "200x50",
    ) -> None:
        self._captures = list(capture_script)
        self.exists = exists
        self.calls: list[list[str]] = []
        # What `display-message -p '#{window_width}x#{window_height}'` reports;
        # the default is already the wanted geometry, so tests that don't care
        # see no resize traffic.
        self.window_size = window_size

    @staticmethod
    def _subcommand(args: list[str]) -> str:
        """The tmux subcommand name, skipping a leading `-f <config>` (B4
        fix: `-f` is a top-level tmux client flag issued BEFORE the
        subcommand, e.g. `tmux -f deploy/tmux.conf new-session ...`)."""
        body = args[1:] if args and args[0] == "tmux" else list(args)
        i = 0
        if body[i : i + 1] == ["-f"]:
            i += 2  # skip "-f" and its config-path value
        return body[i] if i < len(body) else ""

    def __call__(self, args, **kwargs):  # noqa: ANN001
        self.calls.append(args)
        sub = self._subcommand(args)
        out, rc = "", 0
        if sub == "has-session":
            rc = 0 if self.exists else 1
        elif sub == "new-session":
            self.exists = True
        elif sub == "display-message":
            out = self.window_size + "\n"
        elif sub == "capture-pane":
            out = (
                self._captures.pop(0)
                if len(self._captures) > 1
                else (self._captures[0] if self._captures else "")
            )
        return subprocess.CompletedProcess(args, rc, stdout=out, stderr="")

    def sent_subcommands(self) -> list[str]:
        return [self._subcommand(c) for c in self.calls if c and c[0] == "tmux"]

    def sent_keys(self) -> list[list[str]]:
        return [c for c in self.calls if len(c) > 1 and c[1] == "send-keys"]

    def enter_count(self) -> int:
        return sum(1 for c in self.sent_keys() if c[-1] == "Enter")


@pytest.fixture
def clock():
    return {"now": 0.0}


def make_session(
    tmp_path: Path,
    fake: FakeTmux,
    clock: dict,
    *,
    rid: str = "rid00001",
    tmux_config: Path | None = None,
    stall_timeout: float = 10.0,
) -> ClaudeSession:
    def sleep_fn(seconds: float) -> None:
        clock["now"] += seconds

    return ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=sleep_fn,
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=stall_timeout,
        tmux_config=tmux_config,
    )


# ── ensure_session ──────────────────────────────────────────────────────


def test_ensure_session_creates_when_absent(tmp_path, clock):
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    assert "new-session" in fake.sent_subcommands()
    assert "pipe-pane" in fake.sent_subcommands()
    assert (tmp_path / ".dbrain" / "ready").exists()


def test_ensure_session_noop_when_present(tmp_path, clock):
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    assert "new-session" not in fake.sent_subcommands()


def test_ensure_session_handles_trust_prompt(tmp_path, clock):
    fake = FakeTmux([TRUST, READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    assert fake.enter_count() >= 1


def test_ensure_session_does_not_enter_spam_on_trust(tmp_path, clock):
    """Trust persists for several captures; Enter must be debounced to the
    transition, not sent on every poll (M2)."""
    fake = FakeTmux([TRUST, TRUST, TRUST, READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    assert fake.enter_count() == 1


def _digit_sends(fake: FakeTmux, digit: str) -> int:
    return sum(1 for c in fake.sent_keys() if c[-1] == digit)


def test_ensure_session_accepts_bypass_prompt(tmp_path, clock):
    """Fresh config dir under --dangerously-skip-permissions shows the bypass
    accept screen; the session must actively pick "2. Yes, I accept" (the safe
    default sits on "1. No, exit") and then reach READY."""
    fake = FakeTmux([BYPASS, READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    assert _digit_sends(fake, "2") == 1
    assert (tmp_path / ".dbrain" / "ready").exists()


def test_ensure_session_does_not_spam_bypass(tmp_path, clock):
    """Bypass persists for several captures; "2" debounced to the transition."""
    fake = FakeTmux([BYPASS, BYPASS, BYPASS, READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    assert _digit_sends(fake, "2") == 1


def test_ensure_session_raises_if_never_ready(tmp_path, clock):
    fake = FakeTmux([THINKING], exists=False)
    s = make_session(tmp_path, fake, clock)
    with pytest.raises(RuntimeError):
        s.ensure_session()


# ── R1: pinned session id (Fable audit) ──────────────────────────────────


def test_new_session_pins_session_id_via_session_id_flag(tmp_path, clock):
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    session_id_file = tmp_path / ".dbrain" / "session_id"
    assert session_id_file.exists()
    sid = session_id_file.read_text().strip()
    assert sid  # a UUID was generated
    new_session_calls = [c for c in fake.calls if len(c) > 1 and c[1] == "new-session"]
    assert new_session_calls
    start_command = new_session_calls[0][-1]
    assert f"--session-id {sid}" in start_command


def test_history_limit_is_set_globally_before_new_session(tmp_path, clock):
    """R4 fix: history-limit must be applied via `-g` BEFORE `new-session` —
    the previous per-session `-t` call AFTER new-session was a verified
    no-op (see the module docstring / change contract for the live repro)."""
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    calls = [c for c in fake.calls if len(c) > 1 and c[0] == "tmux"]
    set_opt_idx = next(
        i
        for i, c in enumerate(calls)
        if c[1] == "set-option" and "history-limit" in c
    )
    new_session_idx = next(
        i for i, c in enumerate(calls) if "new-session" in c
    )
    assert set_opt_idx < new_session_idx
    assert "-g" in calls[set_opt_idx]


def test_new_session_passes_tmux_config_via_f_flag_for_cold_start(tmp_path, clock):
    """B4 fix: the `-g set-option` above only works once a tmux SERVER
    already exists to hold the global default — on a genuinely cold start
    that call itself fails before any server exists, and `new-session`
    would then fall back to tmux's own built-in history-limit (2000).
    Verified live: `-f <config>` on the SAME invocation that creates the
    session covers the cold case too, since it makes tmux read the config
    as part of starting the new server."""
    conf = tmp_path / "tmux.conf"
    conf.write_text("set -g history-limit 50000\n")
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock, tmux_config=conf)
    s.ensure_session()
    new_session_calls = [
        c for c in fake.calls if len(c) > 1 and c[0] == "tmux" and "new-session" in c
    ]
    assert new_session_calls
    call = new_session_calls[0]
    assert call[1] == "-f"
    assert call[2] == str(conf)
    assert call[3] == "new-session"


def test_new_session_skips_f_flag_when_tmux_config_missing(tmp_path, clock, caplog):
    """A configured-but-missing tmux_config must degrade to the `-g`
    set-option path alone (still correct on warm starts) rather than crash
    or silently pass a bogus -f path to tmux."""
    conf = tmp_path / "does-not-exist.conf"
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock, tmux_config=conf)
    with caplog.at_level("WARNING"):
        s.ensure_session()
    new_session_calls = [
        c for c in fake.calls if len(c) > 1 and c[0] == "tmux" and "new-session" in c
    ]
    assert new_session_calls
    assert new_session_calls[0][1] == "new-session"  # no -f prepended
    assert "not found" in caplog.text


def test_new_session_omits_f_flag_when_no_tmux_config_given(tmp_path, clock):
    """The common/default case (tmux_config=None) must be byte-identical to
    pre-B4 behavior — no -f anywhere in the new-session call."""
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)  # tmux_config defaults to None
    s.ensure_session()
    new_session_calls = [
        c for c in fake.calls if len(c) > 1 and c[0] == "tmux" and "new-session" in c
    ]
    assert new_session_calls
    assert new_session_calls[0][1] == "new-session"


def test_existing_session_keeps_its_pinned_session_id(tmp_path, clock):
    """A tmux session that already exists (the common case: the bot process
    merely restarted) must NOT get a fresh session id — the `claude` process
    still running inside it is still writing to the ORIGINAL transcript."""
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    session_id_file = tmp_path / ".dbrain" / "session_id"
    session_id_file.write_text("pre-existing-id\n")
    s.ensure_session()
    assert session_id_file.read_text().strip() == "pre-existing-id"
    assert not any(len(c) > 1 and c[1] == "new-session" for c in fake.calls)


def test_current_transcript_path_none_before_any_session(tmp_path, clock):
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    assert s.current_transcript_path() is None


def test_current_transcript_path_after_session_start(tmp_path, clock):
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    sid = (tmp_path / ".dbrain" / "session_id").read_text().strip()
    path = s.current_transcript_path()
    assert path is not None
    assert path.name == f"{sid}.jsonl"


def test_clear_resyncs_session_id_to_a_new_transcript_file(
    tmp_path, clock, monkeypatch
):
    """VERIFIED LIVE (2026-08-22, isolated throwaway session): `/clear`
    makes the CLI start an entirely NEW internal session id/transcript.
    clear() must re-pin runtime_dir/session_id to whichever NEW file shows
    up, not the cron session's (which shares the same project directory).

    B1 fix (2026-08-22): the resync now lives in ``send_control()`` itself
    (``clear()`` is a thin wrapper over it), so this monkeypatches the
    actual typing primitive (``_send_text``) rather than ``send_control`` —
    stubbing out ``send_control`` entirely would skip the very code path
    this test exists to exercise.
    """
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    (tmp_path / ".dbrain" / "session_id").write_text("old-id\n")

    project_dir = tmp_path / "claude-projects"
    monkeypatch.setattr(s, "_transcript_project_dir", lambda: project_dir)
    project_dir.mkdir()
    (project_dir / "old-id.jsonl").touch()
    (project_dir / "concurrent-cron-session.jsonl").touch()  # pre-existing

    def fake_send_text(text):
        assert text == "/clear"
        # Simulate the CLI creating a brand-new transcript file the instant
        # /clear runs.
        (project_dir / "new-id-after-clear.jsonl").touch()

    monkeypatch.setattr(s, "_send_text", fake_send_text)
    monkeypatch.setattr(s, "_send_enter", lambda: None)
    s.clear()

    resynced = (tmp_path / ".dbrain" / "session_id").read_text().strip()
    assert resynced == "new-id-after-clear"


def test_clear_resync_ignores_a_pre_existing_concurrent_session_file(
    tmp_path, clock, monkeypatch
):
    """The cron brain shares this exact same project directory — resync must
    key off a genuinely NEW file, not merely the newest mtime, or a
    concurrently-active cron turn could be mistaken for this session's
    post-clear transcript."""
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    (tmp_path / ".dbrain" / "session_id").write_text("old-id\n")
    project_dir = tmp_path / "claude-projects"
    monkeypatch.setattr(s, "_transcript_project_dir", lambda: project_dir)
    project_dir.mkdir()
    (project_dir / "old-id.jsonl").touch()
    cron_file = project_dir / "cron-session.jsonl"
    cron_file.touch()

    def fake_send_text(text):
        # The cron session touches its OWN pre-existing file (newer mtime)
        # but does NOT create a new file — must not be mistaken for /clear's
        # new session.
        import time as _time

        _time.sleep(0.01)
        cron_file.write_text("more cron activity")

    monkeypatch.setattr(s, "_send_text", fake_send_text)
    monkeypatch.setattr(s, "_send_enter", lambda: None)
    s.clear()
    # No genuinely new file ever appeared, so the pin must be left alone
    # (not clobbered with the cron file) — logged as a failure to resync.
    assert (tmp_path / ".dbrain" / "session_id").read_text().strip() == "old-id"


def test_send_control_clear_resyncs_session_id_directly(tmp_path, clock, monkeypatch):
    """B1 fix: the bot's real `/clear` command (bot/handlers/chat.py's
    `_CONTROL` set) and cron_runner.py's post-job `/clear` both call
    ``send_control("/clear")`` directly — neither ever goes through
    ``clear()``. The resync must fire from ``send_control`` itself, not
    only when reached via ``clear()``'s wrapper."""
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    (tmp_path / ".dbrain" / "session_id").write_text("old-id\n")
    project_dir = tmp_path / "claude-projects"
    monkeypatch.setattr(s, "_transcript_project_dir", lambda: project_dir)
    project_dir.mkdir()
    (project_dir / "old-id.jsonl").touch()

    def fake_send_text(text):
        (project_dir / "new-id-after-clear.jsonl").touch()

    monkeypatch.setattr(s, "_send_text", fake_send_text)
    monkeypatch.setattr(s, "_send_enter", lambda: None)

    s.send_control("/clear")

    resynced = (tmp_path / ".dbrain" / "session_id").read_text().strip()
    assert resynced == "new-id-after-clear"


def test_send_control_non_clear_does_not_resync(tmp_path, clock, monkeypatch):
    """Only the literal `/clear` text triggers the resync scan — other
    control commands (e.g. `/model`) must not pay that cost or touch the
    pinned session id."""
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    (tmp_path / ".dbrain" / "session_id").write_text("old-id\n")
    project_dir = tmp_path / "claude-projects"
    monkeypatch.setattr(s, "_transcript_project_dir", lambda: project_dir)
    project_dir.mkdir()

    def fake_send_text(text):
        (project_dir / "new-id-after-clear.jsonl").touch()

    monkeypatch.setattr(s, "_send_text", fake_send_text)
    monkeypatch.setattr(s, "_send_enter", lambda: None)

    s.send_control("/model")

    assert (tmp_path / ".dbrain" / "session_id").read_text().strip() == "old-id"


# ── R1 shadow mode (diagnostic-only, Fable audit) ─────────────────────────


def _shadow_record(rid: str, text: str, *, is_sidechain: bool = False) -> str:
    import json

    return (
        json.dumps(
            {
                "type": "assistant",
                "isSidechain": is_sidechain,
                "message": {"content": [{"type": "text", "text": text}]},
            }
        )
        + "\n"
    )


def _make_shadow_session(
    tmp_path, fake, clock, rid, transcript_file, monkeypatch, **kw
):
    from d_brain.services.transcript import TranscriptTail

    monkeypatch.setattr(
        "d_brain.services.claude_session.transcript_path",
        lambda work_dir, sid: transcript_file,
    )
    # Anchor the tail at the START of the (pre-written) fixture file rather
    # than its real end — lets a test pre-seed the exact record ask() should
    # see without racing real file-append timing inside a synchronous call.
    monkeypatch.setattr(TranscriptTail, "at_end", classmethod(lambda cls, p: cls(p)))
    (tmp_path / ".dbrain").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".dbrain" / "session_id").write_text("sid-1\n")
    return ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        transcript_shadow_mode=True,
        **kw,
    )


def test_shadow_mode_logs_agreement_between_panel_and_transcript(
    tmp_path, clock, caplog, monkeypatch
):
    rid = "shadow01"
    transcript_file = tmp_path / "shadow.jsonl"
    transcript_file.write_text(
        _shadow_record(rid, f"<<<R:{rid}>>>\nHello world\n<<<E:{rid}>>>\n")
    )
    fake = FakeTmux([READY, _complete(rid, "Hello world")], exists=True)
    s = _make_shadow_session(tmp_path, fake, clock, rid, transcript_file, monkeypatch)
    with caplog.at_level(logging.INFO):
        res = s.ask("ping", timeout=60)
    assert res.status == "ok"
    assert any(
        "shadow mode: panel and transcript agree" in r.message for r in caplog.records
    )


def test_shadow_mode_logs_mismatch_when_text_differs(
    tmp_path, clock, caplog, monkeypatch
):
    rid = "shadow02"
    transcript_file = tmp_path / "shadow.jsonl"
    transcript_file.write_text(
        _shadow_record(
            rid, f"<<<R:{rid}>>>\nDifferent transcript text\n<<<E:{rid}>>>\n"
        )
    )
    fake = FakeTmux([READY, _complete(rid, "Panel text")], exists=True)
    s = _make_shadow_session(tmp_path, fake, clock, rid, transcript_file, monkeypatch)
    with caplog.at_level(logging.WARNING):
        res = s.ask("ping", timeout=60)
    assert res.status == "ok"
    assert res.reply == "Panel text"  # shadow mode NEVER changes delivery
    assert any(
        "shadow mode: panel/transcript MISMATCH" in r.message for r in caplog.records
    )


def test_shadow_mode_rejects_sidechain_records(tmp_path, clock, caplog, monkeypatch):
    """A background subagent's own text (isSidechain) must never be treated
    as a match, even if it happens to carry the same rid/text."""
    rid = "shadow03"
    transcript_file = tmp_path / "shadow.jsonl"
    transcript_file.write_text(
        _shadow_record(
            rid, f"<<<R:{rid}>>>\nHello world\n<<<E:{rid}>>>\n", is_sidechain=True
        )
    )
    fake = FakeTmux([READY, _complete(rid, "Hello world")], exists=True)
    s = _make_shadow_session(tmp_path, fake, clock, rid, transcript_file, monkeypatch)
    with caplog.at_level(logging.WARNING):
        s.ask("ping", timeout=60)
    assert any(
        "transcript path found no matching record" in r.message for r in caplog.records
    )


def test_shadow_mode_logs_consistent_f2_when_neither_side_has_a_marker(
    tmp_path, clock, caplog, monkeypatch
):
    """R1 step 4: when the panel shows no R marker at all (F2) AND the
    transcript shows none either, log this distinctly as 'consistent with
    F2 (nothing to compare)' rather than a mismatch warning."""
    rid = "shadow04"
    transcript_file = tmp_path / "shadow.jsonl"
    transcript_file.write_text("")
    pane = (
        "Worked for 2m 22s · 0 background tasks still running\n"
        f"{_INCIDENT_BOX}\n❯\n{_INCIDENT_BOX}\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    fake = FakeTmux([READY, pane], exists=True)
    s = _make_shadow_session(
        tmp_path,
        fake,
        clock,
        rid,
        transcript_file,
        monkeypatch,
        no_main_turn_ceiling=5.0,
        salvage_stable=5.0,
        stall_timeout=1e6,
    )
    with caplog.at_level(logging.INFO):
        res = s.ask("ping", timeout=20)
    assert res.status == "timeout"
    assert any(
        "consistent with the F2 class" in r.message for r in caplog.records
    )


# ── sending (buffer) ─────────────────────────────────────────────────────


def test_send_text_pipes_long_payload_via_stdin(tmp_path, clock):
    """Regression: a long prompt must be streamed to `tmux load-buffer -` over
    stdin, never passed as an argv element — argv data trips tmux's
    `set-buffer: command too long` and the prompt is silently dropped."""
    recorded: list[tuple[list[str], dict]] = []

    def runner(args, **kwargs):
        recorded.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    s = make_session(tmp_path, runner, clock)
    big = "x" * 200_000
    s._send_text(big)

    loads = [
        (a, k)
        for (a, k) in recorded
        if len(a) > 1 and a[1] in ("set-buffer", "load-buffer")
    ]
    assert loads, "expected a buffer-load tmux call"
    args, kwargs = loads[0]
    assert big not in args, "payload must not be passed as an argv element"
    assert kwargs.get("input") == big, "payload must be piped via stdin"
    assert args[1] == "load-buffer" and "-" in args


def test_send_text_noop_on_empty(tmp_path, clock):
    """`load-buffer -` of zero bytes creates no buffer, so a following
    paste-buffer would fail `no buffer` — empty text must send nothing."""
    recorded: list[list[str]] = []

    def runner(args, **kwargs):
        recorded.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    s = make_session(tmp_path, runner, clock)
    s._send_text("")
    subs = [a[1] for a in recorded if len(a) > 1 and a[0] == "tmux"]
    assert "load-buffer" not in subs and "paste-buffer" not in subs


# ── ask ─────────────────────────────────────────────────────────────────


def test_ask_returns_reply_on_completion(tmp_path, clock):
    rid = "abcd0001"
    fake = FakeTmux([READY, THINKING, _complete(rid)], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    res = s.ask("ping", timeout=60)
    assert isinstance(res, AskResult)
    assert res.ok
    assert res.reply == "PONG"


def test_ask_ignores_inline_echo_contamination(tmp_path, clock):
    """The echoed prompt has inline markers; ask must NOT complete on it,
    only on the real line-anchored answer (H1 at session level)."""
    rid = "echo0001"
    fake = FakeTmux([READY, _inline_echo(rid), _complete(rid)], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    res = s.ask("ping", timeout=60)
    assert res.reply == "PONG"


def test_ask_detects_rate_limit_without_hanging(tmp_path, clock):
    """A banner that persists across polls is trusted (see
    _RATE_LIMIT_CONFIRM_POLLS): FakeTmux repeats its last frame forever, so
    this exercises the persistent case, not a single sample."""
    fake = FakeTmux([RATE], exists=True)
    s = make_session(tmp_path, fake, clock)
    res = s.ask("ping", timeout=60)
    assert res.status == "rate_limited"
    assert not res.ok


def test_ask_ignores_a_transient_rate_limit_looking_frame_before_send(
    tmp_path, clock
):
    """2026-08-20 review, reproduced live: raw tool output (Read/grep/cat) or
    an open, unterminated marker span is never stripped by
    strip_reply_bodies(), so a single poll landing on text that merely
    resembles the banner must not short-circuit an otherwise-healthy turn.
    A real banner is a static full-screen state and survives the next poll;
    this one does not."""
    rid = "rate0001"
    fake = FakeTmux([RATE, READY, _complete(rid)], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    res = s.ask("ping", timeout=60)
    assert res.ok
    assert res.reply == "PONG"


def test_suppressed_rate_limit_false_positive_is_logged(tmp_path, clock, caplog):
    """Round-2 finding: when the RATE_LIMITED signature is seen but does not
    persist across _RATE_LIMIT_CONFIRM_POLLS, _confirmed_rate_limited()
    suppressed it with no log line at all — the one signal that would show,
    after deploy, whether the narrowed _RATE_RE is actually doing its job.
    Must be logged (not silently swallowed) when the signature fails to
    confirm."""
    rid = "rate0003"
    fake = FakeTmux([RATE, READY, _complete(rid)], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    with caplog.at_level("INFO"):
        res = s.ask("ping", timeout=60)
    assert res.ok
    assert "false positive" in caplog.text.lower()


def test_ask_returns_rate_limited_once_it_persists_after_send(tmp_path, clock):
    """The banner appearing only AFTER the prompt was sent (not caught by the
    pre-send check) must still be honored once it persists for
    _RATE_LIMIT_CONFIRM_POLLS polls."""
    fake = FakeTmux([READY, RATE], exists=True)  # RATE repeats after send
    s = make_session(tmp_path, fake, clock)
    res = s.ask("ping", timeout=60)
    assert res.status == "rate_limited"


def test_ask_rate_limit_streak_resets_on_a_transient_mid_turn_frame(
    tmp_path, clock
):
    """A single RATE-looking poll mid-turn that then reverts must not add up
    with an unrelated later occurrence — the streak counts CONSECUTIVE polls
    only."""
    rid = "rate0002"
    fake = FakeTmux([READY, RATE, READY, _complete(rid)], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    res = s.ask("ping", timeout=60)
    assert res.ok
    assert res.reply == "PONG"


def test_ask_detects_logged_out(tmp_path, clock):
    fake = FakeTmux([LOGGED_OUT], exists=True)
    s = make_session(tmp_path, fake, clock)
    res = s.ask("ping", timeout=60)
    assert res.status == "logged_out"


def test_ask_times_out_when_no_reply(tmp_path, clock):
    fake = FakeTmux([READY, THINKING], exists=True)
    # the working spinner is visible, so it's a timeout, not a stall
    s = make_session(tmp_path, fake, clock)
    res = s.ask("ping", timeout=3)
    assert res.status == "timeout"


def test_ask_stall_interrupts_via_escape_not_ctrl_c(tmp_path, clock):
    """A stall releases the pane lock, reports an error, and sends Escape to
    stop the (presumed-wedged) turn — never C-c (2026-08-20, revised after
    review).

    C-c is never used here: a stall is declared exactly when the pane LOOKS
    idle, and C-c on an idle prompt starts Claude Code's exit sequence.
    Escape only ever cancels the response. Interrupting (rather than leaving
    the turn running, as an earlier revision did) matters because a wedged
    turn left running is exactly what later trips watchdog._is_hung() —
    whose only recovery is force_recover(), i.e. `tmux kill-session`,
    destroying the whole long-lived brain session.
    """
    fake = FakeTmux([READY, READY], exists=True)  # never shows work
    s = make_session(tmp_path, fake, clock)
    res = s.ask("ping", timeout=600)
    assert res.status == "error"
    assert "stall" in (res.detail or "").lower()
    assert not any(c[-1] == "C-c" for c in fake.sent_keys())
    assert any(c[-1] == "Escape" for c in fake.sent_keys())
    assert not (tmp_path / ".dbrain" / "inflight").exists()


def test_ask_long_silent_work_not_interrupted(tmp_path, clock):
    """The working spinner is visible → the turn is ALIVE however quiet it
    is. No C-c; the hard timeout (not a stall) ends the wait."""
    fake = FakeTmux([READY, THINKING], exists=True)
    s = make_session(tmp_path, fake, clock)
    res = s.ask("ping", timeout=30)
    assert res.status == "timeout"
    assert not any(c[-1] == "C-c" for c in fake.sent_keys())


# ── bounded trust in the static hint (backlog item 13) ──────────────────
#
# "esc to interrupt" is a footer segment this CLI version shows whenever
# ANYTHING is interruptible (99.6% of its occurrences on a real pane.log),
# so a pane frozen mid-turn kept reporting "working" forever and disarmed
# the hang detectors. Trust in that hint alone now expires after
# STATIC_TRUST_WINDOW of a pane that changed NOTHING — neither chrome nor
# pane.log. Both reset signals are load-bearing: see the streaming test.


def _grow_pane_log(tmp_path: Path, text: str) -> None:
    """Append to the piped transcript the session watches for growth."""
    log = tmp_path / ".dbrain" / "pane.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(text)


def test_is_working_false_once_static_trust_window_expires(tmp_path, clock):
    """Regression (item 13): a pane holding the static hint byte-identical,
    with no pane.log growth, is a FROZEN pane once the window is spent —
    the watchdog's hang detector must be able to see it."""
    fake = FakeTmux([THINKING], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.is_working() is True  # baseline observation
    clock["now"] += 1.0
    assert s.is_working() is True  # frozen from here on
    clock["now"] += STATIC_TRUST_WINDOW
    assert s.is_working() is False


def test_is_working_still_true_inside_static_trust_window(tmp_path, clock):
    """The anti-B3 guard: a legitimately quiet long turn (well past the
    900s stall threshold, still inside the window) must NOT be called dead."""
    fake = FakeTmux([THINKING], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.is_working() is True
    clock["now"] += 1.0
    assert s.is_working() is True
    clock["now"] += DEFAULT_STALL_TIMEOUT + 100.0  # 1000s: past stall, inside window
    assert s.is_working() is True


def test_is_working_static_frame_but_growing_log_stays_true(tmp_path, clock):
    """strip_reply_bodies() removes a streaming reply from chrome, so a
    live turn CAN hold chrome byte-identical while its log grows. Log
    growth resets the window — a "chrome only" implementation would call
    this live turn frozen (a fresh B3-class regression)."""
    fake = FakeTmux([THINKING], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.is_working() is True
    for _ in range(4):
        clock["now"] += STATIC_TRUST_WINDOW / 2
        _grow_pane_log(tmp_path, "streamed reply text\n")
        assert s.is_working() is True
    # Total elapsed is far past the window, yet the pane never went quiet.
    assert clock["now"] > STATIC_TRUST_WINDOW


def test_is_working_ticking_progress_keeps_resetting_the_window(tmp_path, clock):
    """A PROGRESS signature that actually ticks is real movement: every poll
    resets the freeze clock, so trust never expires however long it runs."""
    frames = [f"✢ Razzle-dazzling… ({i}s · ↓1.8k tokens)\n" for i in range(1, 7)]
    fake = FakeTmux(frames, exists=True)
    s = make_session(tmp_path, fake, clock)
    for _ in range(5):
        assert s.is_working() is True
        clock["now"] += STATIC_TRUST_WINDOW * 2


def test_ask_frozen_static_pane_eventually_stalls(tmp_path, clock):
    """Regression (item 13), the practical win: a pane frozen mid-turn with
    the static footer present used to hold ask()'s stall detector disarmed
    for the WHOLE turn budget. It now interrupts about
    (STATIC_TRUST_WINDOW + stall_timeout) in — late, deliberately, but not
    never. Contrast with test_ask_long_silent_work_not_interrupted, which
    proves the short quiet turn is still left alone."""
    fake = FakeTmux([READY, THINKING], exists=True)
    s = make_session(tmp_path, fake, clock)
    res = s.ask("ping", timeout=DEFAULT_TIMEOUT)
    assert res.status == "error"
    assert "stall" in (res.detail or "").lower()
    assert any(c[-1] == "Escape" for c in fake.sent_keys())
    # Fired after the window, and well before the turn budget ran out.
    assert STATIC_TRUST_WINDOW < clock["now"] < DEFAULT_TIMEOUT


def test_ask_frozen_static_pane_stalls_at_production_constants(tmp_path, clock):
    """The same frozen pane at the EXACT shape every production call site
    uses — chat_session.py calls ask(prompt) with no overrides and
    processor.py passes timeout=DEFAULT_TIMEOUT, both on the service's
    DEFAULT_STALL_TIMEOUT.

    This is the test the first cut of item 13 was missing. That one ran at
    timeout=STATIC_TRUST_WINDOW*2, a configuration no caller ever uses, and
    so it stayed green while the shipped constants (2700 + 900 == 3600 ==
    DEFAULT_TIMEOUT exactly) made the interrupt UNREACHABLE in production:
    old and new code both just sat out the hard timeout, no Escape ever
    sent. Assert the interrupt itself, with real headroom — the window plus
    the stall threshold must land strictly inside the turn budget, so this
    goes red if either constant moves onto the collision again."""
    fake = FakeTmux([READY, THINKING], exists=True)
    s = make_session(tmp_path, fake, clock, stall_timeout=DEFAULT_STALL_TIMEOUT)
    res = s.ask("ping", timeout=DEFAULT_TIMEOUT)

    escape_sent = any(c[-1] == "Escape" for c in fake.sent_keys())
    assert escape_sent is True, "stall interrupt never fired before the hard timeout"
    assert res.status == "error"
    assert "stall" in (res.detail or "").lower()
    # The interrupt, not the hard timeout, is what ended the turn — and with
    # room to spare, not by a photo finish.
    assert clock["now"] < DEFAULT_TIMEOUT
    assert STATIC_TRUST_WINDOW < clock["now"]
    # The headroom is a property of the constants themselves; state it
    # directly so a future edit to either one fails HERE, loudly, instead of
    # silently turning the mechanism back into dead code.
    assert STATIC_TRUST_WINDOW + DEFAULT_STALL_TIMEOUT < DEFAULT_TIMEOUT
    # ...and the anti-B3 floor: still longer than the longest legitimately
    # quiet held turn on record (1440s, item-23 investigation).
    assert STATIC_TRUST_WINDOW > 1440


def test_ask_returns_error_when_ensure_fails(tmp_path, clock):
    """ensure_session failing must surface as AskResult('error'), never an
    exception out of ask() (C2)."""
    fake = FakeTmux([THINKING], exists=False)  # never becomes READY
    s = make_session(tmp_path, fake, clock)
    res = s.ask("ping", timeout=60)
    assert res.status == "error"


def test_ask_leaves_inflight_orphan_on_timeout(tmp_path, clock):
    """A timed-out prompt is still physically in the pane; inflight marker
    must persist as a stuck-signal (M6)."""
    fake = FakeTmux([READY, THINKING], exists=True)
    s = make_session(tmp_path, fake, clock)
    s.ask("ping", timeout=3)
    assert (tmp_path / ".dbrain" / "inflight").exists()


def test_ask_clears_inflight_on_success(tmp_path, clock):
    rid = "abcd0005"
    fake = FakeTmux([READY, _complete(rid)], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    s.ask("ping", timeout=60)
    assert not (tmp_path / ".dbrain" / "inflight").exists()


def test_ask_sends_prompt_via_buffer_then_enter(tmp_path, clock):
    rid = "abcd0006"
    fake = FakeTmux([READY, _complete(rid)], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    s.ask("do the thing", timeout=60)
    assert "paste-buffer" in fake.sent_subcommands()
    assert any(c[-1] == "Enter" for c in fake.sent_keys())


def test_ask_success_records_rid_as_handled(tmp_path, clock):
    """A completed ask() must mark its rid handled so pop_orphan_replies()
    never re-delivers the same reply once it's the pane's latest pair."""
    rid = "abcd0007"
    fake = FakeTmux([READY, _complete(rid)], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    s.ask("ping", timeout=60)
    assert (tmp_path / ".dbrain" / "last_handled_rid").read_text().strip() == rid


# ── pop_orphan_replies ─────────────────────────────────────────────────────


def test_pop_orphan_reply_none_when_pane_idle_with_no_markers(tmp_path, clock):
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.pop_orphan_replies() == []


def test_pop_orphan_reply_returns_self_marked_reply(tmp_path, clock):
    """A background subagent's reply, self-wrapped in a fresh marker pair
    with no ask() ever having asked for it, must be surfaced once."""
    rid = "auto0001"
    fake = FakeTmux([_complete(rid, "Landing page shipped to staging.")], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.pop_orphan_replies() == ["Landing page shipped to staging."]


def test_pop_orphan_reply_does_not_redeliver_same_rid(tmp_path, clock):
    rid = "auto0002"
    fake = FakeTmux([_complete(rid, "done")], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.pop_orphan_replies() == ["done"]
    assert s.pop_orphan_replies() == []  # second tick, same pane: no repeat


def test_pop_orphan_reply_does_not_redeliver_rid_already_consumed_by_ask(
    tmp_path, clock
):
    """The exact race the design has to avoid: watchdog ticks right after a
    normal ask() completes and must NOT re-send that reply as if orphaned."""
    rid = "abcd0008"
    fake = FakeTmux([READY, _complete(rid, "PONG")], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    res = s.ask("ping", timeout=60)
    assert res.reply == "PONG"
    assert s.pop_orphan_replies() == []


def test_pop_orphan_reply_none_while_turn_active(tmp_path, clock):
    """Must never read the pane as if a reply is final while a real ask()
    might still be mid-turn — refuse whenever the pane lock is held."""
    import fcntl
    import os

    rid = "auto0003"
    fake = FakeTmux([_complete(rid, "done")], exists=True)
    s = make_session(tmp_path, fake, clock)

    lock_fd = os.open(tmp_path / ".dbrain" / "pane.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        assert s.pop_orphan_replies() == []
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


# ── R2a: orphan-salvage for an UNCLOSED span (Fable audit) ───────────────


def _open_span_pane(rid: str, body: str) -> str:
    return (
        f"⏺ <<<R:{rid}>>>\n{body}\n"
        "Worked for 2m 22s · 0 background tasks still running\n"
        f"{'─' * 20}\n❯\n{'─' * 20}\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )


def test_pop_orphan_replies_salvages_unclosed_span_stable_across_two_ticks(
    tmp_path, clock
):
    """F1 (Fable audit): before this, the ceiling's log line PROMISED 'rid
    left unhandled for the orphan poller' but pop_orphan_replies() could
    only ever recover COMPLETE R/E pairs — an unclosed span was lost
    forever. Bar: byte-identical extract_open_reply region across two
    consecutive poller ticks."""
    rid = "orphanopen1"
    pane = _open_span_pane(rid, "Unclosed reply text.")
    fake = FakeTmux([pane], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.pop_orphan_replies() == []  # tick 1: baseline only, not delivered
    out = s.pop_orphan_replies()  # tick 2: byte-identical -> salvaged
    assert len(out) == 1
    assert "Unclosed reply text." in out[0]
    assert "⚠️" in out[0]
    handled = (tmp_path / ".dbrain" / "handled_rids").read_text().split()
    assert rid in handled
    assert s.pop_orphan_replies() == []  # tick 3: never redelivered


def test_pop_orphan_replies_does_not_salvage_while_region_still_changing(
    tmp_path, clock
):
    rid = "orphanopen2"
    fake = FakeTmux(
        [_open_span_pane(rid, "growing 1"), _open_span_pane(rid, "growing 2")],
        exists=True,
    )
    s = make_session(tmp_path, fake, clock)
    assert s.pop_orphan_replies() == []  # tick 1: baseline "growing 1"
    assert s.pop_orphan_replies() == []  # tick 2: changed -> update, no deliver
    fake._captures = [_open_span_pane(rid, "growing 2")]  # tick 3: repeats tick 2
    out = s.pop_orphan_replies()
    assert out and "growing 2" in out[0]


def test_pop_orphan_replies_does_not_salvage_while_main_turn_still_active(
    tmp_path, clock
):
    """Gate is `main_turn_finished` — a genuinely live turn's in-progress
    text must never be salvaged mid-generation, however stable it looks."""
    rid = "orphanopen3"
    pane = f"⏺ <<<R:{rid}>>>\nStill generating...\nWarping… (5s · ↓1.2k tokens)\n"
    fake = FakeTmux([pane], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.pop_orphan_replies() == []
    assert s.pop_orphan_replies() == []  # still refuses despite byte-identical
    handled_path = tmp_path / ".dbrain" / "handled_rids"
    handled = handled_path.read_text().split() if handled_path.exists() else []
    assert rid not in handled


def test_pop_orphan_replies_open_span_rescue_does_not_touch_complete_pairs(
    tmp_path, clock
):
    """A complete R/E pair keeps going through the existing path — the new
    rescue loop only ever considers rids NOT already handled by it."""
    rid = "orphancomplete1"
    fake = FakeTmux([_complete(rid, "normal complete reply")], exists=True)
    s = make_session(tmp_path, fake, clock)
    out = s.pop_orphan_replies()
    assert out == ["normal complete reply"]
    assert "⚠️" not in out[0]


# ── health ──────────────────────────────────────────────────────────────


def test_is_healthy_true_when_session_exists(tmp_path, clock):
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.is_healthy() is True


def test_is_healthy_false_when_absent(tmp_path, clock):
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    assert s.is_healthy() is False


def test_kill_sends_kill_session(tmp_path, clock):
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    s.kill()
    assert "kill-session" in fake.sent_subcommands()




# ── optional markers (wrap=False) + idle-based completion ──────────────────


class FakeTmuxText(FakeTmux):
    """FakeTmux that also records text streamed to load-buffer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.texts: list[str] = []

    def __call__(self, args, **kwargs):  # noqa: ANN001
        if kwargs.get("input") is not None:
            self.texts.append(kwargs["input"])
        return super().__call__(args, **kwargs)


def test_ask_wrap_false_does_not_append_marker_instruction(tmp_path, clock):
    fake = FakeTmuxText(
        [READY, THINKING, "ответ\n" + READY, "ответ\n" + READY], exists=True
    )
    (tmp_path / ".dbrain").mkdir()
    (tmp_path / ".dbrain" / "ready").touch()
    s = make_session(tmp_path, fake, clock)
    s.ask("/clear", timeout=30, wrap=False)
    assert fake.texts, "prompt was not streamed to the pane"
    assert "<<<R:" not in fake.texts[0]
    assert "When done" not in fake.texts[0]


def test_ask_wrap_false_completes_on_idle(tmp_path, clock):
    fake = FakeTmuxText(
        [READY, THINKING, THINKING, "ответ модели\n" + READY, "ответ модели\n" + READY],
        exists=True,
    )
    (tmp_path / ".dbrain").mkdir()
    (tmp_path / ".dbrain" / "ready").touch()
    s = make_session(tmp_path, fake, clock)
    res = s.ask("сделай дело", timeout=30, wrap=False)
    assert res.ok, res
    assert "ответ модели" in (res.reply or "")
    assert "bypass permissions" not in (res.reply or "")


def test_ask_wrap_true_still_appends_markers(tmp_path, clock):
    rid = "rid00001"
    fake = FakeTmuxText([READY, _complete(rid)], exists=True)
    (tmp_path / ".dbrain").mkdir()
    (tmp_path / ".dbrain" / "ready").touch()
    s = make_session(tmp_path, fake, clock)
    res = s.ask("ping", timeout=30)
    assert res.ok and res.reply == "PONG"
    assert fake.texts and f"<<<R:{rid}>>>" in fake.texts[0]


# ── steering: inject input into a LIVE turn (no exclusive lock) ─────────────


def test_steer_sends_text_while_lock_is_held(tmp_path, clock):
    """steer() must work WHILE an ask() holds the pane lock — it types into
    the live turn, so it must not take the blocking lock (no deadlock)."""
    import fcntl
    import os

    fake = FakeTmuxText([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    lock_fd = os.open(tmp_path / ".dbrain" / "pane.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)  # simulate an in-flight ask()
    try:
        s.steer("уточнение: пиши короче")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    assert fake.texts and "уточнение" in fake.texts[0]
    assert fake.enter_count() >= 1


def test_interrupt_sends_escape(tmp_path, clock):
    """interrupt() uses the TUI-native Escape (stops the current response);
    C-c is reserved for the stall path (double C-c would begin app exit)."""
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    s.interrupt()
    assert any(c[-1] == "Escape" for c in fake.sent_keys())
    assert not any(c[-1] == "C-c" for c in fake.sent_keys())


def test_is_turn_active_reflects_lock_state(tmp_path, clock):
    import fcntl
    import os

    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.is_turn_active() is False  # lock free → no ask in flight

    lock_fd = os.open(tmp_path / ".dbrain" / "pane.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        assert s.is_turn_active() is True
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


# ── runtime privacy: the pane transcript is owner-only ──────────────────


def test_runtime_dir_is_owner_only(tmp_path, clock):
    fake = FakeTmux([READY], exists=False)
    make_session(tmp_path, fake, clock)
    mode = (tmp_path / ".dbrain").stat().st_mode & 0o777
    assert mode == 0o700


def test_pane_log_precreated_owner_only_before_pipe(tmp_path, clock):
    # pipe-pane appends the FULL Claude transcript via `cat >>` under the
    # tmux server's umask — the file must ALREADY be 0600 when pipe-pane
    # starts, not fixed up afterwards.
    log = tmp_path / ".dbrain" / "pane.log"
    seen = {}

    class Spy(FakeTmux):
        def __call__(self, args, **kwargs):  # noqa: ANN001
            if len(args) > 1 and args[1] == "pipe-pane":
                seen["mode_at_pipe"] = (
                    log.stat().st_mode & 0o777 if log.exists() else None
                )
            return super().__call__(args, **kwargs)

    s = make_session(tmp_path, Spy([READY], exists=False), clock)
    s.ensure_session()
    assert seen["mode_at_pipe"] == 0o600
    assert (log.stat().st_mode & 0o777) == 0o600


# ── steering gate: maintenance turns must not swallow user input ────────


def _hold_pane_lock(s):
    import fcntl

    fh = open(s._pane_lock, "w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def test_is_steerable_turn_distinguishes_chat_from_maintenance(tmp_path, clock):
    s = make_session(tmp_path, FakeTmux([READY]), clock)
    inflight = tmp_path / ".dbrain" / "inflight"

    assert s.is_steerable_turn() is False  # idle: nothing to steer

    fh = _hold_pane_lock(s)
    try:
        # lock held but no inflight → startup/recovery/control, not a turn
        assert s.is_steerable_turn() is False
        inflight.write_text("rid12345\n0.0\n")
        assert s.is_steerable_turn() is True  # a chat turn
        inflight.write_text("maint-daily\n0.0\n")
        assert s.is_steerable_turn() is False  # the nightly pipeline
    finally:
        fh.close()


def test_inflight_claimed_before_session_startup(tmp_path, clock):
    # A stale inflight left by a timed-out chat turn must not misrepresent
    # the new holder to the steering gate while session startup (up to
    # startup_timeout) is still running.
    inflight = tmp_path / ".dbrain" / "inflight"
    seen = {}

    class Spy(FakeTmux):
        def __call__(self, args, **kwargs):  # noqa: ANN001
            if len(args) > 1 and args[1] == "new-session":
                seen["at_startup"] = (
                    inflight.read_text() if inflight.exists() else None
                )
            return super().__call__(args, **kwargs)

    fake = Spy([READY, READY, _complete("rid00001")], exists=False)
    s = make_session(tmp_path, fake, clock)
    inflight.write_text("stale-chat-rid\n0.0\n")  # leftover from a timeout

    s.ask("nightly run", request_id="maint-process")

    assert seen["at_startup"] is not None
    assert seen["at_startup"].startswith("maint-process")


def test_ask_dismisses_feedback_survey_instead_of_stalling(tmp_path, clock):
    # Claude Code periodically shows "How is Claude doing this session?" —
    # it polluted the chrome, the stall detector fired Escape and the user
    # got a session error. The session must press 0 (Dismiss) and carry on.
    survey = (
        "● How is Claude doing this session? (optional)\n"
        "  1: Bad    2: Fine   3: Good   0: Dismiss\n" + READY
    )
    fake = FakeTmux(
        [READY, survey, THINKING, _complete("rid00001")], exists=True
    )
    s = make_session(tmp_path, fake, clock)
    res = s.ask("привет")
    assert res.status == "ok"
    pressed = [c for c in fake.sent_keys() if c[-1] == "0"]
    assert pressed, "survey was not dismissed with 0"
    escapes = [c for c in fake.sent_keys() if c[-1] == "Escape"]
    assert not escapes, "stall interrupt fired instead of dismissing survey"


def test_growing_pane_log_prevents_false_stall(tmp_path, clock):
    # Version-proof liveness: even if a future spinner format is
    # unrecognized, a transcript that keeps growing means the brain is
    # working — it must not be killed as a stall.
    log = tmp_path / ".dbrain" / "pane.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("")
    busy = "thinking, but with a spinner we do not recognize\n"
    fake = FakeTmux([READY] + [busy] * 20 + [_complete("rid00001")], exists=True)
    s = make_session(tmp_path, fake, clock)  # stall_timeout=10, poll=1

    def grow_sleep(seconds: float) -> None:
        clock["now"] += seconds
        with log.open("a") as f:
            f.write("x" * 100)  # transcript advancing → alive

    s._sleep = grow_sleep
    res = s.ask("long task")
    assert res.status == "ok"
    assert not any(c[-1] in ("Escape", "C-c") for c in fake.sent_keys())


def test_static_pane_log_and_no_spinner_still_stalls(tmp_path, clock):
    # The fallback must not disable stall detection: a genuinely wedged
    # turn (no spinner, transcript frozen) still gets interrupted.
    log = tmp_path / ".dbrain" / "pane.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("frozen")
    busy = "no spinner, no markers, nothing changing\n"
    fake = FakeTmux([READY] + [busy] * 40, exists=True)
    s = make_session(tmp_path, fake, clock)
    res = s.ask("x", timeout=600)
    assert res.status == "error"
    assert "stall" in (res.detail or "").lower()


# ── 2026-08-20 reliability fixes ─────────────────────────────────────────


def test_completed_reply_wins_over_a_limit_looking_pane(tmp_path, clock):
    """Ordering fix: our OWN finished answer outranks any fault signature.

    With the checks the other way round, a reply that merely mentioned a
    limit made ask() return "rate_limited" and throw the finished answer
    away — the user got "⏳ Лимит подписки исчерпан" and then the real
    answer seconds later from the orphan poller (the double-message bug).
    """
    rid = "dual0001"
    pane = (
        f"<<<R:{rid}>>>\nПро лимиты: usage limit reached бывает ложным.\n"
        f"<<<E:{rid}>>>\n"
        "  Claude usage limit reached. Your limit will reset at 3pm (UTC).\n❯\n"
    )
    fake = FakeTmux([READY, pane], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    res = s.ask("ping", timeout=60)
    assert res.status == "ok"
    assert "ложным" in (res.reply or "")


def test_ask_records_rid_in_the_handled_set(tmp_path, clock):
    rid = "abcd0009"
    fake = FakeTmux([READY, _complete(rid)], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    s.ask("ping", timeout=60)
    handled = (tmp_path / ".dbrain" / "handled_rids").read_text().split()
    assert rid in handled


def test_pop_orphan_replies_delivers_a_superseded_orphan(tmp_path, clock):
    """The loss case: a subagent's reply is followed by a newer turn's pair
    before the poller runs. Latest-only lookup dropped the older one."""
    pane = (
        "<<<R:sub00001>>>\nSubagent result nobody delivered.\n<<<E:sub00001>>>\n"
        "<<<R:live0002>>>\nLive turn answer.\n<<<E:live0002>>>\n❯\n"
    )
    fake = FakeTmux([pane], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.pop_orphan_replies() == [
        "Subagent result nobody delivered.",
        "Live turn answer.",
    ]
    assert s.pop_orphan_replies() == []  # both consumed


def test_repainted_older_pair_is_never_redelivered(tmp_path, clock):
    """The duplication case: the TUI repaints its transcript, so an
    already-delivered pair can become the LAST one on the pane. A single
    "last rid" could not tell that apart from a fresh reply."""
    first = "<<<R:aaa00001>>>\nПервый ответ.\n<<<E:aaa00001>>>\n❯\n"
    second = first + "<<<R:bbb00002>>>\nВторой ответ.\n<<<E:bbb00002>>>\n❯\n"
    repainted = second + "<<<R:aaa00001>>>\nПервый ответ.\n<<<E:aaa00001>>>\n❯\n"
    fake = FakeTmux([first], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.pop_orphan_replies() == ["Первый ответ."]
    fake._captures = [second]
    assert s.pop_orphan_replies() == ["Второй ответ."]
    fake._captures = [repainted]
    assert s.pop_orphan_replies() == []


def test_upgrade_does_not_flush_existing_scrollback(tmp_path, clock):
    """Migration guard: an install that predates the handled-rid set has a
    pane full of replies already delivered the old way. The first tick must
    adopt them silently, not dump the scrollback into the chat."""
    runtime = tmp_path / ".dbrain"
    runtime.mkdir(parents=True)
    (runtime / "last_handled_rid").write_text("old00002")
    pane = (
        "<<<R:old00001>>>\nОтвет вчерашнего дня.\n<<<E:old00001>>>\n"
        "<<<R:old00002>>>\nПоследний доставленный.\n<<<E:old00002>>>\n❯\n"
    )
    fake = FakeTmux([pane], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.pop_orphan_replies() == []
    assert s.pop_orphan_replies() == []


def test_fresh_runtime_dir_still_delivers(tmp_path, clock):
    """Seeding is keyed on the OLD file existing — a genuinely fresh runtime
    dir has nothing delivered yet, so a pending reply must still go out."""
    fake = FakeTmux([_complete("new00001", "Готово.")], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.pop_orphan_replies() == ["Готово."]


def test_orphan_superseded_by_a_completing_ask_is_not_lost(tmp_path, clock):
    """The exact loss found in review 2026-08-20: an orphan reply sits on
    the pane, unhandled; a LATER ask() turn finishes and marks its OWN
    (newer) rid handled directly (never going through pop_orphan_replies).
    find_pending_replies' watermark then treats anything above that newest
    handled pair as "already delivered, however it got there" — true for a
    repainted pair, false here. Without queuing the older pair explicitly,
    it falls below the watermark and is never delivered again."""
    rid = "live0002"
    pane_with_both = (
        "<<<R:sub00001>>>\nSubagent result nobody delivered.\n<<<E:sub00001>>>\n"
        "<<<R:live0002>>>\nLive turn answer.\n<<<E:live0002>>>\n❯\n"
    )
    fake = FakeTmux([READY, pane_with_both], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    res = s.ask("ping", timeout=60)
    assert res.status == "ok"
    assert res.reply == "Live turn answer."
    # The orphan below the watermark must still be recoverable...
    assert s.pop_orphan_replies() == ["Subagent result nobody delivered."]
    # ...exactly once.
    assert s.pop_orphan_replies() == []


def test_pop_orphan_replies_skips_delivery_when_the_write_fails(
    tmp_path, clock, monkeypatch
):
    """The mark-before-deliver invariant: a rid must never be handed out
    before it is durably recorded, or a crash right after (e.g.
    delivery_guard's SIGKILL restart) reopens the redelivery hole this store
    exists to close. Must NOT silently deliver on a failed write."""
    rid = "auto0009"
    fake = FakeTmux([_complete(rid, "should not be delivered")], exists=True)
    s = make_session(tmp_path, fake, clock)
    monkeypatch.setattr(s, "_atomic_write", lambda *a, **k: False)
    assert s.pop_orphan_replies() == []


def test_pending_orphan_no_longer_extractable_is_logged_and_dropped(
    tmp_path, clock, caplog
):
    """Round-2 finding: a queued pending-orphan rid whose marker pair has
    scrolled out of the capture window (extract_reply returns None) used to
    be silently `continue`d — no log line, and never cleared from
    pending_orphans, so it was retried every tick forever with nothing to
    show for the loss. Must be logged (loudly, once) and removed from the
    queue instead of retried silently forever."""
    runtime = tmp_path / ".dbrain"
    runtime.mkdir(parents=True)
    (runtime / "pending_orphans").write_text("gone0001\n")
    fake = FakeTmux([READY], exists=True)  # rid's marker pair is nowhere on screen
    s = make_session(tmp_path, fake, clock)
    with caplog.at_level("ERROR"):
        assert s.pop_orphan_replies() == []
    assert "gone0001" in caplog.text
    # Must not be retried forever: gone from the queue after this tick.
    assert (runtime / "pending_orphans").read_text().split() == []


def test_migration_via_ask_first_does_not_flush_scrollback(tmp_path, clock):
    """The precise bug found re-driving the consolidated plan (2026-08-20):
    the old _seed_handled() ran ONLY inside pop_orphan_replies(). On an
    upgraded install the FIRST turn to complete after the upgrade can just as
    easily be ask()'s own completion path — and ask() treated an empty
    handled_rids as "nothing delivered yet", unconditionally queuing the
    WHOLE pre-existing scrollback into pending_orphans, which
    pop_orphan_replies() then handed out on its very next tick regardless of
    the watermark. _ensure_migrated() must run at the START of ask()'s
    completion path too, not only inside pop_orphan_replies()."""
    runtime = tmp_path / ".dbrain"
    runtime.mkdir(parents=True)
    (runtime / "last_handled_rid").write_text("old00002")
    rid = "new00003"
    pane = (
        "<<<R:old00001>>>\nОтвет вчерашнего дня.\n<<<E:old00001>>>\n"
        "<<<R:old00002>>>\nПоследний доставленный по старой схеме.\n"
        "<<<E:old00002>>>\n"
        f"<<<R:{rid}>>>\nСвежий ответ.\n<<<E:{rid}>>>\n❯\n"
    )
    fake = FakeTmux([READY, pane], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    res = s.ask("ping", timeout=60)
    assert res.status == "ok"
    assert res.reply == "Свежий ответ."
    # The two pre-existing pairs must NOT come back as orphans.
    assert s.pop_orphan_replies() == []


def test_migration_reseeds_when_handled_rids_is_stale(tmp_path, clock):
    """Round-2 blocker, reproduced against the real deploy target
    (2026-08-20): a stray ``handled_rids`` left behind by an aborted prior
    deploy can exist on disk while missing the rid ``last_handled_rid`` has
    since moved on to. The old trigger (``handled_rids.exists()``) treated
    ANY existing file as "already migrated, forever" and never looked again
    — so every one of the (already-delivered) pairs still on the pane would
    have been resent. Migration must re-seed instead of permanently
    no-op'ing when the two disagree."""
    runtime = tmp_path / ".dbrain"
    runtime.mkdir(parents=True)
    # Stale store from a botched deploy: none of these rids are on the pane
    # below, and it does not know about the rid last_handled_rid mirrors.
    (runtime / "handled_rids").write_text("manualfix1\nstale0001\n")
    (runtime / "last_handled_rid").write_text("aaaa1111")
    pane = (
        "<<<R:aaaa1111>>>\nОтвет первый.\n<<<E:aaaa1111>>>\n"
        "<<<R:bbbb2222>>>\nОтвет второй.\n<<<E:bbbb2222>>>\n"
        "<<<R:cccc3333>>>\nОтвет третий.\n<<<E:cccc3333>>>\n❯\n"
    )
    fake = FakeTmux([pane], exists=True)
    s = make_session(tmp_path, fake, clock)
    # First tick re-seeds instead of delivering — same contract as a fresh
    # migration.
    assert s.pop_orphan_replies() == []
    # Re-seeding only ADDS rids: the pre-existing (unrelated) entries survive
    # alongside everything now visible on the pane.
    handled = (runtime / "handled_rids").read_text().split()
    assert set(handled) == {
        "manualfix1",
        "stale0001",
        "aaaa1111",
        "bbbb2222",
        "cccc3333",
    }
    # Idempotent: now that handled_rids contains the legacy rid, a second
    # tick must not re-trigger migration or resend anything.
    assert s.pop_orphan_replies() == []


def test_pending_orphan_queue_respects_watermark(tmp_path, clock):
    """Round-2 finding: pending_orphans used to be filled from
    find_unhandled_replies(), which has no watermark rule at all — any pair
    that predates the newest already-handled pair on screen (e.g. resurfaced
    by a pane reflow) got queued and then handed out by pop_orphan_replies()
    unconditionally, resending an already-delivered reply. The queuing must
    use the same watermark-aware find_pending_replies() the rest of the
    delivery path relies on."""
    watermark_rid = "wmk00001"
    fake = FakeTmux([READY, _complete(watermark_rid, "Watermark reply.")], exists=True)
    s = make_session(tmp_path, fake, clock, rid=watermark_rid)
    res = s.ask("ping", timeout=60)
    assert res.status == "ok"

    # A later turn completes; the pane now also shows an OLDER pair, ABOVE
    # the watermark pair, that was never marked handled (a reflow-resurfaced,
    # already-delivered reply — not a genuine orphan).
    new_rid = "new00002"
    pane2 = (
        "<<<R:anc00001>>>\nAncient reply nobody marked.\n<<<E:anc00001>>>\n"
        f"<<<R:{watermark_rid}>>>\nWatermark reply.\n<<<E:{watermark_rid}>>>\n"
        f"<<<R:{new_rid}>>>\nFresh reply.\n<<<E:{new_rid}>>>\n❯\n"
    )
    fake._captures = [READY, pane2]
    s2 = make_session(tmp_path, fake, clock, rid=new_rid)
    res2 = s2.ask("ping again", timeout=60)
    assert res2.status == "ok"
    assert res2.reply == "Fresh reply."

    # The below-watermark pair must NOT have been queued as a pending orphan.
    assert s2.pop_orphan_replies() == []


def test_concurrent_mark_handled_does_not_lose_a_write(tmp_path, clock):
    """_mark_handled's read-modify-write must be safe against a genuinely
    concurrent writer — the bot's ask() and the watchdog's
    pop_orphan_replies() call it from SEPARATE processes in production. A
    lost update here means a rid falls out of the set and its reply is
    redelivered on the next tick (review 2026-08-20)."""
    import threading
    import time as real_time

    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)

    orig_read = s._read_handled

    def slow_read():
        result = orig_read()
        real_time.sleep(0.05)  # widen the read→write window
        return result

    s._read_handled = slow_read

    results: dict[str, bool] = {}

    def mark(rid: str) -> None:
        results[rid] = s._mark_handled(rid)

    threads = [
        threading.Thread(target=mark, args=("thread0001",)),
        threading.Thread(target=mark, args=("thread0002",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert results == {"thread0001": True, "thread0002": True}
    handled = (tmp_path / ".dbrain" / "handled_rids").read_text().split()
    assert "thread0001" in handled
    assert "thread0002" in handled


# ── busy-pane guard (never type over a leftover live turn) ──────────────


def test_ask_waits_out_a_stale_leftover_turn_before_sending(tmp_path, clock):
    """A previous ask() can release the pane lock on a stall while its own
    turn keeps running (2026-08-20 fix). is_turn_active() is lock-based, so
    a new ask() acquiring the free lock must not type over that still-live
    turn — it must wait for the pane to clear first."""
    rid = "wait0001"
    fake = FakeTmux([THINKING, THINKING, READY, _complete(rid)], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    res = s.ask("ping", timeout=60)
    assert res.status == "ok"
    assert res.reply == "PONG"


def test_ask_gives_up_if_the_leftover_turn_never_clears(tmp_path, clock):
    """Busy-panel UX finding (2026-08-22): a legitimately busy panel is not
    an error — status is the distinct 'busy', not 'error', so chat_session.py
    can show an honest message instead of '❌ Ошибка сессии'.

    Status is plain 'busy', NOT 'busy_active' (blind-review F1 fix,
    2026-09): the THINKING fixture holds the legacy "esc to interrupt" hint
    byte-identical for the whole wait — a REAL wedge (dead subagent, D-state
    process) can show this exact static, unchanging chrome forever too, so
    it must still count as a delivery failure. The busy/busy_active split
    tracks GENUINE, RECENT chrome/log change only — see
    _RECENT_PROGRESS_POLLS' docstring in claude_session.py. An earlier
    version of this fix mistakenly reused is_working_progressing() (which
    DOES exempt this hint, for the different purpose of not killing a
    silently-alive turn via the stall-timeout break) for the classification
    too, which is exactly what this test guards against regressing to."""
    fake = FakeTmux([THINKING], exists=True)
    s = make_session(tmp_path, fake, clock)
    res = s.ask("ping", timeout=600)
    assert res.status == "busy"
    assert "busy" in (res.detail or "").lower()
    assert "paste-buffer" not in fake.sent_subcommands()


def test_busy_wait_and_stall_share_one_overall_deadline(tmp_path, clock):
    """The busy-wait for a leftover turn must draw down the SAME budget as
    the rest of the call, not a separate stall_timeout budget stacked on top
    of timeout — that used to let the worst case reach
    stall_timeout + timeout instead of the documented ceiling (review
    2026-08-20). The static "esc to interrupt" hint never lapses the
    STALL-TIMEOUT break on its own (a real silent turn holds it static for
    its whole duration), so this fixture only terminates via the OVERALL
    deadline — the wall-clock spent must never exceed `timeout`.

    Status stays plain 'busy' (blind-review F1 fix, 2026-09): see
    test_ask_gives_up_if_the_leftover_turn_never_clears above — a static,
    byte-identical hint is not evidence of a LIVE pane for classification
    purposes, only of "don't abandon it via the stall break"."""
    fake = FakeTmux([THINKING], exists=True)  # never clears
    s = make_session(tmp_path, fake, clock)  # stall_timeout=10
    timeout = 20
    res = s.ask("ping", timeout=timeout)
    assert res.status == "busy"
    assert "busy" in (res.detail or "").lower()
    assert clock["now"] <= timeout


def test_ask_not_steerable_while_waiting_out_a_leftover_turn(tmp_path, clock):
    """During the busy-wait for a leftover turn, `inflight` must stay
    maint-prefixed (not the real request id): writing the real log_id before
    the pane is confirmed free used to make is_steerable_turn() return True
    for the OTHER, still-running turn — an incoming message arriving during
    the wait would be steered straight into it, interleaving both turns'
    text in one pane (review 2026-08-20)."""
    rid = "wait0002"
    fake = FakeTmux([THINKING, THINKING, READY, _complete(rid)], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    seen: list[str] = []
    prompt_sent = {"done": False}

    orig_sleep = s._sleep

    def spy_sleep(seconds: float) -> None:
        if not prompt_sent["done"]:
            inflight = tmp_path / ".dbrain" / "inflight"
            if inflight.exists():
                seen.append(inflight.read_text().splitlines()[0])
        orig_sleep(seconds)

    orig_send_prompt = s._send_prompt

    def spy_send_prompt(*a, **k):
        prompt_sent["done"] = True
        return orig_send_prompt(*a, **k)

    s._sleep = spy_sleep
    s._send_prompt = spy_send_prompt
    res = s.ask("ping", timeout=60)
    assert res.status == "ok"
    assert seen, "expected at least one sleep during the busy-wait"
    assert all(line.startswith(MAINT_PREFIX) for line in seen)


def test_busy_wait_has_own_budget_despite_leftover_turn_progressing(tmp_path, clock):
    """Round-2 finding: the busy-wait used to reuse the full stall_timeout
    (900s in production) as its own budget, so a leftover turn that kept
    emitting fresh progress frames (never tripping the "no progress for
    stall_timeout" break) could hold BOTH pane.lock and the process-wide
    ask-lock for the entire remaining call deadline — up to an hour — while
    chat.py turned incoming messages away with a false "couple of minutes"
    promise and the watchdog's non-blocking force_recover() could not get
    in either way. The busy-wait must give up on its own, dedicated
    (smaller) budget regardless of how much progress the OTHER turn keeps
    showing.

    Status updated to 'busy_active' (agent-infra-backlog item 22, 2026-09):
    this fixture's frames DO show genuine, changing progress every poll —
    exactly the case the new progress-aware classification distinguishes
    from a frozen pane. The budget-discipline assertion below (own budget,
    not the full stall_timeout) is unaffected by the status rename."""

    def sleep_fn(seconds: float) -> None:
        clock["now"] += seconds

    # Paren-anchored main-turn spinner (is_main_turn_active), not a bare
    # background-agent-row shape (which is_main_turn_active deliberately
    # excludes — see backlog item 10, 2026-08-21): this test is about the
    # busy-wait's OWN budget for a leftover turn that is genuinely still the
    # MAIN turn, not about background-agent rows.
    frames = [f"Warping… ({i}s · ↓{i}k tokens)\n" for i in range(1, 200)]
    fake = FakeTmux(frames, exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=sleep_fn,
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: "budget01",
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=5000.0,  # never trips: progress never lapses
        busy_wait_budget=50.0,
    )
    res = s.ask("ping", timeout=5000)
    assert res.status == "busy_active"
    assert "busy" in (res.detail or "").lower()
    # Gave up close to its own 50s budget, nowhere near the 5000s call
    # deadline or the 5000s stall_timeout it must not have fallen back to.
    assert clock["now"] <= 60


# ── busy vs. busy_active classification (agent-infra-backlog item 22) ────


def test_busy_with_progressing_pane_returns_busy_active(tmp_path, clock):
    """Core Step A fix: a leftover turn whose progress signature genuinely
    CHANGES frame-to-frame is a live, working turn — not evidence of a
    delivery failure. Status must be the neutral 'busy_active', not plain
    'busy'."""

    def sleep_fn(seconds: float) -> None:
        clock["now"] += seconds

    frames = [f"Warping… ({i}s · ↓{i}k tokens)\n" for i in range(1, 100)]
    fake = FakeTmux(frames, exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=sleep_fn,
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: "prog0001",
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=5000.0,
        busy_wait_budget=30.0,
    )
    res = s.ask("ping", timeout=5000)
    assert res.status == "busy_active"
    assert res.busy_seconds is not None
    assert res.busy_seconds > 0


def test_busy_with_frozen_pane_still_returns_busy(tmp_path, clock):
    """B3 regression guard: a pane that shows NO real change across the
    whole busy-wait must still return plain 'busy', detail unchanged from
    before this plan. Uses a background-agent-wait frame held
    byte-identical.

    Note on the assertion below: with make_session's stall_timeout=10, the
    loop actually exits via the STALL-TIMEOUT break (~11s in), not by
    riding out the full busy_wait_budget (300s, default) — that break is
    unaffected by this plan (it still legitimately treats the frozen
    background-agent-wait shape as "no liveness signal at all", the case it
    was built for). The upper-bound assertion is still a valid (looser)
    check either way."""
    frozen = (
        "✻ Waiting for 1 background agent to finish\n"
        "  agent  3m 5s · ↓ 88.3k tokens\n❯\n"
    )
    fake = FakeTmux([frozen], exists=True)
    s = make_session(tmp_path, fake, clock)  # busy_wait_budget default (300)
    res = s.ask("ping", timeout=600)
    assert res.status == "busy"
    assert res.detail == "pane still busy with a previous turn"
    assert res.busy_seconds is not None
    assert clock["now"] <= DEFAULT_BUSY_WAIT_BUDGET + 5


def test_static_esc_to_interrupt_hint_frozen_for_whole_wait_returns_busy(
    tmp_path, clock
):
    """THE critical F1 regression test (blind-review round, 2026-09) — make
    sure this would actually FAIL against the broken implementation before
    trusting it: a naive classifier that reuses is_working_progressing()
    (which exempts the static "esc to interrupt" hint from the
    change-required check, for the SEPARATE purpose of not killing a
    silently-alive turn via the stall-timeout break) sets saw_progress=True
    on the very FIRST poll here and never gets a chance to reconsider —
    reporting 'busy_active' for a pane that is, from this test's point of
    view, indistinguishable from a genuinely wedged one (dead subagent,
    D-state process) that happens to still show this exact static chrome.

    The fixed classifier tracks a REAL chrome change (or pane.log growth)
    with a RECENCY requirement, completely independent of
    is_working_progressing() — so a byte-identical hint, held for the whole
    wait, must never count as "this pane is alive" for classification
    purposes, even though the SAME hint correctly keeps the stall-timeout
    break from firing (see busy_last_active's own tracking, unaffected by
    this test)."""
    fake = FakeTmux([THINKING], exists=True)  # never changes, never clears
    s = make_session(tmp_path, fake, clock)  # stall_timeout=10
    res = s.ask("ping", timeout=600)
    assert res.status == "busy"
    assert res.detail == "pane still busy with a previous turn"


def test_progress_then_freeze_does_not_latch_busy_active(tmp_path, clock):
    """F1(b) regression: a few genuine early changes must NOT be treated as
    permanent evidence of liveness for the rest of a busy-wait that then
    goes solid. A pane that ticks a handful of times and then wedges is
    still a wedge — the classification must require RECENT progress at
    give-up time, not a one-shot latch that survives the rest of the wait.
    stall_timeout is set well above busy_wait_budget so the stall break
    cannot rescue this within the busy budget (matching the live incident
    shape), and the fixture freezes after a few real changes."""

    def sleep_fn(seconds: float) -> None:
        clock["now"] += seconds

    changing = [f"Warping… ({i}s · ↓{i}k tokens)\n" for i in range(1, 4)]
    frozen = changing[-1]
    frames = changing + [frozen] * 400  # changes a few times, then wedges
    fake = FakeTmux(frames, exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=sleep_fn,
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: "latch0001",
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=900.0,  # far above busy_wait_budget — can't rescue this
        busy_wait_budget=300.0,
    )
    res = s.ask("ping", timeout=5000)
    assert res.status == "busy"
    assert res.detail == "pane still busy with a previous turn"


def test_fresh_long_run_marker_short_circuits_the_busy_wait(tmp_path, clock):
    """B3 fast path: a FRESH long-run.json marker plus a genuinely
    progressing pane lets ask() return busy_active in a handful of polls,
    well under the full busy-wait budget — instead of riding it out to
    ~300s like a naive implementation (pre-fix / stale-marker) would."""
    from d_brain.services import long_run

    def sleep_fn(seconds: float) -> None:
        clock["now"] += seconds

    runtime_dir = tmp_path / ".dbrain"
    frames = [f"Warping… ({i}s · ↓{i}k tokens)\n" for i in range(1, 50)]
    fake = FakeTmux(frames, exists=True)
    # Fresh marker: written "now" in wall-clock terms, well inside
    # long_run_stale_after.
    long_run.write(
        runtime_dir, long_run.LongRun(since=time.time() - 200.0, updated_ts=time.time())
    )
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=runtime_dir,
        runner=fake,
        sleep_fn=sleep_fn,
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: "fast0001",
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=5000.0,
        busy_wait_budget=300.0,
    )
    res = s.ask("ping", timeout=5000)
    assert res.status == "busy_active"
    # Short-circuited: nowhere near the 300s busy_wait_budget.
    assert clock["now"] < 30


def test_stale_long_run_marker_falls_back_to_full_busy_wait(tmp_path, clock):
    """A STALE long-run.json (watchdog writer presumed dead) must degrade to
    exactly today's behavior — the fast path never trusts it and is never
    even entered, so a genuinely frozen pane is governed purely by the
    normal busy-wait/stall discipline (unaffected by this plan) and ends
    'busy', not the fast-path 'busy_active'."""
    from d_brain.services import long_run

    runtime_dir = tmp_path / ".dbrain"
    frozen = (
        "✻ Waiting for 1 background agent to finish\n"
        "  agent  3m 5s · ↓ 88.3k tokens\n❯\n"
    )
    fake = FakeTmux([frozen], exists=True)
    # Stale marker: updated_ts far in the past relative to "now".
    long_run.write(
        runtime_dir,
        long_run.LongRun(
            since=time.time() - 10_000.0, updated_ts=time.time() - 9_000.0
        ),
    )
    s = make_session(tmp_path, fake, clock, tmux_config=None)
    # make_session parents runtime_dir at tmp_path/.dbrain already — reuse it.
    res = s.ask("ping", timeout=600)
    assert res.status == "busy"
    assert clock["now"] <= DEFAULT_BUSY_WAIT_BUDGET + 5


# ── stall detector vs. a frozen "working" frame ──────────────────────────


def test_frozen_background_agent_frame_still_stalls(tmp_path, clock):
    """Proof of the review-2026-08-20 finding: the background-agent-wait /
    bare-elapsed-timer signatures matched a STATIC frame forever (a dead
    subagent's last rendered frame), which reported is_working()==True on
    every poll and disarmed the stall detector entirely. Must still stall."""
    frozen = (
        "✻ Waiting for 1 background agent to finish\n"
        "  agent  3m 5s · ↓ 88.3k tokens\n❯\n"
    )
    fake = FakeTmux([READY] + [frozen] * 40, exists=True)
    s = make_session(tmp_path, fake, clock)  # stall_timeout=10, poll=1
    res = s.ask("x", timeout=600)
    assert res.status == "error"
    assert "stall" in (res.detail or "").lower()


def test_is_working_requires_change_for_the_progress_signature(tmp_path, clock):
    frozen = (
        "✻ Waiting for 1 background agent to finish\n"
        "  agent  3m 5s · ↓ 88.3k tokens\n❯\n"
    )
    fake = FakeTmux([frozen], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.is_working() is True  # first sighting: no baseline yet
    assert s.is_working() is False  # unchanged since the previous call
    assert s.is_working() is False


def test_is_working_legacy_hint_does_not_need_change(tmp_path, clock):
    """The legacy "esc to interrupt" hint is a whole-turn marker, not a
    per-tick counter — a real silent task holds it static for its entire
    duration, so it must keep counting as working every call."""
    fake = FakeTmux([THINKING], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.is_working() is True
    assert s.is_working() is True


# ── nudge (subscription-limit wake-up) ───────────────────────────────────


def test_nudge_types_into_a_parked_session(tmp_path, clock):
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.nudge("Continue") is True
    assert "paste-buffer" in fake.sent_subcommands()
    assert fake.enter_count() == 1


def test_nudge_marks_inflight_maintenance_before_typing(tmp_path, clock):
    """A concurrent user message arriving right after nudge() releases the
    lock must be able to see that a maintenance turn was just started here
    (2026-08-20 review) rather than pasting a second prompt over it."""
    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    assert s.nudge("Continue") is True
    inflight = (tmp_path / ".dbrain" / "inflight").read_text()
    assert inflight.startswith(MAINT_PREFIX)


def test_nudge_refuses_while_a_turn_holds_the_lock(tmp_path, clock):
    """Never type into a live turn: the wake-up is for a parked session."""
    import fcntl
    import os

    fake = FakeTmux([READY], exists=True)
    s = make_session(tmp_path, fake, clock)
    lock_fd = os.open(tmp_path / ".dbrain" / "pane.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        assert s.nudge("Continue") is False
        assert "paste-buffer" not in fake.sent_subcommands()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


# ── pane geometry ────────────────────────────────────────────────────────


def test_existing_session_is_resized_when_it_drifted(tmp_path, clock):
    """Measured live 2026-08-20: the brain pane had drifted to 80x23 (an
    attached 80-column client shrinks the window and the size sticks), which
    both hard-wrapped replies mid-sentence and made the chrome window cover
    the model's text."""
    fake = FakeTmux([READY], exists=True, window_size="80x23")
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    assert "resize-window" in fake.sent_subcommands()
    resize = next(c for c in fake.calls if len(c) > 1 and c[1] == "resize-window")
    assert "200" in resize and "50" in resize


def test_correctly_sized_session_is_left_alone(tmp_path, clock):
    fake = FakeTmux([READY], exists=True, window_size="200x50")
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    assert "resize-window" not in fake.sent_subcommands()


# ── backlog item 10 (2026-08-21): reply lost when the closing ────────────
# <<<E:id>>> marker never appears — salvage / no-main-turn ceiling.
#
# The "incident pane": a line-anchored <<<R:rid>>> marker, a multi-line
# reply body, NO closing E line, a "Worked for … · N background tasks still
# running" summary, a blank line, a box rule, a bare idle prompt line,
# another box rule, the bypass-permissions footer, and a background-agent
# list row whose elapsed counter INCREMENTS from poll to poll — reproducing
# the real live frame shape that hung for 22 minutes with zero log output
# before this fix. Verified (2026-08-21, throwaway script against `main`'s
# claude_session.py/tmux_parse.py, not shipped as a test): this exact
# fixture makes unpatched ask() return status="timeout" after the FULL
# DEFAULT_TIMEOUT (3600s) — reproducing the incident precisely.

# Production forms captured live 2026-08-22 (backlog item 12 plan §2.4):
# the box rule is 200 columns wide (pane width), the idle input line is
# "❯ ", and the footer — with "esc to interrupt" INSIDE it, the mandatory
# form for catching defect C2 (a footer-only hint must no longer pin
# is_main_turn_active()) — carries whatever is interruptible (background
# tasks, monitors), not just the main turn.
_INCIDENT_BOX = "─" * 200
_INCIDENT_FOOTER = (
    "  ⏵⏵ bypass permissions on · 2 background tasks · esc to interrupt · "
    "← for agents · ↓ to manage\n"
)

# The TUI's turn-summary verb is RANDOMIZED — measured 2026-08-22 over 258
# real summary lines in ~/.dbrain/pane.log (see tmux_parse._TURN_SUMMARY_RE
# docstring). Every fixture below that used to hardcode "Worked for" is
# parametrized over all seven measured verbs so a test suite never again
# idealizes the one verb that happened to be sampled.
_SUMMARY_VERBS = (
    "Worked",
    "Brewed",
    "Baked",
    "Cooked",
    "Crunched",
    "Churned",
    "Cogitated",
)


def _incident_frame(rid: str, counter: int, verb: str = "Worked") -> str:
    return (
        f"⏺ <<<R:{rid}>>>\n"
        "This is the real answer text that never got its closing marker.\n"
        f"{verb} for 2m 22s · 1 background task still running\n"
        "\n"
        f"{_INCIDENT_BOX}\n"
        "❯ \n"
        f"{_INCIDENT_BOX}\n"
        "  ● main\n"
        f"  ◯ general-purpose  still going   {counter}s · ↓ {counter}k tokens\n"
        f"{_INCIDENT_FOOTER}"
    )


# ── F5 (Fable audit): the "main turn finished ... 0.0s after send" latch ──


def test_f5_first_poll_does_not_falsely_log_main_turn_finished_at_0s(
    tmp_path, clock, caplog
):
    """F5: the very first poll after send can still be looking at the
    PREVIOUS (already-finished) turn's UNCHANGED frame — must not log 'main
    turn finished ... 0.0s after send' until EITHER the main spinner has
    been observed active at least once, or the pane has visibly changed
    since send."""
    rid = "f5test01"
    stale_finished_frame = (
        "Worked for 1m 0s · 0 background tasks still running\n"
        f"{_INCIDENT_BOX}\n❯\n{_INCIDENT_BOX}\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    # pre_cap and the FIRST loop capture are BYTE-IDENTICAL (the pane has
    # not visibly redrawn yet on the very first poll); the pane genuinely
    # changes on the next poll, then completes.
    frames = [
        stale_finished_frame,  # pre_cap
        stale_finished_frame,  # first loop poll: STILL the stale frame
        "Warping… (2s · ↓1k tokens)\n",  # second poll: main turn now active
        _complete(rid, "real answer"),
    ]
    fake = FakeTmux(frames, exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    with caplog.at_level(logging.INFO):
        res = s.ask("ping", timeout=60)
    assert res.status == "ok"
    assert not any(
        "main turn finished" in r.message and "0.0s after send" in r.message
        for r in caplog.records
    )


def test_f5_main_turn_finished_logs_once_pane_has_visibly_changed(
    tmp_path, clock, caplog
):
    """Positive case: once the pane HAS visibly changed since send (a
    genuinely new, finished-looking frame — not a leftover stale one), the
    diagnostic log line must still fire, exactly once."""
    rid = "f5test02"
    finished_frame = (
        "Worked for 1m 0s · 0 background tasks still running\n"
        f"{_INCIDENT_BOX}\n❯\n{_INCIDENT_BOX}\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    fake = FakeTmux([READY, finished_frame], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)
    with caplog.at_level(logging.INFO):
        s.ask("ping", timeout=310)
    finished_logs = [r for r in caplog.records if "main turn finished" in r.message]
    assert len(finished_logs) == 1


@pytest.mark.parametrize("verb", _SUMMARY_VERBS)
def test_ask_salvages_the_incident_reply_when_the_closing_marker_never_appears(
    tmp_path, clock, verb
):
    """Headline regression test for the 2026-08-21 incident."""
    rid = "incid001"
    frames = [READY] + [_incident_frame(rid, i, verb) for i in range(1, 400)]
    fake = FakeTmux(frames, exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=DEFAULT_STALL_TIMEOUT,
        busy_wait_budget=DEFAULT_BUSY_WAIT_BUDGET,
        salvage_stable=DEFAULT_SALVAGE_STABLE,
        no_main_turn_ceiling=DEFAULT_NO_MAIN_TURN_CEILING,
    )
    res = s.ask("please answer", timeout=DEFAULT_TIMEOUT)

    assert res.status == "ok"
    assert res.salvaged is True
    assert res.reply is not None
    assert "real answer text" in res.reply
    assert "<<<R:" not in res.reply and "<<<E:" not in res.reply

    # Resolves at roughly DEFAULT_SALVAGE_STABLE virtual time — drastically
    # less than stall_timeout (900s) and orders of magnitude less than
    # timeout (3600s).
    assert clock["now"] < DEFAULT_STALL_TIMEOUT / 4
    assert clock["now"] < DEFAULT_TIMEOUT / 10
    assert DEFAULT_SALVAGE_STABLE <= clock["now"] <= DEFAULT_SALVAGE_STABLE + 5

    # No interrupt/Escape/Ctrl-C was ever sent.
    assert not any(c[-1] == "Escape" for c in fake.sent_keys())
    assert not any(c[-1] == "C-c" for c in fake.sent_keys())

    # The rid IS present in handled_rids afterward.
    handled = (tmp_path / ".dbrain" / "handled_rids").read_text().split()
    assert rid in handled


@pytest.mark.parametrize("verb", _SUMMARY_VERBS)
def test_salvage_refuses_early_but_ceiling_salvages_when_region_has_a_working_signature(
    tmp_path, clock, verb
):
    """Round-2 blind-review finding: is_main_turn_active() can false-negative
    on a live main turn whose spinner is rendered WITHOUT the paren anchor
    _MAIN_SPINNER_RE requires (a real CLI shape, not hypothetical — reviewer
    demonstrated '✢ Razzle-dazzling…  44s · ↓1.8k tokens' with no parens).
    The FAST 120s salvage path still refuses this (the _WORKING_RE conjunct
    is a hard gate there — see the ceiling-salvage comment in ask() for why):
    a TRUNCATED, still-generating reply must never be marked delivered THAT
    early, since the real full answer would then be permanently blocked by
    handled_rids.

    R2b (Fable audit, 2026-08-22) changes what happens AFTER the full
    DEFAULT_NO_MAIN_TURN_CEILING with still no closing marker: at that point
    the audit explicitly judges a possibly-truncated delivery better than
    permanent loss, so ask() now delivers the open span regardless of the
    working-signature conjunct — this is an ACCEPTED, STATED risk (see the
    comment in ask()), not an oversight. This fixture — an ambiguous,
    ultimately-static frame — proves the ceiling reaches that branch and
    delivers rather than timing out."""
    rid = "spin0001"
    pane = (
        f"⏺ <<<R:{rid}>>>\n"
        "This is the real answer text that never got its closing marker.\n"
        "✢ Razzle-dazzling…  44s · ↓1.8k tokens\n"
        f"{verb} for 2m 22s · 1 background task still running\n"
        f"{_INCIDENT_BOX}\n❯\n{_INCIDENT_BOX}\n"
        "  hello | Opus 4.8 (1M context) | ~/p\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    fake = FakeTmux([READY, pane], exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=DEFAULT_STALL_TIMEOUT,
        busy_wait_budget=DEFAULT_BUSY_WAIT_BUDGET,
        salvage_stable=DEFAULT_SALVAGE_STABLE,
        no_main_turn_ceiling=DEFAULT_NO_MAIN_TURN_CEILING,
    )
    res = s.ask("please answer", timeout=DEFAULT_TIMEOUT)

    # Ceiling salvage (R2b): delivered, marked salvaged, NOT a bare timeout.
    assert res.status == "ok"
    assert res.salvaged is True
    assert res.reply is not None and "real answer text" in res.reply
    handled_path = tmp_path / ".dbrain" / "handled_rids"
    handled = handled_path.read_text().split() if handled_path.exists() else []
    assert rid in handled
    assert (
        DEFAULT_NO_MAIN_TURN_CEILING
        <= clock["now"]
        <= DEFAULT_NO_MAIN_TURN_CEILING + 5
    )
    assert not any(c[-1] == "Escape" for c in fake.sent_keys())
    assert not any(c[-1] == "C-c" for c in fake.sent_keys())


@pytest.mark.parametrize("verb", _SUMMARY_VERBS)
def test_salvaged_rid_never_redelivered_by_orphan_poller(tmp_path, clock, verb):
    """After a salvage, the orphan poller must never re-deliver the same
    rid — even once a (delayed) proper E marker appears on screen."""
    rid = "incid002"
    frames = [READY] + [_incident_frame(rid, i, verb) for i in range(1, 400)]
    fake = FakeTmux(frames, exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=DEFAULT_STALL_TIMEOUT,
        salvage_stable=DEFAULT_SALVAGE_STABLE,
        no_main_turn_ceiling=DEFAULT_NO_MAIN_TURN_CEILING,
    )
    res = s.ask("please answer", timeout=DEFAULT_TIMEOUT)
    assert res.status == "ok" and res.salvaged is True

    # A delayed repaint now shows the pair as complete.
    salvaged_text = "This is the real answer text that never got its closing marker."
    fake._captures = [_complete(rid, salvaged_text)]
    assert s.pop_orphan_replies() == []


def test_ask_does_not_salvage_a_streaming_reply_still_in_progress(tmp_path, clock):
    """R marker present, no E, body actually GROWING frame to frame, main
    spinner present (paren-anchored) — must not salvage early even well
    past DEFAULT_SALVAGE_STABLE; once a proper E marker appears, returns
    the COMPLETE final body, not a partial early salvage."""
    rid = "stream01"

    def growing_frame(i: int) -> str:
        lines = "\n".join(f"line {j}" for j in range(1, i + 1))
        return f"<<<R:{rid}>>>\n{lines}\nWarping… ({i}s · ↓{i}k tokens)\n"

    n_polls = int(DEFAULT_SALVAGE_STABLE) + 50
    growth_frames = [growing_frame(i) for i in range(1, n_polls)]
    final_body = "\n".join(f"line {j}" for j in range(1, n_polls))
    final_frame = (
        f"<<<R:{rid}>>>\n{final_body}\n<<<E:{rid}>>>\n❯\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    frames = [READY, *growth_frames, final_frame]
    fake = FakeTmux(frames, exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=DEFAULT_STALL_TIMEOUT,
        salvage_stable=DEFAULT_SALVAGE_STABLE,
        no_main_turn_ceiling=DEFAULT_NO_MAIN_TURN_CEILING,
    )
    res = s.ask("ping", timeout=DEFAULT_TIMEOUT)
    assert res.status == "ok"
    assert res.salvaged is False
    assert res.reply == final_body


def test_ask_does_not_salvage_while_esc_to_interrupt_present_outside_footer(
    tmp_path, clock
):
    """Legacy hint semantics survive C2 (backlog item 12): a standalone,
    non-footer "(esc to interrupt)" line is still the whole-turn hint it
    always was — is_main_turn_active() stays True and salvage stays
    refused. Only a footer-FUSED occurrence of the same string is affected
    by C2 (see test_ask_salvages_when_only_footer_esc_to_interrupt_present
    below)."""
    _assert_no_early_salvage_or_ceiling(
        tmp_path, clock, "  ✻ Working…  (esc to interrupt)\n"
    )


def test_ask_salvages_when_only_footer_esc_to_interrupt_present(tmp_path, clock):
    """C2 fix (backlog item 12, defect P3): this CLI version renders "esc to
    interrupt" INSIDE its persistent footer whenever ANYTHING is
    interruptible — background tasks, monitors — with the main turn long
    finished (99.6% of 6512 real occurrences measured 2026-08-22). Pre-fix,
    is_main_turn_active() treated that footer text as main-turn liveness and
    pinned last_main_turn_active forever, so NEITHER salvage NOR the ceiling
    could ever fire — this is the literal production bug (salvage never
    fires). With only the footer form on screen (no paren spinner, no
    waiting-for-agents line) the main turn is demonstrably finished and
    salvage must now be allowed."""
    rid = "footeronly1"
    pane = (
        f"⏺ <<<R:{rid}>>>\n"
        "This is the real answer text that never got its closing marker.\n"
        "Worked for 2m 22s · 2 background tasks still running\n"
        f"{_INCIDENT_BOX}\n"
        "❯ \n"
        f"{_INCIDENT_BOX}\n"
        f"{_INCIDENT_FOOTER}"
    )
    assert is_main_turn_active(pane) is False

    fake = FakeTmux([READY, pane], exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=DEFAULT_STALL_TIMEOUT,
        salvage_stable=DEFAULT_SALVAGE_STABLE,
        no_main_turn_ceiling=DEFAULT_NO_MAIN_TURN_CEILING,
    )
    res = s.ask("please answer", timeout=DEFAULT_TIMEOUT)
    assert res.status == "ok"
    assert res.salvaged is True
    assert (
        res.reply
        == "This is the real answer text that never got its closing marker."
    )
    assert DEFAULT_SALVAGE_STABLE <= clock["now"] <= DEFAULT_SALVAGE_STABLE + 5


_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_ask_salvages_the_golden_incident_fixture(tmp_path, clock):
    """End-to-end proof against the golden fixture (backlog item 12, T3): a
    hand-constructed but byte-faithful ~239-line incident pane (long reply,
    no closing marker, turn-summary line, then the real 11-line bottom
    chrome including the footer-fused "esc to interrupt" hint). ask() must
    salvage it at ~DEFAULT_SALVAGE_STABLE virtual seconds, not ride the
    ceiling all the way to 300s."""
    rid = "gold0001"
    pane = (_FIXTURES_DIR / "pane_salvage_incident.txt").read_text(encoding="utf-8")
    fake = FakeTmux([READY, pane], exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=DEFAULT_STALL_TIMEOUT,
        salvage_stable=DEFAULT_SALVAGE_STABLE,
        no_main_turn_ceiling=DEFAULT_NO_MAIN_TURN_CEILING,
    )
    res = s.ask("please answer", timeout=DEFAULT_TIMEOUT)

    assert res.status == "ok"
    assert res.salvaged is True
    assert res.reply is not None
    assert res.reply.rstrip().endswith("season.")
    assert "<<<R:" not in res.reply and "<<<E:" not in res.reply
    assert DEFAULT_SALVAGE_STABLE <= clock["now"] <= DEFAULT_SALVAGE_STABLE + 5


def test_ask_does_not_salvage_while_paren_spinner_present(tmp_path, clock):
    _assert_no_early_salvage_or_ceiling(
        tmp_path, clock, "Warping… (5s · ↓1.2k tokens)\n"
    )


def test_ask_does_not_salvage_while_waiting_for_background_agent(tmp_path, clock):
    _assert_no_early_salvage_or_ceiling(
        tmp_path, clock, "✻ Waiting for 1 background agent to finish\n"
    )


def _assert_no_early_salvage_or_ceiling(tmp_path, clock, active_signature: str) -> None:
    """Body region byte-identical for far longer than DEFAULT_SALVAGE_STABLE,
    but the main turn is still active — must keep polling in all three
    cases, reaching the ordinary generic timeout, not our new salvage/
    ceiling exits."""
    rid = "active001"
    pane = f"<<<R:{rid}>>>\nBody stable text.\n{active_signature}"
    fake = FakeTmux([READY, pane], exists=True)
    small_timeout = 50.0
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=1e6,  # never trips the generic stall check either
        salvage_stable=10.0,  # deliberately tiny — proves it still never fires
        no_main_turn_ceiling=20.0,  # deliberately tiny — proves it still never fires
    )
    res = s.ask("ping", timeout=small_timeout)
    assert res.status == "timeout"
    assert res.detail == f"no reply in {small_timeout}s"  # the ORIGINAL generic path
    assert res.salvaged is False
    assert clock["now"] >= small_timeout


@pytest.mark.parametrize("verb", _SUMMARY_VERBS)
def test_ask_normal_completion_wins_over_salvage_even_when_salvage_would_qualify(
    tmp_path, clock, verb
):
    """A capture that would otherwise qualify for salvage timing-wise (both
    new constants set to 0) but ALSO contains a complete R/E pair must take
    the normal completion path."""
    rid = "wins0001"
    pane = (
        f"<<<R:{rid}>>>\nComplete answer body.\n<<<E:{rid}>>>\n"
        f"{verb} for 1m 0s · 0 background tasks still running\n"
        f"{_INCIDENT_BOX}\n❯\n{_INCIDENT_BOX}\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    fake = FakeTmux([READY, pane], exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=DEFAULT_STALL_TIMEOUT,
        salvage_stable=0.0,
        no_main_turn_ceiling=0.0,
    )
    res = s.ask("ping", timeout=100)
    assert res.status == "ok"
    assert res.salvaged is False
    assert res.reply == "Complete answer body."


@pytest.mark.parametrize("verb", _SUMMARY_VERBS)
def test_ask_ceiling_fires_when_no_marker_and_no_main_turn_activity(
    tmp_path, clock, verb
):
    """Main turn finished, no E marker, AND the R marker itself is absent
    from the capture (so extract_open_reply returns None, salvage is
    impossible) — the F2 class (Fable audit R2c). Returns timeout status at
    approximately DEFAULT_NO_MAIN_TURN_CEILING virtual time, with the
    DISTINCT honest detail ("no reply markers ever appeared...") since the R
    marker was never seen even once — no Escape/interrupt sent, inflight
    cleared, rid NOT in handled_rids."""
    rid = "ceil0001"
    pane = (
        f"{verb} for 2m 22s · 0 background tasks still running\n"
        f"{_INCIDENT_BOX}\n❯\n{_INCIDENT_BOX}\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    fake = FakeTmux([READY, pane], exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=DEFAULT_STALL_TIMEOUT,
        salvage_stable=DEFAULT_SALVAGE_STABLE,
        no_main_turn_ceiling=DEFAULT_NO_MAIN_TURN_CEILING,
    )
    res = s.ask("ping", timeout=DEFAULT_TIMEOUT)

    assert res.status == "timeout"
    assert res.detail is not None and "no reply markers ever appeared" in res.detail
    assert (
        DEFAULT_NO_MAIN_TURN_CEILING
        <= clock["now"]
        <= DEFAULT_NO_MAIN_TURN_CEILING + 5
    )
    assert not any(c[-1] == "Escape" for c in fake.sent_keys())
    assert not any(c[-1] == "C-c" for c in fake.sent_keys())
    assert not (tmp_path / ".dbrain" / "inflight").exists()
    handled_path = tmp_path / ".dbrain" / "handled_rids"
    handled = handled_path.read_text().split() if handled_path.exists() else []
    assert rid not in handled

    # The reply stays recoverable: a later capture where the pair IS
    # complete must be handed out by the orphan poller exactly once.
    fake._captures = [_complete(rid, "Recovered later.")]
    assert s.pop_orphan_replies() == ["Recovered later."]
    assert s.pop_orphan_replies() == []


def test_ceiling_gives_the_generic_detail_when_r_marker_was_seen_but_lost(
    tmp_path, clock
):
    """Distinguishes the two F2-adjacent ceiling outcomes (R2c): when the R
    marker WAS seen at some point (here: present, then scrolls out of the
    capture window before the ceiling) the detail stays the pre-existing
    generic wording, not the "never appeared" honest message — that message
    is reserved for the case where the marker is truly never seen."""
    rid = "ceil0002"
    with_marker = (
        f"⏺ <<<R:{rid}>>>\nsome text\n"
        f"Worked for 2m 22s · 0 background tasks still running\n"
        f"{_INCIDENT_BOX}\n❯\n{_INCIDENT_BOX}\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    without_marker = (
        "Worked for 2m 22s · 0 background tasks still running\n"
        f"{_INCIDENT_BOX}\n❯\n{_INCIDENT_BOX}\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    # _WORKING_RE would otherwise let the ceiling-salvage branch (R2b) fire
    # since main_turn_finished is True throughout — but fresh_open is None
    # once the marker has scrolled out, so it correctly falls through to the
    # "nothing to salvage" branch this test targets.
    fake = FakeTmux([READY, with_marker, without_marker], exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=DEFAULT_STALL_TIMEOUT,
        salvage_stable=DEFAULT_SALVAGE_STABLE,
        no_main_turn_ceiling=DEFAULT_NO_MAIN_TURN_CEILING,
    )
    res = s.ask("ping", timeout=DEFAULT_TIMEOUT)
    assert res.status == "timeout"
    assert res.detail == "no closing marker and no active main turn"


def test_defect_b_idle_pane_with_background_rows_is_not_treated_as_busy_on_presend(
    tmp_path, clock
):
    """Defect B: an idle pane that merely has ticking background-agent rows
    visible must not be mistaken for "busy" on the pre-send gate — the
    prompt IS typed immediately, no 300s busy-wait triggers, and the result
    is never an error with "busy" in its detail."""
    rid = "defb0001"
    pre_cap = (
        "  ◯ general-purpose  still going   54s · ↓ 79.7k tokens\n❯\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    fake = FakeTmux([pre_cap, _complete(rid, "done")], exists=True)
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=fake,
        sleep_fn=lambda secs: clock.__setitem__("now", clock["now"] + secs),
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: rid,
        poll_interval=1.0,
        startup_timeout=30.0,
        stall_timeout=DEFAULT_STALL_TIMEOUT,
        busy_wait_budget=DEFAULT_BUSY_WAIT_BUDGET,
    )
    res = s.ask("ping", timeout=DEFAULT_TIMEOUT)
    assert res.status == "ok"
    assert "paste-buffer" in fake.sent_subcommands()
    assert "busy" not in (res.detail or "").lower()
    # No 300s busy-wait was triggered — total elapsed is a couple of polls.
    assert clock["now"] < DEFAULT_BUSY_WAIT_BUDGET / 10


def test_defect_b_pre_send_background_rows_never_set_maintenance_placeholder(
    tmp_path, clock
):
    """During the pre-send window with only background-agent rows visible,
    `inflight` must never carry the maint-pending placeholder — the
    steerability check must never falsely report a maintenance turn here."""
    rid = "defb0002"
    pre_cap = (
        "  ◯ general-purpose  still going   54s · ↓ 79.7k tokens\n❯\n"
        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
    )
    fake = FakeTmux([pre_cap, _complete(rid, "done")], exists=True)
    s = make_session(tmp_path, fake, clock, rid=rid)

    seen_inflight_first_lines: list[str] = []
    orig_send_prompt = s._send_prompt

    def spy_send_prompt(prompt, rid_, *, wrap=True):
        inflight = (tmp_path / ".dbrain" / "inflight").read_text().splitlines()[0]
        seen_inflight_first_lines.append(inflight)
        orig_send_prompt(prompt, rid_, wrap=wrap)

    s._send_prompt = spy_send_prompt
    res = s.ask("ping")

    assert res.status == "ok"
    assert seen_inflight_first_lines  # the prompt WAS sent
    assert all(
        not line.startswith(MAINT_PREFIX) for line in seen_inflight_first_lines
    )


# ── last_reply_for_resend (backlog item 14: /resend) ─────────────────────


def _jsonl_assistant_record(text: str) -> dict:
    return {
        "type": "assistant",
        "isSidechain": False,
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _write_transcript(s: ClaudeSession, *records: dict) -> Path:
    path = s.current_transcript_path()
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n" if records else "")
    return path


def test_last_reply_for_resend_unavailable_before_any_session(tmp_path, clock):
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    status, body = s.last_reply_for_resend()
    assert status == "unavailable"
    assert body is None


def test_last_reply_for_resend_ready(tmp_path, clock):
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    _write_transcript(
        s, _jsonl_assistant_record("<<<R:rid1>>>\nHello there.\n<<<E:rid1>>>\n")
    )
    status, body = s.last_reply_for_resend()
    assert status == "ready"
    assert body == "Hello there."


def test_last_reply_for_resend_recovers_unclosed_marker_when_not_active(
    tmp_path, clock
):
    """F1 fix: this is the literal shape of every real delivery-loss
    incident on record (bfbe3335, df4f87ef, 9dd35326) — an open
    <<<R:id>>> span with no closing marker. No turn is active here (the
    fake pane is idle), so the recovered body must be delivered as
    'ready' with a salvage notice, not falsely reported as still in
    progress."""
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    _write_transcript(
        s, _jsonl_assistant_record("<<<R:rid1>>>\nStill writing the answer...")
    )
    status, body = s.last_reply_for_resend()
    assert status == "ready"
    assert body is not None
    assert "Still writing the answer..." in body
    assert "восстановлен без закрывающего маркера" in body


def test_last_reply_for_resend_unclosed_marker_stays_in_progress_when_active(
    tmp_path, clock
):
    """Same unclosed-marker shape as above, but a turn IS genuinely active
    (pane.lock held) — the honest status is still 'in_progress', never a
    premature 'ready'."""
    import fcntl
    import os

    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    _write_transcript(
        s, _jsonl_assistant_record("<<<R:rid1>>>\nStill writing the answer...")
    )

    lock_fd = os.open(tmp_path / ".dbrain" / "pane.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        status, body = s.last_reply_for_resend()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    assert status == "in_progress"
    assert body is None


def test_last_reply_for_resend_no_markers_when_idle(tmp_path, clock):
    """F-1 fix (round 2 review): the most recent assistant record has NO
    marker at all (not even an open one), and no turn is active — this is
    literally the feature's primary target bucket (the audit's "3.3% of
    turns with no marker at all"). It must be reported as the honest
    'no_markers' status, never as 'in_progress' (which would falsely claim
    the session is still working)."""
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    _write_transcript(s, _jsonl_assistant_record("Just a plain reply, no markers."))
    status, body = s.last_reply_for_resend()
    assert status == "no_markers"
    assert body is None


def test_last_reply_for_resend_no_markers_stays_in_progress_when_active(
    tmp_path, clock
):
    """Same no-marker shape as above, but a turn IS genuinely active
    (pane.lock held) — the honest status is 'in_progress', not a false
    'no_markers'/idle verdict. Must not regress this direction."""
    import fcntl
    import os

    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    _write_transcript(s, _jsonl_assistant_record("Just a plain reply, no markers."))

    lock_fd = os.open(tmp_path / ".dbrain" / "pane.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        status, body = s.last_reply_for_resend()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    assert status == "in_progress"
    assert body is None


def test_last_reply_for_resend_reply_split_across_records_is_not_falsely_in_progress(
    tmp_path, clock
):
    """Reviewer-found reverse mis-case: a reply's open <<<R:id>>> marker
    lands in one transcript record and its closing <<<E:id>>> lands in the
    NEXT record. latest_reply() only ever inspects the single most recent
    assistant text record, so from its point of view the last record has no
    marker at all (same shape as "no markers"). With no turn active, this
    must not be falsely reported as 'in_progress' — it should come back as
    the honest 'no_markers' status (no cross-record marker stitching is
    attempted)."""
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    _write_transcript(
        s,
        _jsonl_assistant_record("<<<R:rid1>>>\nThe answer, split across"),
        _jsonl_assistant_record(" records.\n<<<E:rid1>>>\n"),
    )
    status, body = s.last_reply_for_resend()
    assert status != "in_progress"
    assert status == "no_markers"
    assert body is None


def test_last_reply_for_resend_empty_when_no_history_and_no_active_turn(
    tmp_path, clock
):
    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    _write_transcript(s)  # transcript file exists, but has no records at all
    status, body = s.last_reply_for_resend()
    assert status == "empty"
    assert body is None


def test_last_reply_for_resend_in_progress_when_turn_active_but_no_text_yet(
    tmp_path, clock
):
    """No assistant text has landed in the transcript yet, but a turn is
    genuinely in flight (pane.lock held) — 'still working' is the honest
    status, not 'empty'."""
    import fcntl
    import os

    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    _write_transcript(s)

    lock_fd = os.open(tmp_path / ".dbrain" / "pane.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        status, body = s.last_reply_for_resend()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    assert status == "in_progress"
    assert body is None


def test_last_reply_for_resend_never_waits_on_pane_lock(tmp_path, clock):
    """MUST NOT WAIT (block) on pane.lock under any circumstance — the
    whole point is that /resend keeps working while ask() holds it for up
    to an hour. is_turn_active() legitimately attempts a brief NON-blocking
    (LOCK_NB) acquire+release when the pane is idle (F4 fix, 2026-08-22);
    the true contract is "never blocking", not "never touches the lock at
    all".

    Proven by recording the actual `blocking=` kwarg every `_locked()` call
    receives, rather than by making the double raise a plain
    ``AssertionError`` — that used to be silently swallowed by
    last_reply_for_resend()'s own ``except Exception``, so the old version
    of this test passed even if the code took the lock BLOCKING."""
    from contextlib import contextmanager

    fake = FakeTmux([READY], exists=False)
    s = make_session(tmp_path, fake, clock)
    s.ensure_session()
    _write_transcript(s)  # forces the "empty" branch to consult is_turn_active()

    real_locked = s._locked
    calls: list[bool] = []

    @contextmanager
    def _spy(*args, blocking: bool = True, **kwargs):
        calls.append(blocking)
        with real_locked(*args, blocking=blocking, **kwargs) as got:
            yield got

    s._locked = _spy  # type: ignore[method-assign]

    status, body = s.last_reply_for_resend()
    assert status == "empty"
    assert body is None
    assert calls, (
        "expected last_reply_for_resend to consult _locked() via is_turn_active()"
    )
    assert all(blocking is False for blocking in calls), (
        f"last_reply_for_resend must only ever attempt a non-blocking lock, got {calls}"
    )


def test_start_command_pins_claude_config_dir(tmp_path, monkeypatch):
    # Second rehearsal review: a tmux server started without the variable
    # does not pass it to new sessions, so the command itself carries it.
    s = ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=FakeTmux([""], exists=True),
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg dir"))
    cmd = s._start_command("00000000-0000-0000-0000-000000000000")
    assert f"CLAUDE_CONFIG_DIR='{tmp_path / 'cfg dir'}' " in cmd
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    plain = s._start_command("00000000-0000-0000-0000-000000000000")
    assert "CLAUDE_CONFIG_DIR" not in plain
