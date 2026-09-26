"""Tests for CodexExecDriver (Codex engine, phase 2).

The real `codex exec` subprocess is replaced by FakeCodexProc, which plays a
scripted JSONL event stream — the same shape the phase-0 spike recorded from
the live CLI (thread.started / turn.started / item.completed / turn.completed)
and re-verified against codex-cli 0.153.4 while writing this driver. Clock,
sleep and rid generation are injected so timeout and stall detection are
deterministic and fast, mirroring how tests/test_claude_session.py fakes tmux.

Nothing here shells out. The one genuinely live check is a standalone script,
`scripts/codex-smoke.py`, run by hand — see its docstring.
"""

import json
import signal
import threading
import time
from pathlib import Path

import pytest

from d_brain.services.claude_session import AskResult
from d_brain.services.codex_driver import CodexExecDriver
from d_brain.services.engine import EngineDriver
from d_brain.services.tmux_parse import PaneState, classify_state

THREAD = "01a07118-7ed5-7d33-8b49-7081f69fed5a"
THREAD_2 = "01a07118-dead-beef-8b49-7081f69fed5a"


def ev(**kw) -> str:
    return json.dumps(kw) + "\n"


def stream_ok(reply: str = "PONG", thread: str = THREAD) -> list[str]:
    """The exact event sequence a healthy turn produces (spike step 1)."""
    return [
        ev(type="thread.started", thread_id=thread),
        ev(type="turn.started"),
        ev(
            type="item.completed",
            item={"id": "item_0", "type": "agent_message", "text": reply},
        ),
        ev(
            type="turn.completed",
            usage={"input_tokens": 15868, "output_tokens": 6},
        ),
    ]


class FakeStream:
    """A pipe-like object the driver's reader thread can readline() on."""

    def __init__(self, lines: list[str], hang: threading.Event | None = None) -> None:
        self._lines = list(lines)
        self._hang = hang

    def readline(self) -> str:
        if self._lines:
            return self._lines.pop(0)
        if self._hang is not None:
            # Model a process that produces nothing further and does not exit
            # until it is signalled — the stall / hard-timeout shape. A test
            # may append late output at signal time (the "reply landed during
            # the kill" case), so re-check the buffer after waking.
            self._hang.wait(timeout=5.0)
            if self._lines:
                return self._lines.pop(0)
        return ""


class FakeCodexProc:
    """Stand-in for the Popen object `codex exec` would produce."""

    def __init__(
        self,
        argv: list[str],
        lines: list[str],
        *,
        returncode: int = 0,
        stderr: str = "",
        hang: bool = False,
    ) -> None:
        self.argv = argv
        self.pid = 424242
        self._hang_event = threading.Event() if hang else None
        self.stdout = FakeStream(lines, self._hang_event)
        self.stderr = FakeStream([stderr] if stderr else [])
        self._returncode = returncode
        self.returncode = None if hang else returncode
        self.signals: list[int] = []
        self.killed = False

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self._finish()

    def kill(self) -> None:
        self.killed = True
        self._finish()

    def _finish(self) -> None:
        self.returncode = self._returncode
        if self._hang_event is not None:
            self._hang_event.set()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self._finish()
        return self.returncode


class FakePopen:
    """Callable factory: hands out scripted processes, records argv."""

    def __init__(self, *procs: FakeCodexProc | dict) -> None:
        self._script = list(procs)
        self.argvs: list[list[str]] = []
        self.kwargs: list[dict] = []
        self.prompts: list[str] = []
        self.made: list[FakeCodexProc] = []

    def __call__(self, argv, **kwargs):
        self.argvs.append(list(argv))
        self.kwargs.append(kwargs)
        self.prompts.append(argv[-1])
        spec = self._script.pop(0) if self._script else {"lines": stream_ok()}
        proc = spec if isinstance(spec, FakeCodexProc) else FakeCodexProc(argv, **spec)
        self.made.append(proc)
        return proc


class Clock:
    """Monotonic clock that advances a fixed step per read, so the driver's
    deadline checks fire deterministically without any real waiting."""

    def __init__(self, step: float = 0.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def make_driver(tmp_path: Path, popen: FakePopen, **over) -> CodexExecDriver:
    clock = over.pop("clock", Clock())
    persona = over.pop("instructions_file", None)
    kwargs = dict(
        session_name="dbrain_codex_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / "rt",
        instructions_file=persona,
        popen=popen,
        sleep_fn=lambda _s: None,
        clock_fn=clock,
        rid_factory=lambda: "rid0001",
        poll_interval=0.001,
        interrupt_grace=0.0,
    )
    kwargs.update(over)
    (tmp_path / "vault").mkdir(parents=True, exist_ok=True)
    drv = CodexExecDriver(**kwargs)
    drv._clock_obj = clock  # test convenience handle
    return drv


@pytest.fixture(autouse=True)
def _codex_on_path(monkeypatch):
    """ensure_session() refuses to run without the binary; every test that
    isn't ABOUT that check gets a stubbed which()."""
    monkeypatch.setattr(
        "d_brain.services.codex_driver.shutil.which", lambda name: f"/usr/bin/{name}"
    )


# ── 1. the Protocol claim, same bar phase 1 held ClaudeSession to ────────

_PROTOCOL_METHODS = (
    "ask",
    "is_turn_active",
    "is_pane_turn_active",
    "is_steerable_turn",
    "steer",
    "interrupt",
    "send_control",
    "is_working",
    "force_recover",
    "ensure_session",
    "is_healthy",
    "kill",
    "capture_text",
    "last_reply_for_resend",
    "pop_orphan_replies",
)


@pytest.mark.parametrize("name", _PROTOCOL_METHODS)
def test_matches_protocol_signature(name):
    import inspect

    assert hasattr(CodexExecDriver, name), f"CodexExecDriver is missing {name}()"
    got = inspect.signature(getattr(CodexExecDriver, name))
    want = inspect.signature(getattr(EngineDriver, name))
    assert got == want, f"{name}(): {got} != protocol {want}"


def test_is_an_engine_driver_instance(tmp_path):
    drv = make_driver(tmp_path, FakePopen())
    assert isinstance(drv, EngineDriver)


# ── 2. the happy path ────────────────────────────────────────────────────


def test_ask_round_trip_maps_to_ok(tmp_path):
    popen = FakePopen({"lines": stream_ok("Привет, сделал.")})
    drv = make_driver(tmp_path, popen)

    res = drv.ask("как дела?")

    assert res == AskResult("ok", reply="Привет, сделал.")
    assert res.ok
    # A brand-new thread carries the sandbox and the working root; both are
    # rejected by `resume`, so they can only ever appear here.
    argv = popen.argvs[0]
    assert argv[:2] == ["codex", "exec"]
    assert "--json" in argv and "-s" in argv and "-C" in argv
    assert "resume" not in argv
    assert popen.kwargs[0]["cwd"] == str(tmp_path / "vault")
    # stdin MUST be DEVNULL: with an inherited stdin the CLI blocks on
    # "Reading additional input from stdin..." and every turn hangs.
    import subprocess

    assert popen.kwargs[0]["stdin"] is subprocess.DEVNULL


def test_ask_argv_separates_dash_prefixed_prompt_from_flags(tmp_path):
    # Found live in independent review: without "--", clap parses a prompt
    # starting with "-" as a flag ("- надо купить молока" -> rc=2 "unexpected
    # argument"). The persona preamble masked this on a thread's first turn
    # only, so every resumed turn was exposed.
    popen = FakePopen({"lines": stream_ok("ок")})
    drv = make_driver(tmp_path, popen)

    drv.ask("- надо купить молока")

    argv = popen.argvs[0]
    assert argv[-2:] == ["--", "- надо купить молока"]


def test_ask_records_the_reply_for_resend(tmp_path):
    drv = make_driver(tmp_path, FakePopen({"lines": stream_ok("ответ")}))
    assert drv.last_reply_for_resend() == ("unavailable", None)
    drv.ask("q")
    assert drv.last_reply_for_resend() == ("ready", "ответ")


def test_ask_strips_markers_a_shared_skill_may_have_asked_for(tmp_path):
    reply = "<<<R:abc123>>>\n<b>готово</b>\n<<<E:abc123>>>"
    drv = make_driver(tmp_path, FakePopen({"lines": stream_ok(reply)}))
    assert drv.ask("q").reply == "<b>готово</b>"


def test_completed_turn_with_no_message_is_an_error_not_an_empty_ok(tmp_path):
    lines = [
        ev(type="thread.started", thread_id=THREAD),
        ev(type="turn.completed", usage={}),
    ]
    drv = make_driver(tmp_path, FakePopen({"lines": lines}))
    res = drv.ask("q")
    assert res.status == "error"
    assert "no reply" in (res.detail or "")


# ── 3. thread-id persistence: the thing that makes multi-turn work ───────


def test_thread_id_persists_across_two_sequential_asks(tmp_path):
    popen = FakePopen(
        {"lines": stream_ok("one")},
        {"lines": stream_ok("two")},
    )
    drv = make_driver(tmp_path, popen)

    assert drv.ask("first").reply == "one"
    assert (tmp_path / "rt" / "thread_id").read_text().strip() == THREAD
    assert drv.ask("second").reply == "two"

    first, second = popen.argvs
    assert "resume" not in first
    # Second turn resumes the SAME thread, and carries neither -s nor -C:
    # `codex exec resume` rejects both ("error: unexpected argument").
    assert second[:5] == ["codex", "exec", "resume", THREAD, "--json"]
    assert "-s" not in second and "-C" not in second


def test_a_fresh_driver_reuses_the_thread_left_on_disk(tmp_path):
    """Bot restarts must not cost the conversation — the thread lives in
    runtime_dir, not in the process."""
    make_driver(tmp_path, FakePopen({"lines": stream_ok()})).ask("first")
    popen2 = FakePopen({"lines": stream_ok("again")})
    assert make_driver(tmp_path, popen2).ask("second").reply == "again"
    assert popen2.argvs[0][2:4] == ["resume", THREAD]


def test_persona_is_injected_only_on_the_first_turn_of_a_thread(tmp_path):
    persona = tmp_path / "codex-agents.md"
    persona.write_text("# d-brain codex agent contract\nТы — ассистент.\n")
    popen = FakePopen({"lines": stream_ok()}, {"lines": stream_ok()})
    drv = make_driver(tmp_path, popen, instructions_file=persona)

    drv.ask("первый вопрос")
    drv.ask("второй вопрос")

    assert "Ты — ассистент." in popen.prompts[0]
    assert popen.prompts[0].endswith("первый вопрос")
    # The thread carries the persona in its replayed context from here on;
    # re-sending it every turn would be pure waste.
    assert "Ты — ассистент." not in popen.prompts[1]
    assert popen.prompts[1] == "второй вопрос"


def test_persona_inlines_vault_rules_and_skips_missing_ones(tmp_path):
    """Codex does not auto-load .claude/rules; a vault-include line makes a
    rule reach the prompt from its single source in the instance's vault."""
    rules = tmp_path / "vault" / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "style.md").write_text("# Стиль\nКоротко.\n")
    persona = tmp_path / "codex-agents.md"
    persona.write_text(
        "# d-brain codex agent contract\n"
        "<!-- vault-include: .claude/rules/style.md -->\n"
        "<!-- vault-include: .claude/rules/absent.md -->\n"
        "<!-- vault-include: ../secret.md -->\n"
        "Хвост.\n"
    )
    popen = FakePopen({"lines": stream_ok()})
    make_driver(tmp_path, popen, instructions_file=persona).ask("вопрос")

    prompt = popen.prompts[0]
    assert "# Стиль\nКоротко." in prompt
    assert "vault-include" not in prompt
    assert "Хвост." in prompt and prompt.endswith("вопрос")


def test_wrap_never_alters_the_prompt_on_this_engine(tmp_path):
    """The marker contract exists to find a reply in a screen scrape. Here
    the reply is a discrete event, so wrap must be a genuine no-op."""
    popen = FakePopen({"lines": stream_ok()}, {"lines": stream_ok()})
    drv = make_driver(tmp_path, popen)
    drv.ask("вопрос", wrap=True)
    drv.ask("вопрос", wrap=False)
    assert popen.prompts == ["вопрос", "вопрос"]
    assert "<<<R:" not in popen.prompts[0]


# ── 4. timeout / stall ───────────────────────────────────────────────────


def test_hard_timeout_interrupts_and_maps_to_timeout(tmp_path):
    proc = FakeCodexProc([], [], returncode=1, hang=True)
    popen = FakePopen(proc)
    drv = make_driver(tmp_path, popen, clock=Clock(step=1.0))

    res = drv.ask("долгий вопрос", timeout=5.0)

    assert res.status == "timeout"
    assert "no reply in 5.0s" in (res.detail or "")
    # Spike step 5: SIGINT is enough, the process dies on its own.
    assert signal.SIGINT in proc.signals


def test_no_completion_event_at_all_maps_to_timeout(tmp_path):
    """The stall shape: the process is alive but the stream has gone quiet.
    Nothing crashed and nothing was delivered — 'timeout' is the honest
    status, and it is the one the plan specifies."""
    proc = FakeCodexProc([], [ev(type="thread.started", thread_id=THREAD)], hang=True)
    drv = make_driver(
        tmp_path, FakePopen(proc), clock=Clock(step=1.0), stall_timeout=3.0
    )

    res = drv.ask("вопрос", timeout=1000.0)

    assert res.status == "timeout"
    assert "no codex stream event" in (res.detail or "")
    assert signal.SIGINT in proc.signals
    # The thread id seen before the stall is still persisted — a wedged turn
    # must not cost the conversation.
    assert (tmp_path / "rt" / "thread_id").read_text().strip() == THREAD


def test_a_reply_that_lands_during_the_kill_becomes_an_orphan(tmp_path):
    """The one real orphan case on this engine: the turn actually finished,
    but after this caller had given up. The answer is real — queue it for the
    watchdog rather than dropping it."""
    proc = FakeCodexProc([], [], returncode=1, hang=True)
    popen = FakePopen(proc)
    drv = make_driver(tmp_path, popen, clock=Clock(step=1.0))

    # Once the driver signals the process, let the buffered completion appear.
    original_signal = proc.send_signal

    def signal_then_emit(sig):
        proc.stdout._lines.extend(stream_ok("поздний ответ"))
        original_signal(sig)

    proc.send_signal = signal_then_emit

    res = drv.ask("вопрос", timeout=3.0)

    assert res.status == "timeout"
    assert "arrived after" in (res.detail or "")
    assert drv.pop_orphan_replies() == ["поздний ответ"]
    assert drv.pop_orphan_replies() == []  # consumed exactly once


# ── 5. interrupt ─────────────────────────────────────────────────────────


def test_interrupt_signals_the_live_turn_and_the_thread_survives(tmp_path):
    """The full spike step-5 round trip: SIGINT mid-turn, then resume the
    same thread and get a normal answer back."""
    sent: list[tuple[int, int]] = []

    proc = FakeCodexProc([], [ev(type="thread.started", thread_id=THREAD)], hang=True)
    popen = FakePopen(proc, {"lines": stream_ok("после прерывания", thread=THREAD)})
    drv = make_driver(tmp_path, popen, clock=Clock(step=1.0), stall_timeout=3.0)

    # A concurrent interrupt() from another process reads the pid file.
    def fake_kill(pid, sig):
        if sig == 0:
            return None
        sent.append((pid, sig))

    import d_brain.services.codex_driver as mod

    monkey = mod.os.kill
    mod.os.kill = fake_kill
    try:
        drv._atomic_write(drv._pid_file, "424242\n")
        assert drv.is_pane_turn_active() is True
        drv.interrupt()
        assert sent == [(424242, signal.SIGINT)]
    finally:
        mod.os.kill = monkey

    drv._pid_file.unlink(missing_ok=True)
    # The turn was interrupted; the thread is intact and resumable.
    drv.ask("вопрос", timeout=1000.0)  # the stalled one
    res = drv.ask("и что получилось?")
    assert res == AskResult("ok", reply="после прерывания")
    assert popen.argvs[1][2:4] == ["resume", THREAD]


def test_a_cancelled_turn_is_not_blamed_on_a_failure_that_never_happened(tmp_path):
    """Observed live while building this driver: an externally SIGINT'ed turn
    exits rc=1 having emitted thread.started + turn.started and NOTHING else —
    no `error`, no `turn.failed`. A genuine API failure emits both. That makes
    the two distinguishable from the stream alone, so the detail says
    "interrupted" instead of inventing a failure."""
    lines = [
        ev(type="thread.started", thread_id=THREAD),
        ev(type="turn.started"),
    ]
    drv = make_driver(tmp_path, FakePopen({"lines": lines, "returncode": 1}))
    res = drv.ask("считай до 400")
    assert res.status == "error"  # parity: the tmux engine's interrupt path too
    assert "interrupted before it produced a reply" in (res.detail or "")
    assert "reason=cancelled" in drv.capture_text()


def test_interrupt_without_a_live_turn_is_a_quiet_no_op(tmp_path):
    drv = make_driver(tmp_path, FakePopen())
    drv.interrupt()  # must not raise
    assert drv.is_pane_turn_active() is False


# ── 6. error classification ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("You've hit your usage limit.", "rate_limited"),
        ("stream error: 429 too many requests", "rate_limited"),
        ("rate limit exceeded", "rate_limited"),
        ("usage not included in your plan", "rate_limited"),
        ("401 unauthorized", "logged_out"),
        ("not logged in; run codex login", "logged_out"),
        ("server overloaded", "error"),
        ("context window exceeded", "error"),
    ],
)
def test_turn_failed_maps_onto_the_shared_status_vocabulary(
    tmp_path, message, expected
):
    lines = [
        ev(type="thread.started", thread_id=THREAD),
        ev(type="error", message=message),
        ev(type="turn.failed", error={"message": message}),
    ]
    drv = make_driver(tmp_path, FakePopen({"lines": lines, "returncode": 1}))
    res = drv.ask("q")
    assert res.status == expected
    assert message[:40] in (res.detail or "")


def test_rate_limit_is_recovered_from_the_rollout_file(tmp_path):
    """Phase-0's explicit instruction to phase 2: limit telemetry is NOT in
    the --json stream. A generic failure must still be recognised as a limit
    via the thread's rollout JSONL."""
    home = tmp_path / "codexhome"
    day = home / "sessions" / "2026" / "09" / "05"
    day.mkdir(parents=True)
    (day / f"rollout-2026-09-05T12-23-45-{THREAD}.jsonl").write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "rate_limits": {
                        "primary": {"used_percent": 100.0},
                        "plan_type": "prolite",
                        "rate_limit_reached_type": "primary",
                    },
                },
            }
        )
        + "\n"
    )
    lines = [
        ev(type="thread.started", thread_id=THREAD),
        ev(type="turn.failed", error={"message": "stream disconnected"}),
    ]
    drv = make_driver(
        tmp_path,
        FakePopen({"lines": lines, "returncode": 1}),
        codex_home=home,
    )
    assert drv.ask("q").status == "rate_limited"


def test_a_healthy_rollout_file_does_not_invent_a_rate_limit(tmp_path):
    home = tmp_path / "codexhome"
    day = home / "sessions" / "2026" / "09" / "05"
    day.mkdir(parents=True)
    (day / f"rollout-x-{THREAD}.jsonl").write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "rate_limits": {"rate_limit_reached_type": None},
                },
            }
        )
        + "\n"
    )
    lines = [
        ev(type="thread.started", thread_id=THREAD),
        ev(type="turn.failed", error={"message": "stream disconnected"}),
    ]
    drv = make_driver(
        tmp_path, FakePopen({"lines": lines, "returncode": 1}), codex_home=home
    )
    assert drv.ask("q").status == "error"


def test_a_missing_binary_is_an_error_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr("d_brain.services.codex_driver.shutil.which", lambda name: None)
    drv = make_driver(tmp_path, FakePopen())
    res = drv.ask("q")
    assert res.status == "error"
    assert "codex binary" in (res.detail or "")
    assert drv.is_healthy() is False


# ── 7. busy / busy_active ────────────────────────────────────────────────


def test_a_contended_lock_with_no_progress_is_busy(tmp_path):
    drv = make_driver(
        tmp_path, FakePopen(), clock=Clock(step=1.0), busy_wait_budget=3.0
    )
    held = drv._try_lock()
    assert held is not None
    try:
        res = drv.ask("q")
    finally:
        drv._unlock(held)
    assert res.status == "busy"
    assert res.busy_seconds is not None


def test_a_contended_lock_with_a_growing_stream_is_busy_active(tmp_path):
    """A live, working turn must not be scored as a delivery failure — the
    same distinction ask_health draws for the tmux engine."""
    drv = make_driver(
        tmp_path, FakePopen(), clock=Clock(step=1.0), busy_wait_budget=3.0
    )
    held = drv._try_lock()
    assert held is not None
    drv._append_event_line(ev(type="turn.started"))

    real_sleep = drv._sleep
    calls = {"n": 0}

    def grow(_seconds):
        calls["n"] += 1
        drv._append_event_line(ev(type="item.completed", item={"n": calls["n"]}))
        real_sleep(_seconds)

    drv._sleep = grow
    try:
        res = drv.ask("q")
    finally:
        drv._unlock(held)
    assert res.status == "busy_active"


def test_is_turn_active_tracks_the_lock(tmp_path):
    drv = make_driver(tmp_path, FakePopen())
    assert drv.is_turn_active() is False
    fd = drv._try_lock()
    # Same process, different fd: flock is per-open-file-description, so this
    # models a genuinely concurrent holder.
    assert drv.is_turn_active() is True
    drv._unlock(fd)
    assert drv.is_turn_active() is False


# ── 8. control commands ──────────────────────────────────────────────────


def test_clear_drops_the_thread_so_the_next_turn_starts_fresh(tmp_path):
    popen = FakePopen(
        {"lines": stream_ok("one")},
        {"lines": stream_ok("two", thread=THREAD_2)},
    )
    drv = make_driver(tmp_path, popen)
    drv.ask("first")
    drv.send_control("/clear")
    assert not (tmp_path / "rt" / "thread_id").exists()
    drv.ask("second")
    assert "resume" not in popen.argvs[1]
    assert (tmp_path / "rt" / "thread_id").read_text().strip() == THREAD_2


def test_model_control_sets_an_override_used_on_the_next_turn(tmp_path):
    popen = FakePopen({"lines": stream_ok()})
    drv = make_driver(tmp_path, popen)
    drv.send_control("/model gpt-5.6-sol")
    drv.ask("q")
    argv = popen.argvs[0]
    assert argv[argv.index("-m") + 1] == "gpt-5.6-sol"


def test_bare_model_and_unknown_controls_are_refused_not_faked(tmp_path, caplog):
    drv = make_driver(tmp_path, FakePopen())
    with caplog.at_level("WARNING"):
        drv.send_control("/model")
        drv.send_control("/compact")
    assert not (tmp_path / "rt" / "model").exists()
    text = caplog.text
    assert "interactive-only" in text
    assert "/compact" in text


def test_steering_is_refused_rather_than_silently_dropped(tmp_path, caplog):
    drv = make_driver(tmp_path, FakePopen())
    assert drv.is_steerable_turn() is False
    with caplog.at_level("ERROR"):
        drv.steer("допиши ещё вот это")
    # chat.py never reaches steer() while is_steerable_turn() is False, but if
    # anything does, the text must be recoverable from the journal.
    assert "допиши ещё вот это" in caplog.text
    assert "NOT delivered" in caplog.text


# ── 9. health / recovery ─────────────────────────────────────────────────


def test_is_working_needs_a_live_process_and_a_growing_stream(tmp_path):
    clock = Clock(step=1.0)
    drv = make_driver(tmp_path, FakePopen(), clock=clock, stall_timeout=3.0)
    assert drv.is_working() is False  # nothing running

    import d_brain.services.codex_driver as mod

    real_kill = mod.os.kill
    mod.os.kill = lambda pid, sig: None
    try:
        drv._atomic_write(drv._pid_file, "424242\n")
        drv._append_event_line(ev(type="turn.started"))
        assert drv.is_working() is True
        drv._append_event_line(ev(type="item.completed", item={}))
        assert drv.is_working() is True  # stream grew
        # Now freeze the stream and let the clock run past stall_timeout.
        assert drv.is_working() is True  # first frozen observation
        for _ in range(6):
            drv.is_working()
        assert drv.is_working() is False
    finally:
        mod.os.kill = real_kill


def test_force_recover_yields_to_a_live_turn(tmp_path):
    drv = make_driver(tmp_path, FakePopen())
    fd = drv._try_lock()
    try:
        assert drv.force_recover() is False
    finally:
        drv._unlock(fd)
    assert drv.force_recover() is True


def test_force_recover_keeps_the_thread(tmp_path):
    """Deliberately less destructive than the tmux engine's kill-session:
    the wedge is a process, the conversation is a durable thread id."""
    drv = make_driver(tmp_path, FakePopen({"lines": stream_ok()}))
    drv.ask("q")
    assert drv.force_recover() is True
    assert (tmp_path / "rt" / "thread_id").read_text().strip() == THREAD


def test_ensure_session_refuses_a_missing_persona_file(tmp_path):
    drv = make_driver(tmp_path, FakePopen(), instructions_file=tmp_path / "nope.md")
    with pytest.raises(RuntimeError, match="personality-less"):
        drv.ensure_session()


def test_runtime_dir_is_owner_only(tmp_path):
    drv = make_driver(tmp_path, FakePopen())
    assert (drv.runtime_dir.stat().st_mode & 0o777) == 0o700


# ── 10. capture_text must not trip Claude-shaped classifiers ─────────────


def test_capture_text_never_trips_tmux_classifiers(tmp_path):
    """cron_runner._limit_recovery feeds capture_text() straight into
    tmux_parse.classify_state, whose rate-limit regex matches Codex's own
    user-facing wording verbatim ("You've hit your usage limit."). Returning
    raw vendor prose would make the cron runner fire send_control("/clear")
    and silently throw the thread away."""
    lines = [
        ev(type="thread.started", thread_id=THREAD),
        ev(type="error", message="You've hit your usage limit. Upgrade at ..."),
        ev(type="turn.failed", error={"message": "You've hit your usage limit."}),
    ]
    drv = make_driver(tmp_path, FakePopen({"lines": lines, "returncode": 1}))
    assert drv.ask("q").status == "rate_limited"

    cap = drv.capture_text()
    assert "rate_limited" in cap  # our own vocabulary is present
    assert "usage limit" not in cap  # the vendor's prose is not
    assert classify_state(cap) is not PaneState.RATE_LIMITED
    assert classify_state(cap) is not PaneState.LOGGED_OUT
    # The raw stream is still on disk for forensics, just not on this surface.
    assert "usage limit" in (tmp_path / "rt" / "codex.log").read_text()


def test_current_state_is_always_ready(tmp_path):
    """READY keeps the watchdog on its idle branch (which delivers orphan
    replies). UNKNOWN would arm _is_hung — and its only recovery is
    force_recover — on a signal this engine cannot produce."""
    drv = make_driver(tmp_path, FakePopen())
    assert drv.current_state() is PaneState.READY
    import d_brain.services.codex_driver as mod

    real_kill = mod.os.kill
    mod.os.kill = lambda pid, sig: None
    try:
        drv._atomic_write(drv._pid_file, "424242\n")
        assert drv.current_state() is PaneState.READY
        assert drv.is_pane_turn_active() is True  # liveness lives here instead
    finally:
        mod.os.kill = real_kill


def test_nudge_is_honestly_false(tmp_path):
    assert make_driver(tmp_path, FakePopen()).nudge() is False


def test_last_reply_for_resend_reports_in_progress_under_a_live_lock(tmp_path):
    drv = make_driver(tmp_path, FakePopen({"lines": stream_ok("x")}))
    drv.ask("q")
    fd = drv._try_lock()
    try:
        assert drv.last_reply_for_resend() == ("in_progress", None)
    finally:
        drv._unlock(fd)
    assert drv.last_reply_for_resend() == ("ready", "x")


def test_state_files_do_not_leak_between_turns(tmp_path):
    """inflight and turn.pid are turn-scoped; a finished turn must leave
    neither behind, or is_steerable_turn/is_pane_turn_active would lie."""
    drv = make_driver(tmp_path, FakePopen({"lines": stream_ok()}))
    drv.ask("q")
    assert not (tmp_path / "rt" / "inflight").exists()
    assert not (tmp_path / "rt" / "turn.pid").exists()
    assert drv.is_turn_active() is False


def test_spawn_failure_is_an_error_result_not_an_exception(tmp_path):
    def boom(*_a, **_k):
        raise OSError("no such file")

    drv = make_driver(tmp_path, boom)
    res = drv.ask("q")
    assert res.status == "error"
    assert "could not launch codex" in (res.detail or "")


def test_ask_never_raises_on_a_garbage_stream(tmp_path):
    lines = ["not json at all\n", "{broken\n", ev(type="turn.completed", usage={})]
    drv = make_driver(tmp_path, FakePopen({"lines": lines}))
    res = drv.ask("q")
    assert isinstance(res, AskResult)
    assert res.status == "error"  # completed, but nothing to deliver


def test_smoke_script_exists_and_is_not_wired_into_the_bot():
    """The live check is a standalone script, deliberately not reachable
    from the bot's engine selection."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "codex-smoke.py"
    assert script.exists()
    body = script.read_text()
    assert "CodexExecDriver" in body
    assert "chat_engine" not in body and "cron_engine" not in body


def test_watchdog_long_run_cap_never_fires_on_this_engine(tmp_path):
    """'s auto-close is Claude-engine-only today — and, crucially,
    it can never FALSELY close a live Codex turn.

    The cap hangs off long-run tracking, which classifies ``capture_text()``
    with ``tmux_parse.is_main_turn_active`` — a Claude-pane classifier. This
    driver's ``capture_text()`` deliberately emits only its own vocabulary
    (see ``test_capture_text_never_trips_tmux_classifiers`` above and the
    method's docstring), so that classifier reads "no main turn active", no
    run is ever tracked, and neither the notice nor the interrupt fires.
    The accepted consequence, stated rather than papered over: with
    ``chat_engine=codex`` the hard cap is simply inactive.
    """
    from d_brain.services import long_run
    from d_brain.services.tmux_parse import is_main_turn_active
    from d_brain.services.watchdog import Watchdog

    drv = make_driver(tmp_path, FakePopen())
    assert drv.ask("q").status == "ok"
    assert is_main_turn_active(drv.capture_text()) is False

    rt = tmp_path / "rt"
    clock = {"now": 1000.0}
    alerts: list[str] = []
    wd = Watchdog(
        drv,
        runtime_dir=rt,
        disk_free_fn=lambda: 10_000_000_000,
        clock_fn=lambda: clock["now"],
        alert_fn=alerts.append,
        min_disk_bytes=500_000_000,
        long_run_alert_seconds=60.0,
        long_run_max_seconds=300.0,
    )
    for t in range(1000, 6000, 250):
        clock["now"] = float(t)
        wd._track_long_run()

    assert alerts == []
    assert long_run.read(rt).since == 0.0


def test_no_sleep_leaks_into_the_suite():
    """Guard against a future edit re-introducing a real time.sleep in the
    driver's hot loops — the whole suite must stay sub-second."""
    started = time.monotonic()
    assert time.monotonic() - started < 1.0
