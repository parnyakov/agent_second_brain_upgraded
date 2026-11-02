"""Reply ledger (2026-10-29): binding a reply to its request by CLI hooks.

FakeCLI below stands in for tmux AND the Claude Code CLI behind it: a prompt
pasted + Enter becomes a real ``scripts/turn_hook.py`` UserPromptSubmit (the
hook module itself runs, in-process), and scripted events fire Stop /
StopFailure / SessionStart through the same hook at chosen times on the
injected clock. Payload shapes are the ones captured live from Claude Code
2.1.312: a message typed into a running turn shares that turn's prompt_id;
an Escape fires no Stop; /compact keeps the session id.
"""

from __future__ import annotations

import asyncio
import importlib.util
import itertools
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from d_brain.services import (
    claude_session,
    reply_metrics,
    reply_pump,
    turn_ledger,
    turn_state,
)
from d_brain.services.claude_session import (
    LEDGER_ALREADY_SENT,
    LEDGER_EMPTY_REPLY,
    MAINT_PREFIX,
    ClaudeSession,
)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "turn_hook_mod", ROOT / "scripts" / "turn_hook.py"
)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

READY = (
    "────────────────────\n❯\n────────────────────\n"
    "  hello | Opus 4.8 (1M context) | ~/p\n"
    "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
)
THINKING = "  ✻ Working…  (esc to interrupt)\n"


class FakeCLI:
    """tmux + the CLI's hook behavior, on an injected clock."""

    def __init__(self, runtime_dir: Path, clock: dict, base: float) -> None:
        self.rt = runtime_dir
        self.clock = clock
        self.base = base
        self.sid = "test-sid"
        self.open_pid: str | None = None
        self.prompts: list[tuple[str, str]] = []
        self.calls: list[list[str]] = []
        self._buf = ""
        self._box: str | None = None
        self._events: list[tuple[float, int, object]] = []
        self._seq = itertools.count()
        self._pids = itertools.count(1)
        self.on_submit = None  # callable(cli, pid, text)
        self.hook_dead = False  # simulate a hook that crashed / timed out
        # Input-box rendering (off by default: the older tests never look).
        self.show_box = False
        self.hide_box_captures = 0  # the CLI draws the paste late
        self.lost_enters = 0  # Enters the CLI swallows while text is in the box
        self.mangle = None  # callable(text) -> text, applied on submit

    # time & hooks
    def wall(self) -> float:
        return self.base + self.clock["now"]

    def fire(self, payload: dict) -> None:
        if self.hook_dead:
            return
        hook.apply(str(self.rt), {"session_id": self.sid, **payload}, self.wall())

    def at(self, dt: float, fn) -> None:
        self._events.append((self.clock["now"] + dt, next(self._seq), fn))
        self._events.sort(key=lambda e: (e[0], e[1]))

    def _flush(self) -> None:
        while self._events and self._events[0][0] <= self.clock["now"]:
            _, _, fn = self._events.pop(0)
            fn()

    # CLI actions
    def submit(self, text: str) -> str:
        pid = self.open_pid or f"p{next(self._pids)}"
        self.open_pid = pid
        self.prompts.append((pid, text))
        self.fire(
            {"hook_event_name": "UserPromptSubmit", "prompt_id": pid, "prompt": text}
        )
        if self.on_submit is not None:
            self.on_submit(self, pid, text)
        return pid

    def stop(
        self, pid: str | None, text: str | None, *, bg: int = 0, prompt_id=True
    ) -> None:
        payload = {
            "hook_event_name": "Stop",
            "last_assistant_message": text,
            "background_tasks": [{"id": f"b{i}"} for i in range(bg)],
        }
        if prompt_id:
            payload["prompt_id"] = pid
        self.fire(payload)
        if self.open_pid == pid or not prompt_id:
            self.open_pid = None

    def fail(self, pid: str, error: str) -> None:
        self.fire({"hook_event_name": "StopFailure", "prompt_id": pid, "error": error})
        if self.open_pid == pid:
            self.open_pid = None

    def notify(self, text: str = "<task-notification>\n<task-id>t1</task-id>") -> str:
        return self.submit(text)

    # tmux surface
    def __call__(self, args, **kwargs):  # noqa: ANN001
        self.calls.append(args)
        body = args[1:]
        if body[:1] == ["-L"]:
            body = body[2:]
        sub = body[0] if body else ""
        out, rc = "", 0
        if sub == "has-session":
            rc = 0
        elif sub == "display-message":
            out = "200x50\n"
        elif sub == "load-buffer":
            self._buf = kwargs.get("input") or ""
        elif sub == "paste-buffer":
            self._box = self._buf
        elif sub == "send-keys":
            if body[-1] == "Enter" and self._box is not None:
                if self.lost_enters > 0:
                    self.lost_enters -= 1
                else:
                    text, self._box = self._box, None
                    if self.mangle is not None:
                        text = self.mangle(text)
                    self.submit(text)
            elif body[-1] in ("C-k", "C-u"):
                self._box = None
            elif body[-1] == "Escape":
                self.open_pid = None  # interrupted: no Stop hook
        elif sub == "capture-pane":
            self._flush()
            out = THINKING if self.open_pid else READY
            if self.show_box and self._box is not None:
                if self.hide_box_captures > 0:
                    self.hide_box_captures -= 1
                else:
                    rule = "────────────────────"
                    rows = "\n  ".join(self._box.splitlines())
                    out = (
                        f"{rule}\n❯ {rows}\n{rule}\n"
                        "  ⏵⏵ bypass permissions on (shift+tab to cycle)\n"
                    )
        return subprocess.CompletedProcess(args, rc, stdout=out, stderr="")


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # The hook writes the test runner's pid as cli_pid; the steer gate only
    # trusts the state file for a live `claude` process.
    monkeypatch.setattr(turn_state, "_is_claude_proc", lambda pid: True)


@pytest.fixture
def clock():
    return {"now": 0.0}


def make(
    tmp_path: Path,
    clock: dict,
    *,
    name: str = ".dbrain",
    ledger: bool = True,
    stall_timeout: float = 60.0,
    grace: float = 3.0,
):
    rt = tmp_path / name
    rt.mkdir(parents=True, exist_ok=True)
    (rt / "session_id").write_text("test-sid\n")
    cli = FakeCLI(rt, clock, base=time.time())
    rids = (f"r{i:04d}" for i in itertools.count(1))

    def sleep(sec: float) -> None:
        clock["now"] += sec

    s = ClaudeSession(
        session_name=f"dbrain_{name.strip('.') or 'x'}",
        work_dir=tmp_path / "vault",
        runtime_dir=rt,
        runner=cli,
        which_fn=lambda n: "/opt/bin/claude",
        sleep_fn=sleep,
        clock_fn=lambda: clock["now"],
        rid_factory=lambda: next(rids),
        poll_interval=1.0,
        submit_settle=0.0,
        paste_settle=0.0,
        stall_timeout=stall_timeout,
        reply_ledger=ledger,
        ledger_stop_grace=grace,
    )
    return s, cli


def reply_after(dt: float, text: str):
    """on_submit: every NEW turn ends dt seconds later with ``text``."""

    def cb(cli: FakeCLI, pid: str, prompt: str) -> None:
        if sum(1 for p, _ in cli.prompts if p == pid) == 1:
            cli.at(dt, lambda: cli.stop(pid, text))

    return cb


def idle(s: ClaudeSession, clock: dict, seconds: int) -> None:
    """Let the fake CLI run while the bot only watches the pane."""
    for _ in range(seconds):
        s.capture_text()
        clock["now"] += 1


def make_pump(rt: Path, cli: FakeCLI, sent: list[str], **kw):
    async def deliver(text: str) -> None:
        sent.append(text)

    return reply_pump.ReplyPump(rt, deliver, clock=cli.wall, **kw)


def pump_once(pump) -> int:
    return asyncio.run(pump.step())


def claims(rt: Path) -> list[dict]:
    return turn_ledger.Claims(rt).records()


# ── the bound turn ──────────────────────────────────────────────────────


def test_bound_turn_returns_the_final_message(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    cli.on_submit = reply_after(3, "<b>Готово</b>: файл записан.")
    res = s.ask("запиши файл", timeout=60)
    assert res.ok and res.reply == "<b>Готово</b>: файл записан."
    pid, text = cli.prompts[0]
    assert text.endswith('<dbrain-turn rid="r0001" kind="ask"/>')
    assert "<<<R:" not in text and "wrap your ENTIRE reply" not in text
    (c,) = [c for c in claims(s.runtime_dir) if c.get("state") == "delivered"]
    assert c["prompt_id"] == pid and c["by"] == "ask" and c["rid"] == "r0001"
    assert not (s.runtime_dir / "inflight").exists()


def test_maint_ask_is_tagged_maint(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    cli.on_submit = reply_after(1, "ok")
    assert s.ask("pipeline", request_id=f"{MAINT_PREFIX}process", timeout=30).ok
    assert cli.prompts[0][1].endswith('kind="maint"/>')


def test_reply_with_old_marker_strings_is_delivered_clean(tmp_path, clock):
    """The incident shape: the model echoes an OLD rid. Nothing depends on
    it any more; stray marker lines are stripped from what is sent."""
    s, cli = make(tmp_path, clock)
    cli.on_submit = reply_after(
        2, "<<<R:deadbeef>>>\nОтвет на вопрос.\nСм. <<<E:deadbeef>>>\n<<<E:deadbeef>>>"
    )
    res = s.ask("вопрос", timeout=60)
    assert res.ok and res.reply == "Ответ на вопрос.\nСм."


def test_echoed_tag_in_reply_is_stripped(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    cli.on_submit = reply_after(2, 'Ок <dbrain-turn rid="r0001" kind="ask"/>')
    assert s.ask("x", timeout=30).reply == "Ок"


def test_wrong_turn_stop_is_never_taken(tmp_path, clock):
    """A task-notification turn that ends while our prompt waits in the CLI
    queue is someone else's turn: ask() waits for the Stop of the turn that
    carries ITS rid; the other one goes to the pump."""
    s, cli = make(tmp_path, clock)

    def cb(c, pid, prompt):
        if 'kind="ask"' in prompt:
            # A notification turn of its own (another prompt_id) submits and
            # ends while ours is still running; then ours ends.
            def other():
                c.fire(
                    {
                        "hook_event_name": "UserPromptSubmit",
                        "prompt_id": "p-bg",
                        "prompt": "<task-notification>\n<task-id>t2</task-id>",
                    }
                )
                c.fire(
                    {
                        "hook_event_name": "Stop",
                        "prompt_id": "p-bg",
                        "last_assistant_message": "Фоновая задача закончилась.",
                    }
                )

            c.at(1, other)
            c.at(3, lambda: c.stop(pid, "Наш ответ"))

    cli.on_submit = cb
    res = s.ask("наш вопрос", timeout=60)
    assert res.ok and res.reply == "Наш ответ"
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent, max_age=1e9)
    pump._since = 0  # this test predates the pump's first start on purpose
    clock["now"] += 30
    pump_once(pump)
    assert sent == ["Фоновая задача закончилась."]


def test_missing_rid_unbound_stop_does_not_satisfy_ask(tmp_path, clock):
    """A Stop for an untagged turn (human typed in attach) while our ask
    waits: not ours. Our own Stop then arrives and is the reply."""
    s, cli = make(tmp_path, clock)

    def cb(c, pid, prompt):
        if 'kind="ask"' in prompt:
            c.at(
                1,
                lambda: c.fire(
                    {
                        "hook_event_name": "Stop",
                        "prompt_id": "untagged-1",
                        "last_assistant_message": "чужое",
                    }
                ),
            )
            c.at(3, lambda: c.stop(pid, "моё"))

    cli.on_submit = cb
    assert s.ask("вопрос", timeout=30).reply == "моё"


def test_stop_without_prompt_id_binds_positionally(tmp_path, clock):
    """An older CLI without prompt_id: the stop closes the newest open turn."""
    s, cli = make(tmp_path, clock)

    def cb(c, pid, prompt):
        c.at(2, lambda: c.stop(pid, "позиционно", prompt_id=False))

    def submit_no_pid(text):
        cli.prompts.append(("?", text))
        cli.open_pid = "?"
        cli.fire({"hook_event_name": "UserPromptSubmit", "prompt": text})
        cb(cli, "?", text)
        return "?"

    cli.submit = submit_no_pid
    assert s.ask("старый CLI", timeout=30).reply == "позиционно"


def test_empty_final_message_is_silent_by_default(tmp_path, clock):
    """Default (empty_reply_notice off): an empty turn is like a bare
    [SILENT]: ok, silent, no text, and logged as "empty" in the claims."""
    s, cli = make(tmp_path, clock)
    cli.on_submit = reply_after(2, "")
    res = s.ask("сделай молча", timeout=30)
    assert res.ok and res.reply == "" and res.silent
    states = [c.get("state") for c in claims(s.runtime_dir) if c.get("by") == "ask"]
    assert "empty" in states and "delivered" not in states


def test_bare_sentinel_on_a_bound_turn_is_silent_by_default(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    cli.on_submit = reply_after(2, "[SILENT]")
    res = s.ask("x", timeout=30)
    assert res.ok and res.reply == "" and res.silent
    states = [c.get("state") for c in claims(s.runtime_dir) if c.get("by") == "ask"]
    assert "sentinel" in states


def test_empty_final_message_gets_the_notice_when_enabled(tmp_path, clock):
    """empty_reply_notice on (an operator instance): the visible notice,
    not silent, so it reaches the chat as before."""
    s, cli = make(tmp_path, clock)
    s.empty_reply_notice = True
    cli.on_submit = reply_after(2, "")
    res = s.ask("сделай молча", timeout=30)
    assert res.ok and res.reply == LEDGER_EMPTY_REPLY and not res.silent


def test_sentinel_on_a_bound_turn_is_stripped(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    cli.on_submit = reply_after(2, "[SILENT] записал")
    res = s.ask("x", timeout=30)
    assert res.reply == "записал" and res.silent


def test_plain_bound_reply_is_not_silent(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    cli.on_submit = reply_after(2, "записал")
    assert not s.ask("x", timeout=30).silent


# ── steering ────────────────────────────────────────────────────────────


def test_steer_mid_turn_joins_and_one_final_message_answers_both(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []

    def cb(c, pid, prompt):
        if 'kind="ask"' in prompt:
            c.at(2, lambda: s.steer("и ещё: сколько будет 2+2?"))
            c.at(5, lambda: c.stop(pid, "Файл готов. 2+2 = 4."))

    cli.on_submit = cb
    res = s.ask("сделай файл", timeout=60)
    assert res.reply == "Файл готов. 2+2 = 4."
    pids = {p for p, _ in cli.prompts}
    assert len(pids) == 1  # the steer joined the running turn
    assert 'kind="steer"' in cli.prompts[1][1]
    pump = make_pump(s.runtime_dir, cli, sent)
    clock["now"] += 60
    pump_once(pump)
    assert sent == []  # no second copy


def test_several_steers_in_a_row_into_an_unbound_turn(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    note = cli.notify()
    assert s.is_steerable_turn()  # open notification turn, lock free
    s.steer("первое")
    s.steer("второе")
    s.steer("третье")
    cli.stop(note, "Ответ на всё сразу.")
    clock["now"] += 5
    pump_once(pump)
    assert sent == ["Ответ на всё сразу."]
    book = turn_ledger.TurnBook()
    book.extend(turn_ledger.read_all(s.runtime_dir))
    (turn,) = book.ordered()
    assert turn.kinds == ["steer", "steer", "steer"]


def test_steer_gate_follows_the_cli_turn_not_the_lock(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    assert not s.is_steerable_turn()  # nothing open
    p = cli.submit('x <dbrain-turn rid="a" kind="ask"/>')
    assert s.is_steerable_turn()  # open chat turn, lock free (ask gave up)
    cli.stop(p, "done")
    assert not s.is_steerable_turn()  # closed → it becomes a new request
    m = cli.submit('<dbrain-turn rid="m" kind="maint"/>')
    assert not s.is_steerable_turn()  # pipeline turn takes no user text
    cli.stop(m, "ok")
    # lock held by an ask still readying the pane (maint- placeholder)
    cli.submit('y <dbrain-turn rid="b" kind="ask"/>')
    (s.runtime_dir / "inflight").write_text(f"{MAINT_PREFIX}pending-x\n0\n")
    with s._locked():
        assert not s.is_steerable_turn()


def test_steer_gate_without_hook_state_falls_back_to_the_lock(tmp_path, clock):
    s, _ = make(tmp_path, clock)
    assert not s.is_steerable_turn()
    (s.runtime_dir / "inflight").write_text("chat-1\n0\n")
    with s._locked():
        assert s.is_steerable_turn()


def test_steer_into_a_turn_that_just_closed_is_delivered_by_the_pump(tmp_path, clock):
    """The race: the gate said open, the turn closed before the text landed.
    The text opens its own turn; nobody's ask() waits; the pump delivers."""
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    p = cli.notify()
    cli.stop(p, "[SILENT] агент закончил")
    cli.on_submit = reply_after(2, "Ответ на позднее сообщение")
    s.steer("а что с отчётом?")
    idle(s, clock, 10)
    pump_once(pump)
    assert sent == ["Ответ на позднее сообщение"]
    states = turn_ledger.Claims(s.runtime_dir).states()
    assert {c["state"] for c in states.values()} == {"sentinel", "delivered"}


# ── task notifications & the sentinel ───────────────────────────────────


def test_task_notification_while_idle_is_delivered(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    p = cli.notify()
    clock["now"] += 3
    cli.stop(p, "Рендер готов: <a href='https://x'>ссылка</a>")
    clock["now"] += 0.5
    assert pump_once(pump) == 0  # inside the unbound grace
    clock["now"] += 1
    assert pump_once(pump) == 1
    assert pump_once(pump) == 0  # deduped by prompt_id
    assert sent == ["Рендер готов: <a href='https://x'>ссылка</a>"]


def test_sentinel_keeps_a_background_turn_quiet(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    p = cli.notify()
    cli.stop(p, "  [SILENT] субагент закончил, всё записано")
    clock["now"] += 5
    pump_once(pump)
    assert sent == []
    (c,) = turn_ledger.Claims(s.runtime_dir).states().values()
    assert c["state"] == "sentinel" and c["by"] == "pump"


def test_task_notification_merged_into_a_bound_turn_goes_to_the_ask(tmp_path, clock):
    s, cli = make(tmp_path, clock)

    def cb(c, pid, prompt):
        if 'kind="ask"' in prompt:
            c.at(2, lambda: c.notify())  # joins the running turn (same pid)
            c.at(4, lambda: c.stop(pid, "Ответ + итог фоновой задачи"))

    cli.on_submit = cb
    assert s.ask("вопрос", timeout=30).reply == "Ответ + итог фоновой задачи"


def test_task_notification_after_a_bound_turn_is_its_own_delivery(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []

    def cb(c, pid, prompt):
        if 'kind="ask"' in prompt:

            def end():
                c.stop(pid, "Запустил агента, вернусь с итогом.", bg=1)
                c.at(5, lambda: c.notify())
                c.at(8, lambda: c.stop(c.open_pid, "Агент закончил: 3 файла."))

            c.at(2, end)

    cli.on_submit = cb
    assert s.ask("запусти агента", timeout=30).reply.startswith("Запустил агента")
    pump = make_pump(s.runtime_dir, cli, sent)
    for _ in range(12):  # the bot keeps polling the pane meanwhile
        s.capture_text()
        clock["now"] += 1
    pump_once(pump)
    assert sent == ["Агент закончил: 3 файла."]


# ── compaction, interrupts, failures ────────────────────────────────────


def test_auto_compaction_and_session_id_change_mid_turn(tmp_path, clock):
    s, cli = make(tmp_path, clock)

    def cb(c, pid, prompt):
        def compact():
            c.sid = "sid-after"
            (s.runtime_dir / "session_id").write_text("sid-after\n")
            c.fire(
                {
                    "hook_event_name": "SessionStart",
                    "source": "compact",
                    "prompt_id": pid,
                }
            )

        c.at(2, compact)
        c.at(4, lambda: c.stop(pid, "после компакта"))

    cli.on_submit = cb
    assert s.ask("долгое", timeout=30).reply == "после компакта"
    st = turn_state.read(s.runtime_dir)
    assert st.session_id == "sid-after" and st.state == "closed"


def test_compaction_keeps_the_turn_open_for_the_steer_gate(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    p = cli.submit('x <dbrain-turn rid="a" kind="ask"/>')
    cli.fire({"hook_event_name": "SessionStart", "source": "compact", "prompt_id": p})
    assert s.is_steerable_turn()


def test_interrupted_turn_ends_the_ask_without_a_stop(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []

    def cb(c, pid, prompt):
        c.at(2, s.interrupt)  # /stop from the chat: Escape, no Stop hook

    cli.on_submit = cb
    res = s.ask("долго", timeout=60)
    assert res.status == "error" and res.detail == "turn interrupted"
    pump = make_pump(s.runtime_dir, cli, sent)
    clock["now"] += 60
    pump_once(pump)
    assert sent == []
    outcome = [c for c in claims(s.runtime_dir) if c.get("state") == "ask_outcome"]
    assert outcome and outcome[-1]["status"] == "error"


@pytest.mark.parametrize(
    "error,status", [("rate_limit_error", "rate_limited"), ("overloaded", "error")]
)
def test_stop_failure_ends_the_ask(tmp_path, clock, error, status):
    s, cli = make(tmp_path, clock)
    cli.on_submit = lambda c, pid, p: c.at(2, lambda: c.fail(pid, error))
    res = s.ask("x", timeout=30)
    assert res.status == status and error in res.detail


def test_stop_failure_on_a_steered_unbound_turn_gets_an_honest_notice(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    p = cli.submit('вопрос <dbrain-turn rid="s9" kind="steer"/>')
    cli.fail(p, "overloaded")
    clock["now"] += 5
    pump_once(pump)
    assert sent == [reply_pump.FAILED_NOTICE]


def test_hook_stop_missing_falls_back_to_the_transcript(tmp_path, clock):
    """The Stop hook crashed or hit its 5 s timeout: no stop line. The
    transcript closes the turn (system/turn_duration); after the grace the
    final message is taken from there."""
    s, cli = make(tmp_path, clock, grace=3.0)
    path = s.current_transcript_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    def cb(c, pid, prompt):
        def finish():
            recs = [
                {
                    "type": "user",
                    "isSidechain": False,
                    "promptId": pid,
                    "message": {"role": "user", "content": prompt},
                },
                {
                    "type": "assistant",
                    "isSidechain": False,
                    "message": {"content": [{"type": "text", "text": "промежуточное"}]},
                },
                {
                    "type": "assistant",
                    "isSidechain": True,
                    "message": {"content": [{"type": "text", "text": "субагент"}]},
                },
                {
                    "type": "assistant",
                    "isSidechain": False,
                    "message": {
                        "content": [{"type": "text", "text": "Итоговый ответ"}]
                    },
                },
                {"type": "system", "subtype": "turn_duration", "isSidechain": False},
            ]
            with path.open("a") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            c.open_pid = None  # the turn is over; only the hook line is missing

        c.at(2, finish)

    cli.on_submit = cb
    cli.hook_dead = False

    real_fire = cli.fire

    def fire_no_stop(payload):
        if payload.get("hook_event_name") != "Stop":
            real_fire(payload)

    cli.fire = fire_no_stop
    res = s.ask("вопрос", timeout=60)
    assert res.ok and res.reply == "Итоговый ответ"
    (c,) = [c for c in claims(s.runtime_dir) if c.get("state") == "delivered"]
    assert c["source"] == "transcript"


def test_older_cli_stop_without_text_reads_the_transcript(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    path = s.current_transcript_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    def cb(c, pid, prompt):
        def finish():
            with path.open("a") as f:
                for r in (
                    {"type": "user", "promptId": pid, "message": {"content": prompt}},
                    {"type": "assistant", "message": {"content": "из стенограммы"}},
                    {"type": "system", "subtype": "turn_duration"},
                ):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            c.fire(
                {
                    "hook_event_name": "Stop",
                    "prompt_id": pid,
                    "transcript_path": str(path),
                }
            )
            c.open_pid = None

        c.at(2, finish)

    cli.on_submit = cb
    assert s.ask("x", timeout=30).reply == "из стенограммы"


def test_hook_crash_or_garbage_never_breaks_the_cli(tmp_path):
    r = subprocess.run(
        ["python3", str(ROOT / "scripts" / "turn_hook.py"), str(tmp_path)],
        input="\x00garbage",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r.returncode == 0 and r.stdout == ""


def test_no_hooks_at_all_times_out_and_says_why(tmp_path, clock, caplog):
    s, cli = make(tmp_path, clock)
    cli.hook_dead = True
    res = s.ask("x", timeout=8)
    assert res.status == "timeout"
    assert "no UserPromptSubmit hook ever reported" in caplog.text
    assert (s.runtime_dir / "inflight").exists()  # the turn may still run


def test_duplicate_stop_events_deliver_once(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []

    def cb(c, pid, prompt):
        c.at(2, lambda: c.stop(pid, "один раз"))
        c.at(
            2.2,
            lambda: c.fire(
                {
                    "hook_event_name": "Stop",
                    "prompt_id": pid,
                    "last_assistant_message": "один раз",
                }
            ),
        )

    cli.on_submit = cb
    assert s.ask("x", timeout=30).reply == "один раз"
    clock["now"] += 60
    pump = make_pump(s.runtime_dir, cli, sent)
    pump_once(pump)
    pump_once(pump)
    assert sent == []


def test_ask_that_loses_the_claim_never_sends_a_second_copy(tmp_path, clock):
    s, cli = make(tmp_path, clock)

    def cb(c, pid, prompt):
        def stop_and_pump_first():
            c.stop(pid, "ответ")
            turn_ledger.Claims(s.runtime_dir).claim(pid, by="pump", state="delivered")

        c.at(2, stop_and_pump_first)

    cli.on_submit = cb
    res = s.ask("x", timeout=30)
    assert res.ok and res.reply == LEDGER_ALREADY_SENT


# ── long replies, restarts, late replies ────────────────────────────────


class _Bot:
    def __init__(self):
        self.sent: list[str] = []

    async def send_message(self, chat_id, text, **kw):
        self.sent.append(text)
        return SimpleNamespace(message_id=len(self.sent))


def test_long_unsolicited_reply_is_chunked_like_any_reply(tmp_path, clock, monkeypatch):
    from d_brain.bot import formatters
    from d_brain.services import outbox

    monkeypatch.setattr(outbox, "SEND_SPACING", 0)
    outbox.reset()
    s, cli = make(tmp_path, clock)
    bot = _Bot()

    async def deliver(text):
        await formatters.send_response(bot, 42, text)

    pump = reply_pump.ReplyPump(s.runtime_dir, deliver, clock=cli.wall)
    body = ("<b>Раздел</b>\n" + "строка отчёта " * 40 + "\n\n") * 20
    assert len(body) > 4096
    p = cli.notify()
    cli.stop(p, body)
    clock["now"] += 5
    pump_once(pump)
    assert len(bot.sent) >= 2 and all(len(x) <= 4096 for x in bot.sent)
    assert "".join(bot.sent).count("строка отчёта") == body.count("строка отчёта")


def test_long_bound_reply_is_returned_whole(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    body = "абзац " * 2000
    cli.on_submit = reply_after(2, body)
    assert s.ask("x", timeout=30).reply == body.strip()


def test_bot_restart_mid_turn_reply_arrives_via_the_pump(tmp_path, clock):
    """The bot dies while its ask() waits; the CLI finishes the turn while
    the bot is down; the new process's pump delivers it (late, marked)."""
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    make_pump(s.runtime_dir, cli, sent)  # the pump ran before: since is set
    clock["now"] += 1
    p = cli.submit('вопрос <dbrain-turn rid="r1" kind="ask"/>')  # ask() then died
    clock["now"] += 30
    cli.stop(p, "Ответ пришёл, пока бот перезапускался.")
    clock["now"] += 5
    fresh = make_pump(s.runtime_dir, cli, sent)  # new process
    assert pump_once(fresh) == 0  # bound: the grace lets a live ask() win
    clock["now"] += reply_pump.BOUND_GRACE
    assert pump_once(fresh) == 1
    assert sent == [reply_pump.LATE_PREFIX + "Ответ пришёл, пока бот перезапускался."]


def test_pump_crash_between_claim_and_send_resends_once(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    boom = {"n": 0}

    async def flaky(text):
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError("process died mid-send")
        sent.append(text)

    pump = reply_pump.ReplyPump(s.runtime_dir, flaky, clock=cli.wall)
    p = cli.notify()
    cli.stop(p, "итог")
    clock["now"] += 5
    pump_once(pump)
    assert sent == []
    restarted = reply_pump.ReplyPump(s.runtime_dir, flaky, clock=cli.wall)
    pump_once(restarted)
    pump_once(restarted)
    assert sent == ["итог"]


def test_ask_timeout_then_the_late_reply_is_delivered_once(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    cli.on_submit = reply_after(40, "Поздний ответ")
    res = s.ask("долгий вопрос", timeout=20)
    assert res.status == "timeout" and "no reply in" in res.detail
    for _ in range(25):
        s.capture_text()
        clock["now"] += 1
    clock["now"] += reply_pump.BOUND_GRACE
    pump_once(pump)
    pump_once(pump)
    assert sent == [reply_pump.LATE_PREFIX + "Поздний ответ"]


def test_pump_ignores_history_from_before_it_first_ran(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    p = cli.notify()
    cli.stop(p, "старьё из маркерной эпохи")
    clock["now"] += 10
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    pump_once(pump)
    assert sent == []
    since = (s.runtime_dir / reply_pump.SINCE_FILE).read_text()
    clock["now"] += 100
    make_pump(s.runtime_dir, cli, sent)
    assert (s.runtime_dir / reply_pump.SINCE_FILE).read_text() == since


# ── isolation: main / cron / duty ───────────────────────────────────────


def test_cron_and_duty_ledgers_are_isolated_from_main(tmp_path, clock):
    main, mcli = make(tmp_path, clock, name=".dbrain")
    cron, ccli = make(tmp_path, clock, name=".dbrain/cron")
    duty, dcli = make(tmp_path, clock, name=".dbrain/duty")
    # Hook settings of each session point at its OWN runtime dir.
    for sess in (main, cron, duty):
        cmd = sess._start_command("sid")
        cfg = json.loads((sess.runtime_dir / "turn-hooks.json").read_text())
        hook_cmd = cfg["hooks"]["Stop"][0]["hooks"][0]["command"]
        assert hook_cmd.endswith(str(sess.runtime_dir.resolve()))
        assert "--settings" in cmd

    def cron_busy(c, pid, prompt):
        c.at(1, lambda: c.stop(pid, "cron-ответ"))

    ccli.on_submit = cron_busy
    assert cron.ask("cron job", timeout=30).reply == "cron-ответ"

    def main_cb(c, pid, prompt):
        # A duty turn and a cron turn end while main waits: not main's.
        c.at(1, lambda: dcli.stop(dcli.notify(), "duty-фон"))
        c.at(2, lambda: ccli.stop(ccli.notify(), "cron-фон"))
        c.at(3, lambda: c.stop(pid, "main-ответ"))

    mcli.on_submit = main_cb
    assert main.ask("вопрос", timeout=30).reply == "main-ответ"
    sent: list[str] = []
    pump = make_pump(main.runtime_dir, mcli, sent)
    clock["now"] += 60
    pump_once(pump)
    assert sent == []  # the main pump never relays cron/duty turns


# ── switch off = marker path untouched ──────────────────────────────────


def test_switch_off_keeps_the_marker_contract(tmp_path, clock):
    s, cli = make(tmp_path, clock, ledger=False)
    assert s.reply_ledger is False
    s.ask("x", timeout=3)
    text = cli.prompts[0][1]
    assert "wrap your ENTIRE reply between a line containing only <<<R:r0001>>>" in text
    assert "<dbrain-turn" not in text


def test_persona_carries_the_marker_contract_only_while_switched_off(tmp_path, clock):
    persona = ROOT / "deploy" / "brain-system.md"
    s_old, _ = make(tmp_path, clock, name="old", ledger=False)
    s_new, _ = make(tmp_path, clock, name="new", ledger=True)
    s_old.system_prompt_file = persona
    s_new.system_prompt_file = persona
    eff = s_old._effective_system_prompt()
    assert eff == persona.with_name("brain-system-markers.md")
    assert str(eff) in s_old._start_command("sid")
    # The marker path is told exactly what it was told before the ledger:
    # byte-for-byte the persona that was on this branch (skipped when the
    # commit is not in the history, e.g. in a fresh clone).
    before = subprocess.run(
        ["git", "-C", str(ROOT), "show", "ad4dcfd:deploy/brain-system.md"],
        capture_output=True,
        text=True,
    )
    if before.returncode == 0:
        assert eff.read_text() == before.stdout
    assert "The final message of your turn IS the reply" not in eff.read_text()
    assert s_new._effective_system_prompt() == persona
    assert str(persona) in s_new._start_command("sid")


def test_orphan_scraping_is_off_in_ledger_mode(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    assert s.pop_orphan_replies() == []
    assert not any("capture-pane" in c for c in cli.calls)


def test_resend_reads_the_ledger(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    assert s.last_reply_for_resend() == ("empty", None)
    p = cli.notify()
    assert s.last_reply_for_resend() == ("in_progress", None)
    cli.stop(p, "<<<R:x>>>\nпоследний ответ\n<<<E:x>>>")
    assert s.last_reply_for_resend() == ("ready", "последний ответ")
    p2 = cli.notify()
    cli.stop(p2, "[SILENT] тихо")
    assert s.last_reply_for_resend() == ("ready", "последний ответ")


# ── the chat-side gate ──────────────────────────────────────────────────


def test_chat_gate_reads_the_ledger_only_when_it_is_on():
    from unittest.mock import MagicMock

    from d_brain.bot.handlers.chat import _cli_turn_steerable

    assert _cli_turn_steerable(MagicMock()) is False  # duck-typed ⇒ marker path
    on = SimpleNamespace(reply_ledger=True, is_steerable_turn=lambda: True)
    off = SimpleNamespace(reply_ledger=False, is_steerable_turn=lambda: True)
    assert _cli_turn_steerable(on) is True
    assert _cli_turn_steerable(off) is False


# ── the metric ──────────────────────────────────────────────────────────


def test_metric_counts_delivered_sentinel_and_lost(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    t0 = cli.wall() - 1
    cli.on_submit = reply_after(2, "ок")
    assert s.ask("1", timeout=30).ok  # bound, delivered by ask
    cli.on_submit = None
    n = cli.notify()
    cli.stop(n, "[SILENT] нечего")  # sentinel, does not need a reply
    n2 = cli.notify()
    cli.stop(n2, "фоновый итог")  # unbound, delivered by pump
    clock["now"] += 3
    pump_once(pump)
    lost = cli.submit('x <dbrain-turn rid="zz" kind="steer"/>')  # nobody delivers
    cli.stop(lost, "потерян")
    turn_ledger.Claims(s.runtime_dir).claim(lost, by="test", state="failed")
    clock["now"] += 300
    rep = reply_metrics.compute(s.runtime_dir, since=t0, now=cli.wall())
    counts = rep.counts()
    assert counts["bound"] == {"delivered": 1}
    assert counts["notify"] == {"sentinel": 1, "delivered": 1}
    assert counts["steer"] == {"failed": 1}
    assert rep.needs_reply == 3 and rep.not_delivered == 1
    assert "33.3%" in reply_metrics.render(rep)
    assert rep.latencies()["p50"] >= 0


def test_metric_measures_the_marker_path_too(tmp_path, clock):
    """Switch off: the hook binds the marker rid, ask() logs its outcome."""
    s, cli = make(tmp_path, clock, ledger=False)
    t0 = cli.wall() - 1
    s.ask("x", timeout=3)  # no markers ever appear → timeout
    clock["now"] += 4 * 3600
    rep = reply_metrics.compute(s.runtime_dir, since=t0, now=cli.wall())
    assert rep.counts() == {"bound": {"lost": 1}}
    assert rep.rate == 1.0


def test_metric_script_prints_a_rate(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    cli.on_submit = reply_after(1, "ок")
    s.ask("x", timeout=30)
    r = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(ROOT / "scripts" / "reply_metrics.py"),
            "--runtime-dir",
            str(s.runtime_dir),
            "--since",
            "1h",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["needs_reply"] == 1 and data["not_delivered"] == 0


def test_parse_window():
    assert reply_metrics.parse_window("90m") == 5400
    assert reply_metrics.parse_window("2d") == 172800


# ── review round 1 (2026-10-29) ─────────────────────────────────────────


def test_interrupted_turn_is_claimed_once_not_every_pass(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    cli.notify()
    s.interrupt()
    for _ in range(30):
        clock["now"] += 2
        pump_once(pump)
    lines = [c for c in claims(s.runtime_dir) if c.get("state") == "interrupted"]
    assert len(lines) == 1 and sent == []


@pytest.mark.parametrize("notice", [False, True])
def test_silent_sentinel_never_swallows_a_steered_message(tmp_path, clock, notice):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent, empty_notice=notice)
    p = cli.notify()
    s.steer("а что с отчётом?")
    cli.stop(p, "[SILENT] Отчёт готов, лежит в daily.")
    q = cli.notify()
    s.steer("ещё вопрос")
    cli.stop(q, "[SILENT]")
    clock["now"] += 5
    pump_once(pump)
    if notice:
        assert sent == ["Отчёт готов, лежит в daily.", turn_ledger.EMPTY_REPLY]
    else:
        # No text for the second message: logged as an empty turn, not sent.
        assert sent == ["Отчёт готов, лежит в daily."]
        (c,) = [c for c in claims(s.runtime_dir) if c.get("prompt_id") == q]
        assert c["state"] == "empty"


def test_stop_whose_submit_was_never_logged_is_still_delivered(tmp_path, clock):
    """The submit hook failed (crash/timeout) but the Stop landed."""
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    cli.fire(
        {
            "hook_event_name": "Stop",
            "prompt_id": "p-nosubmit",
            "last_assistant_message": "без submit",
        }
    )
    clock["now"] += 5
    pump_once(pump)
    assert sent == ["без submit"]


def test_input_after_escape_under_the_same_prompt_id_reopens_the_turn():
    book = turn_ledger.TurnBook()
    book.extend(
        [
            {"ev": "submit", "ts": 1, "prompt_id": "p", "rid": "a", "kind": "ask"},
            {"ev": "interrupt", "ts": 2, "prompt_id": "p"},
            {"ev": "submit", "ts": 3, "prompt_id": "p", "rid": "b", "kind": "ask"},
        ]
    )
    (turn,) = book.ordered()
    assert not turn.closed
    book.extend([{"ev": "stop", "ts": 4, "prompt_id": "p", "text": "ok"}])
    assert turn.stop["text"] == "ok"


def test_timed_out_maintenance_turn_is_not_sent_to_the_owner(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    p = cli.submit('canary <dbrain-turn rid="m1" kind="maint"/>')
    cli.stop(p, "PONG")
    clock["now"] += reply_pump.BOUND_GRACE + 5
    pump_once(pump)
    assert sent == []
    assert turn_ledger.Claims(s.runtime_dir).states()[p]["state"] == "skipped"


def test_restart_reads_a_turn_that_spans_the_ledger_rotation(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    sent: list[str] = []
    make_pump(s.runtime_dir, cli, sent)  # since is set before the turn
    clock["now"] += 1
    p = cli.notify()
    led = s.runtime_dir / turn_ledger.LEDGER_FILE
    led.rename(led.with_name(led.name + ".1"))  # the hook rotated here
    cli.stop(p, "через ротацию")
    clock["now"] += 5
    fresh = make_pump(s.runtime_dir, cli, sent)
    pump_once(fresh)
    assert sent == ["через ротацию"]


def test_metric_counts_asks_that_never_reached_the_cli(tmp_path, clock):
    s, cli = make(tmp_path, clock)
    t0 = cli.wall() - 1
    cli.hook_dead = True
    assert s.ask("x", timeout=5).status == "timeout"
    c = turn_ledger.Claims(s.runtime_dir)
    c.record(state="ask_outcome", rid="busy1", status="busy_active")  # parked
    rep = reply_metrics.compute(s.runtime_dir, since=t0, now=time.time() + 10)
    assert rep.asks_never_seen == 1 and rep.not_delivered == 1


def test_metric_counts_a_stall_interrupt_as_failed(tmp_path, clock):
    s, cli = make(tmp_path, clock, stall_timeout=5)
    t0 = cli.wall() - 1
    real = FakeCLI.__call__

    def frozen(self, args, **kw):
        res = real(self, args, **kw)
        if "capture-pane" in args:
            res = subprocess.CompletedProcess(args, 0, stdout=READY, stderr="")
        return res

    cli.__class__ = type("Frozen", (FakeCLI,), {"__call__": frozen})
    res = s.ask("x", timeout=60)
    assert res.status == "error" and "stalled" in res.detail
    rep = reply_metrics.compute(s.runtime_dir, since=t0, now=cli.wall() + 10)
    assert rep.counts() == {"bound": {"failed": 1}} and rep.not_delivered == 1


# ── replay of real lost-reply incidents ─────────────────────────────────
# tests/fixtures/reply_ledger_replay.json: the 2026-10-29 06:45–06:58
# incident and 11 earlier cases, reconstructed from bot.log, watchdog.log and
# the CLI transcripts of a production instance (redacted: no message text, only
# lengths, rids and marker strings as the model wrote them). Each event is
# what the CLI did under the OLD path; ``old_path_outcome`` says what reached
# Telegram then. The replay feeds the same human/CLI timeline through the
# NEW path: Telegram messages go through the ledger steer gate (steer into an
# open CLI turn, otherwise a bound ask, queued while an ask holds the lock),
# CLI-side events become hook events, the pump runs every 2 s.

import re  # noqa: E402
from collections import deque  # noqa: E402

REPLAY = json.loads(
    (ROOT / "tests" / "fixtures" / "reply_ledger_replay.json").read_text()
)
_TURN_RE = re.compile(r"\[turn=(\d+)\]")


def replay(tmp_path: Path, clock: dict, case: dict) -> dict:
    s, cli = make(tmp_path, clock, stall_timeout=900)
    sent: list[str] = []
    pump = make_pump(s.runtime_dir, cli, sent)
    pid_of: dict[int, str] = {}
    queue: deque[int] = deque()
    results: list[tuple[int, object]] = []
    raw_submit = cli.submit

    def submit(text: str) -> str:
        pid = raw_submit(text)
        m = _TURN_RE.search(text)
        if m and int(m.group(1)) not in pid_of:
            pid_of[int(m.group(1))] = pid
        return pid

    cli.submit = submit

    def telegram(k: int) -> None:
        # bot/handlers/chat.py with the ledger on: steer iff the CLI turn is
        # open and steerable; otherwise a new bound request (queued behind a
        # running ask, exactly like the per-chat lane).
        if s.is_steerable_turn():
            s.steer(f"сообщение [turn={k}]")
        else:
            queue.append(k)

    expected: dict[int, str] = {}

    def end(k: int, text: str, bg: int) -> None:
        if k not in pid_of:  # its request is still queued: the turn is later
            cli.at(1, lambda: end(k, text, bg))
            return
        # Make every turn's text unique, so the test can tell WHICH reply
        # reached the person, not only that something did.
        text = text.replace("<reply text", f"<reply text t{k}")
        expected[k] = turn_ledger.clean_reply(text)
        cli.stop(pid_of[k], text, bg=bg)

    for e in case["events"]:
        k, kind = e["turn"], e["kind"]
        if kind == "user_prompt" and e["rid"]:
            fn = lambda k=k: telegram(k)  # noqa: E731
        elif kind == "steer":
            fn = lambda k=k: telegram(k)  # noqa: E731
        elif kind == "user_prompt":  # typed straight into the pane
            fn = lambda k=k: cli.submit(f"набрано в tmux [turn={k}]")  # noqa: E731
        elif kind == "task_notification":
            fn = lambda k=k: cli.submit(  # noqa: E731
                f"<task-notification>\n<task-id>[turn={k}]</task-id>"
            )
        elif kind == "turn_end":
            bg = e.get("background_tasks", 0)
            fn = lambda k=k, t=e["final_text"], b=bg: end(k, t, b)  # noqa: E731
        elif kind == "interrupt":
            fn = s.interrupt
        else:
            raise AssertionError(kind)
        cli.at(e["t"], fn)

    def tick() -> None:
        pump_once(pump)
        cli.at(2, tick)

    cli.at(2, tick)
    horizon = max(e["t"] for e in case["events"]) + 60
    while clock["now"] < horizon or queue:
        if queue:
            k = queue.popleft()
            results.append((k, s.ask(f"запрос [turn={k}]", timeout=1500)))
        else:
            s.capture_text()
            clock["now"] += 1
    idle(s, clock, int(reply_pump.BOUND_GRACE) + 10)
    pump_once(pump)
    return {
        "session": s,
        "pid_of": pid_of,
        "sent": sent,
        "results": results,
        "expected": expected,
    }


@pytest.mark.parametrize(
    "case", REPLAY["cases"], ids=[c["name"] for c in REPLAY["cases"]]
)
def test_replay_every_reply_now_reaches_telegram_once(tmp_path, clock, case):
    out = replay(tmp_path, clock, case)
    recs = turn_ledger.Claims(out["session"].runtime_dir).records()
    delivered: dict[str, int] = {}
    for r in recs:
        if r.get("state") == "delivered":
            delivered[r["prompt_id"]] = delivered.get(r["prompt_id"], 0) + 1
    assert all(n == 1 for n in delivered.values()), delivered  # never twice
    ends = [e for e in case["events"] if e["kind"] == "turn_end" and e["final_text"]]
    for e in ends:
        pid = out["pid_of"].get(e["turn"])
        assert pid is not None, f"turn {e['turn']} never reached the CLI"
        assert delivered.get(pid) == 1, f"turn {e['turn']} ({pid}) not delivered"
    # Every bound request got a real answer, never a timeout or a busy notice.
    for k, res in out["results"]:
        assert res.ok, (k, res)
    # The RIGHT text reached the person: each turn's own final message (the
    # last one, where two fixture turns ended up in one CLI turn), with the
    # stale marker lines gone.
    got = [r.reply for _, r in out["results"]] + [
        t.removeprefix(reply_pump.LATE_PREFIX) for t in out["sent"]
    ]
    last_turn_of_pid = {pid: k for k, pid in sorted(out["pid_of"].items())}
    for e in ends:
        k = e["turn"]
        if last_turn_of_pid.get(out["pid_of"][k]) == k and k in out["expected"]:
            assert out["expected"][k] in got, (k, out["expected"][k], got)
    assert not any("<<<" in g for g in got)
    # The turns the old path lost are among the ones just shown delivered.
    lost = [k for k, v in case["old_path_outcome"].items() if v == "lost"]
    for key in lost:
        if key.startswith("turn "):
            assert delivered.get(out["pid_of"][int(key.split()[1])]) == 1


# ── a prompt that never left the input box (2026-10-29) ─────────────────


def test_lost_enter_with_a_late_drawn_box_is_pressed_again(tmp_path, clock, caplog):
    """rid 21687a6e: the first Enter was lost and the CLI drew the paste
    after _confirm_submitted had looked (empty box → "sent"). The ledger
    shows no submit, so ask() looks again and presses Enter."""
    s, cli = make(tmp_path, clock, stall_timeout=900)
    cli.on_submit = reply_after(2, "Отменил.")
    cli.show_box = True
    cli.hide_box_captures = 4
    cli.lost_enters = 1
    started = clock["now"]
    res = s.ask("отмени джоб", timeout=600)
    assert res.ok and res.reply == "Отменил."
    assert clock["now"] - started < 30
    assert "pressing Enter (1/" in caplog.text


def test_prompt_stuck_in_the_box_is_taken_out_and_reported(tmp_path, clock):
    s, cli = make(tmp_path, clock, stall_timeout=900)
    cli.show_box = True
    cli.lost_enters = 10_000
    started = clock["now"]
    res = s.ask("вопрос", timeout=600)
    assert res.status == "error"
    assert res.detail == claude_session.NOT_SUBMITTED_DETAIL
    assert clock["now"] - started < 60
    assert cli._box is None  # taken back out, nothing left to glue onto
    assert not (s.runtime_dir / "inflight").exists()
    assert cli.prompts == []


def test_mangled_tag_is_still_our_turn(tmp_path, clock):
    """rid 3529ff34: a keystroke typed in the attached pane landed inside the
    tag, so the hook saw an untagged user prompt. ask() takes that turn as
    its own instead of waiting for a rid that can never appear."""
    s, cli = make(tmp_path, clock, stall_timeout=900)
    clock["now"] = 5.0  # the fake CLI's wall clock must run past time.time()
    cli.mangle = lambda t: t.replace('kind="ask"/>', 'kind="ask"/о')
    cli.on_submit = reply_after(2, "Нет, не завис.")
    started = clock["now"]
    res = s.ask("Ты завис?", timeout=600)
    assert res.ok and res.reply == "Нет, не завис."
    assert clock["now"] - started < 15


def test_untagged_turn_before_the_paste_is_not_taken(tmp_path, clock):
    book = turn_ledger.TurnBook()
    book.add({"ev": "submit", "ts": 100.0, "prompt_id": "old", "origin": "user"})
    assert ClaudeSession._untagged_submit_since(book, 200.0, "x") is None
    book.add(
        {"ev": "submit", "ts": 201.0, "prompt_id": "n", "origin": "task-notification"}
    )
    assert ClaudeSession._untagged_submit_since(book, 200.0, "x") is None
    book.add({"ev": "submit", "ts": 202.0, "prompt_id": "mine", "origin": "user"})
    assert ClaudeSession._untagged_submit_since(book, 200.0, "x").prompt_id == "mine"
