"""Cron reply pump (2026-11-02): a background agent dispatched by a cron job
reports after the job's turn closed; that task-notification turn of the cron
session is delivered by the same reply pump the main session uses.

The end-to-end tests drive a real ClaudeSession in the cron runtime dir
through test_reply_ledger's FakeCLI (the real turn hook writes the ledger),
a real CronRunner on top of it, and a ReplyPump over the same dir. The
replay tests feed the three reference cases from the live cron ledger
(tests/fixtures/cron_background_reports.json).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_reply_ledger as trl

from d_brain.bot import main as bot_main
from d_brain.config import Settings
from d_brain.services import reply_pump, turn_ledger
from d_brain.services.claude_session import (
    LEDGER_ALREADY_SENT,
    LEDGER_EMPTY_REPLY,
    AskResult,
)
from d_brain.services.cron_runner import (
    BACKGROUND_RULE,
    CRON_LATE_PREFIX,
    CronRunner,
    cron_pump_enabled,
    wrap_job_prompt,
)
from d_brain.services.cron_store import CronJob, CronStore, Schedule

# The reply-ledger harness: FakeCLI behind a real ClaudeSession, its fixtures.
_Bot, idle, make = trl._Bot, trl.idle, trl.make
_home = trl._home  # autouse: HOME and the live-CLI check
clock = trl.clock

FIXTURE = Path(__file__).parent / "fixtures" / "cron_background_reports.json"
NOW = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
NOTE = "<task-notification>\n<task-id>{}</task-id>\n<status>completed</status>"


# ── harness ─────────────────────────────────────────────────────────────


class CronBrain:
    """The cron runner's view of a real ClaudeSession over FakeCLI. /clear
    is what the CLI does with it: a SessionStart(clear) with a new id."""

    def __init__(self, s, cli) -> None:
        self.s, self.cli = s, cli
        self.clears = 0

    def ask(self, prompt, *, timeout=0.0, request_id=None, wrap=True):
        return self.s.ask(prompt, timeout=timeout, request_id=request_id)

    def send_control(self, text):
        if text == "/clear":
            self.clears += 1
            self.cli.sid = f"sid-after-clear-{self.clears}"
            self.cli.fire({"hook_event_name": "SessionStart", "source": "clear"})

    def force_recover(self):
        return True

    def capture_text(self):
        return ""

    def is_turn_active(self):
        return False


class World:
    """One cron session + its runner + its pump, all on one clock."""

    def __init__(
        self, tmp_path: Path, clock: dict, *, empty_notice=False, deliver=None
    ):
        self.clock = clock
        self.s, self.cli = make(tmp_path, clock, name=".dbrain/cron")
        self.rt = self.s.runtime_dir
        self.sent: list[str] = []  # everything that reached "Telegram"
        self.runner_sent: list[str] = []
        self.pump_sent: list[str] = []

        async def runner_deliver(chat_id, text):
            self.runner_sent.append(text)
            self.sent.append(text)

        async def pump_deliver(text):
            self.pump_sent.append(text)
            self.sent.append(text)

        self.pump_deliver = deliver or pump_deliver
        self.empty_notice = empty_notice
        self.store = CronStore(self.rt)
        self.runner = CronRunner(
            self.store,
            CronBrain(self.s, self.cli),
            deliver=runner_deliver,
            alert=_noop,
            default_chat_id=111,
            job_timeout=120.0,
            clock=lambda: NOW,
            background_relayed=True,
        )
        self.pump = self.new_pump()

    def new_pump(self):
        return reply_pump.ReplyPump(
            self.rt,
            self.pump_deliver,
            clock=self.cli.wall,
            empty_notice=self.empty_notice,
            name="cron",
            late_prefix=CRON_LATE_PREFIX,
        )

    def add_job(self, job_id: str, prompt: str = "сделай проверку") -> None:
        job = CronJob(
            id=job_id,
            prompt=prompt,
            schedule=Schedule(kind="every", every_seconds=3600),
        )
        job.state.next_run = NOW.isoformat()
        self.store.mutate(lambda jobs: jobs.append(job))

    def run_due(self) -> None:
        asyncio.run(self.runner.tick())

    def wait(self, seconds: int) -> None:
        idle(self.s, self.clock, seconds)

    def pump_passes(self, n: int = 3, pump=None) -> int:
        pump = pump or self.pump
        total = 0
        for _ in range(n):
            total += asyncio.run(pump.step())
            self.clock["now"] += 2
        return total

    def states(self) -> dict[str, dict]:
        return turn_ledger.Claims(self.rt).states()


async def _noop(*_a, **_k):
    return None


def background_job(cli, *, job_reply, reports, gap=60.0, run=12.0, bg_id="a"):
    """on_submit: the job's turn dispatches len(reports) background agents and
    closes with ``job_reply``; agent i reports ``gap*(i+1)`` seconds later in
    its own task-notification turn (``None`` = a turn with no text)."""

    def cb(c, pid, text):
        if text.lstrip().startswith("<task-notification>"):
            return
        if sum(1 for p, _ in c.prompts if p == pid) != 1:
            return
        c.at(5, lambda: c.stop(pid, job_reply, bg=len(reports)))
        for i, report in enumerate(reports):

            def agent_done(i=i, report=report):
                note_pid = c.notify(NOTE.format(f"{bg_id}{i}"))
                c.at(run, lambda: c.stop(note_pid, report))

            c.at(5 + gap * (i + 1), agent_done)

    return cb


# ── the core case ───────────────────────────────────────────────────────


def test_background_report_after_the_job_closed_is_delivered_once(tmp_path, clock):
    w = World(tmp_path, clock)
    w.add_job("ozon-morning")
    w.cli.on_submit = background_job(
        w.cli,
        job_reply="[SILENT] Отправил фоновому агенту проверку, итог придёт отдельно.",
        reports=["Проверка прошла: <b>все 16 офферов совпадают</b>."],
    )
    w.run_due()
    assert w.runner_sent == []  # the job's own turn was [SILENT]
    w.wait(90)
    w.pump_passes(5)
    assert w.pump_sent == ["Проверка прошла: <b>все 16 офферов совпадают</b>."]
    # more passes, and a restarted bot's fresh pump: still once
    w.pump_passes(5)
    w.pump_passes(3, pump=w.new_pump())
    assert w.sent == ["Проверка прошла: <b>все 16 офферов совпадают</b>."]
    delivered = [
        r for r in turn_ledger.Claims(w.rt).records() if r.get("state") == "delivered"
    ]
    assert [r["by"] for r in delivered] == ["ask", "pump"]


def test_job_prompt_tells_the_brain_the_follow_up_is_delivered(tmp_path, clock):
    w = World(tmp_path, clock)
    w.add_job("j")
    w.cli.on_submit = background_job(w.cli, job_reply="[SILENT] ok", reports=[])
    w.run_due()
    job_prompt = w.cli.prompts[0][1]
    assert BACKGROUND_RULE.strip() in job_prompt
    assert BACKGROUND_RULE not in wrap_job_prompt("j", "p", scheduled_for=None)
    assert BACKGROUND_RULE in wrap_job_prompt(
        "j", "p", scheduled_for=None, background_relayed=True
    )


def test_silent_background_report_is_not_sent(tmp_path, clock):
    w = World(tmp_path, clock)
    w.add_job("mail-watch")
    w.cli.on_submit = background_job(
        w.cli,
        job_reply="[SILENT] агент запущен",
        reports=["[SILENT] агент закончил, нового нет, записал в журнал"],
    )
    w.run_due()
    w.wait(90)
    w.pump_passes(5)
    assert w.sent == []
    note = [t for t in turn_ledger.Claims(w.rt).records() if t.get("by") == "pump"]
    assert [r["state"] for r in note] == ["sentinel"]


def test_job_reply_is_delivered_by_the_runner_not_the_pump(tmp_path, clock):
    w = World(tmp_path, clock)
    w.add_job("digest")
    w.cli.on_submit = background_job(
        w.cli, job_reply="<b>Дайджест</b>: всё спокойно.", reports=[]
    )
    w.run_due()
    w.wait(60)
    w.pump_passes(5)
    assert w.runner_sent == ["<b>Дайджест</b>: всё спокойно."]
    assert w.pump_sent == []


def test_several_agents_in_a_row_each_delivered_once_in_order(tmp_path, clock):
    w = World(tmp_path, clock)
    w.add_job("retro")
    reports = [f"<b>Отчёт агента {i}</b>: готово." for i in range(1, 4)]
    w.cli.on_submit = background_job(
        w.cli, job_reply="[SILENT] запустил трёх агентов", reports=reports, gap=40
    )
    w.run_due()
    for _ in range(4):
        w.wait(40)
        w.pump_passes(2)
    w.pump_passes(5)
    assert w.sent == reports


def test_two_reports_racing_each_other_before_one_pump_pass(tmp_path, clock):
    """Both agents finish between two pump passes (bot was busy)."""
    w = World(tmp_path, clock)
    w.add_job("pair")
    w.cli.on_submit = background_job(
        w.cli,
        job_reply="[SILENT] два агента",
        reports=["первый", "второй"],
        gap=20,
        run=2,
    )
    w.run_due()
    w.wait(80)
    assert w.pump_passes(1) == 2
    w.pump_passes(4)
    assert w.sent == ["первый", "второй"]


# ── a notification and the next job ────────────────────────────────────


def test_report_arriving_while_the_next_job_runs(tmp_path, clock):
    """Job A's agent reports while job B's turn is open; the CLI runs the
    notification as its own turn right after B's. B's reply goes out via
    the runner, A's report via the pump, each once."""
    w = World(tmp_path, clock)
    w.add_job("a-job")
    w.cli.on_submit = background_job(w.cli, job_reply="[SILENT] агент A", reports=[])
    w.run_due()
    w.wait(10)

    def job_b(c, pid, text):
        if text.lstrip().startswith("<task-notification>"):
            return

        def b_done():
            c.stop(pid, "<b>Задание B</b>: готово.")
            note = c.notify(NOTE.format("A"))  # queued, starts right after
            c.at(3, lambda: c.stop(note, "<b>Агент A</b>: нашёл 2 заявки."))

        c.at(8, b_done)

    w.cli.on_submit = job_b
    w.store.mutate(lambda jobs: jobs.clear())
    w.add_job("b-job")
    w.run_due()
    w.wait(10)
    w.pump_passes(5)
    assert sorted(w.sent) == sorted(
        ["<b>Задание B</b>: готово.", "<b>Агент A</b>: нашёл 2 заявки."]
    )
    assert w.runner_sent == ["<b>Задание B</b>: готово."]
    assert w.pump_sent == ["<b>Агент A</b>: нашёл 2 заявки."]


def test_report_folded_into_the_running_job_turn(tmp_path, clock):
    """Job A's agent reports while job B's turn runs and the CLI files the
    notification under B's prompt_id (a joined turn, as seen in a live
    cron ledger). The turn is bound to B's ask(): the runner delivers its
    one final message and the pump does not send it again."""
    w = World(tmp_path, clock)
    w.add_job("b-job")

    def job_b(c, pid, text):
        if text.lstrip().startswith("<task-notification>"):
            return
        c.at(3, lambda: c.notify(NOTE.format("A")))  # joins the open turn
        c.at(6, lambda: c.stop(pid, "<b>Агент A</b> и <b>задание B</b>: всё готово."))

    w.cli.on_submit = job_b
    w.run_due()
    w.wait(40)
    w.pump_passes(5)
    assert w.sent == ["<b>Агент A</b> и <b>задание B</b>: всё готово."]
    assert w.runner_sent == w.sent
    pid = w.cli.prompts[0][0]
    assert [p for p, _ in w.cli.prompts] == [pid, pid]
    assert w.states()[pid]["by"] == "ask"


def test_pump_won_the_race_runner_does_not_send_again():
    """ask() lost the claim to the pump: the runner must not send the
    reply nor the 'already sent' notice meant for a chat."""
    from test_cron_runner import FakeSession, Recorder, _add_job, _runner

    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    store = CronStore(tmp / "cron")
    _add_job(store, "j1", next_run=datetime(2026, 6, 10, 12, 0, tzinfo=UTC))
    sess = FakeSession(
        [AskResult("ok", reply=LEDGER_ALREADY_SENT, detail="already delivered")]
    )
    deliver = Recorder()
    runner = _runner(
        store,
        sess,
        deliver=deliver,
        clock=lambda: datetime(2026, 6, 10, 12, 0, tzinfo=UTC),
    )
    asyncio.run(runner.tick())
    assert deliver.calls == []
    assert store.load()[0].state.last_status == "ok-silent"


# ── restarts and history ────────────────────────────────────────────────


def test_bot_restart_between_dispatch_and_report(tmp_path, clock):
    """The job dispatched an agent, then the bot restarted; the agent
    reported while the bot was down. The new process's pump sends it once."""
    w = World(tmp_path, clock)
    w.add_job("long-crawl")
    w.cli.on_submit = background_job(
        w.cli,
        job_reply="[SILENT] обход запущен",
        reports=["<b>Обход</b>: 3 ошибки."],
        gap=120,
    )
    w.run_due()
    w.pump_passes(2)  # the old process's pump has run: since is set
    w.wait(150)  # bot is down; the CLI keeps working and reports
    fresh = w.new_pump()
    w.pump_passes(4, pump=fresh)
    assert w.sent == ["<b>Обход</b>: 3 ошибки."]


def test_job_turn_finishing_while_the_bot_is_down_comes_late_and_marked(
    tmp_path, clock
):
    """The runner's ask() died with the bot mid-job; the job turn's Stop is
    then a bound turn nobody waits for: the pump delivers it after the
    grace, marked late."""
    w = World(tmp_path, clock)
    w.pump_passes(1)
    pid = w.cli.submit('[CRON JOB x] проверка <dbrain-turn rid="r9" kind="ask"/>')
    w.clock["now"] += 30
    w.cli.stop(pid, "<b>Итог</b>: ок")
    fresh = w.new_pump()
    assert asyncio.run(fresh.step()) == 0  # inside the grace a live ask() wins
    w.clock["now"] += reply_pump.BOUND_GRACE + 1
    w.pump_passes(2, pump=fresh)
    assert w.sent == [CRON_LATE_PREFIX + "<b>Итог</b>: ок"]


def test_first_run_does_not_backfill_history_and_logs_it(tmp_path, clock, caplog):
    """The 16 marker-era rids in pending_orphans and the task-notification
    turns nobody took before the pump existed stay unsent; the first pass
    logs them once."""
    s, cli = make(tmp_path, clock, name=".dbrain/cron")
    rt = s.runtime_dir
    rids = ["840d7cde", "rel0926", "fw27a1", "retro39", "fw0930", "yt1106a",
            "ytw0910a", "yt1809", "yt1105", "fw1013", "spk1810", "ozpf1019",
            "ozpf2210", "ritm2410", "ozpf1026", "ritm2610"]  # fmt: skip
    (rt / "pending_orphans").write_text("\n".join(rids) + "\n")
    for i in range(3):
        p = cli.notify(NOTE.format(f"old{i}"))
        cli.stop(p, f"<<<R:old{i}>>>\nстарый отчёт {i}")
        clock["now"] += 600
    sent: list[str] = []

    async def deliver(t):
        sent.append(t)

    with caplog.at_level(logging.WARNING, logger="d_brain.services.reply_pump"):
        pump = reply_pump.ReplyPump(rt, deliver, clock=cli.wall, name="cron")
        for _ in range(3):
            asyncio.run(pump.step())
            clock["now"] += 30
    assert sent == []
    msgs = [r.getMessage() for r in caplog.records if "first run" in r.getMessage()]
    assert len(msgs) == 1
    assert "3 earlier unclaimed turn(s)" in msgs[0] and "16 legacy" in msgs[0]
    assert (rt / "pending_orphans").read_text().split() == rids  # left as is
    # a restart is not a first run: no second report, still nothing sent
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="d_brain.services.reply_pump"):
        asyncio.run(reply_pump.ReplyPump(rt, deliver, clock=cli.wall).step())
    assert not [r for r in caplog.records if "first run" in r.getMessage()]
    assert sent == []


def test_pump_never_sends_a_report_older_than_max_age(tmp_path, clock):
    w = World(tmp_path, clock)
    w.pump_passes(1)
    p = w.cli.notify(NOTE.format("x"))
    w.cli.stop(p, "отчёт")
    w.clock["now"] += reply_pump.MAX_AGE + 10  # the bot was down 6h+
    w.pump_passes(2, pump=w.new_pump())
    assert w.sent == []


# ── empty turns (DBRAIN_EMPTY_REPLY_NOTICE) ─────────────────────────────


@pytest.mark.parametrize("notice", [False, True])
def test_empty_background_turn_is_never_sent(tmp_path, clock, notice):
    """No text from the agent's turn: nothing reaches the owner, notice or
    not — exactly like an unbound main-session notification turn. The
    notice is only for a turn that carried a person's (or a job's) request."""
    w = World(tmp_path, clock, empty_notice=notice)
    w.add_job("quiet")
    w.cli.on_submit = background_job(w.cli, job_reply="[SILENT] пуск", reports=[None])
    w.run_due()
    w.wait(90)
    w.pump_passes(4)
    assert w.sent == []
    pump_claims = [
        r for r in turn_ledger.Claims(w.rt).records() if r.get("by") == "pump"
    ]
    assert [r["state"] for r in pump_claims] == ["empty"]


@pytest.mark.parametrize("notice", [False, True])
def test_empty_job_turn_follows_the_instance_setting(tmp_path, clock, notice):
    w = World(tmp_path, clock, empty_notice=notice)
    w.s.empty_reply_notice = notice
    w.add_job("quiet-job")
    w.cli.on_submit = background_job(w.cli, job_reply=None, reports=[])
    w.run_due()
    w.wait(30)
    w.pump_passes(3)
    assert w.sent == ([LEDGER_EMPTY_REPLY] if notice else [])


# ── Telegram failures ───────────────────────────────────────────────────


def test_telegram_failure_is_retried_and_sent_once(tmp_path, clock):
    attempts: list[str] = []
    w_sent: list[str] = []

    async def flaky(text):
        attempts.append(text)
        if len(attempts) <= 2:
            raise RuntimeError("Telegram 502")
        w_sent.append(text)

    w = World(tmp_path, clock, deliver=flaky)
    w.add_job("j")
    w.cli.on_submit = background_job(w.cli, job_reply="[SILENT] пуск", reports=["итог"])
    w.run_due()
    w.wait(90)
    w.pump_passes(1)  # fails, stays claimed
    w.pump_passes(1, pump=w.new_pump())  # bot restarted: resumes, fails again
    w.pump_passes(4)
    assert w_sent == ["итог"] and len(attempts) == 3
    assert w.states()[w.cli.prompts[-1][0]]["state"] == "delivered"


def test_long_report_goes_through_send_response_chunks(tmp_path, clock, monkeypatch):
    from d_brain.bot import formatters
    from d_brain.services import outbox

    monkeypatch.setattr(outbox, "SEND_SPACING", 0)
    outbox.reset()
    bot = _Bot()

    async def deliver(text):
        await formatters.send_response(bot, 111, text)

    w = World(tmp_path, clock, deliver=deliver)
    body = ("<b>Раздел</b>\n" + "строка отчёта " * 40 + "\n\n") * 20
    w.add_job("j")
    w.cli.on_submit = background_job(w.cli, job_reply="[SILENT] пуск", reports=[body])
    w.run_due()
    w.wait(90)
    w.pump_passes(3)
    assert len(bot.sent) >= 2 and all(len(x) <= 4096 for x in bot.sent)
    assert "".join(bot.sent).count("строка отчёта") == body.count("строка отчёта")


# ── main and cron at the same time ──────────────────────────────────────


def test_main_and_cron_pumps_deliver_side_by_side(tmp_path, clock):
    main_s, main_cli = make(tmp_path, clock, name=".dbrain")
    w = World(tmp_path, clock)
    sent: list[str] = []

    async def deliver(text):
        sent.append(text)

    main_pump = reply_pump.ReplyPump(main_s.runtime_dir, deliver, clock=main_cli.wall)
    cron_pump = reply_pump.ReplyPump(w.rt, deliver, clock=w.cli.wall, name="cron")
    p1 = main_cli.notify(NOTE.format("m"))
    p2 = w.cli.notify(NOTE.format("c"))
    main_cli.stop(p1, "отчёт основной сессии")
    w.cli.stop(p2, "отчёт крон-сессии")
    clock["now"] += 5

    async def two_passes():
        task = asyncio.create_task(reply_pump.run(main_pump, cron_pump, poll_seconds=0))
        for _ in range(20):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(two_passes())
    assert sorted(sent) == ["отчёт крон-сессии", "отчёт основной сессии"]
    # each dir keeps its own claims; the same prompt_id in both is no clash
    assert {
        r["prompt_id"] for r in turn_ledger.Claims(main_s.runtime_dir).records()
    } == {p1}
    assert {
        r["prompt_id"]
        for r in turn_ledger.Claims(w.rt).records()
        if r.get("by") == "pump"
    } == {p2}


def test_one_failing_pump_does_not_stop_the_other(tmp_path, clock):
    w = World(tmp_path, clock)
    sent: list[str] = []

    class Broken:
        name = "main"

        async def step(self):
            raise RuntimeError("boom")

    async def deliver(text):
        sent.append(text)

    cron_pump = reply_pump.ReplyPump(w.rt, deliver, clock=w.cli.wall, name="cron")
    p = w.cli.notify(NOTE.format("c"))
    w.cli.stop(p, "дошло")
    clock["now"] += 5

    async def go():
        task = asyncio.create_task(reply_pump.run(Broken(), cron_pump, poll_seconds=0))
        for _ in range(10):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert sent == ["дошло"]


# ── the three reference cases from the live cron ledger ─────────────────


def _cases():
    return json.loads(FIXTURE.read_text())["cases"]


def _replay(rt: Path, base: float, case: dict) -> None:
    lines = []
    for rec in case["records"]:
        out = {
            "ev": rec["ev"],
            "ts": base + rec["dt"],
            "session_id": rec["session"],
            "prompt_id": rec["prompt_id"],
        }
        for key in ("rid", "kind", "origin", "text", "background_tasks", "source"):
            if key in rec:
                out[key] = rec[key]
        lines.append(json.dumps(out, ensure_ascii=False))
    with (rt / turn_ledger.LEDGER_FILE).open("a") as f:
        f.write("\n".join(lines) + "\n")
    claims = turn_ledger.Claims(rt)
    for c in case["ask_claims"]:
        claims.claim(
            c["prompt_id"], by="ask", state="delivered", rid=c["rid"], bound=True,
            sentinel=c["sentinel"],
        )  # fmt: skip


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_reference_case_is_delivered_once(tmp_path, case):
    rt = tmp_path / "cron"
    rt.mkdir()
    now = {"t": 1_000_000.0}
    sent: list[str] = []

    async def deliver(text):
        sent.append(text)

    pump = reply_pump.ReplyPump(rt, deliver, clock=lambda: now["t"], name="cron")
    asyncio.run(pump.step())
    _replay(rt, now["t"] + 1, case)
    end = max(r["dt"] for r in case["records"])
    for dt in range(0, int(end) + 30, 7):
        now["t"] = 1_000_000.0 + 1 + dt
        asyncio.run(pump.step())
    assert sent == [case["expected"]]
    states = turn_ledger.Claims(rt).states()
    job = case["ask_claims"][0]["prompt_id"]
    bg = case["records"][-1]["prompt_id"]
    assert states[job]["by"] == "ask" and states[bg]["by"] == "pump"
    assert (
        states[bg]["state"] == "delivered"
        and states[bg]["origin"] == "task-notification"
    )


def test_all_reference_cases_in_one_ledger(tmp_path):
    rt = tmp_path / "cron"
    rt.mkdir()
    now = {"t": 2_000_000.0}
    sent: list[str] = []

    async def deliver(text):
        sent.append(text)

    pump = reply_pump.ReplyPump(rt, deliver, clock=lambda: now["t"], name="cron")
    asyncio.run(pump.step())
    base = now["t"] + 1
    for i, case in enumerate(_cases()):
        _replay(rt, base + i * 3600, case)
    for t in range(0, 3 * 3600, 30):
        now["t"] = base + t
        asyncio.run(pump.step())
    assert sent == [c["expected"] for c in _cases()]


# ── wiring: the instance switch ─────────────────────────────────────────


def _settings(tmp_path, **kw):
    base = dict(
        reply_ledger=True,
        cron_reply_pump=False,
        cron_enabled=True,
        cron_engine="claude",
        chat_engine="claude",
        empty_reply_notice=False,
        runtime_dir=tmp_path / "rt",
        cron_dir=tmp_path / "rt" / "cron",
    )
    base.update(kw)
    return SimpleNamespace(**base)


async def _d(text):
    return None


def test_switch_off_by_default_only_main_pump(tmp_path):
    pumps = bot_main._reply_pumps(_settings(tmp_path), _d)
    assert [p.name for p in pumps] == ["main"]


def test_switch_on_adds_the_cron_pump(tmp_path):
    pumps = bot_main._reply_pumps(
        _settings(tmp_path, cron_reply_pump=True, empty_reply_notice=True), _d
    )
    assert [p.name for p in pumps] == ["main", "cron"]
    cron = pumps[1]
    assert cron.runtime_dir == tmp_path / "rt" / "cron"
    assert cron._empty_notice is True and cron._late_prefix == CRON_LATE_PREFIX


@pytest.mark.parametrize(
    "over",
    [
        {"reply_ledger": False},
        {"cron_enabled": False},
        {"cron_engine": "codex"},
    ],
)
def test_cron_pump_needs_ledger_cron_and_claude(tmp_path, over):
    s = _settings(tmp_path, cron_reply_pump=True, **over)
    assert not cron_pump_enabled(s)
    assert "cron" not in [p.name for p in bot_main._reply_pumps(s, _d)]


def test_cron_pump_runs_even_when_chat_is_on_codex(tmp_path):
    s = _settings(tmp_path, cron_reply_pump=True, chat_engine="codex")
    assert [p.name for p in bot_main._reply_pumps(s, _d)] == ["cron"]


def test_env_flag_parses(monkeypatch, tmp_path):
    monkeypatch.setenv("DBRAIN_CRON_REPLY_PUMP", "1")
    base = dict(
        telegram_bot_token="t",
        deepgram_api_key="d",
        _env_file=None,
        vault_path=tmp_path,
    )
    assert Settings(**base).cron_reply_pump is True
    monkeypatch.delenv("DBRAIN_CRON_REPLY_PUMP")
    assert Settings(**base).cron_reply_pump is False
