"""Stuck-turn metric: journal, summary, and its two readers (/work, doctor)."""

import asyncio
import json

from d_brain.bot.formatters import validate_telegram_html
from d_brain.bot.handlers import work
from d_brain.config import Settings
from d_brain.services import chat_queue, doctor, turn_metrics


def _events(rt):
    path = rt / turn_metrics.EVENTS_FILE
    return [json.loads(x) for x in path.read_text().splitlines()]


# ── journal ───────────────────────────────────────────────────────────────


def test_append_writes_one_json_line_per_event(tmp_path):
    turn_metrics.append_event(
        tmp_path, "queued", ts=10.0, job_id="a", turn_state="open"
    )
    turn_metrics.append_event(tmp_path, "decision", ts=11.0, source="hooks", open=False)
    evs = _events(tmp_path)
    assert [e["kind"] for e in evs] == ["queued", "decision"]
    assert evs[0] == {"ts": 10.0, "kind": "queued", "job_id": "a", "turn_state": "open"}


def test_append_never_raises(tmp_path):
    turn_metrics.append_event(tmp_path / "missing" / "dir", "queued")
    turn_metrics.append_event(None, "queued")


def test_oversized_journal_keeps_the_newer_half(tmp_path, monkeypatch):
    monkeypatch.setattr(turn_metrics, "MAX_BYTES", 2000)
    for i in range(100):
        turn_metrics.append_event(tmp_path, "queued", ts=float(i), job_id=str(i))
    path = tmp_path / turn_metrics.EVENTS_FILE
    assert path.stat().st_size <= 2000
    evs = _events(tmp_path)  # every surviving line still parses
    assert evs[-1]["job_id"] == "99"
    assert evs[0]["ts"] > 0


def test_turn_state_raw(tmp_path):
    assert turn_metrics.read_turn_state_raw(tmp_path) == "unknown"
    f = tmp_path / turn_metrics.STATE_FILE
    f.write_text("{broken")
    assert turn_metrics.read_turn_state_raw(tmp_path) == "unknown"
    f.write_text(json.dumps({"state": "closed"}))
    assert turn_metrics.read_turn_state_raw(tmp_path) == "closed"
    f.write_text(json.dumps({"state": "weird"}))
    assert turn_metrics.read_turn_state_raw(tmp_path) == "unknown"


# ── summary ───────────────────────────────────────────────────────────────


def test_summary_counts_within_the_window_and_skips_broken_lines(tmp_path):
    now = 100_000.0
    a = turn_metrics.append_event
    a(tmp_path, "queued", ts=now - 90_000, turn_state="closed")  # outside window
    a(tmp_path, "queued", ts=now - 10, turn_state="closed")
    a(tmp_path, "queued", ts=now - 9, turn_state="open")
    a(tmp_path, "dispatched", ts=now - 8, delay_s=3.0, turn_state_at_queue="closed")
    a(tmp_path, "dispatched", ts=now - 7, delay_s=40.0, turn_state_at_queue="open")
    a(tmp_path, "stale_override", ts=now - 6)
    a(tmp_path, "decision", ts=now - 5, source="screen", open=True)
    a(tmp_path, "decision", ts=now - 4, source="hooks", open=False)
    with (tmp_path / turn_metrics.EVENTS_FILE).open("a") as fh:
        fh.write('not json\n{"kind": 1}\n{"ts": "x", "kind": "queued"}\n')
    d = turn_metrics.summary(tmp_path, now=now)
    assert d["queued_total"] == 2
    assert d["parked_while_closed"] == 1
    assert d["max_queue_delay_s"] == 40.0
    assert d["max_queue_delay_closed_s"] == 3.0
    assert d["stale_overrides"] == 1
    assert d["screen_decisions_share"] == 0.5


def test_summary_of_nothing(tmp_path):
    d = turn_metrics.summary(tmp_path, now=1.0)
    assert d["queued_total"] == 0
    assert d["max_queue_delay_s"] is None
    assert d["screen_decisions_share"] is None
    assert not turn_metrics.needs_attention(d)


def test_format_summary_is_valid_telegram_html(tmp_path):
    turn_metrics.append_event(tmp_path, "queued", ts=5.0, turn_state="closed")
    d = turn_metrics.summary(tmp_path, now=10.0)
    text = turn_metrics.format_summary(d)
    assert "Застревания за сутки" in text
    assert "при закрытом ходе: <b>1</b>" in text
    assert validate_telegram_html(text)
    assert turn_metrics.needs_attention(d)


# ── integration: the chat queue writes queued / dispatched ────────────────


def test_queue_writes_queued_and_dispatched_with_delay(tmp_path):
    clock = [1000.0]
    (tmp_path / turn_metrics.STATE_FILE).write_text(json.dumps({"state": "closed"}))
    q = chat_queue.ChatQueue(tmp_path, clock_fn=lambda: clock[0])
    job, _ = q.enqueue(10, 7, "привет")
    clock[0] = 1012.5

    async def runner(_bot, _job):
        return True

    try:
        asyncio.run(chat_queue.drain(object(), q, runner))
    finally:
        chat_queue.reset()
    evs = _events(tmp_path)
    queued = [e for e in evs if e["kind"] == "queued"]
    disp = [e for e in evs if e["kind"] == "dispatched"]
    assert queued == [
        {
            "ts": queued[0]["ts"],
            "kind": "queued",
            "job_id": job.id,
            "turn_state": "closed",
        }
    ]
    assert disp[0]["job_id"] == job.id
    assert disp[0]["delay_s"] == 12.5
    assert disp[0]["turn_state_at_queue"] == "closed"


# ── /work and doctor show it ──────────────────────────────────────────────


def _settings(tmp_path):
    s = Settings(
        telegram_bot_token="t",
        deepgram_api_key="d",
        vault_path=tmp_path / "vault",
        runtime_dir=tmp_path / "rt",
        _env_file=None,
    )
    s.runtime_dir.mkdir(parents=True, exist_ok=True)
    return s


class _Idle:
    def is_turn_active(self):
        return False

    def is_pane_turn_active(self):
        return False


def test_work_report_shows_the_summary(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    monkeypatch.setattr(work.runtime, "get_session", lambda _s: _Idle())
    monkeypatch.setattr(work.runtime, "get_cron_session", lambda _s: _Idle())
    turn_metrics.append_event(s.runtime_dir, "queued", ts=990.0, turn_state="closed")
    text = work.build_work_report(s, now=1000.0)
    assert "Застревания за сутки" in text
    assert "при закрытом ходе: <b>1</b>" in text


def test_doctor_check_is_info_and_warns(tmp_path):
    ok = doctor.check_turn_metrics(tmp_path, now=1000.0)
    assert ok.ok and "⚠️" not in ok.detail and "Застревания" in ok.detail
    turn_metrics.append_event(
        tmp_path, "dispatched", ts=999.0, delay_s=30.0, turn_state_at_queue="closed"
    )
    warn = doctor.check_turn_metrics(tmp_path, now=1000.0)
    assert warn.ok  # never fails the doctor
    assert warn.detail.startswith("⚠️")
