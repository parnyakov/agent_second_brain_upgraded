# ruff: noqa: E501
"""Safety gates for the nightly tmux cleanup."""

import gzip
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/server_cleanup.py"
SPEC = importlib.util.spec_from_file_location("server_cleanup", SCRIPT)
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


def session(**changes):
    value = dict(name="old_agent", created=1, activity=1, attached=0, windows=1)
    value.update(changes)
    return value


def test_protected_attached_and_recent_sessions_are_kept():
    now = cleanup.WEEK + 100
    with patch.object(cleanup, "idle_prompt", return_value=True):
        skip = cleanup.reason_to_skip
        assert skip(session(name="dbrain_chat"), {}, now) == "core bot session"
        assert skip(session(), {"old_agent": 1}, now) == "protected by user"
        assert skip(session(attached=1), {}, now) == "attached"
        assert skip(session(activity=now - 10), {}, now) == "recent activity"
        assert cleanup.reason_to_skip(session(), {}, now) is None


def test_unanswered_prompt_and_background_work_are_kept():
    pane = subprocess.CompletedProcess([], 0, "0\tnode\n", "")
    unanswered = subprocess.CompletedProcess([], 0, "❯ закоммить всё это\n", "")
    busy = subprocess.CompletedProcess(
        [], 0,
        "Working (36m) · 1 background terminal running\n› Ask Codex to do anything\n",
        ""
    )
    idle = subprocess.CompletedProcess([], 0, "Done\n› Ask Codex to do anything\n", "")
    for captured, expected in ((unanswered, False), (busy, False), (idle, True)):
        with patch.object(cleanup, "command", side_effect=[pane, captured]):
            assert cleanup.idle_prompt("old_agent") is expected


def test_old_subagent_log_is_verified_and_kept_as_archive(tmp_path):
    session_id = "e92e3859-18b5-4367-ba7a-75e464114e77"
    folder = tmp_path / ".claude/projects/project" / session_id / "subagents"
    folder.mkdir(parents=True)
    source = folder / "agent-test.jsonl"
    payload = b'{"event":"test"}\n' * 80_000
    source.write_bytes(payload)
    os.utime(source, (1, 1))
    with (patch.object(cleanup.Path, "home", return_value=tmp_path),
          patch.object(cleanup, "active_claude_ids", return_value=set()),
          patch.object(cleanup, "open_inodes", return_value=set())):
        count, original, archived = cleanup.archive_old_subagent_logs(
            cleanup.WEEK + 2, False)
    assert count == 1
    assert original == len(payload)
    assert archived < original
    assert not source.exists()
    with gzip.open(source.with_suffix(".jsonl.gz"), "rb") as result:
        assert result.read() == payload


def test_finished_job_temp_needs_no_live_process(tmp_path):
    job = tmp_path / ".claude/jobs/97e59315"
    temp = job / "tmp"
    temp.mkdir(parents=True)
    file = temp / "result.txt"
    file.write_bytes(b"old data")
    (job / "state.json").write_text(json.dumps(
        {"state": "done", "updatedAt": "1970-01-01T00:00:01Z"}
    ))
    os.utime(file, (1, 1))
    os.utime(temp, (1, 1))
    with (patch.object(cleanup.Path, "home", return_value=tmp_path),
          patch.object(cleanup, "active_claude_ids",
                       side_effect=[{"97e59315-0000-0000-0000-000000000000"}, set()]),
          patch.object(cleanup, "open_inodes", return_value=set()),
          patch.object(cleanup, "process_cwds", return_value=[])):
        assert cleanup.prune_finished_job_tmp(cleanup.WEEK + 2, False) == (0, 0)
        count, size = cleanup.prune_finished_job_tmp(cleanup.WEEK + 2, False)
    assert count == 1 and size > 0
    assert not temp.exists()
    assert (job / "state.json").exists()


def test_journal_retention_uses_seven_days():
    result = subprocess.CompletedProcess([], 0, "Vacuuming done, freed 1.0G.\n", "")
    with patch.object(cleanup, "command", return_value=result) as run:
        assert "очищены записи старше 7 дней" in cleanup.vacuum_old_journal(False)
    assert run.call_args.args[:5] == (
        "sudo", "-n", "journalctl", "--rotate", "--vacuum-time=7d"
    )



def proc(pid, ppid, comm, start, cmd="", rss=100 * 1048576):
    return dict(pid=pid, ppid=ppid, comm=comm, start=start, rss=rss, state="S", cmd=cmd or comm)


def test_browsers_close_nightly_except_fresh_agent_work():
    now = 100 * 86400
    table = {
        10: proc(10, 5, "claude", 1, "/usr/bin/claude --session-id x"),
        20: proc(20, 10, "chrome", now - 7200, "chrome --user-data-dir=/tmp/old"),
        21: proc(21, 20, "chrome", now - 7200, "chrome --type=renderer"),
        22: proc(22, 20, "chrome_crashpad", now - 7200),
        30: proc(30, 10, "chrome", now - 600, "chrome --user-data-dir=/tmp/live"),
        31: proc(31, 30, "chrome", now - 600),
        40: proc(40, 1, "headless_shell", now - 300),
        50: proc(50, 1, "Xvfb", now - 4 * 86400, "Xvfb :99"),
        51: proc(51, 1, "Xvfb", now - 86400, "Xvfb :98"),
        52: proc(52, 1, "oosplash", now - 5 * 86400, "oosplash --headless"),
        53: proc(53, 1, "node", now - 5 * 86400, "node /x/playwright/cli.js run-driver"),
        60: proc(60, 1, "postgres", now - 9 * 86400, "postgres -D /tmp/db"),
        61: proc(61, 1, "uv", now - 9 * 86400, "uv run python -m d_brain"),
        62: proc(62, 1, "node", now - 9 * 86400, "node server.js"),
    }
    with patch.object(cleanup, "process_table", return_value=table):
        targets, skipped, noted = cleanup.find_stale_processes(now)
    roots = {t["pids"][0]: t for t in targets}
    assert set(roots) == {20, 40, 50, 52, 53}
    assert sorted(roots[20]["pids"]) == [20, 21, 22]
    assert roots[20]["rss"] == 300 * 1048576
    assert roots[20]["kind"] == "browser" and roots[50]["kind"] == "Xvfb"
    assert sorted(skipped.values()) == ["live agent task", "recent helper"]
    assert [n.split()[0] for n in noted] == ["postgres", "node"]


def test_stop_process_tree_terminates_then_kills_survivors():
    target = dict(pids=[20, 21], starts={20: 5.0, 21: 5.0})
    table = {20: proc(20, 1, "chrome", 5.0), 21: proc(21, 20, "chrome", 5.0)}
    sent = []
    with (patch.object(cleanup, "process_table", return_value=table),
          patch.object(cleanup, "KILL_GRACE", 0),
          patch.object(cleanup.os, "kill", side_effect=lambda pid, sig: sent.append((pid, sig)))):
        cleanup.stop_process_tree(target)
    term, kill = cleanup.signal.SIGTERM, cleanup.signal.SIGKILL
    assert sent == [(20, term), (21, term), (20, kill), (21, kill)]


def test_report_lists_closed_browsers_and_memory():
    report = dict(closed=["coach"], browsers=2, browser_bytes=600 * 1048576,
                  helpers=["Xvfb 50"], helper_bytes=1048576, noted=["postgres 60, 2 МиБ"],
                  pip_files=0, pip_bytes=0, scratchpads=0, scratchpad_bytes=0, job_tmp=0,
                  job_tmp_bytes=0, logs_archived=0, logs_original_bytes=0,
                  logs_archive_bytes=0, uv_result="", journal_result="ok", errors=[],
                  memory_before=1024 * 1048576, memory_after=1700 * 1048576,
                  swap_before=0, swap_after=0, disk_used_percent=50.0, disk_after=0,
                  sessions_left=3, protected_or_active=3, disk_breakdown={})
    text = cleanup.report_text(report)
    assert "Браузеры закрыты: 2 шт., около 600 МиБ." in text
    assert "Xvfb 50" in text and "postgres 60" in text
    assert "Доступная RAM: 1024 МиБ → 1700 МиБ." in text
