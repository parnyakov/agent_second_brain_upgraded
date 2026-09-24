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
