"""Turn-state hook script + settings generator + start command wiring."""

import json
import subprocess
import sys
from pathlib import Path

from d_brain.services import turn_hooks
from d_brain.services.claude_session import ClaudeSession

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "turn_hook.py"


def run_hook(rt: Path, payload, raw: str | None = None):
    data = raw if raw is not None else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(rt)],
        input=data, capture_output=True, text=True, timeout=10,
    )


def state(rt: Path) -> dict:
    return json.loads((rt / "turn-state.json").read_text())


def ev(name, **kw):
    return {"hook_event_name": name, "session_id": "s1", **kw}


def test_user_prompt_opens_turn_with_user_origin(tmp_path):
    r = run_hook(tmp_path, ev("UserPromptSubmit", prompt="привет"))
    assert r.returncode == 0 and r.stdout == ""
    st = state(tmp_path)
    assert st["state"] == "open" and st["origin"] == "user"
    assert st["event"] == "UserPromptSubmit" and st["session_id"] == "s1"
    assert isinstance(st["cli_pid"], int) and st["ts"] == st["last_event_ts"]


def test_stop_with_background_tasks_closes_and_counts(tmp_path):
    run_hook(tmp_path, ev("UserPromptSubmit", prompt="x"))
    bg = [{"id": "af4d", "type": "subagent", "status": "running"},
          {"id": "bxqu", "type": "shell", "status": "running"}]
    run_hook(tmp_path, ev("Stop", background_tasks=bg))
    st = state(tmp_path)
    assert st["state"] == "closed" and st["background_tasks"] == 2


def test_task_notification_opens_with_origin(tmp_path):
    prompt = "  <task-notification>\n<id>a</id>"
    run_hook(tmp_path, ev("UserPromptSubmit", prompt=prompt))
    st = state(tmp_path)
    assert st["state"] == "open" and st["origin"] == "task-notification"
    run_hook(tmp_path, ev("Stop", background_tasks=[]))
    assert state(tmp_path)["state"] == "closed"
    assert state(tmp_path)["background_tasks"] == 0


def test_two_prompts_one_stop_closes(tmp_path):
    run_hook(tmp_path, ev("UserPromptSubmit", prompt="a"))
    ts1 = state(tmp_path)["ts"]
    run_hook(tmp_path, ev("UserPromptSubmit", prompt="b"))
    st = state(tmp_path)
    assert st["state"] == "open" and st["ts"] == ts1  # ts only on state change
    assert st["last_event_ts"] >= ts1
    run_hook(tmp_path, ev("StopFailure"))
    st = state(tmp_path)
    assert st["state"] == "closed" and st["ts"] > ts1


def test_agent_id_events_are_ignored(tmp_path):
    run_hook(tmp_path, ev("UserPromptSubmit", prompt="a"))
    before = state(tmp_path)
    run_hook(tmp_path, ev("Stop", agent_id="af4d", background_tasks=[]))
    run_hook(tmp_path, ev("PreToolUse", agent_id="af4d", tool_name="Bash"))
    assert state(tmp_path) == before


def test_session_start_closes_with_new_session_id(tmp_path):
    run_hook(tmp_path, ev("UserPromptSubmit", prompt="a"))
    run_hook(tmp_path, {"hook_event_name": "SessionStart", "session_id": "s2",
                        "source": "clear"})
    st = state(tmp_path)
    assert st["state"] == "closed" and st["session_id"] == "s2"
    assert st["event"] == "SessionStart"


def test_tool_events_only_pulse(tmp_path):
    run_hook(tmp_path, ev("PreToolUse", tool_name="Bash"))
    assert not (tmp_path / "turn-state.json").exists()  # never created
    run_hook(tmp_path, ev("UserPromptSubmit", prompt="a"))
    before = state(tmp_path)
    r = run_hook(tmp_path, ev("PostToolUse", tool_name="Bash"))
    assert r.stdout == ""
    st = state(tmp_path)
    assert st["state"] == "open" and st["ts"] == before["ts"]
    assert st["event"] == before["event"]
    assert st["last_event_ts"] >= before["last_event_ts"]


def test_broken_stdin_exits_zero_silently(tmp_path):
    r = run_hook(tmp_path, None, raw="{not json")
    assert r.returncode == 0 and r.stdout == ""
    assert not (tmp_path / "turn-state.json").exists()
    assert (tmp_path / "turn-hook.err").exists()


def test_missing_runtime_dir_exits_zero(tmp_path):
    r = run_hook(tmp_path / "nope", ev("Stop"))
    assert r.returncode == 0 and r.stdout == ""


def test_write_hook_settings(tmp_path):
    p = turn_hooks.write_hook_settings(tmp_path, python="/usr/bin/py thon")
    assert p == tmp_path / "turn-hooks.json"
    cfg = json.loads(p.read_text())
    assert cfg["env"] == {"DISABLE_AUTOUPDATER": "1"}
    events = {"UserPromptSubmit", "Stop", "StopFailure", "SessionStart",
              "PreToolUse", "PostToolUse"}
    assert set(cfg["hooks"]) == events
    for name, groups in cfg["hooks"].items():
        h = groups[0]["hooks"][0]
        assert h["type"] == "command" and h["timeout"] == 5
        assert "'/usr/bin/py thon'" in h["command"]
        assert str(SCRIPT) in h["command"] and str(tmp_path) in h["command"]
        assert ("matcher" in groups[0]) == (name in ("PreToolUse", "PostToolUse"))
    assert not list(tmp_path.glob("*.tmp"))


def test_write_hook_settings_without_script(tmp_path, monkeypatch):
    monkeypatch.setattr(turn_hooks, "hook_script_path", lambda: tmp_path / "x.py")
    assert turn_hooks.write_hook_settings(tmp_path) is None
    assert not (tmp_path / "turn-hooks.json").exists()


def _session(tmp_path):
    from test_claude_session import READY, FakeTmux, _auth
    return ClaudeSession(
        session_name="dbrain_test",
        work_dir=tmp_path / "vault",
        runtime_dir=tmp_path / ".dbrain",
        runner=FakeTmux([READY]),
        which_fn=lambda n: "/opt/bin/claude",
        cli_runner=_auth(True),
    )


def test_start_command_wires_hooks_and_disables_autoupdater(tmp_path):
    s = _session(tmp_path)
    cmd = s._start_command("sid")
    assert " && DISABLE_AUTOUPDATER=1 /opt/bin/claude " in cmd
    assert f"--settings {tmp_path / '.dbrain' / 'turn-hooks.json'}" in cmd
    assert (tmp_path / ".dbrain" / "turn-hooks.json").exists()


def test_start_command_survives_hook_generation_failure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(turn_hooks, "write_hook_settings", boom)
    cmd = _session(tmp_path)._start_command("sid")
    assert "--settings" not in cmd and "DISABLE_AUTOUPDATER=1 " in cmd
