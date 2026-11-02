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


# ── reply ledger (2026-10-29) ───────────────────────────────────────────
# Payload shapes below are the ones captured live from Claude Code 2.1.312
# (throwaway session on a private tmux socket): UserPromptSubmit carries
# prompt_id + prompt, Stop carries the same prompt_id, last_assistant_message
# and background_tasks; a steer shares the running turn's prompt_id.


def ledger(rt: Path) -> list[dict]:
    p = rt / "turn-ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


TAG = '<dbrain-turn rid="abc123" kind="ask"/>'


def test_ledger_records_bound_submit_and_stop(tmp_path):
    r = run_hook(tmp_path, ev("UserPromptSubmit", prompt=f"привет\n\n{TAG}",
                              prompt_id="p1"))
    assert r.returncode == 0 and r.stdout == ""
    run_hook(tmp_path, ev("Stop", prompt_id="p1", last_assistant_message="Ответ",
                          background_tasks=[{"id": "x"}]))
    sub, stop = ledger(tmp_path)
    assert sub["ev"] == "submit" and sub["prompt_id"] == "p1"
    assert (sub["rid"], sub["kind"], sub["origin"]) == ("abc123", "ask", "user")
    assert "prompt" not in sub  # the prompt text is never copied
    assert stop["ev"] == "stop" and stop["prompt_id"] == "p1"
    assert stop["text"] == "Ответ" and stop["background_tasks"] == 1
    assert "transcript_path" not in stop


def test_ledger_takes_the_last_tag_in_a_prompt(tmp_path):
    old = '<dbrain-turn rid="old111" kind="ask"/>'
    new = '<dbrain-turn rid="new222" kind="steer"/>'
    run_hook(tmp_path, ev("UserPromptSubmit", prompt=f"цитата {old}\nтекст\n\n{new}",
                          prompt_id="p1"))
    rec = ledger(tmp_path)[0]
    assert (rec["rid"], rec["kind"]) == ("new222", "steer")


def test_ledger_untagged_and_task_notification(tmp_path):
    run_hook(tmp_path, ev("UserPromptSubmit", prompt="набрал руками", prompt_id="p1"))
    run_hook(tmp_path, ev("UserPromptSubmit", prompt="<task-notification>\n<task-id>x",
                          prompt_id="p2"))
    a, b = ledger(tmp_path)
    assert a["rid"] is None and a["origin"] == "user"
    assert b["rid"] is None and b["origin"] == "task-notification"


def test_ledger_binds_legacy_marker_prompt_for_the_metric(tmp_path):
    prompt = ("вопрос\n\nWhen done, wrap your ENTIRE reply between a line containing "
              "only <<<R:leg42>>> and a line containing only <<<E:leg42>>>.")
    run_hook(tmp_path, ev("UserPromptSubmit", prompt=prompt, prompt_id="p1"))
    assert (ledger(tmp_path)[0]["rid"], ledger(tmp_path)[0]["kind"]) == ("leg42", "ask")


def test_ledger_stop_without_message_points_at_transcript(tmp_path):
    """An older CLI without last_assistant_message: the reader resolves the
    text from the transcript later; the hook stays cheap."""
    run_hook(tmp_path, ev("Stop", prompt_id="p1", transcript_path="/t/x.jsonl"))
    rec = ledger(tmp_path)[0]
    assert rec["text"] is None and rec["transcript_path"] == "/t/x.jsonl"


def test_ledger_stop_failure_and_session_start(tmp_path):
    run_hook(tmp_path, ev("StopFailure", prompt_id="p1", error="rate_limit"))
    run_hook(tmp_path, ev("SessionStart", source="clear"))
    f, s = ledger(tmp_path)
    assert f["ev"] == "stop_failure" and f["error"] == "rate_limit"
    assert s["ev"] == "session_start" and s["source"] == "clear"


def test_ledger_ignores_subagent_and_tool_events(tmp_path):
    run_hook(tmp_path, ev("Stop", agent_id="af4d", prompt_id="p1",
                          last_assistant_message="subagent"))
    run_hook(tmp_path, ev("PreToolUse", tool_name="Bash", prompt_id="p1"))
    assert ledger(tmp_path) == []


def test_ledger_rotates_at_the_size_cap(tmp_path):
    big = tmp_path / "turn-ledger.jsonl"
    big.write_text("x" * (8 * 1024 * 1024 + 10) + "\n")
    run_hook(tmp_path, ev("Stop", prompt_id="p9", last_assistant_message="new"))
    assert (tmp_path / "turn-ledger.jsonl.1").exists()
    assert [r["prompt_id"] for r in ledger(tmp_path)] == ["p9"]


def test_ledger_still_written_when_state_file_is_broken(tmp_path):
    (tmp_path / "turn-state.json").mkdir()  # the state write must fail
    r = run_hook(tmp_path, ev("Stop", prompt_id="p1", last_assistant_message="ok"))
    assert r.returncode == 0 and r.stdout == ""
    assert ledger(tmp_path)[0]["text"] == "ok"


def test_compact_does_not_close_an_open_turn(tmp_path):
    """Auto-compaction inside a running turn: the turn goes on."""
    run_hook(tmp_path, ev("UserPromptSubmit", prompt=TAG, prompt_id="p1"))
    run_hook(tmp_path, {"hook_event_name": "SessionStart", "session_id": "s2",
                        "source": "compact", "prompt_id": "p1"})
    st = state(tmp_path)
    assert st["state"] == "open" and st["session_id"] == "s2"
    assert st["prompt_id"] == "p1"


def test_state_tracks_prompt_id_and_tag_kinds_of_the_open_turn(tmp_path):
    run_hook(tmp_path, ev("UserPromptSubmit", prompt=TAG, prompt_id="p1"))
    steer = '<dbrain-turn rid="s1" kind="steer"/>'
    run_hook(tmp_path, ev("UserPromptSubmit", prompt=f"ещё {steer}", prompt_id="p1"))
    st = state(tmp_path)
    assert st["prompt_id"] == "p1" and st["kinds"] == ["ask", "steer"]
    run_hook(tmp_path, ev("Stop", prompt_id="p1", last_assistant_message="x"))
    maint = '<dbrain-turn rid="m1" kind="maint"/>'
    run_hook(tmp_path, ev("UserPromptSubmit", prompt=maint, prompt_id="p2"))
    st = state(tmp_path)
    assert st["prompt_id"] == "p2" and st["kinds"] == ["maint"]


def test_hook_and_module_tag_regex_agree():
    """scripts/turn_hook.py is stdlib-only and keeps its own copy."""
    import importlib.util

    from d_brain.services import turn_ledger

    spec = importlib.util.spec_from_file_location("turn_hook_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TAG_RE.pattern == turn_ledger.TAG_RE.pattern
    tag = turn_ledger.turn_tag("r_1-x", "maint")
    assert mod.parse_tag(f"a\n\n{tag}") == ("r_1-x", "maint")
