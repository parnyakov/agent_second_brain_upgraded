#!/usr/bin/env python3
"""Claude Code hook -> <runtime_dir>/turn-state.json + turn-ledger.jsonl.

Invoked by the CLI as ``<python> turn_hook.py <runtime_dir>`` with the hook
payload as JSON on stdin. Stdlib only (fast start), never prints to stdout
(UserPromptSubmit stdout is injected into the model context), always exits 0.

Two outputs:

* ``turn-state.json`` — "is the CLI turn open right now" (busy detection,
  the steer gate). Atomic rewrite.
* ``turn-ledger.jsonl`` — the append-only turn ledger (reply binding, see
  ``d_brain.services.turn_ledger``). One JSON line per UserPromptSubmit /
  Stop / StopFailure / SessionStart. Field shapes verified live on Claude
  Code 2.1.312 (2026-10-29): UserPromptSubmit carries ``prompt_id`` and the
  full ``prompt``; Stop carries the same ``prompt_id``,
  ``last_assistant_message`` and ``background_tasks``; a message typed into
  a running turn fires another UserPromptSubmit with the SAME ``prompt_id``;
  an Escape interrupt fires no Stop; ``/compact`` fires
  SessionStart(source=compact) with the session id unchanged.
"""

import json
import os
import re
import sys
import time

STATE_FILE = "turn-state.json"
LEDGER_FILE = "turn-ledger.jsonl"
LEDGER_MAX = 8 * 1024 * 1024
ERR_FILE = "turn-hook.err"
ERR_MAX = 64 * 1024
# Keep in sync with d_brain.services.turn_ledger.TAG_RE (a test pins it).
TAG_RE = re.compile(
    r'<dbrain-turn rid="([A-Za-z0-9_-]{1,64})" kind="(ask|steer|maint)"\s*/>'
)


# The marker-path instruction (claude_session._send_prompt, ledger off): its
# rid binds a legacy ask to its CLI turn for the delivery metric only.
LEGACY_RE = re.compile(
    r"wrap your ENTIRE reply between a line containing only "
    r"<<<R:([A-Za-z0-9_-]{1,64})>>>"
)


def parse_tag(prompt: str) -> tuple[str, str] | None:
    """The LAST bot tag in the prompt: the bot appends its own at the end, so
    a tag quoted earlier in the text (an old message pasted back) loses.
    Falls back to the legacy marker instruction (kind "ask")."""
    found = TAG_RE.findall(prompt or "")
    if found:
        return found[-1]
    legacy = LEGACY_RE.findall(prompt or "")
    return (legacy[-1], "ask") if legacy else None


def _append_ledger(runtime_dir: str, record: dict) -> None:
    path = os.path.join(runtime_dir, LEDGER_FILE)
    try:
        if os.path.getsize(path) > LEDGER_MAX:
            os.replace(path, path + ".1")
    except OSError:
        pass
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def ledger_record(payload: dict, now: float) -> dict | None:
    """The ledger line for one hook payload, or None for events it skips."""
    event = payload.get("hook_event_name") or ""
    base = {
        "ts": now,
        "session_id": payload.get("session_id") or "",
        "prompt_id": payload.get("prompt_id") or "",
    }
    if event == "UserPromptSubmit":
        prompt = payload.get("prompt")
        prompt = prompt if isinstance(prompt, str) else ""
        tag = parse_tag(prompt)
        return {
            "ev": "submit",
            **base,
            "rid": tag[0] if tag else None,
            "kind": tag[1] if tag else None,
            "origin": (
                "task-notification"
                if prompt.lstrip().startswith("<task-notification>")
                else "user"
            ),
        }
    if event == "Stop":
        text = payload.get("last_assistant_message")
        bg = payload.get("background_tasks")
        rec = {
            "ev": "stop",
            **base,
            "text": text if isinstance(text, str) else None,
            "background_tasks": len(bg) if isinstance(bg, list) else 0,
        }
        if rec["text"] is None:
            # Older CLI: no last_assistant_message. The reader resolves the
            # final message from the transcript instead (lazily, not here —
            # a hook must stay fast).
            rec["transcript_path"] = payload.get("transcript_path") or ""
        return rec
    if event == "StopFailure":
        err = payload.get("error") or payload.get("error_type") or payload.get("reason")
        return {"ev": "stop_failure", **base, "error": str(err or "")[:300]}
    if event == "SessionStart":
        return {"ev": "session_start", **base, "source": payload.get("source") or ""}
    return None


def _find_cli_pid() -> int:
    ppid = os.getppid()
    pid = ppid
    for _ in range(32):
        if pid <= 1:
            break
        try:
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip()
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
            argv0 = argv[0].decode(errors="replace") if argv else ""
            if "claude" in comm or "claude" in os.path.basename(argv0):
                return pid
            # node-launched CLI: argv[1] is the claude script
            if len(argv) > 1 and b"claude" in argv[1]:
                return pid
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return ppid


def _read_state(path: str) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_state(runtime_dir: str, path: str, state: dict) -> None:
    tmp = os.path.join(runtime_dir, f".{STATE_FILE}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def apply(runtime_dir: str, payload: dict, now: float) -> None:
    if payload.get("agent_id"):
        return
    try:
        _apply_state(runtime_dir, payload, now)
    finally:
        # The ledger is written even when the state update failed: it is the
        # reply path, the state file is only the busy signal.
        record = ledger_record(payload, now)
        if record is not None:
            _append_ledger(runtime_dir, record)


def _apply_state(runtime_dir: str, payload: dict, now: float) -> None:
    event = payload.get("hook_event_name") or ""
    path = os.path.join(runtime_dir, STATE_FILE)
    prev = _read_state(path)
    session_id = payload.get("session_id")

    if event in ("PreToolUse", "PostToolUse"):
        if prev is None:
            return
        prev["last_event_ts"] = now
        if session_id:
            prev["session_id"] = session_id
        _write_state(runtime_dir, path, prev)
        return

    if event == "UserPromptSubmit":
        new_state = "open"
    elif event == "SessionStart" and payload.get("source") == "compact":
        # Auto-compaction can happen INSIDE a running turn, which then goes
        # on: a compact is not the end of a turn. Only the session id moves.
        if prev is None:
            return
        prev["last_event_ts"] = now
        if session_id:
            prev["session_id"] = session_id
        _write_state(runtime_dir, path, prev)
        return
    elif event in ("Stop", "StopFailure", "SessionStart"):
        new_state = "closed"
    else:
        return

    state = dict(prev) if prev else {}
    state.setdefault("origin", None)
    state.setdefault("background_tasks", 0)
    if state.get("state") != new_state or "ts" not in state:
        state["ts"] = now
    state["state"] = new_state
    state["event"] = event
    state["last_event_ts"] = now
    if session_id or "session_id" not in state:
        state["session_id"] = session_id or ""
    state["cli_pid"] = _find_cli_pid()

    if event == "UserPromptSubmit":
        prompt = payload.get("prompt")
        prompt = prompt if isinstance(prompt, str) else ""
        prompt_id = payload.get("prompt_id") or ""
        joined = (
            prev is not None
            and prev.get("state") == "open"
            and bool(prompt_id)
            and prev.get("prompt_id") == prompt_id
        )
        if not joined:
            state["origin"] = (
                "task-notification"
                if prompt.lstrip().startswith("<task-notification>")
                else "user"
            )
            state["kinds"] = []
        tag = parse_tag(prompt)
        kinds = list(state.get("kinds") or [])
        kinds.append(tag[1] if tag else "")
        state["kinds"] = kinds[-16:]
        state["prompt_id"] = prompt_id
    elif event in ("Stop", "StopFailure"):
        bg = payload.get("background_tasks")
        state["background_tasks"] = len(bg) if isinstance(bg, list) else 0
    elif event == "SessionStart":
        state["background_tasks"] = 0
        state["origin"] = None

    _write_state(runtime_dir, path, state)


def _log_error(runtime_dir: str, exc: BaseException) -> None:
    try:
        path = os.path.join(runtime_dir, ERR_FILE)
        try:
            if os.path.getsize(path) > ERR_MAX:
                os.remove(path)
        except OSError:
            pass
        with open(path, "a") as f:
            f.write(f"{time.time():.3f} {type(exc).__name__}: {exc}"[:500] + "\n")
    except Exception:
        pass


def main() -> None:
    runtime_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if not runtime_dir or not os.path.isdir(runtime_dir):
            return
        payload = json.loads(sys.stdin.read() or "{}")
        if isinstance(payload, dict):
            apply(runtime_dir, payload, time.time())
    except BaseException as exc:  # noqa: BLE001 — a hook must never fail
        if runtime_dir:
            _log_error(runtime_dir, exc)


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)
