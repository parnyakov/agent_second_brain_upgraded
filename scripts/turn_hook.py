#!/usr/bin/env python3
"""Claude Code hook -> <runtime_dir>/turn-state.json ("turn open/closed").

Invoked by the CLI as ``<python> turn_hook.py <runtime_dir>`` with the hook
payload as JSON on stdin. Stdlib only (fast start), never prints to stdout
(UserPromptSubmit stdout is injected into the model context), always exits 0.
"""

import json
import os
import sys
import time

STATE_FILE = "turn-state.json"
ERR_FILE = "turn-hook.err"
ERR_MAX = 64 * 1024


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
        state["origin"] = (
            "task-notification"
            if prompt.lstrip().startswith("<task-notification>")
            else "user"
        )
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
