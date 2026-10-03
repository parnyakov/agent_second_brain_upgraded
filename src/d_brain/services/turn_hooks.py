"""Generate the Claude Code settings file that wires turn-state hooks.

The CLI is started with ``--settings <runtime_dir>/turn-hooks.json``; every
hook runs ``scripts/turn_hook.py <runtime_dir>`` which maintains
``<runtime_dir>/turn-state.json`` (see that script for the rules).
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

HOOK_SETTINGS_NAME = "turn-hooks.json"
HOOK_TIMEOUT = 5
_PLAIN_EVENTS = ("UserPromptSubmit", "Stop", "StopFailure", "SessionStart")
_TOOL_EVENTS = ("PreToolUse", "PostToolUse")


def hook_script_path() -> Path:
    # src/d_brain/services/turn_hooks.py -> repo_root/scripts/turn_hook.py
    return Path(__file__).resolve().parents[3] / "scripts" / "turn_hook.py"


def build_hook_settings(runtime_dir: Path, python: str, script: Path) -> dict:
    command = " ".join(
        shlex.quote(p) for p in (python, str(script), str(Path(runtime_dir).resolve()))
    )
    entry = {"type": "command", "command": command, "timeout": HOOK_TIMEOUT}
    hooks: dict[str, list] = {ev: [{"hooks": [dict(entry)]}] for ev in _PLAIN_EVENTS}
    for ev in _TOOL_EVENTS:
        hooks[ev] = [{"matcher": "*", "hooks": [dict(entry)]}]
    return {"hooks": hooks, "env": {"DISABLE_AUTOUPDATER": "1"}}


def write_hook_settings(runtime_dir: Path, python: str = sys.executable) -> Path | None:
    """Write ``<runtime_dir>/turn-hooks.json`` atomically and return its path.

    Returns None (writes nothing) when the hook script is not installed.
    """
    script = hook_script_path()
    if not script.is_file():
        return None
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    target = runtime_dir / HOOK_SETTINGS_NAME
    tmp = runtime_dir / f".{HOOK_SETTINGS_NAME}.{os.getpid()}.tmp"
    settings = build_hook_settings(runtime_dir, python, script)
    tmp.write_text(json.dumps(settings, indent=2))
    os.replace(tmp, target)
    return target
