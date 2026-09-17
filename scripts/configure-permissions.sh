#!/usr/bin/env bash
# Permission profile of the agent on this server.
#
#   bash scripts/configure-permissions.sh status|max|standard
#   (or: dbrain permissions status|max|standard)
#
# The boundary is the server and this Linux user: the bot never runs as root,
# secrets stay in .env (chmod 600), and the private vault repository is the
# only place memory leaves the server.
#
# max       for a DEDICATED server that runs nothing but the agent:
#           * Codex threads use sandbox danger-full-access, no approvals;
#           * Claude Code terminal sessions start in bypassPermissions
#             (the Telegram brain already runs without prompts);
#           * passwordless sudo for this user, so the agent can install
#             packages and configure the server without asking you to type.
# standard  the agent works freely inside the project, vault, its schedules
#           and systemd --user services; Codex writes only inside the vault
#           workspace; no passwordless sudo, system changes are done by you.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env"
SUDOERS_FILE="${DBRAIN_SUDOERS_FILE:-/etc/sudoers.d/90-dbrain-${USER}}"
CLAUDE_SETTINGS="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/settings.json"
CODEX_CONFIG="${CODEX_HOME:-$HOME/.codex}/config.toml"

env_set() {
    local key="$1" value="$2" tmp
    umask 077
    [ -f "$ENV_FILE" ] || cp "$PROJECT_DIR/.env.example" "$ENV_FILE"
    tmp="$(mktemp "$ENV_FILE.XXXXXX")"
    grep -v -E "^$key=" "$ENV_FILE" > "$tmp" || true
    printf '%s=%s\n' "$key" "$value" >> "$tmp"
    chmod 600 "$tmp"
    mv "$tmp" "$ENV_FILE"
}

env_get() {
    [ -f "$ENV_FILE" ] && grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- || true
}

write_claude_settings() {
    local mode="$1"
    mkdir -p "$(dirname "$CLAUDE_SETTINGS")"
    MODE="$mode" python3 - "$CLAUDE_SETTINGS" <<'PY'
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
data = {}
if path.exists():
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        path.replace(path.with_suffix(".json.bak"))
        data = {}
perms = data.setdefault("permissions", {})
mode = os.environ["MODE"]
perms["defaultMode"] = mode
if mode == "bypassPermissions":
    data["skipDangerousModePermissionPrompt"] = True
else:
    data.pop("skipDangerousModePermissionPrompt", None)
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

write_codex_config() {
    local sandbox="$1"
    mkdir -p "$(dirname "$CODEX_CONFIG")"
    SANDBOX="$sandbox" python3 - "$CODEX_CONFIG" <<'PY'
import os, re, sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8") if path.exists() else ""
begin, end = "# dbrain-permissions:begin", "# dbrain-permissions:end"
text = re.sub(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", "", text, flags=re.S)
head = text.split("\n[", 1)[0]
user_keys = {k for k in ("approval_policy", "sandbox_mode") if re.search(rf"^\s*{k}\s*=", head, re.M)}
lines = [begin]
if "approval_policy" not in user_keys:
    lines.append('approval_policy = "never"')
if "sandbox_mode" not in user_keys:
    lines.append(f'sandbox_mode = "{os.environ["SANDBOX"]}"')
lines.append(end)
path.write_text("\n".join(lines) + "\n" + text.lstrip("\n"), encoding="utf-8")
if user_keys:
    print("  note: kept your own " + ", ".join(sorted(user_keys)) + " in " + str(path))
PY
}

enable_sudo() {
    local tmp
    tmp="$(mktemp)"
    printf '# Managed by d-brain (dbrain permissions). Remove with: dbrain permissions standard\n%s ALL=(ALL) NOPASSWD: ALL\n' "$USER" > "$tmp"
    if ! sudo visudo -cf "$tmp" >/dev/null; then
        rm -f "$tmp"
        echo "sudoers validation failed; nothing changed" >&2
        return 1
    fi
    # sudo runs install as root, so the file is root-owned; sudo refuses
    # sudoers.d files that are writable by anyone else.
    sudo install -m 0440 "$tmp" "$SUDOERS_FILE"
    rm -f "$tmp"
}

disable_sudo() {
    if sudo -n test -e "$SUDOERS_FILE" 2>/dev/null || [ -e "$SUDOERS_FILE" ]; then
        sudo rm -f "$SUDOERS_FILE"
    fi
}

status() {
    local profile sandbox sudo_state="нет"
    profile="$(env_get DBRAIN_PERMISSION_PROFILE)"
    sandbox="$(env_get DBRAIN_CODEX_SANDBOX)"
    [ -e "$SUDOERS_FILE" ] && sudo_state="да"
    echo "Профиль прав: ${profile:-не выбран}"
    echo "  Codex sandbox для новых диалогов: ${sandbox:-по умолчанию (workspace-write)}"
    echo "  sudo без пароля для $USER: $sudo_state"
    echo "  Claude Code settings: $CLAUDE_SETTINGS"
}

apply() {
    local profile="$1"
    case "$profile" in
        max)
            env_set DBRAIN_PERMISSION_PROFILE max
            env_set DBRAIN_CODEX_SANDBOX danger-full-access
            write_claude_settings bypassPermissions
            write_codex_config danger-full-access
            enable_sudo
            ;;
        standard)
            env_set DBRAIN_PERMISSION_PROFILE standard
            env_set DBRAIN_CODEX_SANDBOX workspace-write
            write_claude_settings acceptEdits
            write_codex_config workspace-write
            disable_sudo
            ;;
    esac
    echo "[OK] Профиль прав: $profile"
    echo "     Действует с нового диалога: отправьте боту /new."
}

case "${1:-status}" in
    status) status ;;
    max|standard) apply "$1" ;;
    *) echo "Usage: dbrain permissions status|max|standard" >&2; exit 2 ;;
esac
