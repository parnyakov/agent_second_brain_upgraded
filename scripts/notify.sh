#!/bin/bash
# Send a one-line alert to the admin Telegram chat. Standalone (only needs
# curl + .env), so it works as a systemd OnFailure handler even when the bot
# and watchdog are down. Usage: notify.sh "message"
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# C8: DBRAIN_ENV_FILE lets one shared code checkout alert through whichever
# instance's bot token/chat id the caller names. Unset → $PROJECT_DIR/.env,
# exactly as before.
ENV_FILE="${DBRAIN_ENV_FILE:-$PROJECT_DIR/.env}"
# B3: this file is NO LONGER sourced.
#
# With the templated units $DBRAIN_ENV_FILE can point at another instance's
# /etc/dbrain/<inst>/.env — a file owned and writable by that unprivileged
# user. `.`/`source` EXECUTES its contents, so anything placed there would
# run with this script's privileges. Extracting the few keys we actually use
# with grep + parameter expansion executes nothing.
#
# This also handles the malformed-line case the old comment worried about:
# a bad line simply fails to match, instead of aborting the last-resort
# alert handler before it can reach curl.
env_get() {
    local key="$1" line
    [ -f "$ENV_FILE" ] || return 0
    line=$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "$ENV_FILE" 2>/dev/null | tail -n1) || return 0
    [ -n "$line" ] || return 0
    line="${line#*=}"
    line="${line%$'\r'}"                       # tolerate CRLF
    case "$line" in
        \"*\") line="${line#\"}"; line="${line%\"}" ;;
        \'*\') line="${line#\'}"; line="${line%\'}" ;;
    esac
    printf '%s' "$line"
}

# Process environment wins over the file (systemd EnvironmentFile= already
# put these in the environment for the templated units).
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-$(env_get TELEGRAM_BOT_TOKEN)}"
ALLOWED_USER_IDS="${ALLOWED_USER_IDS:-$(env_get ALLOWED_USER_IDS)}"
# Deliberately DBRAIN_RUNTIME_DIR, not RUNTIME_DIR: sourcing exported both,
# but the script below only ever read DBRAIN_RUNTIME_DIR, so honouring
# RUNTIME_DIR here would silently move the debounce stamp dir. The
# $HOME/.dbrain fallback further down already gives each instance its own.
DBRAIN_RUNTIME_DIR="${DBRAIN_RUNTIME_DIR:-$(env_get DBRAIN_RUNTIME_DIR)}"
DBRAIN_NOTIFY_COOLDOWN="${DBRAIN_NOTIFY_COOLDOWN:-$(env_get DBRAIN_NOTIFY_COOLDOWN)}"

MSG="${1:-d-brain alert}"
# Defensive default: under `set -u`, referencing an ENTIRELY unset
# ALLOWED_USER_IDS (missing/misnamed .env, migration that dropped the file)
# aborts the script before it can even report "missing token or chat id"
# below — this is exactly the class of failure seen live 2026-08-14
# ("notify.sh: line 22: ALLOWED_USER_IDS: unbound variable"). ``:-`` makes a
# genuinely-missing var behave the same as an empty one, which the check
# below already handles.
CHAT_ID="${ALLOWED_USER_IDS:-}"
CHAT_ID="${CHAT_ID//[\[\] ]/}"  # strip brackets/spaces
CHAT_ID="${CHAT_ID%%,*}"  # first id only

if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ -z "$CHAT_ID" ]; then
    echo "notify: missing token or chat id" >&2
    exit 0
fi

# Debounce identical alerts: a unit crash-loop must not spam Telegram. Transient
# faults self-heal silently; we only want a rare, meaningful signal. Keyed by
# message text so distinct faults still alert independently.
RUNTIME_DIR="${DBRAIN_RUNTIME_DIR:-$HOME/.dbrain}"
COOLDOWN="${DBRAIN_NOTIFY_COOLDOWN:-1800}"  # seconds (30 min)
mkdir -p "$RUNTIME_DIR" 2>/dev/null || true
STAMP="$RUNTIME_DIR/notify.$(printf '%s' "$MSG" | cksum | cut -d' ' -f1).stamp"
NOW=$(date +%s)
LAST=$(cat "$STAMP" 2>/dev/null || echo 0)
case "$LAST" in *[!0-9]*) LAST=0 ;; esac
if [ "$((NOW - LAST))" -lt "$COOLDOWN" ]; then
    exit 0  # within cooldown — stay silent
fi
echo "$NOW" >"$STAMP" 2>/dev/null || true

curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
    -d "chat_id=$CHAT_ID" -d "text=$MSG" >/dev/null || true
