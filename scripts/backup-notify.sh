#!/bin/bash
# Backup alert channel — deliberately NOT the d-brain bot.
#
# scripts/notify.sh is already independent of the d_brain *code* (curl + .env),
# but it still speaks through the one Telegram bot token the whole system runs
# on. If that token is what is broken — revoked, flood-limited, or answering a
# second poller with 409 — every alert path fails together, which is precisely
# the failure mode we are alerting about.
#
# This uses a SECOND bot token, from a second @BotFather bot, stored outside
# the repo. No python, no .env, no imports: bash + curl. A bug anywhere in
# d_brain cannot reach it.
#
#   Usage: backup-notify.sh "message"
#
# Configuration (this is the part only you can do) —
#   1. Talk to @BotFather, /newbot, e.g. "d-brain backup alerts".
#   2. Open a chat with the new bot and press Start, ONCE. A bot cannot
#      message a user who has never started it, so an unstarted backup bot is
#      a backup channel that silently does not exist.
#   3. Write ~/.dbrain/secrets/backup-notify.env, chmod 600:
#          BACKUP_BOT_TOKEN=123456:AA...
#          BACKUP_CHAT_ID=123456789
#      (BACKUP_CHAT_ID is the same Telegram user id the main bot already uses.)
#
# Until that file exists this exits 0 and stays quiet — an unconfigured backup
# channel must never be the reason a recovery script aborts.
set -uo pipefail

CONF="${DBRAIN_BACKUP_NOTIFY_ENV:-$HOME/.dbrain/secrets/backup-notify.env}"
MSG="${1:-d-brain backup alert}"

if [ ! -f "$CONF" ]; then
    echo "backup-notify: not configured ($CONF missing) — see header" >&2
    exit 0
fi

set -a
# shellcheck disable=SC1090
. "$CONF" 2>/dev/null
set +a

if [ -z "${BACKUP_BOT_TOKEN:-}" ] || [ -z "${BACKUP_CHAT_ID:-}" ]; then
    echo "backup-notify: $CONF has no BACKUP_BOT_TOKEN/BACKUP_CHAT_ID" >&2
    exit 0
fi

# Own debounce, own stamp namespace: the backup channel must not fall silent
# just because the primary one already alerted about the same fault.
RUNTIME_DIR="${DBRAIN_RUNTIME_DIR:-$HOME/.dbrain}"
COOLDOWN="${DBRAIN_BACKUP_NOTIFY_COOLDOWN:-1800}"
mkdir -p "$RUNTIME_DIR" 2>/dev/null || true
STAMP="$RUNTIME_DIR/backup-notify.$(printf '%s' "$MSG" | cksum | cut -d' ' -f1).stamp"
NOW=$(date +%s)
LAST=$(cat "$STAMP" 2>/dev/null || echo 0)
case "$LAST" in *[!0-9]*) LAST=0 ;; esac
if [ "$((NOW - LAST))" -lt "$COOLDOWN" ]; then
    exit 0
fi
echo "$NOW" >"$STAMP" 2>/dev/null || true

curl -s -m 20 -X POST "https://api.telegram.org/bot$BACKUP_BOT_TOKEN/sendMessage" \
    -d "chat_id=$BACKUP_CHAT_ID" -d "text=$MSG" >/dev/null || true
