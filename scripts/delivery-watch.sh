#!/bin/bash
# Out-of-band health check for the delivery channel.
#
# Runs from its own systemd timer, in its own process, using nothing from the
# d_brain package. That independence is the entire point: on 2026-08-20 every
# component that could have noticed the outage was downstream of the component
# that was broken, so the first detection was a human, hours late.
#
# Only POSITIVE fault signals — never "it has been quiet for a while". Quiet is
# normal at night, and an alerter that cries wolf at 3am gets muted, which is
# the same as not having one.
#
#   F1  dbrain-bot.service is not active
#   F2  dbrain-watchdog.service is not active
#   F3  the watchdog's STATUS.md has not been touched in STALE_AFTER seconds
#       (the watchdog ticks every 15s; stale ⇒ it is dead or wedged)
#   F4  ask-health.json shows FAIL_STREAK or more delivered-nothing turns
#   F5  more than one live `python -m d_brain` (the orphan that survived the
#       10:51:40 UTC restart on 2026-08-20)
#
# Alerts go out on BOTH channels: notify.sh (primary bot) and backup-notify.sh
# (second bot, if configured). Exit 0 when healthy, 1 when a fault was found
# and reported.
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_DIR="${DBRAIN_RUNTIME_DIR:-$HOME/.dbrain}"
STALE_AFTER="${DBRAIN_STATUS_STALE_AFTER:-600}"   # 10 min
FAIL_STREAK="${DBRAIN_FAIL_STREAK:-3}"

# C6: unit names and systemd scope are env-configurable; defaults are the
# names this script had hardcoded, so an install that sets nothing behaves
# exactly as before. `is-active`/`show` need no privileges in either scope
# (plan §3.5), so there is no sudo here.
BOT_UNIT="${DBRAIN_BOT_UNIT:-dbrain-bot.service}"
WATCHDOG_UNIT="${DBRAIN_WATCHDOG_UNIT:-dbrain-watchdog.service}"
SYSTEMD_SCOPE="${DBRAIN_SYSTEMD_SCOPE:-user}"

if [ "$SYSTEMD_SCOPE" = "user" ]; then
    SCOPE_FLAG="--user"
else
    SCOPE_FLAG="--system"
fi

faults=()

unit_active() { systemctl "$SCOPE_FLAG" is-active --quiet "$1"; }

unit_active "$BOT_UNIT" || faults+=("$BOT_UNIT не активен")
unit_active "$WATCHDOG_UNIT" || faults+=("$WATCHDOG_UNIT не активен")

STATUS="$RUNTIME_DIR/STATUS.md"
if [ -f "$STATUS" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$STATUS" 2>/dev/null || date +%s) ))
    if [ "$age" -gt "$STALE_AFTER" ]; then
        # The fault text must be BYTE-STABLE across runs. notify.sh and
        # backup-notify.sh both key their 30-min debounce on a cksum of the
        # message, so embedding the live age here would mint a brand-new
        # "message" every 5 minutes and alert forever — the exact cry-wolf
        # behaviour this script's header forbids. Exact age goes to stderr
        # (i.e. the journal) below instead.
        faults+=("watchdog молчит дольше ${STALE_AFTER}s (STATUS.md не обновляется)")
        echo "delivery-watch: STATUS.md age ${age}s (limit ${STALE_AFTER}s)" >&2
    fi
else
    # Right after an install, reboot or restart the watchdog has not written
    # its first STATUS.md yet (the timer's first run can land in the same
    # second). Missing is a fault only once the watchdog has been up longer
    # than STALE_AFTER, otherwise a brand-new user gets a red alert at once.
    # A crash-looping watchdog keeps a fresh start time, so any restart
    # counted by systemd cancels the grace.
    since="$(systemctl "$SCOPE_FLAG" show -p ActiveEnterTimestamp --value "$WATCHDOG_UNIT" 2>/dev/null || true)"
    restarts="$(systemctl "$SCOPE_FLAG" show -p NRestarts --value "$WATCHDOG_UNIT" 2>/dev/null || true)"
    since_s=0
    [ -n "$since" ] && since_s="$(date -d "$since" +%s 2>/dev/null || echo 0)"
    if [ "$since_s" -eq 0 ] || [ "${restarts:-0}" != "0" ] \
       || [ $(( $(date +%s) - since_s )) -gt "$STALE_AFTER" ]; then
        faults+=("нет $STATUS — watchdog ни разу не тикнул")
    fi
fi

HEALTH="$RUNTIME_DIR/ask-health.json"
if [ -f "$HEALTH" ]; then
    streak=$(grep -o '"fail_streak"[: ]*[0-9]*' "$HEALTH" 2>/dev/null | grep -o '[0-9]*$')
    case "${streak:-0}" in
        ''|*[!0-9]*) streak=0 ;;
    esac
    if [ "$streak" -ge "$FAIL_STREAK" ]; then
        faults+=("$streak ответов подряд не доставлено")
    fi
fi

# The python child only. `-x` makes the WHOLE command line have to match, and
# the leading '.*/' excludes the `uv run python -m d_brain` wrapper (no
# "/python" in it) — otherwise a healthy install always reads as two.
#
# C6: `-u <me>` scopes the count to THIS instance's user. Without it, once a
# second instance runs under another account, each install would see the
# other's healthy bot and report a phantom "orphan process" (F5) forever.
# Single-instance result is unchanged: the only matching process is our own.
n_bots=$(pgrep -u "$(id -un)" -cxf '.*/python[0-9.]* -m d_brain' 2>/dev/null || echo 0)
if [ "${n_bots:-0}" -gt 1 ]; then
    faults+=("$n_bots процессов бота одновременно (осиротевший процесс)")
fi

if [ "${#faults[@]}" -eq 0 ]; then
    exit 0
fi

if [ "$SYSTEMD_SCOPE" = "user" ]; then
    JOURNAL_HINT="journalctl --user -u $BOT_UNIT -n 100"
else
    JOURNAL_HINT="journalctl -u $BOT_UNIT -n 100"
fi

MSG="🔴 d-brain: канал доставки под вопросом
$(printf -- '- %s\n' "${faults[@]}")
$JOURNAL_HINT"

"$PROJECT_DIR/scripts/notify.sh" "$MSG" || true
"$PROJECT_DIR/scripts/backup-notify.sh" "$MSG" || true
echo "$MSG" >&2
exit 1
