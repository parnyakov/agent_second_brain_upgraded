#!/bin/bash
# Out-of-band health check for the delivery channel.
#
# Runs from its own systemd timer, in its own process, using nothing from the
# d_brain package. That independence is the entire point: during a real
# production outage, every component that could have noticed it was
# downstream of the component that was broken, so the first detection was a
# human, hours late.
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
#       AND no reply has reached Telegram since the last of them
#   F5  more than one live `python -m d_brain` (the orphan that survived the
#       10:51:40 UTC restart on 2026-08-20)
#   F6  a message this install can PROVE it failed to deliver: a reply in the
#       outbox's dead queue, an accepted message retired unanswered, or an
#       accepted message left stranded
#
# Alerts go out on BOTH channels: notify.sh (primary bot) and backup-notify.sh
# (second bot, if configured). Exit 0 when healthy, 1 when a fault was found
# and reported.
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_DIR="${DBRAIN_RUNTIME_DIR:-$HOME/.dbrain}"
STALE_AFTER="${DBRAIN_STATUS_STALE_AFTER:-600}"   # 10 min
FAIL_STREAK="${DBRAIN_FAIL_STREAK:-3}"
# How long a unit may sit in `activating`/`deactivating` before that is a
# fault rather than a transition. Must clear the bot's whole graceful stop
# (`TimeoutStopSec=360`, deploy/systemd/dbrain-bot@.service) plus
# `RestartSec=10` — past that, "it is starting" and "it never came back" are
# the same picture, and only the second one is true.
TRANSITION_GRACE="${DBRAIN_TRANSITION_GRACE:-420}"
# How long an accepted message may sit UNTOUCHED in the inbox before its
# presence is proof of a loss. Mirrors delivery_proof.DEFAULT_STUCK_AFTER
# (7200s) — expressed in minutes here because `find -mmin` reads it. Two
# hours, because one handler legitimately holds its entry for two full turns
# at chat_turn_timeout plus a busy-wait; see that module for the arithmetic.
STUCK_AFTER_MIN="${DBRAIN_STUCK_AFTER_MIN:-120}"

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

# `is-active` is true only for active/reloading — a unit in `deactivating`
# fails it. That used to be a window of a second or two; since 2026-09-22 a
# graceful stop can hold `deactivating` for up to ~5.5 minutes (the bot lets
# the turn in flight finish before it exits), which
# is LONGER than this check's own 5-minute timer. Keeping the shortcut would
# have alerted "бот не активен" on both channels during essentially every
# restart that paid the grace — cry-wolf, which this script's header forbids,
# and about the bot doing exactly what it was told.
#
# So read the state instead: a unit that is on its way up or down is not a
# fault. `failed`, `inactive` and `dead` still are.
#
# But only for as long as a transition can honestly be called one (item 34).
# Accepting `activating`/`deactivating` unconditionally, as the first cut of
# this fix did, silently deleted the fault it was guarding: a bot that hangs
# on shutdown, or that never finishes coming back up, sits in exactly those
# states forever and would never have been reported again. The requirement
# is not "fewer alerts", it is "no FALSE alerts" — a unit stuck mid-
# transition past TRANSITION_GRACE is the real "бот не поднялся после
# перезапуска", and it must still reach the owner.
#
# StateChangeTimestampMonotonic is the last time this unit changed state at
# all, in CLOCK_MONOTONIC microseconds — the same clock /proc/uptime uses, so
# the two are directly comparable and neither is disturbed by a wall-clock
# step. A systemd too old to report it (or a 0) is treated as "cannot tell",
# which keeps today's permissive behavior rather than inventing a fault.
unit_active() {
    local state changed_us now_us age
    state="$(systemctl "$SCOPE_FLAG" show -p ActiveState --value "$1" 2>/dev/null)"
    case "$state" in
        active|reloading) return 0 ;;
        activating|deactivating)
            changed_us="$(systemctl "$SCOPE_FLAG" show \
                -p StateChangeTimestampMonotonic --value "$1" 2>/dev/null)"
            case "${changed_us:-0}" in
                ''|*[!0-9]*) return 0 ;;   # cannot tell — do not invent a fault
                0) return 0 ;;
            esac
            now_us="$(awk '{printf "%d", $1 * 1000000}' /proc/uptime 2>/dev/null)"
            case "${now_us:-0}" in
                ''|*[!0-9]*|0) return 0 ;;
            esac
            age=$(( (now_us - changed_us) / 1000000 ))
            [ "$age" -le "$TRANSITION_GRACE" ] && return 0
            echo "delivery-watch: $1 stuck in $state for ${age}s" >&2
            return 1
            ;;
        *) return 1 ;;
    esac
}

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

# F4. ONE change here, after a false alarm: this check alerted every five
# minutes while every answer it was worried about had in fact been
# delivered.
#
# THE NUMBER IS OUT OF THE TEXT. notify.sh and backup-notify.sh debounce on
# a cksum of the message, so "$streak ответов подряд" minted a brand-new
# message every time the streak moved and alerted forever. The STATUS.md
# block above already learned this; this one had not. The exact value goes
# to stderr, i.e. the journal.
#
# What is deliberately NOT here is a second gate on "but something WAS
# delivered since" (an outbox receipt newer than the last failed turn). It
# was written and removed in blind review: the apology the user gets for a
# failed turn goes out through the outbox too, so its receipt always
# postdates that turn's ledger row, and the gate would suppress every streak
# it was meant to judge. The false alarm was in the LEDGER, and it is fixed
# where the ledger is written — a turn the duty session covered, or one a
# late reply answered, now scores `ok`. See services/ask_health.py.

HEALTH="$RUNTIME_DIR/ask-health.json"
if [ -f "$HEALTH" ]; then
    streak=$(grep -o '"fail_streak"[: ]*[0-9]*' "$HEALTH" 2>/dev/null | grep -o '[0-9]*$')
    case "${streak:-0}" in
        ''|*[!0-9]*) streak=0 ;;
    esac
    if [ "$streak" -ge "$FAIL_STREAK" ]; then
        faults+=("ответы подряд не доставлены (см. журнал)")
        echo "delivery-watch: fail_streak $streak (limit $FAIL_STREAK)" >&2
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

# ── F6: a message we can PROVE was not delivered ─────────────────────
#
# Every check above this line is an INFERENCE from something adjacent to
# delivery — a unit's state, a file's age, a streak of turn outcomes, a
# process count. Each of them can be true while the person is reading his
# answers, which is how item 34 came to exist. This one is not an inference.
# The three durable queues of item 33 leave a file behind for exactly the
# states that mean a message was lost, and this lists those files:
#
#   outbox/dead/       a reply that was produced and could not be delivered
#   inbox/stale/       a message accepted and retired without an answer
#   chat-queue/stale/  a message prepared and retired without an answer
#   inbox/*.json old   a message accepted, and nothing has touched it since
#
# It is also why the gates added above are safe. They make the noisy checks
# quieter; this one guarantees that the case they exist for — «ответ не
# дошёл» — still reaches the owner on its own evidence, on both channels.
#
# Reported once per LOSS, by id and not by count: a pile he has already been
# told about must not re-alert every half hour forever (this script's header
# forbids exactly that), but a NEW loss must — including one that appears in
# the same interval another was cleared, which a count alone cannot see. The
# ids are the queues' own filenames.
LOSS_STAMP="$RUNTIME_DIR/delivery-watch.loss"
LOSS_NOW="$RUNTIME_DIR/.delivery-watch.loss.$$"
# `|| true` on every find: a queue directory that does not exist yet makes
# find exit non-zero, and under `pipefail` that failed the whole group — the
# first cut of this block silently produced an EMPTY loss list on exactly the
# installs where a queue had never been used.
{
    find "$RUNTIME_DIR/outbox/dead" -maxdepth 1 -type f -name '*.json' \
        -printf 'outbox-dead/%f\n' 2>/dev/null || true
    find "$RUNTIME_DIR/inbox/stale" -maxdepth 1 -type f -name '*.json' \
        -printf 'inbox-stale/%f\n' 2>/dev/null || true
    find "$RUNTIME_DIR/chat-queue/stale" -maxdepth 1 -type f -name '*.json' \
        -printf 'queue-stale/%f\n' 2>/dev/null || true
    # mtime, not the accepted_at inside the file: the question is whether
    # ANYTHING is still working on this entry, and a boot replay rewrites it
    # exactly when it stops being abandoned. services/delivery_proof.py uses
    # the same rule so the two readers cannot disagree.
    find "$RUNTIME_DIR/inbox" -maxdepth 1 -type f -name '*.json' \
        ! -name 'receipts.json' -mmin "+$STUCK_AFTER_MIN" \
        -printf 'inbox-stranded/%f\n' 2>/dev/null || true
} | LC_ALL=C sort >"$LOSS_NOW" 2>/dev/null
[ -f "$LOSS_STAMP" ] || : >"$LOSS_STAMP" 2>/dev/null || true
fresh_losses="$(LC_ALL=C comm -13 "$LOSS_STAMP" "$LOSS_NOW" 2>/dev/null || true)"

loss_reported=0
if [ -n "$fresh_losses" ]; then
    loss_reported=1
    case "$fresh_losses" in
        *outbox-dead/*)
            faults+=("ответ не удалось доставить — лежит в мёртвой очереди") ;;
    esac
    case "$fresh_losses" in
        *inbox-stale/*|*queue-stale/*)
            faults+=("сообщение принято, но ответ так и не ушёл") ;;
    esac
    case "$fresh_losses" in
        *inbox-stranded/*)
            faults+=("принятое сообщение зависло без ответа") ;;
    esac
    echo "delivery-watch: proven loss —" $fresh_losses >&2
fi

if [ "${#faults[@]}" -eq 0 ]; then
    # Nothing to report at all, so nothing can go unreported: record the
    # current picture (including a pile that was cleaned up, so the same
    # kind of loss can alert again).
    mv -f "$LOSS_NOW" "$LOSS_STAMP" 2>/dev/null || rm -f "$LOSS_NOW"
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

# Both channels print `sent` on stdout when a message actually left; their
# diagnostics still go to stderr, i.e. the journal.
primary="$("$PROJECT_DIR/scripts/notify.sh" "$MSG" 2>/dev/null || true)"
backup="$("$PROJECT_DIR/scripts/backup-notify.sh" "$MSG" 2>/dev/null || true)"
echo "$MSG" >&2

# The loss watermark advances ONLY once the report has actually left on at
# least one channel (blind review 3). The failure modes are correlated:
# entries land in outbox/dead/ precisely when Telegram sends are failing,
# which is also when these two fail — and a watermark advanced there would
# consume the one report of the one fault this script guarantees.
if [ "$loss_reported" -eq 1 ]; then
    case "$primary$backup" in
        *sent*) mv -f "$LOSS_NOW" "$LOSS_STAMP" 2>/dev/null || rm -f "$LOSS_NOW" ;;
        *)      rm -f "$LOSS_NOW" ;;
    esac
else
    mv -f "$LOSS_NOW" "$LOSS_STAMP" 2>/dev/null || rm -f "$LOSS_NOW"
fi
exit 1
