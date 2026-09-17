#!/bin/bash
set -e

# PATH for systemd (claude, uv, npx, node)
export PATH="$HOME/.local/bin:$HOME/.nvm/versions/node/$(ls "$HOME/.nvm/versions/node/" 2>/dev/null | tail -1)/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# Paths — auto-detect from script location
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# C5: env-overridable, defaults IDENTICAL to the previous hardcoded values.
# ENV_FILE resolves before the env file is sourced (it names the file), so it
# can only come from the process environment — that is how the templated
# systemd unit points an instance at /etc/dbrain/<inst>/.env.
ENV_FILE="${DBRAIN_ENV_FILE:-$PROJECT_DIR/.env}"

# Load environment variables (set -a survives quoted values and spaces,
# unlike export $(... | xargs), which word-splits and strips quotes)
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi

# VAULT_DIR is resolved AFTER the env file is sourced so that VAULT_PATH set
# either in the process environment or inside the env file takes effect.
# Unset → $PROJECT_DIR/vault, exactly as before.
VAULT_DIR="${VAULT_PATH:-$PROJECT_DIR/vault}"

# B4: VAULT_PATH is RELATIVE in the live .env (`./vault`), not unset. Today it
# resolves correctly only by coincidence — the installed unit's
# WorkingDirectory happens to be the repo root. After phases 5/6 both
# instances' units set WorkingDirectory to the SHARED checkout, so a relative
# VAULT_PATH in /etc/dbrain/<instance>/.env would resolve to ANOTHER instance's vault: the
# worst-case isolation failure, from a one-character config mistake.
# Anchor it to $PROJECT_DIR explicitly instead of trusting the cwd.
case "$VAULT_DIR" in
    /*) : ;;  # already absolute — untouched
    *)
        _rel="$VAULT_DIR"
        VAULT_DIR="$(cd "$PROJECT_DIR" && cd "$_rel" 2>/dev/null && pwd)" \
            || VAULT_DIR="$PROJECT_DIR/${_rel#./}"
        ;;
esac

# Check token
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "ERROR: TELEGRAM_BOT_TOKEN not set"
    exit 1
fi

# Timezone (configure in .env: TZ=Your/Timezone)
export TZ="${TZ:-UTC}"

# Date and chat_id
TODAY=$(date +%Y-%m-%d)
CHAT_ID="${ALLOWED_USER_IDS//[\[\] ]/}"  # strip brackets/spaces from [123, 456]
CHAT_ID="${CHAT_ID%%,*}"  # first id only — the admin gets the report

echo "=== d-brain processing for $TODAY ==="

# ── C5: git sync ────────────────────────────────────────────────────────
# Shared-zone lock (plan §3.6): both instances' nightly pipelines serialize
# their commit+push of the shared projects repo through one flock.
SHARED_LOCK="${DBRAIN_SHARED_LOCK:-/var/lib/dbrain/shared/.locks/commit.lock}"

# A degraded lock is a real operational problem in the shared zone, and a
# per-instance log file nobody reads is not where it should land. Route it to
# Telegram (notify.sh debounces identical messages for 30 min, so a nightly
# job cannot spam). Never fatal.
_warn_shared() {
    echo "WARN: $1" >&2
    "$PROJECT_DIR/scripts/notify.sh" "⚠️ d-brain: $1" >/dev/null 2>&1 || true
}

# Commit+push one repo. Never fatal: `set -e` is on and a failed push must
# not abort the run before the Telegram report is sent.
_commit_repo() {
    local dir="$1"
    git -C "$dir" add -A || true
    git -C "$dir" commit -m "chore: process daily $TODAY" || true
    # Fail closed: an install that requires a PRIVATE memory remote never
    # pushes to a repository that is public or cannot be verified.
    if [ -n "${DBRAIN_REQUIRE_PRIVATE_VAULT:-}" ] && \
       ! (cd "$PROJECT_DIR" && uv run python -m d_brain.services.git check-private "$dir"); then
        _warn_shared "память не отправлена в GitHub: репозиторий не подтверждён как PRIVATE"
        return 0
    fi
    git -C "$dir" push || true
}

# BEFORE the git split (today): $VAULT_DIR has no .git of its own, so this
# takes the else-branch and runs the exact same three commands, from the same
# cwd ($PROJECT_DIR), as the code this replaced. AFTER the split: vault and
# vault/projects are two independent repos and the umbrella `git add -A` must
# NOT run, or the code repo would swallow the vault again.
sync_git() {
    if [ -d "$VAULT_DIR/.git" ]; then
        echo "=== Git: split layout (vault is its own repo) ==="
        _commit_repo "$VAULT_DIR"

        if [ -d "$VAULT_DIR/projects/.git" ]; then
            # Create the lock directory rather than skipping the lock: a
            # missing lock dir is exactly the misconfigured state in which the
            # concurrent-commit race is MOST likely, so silently degrading to
            # unlocked defeats the mechanism precisely when it is needed.
            mkdir -p "$(dirname "$SHARED_LOCK")" 2>/dev/null || true
            # `sg dbrain -c '...'` wraps the WHOLE flock+git chain, not
            # just the git commands: a long-lived session (the hourly
            # auto-vault-sync cron session, in particular) can have been
            # started BEFORE this instance's user was added to dbrain
            # -- supplementary groups are fixed at process start, not re-read
            # live -- so a bare `flock` here would itself fail to even open
            # the lock FILE under the stale group list (found live
            # 2026-08-23: wrapping only the inner git commands still hit
            # `flock: Permission denied` on the lock file itself, before ever
            # reaching git). `sg` re-checks /etc/group at invocation time, so
            # it must be the outermost command. See
            # the multi-instance rollout plan, phase 3
            # status for the original EACCES-on-.git finding this fixes.
            if command -v flock >/dev/null 2>&1 && [ -d "$(dirname "$SHARED_LOCK")" ]; then
                echo "=== Git: shared projects repo (flock) ==="
                # -w 300: never block forever. If the other instance's
                # pipeline wedges while holding the lock, this run gives up
                # and says so instead of hanging the nightly job indefinitely.
                if ! sg dbrain -c \
                    "flock -w 300 '$SHARED_LOCK' bash -c \" \
                     git -C '$VAULT_DIR/projects' add -A; \
                     git -C '$VAULT_DIR/projects' commit -m 'chore: process daily $TODAY' || true; \
                     git -C '$VAULT_DIR/projects' push || true\""; then
                    _warn_shared "не удалось взять блокировку общего репозитория за 300с — коммит общей зоны пропущен"
                fi
            else
                _warn_shared "flock недоступен или нет каталога блокировки ($SHARED_LOCK) — общая зона коммитится БЕЗ блокировки"
                sg dbrain -c \
                    "git -C '$VAULT_DIR/projects' add -A; \
                     git -C '$VAULT_DIR/projects' commit -m 'chore: process daily $TODAY' || true; \
                     git -C '$VAULT_DIR/projects' push || true" || true
            fi
        fi
    else
        # Pre-split behavior, byte-for-byte.
        cd "$PROJECT_DIR"
        git add -A || true
        git commit -m "chore: process daily $TODAY" || true
        git push || true
    fi
}

# ── ORIENT PHASE: pre-flight checks ──
DAILY_FILE="$VAULT_DIR/daily/$TODAY.md"
HANDOFF_FILE="$VAULT_DIR/.session/handoff.md"
GRAPH_FILE="$VAULT_DIR/.graph/vault-graph.json"

# Check daily file exists and has content
if [ ! -f "$DAILY_FILE" ]; then
    echo "ORIENT: daily/$TODAY.md not found — creating empty file"
    echo "# $TODAY" > "$DAILY_FILE"
fi

DAILY_SIZE=$(wc -c < "$DAILY_FILE" 2>/dev/null || echo "0")
if [ "$DAILY_SIZE" -lt 50 ]; then
    echo "ORIENT: daily/$TODAY.md is empty ($DAILY_SIZE bytes) — skipping Claude processing"
    echo "ORIENT: Running graph rebuild only"

    # Still rebuild graph and commit
    cd "$VAULT_DIR"
    uv run .claude/skills/autograph/scripts/graph.py health . || echo "Graph rebuild failed (non-critical)"
    cd "$PROJECT_DIR"

    sync_git
    echo "=== Done (empty daily, graph-only) ==="
    exit 0
fi

# Check handoff exists
if [ ! -f "$HANDOFF_FILE" ]; then
    echo "ORIENT: handoff.md not found — creating stub"
    mkdir -p "$VAULT_DIR/.session"
    echo -e "---\nupdated: $(date -Iseconds)\n---\n\n## Last Session\n(none)\n\n## Observations" > "$HANDOFF_FILE"
fi

# Check graph freshness (warn if >7 days old)
if [ -f "$GRAPH_FILE" ]; then
    GRAPH_AGE=$(( ($(date +%s) - $(stat -c %Y "$GRAPH_FILE" 2>/dev/null || stat -f %m "$GRAPH_FILE" 2>/dev/null || echo 0)) / 86400 ))
    if [ "$GRAPH_AGE" -gt 7 ]; then
        echo "ORIENT: vault-graph.json is $GRAPH_AGE days old (>7)"
    fi
fi

echo "ORIENT: daily=$DAILY_SIZE bytes, handoff=OK, graph=OK"
# ── END ORIENT PHASE ──

# ── PROCESS via the persistent interactive session (NO claude -p) ──
# The 3-phase claude -p pipeline is gone: after 2026-06-15 that bills against
# the Agent SDK credit. We drive the long-lived interactive session instead,
# which stays on the subscription. The Python entrypoint runs the daily
# processing through the shared session and prints the HTML report.
echo "=== Daily processing (interactive session) ==="
set +e
REPORT=$(cd "$PROJECT_DIR" && uv run python -m d_brain.pipeline daily 2>&1)
PIPELINE_EXIT=$?
set -e

echo "=== pipeline output ==="
echo "$REPORT"
echo "======================="

# The pipeline's own stdout is the intended user-facing report. If the
# invocation itself failed (e.g. `uv`/`python` missing, PATH misconfigured
# for this instance), $REPORT is raw shell/interpreter error text instead —
# never forward that to Telegram as if it were the report.
if [ "$PIPELINE_EXIT" -ne 0 ]; then
    echo "Daily pipeline failed (exit $PIPELINE_EXIT) — not forwarding raw output to Telegram"
    REPORT="⚠️ Ежедневная обработка не запустилась (техническая ошибка на сервере, exit $PIPELINE_EXIT). Подробности — в логах, не в этом сообщении."
fi

# Remove HTML comments (break Telegram HTML parser)
REPORT_CLEAN=$(echo "$REPORT" | sed '/<!--/,/-->/d')

# Rebuild vault graph (keeps structure up to date)
echo "=== Rebuilding vault graph ==="
cd "$VAULT_DIR"
uv run .claude/skills/autograph/scripts/graph.py health . || echo "Graph rebuild failed (non-critical)"

# Memory decay (update relevance scores and tiers)
echo "=== Memory decay ==="
uv run .claude/skills/autograph/scripts/engine.py decay . || echo "Memory decay failed (non-critical)"
cd "$PROJECT_DIR"

# Git commit
sync_git

# Send to Telegram
if [ -n "$REPORT_CLEAN" ] && [ -n "$CHAT_ID" ]; then
    echo "=== Sending to Telegram ==="
    RESULT=$(curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d "chat_id=$CHAT_ID" \
        -d "text=$REPORT_CLEAN" \
        -d "parse_mode=HTML")

    # If HTML failed, send without formatting
    if echo "$RESULT" | grep -q '"ok":false'; then
        echo "HTML failed: $RESULT"
        curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
            -d "chat_id=$CHAT_ID" \
            -d "text=$REPORT_CLEAN"
    fi
fi

echo "=== Done ==="
