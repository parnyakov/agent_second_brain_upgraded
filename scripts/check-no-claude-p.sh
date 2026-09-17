#!/bin/bash
# CI / install guard: fail if `claude -p` / `claude --print` is actually
# INVOKED in the hot path. The only allowed use is the escape hatch, marked
# ALLOW-CLAUDE-P. Prose/comments mentioning "claude -p" are ignored — we look
# for real invocations: a "--print" string literal (Python arg list) or a
# `claude ... --print/-p` command on a non-comment shell line.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

py_hits="$(grep -rn --include='*.py' '"--print"' "$ROOT/src" \
    | grep -v 'ALLOW-CLAUDE-P' || true)"

sh_hits="$(grep -rnE --include='*.sh' \
    '^[^#]*claude[[:space:]]+[^#]*(--print|[[:space:]]-p([[:space:]]|$))' \
    "$ROOT/scripts" \
    | grep -v 'ALLOW-CLAUDE-P' \
    | grep -v 'check-no-claude-p' || true)"

hits="$(printf '%s\n%s' "$py_hits" "$sh_hits" | grep -v '^$' || true)"

if [ -n "$hits" ]; then
    echo "❌ Found a real claude -p / --print invocation outside the escape hatch:"
    echo "$hits"
    echo
    echo "Use the interactive ClaudeSession instead, or mark the line"
    echo "ALLOW-CLAUDE-P if it is genuinely the router escape hatch."
    exit 1
fi
echo "✅ No claude -p invocation in the hot path."
