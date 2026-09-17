#!/usr/bin/env python
"""Measure how often the model drops the closing `<<<E:id>>>` marker, bucketed
by how much context the session was carrying at that point.

Read-only: parses the live Claude Code session transcript (JSONL) for turns
that were told to wrap their reply in `<<<R:id>>>`/`<<<E:id>>>`, and checks
whether the closing marker actually appears in that turn's own output. Does
not touch src/d_brain or anything on the delivery path — this only produces
a metric.

    uv run python scripts/marker_compliance.py
    uv run python scripts/marker_compliance.py --transcript /path/to/other.jsonl

Background: agent-infra-backlog.md item 11 found (2026-08-21, manual read of
the raw transcript) that the model drops the closing marker more often as
context grows — 1.5% under 400k tokens, 6.9% at 400-600k, 12.8% at 300k+ in a
provisional sample. This script reproduces that measurement mechanically so
it doesn't require re-reading the transcript by hand each time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# The protocol requires the marker to be ALONE on its own line (a leading
# bullet/indentation from UI rendering is fine) — matching only that, not
# any occurrence of the substring, is what keeps this from false-matching
# later prose that quotes a marker while describing an incident (e.g. a
# daily.md entry saying "closing marker `<<<E:9dd35326>>>` never appeared").
OPEN_RE = re.compile(r"^[\s⏺>*-]*<<<R:([A-Za-z0-9_-]+)>>>[\s]*$", re.MULTILINE)
CLOSE_RE = re.compile(r"^[\s⏺>*-]*<<<E:([A-Za-z0-9_-]+)>>>[\s]*$", re.MULTILINE)
# The instruction line the harness appends to prompts that expect a
# marker-wrapped reply — its presence is what makes a turn "in scope" for
# this metric, not just any turn that happens to mention markers.
INSTRUCTED_RE = re.compile(
    r"wrap your ENTIRE reply between a line containing only <<<R:([A-Za-z0-9_-]+)>>>"
)

BUCKETS = [(0, 400_000), (400_000, 600_000), (600_000, float("inf"))]


def default_transcript_path() -> Path:
    """Resolve the live session's transcript.

    R1 (Fable audit) pinned a `runtime_dir/session_id` file precisely to
    close the footgun this function used to have: mtime-based "newest
    .jsonl" guessing can pick a DIFFERENT concurrent session's file — the
    cron brain (get_cron_session) shares this exact same project directory
    (same work_dir/vault_path), and any other Claude Code session someone
    happens to run against this vault would too (found live 2026-08-22,
    documented in the incident registry). Prefer the pinned id; fall back to
    the old mtime guess ONLY for an install that predates R1, with a loud
    warning so the caller knows the result may be wrong.
    """
    home = Path.home()
    project_dir = home / ".claude" / "projects"
    # Same convention used by every session this vault runs: the project dir
    # name is the vault path with slashes replaced by dashes.
    vault_path = Path(__file__).resolve().parent.parent / "vault"
    slug = str(vault_path).replace("/", "-")
    candidate_dir = project_dir / slug

    runtime_dir = Path(
        os.environ.get("DBRAIN_RUNTIME_DIR", str(Path.home() / ".dbrain"))
    )
    session_id_file = runtime_dir / "session_id"
    try:
        session_id = session_id_file.read_text().strip()
    except OSError:
        session_id = ""
    if session_id:
        pinned = candidate_dir / f"{session_id}.jsonl"
        if pinned.is_file():
            return pinned
        print(
            f"warning: pinned session id {session_id!r} ({session_id_file}) has "
            f"no matching transcript at {pinned} — falling back to mtime guess",
            file=sys.stderr,
        )

    if not candidate_dir.is_dir():
        raise FileNotFoundError(
            f"no transcript directory at {candidate_dir} — pass --transcript explicitly"
        )
    jsonls = sorted(candidate_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not jsonls:
        raise FileNotFoundError(f"no .jsonl transcripts found under {candidate_dir}")
    if len(jsonls) > 1:
        print(
            f"warning: {len(jsonls)} transcripts under {candidate_dir} and no "
            "pinned session id to disambiguate — picked the newest by mtime, "
            "which may be a DIFFERENT concurrent session (e.g. the cron brain)",
            file=sys.stderr,
        )
    return jsonls[-1]


def iter_turns(transcript: Path):
    """Yield (rid, instructed: bool, context_tokens: int, closed: bool) per turn
    that contains a marker-wrap instruction, in transcript order."""
    with transcript.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            usage = (rec.get("message") or {}).get("usage") or {}
            context_tokens = (
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )

            content = rec.get("message", {}).get("content")
            text_parts = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
            text = "\n".join(text_parts)

            instructed_match = INSTRUCTED_RE.search(text)
            if not instructed_match:
                continue
            rid = instructed_match.group(1)
            # The instruction line appears in the USER turn asking for a
            # marker-wrapped reply; the actual open/close markers are emitted
            # by the model's NEXT turn. This generator only flags instructed
            # turns — pairing with the model's answer happens in main().
            yield rid, context_tokens


def find_reply_markers(transcript: Path):
    """Return {rid: (has_open, has_close, context_tokens_at_open)} across the
    whole transcript, keyed by the rid embedded in the markers themselves."""
    results: dict[str, dict] = {}
    with transcript.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = rec.get("message", {}).get("content")
            text_parts = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
            text = "\n".join(text_parts)
            if not text:
                continue

            usage = (rec.get("message") or {}).get("usage") or {}
            context_tokens = (
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )

            for m in OPEN_RE.finditer(text):
                rid = m.group(1)
                entry = results.setdefault(
                    rid, {"open": False, "close": False, "tokens": None}
                )
                entry["open"] = True
                if entry["tokens"] is None:
                    entry["tokens"] = context_tokens
            for m in CLOSE_RE.finditer(text):
                rid = m.group(1)
                entry = results.setdefault(
                    rid, {"open": False, "close": False, "tokens": None}
                )
                entry["close"] = True
    return results


def bucket_for(tokens: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= tokens < hi:
            hi_label = "inf" if hi == float("inf") else f"{int(hi) // 1000}k"
            return f"{lo // 1000}k-{hi_label}"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help=(
            "path to the session .jsonl transcript "
            "(default: this vault's live session)"
        ),
    )
    args = parser.parse_args()

    try:
        transcript = args.transcript or default_transcript_path()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    markers = find_reply_markers(transcript)
    if not markers:
        print("no <<<R:id>>> markers found in transcript — nothing to measure")
        return 0

    bucket_totals: dict[str, list[int]] = {}  # bucket -> [total, dropped]
    dropped_rids = []
    for rid, entry in markers.items():
        if not entry["open"]:
            continue  # not a reply-wrap turn we can attribute
        tokens = entry["tokens"] or 0
        b = bucket_for(tokens)
        bucket_totals.setdefault(b, [0, 0])
        bucket_totals[b][0] += 1
        if not entry["close"]:
            bucket_totals[b][1] += 1
            dropped_rids.append((rid, tokens))

    print(f"transcript: {transcript}")
    print(f"total marker-wrap turns: {sum(t for t, _ in bucket_totals.values())}")
    print()
    print(f"{'bucket':<14}{'turns':>8}{'dropped':>10}{'rate':>10}")
    for b in sorted(bucket_totals, key=lambda k: int(k.split("k-")[0])):
        total, dropped = bucket_totals[b]
        rate = f"{100 * dropped / total:.1f}%" if total else "n/a"
        print(f"{b:<14}{total:>8}{dropped:>10}{rate:>10}")

    if dropped_rids:
        print()
        print("dropped rids (rid, context_tokens_at_open):")
        for rid, tokens in dropped_rids:
            print(f"  {rid}  {tokens}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
