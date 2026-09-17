"""Incremental reader for a Claude Code session's JSONL transcript.

Fable audit R1 (2026-08-22): the pane render is a lossy, TUI-version-fragile
proxy for "what did the model actually say" — the transcript Claude Code
itself writes to ``~/.claude/projects/<slug>/<session-id>.jsonl`` is the raw,
pre-render text with an authoritative ``stop_reason``/``usage``. This module
is pure parsing + incremental file I/O, no tmux/subprocess — mirrors the
tmux_parse/claude_session split so this stays independently unit-testable
against small hand-built JSONL fixtures.

Kept STRICTLY diagnostic (shadow mode) for this round: nothing here is
allowed to become the thing actually delivered to Telegram yet — see
ClaudeSession's ``transcript_shadow_mode`` wiring in claude_session.py.

Format risk (explicitly flagged by the audit): the JSONL schema is not a
published/versioned API and can drift between CLI releases. Every entry
point here is written to fail closed and quiet on a single malformed
line/record (skip it) rather than raise — a caller integrating this into a
live turn must never let a transcript-shape surprise take down real
delivery.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Same line-anchoring contract as tmux_parse's markers, applied to RAW
# (un-rendered) model text — there are no terminal soft-wraps to undo here,
# unlike the pane path.
_OPEN_RE = re.compile(r"(?m)^.*?<<<R:(\w+)>>>[ \t]*\r?$")
_CLOSE_RE = re.compile(r"(?m)^.*?<<<E:(\w+)>>>[ \t]*\r?$")


def transcript_path(work_dir: Path | str, session_id: str) -> Path:
    """Path Claude Code writes this session's transcript to.

    Slug convention verified against the live install (also used by
    ``scripts/marker_compliance.py``): the absolute cwd with every ``/``
    replaced by ``-``.
    """
    slug = str(Path(work_dir).resolve()).replace("/", "-")
    return Path.home() / ".claude" / "projects" / slug / f"{session_id}.jsonl"


def _record_text(rec: dict) -> str | None:
    """Assistant text content of one JSONL record, or ``None`` if this
    record carries no assistant text (tool_use/tool_result/user/meta/...)."""
    if rec.get("type") != "assistant":
        return None
    message = rec.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(p for p in parts if p)
        return text or None
    return None


@dataclass(frozen=True)
class TranscriptReply:
    """One rid's reply as found in the transcript."""

    rid: str
    body: str
    closed: bool  # True iff a matching <<<E:rid>>> line was found


def extract_reply_from_record(rec: dict, rid: str) -> TranscriptReply | None:
    """The ``rid``'s reply from one JSONL record, or ``None``.

    Mirrors :func:`d_brain.services.tmux_parse.extract_reply`'s line-
    anchoring rules against raw model text. ``isSidechain`` records are
    REJECTED unconditionally — a background subagent's own text must never
    be mistaken for the main turn's reply (explicit audit risk note).
    """
    if rec.get("isSidechain"):
        return None
    text = _record_text(rec)
    if not text:
        return None
    my_opens = [m for m in _OPEN_RE.finditer(text) if m.group(1) == rid]
    if not my_opens:
        return None
    start = my_opens[-1]
    my_closes = [
        m
        for m in _CLOSE_RE.finditer(text)
        if m.group(1) == rid and m.start() > start.end()
    ]
    if my_closes:
        body = text[start.end() : my_closes[0].start()].strip()
        return TranscriptReply(rid=rid, body=body, closed=True) if body else None
    body = text[start.end() :].strip()
    return TranscriptReply(rid=rid, body=body, closed=False) if body else None


def latest_reply(
    path: Path | str, tail_bytes: int = 1_048_576
) -> tuple[str, str | None]:
    """The most recent assistant reply in the transcript at ``path``,
    scanning the tail IN REVERSE — WITHOUT requiring a known ``rid`` up
    front (unlike :func:`extract_reply_from_record`, which needs one).

    Backlog item 14: powers the ``/resend`` command, a manual, READ-ONLY
    escape hatch for the ~3.3% of turns measured where the tmux-pane-scrape
    delivery path never produces a closing marker at all — the user can ask
    the bot to re-fetch the last reply straight from the transcript instead.
    Because this is user-triggered and read-only (worst case: it shows
    nothing, or shows something unexpected, and the user just asks again),
    it does not need the same validation rigor as the automatic delivery
    path — but it MUST be honest about what it found, distinguishing the
    real outcomes rather than collapsing them into one generic failure:

    - ``("unavailable", None)`` — the file is missing or unreadable.
    - ``("empty", None)`` — no assistant text record was found in the tail
      window at all (a brand-new session, or one that hasn't answered yet).
    - ``("in_progress", None)`` — the most recent assistant text record has
      no ``<<<R:id>>>`` open marker at all yet — most plausibly still-
      streaming preamble/tool narration ahead of the wrapped answer. Never
      reported as "lost": the model may simply not be done yet. Same as
      ``"unclosed"`` below, this function stays pure and lock-free, so it
      cannot itself tell "genuinely still generating" apart from "finished,
      but this record never got a marker at all" (e.g. a reply split across
      two transcript records, with the open marker landing in an EARLIER
      record than the one being inspected — only the single most recent
      record is ever examined, see below). That distinction is
      ``last_reply_for_resend()``'s job (F-1 fix, round 2, 2026-08-22): it
      consults ``is_turn_active()`` and turns this into either "in_progress"
      (turn genuinely active) or its own separate ``"no_markers"`` status
      (turn idle, nothing usable to recover — honesty only, no salvage
      attempted since there is no marker span to recover from).
    - ``("unclosed", text)`` — the most recent assistant text record has an
      open ``<<<R:id>>>`` marker with NO matching ``<<<E:id>>>`` — this is
      the literal shape of every real delivery-loss incident on record
      (bfbe3335, df4f87ef, 9dd35326: full reply text, end of message, no
      closing marker). ``text`` is the recovered/salvaged body found after
      the open marker. Deliberately kept a DISTINCT outcome from
      ``in_progress`` rather than folded into it: this function stays pure
      and lock-free (no liveness check here), so it cannot itself decide
      whether the turn producing this span is still active — that decision
      belongs to the caller (``last_reply_for_resend`` in
      claude_session.py), which has access to ``is_turn_active()`` and
      turns this into either "ready" (with the recovered body, turn is not
      active) or "in_progress" (turn genuinely still active) accordingly.
    - ``("ready", text)`` — the most recent assistant text record has a
      complete, closed reply; ``text`` is its body.

    Only the SINGLE most recent assistant (non-sidechain) text record is
    examined — an older, already-closed reply further back is deliberately
    NOT surfaced once a newer turn has started producing text, since that
    would resend a stale answer while a fresh one is in flight. Uses the
    same tail-read approach as :func:`latest_context_tokens`; a single
    record longer than ``tail_bytes`` straddling the boundary is the same
    known limitation noted there.

    Per-record parsing (after the JSON itself decodes) is wrapped in a
    broad ``except Exception`` — the JSONL schema is not a
    published/versioned API (see module docstring), so a record shaped
    unexpectedly (e.g. ``message`` present but not a dict) must be skipped,
    never allowed to raise out of this scan and take the whole ``/resend``
    path down with it.
    """
    try:
        size = Path(path).stat().st_size
        with Path(path).open("rb") as f:
            f.seek(max(0, size - tail_bytes))
            chunk = f.read()
    except OSError:
        return "unavailable", None
    for raw in reversed(chunk.split(b"\n")):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        try:
            if not isinstance(rec, dict) or rec.get("isSidechain"):
                continue
            text = _record_text(rec)
            if not text:
                continue
            opens = list(_OPEN_RE.finditer(text))
            if not opens:
                # The most recent assistant text carries no reply marker at
                # all yet — most plausibly still-streaming preamble/tool
                # narration ahead of the wrapped answer. Honest as "still
                # working", not "empty" (something IS happening) and not
                # "ready" (nothing complete to show).
                return "in_progress", None
            start = opens[-1]
            rid = start.group(1)
            closes = [
                m
                for m in _CLOSE_RE.finditer(text)
                if m.group(1) == rid and m.start() > start.end()
            ]
            if not closes:
                body = text[start.end() :].strip()
                if not body:
                    return "in_progress", None
                return "unclosed", body
            body = text[start.end() : closes[0].start()].strip()
            if not body:
                return "in_progress", None
            return "ready", body
        except Exception:
            logger.debug(
                "skipping malformed transcript record while scanning for "
                "latest reply",
                exc_info=True,
            )
            continue
    return "empty", None


def latest_context_tokens(path: Path | str, tail_bytes: int = 262_144) -> int | None:
    """Best-effort estimate of the CURRENT context size (in tokens) from the
    LATEST real assistant record in the transcript at ``path``, reading only
    the last ``tail_bytes`` — R3 (Fable audit): a watchdog tick asking "how
    big is the context right now" must never read a 100MB+ transcript whole
    just to answer that.

    B2 fix (Fable audit fix round, 2026-08-22): the original version
    returned ``usage.cache_read_input_tokens`` alone. That undercounts badly
    on a CACHE-REWRITE turn — verified against a real live transcript where
    a single record showed ``cache_read_input_tokens=30698`` alongside
    ``cache_creation_input_tokens=580621`` (real context ~611k) — because a
    cache miss/rewrite moves most of the context from "read" into
    "creation" accounting for that one turn without shrinking the context
    itself. Reading `cache_read` alone made the R3 latch see a spuriously
    LOW number on these turns and un-arm, then re-fire on the very next
    ordinary (cache-hit) turn — in practice, close to every cache-rewrite
    cycle (~2x/day on real usage patterns). The fix sums all three usage
    fields that together represent the full prompt the model just processed:
    ``cache_read_input_tokens + cache_creation_input_tokens +
    input_tokens``. Field names verified against a real live transcript
    (2026-08-22), not merely assumed from the audit.

    Also skips ``model == "<synthetic>"`` records: these are Claude Code's
    own synthetic/meta assistant entries (e.g. compaction notices), which
    report usage of 0 and would otherwise be mistaken for "the context just
    dropped to zero" by the caller — the same false-unarm failure mode as
    the cache-rewrite case above, from the opposite direction.

    Returns ``None`` if the file is missing/unreadable, or if no usable
    record was found within the tail window (e.g. a single record longer
    than ``tail_bytes`` straddling the boundary) — the caller should treat
    that as "skip this tick", not as "context is zero".
    """
    try:
        size = Path(path).stat().st_size
        with Path(path).open("rb") as f:
            f.seek(max(0, size - tail_bytes))
            chunk = f.read()
    except OSError:
        return None
    for raw in reversed(chunk.split(b"\n")):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if rec.get("type") != "assistant":
            continue
        message = rec.get("message") or {}
        if message.get("model") == "<synthetic>":
            continue
        usage = message.get("usage") or {}
        read = usage.get("cache_read_input_tokens")
        creation = usage.get("cache_creation_input_tokens")
        input_tokens = usage.get("input_tokens")
        parts = [t for t in (read, creation, input_tokens) if isinstance(t, int)]
        if not parts:
            continue
        return sum(parts)
    return None


class TranscriptTail:
    """Byte-offset-tracked incremental reader over a (potentially 100MB+)
    JSONL transcript — re-reading the whole file on every poll is explicitly
    ruled out by the audit (R1 step 2).

    Only ever advances its offset past FULLY newline-terminated lines: a
    partial trailing line (the writer mid-append) is left unconsumed and
    re-read whole on the next call, so a record is never parsed half-written.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._offset = 0

    @classmethod
    def at_end(cls, path: Path | str) -> TranscriptTail:
        """A tail positioned at the CURRENT end of the file — for a caller
        that only cares about records appended from this moment on (e.g. a
        turn just about to be sent)."""
        tail = cls(path)
        try:
            tail._offset = Path(path).stat().st_size
        except OSError:
            tail._offset = 0
        return tail

    def poll_new_records(self) -> list[dict]:
        """Every complete JSON record appended since the last call, in
        order. Returns ``[]`` (never raises) if the file is missing/unreadable
        — the transcript can legitimately not exist yet at turn-send time."""
        try:
            with self.path.open("rb") as f:
                f.seek(self._offset)
                chunk = f.read()
        except OSError:
            return []
        if not chunk:
            return []
        records: list[dict] = []
        consumed = 0
        start = 0
        while True:
            nl = chunk.find(b"\n", start)
            if nl == -1:
                break  # partial trailing line — leave it for next time
            line = chunk[start:nl].strip()
            consumed = nl + 1
            start = nl + 1
            if not line:
                continue
            try:
                records.append(json.loads(line.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.debug(
                    "skipping unparsable transcript line at offset %d", self._offset
                )
        self._offset += consumed
        return records
