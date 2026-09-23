"""Tests for the JSONL transcript reader (Fable audit R1).

Fixtures are hand-built JSONL lines shaped like real Claude Code transcript
records — including the three named incidents' actual shape (bfbe3335,
df4f87ef, 9dd35326): `<<<R:id>>>` + text + end of message, no `<<<E:id>>>`.
"""

import json
import re
from pathlib import Path

from d_brain.services.transcript import (
    ReplyTail,
    TranscriptTail,
    latest_context_tokens,
    latest_reply,
    transcript_path,
)


def _assistant_record(
    text: str,
    *,
    is_sidechain: bool = False,
    usage: dict | None = None,
    model: str | None = None,
) -> dict:
    message: dict = {
        "content": [{"type": "text", "text": text}],
        "usage": usage or {},
    }
    if model is not None:
        message["model"] = model
    return {
        "type": "assistant",
        "isSidechain": is_sidechain,
        "message": message,
    }


# ── transcript_path ───────────────────────────────────────────────────────


def test_transcript_path_slug_convention(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    work_dir = tmp_path / "some" / "vault"
    work_dir.mkdir(parents=True)
    p = transcript_path(work_dir, "abc-123")
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(work_dir.resolve()))
    assert p == Path.home() / ".claude" / "projects" / slug / "abc-123.jsonl"


def test_transcript_path_replaces_every_non_alphanumeric_char(tmp_path, monkeypatch):
    """Live-verified 2026-09-22: `_` and `.` become `-` too, not just `/`.
    A cwd like /tmp/x_y/.claude used to resolve to a directory that simply
    does not exist — and with the reply read from this file, that is a total
    delivery failure, not a degraded diagnostic."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    work_dir = tmp_path / "dbrain-smoke-tfgw_84x" / "work"
    work_dir.mkdir(parents=True)
    p = transcript_path(work_dir, "abc-123")
    assert "_" not in p.parent.name
    assert p.parent.name.endswith("-dbrain-smoke-tfgw-84x-work")


def test_transcript_path_finds_the_pinned_id_when_the_slug_is_wrong(
    tmp_path, monkeypatch
):
    """Belt and braces for a future slug-rule change: the session id is a
    UUID, so if the computed directory holds nothing, the file named after
    the pinned id is looked up wherever it actually is."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    work_dir = tmp_path / "vault"
    work_dir.mkdir()
    elsewhere = home / ".claude" / "projects" / "some-other-slug"
    elsewhere.mkdir(parents=True)
    real = elsewhere / "abc-123.jsonl"
    real.write_text("")
    assert transcript_path(work_dir, "abc-123") == real


def test_transcript_path_falls_back_to_the_computed_path_when_nothing_exists(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    work_dir = tmp_path / "vault"
    work_dir.mkdir()
    p = transcript_path(work_dir, "abc-123")
    assert p.name == "abc-123.jsonl" and not p.exists()


# ── ReplyTail (THE reply source, backlog item 32) ─────────────────────────


def _append(path: Path, *records: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _tail(tmp_path, rid: str = "rid1") -> tuple[ReplyTail, Path]:
    """A ReplyTail anchored at the end of an empty transcript — exactly how
    ask() opens one just before typing the prompt."""
    path = tmp_path / "session.jsonl"
    path.write_text("")
    return ReplyTail(path, rid), path


def test_reply_tail_closed_pair(tmp_path):
    tail, path = _tail(tmp_path)
    _append(path, _assistant_record("<<<R:rid1>>>\nHello there.\n<<<E:rid1>>>\n"))
    reply = tail.poll()
    assert reply is not None
    assert reply.closed is True
    assert reply.body == "Hello there."


def test_reply_tail_open_span_no_closing_marker(tmp_path):
    """The real shape of the three named incidents (bfbe3335, df4f87ef,
    9dd35326): <<<R:id>>>, full reply text, end of message, no <<<E:id>>>."""
    tail, path = _tail(tmp_path, "bfbe3335")
    _append(path, _assistant_record("<<<R:bfbe3335>>>\nThe full answer text.\nend."))
    reply = tail.poll()
    assert reply is not None
    assert reply.closed is False
    assert reply.body == "The full answer text.\nend."


def test_reply_tail_none_for_wrong_rid(tmp_path):
    tail, path = _tail(tmp_path, "other-rid")
    _append(path, _assistant_record("<<<R:rid1>>>\nHello.\n<<<E:rid1>>>\n"))
    assert tail.poll() is None


def test_reply_tail_rejects_sidechain(tmp_path):
    """A background subagent's own text must never be mistaken for the main
    turn's reply (explicit audit risk note)."""
    tail, path = _tail(tmp_path)
    _append(
        path,
        _assistant_record(
            "<<<R:rid1>>>\nSubagent text.\n<<<E:rid1>>>\n", is_sidechain=True
        ),
    )
    assert tail.poll() is None


def test_reply_tail_ignores_non_assistant_records(tmp_path):
    tail, path = _tail(tmp_path)
    _append(
        path,
        {
            "type": "user",
            "message": {"content": "<<<R:rid1>>>\nnot a reply\n<<<E:rid1>>>\n"},
        },
    )
    assert tail.poll() is None


def test_reply_tail_none_when_no_open_marker(tmp_path):
    tail, path = _tail(tmp_path)
    _append(path, _assistant_record("just a normal reply, no markers"))
    assert tail.poll() is None


def test_reply_tail_handles_string_content(tmp_path):
    tail, path = _tail(tmp_path)
    _append(
        path,
        {
            "type": "assistant",
            "isSidechain": False,
            "message": {
                "content": "<<<R:rid1>>>\nplain string content\n<<<E:rid1>>>\n"
            },
        },
    )
    reply = tail.poll()
    assert reply is not None and reply.body == "plain string content"


def test_reply_tail_joins_a_pair_split_across_records(tmp_path):
    """The model opened the pair, stopped to call a tool (which closes the
    assistant record), then finished. Both halves must be joined, or a
    perfectly complete reply reads as 'still open'."""
    tail, path = _tail(tmp_path)
    _append(path, _assistant_record("<<<R:rid1>>>\nFirst half."))
    assert tail.poll().closed is False
    _append(
        path,
        {"type": "user", "message": {"content": "tool result"}},
        _assistant_record("Second half.\n<<<E:rid1>>>"),
    )
    reply = tail.poll()
    assert reply.closed is True
    assert reply.body == "First half.\nSecond half."


def test_reply_tail_ignores_text_written_before_the_anchor(tmp_path):
    """Anchored at send time: an earlier turn's records are invisible even
    when they carry (impossibly) the same rid."""
    path = tmp_path / "session.jsonl"
    path.write_text("")
    _append(path, _assistant_record("<<<R:rid1>>>\nOld answer.\n<<<E:rid1>>>"))
    tail = ReplyTail(path, "rid1")
    assert tail.poll() is None
    _append(path, _assistant_record("<<<R:rid1>>>\nNew answer.\n<<<E:rid1>>>"))
    assert tail.poll().body == "New answer."


def test_reply_tail_skips_synthetic_records(tmp_path):
    """Claude Code's own meta entries (compaction notices) are not the model
    talking and must never be spliced into a delivered reply."""
    tail, path = _tail(tmp_path)
    _append(
        path,
        _assistant_record("<<<R:rid1>>>\nReal answer."),
        _assistant_record("Compacted conversation.", model="<synthetic>"),
        _assistant_record("<<<E:rid1>>>"),
    )
    reply = tail.poll()
    assert reply.closed is True
    assert reply.body == "Real answer."


def test_reply_tail_takes_the_newest_pair_when_a_turn_emits_two(tmp_path):
    """Documented rule: last open marker, first close after it — the newest
    complete answer, never a splice of both."""
    tail, path = _tail(tmp_path)
    _append(
        path,
        _assistant_record(
            "<<<R:rid1>>>\nfirst answer\n<<<E:rid1>>>\n"
            "<<<R:rid1>>>\nsecond answer\n<<<E:rid1>>>"
        ),
    )
    reply = tail.poll()
    assert reply.closed is True
    assert reply.body == "second answer"


def test_reply_tail_ignores_markers_inside_a_thinking_block(tmp_path):
    """Only `text` blocks are the model talking to the user."""
    tail, path = _tail(tmp_path)
    _append(
        path,
        {
            "type": "assistant",
            "isSidechain": False,
            "message": {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "<<<R:rid1>>>\nnot the answer\n<<<E:rid1>>>",
                    }
                ]
            },
        },
    )
    assert tail.poll() is None


def test_reply_tail_skips_a_record_whose_message_is_not_a_dict(tmp_path):
    """Schema drift must be skipped, never raised out of a live turn."""
    tail, path = _tail(tmp_path)
    _append(
        path,
        {"type": "assistant", "isSidechain": False, "message": "surprise"},
        _assistant_record("<<<R:rid1>>>\nGood.\n<<<E:rid1>>>"),
    )
    assert tail.poll().body == "Good."


def test_reply_tail_reports_an_opening_marker_with_an_empty_body(tmp_path):
    """`saw_open` must not hinge on the reply being non-empty: the caller
    uses it to say whether the model started answering at all."""
    tail, path = _tail(tmp_path)
    _append(path, _assistant_record("<<<R:rid1>>>"))
    assert tail.poll() is None
    assert tail.saw_open is True


def test_reply_tail_survives_a_missing_file(tmp_path):
    tail = ReplyTail(tmp_path / "nope.jsonl", "rid1")
    assert tail.poll() is None


def test_reply_tail_skips_malformed_lines(tmp_path):
    tail, path = _tail(tmp_path)
    with path.open("a", encoding="utf-8") as f:
        f.write("{not json at all\n")
    _append(path, _assistant_record("<<<R:rid1>>>\nGood.\n<<<E:rid1>>>"))
    assert tail.poll().body == "Good."


def test_reply_tail_never_returns_a_partially_written_line(tmp_path):
    """A record still being appended must not be parsed half-written."""
    tail, path = _tail(tmp_path)
    rec = json.dumps(_assistant_record("<<<R:rid1>>>\nDone.\n<<<E:rid1>>>"))
    with path.open("a", encoding="utf-8") as f:
        f.write(rec[: len(rec) // 2])  # no trailing newline yet
    assert tail.poll() is None
    with path.open("a", encoding="utf-8") as f:
        f.write(rec[len(rec) // 2 :] + "\n")
    assert tail.poll().body == "Done."


# ── TranscriptTail ────────────────────────────────────────────────────────


def test_transcript_tail_reads_only_new_records(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({"type": "assistant", "n": 1}) + "\n")
    tail = TranscriptTail(path)
    assert [r["n"] for r in tail.poll_new_records()] == [1]
    assert tail.poll_new_records() == []  # nothing new yet

    with path.open("a") as f:
        f.write(json.dumps({"type": "assistant", "n": 2}) + "\n")
    assert [r["n"] for r in tail.poll_new_records()] == [2]


def test_transcript_tail_at_end_skips_pre_existing_records(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({"type": "assistant", "n": 1}) + "\n")
    tail = TranscriptTail.at_end(path)
    assert tail.poll_new_records() == []
    with path.open("a") as f:
        f.write(json.dumps({"type": "assistant", "n": 2}) + "\n")
    assert [r["n"] for r in tail.poll_new_records()] == [2]


def test_transcript_tail_leaves_partial_trailing_line_for_next_call(tmp_path):
    path = tmp_path / "session.jsonl"
    complete = json.dumps({"type": "assistant", "n": 1})
    partial = json.dumps({"type": "assistant", "n": 2})
    path.write_text(complete + "\n" + partial[: len(partial) // 2])  # no trailing \n
    tail = TranscriptTail(path)
    records = tail.poll_new_records()
    assert [r["n"] for r in records] == [1]  # the partial line is NOT parsed yet

    # writer finishes the line later
    with path.open("a") as f:
        f.write(partial[len(partial) // 2 :] + "\n")
    assert [r["n"] for r in tail.poll_new_records()] == [2]


def test_transcript_tail_missing_file_returns_empty(tmp_path):
    tail = TranscriptTail(tmp_path / "does-not-exist.jsonl")
    assert tail.poll_new_records() == []


def test_transcript_tail_skips_unparsable_lines(tmp_path):
    path = tmp_path / "session.jsonl"
    good = json.dumps({"type": "assistant", "n": 1})
    path.write_text(f"not json at all\n{good}\n")
    tail = TranscriptTail(path)
    assert [r["n"] for r in tail.poll_new_records()] == [1]


# ── latest_context_tokens (R3, B2 fix) ────────────────────────────────────


def test_latest_context_tokens_returns_last_assistant_usage(tmp_path):
    path = tmp_path / "session.jsonl"
    lines = [
        _assistant_record("first", usage={"cache_read_input_tokens": 1000}),
        {"type": "user", "message": {"content": "hi"}},
        _assistant_record("second", usage={"cache_read_input_tokens": 420_000}),
    ]
    path.write_text("\n".join(json.dumps(rec) for rec in lines) + "\n")
    assert latest_context_tokens(path) == 420_000


def test_latest_context_tokens_missing_file_returns_none(tmp_path):
    assert latest_context_tokens(tmp_path / "nope.jsonl") is None


def test_latest_context_tokens_no_usage_returns_none(tmp_path):
    path = tmp_path / "session.jsonl"
    rec = {"type": "assistant", "message": {"content": "x"}}
    path.write_text(json.dumps(rec) + "\n")
    assert latest_context_tokens(path) is None


def test_latest_context_tokens_reads_only_the_tail(tmp_path):
    """Must not read the whole file for a 100MB+ transcript — proven by
    giving it a tiny tail_bytes window that only reaches the last record."""
    path = tmp_path / "session.jsonl"
    old = _assistant_record("old " * 5000, usage={"cache_read_input_tokens": 1})
    new = _assistant_record("new", usage={"cache_read_input_tokens": 999})
    path.write_text(json.dumps(old) + "\n" + json.dumps(new) + "\n")
    assert latest_context_tokens(path, tail_bytes=200) == 999


def test_latest_context_tokens_sums_read_creation_and_input(tmp_path):
    """B2 fix (2026-08-22): the real live-transcript shape this bug was
    found against — a cache-rewrite turn reports a low
    cache_read_input_tokens (30698) alongside a much larger
    cache_creation_input_tokens (580621); the real context is the SUM
    (~611k), not the read figure alone."""
    path = tmp_path / "session.jsonl"
    rec = _assistant_record(
        "x",
        usage={
            "cache_read_input_tokens": 30_698,
            "cache_creation_input_tokens": 580_621,
            "input_tokens": 4,
        },
    )
    path.write_text(json.dumps(rec) + "\n")
    assert latest_context_tokens(path) == 30_698 + 580_621 + 4


def test_latest_context_tokens_skips_synthetic_model_records(tmp_path):
    """B2 fix: `model: "<synthetic>"` records (Claude Code's own meta/
    compaction entries) report usage of 0 and must not be mistaken for a
    real turn showing the context has dropped to zero — that would unarm
    the R3 latch exactly as wrongly as the cache-rewrite undercount did."""
    path = tmp_path / "session.jsonl"
    lines = [
        _assistant_record(
            "real turn", usage={"cache_read_input_tokens": 420_000}
        ),
        _assistant_record(
            "synthetic meta entry",
            usage={"cache_read_input_tokens": 0},
            model="<synthetic>",
        ),
    ]
    path.write_text("\n".join(json.dumps(rec) for rec in lines) + "\n")
    assert latest_context_tokens(path) == 420_000


# ── latest_reply (backlog item 14: /resend) ───────────────────────────────


def test_latest_reply_missing_file_is_unavailable(tmp_path):
    status, body = latest_reply(tmp_path / "does-not-exist.jsonl")
    assert status == "unavailable"
    assert body is None


def test_latest_reply_no_assistant_record_is_empty(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n")
    status, body = latest_reply(path)
    assert status == "empty"
    assert body is None


def test_latest_reply_empty_file_is_empty(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text("")
    status, body = latest_reply(path)
    assert status == "empty"
    assert body is None


def test_latest_reply_open_span_no_closing_marker_is_unclosed(tmp_path):
    """The real shape of the three named incidents (bfbe3335, df4f87ef,
    9dd35326): <<<R:id>>>, full reply text, end of message, no <<<E:id>>>.

    latest_reply() itself stays liveness-agnostic (pure/lock-free): it
    reports this as its own distinct "unclosed" outcome carrying the
    recovered body, rather than collapsing it into "in_progress" — F1 fix
    (2026-08-22). last_reply_for_resend() in claude_session.py is what
    turns this into the final ready/in_progress verdict, using
    is_turn_active()."""
    path = tmp_path / "session.jsonl"
    rec = _assistant_record("<<<R:bfbe3335>>>\nThe full answer text.\nend of turn.")
    path.write_text(json.dumps(rec) + "\n")
    status, body = latest_reply(path)
    assert status == "unclosed"
    assert body == "The full answer text.\nend of turn."


def test_latest_reply_no_marker_at_all_is_in_progress(tmp_path):
    """The most recent assistant text has no <<<R:id>>> marker yet — most
    plausibly still-streaming preamble ahead of the wrapped answer. Honest
    as 'still working', not 'empty' and not 'ready'."""
    path = tmp_path / "session.jsonl"
    rec = _assistant_record("Let me check that file first...")
    path.write_text(json.dumps(rec) + "\n")
    status, body = latest_reply(path)
    assert status == "in_progress"
    assert body is None


def test_latest_reply_open_marker_in_earlier_record_reports_in_progress(tmp_path):
    """Reviewer-found (round 2) shape: a reply's open <<<R:id>>> marker
    lands in one record and its closing <<<E:id>>> lands in the NEXT
    assistant record. latest_reply() only ever inspects the single most
    recent assistant text record (by design, see module docstring) — from
    its point of view the last record carries no marker at all, so this
    is indistinguishable from the plain no-marker case and reports
    "in_progress" here too. No cross-record stitching is attempted.
    last_reply_for_resend() in claude_session.py is what turns this into
    the honest "no_markers" status when the session is actually idle
    (F-1 fix, round 2, 2026-08-22) — this function itself is unchanged."""
    path = tmp_path / "session.jsonl"
    lines = [
        _assistant_record("<<<R:rid1>>>\nThe answer, split across"),
        _assistant_record(" records.\n<<<E:rid1>>>\n"),
    ]
    path.write_text("\n".join(json.dumps(rec) for rec in lines) + "\n")
    status, body = latest_reply(path)
    assert status == "in_progress"
    assert body is None


def test_latest_reply_closed_pair_is_ready(tmp_path):
    path = tmp_path / "session.jsonl"
    rec = _assistant_record("<<<R:rid1>>>\nHello there.\n<<<E:rid1>>>\n")
    path.write_text(json.dumps(rec) + "\n")
    status, body = latest_reply(path)
    assert status == "ready"
    assert body == "Hello there."


def test_latest_reply_uses_the_most_recent_assistant_record(tmp_path):
    """An older, already-closed reply must not be resurfaced once a newer
    turn has started producing text — that would resend a stale answer
    while a fresh one is in flight."""
    path = tmp_path / "session.jsonl"
    lines = [
        _assistant_record("<<<R:old>>>\nOld answer.\n<<<E:old>>>\n"),
        {"type": "user", "message": {"content": "another question"}},
        _assistant_record("Still thinking about the new question..."),
    ]
    path.write_text("\n".join(json.dumps(rec) for rec in lines) + "\n")
    status, body = latest_reply(path)
    assert status == "in_progress"
    assert body is None


def test_latest_reply_rejects_sidechain(tmp_path):
    """A background subagent's own text must never be mistaken for the
    main turn's reply (same rule as extract_reply_from_record)."""
    path = tmp_path / "session.jsonl"
    rec = _assistant_record(
        "<<<R:rid1>>>\nSubagent text.\n<<<E:rid1>>>\n", is_sidechain=True
    )
    path.write_text(json.dumps(rec) + "\n")
    status, body = latest_reply(path)
    assert status == "empty"
    assert body is None


def test_latest_reply_skips_unparsable_and_non_dict_lines(tmp_path):
    path = tmp_path / "session.jsonl"
    good = _assistant_record("<<<R:rid1>>>\nHi.\n<<<E:rid1>>>\n")
    path.write_text(
        "not json at all\n" + json.dumps([1, 2, 3]) + "\n" + json.dumps(good) + "\n"
    )
    status, body = latest_reply(path)
    assert status == "ready"


def test_latest_reply_skips_malformed_record_with_string_message(tmp_path):
    """F5 fix: a record where ``message`` is a string (not a dict) must be
    skipped, not raise AttributeError out of the scan and take the whole
    /resend path down with it — the module docstring already promises to
    skip malformed records rather than raise."""
    path = tmp_path / "session.jsonl"
    malformed = {"type": "assistant", "isSidechain": False, "message": "oops"}
    good = _assistant_record("<<<R:rid1>>>\nHi.\n<<<E:rid1>>>\n")
    path.write_text(json.dumps(good) + "\n" + json.dumps(malformed) + "\n")
    status, body = latest_reply(path)
    assert status == "ready"
    assert body == "Hi."
    assert body == "Hi."
