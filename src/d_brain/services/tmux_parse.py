"""Pure parsing of `tmux capture-pane -p` output from an interactive
Claude Code session. No subprocess/tmux access — text in, value out.

Kept separate from claude_session.py so the fragile parsing logic is
fully unit-testable against real capture fixtures.

Design invariants (from live spikes + adversarial review):
* The model's answer markers are LINE-ANCHORED (each on its own line).
  The input echo shows the markers INLINE (mid-sentence), and the model
  quoting the marker syntax also appears inline. So matching only
  line-anchored markers distinguishes the real answer from the echo and
  from inline self-references — no fragile occurrence-counting needed.
* State signatures are matched only against the CHROME region (the bottom
  of the pane: footer/banner/idle line), never the whole transcript, so a
  reply that *mentions* "usage limit" or "/login" cannot be misclassified.

The rate-limit / logged-out signatures are not yet confirmed against a
live session and may need adjustment per CLI version (see tests, open Q #2).
"""

import re
from datetime import UTC, datetime, timedelta
from enum import Enum

# How many trailing lines count as "chrome" (footer/banner/idle region).
# State signatures are matched only here, not against the transcript body.
#
# CAVEAT (2026-08-20, measured live): this only holds if the pane is TALL.
# The brain's pane had drifted to 80x23 (a narrow attached client shrinks the
# window and the size sticks after detach), so 18 trailing lines covered
# nearly the whole SCREEN — i.e. the model's own answer text. That is how a
# reply merely discussing rate limits produced a real "⏳ Лимит подписки
# исчерпан" to the user. Two independent guards now exist: ClaudeSession
# re-asserts the pane geometry, and strip_reply_bodies() below removes
# model-authored spans before any state signature is matched.
_CHROME_LINES = 18


class PaneState(str, Enum):
    """Coarse state of the interactive session, read from the pane text."""

    TRUST_PROMPT = "trust_prompt"  # "Is this a project you trust?" — needs Enter
    BYPASS_PROMPT = "bypass_prompt"  # "Bypass Permissions mode" accept — needs "2"
    STARTING = "starting"  # welcome box visible, not yet idle
    READY = "ready"  # idle prompt / bypass-permissions footer
    RATE_LIMITED = "rate_limited"  # usage limit hit — do NOT kill, wait for reset
    LOGGED_OUT = "logged_out"  # auth lost — needs re-login
    UNKNOWN = "unknown"


def _require_rid(rid: str) -> None:
    if not rid:
        raise ValueError("rid must be a non-empty string")


def _line_anchored(rid: str, kind: str) -> re.Pattern[str]:
    # The marker must be at the END of its line (only whitespace after it).
    # Any prefix is allowed — Claude Code prefixes the first answer line with
    # "⏺ " and indents the rest. The input echo has TEXT after the marker
    # ("<<<R:id>>> and a line..."), so it never matches end-of-line.
    return re.compile(rf"(?m)^.*?<<<{kind}:{re.escape(rid)}>>>[ \t]*\r?$")


def extract_reply(text: str, rid: str) -> str | None:
    """Return the text of the last well-formed, line-anchored
    ``<<<R:rid>>> .. <<<E:rid>>>`` pair, stripped, or ``None``.

    Only line-anchored markers are considered (the input echo and inline
    self-references are mid-line and thus ignored). The chosen span must not
    contain another line-anchored marker of either kind, so a stray end
    marker cannot make the span swallow chrome.
    """
    _require_rid(rid)
    opens = list(_line_anchored(rid, "R").finditer(text))
    ends = list(_line_anchored(rid, "E").finditer(text))
    if not opens or not ends:
        return None

    # Walk end markers from last to first; pair each with the nearest
    # preceding open marker and accept the first span with no inner marker.
    open_starts = [m.start() for m in opens]
    for end_m in reversed(ends):
        end_pos = end_m.start()
        preceding = [s for s in open_starts if s < end_pos]
        if not preceding:
            continue
        start_m = next(m for m in opens if m.start() == preceding[-1])
        inner = text[start_m.end() : end_pos]
        # Reject if another line-anchored marker hides inside the span.
        if _line_anchored(rid, "E").search(inner) or _line_anchored(rid, "R").search(
            inner
        ):
            continue
        # The pane is a RENDERED screen: undo the terminal's soft wraps here,
        # at the single point every delivered reply passes through.
        return unwrap_soft_breaks(inner).strip()
    return None


# A line at least this long was almost certainly cut by the terminal rather
# than by the model: the pane is 200 columns, so soft-wrapped lines come back
# at ~190-198 chars, while deliberate short lines (list items, headings, the
# last line of a paragraph) essentially never reach this length.
_WRAP_MIN_LEN = 140
# Lines that always begin something new — never glue these onto the previous
# line even when it looks full.
_BLOCK_START_RE = re.compile(r"^\s*(?:[-*•>#|]|\d+[.)]\s|```|<pre|<b>|\[)")
_FENCE_RE = re.compile(r"^\s*(?:```|</?pre\b)")


def unwrap_soft_breaks(text: str) -> str:
    """Glue terminal soft-wraps back into single paragraphs.

    `capture-pane` returns the RENDERED screen, so every line break the
    terminal inserted to fit the width is indistinguishable from one the model
    typed — and they reached Telegram as real breaks, chopping sentences into
    a ragged column ("приходится бегать глазами между строчками"). Widening
    the pane only moves the wrap point; it does not remove it.

    Deliberately conservative: a line is treated as a continuation only if the
    PHYSICAL line before it was long enough to have been cut, it does not open
    a list/quote/heading/code block, and neither sits inside a fenced or <pre>
    region — where every break is meaningful and must survive untouched.
    """
    out: list[str] = []
    last_raw_len = 0  # length of the previous PHYSICAL line, not the joined one
    in_block = False
    for raw in text.split("\n"):
        if _FENCE_RE.search(raw):
            marker = raw.strip()
            if marker.startswith("```"):
                in_block = not in_block
            elif marker.startswith("</pre"):
                in_block = False
            else:  # <pre …>
                in_block = True
            out.append(raw)
            last_raw_len = 0
            continue
        if (
            not in_block
            and out
            and raw.strip()
            and last_raw_len >= _WRAP_MIN_LEN
            and not _BLOCK_START_RE.match(raw)
        ):
            out[-1] = out[-1].rstrip() + " " + raw.strip()
        else:
            out.append(raw)
        last_raw_len = len(raw)
    return "\n".join(out)


_ANY_END_MARKER_RE = re.compile(r"(?m)^.*?<<<E:(\w+)>>>[ \t]*\r?$")


def find_latest_reply(text: str) -> tuple[str, str] | None:
    """Return ``(rid, body)`` for the last well-formed, line-anchored marker
    pair in ``text`` for ANY rid, or ``None`` if none is present.

    Generalizes :func:`extract_reply` for callers that don't know the rid in
    advance — e.g. a poller looking for a reply nobody's ``ask()`` is
    currently waiting on (an orphaned proactive reply). Candidate rids are
    read off line-anchored end markers (latest first); each candidate is
    validated with the same pairing/no-inner-marker rules as
    :func:`extract_reply`, so this can never disagree with it.
    """
    for end_m in reversed(list(_ANY_END_MARKER_RE.finditer(text))):
        rid = end_m.group(1)
        body = extract_reply(text, rid)
        if body is not None:
            return rid, body
    return None


def find_unhandled_replies(text: str, handled: set[str]) -> list[tuple[str, str]]:
    """Every complete marker pair in ``text`` whose rid is not in ``handled``,
    oldest first.

    :func:`find_latest_reply` only ever surfaces the LAST pair, which loses a
    reply that a newer turn superseded before the poller got to it (the
    observed "answer never arrived" race). Returning all unhandled pairs makes
    delivery independent of poll timing — dedup is the caller's ``handled``
    set, not "is it currently the newest one".
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for end_m in _ANY_END_MARKER_RE.finditer(text):
        rid = end_m.group(1)
        if rid in handled or rid in seen:
            continue
        body = extract_reply(text, rid)
        if body is None:
            continue
        seen.add(rid)
        out.append((rid, body))
    return out


def find_pending_replies(text: str, handled: set[str]) -> list[tuple[str, str]]:
    """Unhandled pairs that are NEWER than the newest handled pair on screen.

    The pane is append-ordered, so a pair sitting ABOVE one we know we already
    delivered must be older than it — and therefore already delivered too,
    however it got there. Without this rule anything that shifts the capture
    window re-surfaces old replies as "never seen": observed for real on
    2026-08-20, when widening the pane made the TUI reflow its transcript,
    older pairs fit into the 200-line capture, and a reply from hours earlier
    was sent to the user a second time.

    When no handled pair is visible at all there is no watermark to trust, so
    every unhandled pair is returned (a genuinely fresh pane).
    """
    ordered = find_unhandled_replies(text, set())
    watermark = -1
    for i, (rid, _) in enumerate(ordered):
        if rid in handled:
            watermark = i
    return [(rid, body) for rid, body in ordered[watermark + 1 :] if rid not in handled]


def reply_rids(text: str) -> set[str]:
    """Rids of every complete marker pair visible in ``text``.

    Used to seed the delivered-rid store on first run so a fresh install does
    not flush the whole scrollback into the chat.
    """
    return {rid for rid, _ in find_unhandled_replies(text, set())}


_ANY_OPEN_MARKER_RE = re.compile(r"(?m)^.*?<<<R:(\w+)>>>[ \t]*\r?$")


def open_reply_rids(text: str) -> set[str]:
    """Rids of every line-anchored ``<<<R:rid>>>`` in ``text``, closed or not.

    Superset of :func:`reply_rids` (which only returns rids with a COMPLETE
    pair) — feeds the orphan-salvage path (Fable audit R2a) that rescues an
    UNCLOSED span nobody's ``ask()`` is waiting on.
    """
    return {m.group(1) for m in _ANY_OPEN_MARKER_RE.finditer(text)}


def has_marker(text: str, rid: str, kind: str) -> bool:
    """True iff a line-anchored ``<<<{kind}:{rid}>>>`` line is present.

    Exposed so a caller can tell "this marker was seen at some point, even
    though it is not extractable from the CURRENT capture" (e.g. it scrolled
    out of the ``-S`` capture window) apart from "this marker never appeared
    at all" — the distinction the Fable audit's F2 class (3.3% of turns,
    honest ``no reply markers ever appeared`` message) needs (R2c).
    """
    _require_rid(rid)
    return bool(_line_anchored(rid, kind).search(text))


#
# BUG FIXED 2026-08-20 (found by re-driving the plan against a live capture):
# the previous version compiled the WHOLE pattern with re.S, which makes the
# leading `^.*?` cross newlines too. Because `^` under MULTILINE already
# matches position 0 (the very start of the string), the lazy `.*?` never
# needed to restart at the marker's own line — it just swallowed every line
# ABOVE the open marker, including a real rate-limit banner sitting higher in
# the pane, and strip_reply_bodies() deleted the very banner it exists to
# protect. Fix: re.S is dropped globally so `^.*?<<<R:...>>>...$` and the
# closing `^.*?<<<E:...>>>...$` stay confined to their OWN line (the usual
# case is a bare marker line, or the model's answer-prefix "⏺ " before it);
# only the inner `(?s:.*?)` body — the actual multi-line reply text between
# the markers — is allowed to cross newlines.
_REPLY_SPAN_RE = re.compile(
    r"(?m)^.*?<<<R:(\w+)>>>[ \t]*\r?$"
    r"(?s:.*?)"
    r"^.*?<<<E:\1>>>[ \t]*\r?$",
)


def strip_reply_bodies(text: str) -> str:
    """Drop complete ``<<<R:id>>> … <<<E:id>>>`` spans from ``text``.

    Everything inside a marker pair is text the MODEL wrote. State signatures
    ("usage limit reached", "please run /login") describe what the CLI itself
    is showing, so model-authored prose must never feed them — otherwise
    answering a question *about* rate limits reports the session as rate
    limited. Only closed spans are removed: an unterminated one may be the
    turn during which a real limit banner appears, and that must still count.
    """
    return _REPLY_SPAN_RE.sub("", text)


def is_complete(text: str, rid: str) -> bool:
    """True iff a complete line-anchored answer pair is present.

    Replaces the fragile "count >= 2" heuristic: because the echo is inline,
    a single line-anchored pair already means the model's answer is done.
    """
    _require_rid(rid)
    return extract_reply(text, rid) is not None


# Signature tables. Order of checks in classify_state encodes priority.
# TRUST anchors on the numbered menu (structural), not the prose sentence,
# so the model describing the trust prompt cannot trigger it.
_TRUST_MENU_RE = re.compile(r"(?m)^\s*(?:❯\s*)?1\.\s+Yes, I trust this folder")
# First run with --dangerously-skip-permissions on a fresh config dir shows a
# full-screen "WARNING: Claude Code running in Bypass Permissions mode" with a
# numbered menu (1. No, exit / 2. Yes, I accept). classify_state knew nothing
# of it → UNKNOWN → "not ready in 90s" on every fresh install. Accepting needs
# the active choice "2" (the safe default ❯ sits on "1. No, exit").
# DOUBLE anchor: the unique warning TITLE *and* the numbered accept line must
# BOTH be present. "Yes, I accept" alone is reply-natural (a model could emit a
# numbered consent list), so unlike TRUST's single unique line we also require
# the verbatim title — no reply reproduces both at once.
_BYPASS_TITLE_RE = re.compile(
    r"WARNING: Claude Code running in Bypass Permissions mode"
)
_BYPASS_MENU_RE = re.compile(r"(?m)^\s*(?:❯\s*)?2\.\s+Yes, I accept")
# Rate-limit signature. DELIBERATELY NARROW (tightened 2026-08-20 after three
# false "⏳ Лимит подписки исчерпан" reports in one day): the old pattern fired
# on the bare words "usage limit" / "rate limit" / "limit reached" / "resets
# at", every one of which shows up in ordinary conversation and in tool output
# (an API gateway schema has a rate_limit column; discussing this very bug quotes
# the banner). Only the CLI's own "you are blocked NOW" phrasings count — a
# soft "approaching your weekly limit" heads-up must NOT stop the session.
_RATE_RE = re.compile(
    r"(?:usage|session|5-hour|weekly|opus|sonnet)\s+limit\s+(?:reached|exceeded)"
    r"|you'?ve (?:hit|reached) your (?:\w+ ){0,2}limit"
    r"|limit reached\s*[·∙|-]\s*resets"
    r"|limit will reset at",
    re.I,
)
# "resets 12am (UTC)" / "resets at 3pm" / "resets at 15:00" — the wake-up time
# the watchdog schedules its nudge against. Minutes and the meridiem are both
# optional because the CLI renders whole hours as bare "12am".
_RESET_TIME_RE = re.compile(
    r"resets?(?:\s+at)?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    re.I,
)
_LOGGED_OUT_RE = re.compile(
    r"invalid api key|please run /login|logged out|please log ?in|"
    r"authentication (failed|required|expired)|session expired|"
    # First-run onboarding screens (fresh CLAUDE_CONFIG_DIR): they need a
    # human, restarts won't help — alert like a logout, don't kill.
    r"select login method|syntax theme: \w+ \(ctrl\+t to disable\)|"
    # Security notes screen of the first run ("Press Enter to continue…").
    r"security notes:[\s\S]*press enter to continue",
    re.I,
)
# READY signals. The bypass footer is our always-present anchor (we launch
# with --dangerously-skip-permissions); it can sit above a blank bottom, so
# it is matched over the WHOLE pane. The idle ❯ is a secondary signal matched
# only in chrome (a bare ❯ elsewhere could be model output).
_FOOTER_RE = re.compile(r"bypass permissions on")
_IDLE_RE = re.compile(r"(?m)^\s*❯")
_STARTING_RE = re.compile(r"Claude Code v\d", re.I)
# Active-turn marker. The TUI shows "(esc to interrupt)" next to its spinner
# for the whole turn, so its absence + an idle ❯ is the idle signal. The
# bypass footer is ALWAYS on screen under --dangerously-skip-permissions and
# must never be used as an idle signal by itself.
# Two working signatures: the legacy "esc to interrupt" hint, and the newer
# spinner with a live elapsed-time + token counter, e.g.
# "✢ Razzle-dazzling… (44s · ↓1.8k tokens)". Newer Claude Code dropped the
# hint entirely, so matching only the old string blinded the stall detector
# and false-killed every turn longer than stall_timeout.
# Third signature added 2026-08-20 after measuring the LIVE pane mid-turn:
# while the turn waits on a background agent the TUI shows neither the hint
# nor a parenthesised spinner — just "✻ Waiting for 1 background agent to
# finish" plus an agent row "… 3m 5s · ↓ 88.3k tokens". is_working() returned
# False on a demonstrably live turn, which is exactly how a 6-9 minute
# tool-dense turn tripped the stall detector. The elapsed-time clause also
# drops the "(" anchor so the "2m 15s ·" form counts, not only "(44s ·" —
# but that made it match ANY tool-output line shaped like "<Ns> ·", e.g.
# "tool ran 12s · ok" (verified live 2026-08-20 review: is_working() on that
# string was True). The CLI's own elapsed-time readout is always paired with
# its token-count arrow ("12s · ↓1.8k tokens" / "3m 5s · ↓ 88.3k tokens") —
# requiring "↓" right after the "·" keeps both real spinner forms matching
# while excluding arbitrary tool output that merely contains "Ns ·".
# Split so callers that track liveness OVER TIME (ask()'s stall loop, the
# watchdog) can require the PROGRESS signatures to actually show change —
# see is_working_progressing() below. "esc to interrupt" stays exempt: it is
# the whole-turn hint, not a per-tick counter, and a real silent task holds
# it byte-identical on screen for its entire duration (see
# test_ask_long_silent_work_not_interrupted) — only the newer signatures
# that are EXPECTED to tick over need the extra guard.
_WORKING_STATIC_RE = re.compile(r"esc to interrupt", re.I)
_WORKING_PROGRESS_RE = re.compile(
    r"(?:\d+m\s*)?\d+s\s*·\s*↓"
    r"|waiting for \d+ background agents? to finish",
    re.I,
)
_WORKING_RE = re.compile(
    _WORKING_STATIC_RE.pattern + "|" + _WORKING_PROGRESS_RE.pattern, re.I
)
# Idle = a BARE ❯ on its own line (empty input). A menu selector ("❯ 1. Yes…")
# has text after the chevron and must NOT count — otherwise a turn stuck on an
# approval/menu prompt would be mistaken for completion (wrap=False).
_IDLE_BARE_RE = re.compile(r"(?m)^\s*❯\s*$")


def _chrome(text: str) -> str:
    # Model-authored spans are removed FIRST: they are transcript, never
    # chrome, and letting them into this window is what made the CLI's state
    # signatures fire on the model's own prose (see _CHROME_LINES).
    return "\n".join(strip_reply_bodies(text).splitlines()[-_CHROME_LINES:])


_CHROME_LINE_RE = re.compile(
    r"^\s*❯?\s*$"               # empty / bare idle prompt
    r"|^\s*─+\s*$"              # box rule
    r"|bypass permissions on"   # always-present footer
    r"|esc to interrupt"        # working spinner hint
    r"|^\s*⏵⏵"                  # footer arrows
    r"|^\s{2}\S.* \| .* \| "    # status line: "  name | model | path"
)


def strip_chrome(text: str) -> str:
    """Best-effort body of a non-marker turn: drop TUI chrome lines.

    Used for ``wrap=False`` turns where no marker pair exists; the result may
    still contain the input echo — callers treat it as informational text,
    not a structured reply.
    """
    kept = [ln for ln in text.splitlines() if not _CHROME_LINE_RE.search(ln)]
    return "\n".join(kept).strip()


def is_working(text: str) -> bool:
    """True iff the pane shows an ACTIVE turn (the working spinner).

    The shared liveness predicate for ask()'s stall detector and the
    watchdog: silence is not a hang signal — a long task that prints nothing
    still shows '(esc to interrupt)'. Hung == stuck WITHOUT this marker.
    """
    return bool(_WORKING_RE.search(_chrome(text)))


def is_working_progressing(
    text: str, prev_text: str | None, *, trust_static: bool = True
) -> bool:
    """Liveness check for callers that poll repeatedly over time (ask()'s
    stall loop, the watchdog): like :func:`is_working`, but the PROGRESS
    signatures (elapsed-timer, background-agent wait) only count when the
    chrome actually changed since the previous poll.

    Added 2026-08-20 after measuring a real false negative for the stall
    detector: a session frozen mid-turn (dead subagent, a process stuck in
    D-state) still shows "Waiting for 1 background agent to finish" / "3m 5s
    ·" on its LAST rendered frame forever, which made is_working() report
    True on an unchanging screen and disarmed both ask()'s stall timer and
    the watchdog's hang detector — the exact class of hang stall detection
    exists to catch. The legacy "esc to interrupt" hint is exempt: it is a
    whole-turn marker, not a per-tick counter, and a genuinely silent-but-
    alive task holds it byte-identical on screen for its entire duration.

    ``trust_static`` (backlog item 13, 2026-09-04) bounds that exemption in
    TIME without touching it in kind. Default True ⇒ byte-identical to the
    behaviour described above, which is what every legacy call site gets.
    Passing False says "this caller has watched the pane sit byte-identical
    for longer than any single genuinely-quiet turn plausibly lasts" — and
    then the static hint stops being a free pass: the whole ``_WORKING_RE``
    match (static hint included) must be backed by chrome that actually
    changed since ``prev_text``, exactly the rule the PROGRESS signatures
    already live under. The measurement that motivates it: 6485 of 6512
    occurrences (99.6%) of "esc to interrupt" on a real pane.log sit in the
    persistent footer, which this CLI version shows whenever ANYTHING is
    interruptible — so on its own it is not evidence that THIS turn is
    alive. Deciding WHEN to stop trusting is deliberately left to the
    caller (it owns the clock and the pane.log growth signal); this
    function stays pure. ``_WORKING_STATIC_RE`` and :func:`is_working` are
    intentionally left byte-identical — their other consumers rely on the
    unbounded semantics.
    """
    chrome = _chrome(text)
    if trust_static and _WORKING_STATIC_RE.search(chrome):
        return True
    # With trust_static=True the static hint has already been consumed
    # above, so _WORKING_RE here is equivalent to _WORKING_PROGRESS_RE —
    # legacy behaviour is preserved exactly. With trust_static=False the
    # static hint falls through to the same change-required rule.
    if not _WORKING_RE.search(chrome):
        return False
    return prev_text is None or _chrome(prev_text) != chrome


_SURVEY_RE = re.compile(r"How is Claude doing this session\?")


def has_survey_prompt(text: str) -> bool:
    """True iff the periodic feedback survey is on screen.

    Claude Code occasionally shows "How is Claude doing this session?
    1: Bad 2: Fine 3: Good 0: Dismiss" — it pollutes the chrome and must be
    dismissed (key 0), never treated as a stalled turn.
    """
    return bool(_SURVEY_RE.search(text))


def is_idle(text: str) -> bool:
    """True iff the session sits at an idle input prompt (no active turn).

    Unlike READY in classify_state (anchored on the always-present bypass
    footer), this checks the chrome for an idle ``❯`` AND the absence of the
    working spinner — usable as a turn-completion signal for prompts that
    produce no marker pair.
    """
    if not text.strip():
        return False
    chrome = _chrome(text)
    if _WORKING_RE.search(chrome):
        return False
    return bool(_IDLE_BARE_RE.search(chrome))


# ── main-turn liveness / salvage extraction ───────────────────────────────
#
# Added for backlog item 10 (2026-08-21): a reply is lost forever when the
# model finishes its turn but never emits the closing `<<<E:id>>>` marker.
# `is_working()`/`is_working_progressing()` above are DELIBERATELY left
# untouched — they answer "is ANYTHING happening in this pane" (main turn OR
# a background-agent list row), which is exactly right for the stall/hang
# detectors, and widening what counts as "working" there would narrow what
# those detectors can ever catch (see their docstrings). This section answers
# a different, narrower question: is the turn `ask()` itself sent a prompt to
# still running — deliberately EXCLUDING background-agent list rows, which
# tick every second on their own and are not evidence the MAIN turn survives.

# The MAIN spinner's elapsed-time readout is always wrapped in parentheses
# directly after the spinner glyph, e.g. "✢ Razzle-dazzling… (44s · ↓1.8k
# tokens)" or "Warping… (2m 33s · ↓ 8.4k tokens · thought for 38s)". A
# background-agent list row shows the same "<elapsed> · ↓<tokens>" shape but
# WITHOUT the enclosing "(" — e.g. "◯ general-purpose  <text>   54s · ↓
# 79.7k tokens" — so anchoring on the opening paren is what tells the two
# apart.
#
# WIDENED 2026-08-21 (round-2 review finding): the elapsed-time clause only
# accounted for minutes+seconds ("(44s · ↓" / "(2m 33s · ↓"), so an hour-scale
# readout — "(1h 5m 12s · ↓ 9.9k tokens)", which the CLI emits for a long
# turn — silently failed to match, making is_main_turn_active() false-
# negative on a genuinely still-live main turn. The hour segment is optional
# (like the minute segment already was) so the short forms keep matching
# unchanged.
_MAIN_SPINNER_RE = re.compile(r"\((?:\d+h\s*)?(?:\d+m\s*)?\d+s\s*·\s*↓", re.I)
# The main turn can itself be blocked waiting on a subagent — that IS
# main-turn-active (the turn has not returned control yet), unlike a
# "Worked for … · N background tasks still running" summary line below,
# which means the MAIN turn already finished and only background
# bookkeeping remains.
_WAITING_FOR_AGENTS_RE = re.compile(
    r"waiting for \d+ background agents? to finish", re.I
)
# The TUI's turn-summary readout once the main turn returns control. The VERB
# IS RANDOMIZED — measured over 258 real summary lines in ~/.dbrain/pane.log
# (2026-08-22): Worked 53, Brewed 42, Baked 39, Cooked 34, Crunched 33,
# Churned 31, Cogitated 26. The original `Worked for \d` therefore matched
# only 20.5% of finished turns, which is why salvage never fired in
# production while passing its own "Worked for 2m 22s" fixtures.
# Deliberately LINE-ANCHORED with at most one leading glyph (always "✻" in
# practice), which makes it STRICTER than the old substring match: a model's
# own prose "…it worked for 30s…" mid-sentence cannot match.
_TURN_SUMMARY_RE = re.compile(
    r"(?m)^\s*(?:[^\w\s]\s*)?[A-Z][a-z]+ for (?:\d+h\s*)?(?:\d+m\s*)?\d+s\b"
)
# A run of box-drawing characters — the TUI's horizontal rule between the
# transcript and the footer.
_BOX_RULE_RE = re.compile(r"^\s*─+\s*$")


def _static_hint_outside_footer(chrome: str) -> bool:
    """True iff the legacy whole-turn hint "esc to interrupt" appears on a
    line that is NOT the TUI footer.

    Measured 2026-08-22 on ~/.dbrain/pane.log: 6485 of 6512 occurrences of
    "esc to interrupt" (99.6%) sit inside the persistent footer line
    ("⏵⏵ bypass permissions on · N background tasks · esc to interrupt · …"),
    which this CLI version shows whenever ANYTHING is interruptible —
    including background tasks and monitors — with the main turn long
    finished. Reading it as main-turn liveness pinned last_main_turn_active
    forever and blocked BOTH the salvage and the ceiling exits.

    Deliberately scoped to is_main_turn_active(): _WORKING_STATIC_RE itself,
    is_working(), is_working_progressing() and is_idle() are left byte-
    identical so the watchdog and the stall detector are not touched by this
    change.
    """
    return any(
        _WORKING_STATIC_RE.search(ln)
        and not _FOOTER_RE.search(ln)
        and not ln.lstrip().startswith("⏵⏵")
        for ln in chrome.splitlines()
    )


def is_main_turn_active(text: str) -> bool:
    """True iff the pane shows the MAIN turn (the one `ask()` sent a prompt
    to) is still running — deliberately excluding background-agent list
    rows, which tick every second on their own and are not evidence the
    main turn survives.

    Used by `ask()`'s pre-send busy-wait gate (Defect B: an idle pane with
    only background-agent rows visible used to be mistaken for "busy" and
    refused to type at all, via bare `is_working()`) and by the
    salvage/ceiling logic in `claude_session.py` (Defect A: neither
    `is_working()` nor `is_working_progressing()` can ever go False while a
    background task is merely listed, so `ask()`'s stall loop never escaped
    a turn whose MAIN answer had already finished).
    """
    chrome = _chrome(text)
    if _static_hint_outside_footer(chrome):
        return True
    if _MAIN_SPINNER_RE.search(chrome):
        return True
    if _WAITING_FOR_AGENTS_RE.search(chrome):
        return True
    return False


# Auth failure the CLI prints for the current turn ("⎿  API Error: 401 … ·
# Please run /login", "⎿  Invalid API key · …"). The chrome-only LOGGED_OUT
# signatures miss it when the screen is not full, and the turn then ends with
# no markers. Only the CLI's own message templates count.
_TURN_AUTH_ERROR_RE = re.compile(
    r"(?i)^\s*⎿\s+(?:invalid api key\b|api error: 401\b|.*· please run /login\b|"
    r".*oauth token has expired)"
)


def _auth_error_block(lines: list[str]) -> bool:
    """The last "⎿" block of the turn is a CLI auth error, not tool output.

    Layout rules, all conservative (any doubt -> False, so the ask falls back
    to its old timeout path instead of a false "log in again"):
    * the block text is the "⎿" line plus its indented wrap continuations
      (the pane is 200 columns, the real 401 message is longer);
    * walking up from the block over indented lines and earlier "⎿" lines must
      reach a blank line or the start of the turn, never a "●" header (tool
      results, wrapped tool headers and nested sub-agent output all hang
      under one);
    * below the block only blank lines and a turn summary may appear before
      the "❯" prompt; everything under the prompt is footer.
    """
    idx = next(
        (i for i in range(len(lines) - 1, -1, -1) if lines[i].lstrip().startswith("⎿")),
        None,
    )
    if idx is None:
        return False
    end = idx + 1
    while end < len(lines) and lines[end][:1].isspace() and lines[end].strip() \
            and not lines[end].lstrip().startswith(("⎿", "●", "❯", "✻")):
        end += 1
    block = " ".join(ln.strip() for ln in lines[idx:end])
    if not _TURN_AUTH_ERROR_RE.match(block):
        return False
    up = idx - 1
    while up >= 0 and lines[up].strip():
        if lines[up].lstrip().startswith("●"):
            return False
        up -= 1
    for line in lines[end:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("✻") or _TURN_SUMMARY_RE.match(line):
            continue
        return stripped.startswith("❯") or bool(_BOX_RULE_RE.match(line))
    return False


def turn_auth_error(text: str, rid: str) -> bool:
    """True when the CLI reported an auth failure after the prompt of ``rid``.

    The prompt echo carries ``<<<E:rid>>>`` inline, so only the text after
    its LAST occurrence (the echo, or a closing marker) is inspected: an
    older turn's error never counts for this one. Only once the main turn
    is over."""
    marker = f"<<<E:{rid}>>>"
    if marker not in text or is_main_turn_active(text):
        return False
    after = text.rsplit(marker, 1)[-1]
    # Drop the rest of the echoed prompt line ("… line — end with it.").
    lines = after.splitlines()[1:]
    while lines and lines[0].strip() and not lines[0].lstrip().startswith(("⎿", "●")):
        lines.pop(0)
    return _auth_error_block(lines)


def main_turn_finished(text: str) -> bool:
    """True iff the main turn is demonstrably DONE: not active (see
    :func:`is_main_turn_active`) AND at least one POSITIVE finished signal
    is present in chrome. Requiring a positive signal (not just the absence
    of activity) is deliberate insurance: an empty, garbled, or failed
    capture must never read as "finished" — fails closed on empty/whitespace
    input.
    """
    if not text.strip():
        return False
    if is_main_turn_active(text):
        return False
    chrome = _chrome(text)
    return bool(_TURN_SUMMARY_RE.search(chrome) or _IDLE_BARE_RE.search(chrome))


def _is_boundary_line(line: str) -> bool:
    """True iff ``line`` is TUI boundary chrome that ends an open reply span
    (see :func:`extract_open_reply`): a box rule, the turn-summary line, the
    bypass-permissions footer, a bare idle prompt, or a well-formed main-turn
    spinner line.

    ROUND-2 ADDITION (2026-08-21 review): recognizing a live spinner as a
    boundary keeps it from being swallowed into a salvage candidate's text.
    Deliberately uses ONLY the narrow, paren-anchored `_MAIN_SPINNER_RE` here
    — NOT the broader `_WORKING_RE` (which also matches the non-paren
    spinner shape and the legacy "esc to interrupt" hint). This is
    intentional, not an oversight: `ask()`'s salvage condition
    (claude_session.py) separately checks the FULL `_WORKING_RE` against the
    region this function returns, as a broader safety net. If this function
    used the broad pattern too, it would strip the non-paren spinner text out
    of the candidate before that safety net ever saw it — making it dead
    code for exactly the shape it exists to catch (verified by hand-tracing
    both orderings). Keep this function narrow; let the caller's check stay
    broad."""
    return bool(
        _BOX_RULE_RE.search(line)
        or _TURN_SUMMARY_RE.search(line)
        or _FOOTER_RE.search(line)
        or _IDLE_BARE_RE.search(line)
        or _MAIN_SPINNER_RE.search(line)
    )


def extract_open_reply(text: str, rid: str) -> str | None:
    """Return the body of an UNTERMINATED ``<<<R:rid>>>`` span (no closing
    ``<<<E:rid>>>``), or ``None``.

    The salvage path for backlog item 10: the model occasionally never
    emits the closing marker for an otherwise-complete answer, and with no
    E line :func:`extract_reply` (which requires a complete pair) has
    nothing to return — there is otherwise no delivery path at all for that
    turn. Deliberately a SEPARATE function rather than a mode of
    :func:`extract_reply`: a complete pair is always preferred, so this
    never shadows or duplicates that authoritative path (rule 1 below).

    Rules:
    1. Returns ``None`` if a COMPLETE pair for ``rid`` already exists via
       :func:`extract_reply`.
    2. Finds the LAST line-anchored ``<<<R:rid>>>`` line (same
       line-anchoring as :func:`extract_reply`, so the marker-instruction
       ECHO — which has text after the marker on the same line — can never
       be mistaken for a real one). Returns ``None`` if absent.
    3. The body starts after that line and ends at the FIRST line at or
       after it that is TUI boundary chrome (see :func:`_is_boundary_line`)
       — or end-of-text.
    4. Runs the result through :func:`unwrap_soft_breaks` and ``.strip()``,
       the same normalization every normally-delivered reply already goes
       through.
    5. Returns ``None`` if the result is empty after stripping.
    """
    _require_rid(rid)
    if extract_reply(text, rid) is not None:
        return None  # a complete pair exists — defer to the authoritative path
    opens = list(_line_anchored(rid, "R").finditer(text))
    if not opens:
        return None
    tail = text[opens[-1].end() :]
    # `tail` starts with the newline right after the marker's own line (or
    # is empty at end-of-text); drop that leading (empty) element.
    lines = tail.split("\n")[1:]
    body_lines: list[str] = []
    for line in lines:
        if _is_boundary_line(line):
            break
        body_lines.append(line)
    body = unwrap_soft_breaks("\n".join(body_lines)).strip()
    return body or None


def strip_open_reply_body(text: str, rid: str) -> str:
    """Remove the BODY of an UNTERMINATED ``<<<R:rid>>>`` span from ``text``,
    keeping the marker line and any boundary chrome that follows intact.

    Companion to :func:`strip_reply_bodies` (which only ever removes CLOSED
    pairs, by design — see its docstring). An OPEN span is the model's own
    in-progress or marker-dropped prose; if it happens to quote rate-limit-
    looking text, that text sat in the chrome window forever (until the next
    turn), and ``classify_state()`` reported a false, poll-resistant
    RATE_LIMITED that no number of confirm-polls could clear (Fable audit F4
    / R5) — the frame is static, so the signature "survives" indefinitely.

    Walks the body the SAME way :func:`extract_open_reply` does (stopping at
    the first :func:`_is_boundary_line`), so the two functions can never
    disagree about where the body ends — a REAL rate-limit banner appearing
    AFTER the open span (outside the region this strips) stays fully visible
    to ``classify_state()``.
    """
    _require_rid(rid)
    if extract_reply(text, rid) is not None:
        return text  # a complete pair exists — strip_reply_bodies handles it
    opens = list(_line_anchored(rid, "R").finditer(text))
    if not opens:
        return text
    marker_end = opens[-1].end()
    nl = text.find("\n", marker_end)
    if nl == -1:
        return text  # the marker is the very last line — no body to strip
    body_start = nl + 1
    pos = body_start
    while pos <= len(text):
        line_end = text.find("\n", pos)
        line = text[pos : line_end if line_end != -1 else len(text)]
        if _is_boundary_line(line):
            break
        if line_end == -1:
            pos = len(text)
            break
        pos = line_end + 1
    return text[:body_start] + text[pos:]


def parse_reset_time(text: str) -> tuple[int, int] | None:
    """``(hour, minute)`` in 24h UTC from a limit banner's "resets …" clause.

    The banner names a wall-clock time with no date ("resets 12am (UTC)"), so
    this returns the time-of-day only; turning it into an instant needs the
    moment the banner was observed, which is the watchdog's job. Returns None
    when no limit banner is present or its time is unparseable — the caller
    must then keep waiting rather than guess.

    UTC is required, and only when it is on the SAME LINE as the matched
    time: the watchdog unconditionally treats the returned (hour, minute) as
    UTC (see watchdog._reset_deadline), but the CLI's rendering has changed
    format before (see _WORKING_RE's history) and could print a localized
    time with no marker at all. Silently mis-anchoring by hours with no log
    line to explain it is a worse failure than the caller's limit_max_wait
    fallback, so an unmarked time is treated as unparseable (caller logs it).
    """
    stripped = strip_reply_bodies(text)
    chrome = _chrome(stripped)
    if not _RATE_RE.search(chrome):
        return None
    m = _RESET_TIME_RE.search(chrome)
    if not m:
        return None
    line_start = chrome.rfind("\n", 0, m.start()) + 1
    line_end = chrome.find("\n", m.end())
    line = chrome[line_start : line_end if line_end != -1 else len(chrome)]
    if "utc" not in line.lower():
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    meridiem = (m.group(3) or "").lower()
    if meridiem == "am":
        hour = 0 if hour == 12 else hour
    elif meridiem == "pm":
        hour = 12 if hour == 12 else hour + 12
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def reset_epoch(text: str, *, seen_at: float) -> float | None:
    """Epoch of the limit's reset for a banner captured at ``seen_at``, or
    None if :func:`parse_reset_time` can't read a reset time from ``text``.

    The banner names a wall-clock time with no date, so turning it into an
    instant needs the moment it was observed: the reset is the first
    occurrence of that UTC time at or after ``seen_at`` (rolling forward a
    day if that time-of-day has already passed today). Shared anchoring math
    for both callers that need it — the main-session watchdog
    (``watchdog._reset_deadline``) and the cron runner
    (``cron_runner.CronRunner._limit_recovery``) — so there is exactly one
    implementation of "roll forward by a day" to keep in sync.
    """
    parsed = parse_reset_time(text)
    if parsed is None:
        return None
    hour, minute = parsed
    seen = datetime.fromtimestamp(seen_at, tz=UTC)
    cand = seen.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if cand < seen:
        cand += timedelta(days=1)
    return cand.timestamp()


def unmarked_reset_banner(text: str) -> str | None:
    """The banner LINE when a rate-limit banner names a reset time but the
    UTC marker :func:`parse_reset_time` requires is missing from that line,
    or ``None`` otherwise (no banner at all, or the time WAS parseable).

    Diagnostic-only, for the watchdog: :func:`parse_reset_time` returning
    ``None`` is ambiguous by itself — no banner present, a banner with no
    time-of-day clause at all, and a banner whose time IS there but unmarked
    all look identical to a caller. The last case is the one worth a loud
    log line (the watchdog otherwise falls back to ``limit_max_wait`` with no
    explanation at all — a silent multi-hour mis-wait, see review 2026-08-20).
    """
    stripped = strip_reply_bodies(text)
    chrome = _chrome(stripped)
    if not _RATE_RE.search(chrome):
        return None
    m = _RESET_TIME_RE.search(chrome)
    if not m:
        return None
    line_start = chrome.rfind("\n", 0, m.start()) + 1
    line_end = chrome.find("\n", m.end())
    line = chrome[line_start : line_end if line_end != -1 else len(chrome)]
    if "utc" in line.lower():
        return None  # was parseable; not this diagnostic's concern
    return line.strip()


def classify_state(text: str) -> PaneState:
    """Classify the pane into a coarse state.

    State signatures are matched against the chrome region only; STARTING is
    matched against the whole text (its banner can sit above the fold during
    boot). Priority: TRUST > RATE_LIMITED > LOGGED_OUT > READY > STARTING.
    """
    if not text.strip():
        return PaneState.UNKNOWN
    # Classify the CLI's own output only — never the model's answers. Without
    # this a reply that quotes a limit banner (e.g. one discussing this bug)
    # classifies the healthy session as RATE_LIMITED.
    text = strip_reply_bodies(text)
    # TRUST is a full-screen modal whose menu sits at the TOP; on a tall pane
    # the chrome (bottom) is blank, so match it over the WHOLE pane. Safe
    # because it anchors on the numbered menu line, which the model cannot
    # reproduce verbatim in a reply.
    if _TRUST_MENU_RE.search(text):
        return PaneState.TRUST_PROMPT
    # BYPASS is also a full-screen modal at the TOP — match over the whole pane
    # (chrome below is blank). Require BOTH the unique title and the numbered
    # accept line, AND the ABSENCE of the idle footer: the real modal appears
    # BEFORE the normal TUI (no "bypass permissions on" footer yet), while a
    # working/idle session always shows that footer. Without this guard a reply
    # that quotes the warning from scrollback (capture spans -S -200) would be
    # taken for the modal — and the watchdog, which does not list BYPASS_PROMPT
    # as serviceable, would force-recover a perfectly healthy session.
    # Checked before READY so the modal's `❯ 1. No, exit` selector (which
    # matches _IDLE_RE) can't be mistaken for an idle prompt.
    if (
        _BYPASS_TITLE_RE.search(text)
        and _BYPASS_MENU_RE.search(text)
        and not _FOOTER_RE.search(text)
    ):
        return PaneState.BYPASS_PROMPT
    chrome = _chrome(text)
    if _RATE_RE.search(chrome):
        return PaneState.RATE_LIMITED
    if _LOGGED_OUT_RE.search(chrome):
        return PaneState.LOGGED_OUT
    if _FOOTER_RE.search(text) or _IDLE_RE.search(chrome):
        return PaneState.READY
    if _STARTING_RE.search(text):
        return PaneState.STARTING
    return PaneState.UNKNOWN
